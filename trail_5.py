#!/usr/bin/env python3
import os
import argparse
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import networkx as nx
import community as community_louvain
import json
import time
from pathlib import Path

def open_text_file(path):
    p = Path(path)
    if p.suffix == '.gz':
        import gzip
        return gzip.open(p, 'rt', errors='ignore')
    else:
        return open(p, 'rt', errors='ignore')

def guess_and_read_edges(path):
    edges = []
    p = Path(path)
    with open_text_file(p) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 2:
                parts = [x.strip() for x in s.split(',') if x.strip()]
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
    return edges

def build_adj_from_edges(edges, node_index=None, sym=True):
    if node_index is None:
        nodes = sorted({u for u, v in edges} | {v for u, v in edges})
        node_index = {nid: i for i, nid in enumerate(nodes)}
    n = len(node_index)
    rows, cols, data = [], [], []
    for u, v in edges:
        if u in node_index and v in node_index:
            i, j = node_index[u], node_index[v]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            if sym and i != j:
                rows.append(j)
                cols.append(i)
                data.append(1.0)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A.setdiag(0)
    A.eliminate_zeros()
    return A, node_index

def align_layers(edgefile_paths, align='intersection', nmax=None, seed=0):
    all_edges = []
    node_sets = []
    for p in edgefile_paths:
        edges = guess_and_read_edges(p)
        all_edges.append(edges)
        node_sets.append({u for u, v in edges} | {v for u, v in edges})
    
    if align == 'intersection':
        node_ids = sorted(set.intersection(*node_sets))
    else:
        node_ids = sorted(set.union(*node_sets))
    
    if nmax is not None and len(node_ids) > nmax:
        import random
        random.seed(seed)
        node_ids = random.sample(node_ids, nmax)
        node_ids = sorted(node_ids)
    
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    A_list = []
    for edges in all_edges:
        A, _ = build_adj_from_edges(edges, node_index=node_index, sym=True)
        A_list.append(A)
    
    return A_list, node_ids

def symmetric_normalize(A):
    deg = np.array(A.sum(axis=1)).flatten()
    deg_safe = np.maximum(deg, 1.0) # to avoid division w zero
    inv_sqrt = 1.0 / np.sqrt(deg_safe)
    Dinv = sp.diags(inv_sqrt)
    A_norm = Dinv @ A @ Dinv
    A_norm.setdiag(0)
    A_norm.eliminate_zeros()
    return A_norm

def compute_layer_weights(A_list, method='inv_nnz'):
    if method == 'inv_nnz':
        nnzs = np.array([A.nnz for A in A_list], dtype=float)
        w = 1.0 / (nnzs + 1e-9)
        w = w / w.sum()
        return w
    else:
        L = len(A_list)
        return np.ones(L) / L

def initialize_with_louvain(A_list, k, weights=None):
    # creating an aggregated graph and then converting it to a networkx graph
    if weights is None:
        weights = np.ones(len(A_list)) / len(A_list)
    A_agg = sum(weights[i] * A_list[i] for i in range(len(A_list)))
    if hasattr(nx, "from_scipy_sparse_array"):
        G = nx.from_scipy_sparse_array(A_agg)
    else:
        G = nx.from_scipy_sparse_matrix(A_agg)
    
    partition = community_louvain.best_partition(G)
    n = A_agg.shape[0]
    
    communities = set(partition.values())
    community_map = {comm: idx for idx, comm in enumerate(communities)}
    
    H = np.zeros((n, k))
    for node, comm in partition.items():
        if node < n:
            comm_idx = community_map[comm] % k
            H[node, comm_idx] = 1.0
    
    H += 0.1 * np.random.rand(n, k)
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    H = H / row_sums
    
    return H

def layer_aware_symnmf_improved(A_list, k=20, w=None, nu=1e-6, mu=1e-2, lam=1e-3, dclip=(0.0, 1e3), eps=1e-12, max_iter=300, tol=1e-5, use_louvain_init=True, verbose=True):
    L = len(A_list)
    n = A_list[0].shape[0]
    
    if w is None:
        w = compute_layer_weights(A_list)
    
    if use_louvain_init:
        H = initialize_with_louvain(A_list, k, w)
    else:
        H = np.abs(np.random.randn(n, k)).astype(float)
        # Normalize rows
        row_sums = H.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        H = H / row_sums
    
    D_list = [np.eye(k) for _ in range(L)]
    prev_obj = np.inf
    history = {"obj": []}
    
    # Adaptive learning rate parameters
    alpha = 0.1
    alpha_min = 0.01
    alpha_decay = 0.99
    
    for it in range(1, max_iter + 1):
        if verbose:
            print(f"[SymNMF] iter {it}/{max_iter}  n={n} k={k} alpha={alpha:.4f}")
        
        for ell in range(L):
            A = A_list[ell]
            D = np.zeros((k, k), dtype=float)
            for r in range(k):
                h_r = H[:, r]
                Ah = A @ h_r
                num = h_r @ Ah
                denom = (h_r @ h_r) ** 2 + nu
                d_rr = max(0.0, num / denom) if denom > 0 else 0.0
                d_rr = min(max(d_rr, dclip[0]), dclip[1])
                D[r, r] = d_rr
            D_list[ell] = D
        
        Num = np.zeros((n, k), dtype=float)
        Den = np.zeros((n, k), dtype=float)
        
        for ell in range(L):
            A = A_list[ell]
            D = D_list[ell]
            HD = H @ D
            
            Num += w[ell] * (A @ HD)
            
            # We use iterative approach for large matrices to avoid memory issues
            batch_size = min(500, n)
            for i in range(0, n, batch_size):
                i_end = min(i + batch_size, n)
                HD_batch = HD[i:i_end, :]
                H_batch = H[i:i_end, :]
                tmp = HD_batch @ H.T
                Den[i:i_end, :] += w[ell] * (tmp @ HD)
        
        update_ratio = Num / (Den + eps)
        H_new = H * (1 - alpha) + H * update_ratio * alpha
        
        if lam > 0:
            H_new = np.maximum(0.0, H_new - lam)
        
        row_sums = H_new.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        H_new = H_new / row_sums
        
        obj = 0.0
        for ell in range(L):
            A = A_list[ell]
            D = D_list[ell]
            HD = H_new @ D
            R = HD @ H_new.T
            diff = A - R

            if sp.issparse(diff):
                obj += w[ell] * float(diff.multiply(diff).sum())
            else:
                arr = np.asarray(diff)
                obj += w[ell] * float(np.sum(arr * arr))
        if mu > 0:
            for D in D_list:
                obj += mu * np.linalg.norm(D, 'fro') ** 2
        
        history["obj"].append(obj)
        
        if verbose:
            print(f"  obj={obj:.6e}")
        if prev_obj < np.inf:
            rel_change = abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)
            if rel_change < tol:
                if verbose:
                    print(f"[SymNMF] Converged (relative change {rel_change:.6e} < {tol})")
                break
            if obj < prev_obj:
                alpha = min(alpha * 1.05, 0.5)
            else:
                alpha = max(alpha * alpha_decay, alpha_min)
        
        H = H_new
        prev_obj = obj
    
    return H, D_list, history

def evaluate_partition(labels_pred, A_list=None, labels_true=None):
    out = {}
    if labels_true is not None:
        out['NMI'] = normalized_mutual_info_score(labels_true, labels_pred)
        out['ARI'] = adjusted_rand_score(labels_true, labels_pred)
    
    if A_list is not None:
        modularities = []
        for A in A_list:
            try:
                if hasattr(nx, "from_scipy_sparse_array"):
                    G = nx.from_scipy_sparse_array(A)
                else:
                    G = nx.from_scipy_sparse_matrix(A)

                communities = []
                unique_labels = np.unique(labels_pred)
                for c in unique_labels:
                    members = np.where(labels_pred == c)[0].tolist()
                    if members:
                        communities.append(set(members))

                # Calculate modularity
                q = nx.algorithms.community.quality.modularity(G, communities)
                modularities.append(q)
            except Exception as e:
                print(f"Error calculating modularity: {e}")
                modularities.append(float('nan'))

        out['modularity_per_layer'] = modularities
        out['modularity_mean'] = float(np.nanmean(modularities))

    return out

def baseline_louvain(A_list, w=None):
    if w is None:
        w = np.ones(len(A_list)) / len(A_list)
    
    A_agg = sum(w[i] * A_list[i] for i in range(len(A_list)))
    
    if hasattr(nx, "from_scipy_sparse_array"):
        G = nx.from_scipy_sparse_array(A_agg)
    else:
        G = nx.from_scipy_sparse_matrix(A_agg)
    
    partition = community_louvain.best_partition(G)
    labels = np.array([partition.get(i, 0) for i in range(A_agg.shape[0])], dtype=int)
    return labels

def main():
    parser = argparse.ArgumentParser(description="Improved Layer-Aware SymNMF for multiplex networks")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing graph files")
    parser.add_argument("--k", type=int, default=20, help="Number of communities")
    parser.add_argument("--nmax", type=int, default=5000, help="Maximum number of nodes to use")
    parser.add_argument("--max_iter", type=int, default=300, help="Maximum number of iterations")
    parser.add_argument("--tol", type=float, default=1e-5, help="Convergence tolerance")
    parser.add_argument("--nu", type=float, default=1e-6, help="Ridge parameter for D update")
    parser.add_argument("--mu", type=float, default=1e-2, help="Regularization for D matrices")
    parser.add_argument("--lam", type=float, default=1e-3, help="Sparsity regularization for H")
    parser.add_argument("--dclip_max", type=float, default=1e3, help="Maximum value for D entries")
    parser.add_argument("--normalize", action="store_true", help="Apply symmetric normalization to adjacency matrices")
    parser.add_argument("--no_louvain_init", action="store_true", help="Don't use Louvain initialization")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist")
        return
    
    files = list(data_dir.iterdir())
    if len(files) < 2:
        print("Need at least two graph files in the data directory")
        return
    
    print(f"Found {len(files)} files in {data_dir}")
    file1, file2 = files[0], files[1]
    print(f"Using files: {file1.name}, {file2.name}")
    A_list, node_ids = align_layers([file1, file2], align='intersection', nmax=args.nmax)
    n = len(node_ids)
    print(f"Aligned multiplex network with {n} nodes")
    if args.normalize:
        A_list = [symmetric_normalize(A) for A in A_list]
        print("Applied symmetric normalization to adjacency matrices")
    start_time = time.time()
    H, D_list, history = layer_aware_symnmf_improved(
        A_list, k=args.k, nu=args.nu, mu=args.mu, lam=args.lam, dclip=(0.0, args.dclip_max), max_iter=args.max_iter, tol=args.tol,use_louvain_init=not args.no_louvain_init, verbose=args.verbose
    )
    runtime = time.time() - start_time
    print(f"SymNMF completed in {runtime:.2f} seconds")
    labels = np.argmax(H, axis=1)
    print("Cluster sizes:", np.bincount(labels))
    eval_result = evaluate_partition(labels, A_list=A_list)
    print("Evaluation results:")
    print(f"  Mean modularity: {eval_result['modularity_mean']:.4f}")
    for i, mod in enumerate(eval_result['modularity_per_layer']):
        print(f"  Layer {i+1} modularity: {mod:.4f}")
    louvain_labels = baseline_louvain(A_list)
    louvain_eval = evaluate_partition(louvain_labels, A_list=A_list)
    print(f"Louvain baseline mean modularity: {louvain_eval['modularity_mean']:.4f}")
    
    results = {
        "n_nodes": n,
        "n_communities": args.k,
        "runtime_seconds": runtime,
        "modularity_mean": eval_result['modularity_mean'],
        "modularity_per_layer": eval_result['modularity_per_layer'],
        "louvain_modularity_mean": louvain_eval['modularity_mean'],
        "cluster_sizes": np.bincount(labels).tolist(),
        "convergence_history": history["obj"],
        "D_matrices": [D.tolist() for D in D_list],
        "parameters": vars(args)
    }
    
    output_file = data_dir / f"trail_5_k{args.k}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()