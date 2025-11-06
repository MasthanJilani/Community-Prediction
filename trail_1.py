#!/usr/bin/env python3
import os
import sys
import gzip
import zipfile
import random
import argparse
from pathlib import Path
import tempfile
import shutil
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import networkx as nx
import community as community_louvain
from tqdm import tqdm
import json
import time
import math

def sparse_to_nx(A):
    # Converting adjacency matrix to a networkx Graph for compatibility
    if hasattr(nx, "from_scipy_sparse_array"):
        return nx.from_scipy_sparse_array(A)
    elif hasattr(nx, "from_scipy_sparse_matrix"):
        return nx.from_scipy_sparse_matrix(A)
    else:
        # converting to dense numpy array (for dense graphs)
        arr = A.todense() if sp.issparse(A) else np.asarray(A)
        return nx.from_numpy_array(np.asarray(arr))
    
def find_dataset_files(data_dir):
    p = Path(data_dir)
    files = list(p.iterdir())
    dblp_files = [f for f in files if 'dblp' in f.name.lower() or 'db' in f.name.lower() and 'dblp' in f.name.lower()]
    enron_files = [f for f in files if 'enron' in f.name.lower() or 'email' in f.name.lower()]

    if not dblp_files:
        dblp_files = [f for f in files if 'db' in f.name.lower() or 'dbl' in f.name.lower()]
    if not enron_files:
        enron_files = [f for f in files if 'enron' in f.name.lower() or 'email' in f.name.lower()]
    return dblp_files, enron_files, files

def extract_zip_to_temp(zip_path):
    tmpdir = tempfile.mkdtemp(prefix='symnmf_zip_')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmpdir)
    cand = []
    for root, _, filenames in os.walk(tmpdir):
        for fn in filenames:
            if fn.endswith(('.txt', '.csv', '.edges')) or not '.' in fn:
                cand.append(Path(root) / fn)
    return tmpdir, cand

def open_text_file(path):
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt', errors='ignore')
    else:
        return open(path, 'rt', errors='ignore')

def guess_and_read_edges(path):
    edges = []
    p = Path(path)
    if p.suffix == '.zip':
        tmpdir, cand = extract_zip_to_temp(p)
        for f in cand:
            try:
                with open_text_file(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith('#'): continue
                        parts = line.split()
                        if len(parts) < 2:
                            parts = [x.strip() for x in line.split(',') if x.strip()]
                        if len(parts) >= 2:
                            edges.append((parts[0], parts[1]))
            except Exception:
                continue
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return edges
    else:
        try:
            with open_text_file(p) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split()
                    if len(parts) < 2:
                        parts = [x.strip() for x in line.split(',') if x.strip()]
                    if len(parts) >= 2:
                        edges.append((parts[0], parts[1]))
        except Exception as e:
            print(f"Error reading {p}: {e}")
            raise
        return edges

def build_adj_from_edges(edges, node_index=None, sym=True):
    # Returns: (A_sparse, node_index)
    if node_index is None:
        nodes = set()
        for u,v in edges:
            nodes.add(u); nodes.add(v)
        nodes = sorted(nodes)
        node_index = {nid: i for i,nid in enumerate(nodes)}
    n = len(node_index)
    rows = []
    cols = []
    data = []
    for u,v in edges:
        if u in node_index and v in node_index:
            i = node_index[u]; j = node_index[v]
            rows.append(i); cols.append(j); data.append(1.0)
            if sym:
                rows.append(j); cols.append(i); data.append(1.0)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A.setdiag(0); A.eliminate_zeros()
    return A, node_index

def align_layers(edgefile_paths, align='intersection', nmax=None, seed=0):
    # Returns: A_list (aligned), node_ids (list)
    all_edges = []
    node_sets = []
    for p in edgefile_paths:
        edges = guess_and_read_edges(p)
        all_edges.append(edges)
        node_sets.append(set(u for e in edges for u in e))
    if align == 'intersection':
        common = set.intersection(*node_sets)
        node_ids = sorted(common)
    else:
        node_ids = sorted(set.union(*node_sets))
    if nmax is not None and len(node_ids) > nmax:
        random.seed(seed)
        node_ids = random.sample(node_ids, nmax)
        node_ids = sorted(node_ids)
    node_index = {nid:i for i,nid in enumerate(node_ids)}
    A_list = []
    for edges in all_edges:
        A, _ = build_adj_from_edges(edges, node_index=node_index, sym=True)
        A_list.append(A)
    return A_list, node_ids

# Layer-aware SymNMF core
def layer_aware_symnmf(A_list, k=20, w=None, nu=1e-6, mu=1e-6, lam=0.0, eps=1e-12, max_iter=200, tol=1e-4, verbose=True, sample_obj_limit=3000):
    # returns: H (n x k numpy), D_list (list of k x k arrays), history dict
    L = len(A_list)
    n = A_list[0].shape[0]
    if w is None:
        w = np.ones(L) / L
    H = np.abs(np.random.randn(n, k)).astype(float)
    D_list = [np.eye(k) for _ in range(L)]
    prev_obj = np.inf
    history = {"obj": []}
    for it in range(1, max_iter+1):
        if verbose:
            print(f"[SymNMF] iter {it}/{max_iter}  n={n} k={k}")
        for ell in range(L):
            A = A_list[ell]
            D = np.zeros((k,k), dtype=float)
            for r in range(k):
                h_r = H[:, r]
                Ah = A.dot(h_r)
                num = float(h_r.dot(Ah))
                denom = (float(h_r.dot(h_r))**2) + nu
                d_rr = max(0.0, num / denom) if denom>0 else 0.0
                D[r,r] = d_rr
            D_list[ell] = D
        Num = np.zeros((n,k), dtype=float)
        Den = np.zeros((n,k), dtype=float)
        for ell in range(L):
            A = A_list[ell]
            D = D_list[ell]
            HD = H.dot(D)
            Num += w[ell] * (A.dot(HD))
            Den += w[ell] * ((HD.dot(H.T)).dot(HD))
        H *= (Num / (Den + eps))
        H[H < 0] = 0.0
        obj = 0.0
        if n <= sample_obj_limit:
            for ell in range(L):
                A = A_list[ell]
                D = D_list[ell]
                HD = H.dot(D)
                R = HD.dot(H.T)
                if sp.issparse(A):
                    diff = (A.todense() - R)
                    obj += w[ell] * (np.linalg.norm(diff, 'fro')**2)
                else:
                    diff = A - R
                    obj += w[ell] * (np.linalg.norm(diff, 'fro')**2)
        else:
            samples = 5000
            for ell in range(L):
                A = A_list[ell]
                nz_rows, nz_cols = A.nonzero()
                m = len(nz_rows)
                s = min(samples, m) if m>0 else 0
                s_idx = np.random.choice(m, s, replace=False) if s>0 else []
                sqsum = 0.0
                for idx in s_idx:
                    i = nz_rows[idx]; j = nz_cols[idx]
                    aij = A[i,j]
                    # compute r_ij via HD H^T: r_ij = (H[i,:] D) dot H[j,:].T
                    r_ij = float((H[i,:].dot(D_list[ell])).dot(H[j,:].T))
                    sqsum += (aij - r_ij)**2
                if m>0:
                    est = (m/s) * sqsum if s>0 else 0.0
                else:
                    est = 0.0
                obj += w[ell] * est
        if lam > 0:
            obj += lam * np.sum(np.abs(H))
        if mu > 0:
            ssum = 0.0
            for D in D_list:
                ssum += np.linalg.norm(D, 'fro')**2
            obj += mu * ssum
        history["obj"].append(obj)
        if verbose:
            print(f"  obj={obj:.6e}")
        if prev_obj < np.inf:
            rel = abs(prev_obj - obj) / (prev_obj + 1e-12)
            if rel < tol:
                if verbose: print("[SymNMF] Converged (relative obj change < tol).")
                break
        prev_obj = obj
    return H, D_list, history

# Postprocessing & evaluation
def hard_assignments_from_H(H, method='argmax', n_clusters=None, random_state=0):
    if method == 'argmax':
        labels = np.argmax(H, axis=1)
    elif method == 'kmeans':
        assert n_clusters is not None
        km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(H)
        labels = km.labels_
    else:
        raise ValueError("unknown method")
    return labels

def evaluate(labels_pred, A_list=None, labels_true=None):
    out = {}
    if labels_true is not None:
        out['NMI'] = normalized_mutual_info_score(labels_true, labels_pred)
        out['ARI'] = adjusted_rand_score(labels_true, labels_pred)
    if A_list is not None:
        modularities = []
        for A in A_list:
            try:
                G = sparse_to_nx(A)
            except Exception as e:
                rows, cols = A.nonzero()
                G = nx.Graph()
                G.add_nodes_from(range(A.shape[0]))
                G.add_edges_from(zip(rows.tolist(), cols.tolist()))
            # build communities as list of sets
            comms = []
            unique_labels = np.unique(labels_pred)
            for c in unique_labels:
                members = set(np.where(labels_pred == c)[0].tolist())
                if members:
                    comms.append(members)
            try:
                q = nx.community.quality.modularity(G, comms)
            except Exception:
                q = float('nan')
            modularities.append(q)
        out['modularity_per_layer'] = modularities
        out['modularity_mean'] = np.nanmean(modularities)
    return out

def baseline_louvain(A_list, w=None):
    if w is None:
        w = np.ones(len(A_list)) / len(A_list)
    Aagg = sum(w[i] * A_list[i] for i in range(len(A_list)))
    # convert aggregate adjacency to networkx Graph
    G = sparse_to_nx(Aagg)
    partition = community_louvain.best_partition(G)
    # convert to labels in node order 0..n-1
    labels = np.array([partition.get(i, 0) for i in range(Aagg.shape[0])], dtype=int)
    return labels

def main(args):
    dd = Path(args.data_dir)
    if not dd.exists():
        print("data_dir not found:", dd)
        return
    dblp_files, enron_files, all_files = find_dataset_files(dd)
    print("Detected files in data/:", [f.name for f in all_files])
    if dblp_files and enron_files:
        dblp_path = dblp_files[0]
        enron_path = enron_files[0]
    else:
        if len(all_files) >= 2:
            dblp_path = all_files[0]
            enron_path = all_files[1]
            print("Warning: couldn't auto-detect DBLP/Enron specifically; using first two files.")
        else:
            print("Need at least two files in data/. Found:", len(all_files))
            return
    print("Using layer files:", dblp_path.name, enron_path.name)

    edgefiles = [dblp_path, enron_path]
    print("Align strategy:", args.align, "nmax:", args.nmax)
    A_list, node_ids = align_layers(edgefiles, align=args.align, nmax=args.nmax, seed=args.seed)
    n = len(node_ids)
    print(f"Built multiplex: L={len(A_list)}, n={n}")

    for i,A in enumerate(A_list):
        print(f" Layer {i+1}: nnz={A.nnz} density={A.nnz/(n*n):.3e}")

    k = args.k
    print("Running SymNMF k=", k)
    start = time.time()
    H, D_list, hist = layer_aware_symnmf(A_list, k=k, max_iter=args.max_iter, tol=args.tol,
                                         nu=args.nu, mu=args.mu, lam=args.lam, eps=1e-12, verbose=True)
    elapsed = time.time() - start
    print(f"SymNMF finished in {elapsed:.1f}s. iterations: {len(hist['obj'])}")

    labels = hard_assignments_from_H(H, method='argmax')
    print("Cluster sizes:", np.bincount(labels))

    res = evaluate(labels, A_list=A_list, labels_true=None)
    print("Evaluation:", json.dumps(res, indent=2))

    lab_louv = baseline_louvain(A_list)
    res_louv = evaluate(lab_louv, A_list=A_list, labels_true=None)
    print("Louvain baseline evaluation:", json.dumps(res_louv, indent=2))

    out = {
        "node_count": n,
        "k": k,
        "labels_symnmf": labels.tolist(),
        "labels_louvain": lab_louv.tolist(),
        "D_list": [np.diag(D).tolist() for D in D_list],
        "history_obj": hist['obj'],
        "eval_symnmf": res,
        "eval_louvain": res_louv,
        "runtime_seconds": elapsed
    }
    out_path = dd / f"trail_1_k{k}.json"
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print("Saved results to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--align", type=str, choices=['intersection','union'], default='intersection')
    parser.add_argument("--nmax", type=int, default=5000, help="max number of nodes (sample) to keep for experiments")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--nu", type=float, default=1e-6)
    parser.add_argument("--mu", type=float, default=1e-6)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
