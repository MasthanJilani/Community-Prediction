#!/usr/bin/env python3
import os
import argparse
import random
import gzip
import zipfile
import shutil
import tempfile
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import networkx as nx
import community as community_louvain
from tqdm import tqdm

def sparse_to_nx(A):
    if hasattr(nx, "from_scipy_sparse_array"):
        return nx.from_scipy_sparse_array(A)
    elif hasattr(nx, "from_scipy_sparse_matrix"):
        return nx.from_scipy_sparse_matrix(A)
    else:
        arr = A.todense() if sp.issparse(A) else np.asarray(A)
        return nx.from_numpy_array(np.asarray(arr))

def open_text_file(p):
    p = Path(p)
    if p.suffix == '.gz':
        return gzip.open(p, 'rt', errors='ignore')
    else:
        return open(p, 'rt', errors='ignore')

def extract_zip_to_temp(zip_path):
    tmpdir = tempfile.mkdtemp(prefix='symnmf_zip_')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmpdir)
    cand = []
    for root, _, filenames in os.walk(tmpdir):
        for fn in filenames:
            if fn.endswith(('.txt', '.csv', '.edges')) or '.' not in fn:
                cand.append(Path(root) / fn)
    return tmpdir, cand

def guess_and_read_edges(path):
    edges = []
    p = Path(path)
    if p.suffix == '.zip':
        tmpdir, cand = extract_zip_to_temp(p)
        for f in cand:
            try:
                with open_text_file(f) as fh:
                    for line in fh:
                        s = line.strip()
                        if not s or s.startswith('#'): continue
                        parts = s.split()
                        if len(parts) < 2:
                            parts = [x.strip() for x in s.split(',') if x.strip()]
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
        with open_text_file(p) as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith('#'): continue
                parts = s.split()
                if len(parts) < 2:
                    parts = [x.strip() for x in s.split(',') if x.strip()]
                if len(parts) >= 2:
                    edges.append((parts[0], parts[1]))
        return edges

def build_adj_from_edges(edges, node_index=None, sym=True):
    if node_index is None:
        nodes = sorted({u for u,v in edges} | {v for u,v in edges})
        node_index = {nid:i for i,nid in enumerate(nodes)}
    n = len(node_index)
    rows=[]; cols=[]; data=[]
    for u,v in edges:
        if u in node_index and v in node_index:
            i=node_index[u]; j=node_index[v]
            rows.append(i); cols.append(j); data.append(1.0)
            if sym:
                rows.append(j); cols.append(i); data.append(1.0)
    A = sp.csr_matrix((data,(rows,cols)), shape=(n,n))
    A.setdiag(0); A.eliminate_zeros()
    return A, node_index

def align_layers(edgefile_paths, align='intersection', nmax=None, seed=0):
    all_edges = []
    node_sets = []
    for p in edgefile_paths:
        edges = guess_and_read_edges(p)
        all_edges.append(edges)
        node_sets.append({u for u,v in edges} | {v for u,v in edges})
    if align == 'intersection':
        node_ids = sorted(set.intersection(*node_sets))
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

# Preprocessing: normalization and weights
def symmetric_normalize(A):
    deg = np.array(A.sum(axis=1)).flatten()
    deg_safe = np.maximum(deg, 1.0)
    inv_sqrt = 1.0 / np.sqrt(deg_safe)
    Dinv = sp.diags(inv_sqrt)
    A_norm = Dinv.dot(A).dot(Dinv)
    A_norm.setdiag(0); A_norm.eliminate_zeros()
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

# Blockwise denominator
def compute_den_blockwise(HD, H, block_size=500):
    n,k = HD.shape
    Den = np.zeros((n,k), dtype=float)
    HT = H.T
    for i in range(0, n, block_size):
        i2 = min(n, i+block_size)
        HD_block = HD[i:i2, :]
        tmp = HD_block.dot(HT)   # (b, n)
        Den[i:i2, :] = tmp.dot(HD)
    return Den

# Core SymNMF (operates on preprocessed A_list and weights w)
def layer_aware_symnmf_core(A_proc, w, k=20, nu=1e-6, mu=1e-2, lam=1e-3, dclip=(0.0, 1e3), eps=1e-12, max_iter=200, tol=1e-4, block_size=500, seed=0, verbose=False):

    rng = np.random.RandomState(seed)
    L = len(A_proc)
    n = A_proc[0].shape[0]
    H = np.abs(rng.randn(n, k)).astype(float)
    D_list = [np.eye(k) for _ in range(L)]
    prev_obj = np.inf
    history = {"obj": []}

    for it in range(1, max_iter+1):
        if verbose:
            print(f"[SymNMF core] iter {it}/{max_iter} n={n} k={k}")
        # update D
        for ell in range(L):
            A = A_proc[ell]
            D = np.zeros((k,k), dtype=float)
            for r in range(k):
                h_r = H[:, r]
                Ah = A.dot(h_r)
                num = float(h_r.dot(Ah))
                denom = (float(h_r.dot(h_r))**2) + nu
                d_rr = max(0.0, num / denom) if denom > 0 else 0.0
                # clip
                d_rr = min(max(d_rr, dclip[0]), dclip[1])
                D[r,r] = d_rr
            D_list[ell] = D

        # numerator and denominator (blockwise)
        Num = np.zeros((n,k), dtype=float)
        Den = np.zeros((n,k), dtype=float)
        for ell in range(L):
            A = A_proc[ell]
            D = D_list[ell]
            HD = H.dot(D)
            Num += w[ell] * (A.dot(HD))
            Den += w[ell] * compute_den_blockwise(HD, H, block_size=block_size)

        H *= (Num / (Den + eps))
        if lam > 0:
            H = np.maximum(0.0, H - lam)
        # row-normalize to avoid collapse
        row_sums = H.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        H = H / row_sums

        obj = 0.0
        if n <= 3000:
            for ell in range(L):
                A = A_proc[ell]
                D = D_list[ell]
                HD = H.dot(D)
                R = HD.dot(H.T)
                diff = (A.todense() if sp.issparse(A) else A) - R
                obj += w[ell] * (np.linalg.norm(diff, 'fro')**2)
        else:
            samp = 5000
            for ell in range(L):
                A = A_proc[ell]
                nz_rows, nz_cols = A.nonzero()
                m = len(nz_rows)
                if m > 0:
                    s = min(samp, m)
                    idxs = np.random.choice(m, s, replace=False)
                    sqsum = 0.0
                    for idx in idxs:
                        i = nz_rows[idx]; j = nz_cols[idx]
                        aij = A[i,j]
                        r_ij = float((H[i,:].dot(D_list[ell])).dot(H[j,:].T))
                        sqsum += (aij - r_ij)**2
                    est = (m/s) * sqsum
                    obj += w[ell] * est

        if mu > 0:
            s = sum(np.linalg.norm(D, 'fro')**2 for D in D_list)
            obj += mu * s

        history["obj"].append(obj)
        if verbose:
            print(f"  obj={obj:.6e}")
        if prev_obj < np.inf:
            rel = abs(prev_obj - obj) / (prev_obj + 1e-12)
            if rel < tol:
                if verbose:
                    print("[SymNMF core] converged (relative obj change < tol).")
                break
        prev_obj = obj

    return H, D_list, history

def hard_assignments_from_H_argmax(H):
    return np.argmax(H, axis=1)

def hard_assignments_from_H_kmeans_norm(H, n_clusters, seed=0):
    H_rows = H.copy()
    row_norms = np.linalg.norm(H_rows, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    H_norm = H_rows / row_norms
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(H_norm)
    return km.labels_

def evaluate_partition(labels_pred, A_list=None, labels_true=None):
    out = {}
    if labels_true is not None:
        out['NMI'] = normalized_mutual_info_score(labels_true, labels_pred)
        out['ARI'] = adjusted_rand_score(labels_true, labels_pred)
    if A_list is not None:
        modularities = []
        for A in A_list:
            try:
                G = sparse_to_nx(A)
                comms = [set(np.where(labels_pred==c)[0].tolist()) for c in np.unique(labels_pred)]
                comms = [c for c in comms if len(c) > 0]
                q = nx.community.quality.modularity(G, comms)
            except Exception:
                q = float('nan')
            modularities.append(q)
        out['modularity_per_layer'] = modularities
        out['modularity_mean'] = float(np.nanmean(modularities))
    return out

def baseline_louvain(A_list, w=None):
    if w is None:
        w = np.ones(len(A_list)) / len(A_list)
    Aagg = sum(w[i] * A_list[i] for i in range(len(A_list)))
    G = sparse_to_nx(Aagg)
    partition = community_louvain.best_partition(G)
    labels = np.array([partition.get(i, 0) for i in range(Aagg.shape[0])], dtype=int)
    return labels

def find_files_in_data(data_dir):
    p = Path(data_dir)
    files = sorted([f for f in p.iterdir() if f.is_file()])
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--align", choices=['intersection','union'], default='intersection')
    parser.add_argument("--nmax", type=int, default=2000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--nu", type=float, default=1e-6)
    parser.add_argument("--mu", type=float, default=1e-2)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--dclip_max", type=float, default=1e3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--normalize", action='store_true', help="apply symmetric degree normalization")
    parser.add_argument("--block_size", type=int, default=500)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--save_H", action='store_true', help="save H matrix in output json (may be large)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print("data_dir does not exist:", data_dir)
        return

    files = find_files_in_data(data_dir)
    if len(files) < 2:
        print("Please place at least two edge files in data_dir. Found:", files)
        return

    print("Files found in data_dir:", [f.name for f in files])
    file1, file2 = files[0], files[1]
    print("Using files:", file1.name, file2.name)

    A_list_raw, node_ids = align_layers([file1, file2], align=args.align, nmax=args.nmax, seed=args.seed)
    n = len(node_ids)
    print(f"Built multiplex: L={len(A_list_raw)} n={n}")
    for i,A in enumerate(A_list_raw):
        print(f" Layer {i+1}: nnz={A.nnz} density={A.nnz/(n*n):.3e}")

    if args.normalize:
        A_proc = [symmetric_normalize(A) for A in A_list_raw]
    else:
        A_proc = [A.copy() for A in A_list_raw]
    w = compute_layer_weights(A_proc, method='inv_nnz')
    print("Layer weights (inv_nnz):", w.tolist())

    best_overall = None
    start_all = time.time()
    for r in range(args.restarts):
        seed_r = args.seed + r
        if args.verbose:
            print(f"\n=== Restart {r+1}/{args.restarts} seed={seed_r} ===")
        H_r, D_list_r, hist_r = layer_aware_symnmf_core(
            A_proc,
            w,
            k=args.k,
            nu=args.nu,
            mu=args.mu,
            lam=args.lam,
            dclip=(0.0, args.dclip_max),
            eps=1e-12,
            max_iter=args.max_iter,
            tol=args.tol,
            block_size=args.block_size,
            seed=seed_r,
            verbose=args.verbose
        )

        cand_labels = {}
        cand_labels['argmax'] = hard_assignments_from_H_argmax(H_r)
        try:
            cand_labels['kmeans_norm'] = hard_assignments_from_H_kmeans_norm(H_r, n_clusters=args.k, seed=seed_r)
        except Exception:
            cand_labels['kmeans_norm'] = cand_labels['argmax']

        best_local = None
        for method_name, labs in cand_labels.items():
            eval_res = evaluate_partition(labs, A_list=A_proc, labels_true=None)
            mod_mean = eval_res.get('modularity_mean', float('-inf'))
            if best_local is None or mod_mean > best_local[0]:
                best_local = (mod_mean, {
                    "seed": seed_r,
                    "method": method_name,
                    "labels": labs,
                    "D_list": [np.diag(D).tolist() for D in D_list_r],
                    "history": hist_r["obj"],
                    "eval": eval_res
                })
        print(f" Restart {r+1} seed={seed_r}: best_method={best_local[1]['method']} mod_mean={best_local[0]:.4f}")

        if best_overall is None or best_local[0] > best_overall[0]:
            best_overall = best_local

    total_time = time.time() - start_all
    if best_overall is None:
        print("No successful runs.")
        return

    best_mod, best_res = best_overall
    labels_best = best_res["labels"]
    D_list_best = best_res["D_list"]
    hist_best = best_res["history"]
    print("\n== Selected best run ==")
    print(f" seed={best_res['seed']}, method={best_res['method']}, modularity_mean={best_mod:.4f}")
    print(" Cluster sizes:", np.bincount(labels_best))

    lab_louv = baseline_louvain(A_proc, w=w)
    res_louv = evaluate_partition(lab_louv, A_list=A_proc, labels_true=None)
    print("Louvain baseline (on processed layers) modularity mean:", res_louv.get('modularity_mean'))

    out = {
        "n": n,
        "k": args.k,
        "best_seed": best_res['seed'],
        "best_method": best_res['method'],
        "labels_symnmf": labels_best.tolist(),
        "labels_louvain": lab_louv.tolist(),
        "D_list": D_list_best,
        "history_obj": hist_best,
        "eval_symnmf": best_res['eval'],
        "eval_louvain": res_louv,
        "w": w.tolist(),
        "files_used": [file1.name, file2.name],
        "runtime_seconds": total_time
    }
    out_path = data_dir / f"trail_4_k{args.k}.json"
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print("Saved results to", out_path)

if __name__ == "__main__":
    main()
