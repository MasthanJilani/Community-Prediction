#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import random
import gzip, zipfile, shutil, tempfile
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import networkx as nx
import community as community_louvain
import json, time

def sparse_to_nx(A):
    if hasattr(nx, "from_scipy_sparse_array"):
        return nx.from_scipy_sparse_array(A)
    elif hasattr(nx, "from_scipy_sparse_matrix"):
        return nx.from_scipy_sparse_matrix(A)
    else:
        arr = A.todense() if sp.issparse(A) else np.asarray(A)
        return nx.from_numpy_array(np.asarray(arr))

def open_text_file(path):
    p = Path(path)
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
        try: shutil.rmtree(tmpdir)
        except Exception: pass
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
    all_edges=[]; node_sets=[]
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
    A_list=[]
    for edges in all_edges:
        A, _ = build_adj_from_edges(edges, node_index=node_index, sym=True)
        A_list.append(A)
    return A_list, node_ids

# normalization and computing layer weights
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
        L = len(A_list); return np.ones(L) / L

def compute_den_blockwise(HD, H, block_size=500):
    # Computes (HD H^T) @ HD in blocks returning an (n,k) array
    # HD: (n,k), H: (n,k)
    n,k = HD.shape
    Den = np.zeros((n,k), dtype=float)
    HT = H.T
    for i in range(0, n, block_size):
        i2 = min(n, i+block_size)
        HD_block = HD[i:i2, :]          # (b,k)
        # (HD_block @ H.T) -> (b, n)
        tmp = HD_block.dot(HT)          # (b, n)
        # tmp @ HD -> (b,k)
        Den[i:i2, :] = tmp.dot(HD)
    return Den

def layer_aware_symnmf_improved(A_list, k=20, normalize=True, weight_method='inv_nnz', nu=1e-6, mu=1e-2, lam=1e-3, dclip=(0,1e3), eps=1e-12, max_iter=200, tol=1e-4, block_size=500, seed=0, verbose=True):
    random_state = np.random.RandomState(seed)
    L = len(A_list)
    n = A_list[0].shape[0]
    A_proc = []
    for A in A_list:
        if normalize:
            A_proc.append(symmetric_normalize(A))
        else:
            A_proc.append(A.copy())

    w = compute_layer_weights(A_proc, method=weight_method)

    H = np.abs(random_state.randn(n, k)).astype(float)
    D_list = [np.eye(k) for _ in range(L)]
    prev_obj = np.inf
    history = {"obj": []}

    for it in range(1, max_iter+1):
        if verbose:
            print(f"[SymNMF] iter {it}/{max_iter}  n={n} k={k}")
        for ell in range(L):
            A = A_proc[ell]
            D = np.zeros((k,k), dtype=float)
            for r in range(k):
                h_r = H[:, r]
                Ah = A.dot(h_r)
                num = float(h_r.dot(Ah))
                denom = (float(h_r.dot(h_r))**2) + nu
                d_rr = max(0.0, num/denom) if denom>0 else 0.0
                # clip to avoid explosion
                d_rr = min(max(d_rr, dclip[0]), dclip[1])
                D[r,r] = d_rr
            D_list[ell] = D

        Num = np.zeros((n,k), dtype=float)
        Den = np.zeros((n,k), dtype=float)
        for ell in range(L):
            A = A_proc[ell]
            D = D_list[ell]
            HD = H.dot(D)
            Num += w[ell] * (A.dot(HD))
            # compute denominator blockwise to avoid n x n dense formation
            Den += w[ell] * compute_den_blockwise(HD, H, block_size=block_size)

        ratio = Num / (Den + eps)
        H *= ratio
        if lam > 0:
            H = np.maximum(0.0, H - lam)
        # row-normalize to avoid blow-up / trivial solutions
        row_sums = H.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
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
                if m>0:
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
                if verbose: print("[SymNMF] Converged (relative obj change < tol).")
                break
        prev_obj = obj

    return H, D_list, history, A_proc, w

# Postprocessing & evaluation
def hard_assignments_from_H(H, method='argmax', n_clusters=None, random_state=0):
    if method == 'argmax':
        labels = np.argmax(H, axis=1)
    elif method == 'kmeans':
        km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(H)
        labels = km.labels_
    else:
        raise ValueError("unknown method")
    return labels

def evaluate(labels_pred, A_list=None, labels_true=None):
    out={}
    if labels_true is not None:
        out['NMI'] = normalized_mutual_info_score(labels_true, labels_pred)
        out['ARI'] = adjusted_rand_score(labels_true, labels_pred)
    if A_list is not None:
        modularities=[]
        for A in A_list:
            try:
                G = sparse_to_nx(A)
                comms = [set(np.where(labels_pred==c)[0].tolist()) for c in np.unique(labels_pred)]
                modularities.append(nx.community.quality.modularity(G, [c for c in comms if c]))
            except Exception:
                modularities.append(float('nan'))
        out['modularity_per_layer'] = modularities
        out['modularity_mean'] = float(np.nanmean(modularities))
    return out

def baseline_louvain(A_list, w=None):
    if w is None: w = np.ones(len(A_list))/len(A_list)
    Aagg = sum(w[i]*A_list[i] for i in range(len(A_list)))
    G = sparse_to_nx(Aagg)
    part = community_louvain.best_partition(G)
    labels = np.array([part.get(i,0) for i in range(Aagg.shape[0])], dtype=int)
    return labels

def find_files_in_data(data_dir):
    p=Path(data_dir)
    files=list(p.iterdir())
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--align", choices=['intersection','union'], default='intersection')
    parser.add_argument("--nmax", type=int, default=2000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mu", type=float, default=1e-2)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--dclip_max", type=float, default=1e3)
    parser.add_argument("--normalize", action='store_true', help="apply symmetric degree normalization")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    files = find_files_in_data(args.data_dir)
    print("Found in data/:", [f.name for f in files])
    if len(files) < 2:
        print("Please put at least two edge files in the data/ directory.")
        return
    f1=files[0]; f2=files[1]
    print("Using files:", f1.name, f2.name)
    A_list, node_ids = align_layers([f1, f2], align=args.align, nmax=args.nmax, seed=args.seed)
    print(f"Built multiplex L={len(A_list)} n={len(node_ids)}")
    for i,A in enumerate(A_list):
        print(f"  layer {i+1}: nnz={A.nnz} density={A.nnz/(A.shape[0]**2):.3e}")

    H, D_list, hist, A_proc, w = layer_aware_symnmf_improved(
        A_list,
        k=args.k,
        normalize=args.normalize,
        weight_method='inv_nnz',
        nu=1e-6,
        mu=args.mu,
        lam=args.lam,
        dclip=(0.0, args.dclip_max),
        eps=1e-12,
        max_iter=args.max_iter,
        tol=1e-4,
        block_size=500,
        seed=args.seed,
        verbose=args.verbose or True
    )

    labels = hard_assignments_from_H(H, method='argmax')
    print("Cluster distribution:", np.bincount(labels))
    res = evaluate(labels, A_list=A_proc, labels_true=None)
    print("SymNMF eval:", res)

    lab_louv = baseline_louvain(A_proc, w=w)
    res_louv = evaluate(lab_louv, A_list=A_proc, labels_true=None)
    print("Louvain eval (on normalized/weighted agg):", res_louv)

    out = {
        "n": len(node_ids),
        "k": args.k,
        "labels_symnmf": labels.tolist(),
        "labels_louvain": lab_louv.tolist(),
        "D_list": [np.diag(D).tolist() for D in D_list],
        "history_obj": hist["obj"],
        "eval_symnmf": res,
        "eval_louvain": res_louv,
        "w": w.tolist()
    }
    out_path = Path(args.data_dir) / f"trail_2_k{args.k}.json"
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print("Saved results to", out_path)

if __name__ == "__main__":
    main()
