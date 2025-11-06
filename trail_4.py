#!/usr/bin/env python3
import argparse, json, time, random, gzip, zipfile, shutil, tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
import networkx as nx
import community as community_louvain

def open_text_file(p):
    p = Path(p)
    if p.suffix == '.gz':
        return gzip.open(p, 'rt', errors='ignore')
    return open(p, 'rt', errors='ignore')

def extract_zip_to_temp(zip_path):
    tmpdir = tempfile.mkdtemp(prefix='symnmf_zip_')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmpdir)
    cand = []
    import os
    for root, _, files in os.walk(tmpdir):
        for fn in files:
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

# Preprocessing
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
        L = len(A_list); return np.ones(L)/L

def compute_den_blockwise(HD, H, block_size=500):
    n,k = HD.shape
    Den = np.zeros((n,k), dtype=float)
    HT = H.T
    for i in range(0, n, block_size):
        i2 = min(n, i+block_size)
        HD_block = HD[i:i2, :]
        tmp = HD_block.dot(HT)   # (b,n)
        Den[i:i2, :] = tmp.dot(HD)
    return Den

def sparse_to_nx(A):
    if hasattr(nx, "from_scipy_sparse_array"):
        return nx.from_scipy_sparse_array(A)
    elif hasattr(nx, "from_scipy_sparse_matrix"):
        return nx.from_scipy_sparse_matrix(A)
    else:
        arr = A.todense() if sp.issparse(A) else np.asarray(A)
        return nx.from_numpy_array(np.asarray(arr))

def build_Aagg_and_Bdot(A_proc, w):
    A_agg = sum(w[i]*A_proc[i] for i in range(len(A_proc)))
    deg = np.array(A_agg.sum(axis=1)).flatten()
    m = float(deg.sum())/2.0 + 1e-12
    def B_dot(X):
        term1 = A_agg.dot(X)
        s = deg.dot(X)
        term2 = np.outer(deg, s) / (2.0*m)
        return term1 - term2
    return A_agg, B_dot, deg, m

# core SymNMF with modularity PGD
def symnmf_with_modularity(A_proc, w, k=20, nu=1e-6, mu=1e-2, lam=1e-3,
                           dclip=(0.0,1e3), eps=1e-12, max_iter=200, tol=1e-4,
                           block_size=500, seed=0, verbose=False,
                           gamma=1.0, pgd_steps=5, alpha=1e-3,
                           seed_nodes=None, eta_seed=0.0):
    rng = np.random.RandomState(seed)
    L = len(A_proc); n = A_proc[0].shape[0]
    H = np.abs(rng.randn(n,k)).astype(float)
    D_list = [np.eye(k) for _ in range(L)]
    prev_obj = np.inf
    history = {"obj": []}

    A_agg, B_dot, deg, m = build_Aagg_and_Bdot(A_proc, w)

    for it in range(1, max_iter+1):
        if verbose: print(f"[symnmf] iter {it}/{max_iter} n={n} k={k}")
        # update D closed form + clip
        for ell in range(L):
            A = A_proc[ell]
            D = np.zeros((k,k), dtype=float)
            for r in range(k):
                h_r = H[:, r]
                Ah = A.dot(h_r)
                num = float(h_r.dot(Ah))
                denom = (float(h_r.dot(h_r))**2) + nu
                d_rr = max(0.0, num/denom) if denom>0 else 0.0
                d_rr = min(max(d_rr, dclip[0]), dclip[1])
                D[r,r] = d_rr
            D_list[ell] = D
        # multiplicative numerator & denominator (blockwise)
        Num = np.zeros((n,k), dtype=float)
        Den = np.zeros((n,k), dtype=float)
        for ell in range(L):
            A = A_proc[ell]; D = D_list[ell]
            HD = H.dot(D)
            Num += w[ell] * (A.dot(HD))
            Den += w[ell] * compute_den_blockwise(HD, H, block_size=block_size)

        # multiplicative update
        H *= (Num / (Den + eps))
        if lam > 0:
            H = np.maximum(0.0, H - lam)
        row_sums = H.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        H = H / row_sums

        if pgd_steps > 0:
            for _ in range(pgd_steps):
                # g_recon ~ 2 * sum_l w_l * ( (HD H^T) H D - A H D )
                g_recon = np.zeros_like(H)
                for ell in range(L):
                    D = D_list[ell]; A = A_proc[ell]
                    HD = H.dot(D)
                    g_recon += 2.0 * w[ell] * ( (HD.dot(H.T)).dot(HD) - A.dot(HD) )
                g_mod = -2.0 * gamma * B_dot(H)
                g_seed = np.zeros_like(H)
                if eta_seed > 0 and seed_nodes is not None and len(seed_nodes)>0:
                    for i, lab in seed_nodes.items():
                        y = np.zeros((k,))
                        y[lab % k] = 1.0
                        g_seed[i,:] = 2.0 * eta_seed * (H[i,:] - y)
                g_total = g_recon + g_mod + g_seed
                H = H - alpha * g_total
                H = np.maximum(0.0, H)
                row_sums = H.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                H = H / row_sums

        obj = 0.0
        if n <= 3000:
            for ell in range(L):
                A = A_proc[ell]; D = D_list[ell]
                HD = H.dot(D)
                R = HD.dot(H.T)
                diff = (A.todense() if sp.issparse(A) else A) - R
                obj += w[ell] * (np.linalg.norm(diff, 'fro')**2)
        else:
            samp = 5000
            for ell in range(L):
                A = A_proc[ell]
                nz_rows, nz_cols = A.nonzero()
                m_nz = len(nz_rows)
                if m_nz>0:
                    s = min(samp, m_nz)
                    idxs = np.random.choice(m_nz, s, replace=False)
                    sqsum = 0.0
                    for idx in idxs:
                        i = nz_rows[idx]; j = nz_cols[idx]
                        aij = A[i,j]
                        r_ij = float((H[i,:].dot(D_list[ell])).dot(H[j,:].T))
                        sqsum += (aij - r_ij)**2
                    obj += w[ell] * (m_nz/s) * sqsum
        BH = None
        if gamma != 0:
            BH = build_BH_for_obj(H, A_proc, w)
            tr = np.sum(H * BH)
            obj -= gamma * tr

        if mu > 0:
            sD = sum(np.linalg.norm(D, 'fro')**2 for D in D_list)
            obj += mu * sD

        history["obj"].append(float(obj))
        if verbose:
            print(f"  obj={obj:.6e}")
        if prev_obj < np.inf:
            rel = abs(prev_obj - obj) / (prev_obj + 1e-12)
            if rel < tol:
                if verbose: print("[symnmf] converged.")
                break
        prev_obj = obj

    return H, D_list, history

def build_BH_for_obj(H, A_proc, w):
    A_agg = sum(w[i]*A_proc[i] for i in range(len(A_proc)))
    deg = np.array(A_agg.sum(axis=1)).flatten()
    m = float(deg.sum())/2.0 + 1e-12
    term1 = A_agg.dot(H)
    s = deg.dot(H)
    term2 = np.outer(deg, s) / (2.0*m)
    return term1 - term2

# Postprocessing & evaluation 
def argmax_labels(H):
    return np.argmax(H, axis=1)

def kmeans_norm_labels(H, k, seed=0):
    H_norm = H.copy()
    row_norms = np.linalg.norm(H_norm, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    H_norm = H_norm / row_norms
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(H_norm)
    return km.labels_

def evaluate_partition(labels, A_proc):
    res = {}
    modularities = []
    for A in A_proc:
        try:
            G = sparse_to_nx(A)
            comms = [set(np.where(labels==c)[0].tolist()) for c in np.unique(labels)]
            comms = [c for c in comms if len(c)>0]
            q = nx.community.quality.modularity(G, comms)
        except Exception:
            q = float('nan')
        modularities.append(q)
    res['modularity_per_layer'] = modularities
    res['modularity_mean'] = float(np.nanmean(modularities))
    return res

def sparse_to_nx(A):
    if hasattr(nx, "from_scipy_sparse_array"):
        return nx.from_scipy_sparse_array(A)
    elif hasattr(nx, "from_scipy_sparse_matrix"):
        return nx.from_scipy_sparse_matrix(A)
    else:
        arr = A.todense() if sp.issparse(A) else np.asarray(A)
        return nx.from_numpy_array(np.asarray(arr))

def compute_louvain_seeds(A_proc, w, max_seeds_fraction=0.05, small_comm_threshold=50):
    A_agg = sum(w[i]*A_proc[i] for i in range(len(A_proc)))
    G = sparse_to_nx(A_agg)
    part = community_louvain.best_partition(G)
    from collections import Counter
    counts = Counter(part.values())
    seeds = {}
    for node, lab in part.items():
        if counts[lab] <= small_comm_threshold:
            seeds[node] = lab
    max_seeds = int(max_seeds_fraction * A_agg.shape[0])
    if len(seeds) > max_seeds:
        items = list(seeds.items())
        random.shuffle(items)
        items = items[:max_seeds]
        seeds = dict(items)
    return seeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--align", choices=['intersection','union'], default='intersection')
    parser.add_argument("--nmax", type=int, default=2000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=1.0, help="modularity reg weight")
    parser.add_argument("--pgd_steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1e-3, help="PGD step size")
    parser.add_argument("--nu", type=float, default=1e-6)
    parser.add_argument("--mu", type=float, default=1e-2)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--dclip_max", type=float, default=1e3)
    parser.add_argument("--block_size", type=int, default=500)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--seed_supervision", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted([f for f in data_dir.iterdir() if f.is_file()])
    if len(files) < 2:
        print("Please put at least two edge files into", data_dir)
        return
    file1, file2 = files[0], files[1]
    print("Using files:", file1.name, file2.name)

    A_list_raw, node_ids = align_layers([file1, file2], align=args.align, nmax=args.nmax, seed=args.seed)
    n = len(node_ids)
    print(f"Built multiplex L={len(A_list_raw)} n={n}")
    for i,A in enumerate(A_list_raw):
        print(f" layer {i+1}: nnz={A.nnz} dens={A.nnz/(n*n):.3e}")

    if args.normalize:
        A_proc = [symmetric_normalize(A) for A in A_list_raw]
    else:
        A_proc = [A.copy() for A in A_list_raw]
    w = compute_layer_weights(A_proc, method='inv_nnz')
    print("Layer weights:", w.tolist())

    seed_nodes = None
    if args.seed_supervision:
        seed_nodes = compute_louvain_seeds(A_proc, w, max_seeds_fraction=0.05, small_comm_threshold=50)
        print("Seed nodes count:", len(seed_nodes))
    best_overall = None
    t0 = time.time()
    for r in range(args.restarts):
        seed_r = args.seed + r
        if args.verbose: print(f"\nRestart {r+1}/{args.restarts} seed={seed_r}")
        H, D_list, history = symnmf_with_modularity(
            A_proc, w, k=args.k,
            nu=args.nu, mu=args.mu, lam=args.lam,
            dclip=(0.0, args.dclip_max), eps=1e-12,
            max_iter=args.max_iter, tol=1e-4,
            block_size=args.block_size, seed=seed_r, verbose=args.verbose,
            gamma=args.gamma, pgd_steps=args.pgd_steps, alpha=args.alpha,
            seed_nodes=seed_nodes, eta_seed=(1.0 if args.seed_supervision else 0.0)
        )
        labs_arg = argmax_labels(H)
        labs_km = kmeans_norm_labels(H, args.k, seed=seed_r)
        eval_arg = evaluate_partition(labs_arg, A_proc)
        eval_km = evaluate_partition(labs_km, A_proc)
        if eval_km['modularity_mean'] >= eval_arg['modularity_mean']:
            best_local = (eval_km['modularity_mean'], {"labels": labs_km, "method": "kmeans_norm", "eval": eval_km, "D": [np.diag(D).tolist() for D in D_list], "history": history["obj"], "seed": seed_r})
        else:
            best_local = (eval_arg['modularity_mean'], {"labels": labs_arg, "method": "argmax", "eval": eval_arg, "D": [np.diag(D).tolist() for D in D_list], "history": history["obj"], "seed": seed_r})
        print(f" Restart {r+1} best method={best_local[1]['method']} mod_mean={best_local[0]:.4f}")
        if best_overall is None or best_local[0] > best_overall[0]:
            best_overall = best_local

    total_time = time.time() - t0
    if best_overall is None:
        print("No successful runs.")
        return

    best_mod, best_res = best_overall
    labels_best = best_res["labels"]
    print("\nSelected best run:", best_res["seed"], "method", best_res["method"], "mod_mean", best_mod)
    print("Cluster sizes:", np.bincount(labels_best))

    A_agg = sum(w[i]*A_proc[i] for i in range(len(A_proc)))
    G_agg = sparse_to_nx(A_agg)
    louv_labels = community_louvain.best_partition(G_agg)
    lab_vec = np.array([louv_labels.get(i,0) for i in range(len(node_ids))])
    louv_eval = evaluate_partition(lab_vec, A_proc)
    print("Louvain eval (processed A):", louv_eval)

    out = {
        "n": n,
        "k": args.k,
        "best_seed": best_res["seed"],
        "best_method": best_res["method"],
        "labels_symnmf": labels_best.tolist(),
        "labels_louvain": lab_vec.tolist(),
        "D_list": best_res["D"],
        "history": best_res["history"],
        "eval_symnmf": best_res["eval"],
        "eval_louvain": louv_eval,
        "w": w.tolist(),
        "files_used": [file1.name, file2.name],
        "runtime_seconds": total_time
    }
    out_path = Path(args.data_dir) / f"trail_4_k{args.k}.json"
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print("Saved result to", out_path)

if __name__ == "__main__":
    main()
