#!/usr/bin/env python3
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
import networkx as nx
from itertools import combinations
try:
    import community as community_louvain
except Exception:
    community_louvain = None

EPS = 1e-12

def sparse_to_networkx(A, weighted=True):
    if sp.issparse(A):
        Acoo = A.tocoo()
        G = nx.Graph()
        G.add_nodes_from(range(A.shape[0]))
        if weighted and getattr(Acoo, "data", None) is not None:
            edges = list(zip(map(int, Acoo.row), map(int, Acoo.col), map(float, Acoo.data)))
            G.add_weighted_edges_from(edges)
        else:
            edges = list(zip(map(int, Acoo.row), map(int, Acoo.col)))
            G.add_edges_from(edges)
        return G
    else:
        arr = np.asarray(A)
        return nx.from_numpy_array(arr)

def load_adj(path):
    path = str(path)
    if path.endswith('.npz'):
        return sp.load_npz(path).tocsr()
    else:
        rows, cols = [], []
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip()=="" or line.startswith('#'): continue
                parts = line.split()
                if len(parts) < 2: continue
                u, v = parts[0], parts[1]
                try:
                    ui = int(u); vi = int(v)
                except:
                    ui = int(u); vi = int(v)
                rows.append(ui); cols.append(vi)
        if len(rows)==0:
            n = 0
        else:
            n = max(max(rows), max(cols)) + 1
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n,n))
        A = A + A.T
        if A.data.size > 0:
            A.data = np.minimum(A.data, 1.0)
        return A.tocsr()

def ensure_same_nodes(A_list):
    ns = [A.shape[0] for A in A_list]
    if len(set(ns)) != 1:
        raise ValueError("Adjacency matrices must have same number of nodes (n). Found: "+str(ns))
    return ns[0]

def compute_layer_weights(A_list, method='none', custom=None):
    if method is None or method == 'none':
        return None
    L = len(A_list)
    raw = []
    for A in A_list:
        nnz = A.nnz if sp.issparse(A) else int(np.count_nonzero(A))
        mean_deg = (2.0 * (nnz/2.0) / A.shape[0]) if A.shape[0]>0 else 0.0
        if method == 'inv_nnz':
            raw.append(1.0 / max(1, nnz))
        elif method == 'sqrt_inv_nnz':
            raw.append(1.0 / np.sqrt(max(1, nnz)))
        elif method == 'degree_inv':
            raw.append(1.0 / max(1e-6, mean_deg))
        elif method == 'uniform':
            raw.append(1.0)
        elif method == 'custom':
            if custom is None:
                raise ValueError("custom weights requested but none provided")
            raw = list(custom)
            break
        else:
            raise ValueError("unknown weight method: "+str(method))
    raw = np.array(raw, dtype=float)
    # normalize so sum(weights) == L
    if raw.sum() == 0:
        weights = np.ones_like(raw)
    else:
        weights = raw * (len(raw) / raw.sum())
    return weights.tolist()

def louvain_init(Hshape, A):
    if community_louvain is None:
        raise ImportError("python-louvain required for louvain_init")
    G = sparse_to_networkx(A)
    partition = community_louvain.best_partition(G)
    labels = [partition.get(i, 0) for i in range(len(partition))]
    unique = sorted(list(set(labels)))
    mapping = {c:i for i,c in enumerate(unique)}
    H0 = np.zeros((len(labels), Hshape[1]))
    for i,l in enumerate(labels):
        idx = mapping[l]
        if idx < Hshape[1]:
            H0[i, idx] = 1.0
    for c in range(Hshape[1]):
        if H0[:,c].sum() == 0:
            H0[:,c] = np.abs(0.1 * np.random.randn(Hshape[0]))
    return H0

def compute_Ds_from_H(H, A_list, reg=1e-6, d_clip=None):
    k = H.shape[1]
    HH = H.T.dot(H)
    M = (HH ** 2)
    M_reg = M + reg * np.eye(k)
    Ds = []
    for A in A_list:
        AH = A.dot(H)
        b = np.sum(H * AH, axis=0)
        try:
            d = np.linalg.solve(M_reg, b)
        except np.linalg.LinAlgError:
            d, *_ = np.linalg.lstsq(M_reg, b, rcond=None)
        d = np.maximum(d, 0.0)
        if d_clip is not None and d_clip > 0:
            d = np.minimum(d, float(d_clip))
        Ds.append(d)
    return Ds

def symnmf_train(A_list, k, init='random', mu=1e-6, max_iter=500, tol=1e-4, normalize_H=False, verbose=False, d_clip=None):
    n = ensure_same_nodes(A_list)
    if init == 'random':
        H = np.abs(np.random.randn(n, k)).astype(float) + 1e-6
    elif init == 'louvain':
        H = louvain_init((n,k), A_list[0])
    else:
        raise ValueError("init must be 'random' or 'louvain'")
    H = np.maximum(H, 1e-8)
    if normalize_H:
        row_sums = H.sum(axis=1, keepdims=True) + EPS
        H = H / row_sums

    prev_obj = None
    history = []
    for it in range(max_iter):
        Ds = compute_Ds_from_H(H, A_list, reg=1e-8, d_clip=d_clip)
        numer = np.zeros_like(H)
        denom = np.zeros_like(H)
        HTH = H.T.dot(H)
        for d, A in zip(Ds, A_list):
            Ddiag = np.diag(d)
            numer += A.dot(H.dot(Ddiag))
            S = Ddiag.dot(HTH.dot(Ddiag))
            denom += H.dot(S)
        denom += mu * H
        ratio = numer / (denom + EPS)
        H = H * ratio
        H = np.maximum(H, 1e-12)
        if normalize_H:
            row_sums = H.sum(axis=1, keepdims=True) + EPS
            H = H / row_sums

        # memory-efficient objective
        obj = 0.0
        for d, A in zip(Ds, A_list):
            # HD = H * d[np.newaxis, :]       # elementwise scale columns (n x k)
            # R = HD.dot(H.T)                 # dense n x n  <-- expensive!
            M = H.T.dot(H)
            AH = A.dot(H)
            B = H.T.dot(AH)
            diagB = np.diag(B)
            traceAR = float(np.sum(diagB * d))
            DM = (d[:, None] * M)
            normR2 = float(np.trace(DM.dot(DM)))
            if sp.issparse(A):
                normA2 = float((A.data ** 2).sum()) if getattr(A, "data", None) is not None else 0.0
            else:
                normA2 = float(np.sum(A * A))

                # actual implementation (very expensive!)
            # if sp.issparse(A):
            #     traceAR = float((A.multiply(R)).sum())
            # else:
            #     traceAR = float(np.sum(A * R))
            # normR2 = float(np.sum(R * R))
            # obj += (normA2 + normR2 - 2.0 * traceAR)
            obj += (normA2 + normR2 - 2.0 * traceAR)

        obj += mu * np.sum(H*H)

        if not np.isfinite(obj):
            if verbose: print(f"[symnmf_train] Iter {it}: objective not finite ({obj}). Stopping.")
            history.append(float('nan'))
            break
        if np.any(~np.isfinite(H)):
            if verbose: print(f"[symnmf_train] Iter {it}: H contains non-finite. Stopping.")
            history.append(float('nan'))
            break

        history.append(float(obj))
        if verbose and (it % 50 == 0 or it == max_iter - 1):
            print(f"iter {it:03d} obj {obj:.6e}")

        if prev_obj is not None and np.isfinite(prev_obj):
            relchg = abs(prev_obj - obj) / (abs(prev_obj) + EPS)
            if relchg < tol:
                if verbose: print(f"[symnmf_train] converged iter {it} relchg={relchg:.3e}")
                break
        prev_obj = float(obj)
    Ds = compute_Ds_from_H(H, A_list, reg=1e-8, d_clip=d_clip)
    return H, Ds, history

def labels_from_H(H, method='argmax'):
    if method == 'argmax':
        return np.argmax(H, axis=1)
    elif method == 'kmeans':
        km = KMeans(n_clusters=H.shape[1], n_init=10, random_state=0).fit(H)
        return km.labels_
    else:
        raise ValueError()

def modularity_per_layer(labels, A_list):
    mods = []
    for A in A_list:
        G = sparse_to_networkx(A)
        communities = []
        for c in np.unique(labels):
            nodes = list(np.where(labels==c)[0])
            if len(nodes)>0:
                communities.append(set(nodes))
        try:
            m = nx.community.quality.modularity(G, communities)
        except Exception:
            m = float('nan')
        mods.append(float(m))
    return mods, float(np.nanmean(mods))

def conductance_for_partition(A, labels):
    G = sparse_to_networkx(A)
    vol = dict(G.degree())
    comms = [set(np.where(labels==c)[0]) for c in np.unique(labels)]
    conds = []
    for S in comms:
        if len(S) == 0 or len(S) == G.number_of_nodes():
            continue
        cut = 0
        for u in S:
            for v in G[u]:
                if v not in S:
                    cut += 1
        volS = sum(vol[n] for n in S)
        volT = sum(vol[n] for n in set(G.nodes())-S)
        denom = min(volS, volT) if min(volS,volT)>0 else 1.0
        conds.append(cut/denom)
    return float(np.nanmean(conds)) if conds else float('nan')

def conductance_per_layer(labels, A_list):
    conds = []
    for A in A_list:
        conds.append(conductance_for_partition(A, labels))
    return conds, float(np.nanmean(conds))

def pairwise_nmi(labels_list):
    pairs = list(combinations(range(len(labels_list)), 2))
    if not pairs: return float('nan'), float('nan')
    vals = [normalized_mutual_info_score(labels_list[i], labels_list[j]) for i,j in pairs]
    return float(np.mean(vals)), float(np.std(vals))

def link_prediction_auc_from_H(H, A, holdout_frac=0.1, num_neg=None, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    Acoo = sp.triu(A, k=1).tocoo()
    edges = list(zip(Acoo.row.tolist(), Acoo.col.tolist()))
    m = len(edges)
    if m == 0: return float('nan')
    num_pos = max(1, int(m * holdout_frac))
    pos_idx = rng.choice(range(m), size=num_pos, replace=False)
    pos = [edges[i] for i in pos_idx]
    if num_neg is None: num_neg = num_pos
    n = A.shape[0]
    neg = []
    while len(neg) < num_neg:
        u = int(rng.randint(n)); v = int(rng.randint(n))
        if u==v: continue
        if A[u, v] != 0 or A[v, u] != 0: continue
        neg.append((u,v))
    pairs = pos + neg
    y = [1]*len(pos) + [0]*len(neg)
    scores = [float(H[u].dot(H[v])) for (u,v) in pairs]
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return float('nan')

def run_experiment(adj_files, k=20, restarts=10, init='random', mu=1e-6, max_iter=500,
                   normalize_H=False, holdout_frac=0.1, save_dir='results', verbose=False,
                   weight_method='none', custom_weights=None, d_clip=None):
    A_list = [load_adj(p) for p in adj_files]
    n = ensure_same_nodes(A_list)
    weights = compute_layer_weights(A_list, method=weight_method, custom=custom_weights)
    if verbose:
        print("Layer weights:", weights)
    if weights is None:
        A_list_weighted = A_list
    else:
        A_list_weighted = []
        for w, A in zip(weights, A_list):
            if w == 1.0:
                A_list_weighted.append(A.copy())
            else:
                A_list_weighted.append(A.multiply(float(w)).tocsr())

    os.makedirs(save_dir, exist_ok=True)

    all_runs = []
    rng = np.random.RandomState(0)
    for seed in range(restarts):
        if verbose: print(f"Restart {seed+1}/{restarts}")
        np.random.seed(seed)
        A_train_list = []
        for A in A_list_weighted:
            if holdout_frac is None or holdout_frac<=0:
                A_train_list.append(A.copy())
            else:
                Acoo = sp.triu(A, k=1).tocoo()
                edges = list(zip(Acoo.row, Acoo.col))
                m = len(edges)
                num_hold = int(max(1, m*holdout_frac))
                if m <= 1:
                    A_train_list.append(A.copy()); continue
                hold_idx = rng.choice(range(m), size=num_hold, replace=False)
                hold_set = set(hold_idx)
                rows=[]; cols=[]; data=[]
                for i,(u,v) in enumerate(edges):
                    if i in hold_set: continue
                    rows.append(u); cols.append(v); data.append(1)
                    rows.append(v); cols.append(u); data.append(1)
                if len(rows)==0:
                    A_new = sp.csr_matrix(A.shape, dtype=np.float32)
                else:
                    A_new = sp.csr_matrix((data,(rows,cols)), shape=A.shape)
                A_train_list.append(A_new)
        try:
            H_init = np.abs(np.random.randn(n,k)) if init=='random' else (louvain_init((n,k), A_train_list[0]) if community_louvain is not None else np.abs(np.random.randn(n,k)))
        except Exception:
            H_init = np.abs(np.random.randn(n,k))
        H, Ds, history = symnmf_train(A_train_list, k, init=init, mu=mu, max_iter=max_iter, tol=1e-4, normalize_H=normalize_H, verbose=verbose, d_clip=d_clip)
        labels_arg = labels_from_H(H, method='argmax')
        labels_km = labels_from_H(H, method='kmeans')
        mods_arg, mods_mean_arg = modularity_per_layer(labels_arg, A_list)
        mods_km, mods_mean_km = modularity_per_layer(labels_km, A_list)
        conds_arg, cond_mean_arg = conductance_per_layer(labels_arg, A_list)
        conds_km, cond_mean_km = conductance_per_layer(labels_km, A_list)
        lp_auc = link_prediction_auc_from_H(H, A_list[0], holdout_frac=holdout_frac, rng=rng) if holdout_frac and holdout_frac>0 else float('nan')
        run_info = {
            "seed": int(seed),
            "weights_used": weights,
            "labels_argmax": labels_arg.tolist(),
            "labels_kmeans": labels_km.tolist(),
            "modularity_per_layer_argmax": mods_arg,
            "modularity_mean_argmax": mods_mean_arg,
            "modularity_per_layer_kmeans": mods_km,
            "modularity_mean_kmeans": mods_mean_km,
            "conductance_per_layer_argmax": conds_arg,
            "conductance_mean_argmax": cond_mean_arg,
            "conductance_per_layer_kmeans": conds_km,
            "conductance_mean_kmeans": cond_mean_km,
            "linkpred_auc_first_layer": lp_auc,
            "Ds": [d.tolist() for d in Ds],
            "history": history,
            "H_norm": float(np.linalg.norm(H)),
        }
        fname = os.path.join(save_dir, f"symnmf_k{k}_seed{seed}.json")
        with open(fname, 'w') as fh:
            json.dump(run_info, fh, indent=2)
        all_runs.append(run_info)
        if verbose: print(f"Saved run to {fname}")

    def gather(key):
        vals=[]
        for r in all_runs:
            v = r.get(key)
            if v is None: continue
            if isinstance(v, list):
                try: vals.append(float(np.mean(v)))
                except: pass
            else:
                try: vals.append(float(v))
                except: pass
        vals = np.array(vals, dtype=float) if len(vals)>0 else np.array([])
        if vals.size==0:
            return {"mean": None, "std": None}
        return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    summary = {
        "k": k,
        "num_restarts": restarts,
        "weight_method": weight_method,
        "weights": weights,
        "modularity_mean_argmax": gather("modularity_mean_argmax"),
        "modularity_mean_kmeans": gather("modularity_mean_kmeans"),
        "conductance_mean_argmax": gather("conductance_mean_argmax"),
        "conductance_mean_kmeans": gather("conductance_mean_kmeans"),
        "linkpred_auc_first_layer": gather("linkpred_auc_first_layer"),
    }
    labs_arg_all = [np.array(r['labels_argmax']) for r in all_runs]
    labs_km_all = [np.array(r['labels_kmeans']) for r in all_runs]
    summary["stability_argmax_mean_nmi"], summary["stability_argmax_std_nmi"] = pairwise_nmi(labs_arg_all)
    summary["stability_kmeans_mean_nmi"], summary["stability_kmeans_std_nmi"] = pairwise_nmi(labs_km_all)

    sum_path = os.path.join(save_dir, f"summary_symnmf_k{k}.json")
    with open(sum_path, 'w') as fh:
        json.dump({"summary": summary, "runs": [ {"seed": int(r["seed"]), "modularity_mean_argmax": r.get("modularity_mean_argmax", None)} for r in all_runs]}, fh, indent=2)
    if verbose: print("Saved summary to", sum_path)
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adj_files", nargs='+', required=True)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--init", choices=['random','louvain'], default='louvain')
    p.add_argument("--mu", type=float, default=1e-6)
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--normalize_H", action='store_true')
    p.add_argument("--holdout_frac", type=float, default=0.1)
    p.add_argument("--save_dir", type=str, default="results_weighted")
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--weight_method", type=str, default="none", choices=['none','inv_nnz','sqrt_inv_nnz','degree_inv','uniform','custom'])
    p.add_argument("--weights", type=str, default=None, help="comma-separated custom weights when --weight_method custom")
    p.add_argument("--d_clip", type=float, default=None, help="optional clip for D entries (e.g. 10.0)")
    args = p.parse_args()

    custom_weights = None
    if args.weight_method == 'custom':
        if args.weights is None:
            raise ValueError("Please pass --weights 'w1,w2,...' when --weight_method custom")
        custom_weights = [float(x) for x in args.weights.split(',')]

    run_experiment(args.adj_files, k=args.k, restarts=args.restarts, init=args.init, mu=args.mu, max_iter=args.max_iter,
                   normalize_H=args.normalize_H, holdout_frac=args.holdout_frac, save_dir=args.save_dir, verbose=args.verbose,
                   weight_method=args.weight_method, custom_weights=custom_weights, d_clip=args.d_clip)
