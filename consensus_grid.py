#!/usr/bin/env python3
import argparse, os, glob, json, csv, math, random, time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import networkx as nx
import community as community_louvain
from itertools import combinations

# Run this file ONLY IF RAM >= 32 GB!!
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

def load_run_labels(save_dir, k):
    pattern = os.path.join(save_dir, f"symnmf_k{k}_seed*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    labels_list = []
    for f in files:
        j = json.load(open(f, 'r'))
        if "labels_argmax" in j:
            labels_list.append(np.array(j["labels_argmax"], dtype=int))
        elif "labels_kmeans" in j:
            labels_list.append(np.array(j["labels_kmeans"], dtype=int))
    return labels_list

def build_coassoc_dense(labels_list):
    m = len(labels_list)
    n = labels_list[0].shape[0]
    C = np.zeros((n,n), dtype=np.float32)
    for lab in labels_list:
        # build indicator matrix U of shape (n, k_run)
        uniq = np.unique(lab)
        k_run = len(uniq)
        # map labels to 0..k_run-1
        mapping = {c:i for i,c in enumerate(uniq)}
        U = np.zeros((n, k_run), dtype=np.float32)
        for i,c in enumerate(lab):
            U[i, mapping[c]] = 1.0
        C += U.dot(U.T)
    C = C / float(m)
    return C

def build_coassoc_sparse(labels_list, max_exact=2000, max_sample_pairs=200000):
    n = labels_list[0].shape[0]
    num_runs = len(labels_list)
    rows = []
    cols = []
    data = []
    t0 = time.time()
    total_pairs = 0
    for run_idx, lab in enumerate(labels_list):
        uniq = np.unique(lab)
        for c in uniq:
            nodes = np.where(lab == c)[0]
            m = len(nodes)
            if m <= 1:
                continue
            if m <= max_exact:
                idx0, idx1 = np.triu_indices(m, k=1)
                if idx0.size > 0:
                    r = nodes[idx0]; c_ = nodes[idx1]
                    rows.append(r); cols.append(c_); data.append(np.ones_like(r, dtype=np.int32))
                    total_pairs += r.size
            else:
                # sample pairs to limit memory
                S = min(max_sample_pairs, m * (m-1) // 2)
                if S <= 0:
                    continue
                u = np.random.randint(0, m, size=S)
                v = np.random.randint(0, m, size=S)
                mask = u != v
                u = u[mask]; v = v[mask]
                if u.size == 0: continue
                a = np.minimum(u, v); b = np.maximum(u, v)
                r = nodes[a]; c_ = nodes[b]
                rows.append(r); cols.append(c_); data.append(np.ones_like(r, dtype=np.int32))
                total_pairs += r.size
    if len(rows) == 0:
        return sp.coo_matrix((n,n), dtype=np.float32)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    rows_full = np.concatenate([rows, cols])
    cols_full = np.concatenate([cols, rows])
    data_full = np.concatenate([data, data])
    Ccoo = sp.coo_matrix((data_full, (rows_full, cols_full)), shape=(n,n), dtype=np.float32)
    Ccoo.sum_duplicates()
    Ccoo = Ccoo.tocsr()
    Ccoo = Ccoo.astype(np.float32) / float(len(labels_list))
    elapsed = time.time() - t0
    print(f"[build_coassoc_sparse] done. approx pairs added: {data_full.size}, time {elapsed:.1f}s")
    return Ccoo

def run_louvain_on_sparse_coassoc(C_sparse, threshold=0.1):
    C = C_sparse.tocoo()
    mask = C.data >= threshold
    if mask.sum() == 0:
        mask = C.data >= 0.0
    rows = C.row[mask]; cols = C.col[mask]; vals = C.data[mask]
    G = nx.Graph()
    n = C_sparse.shape[0]
    G.add_nodes_from(range(n))
    edges = [(int(r), int(c), float(w)) for r,c,w in zip(rows, cols, vals) if r != c]
    G.add_weighted_edges_from(edges)
    if G.number_of_edges() == 0:
        part = {i:0 for i in range(n)}
    else:
        part = community_louvain.best_partition(G, weight='weight')
    labels = np.array([part.get(i, 0) for i in range(n)], dtype=int)
    return labels, G

def eval_modularity_per_layers(labels, adj_files):
    out = {}
    for p in adj_files:
        A = sp.load_npz(p).tocsr()
        G = sparse_to_networkx(A)
        comms = []
        for c in np.unique(labels):
            nodes = list(np.where(labels==c)[0])
            if len(nodes)>0:
                comms.append(set(nodes))
        try:
            m = nx.community.quality.modularity(G, comms)
        except Exception:
            m = float('nan')
        out[os.path.basename(p)] = float(m)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_csv", required=True, help="path to grid_results.csv")
    parser.add_argument("--k", type=int, required=True, help="k used in runs")
    parser.add_argument("--adj_files", nargs='+', required=True, help="original .npz adjacency files")
    parser.add_argument("--mode", choices=['sparse','dense'], default='sparse')
    parser.add_argument("--thresholds", nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.5])
    parser.add_argument("--max_exact", type=int, default=2000, help="max cluster size for exact pair expansion")
    parser.add_argument("--max_sample_pairs", type=int, default=200000, help="max sampled pairs for large clusters")
    parser.add_argument("--out_csv", default="grid_consensus_results.csv")
    args = parser.parse_args()

    rows = []
    with open(args.grid_csv, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                rk = int(float(r.get('k', r.get('K', 0))))
            except:
                rk = None
            if rk == args.k:
                sd = r.get('save_dir') or r.get('save_dir'.lower())
                if sd and os.path.isdir(sd):
                    rows.append((r, sd))
                else:
                    sd2 = os.path.join(os.path.dirname(args.grid_csv), sd) if sd else None
                    if sd2 and os.path.isdir(sd2):
                        rows.append((r, sd2))
                    else:
                        print(f"[warn] save_dir not found or missing for row: {r}")

    if not rows:
        raise SystemExit("No matching rows with k={}. Check grid CSV and save_dir paths.".format(args.k))

    with open(args.out_csv, 'w', newline='') as outf:
        writer = csv.writer(outf)
        header = ["save_dir","mode","threshold","num_runs","n_nodes","num_edges_in_consensus_graph","per_layer_modularity","runtime_sec"]
        writer.writerow(header)

        for rowdict, save_dir in rows:
            print("Processing", save_dir)
            labels_list = load_run_labels(save_dir, args.k)
            if not labels_list:
                print("  no run JSONs found in", save_dir)
                continue
            n = labels_list[0].shape[0]
            t0_all = time.time()
            if args.mode == 'dense':
                print("  Building dense coassociation (memory heavy)...")
                C = build_coassoc_dense(labels_list)  # numpy dense
                Csp = sp.csr_matrix(C)
            else:
                print(f"  Building sparse coassociation (max_exact={args.max_exact}, max_sample_pairs={args.max_sample_pairs})")
                Csp = build_coassoc_sparse(labels_list, max_exact=args.max_exact, max_sample_pairs=args.max_sample_pairs)
            for thresh in args.thresholds:
                t0 = time.time()
                labels_cons, Gc = run_louvain_on_sparse_coassoc(Csp, threshold=thresh)
                evals = eval_modularity_per_layers(labels_cons, args.adj_files)
                num_edges = Gc.number_of_edges() if hasattr(Gc, "number_of_edges") else 0
                runtime = time.time() - t0
                print(f"  thresh={thresh:.3f} -> communities={len(np.unique(labels_cons))}, edges={num_edges}, runtime {runtime:.1f}s")
                outpath = os.path.join(save_dir, f"consensus_labels_k{args.k}_th{thresh:.2f}.json")
                with open(outpath, 'w') as fh:
                    json.dump({"labels": labels_cons.tolist(), "per_layer_modularity": evals}, fh, indent=2)
                writer.writerow([save_dir, args.mode, thresh, len(labels_list), n, num_edges, json.dumps(evals), runtime])
                outf.flush()
            print("  Done folder; total time: {:.1f}s".format(time.time() - t0_all))

    print("All done. Results saved to", args.out_csv)

if __name__ == "__main__":
    main()
