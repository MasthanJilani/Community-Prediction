#!/usr/bin/env python3
import os, json, glob, argparse
import numpy as np
import scipy.sparse as sp
import networkx as nx
import community as community_louvain
from collections import defaultdict

def load_runs_jsons(res_dir, k):
    pattern = os.path.join(res_dir, f"symnmf_k{k}_seed*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError("no files found: " + pattern)
    labels = []
    for f in files:
        j = json.load(open(f))
        if "labels_argmax" in j:
            labels.append(np.array(j["labels_argmax"], dtype=int))
        elif "labels_sym" in j:
            labels.append(np.array(j["labels_sym"], dtype=int))
        else:
            raise ValueError("file missing labels: " + f)
    return labels, files

def build_coassoc(labels_list):
    n = labels_list[0].shape[0]
    m = len(labels_list)
    C = np.zeros((n,n), dtype=float)
    for lab in labels_list:
        for c in np.unique(lab):
            nodes = np.where(lab==c)[0]
            if nodes.size == 0: continue
            idx = np.ix_(nodes, nodes)
            C[idx] += 1.0
    C = C / float(m)
    return C

def run_louvain_on_coassoc(C, threshold=0.0):
    n = C.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(np.triu(C, k=1) > threshold)
    edges = []
    for r,c in zip(rows, cols):
        edges.append((int(r), int(c), float(C[r,c])))
    G.add_weighted_edges_from(edges)
    partition = community_louvain.best_partition(G, weight='weight')
    labels = np.array([partition[i] for i in range(n)], dtype=int)
    return labels, G

def evaluate_labels_on_layers(labels, adj_paths):
    res = {}
    for p in adj_paths:
        A = sp.load_npz(p).tocsr()
        G = nx.from_scipy_sparse_array(A) if hasattr(nx, 'from_scipy_sparse_array') else nx.from_numpy_array(A.toarray())
        # building communities for modularity
        comms = []
        for c in np.unique(labels):
            nodes = set(np.where(labels==c)[0].tolist())
            if len(nodes) > 0:
                comms.append(nodes)
        mod = nx.community.quality.modularity(G, comms)
        res[os.path.basename(p)] = float(mod)
    return res

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, help="folder containing symnmf_k{k}_seed*.json")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--adj_files", nargs='+', required=True, help="original .npz adjacency files (same order used in runs)")
    p.add_argument("--threshold", type=float, default=0.0, help="threshold for coassoc edges (0.0 => keep all)")
    args = p.parse_args()

    labels_list, files = load_runs_jsons(args.results_dir, args.k)
    print(f"Loaded {len(labels_list)} runs from {args.results_dir}")

    C = build_coassoc(labels_list)
    print("Built co-association matrix (shape {})".format(C.shape))
    np.save(os.path.join(args.results_dir, f"coassoc_k{args.k}.npy"), C)
    print("Saved coassoc matrix.")

    cons_labels, Gc = run_louvain_on_coassoc(C, threshold=args.threshold)
    print("Consensus Louvain generated with {} communities".format(len(np.unique(cons_labels))))
    out = evaluate_labels_on_layers(cons_labels, args.adj_files)
    print("Consensus modularity per layer:")
    for k,v in out.items():
        print(f"  {k}: modularity = {v:.4f}")

    with open(os.path.join(args.results_dir, f"consensus_labels_k{args.k}.json"), 'w') as fh:
        json.dump({"labels": cons_labels.tolist(), "per_layer_modularity": out}, fh, indent=2)
    print("Saved consensus labels & eval.")
