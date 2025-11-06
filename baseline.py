import scipy.sparse as sp
import networkx as nx
import community as community_louvain
import numpy as np
import sys

files = ["data_npz/email-Enron.npz","data_npz/ca-HepPh.npz","data_npz/ca-GrQc.npz"]
for f in files:
    A = sp.load_npz(f).tocsr()
    G = nx.Graph()
    Acoo = sp.triu(A, k=1).tocoo()
    edges = list(zip(map(int, Acoo.row), map(int, Acoo.col)))
    G.add_nodes_from(range(A.shape[0]))
    G.add_edges_from(edges)
    part = community_louvain.best_partition(G)
    from collections import defaultdict
    comms = defaultdict(list)
    for n, c in part.items():
        comms[c].append(n)
    communities = [set(v) for v in comms.values()]
    m = nx.community.quality.modularity(G, communities)
    print(f"{f}: nodes={A.shape[0]}, edges={A.nnz//2}, louvain_modularity={m:.4f}")
