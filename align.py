#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import scipy.sparse as sp

# python align.py --mode union --out_dir data_npz file1.txt file2.txt ...
def iter_edges(path):
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            yield u, v

def collect_node_sets(files):
    node_sets = []
    for p in files:
        s = set()
        for u,v in iter_edges(p):
            s.add(u); s.add(v)
        node_sets.append(s)
    return node_sets

def build_and_save(files, mode='union', out_dir='data_npz', make_undirected=True):
    os.makedirs(out_dir, exist_ok=True)
    node_sets = collect_node_sets(files)
    if mode == 'union':
        all_nodes = set().union(*node_sets)
    elif mode == 'intersection':
        all_nodes = set(node_sets[0])
        for s in node_sets[1:]:
            all_nodes &= s
    else:
        raise ValueError("mode must be 'union' or 'intersection'")

    all_nodes = sorted(all_nodes)
    n = len(all_nodes)
    print(f"Mode={mode}. total nodes = {n}")
    mapping = {node: i for i, node in enumerate(all_nodes)}

    with open(os.path.join(out_dir, 'node_mapping.json'), 'w', encoding='utf-8') as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)
    print("Saved node mapping to", os.path.join(out_dir, 'node_mapping.json'))

    for p in files:
        rows = []
        cols = []
        for u, v in iter_edges(p):
            if u not in mapping or v not in mapping:
                # skip edges with nodes excluded by intersection mode
                continue
            ui = mapping[u]; vi = mapping[v]
            if ui == vi:
                continue
            rows.append(ui); cols.append(vi)
            if make_undirected:
                rows.append(vi); cols.append(ui)
        if len(rows) == 0:
            A = sp.csr_matrix((n, n), dtype=np.float32)
        else:
            data = np.ones(len(rows), dtype=np.float32)
            A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
            A.sum_duplicates()
            A.data = np.minimum(A.data, 1.0)
            A = A.tocsr()
        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, base + '.npz')
        sp.save_npz(out_path, A)
        print(f"Saved {out_path} (shape={A.shape}, nnz={A.nnz})")

if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='edge list files (u v per line)')
    parser.add_argument('--mode', choices=['union','intersection'], default='union', help='union (default) or intersection of node sets')
    parser.add_argument('--out_dir', default='data_npz', help='where to save .npz and mapping')
    args = parser.parse_args()
    build_and_save(args.files, mode=args.mode, out_dir=args.out_dir)
