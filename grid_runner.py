#!/usr/bin/env python3
import argparse, subprocess, json, os, sys, time, csv
from itertools import product

DEFAULT_SCRIPT = "symm_nmf.py"
# Runs the python file over a small hyperparameter grid and collect summary stats.

def run_cmd(cmd):
    print("RUN:", " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def read_summary(save_dir, k):
    path = os.path.join(save_dir, f"summary_symnmf_k{k}.json")
    if not os.path.exists(path):
        return None
    with open(path,'r') as fh:
        j = json.load(fh)
    return j.get("summary", j)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", type=str, default=DEFAULT_SCRIPT, help="SymNMF runner script")
    p.add_argument("--adj_files", nargs='+', required=True)
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--save_root", type=str, default="grid_results")
    p.add_argument("--ks", type=str, default="10,20,50")
    p.add_argument("--mus", type=str, default="1e-6,1e-4,1e-2")
    p.add_argument("--norms", type=str, default="0,1")
    p.add_argument("--inits", type=str, default="louvain,random")
    p.add_argument("--weight_methods", type=str, default="none,inv_nnz,sqrt_inv_nnz,degree_inv,uniform,custom")
    p.add_argument("--d_clips", type=str, default="None,5,10")
    p.add_argument("--custom_weights", type=str, default=None)
    p.add_argument("--max_runs", type=int, default=None)
    args = p.parse_args()

    ks = [int(x) for x in args.ks.split(',') if x!='']
    mus = [float(x) for x in args.mus.split(',') if x!='']
    norms = [bool(int(x)) for x in args.norms.split(',') if x!='']
    inits = [x for x in args.inits.split(',') if x!='']
    weight_methods = [x for x in args.weight_methods.split(',') if x!='']
    d_clips_raw = [x for x in args.d_clips.split(',') if x!='']
    d_clips = []
    for v in d_clips_raw:
        if v == 'None':
            d_clips.append(None)
        else:
            d_clips.append(float(v))
    restarts = args.restarts

    os.makedirs(args.save_root, exist_ok=True)
    csv_path = os.path.join(args.save_root, "grid_results.csv")

    header = [
        "k","mu","normalize_H","init","weight_method","d_clip","save_dir",
        "modularity_mean_argmax_mean","modularity_mean_argmax_std",
        "stability_argmax_mean_nmi","stability_argmax_std_nmi",
        "conductance_mean_argmax_mean","conductance_mean_argmax_std",
        "linkpred_auc_first_layer_mean","linkpred_auc_first_layer_std",
        "runtime_sec","returncode"
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        exp_idx = 0
        for (k, mu, normalize_H, init, weight_method, d_clip) in product(ks, mus, norms, inits, weight_methods, d_clips):
            exp_idx += 1
            if args.max_runs is not None and exp_idx > args.max_runs:
                print("Reached max_runs", args.max_runs)
                break

            wm_tag = weight_method if weight_method is not None else "none"
            dc_tag = "None" if d_clip is None else str(d_clip)
            save_dir = os.path.join(args.save_root, f"k{k}_mu{mu}_norm{int(normalize_H)}_init{init}_w{wm_tag}_d{dc_tag}")
            os.makedirs(save_dir, exist_ok=True)

            cmd = [sys.executable, args.script, "--adj_files"] + args.adj_files + [
                   "--k", str(k),
                   "--restarts", str(restarts),
                   "--init", init,
                   "--mu", str(mu),
                   "--save_dir", save_dir,
                   "--verbose"
            ]
            if normalize_H:
                cmd.append("--normalize_H")
            if weight_method is not None and weight_method != 'none':
                cmd += ["--weight_method", weight_method]
            if d_clip is not None:
                cmd += ["--d_clip", str(d_clip)]
            if weight_method == 'custom':
                if args.custom_weights is None:
                    print("Skipping custom weight_method because --custom_weights not provided.")
                    continue
                cmd += ["--weights", args.custom_weights]

            t0 = time.time()
            rc, out, err = run_cmd(cmd)
            dt = time.time() - t0

            if out and out.strip():
                print(out)
            if err and err.strip():
                print("STDERR:", err, file=sys.stderr)

            summary = read_summary(save_dir, k)
            if summary is None:
                writer.writerow([k,mu,normalize_H,init,weight_method,d_clip,save_dir] + [""]*10 + [round(dt,2), rc])
                csvfile.flush()
                continue

            mm = summary.get("modularity_mean_argmax", {})
            cm = summary.get("conductance_mean_argmax", {})
            la = summary.get("linkpred_auc_first_layer", {})
            stab_mean = summary.get("stability_argmax_mean_nmi", None)
            stab_std = summary.get("stability_argmax_std_nmi", None)

            row = [
                k, mu, normalize_H, init, weight_method, d_clip, save_dir,
                mm.get("mean", ""), mm.get("std",""),
                stab_mean, stab_std,
                cm.get("mean",""), cm.get("std",""),
                la.get("mean",""), la.get("std",""),
                round(dt,2), rc
            ]
            writer.writerow(row)
            csvfile.flush()

    print("Grid finished. Results saved to", csv_path)

if __name__ == "__main__":
    main()
