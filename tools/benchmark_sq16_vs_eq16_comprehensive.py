#!/usr/bin/env python3
"""
benchmark_sq16_vs_eq16_comprehensive.py - Comprehensive benchmark comparing
SQ16 (16-bit Cubic Scalar Quantization) vs EQ16 (16-bit E8 Block Lattice Quantization)
for both gric-cluster and gric-knn.
"""

import os
import sys
import re
import time
import subprocess
import numpy as np

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
TMP_DIR = "/tmp/bench_sq16_eq16"
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace"))

os.makedirs(TMP_DIR, exist_ok=True)

def run_cmd(cmd):
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if p.returncode != 0:
        print(f"Error running: {cmd}\nStderr: {p.stderr}", file=sys.stderr)
    return p.stdout, p.stderr, p.returncode, wall_ms

def parse_cluster_output(stdout, wall_ms):
    stats = {
        "wall_ms": wall_ms,
        "proc_ms": 0.0,
        "fps": 0.0,
        "clusters": 0,
        "framedists": 0,
        "sample_dists": 0,
        "inter_dists": 0,
        "pruned": 0,
        "step3a_ms": 0.0,
        "quant_pruned": 0
    }
    m_proc = re.search(r"Processing time:\s+([\d\.]+)\s+ms", stdout)
    if m_proc:
        stats["proc_ms"] = float(m_proc.group(1))
    m_cl = re.search(r"Total clusters:\s+(\d+)", stdout)
    if m_cl:
        stats["clusters"] = int(m_cl.group(1))
    m_fd = re.search(r"Framedist calls:\s+(\d+)\s+\(sample-to-cluster:\s+(\d+),\s+inter-cluster:\s+(\d+)\)", stdout)
    if m_fd:
        stats["framedists"] = int(m_fd.group(1))
        stats["sample_dists"] = int(m_fd.group(2))
        stats["inter_dists"] = int(m_fd.group(3))
    m_sq_diag = re.search(r"SQ16 Pruned:\s+(\d+)", stdout)
    if m_sq_diag:
        stats["quant_pruned"] = int(m_sq_diag.group(1))
    m_eq_diag = re.search(r"EQ16 Pruned:\s+(\d+)", stdout)
    if m_eq_diag:
        stats["quant_pruned"] = int(m_eq_diag.group(1))
    m_pruned = re.search(r"pruned away:\s+(\d+)", stdout)
    if m_pruned:
        stats["pruned"] = int(m_pruned.group(1))
    m_3a = re.search(r"Step 3a \(Priors/Prune\):\s+([\d\.]+)\s+ms", stdout)
    if m_3a:
        stats["step3a_ms"] = float(m_3a.group(1))
    return stats

def parse_knn_output(stdout, wall_ms):
    stats = {
        "wall_ms": wall_ms,
        "search_ms": 0.0,
        "fps": 0.0,
        "framedists": 0,
        "prune_pct": 0.0,
        "l2_anchors_pruned": 0,
        "l3_annular_pruned": 0,
        "quant_evals": 0,
        "quant_pruned": 0
    }
    m_search = re.search(r"Search Wall Time:\s+([\d\.]+)\s+ms\s+\(([\d\.]+)\s+fps\)", stdout)
    if m_search:
        stats["search_ms"] = float(m_search.group(1))
        stats["fps"] = float(m_search.group(2))
    m_fd = re.search(r"Framedist Computations:\s+(\d+)", stdout)
    if m_fd:
        stats["framedists"] = int(m_fd.group(1))
    m_pct = re.search(r"Metric Pruning Efficiency:\s+([\d\.]+)%", stdout)
    if m_pct:
        stats["prune_pct"] = float(m_pct.group(1))
    m_l2 = re.search(r"Level 2 Anchors Pruned:\s+(\d+)", stdout)
    if m_l2:
        stats["l2_anchors_pruned"] = int(m_l2.group(1))
    m_l3 = re.search(r"Level 3 Annular Pruned:\s+(\d+)", stdout)
    if m_l3:
        stats["l3_annular_pruned"] = int(m_l3.group(1))
    m_sq_ev = re.search(r"SQ16 Evaluations:\s+(\d+)", stdout)
    if m_sq_ev:
        stats["quant_evals"] = int(m_sq_ev.group(1))
    m_eq_ev = re.search(r"EQ16 Evaluations:\s+(\d+)", stdout)
    if m_eq_ev:
        stats["quant_evals"] = int(m_eq_ev.group(1))
    m_sq_pr = re.search(r"SQ16 Members Pruned:\s+(\d+)", stdout)
    if m_sq_pr:
        stats["quant_pruned"] += int(m_sq_pr.group(1))
    m_eq_pr = re.search(r"EQ16 Member Pruned:\s+(\d+)", stdout)
    if m_eq_pr:
        stats["quant_pruned"] += int(m_eq_pr.group(1))
    return stats

def load_knn_results(path):
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                qid = int(parts[0])
                n_ids = [int(parts[i]) for i in range(1, len(parts), 2)]
                results[qid] = n_ids
    return results

def compute_recall(gt_dict, cand_dict, k=10):
    if not gt_dict or not cand_dict:
        return 0.0
    common = set(gt_dict.keys()).intersection(set(cand_dict.keys()))
    if not common:
        return 0.0
    total_match = 0
    total_needed = 0
    for q in common:
        s_gt = set(gt_dict[q][:k])
        s_cand = set(cand_dict[q][:k])
        total_match += len(s_gt.intersection(s_cand))
        total_needed += len(s_gt)
    return (total_match / total_needed * 100.0) if total_needed > 0 else 0.0

def run_experiment(name, dataset_path, dim, n_samples, rlim, k=10):
    print(f"\n================================================================================")
    print(f" EXPERIMENT: {name} (D = {dim}, N = {n_samples}, k = {k}, rlim = {rlim:.4f})")
    print(f"================================================================================")

    cl_sq16_dir = f"{TMP_DIR}/cl_sq16_{dim}D"
    cl_eq16_dir = f"{TMP_DIR}/cl_eq16_{dim}D"
    gt_out = f"{TMP_DIR}/knn_gt_{dim}D.txt"
    sq16_out = f"{TMP_DIR}/knn_sq16_{dim}D.txt"
    eq16_out = f"{TMP_DIR}/knn_eq16_{dim}D.txt"

    # A. Clustering with SQ16
    print(f"  [1/4] Clustering with SQ16...")
    cmd_cl_sq = f"{BUILD_DIR}/gric-cluster {rlim:.4f} {dataset_path} -maxim {n_samples} -sq16 -outdir {cl_sq16_dir}"
    out_sq, _, _, wall_cl_sq = run_cmd(cmd_cl_sq)
    res_cl_sq = parse_cluster_output(out_sq, wall_cl_sq)

    # B. Clustering with EQ16
    print(f"  [2/4] Clustering with EQ16...")
    cmd_cl_eq = f"{BUILD_DIR}/gric-cluster {rlim:.4f} {dataset_path} -maxim {n_samples} -eq16 -outdir {cl_eq16_dir}"
    out_eq, _, _, wall_cl_eq = run_cmd(cmd_cl_eq)
    res_cl_eq = parse_cluster_output(out_eq, wall_cl_eq)

    # C. Ground Truth exact search
    print(f"  [3/4] Computing Ground Truth exact k-NN...")
    cmd_gt = f"{BUILD_DIR}/gric-knn {dataset_path} {cl_eq16_dir} -k {k} -no-sq16 -no-eq16 -o {gt_out}"
    run_cmd(cmd_gt)
    gt_results = load_knn_results(gt_out)

    # D. k-NN Search with SQ16
    print(f"  [4/4] Running k-NN with SQ16 vs EQ16...")
    cmd_knn_sq = f"{BUILD_DIR}/gric-knn {dataset_path} {cl_sq16_dir} -k {k} -sq16 -o {sq16_out}"
    out_knn_sq, _, _, wall_knn_sq = run_cmd(cmd_knn_sq)
    res_knn_sq = parse_knn_output(out_knn_sq, wall_knn_sq)
    knn_sq_results = load_knn_results(sq16_out)
    recall_sq = compute_recall(gt_results, knn_sq_results, k)

    cmd_knn_eq = f"{BUILD_DIR}/gric-knn {dataset_path} {cl_eq16_dir} -k {k} -eq16 -o {eq16_out}"
    out_knn_eq, _, _, wall_knn_eq = run_cmd(cmd_knn_eq)
    res_knn_eq = parse_knn_output(out_knn_eq, wall_knn_eq)
    knn_eq_results = load_knn_results(eq16_out)
    recall_eq = compute_recall(gt_results, knn_eq_results, k)

    # Print clean report table
    print(f"\n  --- CLUSTERING PERFORMANCE ---")
    print(f"  {'Metric':<28} | {'SQ16 (Cubic)':>18} | {'EQ16 (E8 Lattice)':>18} | {'Delta / Advantage':>18}")
    print(f"  {'-'*28}-|-{'-'*18}-|-{'-'*18}-|-{'-'*18}")
    cl_t_diff = res_cl_eq['proc_ms'] - res_cl_sq['proc_ms']
    cl_t_pct = (cl_t_diff / res_cl_sq['proc_ms'] * 100.0) if res_cl_sq['proc_ms'] > 0 else 0.0
    print(f"  {'Processing Time (Core)':<28} | {res_cl_sq['proc_ms']:>15.2f} ms | {res_cl_eq['proc_ms']:>15.2f} ms | {cl_t_diff:>+14.2f} ms ({cl_t_pct:+.1f}%)")
    print(f"  {'Wall Time (Total)':<28} | {res_cl_sq['wall_ms']:>15.2f} ms | {res_cl_eq['wall_ms']:>15.2f} ms | {res_cl_eq['wall_ms'] - res_cl_sq['wall_ms']:>+14.2f} ms")
    print(f"  {'Clusters Created (M)':<28} | {res_cl_sq['clusters']:>18} | {res_cl_eq['clusters']:>18} | {res_cl_eq['clusters'] - res_cl_sq['clusters']:>+18}")
    print(f"  {'Total Framedists':<28} | {res_cl_sq['framedists']:>18} | {res_cl_eq['framedists']:>18} | {res_cl_eq['framedists'] - res_cl_sq['framedists']:>+18}")
    print(f"  {'  - Sample-to-Cluster':<28} | {res_cl_sq['sample_dists']:>18} | {res_cl_eq['sample_dists']:>18} | {res_cl_eq['sample_dists'] - res_cl_sq['sample_dists']:>+18}")
    print(f"  {'  - Inter-Cluster':<28} | {res_cl_sq['inter_dists']:>18} | {res_cl_eq['inter_dists']:>18} | {res_cl_eq['inter_dists'] - res_cl_sq['inter_dists']:>+18}")
    print(f"  {'Quant Candidates Pruned':<28} | {res_cl_sq['quant_pruned']:>18} | {res_cl_eq['quant_pruned']:>18} | {res_cl_eq['quant_pruned'] - res_cl_sq['quant_pruned']:>+18}")

    print(f"\n  --- k-NN SEARCH PERFORMANCE ---")
    print(f"  {'Metric':<28} | {'SQ16 (Cubic)':>18} | {'EQ16 (E8 Lattice)':>18} | {'Delta / Advantage':>18}")
    print(f"  {'-'*28}-|-{'-'*18}-|-{'-'*18}-|-{'-'*18}")
    knn_t_diff = res_knn_eq['search_ms'] - res_knn_sq['search_ms']
    knn_t_pct = (knn_t_diff / res_knn_sq['search_ms'] * 100.0) if res_knn_sq['search_ms'] > 0 else 0.0
    print(f"  {'Search Time':<28} | {res_knn_sq['search_ms']:>15.2f} ms | {res_knn_eq['search_ms']:>15.2f} ms | {knn_t_diff:>+14.2f} ms ({knn_t_pct:+.1f}%)")
    fps_ratio = (res_knn_eq['fps'] / res_knn_sq['fps']) if res_knn_sq['fps'] > 0 else 1.0
    print(f"  {'Query Throughput':<28} | {res_knn_sq['fps']:>14.0f} fps | {res_knn_eq['fps']:>14.0f} fps | {fps_ratio:>17.2f}x")
    fd_diff = res_knn_eq['framedists'] - res_knn_sq['framedists']
    fd_pct = (fd_diff / res_knn_sq['framedists'] * 100.0) if res_knn_sq['framedists'] > 0 else 0.0
    print(f"  {'Framedist Calculations':<28} | {res_knn_sq['framedists']:>18} | {res_knn_eq['framedists']:>18} | {fd_diff:>+18} ({fd_pct:+.1f}%)")
    print(f"  {'Pruning Efficiency':<28} | {res_knn_sq['prune_pct']:>17.2f}% | {res_knn_eq['prune_pct']:>17.2f}% | {res_knn_eq['prune_pct'] - res_knn_sq['prune_pct']:>+17.2f}%")
    print(f"  {'Level 2 Anchors Pruned':<28} | {res_knn_sq['l2_anchors_pruned']:>18} | {res_knn_eq['l2_anchors_pruned']:>18} | {res_knn_eq['l2_anchors_pruned'] - res_knn_sq['l2_anchors_pruned']:>+18}")
    print(f"  {'Exact Recall @ 10':<28} | {recall_sq:>17.2f}% | {recall_eq:>17.2f}% | {recall_eq - recall_sq:>+17.2f}%")

    return {
        "dim": dim,
        "n": n_samples,
        "cl_sq": res_cl_sq,
        "cl_eq": res_cl_eq,
        "knn_sq": res_knn_sq,
        "knn_eq": res_knn_eq,
        "recall_sq": recall_sq,
        "recall_eq": recall_eq
    }

def main():
    results = []

    # 1. 8-Dimensional Random Walk (N = 10,000)
    d8_path = f"{TMP_DIR}/data_8D.txt"
    run_cmd(f"{BUILD_DIR}/gric-mktxtseq 10000 {d8_path} 8Dwalk")
    results.append(run_experiment("8D Walk", d8_path, 8, 10000, 0.40, k=10))

    # 2. 16-Dimensional Random Walk (N = 10,000)
    d16_path = f"{TMP_DIR}/data_16D.txt"
    run_cmd(f"{BUILD_DIR}/gric-mktxtseq 10000 {d16_path} 16Dwalk")
    results.append(run_experiment("16D Walk", d16_path, 16, 10000, 0.55, k=10))

    # 3. 32-Dimensional Random Walk (N = 10,000)
    d32_path = f"{TMP_DIR}/data_32D.txt"
    run_cmd(f"{BUILD_DIR}/gric-mktxtseq 10000 {d32_path} 32Dwalk")
    results.append(run_experiment("32D Walk", d32_path, 32, 10000, 0.75, k=10))

    # 4. 64-Dimensional Random Walk (N = 10,000)
    d64_path = f"{TMP_DIR}/data_64D.txt"
    run_cmd(f"{BUILD_DIR}/gric-mktxtseq 10000 {d64_path} 64Dwalk")
    results.append(run_experiment("64D Walk", d64_path, 64, 10000, 1.05, k=10))

    # 5. 512-Dimensional Torus Knot Dataset (N = 10,000)
    torus_bin = f"{WORKSPACE_DIR}/512Dtorus.bin"
    if os.path.exists(torus_bin):
        results.append(run_experiment("512D Torus Knot", torus_bin, 512, 10000, 0.92, k=10))

    # Grand Summary Table
    print("\n" + "=" * 90)
    print(" GRAND BENCHMARK SUMMARY: SQ16 vs EQ16 ACROSS ALL TEST CASES")
    print("=" * 90)
    print(f" {'Dataset':<16} | {'Clust Time (Core)':<21} | {'k-NN Time':<17} | {'k-NN Framedists':<17} | {'Recall':<6}")
    print(f" {'':<16} | {'SQ16 -> EQ16':<21} | {'SQ16 -> EQ16':<17} | {'SQ16 -> EQ16 (Saved)':<17} | {'EQ16':<6}")
    print("-" * 90)
    for r in results:
        cl_sq_t = r['cl_sq']['proc_ms']
        cl_eq_t = r['cl_eq']['proc_ms']
        knn_sq_t = r['knn_sq']['search_ms']
        knn_eq_t = r['knn_eq']['search_ms']
        fd_sq = r['knn_sq']['framedists']
        fd_eq = r['knn_eq']['framedists']
        saved = fd_sq - fd_eq
        pct = (saved / fd_sq * 100.0) if fd_sq > 0 else 0.0
        name = f"{r['dim']}D (N={r['n']})"
        print(f" {name:<16} | {cl_sq_t:>8.1f} -> {cl_eq_t:<8.1f} ms | {knn_sq_t:>6.1f} -> {knn_eq_t:<6.1f} ms | {fd_sq:>6} -> {fd_eq:<6} (-{pct:4.1f}%) | {r['recall_eq']:>5.1f}%")
    print("=" * 90)

if __name__ == "__main__":
    main()
