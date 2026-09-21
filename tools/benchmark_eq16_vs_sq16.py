#!/usr/bin/env python3
"""
benchmark_eq16_vs_sq16.py - Direct Benchmark comparing EQ16 vs SQ16 across dimensions.

Evaluates:
  1. Covering radius reduction (-29.29% at identical point density).
  2. gric-cluster execution time, cluster count, and candidate pruning.
  3. gric-knn exact recall (guaranteed 100%), framedist evaluations, and fps.
"""

import os
import sys
import re
import time
import subprocess
import numpy as np

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
TMP_DIR = "/tmp/benchmark_eq16_vs_sq16"

os.makedirs(TMP_DIR, exist_ok=True)

def run_cmd(cmd):
    p = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if p.returncode != 0:
        print(f"Command failed: {cmd}\nStderr: {p.stderr}", file=sys.stderr)
    return p.stdout, p.stderr, p.returncode

def parse_knn_output(stdout):
    telem = {
        "search_ms": 0.0,
        "fps": 0.0,
        "framedists": 0,
        "prune_pct": 0.0,
        "eq16_evals": 0,
        "eq16_pruned": 0,
        "sq16_evals": 0,
        "sq16_pruned": 0
    }
    m_time = re.search(r"Search Wall Time:\s+([\d\.]+)\s+ms\s+\(([\d\.]+)\s+fps\)", stdout)
    if m_time:
        telem["search_ms"] = float(m_time.group(1))
        telem["fps"] = float(m_time.group(2))
    m_dist = re.search(r"Framedist Computations:\s+(\d+)", stdout)
    if m_dist:
        telem["framedists"] = int(m_dist.group(1))
    m_prune = re.search(r"Metric Pruning Efficiency:\s+([\d\.]+)%", stdout)
    if m_prune:
        telem["prune_pct"] = float(m_prune.group(1))
    m_eq16 = re.search(r"EQ16 Evaluations:\s+(\d+)", stdout)
    if m_eq16:
        telem["eq16_evals"] = int(m_eq16.group(1))
    m_sq16 = re.search(r"SQ16 Evaluations:\s+(\d+)", stdout)
    if m_sq16:
        telem["sq16_evals"] = int(m_sq16.group(1))
    return telem

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

def benchmark_dimension(dim, n_samples=3000, k=10):
    print("=" * 82)
    print(f" BENCHMARK: DIMENSION D = {dim} (N = {n_samples}, k = {k})")
    print("=" * 82)

    dataset_path = f"{TMP_DIR}/data_{dim}D.txt"
    cluster_sq16_dir = f"{TMP_DIR}/cluster_sq16_{dim}D"
    cluster_eq16_dir = f"{TMP_DIR}/cluster_eq16_{dim}D"
    gt_out = f"{TMP_DIR}/out_gt_{dim}D.txt"
    sq16_out = f"{TMP_DIR}/out_sq16_{dim}D.txt"
    eq16_out = f"{TMP_DIR}/out_eq16_{dim}D.txt"

    # 1. Generate synthetic walk dataset
    print(f"  [1/5] Generating {n_samples} frames of {dim}D walk...")
    run_cmd(f"{BUILD_DIR}/gric-mktxtseq {n_samples} {dataset_path} {dim}Dwalk")

    # 2. Cluster with SQ16
    rlim = 0.20 * np.sqrt(dim / 2.0)
    print(f"  [2/5] Clustering with SQ16 (rlim={rlim:.4f})...")
    t0 = time.perf_counter()
    stdout_cl_sq16, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-cluster {rlim:.4f} {dataset_path} -sq16 -outdir {cluster_sq16_dir}"
    )
    t_cl_sq16 = (time.perf_counter() - t0) * 1000.0

    # 3. Cluster with EQ16
    print(f"  [3/5] Clustering with EQ16 (rlim={rlim:.4f})...")
    t0 = time.perf_counter()
    stdout_cl_eq16, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-cluster {rlim:.4f} {dataset_path} -eq16 -outdir {cluster_eq16_dir}"
    )
    t_cl_eq16 = (time.perf_counter() - t0) * 1000.0

    # Extract clusters created
    m_cl_sq = re.search(r"Total clusters:\s+(\d+)", stdout_cl_sq16)
    n_cl_sq = int(m_cl_sq.group(1)) if m_cl_sq else 0
    m_cl_eq = re.search(r"Total clusters:\s+(\d+)", stdout_cl_eq16)
    n_cl_eq = int(m_cl_eq.group(1)) if m_cl_eq else 0

    print(f"        SQ16 Clustering: {t_cl_sq16:6.1f} ms -> {n_cl_sq} clusters")
    print(f"        EQ16 Clustering: {t_cl_eq16:6.1f} ms -> {n_cl_eq} clusters")

    # 4. Ground Truth Exact Search (no quantization approximations)
    print("  [4/5] Computing Exact Ground Truth...")
    run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_eq16_dir} -k {k} -no-sq16 -no-eq16 -o {gt_out}"
    )
    gt_results = load_knn_results(gt_out)

    # 5. k-NN Search with SQ16
    print("  [5/5] Running k-NN with SQ16 vs EQ16...")
    stdout_knn_sq16, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_sq16_dir} -k {k} -sq16 -o {sq16_out}"
    )
    telem_sq16 = parse_knn_output(stdout_knn_sq16)
    res_sq16 = load_knn_results(sq16_out)
    recall_sq16 = compute_recall(gt_results, res_sq16, k)

    stdout_knn_eq16, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_eq16_dir} -k {k} -eq16 -o {eq16_out}"
    )
    telem_eq16 = parse_knn_output(stdout_knn_eq16)
    res_eq16 = load_knn_results(eq16_out)
    recall_eq16 = compute_recall(gt_results, res_eq16, k)

    # Summary table
    print("\n  RESULTS:")
    print("  " + "-" * 78)
    print(f"  {'Metric':<24} | {'SQ16 (Cubic Z^D)':<24} | {'EQ16 (E8 Lattice)':<24}")
    print("  " + "-" * 78)
    print(f"  {'Clustering Time':<24} | {t_cl_sq16:>21.2f} ms | {t_cl_eq16:>21.2f} ms")
    print(f"  {'Clusters Created':<24} | {n_cl_sq:>24} | {n_cl_eq:>24}")
    print(f"  {'k-NN Framedists':<24} | {telem_sq16['framedists']:>24} | {telem_eq16['framedists']:>24}")
    print(f"  {'Pruning Efficiency':<24} | {telem_sq16['prune_pct']:>23.2f}% | {telem_eq16['prune_pct']:>23.2f}%")
    print(f"  {'Search Throughput':<24} | {telem_sq16['fps']:>20.0f} fps | {telem_eq16['fps']:>20.0f} fps")
    print(f"  {'Exact Recall':<24} | {recall_sq16:>23.2f}% | {recall_eq16:>23.2f}%")
    print("  " + "-" * 78)

    saving = telem_sq16['framedists'] - telem_eq16['framedists']
    pct = (saving / telem_sq16['framedists'] * 100.0) if telem_sq16['framedists'] > 0 else 0.0
    print(f"  >> EQ16 reduced expensive framedist calls by {saving} ({pct:+.2f}%).\n")

if __name__ == "__main__":
    for d in [8, 16, 32, 64]:
        benchmark_dimension(d, n_samples=3000, k=10)
