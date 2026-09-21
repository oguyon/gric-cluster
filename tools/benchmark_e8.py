#!/usr/bin/env python3
"""
benchmark_e8.py - Benchmark E8 root lattice vs standard cubic quantization & topology.

Measures:
  1. Covering radius and lower bound tightening across dimensions 8 to 128.
  2. End-to-end gric-knn performance on 8D, 16D, and 32D datasets:
     - Baseline Cubic RQ8 (-rq8 -no-sq16)
     - E8 Lattice RQ8 (-rq8 -e8-quant -no-sq16)
     - E8 Lattice + 240-NN Gosset Graph (-rq8 -e8 -no-sq16)
  3. Pruning efficiency, distance computations saved, search wall time, and exact recall.
"""

import os
import sys
import re
import time
import subprocess
import numpy as np

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
TMP_DIR = "/tmp/benchmark_e8"

os.makedirs(TMP_DIR, exist_ok=True)

def run_cmd(cmd):
    p = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if p.returncode != 0:
        print(f"Error running: {cmd}\nStderr: {p.stderr}", file=sys.stderr)
    return p.stdout, p.stderr, p.returncode

def parse_knn_output(stdout):
    telem = {
        "search_ms": 0.0,
        "fps": 0.0,
        "framedists": 0,
        "prune_pct": 0.0,
        "rq8_evals": 0,
        "rq8_pruned": 0
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
    m_rq8_eval = re.search(r"RQ8 Evaluations:\s+(\d+)", stdout)
    if m_rq8_eval:
        telem["rq8_evals"] = int(m_rq8_eval.group(1))
    m_rq8_prune = re.search(r"RQ8 Member Pruned:\s+(\d+)", stdout)
    if m_rq8_prune:
        telem["rq8_pruned"] = int(m_rq8_prune.group(1))
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

def run_e8_microbenchmark():
    print("=" * 80)
    print(" 1. MATHEMATICAL MICRO-BENCHMARK: E8 VS CUBIC COVERING RADIUS")
    print("=" * 80)
    out, _, code = run_cmd(f"{BUILD_DIR}/test_e8_lattice")
    if code == 0:
        for line in out.splitlines():
            if "Dimension" in line or "covering radius" in line or "ratio" in line or "tightening" in line:
                print(f"  {line}")
    print()

    out, _, code = run_cmd(f"{BUILD_DIR}/test_rq8_e8")
    if code == 0:
        for line in out.splitlines():
            if "Trial" in line or "bound" in line or "tighter" in line or "PASSED" in line or "Passed" in line:
                print(f"  {line}")
    print()

def run_dataset_benchmark(dim, n_samples=3000, k=10):
    print("=" * 80)
    print(f" 2. END-TO-END k-NN BENCHMARK: {dim}D DATASET (N={n_samples}, k={k})")
    print("=" * 80)

    dataset_path = f"{TMP_DIR}/dataset_{dim}D.txt"
    cluster_dir = f"{TMP_DIR}/cluster_{dim}D"
    gt_out = f"{TMP_DIR}/out_gt_{dim}D.txt"
    cubic_out = f"{TMP_DIR}/out_cubic_{dim}D.txt"
    e8_out = f"{TMP_DIR}/out_e8_{dim}D.txt"
    e8_graph_out = f"{TMP_DIR}/out_e8_graph_{dim}D.txt"

    # 1. Generate synthetic ND walk sequence
    print(f"  Generating {n_samples} frames of {dim}Dwalk...")
    run_cmd(f"{BUILD_DIR}/gric-mktxtseq {n_samples} {dataset_path} {dim}Dwalk")

    # 2. Cluster dataset
    rlim = 0.25 * np.sqrt(dim / 2.0)
    print(f"  Clustering with rlim={rlim:.4f}...")
    run_cmd(f"{BUILD_DIR}/gric-cluster {rlim:.4f} {dataset_path} -outdir {cluster_dir}")

    # 3. Ground truth exact search
    print("  Running Ground Truth exact baseline...")
    run_cmd(f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -no-sq16 -no-rq8 -no-cluster-graph -o {gt_out}")
    gt_results = load_knn_results(gt_out)

    # 4. Standard Cubic RQ8
    print("  Running Standard Cubic RQ8 (-rq8 -no-sq16)...")
    t0 = time.perf_counter()
    stdout_cubic, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -rq8 -no-sq16 -no-cluster-graph -o {cubic_out}"
    )
    t_cubic = time.perf_counter() - t0
    telem_cubic = parse_knn_output(stdout_cubic)
    results_cubic = load_knn_results(cubic_out)
    recall_cubic = compute_recall(gt_results, results_cubic, k)

    # 5. E8 Lattice RQ8
    print("  Running E8 Lattice RQ8 (-rq8 -e8-quant -no-sq16)...")
    t0 = time.perf_counter()
    stdout_e8, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -rq8 -e8-quant -no-sq16 -no-cluster-graph -o {e8_out}"
    )
    t_e8 = time.perf_counter() - t0
    telem_e8 = parse_knn_output(stdout_e8)
    results_e8 = load_knn_results(e8_out)
    recall_e8 = compute_recall(gt_results, results_e8, k)

    # 6. E8 Lattice + E8 240-NN Graph Routing
    print("  Running E8 Lattice + 240-NN Graph (-rq8 -e8 -no-sq16)...")
    t0 = time.perf_counter()
    stdout_e8_graph, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -rq8 -e8 -no-sq16 -o {e8_graph_out}"
    )
    t_e8_graph = time.perf_counter() - t0
    telem_e8_graph = parse_knn_output(stdout_e8_graph)
    results_e8_graph = load_knn_results(e8_graph_out)
    recall_e8_graph = compute_recall(gt_results, results_e8_graph, k)

    # Table output
    print("\n  RESULTS SUMMARY:")
    print("  " + "-" * 76)
    print(f"  {'Configuration':<28} | {'Framedists':>11} | {'Pruned %':>9} | {'Throughput':>11} | {'Recall':>7}")
    print("  " + "-" * 76)
    print(f"  {'Cubic RQ8 (Baseline)':<28} | {telem_cubic['framedists']:>11} | {telem_cubic['prune_pct']:>8.2f}% | {telem_cubic['fps']:>8.0f} fps | {recall_cubic:>6.1f}%")
    print(f"  {'E8 Lattice RQ8':<28} | {telem_e8['framedists']:>11} | {telem_e8['prune_pct']:>8.2f}% | {telem_e8['fps']:>8.0f} fps | {recall_e8:>6.1f}%")
    print(f"  {'E8 RQ8 + 240-NN Graph':<28} | {telem_e8_graph['framedists']:>11} | {telem_e8_graph['prune_pct']:>8.2f}% | {telem_e8_graph['fps']:>8.0f} fps | {recall_e8_graph:>6.1f}%")
    print("  " + "-" * 76)

    dist_savings = telem_cubic['framedists'] - telem_e8['framedists']
    pct_savings = (dist_savings / telem_cubic['framedists'] * 100.0) if telem_cubic['framedists'] > 0 else 0.0
    speedup = (telem_e8_graph['fps'] / telem_cubic['fps']) if telem_cubic['fps'] > 0 else 1.0

    print(f"  >> E8 Lattice saved {dist_savings} exact framedist calls ({pct_savings:+.2f}% fewer distance evaluations).")
    print(f"  >> Combined E8 Lattice + Graph achieved {speedup:.2f}x throughput speedup at {recall_e8_graph:.1f}% exact recall.\n")

if __name__ == "__main__":
    run_e8_microbenchmark()
    run_dataset_benchmark(8, n_samples=3000, k=10)
    run_dataset_benchmark(16, n_samples=3000, k=10)
    run_dataset_benchmark(32, n_samples=3000, k=10)
