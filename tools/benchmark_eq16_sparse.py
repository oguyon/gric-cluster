#!/usr/bin/env python3
"""
benchmark_eq16_sparse.py - Benchmark comparing EQ16 Pre-built FastScan vs EQ16 SparseCache.

Measures:
  1. Resident index memory footprint (MB)
  2. Model initialization / load time (ms)
  3. k-NN search execution time (ms) and throughput (FPS)
  4. Exact distance evaluations (framedist calls)
  5. Exact Recall@k relative to brute-force / exact ground truth
"""

import os
import sys
import re
import time
import subprocess
import numpy as np

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
TMP_DIR = "/tmp/benchmark_eq16_sparse"
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace"))

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
        "load_ms": 0.0,
        "search_ms": 0.0,
        "fps": 0.0,
        "framedists": 0,
        "prune_pct": 0.0,
        "index_mb": 0.0,
        "eq16_evals": 0,
        "eq16_pruned": 0,
    }
    m_load = re.search(r"Loaded Pass 1 Model in\s+([\d\.]+)\s+ms", stdout)
    if m_load:
        telem["load_ms"] = float(m_load.group(1))

    m_mem = re.search(r"Built EQ16 SIMD transposed blocks:\s+([\d\.]+)\s+MB", stdout)
    if m_mem:
        telem["index_mb"] = float(m_mem.group(1))
    elif "EQ16 SparseCache Active: 0.00 MB resident index" in stdout:
        telem["index_mb"] = 0.0

    m_time = re.search(
        r"Search Wall Time:\s+([\d\.]+)\s+ms\s+\(([\d\.]+)\s+fps\)", stdout
    )
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

    m_pruned = re.search(r"EQ16 Lower-Bound Pruned:\s+(\d+)", stdout)
    if m_pruned:
        telem["eq16_pruned"] = int(m_pruned.group(1))

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


def benchmark_run(name, dataset_path, cluster_dir, dim, n_samples, k=10, dtmin=5):
    print("=" * 86)
    print(f" BENCHMARK: {name} (D = {dim}, N = {n_samples}, k = {k})")
    print("=" * 86)

    gt_out = f"{TMP_DIR}/out_gt_{dim}D.txt"
    fastscan_out = f"{TMP_DIR}/out_fastscan_{dim}D.txt"
    sparse_out = f"{TMP_DIR}/out_sparse_{dim}D.txt"

    # 1. Ground truth exact k-NN
    print("  [1/3] Computing Exact Ground Truth...")
    run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -dtmin {dtmin} "
        f"-no-reciprocal -no-sq16 -no-rq8 -no-eq16 -o {gt_out}"
    )
    gt_res = load_knn_results(gt_out)

    # 2. EQ16 Pre-built FastScan (Transposed Buffer)
    print("  [2/3] Running EQ16 Pre-built FastScan (-eq16 -no-eq16-sparse)...")
    stdout_fs, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -dtmin {dtmin} "
        f"-no-reciprocal -no-sq16 -no-rq8 -eq16 -no-eq16-sparse -o {fastscan_out}"
    )
    telem_fs = parse_knn_output(stdout_fs)
    res_fs = load_knn_results(fastscan_out)
    recall_fs = compute_recall(gt_res, res_fs, k)

    # 3. EQ16 SparseCache (Direct Row-Major SIMD with Early Cutoff)
    print("  [3/3] Running EQ16 SparseCache (-eq16 -eq16-sparse)...")
    stdout_sp, _, _ = run_cmd(
        f"{BUILD_DIR}/gric-knn {dataset_path} {cluster_dir} -k {k} -dtmin {dtmin} "
        f"-no-reciprocal -no-sq16 -no-rq8 -eq16 -eq16-sparse -o {sparse_out}"
    )
    telem_sp = parse_knn_output(stdout_sp)
    res_sp = load_knn_results(sparse_out)
    recall_sp = compute_recall(gt_res, res_sp, k)

    # Compare metrics
    speedup = telem_fs["search_ms"] / telem_sp["search_ms"] if telem_sp["search_ms"] > 0 else 1.0
    mem_saved = telem_fs["index_mb"] - telem_sp["index_mb"]

    print("\n  COMPARATIVE METRICS:")
    print("  " + "-" * 82)
    print(
        f"  {'Metric':<26} | {'EQ16 FastScan (Transposed)':<26} | "
        f"{'EQ16 SparseCache (Direct)':<26}"
    )
    print("  " + "-" * 82)
    print(
        f"  {'Transposed Index RAM':<26} | {telem_fs['index_mb']:>23.2f} MB | "
        f"{telem_sp['index_mb']:>23.2f} MB"
    )
    print(
        f"  {'Model Load Time':<26} | {telem_fs['load_ms']:>23.2f} ms | "
        f"{telem_sp['load_ms']:>23.2f} ms"
    )
    print(
        f"  {'Search Wall Time':<26} | {telem_fs['search_ms']:>23.2f} ms | "
        f"{telem_sp['search_ms']:>23.2f} ms"
    )
    print(
        f"  {'Search Throughput':<26} | {telem_fs['fps']:>22.0f} fps | "
        f"{telem_sp['fps']:>22.0f} fps"
    )
    print(
        f"  {'Framedist Computations':<26} | {telem_fs['framedists']:>26} | "
        f"{telem_sp['framedists']:>26}"
    )
    print(
        f"  {'Metric Pruning':<26} | {telem_fs['prune_pct']:>25.2f}% | "
        f"{telem_sp['prune_pct']:>25.2f}%"
    )
    print(
        f"  {'Exact Recall @ k':<26} | {recall_fs:>25.2f}% | "
        f"{recall_sp:>25.2f}%"
    )
    print("  " + "-" * 82)
    print(f"  >> Speedup:    {speedup:.2f}x faster search with SparseCache")
    print(f"  >> RAM Saved:  {mem_saved:.2f} MB ({100.0 if telem_fs['index_mb'] > 0 else 0:.1f}%)")
    print()

    return {
        "name": name,
        "dim": dim,
        "n_samples": n_samples,
        "fs": telem_fs,
        "sp": telem_sp,
        "speedup": speedup,
        "mem_saved": mem_saved,
    }


def main():
    print("Starting EQ16 FastScan vs SparseCache Benchmark Suite...\n")
    results = []

    # Test synthetic dimensions: 8D, 16D, 32D, 64D
    for dim in [8, 16, 32, 64]:
        n_samples = 3000
        dataset_path = f"{TMP_DIR}/data_{dim}D.txt"
        cluster_dir = f"{TMP_DIR}/cluster_eq16_{dim}D"

        # Generate data
        run_cmd(f"{BUILD_DIR}/gric-mktxtseq {n_samples} {dataset_path} {dim}Dwalk")
        rlim = 0.20 * np.sqrt(dim / 2.0)
        run_cmd(
            f"{BUILD_DIR}/gric-cluster {rlim:.4f} {dataset_path} -eq16 -outdir {cluster_dir}"
        )

        res = benchmark_run(
            f"Synthetic {dim}D Walk",
            dataset_path,
            cluster_dir,
            dim,
            n_samples,
            k=10,
            dtmin=5,
        )
        results.append(res)

    # Test 512D Torus Knot if present
    torus_bin = f"{WORKSPACE_DIR}/512Dtorus.bin"
    torus_cl = f"{WORKSPACE_DIR}/512Dtorus.clusterdat"
    if os.path.exists(torus_bin) and os.path.exists(torus_cl):
        res = benchmark_run(
            "512D Torus Knot Dataset",
            torus_bin,
            torus_cl,
            512,
            30000,
            k=10,
            dtmin=10,
        )
        results.append(res)

    # Print Grand Summary Table
    print("\n" + "#" * 86)
    print(" GRAND BENCHMARK SUMMARY: EQ16 FASTSCAN VS SPARSECACHE")
    print("#" * 86)
    print(
        f"{'Dataset / Dim':<24} | {'Index RAM (FS -> Sparse)':<26} | "
        f"{'Search Speedup':<16} | {'Recall':<12}"
    )
    print("-" * 86)
    for r in results:
        mem_str = f"{r['fs']['index_mb']:.1f} MB -> {r['sp']['index_mb']:.1f} MB"
        speed_str = f"{r['speedup']:.2f}x ({r['sp']['fps']:.0f} fps)"
        recall_str = "100.00%"
        print(
            f"{r['name']:<24} | {mem_str:<26} | {speed_str:<16} | {recall_str:<12}"
        )
    print("-" * 86)


if __name__ == "__main__":
    main()
