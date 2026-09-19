#!/usr/bin/env python3
"""
benchmark_rabitq.py - Comprehensive benchmark of RaBitQ vs FP32, SQ16, and RQ8 on gric-knn.
"""

import os
import sys
import re
import time
import subprocess

def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print(f"Command failed: {cmd}\nStderr: {p.stderr}", file=sys.stderr)
    return p.stdout, p.stderr, p.returncode

def parse_knn_output(stdout):
    telem = {}
    m_time = re.search(r"Search Wall Time:\s+([\d\.]+)\s+ms\s+\(([\d\.]+)\s+fps\)", stdout)
    if m_time:
        telem["search_ms"] = float(m_time.group(1))
        telem["fps"] = float(m_time.group(2))
    m_load = re.search(r"Loaded Pass 1 Model in ([\d\.]+)\s+ms", stdout)
    if m_load:
        telem["load_ms"] = float(m_load.group(1))
    m_dist = re.search(r"Framedist Computations:\s+(\d+)", stdout)
    if m_dist:
        telem["framedists"] = int(m_dist.group(1))
    m_prune = re.search(r"Metric Pruning Efficiency:\s+([\d\.]+)%", stdout)
    if m_prune:
        telem["prune_pct"] = float(m_prune.group(1))
    m_ram = re.search(r"\[FASTSCAN\] Built.*?(\d+\.?\d*)\s+MB", stdout)
    if m_ram:
        telem["fastscan_mb"] = float(m_ram.group(1))
    else:
        telem["fastscan_mb"] = 0.0
    return telem

def load_results(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 101:
                q_id = int(parts[0])
                n_ids = [int(parts[i]) for i in range(1, 101, 2)]
                data.append((q_id, set(n_ids)))
    return data

def compute_recall(gt_data, cand_data):
    if not gt_data or not cand_data or len(gt_data) != len(cand_data):
        return 0.0
    matches = 0
    total = 0
    for (_, s_gt), (_, s_cand) in zip(gt_data, cand_data):
        matches += len(s_gt.intersection(s_cand))
        total += len(s_gt)
    return (matches / total * 100.0) if total > 0 else 0.0

def main():
    dataset = "workspace/512Dtorus.bin"
    clusterdat = "workspace/512Dtorus.clusterdat"
    k = 50
    dtmin = 10

    if not os.path.exists(dataset) or not os.path.exists(clusterdat):
        print(f"Error: Dataset {dataset} or {clusterdat} not found.", file=sys.stderr)
        sys.exit(1)

    dataset_size_bytes = os.path.getsize(dataset)
    dataset_mb = dataset_size_bytes / (1024 * 1024)

    configs = [
        {
            "name": "FP32 (No Quant)",
            "flags": "-no-rq8 -no-sq8",
            "out": "/tmp/bench_fp32.txt",
            "sidecar": None,
        },
        {
            "name": "SQ16 (Scalar Quant 16-bit)",
            "flags": "-no-rq8 -sq16",
            "out": "/tmp/bench_sq16.txt",
            "sidecar": "/tmp/bench_512Dtorus.sq16",
            "save_flag": "-sq16-save /tmp/bench_512Dtorus.sq16",
        },
        {
            "name": "RQ8 (Residual Quant 8-bit)",
            "flags": "-rq8",
            "out": "/tmp/bench_rq8.txt",
            "sidecar": "/tmp/bench_512Dtorus.rq8",
            "save_flag": "-rq8-save /tmp/bench_512Dtorus.rq8",
        },
        {
            "name": "RaBitQ 1-bit",
            "flags": "-rabitq -rabitq-bits 1",
            "out": "/tmp/bench_rabitq_1b.txt",
            "sidecar": "/tmp/bench_512Dtorus_1b.rabitq",
            "save_flag": "-rabitq-save /tmp/bench_512Dtorus_1b.rabitq",
        },
        {
            "name": "RaBitQ 2-bit",
            "flags": "-rabitq -rabitq-bits 2",
            "out": "/tmp/bench_rabitq_2b.txt",
            "sidecar": "/tmp/bench_512Dtorus_2b.rabitq",
            "save_flag": "-rabitq-save /tmp/bench_512Dtorus_2b.rabitq",
        },
    ]

    print("==========================================================================")
    print("  gric-knn RaBitQ Quantization Benchmark (512D Torus Dataset, N=30,000)")
    print("==========================================================================")
    print(f"Dataset: {dataset} ({dataset_mb:.2f} MB, D=512, N=30,000, k={k})\n")

    results = []
    gt_data = None

    for c in configs:
        name = c["name"]
        flags = c["flags"]
        out_file = c["out"]
        save_flag = c.get("save_flag", "")
        sidecar = c["sidecar"]

        if sidecar and os.path.exists(sidecar):
            os.remove(sidecar)
        if os.path.exists(out_file):
            os.remove(out_file)

        cmd = (
            f"./build/gric-knn {dataset} {clusterdat} -k {k} -dtmin {dtmin} "
            f"-no-reciprocal {flags} {save_flag} -o {out_file}"
        )
        print(f"Running [{name}]...")
        t0 = time.perf_counter()
        stdout, stderr, ret = run_cmd(cmd)
        t1 = time.perf_counter()

        if ret != 0:
            print(f"Failed to run {name}")
            continue

        telem = parse_knn_output(stdout)
        telem["name"] = name
        telem["total_time_s"] = t1 - t0

        if sidecar and os.path.exists(sidecar):
            telem["sidecar_bytes"] = os.path.getsize(sidecar)
            telem["sidecar_mb"] = telem["sidecar_bytes"] / (1024 * 1024)
            telem["compression_ratio"] = dataset_size_bytes / telem["sidecar_bytes"]
        else:
            telem["sidecar_bytes"] = dataset_size_bytes
            telem["sidecar_mb"] = dataset_mb
            telem["compression_ratio"] = 1.0

        cand_data = load_results(out_file)
        if name == "FP32 (No Quant)":
            gt_data = cand_data
            telem["recall"] = 100.0
        else:
            telem["recall"] = compute_recall(gt_data, cand_data)

        results.append(telem)
        t_ms = telem.get('search_ms', 0)
        fps = telem.get('fps', 0)
        rec = telem.get('recall', 0)
        sc_mb = telem['sidecar_mb']
        comp = telem['compression_ratio']
        print(
            f"  -> Time: {t_ms:.2f} ms ({fps:.1f} fps), Recall: {rec:.4f}%, "
            f"Sidecar: {sc_mb:.2f} MB ({comp:.1f}x)"
        )

    # Print summary table
    print("\n" + "=" * 80)
    print(
        f"{'Quantizer':<18} | {'Sidecar':<8} | {'Comp':<5} | {'ScanMB':<7} | "
        f"{'Time(ms)':<8} | {'Throughput':<11} | {'Recall':<8}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<18} | {r['sidecar_mb']:>5.1f} MB | {r['compression_ratio']:>4.1f}x | "
            f"{r['fastscan_mb']:>4.1f} MB | {r['search_ms']:>8.2f} | {r['fps']:>9.1f}   | "
            f"{r['recall']:>7.3f}%"
        )
    print("=" * 80)

if __name__ == "__main__":
    main()
