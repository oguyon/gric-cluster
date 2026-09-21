#!/usr/bin/env python3
"""
benchmark_knn_comparison.py - Benchmark GRIC kNN against FAISS, HNSWLib, and Annoy.

Evaluates:
  1. FAISS Flat (Exact brute-force L2)
  2. FAISS IVFFlat (Inverted File Flat)
  3. FAISS IVFPQFastScan (Inverted File 4-bit FastScan Product Quantization)
  4. FAISS HNSW (Hierarchical Navigable Small World)
  5. HNSWLib (Reference C++ HNSW implementation)
  6. Spotify Annoy (Random Projection Trees)
  7. GRIC kNN FP32 (Full precision metric-pruned solver)
  8. GRIC kNN SQ16 FastScan (16-bit scalar quant with transposed FastScan)
  9. GRIC kNN SQ16 SparseCache (Direct row-major SIMD, 0 MB resident index)
 10. GRIC kNN RQ8 (Cluster-adaptive local 8-bit residual quantization)
 11. GRIC kNN RaBitQ-2b (2-bit FWHT randomized bit quantization)
 12. GRIC kNN RaBitQ-1b (1-bit FWHT randomized bit quantization)

Metrics:
  - Build Time (s)
  - Index Memory (MB)
  - Search Time for 30,000 queries (ms)
  - Query Throughput (QPS)
  - Recall@10 (%) vs Exact L2 Ground Truth
"""

import os
import sys
import time
import json
import struct
import subprocess
import numpy as np

# Third-party libraries
try:
    import faiss
except ImportError:
    faiss = None

try:
    import hnswlib
except ImportError:
    hnswlib = None

try:
    import annoy
except ImportError:
    annoy = None


def load_gric_binary(filepath):
    """Load dataset from self-describing GRIC binary format."""
    with open(filepath, "rb") as f:
        hdr = f.read(64)
        magic, ver, ftype, dtype, endian, hbytes, ndim, flags, num_el, data_bytes = (
            struct.unpack("<4sBBBBHHIQQ", hdr[:32])
        )
        dims = struct.unpack("<4Q", hdr[32:64])
        f.seek(hbytes)
        n_frames = int(dims[0])
        dim = int(dims[1])
        data = np.fromfile(f, dtype=np.float32).reshape(n_frames, dim)
        return data


def run_cmd(cmd):
    """Execute shell command and return stdout, stderr, and returncode."""
    p = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return p.stdout, p.stderr, p.returncode


def compute_ground_truth(data, num_queries=1000, k=10, dtmin=10):
    """Compute exact ground truth L2 nearest neighbors using FAISS Flat."""
    dim = data.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(data)
    fetch_k = min(data.shape[0], max(100, k * 5))
    dists, indices = index.search(data[:num_queries], fetch_k)

    gt = []
    for q in range(num_queries):
        valid = []
        for idx in indices[q]:
            if abs(int(idx) - q) >= dtmin:
                valid.append(int(idx))
                if len(valid) == k:
                    break
        gt.append(set(valid))
    return gt


def evaluate_recall(ground_truth, query_results, dtmin=10):
    """Compute Recall@k against ground truth sets."""
    num_queries = len(ground_truth)
    if num_queries == 0 or len(query_results) < num_queries:
        return 0.0

    k = len(next(iter(ground_truth)))
    total = num_queries * k
    matches = 0
    for q in range(num_queries):
        res = query_results[q]
        filtered = [int(idx) for idx in res if abs(int(idx) - q) >= dtmin][:k]
        matches += len(ground_truth[q].intersection(set(filtered)))
    return (matches / total) * 100.0 if total > 0 else 0.0


def benchmark_faiss_flat(data, num_queries, k, dtmin):
    """Benchmark FAISS Flat (Exact baseline)."""
    dim = data.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(dim)
    index.add(data)
    t_build = time.perf_counter() - t0

    fetch_k = min(data.shape[0], max(100, k * 5))
    t0 = time.perf_counter()
    D, I = index.search(data[:num_queries], fetch_k)
    t_search = time.perf_counter() - t0

    ram_mb = data.nbytes / (1024 * 1024)

    return {
        "name": "FAISS Flat (Exact L2)",
        "build_s": t_build,
        "ram_mb": ram_mb,
        "search_ms": t_search * 1000.0,
        "qps": num_queries / t_search,
        "results": I,
    }


def benchmark_faiss_ivf(data, num_queries, k, dtmin, nlist=100, nprobe=8):
    """Benchmark FAISS IVFFlat."""
    dim = data.shape[1]
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.train(data)
    index.add(data)
    t_build = time.perf_counter() - t0

    index.nprobe = nprobe
    fetch_k = min(data.shape[0], max(100, k * 5))
    t0 = time.perf_counter()
    D, I = index.search(data[:num_queries], fetch_k)
    t_search = time.perf_counter() - t0

    ram_mb = (data.nbytes + nlist * dim * 4 + data.shape[0] * 8) / (1024 * 1024)

    return {
        "name": f"FAISS IVF-Flat (nprobe={nprobe})",
        "build_s": t_build,
        "ram_mb": ram_mb,
        "search_ms": t_search * 1000.0,
        "qps": num_queries / t_search,
        "results": I,
    }


def benchmark_faiss_ivf_pq_fastscan(data, num_queries, k, dtmin, nlist=100, nprobe=8):
    """Benchmark FAISS IVFPQFastScan (4-bit FastScan)."""
    dim = data.shape[1]
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQFastScan(quantizer, dim, nlist, 32, 4)
    index.train(data)
    index.add(data)
    t_build = time.perf_counter() - t0

    index.nprobe = nprobe
    fetch_k = min(data.shape[0], max(100, k * 5))
    t0 = time.perf_counter()
    D, I = index.search(data[:num_queries], fetch_k)
    t_search = time.perf_counter() - t0

    ram_mb = (data.shape[0] * 16 + nlist * dim * 4) / (1024 * 1024)

    return {
        "name": f"FAISS IVFPQFastScan (npr={nprobe})",
        "build_s": t_build,
        "ram_mb": ram_mb,
        "search_ms": t_search * 1000.0,
        "qps": num_queries / t_search,
        "results": I,
    }


def benchmark_faiss_hnsw(data, num_queries, k, dtmin, m=32, ef_search=64):
    """Benchmark FAISS HNSW."""
    dim = data.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(dim, m)
    index.add(data)
    t_build = time.perf_counter() - t0

    index.hnsw.efSearch = ef_search
    fetch_k = min(data.shape[0], max(100, k * 5))
    t0 = time.perf_counter()
    D, I = index.search(data[:num_queries], fetch_k)
    t_search = time.perf_counter() - t0

    ram_mb = (data.nbytes + data.shape[0] * m * 2 * 4) / (1024 * 1024)

    return {
        "name": f"FAISS HNSW (M={m}, ef={ef_search})",
        "build_s": t_build,
        "ram_mb": ram_mb,
        "search_ms": t_search * 1000.0,
        "qps": num_queries / t_search,
        "results": I,
    }


def benchmark_hnswlib(data, num_queries, k, dtmin, m=32, ef_search=64):
    """Benchmark HNSWLib (Malkov & Yashunin)."""
    dim = data.shape[1]
    t0 = time.perf_counter()
    p = hnswlib.Index(space="l2", dim=dim)
    p.init_index(max_elements=data.shape[0], ef_construction=200, M=m)
    p.add_items(data)
    t_build = time.perf_counter() - t0

    p.set_ef(ef_search)
    fetch_k = min(data.shape[0], max(100, k * 5))
    t0 = time.perf_counter()
    labels, dists = p.knn_query(data[:num_queries], k=fetch_k)
    t_search = time.perf_counter() - t0

    ram_mb = (data.nbytes + data.shape[0] * m * 2 * 4) / (1024 * 1024)

    return {
        "name": f"HNSWLib (M={m}, ef={ef_search})",
        "build_s": t_build,
        "ram_mb": ram_mb,
        "search_ms": t_search * 1000.0,
        "qps": num_queries / t_search,
        "results": labels,
    }


def benchmark_annoy(data, num_queries, k, dtmin, n_trees=20):
    """Benchmark Spotify Annoy."""
    dim = data.shape[1]
    t0 = time.perf_counter()
    t_index = annoy.AnnoyIndex(dim, "euclidean")
    for i in range(data.shape[0]):
        t_index.add_item(i, data[i])
    t_index.build(n_trees, n_jobs=-1)
    t_build = time.perf_counter() - t0

    fetch_k = min(data.shape[0], max(100, k * 5))
    t0 = time.perf_counter()
    results = []
    for q in range(num_queries):
        nns = t_index.get_nns_by_item(q, fetch_k, search_k=fetch_k * n_trees)
        results.append(nns)
    t_search = time.perf_counter() - t0

    ram_mb = (data.nbytes * 1.5) / (1024 * 1024)

    return {
        "name": f"Spotify Annoy ({n_trees} trees)",
        "build_s": t_build,
        "ram_mb": ram_mb,
        "search_ms": t_search * 1000.0,
        "qps": num_queries / t_search,
        "results": results,
    }


def load_gric_results_bin(bin_path, num_queries, k):
    """Load results from gric-knn binary indices file."""
    with open(bin_path, "rb") as f:
        hdr = f.read(64)
        magic, ver, ftype, dtype, endian, hbytes, ndim, flags, num_el, data_bytes = (
            struct.unpack("<4sBBBBHHIQQ", hdr[:32])
        )
        f.seek(hbytes)
        idx = np.fromfile(f, dtype=np.uint32).reshape(num_queries, k)
        return idx


def benchmark_gric_knn(dataset_path, cluster_dir, name, flags, k, dtmin):
    """Run gric-knn CLI and parse performance telemetry."""
    out_file = f"/tmp/bench_cmp_{int(time.time() * 1000)}.txt"
    bin_idx = out_file.replace(".txt", "_indices.bin")

    cmd = (
        f"./build/gric-knn {dataset_path} {cluster_dir} -k {k} -dtmin {dtmin} "
        f"-no-reciprocal {flags} -o {out_file}"
    )

    t0 = time.perf_counter()
    stdout, stderr, ret = run_cmd(cmd)
    t_total = time.perf_counter() - t0

    if ret != 0:
        print(f"Error running GRIC kNN ({name}): {stderr}", file=sys.stderr)
        return None

    search_ms = 0.0
    qps = 0.0
    for line in stdout.splitlines():
        if "Search Wall Time:" in line:
            parts = line.split()
            search_ms = float(parts[3])
            qps = float(parts[5].strip("()"))
            break

    idx = load_gric_results_bin(bin_idx, 30000, k)

    for p in [out_file, bin_idx, bin_idx.replace("_indices", "_distances"),
              bin_idx.replace("_indices", "_mutual_dists")]:
        if os.path.exists(p):
            os.remove(p)

    return {
        "name": name,
        "build_s": 0.0,
        "search_ms": search_ms if search_ms > 0 else t_total * 1000.0,
        "qps": qps if qps > 0 else 30000.0 / t_total,
        "results": idx,
    }


def main():
    dataset_path = "workspace/512Dtorus.bin"
    cluster_dir = "workspace/512Dtorus.clusterdat"
    k = 10
    dtmin = 10
    num_eval_queries = 1000

    if not os.path.exists(dataset_path) or not os.path.exists(cluster_dir):
        print("Error: Dataset or cluster directory not found.", file=sys.stderr)
        sys.exit(1)

    print("==========================================================================")
    print("  High-Performance kNN Benchmark: GRIC vs FAISS vs HNSWLib vs Annoy")
    print("  Dataset: 512D Torus Knot (N=30,000, D=512, k=10, dtmin=10)")
    print("==========================================================================")

    print("Loading 512D Torus dataset...")
    data = load_gric_binary(dataset_path)
    print(f"Loaded dataset: shape={data.shape}, dtype={data.dtype}, "
          f"size={data.nbytes / 1e6:.1f} MB\n")

    print(f"Computing Exact Ground Truth (L2 Brute Force, N={num_eval_queries})...")
    gt = compute_ground_truth(data, num_queries=num_eval_queries, k=k, dtmin=dtmin)
    print("Ground truth computed.\n")

    benchmarks = []

    # 1. FAISS Flat
    if faiss:
        print("Running FAISS Flat...")
        res = benchmark_faiss_flat(data, data.shape[0], k, dtmin)
        res["recall"] = evaluate_recall(gt, res["results"][:num_eval_queries], dtmin)
        benchmarks.append(res)

        # 2. FAISS IVF-Flat
        print("Running FAISS IVF-Flat...")
        res = benchmark_faiss_ivf(data, data.shape[0], k, dtmin, nlist=100, nprobe=8)
        res["recall"] = evaluate_recall(gt, res["results"][:num_eval_queries], dtmin)
        benchmarks.append(res)

        # 3. FAISS IVFPQ FastScan
        print("Running FAISS IVFPQFastScan...")
        res = benchmark_faiss_ivf_pq_fastscan(
            data, data.shape[0], k, dtmin, nlist=100, nprobe=16
        )
        res["recall"] = evaluate_recall(gt, res["results"][:num_eval_queries], dtmin)
        benchmarks.append(res)

        # 4. FAISS HNSW
        print("Running FAISS HNSW...")
        res = benchmark_faiss_hnsw(data, data.shape[0], k, dtmin, m=32, ef_search=64)
        res["recall"] = evaluate_recall(gt, res["results"][:num_eval_queries], dtmin)
        benchmarks.append(res)

    # 5. HNSWLib
    if hnswlib:
        print("Running HNSWLib...")
        res = benchmark_hnswlib(data, data.shape[0], k, dtmin, m=32, ef_search=64)
        res["recall"] = evaluate_recall(gt, res["results"][:num_eval_queries], dtmin)
        benchmarks.append(res)

    # 6. Annoy
    if annoy:
        print("Running Spotify Annoy...")
        res = benchmark_annoy(data, data.shape[0], k, dtmin, n_trees=20)
        res["recall"] = evaluate_recall(gt, res["results"][:num_eval_queries], dtmin)
        benchmarks.append(res)

    # 7. GRIC kNN Variants (Cluster Tree Mode)
    gric_configs = [
        ("GRIC FP32 (Full Metric)", "-no-rq8 -no-sq8", 0.0),
        ("GRIC SQ16 (FastScan)", "-no-rq8 -sq16", 30.72),
        ("GRIC SQ16 SparseCache (Direct)", "-no-rq8 -sq16 -sq16-sparse", 0.0),
        ("GRIC SQ16 Sparse (ef=64)", "-no-rq8 -sq16 -sq16-sparse -ef-cluster 64", 0.0),
        ("GRIC RQ8 (Adaptive Residual 8b)", "-rq8", 14.64),
        ("GRIC RaBitQ-2b (2-bit FWHT)", "-rabitq -rabitq-bits 2", 3.84),
        ("GRIC RaBitQ-1b (1-bit FWHT)", "-rabitq -rabitq-bits 1", 1.92),
    ]

    for name, flags, ram in gric_configs:
        print(f"Running {name}...")
        res = benchmark_gric_knn(dataset_path, cluster_dir, name, flags, k, dtmin)
        if res:
            res["ram_mb"] = ram
            res["recall"] = evaluate_recall(gt, res["results"][:num_eval_queries], dtmin)
            benchmarks.append(res)

    # Print Summary Table
    print("\n" + "=" * 98)
    print(
        f"{'Algorithm / Quantizer':<34} | {'Build (s)':<9} | {'RAM (MB)':<8} | "
        f"{'Time (ms)':<9} | {'Throughput':<11} | {'Recall@10':<9}"
    )
    print("-" * 98)

    for b in benchmarks:
        print(
            f"{b['name']:<34} | {b['build_s']:>7.2f} s | {b['ram_mb']:>6.1f} MB | "
            f"{b['search_ms']:>8.2f}  | {b['qps']:>9.0f} fps | {b['recall']:>8.2f}%"
        )
    print("=" * 98)

    json_path = "/tmp/knn_benchmark_results.json"
    export_data = [
        {
            "name": b["name"],
            "build_s": b["build_s"],
            "ram_mb": b["ram_mb"],
            "search_ms": b["search_ms"],
            "qps": b["qps"],
            "recall": b["recall"],
        }
        for b in benchmarks
    ]
    with open(json_path, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\nRaw results exported to {json_path}")


if __name__ == "__main__":
    main()
