---
name: high_dim_knn_search
title: High-Dimensional Quantized kNN
---
# Recipe: High-Dimensional Quantized kNN Indexing & Retrieval

1. Encode raw text vectors to self-describing binary format:
   $ gric-ascii2bin -dim 64 raw_vectors.txt data.bin

2. Calibrate dataset geometry and quantization presets:
   Call gric_probe_dataset(dataset_path="data.bin")

3. Run Pass 1 clustering with E8 lattice quantization (EQ16):
   $ gric-cluster a1.0 data.bin -maxcl 1500 -eq16 -outdir run.clusterdat

4. Verify clustering boundary invariants:
   Call gric_verify_invariants(run_dir="run.clusterdat", dataset="data.bin")

5. Run Pass 2 out-of-core metric-pruned kNN retrieval:
   $ gric-knn data.bin run.clusterdat -k 10 -multipivot -angular
