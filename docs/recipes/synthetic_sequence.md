---
name: synthetic_sequence
title: Synthetic Sequence Generation
---
# Recipe: Generating Synthetic Manifolds & Benchmarking

1. Generate synthetic coordinate sequence:
   $ gric-mktxtseq 5000 /tmp/spiral.txt 2Dspiral

2. Cluster synthetic sequence:
   $ gric-cluster 0.15 /tmp/spiral.txt -outdir /tmp/spiral.clusterdat

3. Inspect results and plot:
   Call gric_inspect_run(run_dir="/tmp/spiral.clusterdat")
   $ gric-plot /tmp/spiral.txt /tmp/spiral.clusterdat/cluster_run.log /tmp/plot.png
