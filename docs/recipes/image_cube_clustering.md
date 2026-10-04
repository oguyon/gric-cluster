---
name: image_cube_clustering
title: Image Cube Tiled Clustering
---
# Recipe: FITS Image Cube Clustering with Spatial Tiling

1. Probe image dimensions and noise floor:
   Call gric_probe_dataset(dataset_path="image_cube.fits")

2. Cluster with 2x2 spatial tiling to capture localized patterns:
   $ gric-cluster a1.2 image_cube.fits -tiles 2x2 -maxcl 1000 -outdir cube.clusterdat

3. Inspect clustering telemetry and pruning efficiency:
   Call gric_inspect_run(run_dir="cube.clusterdat")
