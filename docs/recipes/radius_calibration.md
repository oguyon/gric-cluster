---
name: radius_calibration
title: Radius Calibration Guide
---
# Recipe: Automatic Radius Threshold Calibration

1. Rapid distance spectrum scan:
   $ gric-cluster -scandist input_data.bin

2. Select adaptive median factor:
   - a0.5 : Fine granularity, many small clusters (high fidelity)
   - a1.0 : Balanced segmentation (recommended default)
   - a1.5 : Coarse grouping, fewer clusters

3. Or use gric_probe_dataset for percentile recommendations (D1%, D10%, D25%).
