# analysis

## OVERVIEW
Analysis and debugging options help calibrate
parameters and monitor the clustering run.

## PRE-RUN CALIBRATION
-scandist
  Reads frames and computes distance statistics
  (min, max, median, 20%, 80% percentiles)
  without clustering. Use the median or 20%
  value to pick a good rlim.

  Example:
    gric-cluster -scandist input.txt

## RUNTIME MONITORING
-progress
  Print periodic progress information.
  Enabled by default.

## SEE ALSO
- `-scandist`: Measure distance stats
- `-progress`: Print progress (default: enabled)
- `-conf`: Read options from configuration file
- `-confw`: Write current options to file
