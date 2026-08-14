# output

## OVERVIEW
All output files are written to a directory named
<input>.clusterdat/ by default (override with
-outdir). Most outputs are disabled by default;
enable them with the flags below.

## CORE OUTPUTS (enabled by default)
frame_membership.txt
  FrameIndex ClusterIndex per line.
  For streams: also includes cnt0 and timestamp.

dcc.txt
  Inter-cluster distance matrix:
  Cluster_i  Cluster_j  Distance

## OPTIONAL OUTPUTS
transition_matrix.txt  (-tm_out)
  From_Cluster  To_Cluster  Count

cluster_counts.txt     (-counts)
  Number of frames assigned to each cluster

anchors.fits/txt/png   (-anchors)
  Anchor frame of each cluster

average.fits/txt/png   (-avg)
  Mean frame per cluster (lucky imaging)

*.clustered.txt        (-clustered)
  All input data grouped by cluster

cluster_X/             (-clusters)
  Individual directories with member frames

discarded_frames.txt   (-discarded)
  Frame indices from evicted clusters

distall.txt            (-distall)
  Every computed distance with metadata

## SEE ALSO
- `-outdir`: Specify output directory
- `-avg`: Compute average frame per cluster
- `-pngout`: Write output as PNG images
- `-fitsout`: Force FITS output format
- `-dcc`: Enable dcc.txt output
- `-no_dcc`: Disable dcc.txt output
- `-tm_out`: Enable transition_matrix.txt output
- `-anchors`: Enable anchors output
- `-counts`: Enable cluster_counts.txt output
- `-membership`: Enable frame_membership.txt output
- `-no_membership`: Disable frame_membership.txt output
- `-discarded`: Enable discarded_frames.txt output
- `-clustered`: Enable *.clustered.txt output
- `-clusters`: Enable individual cluster files
- `-shm`: Enable shared-memory status output
