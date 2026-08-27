# Overview

This file compiles tests and benchmarks for the gric-cluster program, with a short discussion
of results in comments.

---

# Simple 2D Patterns

## 2D Spiral

In this test, a 2D point slowly moves outward in a spiral pattern.

The pattern is written to a text file with:
```bash
./build/gric-mktxtseq 20000 2Dspiral.txt 2Dspiral
# OUTPUT: 2Dspiral.txt (20000 samples)
```

### Simple Example: Clustering from the 2D txt Input

The samples are clustered with:
```bash
./build/gric-cluster 0.095 2Dspiral.txt
# OUTPUT:
# 2Dspiral.clusterdat/cluster_run.log
# 2Dspiral.clusterdat/frame_membership.txt
# 2Dspiral.clusterdat/dcc.txt
```
Here, the cluster radius value has been adjusted to get 100 clusters.

Results can be visualized with the plot utility:
```bash
./build/gric-plot 2Dspiral.txt 2Dspiral.clusterdat/cluster_run.log ./plots/plot.2Dspiral.png
```

The 20,000 samples are clustered into 100 clusters with 25,091 distance computations (average:
1.255 distance computations per sample). Most samples are resolved with a single distance
computation thanks to the slow-moving sample coordinates.

**GRIC first tests if the current frame belongs to the same cluster as the previous frame.
With slow-moving input, as is often the case in video streams, most samples are quickly
resolved/confirmed with a single distcomp.**

### High Dimension Input (Manifold Embeddings)

Operating in high dimension (256x256 pixel images = 65,536 dimensions) derived from a low-D input.
We use `gric-ascii-spot-2-video` to convert the 2D input into a high-D image stream.

Writer (write 2D spot to stream, with cnt2sync):
```bash
./build/gric-ascii-spot-2-video -isio -cnt2sync 256 0.1 2Dspiral.txt spot2d
```

Reader (run clustering):
```bash
./build/gric-cluster -stream -cnt2sync 2560 spot2d
```

And results are plotted with:
```bash
./build/gric-plot 2Dspiral.txt spot2d.clusterdat/cluster_run.log ./plots/plot.2Dspiral.im256.png
```

***GRIC's efficiency (number of distcomps required for solving) is preserved in high dimension
if a manifold embedding to lower dimension exists.***

---

# Benchmark Script Recipes

```bash
MKSEQEXEC="./build/gric-mktxtseq"
RNUCLEXEC="./build/gric-cluster"
CLPLOT="./build/gric-plot"

NBSAMPLE=1000000
RLIM="0.10"
OPTIONS="-maxim $NBSAMPLE -outdir clusteroutdir"

# 1. Slow moving point on spiral (tests short-term recency memory, ~1.00 dist/frame)
$MKSEQEXEC $NBSAMPLE 2Dspiral.txt 2Dspiral
$RNUCLEXEC $RLIM $OPTIONS 2Dspiral.txt
$CLPLOT 2Dspiral.txt clusteroutdir/cluster_run.log

# 2. Random point on circle (tests 1D geometric solving, ~2.73 dist/frame)
$MKSEQEXEC $NBSAMPLE 2Dcircle-shuffle.txt 2Dcircle -shuffle
$RNUCLEXEC $RLIM $OPTIONS 2Dcircle-shuffle.txt
$CLPLOT 2Dcircle-shuffle.txt clusteroutdir/cluster_run.log

# 3. Random points on spiral (tests 2D geometric manifold solving, ~3.0 dist/frame)
$MKSEQEXEC $NBSAMPLE 2Dspiral-shuffle.txt 2Dspiral -shuffle
$RNUCLEXEC $RLIM $OPTIONS 2Dspiral-shuffle.txt
$CLPLOT 2Dspiral-shuffle.txt clusteroutdir/cluster_run.log

# 4. Learning fine geometrical structure via -gprob (~2.52 dist/frame)
$RNUCLEXEC $RLIM -maxim $NBSAMPLE -gprob -fmatcha 1.0 -fmatchb 0.0 2Dspiral-shuffle.txt

# 5. Random points in 2D (unstructured metric packing, ~3.53 dist/frame)
$MKSEQEXEC $NBSAMPLE 2Drand.txt 2Drand
$RNUCLEXEC $RLIM $OPTIONS 2Drand.txt
$CLPLOT 2Drand.txt clusteroutdir/cluster_run.log

# 6. Random points in 3D (3D volume metric bounds, ~9.78 dist/frame)
$MKSEQEXEC $NBSAMPLE 3Drand.txt 3Drand
$RNUCLEXEC $RLIM -maxcl 10000 $OPTIONS 3Drand.txt
$CLPLOT 3Drand.txt clusteroutdir/cluster_run.log

# 7. Recurring periodic sequence with noise (tests cyclic transition / -pred)
$MKSEQEXEC $NBSAMPLE 2DcircleP10n.txt 2Dcircle10 -noise 0.04
$RNUCLEXEC $RLIM $OPTIONS 2DcircleP10n.txt
$CLPLOT 2DcircleP10n.txt clusteroutdir/cluster_run.log

# With sequence predictor (-pred[10,100,1]):
$RNUCLEXEC $RLIM -maxcl 10000 -pred[10,100,1] -maxim $NBSAMPLE 2DcircleP10n.txt
```
