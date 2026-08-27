# Data Generators, Media Converters & Diagnostics

Overview of synthetic dataset generators, rendering tools, format converters, and diagnostics
in the GRIC suite.

---

## 1. `gric-mktxtseq` (Synthetic Coordinate Generator)

Generates synthetic coordinate benchmark sequences in arbitrary dimensions.

```bash
gric-mktxtseq <N> <output_file> <pattern> [options]
```

### Supported Patterns
* `[ND]random`: Uniform random coordinates in unit hypercube/hypersphere
* `[ND]walk[S]`: Random walk with step size `S` (default: `0.1`)
* `[ND]spiral[L]`: Continuous spiral with `L` loops (default: `3.0`)
* `[ND]circle[P]`: Circular periodic orbit with period `P`
* `[ND]sphere`: Random points distributed on hypersphere surface

### Options
* `-repeat <M>`: Repeat pattern $M$ consecutive cycles
* `-noise <R>`: Add Gaussian/uniform noise with standard deviation $R$
* `-shuffle`: Randomize point sequence order (testing non-sequential geometric solving)

---

## 2. `gric-plot` (Cluster Manifold Visualization)

Generates 2D/3D scatter plots, Voronoi tessellations, candidate query breakdown charts, and
transition diagrams in PNG or SVG format.

```bash
gric-plot <points_file> <log_file> [output.png] [options]
```

### Options
* `-svg`: Output high-resolution scalable vector graphic (SVG) format
* `-fs <size>`: Axis and label font size (default: `18.0`)

---

## 3. `gric-gen-balls` (Kinematic Bouncing Balls Generator)

Synthesizes 3D FITS image cubes containing multi-body physics simulations of elastic bouncing
balls with collisions.

```bash
gric-gen-balls -n <num_balls> -W <width> -H <height> -f <num_frames> [options] <output.fits>
```

---

## 4. `gric-NDmodel` (N-Dimensional Space Reconstruction)

Reconstructs Cartesian coordinate geometries from pairwise cluster distance matrices (`dcc.txt`)
using Simulated Annealing optimization.

```bash
gric-NDmodel <dcc_file> <dimensions> <output_file> [options]
```

---

## 5. `gric-ascii-spot-2-video` (Trajectory to Video/Stream)

Renders moving Gaussian spot video files (MP4 via FFmpeg) or shared-memory streams from 2D/3D
coordinate paths.

```bash
gric-ascii-spot-2-video [options] <pixel_size> <alpha> <input.txt> <output>
```

---

## 6. `gric-txt2stream` & `gric-stream-to-pipe` (Stream Utilities)

* `gric-txt2stream <input.txt> <stream_name> [options]`: Ingests ASCII coordinate text into a live
  `ImageStreamIO` shared memory circular ring buffer.
* `gric-stream-to-pipe <stream_name> [max_frames]`: Pipes raw floating-point frame data from an
  `ImageStreamIO` stream directly to stdout.

---

## 7. `gric-mkclusteredfile` (Image Cube Reconstruction)

Reconstructs a full clustered file from the original data input and the `frame_membership.txt` file.

```bash
gric-mkclusteredfile <input_file> <membership_file> <output_file> [options]
```

---

## 8. `gric-info` & `gric-help` (Diagnostics & CLI Manual)

* `gric-info`: Queries and displays compile-time feature flags, optional dependency statuses
  (CFITSIO, PNG, FFmpeg, ImageStreamIO, OpenMP), and library paths.
* `gric-help [program]`: Terminal onboarding guide and manual page viewer for any GRIC executable.
