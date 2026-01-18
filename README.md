<img src="gric.png" alt="GRIC Logo" width="700">

# High-speed distance-based clustering

**Fast Image Clustering**

A high-speed image clustering tool written in C, optimized for performance. It groups frames (images) into clusters based on Euclidean distance.

## Features

- **Fast & Optimized**: Written in C17 for performance.
- **Multiple Formats**: Supports FITS, MP4 (video), and ASCII text files.
- **Customizable**: Tunable distance limits (`rlim`), probability rewards, and geometric matching parameters.
- **Plotting**: Includes `gric-plot` to visualize clusters (SVG/PNG).
- **Modeling**: Includes `gric-NDmodel` to reconstruct N-dimensional models from distance matrices.

## Programs

The project builds several executables, each serving a specific role:

| Program | Role | Description |
| :--- | :--- | :--- |
| **gric-cluster** | Core Tool | Main clustering executable. Groups frames from files or streams based on Euclidean distance. |
| **gric-info** | Info | Displays status, versions, and paths of optional libraries (CFITSIO, PNG, FFmpeg, etc.). |
| **gric-plot** | Visualization | Generates summary plots (PNG/SVG) of clustering results, including scatter plots and histograms. |
| **gric-ascii-spot-2-video** | Simulation | Converts coordinate text files into MP4 video or ImageStreamIO streams (simulating a moving spot). |
| **gric-mktxtseq** | Test Data | Generates synthetic coordinate sequences (random, walk, spiral, etc.) for testing. |
| **gric-mkclusteredfile** | Post-processing | Reconstructs a full clustered file from input data and a membership list. |
| **gric-NDmodel** | Modeling | Reconstructs N-dimensional coordinates from a cluster distance matrix (`dcc.txt`) using Simulated Annealing. |
| **gric-stream-to-pipe** | Utility | (ImageStreamIO only) Pipes raw data from a shared memory stream to stdout. |

## Dependencies

### Mandatory
*   **CMake** (Build system)
*   **C Compiler** (GCC/Clang) with C17 support
*   **pkg-config**

### Optional
The build system automatically detects these libraries. If missing, the corresponding features are disabled.

*   **CFITSIO** (`libcfitsio-dev`): Required for FITS file input/output.
*   **LibPNG** (`libpng-dev`): Required for PNG output in `gric-plot` and `gric-cluster`.
*   **FFmpeg** (`libav*-dev`): Required for processing MP4 video files.
*   **ImageStreamIO**: Required for low-latency shared memory streaming.
*   **OpenMP**: Enables multi-threading support for faster processing.

### Installation (Debian/Ubuntu)

```bash
# Mandatory
sudo apt update
sudo apt install build-essential cmake pkg-config

# Recommended (for full features)
sudo apt install libcfitsio-dev libpng-dev \
    libavformat-dev libavcodec-dev libswscale-dev libavutil-dev \
    libomp-dev
```

## Build

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

Check installed features:
```bash
./gric-info
```

Run clustering:
```bash
./gric-cluster [options] <rlim> <input_file>
```

See [Wiki](https://github.com/oguyon/stream-cluster/wiki) for detailed documentation.
