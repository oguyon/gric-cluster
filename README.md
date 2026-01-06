# High-speed distance-based clustering

**Fast Image Clustering**

A high-speed image clustering tool written in C, optimized for performance. It groups frames (images) into clusters based on Euclidean distance.

## Features

- **Fast & Optimized**: Written in C99 for performance.
- **Multiple Formats**: Supports FITS, MP4 (video), and ASCII text files.
- **Customizable**: Tunable distance limits (`rlim`), probability rewards, and geometric matching parameters.
- **Plotting**: Includes `image-cluster-plot` to visualize clusters (SVG/PNG).
- **Modeling**: Includes `image-cluster-NDmodel` to reconstruct N-dimensional models from distance matrices.

## Dependencies

Required packages (Debian/Ubuntu):

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config \
    libcfitsio-dev \
    libpng-dev \
    libavformat-dev libavcodec-dev libswscale-dev libavutil-dev
```

If libraries are missing, the build system will automatically disable the corresponding features (e.g., FITS support, MP4 support, PNG output).

## Build

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

```bash
./image-cluster <rlim> [options] <input_file>
```

See [Wiki](https://github.com/oguyon/stream-cluster/wiki) for detailed documentation.
