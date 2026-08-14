# Satellite Earth Observation Clustering Demo

This guide demonstrates how to download, prepare, and cluster long-term satellite image
time series of planet Earth (e.g. from **NASA Worldview / MODIS / VIIRS / GOES**)
using `gric-cluster`.

---

## 1. Overview & Objectives

Satellite Earth observations from low Earth orbit (MODIS / VIIRS) and geostationary orbit (GOES)
provide continuous multi-year image streams of Earth's surface and atmosphere.

Clustering satellite Earth imagery with `gric-cluster` demonstrates several key capabilities:

1. **Continuous Seasonal & Rotational Recurrence**:
   Tracks 365-day annual seasonal cycles, vegetation greening, polar ice expansion/retreat,
   and cloud weather systems across the globe.
2. **Multi-Tile Spatial Decomposition**:
   By subdividing the $128 \times 128$ image into $2 \times 2$ or $4 \times 4$ spatial tiles,
   the engine clusters Northern vs Southern hemisphere weather systems and regional continents
   in parallel across CPU cores.
3. **Metric Triangle Inequality Pruning**:
   Eliminates $>95\%$ of distance calculations during image classification, enabling processing
   speeds of **$>50,000\text{ frames/sec}$** on multi-core workstations.

---

## 2. Dataset Downloader Helper Script

The repository includes a dedicated helper tool, `tools/fetch_satellite_dataset.py`, to
automatically query, download, downsample, and package satellite time series into 3D FITS cubes
ready for clustering.

### Key CLI Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--source` | `worldview` | `worldview` (NASA Worldview/GIBS), `epic`, or `synthetic` |
| `--layer` | `MODIS_Terra...` | Imagery layer name in NASA Worldview / GIBS |
| `--start` | `2023-01-01` | Start timestamp (`YYYY-MM-DD` or ISO-8601) |
| `--end` | `2023-12-31` | End timestamp (`YYYY-MM-DD` or ISO-8601) |
| `--cadence` | `auto` | Step size (`auto` = native dataset cadence, or `1d`, `1h`, `10m`) |
| `--bbox` | `auto` | Bounding box (`auto` = native framing, or `minLat,minLon,maxLat,maxLon`) |
| `--size` | `128` | Target image size in pixels ($W \times H$) |
| `--max-frames` | `10000` | Maximum number of frames to download/generate |
| `--output`, `-o` | `earth_128x128.fits` | Output 3D FITS cube filepath |
| `--dry-run` | `false` | Inspect catalog counts and dates without downloading |

---

## 3. Supported Satellite Layers (`--layer`)

The `--layer` option gives direct access to hundreds of public NASA Earthdata / GIBS products:

| Layer Identifier | Satellite | Cadence | Details |
| :--- | :--- | :--- | :--- |
| `MODIS_Terra_CorrectedReflectance_TrueColor` | Terra | Daily | 2000–present global true color |
| `MODIS_Aqua_CorrectedReflectance_TrueColor` | Aqua | Daily | 2002–present afternoon color |
| `VIIRS_SNPP_CorrectedReflectance_TrueColor` | Suomi NPP | Daily | 2012–present high resolution |
| `MODIS_Terra_CorrectedReflectance_Bands721` | Terra | Daily | False color (snow, ice, fires) |
| `GOES-East_ABI_GeoColor` | GOES-16 | 10m / 1h | Americas full-disk true color |
| `GOES-West_ABI_GeoColor` | GOES-18 | 10m / 1h | Pacific full-disk true color |

---

## 4. Step-by-Step Workflow

### Example A: 1-Year Daily Global Time Series (MODIS Terra)

```bash
# 1. Download 365 daily global maps (128x128)
python3 tools/fetch_satellite_dataset.py \
    --source worldview \
    --layer MODIS_Terra_CorrectedReflectance_TrueColor \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --size 128 \
    --output earth_2023_128x128.fits

# 2. Run multi-tile clustering
./build/gric-cluster 2.5 -maxcl 5000 -tiles 2x2 -ncpu 4 -clustered earth_2023_128x128.fits
```

### Example B: 1-Month Hourly Geostationary Time Series (GOES-East)

```bash
# 1. Download ~720 hourly frames of Americas geostationary full-disk
python3 tools/fetch_satellite_dataset.py \
    --source worldview \
    --layer GOES-East_ABI_GeoColor \
    --cadence 1h \
    --bbox -80,-140,80,-20 \
    --start 2023-06-01 \
    --end 2023-06-30 \
    --size 128 \
    --output goes_east_hourly.fits

# 2. Run multi-tile clustering
./build/gric-cluster 2.5 -maxcl 5000 -tiles 2x2 -ncpu 4 -clustered goes_east_hourly.fits
```

---

## 5. Analyzing Clustering Results

Clustering produces structured outputs in the `<filename>.clusterdat/` directory:

* `frame_membership.txt`: Active cluster ID assignment for each sequential frame across all tiles.
* `tuple_history.txt`: Global multi-tile joint state tuples $(c_0, c_1, c_2, c_3)$ over time.
* `dcc.txt`: Inter-cluster pairwise metric distance matrix ($D_{CC}$).
* `cluster_run.log`: Execution metrics including throughput (fps), RMS fit, and distance calls.

---

## 6. Offline & Synthetic Planet Generation Mode

In isolated environments or without internet access, `tools/fetch_satellite_dataset.py`
provides a procedural 3D rotating planetary Earth generator (`--source synthetic`):

```bash
# 1. Generate 10,000-frame synthetic Earth FITS cube
python3 tools/fetch_satellite_dataset.py \
    --source synthetic \
    --size 128 \
    --max-frames 10000 \
    --output earth_synthetic_128x128.fits

# 2. Run clustering with 2x2 spatial tiling
./build/gric-cluster 2.5 -maxcl 5000 -tiles 2x2 -ncpu 4 -clustered earth_synthetic_128x128.fits
```
