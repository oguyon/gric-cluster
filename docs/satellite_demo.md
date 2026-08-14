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

| Layer Identifier | Satellite | Cadence | Retention Archive |
| :--- | :--- | :--- | :--- |
| `MODIS_Terra_CorrectedReflectance_TrueColor` | Terra | Daily (`1d`) | 2000–present (Permanent) |
| `MODIS_Aqua_CorrectedReflectance_TrueColor` | Aqua | Daily (`1d`) | 2002–present (Permanent) |
| `VIIRS_SNPP_CorrectedReflectance_TrueColor` | Suomi NPP | Daily (`1d`) | 2012–present (Perm.) |
| `MODIS_Terra_CorrectedReflectance_Bands721` | Terra | Daily (`1d`) | 2000–present (Permanent) |
| `GOES-East_ABI_GeoColor` | GOES-16 | 10m / 1h | Last 90 days (Rolling) |
| `GOES-West_ABI_GeoColor` | GOES-18 | 10m / 1h | Last 90 days (Rolling) |

> **Note on Geostationary Retention**: Geostationary layers (GOES-East/West, Himawari) produce
> massive data volumes at 10-minute cadence, so NASA GIBS retains them for the **rolling last
> 90 days**. For dates within the last 90 days, real high-rate images are returned; older dates
> return blank placeholders. For historical multi-year time series (2000–present), use daily
> polar-orbiting layers (MODIS Terra/Aqua or VIIRS).

---

## 4. Step-by-Step Workflow

### Example A: 1-Year Daily Global Time Series (MODIS Terra, Historical 2023)

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

### Example B: 1-Week Hourly Geostationary Time Series (GOES-East, Recent Date)

```bash
# 1. Download 168 hourly frames of Americas geostationary full-disk
python3 tools/fetch_satellite_dataset.py \
    --source worldview \
    --layer GOES-East_ABI_GeoColor \
    --cadence 1h \
    --start 2026-08-01 \
    --end 2026-08-07 \
    --size 128 \
    --output goes_east_hourly.fits

# 2. Run multi-tile clustering
./build/gric-cluster 2.5 -maxcl 5000 -tiles 2x2 -ncpu 4 -clustered goes_east_hourly.fits
```

### Example C: >10,000 Historical Frames from NOAA AWS S3 (NODD)

```bash
# 1. Download ~70 days of 10-minute continuous thermal infrared (10,000 frames)
python3 tools/fetch_goes_s3.py \
    --satellite goes16 \
    --start 2023-06-01 \
    --end 2023-08-10 \
    --channel CMI_C13 \
    --size 128 \
    --max-frames 10000 \
    --output goes16_10k_128x128.fits

# 2. Run multi-tile clustering
./build/gric-cluster 2.5 -maxcl 5000 -tiles 2x2 -ncpu 8 -clustered goes16_10k_128x128.fits
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
