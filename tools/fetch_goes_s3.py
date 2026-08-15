#!/usr/bin/env python3
"""
tools/fetch_goes_s3.py
High-performance downloader for NOAA GOES-16/18 Full-Disk NetCDF observation files
from public unauthenticated AWS S3 buckets (NOAA Open Data Dissemination - NODD).
Extracts thermal infrared (e.g. Channel 13) or visible (e.g. Channel 2) bands,
normalizes frames, and builds a 3D FITS cube for gric-cluster.

Optimizations:
  1. Single-Channel S3 Targeting (ABI-L2-CMIPF): ~22 MB/file vs ~267 MB (12x less data).
  2. Multi-Process Parallelism: Bypasses NetCDF/HDF5 C-library locks and GIL.
  3. Persistent HTTP Sessions: Eliminates TCP/TLS handshake overhead per request.
  4. Direct Strided NetCDF Slicing: Subsamples during read for fast resizing.
  5. Fast Percentile Estimation: Subsampled percentile computation on valid Earth pixels.
  6. Filtered S3 Catalog Indexing: Queries hourly folders with specific channel prefix.

Usage:
  python3 tools/fetch_goes_s3.py --satellite goes16 \
      --start 2023-06-01 --end 2023-07-01 --channel CMI_C13 --size 512 \
      --max-frames 10000 --output goes16_512.fits --workers 16
"""

import argparse
import io
import multiprocessing as mp
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

# ANSI Color Codes
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"


def print_error(msg):
    print(f"{COLOR_RED}{msg}{COLOR_RESET}", file=sys.stderr)


def print_warning(msg):
    print(f"{COLOR_YELLOW}{msg}{COLOR_RESET}", file=sys.stderr)


try:
    import numpy as np
    from PIL import Image
    from astropy.io import fits
except ImportError as e:
    print_error(f"Error: Missing required Python packages ({e}). "
                f"Please run: pip install numpy pillow astropy")
    sys.exit(1)

# Check for requests
HAVE_REQUESTS = False
try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    pass

# Check for NetCDF loader (netCDF4 or h5py)
HAVE_NETCDF4 = False
HAVE_H5PY = False
try:
    import netCDF4
    HAVE_NETCDF4 = True
except ImportError:
    try:
        import h5py
        HAVE_H5PY = True
    except ImportError:
        pass

# Global session variable for worker processes
WORKER_SESSION = None


def init_worker_session():
    """Initialize a persistent HTTP session for each worker process."""
    global WORKER_SESSION
    if HAVE_REQUESTS:
        WORKER_SESSION = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=2)
        WORKER_SESSION.mount("https://", adapter)
        WORKER_SESSION.mount("http://", adapter)


def parse_channel_config(channel_str, product_override=None):
    """
    Parse channel identifier (e.g. 'CMI_C13', 'C13', '13', 'CMI_C02')
    into channel number, channel string, variable name, and product prefix.
    """
    m = re.search(r"(\d+)", channel_str)
    if m:
        ch_num = int(m.group(1))
    else:
        ch_num = 13  # Default to Channel 13 (Clean IR Longwave Window 10.3 um)

    ch_str = f"C{ch_num:02d}"
    var_mcmip = f"CMI_{ch_str}"

    if product_override:
        prod = product_override.upper()
    else:
        # Default to single-channel CMIPF (Full Disk) for 12x smaller downloads
        prod = "CMIPF"

    return ch_num, ch_str, var_mcmip, prod


def parse_args():
    default_workers = min(24, max(4, (os.cpu_count() or 8)))
    parser = argparse.ArgumentParser(
        description="Download historical NOAA GOES-16/18 data from public AWS S3 buckets (accelerated).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--satellite", default="goes16", choices=["goes16", "goes18"],
        help="NOAA satellite constellation (goes16 for East, goes18 for West)"
    )
    parser.add_argument(
        "--start", default="2023-06-01",
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end", default="2023-08-10",
        help="End date in YYYY-MM-DD format (70 days = ~10,000 frames at 10-min cadence)"
    )
    parser.add_argument(
        "--channel", default="CMI_C13",
        help="Spectral band (e.g. CMI_C13 for Clean Thermal IR 10.3um, CMI_C02 for Red 0.64um)"
    )
    parser.add_argument(
        "--product", default="auto", choices=["auto", "CMIPF", "MCMIPF", "CMIPC"],
        help="S3 product: auto (CMIPF for single channel, ~22MB/file), CMIPF, MCMIPF (~267MB/file), CMIPC (CONUS)"
    )
    parser.add_argument(
        "--size", type=int, default=128,
        help="Image size in pixels (e.g. 128 for 128x128, or 512 for 512x512)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=10000,
        help="Maximum number of frames to download"
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Subsampling step (1 = all files, 2 = every 2nd file)"
    )
    parser.add_argument(
        "--workers", type=int, default=default_workers,
        help="Concurrent download and decoding worker processes"
    )
    parser.add_argument(
        "--output", "-o", default="goes16_10k_128x128.fits",
        help="Output FITS file path"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List available files without downloading"
    )
    return parser.parse_args()


def list_s3_keys_for_hour(args):
    """Query AWS S3 REST API for NetCDF keys in a given hour slot."""
    bucket, year, day_of_year, hour, prod, ch_str = args
    if prod == "MCMIPF":
        prefix = f"ABI-L2-MCMIPF/{year}/{day_of_year:03d}/{hour:02d}/"
    else:
        prefix = f"ABI-L2-{prod}/{year}/{day_of_year:03d}/{hour:02d}/OR_ABI-L2-{prod}-M6{ch_str}"

    url = f"https://{bucket}.s3.amazonaws.com/?prefix={prefix}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (gric-cluster)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                tree = ET.fromstring(resp.read())
                ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
                return [
                    elem.text for elem in tree.findall(".//s3:Key", ns)
                    if elem.text and elem.text.endswith(".nc")
                ]
        except Exception:
            time.sleep(0.2 * (attempt + 1))
    return []


def decode_and_normalize_frame(data_bytes, channel_mcmip, target_size):
    """Extract, normalize, and resize NetCDF image array from memory bytes."""
    tid = mp.current_process().pid
    mem_name = f"mem_{tid}_{int(time.time() * 1000)}.nc"

    if HAVE_NETCDF4:
        with netCDF4.Dataset(mem_name, memory=data_bytes) as nc:
            # Single-channel files use 'CMI', multi-channel files use 'CMI_C13'
            if "CMI" in nc.variables:
                var = nc.variables["CMI"]
            elif channel_mcmip in nc.variables:
                var = nc.variables[channel_mcmip]
            else:
                cmi_candidates = [v for v in nc.variables if v.startswith("CMI")]
                if cmi_candidates:
                    var = nc.variables[cmi_candidates[0]]
                else:
                    return None, f"Channel variable '{channel_mcmip}' not found in NetCDF"

            shape = var.shape
            # Efficient strided read if raw resolution is much larger than target size
            step = max(1, shape[0] // (target_size * 2))
            raw_var = var[::step, ::step]
    elif HAVE_H5PY:
        with h5py.File(io.BytesIO(data_bytes), "r") as h5f:
            if "CMI" in h5f:
                var = h5f["CMI"]
            elif channel_mcmip in h5f:
                var = h5f[channel_mcmip]
            else:
                return None, f"Channel variable '{channel_mcmip}' not found in HDF5"

            shape = var.shape
            step = max(1, shape[0] // (target_size * 2))
            raw_var = var[::step, ::step]
    else:
        return None, "Missing NetCDF library. Please run: pip install netCDF4"

    # Distinguish valid Earth pixels from masked space background
    if hasattr(raw_var, "mask") and np.ma.is_masked(raw_var):
        earth_mask = ~raw_var.mask
        earth_data = np.array(raw_var.data, dtype=np.float32)
    else:
        earth_mask = ~np.isnan(raw_var) & (raw_var > 0.0)
        earth_data = np.array(raw_var, dtype=np.float32)

    valid_vals = earth_data[earth_mask]
    if len(valid_vals) == 0:
        return np.zeros((target_size, target_size), dtype=np.float32), None

    # Subsampled fast percentile estimation for performance
    sample_vals = valid_vals[::4] if len(valid_vals) > 10000 else valid_vals
    v_min, v_max = np.percentile(sample_vals, (1.0, 99.0))
    denom = max(1e-5, (v_max - v_min))

    # Earth disk normalized to [0.05, 1.0], space background is 0.0
    norm = np.zeros(earth_data.shape, dtype=np.float32)
    norm[earth_mask] = np.clip((valid_vals - v_min) / denom, 0.05, 1.0)

    # High-quality bilinear resize to target resolution
    im = Image.fromarray((norm * 255.0).astype(np.uint8))
    im_res = im.resize((target_size, target_size), Image.Resampling.BILINEAR)
    arr = np.array(im_res, dtype=np.float32) / 255.0
    return arr, None


def process_s3_task(args):
    """Worker task: Download NetCDF from S3 and decompress/resize in parallel."""
    global WORKER_SESSION
    bucket, key, channel_mcmip, target_size = args
    url = f"https://{bucket}.s3.amazonaws.com/{key}"

    data_bytes = None
    for attempt in range(3):
        try:
            if WORKER_SESSION is not None:
                resp = WORKER_SESSION.get(url, timeout=30)
                if resp.status_code == 200:
                    data_bytes = resp.content
                    break
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (gric-cluster)"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data_bytes = resp.read()
                    break
        except Exception:
            if attempt == 2:
                return None, key, "HTTP download failed after 3 attempts"
            time.sleep(0.3 * (attempt + 1))

    if data_bytes is None:
        return None, key, "Empty data received"

    arr, err = decode_and_normalize_frame(data_bytes, channel_mcmip, target_size)
    return arr, key, err


def main():
    args = parse_args()
    if not (HAVE_NETCDF4 or HAVE_H5PY) and not args.dry_run:
        print_error("Error: Python package 'netCDF4' (or 'h5py') is required to read GOES data.")
        print("Please install it with: pip install --user --break-system-packages netCDF4")
        sys.exit(1)

    product_override = None if args.product == "auto" else args.product
    ch_num, ch_str, var_mcmip, prod = parse_channel_config(args.channel, product_override)
    bucket = f"noaa-{args.satellite}"
    cur_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    print("==================================================")
    print(" NOAA GOES-16/18 AWS S3 Downloader & FITS Builder ")
    print("==================================================")
    print(f"Bucket:        s3://{bucket}/ (NOAA Open Data Dissemination)")
    print(f"Time Window:   {args.start} to {args.end}")
    print(f"Channel:       {args.channel} (Channel {ch_num:02d} -> S3 Product: ABI-L2-{prod})")
    print(f"Image Size:    {args.size} x {args.size} pixels")
    print(f"Max Frames:    {args.max_frames:,} (step: {args.step})")
    print(f"Target Output: {args.output}")
    print(f"Workers:       {args.workers} parallel processes (multi-process accelerated)")
    print("==================================================")

    print("Querying NOAA AWS S3 catalog across hourly folders...")
    hour_slots = []
    while cur_dt <= end_dt:
        doy = cur_dt.timetuple().tm_yday
        for h in range(24):
            hour_slots.append((bucket, cur_dt.year, doy, h, prod, ch_str))
        cur_dt += timedelta(days=1)

    all_keys = []
    with ThreadPoolExecutor(max_workers=min(32, max(8, args.workers * 2))) as ex:
        for keys in ex.map(list_s3_keys_for_hour, hour_slots):
            all_keys.extend(keys)

    # Fallback to MCMIPF if CMIPF had 0 files
    if len(all_keys) == 0 and prod == "CMIPF":
        print_warning("No single-channel files found under CMIPF, trying MCMIPF...")
        prod = "MCMIPF"
        hour_slots = [(bucket, y, d, h, prod, ch_str) for (bucket, y, d, h, _, ch_str) in hour_slots]
        with ThreadPoolExecutor(max_workers=min(32, max(8, args.workers * 2))) as ex:
            for keys in ex.map(list_s3_keys_for_hour, hour_slots):
                all_keys.extend(keys)

    all_keys.sort()
    selected_keys = all_keys[::args.step][:args.max_frames]
    n_frames = len(selected_keys)
    est_size_mb = (n_frames * args.size * args.size * 4) / (1024 * 1024)

    print(f"\nTotal NOAA GOES files cataloged: {len(all_keys):,}")
    print(f"Selected files for download:     {n_frames:,}")
    print(f"Estimated FITS file size:        {est_size_mb:.2f} MB")

    if args.dry_run:
        print("\n[Dry Run]: Inspection complete. No files downloaded.")
        if selected_keys:
            print(f"Earliest file: {selected_keys[0]}")
            print(f"Latest file:   {selected_keys[-1]}")
        return

    if n_frames == 0:
        print_error("Error: No files found in the specified date range.")
        return

    print(f"\nDownloading & processing {n_frames:,} NetCDF files ({args.workers} workers)...")
    cube = np.zeros((n_frames, args.size, args.size), dtype=np.float32)
    t_start = time.time()
    downloaded_count = 0

    # Multi-Process Pool to decompress NetCDF in parallel across all CPU cores
    mp_ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp_ctx,
        initializer=init_worker_session
    ) as ex:
        tasks = [
            (bucket, k, var_mcmip, args.size)
            for k in selected_keys
        ]
        future_to_idx = {
            ex.submit(process_s3_task, task): i
            for i, task in enumerate(tasks)
        }
        last_report = time.time()
        for f in as_completed(future_to_idx):
            idx = future_to_idx[f]
            arr, k, err = f.result()
            if arr is not None:
                cube[idx] = arr
                downloaded_count += 1
            else:
                print_error(f"\nWarning: Failed to process {k}: {err}")

            now = time.time()
            if now - last_report > 0.5 or downloaded_count == n_frames:
                elapsed = now - t_start
                rate = downloaded_count / max(0.001, elapsed)
                pct = (downloaded_count / float(n_frames)) * 100.0
                eta = (n_frames - downloaded_count) / max(0.001, rate)
                sys.stdout.write(
                    f"\r  [{downloaded_count:,}/{n_frames:,}] ({pct:5.1f}%) "
                    f"| Speed: {rate:5.1f} fps | ETA: {eta:4.1f}s"
                )
                sys.stdout.flush()
                last_report = now

    t_total = time.time() - t_start
    print(f"\nDownload completed: {downloaded_count:,} frames in {t_total:.2f}s "
          f"({downloaded_count / max(0.001, t_total):.1f} fps)")

    # Write FITS Cube
    print(f"\nWriting FITS cube to {args.output}...")
    hdu = fits.PrimaryHDU(cube)
    hdr = hdu.header
    hdr["SOURCE"] = (f"NOAA {args.satellite.upper()} (Channel {ch_num:02d})", "Data provider")
    hdr["CHANNEL"] = (args.channel, "Requested spectral channel")
    hdr["PRODUCT"] = (f"ABI-L2-{prod}", "S3 product type")
    hdr["DATE-BEG"] = (args.start, "Start date")
    hdr["DATE-END"] = (args.end, "End date")
    hdr["NFRAMES"] = (n_frames, "Number of 2D image frames in cube")
    hdr["NAXIS1"] = (args.size, "Image width in pixels")
    hdr["NAXIS2"] = (args.size, "Image height in pixels")
    hdr["NAXIS3"] = (n_frames, "Time axis length")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(out_path, overwrite=True)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Successfully generated FITS file: {out_path} ({file_size_mb:.2f} MB)")

    print("\nNext step: Run gric-cluster on this dataset:")
    print(f"  ./build/gric-cluster 2.5 -maxcl 5000 -tiles 2x2 -ncpu 8 -clustered {args.output}")


if __name__ == "__main__":
    main()
