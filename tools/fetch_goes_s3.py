#!/usr/bin/env python3
"""
tools/fetch_goes_s3.py
Downloads NOAA GOES-16/18 Full-Disk NetCDF observation files from public
unauthenticated AWS S3 buckets (NOAA Open Data Dissemination - NODD),
extracts thermal infrared (Channel 13) or visible (Channel 2) bands,
and builds a 128x128 3D FITS cube with >10,000 frames for gric-cluster.

Usage:
  python3 tools/fetch_goes_s3.py --satellite goes16 \
      --start 2023-06-01 --end 2023-08-10 --size 128 --max-frames 10000 \
      --output goes16_10k_128x128.fits
"""

import argparse
import io
import os
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

# ANSI Color Codes
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"

# Mutex to ensure non-thread-safe C NetCDF4 / HDF5 libraries do not corrupt heap
NETCDF_C_LOCK = threading.Lock()


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download historical NOAA GOES-16/18 data from public AWS S3 buckets.",
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
        help="Spectral band (CMI_C13: Clean Thermal IR 10.3um 24/7, CMI_C02: Red Visible 0.64um)"
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
        "--workers", type=int, default=8,
        help="Concurrent download threads"
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


def list_s3_keys_for_hour(bucket, year, day_of_year, hour):
    prefix = f"ABI-L2-MCMIPF/{year}/{day_of_year:03d}/{hour:02d}/"
    url = f"https://{bucket}.s3.amazonaws.com/?prefix={prefix}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (gric-cluster)'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                tree = ET.fromstring(resp.read())
                ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
                return [
                    elem.text for elem in tree.findall('.//s3:Key', ns)
                    if elem.text and elem.text.endswith('.nc')
                ]
        except Exception:
            time.sleep(0.3 * (attempt + 1))
    return []


def read_netcdf_variable(data_bytes, channel):
    """Extract and normalize 2D array from in-memory NetCDF bytes (mutex-protected)."""
    tid = threading.get_ident()
    mem_name = f"mem_{tid}_{int(time.time() * 1000)}.nc"

    with NETCDF_C_LOCK:
        if HAVE_NETCDF4:
            with netCDF4.Dataset(mem_name, memory=data_bytes) as nc:
                if channel not in nc.variables:
                    return None, f"Channel {channel} not found in NetCDF"
                var = np.array(nc.variables[channel][:], dtype=np.float32)
        elif HAVE_H5PY:
            with h5py.File(io.BytesIO(data_bytes), 'r') as h5f:
                if channel not in h5f:
                    return None, f"Channel {channel} not found in HDF5"
                var = np.array(h5f[channel][:], dtype=np.float32)
        else:
            return None, "Missing NetCDF library. Please run: pip install netCDF4"

    valid_mask = ~np.isnan(var) & (var > -999.0)
    if not np.any(valid_mask):
        return np.zeros((128, 128), dtype=np.float32), None

    v_min, v_max = np.percentile(var[valid_mask], (1.0, 99.0))
    denom = max(1e-5, (v_max - v_min))
    norm = np.clip((var - v_min) / denom, 0.0, 1.0)
    norm[~valid_mask] = 0.0
    return norm, None


def process_s3_file(bucket, key, channel, target_size):
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (gric-cluster)'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data_bytes = resp.read()
                raw_norm, err = read_netcdf_variable(data_bytes, channel)
                if raw_norm is None:
                    return None, key, err

                im = Image.fromarray((raw_norm * 255.0).astype(np.uint8))
                im_res = im.resize((target_size, target_size), Image.Resampling.BILINEAR)
                arr = np.array(im_res, dtype=np.float32) / 255.0
                return arr, key, None
        except Exception as e:
            if attempt == 2:
                return None, key, str(e)
            time.sleep(0.5 * (attempt + 1))
    return None, key, "Download failed"


def main():
    args = parse_args()
    if not (HAVE_NETCDF4 or HAVE_H5PY) and not args.dry_run:
        print_error("Error: Python package 'netCDF4' (or 'h5py') is required to read GOES data.")
        print("Please install it with: pip install --user --break-system-packages netCDF4")
        sys.exit(1)

    bucket = f"noaa-{args.satellite}"
    cur_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    print("==================================================")
    print(" NOAA GOES-16/18 AWS S3 Downloader & FITS Builder")
    print("==================================================")
    print(f"Bucket:        s3://{bucket}/ (NOAA Open Data Dissemination)")
    print(f"Time Window:   {args.start} to {args.end}")
    print(f"Channel:       {args.channel}")
    print(f"Image Size:    {args.size} x {args.size} pixels")
    print(f"Max Frames:    {args.max_frames:,} (step: {args.step})")
    print(f"Target Output: {args.output}")
    print(f"Workers:       {args.workers} threads")
    print("==================================================")

    print("Querying NOAA AWS S3 catalog across hourly folders...")
    hour_slots = []
    while cur_dt <= end_dt:
        doy = cur_dt.timetuple().tm_yday
        for h in range(24):
            hour_slots.append((cur_dt.year, doy, h))
        cur_dt += timedelta(days=1)

    all_keys = []
    with ThreadPoolExecutor(max_workers=min(16, args.workers * 2)) as ex:
        futures = [ex.submit(list_s3_keys_for_hour, bucket, y, d, h) for y, d, h in hour_slots]
        for f in as_completed(futures):
            all_keys.extend(f.result())

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

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_idx = {
            ex.submit(process_s3_file, bucket, k, args.channel, args.size): i
            for i, k in enumerate(selected_keys)
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
            if now - last_report > 1.0 or downloaded_count == n_frames:
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
    hdr["SOURCE"] = (f"NOAA {args.satellite.upper()} ({args.channel})", "Data provider")
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
