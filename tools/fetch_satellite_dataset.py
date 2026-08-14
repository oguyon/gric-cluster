#!/usr/bin/env python3
"""
tools/fetch_satellite_dataset.py
Downloads satellite Earth observation image time series (e.g. NASA DSCOVR/EPIC)
over long periods (e.g. 1-2 years, ~10,000 frames) or generates high-fidelity
synthetic rotating planetary Earth cubes, formatting them into standard 3D FITS
cubes for streaming clustering with gric-cluster.

Usage:
  python3 tools/fetch_satellite_dataset.py --start 2023-01-01 --end 2023-12-31 \
      --size 128 --max-frames 10000 --output earth_2023_128x128.fits --run-demo
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
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


def print_error(msg):
    """Print an error message formatted in bright red."""
    print(f"{COLOR_RED}{msg}{COLOR_RESET}", file=sys.stderr)


def print_warning(msg):
    """Print a warning message formatted in yellow."""
    print(f"{COLOR_YELLOW}{msg}{COLOR_RESET}", file=sys.stderr)


try:
    import numpy as np
    from PIL import Image
    from astropy.io import fits
except ImportError as e:
    print_error(f"Error: Missing required Python packages ({e}). "
                f"Please run: pip install numpy pillow astropy")
    sys.exit(1)

EPIC_API_DATES_URL = "https://epic.gsfc.nasa.gov/api/natural/all"
EPIC_API_DATE_URL = "https://epic.gsfc.nasa.gov/api/natural/date/{date_str}"
EPIC_IMG_URL = "https://epic.gsfc.nasa.gov/archive/natural/{date_path}/{ext}/{image_id}.{ext}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download or generate satellite Earth time series for gric-cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source", type=str, default="epic", choices=["epic", "synthetic"],
        help="Data source: 'epic' (NASA DSCOVR/EPIC satellite) or 'synthetic' (procedural Earth)"
    )
    parser.add_argument(
        "--start", type=str, default="2023-01-01",
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end", type=str, default="2023-12-31",
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--size", type=str, default="128",
        help="Image size in pixels (e.g. '128' for 128x128, or '64x64')"
    )
    parser.add_argument(
        "--max-frames", type=int, default=10000,
        help="Maximum number of frames to download/generate"
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Subsampling step (e.g. 1 = every frame, 2 = every 2nd frame)"
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Number of concurrent download threads"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="earth_satellite_128x128.fits",
        help="Output FITS file path"
    )
    parser.add_argument(
        "--format", type=str, default="jpg", choices=["jpg", "png", "thumbs"],
        help="Image quality format from EPIC archive ('jpg', 'png', or 'thumbs')"
    )
    parser.add_argument(
        "--timeout", type=int, default=15,
        help="HTTP request timeout in seconds"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Inspect date list and frame counts without downloading"
    )
    parser.add_argument(
        "--run-demo", action="store_true",
        help="Execute gric-cluster on the generated FITS cube after creation"
    )
    parser.add_argument(
        "--rlim", type=str, default="2.5",
        help="Radius threshold rlim for the demo clustering run"
    )
    parser.add_argument(
        "--tiles", type=str, default="2x2",
        help="Multi-tile layout for the demo clustering run (e.g. '2x2' or '1x1')"
    )
    parser.add_argument(
        "--ncpu", type=str, default="4",
        help="OpenMP thread count for the demo clustering run"
    )
    return parser.parse_args()


def parse_size(size_str):
    if 'x' in size_str.lower():
        parts = size_str.lower().split('x')
        return int(parts[0]), int(parts[1])
    val = int(size_str)
    return val, val


def fetch_url_json(url, timeout=15, max_retries=3):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (gric-cluster)'})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.3 * (attempt + 1))


def fetch_epic_catalog(start_date, end_date, timeout=15):
    print("Querying NASA EPIC date catalog...")
    all_dates_meta = fetch_url_json(EPIC_API_DATES_URL, timeout=timeout)
    all_dates = {item["date"] for item in all_dates_meta}

    cur_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    target_dates = []
    while cur_dt <= end_dt:
        d_str = cur_dt.strftime("%Y-%m-%d")
        if d_str in all_dates:
            target_dates.append(d_str)
        cur_dt += timedelta(days=1)

    print(f"Found {len(target_dates)} observation dates between {start_date} and {end_date}.")
    return target_dates


def fetch_day_metadata(date_str, timeout=15):
    url = EPIC_API_DATE_URL.format(date_str=date_str)
    try:
        items = fetch_url_json(url, timeout=timeout, max_retries=3)
        return date_str, items
    except Exception as e:
        print_error(f"Error: Failed to fetch metadata for {date_str}: {e}")
        return date_str, []


def download_epic_frame(item, target_size, img_format, timeout=20):
    img_id = item["image"]
    date_str = item["date"].split()[0]
    date_path = date_str.replace("-", "/")
    ext = "jpg" if img_format != "png" else "png"
    sub_folder = img_format

    img_url = EPIC_IMG_URL.format(
        date_path=date_path,
        ext=ext if sub_folder != "thumbs" else "jpg",
        sub_folder=sub_folder,
        image_id=img_id
    )
    if sub_folder == "thumbs":
        img_url = (f"https://epic.gsfc.nasa.gov/archive/natural/{date_path}/thumbs/"
                   f"{img_id}.jpg")

    req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0 (gric-cluster)'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                img = Image.open(io.BytesIO(data)).convert('L')
                img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                return arr, item["date"], None
        except Exception as e:
            if attempt == 2:
                return None, item.get("date", ""), str(e)
            time.sleep(0.5 * (attempt + 1))
    return None, item.get("date", ""), "Unknown error"


def generate_synthetic_earth_cube(n_frames, size, start_date="2023-01-01"):
    w, h = size
    y, x = np.ogrid[:h, :w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = min(w, h) * 0.42

    dx = (x - cx) / r
    dy = (y - cy) / r
    dist2 = dx**2 + dy**2
    mask = dist2 <= 1.0
    dz = np.sqrt(np.maximum(0.0, 1.0 - dist2))

    cube = np.zeros((n_frames, h, w), dtype=np.float32)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    print(f"Generating {n_frames:,} synthetic planetary Earth frames ({w}x{h})...")
    t0 = time.time()

    for f in range(n_frames):
        lon_rot = (f * 0.25) % (2.0 * np.pi)
        x_rot = dx * np.cos(lon_rot) - dz * np.sin(lon_rot)
        z_rot = dx * np.sin(lon_rot) + dz * np.cos(lon_rot)
        y_rot = dy

        continents = (
            np.sin(3.0 * x_rot) * np.cos(2.5 * y_rot) +
            0.5 * np.cos(5.0 * z_rot) * np.sin(4.0 * y_rot) +
            0.3 * np.sin(7.0 * x_rot + 3.0 * y_rot)
        ) > 0.25

        albedo = np.where(continents, 0.45, 0.15)
        albedo = np.where(np.abs(dy) > 0.78, 0.85, albedo)

        cloud_rot = (f * 0.18 + 0.5 * np.sin(dy * 4.0)) % (2.0 * np.pi)
        xc_rot = dx * np.cos(cloud_rot) - dz * np.sin(cloud_rot)
        clouds = np.maximum(0.0, np.sin(6.0 * xc_rot) * np.cos(4.0 * y_rot) - 0.2) * 1.2
        albedo = np.minimum(1.0, albedo + clouds * 0.5)

        solar_tilt = 0.4 * np.sin(2.0 * np.pi * f / float(max(1, n_frames)))
        sun_x, sun_y, sun_z = 0.3, solar_tilt, 1.0
        sun_norm = np.sqrt(sun_x**2 + sun_y**2 + sun_z**2)
        sun_x, sun_y, sun_z = sun_x / sun_norm, sun_y / sun_norm, sun_z / sun_norm

        illum = np.maximum(0.0, dx * sun_x + dy * sun_y + dz * sun_z)

        frame = np.zeros((h, w), dtype=np.float32)
        frame[mask] = albedo[mask] * illum[mask]
        cube[f] = frame

    print(f"Generated {n_frames:,} frames in {time.time() - t0:.2f}s.")
    end_dt = start_dt + timedelta(hours=f)
    return cube, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def main():
    args = parse_args()
    w, h = parse_size(args.size)

    print("==================================================")
    print(" GRIC Satellite Earth Observation Dataset Builder")
    print("==================================================")
    print(f"Source Mode:   {args.source.upper()}")
    print(f"Time Window:   {args.start} to {args.end}")
    print(f"Image Size:    {w} x {h} pixels (grayscale normalized [0, 1])")
    print(f"Max Frames:    {args.max_frames:,} (step: {args.step})")
    print(f"Target Output: {args.output}")
    print(f"Workers:       {args.workers} threads")
    print("==================================================")

    cube = None
    date_beg, date_end = args.start, args.end
    n_frames = 0

    if args.source == "epic":
        try:
            dates = fetch_epic_catalog(args.start, args.end, timeout=args.timeout)
        except Exception as e:
            print_error(f"\nWarning: Could not connect to NASA EPIC API ({e}).")
            print(f"{COLOR_YELLOW}Network note: Direct outbound connection to NASA was blocked "
                  f"or unreachable.{COLOR_RESET}")
            print(f"{COLOR_CYAN}Falling back to high-fidelity synthetic planetary Earth "
                  f"simulation...{COLOR_RESET}")
            args.source = "synthetic"
            dates = []

    if args.source == "epic" and dates:
        print(f"\nQuerying metadata for {len(dates)} dates in parallel...")
        all_images = []
        with ThreadPoolExecutor(max_workers=min(16, args.workers)) as executor:
            futures = [executor.submit(fetch_day_metadata, d, args.timeout) for d in dates]
            for f in as_completed(futures):
                d_str, items = f.result()
                all_images.extend(items)

        all_images.sort(key=lambda x: x["date"])
        selected_images = all_images[::args.step][:args.max_frames]
        n_frames = len(selected_images)
        est_size_mb = (n_frames * w * h * 4) / (1024 * 1024)

        print(f"\nTotal frames cataloged: {len(all_images):,}")
        print(f"Frames selected for download: {n_frames:,}")
        print(f"Estimated FITS file size: {est_size_mb:.2f} MB")

        if args.dry_run:
            print("\n[Dry Run]: Inspection complete. No files downloaded.")
            if selected_images:
                first_s = selected_images[0]
                last_s = selected_images[-1]
                print(f"Earliest sample: {first_s['date']} ({first_s['image']})")
                print(f"Latest sample:   {last_s['date']} ({last_s['image']})")
            return

        if n_frames > 0:
            print(f"\nDownloading {n_frames:,} frames ({args.workers} concurrent workers)...")
            cube = np.zeros((n_frames, h, w), dtype=np.float32)
            t_start = time.time()
            downloaded_count = 0

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_idx = {
                    executor.submit(download_epic_frame, itm, (w, h), args.format,
                                    args.timeout): i
                    for i, itm in enumerate(selected_images)
                }
                last_report = time.time()
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    arr, ts, err = future.result()
                    if arr is not None:
                        cube[idx] = arr
                        downloaded_count += 1
                    else:
                        print_error(f"\nWarning: Frame {idx} failed to download: {err}")

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
            date_beg = selected_images[0]["date"]
            date_end = selected_images[-1]["date"]

    if cube is None or n_frames == 0:
        n_frames = args.max_frames
        est_size_mb = (n_frames * w * h * 4) / (1024 * 1024)
        print(f"\nSynthetic mode: {n_frames:,} frames ({w}x{h}) ~ {est_size_mb:.2f} MB")
        if args.dry_run:
            print("[Dry Run]: Synthetic planet simulation ready for generation.")
            return
        cube, date_beg, date_end = generate_synthetic_earth_cube(n_frames, (w, h), args.start)

    # Write FITS Cube
    print(f"\nWriting FITS cube to {args.output}...")
    hdu = fits.PrimaryHDU(cube)
    hdr = hdu.header
    hdr["SOURCE"] = ("NASA DSCOVR/EPIC" if args.source == "epic" else "Synthetic Planet Earth",
                     "Data provider")
    hdr["SATELLIT"] = ("DSCOVR" if args.source == "epic" else "Simulation", "Platform")
    hdr["INSTRUME"] = ("EPIC" if args.source == "epic" else "Procedural", "Instrument")
    hdr["DATE-BEG"] = (date_beg, "Earliest observation timestamp")
    hdr["DATE-END"] = (date_end, "Latest observation timestamp")
    hdr["NFRAMES"] = (n_frames, "Number of 2D image frames in cube")
    hdr["NAXIS1"] = (w, "Image width in pixels")
    hdr["NAXIS2"] = (h, "Image height in pixels")
    hdr["NAXIS3"] = (n_frames, "Time axis length")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(out_path, overwrite=True)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Successfully generated FITS file: {out_path} ({file_size_mb:.2f} MB)")

    # Optional Auto-Demo Run
    if args.run_demo:
        gric_exe = Path(__file__).resolve().parent.parent / "build" / "gric-cluster"
        if not gric_exe.exists():
            print_warning(f"Warning: Executable not found at {gric_exe}. Skipping demo run.")
            return

        cmd = [
            str(gric_exe), args.rlim,
            "-maxcl", "5000",
            "-maxim", str(n_frames),
            "-tiles", args.tiles,
            "-ncpu", args.ncpu,
            "-clustered",
            str(out_path)
        ]
        print("\n==================================================")
        print(" Launching GRIC-Cluster Demo Run")
        print(f" Command: {' '.join(cmd)}")
        print("==================================================")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
