#!/usr/bin/env python
"""
Benchmark script for gric-cluster and gric-locate, focusing on performance
and distance computation logging.
"""

import os
import subprocess
import time
import sys

# Ensure local gric package is prioritized for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import gric.image_cluster as gric_ic

# --- Helper to locate binaries ---
build_dir = os.path.join(project_root, 'build')

def get_binary_path(name):
    return os.path.join(build_dir, name)

if not os.path.isdir(build_dir):
    print(f"Build directory not found at '{build_dir}'")
    print("Please build the C project first (run 'mkdir build && cd build && cmake .. && make' in the project root).")
    sys.exit(1)
else:
    print(f"Using binaries from: {build_dir}")

# --- Input Files ---
# Assuming cube_test and cube_test2 are in the 'test/' directory
# relative to the project root.
# Input Files from the gric/ directory
gric_data_dir = os.path.join(project_root, 'gric')
cube_test_file = os.path.join(gric_data_dir, 'cube.fits')
cube_test2_file = os.path.join(gric_data_dir, 'imgsci.fits')

# --- Output Directory for benchmark results ---
benchmark_output_dir = os.path.join(project_root, 'out_benchmark')
os.makedirs(benchmark_output_dir, exist_ok=True)


print(f"\n--- Running gric-cluster benchmark on {os.path.basename(cube_test_file)} ---")
try:
    cluster_map_tool = gric_ic.ImageCluster(
        rlim=0.005,
        binary_path=get_binary_path("gric-cluster"),
        anchors=True, # Create anchors.fits for gric-locate
        avg=True      # Create average_xxxx.fits files
    )
    cluster_results_dir = os.path.join(benchmark_output_dir, "cube_test.clusterdat")
    cluster_map_results = cluster_map_tool.run(cube_test_file, output_dir=cluster_results_dir)

    print("gric-cluster complete!")
    print(f"  - Total clusters found: {cluster_map_results.get('total_clusters')}")
    print(f"  - Output map directory: {cluster_map_results.get('output_dir')}")
    # Additional benchmark metrics can be added here from cluster_map_results
    # e.g., print(f"  - Clustering time: {cluster_map_results.get('time_ms')} ms")

except Exception as e:
    print(f"Error during gric-cluster benchmark: {e}", file=sys.stderr)
    sys.exit(1)

print(f"\n--- Running gric-locate benchmark on {os.path.basename(cube_test2_file)} ---")
try:
    # Paths for anchors and dcc are relative to the cluster_results_dir
    anchors_path = os.path.join(cluster_results_dir, 'anchors.fits')
    dcc_path = os.path.join(cluster_results_dir, 'dcc.txt')

    locate_results = gric_ic.locate_points(
        anchors_file=anchors_path,
        dcc_file=dcc_path,
        new_input_file=cube_test2_file,
        num_neighbors=1,
        binary_path=get_binary_path("gric-locate"),
        output_dir=cluster_results_dir # This is where locate_run.log will be written
    )

    print("gric-locate complete!")
    locate_stats = locate_results['locate_stats']
    print(f"  - Total frames processed: {locate_stats['total_frames_processed']}")
    print(f"  - Distance computation histogram: {locate_stats['distance_histogram']}")
    print(f"  - First 5 frame-to-cluster associations: {list(locate_results['locations'].items())[:5]}")
    # Additional benchmark metrics can be added here from locate_results

    # Generate the histogram plot for gric-locate results
    print("\n--- Generating gric-locate histogram plot ---")
    locate_log_file = os.path.join(cluster_results_dir, 'locate_run.log')
    locate_plot_output_path = os.path.join(benchmark_output_dir, 'locate_benchmark_histogram.png')

    plot_output_path_locate = gric_ic.plot_locate_results(
        log_file=locate_log_file,
        output_file=locate_plot_output_path,
        binary_path=get_binary_path("gric-locate-plot")
    )
    if plot_output_path_locate:
        print(f"gric-locate results histogram saved to: {plot_output_path_locate}")
    else:
        print("Failed to generate gric-locate results histogram or retrieve output path.")

except Exception as e:
    print(f"Error during gric-locate benchmark or plot generation: {e}", file=sys.stderr)
    sys.exit(1)

print("\nBenchmark script finished.")
