#!/usr/bin/env python
"""
This Python script is converted from gric/user_example.ipynb.
It demonstrates how to use the gric-cluster Python wrapper.
"""

import os
import shutil
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import gric.image_cluster as gric_ic


print("Libraries imported successfully!")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
build_dir = os.path.join(project_root, 'build')

def get_binary_path(name):
    return os.path.join(build_dir, name)

if not os.path.isdir(build_dir):
    print(f"Build directory not found at '{build_dir}'")
    print("Please build the C project first (run 'mkdir build && cd build && cmake .. && make' in the project root).")
else:
    print(f"Using binaries from: {build_dir}")

reference_cube = 'gric/cube.fits'
print(f"Running clustering on reference file: {reference_cube}...")

cluster_map_tool = gric_ic.ImageCluster(
    rlim=0.005,
    binary_path=get_binary_path("gric-cluster"),
    anchors=True, # Create anchors.txt for gric-locate
    avg=True      # Create average_xxxx.fits files for plotting
)

cluster_map_results = cluster_map_tool.run(reference_cube)

print("Reference clustering complete!")
print(f"  - Total clusters found: {cluster_map_results.get('total_clusters')}")
print(f"  - Output map directory: {cluster_map_results.get('output_dir')}")



print("Loading anchor images for plotting...")
global loaded_anchors_data
loaded_anchors_data = {}
try:
    # For this notebook's scenario (cube.fits input), it's anchors.fits.
    full_anchors_path = os.path.join(cluster_map_results['output_dir'], 'anchors.fits')

    with fits.open(full_anchors_path) as hdul:
        # Assuming 3D FITS (num_clusters, height, width) or 2D FITS (height, width) for single anchor
        if hdul[0].data.ndim == 3:
            for i in range(hdul[0].data.shape[0]):
                loaded_anchors_data[i] = hdul[0].data[i, :, :]
        elif hdul[0].data.ndim == 2:
            loaded_anchors_data[0] = hdul[0].data[:, :]
        else:
            print(f"Warning: Unexpected FITS dimensions for anchors: {hdul[0].data.ndim}. Cannot load for plotting.")
except FileNotFoundError:
    print(f"Error: Anchor file not found at {full_anchors_path}. Ensure `anchors=True` was set and gric-cluster ran successfully.")
except Exception as e:
    print(f"Error loading anchor images: {e}")

print("Anchor images loaded.")

reference_input_file_for_plot = reference_cube # This was defined in Step 1
clustering_log_file = os.path.join(cluster_map_results['output_dir'], 'cluster_run.log')

print(f"Plotting reference clusters from {reference_input_file_for_plot} using log file {clustering_log_file}...")

try:
    plot_output_path_ref = gric_ic.plot_clusters(
        points_file=reference_input_file_for_plot,
        log_file=clustering_log_file,
        output_file=os.path.join(cluster_map_results['output_dir'], 'reference_clusters.png'),
        binary_path=get_binary_path("gric-plot")
    )
    if plot_output_path_ref:
        print(f"Plot saved to: {plot_output_path_ref}")
         # Display in notebook
    else:
        print("Failed to generate reference cluster plot or retrieve output path.")
except Exception as e:
    print(f"Error plotting reference clusters: {e}")


science_file = 'gric/imgsci.fits'
output_dir = cluster_map_results['output_dir']

# Ensure anchors.txt and dcc.txt paths are correctly set
anchors_path = os.path.join(output_dir, 'anchors.fits') # gric-cluster writes anchors as .txt
dcc_path = os.path.join(output_dir, 'dcc.txt')

print(f"Locating frames from {science_file}...")
locate_results = gric_ic.locate_points(
    anchors_file='./'+anchors_path,
    dcc_file=dcc_path,
    new_input_file=science_file,
    num_neighbors=1,
    binary_path=get_binary_path("gric-locate"),
    output_dir=output_dir
)

located_science_frames = locate_results['locations']

print(f"Location analysis complete. Found mappings for {len(located_science_frames)} frames.")
first_five = list(located_science_frames.items())[:5]
for frame_idx, neighbors in first_five:
    print(f"- Science frame {frame_idx} -> Closest clusters: {neighbors}")

print("\nLocate Stats:")
locate_stats = locate_results['locate_stats']
print(f"  - Total frames processed: {locate_stats['total_frames_processed']}")
print(f"  - Distance computation histogram: {locate_stats['distance_histogram']}")

print("\nPlotting gric-locate results...")
locate_log_file = os.path.join(output_dir, 'locate_run.log')
locate_plot_output_path = os.path.join(output_dir, 'locate_results.png')

try:
    plot_output_path_locate = gric_ic.plot_locate_results(
        log_file=locate_log_file,
        output_file=locate_plot_output_path,
        binary_path=get_binary_path("gric-locate-plot")
    )
    if plot_output_path_locate:
        print(f"gric-locate results plot saved to: {plot_output_path_locate}")
        # In a script, we might just save and not display interactively
        # For interactive display, you could use matplotlib.image.imread and plt.imshow
    else:
        print("Failed to generate gric-locate results plot or retrieve output path.")
except Exception as e:
    print(f"Error plotting gric-locate results: {e}")

print("Preparing data for plots...")
# Open the science FITS file to access its frames
science_hdul = fits.open(science_file)
science_data = science_hdul[0].data

# Get the sample of 5 frames we printed before
sample_frames_to_plot = list(located_science_frames.items())[:5]

def get_mean_cluster_image(cluster_id, base_dir=None):
    global loaded_anchors_data
    if cluster_id in loaded_anchors_data:
        # Reshape if necessary - anchors are loaded as (height, width)
        # Ensure consistency with plotting expectations (e.g., 2D array)
        return loaded_anchors_data[cluster_id]
    else:
        print(f"Warning: Anchor data for cluster {cluster_id} not found in loaded_anchors_data. Returning None.")
        return None

print("Data ready.")

fig, axes = plt.subplots(5, 2, figsize=(8, 20))
fig.suptitle('Comparison: Science Frame vs. Closest Mean Cluster', fontsize=16)

for i, (frame_idx, neighbors) in enumerate(sample_frames_to_plot):
    # Get science frame data
    target_image = science_data[frame_idx]

    # Get the #1 closest cluster
    closest_cluster_id, dist = neighbors[0]
    mean_image = get_mean_cluster_image(closest_cluster_id, output_dir)

    # Plot Mean Cluster
    ax = axes[i, 0]
    if mean_image is not None:
        im = ax.imshow(mean_image, cmap='viridis')
        fig.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_title(f"Mean of Cluster {closest_cluster_id}")
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Science Frame
    ax = axes[i, 1]
    im = ax.imshow(target_image, cmap='viridis')
    fig.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_title(f"Science Frame {frame_idx} (Dist: {dist:.4f})")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

fig, axes = plt.subplots(5, 3, figsize=(12, 20))
fig.suptitle('Comparison: Science Frame vs. Two Closest Mean Clusters', fontsize=16)

for i, (frame_idx, neighbors) in enumerate(sample_frames_to_plot):
    # Get science frame data
    target_image = science_data[frame_idx]

    # --- Plot #1 Closest --- #
    ax = axes[i, 0]
    c1_id, d1 = neighbors[0]
    mean1_img = get_mean_cluster_image(c1_id, output_dir)
    if mean1_img is not None:
        im = ax.imshow(mean1_img, cmap='viridis')
        fig.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_title(f"Closest: Cl. {c1_id} (d={d1:.4f})")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Plot #2 Closest --- #
    ax = axes[i, 1]
    if len(neighbors) > 1:
        c2_id, d2 = neighbors[1]
        mean2_img = get_mean_cluster_image(c2_id, output_dir)
        if mean2_img is not None:
            im = ax.imshow(mean2_img, cmap='viridis')
            fig.colorbar(im, ax=ax, orientation='horizontal')
        ax.set_title(f"2nd Closest: Cl. {c2_id} (d={d2:.4f})")
    else:
        ax.set_title("No 2nd neighbor found")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Plot Science Frame --- #
    ax = axes[i, 2]
    im = ax.imshow(target_image, cmap='viridis')
    fig.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_title(f"Science Frame {frame_idx}")
    ax.set_xticks([]); ax.set_yticks([])

science_hdul.close()
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()