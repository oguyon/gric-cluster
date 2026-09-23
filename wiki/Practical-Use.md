# Practical Use Cases

This tool is designed for high-speed clustering of sequential image data or high-dimensional
vectors. Below are some practical applications.

## 1. Astronomical Imaging

**Scenario**: You have a "cube" of FITS images (e.g., a time-series observation of a star field)
and want to group frames based on seeing conditions or shifting alignment.

**Workflow**:
1.  **Input**: A 3D FITS cube where each slice is a frame.
2.  **Run**: Use `gric-cluster` to group similar frames.
    ```bash
    ./gric-cluster a1.5 input.fits -outdir results -avg
    ```
    *   `a1.5`: Auto-sets the radius to 1.5x the median frame-to-frame distance.
    *   `-avg`: Computes the average image (stack) for each cluster, which can improve SNR.
3.  **Result**: You get `average.fits` containing the "lucky imaging" stacks for each cluster.

## 2. Video Stream Analysis (Data Reduction)

**Scenario**: You have a stream of video frames (converted to feature vectors or raw pixels)
and want to identify unique scenes or remove near-duplicates.

**Workflow**:
1.  **Preprocessing**: Convert frames to downscaled feature vectors.
2.  **Run**:
    ```bash
    ./gric-cluster 500.0 video_feats.txt -maxcl 100 -tm 0.8
    ```
    *   `-tm 0.8`: Uses the transition matrix to predict the next scene based on history (80%
        weight), optimizing speed for structured video.
3.  **Result**: The tool identifies unique "anchor" frames. Frames within distance `500.0` of an
    anchor are grouped. Use `cluster_counts.txt` or `counts.bin` to find common scenes.

## 3. Noisy Data Categorization

**Scenario**: You have experimental sensor data (1D vectors) that drift over time.

**Workflow**:
1.  **Generate Test Data** (to tune parameters):
    ```bash
    ./gric-mktxtseq 100 test.txt 2Dwalk -repeat 10 -noise 0.1
    ```
2.  **Tune Radius**:
    ```bash
    ./gric-cluster -scandist test.txt
    ```
    Use the "Median" or "20%" percentile output to choose a tight `rlim`.
3.  **Cluster**:
    ```bash
    ./gric-cluster <chosen_rlim> sensor_data.txt -gprob
    ```
    `-gprob` is useful here if drift is continuous, as it learns the trajectory of sensor data.

## 4. High-Dimensional / Expensive Metric Clustering

**Scenario**: You are clustering vectors where distance computation is expensive (e.g., $D > 1000$).

**Workflow**:
1.  **Run**:
    ```bash
    ./gric-cluster <rlim> vectors.txt -te5 -eq16
    ```
    *   `-te5`: Enables 5-point pruning to reduce distance calculations by ~45%.
    *   `-eq16`: Enables 16-bit E8 lattice quantization for $4\times$ memory bandwidth savings.

## 5. Continuous Stream Monitoring (Managing Max Clusters)

**Scenario**: You are processing an infinite stream of data and want to maintain a fixed-size
dictionary of clusters without running out of memory.

**Strategies (`-maxcl_strategy`)**:
*   **Stop** (default): The program exits when `-maxcl` is reached.
*   **Discard** (`-maxcl_strategy discard`): Deletes the oldest/smallest cluster to make room.
*   **Merge** (`-maxcl_strategy merge`): Two closest clusters are merged into one.

**Workflow**:
```bash
./gric-cluster 0.5 stream_data.txt -maxcl 100 -maxcl_strategy discard -discard_frac 0.5
```
*   `-discard_frac 0.5`: Only consider the oldest 50% of clusters for deletion.

## Tips for Best Results

*   **Auto-Tuning**: Run `gric-probe <input>` or `-scandist` to understand dataset scale.
*   **Geometric Probability**: Use `-gprob` for time-series data with smooth trajectories.
*   **Transition Matrix**: Use `-tm` for repeating sequences (e.g., video loops, cyclic phases).
