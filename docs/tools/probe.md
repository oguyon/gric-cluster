# gric-probe: Dataset Characterization & Profiler

`gric-probe` (also available via the alias symlink `gric-cluster-probe`) is a fast dataset
profiling utility. It conducts a single-pass geometric, statistical, and quantization scan over
raw dataset frames to derive optimal clustering parameters and acceleration presets.

---

## 1. Overview & Capabilities

Before running clustering on an unknown dataset, choosing the radius threshold $r_{\text{lim}}$ or
determining whether quantization acceleration (`-sq16`, `-eq16`) is beneficial often requires trial
and error. `gric-probe` automates this process by analyzing:

1. **Distance Scale & Metric Geometry**: Computes step-wise and pairwise distance
   distributions (percentiles, min, median, max) to derive presets (`fine`, `balanced`, `coarse`).
2. **Quantization Fidelity**: Evaluates distortion bounds for 16-bit scalar (`SQ16`) and E8 lattice
   (`EQ16`) quantization, recommending the optimal acceleration mode.
3. **Spectral Variance & Sparsity**: Measures per-dimension energy and variance distribution across
   channels or pixels.
4. **Autonomous Integration**: Writes a self-describing `.gricprof` JSON profile that `gric-cluster`
   and `gric-knn` automatically discover and consume when processing the dataset.

---

## 2. Usage & Options

```bash
gric-probe <dataset_path> [options]
```

### Options

| Option | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| `-o` | `<path>` | Output path for `.gricprof` JSON profile | `<dataset>.gricprof` |
| `-n` | `<count>` | Number of sample frames to evaluate | Adaptive (max 5,000) |
| `-all` | Flag | Scan 100% of dataset without subsampling | Subsample |
| `-warmup` | `<count>` | Warmup frames to ignore for live streams | `0` |
| `-double` | Flag | Process dataset in 64-bit double precision | `float32` |
| `-q`, `--quiet` | Flag | Suppress interactive terminal dashboard | Verbose |
| `-progress` | Flag | Emit `[PROBE: XX%]` progress updates to stderr | Disabled |
| `-json` | Flag | Print raw JSON profile directly to stdout | Formatted text |
| `--env` | Flag | Output recommended shell environment variables | Disabled |
| `-h`, `--help` | Flag | Display CLI help message and exit | - |

---

## 3. Profile Output & Presets (`.gricprof`)

`gric-probe` produces a JSON file containing comprehensive dataset metrics:

```json
{
  "version": 1,
  "dataset": "observations.fits",
  "dimensions": 1024,
  "frame_count": 5000,
  "median_step_dist": 1.428,
  "presets": {
    "fine":     {"rlim": 0.714, "desc": "High fidelity, many granular clusters"},
    "balanced": {"rlim": 1.428, "desc": "Recommended default operational point"},
    "coarse":   {"rlim": 2.856, "desc": "Broad geometric aggregation"}
  },
  "recommended_flags": "-eq16 -te4"
}
```

### Automatic Profile Ingestion in `gric-cluster`
When running `gric-cluster` or `gric-knn`:
- If `observations.fits.gricprof` exists in the same directory, it is **automatically loaded**.
- You can specify radius presets directly via `-preset fine`, `-preset balanced`, or
  `-preset coarse` instead of guessing numeric $r_{\text{lim}}$ values.
- Override or disable profile loading with `-prof <path>` or `-no-prof`.

---

## 4. Examples

```bash
# 1. Fast profile scan of high-dimensional FITS cube
gric-probe observations.fits

# 2. Complete non-subsampled scan of small coordinate file
gric-probe coordinates.txt -all -o coords.gricprof

# 3. Direct JSON output piped to jq for automated pipelines
gric-probe input.mp4 -json | jq '.presets.balanced.rlim'

# 4. Run clustering using probe-derived balanced preset
gric-cluster input.fits -preset balanced -outdir results/
```
