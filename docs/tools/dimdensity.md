# gric-dimdensity: Local Intrinsic Dimension & Density Estimator

`gric-dimdensity` is an analytical tool that estimates the Local Intrinsic Dimension (LID) and
local probability density around each point on a manifold using $k$-nearest neighbor distance
distributions.

---

## 1. Overview & Theoretical Principles

High-dimensional data (e.g., $D = 1024$ pixels) often lives on a much lower-dimensional non-linear
submanifold (e.g., $d \approx 3 \sim 12$). `gric-dimdensity` quantifies this geometry:

### Levina-Bickel Maximum Likelihood Estimator (MLE)
Estimates the local intrinsic dimensionality around query sample $x_i$ using distances to its
$k$ nearest neighbors:

$$\hat{d}(x_i) = \frac{k - 2}{\sum_{j=1}^{k-1} \left[ \ln(r_k(x_i)) - \ln(r_j(x_i)) \right]}$$

The $(k - 2)$ numerator provides a finite-sample unbiased correction, preventing significant
overestimation on moderate neighborhood sizes ($k \le 50$).

### Mack-Rosenblatt Variable-Bandwidth Density
Computes local manifold probability density by scaling the neighborhood radius to the locally
estimated intrinsic dimension $\hat{d}$:

$$\hat{f}(x_i) = \frac{k - 1}{N \cdot V_{\hat{d}} \cdot r_k(x_i)^{\hat{d}}}$$

where $V_d = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}$ is the volume of the $d$-dimensional unit ball.

---

## 2. Usage & Options

```bash
gric-dimdensity <knn_input_or_dir> [options]
```

### Options

| Option | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| `-k` | `<int>` | Number of nearest neighbors to evaluate | All available |
| `-kmin` | `<int>` | Minimum $k$ for multi-scale range averaging | - |
| `-kmax` | `<int>` | Maximum $k$ for multi-scale range averaging | - |
| `-range` | Flag | Multi-scale range averaging over $[k_{\text{min}}, k_{\text{max}}]$ | Off |
| `-classic` | Flag | Use classic $(k-1)$ MLE instead of $(k-2)$ unbiased | Unbiased |
| `-kernel` | `<type>` | Kernel weighting: `uniform`, `epanechnikov`, `gaussian` | `uniform` |
| `-o` | `<path>` | Output destination path (`.txt`, `.bin`, `.fits`) | Auto |
| `-json` | Flag | Emit comprehensive statistical JSON report | Text output |
| `-h`, `--help` | Flag | Display CLI help screen and exit | - |

---

## 3. Examples

```bash
# 1. Run k-NN search and estimate intrinsic dimension
gric-knn observations.fits obs_out/ -k 30 -o knn_obs.bin
gric-dimdensity knn_obs.bin -k 20

# 2. Multi-scale smoothed estimation using Epanechnikov kernel
gric-dimdensity knn_obs.bin -kmin 10 -kmax 25 -range -kernel epanechnikov -o lid_results.bin

# 3. Output structured summary report in JSON
gric-dimdensity obs_out/ -k 15 -json > lid_summary.json
```
