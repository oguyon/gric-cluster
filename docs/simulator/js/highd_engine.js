/**
 * @file highd_engine.js
 * @brief High-Dimensional (32D+) Visualization and Linear Algebra Engine for GRIC Simulator.
 *
 * Implements real-time Principal Component Analysis (PCA) with cyclic Jacobi eigen-solver,
 * the Grand Tour continuous 60 FPS geodesic rotator, Fisher LDA cluster-discriminant projection,
 * interactive 3D Biplot rays, tomographic slicing, 32-axis Parallel Coordinates (PCP),
 * 32-bar Equalizer HUD, and anchor sparkline glyphs.
 */

(function (global) {
  'use strict';

  const HighDEngine = {};

  // =========================================================================
  // 1. LINEAR ALGEBRA & COVARIANCE COMPUTATION
  // =========================================================================

  /**
   * Extract coordinate array from a point or cluster object.
   * @param {Object|Array} p
   * @param {number} dim
   * @returns {Float64Array|null}
   */
  HighDEngine.getCoords = function (p, dim) {
    if (!p) return null;
    if (Array.isArray(p) || ArrayBuffer.isView(p)) {
      const out = new Float64Array(dim);
      for (let i = 0; i < dim && i < p.length; i++) out[i] = p[i] || 0.0;
      return out;
    }
    if (p.coords) {
      const out = new Float64Array(dim);
      const len = p.coords.length;
      for (let i = 0; i < dim && i < len; i++) out[i] = p.coords[i] || 0.0;
      return out;
    }
    if (p.anchor) {
      const out = new Float64Array(dim);
      const len = p.anchor.length;
      for (let i = 0; i < dim && i < len; i++) out[i] = p.anchor[i] || 0.0;
      return out;
    }
    // Fallback for 2D/3D objects with .x, .y, .z
    const out = new Float64Array(dim);
    out[0] = p.x || 0.0;
    out[1] = p.y || 0.0;
    if (dim > 2) out[2] = p.z || 0.0;
    return out;
  };

  /**
   * Compute mean vector and sample covariance matrix for D-dimensional points.
   * @param {Array} points - Array of point/cluster objects.
   * @param {number} dim - Dimension D (e.g. 32).
   * @returns {{ mean: Float64Array, cov: Array<Float64Array>, n: number }}
   */
  HighDEngine.computeMeanAndCovariance = function (points, dim) {
    const N = points.length;
    const mean = new Float64Array(dim);
    const cov = [];
    for (let i = 0; i < dim; i++) {
      cov.push(new Float64Array(dim));
    }

    if (N === 0) return { mean, cov, n: 0 };

    // 1. Mean computation
    let validCount = 0;
    for (let i = 0; i < N; i++) {
      const c = HighDEngine.getCoords(points[i], dim);
      if (!c) continue;
      validCount++;
      for (let d = 0; d < dim; d++) {
        mean[d] += c[d];
      }
    }

    if (validCount === 0) return { mean, cov, n: 0 };

    for (let d = 0; d < dim; d++) {
      mean[d] /= validCount;
    }

    if (validCount === 1) return { mean, cov, n: 1 };

    // 2. Covariance computation (symmetric)
    const factor = 1.0 / (validCount - 1);
    const diff = new Float64Array(dim);

    for (let i = 0; i < N; i++) {
      const c = HighDEngine.getCoords(points[i], dim);
      if (!c) continue;
      for (let d = 0; d < dim; d++) {
        diff[d] = c[d] - mean[d];
      }
      for (let r = 0; r < dim; r++) {
        const vr = diff[r];
        for (let col = r; col < dim; col++) {
          cov[r][col] += vr * diff[col];
        }
      }
    }

    // Mirror upper triangle to lower triangle and scale
    for (let r = 0; r < dim; r++) {
      for (let col = r; col < dim; col++) {
        cov[r][col] *= factor;
        cov[col][r] = cov[r][col];
      }
    }

    return { mean, cov, n: validCount };
  };

  /**
   * Exact cyclic Jacobi eigen-decomposition for real symmetric D x D matrix.
   * A * V = V * diag(eigenvalues)
   *
   * @param {Array<Float64Array>} matrix - Symmetric D x D matrix.
   * @param {number} dim - Dimension D.
   * @param {number} [maxSweeps=50] - Maximum Jacobi sweeps.
   * @param {number} [tol=1e-12] - Convergence tolerance.
   * @returns {{ eigenvalues: Float64Array, eigenvectors: Array<Float64Array> }}
   */
  HighDEngine.jacobiEigenSymmetric = function (matrix, dim, maxSweeps = 50, tol = 1e-12) {
    // Copy matrix to avoid modifying original
    const A = [];
    const V = [];
    for (let i = 0; i < dim; i++) {
      A.push(new Float64Array(matrix[i]));
      const vRow = new Float64Array(dim);
      vRow[i] = 1.0; // Identity matrix
      V.push(vRow);
    }

    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let maxOffDiag = 0.0;
      for (let i = 0; i < dim - 1; i++) {
        for (let j = i + 1; j < dim; j++) {
          const absVal = Math.abs(A[i][j]);
          if (absVal > maxOffDiag) maxOffDiag = absVal;
        }
      }

      if (maxOffDiag < tol) break;

      for (let p = 0; p < dim - 1; p++) {
        for (let q = p + 1; q < dim; q++) {
          const apq = A[p][q];
          if (Math.abs(apq) < 1e-15) continue;

          const app = A[p][p];
          const aqq = A[q][q];
          const tau = (aqq - app) / (2.0 * apq);
          let t;
          if (tau >= 0.0) {
            t = 1.0 / (tau + Math.sqrt(1.0 + tau * tau));
          } else {
            t = -1.0 / (-tau + Math.sqrt(1.0 + tau * tau));
          }

          const c = 1.0 / Math.sqrt(1.0 + t * t);
          const s = t * c;
          const tauSin = s / (1.0 + c);

          // Update diagonal elements
          A[p][p] = app - t * apq;
          A[q][q] = aqq + t * apq;
          A[p][q] = 0.0;
          A[q][p] = 0.0;

          // Update other rows/columns of A
          for (let r = 0; r < dim; r++) {
            if (r !== p && r !== q) {
              const arp = A[r][p];
              const arq = A[r][q];
              A[r][p] = arp - s * (arq + tauSin * arp);
              A[p][r] = A[r][p];
              A[r][q] = arq + s * (arp - tauSin * arq);
              A[q][r] = A[r][q];
            }
          }

          // Accumulate eigenvectors in columns of V
          for (let r = 0; r < dim; r++) {
            const vrp = V[r][p];
            const vrq = V[r][q];
            V[r][p] = vrp - s * (vrq + tauSin * vrp);
            V[r][q] = vrq + s * (vrp - tauSin * vrq);
          }
        }
      }
    }

    // Extract eigenvalues and convert eigenvectors to column arrays
    const rawEvals = new Float64Array(dim);
    for (let i = 0; i < dim; i++) {
      rawEvals[i] = A[i][i];
    }

    // Sort eigenvalues in descending order and permute eigenvector columns
    const indices = [];
    for (let i = 0; i < dim; i++) indices.push(i);
    indices.sort((a, b) => rawEvals[b] - rawEvals[a]);

    const eigenvalues = new Float64Array(dim);
    const eigenvectors = []; // Array of D length-D column vectors
    for (let col = 0; col < dim; col++) {
      eigenvectors.push(new Float64Array(dim));
    }

    for (let j = 0; j < dim; j++) {
      const origCol = indices[j];
      eigenvalues[j] = Math.max(0.0, rawEvals[origCol]);
      for (let r = 0; r < dim; r++) {
        eigenvectors[j][r] = V[r][origCol];
      }
    }

    return { eigenvalues, eigenvectors };
  };

  /**
   * Full Principal Component Analysis (PCA) for high-dimensional dataset.
   * @param {Array} points - Data points.
   * @param {number} dim - Dimensionality D.
   * @returns {{
   *   mean: Float64Array,
   *   eigenvalues: Float64Array,
   *   components: Array<Float64Array>,
   *   explainedVarianceRatio: Float64Array,
   *   cumulativeVarianceRatio: Float64Array,
   *   totalVariance: number
   * }}
   */
  HighDEngine.computePCA = function (points, dim) {
    const { mean, cov, n } = HighDEngine.computeMeanAndCovariance(points, dim);
    if (n === 0) {
      return {
        mean: new Float64Array(dim),
        eigenvalues: new Float64Array(dim),
        components: Array.from({ length: dim }, () => new Float64Array(dim)),
        explainedVarianceRatio: new Float64Array(dim),
        cumulativeVarianceRatio: new Float64Array(dim),
        totalVariance: 0.0
      };
    }

    const { eigenvalues, eigenvectors } = HighDEngine.jacobiEigenSymmetric(cov, dim);

    let totalVariance = 0.0;
    for (let i = 0; i < dim; i++) totalVariance += eigenvalues[i];

    const explainedVarianceRatio = new Float64Array(dim);
    const cumulativeVarianceRatio = new Float64Array(dim);
    let runningSum = 0.0;

    for (let i = 0; i < dim; i++) {
      explainedVarianceRatio[i] = totalVariance > 1e-12 ? eigenvalues[i] / totalVariance : 0.0;
      runningSum += explainedVarianceRatio[i];
      cumulativeVarianceRatio[i] = Math.min(1.0, runningSum);
    }

    return {
      mean,
      eigenvalues,
      components: eigenvectors, // components[k] is eigenvector for PC k+1
      explainedVarianceRatio,
      cumulativeVarianceRatio,
      totalVariance
    };
  };

  // =========================================================================
  // 2. CLUSTER-DISCRIMINANT FISHER / LDA PROJECTION
  // =========================================================================

  /**
   * Compute cluster-discriminant projection vectors (Fisher / LDA)
   * that maximize separation between clusters detected by gric-cluster.
   *
   * @param {Array} clusters - Array of cluster objects with .anchor / .coords and .members
   * @param {number} dim - Dimension D.
   * @returns {{ components: Array<Float64Array>, mean: Float64Array }}
   */
  HighDEngine.computeFisherLDA = function (clusters, dim) {
    if (!clusters || clusters.length < 2) {
      // Fallback to identity
      const comps = [];
      for (let i = 0; i < 3; i++) {
        const v = new Float64Array(dim);
        if (i < dim) v[i] = 1.0;
        comps.push(v);
      }
      return { components: comps, mean: new Float64Array(dim) };
    }

    const K = clusters.length;
    const mean = new Float64Array(dim);
    let totalMembers = 0;

    // 1. Overall centroid
    for (let k = 0; k < K; k++) {
      const c = HighDEngine.getCoords(clusters[k], dim);
      if (!c) continue;
      const cnt = typeof clusters[k].members === 'number'
        ? clusters[k].members
        : (clusters[k].members ? clusters[k].members.length : 1);
      totalMembers += cnt;
      for (let d = 0; d < dim; d++) {
        mean[d] += c[d] * cnt;
      }
    }

    if (totalMembers > 0) {
      for (let d = 0; d < dim; d++) mean[d] /= totalMembers;
    }

    // 2. Between-cluster scatter matrix S_B = sum_k N_k (c_k - mean)(c_k - mean)^T
    const Sb = [];
    for (let r = 0; r < dim; r++) Sb.push(new Float64Array(dim));

    for (let k = 0; k < K; k++) {
      const c = HighDEngine.getCoords(clusters[k], dim);
      if (!c) continue;
      const cnt = typeof clusters[k].members === 'number'
        ? clusters[k].members
        : (clusters[k].members ? clusters[k].members.length : 1);

      const dVec = new Float64Array(dim);
      for (let d = 0; d < dim; d++) dVec[d] = c[d] - mean[d];

      for (let r = 0; r < dim; r++) {
        const vr = dVec[r];
        for (let col = r; col < dim; col++) {
          Sb[r][col] += cnt * vr * dVec[col];
        }
      }
    }

    for (let r = 0; r < dim; r++) {
      for (let col = r; col < dim; col++) {
        Sb[col][r] = Sb[r][col];
      }
    }

    // Solve for top eigenvectors of Between-cluster scatter
    const { eigenvectors } = HighDEngine.jacobiEigenSymmetric(Sb, dim);

    return {
      components: eigenvectors.slice(0, 3),
      mean
    };
  };

  // =========================================================================
  // 3. THE "GRAND TOUR" 60 FPS CONTINUOUS GEODESIC ROTATOR
  // =========================================================================

  /**
   * GrandTour Engine: continuously rotates a 3D projection frame
   * through D-dimensional space at 60 FPS.
   *
   * @param {number} dim - Dimensionality D.
   */
  HighDEngine.GrandTour = function (dim = 32) {
    this.dim = Math.max(3, dim);
    // 3 x D orthonormal projection frame
    this.frame = [
      new Float64Array(this.dim),
      new Float64Array(this.dim),
      new Float64Array(this.dim)
    ];

    this.reset();

    this.speed = 1.0;
    this.isPlaying = false;
    this.time = 0.0;

    // Active rotation plane indices
    this.pairIdx = 0;
    this.rotationPairs = [];
    this._initPairs();
  };

  HighDEngine.GrandTour.prototype.reset = function () {
    for (let r = 0; r < 3; r++) {
      this.frame[r].fill(0.0);
      if (r < this.dim) this.frame[r][r] = 1.0;
    }
  };

  HighDEngine.GrandTour.prototype._initPairs = function () {
    this.rotationPairs = [];
    // Generate a diverse sequence of (p, q) planes
    const D = this.dim;
    for (let p = 0; p < D - 1; p++) {
      for (let q = p + 1; q < D; q++) {
        // Interleave primary axes with higher modes
        if (p < 4 || q < 8 || (p + q) % 3 === 0) {
          this.rotationPairs.push([p, q]);
        }
      }
    }
    // Shuffle pairs slightly for organic tumbling motion
    for (let i = this.rotationPairs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = this.rotationPairs[i];
      this.rotationPairs[i] = this.rotationPairs[j];
      this.rotationPairs[j] = tmp;
    }
  };

  /**
   * Step the continuous rotation forward by dt seconds.
   * @param {number} [dt=0.016] - Elapsed time in seconds.
   */
  HighDEngine.GrandTour.prototype.step = function (dt = 0.016) {
    if (!this.isPlaying) return;

    const baseAngle = 0.8 * this.speed * dt;
    this.time += dt;

    if (this.rotationPairs.length === 0) this._initPairs();

    // Rotate across 2-3 simultaneous orthogonal/interleaved planes for rich 3D motion
    for (let k = 0; k < 2; k++) {
      const pair = this.rotationPairs[(this.pairIdx + k) % this.rotationPairs.length];
      const p = pair[0];
      const q = pair[1];
      const theta = baseAngle * (k === 0 ? 1.0 : -0.73);
      const c = Math.cos(theta);
      const s = Math.sin(theta);

      // Apply Givens rotation to the 3 frame row vectors
      for (let r = 0; r < 3; r++) {
        const u_p = this.frame[r][p];
        const u_q = this.frame[r][q];
        this.frame[r][p] = c * u_p - s * u_q;
        this.frame[r][q] = s * u_p + c * u_q;
      }
    }

    this.pairIdx = (this.pairIdx + 1) % this.rotationPairs.length;

    // Gram-Schmidt orthonormalization every step to prevent numerical drift
    this._reorthonormalize();
  };

  HighDEngine.GrandTour.prototype._reorthonormalize = function () {
    const u0 = this.frame[0];
    const u1 = this.frame[1];
    const u2 = this.frame[2];
    const D = this.dim;

    // Normalize u0
    let norm0 = 0.0;
    for (let d = 0; d < D; d++) norm0 += u0[d] * u0[d];
    norm0 = Math.sqrt(norm0) || 1.0;
    for (let d = 0; d < D; d++) u0[d] /= norm0;

    // u1 = u1 - (u1 . u0) u0
    let dot01 = 0.0;
    for (let d = 0; d < D; d++) dot01 += u1[d] * u0[d];
    for (let d = 0; d < D; d++) u1[d] -= dot01 * u0[d];
    let norm1 = 0.0;
    for (let d = 0; d < D; d++) norm1 += u1[d] * u1[d];
    norm1 = Math.sqrt(norm1) || 1.0;
    for (let d = 0; d < D; d++) u1[d] /= norm1;

    // u2 = u2 - (u2 . u0) u0 - (u2 . u1) u1
    let dot02 = 0.0, dot12 = 0.0;
    for (let d = 0; d < D; d++) {
      dot02 += u2[d] * u0[d];
      dot12 += u2[d] * u1[d];
    }
    for (let d = 0; d < D; d++) {
      u2[d] -= dot02 * u0[d] + dot12 * u1[d];
    }
    let norm2 = 0.0;
    for (let d = 0; d < D; d++) norm2 += u2[d] * u2[d];
    norm2 = Math.sqrt(norm2) || 1.0;
    for (let d = 0; d < D; d++) u2[d] /= norm2;
  };

  /**
   * Project a D-dimensional point onto the current 3D tour frame.
   * @param {Object|Array} p
   * @param {Float64Array} [mean]
   * @returns {{ x: number, y: number, z: number }}
   */
  HighDEngine.GrandTour.prototype.projectPoint = function (p, mean) {
    const c = HighDEngine.getCoords(p, this.dim);
    if (!c) return { x: 0, y: 0, z: 0 };

    let x = 0.0, y = 0.0, z = 0.0;
    const D = this.dim;
    const u0 = this.frame[0];
    const u1 = this.frame[1];
    const u2 = this.frame[2];

    if (mean && mean.length >= D) {
      for (let d = 0; d < D; d++) {
        const diff = c[d] - mean[d];
        x += u0[d] * diff;
        y += u1[d] * diff;
        z += u2[d] * diff;
      }
    } else {
      for (let d = 0; d < D; d++) {
        const val = c[d];
        x += u0[d] * val;
        y += u1[d] * val;
        z += u2[d] * val;
      }
    }

    return { x, y, z };
  };

  // =========================================================================
  // 4. INTERACTIVE 3D BIPLOT RAYS
  // =========================================================================

  /**
   * Compute 3D biplot vectors for standard basis axes e_0 ... e_{D-1}.
   * @param {Array<Float64Array>} projFrame - 3 x D projection matrix.
   * @param {number} dim - Dimension D.
   * @param {number} [scale=1.0] - Visual scale factor.
   * @returns {Array<{ dim: number, x: number, y: number, z: number, mag: number }>}
   */
  HighDEngine.getBiplotRays = function (projFrame, dim, scale = 1.0) {
    if (!projFrame || projFrame.length < 3) return [];
    const rays = [];
    const u0 = projFrame[0];
    const u1 = projFrame[1];
    const u2 = projFrame[2];

    for (let d = 0; d < dim; d++) {
      const vx = (u0 && u0[d] !== undefined) ? u0[d] : (d === 0 ? 1 : 0);
      const vy = (u1 && u1[d] !== undefined) ? u1[d] : (d === 1 ? 1 : 0);
      const vz = (u2 && u2[d] !== undefined) ? u2[d] : (d === 2 ? 1 : 0);
      const mag = Math.hypot(vx, vy, vz);

      rays.push({
        dim: d,
        x: vx * scale,
        y: vy * scale,
        z: vz * scale,
        mag
      });
    }

    return rays;
  };

  // =========================================================================
  // 5. TOMOGRAPHIC SLICING FILTER
  // =========================================================================

  /**
   * Evaluate whether a point lies within a tomographic slice window.
   * @param {Object|Array} point
   * @param {number} sliceDim - Dimension index to slice (-1 = off).
   * @param {number} sliceCenter - Center coordinate of slice.
   * @param {number} sliceThickness - Half-width of slice window.
   * @returns {{ inside: boolean, alpha: number }}
   */
  HighDEngine.isPointInSlice = function (point, sliceDim, sliceCenter, sliceThickness) {
    if (sliceDim < 0 || sliceThickness <= 0) {
      return { inside: true, alpha: 1.0 };
    }

    const c = HighDEngine.getCoords(point, sliceDim + 1);
    if (!c || c.length <= sliceDim) {
      return { inside: true, alpha: 1.0 };
    }

    const val = c[sliceDim];
    const dist = Math.abs(val - sliceCenter);

    if (dist <= sliceThickness) {
      // Smooth falloff towards boundary
      const frac = dist / sliceThickness;
      const alpha = Math.max(0.2, 1.0 - 0.7 * frac * frac);
      return { inside: true, alpha };
    }

    // Ghost point outside window
    return { inside: false, alpha: 0.06 };
  };

  // =========================================================================
  // 6. "AUTO-PICK BEST 3D" OPTIMIZER
  // =========================================================================

  /**
   * Scan dimensions and pick the triplet (dx, dy, dz) that maximizes
   * cluster separability or overall variance.
   * @param {Array} clusters - Clusters detected by gric-cluster.
   * @param {Array} points - Sample points.
   * @param {number} dim - Dimension D.
   * @returns {[number, number, number]}
   */
  HighDEngine.findBestSeparatingTriplet = function (clusters, points, dim) {
    if (dim <= 3) return [0, 1, Math.min(2, dim - 1)];

    const D = dim;
    const scores = new Float64Array(D);

    if (clusters && clusters.length >= 2) {
      // Score based on cluster centroid variance
      const K = clusters.length;
      const means = new Float64Array(D);

      for (let k = 0; k < K; k++) {
        const c = HighDEngine.getCoords(clusters[k], D);
        if (!c) continue;
        for (let d = 0; d < D; d++) means[d] += c[d];
      }
      for (let d = 0; d < D; d++) means[d] /= K;

      for (let k = 0; k < K; k++) {
        const c = HighDEngine.getCoords(clusters[k], D);
        if (!c) continue;
        for (let d = 0; d < D; d++) {
          const diff = c[d] - means[d];
          scores[d] += diff * diff;
        }
      }
    } else if (points && points.length > 10) {
      // Score based on sample variance
      const N = Math.min(points.length, 1000);
      const means = new Float64Array(D);
      for (let i = 0; i < N; i++) {
        const c = HighDEngine.getCoords(points[i], D);
        if (!c) continue;
        for (let d = 0; d < D; d++) means[d] += c[d];
      }
      for (let d = 0; d < D; d++) means[d] /= N;

      for (let i = 0; i < N; i++) {
        const c = HighDEngine.getCoords(points[i], D);
        if (!c) continue;
        for (let d = 0; d < D; d++) {
          const diff = c[d] - means[d];
          scores[d] += diff * diff;
        }
      }
    }

    const indices = [];
    for (let d = 0; d < D; d++) indices.push(d);
    indices.sort((a, b) => scores[b] - scores[a]);

    const dx = indices[0] !== undefined ? indices[0] : 0;
    const dy = indices[1] !== undefined ? indices[1] : 1;
    const dz = indices[2] !== undefined ? indices[2] : 2;

    return [dx, dy, dz];
  };

  // =========================================================================
  // 7. 32-AXIS PARALLEL COORDINATES PLOT (PCP) RENDERER
  // =========================================================================

  /**
   * Render a Parallel Coordinates Plot across all D dimensions inside rect.
   *
   * @param {CanvasRenderingContext2D} ctx
   * @param {{ x: number, y: number, w: number, h: number }} rect
   * @param {Array} clusters - Cluster centroids.
   * @param {Array} samplePoints - Sample points.
   * @param {number} hoveredClusterId
   * @param {number} selectedClusterId
   * @param {number} dim - Dimension D (e.g. 32).
   */
  HighDEngine.renderParallelCoordinates = function (
    ctx, rect, clusters, samplePoints, hoveredClusterId, selectedClusterId, dim = 32
  ) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
    ctx.clip();

    // Background fill
    ctx.fillStyle = '#090d16';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

    const D = Math.max(4, dim);
    const padL = 36;
    const padR = 24;
    const padT = 32;
    const padB = 28;
    const plotW = rect.w - padL - padR;
    const plotH = rect.h - padT - padB;

    if (plotW <= 20 || plotH <= 20) {
      ctx.restore();
      return;
    }

    const axisX = [];
    for (let d = 0; d < D; d++) {
      axisX.push(rect.x + padL + (d / (D - 1)) * plotW);
    }

    const yMinVal = -1.2;
    const yMaxVal = 1.2;
    const valRange = yMaxVal - yMinVal;

    function valToY(v) {
      const clamped = Math.max(yMinVal, Math.min(yMaxVal, v));
      const frac = (clamped - yMinVal) / valRange;
      return rect.y + padT + (1.0 - frac) * plotH;
    }

    // 1. Draw Axis Lines & Labels
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1.0;
    const stepLabel = Math.ceil(D / 16);

    for (let d = 0; d < D; d++) {
      const ax = axisX[d];
      ctx.beginPath();
      ctx.moveTo(ax, rect.y + padT);
      ctx.lineTo(ax, rect.y + padT + plotH);
      ctx.stroke();

      // Axis bottom label
      if (d % stepLabel === 0 || d === D - 1) {
        ctx.fillStyle = '#64748b';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`D${d}`, ax, rect.y + padT + plotH + 14);
      }
    }

    // Zero guide line
    const zeroY = valToY(0.0);
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.4)';
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(rect.x + padL, zeroY);
    ctx.lineTo(rect.x + padL + plotW, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Draw Sample Point Polylines (subsampled for performance)
    if (samplePoints && samplePoints.length > 0) {
      const stepSample = Math.max(1, Math.floor(samplePoints.length / 80));
      ctx.lineWidth = 0.7;
      ctx.globalAlpha = 0.12;

      for (let i = 0; i < samplePoints.length; i += stepSample) {
        const c = HighDEngine.getCoords(samplePoints[i], D);
        if (!c) continue;

        ctx.strokeStyle = '#94a3b8';
        ctx.beginPath();
        for (let d = 0; d < D; d++) {
          const px = axisX[d];
          const py = valToY(c[d]);
          if (d === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;
    }

    // 3. Draw Cluster Centroid Polylines
    if (clusters && clusters.length > 0) {
      clusters.forEach(cl => {
        const isHovered = (hoveredClusterId === cl.id);
        const isSelected = (selectedClusterId === cl.id);
        if (isHovered || isSelected) return; // Draw highlighted on top

        const c = HighDEngine.getCoords(cl, D);
        if (!c) return;

        ctx.strokeStyle = cl.color || '#38bdf8';
        ctx.lineWidth = 1.6;
        ctx.globalAlpha = 0.65;
        ctx.beginPath();
        for (let d = 0; d < D; d++) {
          const px = axisX[d];
          const py = valToY(c[d]);
          if (d === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      });
      ctx.globalAlpha = 1.0;

      // Draw Selected / Hovered Clusters with glowing halo
      [selectedClusterId, hoveredClusterId].forEach(hlId => {
        if (hlId === -1 || hlId === null || hlId === undefined) return;
        const cl = clusters.find(c => c.id === hlId);
        if (!cl) return;

        const c = HighDEngine.getCoords(cl, D);
        if (!c) return;

        const isSel = (selectedClusterId === cl.id);
        const glowColor = isSel ? '#facc15' : '#38bdf8';

        ctx.save();
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 8;
        ctx.strokeStyle = glowColor;
        ctx.lineWidth = 3.2;

        ctx.beginPath();
        for (let d = 0; d < D; d++) {
          const px = axisX[d];
          const py = valToY(c[d]);
          if (d === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // Node circles at each axis intersection
        for (let d = 0; d < D; d++) {
          const px = axisX[d];
          const py = valToY(c[d]);
          ctx.beginPath();
          ctx.arc(px, py, 3.5, 0, Math.PI * 2);
          ctx.fillStyle = glowColor;
          ctx.fill();
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 1.0;
          ctx.stroke();
        }
        ctx.restore();
      });
    }

    // 4. Header Bar
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(rect.x, rect.y, rect.w, 24);
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
    ctx.beginPath();
    ctx.moveTo(rect.x, rect.y + 24);
    ctx.lineTo(rect.x + rect.w, rect.y + 24);
    ctx.stroke();

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`📈 32D Parallel Coordinates (${D} Axes)`, rect.x + 8, rect.y + 16);

    ctx.font = '10px monospace';
    ctx.fillStyle = '#94a3b8';
    const rangeStr = `Range: [${yMinVal.toFixed(1)}, +${yMaxVal.toFixed(1)}]`;
    ctx.fillText(rangeStr, rect.x + plotW - 40, rect.y + 16);

    ctx.restore();
  };

  // =========================================================================
  // 8. 32-BAR EQUALIZER HUD & ANCHOR SPARKLINES
  // =========================================================================

  /**
   * Render an interactive 32-bar Equalizer HUD card for a hovered point/cluster.
   *
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x - HUD top-left X.
   * @param {number} y - HUD top-left Y.
   * @param {Object|Array} item - Hovered point or cluster object.
   * @param {string} label - Title label (e.g. "Cluster C2" or "Sample #42").
   * @param {string} color - Accent color.
   * @param {number} dim - Dimension D (e.g. 32).
   */
  HighDEngine.renderEqualizerHUD = function (ctx, x, y, item, label, color = '#38bdf8', dim = 32) {
    const coords = HighDEngine.getCoords(item, dim);
    if (!coords) return;

    const D = Math.max(4, dim);
    const cardW = 280;
    const cardH = 76;
    const barAreaH = 34;

    ctx.save();
    // Card background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.roundRect(x, y, cardW, cardH, 6);
    ctx.fill();
    ctx.stroke();

    // Title & Badge
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 10.5px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`📊 ${label}`, x + 8, y + 16);

    ctx.fillStyle = color;
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${D}-D Signature`, x + cardW - 8, y + 16);

    // Bars Area
    const barPadL = 8;
    const barPadR = 8;
    const totalBarW = cardW - barPadL - barPadR;
    const barW = Math.max(1.5, (totalBarW / D) - 1.5);
    const midY = y + 26 + barAreaH / 2;

    // Center baseline
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.5)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(x + barPadL, midY);
    ctx.lineTo(x + cardW - barPadR, midY);
    ctx.stroke();

    // Bars
    for (let d = 0; d < D; d++) {
      const val = coords[d] || 0.0;
      const bx = x + barPadL + (d / D) * totalBarW;
      const normH = Math.min(barAreaH / 2, Math.abs(val) * (barAreaH / 2));

      if (val >= 0) {
        ctx.fillStyle = color;
        ctx.fillRect(bx, midY - normH, barW, normH);
      } else {
        ctx.fillStyle = '#f97316'; // Orange for negative
        ctx.fillRect(bx, midY, barW, normH);
      }
    }

    // Bottom labels: D0 and D{D-1}
    ctx.fillStyle = '#64748b';
    ctx.font = '8.5px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('D0', x + barPadL, y + cardH - 4);
    ctx.textAlign = 'right';
    ctx.fillText(`D${D - 1}`, x + cardW - barPadR, y + cardH - 4);

    ctx.restore();
  };

  /**
   * Render a micro 32-bar sparkline glyph directly above an anchor in 2D/3D space.
   *
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} px - Screen X.
   * @param {number} py - Screen Y.
   * @param {Object|Array} cluster - Cluster anchor object.
   * @param {string} color - Cluster color.
   * @param {number} dim - Dimension D (e.g. 32).
   */
  HighDEngine.renderAnchorSparkline = function (ctx, px, py, cluster, color, dim = 32) {
    const coords = HighDEngine.getCoords(cluster, dim);
    if (!coords) return;

    const D = Math.min(32, dim);
    const glyphW = 34;
    const glyphH = 14;
    const gx = px - glyphW / 2;
    const gy = py - glyphH - 10;

    ctx.save();
    // Backdrop pill
    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.strokeStyle = color || '#38bdf8';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.roundRect(gx, gy, glyphW, glyphH, 2.5);
    ctx.fill();
    ctx.stroke();

    const midY = gy + glyphH / 2;
    const barW = Math.max(0.6, (glyphW - 4) / D);

    for (let d = 0; d < D; d++) {
      const val = coords[d] || 0.0;
      const bx = gx + 2 + (d / D) * (glyphW - 4);
      const bH = Math.min(glyphH / 2 - 1, Math.abs(val) * (glyphH / 2));

      ctx.fillStyle = val >= 0 ? color : '#f97316';
      if (val >= 0) {
        ctx.fillRect(bx, midY - bH, barW, bH);
      } else {
        ctx.fillRect(bx, midY, barW, bH);
      }
    }
    ctx.restore();
  };

  // =========================================================================
  // EXPORT
  // =========================================================================

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = HighDEngine;
  }
  if (typeof window !== 'undefined') {
    window.HighDEngine = HighDEngine;
  }
})(typeof window !== 'undefined' ? window : globalThis);
