/**
 * @file highd_engine_test.js
 * @brief Automated verification suite for HighDEngine linear algebra & projection functions.
 */

const assert = require('assert');
const HighDEngine = require('../docs/simulator/js/highd_engine.js');

console.log('--- Running HighDEngine Test Suite ---');

// -----------------------------------------------------------------------------
// 1. Mean & Covariance Test
// -----------------------------------------------------------------------------
{
  const dim = 4;
  const pts = [
    { coords: [1, 2, 3, 4] },
    { coords: [3, 4, 5, 6] },
    { coords: [5, 6, 7, 8] }
  ];

  const { mean, cov, n } = HighDEngine.computeMeanAndCovariance(pts, dim);
  assert.strictEqual(n, 3, 'Sample count must be 3');
  assert.strictEqual(mean[0], 3.0, 'Mean[0] should be 3.0');
  assert.strictEqual(mean[1], 4.0, 'Mean[1] should be 4.0');
  assert.strictEqual(mean[2], 5.0, 'Mean[2] should be 5.0');
  assert.strictEqual(mean[3], 6.0, 'Mean[3] should be 6.0');

  // Covariance between coords (diff is [-2, 0, 2] on all axes)
  // Var = (-2)^2 + 0^2 + 2^2 = 8 / 2 = 4.0
  for (let r = 0; r < dim; r++) {
    for (let c = 0; c < dim; c++) {
      assert(Math.abs(cov[r][c] - 4.0) < 1e-10, `cov[${r}][${c}] should be 4.0, got ${cov[r][c]}`);
    }
  }
  console.log('✅ Mean and Covariance calculation verified');
}

// -----------------------------------------------------------------------------
// 2. Jacobi Eigen-decomposition on Symmetric Matrix
// -----------------------------------------------------------------------------
{
  // Known 2x2 matrix: [[2, 1], [1, 2]] with eigenvalues 3 and 1
  const m2 = [
    new Float64Array([2.0, 1.0]),
    new Float64Array([1.0, 2.0])
  ];
  const res2 = HighDEngine.jacobiEigenSymmetric(m2, 2);
  assert(Math.abs(res2.eigenvalues[0] - 3.0) < 1e-10,
    `Eigenvalue 0 must be 3, got ${res2.eigenvalues[0]}`);
  assert(Math.abs(res2.eigenvalues[1] - 1.0) < 1e-10,
    `Eigenvalue 1 must be 1, got ${res2.eigenvalues[1]}`);

  // Test 32x32 random symmetric matrix
  const dim = 32;
  const randSym = [];
  for (let i = 0; i < dim; i++) randSym.push(new Float64Array(dim));
  for (let i = 0; i < dim; i++) {
    for (let j = i; j < dim; j++) {
      const v = (Math.random() - 0.5) * 2.0;
      randSym[i][j] = v;
      randSym[j][i] = v;
    }
    randSym[i][i] += 10.0; // Positive definite shift
  }

  const { eigenvalues, eigenvectors } = HighDEngine.jacobiEigenSymmetric(randSym, dim);

  // Check eigenvalues are sorted descending
  for (let i = 0; i < dim - 1; i++) {
    assert(eigenvalues[i] >= eigenvalues[i + 1] - 1e-12, `Eigenvalues must be sorted: idx ${i}`);
  }

  // Check eigenvectors are orthonormal: V^T * V = I
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      let dot = 0.0;
      for (let k = 0; k < dim; k++) {
        dot += eigenvectors[i][k] * eigenvectors[j][k];
      }
      const expected = (i === j) ? 1.0 : 0.0;
      assert(Math.abs(dot - expected) < 1e-9,
        `Eigenvectors must be orthonormal at (${i}, ${j}): got ${dot}`);
    }
  }
  console.log('✅ Jacobi Eigen-decomposition (32x32 orthonormal) verified');
}

// -----------------------------------------------------------------------------
// 3. Full 32D PCA Test
// -----------------------------------------------------------------------------
{
  const D = 32;
  const N = 200;
  const dataset = [];

  // Generate synthetic anisotropic 32D data (axis 0 has high variance, axis 1 moderate, etc.)
  for (let i = 0; i < N; i++) {
    const coords = new Float64Array(D);
    for (let d = 0; d < D; d++) {
      const scale = 1.0 / (d + 1); // Declining variance
      coords[d] = (Math.random() - 0.5) * scale;
    }
    dataset.push({ coords });
  }

  const pca = HighDEngine.computePCA(dataset, D);

  assert.strictEqual(pca.components.length, D, 'Components count must match D');
  assert.strictEqual(pca.explainedVarianceRatio.length, D, 'Ratio count must match D');

  // Total cumulative variance of all components must sum to 1.0
  const finalCumVar = pca.cumulativeVarianceRatio[D - 1];
  assert(Math.abs(finalCumVar - 1.0) < 1e-9,
    `Final cumulative variance must be 1.0, got ${finalCumVar}`);

  // PC1 should capture significantly more variance than PC31
  assert(pca.explainedVarianceRatio[0] > pca.explainedVarianceRatio[D - 1],
    'PC1 variance should be > PC31');
  const pc1Pct = (pca.explainedVarianceRatio[0] * 100).toFixed(2);
  console.log(`✅ 32D PCA verified: PC1 variance ratio = ${pc1Pct}%`);
}

// -----------------------------------------------------------------------------
// 4. Grand Tour 60 FPS Geodesic Rotator Orthonormality Test
// -----------------------------------------------------------------------------
{
  const D = 32;
  const tour = new HighDEngine.GrandTour(D);
  tour.isPlaying = true;
  tour.speed = 1.5;

  // Step 500 times (simulating ~8.3 seconds of 60 FPS animation)
  for (let step = 0; step < 500; step++) {
    tour.step(0.016);
  }

  // Verify frame rows u0, u1, u2 maintain exact orthonormality: U * U^T = I_3
  for (let r1 = 0; r1 < 3; r1++) {
    for (let r2 = 0; r2 < 3; r2++) {
      let dot = 0.0;
      for (let d = 0; d < D; d++) {
        dot += tour.frame[r1][d] * tour.frame[r2][d];
      }
      const expected = (r1 === r2) ? 1.0 : 0.0;
      assert(Math.abs(dot - expected) < 1e-10,
        `Tour frame rows must be orthonormal at (${r1}, ${r2}): got ${dot}`);
    }
  }

  // Test point projection through tour frame
  const testPt = { coords: new Float64Array(D).fill(1.0) };
  const proj = tour.projectPoint(testPt);
  assert(typeof proj.x === 'number' && !isNaN(proj.x), 'Projected X must be a valid number');
  assert(typeof proj.y === 'number' && !isNaN(proj.y), 'Projected Y must be a valid number');
  assert(typeof proj.z === 'number' && !isNaN(proj.z), 'Projected Z must be a valid number');

  console.log('✅ Grand Tour continuous rotation and frame orthonormality verified (500 steps)');
}

// -----------------------------------------------------------------------------
// 5. Fisher LDA Cluster Discriminant Test
// -----------------------------------------------------------------------------
{
  const D = 32;
  // Two distinct clusters separated primarily along dimension 7
  const cl1 = { id: 0, anchor: new Float64Array(D), members: 50 };
  const cl2 = { id: 1, anchor: new Float64Array(D), members: 50 };
  cl1.anchor[7] = -2.0;
  cl2.anchor[7] = 2.0;

  const lda = HighDEngine.computeFisherLDA([cl1, cl2], D);
  assert.strictEqual(lda.components.length, 3, 'LDA must return 3 projection components');

  // The primary component should align with dimension 7
  const v0 = lda.components[0];
  assert(Math.abs(Math.abs(v0[7]) - 1.0) < 1e-9,
    `Primary LDA vector must align with separation dimension 7, got ${v0[7]}`);
  console.log('✅ Fisher LDA cluster separation projection verified');
}

// -----------------------------------------------------------------------------
// 6. Tomographic Slicing Filter Test
// -----------------------------------------------------------------------------
{
  const ptInside = { coords: [0.0, 0.0, 0.0, 0.15] };
  const ptOutside = { coords: [0.0, 0.0, 0.0, 0.85] };

  const resIn = HighDEngine.isPointInSlice(ptInside, 3, 0.0, 0.3);
  assert.strictEqual(resIn.inside, true, 'ptInside must be inside window');
  assert(resIn.alpha > 0.5, 'Alpha must be high for inside point');

  const resOut = HighDEngine.isPointInSlice(ptOutside, 3, 0.0, 0.3);
  assert.strictEqual(resOut.inside, false, 'ptOutside must be outside window');
  assert.strictEqual(resOut.alpha, 0.06, 'Alpha must be ghost alpha for outside point');

  console.log('✅ Tomographic slicing filter bounds and falloff verified');
}

// -----------------------------------------------------------------------------
// 7. Auto-Pick Best 3D Triplet Test
// -----------------------------------------------------------------------------
{
  const D = 32;
  const clA = { id: 0, anchor: new Float64Array(D) };
  const clB = { id: 1, anchor: new Float64Array(D) };
  // Large variance in dims 4, 9, 15
  clA.anchor[4] = -5.0; clB.anchor[4] = 5.0;
  clA.anchor[9] = -4.0; clB.anchor[9] = 4.0;
  clA.anchor[15] = -3.0; clB.anchor[15] = 3.0;

  const triplet = HighDEngine.findBestSeparatingTriplet([clA, clB], null, D);
  assert.deepStrictEqual(triplet, [4, 9, 15],
    `Best triplet should be [4, 9, 15], got ${JSON.stringify(triplet)}`);
  console.log('✅ Auto-Pick Best 3D separation optimizer verified');
}

// -----------------------------------------------------------------------------
// 8. 3D Biplot Rays Test
// -----------------------------------------------------------------------------
{
  const D = 32;
  const frame = [
    new Float64Array(D),
    new Float64Array(D),
    new Float64Array(D)
  ];
  frame[0][0] = 1.0;
  frame[1][1] = 1.0;
  frame[2][2] = 1.0;

  const rays = HighDEngine.getBiplotRays(frame, D, 2.0);
  assert.strictEqual(rays.length, D, 'Must return D rays');
  assert.strictEqual(rays[0].x, 2.0, 'Ray 0 x should be 2.0');
  assert.strictEqual(rays[1].y, 2.0, 'Ray 1 y should be 2.0');
  assert.strictEqual(rays[2].z, 2.0, 'Ray 2 z should be 2.0');
  console.log('✅ 3D Biplot basis rays verified');
}

console.log('--- ALL HIGH-D ENGINE TESTS PASSED SUCCESFULLY ---');
