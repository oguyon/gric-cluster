/**
 * Test for Image Mode & Bouncing Balls Generator with WASM Clustering (ndim=1024).
 * Run with: node tests/wasm_image_test.js
 */

const path = require('path');
const fs = require('fs');

// Load image_benchmarks.js in Node context
const imgBenchCode = fs.readFileSync(
  path.resolve(__dirname, '../docs/simulator/js/image_benchmarks.js'),
  'utf8'
);
eval(imgBenchCode);

async function main() {
  console.log('--- 1. Testing Image Benchmarks Generator ---');

  const N = 100;
  const frames1 = generateImageBenchmark('img-ball-1', N);
  console.log(`Generated ${frames1.length} frames for img-ball-1`);
  if (frames1.length !== N) {
    throw new Error(`Expected ${N} frames, got ${frames1.length}`);
  }
  if (frames1[0].length !== 1024) {
    throw new Error(`Expected 1024 pixels per frame, got ${frames1[0].length}`);
  }

  // Check pixel intensity bounds
  let maxIntensity1 = 0;
  for (let i = 0; i < N; i++) {
    for (let p = 0; p < 1024; p++) {
      if (frames1[i][p] > maxIntensity1) maxIntensity1 = frames1[i][p];
    }
  }
  console.log(`img-ball-1 Max Intensity: ${maxIntensity1.toFixed(3)} (expected <= 1.0)`);
  if (maxIntensity1 <= 0 || maxIntensity1 > 1.0) {
    throw new Error(`Unexpected max intensity for 1-ball: ${maxIntensity1}`);
  }

  const frames3 = generateImageBenchmark('img-ball-3', N);
  console.log(`Generated ${frames3.length} frames for img-ball-3`);
  if (frames3.length !== N || frames3[0].length !== 1024) {
    throw new Error('img-ball-3 frame dimension mismatch');
  }

  console.log('Image benchmark generation: PASS');

  console.log('\n--- 2. Testing WASM 1024-D Image Clustering ---');

  // Try locating WASM module in docs/simulator/wasm or site/simulator/wasm
  let wasmPath = path.resolve(__dirname, '../docs/simulator/wasm/gric_cluster.js');
  if (!fs.existsSync(wasmPath)) {
    wasmPath = path.resolve(__dirname, '../site/simulator/wasm/gric_cluster.js');
  }

  const GricClusterModule = require(wasmPath);
  const Module = await GricClusterModule();

  const init = Module.cwrap('wasm_cluster_init', 'number', [
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number','number'
  ]);

  const processFrame = Module.cwrap(
    'wasm_cluster_process_frame', 'number',
    ['number', 'number', 'number']
  );

  const getNumClusters = Module.cwrap(
    'wasm_cluster_get_num_clusters', 'number',
    ['number']
  );

  const getAnchors = Module.cwrap(
    'wasm_cluster_get_anchors', null,
    ['number', 'number', 'number', 'number']
  );

  const clusterFree = Module.cwrap(
    'wasm_cluster_free', null, ['number']
  );

  const ndim = 1024;
  const rlim = 2.5; // typical distance threshold for 1024-D image
  const maxnbclust = 256;
  const maxnbfr = 10000;

  const handle = init(
    rlim,
    maxnbclust,
    maxnbfr,
    ndim,
    0, 1, 0, // entropy=0, te4=1, te5=0
    0, 2, 0, // pred=0, gprob=0
    0.0,
    0, 0, 0,
    0, 0,
    0.75, 1.5,
    0,
    1.0,
    0,
    0.1,
    20
  );

  if (!handle) {
    throw new Error('wasm_cluster_init returned 0 for ndim=1024');
  }

  const coordsPtr = Module._malloc(ndim * 8);
  const numTestFrames = 200;
  const testFrames = generateImageBenchmark('img-ball-1', numTestFrames);

  console.log(`Processing ${numTestFrames} 32x32 frames through WASM...`);
  const assignments = [];

  for (let f = 0; f < numTestFrames; f++) {
    const frame = testFrames[f];
    for (let p = 0; p < ndim; p++) {
      Module.setValue(coordsPtr + p * 8, frame[p], 'double');
    }
    const assigned = processFrame(handle, coordsPtr, ndim);
    assignments.push(assigned);
  }

  const numClusters = getNumClusters(handle);
  console.log(`Completed processing. Total clusters formed: ${numClusters}`);
  if (numClusters <= 0) {
    throw new Error('Expected at least 1 cluster formed');
  }

  // Verify anchors read
  const anchorsPtr = Module._malloc(numClusters * ndim * 8);
  const membersPtr = Module._malloc(numClusters * 4);
  getAnchors(handle, anchorsPtr, membersPtr, ndim);

  let anchorNonZero = 0;
  for (let p = 0; p < ndim; p++) {
    if (Module.getValue(anchorsPtr + p * 8, 'double') > 0) {
      anchorNonZero++;
    }
  }
  console.log(`Anchor C0 has ${anchorNonZero} non-zero pixels out of 1024`);
  if (anchorNonZero === 0) {
    throw new Error('Anchor C0 has zero pixels');
  }

  // Cleanup
  Module._free(coordsPtr);
  Module._free(anchorsPtr);
  Module._free(membersPtr);
  clusterFree(handle);

  console.log('WASM 1024-D clustering test: PASS');
}

main().catch(err => {
  console.error('Test FAILED:', err);
  process.exit(1);
});
