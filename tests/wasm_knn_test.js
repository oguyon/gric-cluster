/**
 * Unit test for the GRIC WASM k-NN module (gric-knn parity).
 * Run with: node tests/wasm_knn_test.js
 */

const path = require('path');
const assert = require('assert');

async function main() {
  const wasmPath = path.resolve(__dirname, '../docs/simulator/wasm/gric_cluster.js');
  const GricClusterModule = require(wasmPath);

  console.log('Loading WASM module...');
  const Module = await GricClusterModule();
  console.log('WASM module loaded.');

  const init = Module.cwrap('wasm_cluster_init', 'number', [
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number',
  ]);

  const processFrame = Module.cwrap(
    'wasm_cluster_process_frame', 'number',
    ['number', 'number', 'number']
  );

  const getNumClusters = Module.cwrap(
    'wasm_cluster_get_num_clusters', 'number',
    ['number']
  );

  const clusterFree = Module.cwrap(
    'wasm_cluster_free', null, ['number']
  );

  const knnRunSearch = Module.cwrap('wasm_knn_run_search', 'number', [
    'number', 'number', 'number', 'number', 'number', 'number',
    'number', 'number', 'number', 'number', 'number', 'number',
    'number'
  ]);

  // 1. Ingest 200 spiral frames into clustering engine
  const handle = init(
    0.1, 256, 10000, 2,
    0, 1, 0, 0, 2, 0, 0.0, 0, 0, 0,
    0.75, 1.5, 0, 1.0, 0, 0.1, 20
  );

  const N = 200;
  const points = [];
  const coordsPtr = Module._malloc(2 * 8);

  for (let i = 0; i < N; i++) {
    const t = (i / 100) * 4 * Math.PI;
    const r = 0.1 + 0.08 * t;
    const x = r * Math.cos(t);
    const y = r * Math.sin(t);
    points.push({ x, y });

    Module.HEAPF64[coordsPtr >> 3] = x;
    Module.HEAPF64[(coordsPtr >> 3) + 1] = y;
    processFrame(handle, coordsPtr, 2);
  }
  Module._free(coordsPtr);

  const numClusters = getNumClusters(handle);
  console.log(`Clustering formed ${numClusters} clusters over ${N} frames.`);
  assert(numClusters > 5, 'Should form at least 5 clusters');

  // 2. Prepare k-NN search buffers
  const k = 10;
  const dtmin = 1;
  const pointsPtr = Module._malloc(N * 2 * 8);
  const indicesPtr = Module._malloc(N * k * 4);
  const distsPtr = Module._malloc(N * k * 8);
  const telemPtr = Module._malloc(8 * 8);

  for (let i = 0; i < N; i++) {
    Module.HEAPF64[(pointsPtr >> 3) + i * 2] = points[i].x;
    Module.HEAPF64[(pointsPtr >> 3) + i * 2 + 1] = points[i].y;
  }

  // 3. Run wasm_knn_run_search
  const res = knnRunSearch(
    handle,
    pointsPtr,
    N,
    2,
    k,
    dtmin,
    0, // past_only
    0, // future_only
    0.0, // epsilon
    0.0, // rlim_cutoff
    indicesPtr,
    distsPtr,
    telemPtr
  );

  assert.strictEqual(res, 0, 'wasm_knn_run_search returned non-zero error');

  const indices = new Int32Array(N * k);
  const distances = new Float64Array(N * k);
  indices.set(Module.HEAP32.subarray(indicesPtr >> 2, (indicesPtr >> 2) + N * k));
  distances.set(Module.HEAPF64.subarray(distsPtr >> 3, (distsPtr >> 3) + N * k));

  const tOffset = telemPtr >> 3;
  const queries = Module.HEAPF64[tOffset];
  const distCalls = Module.HEAPF64[tOffset + 1];
  const l1Pruned = Module.HEAPF64[tOffset + 2];
  const l2Pruned = Module.HEAPF64[tOffset + 3];
  const l3Pruned = Module.HEAPF64[tOffset + 4];
  const searchTimeMs = Module.HEAPF64[tOffset + 7];

  console.log(`k-NN Search Telemetry:`);
  console.log(`- Queries: ${queries}`);
  console.log(`- Distance Calculations: ${distCalls} (vs ${N * N} brute-force)`);
  console.log(`- Pruning Efficiency: ${((1 - distCalls / (N * N)) * 100).toFixed(2)}%`);
  console.log(`- L1 Clusters Pruned: ${l1Pruned}`);
  console.log(`- L2 Anchors Pruned: ${l2Pruned}`);
  console.log(`- L3 Annular Pruned: ${l3Pruned}`);
  console.log(`- Search Time: ${searchTimeMs.toFixed(3)} ms`);

  assert.strictEqual(queries, N, 'Queries count must match N');
  assert(distCalls < N * N * 0.25, 'Metric pruning should eliminate >75% of distance calls');

  // Verify distance ordering for each query
  for (let q = 0; q < N; q++) {
    for (let r = 0; r < k - 1; r++) {
      const d1 = distances[q * k + r];
      const d2 = distances[q * k + r + 1];
      assert(d1 <= d2 + 1e-9, `Neighbors for query ${q} must be sorted by ascending distance`);
    }
  }

  console.log(`✅ Sample #0 Top-5 Nearest Neighbors:`);
  for (let r = 0; r < 5; r++) {
    console.log(`  #${r + 1}: Neighbor Frame #${indices[r]} (Distance = ${distances[r].toFixed(5)})`);
  }

  Module._free(pointsPtr);
  Module._free(indicesPtr);
  Module._free(distsPtr);
  Module._free(telemPtr);
  clusterFree(handle);

  console.log('\n✅ ALL WASM k-NN TESTS PASSED SUCCESSFULLY!');
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
