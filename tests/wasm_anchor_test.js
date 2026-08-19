/**
 * Verify WASM cluster anchors have correct coordinates.
 * This tests the fix for the data ownership bug where
 * all clusters were created at (0,0).
 */

const path = require('path');

async function main() {
  const wasmPath = path.resolve(
    __dirname, '../site/simulator/wasm/gric_cluster.js'
  );
  const GricClusterModule = require(wasmPath);
  const Module = await GricClusterModule();

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
    'wasm_cluster_get_num_clusters', 'number', ['number']
  );
  const getAnchors = Module.cwrap(
    'wasm_cluster_get_anchors', null,
    ['number', 'number', 'number', 'number']
  );
  const clusterFree = Module.cwrap(
    'wasm_cluster_free', null, ['number']
  );

  // Init with rlim=0.05 to get multiple clusters
  const handle = init(
    0.05, 256, 10000, 2,
    0, 0, 0, 0, 2, 0, 0.0, 0, 0, 0,
    0.75, 1.5, 0, 1.0, 0, 0.1, 20
  );

  const coordsPtr = Module._malloc(2 * 8);

  // Process distinct well-separated points
  const points = [
    [0.1, 0.1],   // cluster 0
    [0.5, 0.5],   // cluster 1 (far from 0)
    [-0.3, 0.4],  // cluster 2 (far from both)
    [0.11, 0.11], // should match cluster 0
    [0.51, 0.49], // should match cluster 1
  ];

  for (let i = 0; i < points.length; i++) {
    Module.setValue(coordsPtr, points[i][0], 'double');
    Module.setValue(coordsPtr + 8, points[i][1], 'double');
    const a = processFrame(handle, coordsPtr, 2);
    console.log(
      `Frame ${i}: (${points[i][0]}, ${points[i][1]})` +
      ` -> Cluster ${a}`
    );
  }

  const K = getNumClusters(handle);
  console.log(`\nTotal clusters: ${K}`);

  // Read back anchor coordinates
  const anchorsPtr = Module._malloc(K * 2 * 8);
  const membersPtr = Module._malloc(K * 4);
  getAnchors(handle, anchorsPtr, membersPtr, 2);

  let allCorrect = true;
  for (let i = 0; i < K; i++) {
    const ax = Module.getValue(anchorsPtr + i * 16, 'double');
    const ay = Module.getValue(anchorsPtr + i * 16 + 8, 'double');
    const m = Module.getValue(membersPtr + i * 4, 'i32');
    console.log(
      `  Cluster ${i}: anchor=(${ax.toFixed(3)}, ${ay.toFixed(3)})` +
      ` members=${m}`
    );
    if (Math.abs(ax) < 0.001 && Math.abs(ay) < 0.001 && i > 0) {
      console.log(`  ❌ WRONG: Cluster ${i} at origin!`);
      allCorrect = false;
    }
  }

  Module._free(coordsPtr);
  Module._free(anchorsPtr);
  Module._free(membersPtr);
  clusterFree(handle);

  if (allCorrect && K >= 3) {
    console.log('\n✅ Anchor coordinate test PASSED');
  } else {
    console.log('\n❌ Anchor coordinate test FAILED');
    process.exit(1);
  }
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
