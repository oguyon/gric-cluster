/**
 * Quick smoke test for the GRIC WASM clustering module.
 * Run with: node tests/wasm_smoke_test.js
 */

const path = require('path');

async function main() {
  const wasmPath = path.resolve(
    __dirname, '../site/simulator/wasm/gric_cluster.js'
  );
  const GricClusterModule = require(wasmPath);

  console.log('Loading WASM module...');
  const Module = await GricClusterModule();
  console.log('WASM module loaded.');

  // Wrap C functions
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

  // Init: rlim=0.1, maxcl=256, maxfr=10000, ndim=2
  // All acceleration off for smoke test
  const handle = init(
    0.1,   // rlim
    256,   // maxnbclust
    10000, // maxnbfr
    2,     // ndim
    0, 0, 0,         // entropy, te4, te5
    0,               // pred_mode
    2,               // pred_h
    0,               // gprob
    0.0,             // tm_mixing
    0, 0, 0,         // soft_bayesian, xtile, sparse
    0.75, 1.5,       // entropy gates
    0,               // entropy_fast
    1.0,             // bayesian sigma
    0,               // maxcl_strategy
    0.1,             // discard_fraction
    20               // max_gprob_visitors
  );

  if (!handle) {
    console.error('FAIL: wasm_cluster_init returned null');
    process.exit(1);
  }
  console.log('Handle:', handle);

  // Allocate coords buffer on WASM heap
  const coordsPtr = Module._malloc(2 * 8); // 2 doubles

  // Process a 2D spiral dataset (100 points)
  const N = 100;
  for (let i = 0; i < N; i++) {
    const t = (i / N) * Math.PI * 4;
    const r = 0.1 + 0.4 * (i / N);
    const x = r * Math.cos(t);
    const y = r * Math.sin(t);

    Module.setValue(coordsPtr, x, 'double');
    Module.setValue(coordsPtr + 8, y, 'double');

    const assigned = processFrame(handle, coordsPtr, 2);
    if (i < 5 || i === N - 1) {
      console.log(
        `Frame ${i}: (${x.toFixed(3)}, ${y.toFixed(3)})` +
        ` -> Cluster ${assigned}`
      );
    }
  }

  const K = getNumClusters(handle);
  console.log(`\nTotal clusters after ${N} frames: ${K}`);

  // Cleanup
  Module._free(coordsPtr);
  clusterFree(handle);

  if (K > 0 && K < N) {
    console.log('\n✅ WASM smoke test PASSED');
  } else {
    console.log('\n❌ WASM smoke test FAILED');
    process.exit(1);
  }
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
