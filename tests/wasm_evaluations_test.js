/**
 * Test for wasm_cluster_get_evaluations API.
 * Run with: node tests/wasm_evaluations_test.js
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

  const init = Module.cwrap('wasm_cluster_init', 'number', [
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number', 'number',
  ]);

  const processFrame = Module.cwrap(
    'wasm_cluster_process_frame', 'number',
    ['number', 'number', 'number']
  );

  const getEvaluations = Module.cwrap(
    'wasm_cluster_get_evaluations', 'number',
    ['number', 'number', 'number', 'number']
  );

  const clusterFree = Module.cwrap(
    'wasm_cluster_free', null, ['number']
  );

  const rlim = 0.1;
  const handle = init(
    rlim,  // rlim
    256,   // maxnbclust
    10000, // maxnbfr
    2,     // ndim
    0, 0, 0,         // entropy, te4, te5
    0,               // pred_mode
    2,               // pred_h
    0,               // gprob
    0.0,             // tm_mixing
    0, 0, 0, 0,      // soft_bayesian, xtile, sparse, sparse_extra
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

  const coordsPtr = Module._malloc(2 * 8);
  const evalIndicesPtr = Module._malloc(256 * 4);
  const evalDistsPtr = Module._malloc(256 * 8);

  // Frame 0: (0.0, 0.0) -> creates Cluster 0
  Module.setValue(coordsPtr, 0.0, 'double');
  Module.setValue(coordsPtr + 8, 0.0, 'double');
  let assigned = processFrame(handle, coordsPtr, 2);
  let numEvals = getEvaluations(handle, evalIndicesPtr, evalDistsPtr, 256);
  console.log(`Frame 0: assigned to C${assigned}, numEvals=${numEvals}`);
  if (numEvals !== 0) {
    console.error(`FAIL: Expected 0 evaluations on frame 0, got ${numEvals}`);
    process.exit(1);
  }

  // Frame 1: (0.5, 0.5) -> tested against C0 (dist ~0.707 > rlim), creates C1
  Module.setValue(coordsPtr, 0.5, 'double');
  Module.setValue(coordsPtr + 8, 0.5, 'double');
  assigned = processFrame(handle, coordsPtr, 2);
  numEvals = getEvaluations(handle, evalIndicesPtr, evalDistsPtr, 256);
  console.log(`Frame 1: assigned to C${assigned}, numEvals=${numEvals}`);
  if (numEvals !== 1) {
    console.error(`FAIL: Expected 1 evaluation on frame 1, got ${numEvals}`);
    process.exit(1);
  }
  let c0Id = Module.getValue(evalIndicesPtr, 'i32');
  let c0Dist = Module.getValue(evalDistsPtr, 'double');
  console.log(`  Eval 0: Target C${c0Id}, dist=${c0Dist.toFixed(4)} (mismatch: > rlim)`);
  if (c0Id !== 0 || c0Dist <= rlim) {
    console.error('FAIL: Evaluation 0 mismatch');
    process.exit(1);
  }

  // Frame 2: (0.52, 0.51) -> tested against C1 (dist ~0.0224 <= rlim), matches C1
  Module.setValue(coordsPtr, 0.52, 'double');
  Module.setValue(coordsPtr + 8, 0.51, 'double');
  assigned = processFrame(handle, coordsPtr, 2);
  numEvals = getEvaluations(handle, evalIndicesPtr, evalDistsPtr, 256);
  console.log(`Frame 2: assigned to C${assigned}, numEvals=${numEvals}`);
  if (numEvals < 1) {
    console.error(`FAIL: Expected at least 1 evaluation on frame 2, got ${numEvals}`);
    process.exit(1);
  }
  let lastEvalId = Module.getValue(evalIndicesPtr + (numEvals - 1) * 4, 'i32');
  let lastEvalDist = Module.getValue(evalDistsPtr + (numEvals - 1) * 8, 'double');
  console.log(`  Last eval: Target C${lastEvalId}, dist=${lastEvalDist.toFixed(4)} (match: <= rlim)`);
  if (lastEvalId !== assigned || lastEvalDist > rlim) {
    console.error('FAIL: Matched evaluation mismatch');
    process.exit(1);
  }

  Module._free(coordsPtr);
  Module._free(evalIndicesPtr);
  Module._free(evalDistsPtr);
  clusterFree(handle);

  console.log('\n✅ wasm_cluster_get_evaluations test PASSED');
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
