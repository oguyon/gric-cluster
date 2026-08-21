/**
 * WASM Growth Test
 *
 * Verifies that when unlimited mode is enabled,
 * the WASM engine grows its internal arrays
 * beyond the initial allocation (256) without
 * crashing or losing data.
 */

const { join } = require('path');

async function main() {
  const wasmPath = join(__dirname, '..', 'site',
    'simulator', 'wasm', 'gric_cluster.js');
  const factory = require(wasmPath);
  const Module = await factory();

  const init = Module.cwrap('wasm_cluster_init',
    'number', [
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
    ]);

  const processFrame = Module.cwrap(
    'wasm_cluster_process_frame', 'number',
    ['number', 'number', 'number']);

  const getNumClusters = Module.cwrap(
    'wasm_cluster_get_num_clusters', 'number',
    ['number']);

  const setUnlimited = Module.cwrap(
    'wasm_cluster_set_unlimited', null,
    ['number', 'number']);

  const getCapacity = Module.cwrap(
    'wasm_cluster_get_capacity', 'number',
    ['number']);

  const freeSession = Module.cwrap(
    'wasm_cluster_free', null, ['number']);

  /* Init with small capacity (64) + unlimited */
  const handle = init(
    0.01, 64, 100000, 2,
    0, 0, 0, 0, 2, 0, 0.0,
    0, 0, 0, 0.75, 1.5, 0, 1.0, 0, 0.1
  );
  if (!handle) {
    console.error('ERROR: init returned NULL');
    process.exit(1);
  }

  setUnlimited(handle, 1);

  const initCap = getCapacity(handle);
  console.log(`Initial capacity: ${initCap}`);
  if (initCap !== 64) {
    console.error(
      `FAIL: expected capacity 64, got ${initCap}`);
    process.exit(1);
  }

  /* Use a tiny rlim (0.01) so each point creates
   * a new cluster. Feed 200 points spread across
   * the space to exceed the initial 64 capacity. */
  const coordsPtr = Module._malloc(2 * 8);
  let growthDetected = false;

  for (let i = 0; i < 200; i++) {
    /* Place points far apart: grid pattern */
    const x = (i % 20) * 0.1;
    const y = Math.floor(i / 20) * 0.1;
    Module.setValue(coordsPtr, x, 'double');
    Module.setValue(coordsPtr + 8, y, 'double');

    const assigned = processFrame(handle, coordsPtr, 2);
    if (assigned < -1) {
      console.error(
        `FAIL: processFrame returned ${assigned} ` +
        `at frame ${i} (OOM or stop?)`);
      process.exit(1);
    }

    const cap = getCapacity(handle);
    if (cap > initCap && !growthDetected) {
      console.log(`  Growth at frame ${i}: ` +
        `capacity ${initCap} -> ${cap}`);
      growthDetected = true;
    }
  }

  const finalK = getNumClusters(handle);
  const finalCap = getCapacity(handle);
  console.log(
    `After 200 frames: ${finalK} clusters, ` +
    `capacity ${finalCap}`);

  if (!growthDetected) {
    console.error(
      'FAIL: capacity never grew beyond ' + initCap);
    process.exit(1);
  }

  if (finalK <= initCap) {
    console.error(
      `FAIL: expected > ${initCap} clusters, ` +
      `got ${finalK}`);
    process.exit(1);
  }

  if (finalCap < finalK) {
    console.error(
      `FAIL: capacity (${finalCap}) < clusters ` +
      `(${finalK})`);
    process.exit(1);
  }

  Module._free(coordsPtr);
  freeSession(handle);

  console.log('\n✅ WASM growth test PASSED');
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
