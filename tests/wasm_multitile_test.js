/**
 * Multi-tile WASM API test.
 * Exercises the axis-decomposition multi-tile
 * clustering for 2D and 3D data.
 *
 * Run with: node tests/wasm_multitile_test.js
 */

const path = require('path');

async function main() {
  const wasmPath = path.resolve(
    __dirname,
    '../site/simulator/wasm/gric_cluster.js'
  );
  const GricClusterModule = require(wasmPath);

  console.log('Loading WASM module...');
  const M = await GricClusterModule();
  console.log('WASM module loaded.\n');

  /* ---- Wrap multi-tile C functions ---- */
  const mtInit = M.cwrap(
    'wasm_multitile_init', 'number', [
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
      'number', 'number', 'number', 'number',
      'number', 'number',
    ]
  );
  const mtProcess = M.cwrap(
    'wasm_multitile_process_frame', 'number',
    ['number', 'number', 'number']
  );
  const mtGetNumTiles = M.cwrap(
    'wasm_multitile_get_num_tiles', 'number',
    ['number']
  );
  const mtGetNumTileCl = M.cwrap(
    'wasm_multitile_get_num_tile_clusters',
    'number', ['number', 'number']
  );
  const mtGetTileCl = M.cwrap(
    'wasm_multitile_get_tile_clusters', null,
    ['number', 'number', 'number', 'number']
  );
  const mtGetTuples = M.cwrap(
    'wasm_multitile_get_tuples', 'number',
    ['number', 'number', 'number',
     'number', 'number']
  );
  const mtReset = M.cwrap(
    'wasm_multitile_reset', null, ['number']
  );
  const mtFree = M.cwrap(
    'wasm_multitile_free', null, ['number']
  );

  let failures = 0;

  function check(name, cond) {
    if (cond) {
      console.log(`  ✅ ${name}`);
    } else {
      console.log(`  ❌ ${name}`);
      failures++;
    }
  }

  /* ========================================
   * TEST 1: 2D axis decomposition (2 tiles)
   * ======================================== */
  console.log('TEST 1: 2D axis decomposition');
  {
    const handle = mtInit(
      0.1, 256, 10000, 2,
      0, 0, 0,
      0, 2, 0,
      0.0, 0,
      0, 0, 0,
      0.75, 1.5,
      0, 1.0,
      0, 0.1, 20
    );
    check('Init returns handle', handle !== 0);

    const numTiles = mtGetNumTiles(handle);
    check('2 tiles for 2D', numTiles === 2);

    /* Feed a 2D spiral (100 points) */
    const coordsPtr = M._malloc(2 * 8);
    const N = 100;
    for (let i = 0; i < N; i++) {
      const t = (i / N) * Math.PI * 4;
      const r = 0.1 + 0.4 * (i / N);
      const x = r * Math.cos(t);
      const y = r * Math.sin(t);
      M.setValue(coordsPtr, x, 'double');
      M.setValue(coordsPtr + 8, y, 'double');
      mtProcess(handle, coordsPtr, 2);
    }
    M._free(coordsPtr);

    const kX = mtGetNumTileCl(handle, 0);
    const kY = mtGetNumTileCl(handle, 1);
    console.log(`  Tile X clusters: ${kX}`);
    console.log(`  Tile Y clusters: ${kY}`);
    check('Tile X has clusters', kX > 0);
    check('Tile Y has clusters', kY > 0);
    check(
      '1D clusters < total frames',
      kX < N && kY < N
    );

    /* Read per-tile anchors */
    const anchorPtr = M._malloc(256 * 8);
    const memberPtr = M._malloc(256 * 4);
    mtGetTileCl(handle, 0, anchorPtr, memberPtr);
    const x0 = M.getValue(anchorPtr, 'double');
    const m0 = M.getValue(memberPtr, 'i32');
    check(
      'Tile X anchor 0 has data',
      !isNaN(x0) && m0 > 0
    );
    M._free(anchorPtr);
    M._free(memberPtr);

    /* Read joint tuples */
    const maxT = 1024;
    const flatPtr =
      M._malloc(maxT * numTiles * 4);
    const countPtr = M._malloc(maxT * 4);
    const activePtr = M._malloc(maxT * 4);
    const numUnique = mtGetTuples(
      handle, flatPtr, countPtr, activePtr, maxT
    );
    console.log(`  Unique tuples: ${numUnique}`);
    check('Has joint tuples', numUnique > 0);
    check(
      'Unique tuples <= total frames',
      numUnique <= N
    );

    /* Sum tuple counts == total frames */
    let sumCounts = 0;
    for (let u = 0; u < numUnique; u++) {
      sumCounts +=
        M.getValue(countPtr + u * 4, 'i32');
    }
    check(
      'Tuple counts sum to N',
      sumCounts === N
    );
    M._free(flatPtr);
    M._free(countPtr);
    M._free(activePtr);

    /* Test reset */
    mtReset(handle);
    const kAfter = mtGetNumTileCl(handle, 0);
    check('Reset clears tile X', kAfter === 0);

    mtFree(handle);
    console.log('');
  }

  /* ========================================
   * TEST 2: 3D axis decomposition (3 tiles)
   * ======================================== */
  console.log('TEST 2: 3D axis decomposition');
  {
    const handle = mtInit(
      0.1, 256, 10000, 3,
      0, 0, 0,
      0, 2, 0,
      0.0, 0,
      0, 0, 0,
      0.75, 1.5,
      0, 1.0,
      0, 0.1, 20
    );
    check('Init 3D handle', handle !== 0);

    const numTiles = mtGetNumTiles(handle);
    check('3 tiles for 3D', numTiles === 3);

    /* Feed a 3D torus (200 points) */
    const coordsPtr = M._malloc(3 * 8);
    const N = 200;
    const R = 0.3;
    const r = 0.1;
    for (let i = 0; i < N; i++) {
      const theta =
        (i / N) * Math.PI * 2;
      const phi =
        (i / N) * Math.PI * 6;
      const x =
        (R + r * Math.cos(phi)) * Math.cos(theta);
      const y =
        (R + r * Math.cos(phi)) * Math.sin(theta);
      const z = r * Math.sin(phi);
      M.setValue(coordsPtr, x, 'double');
      M.setValue(coordsPtr + 8, y, 'double');
      M.setValue(coordsPtr + 16, z, 'double');
      mtProcess(handle, coordsPtr, 3);
    }
    M._free(coordsPtr);

    const kX = mtGetNumTileCl(handle, 0);
    const kY = mtGetNumTileCl(handle, 1);
    const kZ = mtGetNumTileCl(handle, 2);
    console.log(
      `  Tiles: X=${kX}, Y=${kY}, Z=${kZ}`
    );
    check('All 3 tiles have clusters',
      kX > 0 && kY > 0 && kZ > 0
    );

    /* Joint tuples */
    const maxT = 2048;
    const flatPtr =
      M._malloc(maxT * numTiles * 4);
    const countPtr = M._malloc(maxT * 4);
    const activePtr = M._malloc(maxT * 4);
    const numUnique = mtGetTuples(
      handle, flatPtr, countPtr, activePtr, maxT
    );
    console.log(`  Unique 3D tuples: ${numUnique}`);
    check(
      '3D tuples exist', numUnique > 0
    );

    let sumCounts = 0;
    for (let u = 0; u < numUnique; u++) {
      sumCounts +=
        M.getValue(countPtr + u * 4, 'i32');
    }
    check(
      '3D tuple counts sum to N',
      sumCounts === N
    );
    M._free(flatPtr);
    M._free(countPtr);
    M._free(activePtr);

    mtFree(handle);
    console.log('');
  }

  /* ========================================
   * TEST 3: Cross-tile priors (xtile_mode=1)
   * ======================================== */
  console.log('TEST 3: Cross-tile CPT priors');
  {
    const handle = mtInit(
      0.1, 256, 10000, 2,
      0, 0, 0,
      0, 2, 0,
      0.0, 0,
      1, 0, 0,   // xtile_mode=1
      0.75, 1.5,
      0, 1.0,
      0, 0.1, 20
    );
    check('Init xtile handle', handle !== 0);

    const coordsPtr = M._malloc(2 * 8);
    for (let i = 0; i < 50; i++) {
      const t = (i / 50) * Math.PI * 2;
      M.setValue(
        coordsPtr, 0.3 * Math.cos(t), 'double'
      );
      M.setValue(
        coordsPtr + 8, 0.3 * Math.sin(t), 'double'
      );
      mtProcess(handle, coordsPtr, 2);
    }
    M._free(coordsPtr);

    const kX = mtGetNumTileCl(handle, 0);
    const kY = mtGetNumTileCl(handle, 1);
    check(
      'xtile tiles have clusters',
      kX > 0 && kY > 0
    );

    mtFree(handle);
    console.log('');
  }

  /* ---- Summary ---- */
  console.log('='.repeat(50));
  if (failures === 0) {
    console.log(
      '✅ ALL MULTI-TILE WASM TESTS PASSED'
    );
  } else {
    console.log(
      `❌ ${failures} CHECK(S) FAILED`
    );
    process.exit(1);
  }
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
