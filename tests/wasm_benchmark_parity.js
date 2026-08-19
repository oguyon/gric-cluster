/**
 * tests/wasm_benchmark_parity.js
 * Multi-benchmark parity validation between C/WASM and JavaScript engines.
 */

const path = require('path');
const wasmPath = path.resolve(__dirname, '../site/simulator/wasm/gric_cluster.js');
const GricClusterModule = require(wasmPath);

function generateBenchmark(type, N = 1000) {
  const points = [];
  if (type === '2Dspiral') {
    for (let i = 0; i < N; i++) {
      const t = i / N;
      const r = t * 0.92;
      const theta = 2.0 * Math.PI * 2.0 * t;
      points.push({ x: r * Math.cos(theta), y: r * Math.sin(theta), z: 0.0 });
    }
  } else if (type === '3Dspiral') {
    const cos45 = 0.70710678, sin45 = 0.70710678;
    for (let i = 0; i < N; i++) {
      const t = i / N;
      const r = 0.12 + 0.68 * t;
      const theta = 2.0 * Math.PI * 4.0 * t;
      const x0 = r * Math.cos(theta);
      const y0 = r * Math.sin(theta);
      const z0 = (t - 0.5) * 1.4;
      points.push({ x: x0 * cos45 - z0 * sin45, y: y0, z: x0 * sin45 + z0 * cos45 });
    }
  } else if (type === '2Dwalk') {
    let cx = 0, cy = 0;
    for (let i = 0; i < N; i++) {
      const theta = Math.sin(i * 12.9898) * Math.PI * 2.0;
      cx = Math.max(-0.9, Math.min(0.9, cx + 0.04 * Math.cos(theta)));
      cy = Math.max(-0.9, Math.min(0.9, cy + 0.04 * Math.sin(theta)));
      points.push({ x: cx, y: cy, z: 0.0 });
    }
  } else if (type === '3Dtorus') {
    for (let i = 0; i < N; i++) {
      const t = i / N;
      const u = t * Math.PI * 2 * 3;
      const v = t * Math.PI * 2 * 7;
      const R = 0.60, r = 0.24;
      points.push({
        x: (R + r * Math.cos(v)) * Math.cos(u),
        y: (R + r * Math.cos(v)) * Math.sin(u),
        z: r * Math.sin(v)
      });
    }
  }
  return points;
}

async function runParitySuite() {
  const Module = await GricClusterModule();
  const init = Module.cwrap('wasm_cluster_init', 'number', [
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number','number','number','number',
    'number'
  ]);
  const processFrame = Module.cwrap('wasm_cluster_process_frame', 'number', ['number', 'number', 'number']);
  const getNumClusters = Module.cwrap('wasm_cluster_get_num_clusters', 'number', ['number']);
  const getTelemetry = Module.cwrap('wasm_cluster_get_telemetry', null, ['number', 'number', 'number']);
  const clusterFree = Module.cwrap('wasm_cluster_free', null, ['number']);

  const benchmarks = ['2Dspiral', '3Dspiral', '2Dwalk', '3Dtorus'];
  const coordsPtr = Module._malloc(3 * 8);
  const telemPtr = Module._malloc(32 * 8);
  const lenPtr = Module._malloc(4);

  console.log('================================================================================');
  console.log('BENCHMARK PARITY VALIDATION: C/WASM vs JAVASCRIPT (Greedy + 3P Pruning)');
  console.log('================================================================================');

  let allPassed = true;

  for (const bName of benchmarks) {
    const is3D = bName.startsWith('3D');
    const ndim = is3D ? 3 : 2;
    const pts = generateBenchmark(bName, 1000);

    // WASM run
    const handle = init(
      0.1, 256, 10000, ndim,
      0, 0, 0, 0, 2, 0, 0.0, 0, 0, 0,
      0.75, 1.5, 0, 1.0, 0, 0.1, 20
    );

    const wasmAssignments = [];
    for (let i = 0; i < pts.length; i++) {
      Module.setValue(coordsPtr, pts[i].x, 'double');
      Module.setValue(coordsPtr + 8, pts[i].y, 'double');
      if (is3D) Module.setValue(coordsPtr + 16, pts[i].z, 'double');
      wasmAssignments.push(processFrame(handle, coordsPtr, ndim));
    }
    const wasmK = getNumClusters(handle);
    getTelemetry(handle, telemPtr, lenPtr);
    const wasmTotal = Module.getValue(telemPtr, 'double');
    const wasmDfc = Module.getValue(telemPtr + 8, 'double');
    const wasmDcc = Module.getValue(telemPtr + 16, 'double');
    clusterFree(handle);

    // JS run
    const jsClusters = [];
    const jsDccMatrix = [];
    const jsAssignments = [];
    let jsDfc = 0, jsDccCount = 0;

    for (let i = 0; i < pts.length; i++) {
      const pt = pts[i];
      const K = jsClusters.length;
      if (K === 0) {
        jsClusters.push({ id: 0, x: pt.x, y: pt.y, z: pt.z, prob: 1.0 });
        jsDccMatrix.push([0.0]);
        jsAssignments.push(0);
        continue;
      }
      const clmembflag = new Uint8Array(K).fill(1);

      let sumProb = 0;
      for (let j = 0; j < K; j++) sumProb += (jsClusters[j].prob || 1.0);
      if (sumProb > 0) {
        for (let j = 0; j < K; j++) jsClusters[j].prob = (jsClusters[j].prob || 1.0) / sumProb;
      }

      let found = false, assigned = -1;
      while (true) {
        let maxP = -1, chosen = -1;
        for (let j = 0; j < K; j++) {
          if (clmembflag[j] && jsClusters[j].prob > maxP) {
            maxP = jsClusters[j].prob;
            chosen = j;
          }
        }
        if (chosen === -1) break;

        jsDfc++;
        const c = jsClusters[chosen];
        const dx = pt.x - c.x, dy = pt.y - c.y, dz = is3D ? (pt.z - c.z) : 0.0;
        const dfc = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dfc <= 0.1) {
          assigned = chosen;
          found = true;
          break;
        } else {
          clmembflag[chosen] = 0;
          for (let cl = 0; cl < K; cl++) {
            if (!clmembflag[cl]) continue;
            if (Math.abs(dfc - jsDccMatrix[chosen][cl]) > 0.1) clmembflag[cl] = 0;
          }
        }
      }

      if (!found) {
        assigned = jsClusters.length;
        const newRow = [];
        for (let j = 0; j < assigned; j++) {
          const c = jsClusters[j];
          const dist = Math.sqrt((pt.x - c.x)**2 + (pt.y - c.y)**2 + (is3D ? (pt.z - c.z)**2 : 0.0));
          newRow.push(dist);
          jsDccMatrix[j].push(dist);
          jsDccCount++;
        }
        newRow.push(0.0);
        jsDccMatrix.push(newRow);
        jsClusters.push({ id: assigned, x: pt.x, y: pt.y, z: pt.z, prob: 1.0 });
      }
      jsAssignments.push(assigned);
    }

    let matchCount = 0;
    for (let i = 0; i < pts.length; i++) {
      if (wasmAssignments[i] === jsAssignments[i]) matchCount++;
    }

    const matchRate = ((matchCount / pts.length) * 100).toFixed(1);
    console.log(`[${bName.padEnd(10)}] K: WASM=${wasmK}, JS=${jsClusters.length} | Match: ${matchCount}/${pts.length} (${matchRate}%) | DFC: WASM=${wasmDfc}, JS=${jsDfc} | Total: WASM=${wasmTotal}, JS=${jsDfc + jsDccCount}`);

    if (matchCount < 990) {
      allPassed = false;
    }
  }

  Module._free(coordsPtr);
  Module._free(telemPtr);
  Module._free(lenPtr);

  console.log('================================================================================');
  if (allPassed) {
    console.log('✅ ALL BENCHMARK PARITY CHECKS PASSED');
  } else {
    console.log('❌ SOME BENCHMARK PARITY CHECKS FAILED');
    process.exit(1);
  }
}

runParitySuite().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
