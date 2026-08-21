/**
 * GRIC Simulator - wasm_worker.js
 * Dedicated Web Worker for background C/WebAssembly clustering execution.
 *
 * Runs the compiled C/WASM engine off the main thread, processing heavy batches
 * at maximum CPU speed while streaming state snapshots back to the main UI thread.
 */

/* global importScripts, GricClusterModule */

'use strict';

let _module = null;
let _handle = null;
let _ready = false;
let _ndim = 3;
let _maxK = 256;

// Pre-allocated heap buffer pointers
let _coordsPtr = 0;
let _anchorsPtr = 0;
let _membersPtr = 0;
let _dccPtr = 0;
let _probsPtr = 0;
let _telemetryPtr = 0;
let _telemetryLenPtr = 0;

// Wrapped C functions
const _fn = {};

// Batch processing control
let _batchActive = false;
let _batchPaused = false;
let _batchData = null; // Float64Array of [x0, y0, z0, x1, y1, z1, ...]
let _batchIndex = 0;
let _batchTotal = 0;
let _batchChunkSize = 250; // frames per slice between yielding to message queue

/**
 * Load the Emscripten WASM module in Worker context.
 */
async function loadModule() {
  try {
    importScripts('../wasm/gric_cluster.js');
    if (typeof GricClusterModule !== 'function') {
      self.postMessage({ type: 'ERROR', message: 'GricClusterModule not defined in worker' });
      return;
    }
    _module = await GricClusterModule({
      locateFile: function (path) {
        if (path.endsWith('.wasm')) {
          return '../wasm/' + path;
        }
        return path;
      }
    });
    _wrapFunctions();
    _ready = true;
    self.postMessage({ type: 'READY' });
  } catch (err) {
    self.postMessage({ type: 'ERROR', message: err.message || String(err) });
  }
}

function _wrapFunctions() {
  const M = _module;
  _fn.init = M.cwrap('wasm_cluster_init', 'number', [
    'number', 'number', 'number', 'number',
    'number', 'number', 'number', 'number',
    'number', 'number', 'number', 'number',
    'number', 'number', 'number', 'number',
    'number', 'number', 'number', 'number',
    'number'
  ]);
  _fn.processFrame = M.cwrap('wasm_cluster_process_frame', 'number', ['number', 'number', 'number']);
  _fn.processBatch = M.cwrap('wasm_cluster_process_batch', 'number', ['number', 'number', 'number', 'number', 'number']);
  _fn.getNumClusters = M.cwrap('wasm_cluster_get_num_clusters', 'number', ['number']);
  _fn.getAnchors = M.cwrap('wasm_cluster_get_anchors', null, ['number', 'number', 'number', 'number']);
  _fn.getDcc = M.cwrap('wasm_cluster_get_dcc', null, ['number', 'number', 'number']);
  _fn.getProbs = M.cwrap('wasm_cluster_get_probs', null, ['number', 'number', 'number']);
  _fn.getTelemetry = M.cwrap('wasm_cluster_get_telemetry', null, ['number', 'number', 'number']);
  _fn.reset = M.cwrap('wasm_cluster_reset', null, ['number']);
  _fn.free = M.cwrap('wasm_cluster_free', null, ['number']);
}

let _batchBufferPtr = 0;
let _batchBufferCapacity = 0;

function _allocBuffers() {
  const M = _module;
  const DOUBLE = 8;
  const INT = 4;
  _coordsPtr = M._malloc(_ndim * DOUBLE);
  _anchorsPtr = M._malloc(_maxK * _ndim * DOUBLE);
  _membersPtr = M._malloc(_maxK * INT);
  _dccPtr = M._malloc(_maxK * _maxK * DOUBLE);
  _probsPtr = M._malloc(_maxK * DOUBLE);
  _telemetryPtr = M._malloc(32 * DOUBLE);
  _telemetryLenPtr = M._malloc(INT);

  _batchBufferCapacity = 1000;
  _batchBufferPtr = M._malloc(_batchBufferCapacity * _ndim * DOUBLE);
}

function _freeBuffers() {
  const M = _module;
  if (_coordsPtr) M._free(_coordsPtr);
  if (_anchorsPtr) M._free(_anchorsPtr);
  if (_membersPtr) M._free(_membersPtr);
  if (_dccPtr) M._free(_dccPtr);
  if (_probsPtr) M._free(_probsPtr);
  if (_telemetryPtr) M._free(_telemetryPtr);
  if (_telemetryLenPtr) M._free(_telemetryLenPtr);
  if (_batchBufferPtr) M._free(_batchBufferPtr);
  _coordsPtr = 0;
  _anchorsPtr = 0;
  _membersPtr = 0;
  _dccPtr = 0;
  _probsPtr = 0;
  _telemetryPtr = 0;
  _telemetryLenPtr = 0;
  _batchBufferPtr = 0;
}

function initSession(params) {
  if (!_ready) return false;
  if (_handle) {
    _fn.free(_handle);
    _freeBuffers();
    _handle = null;
  }

  _ndim = params.ndim || 3;
  _maxK = params.maxnbclust || 256;

  const te4 = (params.pruneMode === '4P' || params.pruneMode === '5P') ? 1 : 0;
  const te5 = (params.pruneMode === '5P') ? 1 : 0;
  let strategyEnum = 0;
  if (params.maxclStrategy === 'discard') strategyEnum = 1;
  else if (params.maxclStrategy === 'merge') strategyEnum = 2;

  const effectiveMaxcl = params.maxcl > 0 ? params.maxcl : _maxK;

  _handle = _fn.init(
    params.rlim || 0.1,
    effectiveMaxcl,
    params.maxnbfr || 100000,
    _ndim,
    params.entropyMode ? 1 : 0,
    te4,
    te5,
    params.predMode ? 1 : 0,
    params.predHorizon || 2,
    params.gprobMode ? 1 : 0,
    params.tmMixingCoeff || 0.0,
    params.softBayesian ? 1 : 0,
    params.xtileMode ? 1 : 0,
    params.sparseDcc ? 1 : 0, // sparse_dcc_mode
    params.sparseDccExtraEvals || 0, // sparse_dcc_extra_evals
    params.entropyGate || 0.75,
    params.entropyFirstGate || 1.5,
    params.entropyFast ? 1 : 0,
    params.softBayesianSigma || 1.0,
    strategyEnum,
    params.discardFraction || 0.1,
    params.maxVisitors || 20
  );

  if (!_handle) {
    self.postMessage({ type: 'ERROR', message: 'wasm_cluster_init returned NULL' });
    return false;
  }

  _allocBuffers();
  self.postMessage({ type: 'INIT_OK' });
  return true;
}

function getSnapshot() {
  if (!_handle) return null;
  const M = _module;
  const K = _fn.getNumClusters(_handle);

  _fn.getAnchors(_handle, _anchorsPtr, _membersPtr, _ndim);
  const anchors = [];
  for (let i = 0; i < K; i++) {
    const base = _anchorsPtr + i * _ndim * 8;
    anchors.push({
      id: i,
      x: M.getValue(base, 'double'),
      y: M.getValue(base + 8, 'double'),
      z: _ndim >= 3 ? M.getValue(base + 16, 'double') : 0.0,
      members: M.getValue(_membersPtr + i * 4, 'i32')
    });
  }

  _fn.getDcc(_handle, _dccPtr, K);
  const dccMatrix = [];
  const heapF64 = M.HEAPF64;
  if (heapF64) {
    const dccOffset = _dccPtr >> 3;
    for (let i = 0; i < K; i++) {
      dccMatrix.push(Array.from(heapF64.subarray(dccOffset + i * K, dccOffset + (i + 1) * K)));
    }
  } else {
    for (let i = 0; i < K; i++) {
      const row = [];
      for (let j = 0; j < K; j++) {
        row.push(M.getValue(_dccPtr + (i * K + j) * 8, 'double'));
      }
      dccMatrix.push(row);
    }
  }

  _fn.getProbs(_handle, _probsPtr, K);
  const probs = [];
  if (heapF64) {
    const probsOffset = _probsPtr >> 3;
    probs.push(...heapF64.subarray(probsOffset, probsOffset + K));
  } else {
    for (let i = 0; i < K; i++) {
      probs.push(M.getValue(_probsPtr + i * 8, 'double'));
    }
  }

  _fn.getTelemetry(_handle, _telemetryPtr, _telemetryLenPtr);
  const tLen = M.getValue(_telemetryLenPtr, 'i32');
  const tArr = [];
  if (heapF64) {
    const telOffset = _telemetryPtr >> 3;
    tArr.push(...heapF64.subarray(telOffset, telOffset + tLen));
  } else {
    for (let i = 0; i < tLen; i++) {
      tArr.push(M.getValue(_telemetryPtr + i * 8, 'double'));
    }
  }

  return {
    numClusters: K,
    anchors: anchors,
    dcc: dccMatrix,
    probs: probs,
    telemetry: {
      framedistCalls: tArr[0] || 0,
      framedistSample: tArr[1] || 0,
      framedistIntercluster: tArr[2] || 0,
      clustersPruned: tArr[3] || 0,
      totalFrames: tArr[4] || 0,
      lastFrameDists: tArr[5] || 0,
      lastFrameDfc: tArr[6] || 0,
      lastFrameDcc: tArr[7] || 0,
      lastAssignmentDist: tArr[8] || 0,
      numNewClusters: tArr[9] || 0,
      predAttempts: tArr[10] || 0,
      predHits: tArr[11] || 0,
      entropyGated: tArr[12] || 0,
      entropyEvaluated: tArr[13] || 0,
      entropySumInitial: tArr[14] || 0,
      entropyMaxInitial: tArr[15] || 0,
      entropyLastInitial: tArr[16] || 0,
    }
  };
}

function processSingleFrame(x, y, z) {
  if (!_handle) return -1;
  const M = _module;
  M.setValue(_coordsPtr, x, 'double');
  M.setValue(_coordsPtr + 8, y, 'double');
  if (_ndim >= 3) M.setValue(_coordsPtr + 16, z || 0.0, 'double');
  return _fn.processFrame(_handle, _coordsPtr, _ndim);
}

let _lastYieldTime = 0;

function pumpBatch() {
  if (!_batchActive || _batchPaused || !_batchData || !_handle) return;

  const M = _module;
  const startTime = performance.now();
  let isDone = false;

  while (true) {
    const count = Math.min(_batchChunkSize, _batchTotal - _batchIndex);
    if (count <= 0) {
      isDone = true;
      break;
    }

    // Ensure batch buffer capacity
    if (count > _batchBufferCapacity) {
      if (_batchBufferPtr) M._free(_batchBufferPtr);
      _batchBufferCapacity = count;
      _batchBufferPtr = M._malloc(_batchBufferCapacity * _ndim * 8);
    }

    // Copy chunk to WASM heap
    const offset = _batchIndex * _ndim;
    for (let i = 0; i < count * _ndim; i++) {
      M.setValue(_batchBufferPtr + i * 8, _batchData[offset + i], 'double');
    }

    // Process entire chunk in C in a single call
    _fn.processBatch(_handle, _batchBufferPtr, 0, count, _ndim);
    _batchIndex += count;

    isDone = (_batchIndex >= _batchTotal);
    if (isDone) {
      break;
    }

    // Check if we should yield (e.g., spent more than 30ms in this pump)
    if (performance.now() - startTime > 30) {
      break;
    }
  }

  const snapshot = getSnapshot();

  self.postMessage({
    type: isDone ? 'BATCH_DONE' : 'BATCH_PROGRESS',
    progress: _batchIndex / _batchTotal,
    processedFrames: _batchIndex,
    totalFrames: _batchTotal,
    snapshot: snapshot
  });

  if (isDone) {
    _batchActive = false;
    _batchData = null;
  } else {
    // Yield briefly to worker event loop to remain responsive to pause/stop messages
    setTimeout(pumpBatch, 0);
  }
}

self.onmessage = function (e) {
  const msg = e.data;
  if (!msg || !msg.type) return;

  switch (msg.type) {
    case 'LOAD':
      loadModule();
      break;

    case 'INIT':
      initSession(msg.params || {});
      break;

    case 'PROCESS_FRAME': {
      const assigned = processSingleFrame(msg.x, msg.y, msg.z || 0.0);
      const snapshot = msg.needSnapshot ? getSnapshot() : null;
      self.postMessage({ type: 'FRAME_PROCESSED', assigned: assigned, snapshot: snapshot });
      break;
    }

    case 'START_BATCH':
      _batchData = msg.data; // Float64Array
      _batchTotal = msg.totalFrames || Math.floor(_batchData.length / _ndim);
      _batchIndex = msg.startIndex || 0;
      _batchChunkSize = msg.chunkSize || 250;
      _batchActive = true;
      _batchPaused = false;
      pumpBatch();
      break;

    case 'PAUSE_BATCH':
      _batchPaused = true;
      self.postMessage({ type: 'BATCH_PAUSED', processedFrames: _batchIndex, snapshot: getSnapshot() });
      break;

    case 'RESUME_BATCH':
      if (_batchActive) {
        _batchPaused = false;
        pumpBatch();
      }
      break;

    case 'GET_SNAPSHOT':
      self.postMessage({ type: 'SNAPSHOT', snapshot: getSnapshot() });
      break;

    case 'RESET':
      if (_handle) _fn.reset(_handle);
      _batchActive = false;
      _batchPaused = false;
      _batchData = null;
      self.postMessage({ type: 'RESET_OK', snapshot: getSnapshot() });
      break;

    case 'DESTROY':
      if (_handle) {
        _fn.free(_handle);
        _freeBuffers();
        _handle = null;
      }
      _batchActive = false;
      _batchPaused = false;
      _batchData = null;
      self.postMessage({ type: 'DESTROY_OK' });
      break;

    default:
      console.warn('[WasmWorker] Unknown message type:', msg.type);
  }
};
