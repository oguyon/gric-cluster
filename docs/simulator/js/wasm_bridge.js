/**
 * GRIC Simulator - wasm_bridge.js
 * WebAssembly bridge layer between the GRIC C clustering engine
 * (compiled via Emscripten) and the JavaScript simulator UI.
 *
 * This module provides:
 * - GricWasm.load()        — async WASM module initialization
 * - GricWasm.init(params)  — create a new clustering session
 * - GricWasm.processFrame(x, y, z) — process one coordinate frame
 * - GricWasm.syncState()   — pull cluster state from WASM → JS
 * - GricWasm.reset()       — reset clustering state
 * - GricWasm.destroy()     — free WASM memory
 * - GricWasm.isReady()     — check if WASM engine is loaded
 */

// eslint-disable-next-line no-unused-vars
const GricWasm = (function () {
  'use strict';

  let _module = null;    // Emscripten Module instance
  let _handle = null;    // Opaque C pointer to WasmHandle
  let _ready = false;
  let _ndim = 3;
  let _maxK = 256;

  // Pre-allocated WASM heap pointers for coordinate passing
  let _coordsPtr = 0;    // double[3] on WASM heap
  let _anchorsPtr = 0;   // double[maxK * ndim]
  let _membersPtr = 0;   // int[maxK]
  let _dccPtr = 0;       // double[maxK * maxK]
  let _tmPtr = 0;        // (unused for now, JS uses int64)
  let _probsPtr = 0;     // double[maxK]
  let _evalIndicesPtr = 0; // int[maxK]
  let _evalDistsPtr = 0;   // double[maxK]
  let _telemetryPtr = 0; // double[32]
  let _telemetryLenPtr = 0; // int[1]

  // Wrapped C functions (populated after module load)
  const _fn = {};

  /**
   * Load the Emscripten WASM module.
   * Returns a Promise that resolves when the module is ready.
   */
  async function load() {
    if (_ready) return;

    if (typeof GricClusterModule !== 'function') {
      console.warn(
        '[GricWasm] GricClusterModule not found. ' +
        'Falling back to JS clustering engine.'
      );
      return;
    }

    try {
      _module = await GricClusterModule();
      _wrapFunctions();
      _ready = true;
      console.log('[GricWasm] WASM clustering engine loaded.');

      // Populate header with WASM build info
      try {
        const ver = getVersion();
        let hash = 'unknown';
        let dateStr = '';
        if (ver.includes('|')) {
          const parts = ver.split('|');
          hash = parts[0].trim();
          dateStr = parts.slice(1).join('|').trim();
        } else {
          const parts = ver.split(' ');
          hash = parts[0] || 'unknown';
          dateStr = parts.slice(1).join(' ');
        }

        const badge = document.getElementById('versionBadge');
        if (badge) {
          badge.textContent = '🔧 ' + hash + (dateStr ? ' | ' + dateStr : '');
        }
        const info = document.getElementById('headerBuildInfo');
        if (info) {
          info.textContent = 'Build: ' + ver;
        }
        const hashEl = document.getElementById('wasmBuildHash');
        if (hashEl) {
          hashEl.textContent = ver;
        }
      } catch (e) {
        /* non-critical */
      }
    } catch (err) {
      console.error(
        '[GricWasm] Failed to load WASM module:', err
      );
      _ready = false;
    }
  }

  /**
   * Wrap all exported C functions via Module.cwrap.
   */
  function _wrapFunctions() {
    const M = _module;

    _fn.init = M.cwrap('wasm_cluster_init', 'number', [
      'number', // rlim
      'number', // maxnbclust
      'number', // maxnbfr
      'number', // ndim
      'number', // entropy_mode
      'number', // te4_mode
      'number', // te5_mode
      'number', // pred_mode
      'number', // pred_h
      'number', // gprob_mode
      'number', // tm_mixing_coeff
      'number', // soft_bayesian_mode
      'number', // xtile_mode
      'number', // sparse_dcc_mode
      'number', // sparse_dcc_extra_evals
      'number', // entropy_gate_bits
      'number', // entropy_first_gate_bits
      'number', // entropy_fast_mode
      'number', // soft_bayesian_sigma_coeff
      'number', // maxcl_strategy
      'number', // discard_fraction
      'number', // max_gprob_visitors
    ]);

    _fn.processFrame = M.cwrap(
      'wasm_cluster_process_frame',
      'number',
      ['number', 'number', 'number']
    );

    _fn.getNumClusters = M.cwrap(
      'wasm_cluster_get_num_clusters',
      'number',
      ['number']
    );

    _fn.getAnchors = M.cwrap(
      'wasm_cluster_get_anchors',
      null,
      ['number', 'number', 'number', 'number']
    );

    _fn.getDcc = M.cwrap(
      'wasm_cluster_get_dcc',
      null,
      ['number', 'number', 'number']
    );

    _fn.getProbs = M.cwrap(
      'wasm_cluster_get_probs',
      null,
      ['number', 'number', 'number']
    );

    _fn.getEvaluations = M.cwrap(
      'wasm_cluster_get_evaluations',
      'number',
      ['number', 'number', 'number', 'number']
    );

    _fn.getTelemetry = M.cwrap(
      'wasm_cluster_get_telemetry',
      null,
      ['number', 'number', 'number']
    );

    _fn.reset = M.cwrap(
      'wasm_cluster_reset',
      null,
      ['number']
    );

    _fn.free = M.cwrap(
      'wasm_cluster_free',
      null,
      ['number']
    );

    // Trace / Explain API
    _fn.setTrace = M.cwrap(
      'wasm_cluster_set_trace',
      null,
      ['number', 'number', 'number']
    );

    _fn.getTraceCount = M.cwrap(
      'wasm_cluster_get_trace_count',
      'number',
      ['number']
    );

    _fn.getTraceEvents = M.cwrap(
      'wasm_cluster_get_trace_events',
      'number',
      ['number']
    );

    _fn.getTraceEventSize = M.cwrap(
      'wasm_cluster_get_trace_event_size',
      'number',
      []
    );

    _fn.getTraceHead = M.cwrap(
      'wasm_cluster_get_trace_head',
      'number',
      ['number']
    );

    _fn.getTraceFrameStart = M.cwrap(
      'wasm_cluster_get_trace_frame_start',
      'number',
      ['number']
    );

    _fn.clearTrace = M.cwrap(
      'wasm_cluster_clear_trace',
      null,
      ['number']
    );
    _fn.setUnlimited = M.cwrap(
      'wasm_cluster_set_unlimited',
      null,
      ['number', 'number']
    );
    _fn.getCapacity = M.cwrap(
      'wasm_cluster_get_capacity',
      'number',
      ['number']
    );
  }

  /**
   * Allocate WASM heap buffers for data exchange.
   */
  function _allocBuffers() {
    const M = _module;
    const DOUBLE = 8;
    const INT = 4;

    // Coordinate input buffer: double[ndim]
    _coordsPtr = M._malloc(_ndim * DOUBLE);

    // Anchor output: double[maxK * ndim]
    _anchorsPtr = M._malloc(_maxK * _ndim * DOUBLE);

    // Member counts: int[maxK]
    _membersPtr = M._malloc(_maxK * INT);

    // DCC matrix: double[maxK * maxK]
    _dccPtr = M._malloc(_maxK * _maxK * DOUBLE);

    // Probabilities: double[maxK]
    _probsPtr = M._malloc(_maxK * DOUBLE);

    // Evaluations: int[maxK] indices, double[maxK] distances
    _evalIndicesPtr = M._malloc(_maxK * INT);
    _evalDistsPtr = M._malloc(_maxK * DOUBLE);

    // Telemetry: double[32] + int[1] for length
    _telemetryPtr = M._malloc(32 * DOUBLE);
    _telemetryLenPtr = M._malloc(INT);
  }

  /**
   * Free WASM heap buffers.
   */
  function _freeBuffers() {
    const M = _module;
    if (_coordsPtr) M._free(_coordsPtr);
    if (_anchorsPtr) M._free(_anchorsPtr);
    if (_membersPtr) M._free(_membersPtr);
    if (_dccPtr) M._free(_dccPtr);
    if (_probsPtr) M._free(_probsPtr);
    if (_evalIndicesPtr) M._free(_evalIndicesPtr);
    if (_evalDistsPtr) M._free(_evalDistsPtr);
    if (_telemetryPtr) M._free(_telemetryPtr);
    if (_telemetryLenPtr) M._free(_telemetryLenPtr);
    _coordsPtr = 0;
    _anchorsPtr = 0;
    _membersPtr = 0;
    _dccPtr = 0;
    _probsPtr = 0;
    _evalIndicesPtr = 0;
    _evalDistsPtr = 0;
    _telemetryPtr = 0;
    _telemetryLenPtr = 0;
  }

  /**
   * Initialize a new WASM clustering session.
   * Maps JS simulator state variables to C config.
   *
   * @param {Object} params — algorithm parameters
   *   from the JS simulator global state
   */
  function init(params) {
    if (!_ready) return false;

    // Destroy any existing session
    if (_handle) {
      _fn.free(_handle);
      _freeBuffers();
      _handle = null;
    }

    _ndim = params.ndim || 3;
    const isUnlimited = !(params.maxcl > 0);
    _maxK = isUnlimited ? 256 : params.maxnbclust;

    // Map JS pruneMode string to te4/te5 booleans
    const te4 = (params.pruneMode === '4P' ||
                 params.pruneMode === '5P') ? 1 : 0;
    const te5 = (params.pruneMode === '5P') ? 1 : 0;

    // Map maxcl strategy string to enum
    let strategyEnum = 0; // MAXCL_STOP
    if (params.maxclStrategy === 'discard') {
      strategyEnum = 1;
    } else if (params.maxclStrategy === 'merge') {
      strategyEnum = 2;
    }

    const effectiveMaxcl = isUnlimited ? 256 : _maxK;

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
      console.error(
        '[GricWasm] wasm_cluster_init returned NULL'
      );
      return false;
    }

    // Enable geometric growth for unlimited mode
    if (isUnlimited) {
      _fn.setUnlimited(_handle, 1);
    }

    if (_traceEnabled || (params && params.isExplainMode) || (typeof isExplainMode !== 'undefined' && isExplainMode)) {
      _traceEnabled = true;
      if (_fn.setTrace) {
        _fn.setTrace(_handle, 1, 2048);
        if (!_eventSize) {
          _eventSize = _fn.getTraceEventSize();
        }
      }
    }

    _allocBuffers();
    return true;
  }

  /**
   * Process a single coordinate frame through
   * the WASM clustering engine.
   *
   * @param {number} x — X coordinate
   * @param {number} y — Y coordinate
   * @param {number} z — Z coordinate (0 for 2D)
   * @returns {number} Assigned cluster index,
   *   or -1 if unassigned
   */
  function processFrame(x, y, z) {
    if (!_handle) return -1;

    const M = _module;
    M.setValue(_coordsPtr, x, 'double');
    M.setValue(_coordsPtr + 8, y, 'double');
    if (_ndim >= 3) {
      M.setValue(_coordsPtr + 16, z || 0.0, 'double');
    }

    return _fn.processFrame(
      _handle, _coordsPtr, _ndim
    );
  }

  /**
   * Pull cluster state from WASM heap into JS objects.
   * Updates the global JS `clusters`, `dcc`, and
   * `transitionCounts` arrays used by the renderer
   * and telemetry.
   *
   * @returns {Object} Snapshot of current WASM state
   */
  function syncState() {
    if (!_handle) return null;

    const M = _module;

    // Detect if C engine grew its arrays
    const cCapacity = _fn.getCapacity(_handle);
    if (cCapacity > _maxK) {
      _freeBuffers();
      _maxK = cCapacity;
      _allocBuffers();
    }

    const K = _fn.getNumClusters(_handle);

    // Read anchor positions and member counts
    _fn.getAnchors(
      _handle, _anchorsPtr, _membersPtr, _ndim
    );

    const anchors = [];
    for (let i = 0; i < K; i++) {
      const coordBase = _anchorsPtr + i * _ndim * 8;
      anchors.push({
        id: i,
        x: M.getValue(coordBase, 'double'),
        y: M.getValue(coordBase + 8, 'double'),
        z: _ndim >= 3
          ? M.getValue(coordBase + 16, 'double')
          : 0.0,
        members: M.getValue(
          _membersPtr + i * 4, 'i32'
        )
      });
    }

    // Read DCC matrix
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

    // Read probabilities
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

    // Read telemetry
    _fn.getTelemetry(_handle, _telemetryPtr, _telemetryLenPtr);
    const tLen = M.getValue(_telemetryLenPtr, 'i32');
    const telemetryArr = [];
    if (heapF64) {
      const telOffset = _telemetryPtr >> 3;
      telemetryArr.push(...heapF64.subarray(telOffset, telOffset + tLen));
    } else {
      for (let i = 0; i < tLen; i++) {
        telemetryArr.push(M.getValue(_telemetryPtr + i * 8, 'double'));
      }
    }

    // Read candidate distance evaluations for active query frame
    const evaluations = [];
    if (_fn.getEvaluations && _evalIndicesPtr && _evalDistsPtr) {
      const numEvals = _fn.getEvaluations(
        _handle, _evalIndicesPtr, _evalDistsPtr, _maxK
      );
      for (let i = 0; i < numEvals; i++) {
        const cId = M.getValue(_evalIndicesPtr + i * 4, 'i32');
        const dist = M.getValue(_evalDistsPtr + i * 8, 'double');
        if (cId >= 0 && cId < K) {
          evaluations.push({
            target: anchors[cId],
            dist: dist,
            match: dist <= (rlim || 0.1),
          });
        }
      }
    }

    return {
      numClusters: K,
      anchors: anchors,
      dcc: dccMatrix,
      probs: probs,
      evaluations: evaluations,
      telemetry: {
        framedistCalls: telemetryArr[0] || 0,
        framedistSample: telemetryArr[1] || 0,
        framedistIntercluster: telemetryArr[2] || 0,
        clustersPruned: telemetryArr[3] || 0,
        totalFrames: telemetryArr[4] || 0,
        lastFrameDists: telemetryArr[5] || 0,
        lastFrameDfc: telemetryArr[6] || 0,
        lastFrameDcc: telemetryArr[7] || 0,
        lastAssignmentDist: telemetryArr[8] || 0,
        numNewClusters: telemetryArr[9] || 0,
        predAttempts: telemetryArr[10] || 0,
        predHits: telemetryArr[11] || 0,
        entropyGated: telemetryArr[12] || 0,
        entropyEvaluated: telemetryArr[13] || 0,
        entropySumInitial: telemetryArr[14] || 0,
        entropyMaxInitial: telemetryArr[15] || 0,
        entropyLastInitial: telemetryArr[16] || 0,
      }
    };
  }

  /**
   * Reset WASM clustering state
   * (keeps config, clears clusters + telemetry).
   */
  function resetState() {
    _lastSyncFrames = 0;
    _lastSyncTime = 0;
    _lastSyncDists = 0;
    if (_handle) {
      _fn.reset(_handle);
    }
  }

  /**
   * Free all WASM resources.
   */
  function destroy() {
    _lastSyncFrames = 0;
    _lastSyncTime = 0;
    _lastSyncDists = 0;
    if (_handle) {
      _fn.free(_handle);
      _freeBuffers();
      _handle = null;
    }
  }

  /**
   * Check if the WASM engine is loaded and ready.
   */
  function isReady() {
    return _ready && _handle !== null;
  }

  /**
   * Check if the WASM module was loaded
   * (even if no session is active).
   */
  function isLoaded() {
    return _ready;
  }

  /**
   * Build init params from current JS global state.
   * Call this from main.js to create the params
   * object from current simulator settings.
   */
  function buildParamsFromState() {
    return {
      rlim: rlim,
      maxnbclust: maxcl > 0 ? maxcl : 256,
      maxnbfr: 100000,
      ndim: currentDim,
      entropyMode: (targetMode === 'entropy'),
      pruneMode: pruneMode,
      predMode: usePred,
      predHorizon: predHorizon,
      gprobMode: useGprob,
      tmMixingCoeff: useTM ? tmMixingCoeff : 0.0,
      softBayesian: useSoftBayesian,
      xtileMode: useXTile,
      sparseDcc: useSparseDcc,
      sparseDccExtraEvals: sparseDccExtraEvals,
      entropyGate: entropyGate,
      entropyFirstGate: entropyFirstGate,
      entropyFast: entropyFastMode,
      softBayesianSigma: softBayesianSigmaCoeff,
      maxcl: maxcl,
      maxclStrategy: maxclStrategy,
      discardFraction: discardFraction,
      maxVisitors: maxVisitors,
      isExplainMode: typeof isExplainMode !== 'undefined' ? isExplainMode : false,
    };
  }

  let _lastSyncFrames = 0;
  let _lastSyncTime = 0;
  let _lastSyncDists = 0;

  /**
   * Apply WASM state snapshot to JS global variables.
   * Maps WASM cluster data back to the JS renderer's
   * expected format.
   */
  function applyToJsState(snapshot) {
    if (!snapshot) return;

    const K = snapshot.numClusters;

    // Grow or shrink the JS clusters array
    while (clusters.length < K) {
      clusters.push({
        id: clusters.length,
        x: 0, y: 0, z: 0,
        members: 0,
        prob: 0,
        scDists: 0,
        lastActive: totalFrames,
        color: getClusterColor(clusters.length),
      });
    }
    clusters.length = K;

    // Update cluster properties
    for (let i = 0; i < K; i++) {
      const a = snapshot.anchors[i];
      const c = clusters[i];
      c.id = i;
      c.x = a.x;
      c.y = a.y;
      c.z = a.z;
      c.members = a.members;
      c.prob = snapshot.probs[i] || 0;
      c.lastActive = totalFrames;
      if (!c.color) {
        c.color = getClusterColor(i);
      }
    }

    // Update DCC matrix
    dcc = snapshot.dcc;

    // Grow transitionCounts to match cluster count
    while (transitionCounts.length < K) {
      const newRow = new Array(K).fill(0);
      transitionCounts.push(newRow);
    }
    // Extend existing rows
    for (let i = 0; i < transitionCounts.length; i++) {
      while (transitionCounts[i].length < K) {
        transitionCounts[i].push(0);
      }
    }

    // Update active frame distance evaluations for visual renderer
    if (snapshot.evaluations && snapshot.evaluations.length > 0) {
      currentEvaluations = snapshot.evaluations.map(ev => ({
        target: clusters[ev.target.id] || ev.target,
        dist: ev.dist,
        match: ev.match,
      }));
    } else {
      currentEvaluations = [];
    }

    // Update telemetry counters
    const t = snapshot.telemetry;
    if (t) {
      if (t.totalFrames > totalFrames) {
        totalFrames = t.totalFrames;
      }
      distSampleCluster = t.framedistSample;
      distClusterCluster = t.framedistIntercluster;
      distSampleClusterLast = t.lastFrameDfc;
      distClusterClusterLast = t.lastFrameDcc;

      // Update pruning breakdown based on active pruneMode
      if (pruneMode === '4P') {
        pruneCount4P = t.clustersPruned;
        pruneCount3P = 0;
        pruneCount5P = 0;
      } else if (pruneMode === '5P') {
        pruneCount5P = t.clustersPruned;
        pruneCount3P = 0;
        pruneCount4P = 0;
      } else {
        pruneCount3P = t.clustersPruned;
        pruneCount4P = 0;
        pruneCount5P = 0;
      }

      // Sequence / TM predictor hits
      predHitCount = t.predHits || 0;

      // Shannon entropy metrics
      totalEntropyGated = t.entropyGated || 0;
      totalEntropyEvals = t.entropyEvaluated || 0;
      if (t.entropySumInitial > 0) {
        totalInitialEntropyBits = t.entropySumInitial;
        totalEntropyReducedBits = t.entropySumInitial;
      }
      if (t.entropyMaxInitial > 0) {
        maxInitialEntropyObserved = t.entropyMaxInitial;
      }
      if (t.entropyLastInitial > 0) {
        lastInitialEntropy = t.entropyLastInitial;
      }

      naiveEvals = totalFrames * K;

      // Live distance curve history tracking
      if (t.lastFrameDfc > 0 || t.lastFrameDcc > 0) {
        if (distHistoryDFC.length === 0 || distHistoryDFC[distHistoryDFC.length - 1] !== t.lastFrameDfc || distHistoryDCC[distHistoryDCC.length - 1] !== t.lastFrameDcc) {
          distHistoryDFC.push(t.lastFrameDfc);
          distHistoryDCC.push(t.lastFrameDcc);
          if (distHistoryDFC.length > 200) distHistoryDFC.shift();
          if (distHistoryDCC.length > 200) distHistoryDCC.shift();
        }
      }

      // Rolling performance telemetry for CPU / Throughput / Sparkline
      const now = performance.now();
      if (_lastSyncTime === 0) {
        _lastSyncTime = now;
        _lastSyncFrames = t.totalFrames;
        _lastSyncDists = t.framedistSample + t.framedistIntercluster;
      } else {
        const frameDelta = t.totalFrames - _lastSyncFrames;
        const distDelta = (t.framedistSample + t.framedistIntercluster) - _lastSyncDists;
        const timeDelta = Math.max(0.001, now - _lastSyncTime);

        if (frameDelta > 0 && isRunning) {
          const avgPerFrameMs = timeDelta / frameDelta;
          lastComputeTimeMs = avgPerFrameMs;
          totalComputeTimeMs += timeDelta;
          avgComputeTimeMs = totalFrames > 0 ? (totalComputeTimeMs / totalFrames) : avgPerFrameMs;

          sparklineHistory.push(avgPerFrameMs);
          if (sparklineHistory.length > 60) sparklineHistory.shift();

          rollingHistory.push({
            time: now,
            computeMs: timeDelta,
            frames: frameDelta,
            dists: Math.max(0, distDelta)
          });
          if (rollingHistory.length > 1000) {
            rollingHistory = rollingHistory.slice(-500);
          }
        }
        _lastSyncFrames = t.totalFrames;
        _lastSyncTime = now;
        _lastSyncDists = t.framedistSample + t.framedistIntercluster;
      }
    }
  }

  /* -------------------------------------------------------
   * Trace event type enum values (must match cluster_trace.h)
   * ------------------------------------------------------- */
  const _TRACE_TYPE = {
    FRAME_INGEST: 0,
    INITIAL_CLUSTER: 1,
    TARGET_SELECTED: 2,
    MATCH: 3,
    MISMATCH: 4,
    PRUNE_3P: 5,
    PRUNE_4P: 6,
    PRUNE_5P: 7,
    NEW_CLUSTER: 8,
    EVICT_STOP: 9,
    EVICT_DISCARD: 10,
    EVICT_MERGE: 11,
    ENTROPY_GATE: 12,
    PRIOR_MIXING: 13,
  };

  const _TRACE_REASON = {
    GREEDY_STATIC: 0,
    GREEDY_DYNAMIC: 1,
    PREDICTION: 2,
    ENTROPY_FULL: 3,
    ENTROPY_FAST: 4,
    ENTROPY_GATED: 5,
    LEADER_SHORTCUT: 6,
  };

  let _traceEnabled = false;
  let _eventSize = 0;

  /**
   * Enable or disable the C-side explain trace buffer.
   * @param {boolean} enabled
   */
  function setTrace(enabled) {
    if (!_handle) return;
    _traceEnabled = enabled;
    _fn.setTrace(_handle, enabled ? 1 : 0, 2048);
    if (!_eventSize && enabled) {
      _eventSize = _fn.getTraceEventSize();
    }
  }

  /**
   * Map a trace event type enum to a JS step type string.
   */
  function _traceTypeToStep(t) {
    switch (t) {
      case _TRACE_TYPE.FRAME_INGEST:
      case _TRACE_TYPE.TARGET_SELECTED:
      case _TRACE_TYPE.ENTROPY_GATE:
      case _TRACE_TYPE.PRIOR_MIXING:
        return 'target';
      case _TRACE_TYPE.INITIAL_CLUSTER:
      case _TRACE_TYPE.NEW_CLUSTER:
        return 'new-cluster';
      case _TRACE_TYPE.MATCH:
        return 'match';
      case _TRACE_TYPE.MISMATCH:
      case _TRACE_TYPE.EVICT_STOP:
      case _TRACE_TYPE.EVICT_DISCARD:
      case _TRACE_TYPE.EVICT_MERGE:
        return 'mismatch';
      case _TRACE_TYPE.PRUNE_3P:
      case _TRACE_TYPE.PRUNE_4P:
      case _TRACE_TYPE.PRUNE_5P:
        return 'prune';
      default:
        return 'target';
    }
  }

  /**
   * Build a human-readable title for a trace event.
   */
  function _traceTitle(ev) {
    switch (ev._type) {
      case _TRACE_TYPE.FRAME_INGEST:
        return `📍 Ingesting Frame #${ev.frame_id}`;
      case _TRACE_TYPE.INITIAL_CLUSTER:
        return '✨ Initial Cluster Anchor Created';
      case _TRACE_TYPE.TARGET_SELECTED: {
        const reasons = [
          'Greedy', 'Greedy (dynamic)',
          'Prediction', 'Shannon entropy',
          'Popcount surrogate', 'Entropy-gated',
          'Leader shortcut'
        ];
        const r = reasons[ev.reason] || 'Unknown';
        return `🔍 Measuring: C${ev.cluster_id} (${r})`;
      }
      case _TRACE_TYPE.MATCH:
        return `🎯 Match on C${ev.cluster_id}`;
      case _TRACE_TYPE.MISMATCH:
        return `❌ Mismatch on C${ev.cluster_id}`;
      case _TRACE_TYPE.PRUNE_3P:
        return '📐 3-Point Triangle Inequality Pruning';
      case _TRACE_TYPE.PRUNE_4P:
        return '📐 4-Point Pruning (-te4)';
      case _TRACE_TYPE.PRUNE_5P:
        return '📐 5-Point Pruning (-te5)';
      case _TRACE_TYPE.NEW_CLUSTER:
        return `✨ New Cluster Created: C${ev.cluster_id}`;
      case _TRACE_TYPE.EVICT_STOP:
        return '🛑 Max Clusters Limit Reached (-maxcl stop)';
      case _TRACE_TYPE.EVICT_DISCARD:
        return `♻️ Cluster Eviction: C${ev.cluster_id}`;
      case _TRACE_TYPE.EVICT_MERGE:
        return `🤝 Cluster Merge (-maxcl merge)`;
      case _TRACE_TYPE.ENTROPY_GATE:
        return '⚡ Entropy Gated → Greedy Shortcut';
      case _TRACE_TYPE.PRIOR_MIXING:
        return '📊 Prior Probability Mixing';
      default:
        return 'Unknown Event';
    }
  }

  /**
   * Build a human-readable text body for a trace event.
   */
  function _traceText(ev) {
    switch (ev._type) {
      case _TRACE_TYPE.INITIAL_CLUSTER:
        return `Frame #${ev.frame_id}: initialized C0.`;
      case _TRACE_TYPE.TARGET_SELECTED:
        if (ev.entropy_h > 0) {
          return `Target C${ev.cluster_id}, ` +
            `current H = ${ev.entropy_h.toFixed(3)} bits.`;
        }
        return `Target C${ev.cluster_id} selected.`;
      case _TRACE_TYPE.MATCH:
        return `d = ${ev.distance.toFixed(4)}, ` +
          `rlim = ${ev.rlim.toFixed(4)}. ` +
          `Distance ≤ threshold → assigned.`;
      case _TRACE_TYPE.MISMATCH:
        return `d = ${ev.distance.toFixed(4)} > ` +
          `rlim = ${ev.rlim.toFixed(4)}. Excluded.`;
      case _TRACE_TYPE.PRUNE_3P:
      case _TRACE_TYPE.PRUNE_4P:
      case _TRACE_TYPE.PRUNE_5P:
        return `Pruned ${ev.pruned_count} candidate(s), ` +
          `${ev.active_remaining} remaining.`;
      case _TRACE_TYPE.NEW_CLUSTER:
        return `No match in ${ev.active_remaining || 0} ` +
          `candidates. Spawned new anchor C${ev.cluster_id}.`;
      case _TRACE_TYPE.EVICT_STOP:
        return 'Cluster budget reached. Frame unassigned.';
      case _TRACE_TYPE.EVICT_DISCARD:
        return `Evicted C${ev.cluster_id} (lowest frequency).`;
      case _TRACE_TYPE.EVICT_MERGE:
        return `Merged closest pair (d = ` +
          `${ev.distance.toFixed(4)}).`;
      case _TRACE_TYPE.ENTROPY_GATE:
        return `H = ${ev.entropy_h.toFixed(3)} bits < ` +
          `gate ${ev.lower_bound.toFixed(3)} → greedy.`;
      case _TRACE_TYPE.PRIOR_MIXING:
        return 'Prior probabilities blended with ' +
          'transition history.';
      default:
        return '';
    }
  }

  /**
   * Read trace events from WASM heap for the current
   * frame and convert to JS explain step objects.
   *
   * @returns {Array} Array of step objects matching
   *   the existing JS explain format:
   *   { type, title, text, entropyRankings, currentH }
   */
  function getTrace() {
    if (!_handle || !_traceEnabled) return [];

    const count = _fn.getTraceCount(_handle);
    if (count === 0) return [];

    const eventsPtr = _fn.getTraceEvents(_handle);
    if (!eventsPtr) return [];

    if (!_eventSize) {
      _eventSize = _fn.getTraceEventSize();
    }

    const head = _fn.getTraceHead(_handle);
    const frameStart = _fn.getTraceFrameStart(_handle);
    const capacity = 2048;

    const M = _module;
    const HEAP32 = M.HEAP32;
    const HEAPF64 = M.HEAPF64;
    const steps = [];

    /* Walk from frame_start to head (wrapping) to get
     * only the events for the current frame. */
    let idx = frameStart;
    const endIdx = head;
    const maxIter = capacity + 1; // safety bound
    let iter = 0;

    while (idx !== endIdx && iter < maxIter) {
      iter++;
      const byteOff = eventsPtr + idx * _eventSize;

      /* TraceEvent struct layout (must match C):
       *  0: uint16 type
       *  2: uint16 reason
       *  4: int    frame_id
       *  8: int    cluster_id
       * 12: (pad to 16)
       * 16: double distance
       * 24: double rlim
       * 32: double lower_bound
       * 40: double entropy_h
       * 48: int    active_remaining
       * 52: int    pruned_count
       * 56: int    num_candidates
       * 60: (pad to 64)
       * 64: TraceCandidateEntry[8]
       *   each: int id (4) + pad(4) + 3×double (24) = 32
       *   total: 8 × 32 = 256
       * Total: 64 + 256 = 320 bytes
       *
       * We use HEAP32 (4-byte view) for ints and
       * HEAPF64 (8-byte view) for doubles.
       */
      const i32 = byteOff >> 2;
      const typeAndReason = HEAP32[i32];
      const evType = typeAndReason & 0xFFFF;
      const reason = (typeAndReason >> 16) & 0xFFFF;
      const frameId = HEAP32[i32 + 1];
      const clusterId = HEAP32[i32 + 2];

      const f64 = byteOff >> 3;
      const dist = HEAPF64[f64 + 2];
      const rlim = HEAPF64[f64 + 3];
      const lowerBound = HEAPF64[f64 + 4];
      const entropyH = HEAPF64[f64 + 5];

      const activeRemaining = HEAP32[(byteOff + 48) >> 2];
      const prunedCount = HEAP32[(byteOff + 52) >> 2];
      const numCandidates = HEAP32[(byteOff + 56) >> 2];

      const ev = {
        _type: evType,
        reason: reason,
        frame_id: frameId,
        cluster_id: clusterId,
        distance: dist,
        rlim: rlim,
        lower_bound: lowerBound,
        entropy_h: entropyH,
        active_remaining: activeRemaining,
        pruned_count: prunedCount,
      };

      const step = {
        type: _traceTypeToStep(evType),
        title: _traceTitle(ev),
        text: _traceText(ev),
      };

      /* Populate entropy rankings if present */
      if (numCandidates > 0) {
        const rankings = [];
        const candBase = byteOff + 64;
        const candStride = 32; // 4 + 4pad + 3×8 = 32
        const n = Math.min(numCandidates, 8);
        for (let ci = 0; ci < n; ci++) {
          const cOff = candBase + ci * candStride;
          const cId = HEAP32[cOff >> 2];
          const cF64 = (cOff + 8) >> 3;
          rankings.push({
            id: cId,
            p: HEAPF64[cF64],
            expectedH: HEAPF64[cF64 + 1],
            infoGain: HEAPF64[cF64 + 2],
          });
        }
        step.entropyRankings = rankings;
        step.currentH = entropyH;
      }

      steps.push(step);
      idx = (idx + 1) % capacity;
    } // while trace events

    return steps;
  }

  /**
   * Get the git hash the WASM binary was built from.
   * Returns 'unknown' if not available.
   */
  function getVersion() {
    if (!_ready || !_module) return 'unknown';
    try {
      const fn = _module.cwrap(
        'wasm_cluster_get_version', 'string', []);
      return fn();
    } catch (e) {
      return 'unknown';
    }
  }

  // Public API
  return {
    load: load,
    init: init,
    processFrame: processFrame,
    syncState: syncState,
    resetState: resetState,
    destroy: destroy,
    isReady: isReady,
    isLoaded: isLoaded,
    buildParamsFromState: buildParamsFromState,
    applyToJsState: applyToJsState,
    setTrace: setTrace,
    getTrace: getTrace,
    getVersion: getVersion,
  };
})();

/**
 * Build the equivalent gric-cluster CLI command from
 * current JS simulator state variables.
 * Returns a string like:
 *   gric-cluster 0.100 -entropy -te4 input.fits
 */
function buildCliCommand() {
  const parts = ['gric-cluster'];

  // Positional: rlim
  parts.push(rlim.toFixed(3));

  // Pruning mode
  if (pruneMode === '4P' || pruneMode === '5P') {
    parts.push('-te4');
  }
  if (pruneMode === '5P') {
    parts.push('-te5');
  }

  // Target selection
  if (targetMode === 'entropy') {
    parts.push('-entropy');
    if (entropyGate !== 2.0) {
      parts.push('-entropy_gate', entropyGate.toFixed(2));
    }
    if (entropyFirstGate !== 4.0) {
      parts.push(
        '-entropy_first_gate', entropyFirstGate.toFixed(2)
      );
    }
    if (entropyFastMode) {
      parts.push('-entropy_fast');
    }
  }

  // Transition matrix
  if (useTM && tmMixingCoeff > 0) {
    parts.push('-tm', tmMixingCoeff.toFixed(2));
  }

  // Prediction
  if (usePred) {
    if (predHorizon !== 2) {
      parts.push('-pred[,,' + predHorizon + ']');
    } else {
      parts.push('-pred');
    }
  }

  // Geometric probability
  if (useGprob) {
    parts.push('-gprob');
    if (maxVisitors !== 1000) {
      parts.push('-maxvis', maxVisitors.toString());
    }
  }

  // Soft Bayesian
  if (useSoftBayesian) {
    parts.push('-soft_bayesian');
    if (softBayesianSigmaCoeff !== 1.0) {
      parts.push(
        '-soft_bayesian_sigma',
        softBayesianSigmaCoeff.toFixed(2)
      );
    }
  }

  // Cross-tile
  if (useXTile) {
    parts.push('-xtile');
  }

  // Sparse DCC
  if (useSparseDcc) {
    parts.push('-sparse_dcc');
    if (sparseDccExtraEvals > 0) {
      parts.push(
        '-sparse_dcc_extra_evals',
        sparseDccExtraEvals.toString()
      );
    }
  }

  // Max clusters & eviction
  if (maxcl > 0) {
    parts.push('-maxcl', maxcl.toString());
    if (maxclStrategy !== 'stop') {
      parts.push('-maxcl_strategy', maxclStrategy);
    }
    if (maxclStrategy === 'discard' &&
        discardFraction !== 0.5)
    {
      parts.push(
        '-discard_frac', discardFraction.toFixed(2)
      );
    }
  }

  // Input placeholder
  parts.push('<input.fits>');

  return parts.join(' ');
}

/**
 * GricWasmWorker - Main thread controller for background Web Worker clustering.
 */
// eslint-disable-next-line no-unused-vars
const GricWasmWorker = (function () {
  'use strict';

  let _worker = null;
  let _ready = false;
  let _busy = false;
  let _onProgressCb = null;
  let _onDoneCb = null;

  function isAvailable() {
    return typeof window !== 'undefined' && typeof window.Worker !== 'undefined';
  }

  function initWorker() {
    if (_worker || !isAvailable()) return;

    try {
      _worker = new Worker('simulator/js/wasm_worker.js');
      _worker.onmessage = _handleMessage;
      _worker.onerror = function (err) {
        console.error('[GricWasmWorker] Worker error:', err);
      };
      _worker.postMessage({ type: 'LOAD' });
    } catch (err) {
      console.warn('[GricWasmWorker] Could not instantiate worker:', err);
      _worker = null;
    }
  }

  function _handleMessage(e) {
    const msg = e.data;
    if (!msg || !msg.type) return;

    switch (msg.type) {
      case 'READY':
        _ready = true;
        console.log('[GricWasmWorker] Background Web Worker ready.');
        break;

      case 'INIT_OK':
        break;

      case 'BATCH_PROGRESS':
        if (msg.snapshot) {
          GricWasm.applyToJsState(msg.snapshot);
        }
        if (typeof _onProgressCb === 'function') {
          _onProgressCb(msg.progress, msg.processedFrames, msg.totalFrames);
        }
        break;

      case 'BATCH_DONE':
        _busy = false;
        if (msg.snapshot) {
          GricWasm.applyToJsState(msg.snapshot);
        }
        if (typeof _onDoneCb === 'function') {
          _onDoneCb(msg.totalFrames);
        }
        break;

      case 'BATCH_PAUSED':
        _busy = false;
        if (msg.snapshot) {
          GricWasm.applyToJsState(msg.snapshot);
        }
        break;

      case 'SNAPSHOT':
        if (msg.snapshot) {
          GricWasm.applyToJsState(msg.snapshot);
        }
        break;

      case 'ERROR':
        console.error('[GricWasmWorker] Worker error:', msg.message);
        break;
    }
  }

  function startSession(params) {
    if (!_worker) initWorker();
    if (_worker) {
      _worker.postMessage({ type: 'INIT', params: params || GricWasm.buildParamsFromState() });
    }
  }

  function startBatch(dataset, startIndex = 0, onProgress = null, onDone = null) {
    if (!_worker || !_ready) return false;

    _onProgressCb = onProgress;
    _onDoneCb = onDone;
    _busy = true;

    const ndim = currentDim || 3;
    const totalFrames = dataset.length - startIndex;
    const flat = new Float64Array(totalFrames * ndim);

    for (let i = 0; i < totalFrames; i++) {
      const pt = dataset[startIndex + i];
      flat[i * ndim] = pt.x;
      flat[i * ndim + 1] = pt.y;
      if (ndim >= 3) flat[i * ndim + 2] = pt.z || 0.0;
    }

    _worker.postMessage({
      type: 'START_BATCH',
      data: flat,
      totalFrames: totalFrames,
      startIndex: 0,
      chunkSize: 500
    }, [flat.buffer]);

    return true;
  }

  function pauseBatch() {
    if (_worker && _busy) {
      _worker.postMessage({ type: 'PAUSE_BATCH' });
      _busy = false;
    }
  }

  function resumeBatch() {
    if (_worker) {
      _busy = true;
      _worker.postMessage({ type: 'RESUME_BATCH' });
    }
  }

  function reset() {
    _busy = false;
    if (_worker) {
      _worker.postMessage({ type: 'RESET' });
    }
  }

  function isReady() {
    return _ready;
  }

  function isBusy() {
    return _busy;
  }

  return {
    initWorker: initWorker,
    startSession: startSession,
    startBatch: startBatch,
    pauseBatch: pauseBatch,
    resumeBatch: resumeBatch,
    reset: reset,
    isAvailable: isAvailable,
    isReady: isReady,
    isBusy: isBusy
  };
})();

