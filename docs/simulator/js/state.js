/**
 * GRIC Simulator - state.js
 * Part of the GRIC Interactive Algorithm Simulator
 */

//  4. STATE & ENGINE INITIALIZATION
    // =========================================================================

    const canvas = document.getElementById('simCanvas');
    const ctx = canvas.getContext('2d');

    // Dimensionality & Quadrant Layout State
    let currentDim = 3; // 2 or 3
    let maximizedQuad = null; // null (all 4 quads) or 0, 1, 2, 3

    // 3D Orbit Camera State (Spherical Orbit: Azimuth θ, Elevation φ)
    const orbitCamera = {
      azimuth: -35 * (Math.PI / 180),  // θ
      elevation: 25 * (Math.PI / 180), // φ
      zoom: 1.0,
      panX: 0,
      panY: 0
    };

    // 2D Pan & Zoom per Quadrant [0: Along X, 1: Along Y, 2: Along Z, 3: Custom]
    const quadViews = [
      { panX: 0, panY: 0, zoom: 1.0 },
      { panX: 0, panY: 0, zoom: 1.0 },
      { panX: 0, panY: 0, zoom: 1.0 },
      { panX: 0, panY: 0, zoom: 1.0 }
    ];

    // Mouse Interaction State
    let isDragging = false;
    let dragMode = null; // 'orbit' or 'pan'
    let activeDragQuad = 0;
    let dragStartX = 0, dragStartY = 0;
    let isAddPointMode = false;
    let isExplainMode = false;
    let currentActiveTab = 'narrative';

    // Algorithm & Display Config
    let rlim = 0.100;
    let visualFocus = 20; // 0 (Points Only) to 100 (Clusters Only)
    let samplePointSize = 1.5; // Ingested sample points render radius in px (0.5 to 8.0)
    let showPastSamples = true; // Toggle past sample point cloud visibility
    let maxDrawPoints = 10000;  // Max points drawn per frame (subsampled)
    let sampleBufferCap = 100000; // Rolling buffer capacity for pastSamples
    let batchThinRate = 1;      // In batch mode, keep every Nth frame (1 = all / none)
    let showCircleMembers = false; // Area proportional to points in cluster
    let showCircleSCDists = false; // Area proportional to #SC distances
    let showEntropyMap = false; // Spatial Information Gain / Entropy Reduction Map
    let noiseSigma = 0.020; // Truncated Gaussian noise std dev (0 = off)
    let noiseTruncLimit = 0.100; // Truncation radius limit
    let targetMode = 'greedy';
    let pruneMode = '4P';

    // Prior Acceleration Options
    let useTM = true;
    let tmMixingCoeff = 0.50; // -tm <val> (0.00 to 1.00)
    let usePred = true;
    let predHorizon = 2; // -pred [L,H,N] (1 to 5)
    let useGprob = true;
    let maxVisitors = 20; // -maxvis <int> (5 to 50)
    let fmatchA = 1.0, fmatchB = 2.0;

    // Subspace Partitioning
    let useTiles = false; // -tiles
    let useXTile = false; // -xtile
    let xtileDecay = 0.70; // -xtile_decay (0.1 to 1.0)
    let useSparseDcc = false; // -sparse_dcc
    let sparseDccExtraEvals = 0; // -sparse_dcc_extra_evals

    // Cluster Capacity & Eviction Policy
    let maxcl = 0; // -maxcl <int> (0 = Unlimited)
    let maxclStrategy = 'stop'; // -maxcl_strategy (stop, discard, merge)
    let discardFraction = 0.10; // -discard_frac

    // Entropy Advanced Gating & Leader Shortcut
    let entropyFirstGate = 1.50; // -entropy_first_gate (bits)
    let entropyGate = 0.75; // -entropy_gate (bits)
    let entropyFastMode = false; // -entropy_fast
    let entropyLeaderShortcut = false; // Leader P > cutoff bypass
    let entropyLeaderCutoff = 0.50; // Threshold for leader bypass

    // Entropy Telemetry & Diagnostics
    let totalInitialEntropyBits = 0.0;
    let totalEntropyReducedBits = 0.0;
    let totalEntropyEvals = 0;
    let totalEntropyGated = 0;
    let lastInitialEntropy = 0.0;
    let lastEntropyReduced = 0.0;
    let lastInfoGainRate = 0.0;
    let maxInitialEntropyObserved = 0.0;
    let lastEntropyRankings = [];

    // Soft Bayesian Mode
    let useSoftBayesian = false; // -soft_bayesian
    let softBayesianSigmaCoeff = 1.0; // -soft_bayesian_sigma (multiplier of rlim)

    // WASM Engine Mode
    let useWasm = true; // Use C/WASM backend when available
    let wasmSessionActive = false; // True if GricWasm handle is live

    // Interactive Selection & Hover State
    let selectedClusterId = -1;
    let selectedTupleKey = null;
    let hoveredClusterId = -1;

    // Cluster Table Column Visibility & Sorting State
    const clusterTableCols = {
      cluster: { label: 'Cluster', visible: true },
      centroid: { label: 'Centroid', visible: true },
      frames: { label: 'Frames', visible: true },
      scDists: { label: '#SC dists', visible: true },
      dist: { label: 'Dist d(f,c)', visible: true },
      status: { label: 'Status', visible: true }
    };

    let clusterSortKey = 'cluster'; // 'cluster', 'centroid', 'frames', 'scDists', 'dist', 'status'
    let clusterSortDir = 'asc'; // 'asc' or 'desc'

    function toggleClusterColumn(colKey) {
      if (clusterTableCols[colKey]) {
        const visibleCount = Object.values(clusterTableCols).filter(c => c.visible).length;
        if (clusterTableCols[colKey].visible && visibleCount <= 1) return;
        clusterTableCols[colKey].visible = !clusterTableCols[colKey].visible;
        updateUI();
      }
    }

    function setClusterSort(key) {
      if (key === 'id') key = 'cluster';
      if (key === 'members') key = 'frames';

      if (clusterSortKey === key) {
        clusterSortDir = (clusterSortDir === 'asc' ? 'desc' : 'asc');
      } else {
        clusterSortKey = key;
        clusterSortDir = (key === 'frames' || key === 'scDists') ? 'desc' : 'asc';
      }
      updateUI();
    }

    function resetClusterTableColumns() {
      Object.keys(clusterTableCols).forEach(k => clusterTableCols[k].visible = true);
      clusterSortKey = 'cluster';
      clusterSortDir = 'asc';
      updateUI();
    }

    function setHoveredCluster(clusterId) {
      if (hoveredClusterId !== clusterId) {
        hoveredClusterId = clusterId;
        draw();
      }
    }

    function toggleSelectCluster(clusterId) {
      if (selectedClusterId === clusterId) {
        selectedClusterId = -1;
      } else {
        selectedClusterId = clusterId;
      }
      updateUI();
      draw();
    }

    function toggleSelectTuple(tupleKey) {
      if (selectedTupleKey === tupleKey) {
        selectedTupleKey = null;
      } else {
        selectedTupleKey = tupleKey;
      }
      updateUI();
      draw();
    }

    // Auto-rlim Computation (-scandist parity)
    function computeAutoRlim() {
      let pts = benchmarkDataset;
      if (!pts || pts.length === 0) {
        pts = generateBenchmark(currentBenchmark, 250);
      }
      const N = Math.min(pts.length, 300);
      const dists = [];
      for (let i = 0; i < N; i += 2) {
        for (let j = i + 1; j < N; j += 2) {
          const dx = pts[i].x - pts[j].x;
          const dy = pts[i].y - pts[j].y;
          const dz = currentDim === 3 ? (pts[i].z - pts[j].z) : 0;
          dists.push(Math.sqrt(dx*dx + dy*dy + dz*dz));
          distClusterCluster++;
          distClusterClusterLast++;
        }
      }
      if (dists.length === 0) return;
      dists.sort((a, b) => a - b);
      const median = dists[Math.floor(dists.length * 0.5)];
      const autoR = Math.max(0.02, Math.min(0.28, median * 0.35));
      rlim = parseFloat(autoR.toFixed(3));
      const slR = document.getElementById('sliderRlim');
      if (slR) slR.value = rlim;
      const inpR = document.getElementById('inputRlim');
      if (inpR) inpR.value = rlim.toFixed(3);
      showToast(`⚡ Auto-rlim (-scandist): Median=${median.toFixed(3)} ➔ rlim=${rlim.toFixed(3)}`);
      draw();
    }

    // Truncated Gaussian Random Noise Generator (2D / 3D)
    function generateTruncatedGaussianNoise(dim = 3) {
      if (noiseSigma <= 1e-6) return { dx: 0, dy: 0, dz: 0 };

      const maxRadius = noiseTruncLimit;
      const sigma = noiseSigma;

      for (let attempt = 0; attempt < 50; attempt++) {
        // Box-Muller transform
        const u1 = Math.max(1e-12, Math.random());
        const u2 = Math.random();
        const mag1 = sigma * Math.sqrt(-2.0 * Math.log(u1));
        const z0 = mag1 * Math.cos(2.0 * Math.PI * u2);
        const z1 = mag1 * Math.sin(2.0 * Math.PI * u2);

        let z2 = 0;
        if (dim === 3) {
          const u3 = Math.max(1e-12, Math.random());
          const u4 = Math.random();
          const mag2 = sigma * Math.sqrt(-2.0 * Math.log(u3));
          z2 = mag2 * Math.cos(2.0 * Math.PI * u4);
        }

        const r = Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2);
        if (r <= maxRadius) {
          return { dx: z0, dy: z1, dz: z2 };
        }
      }

      // Rejection limit reached: project to boundary
      const u1 = Math.max(1e-12, Math.random());
      const u2 = Math.random();
      const z0 = sigma * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      const z1 = sigma * Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
      let z2 = 0;
      if (dim === 3) {
        const u3 = Math.max(1e-12, Math.random());
        const u4 = Math.random();
        z2 = sigma * Math.sqrt(-2.0 * Math.log(u3)) * Math.cos(2.0 * Math.PI * u4);
      }
      const r = Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2) || 1.0;
      const scale = maxRadius / r;
      return { dx: z0 * scale, dy: z1 * scale, dz: z2 * scale };
    }

    function applyNoiseToPoint(x, y, z = 0.0) {
      if (noiseSigma <= 1e-6) return { x, y, z };
      const noise = generateTruncatedGaussianNoise(currentDim);
      return {
        x: x + noise.dx,
        y: y + noise.dy,
        z: currentDim === 3 ? (z + noise.dz) : 0.0
      };
    }

    // Simulation & Benchmark State
    let currentBenchmark = "3Dtorus";
    let benchmarkDataset = [];
    let currentFrameIdx = 0;
    let isRunning = false;
    let playTimer = null;
    let playSpeed = -1;
    let loopCount = 10; // 1 = 1 pass, 0 = Infinite, N = N passes
    let currentLoop = 1;
    let sampleCount = 10000; // Number of points generated per pattern

    // Past sample history for point cloud building (x, y, z)
    let pastSamples = [];

    // Monolithic Clustering Engine State (2D / 3D)
    let clusters = [];
    let dcc = [];
    let transitionCounts = [];
    let prevAssignedCluster = -1;
    let lastTransitionFrom = -1;
    let lastTransitionTo = -1;
    let hoveredTMCell = null;
    let topLearnedPathsCache = [];
    let cachedTMRowTotals = null;
    let lastTMRenderTimestamp = 0;
    let lastTMTopPathsTimestamp = 0;
    const tmCanvasDimensions = {
      tmHeatmapCanvas: { w: 420, h: 320 }
    };
    let assignmentHistory = [];
    let frameHistory = [];
    let totalFrames = 0;
    let totalEvals = 0;
    let naiveEvals = 0;
    let currentFrame = null; // { x, y, z }
    let currentEvaluations = [];
    let currentPruned = [];
    let currentPredicted = [];
    let currentEntropyBits = 0;
    let currentExplanation = [];

    // Recent Samples Decision & Search Trace History Buffer
    const MAX_SAMPLE_TRACE_HISTORY = 200;
    let sampleTraceLog = []; // [{ frameIndex, point, assignedCluster, isNewCluster, distSC, distCC, initialEntropy, entropyReduced, steps, entropyRankings }]
    let selectedSampleTraceIndex = -1; // -1: Live (Latest frame), >= 0: Viewing specific frame index
    let hoveredSampleTracePoint = null; // { x, y, z, frameIndex, clusterId } for canvas highlight

    // Fast Pre-computed Unit Circle Trigonometry Lookup Table (for 3D Wireframe Rings)
    const CIRCLE_LUT_STEPS = 16;
    const CIRCLE_COS = new Float32Array(CIRCLE_LUT_STEPS + 1);
    const CIRCLE_SIN = new Float32Array(CIRCLE_LUT_STEPS + 1);
    for (let s = 0; s <= CIRCLE_LUT_STEPS; s++) {
      const ang = (s / CIRCLE_LUT_STEPS) * Math.PI * 2;
      CIRCLE_COS[s] = Math.cos(ang);
      CIRCLE_SIN[s] = Math.sin(ang);
    }

    // Pre-allocated Reusable Scratch Buffers for hot compute loop
    let scratchCap = 256;
    let scratchClMembFlag = new Uint8Array(scratchCap);
    let scratchPBase = new Float64Array(scratchCap);
    let scratchPCurrent = new Float64Array(scratchCap);
    let scratchCurrentGprobs = new Float64Array(scratchCap);

    function ensureScratchCapacity(k) {
      if (k > scratchCap) {
        scratchCap = Math.max(k * 2, 512);
        scratchClMembFlag = new Uint8Array(scratchCap);
        scratchPBase = new Float64Array(scratchCap);
        scratchPCurrent = new Float64Array(scratchCap);
        scratchCurrentGprobs = new Float64Array(scratchCap);
      }
    }

    function clearScratchBuffers() {
      scratchClMembFlag.fill(0);
      scratchPBase.fill(0);
      scratchPCurrent.fill(0);
      scratchCurrentGprobs.fill(0);
    }
