/**
 * GRIC Simulator - state.js
 * Part of the GRIC Interactive Algorithm Simulator
 */

//  4. STATE & ENGINE INITIALIZATION
    // =========================================================================

    const canvas = document.getElementById('simCanvas');
    const ctx = canvas.getContext('2d');

    // Dimensionality & Quadrant Layout State
    let dataMode = 'coord'; // 'coord' (2D/3D points) or 'image' (raster image)
    let currentDim = 3; // 2 or 3 (or 1024 for 32x32 image mode)
    let maximizedQuad = null; // null (all 4 quads) or 0, 1, 2, 3
    let plotDimX = 0; // Visual X-axis plotting dimension (0 .. currentDim-1)
    let plotDimY = 1; // Visual Y-axis plotting dimension (0 .. currentDim-1)
    let plotDimZ = 2; // Visual Z-axis plotting dimension (0 .. currentDim-1)

    // Image Mode State & Retro-Inspection
    let imageWidth = 32;
    let imageHeight = 32;
    let imageDim = 1024; // = imageWidth * imageHeight
    let currentImageFrame = null; // Float32Array or Float64Array for active frame
    let imageGalleryScrollY = 0; // Legacy gallery scroll offset
    let imageMembersScrollY = 0; // Q2 Members gallery scroll offset
    let imageKnnScrollY = 0; // Q2 k-NN gallery scroll offset
    let imageClustersScrollY = 0; // Q3 All clusters gallery scroll offset
    let imageTopRightMode = 'anchor'; // 'anchor', 'residual', 'nn1', 'nn1_diff'
    let imageQ2ViewMode = 'members'; // 'members' (cluster) or 'knn' (neighbors)
    let inspectedImageFrameIdx = -1; // -1 = live, >= 0 = retro-inspected frame
    let inspectedClusterId = -1; // -1 = all clusters, >= 0 = inspecting cluster member frames
    let imageFrameAssignments = []; // imageFrameAssignments[frameIdx] = clusterId
    let imageFrameDists = []; // imageFrameDists[frameIdx] = distance to anchor
    let imageClusterMembers = {}; // imageClusterMembers[clusterId] = [frameIdx, ...]
    let imageClustersSortMode = 'id'; // 'id', 'size_desc', 'size_asc'
    let imageThumbSize = 64; // Thumbnail gallery size in pixels (36 to 200)

    // 3D Orbit Camera State (Spherical Orbit: Azimuth θ, Elevation φ)
    const orbitCamera = {
      azimuth: -35 * (Math.PI / 180),  // θ
      elevation: 25 * (Math.PI / 180), // φ
      zoom: 1.0,
      panX: 0,
      panY: 0,
      isLocked: false,                 // Center lock toggle
      targetX: 0,                      // Locked center target X
      targetY: 0,                      // Locked center target Y
      targetZ: 0,                      // Locked center target Z
      targetIndex: -1,                 // Sample index or cluster id
      targetLabel: ''                  // Display label (e.g. "Sample #42")
    };

    // 2D Pan & Zoom per Quadrant [0: Along X, 1: Along Y, 2: Along Z, 3: Custom]
    const quadViews = [
      { panX: 0, panY: 0, zoom: 1.0 },
      { panX: 0, panY: 0, zoom: 1.0 },
      { panX: 0, panY: 0, zoom: 1.0 },
      { panX: 0, panY: 0, zoom: 1.0 }
    ];
    let viewportZoomBoxRects = [null, null, null, null];

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
    let visibleIndicesBuffer = new Int32Array(500000); // Reusable index buffer for FOV sampling
    let viewportPointStats = [
      { drawn: 0, visible: 0, truncated: false },
      { drawn: 0, visible: 0, truncated: false },
      { drawn: 0, visible: 0, truncated: false },
      { drawn: 0, visible: 0, truncated: false }
    ];
    let clusterSpawnRipples = []; // Active expanding ripple animations for new clusters
    let clusterMilestoneFrames = []; // Frame indices where new cluster anchors were established
    let activeSidebarMode = 'all'; // 'all', 'clustering', 'knn', 'files', 'telemetry'
    let sampleBufferCap = 100000; // Rolling buffer capacity for pastSamples
    let batchThinRate = 1;      // In batch mode, keep every Nth frame (1 = all / none)
    let showDistLines = true;   // Toggle distance evaluation and solving lines
    let showDistLabels = true;  // Toggle distance measurement pills & badges
    let showClusterLabels = true; // Toggle C0, C1... cluster text labels
    let showClusterRadii = true; // Toggle cluster receptive field circles (rlim)
    let showTransitionLines = true; // Toggle Markov transition arcs and paths
    let showKnnLines = true;    // Toggle k-NN graph connection vector lines
    let showMotionTail = false; // Toggle recent points trajectory motion trail (off by default)
    let showGridAxes = true;    // Toggle 2D/3D coordinate grids, axes & bounding box
    let showViewportHUD = true; // Toggle viewport header title, stats & zoom badge
    let showColorPerCluster = true; // Toggle per-cluster point colors vs uniform
    let showPrunedMarks = true; // Toggle pruned cluster crosshair marks
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
    let usePass2Nearest = false; // -pass2nearest

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

    // Multi-Dataset Slots: A, B, C, D (Reconstruction)
    const DATASET_SLOTS = ['A', 'B', 'C', 'D'];
    let activeDatasetSlot = 'A';
    let multiDatasetEnabled = false; // Multi-Dataset mode option: Off by default
    let reconstructionInfo = null; // Staged reconstruction metadata & stats for Slot D
    let reconstructionSourceNeighbors = null; // Mapping of D query -> contributing B neighbors
    let isRecon4PanelView = false; // 4-Panel Synchronized View (A, B, C, D)
    let reconOverlayMode = false; // Overlays A+C in left view and B+D in right view
    let showReconKnn = true; // Toggle k-NN rays & focused bright/grey highlights in 4-Panel view
    let reconHoveredQueryIdx = -1; // Current hover-selected query index in C/D
    let reconLockedQueryIdx = -1; // Click-pinned query index in C/D
    let reconHoveredTrainingIdx = -1; // Current hover-selected training point index in A/B
    let reconLockedTrainingIdx = -1; // Click-pinned training point index in A/B
    let reconHoveredTrainingSlot = 'A'; // 'A' or 'B'
    let reconLockedTrainingSlot = 'A'; // 'A' or 'B'

    // 4-Panel Domain-Separated 3D Cameras (Inputs A&C vs Outputs B&D)
    let reconInputCamera = {
      azimuth: -35 * (Math.PI / 180),
      elevation: 25 * (Math.PI / 180),
      panX: 0,
      panY: 0,
      zoom: 1.0,
      targetX: 0, targetY: 0, targetZ: 0,
      isLocked: false
    };
    let reconOutputCamera = {
      azimuth: -35 * (Math.PI / 180),
      elevation: 25 * (Math.PI / 180),
      panX: 0,
      panY: 0,
      zoom: 1.0,
      targetX: 0, targetY: 0, targetZ: 0,
      isLocked: false
    };

    // Dataset Staging & Ingestion State
    let isDatasetStaged = false; // True when a dataset is staged/loaded in memory
    let stagedDatasetInfo = {
      name: '3Dtorus',
      count: 10000,
      dim: 3,
      passes: 10,
      noise: 0.02
    };

    // Execution Engine Mode & Desktop Workspace
    let engineMode = 'wasm'; // 'wasm' (Interactive) or 'cli' (Native gric-cluster)
    let isDesktopBackend = false; // True if connected to native C gric-server
    let workspacePath = ''; // Current working directory
    let workspaceFiles = []; // Files found in workspace
    let selectedCliDataset = ''; // Active dataset for CLI run
    let isCliRunning = false; // True while native CLI subprocess is active
    let autoLoadCliResults = true; // Auto-load clusters into visualizer when CLI finishes

    // WASM Engine Mode
    let useWasm = true; // Use C/WASM backend when available
    let wasmSessionActive = false; // True if GricWasm handle is live

    // k-Nearest Neighbors (gric-knn) Post-Processing Options
    let enableKnn = false;
    let knnK = 50;
    let knnDtmin = 1;
    let knnDirection = 'all'; // 'all', 'past', 'future'
    let knnEpsilon = 0.0;
    let knnRlim = 0.0;
    let knnMvp = false; // Multi-Anchor Pivot Bounding (AESA)
    let knnUseSq8 = false; // 8-Bit Scalar Quantization Filtering
    let clusterUseSq8 = false; // 8-Bit Scalar Quantization Metric Pre-Filter
    let knnResults = null;
    let selectedKnnQuerySample = -1;
    let hoveredKnnNeighborId = -1;

    // Dim & Density State (gric-dimdensity)
    let dimDensityResults = null;
    let dimDensitySummary = null;
    let isDimDensityComputing = false;
    let pointColorMode = 'cluster';
    let reconQualityColoringEnabled = false;
    let reconQualityThreshold = 1.0;
    let reconQualityMask = null;
    let reconQualityIndices = null;
    let dimDensityTraceOffset = 0;

    // Interactive Selection & Hover State
    let selectedClusterId = -1;
    let selectedTupleKey = null;
    let hoveredClusterId = -1;
    let highlightClosestSample = true; // Toggle closest sample highlight on hover (ON by default)
    let hoveredClosestSample = null;   // { index, point, qIdx, screenX, screenY, distPx, clusterId }
    let lockedClosestSample = null;    // { index, point, qIdx, screenX, screenY, distPx, clusterId } - locked/pinned on click
    let frameEvaluationsLog = [];      // Frame distance evaluation records: [frameIndex -> [{ clusterId, dist, match }]]

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

    function selectImageFrame(frameIdx) {
      if (frameIdx < 0) {
        inspectedImageFrameIdx = -1;
      } else {
        const total = (benchmarkDataset && benchmarkDataset.length > 0)
          ? benchmarkDataset.length
          : (typeof totalFrames !== 'undefined' ? totalFrames : 0);
        if (total > 0) {
          inspectedImageFrameIdx = Math.min(total - 1, Math.max(0, frameIdx));
        } else {
          inspectedImageFrameIdx = 0;
        }
      }
      updateUI();
      draw();
    }

    function inspectClusterMembers(clusterId) {
      if (inspectedClusterId === clusterId) {
        inspectedClusterId = -1;
      } else {
        inspectedClusterId = clusterId;
        selectedClusterId = clusterId;
      }
      updateUI();
      draw();
    }

    function clearImageClusterInspection() {
      inspectedClusterId = -1;
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

    // Programmatic clustering rlim setter with UI synchronization
    function setClusteringRlim(val, notify = false) {
      const v = Math.max(0.001, parseFloat(val));
      if (isNaN(v)) return;
      rlim = parseFloat(v.toFixed(3));
      if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
        datasetSlots[activeDatasetSlot].rlim = rlim;
      }
      const slR = document.getElementById('sliderRlim');
      if (slR) {
        if (rlim > parseFloat(slR.max || "0.30")) {
          slR.max = Math.max(2.00, rlim).toFixed(2);
        } else if (currentDim <= 3 && rlim <= 0.30) {
          slR.max = "0.30";
        }
        slR.value = rlim;
      }
      const inpR = document.getElementById('inputRlim');
      if (inpR) {
        if (rlim > parseFloat(inpR.max || "1.000")) {
          inpR.max = Math.max(2.00, rlim).toFixed(2);
        }
        inpR.value = rlim.toFixed(3);
      }
      if (typeof isRunning !== 'undefined' && !isRunning) {
        if (typeof totalFrames !== 'undefined' && totalFrames > 0) {
          if (typeof resetClustering === 'function') {
            resetClustering(true);
            currentFrameIdx = 0;
          }
        } else if (typeof GricWasm !== 'undefined' && GricWasm.isLoaded()) {
          const params = GricWasm.buildParamsFromState();
          wasmSessionActive = GricWasm.init(params);
          if (typeof updateWasmBadge === 'function') updateWasmBadge();
        }
      }
      if (notify && typeof showToast === 'function') {
        showToast(`⚡ Clustering rlim set to ${rlim.toFixed(3)}`);
      }
      if (typeof draw === 'function') {
        draw();
      }
    }

    // Programmatic noise sigma setter with UI synchronization
    function setNoiseSigma(val, notify = false) {
      const v = Math.max(0.0, parseFloat(val));
      if (isNaN(v)) return;
      noiseSigma = parseFloat(v.toFixed(4));
      if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
        datasetSlots[activeDatasetSlot].noiseSigma = noiseSigma;
      }
      const slN = document.getElementById('sliderNoiseSigma');
      if (slN) {
        slN.value = noiseSigma;
      }
      const inpN = document.getElementById('inputNoiseSigma');
      if (inpN) {
        inpN.value = noiseSigma.toFixed(3);
      }
      if (typeof syncControlDependencies === 'function') {
        syncControlDependencies();
      }
      if (notify && typeof showToast === 'function') {
        showToast(`⚡ Noise σ set to ${noiseSigma.toFixed(3)}`);
      }
    }

    // Auto-rlim Computation (-scandist parity)
    function computeAutoRlim() {
      let pts = benchmarkDataset;
      if (!pts || pts.length === 0) {
        rawBenchmarkDataset = generateBenchmark(currentBenchmark, 250);
        applyNoiseToDataset();
        pts = benchmarkDataset;
      }
      const is32D = (currentDim === 32) ||
                    (currentBenchmark && currentBenchmark.startsWith('32D'));
      if (is32D) {
        setClusteringRlim(1.0, true);
        return;
      }
      const N = Math.min(pts.length, 300);
      const dists = [];
      for (let i = 0; i < N; i += 2) {
        for (let j = i + 1; j < N; j += 2) {
          const dx = pts[i].x - pts[j].x;
          const dy = pts[i].y - pts[j].y;
          const dz = currentDim >= 3 ? (pts[i].z - pts[j].z) : 0;
          dists.push(Math.sqrt(dx*dx + dy*dy + dz*dz));
          distClusterCluster++;
          distClusterClusterLast++;
        }
      }
      if (dists.length === 0) return;
      dists.sort((a, b) => a - b);
      const median = dists[Math.floor(dists.length * 0.5)];
      const autoR = Math.max(0.02, Math.min(0.28, median * 0.35));
      setClusteringRlim(autoR, false);
      showToast(
        `⚡ Auto-rlim (-scandist): Median=${median.toFixed(3)} ➔ rlim=${rlim.toFixed(3)}`
      );
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
        if (dim >= 3) {
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
      if (dim >= 3) {
        const u3 = Math.max(1e-12, Math.random());
        const u4 = Math.random();
        z2 = sigma * Math.sqrt(-2.0 * Math.log(u3)) * Math.cos(2.0 * Math.PI * u4);
      }
      const r = Math.sqrt(z0 * z0 + z1 * z1 + z2 * z2) || 1.0;
      const scale = maxRadius / r;
      return { dx: z0 * scale, dy: z1 * scale, dz: z2 * scale };
    }

    function applyNoiseToPoint(x, y, z = 0.0, dim = currentDim) {
      if (noiseSigma <= 1e-6) return { x, y, z: (dim >= 3) ? z : 0.0 };
      const noise = generateTruncatedGaussianNoise(dim);
      return {
        x: x + noise.dx,
        y: y + noise.dy,
        z: (dim >= 3) ? (z + noise.dz) : 0.0
      };
    }

    /**
     * Extract 3D plotting coordinates (x, y, z) mapped from selected visual dimensions.
     * When currentDim > 3, extracts coordinates from p.coords or p.anchor at [plotDimX, Y, Z].
     * Computations remain over all dimensions.
     * @param {Object} p - Point or cluster object
     * @returns {{ x: number, y: number, z: number }}
     */
    function getPlotCoords(p) {
      if (!p) return { x: 0, y: 0, z: 0 };
      if (currentDim > 3) {
        const c = p.coords || p.anchor;
        if (c && c.length > 0) {
          const len = c.length;
          const ix = (plotDimX >= 0 && plotDimX < len) ? plotDimX : 0;
          const iy = (plotDimY >= 0 && plotDimY < len) ? plotDimY : (len > 1 ? 1 : 0);
          const iz = (plotDimZ >= 0 && plotDimZ < len) ? plotDimZ : (len > 2 ? 2 : 0);
          return {
            x: (c[ix] !== undefined) ? c[ix] : 0.0,
            y: (c[iy] !== undefined) ? c[iy] : 0.0,
            z: (c[iz] !== undefined) ? c[iz] : 0.0
          };
        }
      }
      return {
        x: p.x || 0.0,
        y: p.y || 0.0,
        z: (currentDim >= 3) ? (p.z || 0.0) : 0.0
      };
    }

    /**
     * Clamp plotting dimensions to valid range [0, currentDim - 1].
     */
    function clampPlottingDimensions() {
      const maxDim = (currentDim > 0) ? currentDim - 1 : 2;
      if (plotDimX < 0 || plotDimX > maxDim) plotDimX = 0;
      if (plotDimY < 0 || plotDimY > maxDim) plotDimY = Math.min(1, maxDim);
      if (plotDimZ < 0 || plotDimZ > maxDim) plotDimZ = Math.min(2, maxDim);
    }

    /**
     * Set active 3D plotting dimensions and synchronize UI.
     * @param {number} [dx] - Visual X dimension index
     * @param {number} [dy] - Visual Y dimension index
     * @param {number} [dz] - Visual Z dimension index
     */
    function setPlottingDimensions(dx, dy, dz) {
      const maxDim = (currentDim > 0) ? currentDim - 1 : 2;
      if (typeof dx === 'number' && !isNaN(dx)) {
        plotDimX = Math.max(0, Math.min(maxDim, Math.floor(dx)));
      }
      if (typeof dy === 'number' && !isNaN(dy)) {
        plotDimY = Math.max(0, Math.min(maxDim, Math.floor(dy)));
      }
      if (typeof dz === 'number' && !isNaN(dz)) {
        plotDimZ = Math.max(0, Math.min(maxDim, Math.floor(dz)));
      }
      if (typeof datasetSlots !== 'undefined' && activeDatasetSlot &&
          datasetSlots[activeDatasetSlot]) {
        datasetSlots[activeDatasetSlot].plotDimX = plotDimX;
        datasetSlots[activeDatasetSlot].plotDimY = plotDimY;
        datasetSlots[activeDatasetSlot].plotDimZ = plotDimZ;
      }
      updatePlottingDimSelectorsUI();
      if (typeof draw === 'function') {
        draw();
      }
    }

    /**
     * Synchronize and populate plotting dimension selectors in viewport bar and sidebar.
     */
    function updatePlottingDimSelectorsUI() {
      const isHighD = (currentDim > 3);
      const projGroup = document.getElementById('projDimGroup');
      const sideCard = document.getElementById('projDimSidebarCard');

      if (projGroup) {
        projGroup.style.display = isHighD ? 'inline-flex' : 'none';
      }
      if (sideCard) {
        sideCard.style.display = isHighD ? 'block' : 'none';
      }

      if (!isHighD) return;

      clampPlottingDimensions();

      const selectIds = [
        { el: document.getElementById('selectPlotDimX'), val: plotDimX },
        { el: document.getElementById('selectPlotDimY'), val: plotDimY },
        { el: document.getElementById('selectPlotDimZ'), val: plotDimZ },
        { el: document.getElementById('selectSidePlotDimX'), val: plotDimX },
        { el: document.getElementById('selectSidePlotDimY'), val: plotDimY },
        { el: document.getElementById('selectSidePlotDimZ'), val: plotDimZ }
      ];

      selectIds.forEach(({ el, val }) => {
        if (!el) return;
        if (el.children.length !== currentDim) {
          el.innerHTML = '';
          for (let d = 0; d < currentDim; d++) {
            const opt = document.createElement('option');
            opt.value = String(d);
            opt.textContent = `d${d}`;
            el.appendChild(opt);
          }
        }
        el.value = String(val);
      });
    }

    function applyNoiseToDataset() {
      if (!rawBenchmarkDataset || rawBenchmarkDataset.length === 0) {
        benchmarkDataset = [];
        return;
      }
      if (typeof dataMode !== 'undefined' && dataMode === 'image') {
        benchmarkDataset = rawBenchmarkDataset;
        return;
      }
      const passes = (typeof loopCount === 'number' && loopCount > 0) ? loopCount : 1;
      benchmarkDataset = [];
      for (let p = 0; p < passes; p++) {
        for (let i = 0; i < rawBenchmarkDataset.length; i++) {
          const pt = rawBenchmarkDataset[i];
          if (noiseSigma <= 1e-6) {
            const newPt = {
              x: pt.x,
              y: pt.y,
              z: currentDim >= 3 ? (pt.z || 0.0) : 0.0
            };
            if (pt.coords) {
              newPt.coords = new Float64Array(pt.coords);
            }
            benchmarkDataset.push(newPt);
          } else {
            const n = applyNoiseToPoint(pt.x, pt.y, pt.z || 0.0, currentDim);
            const newPt = {
              x: n.x,
              y: n.y,
              z: currentDim >= 3 ? (n.z || 0.0) : 0.0
            };
            if (pt.coords) {
              const noisyCoords = new Float64Array(pt.coords.length);
              noisyCoords[0] = n.x;
              noisyCoords[1] = n.y;
              if (noisyCoords.length > 2) noisyCoords[2] = n.z;
              for (let d = 3; d < noisyCoords.length; d++) {
                const u1 = Math.max(1e-12, Math.random());
                const u2 = Math.random();
                const g = Math.sqrt(-2.0 * Math.log(u1)) *
                  Math.cos(2.0 * Math.PI * u2);
                noisyCoords[d] = pt.coords[d] + g * noiseSigma;
              }
              newPt.coords = noisyCoords;
            }
            benchmarkDataset.push(newPt);
          }
        }
      }
    }

    // Simulation & Benchmark State
    let currentBenchmark = "3Dtorus";
    let rawBenchmarkDataset = [];
    let benchmarkDataset = [];
    let currentFrameIdx = 0;
    let isRunning = false;
    let isComputeAllRunning = false;
    let abortComputeAllRequested = false;
    let computeAllTimer = null;
    let playTimer = null;
    let computePumpTimer = null;
    let playSpeed = -1;
    let loopCount = 10; // 1 = 1 pass, 0 = Infinite, N = N passes
    let currentLoop = 1;
    let sampleCount = 10000; // Number of points generated per pattern

    // Past sample history for point cloud building (x, y, z)
    let pastSamples = [];

    // Monolithic Clustering Engine State (2D / 3D)
    let clusters = [];
    let dcc = [];
    let dccMin = null;
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

    let currentEvaluationsAlpha = 1.0;
    let evaluationsFadeTimer = null;
    let evaluationsFadeAnimId = null;

    /**
     * Clear or fade out active frame evaluation lines so they do not persist on screen.
     * @param {boolean} immediate - If true, clears immediately; otherwise performs a smooth fade.
     */
    function clearActiveFrameEvaluations(immediate = true) {
      if (evaluationsFadeTimer) {
        clearTimeout(evaluationsFadeTimer);
        evaluationsFadeTimer = null;
      }
      if (evaluationsFadeAnimId) {
        cancelAnimationFrame(evaluationsFadeAnimId);
        evaluationsFadeAnimId = null;
      }

      if (immediate) {
        currentEvaluations = [];
        currentFrame = null;
        currentPruned = [];
        lastTransitionFrom = -1;
        lastTransitionTo = -1;
        currentEvaluationsAlpha = 1.0;
        if (typeof draw === 'function') {
          draw();
        }
        return;
      }

      const fadeStart = performance.now();
      const fadeDuration = 350;
      const startAlpha = currentEvaluationsAlpha;

      function stepFade() {
        if (typeof isRunning !== 'undefined' && isRunning) {
          evaluationsFadeAnimId = null;
          currentEvaluationsAlpha = 1.0;
          return;
        }
        const now = performance.now();
        const elapsed = now - fadeStart;
        const progress = Math.min(1.0, elapsed / fadeDuration);
        currentEvaluationsAlpha = startAlpha * (1.0 - progress);

        if (progress >= 1.0 || currentEvaluationsAlpha <= 0.01) {
          currentEvaluations = [];
          currentFrame = null;
          currentPruned = [];
          lastTransitionFrom = -1;
          lastTransitionTo = -1;
          currentEvaluationsAlpha = 1.0;
          evaluationsFadeAnimId = null;
          if (typeof draw === 'function') draw();
          return;
        }

        if (typeof draw === 'function') draw();
        evaluationsFadeAnimId = requestAnimationFrame(stepFade);
      }
      evaluationsFadeAnimId = requestAnimationFrame(stepFade);
    }

    /**
     * Schedule non-persistent active frame distance lines cleanup after a delay.
     * @param {number} delayMs - Delay before fade out begins (default: 800ms)
     */
    function scheduleActiveFrameCleanup(delayMs = 800) {
      if (evaluationsFadeTimer) {
        clearTimeout(evaluationsFadeTimer);
        evaluationsFadeTimer = null;
      }
      if (evaluationsFadeAnimId) {
        cancelAnimationFrame(evaluationsFadeAnimId);
        evaluationsFadeAnimId = null;
      }
      currentEvaluationsAlpha = 1.0;

      if (typeof isRunning !== 'undefined' && isRunning) return;

      evaluationsFadeTimer = setTimeout(() => {
        evaluationsFadeTimer = null;
        clearActiveFrameEvaluations(false);
      }, delayMs);
    }

    // =========================================================================
    //  MULTI-DATASET SLOTS MANAGEMENT (A, B, C)
    // =========================================================================

    function createInitialDatasetSlot(slotId) {
      let defaultBench = '3Dtorus';
      if (slotId === 'B') defaultBench = '3Dspiral';
      if (slotId === 'C') defaultBench = '3Drand';
      if (slotId === 'D') defaultBench = 'reconstructed';

      return {
        id: slotId,
        workspaceName: `workspace/${slotId}`,
        benchmarkKey: defaultBench,
        sampleCount: 10000,
        loopCount: 1,
        noiseSigma: 0.02,
        dataMode: 'coord',
        currentDim: 3,
        isDatasetStaged: false,
        rawBenchmarkDataset: [],
        benchmarkDataset: [],
        stagedDatasetInfo: {
          name: defaultBench,
          count: 0,
          dim: 3,
          passes: 1,
          noise: 0.02
        },
        customFileName: '',
        customFileDesc: '',
        currentFrameIdx: 0,
        currentLoop: 1,
        pastSamples: [],
        clusters: [],
        dcc: [],
        dccMin: null,
        transitionCounts: [],
        prevAssignedCluster: -1,
        lastTransitionFrom: -1,
        lastTransitionTo: -1,
        assignmentHistory: [],
        frameHistory: [],
        totalFrames: 0,
        totalEvals: 0,
        naiveEvals: 0,
        distSampleCluster: 0,
        distClusterCluster: 0,
        distSampleClusterLast: 0,
        distClusterClusterLast: 0,
        dccPopulated: 0,
        dccPairsTotal: 0,
        pruneCount3P: 0,
        pruneCount4P: 0,
        pruneCount5P: 0,
        predHitCount: 0,
        totalComputeTimeMs: 0.0,
        lastComputeTimeMs: 0.0,
        avgComputeTimeMs: 0.0,
        sparklineHistory: new Array(60).fill(0.0),
        distHistoryDFC: [],
        distHistoryDCC: [],
        rollingHistory: [],
        currentCpuLoadPct: 0.0,
        currentFps: 0.0,
        currentDistRate: 0.0,
        sessionStartTime: 0,
        sessionStartFrames: 0,
        sessionElapsedMs: 0,
        sessionIsActive: false,
        sessionAvgFps: 0.0,
        totalInitialEntropyBits: 0.0,
        totalEntropyReducedBits: 0.0,
        totalEntropyEvals: 0,
        totalEntropyGated: 0,
        lastInitialEntropy: 0.0,
        lastEntropyReduced: 0.0,
        lastInfoGainRate: 0.0,
        maxInitialEntropyObserved: 0.0,
        lastEntropyRankings: [],
        currentFrame: null,
        currentEvaluations: [],
        currentPruned: [],
        currentPredicted: [],
        currentEntropyBits: 0,
        currentExplanation: [],
        sampleTraceLog: [],
        frameEvaluationsLog: [],
        clusterMilestoneFrames: [],
        clusterSpawnRipples: [],
        selectedSampleTraceIndex: -1,
        hoveredSampleTracePoint: null,
        hoveredClosestSample: null,
        lockedClosestSample: null,
        selectedClusterId: -1,
        hoveredClusterId: -1,
        knnResults: null,
        selectedKnnQuerySample: -1,
        hoveredKnnNeighborId: -1,
        enableKnn: false,
        knnK: 50,
        knnDtmin: 1,
        knnDirection: 'all',
        knnEpsilon: 0.0,
        knnRlim: 0.0,
        knnMvp: false,
        knnUseSq8: false,
        clusterUseSq8: false,
        dimDensityResults: null,
        dimDensitySummary: null,
        isDimDensityComputing: false,
        dimDensityTraceOffset: 0,
        selectedCliDataset: '',
        imageFrameAssignments: [],
        imageFrameDists: [],
        imageClusterMembers: {},
        inspectedImageFrameIdx: -1,
        inspectedClusterId: -1,
        reconstructionInfo: null,
        reconstructionSourceNeighbors: null,
        reconKthDist: null,
        reconVariance: null,
        reconKthDistMin: 0,
        reconKthDistMax: 0,
        reconVarianceMin: 0,
        reconVarianceMax: 0,
        plotDimX: 0,
        plotDimY: 1,
        plotDimZ: 2,
        rlim: 0.100
      };
    }

    let datasetSlots = {
      A: createInitialDatasetSlot('A'),
      B: createInitialDatasetSlot('B'),
      C: createInitialDatasetSlot('C'),
      D: createInitialDatasetSlot('D')
    };

    function saveSlotState(slotId) {
      if (!datasetSlots[slotId]) return;
      const slot = datasetSlots[slotId];
      slot.benchmarkKey = currentBenchmark;
      slot.sampleCount = sampleCount;
      slot.loopCount = loopCount;
      slot.noiseSigma = noiseSigma;
      slot.rlim = rlim;
      slot.dataMode = dataMode;
      slot.currentDim = currentDim;
      slot.isDatasetStaged = isDatasetStaged;
      slot.stagedDatasetInfo = stagedDatasetInfo ? { ...stagedDatasetInfo } : {
        name: currentBenchmark,
        count: benchmarkDataset ? benchmarkDataset.length : 0,
        dim: currentDim,
        passes: loopCount || 1,
        noise: noiseSigma
      };
      slot.rawBenchmarkDataset = rawBenchmarkDataset;
      slot.benchmarkDataset = benchmarkDataset;
      slot.currentFrameIdx = currentFrameIdx;
      slot.currentLoop = typeof currentLoop !== 'undefined' ? currentLoop : 1;
      slot.pastSamples = pastSamples;
      slot.clusters = clusters;
      slot.dcc = dcc;
      slot.dccMin = dccMin;
      slot.transitionCounts = transitionCounts;
      slot.prevAssignedCluster = prevAssignedCluster;
      slot.lastTransitionFrom = lastTransitionFrom;
      slot.lastTransitionTo = lastTransitionTo;
      slot.assignmentHistory = assignmentHistory;
      slot.frameHistory = frameHistory;
      slot.totalFrames = totalFrames;
      slot.totalEvals = totalEvals;
      slot.naiveEvals = naiveEvals;
      slot.distSampleCluster = (typeof distSampleCluster !== 'undefined') ? distSampleCluster : 0;
      slot.distClusterCluster = (typeof distClusterCluster !== 'undefined') ? distClusterCluster : 0;
      slot.distSampleClusterLast = (typeof distSampleClusterLast !== 'undefined') ? distSampleClusterLast : 0;
      slot.distClusterClusterLast = (typeof distClusterClusterLast !== 'undefined') ? distClusterClusterLast : 0;
      slot.dccPopulated = (typeof dccPopulated !== 'undefined') ? dccPopulated : 0;
      slot.dccPairsTotal = (typeof dccPairsTotal !== 'undefined') ? dccPairsTotal : 0;
      slot.pruneCount3P = (typeof pruneCount3P !== 'undefined') ? pruneCount3P : 0;
      slot.pruneCount4P = (typeof pruneCount4P !== 'undefined') ? pruneCount4P : 0;
      slot.pruneCount5P = (typeof pruneCount5P !== 'undefined') ? pruneCount5P : 0;
      slot.predHitCount = (typeof predHitCount !== 'undefined') ? predHitCount : 0;
      slot.totalComputeTimeMs = (typeof totalComputeTimeMs !== 'undefined') ? totalComputeTimeMs : 0;
      slot.lastComputeTimeMs = (typeof lastComputeTimeMs !== 'undefined') ? lastComputeTimeMs : 0;
      slot.avgComputeTimeMs = (typeof avgComputeTimeMs !== 'undefined') ? avgComputeTimeMs : 0;
      slot.sparklineHistory = (typeof sparklineHistory !== 'undefined') ? [...sparklineHistory] : new Array(60).fill(0.0);
      slot.distHistoryDFC = (typeof distHistoryDFC !== 'undefined') ? [...distHistoryDFC] : [];
      slot.distHistoryDCC = (typeof distHistoryDCC !== 'undefined') ? [...distHistoryDCC] : [];
      slot.rollingHistory = (typeof rollingHistory !== 'undefined') ? [...rollingHistory] : [];
      slot.currentCpuLoadPct = (typeof currentCpuLoadPct !== 'undefined') ? currentCpuLoadPct : 0;
      slot.currentFps = (typeof currentFps !== 'undefined') ? currentFps : 0;
      slot.currentDistRate = (typeof currentDistRate !== 'undefined') ? currentDistRate : 0;
      slot.sessionStartTime = (typeof sessionStartTime !== 'undefined') ? sessionStartTime : 0;
      slot.sessionStartFrames = (typeof sessionStartFrames !== 'undefined') ? sessionStartFrames : 0;
      slot.sessionElapsedMs = (typeof sessionElapsedMs !== 'undefined') ? sessionElapsedMs : 0;
      slot.sessionIsActive = (typeof sessionIsActive !== 'undefined') ? sessionIsActive : false;
      slot.sessionAvgFps = (typeof sessionAvgFps !== 'undefined') ? sessionAvgFps : 0;
      slot.totalInitialEntropyBits = (typeof totalInitialEntropyBits !== 'undefined') ? totalInitialEntropyBits : 0;
      slot.totalEntropyReducedBits = (typeof totalEntropyReducedBits !== 'undefined') ? totalEntropyReducedBits : 0;
      slot.totalEntropyEvals = (typeof totalEntropyEvals !== 'undefined') ? totalEntropyEvals : 0;
      slot.totalEntropyGated = (typeof totalEntropyGated !== 'undefined') ? totalEntropyGated : 0;
      slot.lastInitialEntropy = (typeof lastInitialEntropy !== 'undefined') ? lastInitialEntropy : 0;
      slot.lastEntropyReduced = (typeof lastEntropyReduced !== 'undefined') ? lastEntropyReduced : 0;
      slot.lastInfoGainRate = (typeof lastInfoGainRate !== 'undefined') ? lastInfoGainRate : 0;
      slot.maxInitialEntropyObserved = (typeof maxInitialEntropyObserved !== 'undefined') ? maxInitialEntropyObserved : 0;
      slot.lastEntropyRankings = (typeof lastEntropyRankings !== 'undefined') ? [...lastEntropyRankings] : [];
      slot.currentFrame = currentFrame;
      slot.currentEvaluations = currentEvaluations;
      slot.currentPruned = currentPruned;
      slot.currentPredicted = currentPredicted;
      slot.currentEntropyBits = currentEntropyBits;
      slot.currentExplanation = currentExplanation;
      slot.sampleTraceLog = sampleTraceLog;
      slot.frameEvaluationsLog = frameEvaluationsLog;
      slot.clusterMilestoneFrames = clusterMilestoneFrames;
      slot.clusterSpawnRipples = clusterSpawnRipples;
      slot.selectedSampleTraceIndex = (typeof selectedSampleTraceIndex !== 'undefined') ? selectedSampleTraceIndex : -1;
      slot.hoveredSampleTracePoint = hoveredSampleTracePoint;
      slot.hoveredClosestSample = hoveredClosestSample;
      slot.lockedClosestSample = lockedClosestSample;
      slot.selectedClusterId = selectedClusterId;
      slot.hoveredClusterId = hoveredClusterId;
      slot.knnResults = knnResults;
      slot.selectedKnnQuerySample = selectedKnnQuerySample;
      slot.hoveredKnnNeighborId = hoveredKnnNeighborId;
      slot.enableKnn = enableKnn;
      slot.knnK = knnK;
      slot.knnDtmin = knnDtmin;
      slot.knnDirection = knnDirection;
      slot.knnEpsilon = knnEpsilon;
      slot.knnRlim = knnRlim;
      slot.knnMvp = knnMvp;
      slot.knnUseSq8 = knnUseSq8;
      slot.clusterUseSq8 = clusterUseSq8;
      slot.dimDensityResults = dimDensityResults;
      slot.dimDensitySummary = dimDensitySummary;
      slot.isDimDensityComputing = isDimDensityComputing;
      slot.dimDensityTraceOffset = dimDensityTraceOffset;
      slot.selectedCliDataset = selectedCliDataset;
      slot.imageFrameAssignments = imageFrameAssignments;
      slot.imageFrameDists = imageFrameDists;
      slot.imageClusterMembers = imageClusterMembers;
      slot.inspectedImageFrameIdx = inspectedImageFrameIdx;
      slot.inspectedClusterId = inspectedClusterId;
      slot.reconstructionInfo = reconstructionInfo;
      slot.reconstructionSourceNeighbors = reconstructionSourceNeighbors;
      slot.reconQualityColoringEnabled = reconQualityColoringEnabled;
      slot.reconQualityThreshold = reconQualityThreshold;
      slot.reconQualityMask = reconQualityMask;
      slot.reconQualityIndices = reconQualityIndices;
      slot.plotDimX = plotDimX;
      slot.plotDimY = plotDimY;
      slot.plotDimZ = plotDimZ;
    }

    function loadSlotState(slotId) {
      if (!datasetSlots[slotId]) return;
      const slot = datasetSlots[slotId];
      currentBenchmark = slot.benchmarkKey || '3Dtorus';
      sampleCount = slot.sampleCount || 10000;
      loopCount = slot.loopCount || 10;
      noiseSigma = (slot.noiseSigma !== undefined) ? slot.noiseSigma : 0.02;
      dataMode = slot.dataMode || 'coord';
      currentDim = slot.currentDim || 3;
      if (typeof slot.rlim === 'number') {
        setClusteringRlim(slot.rlim, false);
      } else if (slot.benchmarkKey &&
                 (slot.benchmarkKey.startsWith('32D') || slot.currentDim === 32)) {
        setClusteringRlim(1.0, false);
      } else {
        setClusteringRlim(0.100, false);
      }
      if (typeof slot.noiseSigma === 'number') {
        setNoiseSigma(slot.noiseSigma, false);
      } else if (slot.benchmarkKey &&
                 (slot.benchmarkKey.startsWith('32D') || slot.currentDim === 32)) {
        setNoiseSigma(0.005, false);
      } else {
        setNoiseSigma(0.020, false);
      }
      plotDimX = (typeof slot.plotDimX === 'number') ? slot.plotDimX : 0;
      plotDimY = (typeof slot.plotDimY === 'number') ? slot.plotDimY : 1;
      plotDimZ = (typeof slot.plotDimZ === 'number') ? slot.plotDimZ : 2;
      clampPlottingDimensions();
      updatePlottingDimSelectorsUI();
      isDatasetStaged = slot.isDatasetStaged || false;
      stagedDatasetInfo = slot.stagedDatasetInfo || {
        name: currentBenchmark,
        count: 0,
        dim: currentDim,
        passes: loopCount,
        noise: noiseSigma
      };
      rawBenchmarkDataset = slot.rawBenchmarkDataset || [];
      benchmarkDataset = slot.benchmarkDataset || [];
      currentFrameIdx = slot.currentFrameIdx || 0;
      if (typeof currentLoop !== 'undefined') currentLoop = slot.currentLoop || 1;
      pastSamples = slot.pastSamples || [];
      if ((!pastSamples || pastSamples.length === 0) &&
          benchmarkDataset && benchmarkDataset.length > 0 &&
          dataMode === 'coord') {
        const maxStagedPreview = 100000;
        const stride = benchmarkDataset.length > maxStagedPreview
          ? Math.ceil(benchmarkDataset.length / maxStagedPreview) : 1;
        pastSamples = [];
        for (let i = 0; i < benchmarkDataset.length; i += stride) {
          const pt = benchmarkDataset[i];
          const newPt = {
            x: pt.x,
            y: pt.y,
            z: pt.z || 0.0,
            clusterId: -1,
            frameIndex: i
          };
          if (pt.coords) newPt.coords = pt.coords;
          pastSamples.push(newPt);
        }
        slot.pastSamples = pastSamples;
      }
      clusters = slot.clusters || [];
      dcc = slot.dcc || [];
      dccMin = slot.dccMin || null;
      transitionCounts = slot.transitionCounts || [];
      prevAssignedCluster = (slot.prevAssignedCluster !== undefined) ? slot.prevAssignedCluster : -1;
      lastTransitionFrom = (slot.lastTransitionFrom !== undefined) ? slot.lastTransitionFrom : -1;
      lastTransitionTo = (slot.lastTransitionTo !== undefined) ? slot.lastTransitionTo : -1;
      assignmentHistory = slot.assignmentHistory || [];
      frameHistory = slot.frameHistory || [];
      totalFrames = slot.totalFrames || 0;
      totalEvals = slot.totalEvals || 0;
      naiveEvals = slot.naiveEvals || 0;
      if (typeof distSampleCluster !== 'undefined') distSampleCluster = slot.distSampleCluster || 0;
      if (typeof distClusterCluster !== 'undefined') distClusterCluster = slot.distClusterCluster || 0;
      if (typeof distSampleClusterLast !== 'undefined') distSampleClusterLast = slot.distSampleClusterLast || 0;
      if (typeof distClusterClusterLast !== 'undefined') distClusterClusterLast = slot.distClusterClusterLast || 0;
      if (typeof dccPopulated !== 'undefined') dccPopulated = slot.dccPopulated || 0;
      if (typeof dccPairsTotal !== 'undefined') dccPairsTotal = slot.dccPairsTotal || 0;
      if (typeof pruneCount3P !== 'undefined') pruneCount3P = slot.pruneCount3P || 0;
      if (typeof pruneCount4P !== 'undefined') pruneCount4P = slot.pruneCount4P || 0;
      if (typeof pruneCount5P !== 'undefined') pruneCount5P = slot.pruneCount5P || 0;
      if (typeof predHitCount !== 'undefined') predHitCount = slot.predHitCount || 0;
      if (typeof totalComputeTimeMs !== 'undefined') totalComputeTimeMs = slot.totalComputeTimeMs || 0;
      if (typeof lastComputeTimeMs !== 'undefined') lastComputeTimeMs = slot.lastComputeTimeMs || 0;
      if (typeof avgComputeTimeMs !== 'undefined') avgComputeTimeMs = slot.avgComputeTimeMs || 0;
      if (typeof sparklineHistory !== 'undefined') sparklineHistory = slot.sparklineHistory ? [...slot.sparklineHistory] : new Array(60).fill(0.0);
      if (typeof distHistoryDFC !== 'undefined') distHistoryDFC = slot.distHistoryDFC ? [...slot.distHistoryDFC] : [];
      if (typeof distHistoryDCC !== 'undefined') distHistoryDCC = slot.distHistoryDCC ? [...slot.distHistoryDCC] : [];
      if (typeof rollingHistory !== 'undefined') rollingHistory = slot.rollingHistory ? [...slot.rollingHistory] : [];
      if (typeof currentCpuLoadPct !== 'undefined') currentCpuLoadPct = slot.currentCpuLoadPct || 0;
      if (typeof currentFps !== 'undefined') currentFps = slot.currentFps || 0;
      if (typeof currentDistRate !== 'undefined') currentDistRate = slot.currentDistRate || 0;
      if (typeof sessionStartTime !== 'undefined') sessionStartTime = slot.sessionStartTime || 0;
      if (typeof sessionStartFrames !== 'undefined') sessionStartFrames = slot.sessionStartFrames || 0;
      if (typeof sessionElapsedMs !== 'undefined') sessionElapsedMs = slot.sessionElapsedMs || 0;
      if (typeof sessionIsActive !== 'undefined') sessionIsActive = slot.sessionIsActive || false;
      if (typeof sessionAvgFps !== 'undefined') sessionAvgFps = slot.sessionAvgFps || 0;
      if (typeof totalInitialEntropyBits !== 'undefined') totalInitialEntropyBits = slot.totalInitialEntropyBits || 0;
      if (typeof totalEntropyReducedBits !== 'undefined') totalEntropyReducedBits = slot.totalEntropyReducedBits || 0;
      if (typeof totalEntropyEvals !== 'undefined') totalEntropyEvals = slot.totalEntropyEvals || 0;
      if (typeof totalEntropyGated !== 'undefined') totalEntropyGated = slot.totalEntropyGated || 0;
      if (typeof lastInitialEntropy !== 'undefined') lastInitialEntropy = slot.lastInitialEntropy || 0;
      if (typeof lastEntropyReduced !== 'undefined') lastEntropyReduced = slot.lastEntropyReduced || 0;
      if (typeof lastInfoGainRate !== 'undefined') lastInfoGainRate = slot.lastInfoGainRate || 0;
      if (typeof maxInitialEntropyObserved !== 'undefined') maxInitialEntropyObserved = slot.maxInitialEntropyObserved || 0;
      if (typeof lastEntropyRankings !== 'undefined') lastEntropyRankings = slot.lastEntropyRankings ? [...slot.lastEntropyRankings] : [];
      currentFrame = slot.currentFrame || null;
      currentEvaluations = slot.currentEvaluations || [];
      currentPruned = slot.currentPruned || [];
      currentPredicted = slot.currentPredicted || [];
      currentEntropyBits = slot.currentEntropyBits || 0;
      currentExplanation = slot.currentExplanation || [];
      sampleTraceLog = slot.sampleTraceLog || [];
      frameEvaluationsLog = slot.frameEvaluationsLog || [];
      clusterMilestoneFrames = slot.clusterMilestoneFrames || [];
      clusterSpawnRipples = slot.clusterSpawnRipples || [];
      selectedSampleTraceIndex = (slot.selectedSampleTraceIndex !== undefined) ? slot.selectedSampleTraceIndex : -1;
      hoveredSampleTracePoint = slot.hoveredSampleTracePoint || null;
      hoveredClosestSample = slot.hoveredClosestSample || null;
      lockedClosestSample = slot.lockedClosestSample || null;
      selectedClusterId = (slot.selectedClusterId !== undefined) ? slot.selectedClusterId : -1;
      hoveredClusterId = (slot.hoveredClusterId !== undefined) ? slot.hoveredClusterId : -1;
      knnResults = slot.knnResults || null;
      selectedKnnQuerySample = (slot.selectedKnnQuerySample !== undefined) ? slot.selectedKnnQuerySample : -1;
      hoveredKnnNeighborId = (slot.hoveredKnnNeighborId !== undefined) ? slot.hoveredKnnNeighborId : -1;
      enableKnn = slot.enableKnn || false;
      knnK = slot.knnK || 50;
      knnDtmin = (slot.knnDtmin !== undefined) ? slot.knnDtmin : 1;
      knnDirection = slot.knnDirection || 'all';
      knnEpsilon = (slot.knnEpsilon !== undefined) ? slot.knnEpsilon : 0.0;
      knnRlim = (slot.knnRlim !== undefined) ? slot.knnRlim : 0.0;
      knnMvp = slot.knnMvp || false;
      knnUseSq8 = slot.knnUseSq8 || false;
      clusterUseSq8 = slot.clusterUseSq8 || false;
      dimDensityResults = slot.dimDensityResults || null;
      dimDensitySummary = slot.dimDensitySummary || null;
      isDimDensityComputing = slot.isDimDensityComputing || false;
      dimDensityTraceOffset = slot.dimDensityTraceOffset || 0;
      selectedCliDataset = slot.selectedCliDataset || '';
      imageFrameAssignments = slot.imageFrameAssignments || [];
      imageFrameDists = slot.imageFrameDists || [];
      imageClusterMembers = slot.imageClusterMembers || {};
      inspectedImageFrameIdx = (slot.inspectedImageFrameIdx !== undefined) ? slot.inspectedImageFrameIdx : -1;
      inspectedClusterId = (slot.inspectedClusterId !== undefined) ? slot.inspectedClusterId : -1;
      reconstructionInfo = slot.reconstructionInfo || null;
      reconstructionSourceNeighbors = slot.reconstructionSourceNeighbors || null;
      reconQualityColoringEnabled = slot.reconQualityColoringEnabled || false;
      reconQualityThreshold = (slot.reconQualityThreshold !== undefined)
        ? slot.reconQualityThreshold : 1.0;
      reconQualityMask = slot.reconQualityMask || null;
      reconQualityIndices = slot.reconQualityIndices || null;
    }

    function switchDatasetSlot(newSlotId) {
      if (!DATASET_SLOTS.includes(newSlotId)) return;
      if (newSlotId === activeDatasetSlot) return;

      // Auto-pause active simulation to prevent race conditions
      if (typeof pauseSimulation === 'function' && isRunning) {
        pauseSimulation();
      }
      if (typeof abortComputeAll === 'function' && isComputeAllRunning) {
        abortComputeAll();
      }

      // Save previous slot state
      saveSlotState(activeDatasetSlot);

      // Switch active slot
      activeDatasetSlot = newSlotId;

      // Load new slot state
      loadSlotState(newSlotId);

      // Sync WASM session if available
      if (useWasm && typeof GricWasm !== 'undefined' && GricWasm.isLoaded()) {
        const params = GricWasm.buildParamsFromState();
        wasmSessionActive = GricWasm.init(params);
        if (typeof updateWasmBadge === 'function') updateWasmBadge();
      }

      // Update UI controls & highlights
      if (typeof updateDatasetStatusBadge === 'function') {
        updateDatasetStatusBadge();
      }
      if (typeof updateUI === 'function') {
        updateUI();
      }
      const descEl = document.getElementById('benchmarkDesc');
      if (descEl && typeof BENCHMARK_DESCS !== 'undefined') {
        descEl.innerHTML = BENCHMARK_DESCS[currentBenchmark] || `<b>${currentBenchmark}</b>`;
      }
      if (typeof renderKnnTrace === 'function') {
        renderKnnTrace();
      }
      if (typeof renderDimDensityDashboard === 'function') {
        renderDimDensityDashboard();
      }
      if (typeof renderReconstructionDashboard === 'function') {
        renderReconstructionDashboard();
      }
      if (typeof renderEntropyTrace === 'function') {
        renderEntropyTrace();
      }
      if (typeof renderSampleHistoryUI === 'function') {
        renderSampleHistoryUI();
      }
      if (typeof renderDataStructuresUI === 'function') {
        renderDataStructuresUI();
      }
      if (typeof refreshWorkspaceFiles === 'function') {
        refreshWorkspaceFiles();
      }
      if (typeof draw === 'function') {
        draw();
      }

      if (typeof showToast === 'function') {
        showToast(`⚡ Active Dataset: ${newSlotId} (${datasetSlots[newSlotId].workspaceName})`);
      }
    }

    function updateMultiDatasetUI() {
      const rowB = document.getElementById('datasetRowB');
      const rowC = document.getElementById('datasetRowC');
      const rowD = document.getElementById('datasetRowD');
      const slotBtnA = document.getElementById('btnToggleSlotA');
      const btnToggle = document.getElementById('btnToggleMultiDataset');
      const btnToggleSide = document.getElementById('btnToggleMultiDatasetSide');
      const slotGroupSide = document.getElementById('datasetSlotTogglesSide');

      if (rowB) rowB.style.display = multiDatasetEnabled ? 'flex' : 'none';
      if (rowC) rowC.style.display = multiDatasetEnabled ? 'flex' : 'none';
      if (rowD) rowD.style.display = multiDatasetEnabled ? 'flex' : 'none';
      if (slotBtnA) slotBtnA.style.display = multiDatasetEnabled ? 'inline-flex' : 'none';
      if (slotGroupSide) slotGroupSide.style.display = multiDatasetEnabled ? 'flex' : 'none';

      if (btnToggle) {
        if (multiDatasetEnabled) {
          btnToggle.classList.add('active');
          btnToggle.style.background = 'rgba(56, 189, 248, 0.2)';
          btnToggle.style.color = '#38bdf8';
          btnToggle.style.borderColor = 'rgba(56, 189, 248, 0.5)';
          btnToggle.textContent = '🗂️ Multi-Dataset (A-D): ON';
        } else {
          btnToggle.classList.remove('active');
          btnToggle.style.background = 'rgba(148, 163, 184, 0.1)';
          btnToggle.style.color = 'var(--text-muted)';
          btnToggle.style.borderColor = 'var(--card-border)';
          btnToggle.textContent = '🗂️ Multi-Dataset: OFF';
        }
      }

      if (btnToggleSide) {
        if (multiDatasetEnabled) {
          btnToggleSide.classList.add('active');
          btnToggleSide.style.background = 'rgba(56, 189, 248, 0.2)';
          btnToggleSide.style.color = '#38bdf8';
          btnToggleSide.style.borderColor = 'rgba(56, 189, 248, 0.5)';
          btnToggleSide.textContent = '🗂️ 4 Datasets: ON';
        } else {
          btnToggleSide.classList.remove('active');
          btnToggleSide.style.background = 'rgba(148, 163, 184, 0.1)';
          btnToggleSide.style.color = 'var(--text-muted)';
          btnToggleSide.style.borderColor = 'var(--card-border)';
          btnToggleSide.textContent = '🗂️ 4 Datasets: OFF';
        }
      }
    }

    function populateDefaultMultiDatasets() {
      // 1. Configure Slot A: 3D torus manifold, N=1, 10000 pts
      if (datasetSlots['A']) {
        const slotA = datasetSlots['A'];
        slotA.benchmarkKey = '3Dtorus';
        slotA.sampleCount = 10000;
        slotA.loopCount = 1;
        slotA.noiseSigma = 0.02;
        slotA.currentDim = 3;
        slotA.dataMode = 'coord';
      }

      // 2. Configure Slot B: 3D spiral, N=1, 10000 pts
      if (datasetSlots['B']) {
        const slotB = datasetSlots['B'];
        slotB.benchmarkKey = '3Dspiral';
        slotB.sampleCount = 10000;
        slotB.loopCount = 1;
        slotB.noiseSigma = 0.02;
        slotB.currentDim = 3;
        slotB.dataMode = 'coord';
      }

      // 3. Configure Slot C: 3Drand, N=1, 10000 pts
      if (datasetSlots['C']) {
        const slotC = datasetSlots['C'];
        slotC.benchmarkKey = '3Drand';
        slotC.sampleCount = 10000;
        slotC.loopCount = 1;
        slotC.noiseSigma = 0.02;
        slotC.currentDim = 3;
        slotC.dataMode = 'coord';
      }

      // 4. Stage datasets B and C in background
      if (typeof stageDataset === 'function') {
        stageDataset('3Dspiral', 'B');
        stageDataset('3Drand', 'C');
      }

      // 5. Clear Slot D
      if (typeof clearDatasetSlot === 'function') {
        clearDatasetSlot('D');
      }

      // 6. Stage and load Slot A
      if (activeDatasetSlot === 'A') {
        sampleCount = 10000;
        loopCount = 1;
        currentBenchmark = '3Dtorus';
        currentDim = 3;
        noiseSigma = 0.02;

        const selCount = document.getElementById('selectSampleCount');
        if (selCount) selCount.value = '10000';
        const selCountSide = document.getElementById('selectSampleCountSide');
        if (selCountSide) selCountSide.value = '10000';
        const selLoop = document.getElementById('selectLoop');
        if (selLoop) selLoop.value = '1';
        const selLoopSide = document.getElementById('selectLoopSide');
        if (selLoopSide) selLoopSide.value = '1';
        const selBench = document.getElementById('selectBenchmark');
        if (selBench) selBench.value = '3Dtorus';
        const selBenchSide = document.getElementById('selectBenchmarkSide');
        if (selBenchSide) selBenchSide.value = '3Dtorus';
        const selSlotA = document.getElementById('selectBenchmark_A');
        if (selSlotA) selSlotA.value = '3Dtorus';

        if (typeof stageDataset === 'function') {
          stageDataset('3Dtorus', 'A');
        }
        if (typeof loadSelectedBenchmark === 'function') {
          loadSelectedBenchmark();
        }
      } else {
        if (typeof stageDataset === 'function') {
          stageDataset('3Dtorus', 'A');
        }
      }

      const selSlotB = document.getElementById('selectBenchmark_B');
      if (selSlotB) selSlotB.value = '3Dspiral';
      const selSlotC = document.getElementById('selectBenchmark_C');
      if (selSlotC) selSlotC.value = '3Drand';
      const selSlotD = document.getElementById('selectBenchmark_D');
      if (selSlotD) selSlotD.value = 'reconstructed';

      if (typeof updateDatasetStatusBadge === 'function') {
        updateDatasetStatusBadge();
      }
      if (typeof updateMultiDatasetUI === 'function') {
        updateMultiDatasetUI();
      }
    }

    function setMultiDatasetEnabled(enabled) {
      const wasEnabled = multiDatasetEnabled;
      multiDatasetEnabled = !!enabled;
      if (!multiDatasetEnabled && activeDatasetSlot !== 'A') {
        switchDatasetSlot('A');
      }
      updateMultiDatasetUI();

      if (multiDatasetEnabled && !wasEnabled) {
        populateDefaultMultiDatasets();
      }

      if (typeof showToast === 'function') {
        showToast(multiDatasetEnabled
          ? '🗂️ Multi-Dataset Mode: ON (Defaults loaded: A=3Dtorus, B=3Dspiral, C=3Drand, D=clear)'
          : '🗂️ Multi-Dataset Mode: OFF (Single Dataset active)');
      }
    }

    function clearDatasetSlot(slotId) {
      if (!datasetSlots[slotId]) return;

      // Auto-pause if active simulation is running
      if (activeDatasetSlot === slotId) {
        if (typeof pauseSimulation === 'function' && isRunning) {
          pauseSimulation();
        }
        if (typeof abortComputeAll === 'function' && isComputeAllRunning) {
          abortComputeAll();
        }
      }

      const slot = datasetSlots[slotId];
      slot.isDatasetStaged = false;
      slot.rawBenchmarkDataset = [];
      slot.benchmarkDataset = [];
      slot.pastSamples = [];
      slot.stagedDatasetInfo = {
        name: 'None',
        count: 0,
        dim: slot.currentDim || 2,
        passes: 1,
        noise: 0
      };
      slot.clusters = [];
      slot.dcc = [];
      slot.dccMin = null;
      slot.transitionCounts = [];
      slot.prevAssignedCluster = -1;
      slot.lastTransitionFrom = -1;
      slot.lastTransitionTo = -1;
      slot.assignmentHistory = [];
      slot.frameHistory = [];
      slot.totalFrames = 0;
      slot.totalEvals = 0;
      slot.naiveEvals = 0;
      slot.currentFrameIdx = 0;
      slot.distSampleCluster = 0;
      slot.distClusterCluster = 0;
      slot.distSampleClusterLast = 0;
      slot.distClusterClusterLast = 0;
      slot.dccPopulated = 0;
      slot.dccPairsTotal = 0;
      slot.pruneCount3P = 0;
      slot.pruneCount4P = 0;
      slot.pruneCount5P = 0;
      slot.predHitCount = 0;
      slot.totalComputeTimeMs = 0;
      slot.lastComputeTimeMs = 0;
      slot.avgComputeTimeMs = 0;
      slot.sparklineHistory = new Array(60).fill(0.0);
      slot.distHistoryDFC = [];
      slot.distHistoryDCC = [];
      slot.rollingHistory = [];
      slot.currentCpuLoadPct = 0;
      slot.currentFps = 0;
      slot.currentDistRate = 0;
      slot.sessionIsActive = false;
      slot.sessionAvgFps = 0;
      slot.totalInitialEntropyBits = 0;
      slot.totalEntropyReducedBits = 0;
      slot.totalEntropyEvals = 0;
      slot.totalEntropyGated = 0;
      slot.lastInitialEntropy = 0;
      slot.lastEntropyReduced = 0;
      slot.lastInfoGainRate = 0;
      slot.maxInitialEntropyObserved = 0;
      slot.lastEntropyRankings = [];
      slot.currentFrame = null;
      slot.currentEvaluations = [];
      slot.currentPruned = [];
      slot.currentPredicted = [];
      slot.currentEntropyBits = 0;
      slot.currentExplanation = [];
      slot.sampleTraceLog = [];
      slot.frameEvaluationsLog = [];
      slot.clusterMilestoneFrames = [];
      slot.clusterSpawnRipples = [];
      slot.selectedSampleTraceIndex = -1;
      slot.hoveredSampleTracePoint = null;
      slot.hoveredClosestSample = null;
      slot.lockedClosestSample = null;
      slot.selectedClusterId = -1;
      slot.hoveredClusterId = -1;
      slot.knnResults = null;
      slot.selectedKnnQuerySample = -1;
      slot.hoveredKnnNeighborId = -1;
      slot.dimDensityResults = null;
      slot.dimDensitySummary = null;
      slot.isDimDensityComputing = false;
      slot.dimDensityTraceOffset = 0;
      slot.selectedCliDataset = '';
      slot.imageFrameAssignments = [];
      slot.imageFrameDists = [];
      slot.imageClusterMembers = {};
      slot.inspectedImageFrameIdx = -1;
      slot.inspectedClusterId = -1;
      slot.reconstructionInfo = null;
      slot.reconstructionSourceNeighbors = null;
      slot.reconKthDist = null;
      slot.reconVariance = null;
      slot.reconKthDistMin = 0;
      slot.reconKthDistMax = 0;
      slot.reconVarianceMin = 0;
      slot.reconVarianceMax = 0;
      slot.reconQualityColoringEnabled = false;
      slot.reconQualityThreshold = 1.0;
      slot.reconQualityMask = null;
      slot.reconQualityIndices = null;
      slot._unprunedBackup = null;

      // Update toolbar status pill & indicators
      const pill = document.getElementById(`datasetStatusPill_${slotId}`);
      if (pill) {
        pill.textContent = '⚪ Empty';
        pill.style.background = 'rgba(100, 116, 139, 0.2)';
        pill.style.color = '#94a3b8';
        pill.style.borderColor = 'rgba(100, 116, 139, 0.4)';
      }
      const clustPill = document.getElementById(`datasetClusteredPill_${slotId}`);
      if (clustPill) {
        clustPill.textContent = '⚪ Unclustered';
        clustPill.style.background = 'rgba(100, 116, 139, 0.12)';
        clustPill.style.color = '#64748b';
        clustPill.style.borderColor = 'rgba(100, 116, 139, 0.25)';
      }
      const knnPill = document.getElementById(`datasetKnnPill_${slotId}`);
      if (knnPill) {
        knnPill.textContent = '⚪ No k-NN';
        knnPill.style.background = 'rgba(100, 116, 139, 0.12)';
        knnPill.style.color = '#64748b';
        knnPill.style.borderColor = 'rgba(100, 116, 139, 0.25)';
      }

      // If this is the active slot, synchronize global state and redraw
      if (activeDatasetSlot === slotId) {
        benchmarkDataset = [];
        rawBenchmarkDataset = [];
        pastSamples = [];
        isDatasetStaged = false;
        clusters = [];
        dcc = [];
        dccMin = null;
        transitionCounts = [];
        prevAssignedCluster = -1;
        lastTransitionFrom = -1;
        lastTransitionTo = -1;
        assignmentHistory = [];
        frameHistory = [];
        totalFrames = 0;
        totalEvals = 0;
        naiveEvals = 0;
        currentFrameIdx = 0;
        currentEvaluations = [];
        currentPruned = [];
        currentPredicted = [];

        loadSlotState(slotId);

        // Reset WASM session if available
        if (useWasm && typeof GricWasm !== 'undefined' && GricWasm.isLoaded()) {
          GricWasm.reset();
          const params = GricWasm.buildParamsFromState();
          wasmSessionActive = GricWasm.init(params);
          if (typeof updateWasmBadge === 'function') updateWasmBadge();
        }

        if (typeof resetView === 'function') {
          resetView();
        }
        if (typeof updateDatasetStatusBadge === 'function') {
          updateDatasetStatusBadge();
        }
        if (typeof updateUI === 'function') {
          updateUI();
        }
        if (typeof renderKnnTrace === 'function') {
          renderKnnTrace();
        }
        if (typeof renderDimDensityDashboard === 'function') {
          renderDimDensityDashboard();
        }
        if (typeof renderReconstructionDashboard === 'function') {
          renderReconstructionDashboard();
        }
        if (typeof renderEntropyTrace === 'function') {
          renderEntropyTrace();
        }
        if (typeof renderSampleHistoryUI === 'function') {
          renderSampleHistoryUI();
        }
        if (typeof draw === 'function') {
          draw();
        }
      } else {
        if (typeof renderReconstructionDashboard === 'function') {
          renderReconstructionDashboard();
        }
      }

      if (typeof showToast === 'function') {
        showToast(`🗑️ Cleared Dataset [${slotId}] (data reset & view cleared)`);
      }
    }

    function syncRecon4PanelUI() {
      const is4P = isRecon4PanelView && !reconOverlayMode;
      const isOverlay = isRecon4PanelView && reconOverlayMode;

      const btnPreset = document.getElementById('btnPresetRecon4Panel');
      if (btnPreset) {
        btnPreset.classList.toggle('active', isRecon4PanelView);
        btnPreset.textContent = isOverlay ? '⧉ A+C | B+D' : '⊞ ABCD';
      }
      const btnSide = document.getElementById('btnToggleRecon4PanelView');
      if (btnSide) {
        btnSide.classList.toggle('active', is4P);
        btnSide.classList.toggle('toggle-active', is4P);
      }
      const btnTop = document.getElementById('btnToggleRecon4PanelTop');
      if (btnTop) {
        btnTop.classList.toggle('active', is4P);
        btnTop.classList.toggle('toggle-active', is4P);
      }
      const btnOverlayTop = document.getElementById('btnToggleReconOverlayTop');
      if (btnOverlayTop) {
        btnOverlayTop.classList.toggle('active', isOverlay);
        btnOverlayTop.classList.toggle('toggle-active', isOverlay);
      }
      const btnOverlaySide = document.getElementById('btnToggleReconOverlaySide');
      if (btnOverlaySide) {
        btnOverlaySide.classList.toggle('active', isOverlay);
        btnOverlaySide.classList.toggle('toggle-active', isOverlay);
      }
    }

    function syncReconOverlayUI() {
      syncRecon4PanelUI();
    }

    function toggleRecon4PanelView() {
      if (isRecon4PanelView && reconOverlayMode) {
        // Currently in Overlaid (A+C & B+D) view -> switch back to 4-Panel ABCD view
        reconOverlayMode = false;
        isRecon4PanelView = true;
        if (typeof showToast === 'function') {
          showToast('⊞ ABCD 4-Panel split view restored');
        }
      } else if (isRecon4PanelView) {
        // Currently in 4-Panel ABCD view -> turn off ABCD view completely
        isRecon4PanelView = false;
        reconOverlayMode = false;
        if (typeof showToast === 'function') {
          showToast('⊞ ABCD view closed');
        }
      } else {
        // Currently in single dataset view -> turn on 4-Panel ABCD view
        isRecon4PanelView = true;
        reconOverlayMode = false;
        maximizedQuad = null;
        if (typeof showToast === 'function') {
          showToast('⊞ ABCD 4-Panel synchronized view enabled');
        }
      }
      syncRecon4PanelUI();
      if (typeof updateViewPresetBarPosition === 'function') {
        updateViewPresetBarPosition();
      }
      if (typeof draw === 'function') {
        draw();
      }
    }

    function setRecon4PanelView(enabled) {
      if (typeof enabled === 'undefined') {
        toggleRecon4PanelView();
        return;
      }
      isRecon4PanelView = !!enabled;
      reconOverlayMode = false;
      if (isRecon4PanelView) {
        maximizedQuad = null;
      }
      syncRecon4PanelUI();
      if (typeof updateViewPresetBarPosition === 'function') {
        updateViewPresetBarPosition();
      }
      if (typeof draw === 'function') {
        draw();
      }
    }

    function setReconOverlayMode(enabled) {
      if (typeof enabled === 'boolean') {
        reconOverlayMode = enabled;
      } else {
        if (isRecon4PanelView && reconOverlayMode) {
          reconOverlayMode = false;
          isRecon4PanelView = true;
        } else {
          reconOverlayMode = true;
          isRecon4PanelView = true;
          maximizedQuad = null;
        }
      }
      if (reconOverlayMode) {
        isRecon4PanelView = true;
        maximizedQuad = null;
      }
      syncRecon4PanelUI();
      syncReconKnnUI();

      if (typeof showToast === 'function') {
        showToast(
          reconOverlayMode
            ? '⧉ ABCD Overlay (A+C | B+D) enabled'
            : '⊞ ABCD 4-Panel split view restored'
        );
      }
      if (typeof updateViewPresetBarPosition === 'function') {
        updateViewPresetBarPosition();
      }
      if (typeof draw === 'function') {
        draw();
      }
    }

    function setReconKnn(enabled) {
      if (typeof enabled === 'boolean') {
        showReconKnn = enabled;
      } else {
        showReconKnn = !showReconKnn;
      }
      showKnnLines = showReconKnn;
      syncReconKnnUI();
      if (typeof showToast === 'function') {
        showToast(showReconKnn ? '⚡ k-NN Highlights & Rays enabled' : '⚡ k-NN Highlights hidden');
      }
      if (typeof draw === 'function') {
        draw();
      }
    }

    function syncReconKnnUI() {
      const btnTop = document.getElementById('btnToggleReconKnnTop');
      if (btnTop) {
        btnTop.classList.toggle('active', showReconKnn);
      }
      const btnSide = document.getElementById('btnToggleReconKnnSide');
      if (btnSide) {
        btnSide.classList.toggle('active', showReconKnn);
      }
    }

    window.switchDatasetSlot = switchDatasetSlot;
    window.clearDatasetSlot = clearDatasetSlot;
    window.saveSlotState = saveSlotState;
    window.loadSlotState = loadSlotState;
    window.updateMultiDatasetUI = updateMultiDatasetUI;
    window.setMultiDatasetEnabled = setMultiDatasetEnabled;
    window.populateDefaultMultiDatasets = populateDefaultMultiDatasets;
    window.toggleRecon4PanelView = toggleRecon4PanelView;
    window.setRecon4PanelView = setRecon4PanelView;
    window.syncRecon4PanelUI = syncRecon4PanelUI;
    window.setReconOverlayMode = setReconOverlayMode;
    window.syncReconOverlayUI = syncReconOverlayUI;
    window.setReconKnn = setReconKnn;
    window.syncReconKnnUI = syncReconKnnUI;
    window.reconInputCamera = reconInputCamera;
    window.reconOutputCamera = reconOutputCamera;
    window.reconHoveredTrainingIdx = reconHoveredTrainingIdx;
    window.reconLockedTrainingIdx = reconLockedTrainingIdx;
    window.showReconKnn = showReconKnn;
    Object.defineProperty(window, 'reconOverlayMode', {
      get: () => reconOverlayMode,
      set: (v) => { reconOverlayMode = !!v; },
      configurable: true
    });
    Object.defineProperty(window, 'plotDimX', {
      get: () => plotDimX,
      set: (v) => { plotDimX = v; },
      configurable: true
    });
    Object.defineProperty(window, 'plotDimY', {
      get: () => plotDimY,
      set: (v) => { plotDimY = v; },
      configurable: true
    });
    Object.defineProperty(window, 'plotDimZ', {
      get: () => plotDimZ,
      set: (v) => { plotDimZ = v; },
      configurable: true
    });
    window.getPlotCoords = getPlotCoords;
    window.clampPlottingDimensions = clampPlottingDimensions;
    window.setPlottingDimensions = setPlottingDimensions;
    window.updatePlottingDimSelectorsUI = updatePlottingDimSelectorsUI;
    window.setClusteringRlim = setClusteringRlim;
    window.setNoiseSigma = setNoiseSigma;
    window.datasetSlots = datasetSlots;
