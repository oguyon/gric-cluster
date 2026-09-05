/**
 * GRIC Simulator - main.js
 * Part of the GRIC Interactive Algorithm Simulator
 */

//  WASM ENGINE TOGGLE
    // =========================================================================

    function updateWasmBadge() {
      updateEngineModeUI();
    }

    function updateEngineModeUI() {
      const activeBadge = document.getElementById('activeEngineBadge');
      const banner = document.getElementById('engineModeBanner');
      const bannerIcon = document.getElementById('engineBannerIcon');
      const bannerTitle = document.getElementById('engineBannerTitle');
      const bannerSubtitle = document.getElementById('engineBannerSubtitle');
      const bannerToggle = document.getElementById('btnBannerToggleEngine');
      const btnPlay = document.getElementById('btnPlay');
      const btnStep = document.getElementById('btnStep');
      const btnWasm = document.getElementById('btnEngineWasm');
      const btnCli = document.getElementById('btnEngineCli');
      const badgeWasm = document.getElementById('badgeWasmStatus');
      const labelBackend = document.getElementById('statEngineBackend');
      const badgeNativeTmux = document.getElementById('badgeNativeTmux');

      if (engineMode === 'cli') {
        // Header Badge
        if (activeBadge) {
          activeBadge.textContent = '💻 Native Compiled C (gric-cluster)';
          activeBadge.style.background = 'rgba(74, 222, 128, 0.18)';
          activeBadge.style.color = '#4ade80';
          activeBadge.style.borderColor = 'rgba(74, 222, 128, 0.45)';
        }

        // Visual Banner
        if (banner) {
          banner.className = 'engine-mode-banner engine-cli';
          if (bannerIcon) bannerIcon.textContent = '💻';
          if (bannerTitle) {
            bannerTitle.textContent = 'Active Path: Native Compiled C Executable (gric-cluster)';
            bannerTitle.style.color = 'var(--accent-green)';
          }
          if (bannerSubtitle) {
            bannerSubtitle.textContent = 'Spawning multi-threaded native ELF binary on host CPU with OpenMP & AVX SIMD.';
          }
          if (bannerToggle) {
            bannerToggle.style.display = 'inline-block';
            bannerToggle.textContent = 'Switch to WASM ➔';
          }
        }

        // Toolbar Play / Step buttons
        if (btnPlay) {
          btnPlay.innerHTML = '▶ Run gric-cluster';
          btnPlay.classList.remove('primary');
          btnPlay.classList.add('btn-action');
          btnPlay.style.background = 'rgba(74, 222, 128, 0.25)';
          btnPlay.style.color = '#4ade80';
          btnPlay.style.borderColor = 'rgba(74, 222, 128, 0.5)';
          btnPlay.title = 'Run native compiled gric-cluster executable';
        }
        if (btnStep) {
          btnStep.disabled = true;
          btnStep.title = 'Step inspection is only available in WASM Interactive Simulation mode';
          btnStep.style.opacity = '0.4';
        }

        // Toggle-Slider in Toolbar Row 1
        const engineToggleSlider = document.getElementById('engineToggleSlider');
        if (engineToggleSlider) engineToggleSlider.classList.add('mode-cli');
        if (btnWasm) btnWasm.classList.remove('active');
        if (btnCli) btnCli.classList.add('active');

        if (badgeNativeTmux) badgeNativeTmux.style.display = 'inline-flex';

        // Resource Tracker
        if (badgeWasm) {
          badgeWasm.innerText = 'Native ELF';
          badgeWasm.style.background = 'rgba(74, 222, 128, 0.18)';
          badgeWasm.style.color = '#4ade80';
        }
        if (labelBackend) {
          labelBackend.innerText = 'Native Compiled C (OpenMP / AVX)';
        }
      } else {
        // WASM Mode
        if (activeBadge) {
          activeBadge.textContent = '⚡ In-Browser WASM Simulator';
          activeBadge.style.background = 'rgba(56, 189, 248, 0.18)';
          activeBadge.style.color = '#38bdf8';
          activeBadge.style.borderColor = 'rgba(56, 189, 248, 0.45)';
        }

        if (banner) {
          banner.className = 'engine-mode-banner engine-wasm';
          if (bannerIcon) bannerIcon.textContent = '⚡';
          if (bannerTitle) {
            bannerTitle.textContent = 'Active Path: In-Browser WebAssembly (WASM)';
            bannerTitle.style.color = 'var(--accent-blue)';
          }
          if (bannerSubtitle) {
            bannerSubtitle.textContent = 'Running frame-by-frame simulation inside browser VM with step-by-step HUD telemetry.';
          }
          if (bannerToggle) {
            bannerToggle.style.display = isDesktopBackend ? 'inline-block' : 'none';
            bannerToggle.textContent = 'Switch to Native CLI ➔';
          }
        }

        if (btnPlay) {
          btnPlay.innerHTML = isRunning ? '⏸ Pause' : '▶ Cluster';
          btnPlay.classList.add('primary');
          btnPlay.classList.remove('btn-action');
          btnPlay.style.background = '';
          btnPlay.style.color = '';
          btnPlay.style.borderColor = '';
          btnPlay.title = 'Run / Pause Clustering';
        }
        if (btnStep) {
          btnStep.disabled = false;
          btnStep.title = 'Step Ingest Single Frame';
          btnStep.style.opacity = '1.0';
        }

        const engineToggleSlider = document.getElementById('engineToggleSlider');
        if (engineToggleSlider) engineToggleSlider.classList.remove('mode-cli');
        if (btnWasm) btnWasm.classList.add('active');
        if (badgeNativeTmux) badgeNativeTmux.style.display = 'none';
        if (btnWasm) btnWasm.classList.add('active');
        if (btnCli) btnCli.classList.remove('active');

        // Check if Native CLI is supported on current device / environment
        if (btnCli) {
          if (!DesktopBridge.isNativeSupported()) {
            btnCli.style.opacity = '0.35';
            btnCli.style.cursor = 'not-allowed';
            btnCli.title = DesktopBridge.isMobileDevice()
              ? 'Native CLI is not supported on mobile cell phones (WASM active)'
              : 'Native CLI requires a local desktop gric-server';
          } else {
            btnCli.style.opacity = '1.0';
            btnCli.style.cursor = 'pointer';
            btnCli.title = 'Switch to host Native Compiled C (gric-cluster)';
          }
        }

        const btnNativeQuery = document.getElementById('btnRunNativeReconQuery');
        if (btnNativeQuery) {
          if (DesktopBridge.isNativeSupported()) {
            btnNativeQuery.style.opacity = '1.0';
            btnNativeQuery.style.cursor = 'pointer';
            btnNativeQuery.title = 'Run native gric-knn -query C using cluster anchors of A';
          } else {
            btnNativeQuery.style.opacity = '0.5';
            btnNativeQuery.style.cursor = 'not-allowed';
            btnNativeQuery.title = 'Native CLI requires a local desktop gric-server';
          }
        }

        if (bannerToggle) {
          bannerToggle.style.display = DesktopBridge.isNativeSupported() ? 'inline-block' : 'none';
          bannerToggle.textContent = 'Switch to Native CLI ➔';
        }

        if (badgeWasm) {
          if (isExplainMode) {
            badgeWasm.innerText = 'WASM+Trace';
            badgeWasm.style.background = 'rgba(251, 191, 36, 0.15)';
            badgeWasm.style.color = '#fbbf24';
          } else {
            badgeWasm.innerText = 'WASM';
            badgeWasm.style.background = 'rgba(56, 189, 248, 0.18)';
            badgeWasm.style.color = '#38bdf8';
          }
        }
        if (labelBackend) {
          if (isExplainMode) {
            labelBackend.innerText = 'C/WebAssembly + Trace';
          } else {
            labelBackend.innerText = 'C/WebAssembly (SIMD)';
          }
        }
      }
    }

    function toggleWasmEngine() {
      if (DesktopBridge.isNativeSupported()) {
        setEngineMode(engineMode === 'wasm' ? 'cli' : 'wasm');
      } else if (DesktopBridge.isMobileDevice()) {
        showToast('📱 Cell Phone: Native CLI is disabled (In-Browser WASM active)');
      } else {
        showToast('🌐 Web Mode: Native CLI requires a local desktop gric-server');
      }
    }

// =========================================================================
    //  SYNCHRONIZED ANIMATION & COMPUTE ENGINE (60 FPS)
    // =========================================================================
    function startSimulation() {
      if (isRunning) pauseSimulation();
      if (playTimer) {
        cancelAnimationFrame(playTimer);
        clearTimeout(playTimer);
        clearInterval(playTimer);
        playTimer = null;
      }
      if (isAddPointMode) setAddPointMode(false);

      if (!benchmarkDataset || benchmarkDataset.length === 0) {
        stageDataset();
      }

      if (typeof dataMode !== 'undefined' && dataMode === 'image') {
        if (typeof inspectedImageFrameIdx !== 'undefined') {
          inspectedImageFrameIdx = -1;
        }
        if (typeof inspectedClusterId !== 'undefined') {
          inspectedClusterId = -1;
        }
      }

      if (!hasMoreFrames()) {
        resetClustering(true);
        currentFrameIdx = 0;
      }

      if (useWasm && GricWasm.isLoaded()) {
        const params = GricWasm.buildParamsFromState();
        if (!wasmSessionActive || !GricWasm.isReady() ||
            (GricWasm.isConfigChanged && GricWasm.isConfigChanged(params))) {
          wasmSessionActive = GricWasm.init(params);
          updateWasmBadge();
        }
      }

      isRunning = true;
      sessionStartTime = performance.now();
      sessionStartFrames = totalFrames;
      sessionElapsedMs = 0;
      sessionIsActive = true;
      sessionAvgFps = 0.0;

      const btn = document.getElementById('btnPlay');
      if (btn) {
        btn.innerText = "❚❚ Pause";
        btn.classList.add('danger');
        btn.classList.remove('primary');
      }

      // Mode 1: Instant Run to Completion (playSpeed === 0)
      if (playSpeed === 0) {
        runClusteringToCompletion();
        return;
      }

      // Mode 2: Paced Playback (playSpeed > 0)
      if (playSpeed > 0) {
        runPacedSimulation();
        return;
      }

      // Mode 3: Decoupled High-Throughput Streaming Preview (playSpeed <= -1)
      runDecoupledSimulation();
    }

    function resetComputeAllButton() {
      const btn = document.getElementById('btnComputeAll');
      if (btn) {
        btn.innerHTML = '⚡ Compute All';
        btn.style.background = 'rgba(34, 197, 94, 0.15)';
        btn.style.color = '#4ade80';
        btn.style.borderColor = 'rgba(34, 197, 94, 0.4)';
        btn.classList.remove('danger');
      }
    }

    function setComputeAllButtonActive() {
      const btn = document.getElementById('btnComputeAll');
      if (btn) {
        btn.innerHTML = '⏹ Stop';
        btn.style.background = 'rgba(239, 68, 68, 0.25)';
        btn.style.color = '#f87171';
        btn.style.borderColor = 'rgba(239, 68, 68, 0.5)';
        btn.classList.add('danger');
      }
    }

    function abortComputeAll() {
      if (!isComputeAllRunning) return;
      abortComputeAllRequested = true;
      isComputeAllRunning = false;
      if (computeAllTimer) {
        clearTimeout(computeAllTimer);
        cancelAnimationFrame(computeAllTimer);
        computeAllTimer = null;
      }

      if (useWasm && wasmSessionActive && GricWasm.isReady()) {
        const snapshot = GricWasm.syncState(true);
        if (snapshot) {
          GricWasm.applyToJsState(snapshot);
        }
      }

      pauseSimulation();
      resetComputeAllButton();
      updateUI();
      draw();
      showToast(`⏹ Compute All stopped at frame ${totalFrames.toLocaleString()}`);
    }

    function runClusteringToCompletion() {
      if (isComputeAllRunning) {
        abortComputeAll();
        return;
      }

      if (isRunning) pauseSimulation();
      if (playTimer) {
        cancelAnimationFrame(playTimer);
        clearTimeout(playTimer);
        clearInterval(playTimer);
        playTimer = null;
      }

      if (!benchmarkDataset || benchmarkDataset.length === 0) {
        stageDataset();
      }

      if (!hasMoreFrames()) {
        resetClustering(true);
        currentFrameIdx = 0;
      }

      if (useWasm && GricWasm.isLoaded()) {
        const params = GricWasm.buildParamsFromState();
        if (!wasmSessionActive || !GricWasm.isReady() ||
            (GricWasm.isConfigChanged && GricWasm.isConfigChanged(params))) {
          wasmSessionActive = GricWasm.init(params);
          updateWasmBadge();
        }
      }

      isComputeAllRunning = true;
      abortComputeAllRequested = false;
      isRunning = true;
      knnResults = null;
      setComputeAllButtonActive();

      const btnPlay = document.getElementById('btnPlay');
      if (btnPlay) {
        btnPlay.innerText = "❚❚ Pause";
        btnPlay.classList.add('danger');
        btnPlay.classList.remove('primary');
      }

      const tStart = performance.now();
      const startFrames = totalFrames;
      const isImg = (typeof dataMode !== 'undefined' && dataMode === 'image');
      const totalDatasetCount = benchmarkDataset ? benchmarkDataset.length : 0;

      function computeSlice() {
        if (!isComputeAllRunning || abortComputeAllRequested) {
          isComputeAllRunning = false;
          resetComputeAllButton();
          pauseSimulation();
          return;
        }

        const sliceStart = performance.now();
        const maxSliceMs = 25; // Keep UI responsive at ~40 FPS

        if (useWasm && wasmSessionActive && GricWasm.isReady() && !isImg && !useTiles) {
          // Native WASM batch compute in chunks of 5000 frames
          const N = benchmarkDataset ? benchmarkDataset.length : 0;
          const remaining = N - currentFrameIdx;
          if (remaining > 0) {
            const batchSize = Math.min(remaining, 5000);
            const d = currentDim;
            const flatCoords = new Float64Array(batchSize * d);
            for (let i = 0; i < batchSize; i++) {
              const pt = benchmarkDataset[currentFrameIdx + i];
              if (pt.coords && pt.coords.length >= d) {
                for (let dimIdx = 0; dimIdx < d; dimIdx++) {
                  flatCoords[i * d + dimIdx] = pt.coords[dimIdx];
                }
              } else {
                flatCoords[i * d] = pt.x;
                flatCoords[i * d + 1] = pt.y;
                if (d >= 3) flatCoords[i * d + 2] = pt.z || 0.0;
              }
            }

            const outAssignments = new Int32Array(batchSize);
            const processed = GricWasm.processBatch(
              flatCoords, batchSize, d, outAssignments
            );

            for (let i = 0; i < processed; i++) {
              const frameIdx = currentFrameIdx + i;
              const assigned = outAssignments[i];
              const cId = (assigned >= 0)
                ? assigned
                : Math.max(0, GricWasm.getNumClusters() - 1);
              if (frameIdx < pastSamples.length) {
                pastSamples[frameIdx].clusterId = cId;
              } else if (pastSamples.length < sampleBufferCap) {
                const pt = benchmarkDataset[frameIdx];
                const s = {
                  x: pt.x, y: pt.y, z: pt.z || 0,
                  frameIndex: frameIdx,
                  clusterId: cId
                };
                if (pt.coords) s.coords = pt.coords;
                pastSamples.push(s);
              }
              if (frameIdx < benchmarkDataset.length) {
                benchmarkDataset[frameIdx].clusterId = cId;
              }
            }
            currentFrameIdx += processed;
            totalFrames += processed;
          }
        } else {
          // Standard / Image pipeline in time-budgeted slice
          while (hasMoreFrames() && !abortComputeAllRequested && (performance.now() - sliceStart < maxSliceMs)) {
            stepNextFrame(true);
          }
        }

        // Live progress telemetry update during Compute All
        if (totalDatasetCount > 0) {
          const pct = Math.min(100, Math.max(0, (totalFrames / totalDatasetCount) * 100));
          const fill = document.getElementById('progressFill');
          if (fill) fill.style.width = `${pct.toFixed(1)}%`;
          const fc = document.getElementById('frameCounter');
          if (fc) fc.textContent = `${totalFrames} / ${totalDatasetCount} (${pct.toFixed(1)}%)`;
          const cb = document.getElementById('clusterBadge');
          if (cb) cb.textContent = `${clusters.length} clusters`;
        }

        if (hasMoreFrames() && !abortComputeAllRequested && isComputeAllRunning) {
          computeAllTimer = setTimeout(computeSlice, 0);
        } else {
          // Completion
          const tEnd = performance.now();
          const durationMs = Math.max(0.1, tEnd - tStart);
          const framesComputed = totalFrames - startFrames;
          const ptsPerSec = durationMs > 0.001
            ? (framesComputed / (durationMs / 1000.0))
            : 0.0;

          if (useWasm && wasmSessionActive && GricWasm.isReady()) {
            const snapshot = GricWasm.syncState(true);
            if (snapshot) {
              GricWasm.applyToJsState(snapshot);
            }
          }

          isComputeAllRunning = false;
          resetComputeAllButton();
          pauseSimulation();

          sessionElapsedMs = durationMs;
          sessionAvgFps = ptsPerSec;
          currentFps = ptsPerSec;
          currentCpuLoadPct = 100.0;
          lastComputeTimeMs = durationMs;
          avgComputeTimeMs = framesComputed > 0
            ? (durationMs / framesComputed)
            : 0.0;
          sessionIsActive = false;

          if (typeof recordFrameTelemetry === 'function') {
            recordFrameTelemetry(durationMs, framesComputed, distSampleCluster);
          }

          updateUI();
          draw();

          if (typeof showToast === 'function') {
            const rateStr = Number(ptsPerSec.toFixed(0)).toLocaleString();
            const msg = `⚡ Clustered ${framesComputed.toLocaleString()} pts in ` +
              `${durationMs.toFixed(1)} ms (${rateStr} pts/sec)`;
            showToast(msg);
          }
        }
      }

      computeAllTimer = setTimeout(computeSlice, 0);
    }

    function runDecoupledSimulation() {
      let lastTickTime = performance.now();
      let lastTelemetryFrames = totalFrames;
      let lastTelemetryDists = distSampleCluster;

      const isImg = (typeof dataMode !== 'undefined' && dataMode === 'image');
      const chunkSize = isImg ? 16 : 800;
      const sliceLimitMs = isImg ? 8 : 12;

      // 1. Decoupled Compute Pump (runs at maximum processor speed)
      function computePump() {
        if (!isRunning) return;

        const sliceStart = performance.now();
        let count = 0;
        while (hasMoreFrames()
          && count < chunkSize
          && (performance.now() - sliceStart < sliceLimitMs))
        {
          stepNextFrame(true);
          count++;
        }

        if (!hasMoreFrames()) {
          pauseSimulation();
          return;
        }

        if (isRunning) {
          computePumpTimer = setTimeout(computePump, 0);
        }
      }

      // 2. Decoupled Render Pump (smooth 60 FPS viewport refresh)
      function renderPump() {
        if (!isRunning && !computePumpTimer) return;

        const now = performance.now();
        const delta = Math.max(0, now - lastTickTime);
        lastTickTime = now;

        if (useWasm && wasmSessionActive && GricWasm.isReady()) {
          const needDcc = (typeof currentTab !== 'undefined' &&
            (currentTab === 'tm' || currentTab === 'dist'));
          const snapshot = GricWasm.syncState(needDcc);
          if (snapshot) {
            GricWasm.applyToJsState(snapshot);
          }
        }

        const framesDelta = totalFrames - lastTelemetryFrames;
        const timeDelta = Math.max(0.001, delta);
        if (framesDelta > 0 && typeof recordFrameTelemetry === 'function') {
          const distsDelta = Math.max(0, distSampleCluster - lastTelemetryDists);
          recordFrameTelemetry(timeDelta, framesDelta, distsDelta);
          lastTelemetryFrames = totalFrames;
          lastTelemetryDists = distSampleCluster;
        }

        updateUI();
        draw();

        if (isRunning) {
          playTimer = requestAnimationFrame(renderPump);
        }
      }

      computePumpTimer = setTimeout(computePump, 0);
      playTimer = requestAnimationFrame(renderPump);
    }

    function runPacedSimulation() {
      let lastTickTime = performance.now();
      let lastTelemetryFrames = totalFrames;
      let lastTelemetryDists = distSampleCluster;
      let frameAccumulator = 0;

      function pacedTick() {
        if (!isRunning) return;

        const now = performance.now();
        const delta = Math.max(0, now - lastTickTime);
        lastTickTime = now;

        frameAccumulator += delta;
        const interval = playSpeed;
        let framesToProcess = Math.floor(frameAccumulator / interval);
        if (framesToProcess > 0) {
          frameAccumulator -= framesToProcess * interval;
          framesToProcess = Math.min(framesToProcess, 100);
          for (let f = 0; f < framesToProcess; f++) {
            if (!hasMoreFrames()) break;
            stepNextFrame(true);
          }
        }

        if (useWasm && wasmSessionActive && GricWasm.isReady()) {
          const needDcc = (typeof currentTab !== 'undefined' &&
            (currentTab === 'tm' || currentTab === 'dist'));
          const snapshot = GricWasm.syncState(needDcc);
          if (snapshot) {
            GricWasm.applyToJsState(snapshot);
          }
        }

        const framesDelta = totalFrames - lastTelemetryFrames;
        const timeDelta = Math.max(0.001, delta);
        if (framesDelta > 0 && typeof recordFrameTelemetry === 'function') {
          const distsDelta = Math.max(0, distSampleCluster - lastTelemetryDists);
          recordFrameTelemetry(timeDelta, framesDelta, distsDelta);
          lastTelemetryFrames = totalFrames;
          lastTelemetryDists = distSampleCluster;
        }

        updateUI();
        draw();

        if (!hasMoreFrames()) {
          pauseSimulation();
          return;
        }

        if (isRunning) {
          playTimer = requestAnimationFrame(pacedTick);
        }
      }

      playTimer = requestAnimationFrame(pacedTick);
    }

    function pauseSimulation() {
      isRunning = false;
      if (isComputeAllRunning) {
        abortComputeAllRequested = true;
        isComputeAllRunning = false;
        if (computeAllTimer) {
          clearTimeout(computeAllTimer);
          cancelAnimationFrame(computeAllTimer);
          computeAllTimer = null;
        }
        resetComputeAllButton();
      }
      if (computePumpTimer) {
        clearTimeout(computePumpTimer);
        computePumpTimer = null;
      }
      if (playTimer) {
        cancelAnimationFrame(playTimer);
        clearTimeout(playTimer);
        clearInterval(playTimer);
        playTimer = null;
      }
      if (sessionIsActive) {
        sessionElapsedMs = Math.max(0.0001, performance.now() - sessionStartTime);
        const framesClustered = totalFrames - sessionStartFrames;
        sessionAvgFps = sessionElapsedMs > 0.001
          ? (framesClustered / (sessionElapsedMs / 1000.0))
          : 0.0;
        sessionIsActive = false;
      }

      const btn = document.getElementById('btnPlay');
      if (btn) {
        btn.innerText = "► Cluster";
        btn.classList.remove('danger');
        btn.classList.add('primary');
      }
      if (typeof GricWasmWorker !== 'undefined' && GricWasmWorker.isBusy()) {
        GricWasmWorker.pauseBatch();
      }

      // Final WASM sync
      if (useWasm && wasmSessionActive && GricWasm.isReady()) {
        const snapshot = GricWasm.syncState(true);
        if (snapshot) {
          GricWasm.applyToJsState(snapshot);
        }
      }
      if (typeof clearActiveFrameEvaluations === 'function') {
        clearActiveFrameEvaluations(true);
      }

      if (typeof hasMoreFrames === 'function' && !hasMoreFrames() && usePass2Nearest) {
        if (typeof runSecondPassClustering === 'function') {
          runSecondPassClustering();
        }
      }

      updateUI();
      draw();
      draw();
    }

    function setAddPointMode(enabled) {
      isAddPointMode = enabled;
      const btn = document.getElementById('btnAddPoint');
      const btnSide = document.getElementById('btnAddPointSide');
      if (btn) btn.classList.toggle('toggle-active', isAddPointMode);
      if (btnSide) {
        btnSide.classList.toggle('toggle-active', isAddPointMode);
        btnSide.innerText = isAddPointMode ? "✓ Point Injection Active (Click Canvas)" : "＋ Toggle Add Point Mode";
      }
      if (isAddPointMode) {
        if (isRunning) pauseSimulation();
        canvas.classList.add('crosshair');
        canvas.classList.remove('orbit-cursor');
        canvas.classList.remove('grabbing');
      } else {
        canvas.classList.remove('crosshair');
      }
    }

    function setExplainMode(enabled) {
      isExplainMode = enabled;
      const btn = document.getElementById('btnExplain');
      if (isExplainMode) {
        if (btn) btn.classList.add('toggle-active');
        setTab('narrative');

        // Automatically expand the trace panel so the Decision Narrative is visible
        const card = document.getElementById('cardTrace');
        if (card && card.classList.contains('collapsed')) {
          if (typeof togglePanelCollapse === 'function') {
            togglePanelCollapse('cardTrace');
          } else {
            card.classList.remove('collapsed');
          }
        }

        // Enable C-side trace buffer for WASM explain
        if (typeof GricWasm !== 'undefined' && typeof GricWasm.setTrace === 'function') {
          GricWasm.setTrace(true);
        }
      } else {
        if (btn) btn.classList.remove('toggle-active');
        currentExplanation = [];
        // Disable C-side trace buffer
        if (typeof GricWasm !== 'undefined' && typeof GricWasm.setTrace === 'function') {
          GricWasm.setTrace(false);
        }
      }
      updateWasmBadge();
      updateUI();
    }

    // =========================================================================
    //  8. FILE UPLOAD & 2D / 3D PARSING
    // =========================================================================

    function parseCoordinateFile(text, filename = "dataset") {
      const lines = text.split(/\r?\n/);
      const rawPoints = [];
      let detected3D = false;
      let detectedDim = 2;

      for (let line of lines) {
        line = line.trim();
        if (!line || line.startsWith('#')) continue;

        const tokens = line.split(/[,\s\t]+/).filter(t => t.length > 0);
        if (tokens.length >= 3) {
          const x = parseFloat(tokens[0]);
          const y = parseFloat(tokens[1]);
          const z = parseFloat(tokens[2]);
          if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            const pt = { x, y, z };
            if (tokens.length > 3) {
              const coords = new Float64Array(tokens.length);
              coords[0] = x;
              coords[1] = y;
              coords[2] = z;
              for (let d = 3; d < tokens.length; d++) {
                coords[d] = parseFloat(tokens[d]) || 0.0;
              }
              pt.coords = coords;
              if (tokens.length > detectedDim) detectedDim = tokens.length;
            } else if (detectedDim < 3) {
              detectedDim = 3;
            }
            rawPoints.push(pt);
            detected3D = true;
          }
        } else if (tokens.length >= 2) {
          const x = parseFloat(tokens[0]);
          const y = parseFloat(tokens[1]);
          if (!isNaN(x) && !isNaN(y)) {
            rawPoints.push({ x, y, z: 0.0 });
          }
        }
      }

      if (rawPoints.length === 0) {
        alert(
          "Error: No valid coordinates found in file. Expected 'x y [z...]' per line."
        );
        return;
      }

      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      let minZ = Infinity, maxZ = -Infinity;

      for (const p of rawPoints) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
        if (p.z < minZ) minZ = p.z;
        if (p.z > maxZ) maxZ = p.z;
      }

      const spanX = maxX - minX || 1.0;
      const spanY = maxY - minY || 1.0;
      const spanZ = detected3D ? (maxZ - minZ || 1.0) : 1.0;
      const maxSpan = Math.max(spanX, spanY, detected3D ? spanZ : 0);
      const midX = (minX + maxX) / 2;
      const midY = (minY + maxY) / 2;
      const midZ = (minZ + maxZ) / 2;

      rawBenchmarkDataset = rawPoints.map(p => {
        const item = {
          x: ((p.x - midX) / maxSpan) * 1.76,
          y: ((p.y - midY) / maxSpan) * 1.76,
          z: detected3D ? (((p.z - midZ) / maxSpan) * 1.76) : 0.0
        };
        if (p.coords) {
          const c = new Float64Array(p.coords.length);
          c[0] = item.x;
          c[1] = item.y;
          c[2] = item.z;
          for (let d = 3; d < p.coords.length; d++) {
            c[d] = p.coords[d];
          }
          item.coords = c;
        }
        return item;
      });

      currentDim = detectedDim > 3 ? detectedDim : (detected3D ? 3 : 2);
      if (currentDim === 32) {
        if (typeof setClusteringRlim === 'function') {
          setClusteringRlim(1.0, false);
        } else {
          rlim = 1.0;
        }
        if (typeof setNoiseSigma === 'function') {
          setNoiseSigma(0.005, false);
        } else {
          noiseSigma = 0.005;
        }
      }
      applyNoiseToDataset();

      if (currentDim === 2) {
        maximizedQuad = null;
      }
      const dimLabel = currentDim > 3 ? currentDim + 'D' : (detected3D ? '3D' : '2D');
      BENCHMARK_DESCS["custom"] =
        `<b>Custom File (${filename})</b>: ${benchmarkDataset.length} ` +
        `${dimLabel} frames loaded from upload.`;
      const selMain = document.getElementById('selectBenchmark');
      if (selMain) selMain.value = 'custom';
      const selSlot = document.getElementById(`selectBenchmark_${activeDatasetSlot}`);
      if (selSlot) selSlot.value = 'custom';
      const selSide = document.getElementById('selectBenchmarkSide');
      if (selSide) selSide.value = 'custom';
      const descEl = document.getElementById('benchmarkDesc');
      if (descEl) descEl.innerHTML = BENCHMARK_DESCS["custom"];

      /* Store the original filename on the active slot so that native CLI jobs
       * (e.g. gric-knn -query C) can reference the actual workspace file directly
       * without needing to re-stage the in-memory point data. */
      if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
        if (!datasetSlots[activeDatasetSlot].stagedDatasetInfo) {
          datasetSlots[activeDatasetSlot].stagedDatasetInfo = {};
        }
        datasetSlots[activeDatasetSlot].stagedDatasetInfo.name = filename;
      }

      resetSimulation();
      resetView();
      updateDatasetStatusBadge();
      updateUI();
    }

    DATASET_SLOTS.forEach(sId => {
      const fileIn = document.getElementById(`fileUpload_${sId}`);
      if (fileIn) {
        fileIn.addEventListener('change', (e) => {
          const file = e.target.files[0];
          if (!file) return;
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
          const reader = new FileReader();
          reader.onload = (evt) => parseCoordinateFile(evt.target.result, file.name);
          reader.readAsText(file);
          fileIn.value = '';
        });
      }
      const btnUp = document.getElementById(`btnUpload_${sId}`);
      if (btnUp && fileIn) {
        btnUp.addEventListener('click', (e) => {
          e.stopPropagation();
          fileIn.click();
        });
      }
    });

    const fileInput = document.getElementById('fileUpload');
    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => parseCoordinateFile(evt.target.result, file.name);
        reader.readAsText(file);
        fileInput.value = '';
      });
    }

    const btnUploadLegacy = document.getElementById('btnUpload');
    if (btnUploadLegacy && fileInput) {
      btnUploadLegacy.addEventListener('click', () => fileInput.click());
    }

    const canvasBox = document.getElementById('canvasBox');
    const dropzone = document.getElementById('dropzone');

    ['dragenter', 'dragover'].forEach(eventName => {
      canvasBox.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.style.display = 'flex';
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      canvasBox.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.style.display = 'none';
      });
    });

    canvasBox.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const reader = new FileReader();
        reader.onload = (evt) => parseCoordinateFile(evt.target.result, files[0].name);
        reader.readAsText(files[0]);
      }
    });

    // =========================================================================
    //  9. MOUSE, ORBIT, PAN/ZOOM & INTERACTION LISTENERS
    // =========================================================================

    function getQuadrantAt(clientX, clientY) {
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        const rect = canvas.getBoundingClientRect();
        const px = clientX - rect.left;
        const py = clientY - rect.top;
        const W = rect.width;
        const H = rect.height;
        if (typeof reconOverlayMode !== 'undefined' && reconOverlayMode) {
          return (px < W / 2) ? 0 : 1;
        }
        if (px < W / 2 && py < H / 2) return 0;
        if (px >= W / 2 && py < H / 2) return 1;
        if (px < W / 2 && py >= H / 2) return 2;
        return 3;
      }
      if (currentDim === 2 && (typeof dataMode === 'undefined' || dataMode !== 'image')) {
        return 2;
      }
      if (maximizedQuad !== null) {
        return maximizedQuad;
      }

      const rect = canvas.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      const W = rect.width;
      const H = rect.height;

      if (px < W / 2 && py < H / 2) return 0; // Along X / Top-Left
      if (px >= W / 2 && py < H / 2) return 1; // Along Y / Top-Right
      if (px < W / 2 && py >= H / 2) return 2; // Along Z / Bottom-Left
      return 3; // Custom 3D / Bottom-Right
    }

    let mouseDownTime = 0;
    let mouseDownClientX = 0;
    let mouseDownClientY = 0;

    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const W = rect.width;
      const H = rect.height;
      const qIdx = getQuadrantAt(e.clientX, e.clientY);
      const qRect = getQuadRect(qIdx, W, H);

      mouseDownTime = performance.now();
      mouseDownClientX = e.clientX;
      mouseDownClientY = e.clientY;

      // Check if clicking Maximize / Restore in image mode or 3D mode
      if (dataMode === 'image') {
        if (typeof handleImageModeClick === 'function') {
          const handled = handleImageModeClick(px, py, qIdx, W, H);
          if (handled) {
            syncImageQuadUI();
            return;
          }
        }
      } else if (currentDim >= 3) {
        if (px >= qRect.x + qRect.w - 30 && px <= qRect.x + qRect.w - 4 &&
            py >= qRect.y && py <= qRect.y + 24) {
          maximizedQuad = (maximizedQuad === qIdx) ? null : qIdx;
          syncImageQuadUI();
          draw();
          return;
        }
      }

      // Check if clicking Corner Zoom Box to reset zoom & pan to 1:1
      const zRect = (typeof viewportZoomBoxRects !== 'undefined')
        ? viewportZoomBoxRects[qIdx]
        : null;
      if (showViewportHUD && zRect &&
          px >= zRect.x && px <= zRect.x + zRect.w &&
          py >= zRect.y && py <= zRect.y + zRect.h) {
        if (quadViews && quadViews[qIdx]) {
          quadViews[qIdx].zoom = 1.0;
          quadViews[qIdx].panX = 0;
          quadViews[qIdx].panY = 0;
        }
        if (qIdx === 3 && currentDim >= 3 && typeof orbitCamera !== 'undefined') {
          orbitCamera.zoom = 1.0;
          orbitCamera.panX = 0;
          orbitCamera.panY = 0;
        }
        if (typeof updateZoomBadge === 'function') updateZoomBadge();
        draw();
        return;
      }

      // Point Injection Mode
      if (isAddPointMode) {
        if (isRunning) pauseSimulation();
        const m = mapQuadToMetric(px, py, qIdx, qRect);
        let injX = 0, injY = 0, injZ = 0;

        if (qIdx === 0) {
          // Along X: horizontal is Y, vertical is Z
          injY = m.u;
          injZ = m.v;
          injX = currentFrame ? currentFrame.x : 0.0;
        } else if (qIdx === 1) {
          // Along Y: horizontal is X, vertical is Z
          injX = m.u;
          injZ = m.v;
          injY = currentFrame ? currentFrame.y : 0.0;
        } else if (qIdx === 2) {
          // Along Z / 2D: horizontal is X, vertical is Y
          injX = m.u;
          injY = m.v;
          injZ = currentFrame ? currentFrame.z : 0.0;
        } else if (qIdx === 3) {
          // Custom 3D: inject at projected point with depth=0
          const cosT = Math.cos(orbitCamera.azimuth), sinT = Math.sin(orbitCamera.azimuth);
          injX = m.u * cosT;
          injY = m.u * sinT;
          injZ = m.v;
        }

        clusterFrame(injX, injY, injZ);
        showToast(`Injected: (${injX.toFixed(3)}, ${injY.toFixed(3)}, ${injZ.toFixed(3)})`);
        return;
      }

      // Normal Navigation / Drag Mode
      isDragging = true;
      activeDragQuad = qIdx;
      dragStartX = e.clientX;
      dragStartY = e.clientY;

      const isRecon = (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView);
      if (isRecon) {
        // Quad 0 (A) and Quad 2 (C) -> Input Space
        // Quad 1 (B) and Quad 3 (D) -> Output Space
        const isInputSpace = (qIdx === 0 || qIdx === 2);
        let spaceIs3D = false;
        if (typeof datasetSlots !== 'undefined') {
          if (isInputSpace) {
            spaceIs3D = (datasetSlots.A && datasetSlots.A.currentDim >= 3) ||
                        (datasetSlots.C && datasetSlots.C.currentDim >= 3);
          } else {
            spaceIs3D = (datasetSlots.B && datasetSlots.B.currentDim >= 3) ||
                        (datasetSlots.D && datasetSlots.D.currentDim >= 3);
          }
        }
        dragMode = (spaceIs3D && !e.shiftKey) ? 'orbit' : 'pan';
        canvas.classList.add('grabbing');
      } else {
        const is3DTarget = ((qIdx === 3 || maximizedQuad === 3) && currentDim >= 3);
        if (is3DTarget) {
          dragMode = e.shiftKey ? 'pan' : 'orbit';
          canvas.classList.add('grabbing');
        } else {
          dragMode = 'pan';
          canvas.classList.add('grabbing');
        }
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging || isAddPointMode) return;

      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      dragStartX = e.clientX;
      dragStartY = e.clientY;

      if (dataMode === 'image') {
        if (activeDragQuad === 2 || maximizedQuad === 2) {
          if (imageQ2ViewMode === 'knn') {
            imageKnnScrollY = Math.max(0, (imageKnnScrollY || 0) - dy);
          } else {
            imageMembersScrollY = Math.max(0, (imageMembersScrollY || 0) - dy);
          }
          draw();
        } else if (activeDragQuad === 3 || maximizedQuad === 3) {
          imageClustersScrollY = Math.max(0, (imageClustersScrollY || 0) - dy);
          draw();
        }
        return;
      }

      // --- 4-Panel Reconstruction View Drag Handling ---
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        const isInputSpace = (activeDragQuad === 0 || activeDragQuad === 2);
        const targetViews = isInputSpace
          ? [quadViews[0], quadViews[2]]
          : [quadViews[1], quadViews[3]];
        const targetCamera = isInputSpace
          ? (typeof reconInputCamera !== 'undefined' ? reconInputCamera : orbitCamera)
          : (typeof reconOutputCamera !== 'undefined' ? reconOutputCamera : orbitCamera);

        if (dragMode === 'orbit') {
          targetCamera.azimuth += dx * 0.008;
          targetCamera.elevation = Math.max(
            -1.52, Math.min(1.52, targetCamera.elevation - dy * 0.008)
          );
          draw();
        } else {
          const rect = canvas.getBoundingClientRect();
          const isOverlay = (typeof reconOverlayMode !== 'undefined' && reconOverlayMode);
          const qRect = isOverlay
            ? { x: 0, y: 0, w: rect.width / 2, h: rect.height }
            : { x: 0, y: 0, w: rect.width / 2, h: rect.height / 2 };
          const scale = (Math.min(qRect.w, qRect.h) / 2.35) * (targetViews[0].zoom || 1.0);
          if (scale > 0) {
            targetViews.forEach(v => {
              v.panX -= dx / scale;
              v.panY += dy / scale;
            });
            targetCamera.panX = targetViews[0].panX;
            targetCamera.panY = targetViews[0].panY;
            draw();
          }
        }
        return;
      }

      const is3DTarget = (activeDragQuad === 3 || maximizedQuad === 3) && currentDim >= 3;

      if (is3DTarget) {
        if (e.shiftKey || dragMode === 'pan') {
          // SHIFT + Drag: Pan 3D subpanel
          const targetQuad = maximizedQuad !== null ? maximizedQuad : 3;
          const rect = canvas.getBoundingClientRect();
          const qRect = getQuadRect(targetQuad, rect.width, rect.height);
          const scale = getQuadScale(targetQuad, qRect);

          const v = quadViews[targetQuad];
          v.panX -= dx / scale;
          v.panY += dy / scale;
          draw();
        } else {
          // Normal Drag: Orbit camera in 3D (Azimuth and Elevation)
          orbitCamera.azimuth += dx * 0.008;
          orbitCamera.elevation = Math.max(-1.52, Math.min(1.52, orbitCamera.elevation - dy * 0.008));
          draw();
        }
      } else if (dragMode === 'pan') {
        // Pan 2D Sub-viewport
        const targetQuad = (currentDim === 2) ? 2 : activeDragQuad;
        const rect = canvas.getBoundingClientRect();
        const qRect = getQuadRect(targetQuad, rect.width, rect.height);
        const scale = getQuadScale(targetQuad, qRect);

        const v = quadViews[targetQuad];
        if (v && scale > 0) {
          v.panX -= dx / scale;
          v.panY += dy / scale;
          draw();
        }
      }
    });

    window.addEventListener('mouseup', (e) => {
      if (isDragging) {
        isDragging = false;
        dragMode = null;
        canvas.classList.remove('grabbing');
      }

      // Check if mouseup was a click (not a drag) on the canvas
      const distFromDown = Math.hypot(e.clientX - mouseDownClientX, e.clientY - mouseDownClientY);
      const clickDuration = performance.now() - mouseDownTime;
      if (distFromDown < 6 && clickDuration < 450 && !isAddPointMode) {
        if (e.target === canvas) {
          if (dataMode === 'image') {
            const rect = canvas.getBoundingClientRect();
            const px = e.clientX - rect.left;
            const py = e.clientY - rect.top;
            const qIdx = getQuadrantAt(e.clientX, e.clientY);
            if (typeof handleImageModeClick === 'function') {
              const handled = handleImageModeClick(px, py, qIdx, rect.width, rect.height);
              if (handled) {
                draw();
                return;
              }
            }
          }

          // --- ABCD Reconstruction View Click / Lock ---
          if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
            const qIdx = getQuadrantAt(e.clientX, e.clientY);
            if (reconHoveredTrainingIdx >= 0) {
              if (reconLockedTrainingIdx === reconHoveredTrainingIdx) {
                reconLockedTrainingIdx = -1;
                reconHoveredTrainingIdx = -1;
                if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
                draw();
              } else {
                reconLockedTrainingIdx = reconHoveredTrainingIdx;
                reconLockedTrainingSlot = reconHoveredTrainingSlot ||
                  ((qIdx === 0) ? 'A' : 'B');
                reconLockedQueryIdx = -1;
                if (typeof showToast === 'function') {
                  showToast(
                    `🔒 Locked [${reconLockedTrainingSlot}] Sample #${reconLockedTrainingIdx}`
                  );
                }
                draw();
              }
            } else if (reconHoveredQueryIdx >= 0) {
              if (reconLockedQueryIdx === reconHoveredQueryIdx) {
                reconLockedQueryIdx = -1;
                reconHoveredQueryIdx = -1;
                if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
                draw();
              } else {
                reconLockedQueryIdx = reconHoveredQueryIdx;
                reconLockedTrainingIdx = -1;
                selectedKnnQuerySample = reconLockedQueryIdx;
                if (typeof renderReconstructionDashboard === 'function') {
                  renderReconstructionDashboard();
                }
                if (typeof showToast === 'function') {
                  showToast(`🔒 Locked Query #${reconLockedQueryIdx}`);
                }
                draw();
              }
            } else if (reconLockedTrainingIdx !== -1 || reconLockedQueryIdx !== -1) {
              reconLockedTrainingIdx = -1;
              reconLockedQueryIdx = -1;
              if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
              draw();
            }
            return;
          }

          if (hoveredClosestSample && hoveredClosestSample.point) {
            if (lockedClosestSample && lockedClosestSample.index === hoveredClosestSample.index) {
              // Clicked already locked point -> unlock
              lockedClosestSample = null;
              hoveredClosestSample = null;
              if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
              draw();
            } else {
              // Lock to this hovered sample point
              lockedClosestSample = { ...hoveredClosestSample };
              hoveredClosestSample = lockedClosestSample;
              selectedKnnQuerySample = lockedClosestSample.index;
              if (typeof renderKnnTrace === 'function') renderKnnTrace();

              if (typeof orbitCamera !== 'undefined' && orbitCamera.isLocked &&
                  lockedClosestSample.point) {
                orbitCamera.targetX = lockedClosestSample.point.x || 0;
                orbitCamera.targetY = lockedClosestSample.point.y || 0;
                orbitCamera.targetZ = lockedClosestSample.point.z || 0;
                orbitCamera.targetIndex = lockedClosestSample.index;
                orbitCamera.targetLabel = `#${lockedClosestSample.index}`;
                quadViews[3].panX = 0;
                quadViews[3].panY = 0;
                if (typeof updateLockCenterButtonUI === 'function') updateLockCenterButtonUI();
                if (typeof showToast === 'function') {
                  showToast(`🎯 3D Center locked to Sample #${lockedClosestSample.index}`);
                }
              } else if (typeof showToast === 'function') {
                showToast(`🔒 Locked Sample #${lockedClosestSample.index}`);
              }
              draw();
            }
          } else if (lockedClosestSample !== null) {
            // Clicked empty canvas space while locked -> unlock
            lockedClosestSample = null;
            hoveredClosestSample = null;
            if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
            draw();
          }
        }
      }
    });

    // Closest Sample Hover Inspector
    canvas.addEventListener('mousemove', (e) => {
      if (isDragging || isAddPointMode) return;

      // --- 4-Panel Reconstruction View Hover Handling ---
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (reconLockedQueryIdx >= 0 || reconLockedTrainingIdx >= 0) {
          return;
        }
        const rect = canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;
        const W = rect.width;
        const H = rect.height;

        if (px < 0 || px > W || py < 0 || py > H) {
          if (reconHoveredQueryIdx !== -1 || reconHoveredTrainingIdx !== -1) {
            reconHoveredQueryIdx = -1;
            reconHoveredTrainingIdx = -1;
            draw();
          }
          return;
        }

        function findClosestPointInSlot(
          slotId, px, py, qRect, activeView, activeCam, checkMask = false
        ) {
          const targetSlot = datasetSlots[slotId];
          const pts = targetSlot
            ? (targetSlot.benchmarkDataset || targetSlot.pastSamples) : null;
          if (!pts || pts.length === 0) return { index: -1, distSq: Infinity };

          const is3D = (targetSlot && targetSlot.currentDim >= 3);
          const scale = (Math.min(qRect.w, qRect.h) / 2.35) * (activeView.zoom || 1.0);
          const cx = qRect.x + qRect.w / 2;
          const cy = qRect.y + qRect.h / 2;

          let bestIdx = -1;
          let bestDistSq = Infinity;
          const checkN = Math.min(pts.length, 100000);
          const slotMask = checkMask
            ? (targetSlot.reconQualityMask ||
               (slotId === 'C' ? datasetSlots.C?.reconQualityMask
                               : datasetSlots.D?.reconQualityMask) ||
               reconQualityMask)
            : null;

          for (let i = 0; i < checkN; i++) {
            if (checkMask && reconQualityThreshold < 1.0 && slotMask &&
                i < slotMask.length && !slotMask[i]) {
              continue;
            }
            const pt = pts[i];
            const pCoord = (typeof getPlotCoords === 'function') ? getPlotCoords(pt) : pt;
            let u = pCoord.x, v = pCoord.y;
            if (is3D) {
              let tx = 0, ty = 0, tz = 0;
              if (activeCam && activeCam.isLocked) {
                tx = activeCam.targetX || 0;
                ty = activeCam.targetY || 0;
                tz = activeCam.targetZ || 0;
              }
              const pr = project3DVector(
                pCoord.x - tx, pCoord.y - ty, (pCoord.z || 0.0) - tz,
                activeCam.azimuth, activeCam.elevation
              );
              u = pr.u;
              v = pr.v;
            }
            const screenPx = cx + (u - (activeView.panX || 0)) * scale;
            const screenPy = cy - (v - (activeView.panY || 0)) * scale;
            const dx = screenPx - px;
            const dy = screenPy - py;
            const d2 = dx * dx + dy * dy;
            if (d2 < bestDistSq) {
              bestDistSq = d2;
              bestIdx = i;
            }
          }
          return { index: bestIdx, distSq: bestDistSq };
        }

        const isOverlay = (typeof reconOverlayMode !== 'undefined' && reconOverlayMode);
        const MAX_PICK_DIST_SQ = 70 * 70;

        if (isOverlay) {
          const isLeft = (px < W / 2);
          const qRect = isLeft
            ? { x: 0, y: 0, w: W / 2, h: H }
            : { x: W / 2, y: 0, w: W / 2, h: H };
          const activeView = isLeft ? quadViews[0] : quadViews[1];
          const activeCam = isLeft
            ? (typeof reconInputCamera !== 'undefined' ? reconInputCamera : orbitCamera)
            : (typeof reconOutputCamera !== 'undefined' ? reconOutputCamera : orbitCamera);

          if (isLeft) {
            // Input Space: Check Slot A and Slot C
            const hitA = findClosestPointInSlot(
              'A', px, py, qRect, activeView, activeCam, false
            );
            const hitC = findClosestPointInSlot(
              'C', px, py, qRect, activeView, activeCam, true
            );

            if (hitC.index >= 0 && hitC.distSq <= hitA.distSq &&
                hitC.distSq <= MAX_PICK_DIST_SQ) {
              if (reconHoveredQueryIdx !== hitC.index) {
                reconHoveredQueryIdx = hitC.index;
                reconHoveredTrainingIdx = -1;
                selectedKnnQuerySample = hitC.index;
                if (typeof renderReconstructionDashboard === 'function') {
                  renderReconstructionDashboard();
                }
                draw();
              }
            } else if (hitA.index >= 0 && hitA.distSq <= MAX_PICK_DIST_SQ) {
              if (reconHoveredTrainingIdx !== hitA.index ||
                  reconHoveredTrainingSlot !== 'A') {
                reconHoveredTrainingIdx = hitA.index;
                reconHoveredTrainingSlot = 'A';
                reconHoveredQueryIdx = -1;
                draw();
              }
            } else if (reconHoveredQueryIdx !== -1 || reconHoveredTrainingIdx !== -1) {
              reconHoveredQueryIdx = -1;
              reconHoveredTrainingIdx = -1;
              draw();
            }
          } else {
            // Output Space: Check Slot B and Slot D
            const hitB = findClosestPointInSlot(
              'B', px, py, qRect, activeView, activeCam, false
            );
            const hitD = findClosestPointInSlot(
              'D', px, py, qRect, activeView, activeCam, true
            );

            if (hitD.index >= 0 && hitD.distSq <= hitB.distSq &&
                hitD.distSq <= MAX_PICK_DIST_SQ) {
              if (reconHoveredQueryIdx !== hitD.index) {
                reconHoveredQueryIdx = hitD.index;
                reconHoveredTrainingIdx = -1;
                selectedKnnQuerySample = hitD.index;
                if (typeof renderReconstructionDashboard === 'function') {
                  renderReconstructionDashboard();
                }
                draw();
              }
            } else if (hitB.index >= 0 && hitB.distSq <= MAX_PICK_DIST_SQ) {
              if (reconHoveredTrainingIdx !== hitB.index ||
                  reconHoveredTrainingSlot !== 'B') {
                reconHoveredTrainingIdx = hitB.index;
                reconHoveredTrainingSlot = 'B';
                reconHoveredQueryIdx = -1;
                draw();
              }
            } else if (reconHoveredQueryIdx !== -1 || reconHoveredTrainingIdx !== -1) {
              reconHoveredQueryIdx = -1;
              reconHoveredTrainingIdx = -1;
              draw();
            }
          }
        } else {
          // 4-Panel Quadrant Hover
          const qIdx = getQuadrantAt(e.clientX, e.clientY);
          const qRect = getQuadRect(qIdx, W, H);
          const isInputSpace = (qIdx === 0 || qIdx === 2);
          const activeView = isInputSpace ? quadViews[0] : quadViews[1];
          const activeCam = isInputSpace
            ? (typeof reconInputCamera !== 'undefined' ? reconInputCamera : orbitCamera)
            : (typeof reconOutputCamera !== 'undefined' ? reconOutputCamera : orbitCamera);

          if (qIdx === 0 || qIdx === 1) {
            const slotId = (qIdx === 0) ? 'A' : 'B';
            const hit = findClosestPointInSlot(
              slotId, px, py, qRect, activeView, activeCam, false
            );
            if (hit.index >= 0 && hit.distSq <= MAX_PICK_DIST_SQ) {
              if (reconHoveredTrainingIdx !== hit.index ||
                  reconHoveredTrainingSlot !== slotId) {
                reconHoveredTrainingIdx = hit.index;
                reconHoveredTrainingSlot = slotId;
                reconHoveredQueryIdx = -1;
                draw();
              }
            } else if (reconHoveredTrainingIdx !== -1) {
              reconHoveredTrainingIdx = -1;
              draw();
            }
          } else {
            const slotId = (qIdx === 2) ? 'C' : 'D';
            const hit = findClosestPointInSlot(
              slotId, px, py, qRect, activeView, activeCam, true
            );
            if (hit.index >= 0 && hit.distSq <= MAX_PICK_DIST_SQ) {
              if (reconHoveredQueryIdx !== hit.index) {
                reconHoveredQueryIdx = hit.index;
                reconHoveredTrainingIdx = -1;
                selectedKnnQuerySample = hit.index;
                if (typeof renderReconstructionDashboard === 'function') {
                  renderReconstructionDashboard();
                }
                draw();
              }
            } else if (reconHoveredQueryIdx !== -1) {
              reconHoveredQueryIdx = -1;
              draw();
            }
          }
        }
        return;
      }

      // If user has locked a selection, keep locked selection steady
      if (lockedClosestSample !== null) {
        return;
      }

      if (!highlightClosestSample) {
        if (hoveredClosestSample !== null) {
          hoveredClosestSample = null;
          draw();
        }
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const W = rect.width;
      const H = rect.height;

      if (px < 0 || px > W || py < 0 || py > H) {
        if (hoveredClosestSample !== null) {
          hoveredClosestSample = null;
          draw();
        }
        return;
      }

      const qIdx = getQuadrantAt(e.clientX, e.clientY);
      const qRect = getQuadRect(qIdx, W, H);
      if (px < qRect.x || px > qRect.x + qRect.w || py < qRect.y || py > qRect.y + qRect.h) {
        if (hoveredClosestSample !== null) {
          hoveredClosestSample = null;
          draw();
        }
        return;
      }

      const numPast = pastSamples.length;
      if (numPast === 0) {
        if (hoveredClosestSample !== null) {
          hoveredClosestSample = null;
          draw();
        }
        return;
      }

      function getProjectedCoord(p) {
        const pt = (typeof getPlotCoords === 'function') ? getPlotCoords(p) : p;
        if (currentDim === 2 || qIdx === 2) return { u: pt.x, v: pt.y, depth: pt.z || 0 };
        if (qIdx === 0) return { u: pt.y, v: pt.z, depth: pt.x };
        if (qIdx === 1) return { u: pt.x, v: pt.z, depth: pt.y };
        // CUSTOM_3D (qIdx === 3)
        return project3D(pt.x, pt.y, pt.z, orbitCamera.azimuth, orbitCamera.elevation);
      }

      let bestIdx = -1;
      let bestDistSq = Infinity;
      let bestPos = null;
      let bestPt = null;
      const MAX_PICK_DIST_SQ = 70 * 70; // within 70px

      const maxCheck = Math.min(numPast, 100000);
      for (let i = 0; i < maxCheck; i++) {
        const pt = pastSamples[i];
        const pr = getProjectedCoord(pt);
        const pos = mapMetricToQuad(pr.u, pr.v, qIdx, qRect);

        const dx = pos.px - px;
        const dy = pos.py - py;
        const d2 = dx * dx + dy * dy;

        if (d2 < bestDistSq) {
          bestDistSq = d2;
          bestIdx = (pt.frameIndex !== undefined) ? pt.frameIndex : i;
          bestPos = pos;
          bestPt = pt;
        }
      }

      const prevIdx = hoveredClosestSample ? hoveredClosestSample.index : -1;
      if (bestIdx >= 0 && bestDistSq <= MAX_PICK_DIST_SQ) {
        let cId = -1;
        if (typeof assignmentHistory !== 'undefined' && bestIdx < assignmentHistory.length) {
          cId = assignmentHistory[bestIdx];
        } else if (typeof clustersAssigned !== 'undefined' && clustersAssigned &&
                   bestIdx < clustersAssigned.length) {
          cId = clustersAssigned[bestIdx];
        }

        hoveredClosestSample = {
          index: bestIdx,
          point: bestPt,
          qIdx: qIdx,
          screenX: bestPos.px,
          screenY: bestPos.py,
          distPx: Math.sqrt(bestDistSq),
          clusterId: cId
        };

        if (prevIdx !== bestIdx) {
          selectedKnnQuerySample = bestIdx;
          if (typeof renderKnnTrace === 'function') renderKnnTrace();
          draw();
        }
      } else {
        if (hoveredClosestSample !== null) {
          hoveredClosestSample = null;
          draw();
        }
      }
    });

    canvas.addEventListener('mouseleave', () => {
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (reconLockedQueryIdx >= 0 || reconLockedTrainingIdx >= 0) return;
        if (reconHoveredQueryIdx !== -1 || reconHoveredTrainingIdx !== -1) {
          reconHoveredQueryIdx = -1;
          reconHoveredTrainingIdx = -1;
          draw();
        }
        return;
      }
      if (lockedClosestSample !== null) {
        return;
      }
      if (hoveredClosestSample !== null) {
        hoveredClosestSample = null;
        draw();
      }
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const qIdx = getQuadrantAt(e.clientX, e.clientY);

      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
        const isInputSpace = (qIdx === 0 || qIdx === 2);
        const targetViews = isInputSpace
          ? [quadViews[0], quadViews[2]]
          : [quadViews[1], quadViews[3]];
        const targetCamera = isInputSpace
          ? (typeof reconInputCamera !== 'undefined' ? reconInputCamera : orbitCamera)
          : (typeof reconOutputCamera !== 'undefined' ? reconOutputCamera : orbitCamera);

        targetViews.forEach(v => {
          v.zoom = Math.max(0.05, (v.zoom || 1.0) * zoomFactor);
        });
        targetCamera.zoom = targetViews[0].zoom;
        updateZoomBadge();
        draw();
        return;
      }

      if (dataMode === 'image') {
        if ((e.ctrlKey || e.altKey) &&
            (qIdx === 2 || qIdx === 3 || maximizedQuad === 2 || maximizedQuad === 3)) {
          const delta = e.deltaY < 0 ? 12 : -12;
          const cur = (typeof imageThumbSize !== 'undefined') ? imageThumbSize : 64;
          const next = Math.max(36, Math.min(220, cur + delta));
          imageThumbSize = next;
          const lbl = document.getElementById('lblImgThumbSize');
          if (lbl) lbl.textContent = `${next}px`;
          draw();
          return;
        }

        if (qIdx === 2 || maximizedQuad === 2) {
          if (imageQ2ViewMode === 'knn') {
            imageKnnScrollY = Math.max(0, (imageKnnScrollY || 0) + (e.deltaY > 0 ? 30 : -30));
          } else {
            imageMembersScrollY = Math.max(0, (imageMembersScrollY || 0) + (e.deltaY > 0 ? 30 : -30));
          }
          draw();
        } else if (qIdx === 3 || maximizedQuad === 3) {
          imageClustersScrollY = Math.max(0, (imageClustersScrollY || 0) + (e.deltaY > 0 ? 30 : -30));
          draw();
        }
        return;
      }

      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;

      const v = quadViews[qIdx];
      if (v) {
        v.zoom = Math.max(0.05, (v.zoom || 1.0) * zoomFactor);
        if (qIdx === 3 && currentDim >= 3) {
          orbitCamera.zoom = v.zoom;
        }
      }

      updateZoomBadge();
      draw();
    }, { passive: false });

    canvas.addEventListener('dblclick', (e) => {
      if (isAddPointMode) return;
      const qIdx = getQuadrantAt(e.clientX, e.clientY);

      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        const isInputSpace = (qIdx === 0 || qIdx === 2);
        const targetViews = isInputSpace
          ? [quadViews[0], quadViews[2]]
          : [quadViews[1], quadViews[3]];
        const targetCamera = isInputSpace
          ? (typeof reconInputCamera !== 'undefined' ? reconInputCamera : orbitCamera)
          : (typeof reconOutputCamera !== 'undefined' ? reconOutputCamera : orbitCamera);

        targetCamera.azimuth = -35 * (Math.PI / 180);
        targetCamera.elevation = 25 * (Math.PI / 180);
        targetCamera.panX = 0;
        targetCamera.panY = 0;
        targetCamera.zoom = 1.0;
        targetViews.forEach(v => {
          v.panX = 0;
          v.panY = 0;
          v.zoom = 1.0;
        });
        updateZoomBadge();
        draw();
        return;
      }

      if (dataMode === 'image') {
        maximizedQuad = (maximizedQuad === qIdx) ? null : qIdx;
        draw();
        return;
      }
      if (qIdx === 3 && currentDim >= 3) {
        orbitCamera.azimuth = -35 * (Math.PI / 180);
        orbitCamera.elevation = 25 * (Math.PI / 180);
        orbitCamera.panX = 0;
        orbitCamera.panY = 0;
        orbitCamera.zoom = 1.0;
        quadViews[3].zoom = 1.0;
        quadViews[3].panX = 0;
        quadViews[3].panY = 0;
        updateZoomBadge();
        draw();
      } else {
        resetView();
      }
    });

    // =========================================================================
    //  TOUCH EVENT LISTENERS FOR MOBILE / TABLET GESTURES
    // =========================================================================
    let isTouchDragging = false;
    let isPinching = false;
    let touchStartX = 0;
    let touchStartY = 0;
    let activeTouchQuad = 0;
    let pinchStartDist = 0;
    let pinchQuad = 0;
    let lastTapTime = 0;

    canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      if (typeof hideRichTooltip === 'function') hideRichTooltip();

      if (e.touches.length === 2) {
        // Pinch-to-zoom gesture
        isPinching = true;
        isTouchDragging = false;
        const t1 = e.touches[0];
        const t2 = e.touches[1];
        pinchStartDist = Math.hypot(t1.clientX - t2.clientX, t1.clientY - t2.clientY);
        const midX = (t1.clientX + t2.clientX) / 2;
        const midY = (t1.clientY + t2.clientY) / 2;
        pinchQuad = getQuadrantAt(midX, midY);
        return;
      }

      if (e.touches.length === 1) {
        const t = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const px = t.clientX - rect.left;
        const py = t.clientY - rect.top;
        const W = rect.width;
        const H = rect.height;
        const qIdx = getQuadrantAt(t.clientX, t.clientY);
        const qRect = getQuadRect(qIdx, W, H);

        const now = performance.now();
        const isDoubleTap = (now - lastTapTime < 300);
        lastTapTime = now;

        if (isDoubleTap && !isAddPointMode) {
          if (dataMode === 'image') {
            maximizedQuad = (maximizedQuad === qIdx) ? null : qIdx;
            draw();
            return;
          }
          if (qIdx === 3 && currentDim >= 3) {
            orbitCamera.azimuth = -35 * (Math.PI / 180);
            orbitCamera.elevation = 25 * (Math.PI / 180);
            orbitCamera.panX = 0;
            orbitCamera.panY = 0;
            orbitCamera.zoom = 1.0;
            quadViews[3].zoom = 1.0;
            quadViews[3].panX = 0;
            quadViews[3].panY = 0;
            updateZoomBadge();
            draw();
          } else {
            resetView();
          }
          return;
        }

        // Check if tapping Maximize / Restore Icon in top-right of quadrant
        if (currentDim >= 3 && px >= qRect.x + qRect.w - 36 && px <= qRect.x + qRect.w &&
            py >= qRect.y && py <= qRect.y + 36) {
          maximizedQuad = (maximizedQuad === qIdx) ? null : qIdx;
          draw();
          return;
        }

        // Check if tapping Corner Zoom Box to reset zoom & pan to 1:1
        const zRect = (typeof viewportZoomBoxRects !== 'undefined')
          ? viewportZoomBoxRects[qIdx]
          : null;
        if (showViewportHUD && zRect &&
            px >= zRect.x && px <= zRect.x + zRect.w &&
            py >= zRect.y && py <= zRect.y + zRect.h) {
          if (quadViews && quadViews[qIdx]) {
            quadViews[qIdx].zoom = 1.0;
            quadViews[qIdx].panX = 0;
            quadViews[qIdx].panY = 0;
          }
          if (qIdx === 3 && currentDim >= 3 && typeof orbitCamera !== 'undefined') {
            orbitCamera.zoom = 1.0;
            orbitCamera.panX = 0;
            orbitCamera.panY = 0;
          }
          if (typeof updateZoomBadge === 'function') updateZoomBadge();
          draw();
          return;
        }

        // Point Injection Mode on Touch
        if (isAddPointMode) {
          if (isRunning) pauseSimulation();
          const m = mapQuadToMetric(px, py, qIdx, qRect);
          let injX = 0, injY = 0, injZ = 0;

          if (qIdx === 0) {
            injY = m.u;
            injZ = m.v;
            injX = currentFrame ? currentFrame.x : 0.0;
          } else if (qIdx === 1) {
            injX = m.u;
            injZ = m.v;
            injY = currentFrame ? currentFrame.y : 0.0;
          } else if (qIdx === 2) {
            injX = m.u;
            injY = m.v;
            injZ = currentFrame ? currentFrame.z : 0.0;
          } else if (qIdx === 3) {
            const cosT = Math.cos(orbitCamera.azimuth), sinT = Math.sin(orbitCamera.azimuth);
            injX = m.u * cosT;
            injY = m.u * sinT;
            injZ = m.v;
          }

          clusterFrame(injX, injY, injZ);
          showToast(`Injected: (${injX.toFixed(3)}, ${injY.toFixed(3)}, ${injZ.toFixed(3)})`);
          return;
        }

        // Start Touch Drag / Orbit
        isTouchDragging = true;
        isPinching = false;
        activeTouchQuad = qIdx;
        touchStartX = t.clientX;
        touchStartY = t.clientY;
      }
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();

      if (isPinching && e.touches.length === 2) {
        const t1 = e.touches[0];
        const t2 = e.touches[1];
        const currentDist = Math.hypot(t1.clientX - t2.clientX, t1.clientY - t2.clientY);
        if (pinchStartDist > 0) {
          const factor = currentDist / pinchStartDist;
          const targetQuad = (currentDim === 2) ? 2 : pinchQuad;
          const v = quadViews[targetQuad];
          if (v) {
            v.zoom = Math.max(0.05, (v.zoom || 1.0) * factor);
            if (targetQuad === 3 && currentDim >= 3) orbitCamera.zoom = v.zoom;
          }
          pinchStartDist = currentDist;
          updateZoomBadge();
          draw();
        }
        return;
      }

      if (isTouchDragging && e.touches.length === 1) {
        const t = e.touches[0];
        const dx = t.clientX - touchStartX;
        const dy = t.clientY - touchStartY;
        touchStartX = t.clientX;
        touchStartY = t.clientY;

        if (dataMode === 'image') {
          if (activeTouchQuad === 2 || maximizedQuad === 2) {
            if (imageQ2ViewMode === 'knn') {
              imageKnnScrollY = Math.max(0, (imageKnnScrollY || 0) - dy);
            } else {
              imageMembersScrollY = Math.max(0, (imageMembersScrollY || 0) - dy);
            }
            draw();
          } else if (activeTouchQuad === 3 || maximizedQuad === 3) {
            imageClustersScrollY = Math.max(0, (imageClustersScrollY || 0) - dy);
            draw();
          }
          return;
        }

        const is3DTarget = (activeTouchQuad === 3 || maximizedQuad === 3) && currentDim >= 3;
        if (is3DTarget) {
          // Orbit camera in 3D
          orbitCamera.azimuth += dx * 0.008;
          orbitCamera.elevation = Math.max(-1.52, Math.min(1.52, orbitCamera.elevation - dy * 0.008));
          draw();
        } else {
          // Pan 2D Sub-viewport
          const targetQuad = (currentDim === 2) ? 2 : activeTouchQuad;
          const rect = canvas.getBoundingClientRect();
          const qRect = getQuadRect(targetQuad, rect.width, rect.height);
          const scale = getQuadScale(targetQuad, qRect);

          const v = quadViews[targetQuad];
          if (v && scale > 0) {
            v.panX -= dx / scale;
            v.panY += dy / scale;
            draw();
          }
        }
      }
    }, { passive: false });

    ['touchend', 'touchcancel'].forEach(ev => {
      canvas.addEventListener(ev, () => {
        isTouchDragging = false;
        isPinching = false;
        pinchStartDist = 0;
      });
    });

    // 3D Camera Presets
    document.getElementById('presetIso').addEventListener('click', () => {
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (typeof reconInputCamera !== 'undefined') {
          reconInputCamera.azimuth = -35 * (Math.PI / 180);
          reconInputCamera.elevation = 25 * (Math.PI / 180);
        }
        if (typeof reconOutputCamera !== 'undefined') {
          reconOutputCamera.azimuth = -35 * (Math.PI / 180);
          reconOutputCamera.elevation = 25 * (Math.PI / 180);
        }
      }
      orbitCamera.azimuth = -35 * (Math.PI / 180);
      orbitCamera.elevation = 25 * (Math.PI / 180);
      draw();
    });

    document.getElementById('presetFront').addEventListener('click', () => {
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (typeof reconInputCamera !== 'undefined') {
          reconInputCamera.azimuth = 0;
          reconInputCamera.elevation = 0;
        }
        if (typeof reconOutputCamera !== 'undefined') {
          reconOutputCamera.azimuth = 0;
          reconOutputCamera.elevation = 0;
        }
      }
      orbitCamera.azimuth = 0;
      orbitCamera.elevation = 0;
      draw();
    });

    document.getElementById('presetTop').addEventListener('click', () => {
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (typeof reconInputCamera !== 'undefined') {
          reconInputCamera.azimuth = 0;
          reconInputCamera.elevation = 89 * (Math.PI / 180);
        }
        if (typeof reconOutputCamera !== 'undefined') {
          reconOutputCamera.azimuth = 0;
          reconOutputCamera.elevation = 89 * (Math.PI / 180);
        }
      }
      orbitCamera.azimuth = 0;
      orbitCamera.elevation = 89 * (Math.PI / 180);
      draw();
    });

    document.getElementById('presetSide').addEventListener('click', () => {
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (typeof reconInputCamera !== 'undefined') {
          reconInputCamera.azimuth = 90 * (Math.PI / 180);
          reconInputCamera.elevation = 0;
        }
        if (typeof reconOutputCamera !== 'undefined') {
          reconOutputCamera.azimuth = 90 * (Math.PI / 180);
          reconOutputCamera.elevation = 0;
        }
      }
      orbitCamera.azimuth = 90 * (Math.PI / 180);
      orbitCamera.elevation = 0;
      draw();
    });

    document.getElementById('presetReset3D').addEventListener('click', () => {
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        quadViews.forEach(v => { v.panX = 0; v.panY = 0; v.zoom = 1.0; });
        if (typeof reconInputCamera !== 'undefined') {
          reconInputCamera.azimuth = -35 * (Math.PI / 180);
          reconInputCamera.elevation = 25 * (Math.PI / 180);
          reconInputCamera.panX = 0; reconInputCamera.panY = 0; reconInputCamera.zoom = 1.0;
        }
        if (typeof reconOutputCamera !== 'undefined') {
          reconOutputCamera.azimuth = -35 * (Math.PI / 180);
          reconOutputCamera.elevation = 25 * (Math.PI / 180);
          reconOutputCamera.panX = 0; reconOutputCamera.panY = 0; reconOutputCamera.zoom = 1.0;
        }
      }
      orbitCamera.azimuth = -35 * (Math.PI / 180);
      orbitCamera.elevation = 25 * (Math.PI / 180);
      orbitCamera.panX = 0;
      orbitCamera.panY = 0;
      orbitCamera.zoom = 1.0;
      quadViews[3].zoom = 1.0;
      quadViews[3].panX = 0;
      quadViews[3].panY = 0;
      updateZoomBadge();
      draw();
    });

    // High-D Visual Plotting Dimension Selectors
    ['selectPlotDimX', 'selectSidePlotDimX'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('change', (e) => {
          setPlottingDimensions(parseInt(e.target.value, 10), plotDimY, plotDimZ);
        });
      }
    });

    ['selectPlotDimY', 'selectSidePlotDimY'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('change', (e) => {
          setPlottingDimensions(plotDimX, parseInt(e.target.value, 10), plotDimZ);
        });
      }
    });

    ['selectPlotDimZ', 'selectSidePlotDimZ'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('change', (e) => {
          setPlottingDimensions(plotDimX, plotDimY, parseInt(e.target.value, 10));
        });
      }
    });

    // Grand Tour 60 FPS animation loop
    let tourAnimFrameId = null;
    let lastTourTimestamp = 0;

    function grandTourAnimLoop(now) {
      if (typeof isTourPlaying !== 'undefined' && isTourPlaying && highDProjMode === 'tour') {
        if (!lastTourTimestamp) lastTourTimestamp = now;
        const dt = Math.min(0.1, Math.max(0.001, (now - lastTourTimestamp) / 1000));
        lastTourTimestamp = now;
        if (typeof ensureGrandTour === 'function') {
          const tour = ensureGrandTour();
          if (tour) {
            tour.step(dt);
            draw();
          }
        }
        tourAnimFrameId = requestAnimationFrame(grandTourAnimLoop);
      } else {
        tourAnimFrameId = null;
        lastTourTimestamp = 0;
      }
    }

    function startGrandTourAnimation() {
      if (!tourAnimFrameId) {
        lastTourTimestamp = 0;
        tourAnimFrameId = requestAnimationFrame(grandTourAnimLoop);
      }
    }
    window.startGrandTourAnimation = startGrandTourAnimation;

    // High-D Projection Mode Selectors
    ['selectHighDModeTop', 'selectHighDModeSide'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('change', (e) => {
          setHighDProjMode(e.target.value);
        });
      }
    });

    // Grand Tour Play/Pause button
    const btnTourPlay = document.getElementById('btnHighDTourPlay');
    if (btnTourPlay) {
      btnTourPlay.addEventListener('click', () => {
        isTourPlaying = !isTourPlaying;
        if (isTourPlaying) {
          if (highDProjMode !== 'tour') {
            setHighDProjMode('tour');
          }
          startGrandTourAnimation();
        } else {
          if (grandTour) grandTour.isPlaying = false;
        }
        updatePlottingDimSelectorsUI();
        draw();
      });
    }

    // Auto-3D Dimension Optimization button
    const btnAuto3D = document.getElementById('btnHighDAutoPick');
    if (btnAuto3D) {
      btnAuto3D.addEventListener('click', () => {
        const dataset = (typeof pastSamples !== 'undefined' && pastSamples.length > 0)
          ? pastSamples
          : (typeof benchmarkDataset !== 'undefined' ? benchmarkDataset : []);
        if (typeof HighDEngine !== 'undefined') {
          const triplet = HighDEngine.findBestSeparatingTriplet(clusters, dataset, currentDim);
          if (triplet) {
            setHighDProjMode('raw');
            setPlottingDimensions(triplet[0], triplet[1], triplet[2]);
            if (typeof showToast === 'function') {
              const msg =
                `⚡ Auto-selected best dimensions: d${triplet[0]}, d${triplet[1]}, d${triplet[2]}`;
              showToast(msg);
            }
          }
        }
      });
    }

    // PCA Triplet Select
    const selectPcaTriplet = document.getElementById('selectPcaTriplet');
    if (selectPcaTriplet) {
      selectPcaTriplet.addEventListener('change', (e) => {
        const parts = e.target.value.split(',').map(s => parseInt(s.trim(), 10));
        if (parts.length >= 3 && !parts.some(isNaN)) {
          pcaComponentIndices = parts;
          invalidateHighDCaches();
          updatePlottingDimSelectorsUI();
          draw();
        }
      });
    }

    // Grand Tour Speed Slider
    const sliderTourSpeed = document.getElementById('sliderTourSpeed');
    if (sliderTourSpeed) {
      const onTourSpeedChange = (e) => {
        tourSpeed = parseFloat(e.target.value);
        if (grandTour) grandTour.speed = tourSpeed;
        const lbl = document.getElementById('lblTourSpeed');
        if (lbl) lbl.textContent = `${tourSpeed.toFixed(1)}x`;
      };
      sliderTourSpeed.addEventListener('input', onTourSpeedChange);
      sliderTourSpeed.addEventListener('change', onTourSpeedChange);
    }

    // Tomographic Slicing Controls
    const selectSliceDim = document.getElementById('selectSliceDim');
    if (selectSliceDim) {
      selectSliceDim.addEventListener('change', (e) => {
        sliceDim = parseInt(e.target.value, 10);
        const rowSlice = document.getElementById('rowSliceSlider');
        if (rowSlice) rowSlice.style.display = (sliceDim >= 0) ? 'block' : 'none';
        draw();
      });
    }

    const selectSliceThick = document.getElementById('selectSliceThickness');
    if (selectSliceThick) {
      selectSliceThick.addEventListener('change', (e) => {
        sliceThickness = parseFloat(e.target.value);
        draw();
      });
    }

    const sliderSliceCenter = document.getElementById('sliderSliceCenter');
    if (sliderSliceCenter) {
      const onSliceCenterChange = (e) => {
        sliceCenter = parseFloat(e.target.value);
        const lbl = document.getElementById('lblSliceCenter');
        if (lbl) lbl.textContent = Number(sliceCenter).toFixed(2);
        draw();
      };
      sliderSliceCenter.addEventListener('input', onSliceCenterChange);
      sliderSliceCenter.addEventListener('change', onSliceCenterChange);
    }

    // Viewport & Glyphs Checkboxes
    const chkQuad2PCP = document.getElementById('chkQuad2PCP');
    if (chkQuad2PCP) {
      chkQuad2PCP.addEventListener('change', (e) => {
        quad2Mode = e.target.checked ? 'pcp' : 'along_z';
        draw();
      });
    }

    const chkBiplotRays = document.getElementById('chkBiplotRays');
    if (chkBiplotRays) {
      chkBiplotRays.addEventListener('change', (e) => {
        showBiplotRays = e.target.checked;
        draw();
      });
    }

    const chkAnchorSparklines = document.getElementById('chkAnchorSparklines');
    if (chkAnchorSparklines) {
      chkAnchorSparklines.addEventListener('change', (e) => {
        showAnchorSparklines = e.target.checked;
        draw();
      });
    }

    const chkDecoupledQuads = document.getElementById('chkDecoupledQuads');
    if (chkDecoupledQuads) {
      chkDecoupledQuads.addEventListener('change', (e) => {
        decoupledQuads = e.target.checked;
        draw();
      });
    }

    function updateLockCenterButtonUI() {
      const btn = document.getElementById('btnLockCenter3D');
      const btnSide = document.getElementById('btnLockCenter3DSide');
      const isLocked = !!(orbitCamera && orbitCamera.isLocked);

      [btn, btnSide].forEach(el => {
        if (!el) return;
        if (isLocked) {
          el.classList.add('active');
          el.classList.add('toggle-active');
          const shortLabel = orbitCamera.targetLabel ? ` (${orbitCamera.targetLabel})` : '';
          el.innerHTML = `🎯 Center${shortLabel}`;
          el.style.background = 'rgba(56, 189, 248, 0.25)';
          el.style.borderColor = '#38bdf8';
          el.style.color = '#38bdf8';
        } else {
          el.classList.remove('active');
          el.classList.remove('toggle-active');
          el.innerHTML = el.id === 'btnLockCenter3DSide' ? `🎯 Lock 3D Center` : `🎯 Center`;
          el.style.background = '';
          el.style.borderColor = '';
          el.style.color = '';
        }
      });
    }

    function toggleLockCenter3D(explicitTarget) {
      if (orbitCamera.isLocked && !explicitTarget) {
        // Unlock rotation center
        orbitCamera.isLocked = false;
        orbitCamera.targetX = 0;
        orbitCamera.targetY = 0;
        orbitCamera.targetZ = 0;
        orbitCamera.targetIndex = -1;
        orbitCamera.targetLabel = '';
        updateLockCenterButtonUI();
        if (typeof showToast === 'function') showToast('🔓 3D rotation center reset to origin (0, 0, 0)');
        draw();
        return;
      }

      // Determine target coordinates
      let targetPt = explicitTarget || null;
      let targetIdx = -1;
      let targetLbl = '';

      if (!targetPt) {
        if (lockedClosestSample && lockedClosestSample.point) {
          targetPt = lockedClosestSample.point;
          targetIdx = lockedClosestSample.index;
          targetLbl = `#${targetIdx}`;
        } else if (hoveredClosestSample && hoveredClosestSample.point) {
          targetPt = hoveredClosestSample.point;
          targetIdx = hoveredClosestSample.index;
          targetLbl = `#${targetIdx}`;
        } else if (typeof inspectedClusterId !== 'undefined' && inspectedClusterId >= 0 && clusters[inspectedClusterId]) {
          const c = clusters[inspectedClusterId];
          targetPt = { x: c.x, y: c.y, z: c.z || 0 };
          targetLbl = `Cls #${inspectedClusterId}`;
        } else if (currentFrame && (currentFrame.x !== undefined || currentFrame.y !== undefined)) {
          targetPt = { x: currentFrame.x || 0, y: currentFrame.y || 0, z: currentFrame.z || 0 };
          targetLbl = `Frame`;
        } else if (clusters && clusters.length > 0) {
          targetPt = { x: clusters[0].x, y: clusters[0].y, z: clusters[0].z || 0 };
          targetLbl = `Cls #0`;
        } else if (benchmarkDataset && benchmarkDataset.length > 0) {
          const p = benchmarkDataset[0];
          targetPt = {
            x: Array.isArray(p) ? p[0] : (p.x || 0),
            y: Array.isArray(p) ? p[1] : (p.y || 0),
            z: currentDim >= 3 ? (Array.isArray(p) ? (p[2] || 0) : (p.z || 0)) : 0
          };
          targetLbl = `#0`;
        } else {
          targetPt = { x: 0, y: 0, z: 0 };
          targetLbl = `(0,0,0)`;
        }
      }

      const pCoord = (typeof getPlotCoords === 'function')
        ? getPlotCoords(targetPt) : targetPt;
      orbitCamera.isLocked = true;
      orbitCamera.targetX = pCoord.x || 0;
      orbitCamera.targetY = pCoord.y || 0;
      orbitCamera.targetZ = pCoord.z || 0;
      orbitCamera.targetIndex = targetIdx;
      orbitCamera.targetLabel = targetLbl ||
        `(${orbitCamera.targetX.toFixed(2)}, ${orbitCamera.targetY.toFixed(2)}, ` +
        `${orbitCamera.targetZ.toFixed(2)})`;

      // Reset 2D pan offsets in Quad 3 so target point sits at the exact center of viewport
      quadViews[3].panX = 0;
      quadViews[3].panY = 0;

      updateLockCenterButtonUI();
      if (typeof showToast === 'function') {
        const cxStr = orbitCamera.targetX.toFixed(3);
        const cyStr = orbitCamera.targetY.toFixed(3);
        const czStr = orbitCamera.targetZ.toFixed(3);
        showToast(
          `🎯 Locked 3D center to ${orbitCamera.targetLabel} (${cxStr}, ${cyStr}, ${czStr})`
        );
      }
      draw();
    }
    window.toggleLockCenter3D = toggleLockCenter3D;

    const btnLockCenter = document.getElementById('btnLockCenter3D');
    if (btnLockCenter) {
      btnLockCenter.addEventListener('click', () => toggleLockCenter3D());
    }

    const btnLockCenterSide = document.getElementById('btnLockCenter3DSide');
    if (btnLockCenterSide) {
      btnLockCenterSide.addEventListener('click', () => toggleLockCenter3D());
    }

    // Multi-Dataset (A, B, C) Toolbar & Sidebar Handlers
    DATASET_SLOTS.forEach(sId => {
      // Toggle button in toolbar
      const btnToggle = document.getElementById(`btnToggleSlot${sId}`);
      if (btnToggle) {
        btnToggle.addEventListener('click', (e) => {
          e.stopPropagation();
          switchDatasetSlot(sId);
        });
      }

      // Toggle button in sidebar
      const btnToggleSide = document.getElementById(`btnToggleSlotSide${sId}`);
      if (btnToggleSide) {
        btnToggleSide.addEventListener('click', (e) => {
          e.stopPropagation();
          switchDatasetSlot(sId);
        });
      }

      // Inactive Row click to switch dataset
      const row = document.getElementById(`datasetRow${sId}`);
      if (row) {
        row.addEventListener('click', (e) => {
          if (e.target.tagName === 'SELECT' || e.target.tagName === 'BUTTON' ||
              e.target.tagName === 'INPUT' || e.target.closest('button') ||
              e.target.closest('select')) {
            return;
          }
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
        });
      }

      // Benchmark dropdown per slot
      const selBench = document.getElementById(`selectBenchmark_${sId}`);
      if (selBench) {
        selBench.addEventListener('change', (e) => {
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
          if (e.target.value.startsWith('32D')) {
            if (typeof setClusteringRlim === 'function') {
              setClusteringRlim(1.0, false);
            } else {
              rlim = 1.0;
            }
            if (typeof setNoiseSigma === 'function') {
              setNoiseSigma(0.005, false);
            } else {
              noiseSigma = 0.005;
            }
          }
          stageDataset(e.target.value, sId);
          resetView();
        });
      }

      // Loop count dropdown per slot
      const selLoop = document.getElementById(`selectLoop_${sId}`);
      if (selLoop) {
        selLoop.addEventListener('change', (e) => {
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
          loopCount = parseInt(e.target.value, 10);
          const side = document.getElementById('selectLoopSide');
          if (side) side.value = e.target.value;
          stageDataset(null, sId);
        });
      }

      // Generate / Stage button per slot
      const btnStage = document.getElementById(`btnStageDataset_${sId}`);
      if (btnStage) {
        btnStage.addEventListener('click', (e) => {
          e.stopPropagation();
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
          pauseSimulation();
          stageDataset(null, sId);
          if (typeof showToast === 'function') {
            showToast(`🎲 Generated dataset [${sId}] "${currentBenchmark}" (${benchmarkDataset.length.toLocaleString()} pts)`);
          }
        });
      }

      // Clear button per slot
      const btnClear = document.getElementById(`btnClearDataset_${sId}`);
      if (btnClear) {
        btnClear.addEventListener('click', (e) => {
          e.stopPropagation();
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
          clearDatasetSlot(sId);
        });
      }
    });

    // Multi-Dataset Toggle Buttons
    const btnToggleMulti = document.getElementById('btnToggleMultiDataset');
    if (btnToggleMulti) {
      btnToggleMulti.addEventListener('click', () => {
        setMultiDatasetEnabled(!multiDatasetEnabled);
      });
    }
    const btnToggleMultiSide = document.getElementById('btnToggleMultiDatasetSide');
    if (btnToggleMultiSide) {
      btnToggleMultiSide.addEventListener('click', () => {
        setMultiDatasetEnabled(!multiDatasetEnabled);
      });
    }

    const selectBenchmarkLegacy = document.getElementById('selectBenchmark');
    if (selectBenchmarkLegacy) {
      selectBenchmarkLegacy.addEventListener('change', (e) => {
        if (e.target.value.startsWith('32D')) {
          if (typeof setClusteringRlim === 'function') {
            setClusteringRlim(1.0, false);
          } else {
            rlim = 1.0;
          }
          if (typeof setNoiseSigma === 'function') {
            setNoiseSigma(0.005, false);
          } else {
            noiseSigma = 0.005;
          }
        }
        stageDataset(e.target.value);
        resetView();
      });
    }

    const selectLoopLegacy = document.getElementById('selectLoop');
    if (selectLoopLegacy) {
      selectLoopLegacy.addEventListener('change', (e) => {
        loopCount = parseInt(e.target.value, 10);
        const side = document.getElementById('selectLoopSide');
        if (side) side.value = e.target.value;
        stageDataset();
      });
    }

    document.getElementById('btnPlay').addEventListener('click', () => {
      if (engineMode === 'cli') {
        if (isCliRunning) {
          killNativeCli();
        } else {
          runNativeCli();
        }
        return;
      }
      if (isComputeAllRunning) {
        abortComputeAll();
        return;
      }
      if (isRunning) pauseSimulation();
      else startSimulation();
    });

    const btnComputeAll = document.getElementById('btnComputeAll');
    if (btnComputeAll) {
      btnComputeAll.addEventListener('click', () => {
        if (engineMode === 'cli') {
          if (isCliRunning) {
            killNativeCli();
          } else {
            runNativeCli();
          }
          return;
        }
        if (isComputeAllRunning) {
          abortComputeAll();
          return;
        }
        if (!benchmarkDataset || benchmarkDataset.length === 0) {
          stageDataset();
        }
        if (!hasMoreFrames()) {
          resetClustering(true);
          currentFrameIdx = 0;
        }
        if (useWasm && GricWasm.isLoaded()) {
          const params = GricWasm.buildParamsFromState();
          if (!wasmSessionActive || !GricWasm.isReady() ||
              (GricWasm.isConfigChanged && GricWasm.isConfigChanged(params))) {
            wasmSessionActive = GricWasm.init(params);
            updateWasmBadge();
          }
        }
        runClusteringToCompletion();
      });
    }

    document.getElementById('btnStep').addEventListener('click', () => {
      if (engineMode === 'cli') {
        showToast('Step inspection is only available in WASM Interactive Simulation mode');
        return;
      }
      if (isComputeAllRunning) {
        abortComputeAll();
      }
      if (!benchmarkDataset || benchmarkDataset.length === 0) {
        stageDataset();
      }
      if (!hasMoreFrames()) {
        resetClustering(true);
        currentFrameIdx = 0;
      }
      if (useWasm && GricWasm.isLoaded() && (!wasmSessionActive || !GricWasm.isReady())) {
        const params = GricWasm.buildParamsFromState();
        wasmSessionActive = GricWasm.init(params);
        updateWasmBadge();
      }
      if (typeof dataMode !== 'undefined' && dataMode === 'image') {
        if (typeof inspectedImageFrameIdx !== 'undefined') {
          inspectedImageFrameIdx = -1;
        }
        if (typeof inspectedClusterId !== 'undefined') {
          inspectedClusterId = -1;
        }
      }
      if (isRunning) pauseSimulation();
      stepNextFrame(false);
      if (useWasm && wasmSessionActive && GricWasm.isReady()) {
        const snapshot = GricWasm.syncState(true);
        if (snapshot) {
          GricWasm.applyToJsState(snapshot);
        }
      }
      updateUI();
      draw();
    });

    document.getElementById('btnAddPoint').addEventListener('click', () => {
      setAddPointMode(!isAddPointMode);
    });

    document.getElementById('btnExplain').addEventListener('click', () => {
      setExplainMode(!isExplainMode);
    });

    const btnWasmEl = document.getElementById('btnWasm');
    if (btnWasmEl) {
      btnWasmEl.addEventListener('click', () => {
        toggleWasmEngine();
      });
    }

    const btnPass2El = document.getElementById('btnPass2Nearest');
    if (btnPass2El) {
      btnPass2El.addEventListener('click', () => {
        if (isRunning) pauseSimulation();
        runSecondPassClustering();
      });
    }

    const optPass2El = document.getElementById('optPass2Nearest');
    if (optPass2El) {
      optPass2El.addEventListener('click', () => {
        optPass2El.classList.toggle('active');
        usePass2Nearest = optPass2El.classList.contains('active');
        showToast(usePass2Nearest ? '✓ 2nd Pass (auto-reassign) enabled' : '✗ 2nd Pass disabled');
      });
    }

    document.getElementById('tabNarrative').addEventListener('click', () => {
      setTab('narrative');
      updateUI();
    });
    document.getElementById('tabCandidates').addEventListener('click', () => {
      setTab('candidates');
      updateUI();
    });
    const tabTMEl = document.getElementById('tabTM');
    if (tabTMEl) {
      tabTMEl.addEventListener('click', () => {
        setTab('tm');
        updateUI();
      });
    }
    const tabEntropyTraceEl = document.getElementById('tabEntropyTrace');
    if (tabEntropyTraceEl) {
      tabEntropyTraceEl.addEventListener('click', () => {
        setTab('entropy');
        updateUI();
      });
    }

    // Recent Samples History Navigation Listeners
    const btnPrevSample = document.getElementById('btnPrevSample');
    if (btnPrevSample) {
      btnPrevSample.addEventListener('click', () => {
        if (sampleTraceLog.length === 0) return;
        let currentPos = -1;
        if (selectedSampleTraceIndex === -1) {
          currentPos = sampleTraceLog.length - 1;
        } else {
          currentPos = sampleTraceLog.findIndex(e => e.frameIndex === selectedSampleTraceIndex);
        }
        if (currentPos > 0) {
          selectPastSample(sampleTraceLog[currentPos - 1].frameIndex);
        }
      });
    }

    const btnNextSample = document.getElementById('btnNextSample');
    if (btnNextSample) {
      btnNextSample.addEventListener('click', () => {
        if (sampleTraceLog.length === 0 || selectedSampleTraceIndex === -1) return;
        const currentPos = sampleTraceLog.findIndex(e => e.frameIndex === selectedSampleTraceIndex);
        if (currentPos >= 0 && currentPos < sampleTraceLog.length - 1) {
          selectPastSample(sampleTraceLog[currentPos + 1].frameIndex);
        } else if (currentPos === sampleTraceLog.length - 1) {
          returnToLiveStream();
        }
      });
    }

    const btnLiveSample = document.getElementById('btnLiveSample');
    if (btnLiveSample) {
      btnLiveSample.addEventListener('click', () => {
        returnToLiveStream();
      });
    }

    const selectSampleHistory = document.getElementById('selectSampleHistory');
    if (selectSampleHistory) {
      selectSampleHistory.addEventListener('change', (e) => {
        const val = parseInt(e.target.value, 10);
        if (val === -1) {
          returnToLiveStream();
        } else {
          selectPastSample(val);
        }
      });
    }

    window.addEventListener('keydown', (e) => {
      // Ignore when typing in inputs or selects
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

      if ((e.ctrlKey || e.metaKey) && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        toggleCommandPalette();
        return;
      }

      if (e.key === 'Escape') {
        if (isComputeAllRunning) {
          e.preventDefault();
          abortComputeAll();
          return;
        }
        if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView &&
            (reconLockedQueryIdx >= 0 || reconLockedTrainingIdx >= 0)) {
          e.preventDefault();
          reconLockedQueryIdx = -1;
          reconLockedTrainingIdx = -1;
          showToast('🔓 Selection Unlocked');
          draw();
          return;
        }
        if (maximizedQuad !== null) {
          e.preventDefault();
          maximizedQuad = null;
          syncImageQuadUI();
          draw();
          showToast('⊞ Restored All 4 View Panels');
          return;
        }
      }

      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        const btnPlay = document.getElementById('btnPlay');
        if (btnPlay) btnPlay.click();
      } else if (e.key === 's' || e.key === 'S') {
        const btnStep = document.getElementById('btnStep');
        if (btnStep) btnStep.click();
      } else if (e.key === 'r' || e.key === 'R') {
        const btnReset = document.getElementById('btnReset');
        if (btnReset) btnReset.click();
      } else if (e.key === 'e' || e.key === 'E') {
        const btnExplain = document.getElementById('btnExplain');
        if (btnExplain) btnExplain.click();
      } else if (e.key === 'p' || e.key === 'P' || e.key === '2') {
        const btnPass2 = document.getElementById('btnPass2Nearest');
        if (btnPass2) btnPass2.click();
      } else if (e.key === 'k' || e.key === 'K') {
        const btnRunKnn = document.getElementById('btnRunKnn');
        if (btnRunKnn) btnRunKnn.click();
      } else if (e.key === 'o' || e.key === 'O') {
        setReconOverlayMode();
      } else if (e.key === 'h' || e.key === 'H' || e.key === 't' || e.key === 'T') {
        const btnToggleTooltips = document.getElementById('btnToggleTooltips');
        if (btnToggleTooltips) btnToggleTooltips.click();
      } else if (e.key === 'z' || e.key === 'Z') {
        const btnResetView = document.getElementById('btnResetView');
        if (btnResetView) btnResetView.click();
      } else if (e.key === '[' || e.key === 'ArrowLeft') {
        if (typeof dataMode !== 'undefined' && dataMode === 'image') {
          const total = (benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset.length
            : totalFrames;
          if (total === 0) return;
          const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
            inspectedImageFrameIdx >= 0)
            ? inspectedImageFrameIdx
            : totalFrames - 1;
          const target = Math.max(0, cur - 1);
          if (typeof selectImageFrame === 'function') selectImageFrame(target);
          return;
        }
        if (sampleTraceLog.length === 0) return;
        let currentPos = -1;
        if (selectedSampleTraceIndex === -1) {
          currentPos = sampleTraceLog.length - 1;
        } else {
          currentPos = sampleTraceLog.findIndex(el => el.frameIndex === selectedSampleTraceIndex);
        }
        if (currentPos > 0) {
          selectPastSample(sampleTraceLog[currentPos - 1].frameIndex);
        }
      } else if (e.key === ']' || e.key === 'ArrowRight') {
        if (typeof dataMode !== 'undefined' && dataMode === 'image') {
          const total = (benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset.length
            : totalFrames;
          if (total === 0) return;
          const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
            inspectedImageFrameIdx >= 0)
            ? inspectedImageFrameIdx
            : totalFrames - 1;
          const target = Math.min(total - 1, cur + 1);
          if (typeof selectImageFrame === 'function') selectImageFrame(target);
          return;
        }
        if (sampleTraceLog.length === 0 || selectedSampleTraceIndex === -1) return;
        const currentPos = sampleTraceLog.findIndex(el => el.frameIndex === selectedSampleTraceIndex);
        if (currentPos >= 0 && currentPos < sampleTraceLog.length - 1) {
          selectPastSample(sampleTraceLog[currentPos + 1].frameIndex);
        } else if (currentPos === sampleTraceLog.length - 1) {
          returnToLiveStream();
        }
      } else if (e.key === 'x' || e.key === 'X') {
        if (lockedClosestSample !== null) {
          lockedClosestSample = null;
          hoveredClosestSample = null;
          if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
          draw();
        } else if (hoveredClosestSample !== null && hoveredClosestSample.point) {
          lockedClosestSample = { ...hoveredClosestSample };
          selectedKnnQuerySample = lockedClosestSample.index;
          if (typeof renderKnnTrace === 'function') renderKnnTrace();
          if (typeof showToast === 'function') {
            showToast(`🔒 Locked Sample #${lockedClosestSample.index} (Click or Esc to unlock)`);
          }
          draw();
        }
      } else if (e.key === 'Escape') {
        if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
          if (reconLockedQueryIdx >= 0 || reconHoveredQueryIdx >= 0) {
            reconLockedQueryIdx = -1;
            reconHoveredQueryIdx = -1;
            if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
            draw();
          }
        }
        if (typeof dataMode !== 'undefined' && dataMode === 'image') {
          if (typeof inspectedClusterId !== 'undefined' && inspectedClusterId >= 0) {
            if (typeof clearImageClusterInspection === 'function') {
              clearImageClusterInspection();
            }
          } else if (typeof inspectedImageFrameIdx !== 'undefined' &&
            inspectedImageFrameIdx >= 0) {
            if (typeof selectImageFrame === 'function') selectImageFrame(-1);
          }
        }
        if (lockedClosestSample !== null) {
          lockedClosestSample = null;
          hoveredClosestSample = null;
          if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
          draw();
        }
      } else if (e.key === 'l' || e.key === 'L') {
        if (typeof dataMode !== 'undefined' && dataMode === 'image') {
          if (typeof selectImageFrame === 'function') selectImageFrame(-1);
        } else {
          returnToLiveStream();
        }
      } else if (e.key === 'c' || e.key === 'C') {
        toggleLockCenter3D();
      } else if (e.key === '+' || e.key === '=') {
        if (typeof dataMode !== 'undefined' && dataMode === 'image') {
          e.preventDefault();
          const cur = (typeof imageThumbSize !== 'undefined') ? imageThumbSize : 64;
          const next = Math.min(220, cur + 16);
          imageThumbSize = next;
          const lbl = document.getElementById('lblImgThumbSize');
          if (lbl) lbl.textContent = `${next}px`;
          draw();
          if (typeof showToast === 'function') showToast(`🔍 Thumbnail Size: ${next}px`);
        }
      } else if (e.key === '-' || e.key === '_') {
        if (typeof dataMode !== 'undefined' && dataMode === 'image') {
          e.preventDefault();
          const cur = (typeof imageThumbSize !== 'undefined') ? imageThumbSize : 64;
          const next = Math.max(36, cur - 16);
          imageThumbSize = next;
          const lbl = document.getElementById('lblImgThumbSize');
          if (lbl) lbl.textContent = `${next}px`;
          draw();
          if (typeof showToast === 'function') showToast(`🔍 Thumbnail Size: ${next}px`);
        }
      }
    });

    const btnStageDataset = document.getElementById('btnStageDataset');
    if (btnStageDataset) {
      btnStageDataset.addEventListener('click', () => {
        pauseSimulation();
        stageDataset();
        if (typeof showToast === 'function') {
          showToast(`🎲 Generated dataset "${currentBenchmark}" (${benchmarkDataset.length.toLocaleString()} pts)`);
        }
      });
    }

    const btnStageDatasetSide = document.getElementById('btnStageDatasetSide');
    if (btnStageDatasetSide) {
      btnStageDatasetSide.addEventListener('click', () => {
        pauseSimulation();
        stageDataset();
        if (typeof showToast === 'function') {
          showToast(`🎲 Generated dataset "${currentBenchmark}" (${benchmarkDataset.length.toLocaleString()} pts)`);
        }
      });
    }

    const btnClearDatasetSide = document.getElementById('btnClearDatasetSide');
    if (btnClearDatasetSide) {
      btnClearDatasetSide.addEventListener('click', () => {
        clearDatasetSlot(activeDatasetSlot);
      });
    }

    const btnResetClusters = document.getElementById('btnResetClusters');
    if (btnResetClusters) {
      btnResetClusters.addEventListener('click', () => {
        pauseSimulation();
        resetClustering(true);
        currentFrameIdx = 0;
        updateUI();
        draw();
        if (typeof showToast === 'function') {
          showToast('↺ Cluster models reset. Staged points preserved for next run.');
        }
      });
    }

    document.getElementById('btnReset').addEventListener('click', () => {
      pauseSimulation();
      resetSimulation();
      currentFrameIdx = 0;
      updateUI();
      draw();
      if (typeof showToast === 'function') {
        showToast('⟲ Simulator reset completely to clean blank canvas.');
      }
    });

    document.getElementById('btnResetView').addEventListener('click', resetView);

    const sliderRlim = document.getElementById('sliderRlim');
    const inputRlim = document.getElementById('inputRlim');
    if (sliderRlim) {
      sliderRlim.addEventListener('input', (e) => {
        rlim = parseFloat(e.target.value);
        if (inputRlim) inputRlim.value = rlim.toFixed(3);
        if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
          datasetSlots[activeDatasetSlot].rlim = rlim;
        }
        if (!isRunning) {
          if (totalFrames > 0) {
            resetClustering(true);
            currentFrameIdx = 0;
          } else if (useWasm && GricWasm.isLoaded()) {
            const params = GricWasm.buildParamsFromState();
            wasmSessionActive = GricWasm.init(params);
            updateWasmBadge();
          }
        }
        draw();
      });
    }
    if (inputRlim) {
      inputRlim.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0) {
          rlim = v;
          if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
            datasetSlots[activeDatasetSlot].rlim = rlim;
          }
          if (sliderRlim) {
            if (v > parseFloat(sliderRlim.max || "0.30")) {
              sliderRlim.max = Math.max(2.0, v).toFixed(2);
            }
            sliderRlim.value = Math.max(
              parseFloat(sliderRlim.min || "0.01"),
              Math.min(parseFloat(sliderRlim.max), v)
            );
          }
          if (!isRunning) {
            if (totalFrames > 0) {
              resetClustering(true);
              currentFrameIdx = 0;
            } else if (useWasm && GricWasm.isLoaded()) {
              const params = GricWasm.buildParamsFromState();
              wasmSessionActive = GricWasm.init(params);
              updateWasmBadge();
            }
          }
          draw();
        }
      });
    }

    const sliderFocus = document.getElementById('sliderFocus');
    const inputFocus = document.getElementById('inputFocus');
    function updateFocusDesc(val) {
      const lblUnit = document.getElementById('lblFocusUnit');
      if (lblUnit) {
        if (val === 0) lblUnit.innerText = "% (Points Only)";
        else if (val < 45) lblUnit.innerText = `% (Points Emph)`;
        else if (val <= 55) lblUnit.innerText = `% (Balanced)`;
        else if (val === 100) lblUnit.innerText = "% (Clusters Only)";
        else lblUnit.innerText = `% (Clusters Emph)`;
      }
    }
    if (sliderFocus) {
      sliderFocus.addEventListener('input', (e) => {
        visualFocus = parseInt(e.target.value, 10);
        if (inputFocus) inputFocus.value = visualFocus;
        updateFocusDesc(visualFocus);
        draw();
      });
    }
    if (inputFocus) {
      inputFocus.addEventListener('input', (e) => {
        let v = parseInt(e.target.value, 10);
        if (!isNaN(v)) {
          v = Math.max(0, Math.min(100, v));
          visualFocus = v;
          if (sliderFocus) sliderFocus.value = v;
          updateFocusDesc(visualFocus);
          draw();
        }
      });
    }

    const sliderPointSize = document.getElementById('sliderPointSize');
    const inputPointSize = document.getElementById('inputPointSize');
    if (sliderPointSize) {
      sliderPointSize.addEventListener('input', (e) => {
        samplePointSize = parseFloat(e.target.value);
        if (inputPointSize) inputPointSize.value = samplePointSize.toFixed(2).replace(/\.?0+$/, '');
        draw();
      });
    }
    if (inputPointSize) {
      inputPointSize.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0) {
          samplePointSize = v;
          if (sliderPointSize) sliderPointSize.value = v;
          draw();
        }
      });
    }

    function setupSingleToggle(btnId, getter, setter) {
      const btn = document.getElementById(btnId);
      if (btn) {
        btn.addEventListener('click', () => {
          const newVal = !getter();
          setter(newVal);
          if (window.updateDisplayTogglesUI) {
            window.updateDisplayTogglesUI();
          }
          draw();
        });
      }
    }

    setupSingleToggle('optToggleDistLines', () => showDistLines, v => showDistLines = v);
    setupSingleToggle('optToggleKnnLines', () => showKnnLines, v => showKnnLines = v);
    setupSingleToggle('optToggleKnnLinesKnnCard', () => showKnnLines, v => showKnnLines = v);
    setupSingleToggle('optToggleTransitionLines', () => showTransitionLines,
                      v => showTransitionLines = v);
    setupSingleToggle('optToggleClusterRadii', () => showClusterRadii, v => showClusterRadii = v);
    setupSingleToggle('optToggleGridAxes', () => showGridAxes, v => showGridAxes = v);
    setupSingleToggle('optToggleClusterLabels', () => showClusterLabels,
                      v => showClusterLabels = v);
    setupSingleToggle('optToggleDistLabels', () => showDistLabels, v => showDistLabels = v);
    setupSingleToggle('optToggleViewportHUD', () => showViewportHUD, v => showViewportHUD = v);
    setupSingleToggle('optToggleShowSamples', () => showPastSamples, v => showPastSamples = v);
    setupSingleToggle('optToggleMotionTail', () => showMotionTail, v => showMotionTail = v);
    setupSingleToggle('optToggleColorPerCluster', () => showColorPerCluster, v => {
      showColorPerCluster = v;
      draw();
    });
    setupSingleToggle('optToggleHighlightClosest', () => highlightClosestSample, v => {
      highlightClosestSample = v;
      if (!v) hoveredClosestSample = null;
    });

    const btnToggleMotionTail = document.getElementById('btnToggleMotionTail');
    if (btnToggleMotionTail) {
      btnToggleMotionTail.addEventListener('click', () => {
        showMotionTail = !showMotionTail;
        updateDisplayTogglesUI();
        draw();
      });
    }

    const btnToggleColorPerCluster = document.getElementById('btnToggleColorPerCluster');
    if (btnToggleColorPerCluster) {
      btnToggleColorPerCluster.addEventListener('click', () => {
        showColorPerCluster = !showColorPerCluster;
        updateDisplayTogglesUI();
        draw();
      });
    }

    function updateDisplayTogglesUI() {
      const syncBtn = (id, val) => {
        const el = document.getElementById(id);
        if (el) {
          el.classList.toggle('active', !!val);
          el.classList.toggle('toggle-active', !!val);
        }
      };
      syncBtn('optToggleDistLines', showDistLines);
      syncBtn('optToggleKnnLines', showKnnLines);
      syncBtn('optToggleKnnLinesKnnCard', showKnnLines);
      syncBtn('optToggleTransitionLines', showTransitionLines);
      syncBtn('optToggleClusterRadii', showClusterRadii);
      syncBtn('optToggleGridAxes', showGridAxes);
      syncBtn('optToggleClusterLabels', showClusterLabels);
      syncBtn('optToggleDistLabels', showDistLabels);
      syncBtn('optToggleViewportHUD', showViewportHUD);
      syncBtn('optToggleShowSamples', showPastSamples);
      syncBtn('optToggleMotionTail', showMotionTail);
      syncBtn('btnToggleMotionTail', showMotionTail);
      syncBtn('optToggleColorPerCluster', showColorPerCluster);
      syncBtn('btnToggleColorPerCluster', showColorPerCluster);
      syncBtn('optToggleHighlightClosest', highlightClosestSample);

      syncBtn('optCircleMembers', showCircleMembers);
      syncBtn('optCircleSCDists', showCircleSCDists);
      syncBtn('optEntropyMap', showEntropyMap);
    }
    window.updateDisplayTogglesUI = updateDisplayTogglesUI;

    const btnToggleTooltips = document.getElementById('btnToggleTooltips');
    if (btnToggleTooltips) {
      btnToggleTooltips.addEventListener('click', () => {
        if (typeof toggleTooltips === 'function') {
          toggleTooltips();
        }
      });
    }

    const optTooltipsOn = document.getElementById('optTooltipsOn');
    const optTooltipsOff = document.getElementById('optTooltipsOff');
    if (optTooltipsOn && optTooltipsOff) {
      optTooltipsOn.addEventListener('click', () => {
        if (typeof setTooltipsEnabled === 'function') {
          setTooltipsEnabled(true);
          if (typeof showToast === 'function') showToast('💡 Help Hover Tooltips: ON');
        }
      });
      optTooltipsOff.addEventListener('click', () => {
        if (typeof setTooltipsEnabled === 'function') {
          setTooltipsEnabled(false);
          if (typeof showToast === 'function') showToast('💡 Help Hover Tooltips: OFF');
        }
      });
    }

    // Max Displayed Points slider
    const sliderMaxDrawPts = document.getElementById('sliderMaxDrawPts');
    const inputMaxDrawPts = document.getElementById('inputMaxDrawPts');
    if (sliderMaxDrawPts) {
      sliderMaxDrawPts.addEventListener('input', (e) => {
        maxDrawPoints = parseInt(e.target.value, 10);
        if (inputMaxDrawPts) inputMaxDrawPts.value = maxDrawPoints;
        draw();
      });
    }
    if (inputMaxDrawPts) {
      inputMaxDrawPts.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v > 0) {
          maxDrawPoints = v;
          if (sliderMaxDrawPts) sliderMaxDrawPts.value = v;
          draw();
        }
      });
    }

    // Sample Buffer Capacity slider
    const sliderSampleBufCap = document.getElementById('sliderSampleBufCap');
    const inputSampleBufCap = document.getElementById('inputSampleBufCap');
    if (sliderSampleBufCap) {
      sliderSampleBufCap.addEventListener('input', (e) => {
        sampleBufferCap = parseInt(e.target.value, 10);
        if (inputSampleBufCap) inputSampleBufCap.value = sampleBufferCap;
        if (pastSamples.length > sampleBufferCap) {
          pastSamples = pastSamples.slice(-sampleBufferCap);
        }
      });
    }
    if (inputSampleBufCap) {
      inputSampleBufCap.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v >= 100) {
          sampleBufferCap = v;
          if (sliderSampleBufCap) sliderSampleBufCap.value = v;
          if (pastSamples.length > sampleBufferCap) {
            pastSamples = pastSamples.slice(-sampleBufferCap);
          }
        }
      });
    }

    // Batch Thinning Rate slider
    const sliderBatchThin = document.getElementById('sliderBatchThin');
    const inputBatchThin = document.getElementById('inputBatchThin');
    if (sliderBatchThin) {
      sliderBatchThin.addEventListener('input', (e) => {
        batchThinRate = parseInt(e.target.value, 10);
        if (inputBatchThin) inputBatchThin.value = batchThinRate;
      });
    }
    if (inputBatchThin) {
      inputBatchThin.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v >= 1) {
          batchThinRate = v;
          if (sliderBatchThin) sliderBatchThin.value = v;
        }
      });
    }

    const optCircleMembers = document.getElementById('optCircleMembers');
    if (optCircleMembers) {
      optCircleMembers.addEventListener('click', () => {
        showCircleMembers = !showCircleMembers;
        optCircleMembers.classList.toggle('active', showCircleMembers);
        updateUI();
        draw();
      });
    }

    const optCircleSCDists = document.getElementById('optCircleSCDists');
    if (optCircleSCDists) {
      optCircleSCDists.addEventListener('click', () => {
        showCircleSCDists = !showCircleSCDists;
        optCircleSCDists.classList.toggle('active', showCircleSCDists);
        updateUI();
        draw();
      });
    }

    const optEntropyMap = document.getElementById('optEntropyMap');
    if (optEntropyMap) {
      optEntropyMap.addEventListener('click', () => {
        showEntropyMap = !showEntropyMap;
        optEntropyMap.classList.toggle('active', showEntropyMap);
        updateUI();
        draw();
      });
    }

    const sliderNoiseSigma = document.getElementById('sliderNoiseSigma');
    const inputNoiseSigma = document.getElementById('inputNoiseSigma');
    if (sliderNoiseSigma) {
      sliderNoiseSigma.addEventListener('input', (e) => {
        noiseSigma = parseFloat(e.target.value);
        if (inputNoiseSigma) inputNoiseSigma.value = noiseSigma.toFixed(3);
        if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
          datasetSlots[activeDatasetSlot].noiseSigma = noiseSigma;
        }
        applyNoiseToDataset();
        syncControlDependencies();
        if (!isRunning) {
          resetSimulation();
          draw();
        }
      });
    }
    if (inputNoiseSigma) {
      inputNoiseSigma.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v >= 0) {
          noiseSigma = v;
          if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
            datasetSlots[activeDatasetSlot].noiseSigma = noiseSigma;
          }
          if (sliderNoiseSigma) sliderNoiseSigma.value = v;
          applyNoiseToDataset();
          syncControlDependencies();
          if (!isRunning) {
            resetSimulation();
            draw();
          }
        }
      });
    }

    const sliderNoiseTrunc = document.getElementById('sliderNoiseTrunc');
    const inputNoiseTrunc = document.getElementById('inputNoiseTrunc');
    if (sliderNoiseTrunc) {
      sliderNoiseTrunc.addEventListener('input', (e) => {
        noiseTruncLimit = parseFloat(e.target.value);
        if (inputNoiseTrunc) inputNoiseTrunc.value = noiseTruncLimit.toFixed(3);
        applyNoiseToDataset();
        if (!isRunning) {
          resetSimulation();
          draw();
        }
      });
    }
    if (inputNoiseTrunc) {
      inputNoiseTrunc.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0) {
          noiseTruncLimit = v;
          if (sliderNoiseTrunc) sliderNoiseTrunc.value = v;
          applyNoiseToDataset();
          if (!isRunning) {
            resetSimulation();
            draw();
          }
        }
      });
    }

    document.getElementById('modeGreedy').addEventListener('click', () => {
      targetMode = 'greedy';
      document.getElementById('modeGreedy').classList.add('active');
      document.getElementById('modeEntropy').classList.remove('active');
      syncControlDependencies();
    });

    document.getElementById('modeEntropy').addEventListener('click', () => {
      targetMode = 'entropy';
      document.getElementById('modeEntropy').classList.add('active');
      document.getElementById('modeGreedy').classList.remove('active');
      syncControlDependencies();
    });

    ['3P', '4P', '5P'].forEach(p => {
      document.getElementById(`prune${p}`).addEventListener('click', () => {
        pruneMode = p;
        ['3P', '4P', '5P'].forEach(other => document.getElementById(`prune${other}`).classList.remove('active'));
        document.getElementById(`prune${p}`).classList.add('active');
      });
    });

    const optSq8El = document.getElementById('optSq8');
    if (optSq8El) {
      optSq8El.addEventListener('click', () => {
        clusterUseSq8 = !clusterUseSq8;
        optSq8El.classList.toggle('active', clusterUseSq8);
        updateCliCommand();
        draw();
      });
    }

    document.getElementById('optTM').addEventListener('click', () => {
      useTM = !useTM;
      document.getElementById('optTM').classList.toggle('active', useTM);
      syncControlDependencies();
      updateUI();
      draw();
    });

    document.getElementById('optPred').addEventListener('click', () => {
      usePred = !usePred;
      document.getElementById('optPred').classList.toggle('active', usePred);
      syncControlDependencies();
    });

    document.getElementById('optGprob').addEventListener('click', () => {
      useGprob = !useGprob;
      document.getElementById('optGprob').classList.toggle('active', useGprob);
      syncControlDependencies();
    });

    document.getElementById('optTiles').addEventListener('click', () => {
      useTiles = !useTiles;
      document.getElementById('optTiles').classList.toggle('active', useTiles);
      syncControlDependencies();
      
      const params = GricWasm.buildParamsFromState();
      if (useTiles) {
        GricWasm.destroy();
        if (GricWasm.initMultiTile) {
            GricWasm.initMultiTile(params);
        }
      } else {
        if (GricWasm.destroyMultiTile) {
            GricWasm.destroyMultiTile();
        }
        GricWasm.init(params);
      }

      resetSimulation();
      if (currentBenchmark !== "custom") {
        loadSelectedBenchmark();
      }
      updateWasmBadge();
      updateUI();
      draw();
    });

    // Resource Tracker 4-Tab Switching
    const resourceTabs = [
      { id: 'tabResOverview', panel: 'resOverviewPanel' },
      { id: 'tabResDistances', panel: 'resDistancesPanel' },
      { id: 'tabResMemory', panel: 'resMemoryPanel' },
      { id: 'tabResCompute', panel: 'resComputePanel' }
    ];

    resourceTabs.forEach(t => {
      const tabBtn = document.getElementById(t.id);
      if (tabBtn) {
        tabBtn.addEventListener('click', () => {
          resourceTabs.forEach(other => {
            const btnEl = document.getElementById(other.id);
            const panelEl = document.getElementById(other.panel);
            if (btnEl) btnEl.classList.toggle('active', other.id === t.id);
            if (panelEl) panelEl.style.display = (other.id === t.id) ? 'block' : 'none';
          });
        });
      }
    });

    // k-NN Resource Tracker 4-Tab Switching
    const knnResourceTabs = [
      { id: 'tabKnnResOverview', panel: 'knnResOverviewPanel' },
      { id: 'tabKnnResPruning', panel: 'knnResPruningPanel' },
      { id: 'tabKnnResMemory', panel: 'knnResMemoryPanel' },
      { id: 'tabKnnResSpeed', panel: 'knnResSpeedPanel' }
    ];

    knnResourceTabs.forEach(t => {
      const tabBtn = document.getElementById(t.id);
      if (tabBtn) {
        tabBtn.addEventListener('click', () => {
          knnResourceTabs.forEach(other => {
            const btnEl = document.getElementById(other.id);
            const panelEl = document.getElementById(other.panel);
            if (btnEl) btnEl.classList.toggle('active', other.id === t.id);
            if (panelEl) panelEl.style.display = (other.id === t.id) ? 'block' : 'none';
          });
        });
      }
    });

    // Sync Side Panel Input Stream Selectors
    const selBenchSide = document.getElementById('selectBenchmarkSide');
    if (selBenchSide) {
      selBenchSide.addEventListener('change', (e) => {
        if (e.target.value.startsWith('32D')) {
          if (typeof setClusteringRlim === 'function') {
            setClusteringRlim(1.0, false);
          } else {
            rlim = 1.0;
          }
          if (typeof setNoiseSigma === 'function') {
            setNoiseSigma(0.005, false);
          } else {
            noiseSigma = 0.005;
          }
        }
        stageDataset(e.target.value);
        resetView();
      });
    }

    // Sample Count (N) slider
    const sliderSampleCount = document.getElementById('sliderSampleCount');
    const inputSampleCount = document.getElementById('inputSampleCount');
    if (sliderSampleCount) {
      sliderSampleCount.addEventListener('input', (e) => {
        sampleCount = parseInt(e.target.value, 10);
        if (inputSampleCount) inputSampleCount.value = sampleCount;
      });
      sliderSampleCount.addEventListener('change', () => {
        if (currentBenchmark !== "custom") {
          loadSelectedBenchmark();
        }
      });
    }
    if (inputSampleCount) {
      inputSampleCount.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v > 0) {
          sampleCount = v;
          if (sliderSampleCount) sliderSampleCount.value = Math.max(100, Math.min(10000, v));
        }
      });
      inputSampleCount.addEventListener('change', () => {
        if (currentBenchmark !== "custom") {
          loadSelectedBenchmark();
        }
      });
    }

    const btnUploadSide = document.getElementById('btnUploadSide');
    if (btnUploadSide) {
      btnUploadSide.addEventListener('click', () => {
        document.getElementById('fileUpload').click();
      });
    }

    const selSpeedSide = document.getElementById('selectSpeedSide');
    if (selSpeedSide) {
      selSpeedSide.addEventListener('change', (e) => {
        document.getElementById('selectSpeed').value = e.target.value;
        playSpeed = parseInt(e.target.value);
        if (isRunning) {
          pauseSimulation();
          startSimulation();
        }
      });
    }

    const selLoopSide = document.getElementById('selectLoopSide');
    if (selLoopSide) {
      selLoopSide.addEventListener('change', (e) => {
        document.getElementById('selectLoop').value = e.target.value;
        loopCount = parseInt(e.target.value);
        stageDataset();
      });
    }

    const btnAddPtSide = document.getElementById('btnAddPointSide');
    if (btnAddPtSide) {
      btnAddPtSide.addEventListener('click', () => {
        setAddPointMode(!isAddPointMode);
      });
    }

    // Side Display & 3D Camera Controls
    const sidePresets = [
      { id: 'presetIsoSide', az: -35, el: 25 },
      { id: 'presetFrontSide', az: 0, el: 0 },
      { id: 'presetTopSide', az: 0, el: 89 },
      { id: 'presetSideSide', az: 90, el: 0 }
    ];
    sidePresets.forEach(p => {
      const el = document.getElementById(p.id);
      if (el) {
        el.addEventListener('click', () => {
          orbitCamera.azimuth = p.az * (Math.PI / 180);
          orbitCamera.elevation = p.el * (Math.PI / 180);
          draw();
        });
      }
    });

    const presetReset3DSide = document.getElementById('presetReset3DSide');
    if (presetReset3DSide) {
      presetReset3DSide.addEventListener('click', () => {
        orbitCamera.azimuth = -35 * (Math.PI / 180);
        orbitCamera.elevation = 25 * (Math.PI / 180);
        orbitCamera.panX = 0;
        orbitCamera.panY = 0;
        orbitCamera.zoom = 1.0;
        quadViews[3].zoom = 1.0;
        quadViews[3].panX = 0;
        quadViews[3].panY = 0;
        updateZoomBadge();
        draw();
      });
    }

    const btnResetViewSide = document.getElementById('btnResetViewSide');
    if (btnResetViewSide) {
      btnResetViewSide.addEventListener('click', resetView);
    }

    // Auto-rlim (-scandist)
    document.getElementById('btnAutoRlim').addEventListener('click', computeAutoRlim);

    // Max Clusters & Eviction Policy (-maxcl)
    const sliderMaxcl = document.getElementById('sliderMaxcl');
    const inputMaxcl = document.getElementById('inputMaxcl');
    function updateMaxclUnit(val) {
      const unitEl = document.getElementById('lblMaxclUnit');
      if (unitEl) {
        if (val === 0) unitEl.innerText = "cls (0=∞)";
        else if (val >= 1024) unitEl.innerText = `cls (${(val/1024).toFixed(0)}k)`;
        else unitEl.innerText = "clusters";
      }
    }
    if (sliderMaxcl) {
      sliderMaxcl.addEventListener('input', (e) => {
        const idx = parseInt(e.target.value, 10);
        maxcl = idx === 0 ? 0 : (1 << (idx - 1));
        if (inputMaxcl) inputMaxcl.value = maxcl;
        updateMaxclUnit(maxcl);
        syncControlDependencies();
      });
    }
    if (inputMaxcl) {
      inputMaxcl.addEventListener('input', (e) => {
        let val = parseInt(e.target.value, 10);
        if (isNaN(val) || val < 0) val = 0;
        maxcl = val;
        const idx = (maxcl === 0) ? 0 : Math.min(17, Math.max(1, Math.round(Math.log2(maxcl)) + 1));
        if (sliderMaxcl) sliderMaxcl.value = idx;
        updateMaxclUnit(maxcl);
        syncControlDependencies();
      });
    }

    ['stratStop', 'stratDiscard', 'stratMerge'].forEach(id => {
      document.getElementById(id).addEventListener('click', () => {
        ['stratStop', 'stratDiscard', 'stratMerge'].forEach(other => document.getElementById(other).classList.remove('active'));
        document.getElementById(id).classList.add('active');
        maxclStrategy = id.replace('strat', '').toLowerCase();
        syncControlDependencies();
      });
    });

    const sliderDiscardFrac = document.getElementById('sliderDiscardFrac');
    const inputDiscardFrac = document.getElementById('inputDiscardFrac');
    if (sliderDiscardFrac) {
      sliderDiscardFrac.addEventListener('input', (e) => {
        discardFraction = parseFloat(e.target.value);
        if (inputDiscardFrac) inputDiscardFrac.value = discardFraction.toFixed(2);
      });
    }
    if (inputDiscardFrac) {
      inputDiscardFrac.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0 && v < 1) {
          discardFraction = v;
          if (sliderDiscardFrac) sliderDiscardFrac.value = v;
        }
      });
    }

    // Prior & Acceleration Tuning (-tm, -pred, -maxvis)
    const sliderTmMix = document.getElementById('sliderTmMix');
    const inputTmMix = document.getElementById('inputTmMix');
    if (sliderTmMix) {
      sliderTmMix.addEventListener('input', (e) => {
        tmMixingCoeff = parseFloat(e.target.value);
        if (inputTmMix) inputTmMix.value = tmMixingCoeff.toFixed(2);
        drawTransitionMatrix('tmHeatmapCanvas', false);
      });
    }
    if (inputTmMix) {
      inputTmMix.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v >= 0 && v <= 1) {
          tmMixingCoeff = v;
          if (sliderTmMix) sliderTmMix.value = v;
          drawTransitionMatrix('tmHeatmapCanvas', false);
        }
      });
    }

    const sliderPredHorizon = document.getElementById('sliderPredHorizon');
    const inputPredHorizon = document.getElementById('inputPredHorizon');
    if (sliderPredHorizon) {
      sliderPredHorizon.addEventListener('input', (e) => {
        predHorizon = parseInt(e.target.value, 10);
        if (inputPredHorizon) inputPredHorizon.value = predHorizon;
      });
    }
    if (inputPredHorizon) {
      inputPredHorizon.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v >= 1) {
          predHorizon = v;
          if (sliderPredHorizon) sliderPredHorizon.value = Math.max(1, Math.min(5, v));
        }
      });
    }

    const sliderMaxVis = document.getElementById('sliderMaxVis');
    const inputMaxVis = document.getElementById('inputMaxVis');
    if (sliderMaxVis) {
      sliderMaxVis.addEventListener('input', (e) => {
        maxVisitors = parseInt(e.target.value, 10);
        if (inputMaxVis) inputMaxVis.value = maxVisitors;
      });
    }
    if (inputMaxVis) {
      inputMaxVis.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v >= 1) {
          maxVisitors = v;
          if (sliderMaxVis) sliderMaxVis.value = Math.max(5, Math.min(50, v));
        }
      });
    }

    // Entropy & Soft Bayesian Likelihood (-entropy_first_gate, -entropy_gate, -entropy_fast, -soft_bayesian)
    const sliderEntropyFirstGate = document.getElementById('sliderEntropyFirstGate');
    const inputEntropyFirstGate = document.getElementById('inputEntropyFirstGate');
    if (sliderEntropyFirstGate) {
      sliderEntropyFirstGate.addEventListener('input', (e) => {
        entropyFirstGate = parseFloat(e.target.value);
        if (inputEntropyFirstGate) inputEntropyFirstGate.value = entropyFirstGate.toFixed(2);
      });
    }
    if (inputEntropyFirstGate) {
      inputEntropyFirstGate.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v >= 0) {
          entropyFirstGate = v;
          if (sliderEntropyFirstGate) sliderEntropyFirstGate.value = Math.max(0, Math.min(5.0, v));
        }
      });
    }

    const sliderEntropyGate = document.getElementById('sliderEntropyGate');
    const inputEntropyGate = document.getElementById('inputEntropyGate');
    if (sliderEntropyGate) {
      sliderEntropyGate.addEventListener('input', (e) => {
        entropyGate = parseFloat(e.target.value);
        if (inputEntropyGate) inputEntropyGate.value = entropyGate.toFixed(2);
      });
    }
    if (inputEntropyGate) {
      inputEntropyGate.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v >= 0) {
          entropyGate = v;
          if (sliderEntropyGate) sliderEntropyGate.value = Math.max(0, Math.min(4.0, v));
        }
      });
    }

    document.getElementById('optEntropyFast').addEventListener('click', () => {
      entropyFastMode = !entropyFastMode;
      document.getElementById('optEntropyFast').classList.toggle('active', entropyFastMode);
    });

    document.getElementById('optEntropyLeader').addEventListener('click', () => {
      entropyLeaderShortcut = !entropyLeaderShortcut;
      document.getElementById('optEntropyLeader').classList.toggle('active', entropyLeaderShortcut);
      syncControlDependencies();
    });

    const sliderLeaderCutoff = document.getElementById('sliderLeaderCutoff');
    const inputLeaderCutoff = document.getElementById('inputLeaderCutoff');
    if (sliderLeaderCutoff) {
      sliderLeaderCutoff.addEventListener('input', (e) => {
        entropyLeaderCutoff = parseFloat(e.target.value);
        if (inputLeaderCutoff) inputLeaderCutoff.value = entropyLeaderCutoff.toFixed(2);
      });
    }
    if (inputLeaderCutoff) {
      inputLeaderCutoff.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v >= 0 && v <= 1) {
          entropyLeaderCutoff = v;
          if (sliderLeaderCutoff) sliderLeaderCutoff.value = v;
        }
      });
    }

    document.getElementById('optSoftBayesian').addEventListener('click', () => {
      useSoftBayesian = !useSoftBayesian;
      document.getElementById('optSoftBayesian').classList.toggle('active', useSoftBayesian);
      syncControlDependencies();
    });

    const sliderBayesSigma = document.getElementById('sliderBayesSigma');
    const inputBayesSigma = document.getElementById('inputBayesSigma');
    if (sliderBayesSigma) {
      sliderBayesSigma.addEventListener('input', (e) => {
        softBayesianSigmaCoeff = parseFloat(e.target.value);
        if (inputBayesSigma) inputBayesSigma.value = softBayesianSigmaCoeff.toFixed(1);
      });
    }
    if (inputBayesSigma) {
      inputBayesSigma.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0) {
          softBayesianSigmaCoeff = v;
          if (sliderBayesSigma) sliderBayesSigma.value = v;
        }
      });
    }

    // Cross-Tile Subspace Prior Transfer (-xtile, -xtile_decay)
    document.getElementById('optXTile').addEventListener('click', () => {
      useXTile = !useXTile;
      document.getElementById('optXTile').classList.toggle('active', useXTile);
      syncControlDependencies();
    });

    const sliderXTileDecay = document.getElementById('sliderXTileDecay');
    const inputXTileDecay = document.getElementById('inputXTileDecay');
    if (sliderXTileDecay) {
      sliderXTileDecay.addEventListener('input', (e) => {
        xtileDecay = parseFloat(e.target.value);
        if (inputXTileDecay) inputXTileDecay.value = xtileDecay.toFixed(2);
      });
    }
    if (inputXTileDecay) {
      inputXTileDecay.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0 && v <= 1) {
          xtileDecay = v;
          if (sliderXTileDecay) sliderXTileDecay.value = v;
        }
      });
    }

    // Sparse DCC Distance Bounding (-sparse_dcc)
    document.getElementById('optSparseDcc').addEventListener('click', () => {
      useSparseDcc = !useSparseDcc;
      document.getElementById('optSparseDcc').classList.toggle('active', useSparseDcc);
      syncControlDependencies();
    });

    const sliderSparseDccExtra = document.getElementById('sliderSparseDccExtra');
    const inputSparseDccExtra = document.getElementById('inputSparseDccExtra');
    if (sliderSparseDccExtra) {
      sliderSparseDccExtra.addEventListener('input', (e) => {
        sparseDccExtraEvals = parseInt(e.target.value, 10);
        if (inputSparseDccExtra) inputSparseDccExtra.value = sparseDccExtraEvals;
      });
    }
    if (inputSparseDccExtra) {
      inputSparseDccExtra.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v >= 0) {
          sparseDccExtraEvals = v;
          if (sliderSparseDccExtra) sliderSparseDccExtra.value = Math.max(0, Math.min(10, v));
        }
      });
    }

    // =========================================================================
    //  k-Nearest Neighbors (gric-knn) Controls: Setup & Run Separation
    // =========================================================================

    // =========================================================================
    //  k-Nearest Neighbors (gric-knn) Controls: Setup & Manual Stoppable Run
    // =========================================================================

    let isKnnComputing = false;
    let knnAbortRequested = false;

    function showKnnProgress(pct, processed, total, speed, elapsedSec, etaSec, title) {
      const container = document.getElementById('knnProgressContainer');
      const barFill = document.getElementById('knnProgressBarFill');
      const lblPct = document.getElementById('knnProgressPct');
      const lblFrames = document.getElementById('knnProgFrames');
      const lblSpeed = document.getElementById('knnProgSpeed');
      const lblTime = document.getElementById('knnProgTime');
      const lblTitle = document.getElementById('knnProgressTitle');
      const badgeTop = document.getElementById('knnStatusBadgeTop');
      const badgeCli = document.getElementById('badgeCliStatus');

      const clampedPct = Math.max(0, Math.min(100, pct || 0));

      if (container) container.style.display = 'block';
      if (barFill) barFill.style.width = `${clampedPct.toFixed(1)}%`;
      if (lblPct) lblPct.textContent = `${clampedPct.toFixed(1)}%`;
      if (lblTitle && title) lblTitle.textContent = title;

      if (lblFrames && typeof total === 'number' && total > 0) {
        lblFrames.textContent = `Frames: ${(processed || 0).toLocaleString()} / ${total.toLocaleString()}`;
      }
      if (lblSpeed) {
        lblSpeed.textContent = (speed > 0) ? `${Math.round(speed).toLocaleString()} f/s` : `- f/s`;
      }
      if (lblTime) {
        const elap = (typeof elapsedSec === 'number') ? `${elapsedSec.toFixed(1)}s` : `0.0s`;
        const eta = (typeof etaSec === 'number' && etaSec > 0) ? `${etaSec.toFixed(1)}s` : `-`;
        lblTime.textContent = `Elapsed: ${elap} • ETA: ${eta}`;
      }

      // Update Top Status Badge
      if (badgeTop) {
        const totalStr = (total > 0) ? total.toLocaleString() : '';
        const procStr = (processed > 0) ? processed.toLocaleString() : '';
        badgeTop.textContent = (total > 0)
          ? `⚡ k-NN: ${clampedPct.toFixed(1)}% (${procStr}/${totalStr})`
          : `⚡ Computing k-NN (${clampedPct.toFixed(1)}%)...`;
        badgeTop.style.color = '#4ade80';
        badgeTop.style.borderColor = 'rgba(74, 222, 128, 0.4)';
      }

      // Update Stop Button Labels
      const btnTop = document.getElementById('btnRunKnn');
      const btnSide = document.getElementById('btnRunKnnSide');
      if (btnTop && isKnnComputing) {
        btnTop.innerHTML = `⏹ Stop (${clampedPct.toFixed(0)}%)`;
      }
      if (btnSide && isKnnComputing) {
        btnSide.innerHTML = `⏹ Stop (${clampedPct.toFixed(0)}%)`;
      }

      // Update CLI Badge if in CLI mode
      if (badgeCli && typeof engineMode !== 'undefined' && engineMode === 'cli') {
        badgeCli.textContent = `● k-NN: ${clampedPct.toFixed(1)}%`;
      }

      // Scrubber fill update during search
      const scrubFill = document.getElementById('progressFill');
      if (scrubFill) {
        scrubFill.style.width = `${clampedPct.toFixed(1)}%`;
      }
    }

    function hideKnnProgress() {
      const container = document.getElementById('knnProgressContainer');
      if (container) container.style.display = 'none';
      const barFill = document.getElementById('knnProgressBarFill');
      if (barFill) barFill.style.width = '0%';
    }

    function updateKnnButtonUI(computing, statusText) {
      const btnTop = document.getElementById('btnRunKnn');
      const btnSide = document.getElementById('btnRunKnnSide');
      const badgeTop = document.getElementById('knnStatusBadgeTop');
      const slot = (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot])
        ? datasetSlots[activeDatasetSlot] : null;
      const pts = (slot && slot.benchmarkDataset && slot.benchmarkDataset.length > 0)
        ? slot.benchmarkDataset
        : ((typeof benchmarkDataset !== 'undefined' &&
            benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset
            : (typeof pastSamples !== 'undefined' ? pastSamples : []));
      const hasPoints = (pts && pts.length > 0);

      if (computing) {
        if (btnTop) {
          btnTop.disabled = false;
          btnTop.innerHTML = '⏹ Stop k-NN';
          btnTop.style.background = 'rgba(239, 68, 68, 0.25)';
          btnTop.style.color = '#f87171';
          btnTop.style.borderColor = 'rgba(239, 68, 68, 0.6)';
          btnTop.style.opacity = '1.0';
          btnTop.style.cursor = 'pointer';
          btnTop.title = 'Click to stop / cancel k-NN computation';
        }
        if (btnSide) {
          btnSide.disabled = false;
          btnSide.innerHTML = '⏹ Stop k-NN';
          btnSide.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
          btnSide.style.opacity = '1.0';
          btnSide.style.cursor = 'pointer';
          btnSide.title = 'Click to stop / cancel k-NN computation';
        }
        if (badgeTop) {
          badgeTop.textContent = statusText || 'Computing k-NN...';
          badgeTop.style.color = '#f87171';
          badgeTop.style.borderColor = 'rgba(239, 68, 68, 0.4)';
        }
      } else {
        if (btnTop) {
          btnTop.innerHTML = '▶ Compute k-NN';
          if (!hasPoints) {
            btnTop.disabled = true;
            btnTop.style.background = 'rgba(71, 85, 105, 0.2)';
            btnTop.style.color = '#94a3b8';
            btnTop.style.borderColor = 'rgba(71, 85, 105, 0.4)';
            btnTop.style.opacity = '0.5';
            btnTop.style.cursor = 'not-allowed';
            btnTop.title = 'No dataset staged: Stage or generate a dataset first';
          } else {
            btnTop.disabled = false;
            btnTop.style.background = 'rgba(34, 197, 94, 0.2)';
            btnTop.style.color = '#4ade80';
            btnTop.style.borderColor = 'rgba(34, 197, 94, 0.5)';
            btnTop.style.opacity = '1.0';
            btnTop.style.cursor = 'pointer';
            btnTop.title = 'Compute k-Nearest Neighbors';
          }
        }
        if (btnSide) {
          btnSide.innerHTML = '▶ Compute k-NN';
          if (!hasPoints) {
            btnSide.disabled = true;
            btnSide.style.background = 'linear-gradient(135deg, #475569, #334155)';
            btnSide.style.opacity = '0.5';
            btnSide.style.cursor = 'not-allowed';
            btnSide.title = 'No dataset staged: Stage or generate a dataset first';
          } else {
            btnSide.disabled = false;
            btnSide.style.background = 'linear-gradient(135deg, #10b981, #059669)';
            btnSide.style.opacity = '1.0';
            btnSide.style.cursor = 'pointer';
            btnSide.title = 'Run Out-of-Core k-NN Solver';
          }
        }
        if (badgeTop) {
          if (!hasPoints) {
            badgeTop.textContent = 'No Dataset';
            badgeTop.style.color = '#94a3b8';
            badgeTop.style.borderColor = 'rgba(148, 163, 184, 0.3)';
          } else {
            badgeTop.textContent = `k=${knnK} • ${knnDirection} • dt≥${knnDtmin}`;
            badgeTop.style.color = '#c084fc';
            badgeTop.style.borderColor = 'rgba(192, 132, 252, 0.3)';
          }
        }
      }
    }

    function openKnnSetup() {
      const card = document.getElementById('cardKnnSettings');
      if (card && card.classList.contains('collapsed')) {
        togglePanelCollapse('cardKnnSettings');
      }
      card?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      const btn = document.getElementById('btnKnnSetup');
      if (btn) btn.classList.add('active');
      showToast('⚙️ Opened k-NN Setup Panel in sidebar');
    }

    async function executeKnnComputation() {
      if (isKnnComputing) {
        // User clicked STOP button while running
        knnAbortRequested = true;
        updateKnnButtonUI(true, 'Stopping k-NN...');
        if (engineMode === 'cli' && DesktopBridge.isNativeSupported()) {
          await DesktopBridge.killActiveJob();
        }
        isKnnComputing = false;
        updateKnnButtonUI(false);
        showToast('⏹ k-NN computation stopped');
        return;
      }

      const pts = (typeof benchmarkDataset !== 'undefined' &&
                   benchmarkDataset && benchmarkDataset.length > 0)
        ? benchmarkDataset
        : (typeof pastSamples !== 'undefined' ? pastSamples : []);
      if (!pts || pts.length === 0) {
        showToast('⚠️ No dataset staged to run k-NN. Stage or generate a dataset first.');
        return;
      }

      enableKnn = true;
      showKnnLines = true;
      syncControlDependencies();
      updateCliCommand();
      clearKnnError();

      isKnnComputing = true;
      knnAbortRequested = false;
      updateKnnButtonUI(true, 'Computing k-NN...');
      // Yield briefly to let the browser paint the active "Stop k-NN" button state
      await new Promise(r => setTimeout(r, 40));

      if (knnAbortRequested) {
        isKnnComputing = false;
        updateKnnButtonUI(false);
        return;
      }

      const targetSlot = activeDatasetSlot;

      try {
        const tKnnStart = performance.now();
        const slot = (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot])
          ? datasetSlots[activeDatasetSlot] : null;
        const pts = (slot && slot.benchmarkDataset && slot.benchmarkDataset.length > 0)
          ? slot.benchmarkDataset
          : ((typeof benchmarkDataset !== 'undefined' &&
              benchmarkDataset && benchmarkDataset.length > 0)
              ? benchmarkDataset
              : (typeof pastSamples !== 'undefined' ? pastSamples : []));

        const inputKnnKEl = document.getElementById('inputKnnK');
        if (inputKnnKEl && !isNaN(parseInt(inputKnnKEl.value, 10))) {
          knnK = Math.max(1, parseInt(inputKnnKEl.value, 10));
        }
        if (slot) slot.knnK = knnK;

        if (engineMode === 'cli' && DesktopBridge.isNativeSupported()) {
          let rawDatasetName = (slot && slot.stagedDatasetInfo && slot.stagedDatasetInfo.name)
            ? slot.stagedDatasetInfo.name
            : (slot && slot.benchmarkKey ? slot.benchmarkKey : currentBenchmark);
          if (!rawDatasetName) {
            const selCli = document.getElementById('selectCliDataset');
            rawDatasetName = selCli ? selCli.value : `${currentBenchmark}.txt`;
          }
          const datasetBase = rawDatasetName.replace(/\.(txt|csv|fits|dat|mp4|fits\.fz)$/i, '').replace(/[^a-zA-Z0-9_.-]/g, '_');
          const datasetFile = `${datasetBase}.txt`;
          const clusterDir = `${datasetBase}.clusterdat`;

          // 1. Ensure staged coordinates exist in workspace file
          if (pts && pts.length > 0) {
            await DesktopBridge.stageDatasetFile(datasetBase, pts).catch(() => {});
          }

          // 2. Check if clusterdat exists; if not, create clusters first
          const filesInWorkspace = await DesktopBridge.listFiles().catch(() => []);
          let hasClusterDir = filesInWorkspace.some(
            f => (f.name === clusterDir || f.name === `${datasetBase}.clusterdat`) && (f.is_dir || f.isDir)
          );
          if (!hasClusterDir) {
            const memClusters = (slot && slot.clusters && slot.clusters.length > 0)
              ? slot.clusters : (typeof clusters !== 'undefined' ? clusters : []);
            if (memClusters && memClusters.length > 0) {
              let centroidsText = `# GRIC Cluster Centroids\n# ID X Y Z MEMBERS\n`;
              memClusters.forEach(c => {
                centroidsText += `${c.id} ${Number(c.x || 0).toFixed(6)} ${Number(c.y || 0).toFixed(6)} ` +
                                 `${Number(c.z || 0).toFixed(6)} ${c.members || 1}\n`;
              });
              let dccText = `# GRIC Cluster-to-Cluster Distance Matrix D_cc\n`;
              const memDcc = (slot && slot.dcc && slot.dcc.length > 0)
                ? slot.dcc : (typeof dcc !== 'undefined' ? dcc : []);
              if (memDcc && memDcc.length > 0) {
                memDcc.forEach(row => {
                  dccText += row.map(v => Number(v).toFixed(6)).join(' ') + '\n';
                });
              }
              let memText = `# Frame Membership Assignments\n`;
              const memPast = (slot && slot.pastSamples && slot.pastSamples.length > 0)
                ? slot.pastSamples : (typeof pastSamples !== 'undefined' ? pastSamples : []);
              if (memPast && memPast.length > 0) {
                for (let i = 0; i < memPast.length; i++) {
                  const p = memPast[i];
                  const cId = p.clusterId >= 0 ? p.clusterId : 0;
                  memText += `${i} ${cId} 0.000000\n`;
                }
              }
              let exportFiles = {
                'anchors.txt': centroidsText,
                'dcc.txt': dccText,
                'frame_membership.txt': memText
              };
              await DesktopBridge.exportClusterDat(datasetBase, exportFiles).catch(() => {});
            } else {
              // Run native gric-cluster in workspace first to generate cluster anchors
              showToast(`⚡ Clustering ${datasetFile} for k-NN metric bounds...`);
              await new Promise((resolve) => {
                DesktopBridge.runCliJob({
                  cmd: 'gric-cluster',
                  args: ['-outdir', clusterDir, '0.15', datasetFile],
                  onOutput: (chunk) => {
                    const consoleEl = document.getElementById('cliConsoleLog');
                    if (consoleEl) {
                      consoleEl.textContent += chunk;
                      consoleEl.scrollTop = consoleEl.scrollHeight;
                    }
                  },
                  onTelemetry: () => {},
                  onFinish: () => resolve()
                }).catch(() => resolve());
              });
            }
          }

          const args = [
            datasetFile,
            clusterDir,
            '-k', String(knnK),
            '-dtmin', String(knnDtmin)
          ];
          if (knnDirection === 'past') args.push('-past');
          if (knnDirection === 'future') args.push('-future');
          if (knnEpsilon > 0) args.push('-eps', String(knnEpsilon));
          if (knnRlim > 0) args.push('-rlim', String(knnRlim));
          if (typeof knnMvp !== 'undefined' && knnMvp) args.push('-multipivot');
          if (typeof knnUseSq8 !== 'undefined' && knnUseSq8) args.push('-sq8');
          args.push('-progress');
          args.push('-txt');

          const consoleEl = document.getElementById('cliConsoleLog');
          const btnRunCli = document.getElementById('btnRunCli');
          const btnRunCliKnn = document.getElementById('btnRunCliKnn');
          const btnKillCli = document.getElementById('btnKillCli');
          const badgeStatus = document.getElementById('badgeCliStatus');

          if (btnRunCli) btnRunCli.disabled = true;
          if (btnRunCliKnn) btnRunCliKnn.disabled = true;
          if (btnKillCli) btnKillCli.disabled = false;
          if (badgeStatus) {
            badgeStatus.textContent = '● tmux: gric_knn';
            badgeStatus.style.background = 'rgba(192, 132, 252, 0.2)';
            badgeStatus.style.color = '#c084fc';
          }

          if (consoleEl) {
            consoleEl.textContent = `🚀 Dispatched in tmux session: gric_cli\n` +
              `🖥️ Attach live: tmux attach -t gric_cli\n` +
              `📄 Log stream: /tmp/gric_latest.log\n` +
              `⚙️ Command: gric-knn ${args.join(' ')}\n` +
              `─────────────────────────────────────────────────────────────\n`;
          }

          showToast(`💻 Running native gric-knn (k=${knnK})...`);
          showKnnProgress(0, 0, pts.length, 0, 0, 0, 'Starting native gric-knn solver...');

          let rawCliOutput = '';
          let finalExitCode = 0;
          await new Promise((resolve) => {
            DesktopBridge.runCliJob({
              cmd: 'gric-knn',
              args: args,
              onOutput: (chunk) => {
                rawCliOutput += chunk;
                if (consoleEl) {
                  consoleEl.textContent += chunk;
                  consoleEl.scrollTop = consoleEl.scrollHeight;
                }

                // Match live progress: "Searching k-NN: [====] 50.0% (500 / 1000 frames)"
                const lines = chunk.split(/\r|\n/);
                for (const line of lines) {
                  const m = line.match(
                    /Searching k-NN:\s*\[.*?\]\s*([\d.]+)%\s*\(\s*(\d+)\s*\/\s*(\d+)\s*frames\)/
                  );
                  if (m) {
                    const pct = parseFloat(m[1]);
                    const processed = parseInt(m[2], 10);
                    const total = parseInt(m[3], 10);
                    const now = performance.now();
                    const elapsedSec = Math.max(0.01, (now - tKnnStart) / 1000.0);
                    const speed = (processed > 0) ? (processed / elapsedSec) : 0;
                    const remaining = Math.max(0, total - processed);
                    const etaSec = (speed > 0) ? (remaining / speed) : 0;

                    showKnnProgress(
                      pct, processed, total, speed, elapsedSec, etaSec,
                      'Searching k-Nearest Neighbors...'
                    );
                  }
                }
              },
              onTelemetry: () => {},
              onFinish: (res) => {
                finalExitCode = res?.exitCode ?? 0;
                if (finalExitCode === 0) {
                  showToast(`✅ Native gric-knn finished successfully!`);
                } else {
                  showToast(`⚠️ Native gric-knn finished (Exit: ${finalExitCode})`);
                }
                if (btnRunCli) btnRunCli.disabled = false;
                if (btnRunCliKnn) btnRunCliKnn.disabled = false;
                if (btnKillCli) btnKillCli.disabled = true;
                resolve();
              }
            }).catch((err) => {
              showToast(`⚠️ Failed to start gric-knn: ${err.message}`);
              if (consoleEl) {
                consoleEl.textContent += `\n❌ Error: ${err.message}\n`;
              }
              if (btnRunCli) btnRunCli.disabled = false;
              if (btnRunCliKnn) btnRunCliKnn.disabled = false;
              if (btnKillCli) btnKillCli.disabled = true;
              resolve();
            });
          });

          // Parse telemetry from stdout log
          const parsedTelem = DesktopBridge.parseKnnTelemetryLog(rawCliOutput);

          // Attempt to load top-k neighbors from knn_results.txt
          let nativeData = await DesktopBridge.readKnnResults(clusterDir, knnK);
          if (!nativeData) {
            nativeData = await DesktopBridge.readKnnResults(`${datasetName}.clusterdat`, knnK);
          }
          if (!nativeData || !nativeData.indices || nativeData.indices[0] === -1) {
            // Fallback to WASM / JS engine if native binary produced no valid indices
            const config = {
              k: knnK,
              dtmin: knnDtmin,
              direction: knnDirection,
              epsilon: knnEpsilon,
              rlim: knnRlim,
              multiPivot: (typeof knnMvp !== 'undefined' && knnMvp)
            };
            const wasmRes = GricWasm.runKnn(config, pts);
            if (wasmRes && wasmRes.indices && wasmRes.indices.length > 0) {
              nativeData = wasmRes;
            }
          }
          if (!nativeData) {
            const N = parsedTelem.totalQueries || pts.length;
            nativeData = {
              totalFrames: N,
              k: knnK,
              indices: new Int32Array(N * knnK).fill(-1),
              distances: new Float64Array(N * knnK).fill(0.0)
            };
          }

          const totalWallMs = performance.now() - tKnnStart;
          const timeCompute = parsedTelem.timeSearchMs || 0.0;
          parsedTelem.timeComputeMs = timeCompute;
          parsedTelem.timeTotalMs = totalWallMs;
          parsedTelem.timeIoMs = Math.max(0, totalWallMs - timeCompute);
          nativeData.telemetry = parsedTelem;

          if (datasetSlots[targetSlot]) {
            datasetSlots[targetSlot].knnResults = nativeData;
          }
          if (activeDatasetSlot === targetSlot) {
            knnResults = nativeData;
            {
              const btnDD = document.getElementById(
                'btnRunDimDensity'
              );
              if (btnDD) btnDD.disabled = false;
            }
            if (typeof dataMode !== 'undefined' && dataMode === 'image') {
              imageQ2ViewMode = 'knn';
              if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
            }
            if (typeof renderKnnTrace === 'function') {
              renderKnnTrace();
            }
            if (typeof renderDataStructuresUI === 'function') {
              renderDataStructuresUI();
            }
            if (typeof updateDatasetStatusBadge === 'function') {
              updateDatasetStatusBadge();
            }
            draw();
          }
          showKnnProgress(100, pts.length, pts.length, 0, totalWallMs / 1000.0, 0, 'Completed!');
          showToast(
            `✅ Native k-NN computed: Total ${totalWallMs.toFixed(1)} ms ` +
            `(Compute: ${timeCompute.toFixed(1)} ms, ` +
            `I/O & IPC: ${parsedTelem.timeIoMs.toFixed(1)} ms)`
          );
        } else {
          // WASM / JS Execution
          const config = {
            k: knnK,
            dtmin: knnDtmin,
            direction: knnDirection,
            epsilon: knnEpsilon,
            rlim: knnRlim,
            multiPivot: (typeof knnMvp !== 'undefined' && knnMvp)
          };

          const N = pts.length;
          showKnnProgress(15, 0, N, 0, 0.05, 0.2, 'Building Metric Model & Super-Clusters...');
          await new Promise(r => setTimeout(r, 40));

          showKnnProgress(
            40, Math.floor(N * 0.4), N, 0, 0.1, 0.15, 'Evaluating Multi-Pivot Pruning...'
          );
          await new Promise(r => setTimeout(r, 20));

          const t0 = performance.now();
          const results = GricWasm.runKnn(config, pts);
          const totalWallMs = performance.now() - t0;

          if (knnAbortRequested) {
            return;
          }

          if (results && !results.error && results.indices) {
            const timeCompute = (typeof results.telemetry?.timeComputeMs === 'number')
              ? results.telemetry.timeComputeMs
              : (results.telemetry?.timeSearchMs || 0.0);
            results.telemetry.timeComputeMs = timeCompute;
            results.telemetry.timeTotalMs = totalWallMs;
            results.telemetry.timeIoMs = Math.max(0, totalWallMs - timeCompute);

            clearKnnError();
            if (datasetSlots[targetSlot]) {
              datasetSlots[targetSlot].knnResults = results;
            }
            if (activeDatasetSlot === targetSlot) {
              knnResults = results;
              {
                const btnDD = document.getElementById(
                  'btnRunDimDensity'
                );
                if (btnDD) btnDD.disabled = false;
              }
              if (typeof dataMode !== 'undefined' && dataMode === 'image') {
                imageQ2ViewMode = 'knn';
                if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
              }
              if (typeof renderKnnTrace === 'function') {
                renderKnnTrace();
              }
              if (typeof renderDataStructuresUI === 'function') {
                renderDataStructuresUI();
              }
              if (typeof updateDatasetStatusBadge === 'function') {
                updateDatasetStatusBadge();
              }
              draw();
            }
            showKnnProgress(100, N, N, 0, totalWallMs / 1000.0, 0, 'Completed!');
            showToast(
              `⚡ k-NN computed: Total ${totalWallMs.toFixed(1)} ms ` +
              `(Compute: ${timeCompute.toFixed(1)} ms, ` +
              `I/O: ${results.telemetry.timeIoMs.toFixed(1)} ms)`
            );
          } else {
            const errMsg = (results && results.error)
              ? results.error
              : `WASM k-NN memory limit reached ` +
                `(${pts.length.toLocaleString()} points exceeds WASM heap).`;
            showKnnError('WASM Memory Limit Exceeded', errMsg);
            showToast(`⚠️ ${errMsg}`);
            console.error('[k-NN WASM]', errMsg);
          }
        }
      } catch (err) {
        console.error('[k-NN] Computation error:', err);
        const errMsg = err && err.message ? err.message : 'Unknown execution error';
        showKnnError('k-NN Execution Error', errMsg);
        showToast(`⚠️ k-NN computation failed: ${errMsg}`);
      } finally {
        isKnnComputing = false;
        knnAbortRequested = false;
        setTimeout(() => {
          hideKnnProgress();
          updateKnnButtonUI(false);
        }, 700);
      }
    }

    function showKnnError(title, message) {
      const errConfig = document.getElementById('knnErrorBannerConfig');
      const errTitleConfig = document.getElementById('knnErrorTitleConfig');
      const errMsgConfig = document.getElementById('knnErrorMsgConfig');
      if (errConfig && errMsgConfig) {
        if (errTitleConfig) errTitleConfig.textContent = title || 'k-NN Error';
        errMsgConfig.textContent = message || 'k-NN computation failed.';
        errConfig.style.display = 'block';
      }

      const errRes = document.getElementById('knnErrorBannerRes');
      const errTitleRes = document.getElementById('knnErrorTitleRes');
      const errMsgRes = document.getElementById('knnErrorMsgRes');
      if (errRes && errMsgRes) {
        if (errTitleRes) errTitleRes.textContent = title || 'k-NN Error';
        errMsgRes.textContent = message || 'k-NN computation failed.';
        errRes.style.display = 'block';
      }

      const statusBadge = document.getElementById('knnStatusBadge');
      if (statusBadge) {
        statusBadge.textContent = '⚠️ Memory Limit';
        statusBadge.style.color = '#f87171';
        statusBadge.style.background = 'rgba(239, 68, 68, 0.2)';
        statusBadge.style.border = '1px solid rgba(239, 68, 68, 0.4)';
      }
    }

    function clearKnnError() {
      const errConfig = document.getElementById('knnErrorBannerConfig');
      if (errConfig) errConfig.style.display = 'none';

      const errRes = document.getElementById('knnErrorBannerRes');
      if (errRes) errRes.style.display = 'none';

      const statusBadge = document.getElementById('knnStatusBadge');
      if (statusBadge && statusBadge.textContent.includes('Memory Limit')) {
        statusBadge.textContent = (typeof knnResults !== 'undefined' && knnResults)
          ? 'Active'
          : 'Ready';
        statusBadge.style.color = '';
        statusBadge.style.background = '';
        statusBadge.style.border = '';
      }
    }

    window.dismissKnnError = function() {
      clearKnnError();
    };

    function toggleKnnModule(enable) {
      if (typeof enable !== 'boolean') {
        enable = !enableKnn;
      }
      enableKnn = enable;
      showKnnLines = enable;
      syncControlDependencies();
      updateCliCommand();

      if (enableKnn) {
        showToast('⚡ k-NN enabled. Configure options and click ▶ Compute k-NN.');
      } else {
        showToast('k-NN module disabled');
      }
      draw();
    }

    function resetKnn() {
      knnResults = null;
      if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
        datasetSlots[activeDatasetSlot].knnResults = null;
      }
      selectedKnnQuerySample = -1;
      hoveredKnnNeighborId = -1;
      hoveredClosestSample = null;
      lockedClosestSample = null;
      clearKnnError();

      // Reset dim/density state
      dimDensityResults = null;
      dimDensitySummary = null;
      pointColorMode = 'cluster';
      dimDensityTraceOffset = 0;
      {
        const btnDD = document.getElementById(
          'btnRunDimDensity'
        );
        if (btnDD) btnDD.disabled = true;
        const card7 = document.getElementById(
          'cardDimDensity'
        );
        if (card7) card7.style.display = 'none';
        const cg = document.getElementById(
          'dimDensityColorGroup'
        );
        if (cg) cg.style.display = 'none';
      }

      if (typeof renderKnnTrace === 'function') {
        renderKnnTrace();
      }
      if (typeof renderDataStructuresUI === 'function') {
        renderDataStructuresUI();
      }
      if (typeof updateDatasetStatusBadge === 'function') {
        updateDatasetStatusBadge();
      }
      draw();
      showToast('↺ k-NN results reset (clusters preserved)');
    }

    const btnToggleKnnModule = document.getElementById('btnToggleKnnModule');
    if (btnToggleKnnModule) {
      btnToggleKnnModule.addEventListener('click', () => toggleKnnModule());
    }

    const btnKnnSetup = document.getElementById('btnKnnSetup');
    if (btnKnnSetup) {
      btnKnnSetup.addEventListener('click', openKnnSetup);
    }

    const btnRunKnn = document.getElementById('btnRunKnn');
    if (btnRunKnn) {
      btnRunKnn.addEventListener('click', executeKnnComputation);
    }

    const btnRunKnnSide = document.getElementById('btnRunKnnSide');
    if (btnRunKnnSide) {
      btnRunKnnSide.addEventListener('click', executeKnnComputation);
    }

    const btnResetKnn = document.getElementById('btnResetKnn');
    if (btnResetKnn) {
      btnResetKnn.addEventListener('click', resetKnn);
    }

    const btnResetKnnSide = document.getElementById('btnResetKnnSide');
    if (btnResetKnnSide) {
      btnResetKnnSide.addEventListener('click', resetKnn);
    }

    const sliderKnnK = document.getElementById('sliderKnnK');
    const inputKnnK = document.getElementById('inputKnnK');
    if (sliderKnnK) {
      sliderKnnK.addEventListener('input', (e) => {
        knnK = parseInt(e.target.value, 10);
        if (inputKnnK) inputKnnK.value = knnK;
        if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
          datasetSlots[activeDatasetSlot].knnK = knnK;
        }
        updateCliCommand();
        if (typeof updateDatasetStatusBadge === 'function') {
          updateDatasetStatusBadge();
        }
        draw();
      });
    }
    if (inputKnnK) {
      inputKnnK.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v >= 1) {
          knnK = v;
          if (sliderKnnK) sliderKnnK.value = Math.min(100, v);
          if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
            datasetSlots[activeDatasetSlot].knnK = knnK;
          }
          updateCliCommand();
          if (typeof updateDatasetStatusBadge === 'function') {
            updateDatasetStatusBadge();
          }
          draw();
        }
      });
    }

    const sliderKnnDtmin = document.getElementById('sliderKnnDtmin');
    const inputKnnDtmin = document.getElementById('inputKnnDtmin');
    if (sliderKnnDtmin) {
      sliderKnnDtmin.addEventListener('input', (e) => {
        knnDtmin = parseInt(e.target.value, 10);
        if (inputKnnDtmin) inputKnnDtmin.value = knnDtmin;
        updateCliCommand();
        draw();
      });
    }
    if (inputKnnDtmin) {
      inputKnnDtmin.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v >= 0) {
          knnDtmin = v;
          if (sliderKnnDtmin) sliderKnnDtmin.value = Math.min(25, v);
          updateCliCommand();
          draw();
        }
      });
    }

    ['knnDirAll', 'knnDirPast', 'knnDirFuture'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('click', () => {
          ['knnDirAll', 'knnDirPast', 'knnDirFuture'].forEach(otherId => {
            const other = document.getElementById(otherId);
            if (other) other.classList.remove('active');
          });
          el.classList.add('active');
          if (id === 'knnDirPast') knnDirection = 'past';
          else if (id === 'knnDirFuture') knnDirection = 'future';
          else knnDirection = 'all';
          updateCliCommand();
          draw();
        });
      }
    });

    const sliderKnnEps = document.getElementById('sliderKnnEps');
    const inputKnnEps = document.getElementById('inputKnnEps');
    if (sliderKnnEps) {
      sliderKnnEps.addEventListener('input', (e) => {
        knnEpsilon = parseFloat(e.target.value);
        if (inputKnnEps) inputKnnEps.value = knnEpsilon.toFixed(2);
        updateCliCommand();
        draw();
      });
    }
    if (inputKnnEps) {
      inputKnnEps.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v >= 0.0) {
          knnEpsilon = v;
          if (sliderKnnEps) sliderKnnEps.value = Math.min(0.30, v);
          updateCliCommand();
          draw();
        }
      });
    }

    const btnKnnMultiPivot = document.getElementById('btnKnnMultiPivot');
    if (btnKnnMultiPivot) {
      btnKnnMultiPivot.addEventListener('click', () => {
        knnMvp = !knnMvp;
        btnKnnMultiPivot.classList.toggle('toggle-active', knnMvp);
        btnKnnMultiPivot.classList.toggle('toggle-cyan', knnMvp);
        btnKnnMultiPivot.classList.toggle('active', knnMvp);
        updateCliCommand();
        draw();
      });
    }

    const btnKnnSq8 = document.getElementById('btnKnnSq8');
    if (btnKnnSq8) {
      btnKnnSq8.addEventListener('click', () => {
        knnUseSq8 = !knnUseSq8;
        btnKnnSq8.classList.toggle('toggle-active', knnUseSq8);
        btnKnnSq8.classList.toggle('toggle-cyan', knnUseSq8);
        btnKnnSq8.classList.toggle('active', knnUseSq8);
        updateCliCommand();
        draw();
      });
    }

    // =========================================================
    //  Dim & Density (gric-dimdensity) Controls
    // =========================================================

    /**
     * Execute gric-dimdensity via native CLI.
     */
    async function executeDimDensityComputation() {
      if (isDimDensityComputing) return;
      if (!knnResults) {
        showToast(
          '⚠️ Compute k-NN first before running ' +
          'Dim & Density.'
        );
        return;
      }

      isDimDensityComputing = true;
      const targetSlot = activeDatasetSlot;
      const badge = document.getElementById(
        'dimDensStatusBadge'
      );
      if (badge) {
        badge.textContent = '● Running';
        badge.style.color = '#fbbf24';
        badge.style.background =
          'rgba(251, 191, 36, 0.2)';
      }

      const selCli = document.getElementById(
        'selectCliDataset'
      );
      let datasetName = selCli ? selCli.value : '';
      if (!datasetName) {
        datasetName = `${currentBenchmark}.txt`;
      }
      const dsBase = datasetName.replace(
        /\.(txt|csv|fits|dat|mp4|fits\.fz)$/i, ''
      );
      const clusterDir = `${dsBase}.clusterdat`;

      const args = [
        clusterDir,
        '-json',
        '-o', `${clusterDir}/dimdensity.txt`,
        '-k', String(knnK)
      ];

      const chkRange = document.getElementById(
        'chkDimDensMultiRange'
      );
      const inKmin = document.getElementById(
        'dimDensKmin'
      );
      const inKmax = document.getElementById(
        'dimDensKmax'
      );
      if (chkRange && chkRange.checked) {
        const kmin = parseInt(inKmin?.value || '5', 10) || 5;
        const kmax = parseInt(inKmax?.value || '20', 10) || knnK;
        args.push('-kmin', String(kmin), '-kmax', String(kmax), '-range');
      }

      const consoleEl = document.getElementById(
        'cliConsoleLog'
      );
      if (consoleEl) {
        consoleEl.textContent +=
          `\n🔬 Running: gric-dimdensity ` +
          `${args.join(' ')}\n`;
        consoleEl.scrollTop = consoleEl.scrollHeight;
      }

      showToast(
        '🔬 Running gric-dimdensity estimation...'
      );

      let rawOutput = '';
      let finalExitCode = 0;

      await new Promise((resolve) => {
        DesktopBridge.runCliJob({
          cmd: 'gric-dimdensity',
          args: args,
          onOutput: (chunk) => {
            rawOutput += chunk;
            if (consoleEl) {
              consoleEl.textContent += chunk;
              consoleEl.scrollTop =
                consoleEl.scrollHeight;
            }
          },
          onTelemetry: () => {},
          onFinish: (res) => {
            finalExitCode = res?.exitCode ?? 0;
            resolve();
          }
        }).catch((err) => {
          showToast(
            '⚠️ gric-dimdensity failed: ' +
            err.message
          );
          resolve();
        });
      });

      if (finalExitCode !== 0) {
        showToast(
          `⚠️ gric-dimdensity exited (${finalExitCode})`
        );
        isDimDensityComputing = false;
        if (badge) {
          badge.textContent = 'Error';
          badge.style.color = '#f87171';
        }
        return;
      }

      // Parse JSON summary and read per-sample binary results
      const parsedSummary = DesktopBridge.parseDimDensityLog(rawOutput);
      const parsedResults = await DesktopBridge.readDimDensityResults(clusterDir);

      isDimDensityComputing = false;

      if (datasetSlots[targetSlot]) {
        datasetSlots[targetSlot].dimDensitySummary = parsedSummary;
        datasetSlots[targetSlot].dimDensityResults = parsedResults;
      }

      if (activeDatasetSlot === targetSlot) {
        dimDensitySummary = parsedSummary;
        dimDensityResults = parsedResults;

        if (!dimDensitySummary && !dimDensityResults) {
          showToast('⚠️ Could not parse results.');
          if (badge) {
            badge.textContent = 'Error';
            badge.style.color = '#f87171';
          }
          return;
        }

        if (badge) {
          badge.textContent = '✓ Done';
          badge.style.color = '#4ade80';
          badge.style.background = 'rgba(34, 197, 94, 0.15)';
        }

        // Show Card 7 and color toggle
        const card7 = document.getElementById('cardDimDensity');
        if (card7) {
          card7.style.display = '';
          card7.classList.remove('collapsed');
        }
        const cg = document.getElementById('dimDensityColorGroup');
        if (cg) cg.style.display = 'flex';

        renderDimDensityDashboard();
        draw();
        showToast('✅ Dim & Density analysis complete!');
      }
    }

    /**
     * Populate Card 7 dashboard from summary results.
     */
    function renderDimDensityDashboard() {
      const s = dimDensitySummary;
      const setTxt = (id, txt) => {
        const el = document.getElementById(id);
        if (el) el.textContent = txt;
      };

      if (!s) {
        setTxt('dimDensDimMedian', '--');
        setTxt('dimDensDimMeanUnb', '--');
        setTxt('dimDensLogDensMedian', '--');
        setTxt('dimDensDimRange', '--');
        setTxt('dimDensDensRange', '--');
        setTxt('dimDensExecTime', '-- ms');
        ['dimMedPctP10', 'dimMedPctP25', 'dimMedPctP50', 'dimMedPctP75', 'dimMedPctP90',
         'dimMeanPctP10', 'dimMeanPctP25', 'dimMeanPctP50', 'dimMeanPctP75', 'dimMeanPctP90',
         'ldPctP10', 'ldPctP25', 'ldPctP50', 'ldPctP75', 'ldPctP90'].forEach(id => setTxt(id, '--'));
        const badge = document.getElementById('dimDensStatusBadge');
        if (badge) {
          badge.textContent = 'Idle';
          badge.style.color = 'var(--text-muted)';
        }
        drawDimDensHistograms();
        populateDimDensTraceTable();
        return;
      }

      const dimMed = s.intrinsic_dimension_median_unbiased || s.intrinsic_dimension || {};
      const dimMean = s.intrinsic_dimension_mean_unbiased || s.intrinsic_dimension || {};
      const ld = s.log_density || {};
      const dens = s.local_density || {};
      const dimMedP = dimMed.percentiles || {};
      const dimMeanP = dimMean.percentiles || {};
      const ldP = ld.percentiles || {};

      const iqrMed = ((dimMedP.p75 || 0) - (dimMedP.p25 || 0)).toFixed(2);
      setTxt('dimDensDimMedian',
        `${(dimMedP.p50 || 0).toFixed(2)} ±${iqrMed}`);

      const stdMean = (dimMean.std_dev || 0).toFixed(2);
      setTxt('dimDensDimMeanUnb',
        `${(dimMean.mean || 0).toFixed(2)} ±${stdMean}`);

      setTxt('dimDensLogDensMedian',
        (ldP.p50 || 0).toFixed(2));
      setTxt('dimDensDimRange',
        `${(dimMed.min || 0).toFixed(1)} – ${(dimMed.max || 0).toFixed(1)}`);
      setTxt('dimDensDensRange',
        `${(dens.min || 0).toExponential(1)} – ${(dens.max || 0).toExponential(1)}`);
      setTxt('dimDensExecTime',
        `${(s.execution_time_ms || 0).toFixed(1)} ms`);

      // Percentile table for both estimators
      setTxt('dimMedPctP10', (dimMedP.p10 || 0).toFixed(2));
      setTxt('dimMedPctP25', (dimMedP.p25 || 0).toFixed(2));
      setTxt('dimMedPctP50', (dimMedP.p50 || 0).toFixed(2));
      setTxt('dimMedPctP75', (dimMedP.p75 || 0).toFixed(2));
      setTxt('dimMedPctP90', (dimMedP.p90 || 0).toFixed(2));

      const badge = document.getElementById('dimDensStatusBadge');
      if (badge) {
        badge.textContent = 'Computed';
        badge.style.color = '#4ade80';
      }

      setTxt('dimMeanPctP10', (dimMeanP.p10 || 0).toFixed(2));
      setTxt('dimMeanPctP25', (dimMeanP.p25 || 0).toFixed(2));
      setTxt('dimMeanPctP50', (dimMeanP.p50 || 0).toFixed(2));
      setTxt('dimMeanPctP75', (dimMeanP.p75 || 0).toFixed(2));
      setTxt('dimMeanPctP90', (dimMeanP.p90 || 0).toFixed(2));

      setTxt('ldPctP10', (ldP.p10 || 0).toFixed(2));
      setTxt('ldPctP25', (ldP.p25 || 0).toFixed(2));
      setTxt('ldPctP50', (ldP.p50 || 0).toFixed(2));
      setTxt('ldPctP75', (ldP.p75 || 0).toFixed(2));
      setTxt('ldPctP90', (ldP.p90 || 0).toFixed(2));

      // Draw histograms
      drawDimDensHistograms();

      // Populate trace table
      dimDensityTraceOffset = 0;
      populateDimDensTraceTable();
    }

    // Cached 2D heatmap metadata for interactive hover
    let dimDensHeatmapMeta = null;

    /**
     * Draw 2D heatmap of occurrences centered around average density and dimension.
     */
    function drawDimDensHeatmap(hoverBx = -1, hoverBy = -1) {
      const cvs = document.getElementById('canvasDimDensHeatmap');
      if (!cvs || !dimDensityResults) return;

      const dd = dimDensityResults;
      const N = dd.totalFrames;
      if (N === 0) return;

      const ctx2 = cvs.getContext('2d');
      const W = cvs.width;
      const H = cvs.height;
      ctx2.clearRect(0, 0, W, H);

      // Compute mean and standard deviation for X (logDensity) and Y (localDim)
      let sumX = 0, sumY = 0, validN = 0;
      for (let i = 0; i < N; i++) {
        const xVal = dd.logDensity[i];
        const yVal = dd.localDim[i];
        if (isFinite(xVal) && isFinite(yVal)) {
          sumX += xVal;
          sumY += yVal;
          validN++;
        }
      }
      if (validN === 0) return;

      const xAvg = sumX / validN;
      const yAvg = sumY / validN;

      let varX = 0, varY = 0;
      for (let i = 0; i < N; i++) {
        const xVal = dd.logDensity[i];
        const yVal = dd.localDim[i];
        if (isFinite(xVal) && isFinite(yVal)) {
          varX += (xVal - xAvg) * (xVal - xAvg);
          varY += (yVal - yAvg) * (yVal - yAvg);
        }
      }
      const xStd = Math.sqrt(varX / validN);
      const yStd = Math.sqrt(varY / validN);

      // Focus region: centered around average (+/- 2.5 std devs)
      const halfSpanX = Math.max(0.6, 2.5 * (xStd || 1.0));
      const halfSpanY = Math.max(0.3, 2.5 * (yStd || 1.0));

      const xMin = xAvg - halfSpanX;
      const xMax = xAvg + halfSpanX;
      const yMin = yAvg - halfSpanY;
      const yMax = yAvg + halfSpanY;

      const Nx = 24; // density bins (X axis)
      const Ny = 18; // dimension bins (Y axis)
      const rangeX = xMax - xMin;
      const rangeY = yMax - yMin;

      const counts = [];
      for (let x = 0; x < Nx; x++) {
        counts[x] = new Array(Ny).fill(0);
      }

      let inWindowCount = 0;
      for (let i = 0; i < N; i++) {
        const xVal = dd.logDensity[i];
        const yVal = dd.localDim[i];
        if (!isFinite(xVal) || !isFinite(yVal)) continue;
        if (xVal < xMin || xVal > xMax || yVal < yMin || yVal > yMax) continue;

        let bx = Math.floor((xVal - xMin) / rangeX * Nx);
        let by = Math.floor((yVal - yMin) / rangeY * Ny);
        if (bx < 0) bx = 0;
        if (bx >= Nx) bx = Nx - 1;
        if (by < 0) by = 0;
        if (by >= Ny) by = Ny - 1;
        counts[bx][by]++;
        inWindowCount++;
      }

      let maxC = 0;
      for (let x = 0; x < Nx; x++) {
        for (let y = 0; y < Ny; y++) {
          if (counts[x][y] > maxC) maxC = counts[x][y];
        }
      }
      if (maxC === 0) maxC = 1;

      dimDensHeatmapMeta = {
        xMin, xMax, yMin, yMax, xAvg, yAvg, Nx, Ny, counts, maxC, N,
        rangeX, rangeY, inWindowCount
      };

      const padL = 34;
      const padR = 36;
      const padT = 10;
      const padB = 24;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const cellW = plotW / Nx;
      const cellH = plotH / Ny;

      const stops = [
        [15, 23, 42],    // t=0.0 dark navy
        [40, 11, 84],    // t=0.15 dark purple
        [101, 0, 167],   // t=0.30 violet
        [165, 53, 112],  // t=0.50 magenta
        [224, 116, 38],  // t=0.70 orange
        [250, 219, 67],  // t=0.88 yellow
        [255, 255, 255]  // t=1.00 white
      ];

      function getCellColor(cnt) {
        if (cnt === 0) return 'rgba(30, 41, 59, 0.4)';
        const t = Math.pow(cnt / maxC, 0.55);
        const n = stops.length - 1;
        const fi = Math.max(0, Math.min(1, t)) * n;
        const lo = Math.floor(fi);
        const hi = Math.min(lo + 1, n);
        const f = fi - lo;
        const c0 = stops[lo];
        const c1 = stops[hi];
        const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
        const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
        const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
        return `rgb(${r},${g},${b})`;
      }

      // Draw 2D cells
      for (let bx = 0; bx < Nx; bx++) {
        for (let by = 0; by < Ny; by++) {
          const cnt = counts[bx][by];
          const x = padL + bx * cellW;
          const y = padT + plotH - (by + 1) * cellH;
          ctx2.fillStyle = getCellColor(cnt);
          ctx2.fillRect(x, y, cellW - 0.5, cellH - 0.5);

          if (bx === hoverBx && by === hoverBy) {
            ctx2.strokeStyle = '#38bdf8';
            ctx2.lineWidth = 1.5;
            ctx2.strokeRect(x - 0.5, y - 0.5, cellW, cellH);
          }
        }
      }

      // Center crosshair lines passing through average (xAvg, yAvg)
      ctx2.save();
      ctx2.setLineDash([3, 3]);
      ctx2.strokeStyle = 'rgba(251, 191, 36, 0.35)';
      ctx2.lineWidth = 1;
      // Vertical line at xAvg
      ctx2.beginPath();
      ctx2.moveTo(padL + plotW / 2, padT);
      ctx2.lineTo(padL + plotW / 2, padT + plotH);
      ctx2.stroke();
      // Horizontal line at yAvg
      ctx2.beginPath();
      ctx2.moveTo(padL, padT + plotH / 2);
      ctx2.lineTo(padL + plotW, padT + plotH / 2);
      ctx2.stroke();
      ctx2.restore();

      // Bounding box
      ctx2.strokeStyle = 'rgba(148, 163, 184, 0.3)';
      ctx2.lineWidth = 1;
      ctx2.strokeRect(padL, padT, plotW, plotH);

      // Y-axis ticks & labels (left)
      ctx2.fillStyle = '#94a3b8';
      ctx2.font = '8px monospace';
      ctx2.textAlign = 'right';
      ctx2.fillText(yMax.toFixed(1), padL - 3, padT + 8);
      ctx2.fillStyle = '#fbbf24';
      ctx2.fillText(yAvg.toFixed(1), padL - 3, padT + plotH / 2 + 3);
      ctx2.fillStyle = '#94a3b8';
      ctx2.fillText(yMin.toFixed(1), padL - 3, padT + plotH);

      ctx2.fillStyle = '#fbbf24';
      ctx2.font = '8px sans-serif';
      ctx2.textAlign = 'center';
      ctx2.fillText('Dim ↑', 12, padT + plotH / 2 + 3);

      // X-axis ticks & labels (bottom)
      ctx2.fillStyle = '#94a3b8';
      ctx2.font = '8px monospace';
      ctx2.textAlign = 'left';
      ctx2.fillText(xMin.toFixed(1), padL, H - 12);
      ctx2.textAlign = 'center';
      ctx2.fillStyle = '#c084fc';
      ctx2.fillText(xAvg.toFixed(1), padL + plotW / 2, H - 12);
      ctx2.fillStyle = '#94a3b8';
      ctx2.textAlign = 'right';
      ctx2.fillText(xMax.toFixed(1), padL + plotW, H - 12);

      ctx2.fillStyle = '#c084fc';
      ctx2.font = '8px sans-serif';
      ctx2.textAlign = 'center';
      ctx2.fillText('Log-Density ln(f) →', padL + plotW / 2, H - 2);

      // Right color-bar legend
      const barX = W - padR + 8;
      const barY = padT;
      const barW = 8;
      const barH = plotH;
      const nGrad = 30;
      const dH = barH / nGrad;
      for (let i = 0; i < nGrad; i++) {
        const frac = (nGrad - 1 - i) / (nGrad - 1);
        const cnt = frac * maxC;
        ctx2.fillStyle = getCellColor(cnt);
        ctx2.fillRect(barX, barY + i * dH, barW, dH + 0.5);
      }
      ctx2.strokeStyle = 'rgba(148, 163, 184, 0.3)';
      ctx2.strokeRect(barX, barY, barW, barH);

      ctx2.fillStyle = '#94a3b8';
      ctx2.font = '7px monospace';
      ctx2.textAlign = 'left';
      const maxCLabel = maxC > 999 ? (maxC / 1000).toFixed(1) + 'k' : maxC;
      ctx2.fillText(maxCLabel, barX + 10, barY + 6);
      ctx2.fillText('0', barX + 10, barY + barH);
      ctx2.fillStyle = '#38bdf8';
      ctx2.font = '7px sans-serif';
      ctx2.fillText('Count', barX + 1, barY - 2);
    }

    /**
     * Draw 2D heatmap and 1D histograms.
     */
    function drawDimDensHistograms() {
      if (!dimDensityResults) {
        ['canvasDimDensHeatmap', 'canvasDimHist', 'canvasDensHist'].forEach(id => {
          const cvs = document.getElementById(id);
          if (cvs) {
            const ctx2 = cvs.getContext('2d');
            ctx2.clearRect(0, 0, cvs.width, cvs.height);
          }
        });
        return;
      }
      const dd = dimDensityResults;

      drawDimDensHeatmap();
      drawHistOnCanvas(
        'canvasDimHist',
        dd.localDim, dd.totalFrames,
        '#fbbf24', 'Local Dimension'
      );
      drawHistOnCanvas(
        'canvasDensHist',
        dd.logDensity, dd.totalFrames,
        '#c084fc', 'Log-Density'
      );
    }

    /**
     * Draw a single histogram bar chart centered around the average value.
     */
    function drawHistOnCanvas(
      canvasId, data, N, color, label
    ) {
      const cvs = document.getElementById(canvasId);
      if (!cvs) return;
      const ctx2 = cvs.getContext('2d');
      const W = cvs.width;
      const H = cvs.height;
      ctx2.clearRect(0, 0, W, H);
      if (!data || N === 0) return;

      let sum = 0, validN = 0;
      for (let i = 0; i < N; i++) {
        const v = data[i];
        if (isFinite(v)) {
          sum += v;
          validN++;
        }
      }
      if (validN === 0) return;

      const avg = sum / validN;

      let variance = 0;
      for (let i = 0; i < N; i++) {
        const v = data[i];
        if (isFinite(v)) {
          variance += (v - avg) * (v - avg);
        }
      }
      const std = Math.sqrt(variance / validN);

      // Centered window around average (+/- 2.5 std devs)
      const halfSpan = Math.max(0.4, 2.5 * (std || 1.0));
      const mn = avg - halfSpan;
      const mx = avg + halfSpan;
      const range = mx - mn;

      const bins = 20;
      const counts = new Array(bins).fill(0);
      for (let i = 0; i < N; i++) {
        const v = data[i];
        if (!isFinite(v) || v < mn || v > mx) continue;
        let b = Math.floor((v - mn) / range * bins);
        if (b >= bins) b = bins - 1;
        if (b < 0) b = 0;
        counts[b]++;
      }

      let maxC = 0;
      for (let b = 0; b < bins; b++) {
        if (counts[b] > maxC) maxC = counts[b];
      }
      if (maxC === 0) maxC = 1;

      const padL = 34;
      const padR = 14;
      const padB = 22;
      const padT = 8;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const barW = plotW / bins;

      // Draw background bounding box
      ctx2.strokeStyle = 'rgba(148, 163, 184, 0.2)';
      ctx2.lineWidth = 1;
      ctx2.strokeRect(padL, padT, plotW, plotH);

      // Draw bars
      ctx2.fillStyle = color;
      ctx2.globalAlpha = 0.75;
      for (let b = 0; b < bins; b++) {
        const barH = (counts[b] / maxC) * plotH;
        const x = padL + b * barW + 1;
        const y = padT + plotH - barH;
        ctx2.fillRect(x, y, Math.max(1, barW - 2), barH);
      }
      ctx2.globalAlpha = 1.0;

      // Center dashed line at average
      ctx2.save();
      ctx2.setLineDash([2, 2]);
      ctx2.strokeStyle = 'rgba(251, 191, 36, 0.6)';
      ctx2.lineWidth = 1;
      ctx2.beginPath();
      ctx2.moveTo(padL + plotW / 2, padT);
      ctx2.lineTo(padL + plotW / 2, padT + plotH);
      ctx2.stroke();
      ctx2.restore();

      // Axis labels
      ctx2.fillStyle = '#94a3b8';
      ctx2.font = '8px monospace';
      ctx2.textAlign = 'left';
      ctx2.fillText(mn.toFixed(1), padL, H - 8);

      ctx2.fillStyle = '#fbbf24';
      ctx2.textAlign = 'center';
      ctx2.fillText(`μ=${avg.toFixed(1)}`, padL + plotW / 2, H - 8);

      ctx2.fillStyle = '#94a3b8';
      ctx2.textAlign = 'right';
      ctx2.fillText(mx.toFixed(1), padL + plotW, H - 8);

      // Title/Label on top left
      ctx2.fillStyle = color;
      ctx2.font = '8px sans-serif';
      ctx2.textAlign = 'left';
      ctx2.fillText(label, padL + 4, padT + 10);
    }

    /**
     * Populate per-sample trace table rows.
     */
    function populateDimDensTraceTable() {
      const tbody = document.getElementById(
        'dimDensTraceBody'
      );
      const title = document.getElementById(
        'dimDensTraceTitle'
      );
      const dd = dimDensityResults;

      if (!tbody) return;

      if (!dd) {
        tbody.innerHTML = '<tr><td colspan="5" style="padding: 12px; text-align: center; color: var(--text-muted);">Run gric-dimdensity to compute local dimensionality & density</td></tr>';
        if (title) title.textContent = 'Per-Sample Trace (0 samples)';
        return;
      }

      if (title) {
        title.textContent =
          `Per-Sample Trace (${dd.totalFrames}` +
          ` samples)`;
      }

      if (dimDensityTraceOffset === 0) {
        tbody.innerHTML = '';
      }

      const start = dimDensityTraceOffset;
      const end = Math.min(
        start + 200, dd.totalFrames
      );

      for (let i = start; i < end; i++) {
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        tr.style.color = '#e2e8f0';
        tr.dataset.sampleId = i;
        tr.addEventListener('click', () => {
          selectedKnnQuerySample = i;
          draw();
          if (typeof renderKnnTrace === 'function') {
            renderKnnTrace();
          }
        });
        tr.addEventListener('mouseenter', () => {
          tr.style.background =
            'rgba(251, 191, 36, 0.1)';
        });
        tr.addEventListener('mouseleave', () => {
          tr.style.background = '';
        });

        const mkTd = (txt, align) => {
          const td = document.createElement('td');
          td.style.padding = '3px 6px';
          td.style.textAlign = align || 'right';
          td.textContent = txt;
          return td;
        };

        const dimMedVal = (dd.localDimMed ? dd.localDimMed[i] : dd.localDim[i]) || 0;
        const dimMeanVal = (dd.localDimMean ? dd.localDimMean[i] : dd.localDim[i]) || 0;

        tr.appendChild(mkTd(String(i), 'left'));
        tr.appendChild(mkTd(dimMedVal.toFixed(3)));
        tr.appendChild(mkTd(dimMeanVal.toFixed(3)));
        tr.appendChild(mkTd(dd.density[i].toExponential(2)));
        tr.appendChild(mkTd(dd.logDensity[i].toFixed(3)));
        tr.appendChild(mkTd(dd.rkDist[i].toFixed(4)));
        tbody.appendChild(tr);
      }

      dimDensityTraceOffset = end;

      // Show/hide Load More button
      const btnMore = document.getElementById(
        'btnDimDensLoadMore'
      );
      if (btnMore) {
        btnMore.style.display =
          (end < dd.totalFrames) ? '' : 'none';
      }
    }

    /**
     * Sync color toggle button UI state.
     */
    function syncColorToggleUI() {
      const ids = [
        'colorByCluster',
        'colorByDim',
        'colorByDensity'
      ];
      const modes = [
        'cluster',
        'dimension',
        'density'
      ];
      for (let i = 0; i < ids.length; i++) {
        const el = document.getElementById(ids[i]);
        if (!el) continue;
        if (modes[i] === pointColorMode) {
          el.classList.add('active');
        } else {
          el.classList.remove('active');
        }
      }
    }

    // --- Button Handlers ---

    const btnRunDD = document.getElementById(
      'btnRunDimDensity'
    );
    if (btnRunDD) {
      btnRunDD.addEventListener(
        'click',
        executeDimDensityComputation
      );
    }

    const btnDDMore = document.getElementById(
      'btnDimDensLoadMore'
    );
    if (btnDDMore) {
      btnDDMore.addEventListener('click', () => {
        populateDimDensTraceTable();
      });
    }

    // Color toggle handlers
    ['colorByCluster', 'colorByDim',
     'colorByDensity'].forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('click', () => {
        const map = {
          colorByCluster: 'cluster',
          colorByDim: 'dimension',
          colorByDensity: 'density'
        };
        pointColorMode = map[id] || 'cluster';
        syncColorToggleUI();
        draw();
      });
    });

    // Card 7 tab switching
    ['tabDimDensSummary', 'tabDimDensDistrib',
     'tabDimDensTrace'].forEach((tabId) => {
      const tab = document.getElementById(tabId);
      if (!tab) return;
      tab.addEventListener('click', () => {
        const panels = {
          tabDimDensSummary: 'dimDensSummaryPanel',
          tabDimDensDistrib: 'dimDensDistribPanel',
          tabDimDensTrace: 'dimDensTracePanel'
        };
        // Deactivate all tabs
        ['tabDimDensSummary', 'tabDimDensDistrib',
         'tabDimDensTrace'].forEach((tid) => {
          const t = document.getElementById(tid);
          if (t) t.classList.remove('active');
        });
        // Hide all panels
        Object.values(panels).forEach((pid) => {
          const p = document.getElementById(pid);
          if (p) p.style.display = 'none';
        });
        // Activate clicked tab and panel
        tab.classList.add('active');
        const pid = panels[tabId];
        const p = document.getElementById(pid);
        if (p) p.style.display = '';

        if (tabId === 'tabDimDensDistrib') {
          setTimeout(drawDimDensHistograms, 10);
        }
      });
    });

    // 2D Heatmap interactive hover
    const heatCvs = document.getElementById('canvasDimDensHeatmap');
    if (heatCvs) {
      heatCvs.addEventListener('mousemove', (e) => {
        if (!dimDensHeatmapMeta) return;
        const rect = heatCvs.getBoundingClientRect();
        const scaleX = heatCvs.width / (rect.width || 1);
        const scaleY = heatCvs.height / (rect.height || 1);
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;

        const padL = 34, padR = 36, padT = 10, padB = 24;
        const plotW = heatCvs.width - padL - padR;
        const plotH = heatCvs.height - padT - padB;
        const cellW = plotW / dimDensHeatmapMeta.Nx;
        const cellH = plotH / dimDensHeatmapMeta.Ny;

        if (mx >= padL && mx <= padL + plotW && my >= padT && my <= padT + plotH) {
          const bx = Math.min(
            dimDensHeatmapMeta.Nx - 1,
            Math.max(0, Math.floor((mx - padL) / cellW))
          );
          const by = Math.min(
            dimDensHeatmapMeta.Ny - 1,
            Math.max(0, Math.floor((padT + plotH - my) / cellH))
          );
          const cnt = dimDensHeatmapMeta.counts[bx][by];
          const pct = ((cnt / dimDensHeatmapMeta.N) * 100).toFixed(1);
          const meta = dimDensHeatmapMeta;
          const x0 = (meta.xMin + bx * meta.rangeX / meta.Nx).toFixed(1);
          const x1 = (meta.xMin + (bx + 1) * meta.rangeX / meta.Nx).toFixed(1);
          const y0 = (meta.yMin + by * meta.rangeY / meta.Ny).toFixed(1);
          const y1 = (meta.yMin + (by + 1) * meta.rangeY / meta.Ny).toFixed(1);

          const hoverEl = document.getElementById('dimDensHeatmapHover');
          if (hoverEl) {
            hoverEl.textContent =
              `ln(f):[${x0}..${x1}] Dim:[${y0}..${y1}] ${cnt} (${pct}%)`;
          }
          drawDimDensHeatmap(bx, by);
        } else {
          const hoverEl = document.getElementById('dimDensHeatmapHover');
          if (hoverEl) hoverEl.textContent = 'Hover grid for counts';
          drawDimDensHeatmap(-1, -1);
        }
      });

      heatCvs.addEventListener('mouseleave', () => {
        const hoverEl = document.getElementById('dimDensHeatmapHover');
        if (hoverEl) hoverEl.textContent = 'Hover grid for counts';
        drawDimDensHeatmap(-1, -1);
      });
    }

    // =========================================================================
    //  k-NN RECONSTRUCTION ENGINE & MULTI-DATASET INTERPOLATOR
    // =========================================================================

    /**
     * Executes non-parametric k-NN reconstruction:
     * Mapping Input A -> Output B, evaluating queries C to reconstruct D.
     */
    async function executeDatasetReconstruction() {
      // 1. Validate Slots
      const slotA = datasetSlots['A'];
      const slotB = datasetSlots['B'];
      const slotC = datasetSlots['C'];
      const slotD = datasetSlots['D'];

      const ptsA = slotA ? slotA.benchmarkDataset : null;
      const ptsB = slotB ? slotB.benchmarkDataset : null;
      const ptsC = slotC ? slotC.benchmarkDataset : null;

      if (!ptsA || ptsA.length === 0) {
        showToast('⚠️ Reconstruction Error: Training Input Dataset [A] is empty or not staged.');
        return;
      }
      if (!ptsB || ptsB.length === 0) {
        showToast('⚠️ Reconstruction Error: Training Output Dataset [B] is empty or not staged.');
        return;
      }
      if (!ptsC || ptsC.length === 0) {
        showToast('⚠️ Reconstruction Error: Query Input Dataset [C] is empty or not staged.');
        return;
      }
      if (ptsA.length !== ptsB.length) {
        showToast(`⚠️ Reconstruction Error: Sample count mismatch between [A] (${ptsA.length.toLocaleString()}) and [B] (${ptsB.length.toLocaleString()}). A and B must have identical sample counts.`);
        return;
      }

      function detectSlotDim(slot, pts) {
        if (slot && slot.benchmarkKey && typeof getBenchmarkDim === 'function') {
          return getBenchmarkDim(slot.benchmarkKey);
        }
        if (slot && slot.currentDim) {
          return slot.currentDim;
        }
        if (pts && pts.length > 0) {
          if (pts[0].coords && pts[0].coords.length > 0) {
            return pts[0].coords.length;
          }
          for (let i = 0; i < Math.min(200, pts.length); i++) {
            if (pts[i].z !== undefined && Math.abs(pts[i].z) > 1e-6) return 3;
          }
        }
        return 2;
      }

      const dimA = detectSlotDim(slotA, ptsA);
      const dimB = detectSlotDim(slotB, ptsB);
      const dimC = detectSlotDim(slotC, ptsC);

      if (dimA !== dimC) {
        showToast(`⚠️ Reconstruction Error: Coordinate dimension mismatch between Input [A] (${dimA}D) and Query [C] (${dimC}D). They must match.`);
        return;
      }

      // 2. Read Parameters
      const inputK = document.getElementById('inputReconK');
      const k = inputK ? Math.max(1, Math.min(parseInt(inputK.value, 10) || 30, ptsA.length)) : 30;
      const selectWeight = document.getElementById('selectReconWeight');
      const weightMode = selectWeight ? selectWeight.value : 'uniform';
      const inputAlpha = document.getElementById('inputReconAlpha');
      const alpha = inputAlpha ? Math.max(0.1, parseFloat(inputAlpha.value) || 1.0) : 1.0;

      const numQueries = ptsC.length;
      const numCandidates = ptsA.length;

      // 3. In Native C mode, automatically run the compiled native C gric-knn -query C if not already cached
      if (DesktopBridge.isNativeSupported()) {
        const hasValidNativeKnn = !!(
          slotC && slotC.knnResults && slotC.knnResultsForQuery &&
          slotC.knnResults.indices &&
          slotC.knnResults.totalFrames === numQueries &&
          slotC.knnResults.k >= k
        );

        if (!hasValidNativeKnn) {
          showToast(`⚡ Running native compiled C gric-knn -query C (k=${k})...`);
          await runNativeReconQueryKnn();
        }
      }

      const activeSlotC = datasetSlots['C'] || slotC;
      const nativeKnn = (activeSlotC && activeSlotC.knnResults && activeSlotC.knnResultsForQuery &&
                         activeSlotC.knnResults.indices && activeSlotC.knnResults.totalFrames === numQueries)
        ? activeSlotC.knnResults : null;

      if (DesktopBridge.isNativeSupported() && !nativeKnn) {
        showToast('⚠️ Native k-NN search could not load result indices.');
        return;
      }

      showToast(`⚡ Running k-NN Reconstruction (A: ${ptsA.length} pts in ${dimA}D, B: ${dimB}D target, C: ${ptsC.length} queries, k=${k})...`);

      const tStart = performance.now();
      const reconstructedPoints = new Array(numQueries);
      const sourceNeighbors = new Array(numQueries);
      let totalDistSum = 0.0;
      let totalDistCount = 0;

      if (nativeKnn) {
        /* Fast path: use pre-computed C-in-A indices from gric-knn -query */
        console.log(`[Reconstruction] Using pre-computed native k-NN results ` +
                    `(k=${nativeKnn.k}, N=${numQueries})`);
        const kUsed = Math.min(k, nativeKnn.k);
        for (let i = 0; i < numQueries; i++) {
          const offset = i * nativeKnn.k;
          const topNeighbors = [];

          /* Collect the top-k neighbor indices and distances */
          const rawDists = new Float64Array(kUsed);
          const rawIdx   = new Int32Array(kUsed);
          for (let p = 0; p < kUsed; p++) {
            rawDists[p] = nativeKnn.distances[offset + p];
            rawIdx[p]   = nativeKnn.indices[offset + p];
          }

          let avgX = 0.0, avgY = 0.0, avgZ = 0.0;
          if (weightMode === 'idw') {
            let exactMatchIdx = -1;
            for (let p = 0; p < kUsed; p++) {
              if (rawDists[p] < 1e-9) { exactMatchIdx = p; break; }
            }
            if (exactMatchIdx >= 0) {
              const pb = ptsB[rawIdx[exactMatchIdx]];
              avgX = pb.x; avgY = pb.y;
              avgZ = (dimB >= 3 && typeof pb.z === 'number') ? pb.z : 0.0;
              for (let p = 0; p < kUsed; p++) {
                topNeighbors.push({
                  id: rawIdx[p], dist: rawDists[p],
                  weight: (p === exactMatchIdx) ? 1.0 : 0.0
                });
                totalDistSum += rawDists[p];
                totalDistCount++;
              }
            } else {
              let sumW = 0.0;
              const weights = new Float64Array(kUsed);
              for (let p = 0; p < kUsed; p++) {
                const d = rawDists[p];
                const w = 1.0 / Math.pow(Math.max(d, 1e-7), alpha);
                weights[p] = w; sumW += w;
                totalDistSum += d; totalDistCount++;
              }
              for (let p = 0; p < kUsed; p++) {
                const normW = weights[p] / sumW;
                const pb = ptsB[rawIdx[p]];
                avgX += normW * pb.x; avgY += normW * pb.y;
                if (dimB >= 3 && typeof pb.z === 'number') { avgZ += normW * pb.z; }
                topNeighbors.push({ id: rawIdx[p], dist: rawDists[p], weight: normW });
              }
            }
          } else {
            // 'uniform': Boxcar filter averaging all k nearest neighbors
            const normW = 1.0 / kUsed;
            for (let p = 0; p < kUsed; p++) {
              const pb = ptsB[rawIdx[p]];
              avgX += normW * pb.x; avgY += normW * pb.y;
              if (dimB >= 3 && typeof pb.z === 'number') { avgZ += normW * pb.z; }
              topNeighbors.push({ id: rawIdx[p], dist: rawDists[p], weight: normW });
              totalDistSum += rawDists[p]; totalDistCount++;
            }
          }

          const reconPt = { x: avgX, y: avgY };
          if (dimB >= 3) { reconPt.z = avgZ; }
          reconstructedPoints[i] = reconPt;
          sourceNeighbors[i]     = topNeighbors;
        } // for each query (fast path)

      } else {
        /* Brute-force path: O(N_C × N_A) for each query in C */
        if (nativeKnn === null && slotC && slotC.knnResults) {
          console.log('[Reconstruction] Native k-NN results exist but dimensions mismatch; ' +
                      'falling back to brute-force.');
        }
        for (let i = 0; i < numQueries; i++) {
          const qc = ptsC[i];
          const qx = qc.x;
          const qy = qc.y;
          const qz = (dimA >= 3 && typeof qc.z === 'number') ? qc.z : 0.0;

          /* Compute distance to all candidates in A */
          const dists   = new Float64Array(numCandidates);
          const indices = new Int32Array(numCandidates);
          for (let j = 0; j < numCandidates; j++) {
            const pa = ptsA[j];
            const dx = qx - pa.x;
            const dy = qy - pa.y;
            const dz = (dimA >= 3 && typeof pa.z === 'number') ? (qz - pa.z) : 0.0;
            dists[j]   = Math.sqrt(dx * dx + dy * dy + dz * dz);
            indices[j] = j;
          }

          /* Partial sort top-k smallest distances */
          for (let p = 0; p < k; p++) {
            let minIdx = p;
            for (let j = p + 1; j < numCandidates; j++) {
              if (dists[j] < dists[minIdx]) { minIdx = j; }
            }
            const tmpD = dists[p]; dists[p] = dists[minIdx]; dists[minIdx] = tmpD;
            const tmpI = indices[p]; indices[p] = indices[minIdx]; indices[minIdx] = tmpI;
          }

          /* Calculate weights and average corresponding B samples */
          const topNeighbors = [];
          let avgX = 0.0, avgY = 0.0, avgZ = 0.0;

          if (weightMode === 'idw') {
            let exactMatchIdx = -1;
            for (let p = 0; p < k; p++) {
              if (dists[p] < 1e-9) { exactMatchIdx = p; break; }
            }
            if (exactMatchIdx >= 0) {
              const matchedSampleId = indices[exactMatchIdx];
              const pb = ptsB[matchedSampleId];
              avgX = pb.x; avgY = pb.y;
              avgZ = (dimB >= 3 && typeof pb.z === 'number') ? pb.z : 0.0;
              for (let p = 0; p < k; p++) {
                topNeighbors.push({
                  id: indices[p], dist: dists[p],
                  weight: (p === exactMatchIdx) ? 1.0 : 0.0
                });
                totalDistSum += dists[p]; totalDistCount++;
              }
            } else {
              let sumW = 0.0;
              const weights = new Float64Array(k);
              for (let p = 0; p < k; p++) {
                const d = dists[p];
                const w = 1.0 / Math.pow(Math.max(d, 1e-7), alpha);
                weights[p] = w; sumW += w;
                totalDistSum += d; totalDistCount++;
              }
              for (let p = 0; p < k; p++) {
                const normW = weights[p] / sumW;
                const pb = ptsB[indices[p]];
                avgX += normW * pb.x; avgY += normW * pb.y;
                if (dimB >= 3 && typeof pb.z === 'number') { avgZ += normW * pb.z; }
                topNeighbors.push({ id: indices[p], dist: dists[p], weight: normW });
              }
            }
          } else {
            // 'uniform': Boxcar filter averaging all k nearest neighbors
            const normW = 1.0 / k;
            for (let p = 0; p < k; p++) {
              const pb = ptsB[indices[p]];
              avgX += normW * pb.x; avgY += normW * pb.y;
              if (dimB >= 3 && typeof pb.z === 'number') { avgZ += normW * pb.z; }
              topNeighbors.push({ id: indices[p], dist: dists[p], weight: normW });
              totalDistSum += dists[p]; totalDistCount++;
            }
          }

          const reconPt = { x: avgX, y: avgY };
          if (dimB >= 3) { reconPt.z = avgZ; }
          reconstructedPoints[i] = reconPt;
          sourceNeighbors[i]     = topNeighbors;
        } // for each query (brute-force)
      } // if nativeKnn

      const elapsedMs = performance.now() - tStart;
      const avgNeighborDist =
        totalDistCount > 0 ? (totalDistSum / totalDistCount) : 0.0;

      // Compute quality metrics
      const reconKthDist = new Float64Array(numQueries);
      const reconVariance = new Float64Array(numQueries);
      let kthDistMin = Infinity, kthDistMax = -Infinity;
      let varMin = Infinity, varMax = -Infinity;

      for (let i = 0; i < numQueries; i++)
      {
        const neighbors = sourceNeighbors[i];
        if (!neighbors || neighbors.length === 0)
        {
          continue;
        }

        // k-th NN distance = farthest neighbor dist
        const kthD = neighbors[neighbors.length - 1].dist;
        reconKthDist[i] = kthD;
        if (kthD < kthDistMin) { kthDistMin = kthD; }
        if (kthD > kthDistMax) { kthDistMax = kthD; }

        // Weighted variance of B-samples around mean D[i]
        const rp = reconstructedPoints[i];
        let wvar = 0.0;
        for (let p = 0; p < neighbors.length; p++)
        {
          const nb = neighbors[p];
          if (!nb) continue;
          const pb = ptsB[nb.id];
          if (!pb) continue;
          let dxSq = (pb.x - rp.x) * (pb.x - rp.x)
                   + (pb.y - rp.y) * (pb.y - rp.y);
          if (dimB >= 3)
          {
            const pz = (typeof pb.z === 'number') ? pb.z : 0.0;
            const rz = (typeof rp.z === 'number') ? rp.z : 0.0;
            dxSq += (pz - rz) * (pz - rz);
          }
          const weightVal = (typeof nb.weight === 'number' && !isNaN(nb.weight)) ? nb.weight : (1.0 / neighbors.length);
          wvar += weightVal * dxSq;
        }
        reconVariance[i] = wvar;
        if (wvar < varMin) { varMin = wvar; }
        if (wvar > varMax) { varMax = wvar; }
      } // for quality metrics

      if (kthDistMin === Infinity) { kthDistMin = 0; }
      if (kthDistMax === -Infinity) { kthDistMax = 0; }
      if (varMin === Infinity) { varMin = 0; }
      if (varMax === -Infinity) { varMax = 0; }

      // 4. Save into Slot D
      slotD.benchmarkDataset = reconstructedPoints;
      slotD.rawBenchmarkDataset = reconstructedPoints;
      slotD.pastSamples = reconstructedPoints.map((p, idx) => ({
        x: p.x,
        y: p.y,
        z: (dimB >= 3 && typeof p.z === 'number') ? p.z : 0.0,
        clusterId: -1,
        frameIndex: idx
      }));
      slotD.currentDim = dimB;
      slotD.benchmarkKey = 'reconstructed';
      slotD.sampleCount = numQueries;
      slotD.isDatasetStaged = true;
      slotD.stagedDatasetInfo = {
        name: 'Reconstructed (from A, B, C)',
        count: numQueries,
        dim: dimB,
        passes: 1,
        noise: 0
      };
      slotD.reconstructionInfo = {
        queryCount: numQueries,
        outputDim: dimB,
        k: k,
        weightMode: weightMode,
        alpha: alpha,
        computeTimeMs: elapsedMs,
        avgNeighborDist: avgNeighborDist
      };
      slotD.reconstructionSourceNeighbors = sourceNeighbors;
      slotD._reverseNeighborsIndex = null;
      if (slotC && slotC._onDemandKnnCache) slotC._onDemandKnnCache.clear();

      // Store quality metrics on slots C and D
      slotC.reconKthDist = reconKthDist;
      slotC.reconKthDistMin = kthDistMin;
      slotC.reconKthDistMax = kthDistMax;
      slotD.reconKthDist = reconKthDist;
      slotD.reconKthDistMin = kthDistMin;
      slotD.reconKthDistMax = kthDistMax;
      slotD.reconVariance = reconVariance;
      slotD.reconVarianceMin = varMin;
      slotD.reconVarianceMax = varMax;

      // Update Query k-NN Resource Tracker card with query execution metrics
      if (!nativeKnn) {
        const bruteDistCalls = numQueries * numCandidates;
        updateReconQueryTracker({
          framedistCalls: bruteDistCalls,
          timeSearchMs: elapsedMs,
          timeLoadMs: 0,
          level1ClustersPruned: 0,
          level2AnchorsPruned: 0,
          level3AnnularPruned: 0,
          temporalPruned: 0
        }, numQueries, numCandidates, k);
        const pBadge = document.getElementById('reconQueryPruneEffBadge');
        if (pBadge) pBadge.textContent = '0% (WASM Brute)';
      }

      // Update Toolbar status pill for D
      const pillD = document.getElementById('datasetStatusPill_D');
      if (pillD) {
        pillD.textContent =
          `📦 Reconstructed: ${numQueries.toLocaleString()} pts (${dimB}D, k=${k})`;
        pillD.style.background = 'rgba(168, 85, 247, 0.2)';
        pillD.style.color = '#c084fc';
        pillD.style.borderColor = 'rgba(168, 85, 247, 0.5)';
      }

      // 5. Automatically enable 4-panel view and switch active slot to D
      if (!multiDatasetEnabled) {
        setMultiDatasetEnabled(true);
      }
      setRecon4PanelView(true);

      if (activeDatasetSlot !== 'D') {
        switchDatasetSlot('D');
      } else {
        loadSlotState('D');
        if (typeof updateDatasetStatusBadge === 'function') updateDatasetStatusBadge();
        if (typeof updateUI === 'function') updateUI();
        if (typeof renderReconstructionDashboard === 'function') renderReconstructionDashboard();
      }

      if (typeof resetView === 'function') {
        resetView();
      }
      if (typeof draw === 'function') {
        draw();
      }

      showToast(
        `✅ Reconstruction Complete! Dataset [D] contains ` +
        `${numQueries.toLocaleString()} points (${dimB}D) in ${elapsedMs.toFixed(1)} ms.`
      );
    }

    // =========================================================================
    //  NATIVE gric-knn -query C RUNNER
    // =========================================================================

    /**
     * Populate the Query k-NN Resource Tracker card with parsed telemetry.
     *
     * @param {object} telem    Result of DesktopBridge.parseKnnTelemetryLog()
     * @param {number} numQ     Number of query (C) frames
     * @param {number} numCand  Number of candidate (A) frames
     * @param {number} k        k neighbors
     */
    function updateReconQueryTracker(telem, numQ, numCand, k)
    {
      const bruteForce = numQ * numCand;
      const pruned = Math.max(0, bruteForce - telem.framedistCalls);
      const pruneEff = bruteForce > 0
        ? (100.0 * pruned / bruteForce) : 0.0;
      const speedup = telem.framedistCalls > 0
        ? (bruteForce / telem.framedistCalls) : 0.0;
      const qps = telem.timeSearchMs > 0
        ? (numQ / (telem.timeSearchMs / 1000.0)) : 0.0;
      const distPerQuery = numQ > 0
        ? Math.round(telem.framedistCalls / numQ) : 0;

      /* Compute per-level percentages against total candidate slots */
      const totalSlots = numQ * numCand;
      function pct(val) {
        return totalSlots > 0
          ? `${(100.0 * val / totalSlots).toFixed(1)}%`
          : '0%';
      }
      const l1 = (telem.level0SuperClustersPruned || 0) + (telem.level1ClustersPruned || 0);
      const l3 = telem.level3AnnularPruned || 0;
      const gPruned = telem.graphEdgesPruned || 0;
      const angular = telem.angularPruned || 0;
      const multiPivot = telem.multiPivotPruned || 0;
      const gSeeds = telem.graphSeedsEvaluated || 0;
      const sq8Evals = telem.sq8Evaluations || 0;
      const sq8Members = telem.sq8MembersPruned || 0;
      const sq8Graph = telem.sq8GraphPruned || 0;
      const sq8Pruned = (telem.sq8TotalPruned !== undefined)
        ? telem.sq8TotalPruned
        : (sq8Members + sq8Graph);
      const exact = Math.max(0, telem.framedistCalls || 0);
      const totalAll = l1 + l3 + gPruned + angular + multiPivot + gSeeds + sq8Pruned + exact;

      function barPct(val) {
        return totalAll > 0
          ? `${(100.0 * val / totalAll).toFixed(2)}%`
          : '0%';
      }

      /* Helper to safely set textContent by id */
      function set(id, text) {
        const el = document.getElementById(id);
        if (el) { el.textContent = text; }
      }

      /* -- Overview tab -- */
      set('reconQueryPruneEffVal',     `${pruneEff.toFixed(1)}%`);
      set('reconQuerySpeedupVal',      `${speedup.toFixed(1)}×`);
      set('reconQueryDistCallsVal',    telem.framedistCalls.toLocaleString());
      set('reconQueryDistPerQueryVal', `${distPerQuery.toLocaleString()}/q`);
      set('reconQuerySq8EvalsVal',     sq8Evals > 0 ? sq8Evals.toLocaleString() : '--');
      set('reconQueryFullPrecCallsVal', exact.toLocaleString());
      set('reconQuerySearchTimeVal',   `${telem.timeSearchMs.toFixed(1)} ms`);
      set('reconQueryLoadTimeVal',     telem.timeLoadMs != null
                                         ? `load ${telem.timeLoadMs.toFixed(0)} ms`
                                         : 'load --');
      set('reconQueryQpsVal',          `${Math.round(qps).toLocaleString()} QPS`);
      set('reconQueryTotalQueriesVal', numQ.toLocaleString());
      set('reconQueryKParamVal',       `k=${k}`);

      /* Funnel bar */
      set('reconQueryPruneSummaryTxt', `${pruneEff.toFixed(1)}% Pruned`);
      const barPruned = document.getElementById('barReconQueryPruned');
      const barEval   = document.getElementById('barReconQueryEvaluated');
      if (barPruned) { barPruned.style.width = `${pruneEff.toFixed(2)}%`; }
      if (barEval)   { barEval.style.width   = `${(100 - pruneEff).toFixed(2)}%`; }
      set('lblReconQueryPrunedCount',  pruned.toLocaleString());
      set('lblReconQueryEvalCount',    exact.toLocaleString());

      /* -- Pruning tab -- */
      set('reconQueryL1PrunedVal',          l1.toLocaleString());
      set('reconQueryL1PctVal',             pct(l1));
      set('reconQueryL3PrunedVal',          l3.toLocaleString());
      set('reconQueryL3PctVal',             pct(l3));
      set('reconQueryGraphPrunedVal',       gPruned.toLocaleString());
      set('reconQueryGraphPrunedPctVal',    pct(gPruned));
      set('reconQueryGraphSeedsVal',        gSeeds.toLocaleString());
      set('reconQueryGraphSeedsPctVal',     pct(gSeeds));

      set('reconQueryAngularVal',           angular.toLocaleString());
      set('reconQueryAngularPctVal',        pct(angular));

      set('reconQueryMultiPivotVal',        multiPivot.toLocaleString());
      set('reconQueryMultiPivotPctVal',     pct(multiPivot));

      set('reconQuerySq8PrunedVal',         sq8Pruned.toLocaleString());
      set('reconQuerySq8PctVal',            pct(sq8Pruned));
      set('reconQuerySq8BreakdownVal',
          `Members: ${sq8Members.toLocaleString()} | Graph: ${sq8Graph.toLocaleString()}`);

      const containmentHits = telem.globalContainmentHits || 0;
      const containmentPct = numQ > 0 ? (100.0 * containmentHits / numQ).toFixed(1) : '0.0';
      set('reconQueryContainmentVal',       containmentHits.toLocaleString());
      set('reconQueryContainmentPctVal',    `${containmentPct}% of queries`);

      /* Hierarchy stacked bar */
      const bL1         = document.getElementById('barReconQueryL1');
      const bL3         = document.getElementById('barReconQueryL3');
      const bGPruned    = document.getElementById('barReconQueryGraphPruned');
      const bAngular    = document.getElementById('barReconQueryAngular');
      const bMultiPivot = document.getElementById('barReconQueryMultiPivot');
      const bGSeeds     = document.getElementById('barReconQueryGraphSeeds');
      const bSq8        = document.getElementById('barReconQuerySq8');
      const bExact      = document.getElementById('barReconQueryExact');
      if (bL1)         { bL1.style.width         = barPct(l1); }
      if (bL3)         { bL3.style.width         = barPct(l3); }
      if (bGPruned)    { bGPruned.style.width    = barPct(gPruned); }
      if (bAngular)    { bAngular.style.width    = barPct(angular); }
      if (bMultiPivot) { bMultiPivot.style.width = barPct(multiPivot); }
      if (bGSeeds)     { bGSeeds.style.width     = barPct(gSeeds); }
      if (bSq8)        { bSq8.style.width        = barPct(sq8Pruned); }
      if (bExact)      { bExact.style.width      = barPct(exact); }

      /* Header badges */
      set('reconQueryPruneEffBadge', `${pruneEff.toFixed(1)}% pruned`);
      set('reconQueryQpsBadge',      `${Math.round(qps).toLocaleString()} QPS`);

      /* Show and expand the tracker card */
      const card = document.getElementById('cardReconQueryResources');
      if (card) {
        card.style.display = '';
        if (card.classList.contains('collapsed')) {
          togglePanelCollapse('cardReconQueryResources');
        }
      }
    }

    /**
     * Run native gric-knn -query C via gric-server, showing a live progress
     * bar and populating the Query k-NN Resource Tracker on completion.
     */
    async function runNativeReconQueryKnn()
    {
      if (!DesktopBridge.isNativeSupported()) {
        showToast('⚠️ Native runner not available in Web Mode.');
        return;
      }

      const slotA = datasetSlots['A'];
      const slotC = datasetSlots['C'];

      if (!slotA || !slotA.benchmarkDataset || slotA.benchmarkDataset.length === 0) {
        showToast('⚠️ Dataset A (Training Input) must be staged first.');
        return;
      }
      if (!slotC || !slotC.benchmarkDataset || slotC.benchmarkDataset.length === 0) {
        showToast('⚠️ Dataset C (Query Input) must be staged first.');
        return;
      }

      const ptsA = slotA.benchmarkDataset;
      const ptsC = slotC.benchmarkDataset;

      /* Resolve clean dataset base names and full filenames */
      let rawNameA = (slotA.stagedDatasetInfo && slotA.stagedDatasetInfo.name)
        ? slotA.stagedDatasetInfo.name
        : (slotA.benchmarkKey || 'dataset_A');
      let rawNameC = (slotC.stagedDatasetInfo && slotC.stagedDatasetInfo.name)
        ? slotC.stagedDatasetInfo.name
        : (slotC.benchmarkKey || 'dataset_C');

      const baseA = rawNameA.replace(/\.(txt|csv|fits|dat|mp4|fits\.fz)$/i, '').replace(/[^a-zA-Z0-9_.-]/g, '_');
      const baseC = rawNameC.replace(/\.(txt|csv|fits|dat|mp4|fits\.fz)$/i, '').replace(/[^a-zA-Z0-9_.-]/g, '_');

      const stagedNameC = (baseA === baseC) ? `${baseC}_query_C` : baseC;
      const datasetFileA = `${baseA}.txt`;
      const datasetFileC = `${stagedNameC}.txt`;
      const clusterDir   = `${baseA}.clusterdat`;

      const btnRun  = document.getElementById('btnRunNativeReconQuery');
      const btnKill = document.getElementById('btnKillReconQuery');
      const boxProg = document.getElementById('reconQueryProgressBox');
      const elFill  = document.getElementById('reconQueryProgressFill');
      const elPct   = document.getElementById('reconQueryPct');
      const elFrames= document.getElementById('reconQueryFrames');
      const elEta   = document.getElementById('reconQueryEta');
      const elSpeed = document.getElementById('reconQuerySpeed');
      const elElapsed = document.getElementById('reconQueryElapsed');
      const consoleEl = document.getElementById('cliConsoleLog');

      if (btnRun)  { btnRun.disabled  = true; }
      if (btnKill) { btnKill.disabled = false; }
      if (boxProg) { boxProg.style.display = ''; }
      if (elFill)  { elFill.style.width = '0%'; }
      if (elPct)   { elPct.textContent   = '0%'; }
      if (elFrames){ elFrames.textContent = `0 / ${ptsC.length.toLocaleString()}`; }
      if (elEta)   { elEta.textContent    = 'ETA --'; }
      if (elSpeed) { elSpeed.textContent  = '0 fr/s'; }
      if (elElapsed){ elElapsed.textContent = '0.0 s'; }

      try {
        /* 1. Ensure staged coordinates exist on disk in workspace */
        const dimA = slotA.currentDim || 2;
        const dimC = slotC.currentDim || 2;
        if (ptsA && ptsA.length > 0) {
          await DesktopBridge.stageDatasetFile(baseA, ptsA, dimA).catch(() => {});
        }
        if (ptsC && ptsC.length > 0) {
          await DesktopBridge.stageDatasetFile(stagedNameC, ptsC, dimC).catch(() => {});
        }

        /* 2. Check if cluster directory exists on disk; if not, export clusters of A */
        const filesInWorkspace = await DesktopBridge.listFiles().catch(() => []);
        let hasClusterDir = filesInWorkspace.some(
          f => (f.name === clusterDir || f.name === `${baseA}.clusterdat`) && (f.is_dir || f.isDir)
        );

        if (!hasClusterDir) {
          const memClusters = (slotA && slotA.clusters && slotA.clusters.length > 0)
            ? slotA.clusters : (typeof clusters !== 'undefined' ? clusters : []);
          if (memClusters && memClusters.length > 0) {
            let centroidsText = `# GRIC Cluster Centroids\n# ID X Y Z MEMBERS\n`;
            memClusters.forEach(c => {
              centroidsText += `${c.id} ${Number(c.x || 0).toFixed(6)} ${Number(c.y || 0).toFixed(6)} ` +
                               `${Number(c.z || 0).toFixed(6)} ${c.members || 1}\n`;
            });
            let dccText = `# GRIC Cluster-to-Cluster Distance Matrix D_cc\n`;
            const memDcc = (slotA && slotA.dcc && slotA.dcc.length > 0)
              ? slotA.dcc : (typeof dcc !== 'undefined' ? dcc : []);
            if (memDcc && memDcc.length > 0) {
              memDcc.forEach(row => {
                dccText += row.map(v => Number(v).toFixed(6)).join(' ') + '\n';
              });
            }
            let memText = `# Frame Membership Assignments\n`;
            const memPast = (slotA && slotA.pastSamples && slotA.pastSamples.length > 0)
              ? slotA.pastSamples : (typeof pastSamples !== 'undefined' ? pastSamples : []);
            if (memPast && memPast.length > 0) {
              for (let i = 0; i < memPast.length; i++) {
                const p = memPast[i];
                const cId = p.clusterId >= 0 ? p.clusterId : 0;
                memText += `${i} ${cId} 0.000000\n`;
              }
            }
            let exportFiles = {
              'anchors.txt': centroidsText,
              'dcc.txt': dccText,
              'frame_membership.txt': memText
            };
            await DesktopBridge.exportClusterDat(baseA, exportFiles).catch(() => {});
          } else {
            showToast(`⚡ Clustering Dataset A (${datasetFileA}) for metric bounds...`);
            await new Promise((resolve) => {
              DesktopBridge.runCliJob({
                cmd: 'gric-cluster',
                args: ['-outdir', clusterDir, '0.15', datasetFileA],
                onOutput: (chunk) => {
                  if (consoleEl) {
                    consoleEl.textContent += chunk;
                    consoleEl.scrollTop = consoleEl.scrollHeight;
                  }
                },
                onTelemetry: () => {},
                onFinish: () => resolve()
              }).catch(() => resolve());
            });
          }
        }

        /* 3. Read k from the reconstruction k input */
        const inputK = document.getElementById('inputReconK');
        const k = inputK ? Math.max(1, Math.min(parseInt(inputK.value, 10) || 30, ptsA.length)) : 30;

        /* Distinct output path so we don't overwrite the A-vs-A knn results */
        const queryOutPrefix = `${clusterDir}/knn_query_C`;

        const args = [
          datasetFileA,
          clusterDir,
          '-query', datasetFileC,
          '-k', String(k),
          '-progress',
          '-txt',
          '-o', queryOutPrefix
        ];

        const chkApprox = document.getElementById('chkReconQueryApprox');
        if (chkApprox && chkApprox.checked) {
          args.push('--approx');
        }

        const chkTrajectory = document.getElementById('chkReconQueryTrajectory');
        if (chkTrajectory && chkTrajectory.checked) {
          args.push('--trajectory');
        } else {
          args.push('--no-trajectory');
        }

        const chkSq8 = document.getElementById('chkReconQuerySq8');
        if (chkSq8 && chkSq8.checked) {
          args.push('-sq8');
        }

        if (consoleEl) {
          consoleEl.textContent +=
            `\n🔍 [Reconstruction] Dispatching native gric-knn -query C (k=${k})...\n` +
            `🗂️  Dataset A:    ${datasetFileA}\n` +
            `🗂️  Query C:      ${datasetFileC}\n` +
            `📁  Cluster Dir:  ${clusterDir}\n` +
            `📄  Output Base:  ${queryOutPrefix}\n` +
            `─────────────────────────────────────────────────────────\n`;
          consoleEl.scrollTop = consoleEl.scrollHeight;
        }

        showToast(`🔍 Running native gric-knn -query C (k=${k})...`);

        const tKnnStart = performance.now();
        let rawCliOutput = '';
        let finalExitCode = 0;

        await new Promise((resolve) => {
          DesktopBridge.runCliJob({
            cmd: 'gric-knn',
            args: args,
            onOutput: (chunk) => {
              rawCliOutput += chunk;
              if (consoleEl) {
                consoleEl.textContent += chunk;
                consoleEl.scrollTop = consoleEl.scrollHeight;
              }

              /* Parse live progress: "Searching k-NN: [...] XX.X% (N / M frames)" */
              const lines = chunk.split(/\r|\n/);
              for (const line of lines) {
                const m = line.match(
                  /Searching k-NN:\s*\[.*?\]\s*([\d.]+)%\s*\(\s*(\d+)\s*\/\s*(\d+)\s*frames\)/
                );
                if (!m) { continue; }

                const pct      = parseFloat(m[1]);
                const processed = parseInt(m[2], 10);
                const total    = parseInt(m[3], 10);
                const now      = performance.now();
                const elapsedSec = Math.max(0.01, (now - tKnnStart) / 1000.0);
                const speed    = processed > 0 ? (processed / elapsedSec) : 0;
                const remaining = Math.max(0, total - processed);
                const etaSec   = speed > 0 ? (remaining / speed) : 0;

                if (elFill)   { elFill.style.width     = `${pct.toFixed(1)}%`; }
                if (elPct)    { elPct.textContent       = `${pct.toFixed(1)}%`; }
                if (elFrames) { elFrames.textContent    = `${processed.toLocaleString()} / ${total.toLocaleString()}`; }
                if (elSpeed)  { elSpeed.textContent     = `${Math.round(speed).toLocaleString()} fr/s`; }
                if (elElapsed){ elElapsed.textContent   = `${elapsedSec.toFixed(1)} s`; }
                if (elEta) {
                  elEta.textContent = etaSec > 1
                    ? `ETA ${etaSec.toFixed(0)} s`
                    : 'ETA < 1 s';
                }
              } // for lines
            },
            onTelemetry: () => {},
            onFinish: (res) => {
              finalExitCode = res?.exitCode ?? 0;
              if (finalExitCode === 0) {
                showToast('✅ gric-knn -query C completed successfully!');
                if (elFill) { elFill.style.width = '100%'; }
                if (elPct)  { elPct.textContent   = '100%'; }
                if (elEta)  { elEta.textContent   = 'Done'; }
              } else {
                showToast(`⚠️ gric-knn -query C finished with exit code ${finalExitCode}`);
              }
              resolve();
            }
          }).catch((err) => {
            showToast(`⚠️ Failed to execute gric-knn: ${err.message}`);
            if (consoleEl) {
              consoleEl.textContent += `\n❌ Error: ${err.message}\n`;
              consoleEl.scrollTop = consoleEl.scrollHeight;
            }
            resolve();
          });
        });

        if (finalExitCode === 0) {
          /* Parse final telemetry from stdout */
          const parsedTelem = DesktopBridge.parseKnnTelemetryLog(rawCliOutput);

          /* Populate tracker card */
          updateReconQueryTracker(parsedTelem, ptsC.length, ptsA.length, k);

          /* Load the query k-NN results from distinct output binary files */
          let queryData = await DesktopBridge.readKnnQueryResults(queryOutPrefix, k).catch(() => null);
          if (queryData && queryData.indices && queryData.totalFrames > 0) {
            slotC.knnResults = queryData;
            slotC.knnResultsForQuery = true;
            showToast(
              `📊 Loaded ${queryData.totalFrames.toLocaleString()} query k-NN results ` +
              `(k=${queryData.k}). Auto-reconstructing D...`
            );
            /* Auto-reconstruct Dataset D with the loaded query k-NN neighbors */
            if (typeof executeDatasetReconstruction === 'function') {
              await executeDatasetReconstruction();
            }
          } else {
            showToast('⚠️ Query search completed but result file could not be read.');
          }
        }
      } catch (err) {
        showToast(`⚠️ Error in gric-knn -query: ${err.message}`);
        console.error('[runNativeReconQueryKnn]', err);
      } finally {
        if (btnRun)  { btnRun.disabled  = false; }
        if (btnKill) { btnKill.disabled = true;  }
        if (boxProg) {
          setTimeout(() => {
            boxProg.style.display = 'none';
          }, 800);
        }
      }
    } // runNativeReconQueryKnn


    // =========================================================================
    //  Query k-NN resource tracker tab switching
    // =========================================================================

    {
      const tabOverview = document.getElementById('tabReconQueryOverview');
      const tabPruning  = document.getElementById('tabReconQueryPruning');
      const panelOverview = document.getElementById('reconQueryOverviewPanel');
      const panelPruning  = document.getElementById('reconQueryPruningPanel');

      if (tabOverview && tabPruning) {
        tabOverview.addEventListener('click', () => {
          tabOverview.classList.add('active');
          tabPruning.classList.remove('active');
          if (panelOverview) { panelOverview.style.display = ''; }
          if (panelPruning)  { panelPruning.style.display  = 'none'; }
        });
        tabPruning.addEventListener('click', () => {
          tabPruning.classList.add('active');
          tabOverview.classList.remove('active');
          if (panelPruning)  { panelPruning.style.display  = ''; }
          if (panelOverview) { panelOverview.style.display = 'none'; }
        });
      }
    }

    // Reconstruction UI Listeners
    const btnReconSide = document.getElementById('btnRunReconstructionSide');
    if (btnReconSide) {
      btnReconSide.addEventListener('click', executeDatasetReconstruction);
    }
    const btnReconTop = document.getElementById('btnReconstructDTop');
    if (btnReconTop) {
      btnReconTop.addEventListener('click', executeDatasetReconstruction);
    }

    /* Native query button — only active in desktop mode */
    const btnNativeQuery = document.getElementById('btnRunNativeReconQuery');
    if (btnNativeQuery) {
      btnNativeQuery.addEventListener('click', runNativeReconQueryKnn);
    }
    const btnKillReconQuery = document.getElementById('btnKillReconQuery');
    if (btnKillReconQuery) {
      btnKillReconQuery.addEventListener('click', () => {
        DesktopBridge.killActiveJob().catch(() => {});
        btnKillReconQuery.disabled = true;
        const btnNQ = document.getElementById('btnRunNativeReconQuery');
        if (btnNQ) { btnNQ.disabled = false; }
      });
    }

    const btnToggleRecon4P = document.getElementById('btnToggleRecon4PanelView');
    if (btnToggleRecon4P) {
      btnToggleRecon4P.addEventListener('click', () => {
        toggleRecon4PanelView();
      });
    }
    const btnToggleRecon4PTop = document.getElementById('btnToggleRecon4PanelTop');
    if (btnToggleRecon4PTop) {
      btnToggleRecon4PTop.addEventListener('click', () => {
        toggleRecon4PanelView();
      });
    }
    const btnPresetRecon4P = document.getElementById('btnPresetRecon4Panel');
    if (btnPresetRecon4P) {
      btnPresetRecon4P.addEventListener('click', () => {
        toggleRecon4PanelView();
      });
    }
    const btnToggleReconKnnTop = document.getElementById('btnToggleReconKnnTop');
    if (btnToggleReconKnnTop) {
      btnToggleReconKnnTop.addEventListener('click', () => {
        setReconKnn(!showReconKnn);
      });
    }
    const btnToggleReconKnnSide = document.getElementById('btnToggleReconKnnSide');
    if (btnToggleReconKnnSide) {
      btnToggleReconKnnSide.addEventListener('click', () => {
        setReconKnn(!showReconKnn);
      });
    }
    const btnToggleReconOverlayTop = document.getElementById('btnToggleReconOverlayTop');
    if (btnToggleReconOverlayTop) {
      btnToggleReconOverlayTop.addEventListener('click', () => {
        setReconOverlayMode();
      });
    }
    const btnToggleReconOverlaySide = document.getElementById('btnToggleReconOverlaySide');
    if (btnToggleReconOverlaySide) {
      btnToggleReconOverlaySide.addEventListener('click', () => {
        setReconOverlayMode();
      });
    }
    const btnInspectD = document.getElementById('btnInspectDatasetDSide');
    if (btnInspectD) {
      btnInspectD.addEventListener('click', () => {
        if (!multiDatasetEnabled) setMultiDatasetEnabled(true);
        switchDatasetSlot('D');
      });
    }
    const btnClearDSide = document.getElementById('btnClearDatasetDSide');
    if (btnClearDSide) {
      btnClearDSide.addEventListener('click', () => {
        clearDatasetSlot('D');
      });
    }

    // Quality coloring & filtering helper: compute mask from percentile
    function updateReconQualityMask()
    {
      const slotC = (typeof datasetSlots !== 'undefined') ? datasetSlots['C'] : null;
      const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;

      // 1. Compute for Slot C (k-th NN distance)
      const arrC = (slotC && slotC.reconKthDist) ? slotC.reconKthDist
                 : (slotD && slotD.reconKthDist) ? slotD.reconKthDist : null;
      let visCountC = 0;
      if (arrC && arrC.length > 0 && reconQualityThreshold < 1.0)
      {
        if (!slotC || !slotC.sortedReconKthDist ||
            slotC.sortedReconKthDist.length !== arrC.length)
        {
          if (slotC) { slotC.sortedReconKthDist = Float64Array.from(arrC).sort(); }
        }
        const sortedC = (slotC && slotC.sortedReconKthDist)
          ? slotC.sortedReconKthDist : Float64Array.from(arrC).sort();
        const pctIdxC = Math.min(
          sortedC.length - 1,
          Math.floor(reconQualityThreshold * sortedC.length)
        );
        const threshC = sortedC[pctIdxC];
        const maskC = new Uint8Array(arrC.length);
        const indicesC = [];
        for (let i = 0; i < arrC.length; i++)
        {
          if (arrC[i] <= threshC)
          {
            maskC[i] = 1;
            indicesC.push(i);
            visCountC++;
          }
        }
        if (slotC)
        {
          slotC.reconQualityMask = maskC;
          slotC.reconQualityIndices = new Int32Array(indicesC);
        }
      }
      else
      {
        if (slotC)
        {
          slotC.reconQualityMask = null;
          slotC.reconQualityIndices = null;
        }
        if (arrC) { visCountC = arrC.length; }
      }

      // 2. Compute for Slot D (Recon Variance, fallback to k-th NN dist)
      const arrD = (slotD && slotD.reconVariance) ? slotD.reconVariance
                 : (slotD && slotD.reconKthDist) ? slotD.reconKthDist : null;
      let visCountD = 0;
      if (arrD && arrD.length > 0 && reconQualityThreshold < 1.0)
      {
        if (!slotD || !slotD.sortedReconVariance ||
            slotD.sortedReconVariance.length !== arrD.length)
        {
          if (slotD) { slotD.sortedReconVariance = Float64Array.from(arrD).sort(); }
        }
        const sortedD = (slotD && slotD.sortedReconVariance)
          ? slotD.sortedReconVariance : Float64Array.from(arrD).sort();
        const pctIdxD = Math.min(
          sortedD.length - 1,
          Math.floor(reconQualityThreshold * sortedD.length)
        );
        const threshD = sortedD[pctIdxD];
        const maskD = new Uint8Array(arrD.length);
        const indicesD = [];
        for (let i = 0; i < arrD.length; i++)
        {
          if (arrD[i] <= threshD)
          {
            maskD[i] = 1;
            indicesD.push(i);
            visCountD++;
          }
        }
        if (slotD)
        {
          slotD.reconQualityMask = maskD;
          slotD.reconQualityIndices = new Int32Array(indicesD);
        }
      }
      else
      {
        if (slotD)
        {
          slotD.reconQualityMask = null;
          slotD.reconQualityIndices = null;
        }
        if (arrD) { visCountD = arrD.length; }
      }

      // 3. Set global reconQualityMask & reconQualityIndices according to activeDatasetSlot
      if (activeDatasetSlot === 'C' && slotC)
      {
        reconQualityMask = slotC.reconQualityMask;
        reconQualityIndices = slotC.reconQualityIndices;
      }
      else if (activeDatasetSlot === 'D' && slotD)
      {
        reconQualityMask = slotD.reconQualityMask;
        reconQualityIndices = slotD.reconQualityIndices;
      }
      else if (slotD && slotD.reconQualityMask)
      {
        reconQualityMask = slotD.reconQualityMask;
        reconQualityIndices = slotD.reconQualityIndices;
      }
      else if (slotC && slotC.reconQualityMask)
      {
        reconQualityMask = slotC.reconQualityMask;
        reconQualityIndices = slotC.reconQualityIndices;
      }
      else
      {
        reconQualityMask = null;
        reconQualityIndices = null;
      }

      // 4. Update UI labels
      const targetArr = (activeDatasetSlot === 'C') ? arrC : (arrD || arrC);
      const targetVis = (activeDatasetSlot === 'C') ? visCountC : (visCountD || visCountC);
      updateReconQualityLabels(targetArr, targetVis);

      // 5. Update Sample Pool Action Button State
      const btnUpdatePool = document.getElementById('btnUpdatePoolToPruned');
      if (btnUpdatePool)
      {
        const activeIdxs = (activeDatasetSlot === 'C')
          ? (slotC && slotC.reconQualityIndices)
          : ((slotD && slotD.reconQualityIndices) || (slotC && slotC.reconQualityIndices));
        if (reconQualityThreshold < 1.0 && activeIdxs && activeIdxs.length > 0)
        {
          btnUpdatePool.disabled = false;
          btnUpdatePool.style.opacity = '1.0';
          btnUpdatePool.style.cursor = 'pointer';
          btnUpdatePool.innerHTML =
            `<span>⚡</span> Update Pool to Pruned (${activeIdxs.length.toLocaleString()} pts)`;
        }
        else
        {
          btnUpdatePool.disabled = true;
          btnUpdatePool.style.opacity = '0.45';
          btnUpdatePool.style.cursor = 'not-allowed';
          btnUpdatePool.innerHTML = '<span>⚡</span> Update Pool to Pruned';
        }
      }
    }

    function updateReconQualityLabels(arr, visCount)
    {
      const lblVis = document.getElementById('lblReconQualityVisible');
      const lblTot = document.getElementById('lblReconQualityTotal');
      const lblMetric = document.getElementById('lblReconQualityMetric');

      if (lblTot && arr)
      {
        lblTot.textContent = arr.length.toLocaleString();
      }
      if (lblVis)
      {
        if (visCount !== undefined)
        {
          lblVis.textContent = visCount.toLocaleString();
        }
        else if (arr)
        {
          lblVis.textContent = arr.length.toLocaleString();
        }
        else
        {
          lblVis.textContent = '--';
        }
      }
      if (lblMetric)
      {
        if (activeDatasetSlot === 'C')
        {
          lblMetric.textContent = 'k-th NN Dist (C)';
        }
        else if (activeDatasetSlot === 'D')
        {
          lblMetric.textContent = 'Recon Variance (D)';
        }
        else if (datasetSlots['D'] && datasetSlots['D'].reconVariance)
        {
          lblMetric.textContent = 'Recon Variance (D)';
        }
        else if (datasetSlots['C'] && datasetSlots['C'].reconKthDist)
        {
          lblMetric.textContent = 'k-th NN Dist (C)';
        }
        else
        {
          lblMetric.textContent = '--';
        }
      }
    }

    // Quality checkbox listener
    const chkQuality = document.getElementById('chkReconQuality');
    if (chkQuality)
    {
      chkQuality.addEventListener('change', () => {
        reconQualityColoringEnabled = chkQuality.checked;
        updateReconQualityMask();
        if (typeof draw === 'function') { draw(); }
      });
    }

    // Quality slider listener
    const sliderQuality = document.getElementById('sliderReconQuality');
    if (sliderQuality)
    {
      sliderQuality.addEventListener('input', () => {
        const pct = parseInt(sliderQuality.value, 10);
        reconQualityThreshold = pct / 100.0;
        const lblPct = document.getElementById('lblReconQualityPct');
        if (lblPct) { lblPct.textContent = pct + '%'; }
        updateReconQualityMask();
        if (typeof draw === 'function') { draw(); }
      });
    }

    window.updateReconQualityMask = updateReconQualityMask;

    function updatePoolToPruned()
    {
      const slotC = datasetSlots['C'];
      const slotD = datasetSlots['D'];
      const indices = (slotD && slotD.reconQualityIndices) ? slotD.reconQualityIndices
                    : (slotC && slotC.reconQualityIndices) ? slotC.reconQualityIndices : null;

      if (!indices || indices.length === 0)
      {
        return;
      }

      // Save backups if not already saved
      if (slotC && !slotC._unprunedBackup)
      {
        slotC._unprunedBackup = {
          benchmarkDataset: slotC.benchmarkDataset,
          rawBenchmarkDataset: slotC.rawBenchmarkDataset,
          pastSamples: slotC.pastSamples,
          sampleCount: slotC.sampleCount,
          stagedDatasetInfo: slotC.stagedDatasetInfo ? { ...slotC.stagedDatasetInfo } : null,
          reconKthDist: slotC.reconKthDist,
          reconKthDistMin: slotC.reconKthDistMin,
          reconKthDistMax: slotC.reconKthDistMax,
          sortedReconKthDist: slotC.sortedReconKthDist
        };
      }
      if (slotD && !slotD._unprunedBackup)
      {
        slotD._unprunedBackup = {
          benchmarkDataset: slotD.benchmarkDataset,
          rawBenchmarkDataset: slotD.rawBenchmarkDataset,
          pastSamples: slotD.pastSamples,
          sampleCount: slotD.sampleCount,
          stagedDatasetInfo: slotD.stagedDatasetInfo ? { ...slotD.stagedDatasetInfo } : null,
          reconstructionInfo: slotD.reconstructionInfo ? { ...slotD.reconstructionInfo } : null,
          reconstructionSourceNeighbors: slotD.reconstructionSourceNeighbors,
          reconVariance: slotD.reconVariance,
          reconVarianceMin: slotD.reconVarianceMin,
          reconVarianceMax: slotD.reconVarianceMax,
          reconKthDist: slotD.reconKthDist,
          reconKthDistMin: slotD.reconKthDistMin,
          reconKthDistMax: slotD.reconKthDistMax,
          sortedReconVariance: slotD.sortedReconVariance,
          sortedReconKthDist: slotD.sortedReconKthDist
        };
      }

      const newCount = indices.length;
      const dimB = slotD ? slotD.currentDim : 3;

      if (slotC && slotC.benchmarkDataset)
      {
        const newPtsC = new Array(newCount);
        const newPastC = new Array(newCount);
        const newKthDistC = slotC.reconKthDist ? new Float64Array(newCount) : null;
        let kthMin = Infinity, kthMax = -Infinity;

        for (let j = 0; j < newCount; j++)
        {
          const origIdx = indices[j];
          const pc = slotC.benchmarkDataset[origIdx];
          if (pc)
          {
            newPtsC[j] = { x: pc.x, y: pc.y, z: pc.z };
            newPastC[j] = {
              x: pc.x,
              y: pc.y,
              z: pc.z || 0.0,
              clusterId: -1,
              frameIndex: j
            };
          }
          if (newKthDistC && slotC.reconKthDist)
          {
            const kd = slotC.reconKthDist[origIdx];
            newKthDistC[j] = kd;
            if (kd < kthMin) kthMin = kd;
            if (kd > kthMax) kthMax = kd;
          }
        }
        slotC.benchmarkDataset = newPtsC;
        slotC.rawBenchmarkDataset = newPtsC;
        slotC.pastSamples = newPastC;
        slotC.sampleCount = newCount;
        if (slotC.stagedDatasetInfo)
        {
          slotC.stagedDatasetInfo.count = newCount;
          if (!slotC.stagedDatasetInfo.name.includes('(pruned)'))
          {
            slotC.stagedDatasetInfo.name += ' (pruned)';
          }
        }
        slotC.reconKthDist = newKthDistC;
        slotC.reconKthDistMin = (kthMin === Infinity) ? 0 : kthMin;
        slotC.reconKthDistMax = (kthMax === -Infinity) ? 1 : kthMax;
        slotC.sortedReconKthDist = newKthDistC ? Float64Array.from(newKthDistC).sort() : null;
      }

      if (slotD && slotD.benchmarkDataset)
      {
        const newPtsD = new Array(newCount);
        const newPastD = new Array(newCount);
        const newVarD = slotD.reconVariance ? new Float64Array(newCount) : null;
        const newKthDistD = slotD.reconKthDist ? new Float64Array(newCount) : null;
        const newNeighborsD = slotD.reconstructionSourceNeighbors ? new Array(newCount) : null;
        let varMin = Infinity, varMax = -Infinity;

        for (let j = 0; j < newCount; j++)
        {
          const origIdx = indices[j];
          const pd = slotD.benchmarkDataset[origIdx];
          if (pd)
          {
            newPtsD[j] = { x: pd.x, y: pd.y, z: pd.z };
            newPastD[j] = {
              x: pd.x,
              y: pd.y,
              z: pd.z || 0.0,
              clusterId: -1,
              frameIndex: j
            };
          }
          if (newVarD && slotD.reconVariance)
          {
            const v = slotD.reconVariance[origIdx];
            newVarD[j] = v;
            if (v < varMin) varMin = v;
            if (v > varMax) varMax = v;
          }
          if (newKthDistD && slotD.reconKthDist)
          {
            newKthDistD[j] = slotD.reconKthDist[origIdx];
          }
          if (newNeighborsD && slotD.reconstructionSourceNeighbors)
          {
            newNeighborsD[j] = slotD.reconstructionSourceNeighbors[origIdx];
          }
        }
        slotD.benchmarkDataset = newPtsD;
        slotD.rawBenchmarkDataset = newPtsD;
        slotD.pastSamples = newPastD;
        slotD.sampleCount = newCount;
        if (slotD.stagedDatasetInfo)
        {
          slotD.stagedDatasetInfo.count = newCount;
          if (!slotD.stagedDatasetInfo.name.includes('(pruned)'))
          {
            slotD.stagedDatasetInfo.name += ' (pruned)';
          }
        }
        if (slotD.reconstructionInfo)
        {
          slotD.reconstructionInfo.queryCount = newCount;
        }
        slotD.reconstructionSourceNeighbors = newNeighborsD;
        slotD._reverseNeighborsIndex = null;
        if (slotC && slotC._onDemandKnnCache) slotC._onDemandKnnCache.clear();
        slotD.reconVariance = newVarD;
        slotD.reconVarianceMin = (varMin === Infinity) ? 0 : varMin;
        slotD.reconVarianceMax = (varMax === -Infinity) ? 1 : varMax;
        slotD.sortedReconVariance = newVarD ? Float64Array.from(newVarD).sort() : null;
        if (slotC)
        {
          slotD.reconKthDist = slotC.reconKthDist;
          slotD.reconKthDistMin = slotC.reconKthDistMin;
          slotD.reconKthDistMax = slotC.reconKthDistMax;
          slotD.sortedReconKthDist = slotC.sortedReconKthDist;
        }
      }

      // Reset quality filter slider to 100%
      reconQualityThreshold = 1.0;
      if (sliderQuality) sliderQuality.value = 100;
      const lblPct = document.getElementById('lblReconQualityPct');
      if (lblPct) lblPct.textContent = '100%';

      if (slotC)
      {
        slotC.reconQualityMask = null;
        slotC.reconQualityIndices = null;
      }
      if (slotD)
      {
        slotD.reconQualityMask = null;
        slotD.reconQualityIndices = null;
      }
      reconQualityMask = null;
      reconQualityIndices = null;

      updateReconQualityMask();

      const btnRestore = document.getElementById('btnRestoreOriginalPool');
      if (btnRestore) btnRestore.style.display = 'inline-flex';

      if (typeof updateDatasetStatusBadge === 'function') updateDatasetStatusBadge();
      const pillD = document.getElementById('datasetStatusPill_D');
      if (pillD && slotD)
      {
        pillD.textContent =
          `📦 Reconstructed: ${newCount.toLocaleString()} pts (${dimB}D, pruned)`;
      }
      const pillC = document.getElementById('datasetStatusPill_C');
      if (pillC && slotC && slotC.stagedDatasetInfo)
      {
        pillC.textContent = `📦 ${slotC.stagedDatasetInfo.name}: ${newCount.toLocaleString()} pts`;
      }
      const statPts = document.getElementById('reconStatsPoints');
      if (statPts) statPts.textContent = newCount.toLocaleString();

      if (typeof draw === 'function') draw();
    }

    function restoreOriginalPool()
    {
      const slotC = datasetSlots['C'];
      const slotD = datasetSlots['D'];
      if (slotC && slotC._unprunedBackup)
      {
        Object.assign(slotC, slotC._unprunedBackup);
        delete slotC._unprunedBackup;
      }
      if (slotD && slotD._unprunedBackup)
      {
        Object.assign(slotD, slotD._unprunedBackup);
        delete slotD._unprunedBackup;
      }
      if (slotD) slotD._reverseNeighborsIndex = null;
      if (slotC && slotC._onDemandKnnCache) slotC._onDemandKnnCache.clear();

      const btnRestore = document.getElementById('btnRestoreOriginalPool');
      if (btnRestore) btnRestore.style.display = 'none';

      reconQualityThreshold = 1.0;
      if (sliderQuality) sliderQuality.value = 100;
      const lblPct = document.getElementById('lblReconQualityPct');
      if (lblPct) lblPct.textContent = '100%';

      updateReconQualityMask();
      if (typeof updateDatasetStatusBadge === 'function') updateDatasetStatusBadge();
      const pillD = document.getElementById('datasetStatusPill_D');
      if (pillD && slotD)
      {
        const k = (slotD.reconstructionInfo && slotD.reconstructionInfo.k) || 10;
        pillD.textContent =
          `📦 Reconstructed: ${slotD.sampleCount.toLocaleString()} pts ` +
          `(${slotD.currentDim}D, k=${k})`;
      }
      const pillC = document.getElementById('datasetStatusPill_C');
      if (pillC && slotC && slotC.stagedDatasetInfo)
      {
        pillC.textContent =
          `📦 ${slotC.stagedDatasetInfo.name}: ` +
          `${slotC.sampleCount.toLocaleString()} pts`;
      }
      const statPts = document.getElementById('reconStatsPoints');
      if (statPts && slotD) statPts.textContent = slotD.sampleCount.toLocaleString();

      if (typeof draw === 'function') draw();
    }

    const btnUpdatePool = document.getElementById('btnUpdatePoolToPruned');
    if (btnUpdatePool)
    {
      btnUpdatePool.addEventListener('click', updatePoolToPruned);
    }
    const btnRestorePool = document.getElementById('btnRestoreOriginalPool');
    if (btnRestorePool)
    {
      btnRestorePool.addEventListener('click', restoreOriginalPool);
    }
    window.updatePoolToPruned = updatePoolToPruned;
    window.restoreOriginalPool = restoreOriginalPool;

    const inputReconK = document.getElementById('inputReconK');
    const sliderReconK = document.getElementById('sliderReconK');
    if (inputReconK && sliderReconK) {
      inputReconK.addEventListener('input', () => {
        sliderReconK.value = inputReconK.value;
      });
      sliderReconK.addEventListener('input', () => {
        inputReconK.value = sliderReconK.value;
      });
    }

    const selectReconWeight = document.getElementById('selectReconWeight');
    const reconIdwRow = document.getElementById('reconIdwRow');
    if (selectReconWeight && reconIdwRow) {
      selectReconWeight.addEventListener('change', () => {
        reconIdwRow.style.display = (selectReconWeight.value === 'idw') ? 'flex' : 'none';
      });
    }

    // =========================================================================
    //  SIDEBAR PANELS RESIZING & COLLAPSE CONTROLLER (10 PANELS & 9 RESIZERS)
    // =========================================================================
    const panelConfigs = [
      { id: 'cardInputData', btnId: 'btnCollapseInputData', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardSettings', btnId: 'btnCollapseSettings', defaultFlex: 1.1, savedFlex: 1.1 },
      { id: 'cardDisplay', btnId: 'btnCollapseDisplay', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardCli', btnId: 'btnCollapseCli', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardResources', btnId: 'btnCollapseResources', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardTrace', btnId: 'btnCollapseTrace', defaultFlex: 1.1, savedFlex: 1.1 },
      { id: 'cardKnnSettings', btnId: 'btnCollapseKnnSettings', defaultFlex: 0.9, savedFlex: 0.9 },
      { id: 'cardKnnResources', btnId: 'btnCollapseKnnResources', defaultFlex: 0.9, savedFlex: 0.9 },
      { id: 'cardKnnTrace', btnId: 'btnCollapseKnnTrace', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardDimDensity', btnId: 'btnCollapseDimDensity', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardReconstruction', btnId: 'btnCollapseReconstruction', defaultFlex: 1.1, savedFlex: 1.1 }
    ];

    function getExpandedCardAbove(cardIndex) {
      for (let i = cardIndex; i >= 0; i--) {
        const card = document.getElementById(panelConfigs[i].id);
        if (card && card.style.display !== 'none' && !card.classList.contains('collapsed')) return card;
      }
      return null;
    }

    function getExpandedCardBelow(cardIndex) {
      for (let i = cardIndex; i < panelConfigs.length; i++) {
        const card = document.getElementById(panelConfigs[i].id);
        if (card && card.style.display !== 'none' && !card.classList.contains('collapsed')) return card;
      }
      return null;
    }

    function updateResizersVisibility() {
      for (let i = 0; i < panelConfigs.length - 1; i++) {
        const resizer = document.getElementById(`resizer${i + 1}`);
        if (!resizer) continue;
        const topCard = getExpandedCardAbove(i);
        const botCard = getExpandedCardBelow(i + 1);
        const isVisible = (topCard !== null && botCard !== null);
        resizer.classList.toggle('hidden', !isVisible);
      }
    }

    function togglePanelCollapse(cardId) {
      const card = document.getElementById(cardId);
      if (!card) return;

      const isCurrentlyCollapsed = card.classList.contains('collapsed');
      const shouldCollapse = !isCurrentlyCollapsed;

      card.classList.toggle('collapsed', shouldCollapse);
      card.classList.toggle('expanded', !shouldCollapse);

      card.style.flex = '0 0 auto';
      card.style.height = 'auto';

      updateResizersVisibility();
    }
    window.togglePanelCollapse = togglePanelCollapse;

    // Central Control Enablement & Dependency Synchronization
    function syncControlDependencies() {
      // 1. Target Selection Mode -> Shannon Entropy controls & Display Heatmap
      const isEntropy = (targetMode === 'entropy');
      const cardEntropy = document.getElementById('cardEntropySection');
      if (cardEntropy) {
        cardEntropy.classList.toggle('disabled', !isEntropy);
      }
      const optEntropyMap = document.getElementById('optEntropyMap');
      if (optEntropyMap) {
        optEntropyMap.classList.toggle('disabled', !isEntropy);
      }

      // 2. Prior & Subspace Acceleration Sliders
      const colTmMix = document.getElementById('colTmMix');
      const sliderTmMixEl = document.getElementById('sliderTmMix');
      const inputTmMixEl = document.getElementById('inputTmMix');
      if (colTmMix && sliderTmMixEl) {
        colTmMix.classList.toggle('disabled', !useTM);
        sliderTmMixEl.disabled = !useTM;
        if (inputTmMixEl) inputTmMixEl.disabled = !useTM;
      }

      const colPredHorizon = document.getElementById('colPredHorizon');
      const sliderPredHorizonEl = document.getElementById('sliderPredHorizon');
      const inputPredHorizonEl = document.getElementById('inputPredHorizon');
      if (colPredHorizon && sliderPredHorizonEl) {
        colPredHorizon.classList.toggle('disabled', !usePred);
        sliderPredHorizonEl.disabled = !usePred;
        if (inputPredHorizonEl) inputPredHorizonEl.disabled = !usePred;
      }

      const colMaxVis = document.getElementById('colMaxVis');
      const sliderMaxVisEl = document.getElementById('sliderMaxVis');
      const inputMaxVisEl = document.getElementById('inputMaxVis');
      if (colMaxVis && sliderMaxVisEl) {
        colMaxVis.classList.toggle('disabled', !useGprob);
        sliderMaxVisEl.disabled = !useGprob;
        if (inputMaxVisEl) inputMaxVisEl.disabled = !useGprob;
      }

      // -tiles -> -xtile toggle
      const optXTile = document.getElementById('optXTile');
      if (optXTile) {
        optXTile.classList.toggle('disabled', !useTiles);
      }
      const rowXTileDecay = document.getElementById('rowXTileDecay');
      if (rowXTileDecay) {
        rowXTileDecay.style.display = (useTiles && useXTile) ? 'flex' : 'none';
      }
      const rowSparseDccExtra = document.getElementById(
        'rowSparseDccExtra'
      );
      if (rowSparseDccExtra) {
        rowSparseDccExtra.style.display =
          useSparseDcc ? 'flex' : 'none';
      }

      // Leader shortcut & Bayes Sigma conditional rows
      const rowLeaderCutoff = document.getElementById('rowLeaderCutoff');
      if (rowLeaderCutoff) {
        rowLeaderCutoff.style.display = (isEntropy && entropyLeaderShortcut) ? 'flex' : 'none';
      }
      const rowBayesSigma = document.getElementById('rowBayesSigma');
      if (rowBayesSigma) {
        rowBayesSigma.style.display = useSoftBayesian ? 'flex' : 'none';
      }

      // 3. Cluster Budget & Eviction (-maxcl)
      const isBudgeted = (maxcl > 0);
      const sectionEviction = document.getElementById('sectionEvictionStrategy');
      if (sectionEviction) {
        sectionEviction.classList.toggle('disabled', !isBudgeted);
      }
      const rowDiscardFrac = document.getElementById('rowDiscardFrac');
      if (rowDiscardFrac) {
        rowDiscardFrac.style.display = (isBudgeted && maxclStrategy === 'discard') ? 'flex' : 'none';
      }

      // 4. Sample Truncated Gaussian Noise
      const hasNoise = (noiseSigma > 1e-6);
      const rowNoiseTrunc = document.getElementById('rowNoiseTrunc');
      const sliderNoiseTruncEl = document.getElementById('sliderNoiseTrunc');
      const inputNoiseTruncEl = document.getElementById('inputNoiseTrunc');
      if (rowNoiseTrunc && sliderNoiseTruncEl) {
        rowNoiseTrunc.classList.toggle('disabled', !hasNoise);
        sliderNoiseTruncEl.disabled = !hasNoise;
        if (inputNoiseTruncEl) inputNoiseTruncEl.disabled = !hasNoise;
      }

      // 5. 3D Mode Camera Presets
      const is3D = (currentDim >= 3);
      const card3DPresetsSide = document.getElementById('card3DPresetsSide');
      if (card3DPresetsSide) {
        card3DPresetsSide.classList.toggle('disabled', !is3D);
      }

      // 6. k-NN Post-Processing Controls & Sidebar Panels Visibility
      const btnToggleKnnModule = document.getElementById('btnToggleKnnModule');
      if (btnToggleKnnModule) {
        btnToggleKnnModule.classList.toggle('active', enableKnn);
        btnToggleKnnModule.classList.toggle('toggle-active', enableKnn);
        btnToggleKnnModule.innerHTML = enableKnn ? '✓ k-NN Enabled' : '⚡ Enable k-NN';
      }

      const btnToggleKnn = document.getElementById('btnToggleKnn');
      if (btnToggleKnn) {
        btnToggleKnn.classList.toggle('active', enableKnn);
        btnToggleKnn.classList.toggle('toggle-active', enableKnn);
      }

      const knnExpandedGroup = document.getElementById('knnExpandedGroup');
      if (knnExpandedGroup) {
        knnExpandedGroup.style.display = enableKnn ? 'inline-flex' : 'none';
      }

      const knnStatusBadgeTop = document.getElementById('knnStatusBadgeTop');
      if (knnStatusBadgeTop) {
        if (!isKnnComputing) {
          knnStatusBadgeTop.textContent = `k=${knnK} • ${knnDirection} • dt≥${knnDtmin}`;
        }
      }

      const optEnableKnnEl = document.getElementById('optEnableKnn');
      const knnControlsContainer = document.getElementById('knnControlsContainer');
      if (optEnableKnnEl) {
        optEnableKnnEl.classList.toggle('active', enableKnn);
      }
      if (knnControlsContainer) {
        knnControlsContainer.style.display = enableKnn ? 'flex' : 'none';
      }

      // Show or hide the 3 k-NN sidebar panels and their resizers
      const knnCardIds = ['cardKnnSettings', 'cardKnnResources', 'cardKnnTrace'];
      knnCardIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          el.style.display = enableKnn ? 'flex' : 'none';
        }
      });

      const knnResizerIds = ['resizer6', 'resizer7', 'resizer8'];
      knnResizerIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          el.style.display = enableKnn ? '' : 'none';
        }
      });

      if (typeof updateResizersVisibility === 'function') {
        updateResizersVisibility();
      }

      updateKnnButtonUI(isKnnComputing);

      if (typeof updateCliCommand === 'function') {
        updateCliCommand();
      }
    }

    window.syncControlDependencies = syncControlDependencies;
    syncControlDependencies();

    function initSidebarResizers() {
      function setupResizer(resizerId, getTopCard, getBottomCard) {
        const resizer = document.getElementById(resizerId);
        if (!resizer) return;

        resizer.addEventListener('mousedown', (e) => {
          e.preventDefault();
          const topCard = getTopCard();
          const bottomCard = getBottomCard();
          if (!topCard || !bottomCard) return;
          if (topCard.classList.contains('collapsed') || bottomCard.classList.contains('collapsed')) return;

          resizer.classList.add('dragging');
          document.body.style.cursor = 'row-resize';
          document.body.style.userSelect = 'none';

          const startY = e.clientY;
          const topRect = topCard.getBoundingClientRect();
          const bottomRect = bottomCard.getBoundingClientRect();
          const totalHeight = topRect.height + bottomRect.height;
          const minH = 65;

          const topCfg = panelConfigs.find(p => p.id === topCard.id);
          const botCfg = panelConfigs.find(p => p.id === bottomCard.id);
          const initialCombinedFlex = (topCfg ? topCfg.savedFlex : 1.0) + (botCfg ? botCfg.savedFlex : 1.0);

          function onMouseMove(moveEvent) {
            const deltaY = moveEvent.clientY - startY;
            let newTopH = topRect.height + deltaY;
            let newBottomH = bottomRect.height - deltaY;

            if (newTopH < minH) {
              newTopH = minH;
              newBottomH = totalHeight - minH;
            } else if (newBottomH < minH) {
              newBottomH = minH;
              newTopH = totalHeight - minH;
            }

            const topRatio = newTopH / totalHeight;
            const bottomRatio = newBottomH / totalHeight;

            const flexTop = (topRatio * initialCombinedFlex).toFixed(3);
            const flexBottom = (bottomRatio * initialCombinedFlex).toFixed(3);

            topCard.style.flex = `${flexTop} 1 0px`;
            bottomCard.style.flex = `${flexBottom} 1 0px`;

            if (topCfg) topCfg.savedFlex = parseFloat(flexTop);
            if (botCfg) botCfg.savedFlex = parseFloat(flexBottom);
          }

          function onMouseUp() {
            resizer.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
          }

          window.addEventListener('mousemove', onMouseMove);
          window.addEventListener('mouseup', onMouseUp);
        });
      }

      for (let i = 0; i < panelConfigs.length - 1; i++) {
        const resizerId = `resizer${i + 1}`;
        const topIdx = i;
        const botIdx = i + 1;
        setupResizer(
          resizerId,
          () => getExpandedCardAbove(topIdx),
          () => getExpandedCardBelow(botIdx)
        );
      }

      updateResizersVisibility();
    }

    function initLayoutResizer() {
      const layoutResizer = document.getElementById('layoutResizer');
      const sidePanel = document.getElementById('sidePanel');
      const mainLayout = document.getElementById('mainLayout');
      if (!layoutResizer || !sidePanel || !mainLayout) return;

      let isDragging = false;
      let startX = 0;
      let startWidth = 0;

      layoutResizer.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startWidth = sidePanel.getBoundingClientRect().width;
        layoutResizer.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();

        function onMouseMove(moveEvent) {
          if (!isDragging) return;
          const deltaX = startX - moveEvent.clientX; // Moving left expands sidebar
          const layoutRect = mainLayout.getBoundingClientRect();
          const minW = 280;
          const maxW = Math.max(minW, layoutRect.width - 320);
          const newWidth = Math.round(Math.max(minW, Math.min(maxW, startWidth + deltaX)));

          sidePanel.style.width = `${newWidth}px`;
          sidePanel.style.minWidth = `${newWidth}px`;
          resizeCanvas();
        }

        function onMouseUp() {
          if (isDragging) {
            isDragging = false;
            layoutResizer.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
            resizeCanvas();
          }
        }

        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
      });
    }

    // Ensure hovered cluster highlight is strictly active only when mouse pointer is over the cluster table
    const candidateContainerEl = document.getElementById('candidateContainer');
    if (candidateContainerEl) {
      candidateContainerEl.addEventListener('mouseleave', () => {
        setHoveredCluster(-1);
      });
    }

    document.addEventListener('mousemove', (e) => {
      if (hoveredClusterId !== -1) {
        const inClusterRow = e.target && e.target.closest && e.target.closest('#candidateContainer .cluster-row');
        if (!inClusterRow) {
          setHoveredCluster(-1);
        }
      }
    });

    // Distance Trace Curve Hover Handlers
    const distCanvas = document.getElementById('distCurvesCanvas');
    if (distCanvas) {
      distCanvas.addEventListener('mousemove', (e) => {
        const n = distHistoryDFC.length;
        if (n === 0) return;
        const rect = distCanvas.getBoundingClientRect();
        const mouseX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
        hoverDistIndex = n > 1 ? Math.max(0, Math.min(n - 1, Math.round((mouseX / rect.width) * (n - 1)))) : 0;
        drawDistCurves();
      });
      distCanvas.addEventListener('mouseleave', () => {
        hoverDistIndex = null;
        drawDistCurves();
      });
    }

    const distAvgCanvas = document.getElementById('distAvgCurvesCanvas');
    if (distAvgCanvas) {
      distAvgCanvas.addEventListener('mousemove', (e) => {
        const n = distHistoryDFC.length;
        if (n === 0) return;
        const rect = distAvgCanvas.getBoundingClientRect();
        const mouseX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
        hoverDistAvgIndex = n > 1 ? Math.max(0, Math.min(n - 1, Math.round((mouseX / rect.width) * (n - 1)))) : 0;
        drawDistAvgCurves();
      });
      distAvgCanvas.addEventListener('mouseleave', () => {
        hoverDistAvgIndex = null;
        drawDistAvgCurves();
      });
    }

    // Transition Matrix Heatmap Hover & Touch Handlers
    function setupTMCanvasListeners() {
      const cvs = document.getElementById('tmHeatmapCanvas');
      if (!cvs) return;

      function handleTMInteraction(clientX, clientY) {
        const layout = cvs._tmLayout;
        if (!layout || layout.K === 0 || transitionCounts.length === 0) return;
        const rect = cvs.getBoundingClientRect();
        const mouseX = (clientX - rect.left) * (cvs.width / rect.width);
        const mouseY = (clientY - rect.top) * (cvs.height / rect.height);

        const { gridX, gridY, S, cw, ch, K } = layout;

        if (mouseX >= gridX && mouseX < gridX + S && mouseY >= gridY && mouseY < gridY + S) {
          const col = Math.max(0, Math.min(K - 1, Math.floor((mouseX - gridX) / cw)));
          const row = Math.max(0, Math.min(K - 1, Math.floor((mouseY - gridY) / ch)));

          hoveredTMCell = { i: row, j: col };
          let sum = 0;
          if (transitionCounts[row]) {
            for (let k = 0; k < K; k++) sum += transitionCounts[row][k] || 0;
          }
          const cnt = (transitionCounts[row] && transitionCounts[row][col]) ? transitionCounts[row][col] : 0;
          const prob = sum > 0 ? (cnt / sum) : 0.0;

          const desc = `C${row} → C${col}: Count=${cnt}, P=${(prob * 100).toFixed(1)}% (Row total: ${sum})`;
          const tipEl = document.getElementById('tmCellTooltip');
          if (tipEl) tipEl.innerText = desc;
          drawTransitionMatrix('tmHeatmapCanvas', false);
          draw();
        } else {
          if (hoveredTMCell !== null) {
            hoveredTMCell = null;
            const tipEl = document.getElementById('tmCellTooltip');
            if (tipEl) tipEl.innerText = 'Hover over any cell to inspect transition details';
            drawTransitionMatrix('tmHeatmapCanvas', false);
            draw();
          }
        }
      }

      function clearTMInteraction() {
        if (hoveredTMCell !== null) {
          hoveredTMCell = null;
          const tipEl = document.getElementById('tmCellTooltip');
          if (tipEl) tipEl.innerText = 'Hover over any cell to inspect transition details';
          drawTransitionMatrix('tmHeatmapCanvas', false);
          draw();
        }
      }

      cvs.addEventListener('mousemove', (e) => handleTMInteraction(e.clientX, e.clientY));
      cvs.addEventListener('mouseleave', clearTMInteraction);

      cvs.addEventListener('touchstart', (e) => {
        if (e.touches.length === 1) {
          handleTMInteraction(e.touches[0].clientX, e.touches[0].clientY);
        }
      }, { passive: true });

      cvs.addEventListener('touchmove', (e) => {
        if (e.touches.length === 1) {
          handleTMInteraction(e.touches[0].clientX, e.touches[0].clientY);
        }
      }, { passive: true });

      cvs.addEventListener('touchend', clearTMInteraction);
      cvs.addEventListener('touchcancel', clearTMInteraction);
    }

    setupTMCanvasListeners();

    // CLI Command Display
    function updateCliCommand() {
      const el = document.getElementById('cliCommandOutput');
      if (!el) return;
      const cmd = buildCliCommand();
      el.textContent = cmd;
    }

    // Update CLI and WASM build info
    updateCliCommand();
    const hashEl = document.getElementById('wasmBuildHash');
    if (hashEl && typeof GricWasm !== 'undefined' && GricWasm.isReady && GricWasm.isReady()) {
      hashEl.textContent = GricWasm.getVersion();
    }

    // Copy CLI command to clipboard
    const btnCopy = document.getElementById('btnCopyCli');
    if (btnCopy) {
      btnCopy.addEventListener('click', () => {
        const el = document.getElementById('cliCommandOutput');
        if (!el) return;
        navigator.clipboard.writeText(el.textContent)
          .then(() => {
            btnCopy.textContent = '✅ Copied';
            setTimeout(() => {
              btnCopy.textContent = '📋 Copy';
            }, 1500);
          });
      });
    }

    // =========================================================================
    //  WORKSPACE & DUAL-MODE CONTROLLER (WASM vs Native CLI)
    // =========================================================================

    let currentViewingFile = null;
    let currentDataFilesTab = 'structures';

    function updateStorageModeBanner() {
      const iconMode = document.getElementById('iconStorageMode');
      const titleMode = document.getElementById('txtStorageModeTitle');
      const subMode = document.getElementById('txtStorageModeSubtitle');
      if (!iconMode || !titleMode || !subMode) return;

      if (isDesktopBackend) {
        iconMode.textContent = '💻';
        titleMode.textContent = 'Native Desktop Workspace';
        subMode.textContent = workspacePath || 'Local Filesystem Active';
      } else if (WebFs.isOpen()) {
        iconMode.textContent = '📁';
        titleMode.textContent = 'Linked Local Folder';
        subMode.textContent = `Local Folder: ${WebFs.getDirectoryName()}`;
      } else {
        iconMode.textContent = '⚡';
        titleMode.textContent = 'WebAssembly Sandbox';
        subMode.textContent = 'In-Browser Memory (Save/Download to Disk)';
      }
    }

    function switchDataFilesTab(tab) {
      currentDataFilesTab = tab;
      const tabNavStruct = document.getElementById('tabNavDataStructures');
      const tabNavWs = document.getElementById('tabNavWorkspaceFiles');
      const paneStruct = document.getElementById('tabPaneDataStructures');
      const paneWs = document.getElementById('tabPaneWorkspaceFiles');

      if (tab === 'workspace') {
        if (tabNavStruct) tabNavStruct.classList.remove('active');
        if (tabNavWs) tabNavWs.classList.add('active');
        if (paneStruct) paneStruct.style.display = 'none';
        if (paneWs) paneWs.style.display = 'block';
        refreshWorkspaceFiles();
      } else {
        if (tabNavWs) tabNavWs.classList.remove('active');
        if (tabNavStruct) tabNavStruct.classList.add('active');
        if (paneWs) paneWs.style.display = 'none';
        if (paneStruct) paneStruct.style.display = 'block';
        renderDataStructuresUI();
      }
    }
    window.switchDataFilesTab = switchDataFilesTab;

    function openFileViewerModal(title, category, badge, sizeBytes, content) {
      const modal = document.getElementById('modalFileViewer');
      const lblTitle = document.getElementById('fileViewerTitle');
      const lblSub = document.getElementById('fileViewerSubtitle');
      const preContent = document.getElementById('fileViewerContent');

      currentViewingFile = { title, category, badge, sizeBytes, content };

      if (lblTitle) lblTitle.textContent = title;
      if (lblSub) {
        const sizeStr = (sizeBytes > 1048576)
          ? `${(sizeBytes / 1048576).toFixed(2)} MB`
          : `${(sizeBytes / 1024).toFixed(1)} KB`;
        lblSub.textContent = `${category} • ${badge} • ${sizeStr}`;
      }
      if (preContent) preContent.textContent = content || '(Empty file)';

      if (modal) modal.style.display = 'flex';
    }
    window.openFileViewerModal = openFileViewerModal;

    function closeFileViewerModal() {
      const modal = document.getElementById('modalFileViewer');
      if (modal) modal.style.display = 'none';
    }
    window.closeFileViewerModal = closeFileViewerModal;

    function renderDataStructuresUI() {
      const listEl = document.getElementById('dataStructuresList');
      const badgeCount = document.getElementById('badgeDataFilesCount');
      if (!listEl || typeof DataManager === 'undefined') return;

      const data = DataManager.generateCurrentDataStructures();
      const structures = data.structures;

      let readyCount = 0;
      structures.forEach(s => { if (s.ready) readyCount++; });

      if (badgeCount) {
        badgeCount.textContent = `${readyCount}/${structures.length} ready`;
      }

      let html = `
        <table class="data-files-table">
          <thead>
            <tr>
              <th style="width: 32%;">File</th>
              <th style="width: 22%;">Role</th>
              <th style="width: 20%;">Summary</th>
              <th style="width: 12%; text-align: right;">Size</th>
              <th style="width: 14%; text-align: center;">Actions</th>
            </tr>
          </thead>
          <tbody>
      `;

      structures.forEach((s, idx) => {
        const sizeStr = (s.size > 1048576)
          ? `${(s.size / 1048576).toFixed(2)} MB`
          : `${(s.size / 1024).toFixed(1)} KB`;

        const badgeBg = s.ready ? 'rgba(56, 189, 248, 0.15)' : 'rgba(148, 163, 184, 0.1)';
        const badgeColor = s.ready ? '#38bdf8' : '#94a3b8';
        const statusColor = s.ready ? '#4ade80' : '#64748b';

        html += `
          <tr class="df-row" data-idx="${idx}">
            <td>
              <span class="df-filename btn-view-struct" title="${s.desc}">
                <span>${s.icon}</span>
                <span style="color: ${s.ready ? '#f1f5f9' : '#64748b'};">${s.filename}</span>
              </span>
            </td>
            <td>
              <span class="badge-pill"
                    style="background: ${badgeBg}; color: ${badgeColor};
                           font-size: 0.62rem; padding: 1px 5px;">
                ${s.category}
              </span>
            </td>
            <td>
              <span style="font-size: 0.64rem; color: ${statusColor};" title="${s.badge}">
                ${s.badge}
              </span>
            </td>
            <td style="text-align: right;">
              <span class="df-size">${sizeStr}</span>
            </td>
            <td>
              <div class="df-actions">
                <button class="df-btn-action btn-view-struct"
                        title="View file content" ${s.ready ? '' : 'disabled'}>👁️</button>
                <button class="df-btn-action btn-dl-struct"
                        title="Download file" ${s.ready ? '' : 'disabled'}>💾</button>
                <button class="df-btn-action btn-copy-struct"
                        title="Copy text" ${s.ready ? '' : 'disabled'}>📋</button>
              </div>
            </td>
          </tr>
        `;
      });

      html += `</tbody></table>`;
      listEl.innerHTML = html;

      // Attach event handlers
      listEl.querySelectorAll('.df-row').forEach((row) => {
        const idx = parseInt(row.getAttribute('data-idx'), 10);
        const s = structures[idx];
        if (!s) return;

        row.querySelectorAll('.btn-view-struct').forEach(btn => {
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (s.ready) {
              openFileViewerModal(s.filename, s.category, s.badge, s.size, s.content);
            }
          });
        });

        const btnDl = row.querySelector('.btn-dl-struct');
        if (btnDl) {
          btnDl.addEventListener('click', (e) => {
            e.stopPropagation();
            if (s.binaryBytes) {
              DataManager.downloadBinaryFile(s.filename, s.binaryBytes);
            } else {
              DataManager.downloadTextFile(s.filename, s.content);
            }
            showToast(`💾 Downloaded ${s.filename}`);
          });
        }

        const btnCopy = row.querySelector('.btn-copy-struct');
        if (btnCopy) {
          btnCopy.addEventListener('click', (e) => {
            e.stopPropagation();
            if (navigator.clipboard) {
              navigator.clipboard.writeText(s.content).then(() => {
                showToast(`📋 Copied ${s.filename} to clipboard`);
              }).catch(() => {
                showToast(`📋 Copied ${s.filename}`);
              });
            } else {
              showToast(`📋 Copied ${s.filename}`);
            }
          });
        }
      });
    }
    window.renderDataStructuresUI = renderDataStructuresUI;

    function renderWorkspaceFilesTree() {
      const treeEl = document.getElementById('workspaceFilesTree');
      const headerEl = document.getElementById('lblWorkspaceTreeHeader');
      if (!treeEl) return;

      if (headerEl) {
        if (isDesktopBackend) {
          headerEl.textContent = `💻 Native Workspace (${workspaceFiles.length} items):`;
        } else if (WebFs.isOpen()) {
          headerEl.textContent = `📁 Local Folder (${WebFs.getDirectoryName()}):`;
        } else {
          headerEl.textContent = '⚡ In-Browser Sandbox (No local folder linked):';
        }
      }

      if (!isDesktopBackend && !WebFs.isOpen()) {
        treeEl.innerHTML = `
          <div style="padding: 10px; color: var(--text-muted); text-align: center;
                      line-height: 1.4;">
            <div style="font-size: 1.05rem; margin-bottom: 4px;">⚡ Web Browser Sandbox</div>
            <div style="font-size: 0.70rem;">Working in client-side memory. Click <b>"Save All"</b> or <b>"ZIP"</b>
                 to export results to disk.</div>
            <div style="margin-top: 5px; font-size: 0.66rem; color: #38bdf8;">
              Tip: Click <b>"Open"</b> in the top Workspace bar to link a local folder directly.
            </div>
          </div>
        `;
        return;
      }

      if (workspaceFiles.length === 0) {
        treeEl.innerHTML = `
          <div style="padding: 8px; color: var(--text-muted); text-align: center; font-size: 0.70rem;">
            (Directory is empty)
          </div>
        `;
        return;
      }

      let html = `
        <table class="data-files-table">
          <thead>
            <tr>
              <th style="width: 46%;">Name</th>
              <th style="width: 24%;">Type</th>
              <th style="width: 16%; text-align: right;">Size</th>
              <th style="width: 14%; text-align: center;">Action</th>
            </tr>
          </thead>
          <tbody>
      `;

      workspaceFiles.forEach((f, idx) => {
        const isDir = f.is_dir || f.isDir;
        const icon = isDir
          ? '📁'
          : (f.name.endsWith('.bin') ? '⚡' : (f.name.endsWith('.fits') ? '🌌' : '📄'));
        const sizeStr = !isDir ? `${(f.size / 1024).toFixed(1)} KB` : '—';
        const typeStr = isDir
          ? (f.name.includes('cluster') ? 'Cluster Dir' : 'Directory')
          : (f.name.endsWith('.bin')
             ? 'GRIC Binary'
             : (f.name.endsWith('.fits') ? 'FITS' : 'ASCII Text'));
        const typeBg = isDir
          ? 'rgba(250, 204, 21, 0.12)'
          : (f.name.endsWith('.bin') ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255, 255, 255, 0.06)');
        const typeColor = isDir
          ? '#facc15'
          : (f.name.endsWith('.bin') ? '#38bdf8' : '#94a3b8');

        html += `
          <tr class="df-row" data-ws-idx="${idx}">
            <td>
              <span class="df-filename ${!isDir ? 'btn-view-file' : ''}"
                    style="cursor: ${isDir ? 'default' : 'pointer'};">
                <span>${icon}</span>
                <span style="color: ${isDir ? '#facc15' : '#e2e8f0'};
                             font-weight: ${isDir ? '700' : '400'};">
                  ${f.name}
                </span>
              </span>
            </td>
            <td>
              <span class="badge-pill"
                    style="background: ${typeBg}; color: ${typeColor};
                           font-size: 0.60rem; padding: 1px 4px;">
                ${typeStr}
              </span>
            </td>
            <td style="text-align: right;">
              <span class="df-size">${sizeStr}</span>
            </td>
            <td style="text-align: center;">
              ${isDir && f.name.includes('cluster')
                ? `<button class="df-btn-action btn-load-cluster"
                           style="color: #4ade80; border-color: rgba(74, 222, 128, 0.4);
                                  font-weight: 700;">Load</button>`
                : (!isDir ? `<button class="df-btn-action btn-view-file"
                                     title="View file">👁️</button>` : '')}
            </td>
          </tr>
        `;
      });

      html += `</tbody></table>`;
      treeEl.innerHTML = html;

      // Attach event listeners
      treeEl.querySelectorAll('.df-row').forEach((row) => {
        const idx = parseInt(row.getAttribute('data-ws-idx'), 10);
        const f = workspaceFiles[idx];
        if (!f) return;

        const btnLoad = row.querySelector('.btn-load-cluster');
        if (btnLoad) {
          btnLoad.addEventListener('click', async (e) => {
            e.stopPropagation();
            await loadClusterResults(f.name);
          });
        }

        row.querySelectorAll('.btn-view-file').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
              let text = '';
              if (isDesktopBackend) {
                text = await DesktopBridge.readFile(f.name);
              } else if (WebFs.isOpen()) {
                text = await WebFs.readFile(f.name);
              }
              openFileViewerModal(f.name, 'Workspace File', 'Local Disk', f.size, text);
            } catch (err) {
              showToast(`Failed to read ${f.name}: ${err.message}`);
            }
          });
        });
      });
    }
    window.renderWorkspaceFilesTree = renderWorkspaceFilesTree;

    async function initWorkspaceAndEngine() {
      const lblPath = document.getElementById('lblWorkspacePath');
      const btnCliMode = document.getElementById('btnEngineCli');
      const btnWasmMode = document.getElementById('btnEngineWasm');
      const btnOpenFolder = document.getElementById('btnOpenLocalFolder');
      const btnRefresh = document.getElementById('btnRefreshWorkspace');
      const btnSaveWorkspace = document.getElementById('btnSaveToWorkspace');
      const btnSaveAllData = document.getElementById('btnSaveAllDataStructures');
      const btnDownloadZip = document.getElementById('btnDownloadAllZip');
      const btnRefreshTree = document.getElementById('btnRefreshWorkspaceTree');
      const cliNotice = document.getElementById('cliWebNotice');
      const cliControls = document.getElementById('cliDesktopControls');

      // Probe native C gric-server
      const serverInfo = await DesktopBridge.probe();
      if (serverInfo && !DesktopBridge.isMobileDevice()) {
        isDesktopBackend = true;
        workspacePath = serverInfo.cwd;
        if (lblPath) {
          lblPath.textContent = workspacePath;
          lblPath.title = workspacePath;
        }
        await refreshWorkspaceFiles();

        // Default to Native C Engine mode when desktop backend is available
        await setEngineMode('cli', true);
      } else {
        isDesktopBackend = false;
        if (lblPath) {
          lblPath.textContent = DesktopBridge.isMobileDevice()
            ? '📱 Cell Phone Client (In-Browser Sandbox)'
            : 'Web Browser Sandbox (Client-Side Storage)';
        }
        if (btnOpenFolder && WebFs.isSupported() && !DesktopBridge.isMobileDevice()) {
          btnOpenFolder.style.display = 'inline-block';
        } else if (btnOpenFolder) {
          btnOpenFolder.style.display = 'none';
        }
        if (cliNotice) {
          cliNotice.style.display = 'block';
          if (DesktopBridge.isMobileDevice()) {
            cliNotice.innerHTML = '📱 <b>Mobile Phone Mode:</b> In-Browser WebAssembly (WASM) ' +
              'is active with SIMD hardware acceleration. Native CLI is disabled on mobile.';
          }
        }
        if (cliControls) cliControls.style.display = 'none';

        await setEngineMode('wasm', true);
      }

      updateStorageModeBanner();
      updateEngineModeUI();
      renderDataStructuresUI();

      // Bind Engine Switcher Toggle-Slider
      const engineToggleSlider = document.getElementById('engineToggleSlider');
      if (engineToggleSlider) {
        engineToggleSlider.addEventListener('click', (e) => {
          if (!DesktopBridge.isNativeSupported()) {
            if (DesktopBridge.isMobileDevice()) {
              showToast('📱 Cell Phone: Native CLI is disabled (In-Browser WASM active)');
            } else {
              showToast('🌐 Web Mode: Native CLI requires a local desktop gric-server');
            }
            setEngineMode('wasm');
            return;
          }
          if (e.target.id === 'btnEngineWasm') {
            setEngineMode('wasm');
          } else if (e.target.id === 'btnEngineCli') {
            setEngineMode('cli');
          } else {
            setEngineMode(engineMode === 'wasm' ? 'cli' : 'wasm');
          }
        });
      }
      if (btnWasmMode) {
        btnWasmMode.addEventListener('click', (e) => {
          e.stopPropagation();
          setEngineMode('wasm');
        });
      }
      if (btnCliMode) {
        btnCliMode.addEventListener('click', (e) => {
          e.stopPropagation();
          if (!DesktopBridge.isNativeSupported()) {
            if (DesktopBridge.isMobileDevice()) {
              showToast('📱 Cell Phone: Native CLI is disabled (In-Browser WASM active)');
            } else {
              showToast('🌐 Web Mode: Native CLI requires a local desktop gric-server');
            }
            return;
          }
          setEngineMode('cli');
        });
      }

      // Bind Workspace buttons
      if (btnRefresh) {
        btnRefresh.addEventListener('click', async () => {
          await refreshWorkspaceFiles();
          showToast('🔄 Workspace refreshed');
        });
      }

      if (btnRefreshTree) {
        btnRefreshTree.addEventListener('click', async () => {
          await refreshWorkspaceFiles();
          showToast('🔄 Workspace refreshed');
        });
      }

      if (btnSaveWorkspace) {
        btnSaveWorkspace.addEventListener('click', async () => {
          await DataManager.saveAllToDisk();
        });
      }

      if (btnSaveAllData) {
        btnSaveAllData.addEventListener('click', async () => {
          await DataManager.saveAllToDisk();
        });
      }

      if (btnDownloadZip) {
        btnDownloadZip.addEventListener('click', () => {
          DataManager.downloadZipBundle();
        });
      }

      if (btnOpenFolder) {
        btnOpenFolder.addEventListener('click', async () => {
          const dir = await WebFs.openDirectory();
          if (dir) {
            if (lblPath) {
              lblPath.textContent = `📁 ${dir.name} (Client Local)`;
            }
            updateStorageModeBanner();
            await refreshWorkspaceFiles();
            showToast(`📁 Opened local folder: ${dir.name}`);
          }
        });
      }

      // Modal File Viewer Buttons
      const btnCopyModal = document.getElementById('btnCopyFileViewer');
      if (btnCopyModal) {
        btnCopyModal.addEventListener('click', () => {
          if (currentViewingFile && currentViewingFile.content) {
            if (navigator.clipboard) {
              navigator.clipboard.writeText(currentViewingFile.content).then(() => {
                showToast(`📋 Copied ${currentViewingFile.title} to clipboard`);
              }).catch(() => {
                showToast(`📋 Copied ${currentViewingFile.title}`);
              });
            } else {
              showToast(`📋 Copied ${currentViewingFile.title}`);
            }
          }
        });
      }

      const btnDlModal = document.getElementById('btnDownloadFileViewer');
      if (btnDlModal) {
        btnDlModal.addEventListener('click', () => {
          if (currentViewingFile && currentViewingFile.content) {
            DataManager.downloadTextFile(currentViewingFile.title, currentViewingFile.content);
            showToast(`💾 Downloaded ${currentViewingFile.title}`);
          }
        });
      }

      const btnCloseModal = document.getElementById('btnCloseFileViewer');
      if (btnCloseModal) {
        btnCloseModal.addEventListener('click', closeFileViewerModal);
      }

      // Bind Banner toggle
      const btnBannerToggle = document.getElementById('btnBannerToggleEngine');
      if (btnBannerToggle) {
        btnBannerToggle.addEventListener('click', () => {
          setEngineMode(engineMode === 'wasm' ? 'cli' : 'wasm');
        });
      }

      updateEngineModeUI();
      setupCliRunnerListeners();
    }

    async function refreshWorkspaceFiles() {
      let files = [];
      let shmStreams = [];
      if (isDesktopBackend) {
        files = await DesktopBridge.listFiles();
        shmStreams = await DesktopBridge.listShmStreams();
      } else if (WebFs.isOpen()) {
        files = await WebFs.listFiles();
      }
      workspaceFiles = files;

      updateStorageModeBanner();
      renderWorkspaceFilesTree();

      const selCli = document.getElementById('selectCliDataset');
      if (selCli) {
        const prevVal = selCli.value;
        selCli.innerHTML = '<option value="">(Select dataset from workspace or SHM...)</option>';

        if (shmStreams.length > 0) {
          const shmGroup = document.createElement('optgroup');
          shmGroup.label = '📡 Live Shared Memory Streams (ImageStreamIO)';
          shmStreams.forEach(s => {
            const opt = document.createElement('option');
            opt.value = `shm:${s.name}`;
            opt.textContent = `📡 ${s.name} (${(s.size / (1024 * 1024)).toFixed(1)} MB)`;
            shmGroup.appendChild(opt);
          });
          selCli.appendChild(shmGroup);
        }

        const fileGroup = document.createElement('optgroup');
        fileGroup.label = '📁 Local Workspace Files';
        files.forEach(f => {
          if (!f.isDir) {
            const opt = document.createElement('option');
            opt.value = f.name;
            opt.textContent = `${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
            fileGroup.appendChild(opt);
          }
        });
        selCli.appendChild(fileGroup);

        if (prevVal) selCli.value = prevVal;
      }
    }

    async function setEngineMode(mode, silent = false) {
      if (mode === 'cli' && !DesktopBridge.isNativeSupported()) {
        if (!silent) {
          if (DesktopBridge.isMobileDevice()) {
            showToast('📱 Cell Phone: Native CLI is disabled (In-Browser WASM active)');
          } else {
            showToast('🌐 Web Mode: Native CLI requires a local desktop gric-server');
          }
        }
        mode = 'wasm';
      }

      engineMode = mode;
      useWasm = (mode === 'wasm');
      useJSFallback = (mode === 'js');
      updateEngineModeUI();

      if (mode === 'cli') {
        updateCliCommand();

        const cardCli = document.getElementById('cardCli');
        if (cardCli && cardCli.classList.contains('collapsed')) {
          cardCli.classList.remove('collapsed');
        }

        if (DesktopBridge.isAvailable()) {
          await DesktopBridge.initCliSession();
        }
        if (!silent) {
          showToast('💻 Native CLI mode active (tmux session "gric_cli" ready)');
        }
      } else {
        if (DesktopBridge.isAvailable()) {
          await DesktopBridge.stopCliSession();
        }
        if (typeof GricWasm !== 'undefined' && GricWasm.isLoaded()) {
          const params = GricWasm.buildParamsFromState();
          if (!wasmSessionActive || !GricWasm.isReady() || (GricWasm.isConfigChanged && GricWasm.isConfigChanged(params))) {
            wasmSessionActive = GricWasm.init(params);
            if (typeof updateWasmBadge === 'function') updateWasmBadge();
          }
        }
        if (!silent) {
          showToast('⚡ Switched to In-Browser WebAssembly (WASM)');
        }
      }
    }

    function setupCliRunnerListeners() {
      const btnRun = document.getElementById('btnRunCli');
      const btnKill = document.getElementById('btnKillCli');
      const btnClear = document.getElementById('btnClearCliConsole');
      const btnLoadManual = document.getElementById('btnLoadClusterDatManual');
      const chkAutoLoad = document.getElementById('chkAutoLoadResults');
      const selSideInputMode = document.getElementById('selectInputModeSide');
      const pnlSideStreamConfig = document.getElementById('sideStreamConfigPanel');
      const selSideFps = document.getElementById('selectSideStreamFps');
      const chkSideLoop = document.getElementById('chkSideStreamLoop');
      const chkSideCnt2 = document.getElementById('chkSideStreamCnt2Sync');
      const badgeIngestion = document.getElementById('cliIngestionSourceBadge');

      const updateIngestionStatus = () => {
        const isStream = (selSideInputMode && selSideInputMode.value === 'stream');
        if (pnlSideStreamConfig) {
          pnlSideStreamConfig.style.display = isStream ? 'flex' : 'none';
        }
        if (badgeIngestion) {
          if (isStream) {
            const fpsVal = selSideFps ? selSideFps.value : '0';
            const fpsLabel = (fpsVal === '0') ? 'Lockstep (cnt2sync)' : `${fpsVal} FPS`;
            badgeIngestion.textContent = `⚡ ImageStreamIO (${fpsLabel})`;
            badgeIngestion.style.color = '#38bdf8';
          } else {
            badgeIngestion.textContent = '📁 Direct File / Memory';
            badgeIngestion.style.color = '#4ade80';
          }
        }
        if (isStream && engineMode !== 'cli') {
          setEngineMode('cli');
        }
      };

      if (selSideInputMode) {
        selSideInputMode.addEventListener('change', updateIngestionStatus);
      }
      if (selSideFps) {
        selSideFps.addEventListener('change', updateIngestionStatus);
      }
      updateIngestionStatus();

      if (chkAutoLoad) {
        chkAutoLoad.addEventListener('change', (e) => {
          autoLoadCliResults = e.target.checked;
        });
      }

      if (btnClear) {
        btnClear.addEventListener('click', () => {
          const consoleEl = document.getElementById('cliConsoleLog');
          if (consoleEl) consoleEl.textContent = '[Console cleared]';
        });
      }

      if (btnRun) {
        btnRun.addEventListener('click', async () => {
          await runNativeCli();
        });
      }

      const btnRunCliKnn = document.getElementById('btnRunCliKnn');
      if (btnRunCliKnn) {
        btnRunCliKnn.addEventListener('click', async () => {
          await executeKnnComputation();
        });
      }

      if (btnKill) {
        btnKill.addEventListener('click', async () => {
          await DesktopBridge.killActiveJob();
          showToast('🛑 Abort signal sent to CLI job');
        });
      }

      if (btnLoadManual) {
        btnLoadManual.addEventListener('click', async () => {
          const clusterDirs = workspaceFiles.filter(f => f.isDir && f.name.includes('cluster'));
          if (clusterDirs.length === 0) {
            showToast('No .clusterdat folders found in workspace');
            return;
          }
          const choice = clusterDirs[0].name;
          await loadClusterResults(choice);
        });
      }

      // Tmux attach copy buttons
      const copyTmuxAttachCmd = () => {
        if (navigator.clipboard) {
          navigator.clipboard.writeText('tmux attach -t gric_cli').then(() => {
            showToast('📋 Copied: tmux attach -t gric_cli');
          }).catch(() => {
            showToast('tmux attach -t gric_cli');
          });
        } else {
          showToast('tmux attach -t gric_cli');
        }
      };

      const badgeNativeTmux = document.getElementById('badgeNativeTmux');
      if (badgeNativeTmux) {
        badgeNativeTmux.addEventListener('click', copyTmuxAttachCmd);
      }

      const btnCopyTmux = document.getElementById('btnCopyTmuxCmd');
      if (btnCopyTmux) {
        btnCopyTmux.addEventListener('click', () => {
          copyTmuxAttachCmd();
          btnCopyTmux.textContent = '✅ Copied';
          setTimeout(() => {
            btnCopyTmux.textContent = '📋 Copy Attach';
          }, 1500);
        });
      }
    }

    async function runNativeCli() {
      if (!isDesktopBackend) {
        showToast('Native CLI is only available in Desktop App mode');
        return;
      }

      const selCli = document.getElementById('selectCliDataset');
      let dataset = selCli ? selCli.value : '';
      if (!dataset) {
        dataset = `${currentBenchmark}.txt`;
      }

      let args = [];
      let isStreamInput = false;
      const inputMode = document.getElementById('selectInputModeSide')?.value || 'file';
      const isStreamingMode = (inputMode === 'stream');
      const streamFps = parseFloat(
        document.getElementById('selectSideStreamFps')?.value || '0'
      );
      const streamLoop =
        document.getElementById('chkSideStreamLoop')?.checked || false;
      const streamName = `gric_sim_${Date.now() % 100000}`;
      let streamJobOpts = null;

      if (isStreamingMode) {
        isStreamInput = true;
        const isSynthetic = !selCli || !selCli.value || dataset === `${currentBenchmark}.txt` ||
          (typeof BENCHMARK_DESCS !== 'undefined' &&
           BENCHMARK_DESCS[dataset.replace(/\.[^/.]+$/, '')]);

        if (isSynthetic || !dataset) {
          dataset = `${currentBenchmark}.txt`;
          if (!benchmarkDataset || benchmarkDataset.length === 0) {
            stageDataset();
          }

          let content = '';
          for (let i = 0; i < benchmarkDataset.length; i++) {
            const pt = benchmarkDataset[i];
            if (Array.isArray(pt) || ArrayBuffer.isView(pt) ||
                (pt && typeof pt.length === 'number')) {
              content += Array.from(pt).map(v => Number(v).toFixed(6)).join(' ') + '\n';
            } else if (pt && typeof pt === 'object') {
              if (pt.coords &&
                  (Array.isArray(pt.coords) || ArrayBuffer.isView(pt.coords))) {
                content +=
                  Array.from(pt.coords).map(v => Number(v).toFixed(6)).join(' ') + '\n';
              } else if (currentDim >= 3) {
                content += `${Number(pt.x || 0).toFixed(6)} ` +
                           `${Number(pt.y || 0).toFixed(6)} ` +
                           `${Number(pt.z || 0).toFixed(6)}\n`;
              } else {
                content += `${Number(pt.x || 0).toFixed(6)} ${Number(pt.y || 0).toFixed(6)}\n`;
              }
            }
          }
          try {
            await DesktopBridge.writeFile(dataset, content);
            await refreshWorkspaceFiles();
          } catch (err) {
            console.warn('[CLI] Could not write benchmark dataset file:', err);
          }
        }

        const baseName = dataset.replace(/\.[^/.]+$/, '');
        const clusterDir = `${baseName}.clusterdat`;
        const streamCnt2sync = (streamFps === 0) ||
          (document.getElementById('chkSideStreamCnt2Sync')?.checked ?? false);

        args = [rlim.toFixed(3), streamName, '-stream', '-outdir', clusterDir];
        if (streamCnt2sync) {
          args.push('-cnt2sync');
        }
        if (!streamLoop && benchmarkDataset && benchmarkDataset.length > 0) {
          args.push('-maxim', String(benchmarkDataset.length));
        }

        streamJobOpts = {
          streamFile: dataset,
          streamName: streamName,
          streamFps: streamFps,
          streamLoop: streamLoop,
          streamCnt2sync: streamCnt2sync
        };
      } else if (dataset.startsWith('shm:')) {
        const customStreamName = dataset.substring(4);
        isStreamInput = true;
        args = [rlim.toFixed(3), customStreamName, '-stream', '-outdir', `${customStreamName}.clusterdat`];
      } else {
        // If running active synthetic benchmark, always serialize full sequence with passes
        const isSynthetic = !selCli || !selCli.value || dataset === `${currentBenchmark}.txt` ||
          (typeof BENCHMARK_DESCS !== 'undefined' && BENCHMARK_DESCS[dataset.replace(/\.[^/.]+$/, '')]);

        if (isSynthetic) {
          dataset = `${currentBenchmark}.txt`;
          if (!benchmarkDataset || benchmarkDataset.length === 0) {
            stageDataset();
          }

          let content = '';
          for (let i = 0; i < benchmarkDataset.length; i++) {
            const pt = benchmarkDataset[i];
            if (Array.isArray(pt) || ArrayBuffer.isView(pt) || (pt && typeof pt.length === 'number')) {
              content += Array.from(pt).map(v => Number(v).toFixed(6)).join(' ') + '\n';
            } else if (pt && typeof pt === 'object') {
              if (pt.coords &&
                  (Array.isArray(pt.coords) || ArrayBuffer.isView(pt.coords))) {
                content +=
                  Array.from(pt.coords).map(v => Number(v).toFixed(6)).join(' ') + '\n';
              } else if (currentDim >= 3) {
                content += `${Number(pt.x || 0).toFixed(6)} ` +
                           `${Number(pt.y || 0).toFixed(6)} ` +
                           `${Number(pt.z || 0).toFixed(6)}\n`;
              } else {
                content += `${Number(pt.x || 0).toFixed(6)} ${Number(pt.y || 0).toFixed(6)}\n`;
              }
            }
          }
          try {
            await DesktopBridge.writeFile(dataset, content);
            await refreshWorkspaceFiles();
          } catch (err) {
            console.warn('[CLI] Could not write benchmark dataset file:', err);
          }
        }
        args = [rlim.toFixed(3), dataset];
      }

      if (pruneMode === '4P' || pruneMode === '5P') args.push('-te4');
      if (pruneMode === '5P') args.push('-te5');
      if (targetMode === 'entropy') {
        args.push('-entropy');
        if (entropyGate !== 2.0) args.push('-entropy_gate', entropyGate.toFixed(2));
        if (entropyFirstGate !== 4.0) args.push('-entropy_first_gate', entropyFirstGate.toFixed(2));
        if (entropyFastMode) args.push('-entropy_fast');
      }
      if (useTM && tmMixingCoeff > 0) args.push('-tm', tmMixingCoeff.toFixed(2));
      if (usePred) {
        if (predHorizon !== 2) {
          args.push('-pred[,,' + predHorizon + ']');
        } else {
          args.push('-pred');
        }
      }
      if (useGprob) {
        args.push('-gprob');
        if (maxVisitors !== 1000) args.push('-maxvis', maxVisitors.toString());
      }
      if (useSoftBayesian) {
        args.push('-soft_bayesian');
        if (softBayesianSigmaCoeff !== 1.0) {
          args.push('-soft_bayesian_sigma', softBayesianSigmaCoeff.toFixed(2));
        }
      }
      if (useTiles) args.push('-tiles');
      if (useXTile) args.push('-xtile');
      if (useSparseDcc) {
        args.push('-sparse_dcc');
        if (sparseDccExtraEvals > 0) {
          args.push('-sparse_dcc_extra_evals', sparseDccExtraEvals.toString());
        }
      }
      if (typeof clusterUseSq8 === 'boolean' && clusterUseSq8) {
        args.push('-sq8');
      }
      if (maxcl > 0) {
        args.push('-maxcl', maxcl.toString());
      } else {
        // In GUI, 0 = Unlimited: allocate large capacity headroom for native mode
        args.push('-maxcl', '10000');
      }
      if (maxclStrategy !== 'stop') {
        args.push('-maxcl_strategy', maxclStrategy);
      }
      if (maxclStrategy === 'discard' && discardFraction !== 0.10) {
        args.push('-discard_frac', discardFraction.toFixed(2));
      }
      args.push('-evals');

      const btnRun = document.getElementById('btnRunCli');
      const btnRunKnn = document.getElementById('btnRunCliKnn');
      const btnKill = document.getElementById('btnKillCli');
      const btnPlay = document.getElementById('btnPlay');
      const badgeStatus = document.getElementById('badgeCliStatus');
      const consoleEl = document.getElementById('cliConsoleLog');

      if (btnRun) btnRun.disabled = true;
      if (btnRunKnn) btnRunKnn.disabled = true;
      if (btnKill) btnKill.disabled = false;
      if (btnPlay) {
        btnPlay.innerHTML = '⏳ Running...';
        btnPlay.disabled = true;
      }
      if (badgeStatus) {
        badgeStatus.textContent = '● tmux: gric_cli';
        badgeStatus.style.background = 'rgba(74, 222, 128, 0.2)';
        badgeStatus.style.color = '#4ade80';
      }
      if (consoleEl) {
        const streamNote = isStreamingMode
          ? `📡 Streamer: gric-txt2stream ${dataset} ${streamName} -fps ${streamFps}${streamLoop ? ' -loop' : ''}\n`
          : '';
        consoleEl.textContent = `🚀 Dispatched in tmux session: gric_cli\n` +
          `🖥️ Attach live: tmux attach -t gric_cli\n` +
          `📄 Log stream: /tmp/gric_latest.log\n` +
          streamNote +
          `⚙️ Command: gric-cluster ${args.join(' ')}\n` +
          `─────────────────────────────────────────────────────────────\n`;
      }
      showToast('🚀 Native CLI launched in tmux session "gric_cli" (tmux attach -t gric_cli)');

      const tStart = performance.now();
      isCliRunning = true;
      knnResults = null;

      try {
        await DesktopBridge.runCliJob({
          cmd: 'gric-cluster',
          args: args,
          ...(streamJobOpts || {}),
          onOutput: (chunk) => {
            if (consoleEl) {
              consoleEl.textContent += chunk;
              consoleEl.scrollTop = consoleEl.scrollHeight;
            }
          },
          onTelemetry: (t) => {
            if (!t) return;

            // Live progress & frame counters from SHM
            if (t.total_frames > 0) {
              const pct = Math.min(100, Math.max(0, (t.processed_frames / t.total_frames) * 100));
              const fill = document.getElementById('progressFill');
              if (fill) fill.style.width = `${pct.toFixed(1)}%`;
              const fc = document.getElementById('frameCounter');
              if (fc) fc.textContent = `${t.processed_frames} / ${t.total_frames} (${pct.toFixed(1)}%)`;
            } else if (t.processed_frames > 0) {
              const fc = document.getElementById('frameCounter');
              if (fc) fc.textContent = `${t.processed_frames} frames (streaming)`;
            }

            const cb = document.getElementById('clusterBadge');
            if (cb) cb.textContent = `${t.num_clusters} clusters`;

            const fpsBadge = document.getElementById('fpsBadge');
            if (fpsBadge && t.elapsed_ms > 0) {
              const fps = Math.round(t.processed_frames / (t.elapsed_ms / 1000.0));
              fpsBadge.textContent = `${fps} fps`;
            }

            if (t.elapsed_ms > 0 && t.processed_frames > 0) {
              const liveFps = t.processed_frames / (t.elapsed_ms / 1000.0);
              sessionAvgFps = liveFps;
              avgComputeTimeMs = t.elapsed_ms / t.processed_frames;
              currentCpuLoadPct = 100.0;

              const statAvgFpsEl = document.getElementById('statAvgFps');
              if (statAvgFpsEl) {
                statAvgFpsEl.textContent = liveFps >= 1000
                  ? `${Math.round(liveFps).toLocaleString()} fps`
                  : `${liveFps.toFixed(1)} fps`;
              }
              const statCpuLoadEl = document.getElementById('statCpuLoad');
              if (statCpuLoadEl) {
                statCpuLoadEl.textContent = `${currentCpuLoadPct.toFixed(1)}%`;
              }
              const statComputeMsEl = document.getElementById('statComputeMs');
              if (statComputeMsEl) {
                statComputeMsEl.textContent = `${avgComputeTimeMs.toFixed(2)} ms`;
              }
            }

            // Sync resource tracker panel
            const statFramesEl = document.getElementById('statFrames');
            if (statFramesEl) statFramesEl.textContent = t.processed_frames;

            const statClustersEl = document.getElementById('statClusters');
            if (statClustersEl) statClustersEl.textContent = t.num_clusters;

            const statTotalTimeEl = document.getElementById('statTotalTime');
            if (statTotalTimeEl) statTotalTimeEl.textContent = formatClockTime(t.elapsed_ms);

            const statTotalDistsEl = document.getElementById('statTotalDists');
            if (statTotalDistsEl) statTotalDistsEl.textContent = (t.framedist_calls || 0).toLocaleString();

            const statDistRatioEl = document.getElementById('statDistRatio');
            if (statDistRatioEl) {
              statDistRatioEl.textContent = `${(t.framedist_sample || 0).toLocaleString()} / ${(t.framedist_intercluster || 0).toLocaleString()}`;
            }
            const statDccPopBadge = document.getElementById('statDccPopBadge');
            if (statDccPopBadge && t.dcc_entries_populated !== undefined) {
              statDccPopBadge.textContent = t.dcc_pairs_total > 0
                ? `[${(t.dcc_entries_populated || 0).toLocaleString()}/${(t.dcc_pairs_total || 0).toLocaleString()} pop]`
                : `[${(t.dcc_entries_populated || 0).toLocaleString()} pop]`;
            }

            const statMemoryTotalEl = document.getElementById('statMemoryTotal');
            if (statMemoryTotalEl && t.memory_rss_kb > 0) {
              statMemoryTotalEl.textContent = formatBytes(t.memory_rss_kb * 1024);
            }

            const statPrune3PEl = document.getElementById('statPrune3P');
            if (statPrune3PEl) statPrune3PEl.textContent = (t.clusters_pruned || 0).toLocaleString();

            const statPrune4PEl = document.getElementById('statPrune4P');
            if (statPrune4PEl) statPrune4PEl.textContent = (t.prune_4p_count || 0).toLocaleString();

            const statPrune5PEl = document.getElementById('statPrune5P');
            if (statPrune5PEl) statPrune5PEl.textContent = (t.prune_5p_count || 0).toLocaleString();

            const statPredHitsEl = document.getElementById('statPredHits');
            if (statPredHitsEl) statPredHitsEl.textContent = (t.pred_hits || 0).toLocaleString();

            const statPredAttemptsEl = document.getElementById('statPredAttempts');
            if (statPredAttemptsEl) statPredAttemptsEl.textContent = (t.pred_attempts || 0).toLocaleString();

            const statPredRateEl = document.getElementById('statPredRate');
            if (statPredRateEl && t.pred_attempts > 0) {
              const rate = ((t.pred_hits / t.pred_attempts) * 100).toFixed(1);
              statPredRateEl.textContent = `${rate}%`;
            }

            // Mobile HUD overlay
            const hudSamples = document.getElementById('hudSamples');
            if (hudSamples) hudSamples.textContent = t.processed_frames;
            const hudClusters = document.getElementById('hudClusters');
            if (hudClusters) hudClusters.textContent = t.num_clusters;
            const hudTime = document.getElementById('hudTime');
            if (hudTime) hudTime.textContent = formatClockTime(t.elapsed_ms);
          },
          onFinish: async (res) => {
            isCliRunning = false;
            const elapsed = ((performance.now() - tStart) / 1000).toFixed(2);
            if (btnRun) btnRun.disabled = false;
            if (btnRunKnn) btnRunKnn.disabled = false;
            if (btnKill) btnKill.disabled = true;
            if (btnPlay) {
              btnPlay.innerHTML = '▶ Run gric-cluster';
              btnPlay.disabled = false;
            }

            if (res.exitCode === 0 || res.exitCode === 137) {
              if (badgeStatus) {
                badgeStatus.textContent = `Done (${elapsed}s)`;
                badgeStatus.style.background = 'rgba(56, 189, 248, 0.2)';
                badgeStatus.style.color = '#38bdf8';
              }
              showToast(`✅ gric-cluster completed in ${elapsed}s`);

              if (autoLoadCliResults) {
                const baseName = dataset.startsWith('shm:')
                  ? dataset.substring(4)
                  : dataset.replace(/\.[^/.]+$/, '');
                const clusterDir = `${baseName}.clusterdat`;
                await loadClusterResults(clusterDir);
              }
            } else {
              if (badgeStatus) {
                badgeStatus.textContent = `Failed (exit ${res.exitCode})`;
                badgeStatus.style.background = 'rgba(248, 113, 113, 0.2)';
                badgeStatus.style.color = '#f87171';
              }
              showToast(`❌ CLI execution failed (exit ${res.exitCode})`);
            }

            await refreshWorkspaceFiles();
          }
        });
      } catch (err) {
        isCliRunning = false;
        if (btnRun) btnRun.disabled = false;
        if (btnKill) btnKill.disabled = true;
        if (btnPlay) {
          btnPlay.innerHTML = '▶ Run gric-cluster';
          btnPlay.disabled = false;
        }
        if (consoleEl) consoleEl.textContent += `\n[Error]: ${err.message}\n`;
        showToast(`Error: ${err.message}`);
      }
    }

    async function loadClusterResults(clusterDir) {
      try {
        const data = await DesktopBridge.parseClusterDatDir(clusterDir);
        if (!data.anchors || data.anchors.length === 0) {
          showToast(`No cluster centroids found in ${clusterDir}`);
          return;
        }

        if (isRunning) pauseSimulation();

        clusters = data.anchors.map((a, i) => ({
          id: i,
          x: a.x,
          y: a.y,
          z: a.z,
          anchor: a.anchor,
          members: a.members || 0,
          prob: 0,
          scDists: 0,
          lastActive: 0,
          color: getClusterColor(i)
        }));

        if (data.dcc && data.dcc.length > 0) {
          dcc = data.dcc;
        }

        // Populate pastSamples so points appear across the viewports
        if (benchmarkDataset && benchmarkDataset.length > 0) {
          pastSamples = benchmarkDataset.map((p, idx) => ({
            x: (Array.isArray(p) || ArrayBuffer.isView(p)) ? p[0] : (p.x || 0),
            y: (Array.isArray(p) || ArrayBuffer.isView(p)) ? p[1] : (p.y || 0),
            z: currentDim >= 3
              ? ((Array.isArray(p) || ArrayBuffer.isView(p)) ? (p[2] || 0) : (p.z || 0)) : 0,
            coords: p.coords || (Array.isArray(p) || ArrayBuffer.isView(p) ? p : null),
            frameIndex: idx
          }));
        } else {
          try {
            const baseName = clusterDir.replace(/\.clusterdat\/?$/, '');
            const txtContent = await DesktopBridge.readFile(`${baseName}.txt`);
            if (txtContent) {
              const lines = txtContent.split(/\r?\n/);
              const pts = [];
              for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line || line.startsWith('#')) continue;
                const tokens = line.split(/[,\s\t]+/).filter(t => t.length > 0);
                if (tokens.length >= 2) {
                  const coords = new Float64Array(tokens.length);
                  for (let d = 0; d < tokens.length; d++) {
                    coords[d] = parseFloat(tokens[d]) || 0;
                  }
                  pts.push({
                    x: coords[0],
                    y: coords[1],
                    z: coords.length >= 3 ? coords[2] : 0,
                    coords: coords,
                    frameIndex: pts.length
                  });
                }
                if (pts.length >= 10000) break;
              }
              if (pts.length > 0) {
                pastSamples = pts;
              }
            }
          } catch (e) {
            /* ignore */
          }
        }

        if (data.evals && data.evals.length > 0) {
          frameEvaluationsLog = data.evals;
        }
        if (data.assignments && data.assignments.length > 0) {
          assignmentHistory = data.assignments;
          imageFrameAssignments = data.assignments;
          imageClusterMembers = {};
          data.assignments.forEach((cId, fIdx) => {
            if (!imageClusterMembers[cId]) {
              imageClusterMembers[cId] = [];
            }
            imageClusterMembers[cId].push(fIdx);
          });
        }

        // Apply native execution stats to global telemetry & Resource Tracker
        let sumMembers = 0;
        clusters.forEach(c => { sumMembers += c.members; });

        if (data.stats && data.stats.frames > 0) {
          totalFrames = data.stats.frames;
          sessionStartFrames = 0;
          distSampleCluster = data.stats.sampleDists;
          distClusterCluster = data.stats.interclusterDists;
          distSampleClusterTotal = data.stats.sampleDists;
          distClusterClusterTotal = data.stats.interclusterDists;
          dccPopulated = data.stats.dccPopulated || 0;
          dccPairsTotal = data.stats.dccPairsTotal ||
            (clusters.length > 1 ? (clusters.length * (clusters.length - 1) / 2) : 0);
          totalEvals = data.stats.sampleDists;
          naiveEvals = data.stats.sampleDists + data.stats.pruned;
          sessionElapsedMs = data.stats.timeMs;
          sessionAvgFps = data.stats.timeMs > 0 ? (totalFrames / (data.stats.timeMs / 1000.0)) : 0.0;
          avgComputeTimeMs = totalFrames > 0 ? (data.stats.timeMs / totalFrames) : 0.0;
          currentCpuLoadPct = 100.0;
          sessionIsActive = false;

          if (data.stats.rssKb > 0) {
            const statMemoryTotalEl = document.getElementById('statMemoryTotal');
            if (statMemoryTotalEl) {
              statMemoryTotalEl.textContent = formatBytes(data.stats.rssKb * 1024);
            }
          }
        } else if (sumMembers > 0) {
          totalFrames = sumMembers;
          sessionStartFrames = 0;
          sessionIsActive = false;
        }

        // Compute Shannon entropy over cluster member distribution
        if (totalFrames > 0 && clusters.length > 0) {
          let hBits = 0.0;
          for (let i = 0; i < clusters.length; i++) {
            const p = clusters[i].members / totalFrames;
            clusters[i].prob = p;
            if (p > 0.0) {
              hBits -= p * Math.log2(p);
            }
          }
          currentEntropyBits = hBits;
        }

        // Update progress bar and frame counter HUD
        const progressFill = document.getElementById('progressFill');
        if (progressFill) progressFill.style.width = '100%';
        const frameCounter = document.getElementById('frameCounter');
        if (frameCounter) frameCounter.textContent = `${totalFrames} / ${totalFrames} (100.0%)`;

        // Reset k-NN post-processing results for newly computed/loaded clusters
        knnResults = null;
        if (typeof renderKnnTrace === 'function') {
          renderKnnTrace();
        }

        updateUI();
        if (typeof renderDataStructuresUI === 'function') {
          renderDataStructuresUI();
        }
        resizeCanvas();
        draw();
        requestAnimationFrame(() => {
          resizeCanvas();
          draw();
        });
        setTimeout(() => {
          resizeCanvas();
          draw();
        }, 80);

        showToast(`📊 Loaded ${clusters.length} clusters (${totalFrames.toLocaleString()} frames) from ${clusterDir}`);
      } catch (err) {
        console.error('[LoadResults] Error loading cluster results:', err);
        showToast(`Failed to load ${clusterDir}: ${err.message}`);
      }
    }

    async function exportRunToWorkspace() {
      await DataManager.saveAllToDisk();
    }

    // Initial Startup
    initSidebarResizers();
    initLayoutResizer();
    updateTMCanvasDimensions();
    resizeCanvas();
    if (typeof populateDefaultMultiDatasets === 'function') {
      populateDefaultMultiDatasets();
    } else {
      stageDataset('3Dspiral', 'B');
      stageDataset('3Drand', 'C');
      clearDatasetSlot('D');
    }
    activeDatasetSlot = 'A';
    loadSelectedBenchmark();
    updateZoomBadge();
    setExplainMode(false);
    updateCliCommand();
    initWorkspaceAndEngine();
    updateDatasetStatusBadge();
    if (typeof updateMultiDatasetUI === 'function') {
      updateMultiDatasetUI();
    }

    // =========================================================================
    //  HELP & DOCUMENTATION CENTER (MULTI-TOPIC MODAL & PRESETS)
    // =========================================================================

    const PRESETS = {
      'preset-basic-spiral': {
        name: '2D Spiral Baseline',
        benchmark: '2Dspiral',
        rlim: 0.100,
        targetMode: 'greedy',
        pruneMode: '3P',
        useTM: false,
        usePred: false,
        useTiles: false,
        speed: 50,
        loopCount: 1,
        isExplainMode: false,
        useWasm: true,
        autoPlay: true,
      },
      'preset-entropy-target': {
        name: 'Shannon Entropy Scheduling',
        benchmark: '2Dspiral',
        rlim: 0.100,
        targetMode: 'entropy',
        entropyGate: 0.75,
        entropyFirstGate: 1.50,
        pruneMode: '3P',
        useTM: false,
        usePred: false,
        useTiles: false,
        speed: 50,
        loopCount: 1,
        isExplainMode: false,
        useWasm: true,
        autoPlay: true,
      },
      'preset-pruning-5p': {
        name: '5P Metric Geometric Pruning (3D Torus)',
        benchmark: '3Dtorus',
        rlim: 0.100,
        targetMode: 'greedy',
        pruneMode: '5P',
        useTM: false,
        usePred: false,
        useTiles: false,
        speed: 50,
        loopCount: 1,
        isExplainMode: false,
        useWasm: true,
        autoPlay: true,
      },
      'preset-markov-pred': {
        name: 'Markov Transitions & Sequence Prediction',
        benchmark: '2DcircleP10n',
        rlim: 0.120,
        targetMode: 'greedy',
        pruneMode: '3P',
        useTM: true,
        tmMixingCoeff: 0.70,
        usePred: true,
        predHorizon: 2,
        useTiles: false,
        speed: 50,
        loopCount: 5,
        isExplainMode: false,
        useWasm: true,
        autoPlay: true,
      },
      'preset-explain-walkthrough': {
        name: 'Explain Mode Step-by-Step Walkthrough',
        benchmark: '2Dspiral',
        rlim: 0.100,
        targetMode: 'entropy',
        entropyGate: 0.75,
        pruneMode: '3P',
        useTM: false,
        usePred: false,
        useTiles: false,
        speed: 150,
        loopCount: 1,
        isExplainMode: true,
        useWasm: true,
        autoPlay: false,
      },
      'preset-max-throughput': {
        name: 'Maximum Performance C/WASM Engine',
        benchmark: '3Dspiral',
        rlim: 0.080,
        targetMode: 'entropy',
        pruneMode: '4P',
        useTM: false,
        usePred: true,
        predHorizon: 2,
        useTiles: false,
        speed: 0,
        loopCount: 1,
        isExplainMode: false,
        useWasm: true,
        autoPlay: true,
      }
    };

    function applySimulatorPreset(presetKey) {
      const preset = PRESETS[presetKey];
      if (!preset) return;

      if (isRunning) pauseSimulation();

      // Benchmark
      if (preset.benchmark) {
        currentBenchmark = preset.benchmark;
        const selBench = document.getElementById('selectBenchmark');
        if (selBench) selBench.value = preset.benchmark;
        const selBenchSide = document.getElementById('selectBenchmarkSide');
        if (selBenchSide) selBenchSide.value = preset.benchmark;
        const descEl = document.getElementById('benchmarkDesc');
        if (descEl && BENCHMARK_DESCS[preset.benchmark]) {
          descEl.innerHTML = BENCHMARK_DESCS[preset.benchmark];
        }
      }

      // Radius
      if (preset.rlim !== undefined) {
        rlim = preset.rlim;
        const slRlim = document.getElementById('sliderRlim');
        if (slRlim) slRlim.value = rlim;
        const inpRlim = document.getElementById('inputRlim');
        if (inpRlim) inpRlim.value = rlim.toFixed(3);
      }

      // Target mode
      if (preset.targetMode) {
        targetMode = preset.targetMode;
        const btnGreedy = document.getElementById('modeGreedy');
        const btnEntropy = document.getElementById('modeEntropy');
        if (btnGreedy) btnGreedy.classList.toggle('active', targetMode === 'greedy');
        if (btnEntropy) btnEntropy.classList.toggle('active', targetMode === 'entropy');
      }

      // Pruning mode
      if (preset.pruneMode) {
        pruneMode = preset.pruneMode;
        ['3P', '4P', '5P'].forEach(p => {
          const btn = document.getElementById(`prune${p}`);
          if (btn) btn.classList.toggle('active', p === pruneMode);
        });
      }

      // TM mixing
      if (preset.useTM !== undefined) {
        useTM = preset.useTM;
        const optTM = document.getElementById('optTM');
        if (optTM) optTM.classList.toggle('active', useTM);
        if (preset.tmMixingCoeff !== undefined) {
          tmMixingCoeff = preset.tmMixingCoeff;
          const sl = document.getElementById('sliderTmMix');
          if (sl) sl.value = tmMixingCoeff;
          const inp = document.getElementById('inputTmMix');
          if (inp) inp.value = tmMixingCoeff.toFixed(2);
        }
      }

      // Sequence prediction
      if (preset.usePred !== undefined) {
        usePred = preset.usePred;
        const optPred = document.getElementById('optPred');
        if (optPred) optPred.classList.toggle('active', usePred);
        if (preset.predHorizon !== undefined) {
          predHorizon = preset.predHorizon;
          const sl = document.getElementById('sliderPredHorizon');
          if (sl) sl.value = predHorizon;
          const inp = document.getElementById('inputPredHorizon');
          if (inp) inp.value = predHorizon;
        }
      }

      // Tiling
      if (preset.useTiles !== undefined) {
        useTiles = preset.useTiles;
        const optTiles = document.getElementById('optTiles');
        if (optTiles) optTiles.classList.toggle('active', useTiles);
      }

      // Entropy gates
      if (preset.entropyGate !== undefined) {
        entropyGate = preset.entropyGate;
        const sl = document.getElementById('sliderEntropyGate');
        if (sl) sl.value = entropyGate;
        const inp = document.getElementById('inputEntropyGate');
        if (inp) inp.value = entropyGate.toFixed(2);
      }
      if (preset.entropyFirstGate !== undefined) {
        entropyFirstGate = preset.entropyFirstGate;
        const sl = document.getElementById('sliderEntropyFirstGate');
        if (sl) sl.value = entropyFirstGate;
        const inp = document.getElementById('inputEntropyFirstGate');
        if (inp) inp.value = entropyFirstGate.toFixed(2);
      }

      // Speed
      if (preset.speed !== undefined) {
        playSpeed = preset.speed;
        const selSpeed = document.getElementById('selectSpeed');
        if (selSpeed) selSpeed.value = preset.speed.toString();
        const selSpeedSide = document.getElementById('selectSpeedSide');
        if (selSpeedSide) selSpeedSide.value = preset.speed.toString();
      }

      // Loops
      if (preset.loopCount !== undefined) {
        loopCount = preset.loopCount;
        const selLoop = document.getElementById('selectLoop');
        if (selLoop) selLoop.value = preset.loopCount.toString();
        const selLoopSide = document.getElementById('selectLoopSide');
        if (selLoopSide) selLoopSide.value = preset.loopCount.toString();
      }

      // Explain mode
      if (preset.isExplainMode !== undefined) {
        setExplainMode(preset.isExplainMode);
      }

      // WASM preference
      if (preset.useWasm !== undefined && GricWasm.isLoaded()) {
        useWasm = preset.useWasm;
      }

      syncControlDependencies();
      updateWasmBadge();
      updateCliCommand();

      // Reset and reload dataset
      loadSelectedBenchmark();
      resetView();
      updateUI();
      draw();

      // Close modal
      const modal = document.getElementById('helpModal');
      if (modal) modal.style.display = 'none';

      showToast(`⚡ Loaded: ${preset.name}`);

      if (preset.autoPlay) {
        setTimeout(() => {
          startSimulation();
        }, 250);
      }
    }

    // Help modal lifecycle & topic switching
    const helpModal = document.getElementById('helpModal');
    const toggleHelp = () => {
      if (!helpModal) return;
      helpModal.style.display =
        helpModal.style.display === 'none' ? 'flex' : 'none';
    };

    const btnHelp = document.getElementById('btnHelp');
    if (btnHelp) btnHelp.addEventListener('click', toggleHelp);

    const btnHelpClose = document.getElementById('btnHelpClose');
    if (btnHelpClose) {
      btnHelpClose.addEventListener('click', () => {
        if (helpModal) helpModal.style.display = 'none';
      });
    }

    if (helpModal) {
      helpModal.addEventListener('click', (e) => {
        if (e.target === helpModal) {
          helpModal.style.display = 'none';
        }
      });
    }

    // Help Topic Sidebar Switching
    const topicButtons = document.querySelectorAll('.help-topic-btn');
    topicButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const topic = btn.getAttribute('data-topic');
        topicButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        document.querySelectorAll('.help-topic-content').forEach(pane => {
          pane.classList.remove('active');
        });

        const targetPane = document.getElementById(`topic-${topic}`);
        if (targetPane) {
          targetPane.classList.add('active');
        }
      });
    });

    // Preset Run Buttons inside Help Modal
    document.querySelectorAll('.help-preset-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const key = btn.getAttribute('data-preset');
        if (key) {
          applySimulatorPreset(key);
        }
      });
    });

    const fileModalEl = document.getElementById('modalFileViewer');
    if (fileModalEl) {
      fileModalEl.addEventListener('click', (e) => {
        if (e.target === fileModalEl) {
          fileModalEl.style.display = 'none';
        }
      });
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === '?' && !e.target.matches('input, textarea, select')) {
        e.preventDefault();
        toggleHelp();
      }
      if (e.key === 'Escape') {
        if (helpModal && helpModal.style.display !== 'none') {
          helpModal.style.display = 'none';
        }
        if (fileModalEl && fileModalEl.style.display !== 'none') {
          fileModalEl.style.display = 'none';
        }
        const cmdModal = document.getElementById('modalCommandPalette');
        if (cmdModal && cmdModal.style.display !== 'none') {
          cmdModal.style.display = 'none';
        }
      }
    });

    // -------------------------------------------------------------------------
    // Sidebar Master Mode Switcher
    // -------------------------------------------------------------------------
    function switchSidebarMode(mode) {
      activeSidebarMode = mode;
      document.querySelectorAll('#sidebarModeNav .sidebar-mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
      });

      const cards = {
        cardInputData: ['all', 'clustering'],
        cardSettings: ['all', 'clustering'],
        cardDisplay: ['all', 'clustering'],
        cardDataFiles: ['all', 'files'],
        cardCli: ['all', 'files'],
        cardResources: ['all', 'clustering', 'telemetry'],
        cardTrace: ['all', 'clustering', 'telemetry'],
        cardKnnSettings: ['all', 'knn'],
        cardKnnResources: ['all', 'knn', 'telemetry'],
        cardKnnTrace: ['all', 'knn', 'telemetry'],
        cardDimDensity: ['all', 'knn', 'telemetry'],
        cardReconstruction: ['all', 'knn', 'clustering', 'telemetry', 'files']
      };

      Object.entries(cards).forEach(([cardId, allowedModes]) => {
        const el = document.getElementById(cardId);
        if (!el) return;
        const isKnnCard = cardId.startsWith('cardKnn');
        const allowed = allowedModes.includes(mode);

        if (isKnnCard && !enableKnn) {
          el.style.display = 'none';
        } else {
          el.style.display = allowed ? '' : 'none';
        }
      });
    }
    window.switchSidebarMode = switchSidebarMode;

    // -------------------------------------------------------------------------
    // Interactive Timeline Scrubber & Milestones
    // -------------------------------------------------------------------------
    function addClusterMilestone(frameIdx) {
      if (!clusterMilestoneFrames.includes(frameIdx)) {
        clusterMilestoneFrames.push(frameIdx);
      }
      renderTimelineMilestones();
    }
    window.addClusterMilestone = addClusterMilestone;

    function renderTimelineMilestones() {
      const container = document.getElementById('timelineMilestones');
      if (!container) return;
      const total = (benchmarkDataset && benchmarkDataset.length > 0) ? benchmarkDataset.length : 1;
      container.innerHTML = '';
      clusterMilestoneFrames.forEach(f => {
        const tick = document.createElement('div');
        tick.className = 'timeline-milestone-tick';
        tick.style.left = `${(f / total) * 100}%`;
        container.appendChild(tick);
      });
    }
    window.renderTimelineMilestones = renderTimelineMilestones;

    function initTimelineScrubber() {
      const scrubber = document.getElementById('timelineScrubber');
      const tooltip = document.getElementById('timelineScrubTooltip');
      if (!scrubber) return;

      let isScrubbing = false;

      function scrubToMouse(e) {
        if (!benchmarkDataset || benchmarkDataset.length === 0) return;
        const rect = scrubber.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const total = benchmarkDataset.length;
        const targetFrame = Math.floor(ratio * (total - 1));

        if (typeof dataMode !== 'undefined' && dataMode === 'image') {
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(targetFrame);
          }
        } else {
          selectPastSample(targetFrame);
        }

        if (tooltip) {
          tooltip.style.left = `${ratio * 100}%`;
          tooltip.textContent = `Frame ${targetFrame + 1} / ${total}`;
          tooltip.style.display = 'block';
        }
      }

      scrubber.addEventListener('mousedown', (e) => {
        isScrubbing = true;
        scrubToMouse(e);
      });

      window.addEventListener('mousemove', (e) => {
        if (isScrubbing) {
          scrubToMouse(e);
        }
      });

      window.addEventListener('mouseup', () => {
        if (isScrubbing) {
          isScrubbing = false;
          if (tooltip) tooltip.style.display = 'none';
        }
      });

      scrubber.addEventListener('mousemove', (e) => {
        if (!benchmarkDataset || benchmarkDataset.length === 0) return;
        const rect = scrubber.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const total = benchmarkDataset.length;
        const targetFrame = Math.floor(ratio * (total - 1));

        if (tooltip && !isScrubbing) {
          tooltip.style.left = `${ratio * 100}%`;
          tooltip.textContent = `Frame ${targetFrame + 1} / ${total}`;
          tooltip.style.display = 'block';
        }
      });

      scrubber.addEventListener('mouseleave', () => {
        if (!isScrubbing && tooltip) {
          tooltip.style.display = 'none';
        }
      });
    }

    function initImageScrubber() {
      const slider = document.getElementById('sliderImgFrame');
      const inputFrame = document.getElementById('inputImgFrame');
      const btnPrev = document.getElementById('btnImgPrevFrame');
      const btnNext = document.getElementById('btnImgNextFrame');
      const btnLive = document.getElementById('btnImgLiveStream');

      window.isDraggingImageSlider = false;

      if (slider) {
        const onSliderUpdate = (e) => {
          const val = parseInt(e.target.value, 10);
          if (!isNaN(val) && typeof selectImageFrame === 'function') {
            selectImageFrame(val);
          }
        };

        slider.addEventListener('pointerdown', () => { window.isDraggingImageSlider = true; });
        slider.addEventListener('mousedown', () => { window.isDraggingImageSlider = true; });
        window.addEventListener('pointerup', () => { window.isDraggingImageSlider = false; });
        window.addEventListener('mouseup', () => { window.isDraggingImageSlider = false; });

        slider.addEventListener('input', onSliderUpdate);
        slider.addEventListener('change', onSliderUpdate);
      }

      if (inputFrame) {
        const commitFrameInput = () => {
          const total = (benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset.length
            : (typeof totalFrames !== 'undefined' ? totalFrames : 0);
          if (total === 0) return;
          let val = parseInt(inputFrame.value, 10);
          if (isNaN(val)) {
            const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
              inspectedImageFrameIdx >= 0)
              ? inspectedImageFrameIdx
              : Math.max(0, (typeof totalFrames !== 'undefined' ? totalFrames : 1) - 1);
            inputFrame.value = cur + 1;
            return;
          }
          val = Math.max(1, Math.min(total, val));
          inputFrame.value = val;
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(val - 1);
          }
        };

        inputFrame.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commitFrameInput();
            inputFrame.blur();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
              inspectedImageFrameIdx >= 0)
              ? inspectedImageFrameIdx
              : Math.max(0, (typeof totalFrames !== 'undefined' ? totalFrames : 1) - 1);
            inputFrame.value = cur + 1;
            inputFrame.blur();
          }
        });
        inputFrame.addEventListener('change', commitFrameInput);
      }

      if (btnPrev) {
        btnPrev.addEventListener('click', () => {
          const total = (benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset.length
            : totalFrames;
          if (total === 0) return;
          const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
            inspectedImageFrameIdx >= 0)
            ? inspectedImageFrameIdx
            : totalFrames - 1;
          const target = Math.max(0, cur - 1);
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(target);
          }
        });
      }

      if (btnNext) {
        btnNext.addEventListener('click', () => {
          const total = (benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset.length
            : totalFrames;
          if (total === 0) return;
          const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
            inspectedImageFrameIdx >= 0)
            ? inspectedImageFrameIdx
            : totalFrames - 1;
          const target = Math.min(total - 1, cur + 1);
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(target);
          }
        });
      }

      if (btnLive) {
        btnLive.addEventListener('click', () => {
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(-1);
          }
        });
      }

      const selectSort = document.getElementById('selectImgClusterSort');
      if (selectSort) {
        selectSort.addEventListener('change', (e) => {
          imageClustersSortMode = e.target.value;
          if (typeof draw === 'function') draw();
          const label = (imageClustersSortMode === 'size_desc')
            ? '📊 Sorted by Cluster Size (Descending: Largest first)'
            : (imageClustersSortMode === 'size_asc')
              ? '📉 Sorted by Cluster Size (Ascending: Smallest first)'
              : '🔢 Sorted by Creation ID (Default)';
          if (typeof showToast === 'function') showToast(label);
        });
      }

      const btnShowAll = document.getElementById('btnImgShowAllPanels');
      if (btnShowAll) {
        btnShowAll.addEventListener('click', () => {
          maximizedQuad = null;
          syncImageQuadUI();
          if (typeof draw === 'function') draw();
          if (typeof showToast === 'function') showToast('⊞ Restored All 4 View Panels');
        });
      }

      const selectView = document.getElementById('selectImgViewPanel');
      if (selectView) {
        selectView.addEventListener('change', (e) => {
          const val = e.target.value;
          if (val === 'all') {
            maximizedQuad = null;
            if (typeof showToast === 'function') showToast('⊞ All 4 Panels (Split Grid)');
          } else if (val === '2_knn') {
            maximizedQuad = 2;
            imageQ2ViewMode = 'knn';
            if (typeof showToast === 'function') showToast('⚡ Maximized Q2: k-NN Neighbors');
          } else {
            maximizedQuad = parseInt(val, 10);
            if (maximizedQuad === 2) imageQ2ViewMode = 'members';
            const qNames = ['Current Frame', 'Anchor / Residual', 'Cluster Members', 'All Clusters'];
            if (typeof showToast === 'function') {
              showToast(`🔍 Maximized Q${maximizedQuad}: ${qNames[maximizedQuad]}`);
            }
          }
          syncImageQuadUI();
          if (typeof draw === 'function') draw();
        });
      }

      function setImageThumbSize(size) {
        const clamped = Math.max(36, Math.min(220, Math.round(size)));
        imageThumbSize = clamped;
        const lblSize = document.getElementById('lblImgThumbSize');
        if (lblSize) lblSize.textContent = `${clamped}px`;
        if (typeof draw === 'function') draw();
      }

      const btnThumbSmaller = document.getElementById('btnImgThumbSmaller');
      if (btnThumbSmaller) {
        btnThumbSmaller.addEventListener('click', () => {
          const cur = (typeof imageThumbSize !== 'undefined') ? imageThumbSize : 64;
          setImageThumbSize(cur - 16);
          if (typeof showToast === 'function') showToast(`🔍 Thumbnail Size: ${imageThumbSize}px`);
        });
      }

      const btnThumbBigger = document.getElementById('btnImgThumbBigger');
      if (btnThumbBigger) {
        btnThumbBigger.addEventListener('click', () => {
          const cur = (typeof imageThumbSize !== 'undefined') ? imageThumbSize : 64;
          setImageThumbSize(cur + 16);
          if (typeof showToast === 'function') showToast(`🔍 Thumbnail Size: ${imageThumbSize}px`);
        });
      }
    }

    function syncImageQuadUI() {
      const selectView = document.getElementById('selectImgViewPanel');
      const btnShowAll = document.getElementById('btnImgShowAllPanels');
      if (selectView) {
        if (maximizedQuad === null) {
          selectView.value = 'all';
        } else if (maximizedQuad === 2 && imageQ2ViewMode === 'knn') {
          selectView.value = '2_knn';
        } else {
          selectView.value = String(maximizedQuad);
        }
      }
      if (btnShowAll) {
        if (maximizedQuad === null) {
          btnShowAll.style.background = 'rgba(56, 189, 248, 0.2)';
          btnShowAll.style.color = '#38bdf8';
          btnShowAll.style.borderColor = 'rgba(56, 189, 248, 0.4)';
          btnShowAll.textContent = '⊞ All 4 Panels';
        } else {
          btnShowAll.style.background = 'rgba(34, 197, 94, 0.25)';
          btnShowAll.style.color = '#4ade80';
          btnShowAll.style.borderColor = 'rgba(34, 197, 94, 0.6)';
          btnShowAll.textContent = '⊞ Show All Panels';
        }
      }
    }
    window.syncImageQuadUI = syncImageQuadUI;

    // -------------------------------------------------------------------------
    // Quick Command Palette (Ctrl+K / Cmd+K)
    // -------------------------------------------------------------------------
    const commandPaletteCommands = [
      // Actions
      { id: 'act-play', group: 'Actions', icon: '▶', name: 'Cluster / Play Simulation',
        hint: 'Space', action: () => document.getElementById('btnPlay')?.click() },
      { id: 'act-compute-all', group: 'Actions', icon: '⚡', name: 'Compute All / Run to Completion (or Stop)',
        hint: 'Batch', action: () => document.getElementById('btnComputeAll')?.click() },
      { id: 'act-step', group: 'Actions', icon: '⏭', name: 'Step Single Frame',
        hint: 'S', action: () => document.getElementById('btnStep')?.click() },
      { id: 'act-reset-clusters', group: 'Actions', icon: '↺',
        name: 'Reset Clusters (Keep Dataset)', hint: 'Reset Model',
        action: () => document.getElementById('btnResetClusters')?.click() },
      { id: 'act-reset-all', group: 'Actions', icon: '⟲',
        name: 'Reset All (Clear Dataset & Clusters)', hint: 'R',
        action: () => document.getElementById('btnReset')?.click() },
      { id: 'act-pass2', group: 'Actions', icon: '🔄',
        name: '2nd Pass Nearest Anchor Reallocation', hint: '2 / P',
        action: () => document.getElementById('btnPass2Nearest')?.click() },
      { id: 'act-explain', group: 'Actions', icon: '💬', name: 'Toggle Decision Explain Mode',
        hint: 'E', action: () => document.getElementById('btnExplain')?.click() },
      { id: 'act-knn-toggle', group: 'Actions', icon: '⚡', name: 'Toggle k-NN Module',
        hint: 'K', action: () => document.getElementById('btnToggleKnnModule')?.click() },
      { id: 'act-knn-run', group: 'Actions', icon: '▶', name: 'Compute k-Nearest Neighbors',
        hint: 'k-NN', action: () => document.getElementById('btnRunKnn')?.click() },
      { id: 'act-multi-dataset-toggle', group: 'Actions', icon: '🗂️', name: 'Toggle Multi-Dataset Mode (A, B, C)',
        hint: 'Multi-DS', action: () => {
          if (typeof setMultiDatasetEnabled === 'function') {
            setMultiDatasetEnabled(!multiDatasetEnabled);
          }
        } },
      { id: 'act-img-show-all', group: 'Actions', icon: '⊞', name: 'Show All View Panels (4-Split Grid)',
        hint: 'Esc', action: () => {
          maximizedQuad = null;
          syncImageQuadUI();
          if (typeof draw === 'function') draw();
          if (typeof showToast === 'function') showToast('⊞ Restored All 4 View Panels');
        } },
      { id: 'act-img-sort-desc', group: 'Actions', icon: '📊', name: 'Image Mode: Sort Clusters by Size (Descending)',
        hint: 'Sort Desc', action: () => {
          imageClustersSortMode = 'size_desc';
          const sel = document.getElementById('selectImgClusterSort');
          if (sel) sel.value = 'size_desc';
          if (typeof draw === 'function') draw();
          if (typeof showToast === 'function') showToast('📊 Sorted by Cluster Size (Descending)');
        } },
      { id: 'act-img-sort-asc', group: 'Actions', icon: '📉', name: 'Image Mode: Sort Clusters by Size (Ascending)',
        hint: 'Sort Asc', action: () => {
          imageClustersSortMode = 'size_asc';
          const sel = document.getElementById('selectImgClusterSort');
          if (sel) sel.value = 'size_asc';
          if (typeof draw === 'function') draw();
          if (typeof showToast === 'function') showToast('📉 Sorted by Cluster Size (Ascending)');
        } },
      { id: 'act-img-sort-id', group: 'Actions', icon: '🔢', name: 'Image Mode: Sort Clusters by ID (Default)',
        hint: 'Sort ID', action: () => {
          imageClustersSortMode = 'id';
          const sel = document.getElementById('selectImgClusterSort');
          if (sel) sel.value = 'id';
          if (typeof draw === 'function') draw();
          if (typeof showToast === 'function') showToast('🔢 Sorted by Creation ID (Default)');
        } },
      { id: 'act-motion-tail', group: 'Actions', icon: '〰️',
        name: 'Toggle Recent Points Motion Trail', hint: 'Trail',
        action: () => document.getElementById('btnToggleMotionTail')?.click() },
      { id: 'act-color-per-cluster', group: 'Actions', icon: '🎨',
        name: 'Toggle Per-Cluster Point Colors', hint: 'Colors',
        action: () => document.getElementById('btnToggleColorPerCluster')?.click() },
      { id: 'act-export', group: 'Actions', icon: '💾', name: 'Export Run to Local Workspace',
        hint: '.clusterdat', action: () => document.getElementById('btnSaveToWorkspace')?.click() },

      // Camera Views
      { id: 'cam-11', group: 'Camera', icon: '🔍', name: '1:1 Reset Pan & Zoom',
        hint: 'Z', action: () => document.getElementById('btnResetView')?.click() },
      { id: 'cam-iso', group: 'Camera', icon: '📐', name: 'Isometric 3D Perspective',
        hint: '3D Orbit', action: () => document.getElementById('presetIso')?.click() },
      { id: 'cam-front', group: 'Camera', icon: '🔲', name: 'Front View (XZ Plane)',
        hint: 'Front', action: () => document.getElementById('presetFront')?.click() },
      { id: 'cam-top', group: 'Camera', icon: '🔝', name: 'Top View (XY Plane)',
        hint: 'Top', action: () => document.getElementById('presetTop')?.click() },
      { id: 'cam-side', group: 'Camera', icon: '🔳', name: 'Side View (YZ Plane)',
        hint: 'Side', action: () => document.getElementById('presetSide')?.click() },
      { id: 'cam-reset3d', group: 'Camera', icon: '↺', name: 'Reset 3D Orbit Camera',
        hint: 'Reset 3D', action: () => document.getElementById('presetReset3D')?.click() },
      { id: 'cam-lock-center', group: 'Camera', icon: '🎯', name: 'Lock / Unlock 3D Center of Rotation',
        hint: 'C', action: () => toggleLockCenter3D() },

      // Datasets
      { id: 'ds-slot-a', group: 'Datasets', icon: '🅰️',
        name: 'Switch to Dataset A (workspace/A)', hint: 'Slot A',
        action: () => switchDatasetSlot('A') },
      { id: 'ds-slot-b', group: 'Datasets', icon: '🅱️',
        name: 'Switch to Dataset B (workspace/B)', hint: 'Slot B',
        action: () => switchDatasetSlot('B') },
      { id: 'ds-slot-c', group: 'Datasets', icon: '🅲',
        name: 'Switch to Dataset C (workspace/C)', hint: 'Slot C',
        action: () => switchDatasetSlot('C') },
      { id: 'ds-3dtorus', group: 'Datasets', icon: '🍩',
        name: 'Load 3Dtorus (3D Torus Manifold Knot)', hint: '3D Dataset',
        action: () => loadDatasetByKey('3Dtorus') },
      { id: 'ds-2dspiral', group: 'Datasets', icon: '🌀',
        name: 'Load 2Dspiral (Archimedean Spiral)', hint: '2D Dataset',
        action: () => loadDatasetByKey('2Dspiral') },
      { id: 'ds-3dsphere', group: 'Datasets', icon: '🌐',
        name: 'Load 3Dsphere (Spherical Shell S²)', hint: '3D Dataset',
        action: () => loadDatasetByKey('3Dsphere') },
      { id: 'ds-3dstar', group: 'Datasets', icon: '✨',
        name: 'Load 3Dstar (Multi-Spoke Radial Star)', hint: '3D Dataset',
        action: () => loadDatasetByKey('3Dstar') },
      { id: 'ds-3dlorenz', group: 'Datasets', icon: '🦋',
        name: 'Load 3Dlorenz (Lorenz Strange Attractor)', hint: '3D Dataset',
        action: () => loadDatasetByKey('3Dlorenz') },
      { id: 'ds-2dcircle', group: 'Datasets', icon: '⭕',
        name: 'Load 2Dcircle-shuffle (Circle Shuffled)', hint: '2D Dataset',
        action: () => loadDatasetByKey('2Dcircle-shuffle') },
      { id: 'ds-img-ball1', group: 'Datasets', icon: '⚽',
        name: 'Load Single Bouncing Ball (32×32 Image)', hint: 'Image Dataset',
        action: () => loadDatasetByKey('img-ball-1') },

      // Panels & Modes
      { id: 'nav-clustering', group: 'Navigation', icon: '📊', name: 'Switch to Clustering Mode',
        hint: 'Sidebar', action: () => switchSidebarMode('clustering') },
      { id: 'nav-knn', group: 'Navigation', icon: '⚡', name: 'Switch to k-NN Mode',
        hint: 'Sidebar', action: () => {
          if (!enableKnn) document.getElementById('btnToggleKnnModule')?.click();
          switchSidebarMode('knn');
        } },
      { id: 'nav-files', group: 'Navigation', icon: '📂', name: 'Switch to Files & CLI Mode',
        hint: 'Sidebar', action: () => switchSidebarMode('files') },
      { id: 'nav-telemetry', group: 'Navigation', icon: '📈', name: 'Switch to Telemetry Mode',
        hint: 'Sidebar', action: () => switchSidebarMode('telemetry') },
      { id: 'nav-all', group: 'Navigation', icon: '📑', name: 'Show All Panels',
        hint: 'Sidebar', action: () => switchSidebarMode('all') }
    ];

    function loadDatasetByKey(key) {
      const sel = document.getElementById('selectBenchmark');
      if (sel) {
        sel.value = key;
        sel.dispatchEvent(new Event('change'));
      }
    }

    let cmdSelectedIdx = 0;
    let cmdFilteredList = [];

    function openCommandPalette() {
      const modal = document.getElementById('modalCommandPalette');
      const input = document.getElementById('commandPaletteInput');
      if (!modal || !input) return;
      modal.style.display = 'flex';
      input.value = '';
      renderCommandPaletteResults('');
      setTimeout(() => input.focus(), 20);
    }

    function closeCommandPalette() {
      const modal = document.getElementById('modalCommandPalette');
      if (modal) modal.style.display = 'none';
    }

    function toggleCommandPalette() {
      const modal = document.getElementById('modalCommandPalette');
      if (modal && modal.style.display !== 'none') {
        closeCommandPalette();
      } else {
        openCommandPalette();
      }
    }
    window.openCommandPalette = openCommandPalette;
    window.closeCommandPalette = closeCommandPalette;
    window.toggleCommandPalette = toggleCommandPalette;

    function renderCommandPaletteResults(query) {
      const resultsEl = document.getElementById('commandPaletteResults');
      if (!resultsEl) return;
      const q = query.trim().toLowerCase();

      cmdFilteredList = commandPaletteCommands.filter(c => {
        if (!q) return true;
        return c.name.toLowerCase().includes(q) ||
               c.group.toLowerCase().includes(q) ||
               (c.hint && c.hint.toLowerCase().includes(q));
      });

      cmdSelectedIdx = Math.max(0, Math.min(cmdSelectedIdx, cmdFilteredList.length - 1));

      if (cmdFilteredList.length === 0) {
        resultsEl.innerHTML = `
          <div style="padding: 24px; text-align: center; color: var(--text-muted);
                      font-size: 0.8rem;">
            No matching commands found
          </div>`;
        return;
      }

      const groups = {};
      cmdFilteredList.forEach((cmd, idx) => {
        if (!groups[cmd.group]) groups[cmd.group] = [];
        groups[cmd.group].push({ cmd, flatIdx: idx });
      });

      let html = '';
      Object.entries(groups).forEach(([groupName, items]) => {
        html += `<div class="command-palette-group-title">${groupName}</div>`;
        items.forEach(({ cmd, flatIdx }) => {
          const isActive = (flatIdx === cmdSelectedIdx);
          const badgeHtml = cmd.hint
            ? `<span class="command-palette-item-badge">${cmd.hint}</span>`
            : '';
          html += `
            <div class="command-palette-item ${isActive ? 'active' : ''}"
                 data-idx="${flatIdx}" onclick="executeCommand(${flatIdx})">
              <div class="command-palette-item-left">
                <span class="command-palette-item-icon">${cmd.icon}</span>
                <span class="command-palette-item-text">${cmd.name}</span>
              </div>
              ${badgeHtml}
            </div>
          `;
        });
      });

      resultsEl.innerHTML = html;

      const activeEl = resultsEl.querySelector('.command-palette-item.active');
      if (activeEl) {
        activeEl.scrollIntoView({ block: 'nearest' });
      }
    }

    function executeCommand(idx) {
      if (idx >= 0 && idx < cmdFilteredList.length) {
        const cmd = cmdFilteredList[idx];
        closeCommandPalette();
        if (cmd && typeof cmd.action === 'function') {
          cmd.action();
        }
      }
    }
    window.executeCommand = executeCommand;

    function initCommandPalette() {
      const btn = document.getElementById('btnOpenCommandPalette');
      if (btn) btn.addEventListener('click', openCommandPalette);

      const modal = document.getElementById('modalCommandPalette');
      if (modal) {
        modal.addEventListener('click', (e) => {
          if (e.target === modal) closeCommandPalette();
        });
      }

      const input = document.getElementById('commandPaletteInput');
      if (input) {
        input.addEventListener('input', (e) => {
          cmdSelectedIdx = 0;
          renderCommandPaletteResults(e.target.value);
        });

        input.addEventListener('keydown', (e) => {
          if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (cmdFilteredList.length > 0) {
              cmdSelectedIdx = (cmdSelectedIdx + 1) % cmdFilteredList.length;
              renderCommandPaletteResults(input.value);
            }
          } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (cmdFilteredList.length > 0) {
              const len = cmdFilteredList.length;
              cmdSelectedIdx = (cmdSelectedIdx - 1 + len) % len;
              renderCommandPaletteResults(input.value);
            }
          } else if (e.key === 'Enter') {
            e.preventDefault();
            executeCommand(cmdSelectedIdx);
          } else if (e.key === 'Escape') {
            e.preventDefault();
            closeCommandPalette();
          }
        });
      }
    }

    initTimelineScrubber();
    initImageScrubber();
    initCommandPalette();

    setTimeout(() => { updateTMCanvasDimensions(); resizeCanvas(); }, 50);
    setTimeout(() => { updateTMCanvasDimensions(); resizeCanvas(); }, 250);
