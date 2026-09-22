/**
 * GRIC Simulator - main.js
 * Part of the GRIC Interactive Algorithm Simulator
 */

//  WASM ENGINE TOGGLE
    // =========================================================================

    function updateClusteringButtonUI() {
      const btnPlay = document.getElementById('btnPlay');
      const btnStop = document.getElementById('btnStop');
      if (!btnPlay) return;

      const isCli = (typeof engineMode !== 'undefined' && engineMode === 'cli');
      const isCliActive = (typeof isCliRunning !== 'undefined' && isCliRunning);
      const isBatchActive = (typeof isComputeAllRunning !== 'undefined' && isComputeAllRunning);
      const isSimActive = (typeof isRunning !== 'undefined' && isRunning);
      const isWorkerActive = (typeof GricWasmWorker !== 'undefined' &&
                              typeof GricWasmWorker.isBusy === 'function' &&
                              GricWasmWorker.isBusy());
      const isClusteringActive = isCliActive || isBatchActive || isSimActive || isWorkerActive;

      if (btnStop) {
        btnStop.disabled = !isClusteringActive;
        if (isClusteringActive) {
          btnStop.title = 'Stop clustering (Esc)';
        } else {
          btnStop.title = 'Clustering is not running';
        }
      }

      if (isCli) {
        if (isCliActive) {
          btnPlay.innerHTML = '⏳ Stop gric-cluster';
          btnPlay.title = 'Native gric-cluster is running. Click to terminate.';
          btnPlay.disabled = false;
          btnPlay.classList.add('danger');
          btnPlay.classList.remove('primary', 'btn-clustered');
          btnPlay.style.background = 'rgba(239, 68, 68, 0.25)';
          btnPlay.style.color = '#f87171';
          btnPlay.style.borderColor = 'rgba(239, 68, 68, 0.6)';
        } else {
          btnPlay.innerHTML = '▶ Run gric-cluster';
          btnPlay.title = 'Run native compiled gric-cluster executable';
          btnPlay.disabled = false;
          btnPlay.classList.remove('danger', 'primary', 'btn-clustered');
          btnPlay.classList.add('btn-action');
          btnPlay.style.background = 'rgba(74, 222, 128, 0.25)';
          btnPlay.style.color = '#4ade80';
          btnPlay.style.borderColor = 'rgba(74, 222, 128, 0.5)';
        }
        return;
      }

      if (typeof isComputeAllRunning !== 'undefined' && isComputeAllRunning) {
        btnPlay.innerHTML = '⏳ Computing... (Stop)';
        btnPlay.title = 'Batch clustering in progress. Click to stop.';
        btnPlay.classList.add('danger');
        btnPlay.classList.remove('primary', 'btn-clustered');
        btnPlay.style.background = 'rgba(239, 68, 68, 0.25)';
        btnPlay.style.color = '#f87171';
        btnPlay.style.borderColor = 'rgba(239, 68, 68, 0.6)';
        return;
      }

      if (isRunning) {
        btnPlay.innerHTML = '❚❚ Pause';
        btnPlay.title = 'Clustering in progress. Click to pause.';
        btnPlay.classList.add('danger');
        btnPlay.classList.remove('primary', 'btn-clustered');
        btnPlay.style.background = '';
        btnPlay.style.color = '';
        btnPlay.style.borderColor = '';
        return;
      }

      // Check cluster count in active dataset
      let nClust = 0;
      if (typeof useTiles !== 'undefined' && useTiles) {
        nClust = (typeof jointTuplesMap !== 'undefined' && jointTuplesMap)
          ? jointTuplesMap.size
          : (typeof tileEngineX !== 'undefined' && tileEngineX.clusters
              ? tileEngineX.clusters.length : 0);
      } else if (typeof clusters !== 'undefined' && clusters && clusters.length > 0) {
        nClust = clusters.length;
      } else if (typeof GricWasm !== 'undefined' &&
                 GricWasm.isLoaded() &&
                 GricWasm.isReady() &&
                 wasmSessionActive) {
        nClust = GricWasm.getNumClusters();
      }

      const finished = (typeof hasMoreFrames === 'function') ? !hasMoreFrames() : false;

      if (nClust > 0 && finished) {
        // Clustered and complete: Status pill + Re-cluster trigger
        btnPlay.innerHTML = `🟢 Clustered (${nClust}c) • Re-cluster`;
        btnPlay.title =
          `Dataset is clustered (${nClust} clusters). Click to re-cluster from beginning.`;
        btnPlay.classList.remove('danger', 'primary');
        btnPlay.classList.add('btn-clustered');
        btnPlay.style.background = 'rgba(34, 197, 94, 0.18)';
        btnPlay.style.color = '#4ade80';
        btnPlay.style.borderColor = 'rgba(34, 197, 94, 0.5)';
      } else if (nClust > 0 && !finished &&
                 typeof currentFrameIdx !== 'undefined' && currentFrameIdx > 0) {
        // Paused mid-stream: Resume
        btnPlay.innerHTML = `⏸ Paused (${nClust}c) • Resume`;
        btnPlay.title = 'Clustering is paused. Click to resume stream.';
        btnPlay.classList.remove('danger', 'btn-clustered');
        btnPlay.classList.add('primary');
        btnPlay.style.background = 'rgba(245, 158, 11, 0.2)';
        btnPlay.style.color = '#fbbf24';
        btnPlay.style.borderColor = 'rgba(245, 158, 11, 0.5)';
      } else {
        // Unclustered
        btnPlay.innerHTML = '▶ Cluster (Unclustered)';
        btnPlay.title = 'Click to run clustering on active dataset';
        btnPlay.classList.remove('danger', 'btn-clustered');
        btnPlay.classList.add('primary');
        btnPlay.style.background = '';
        btnPlay.style.color = '';
        btnPlay.style.borderColor = '';
      }
    }
    window.updateClusteringButtonUI = updateClusteringButtonUI;

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
        updateClusteringButtonUI();
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
        const btnToggleGpu = document.getElementById('btnToggleGpu');
        if (btnToggleGpu && gpuAvailable) btnToggleGpu.style.display = 'inline-flex';

        // Resource Tracker
        if (badgeWasm) {
          badgeWasm.innerText = 'Native ELF';
          badgeWasm.style.background = 'rgba(74, 222, 128, 0.18)';
          badgeWasm.style.color = '#4ade80';
        }
        if (labelBackend) {
          labelBackend.innerText = useGpu
            ? 'Native Compiled C (NVIDIA CUDA GPU)'
            : 'Native Compiled C (OpenMP / AVX)';
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

        updateClusteringButtonUI();
        if (btnStep) {
          btnStep.disabled = false;
          btnStep.title = 'Step Ingest Single Frame';
          btnStep.style.opacity = '1.0';
        }

        const engineToggleSlider = document.getElementById('engineToggleSlider');
        if (engineToggleSlider) engineToggleSlider.classList.remove('mode-cli');
        if (btnWasm) btnWasm.classList.add('active');
        if (badgeNativeTmux) badgeNativeTmux.style.display = 'none';
        const btnToggleGpu = document.getElementById('btnToggleGpu');
        if (btnToggleGpu) btnToggleGpu.style.display = 'none';
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

      updateClusteringButtonUI();

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

      updateClusteringButtonUI();

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

      updateClusteringButtonUI();
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

    async function killNativeCli() {
      if (typeof DesktopBridge !== 'undefined' && DesktopBridge.killActiveJob) {
        try {
          await DesktopBridge.killActiveJob();
        } catch (err) {
          console.error('Failed to terminate native CLI job:', err);
        }
      }
      isCliRunning = false;
      const btnRun = document.getElementById('btnRunCli');
      const btnRunKnn = document.getElementById('btnRunCliKnn');
      const btnKill = document.getElementById('btnKillCli');
      const btnPlay = document.getElementById('btnPlay');
      const btnStop = document.getElementById('btnStop');
      const badgeStatus = document.getElementById('badgeCliStatus');
      const consoleEl = document.getElementById('cliConsoleLog');

      if (btnRun) btnRun.disabled = false;
      if (btnRunKnn) btnRunKnn.disabled = false;
      if (btnKill) btnKill.disabled = true;
      if (btnPlay) {
        btnPlay.innerHTML = '▶ Run gric-cluster';
        btnPlay.disabled = false;
        btnPlay.classList.remove('danger');
        btnPlay.classList.add('btn-action');
      }
      if (btnStop) {
        btnStop.disabled = true;
      }
      if (badgeStatus) {
        badgeStatus.textContent = 'Stopped';
        badgeStatus.style.background = 'rgba(239, 68, 68, 0.2)';
        badgeStatus.style.color = '#f87171';
      }
      if (consoleEl) {
        consoleEl.textContent += '\n⏹ CLI clustering stopped by user.\n';
        consoleEl.scrollTop = consoleEl.scrollHeight;
      }
      showToast('🛑 Native CLI clustering stopped');
      updateClusteringButtonUI();
    }
    window.killNativeCli = killNativeCli;

    async function stopClustering() {
      let stoppedAny = false;

      // 1. Native CLI execution
      if (typeof isCliRunning !== 'undefined' && isCliRunning) {
        await killNativeCli();
        stoppedAny = true;
      }

      // 2. Batch compute-all execution
      if (typeof isComputeAllRunning !== 'undefined' && isComputeAllRunning) {
        abortComputeAll();
        stoppedAny = true;
      }

      // 3. Interactive simulation streaming
      if (isRunning) {
        pauseSimulation();
        showToast(`⏹ Clustering stopped at frame ${totalFrames.toLocaleString()}`);
        stoppedAny = true;
      }

      // 4. Background worker
      if (typeof GricWasmWorker !== 'undefined' &&
          typeof GricWasmWorker.isBusy === 'function' &&
          GricWasmWorker.isBusy()) {
        GricWasmWorker.pauseBatch();
        stoppedAny = true;
      }

      if (stoppedAny) {
        updateClusteringButtonUI();
        updateUI();
        draw();
      }
    }
    window.stopClustering = stopClustering;

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
      if (activeDatasetSlot === 'A') {
        const selMain = document.getElementById('selectBenchmark');
        if (selMain) selMain.value = 'custom';
      }
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
      if (currentDim >= 3 && (plotDimZ === plotDimY || plotDimZ < 2)) {
        plotDimX = 0;
        plotDimY = 1;
        plotDimZ = 2;
      }
      if (typeof clampPlottingDimensions === 'function') {
        clampPlottingDimensions();
      }
      if (typeof updatePlottingDimSelectorsUI === 'function') {
        updatePlottingDimSelectorsUI();
      }
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

    let hoverDrawRafId = null;
    function requestHoverDraw() {
      if (hoverDrawRafId === null) {
        hoverDrawRafId = requestAnimationFrame(() => {
          hoverDrawRafId = null;
          if (typeof draw === 'function') draw();
        });
      }
    }
    window.requestHoverDraw = requestHoverDraw;

    function updatePanelPointer(clientX, clientY) {
      if (typeof currentDim === 'undefined' || currentDim < 3) {
        if (hoveredPanelPointer !== null) {
          hoveredPanelPointer = null;
          requestHoverDraw();
        }
        return;
      }
      if (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null) {
        if (hoveredPanelPointer !== null) {
          hoveredPanelPointer = null;
          requestHoverDraw();
        }
        return;
      }
      if (typeof dataMode !== 'undefined' && dataMode === 'image') {
        if (hoveredPanelPointer !== null) {
          hoveredPanelPointer = null;
          requestHoverDraw();
        }
        return;
      }
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (hoveredPanelPointer !== null) {
          hoveredPanelPointer = null;
          requestHoverDraw();
        }
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      const W = rect.width;
      const H = rect.height;

      if (px < 0 || px > W || py < 0 || py > H) {
        if (hoveredPanelPointer !== null) {
          hoveredPanelPointer = null;
          requestHoverDraw();
        }
        return;
      }

      const qIdx = getQuadrantAt(clientX, clientY);
      const qRect = getQuadRect(qIdx, W, H);
      if (px < qRect.x || px > qRect.x + qRect.w || py < qRect.y || py > qRect.y + qRect.h) {
        if (hoveredPanelPointer !== null) {
          hoveredPanelPointer = null;
          requestHoverDraw();
        }
        return;
      }

      const metric = mapQuadToMetric(px, py, qIdx, qRect);
      const u = metric.u;
      const v = metric.v;

      // Check if hovering near a sample point for exact snapping
      let snapped = null;
      const activeHl = (typeof lockedClosestSample !== 'undefined' && lockedClosestSample)
        ? null : (typeof hoveredClosestSample !== 'undefined' ? hoveredClosestSample : null);
      if (activeHl && activeHl.point) {
        const pt = activeHl.point;
        const pCoord = (typeof getPlotCoords === 'function') ? getPlotCoords(pt) : pt;
        snapped = {
          x: pCoord.x,
          y: pCoord.y,
          z: (typeof pCoord.z === 'number') ? pCoord.z : 0.0
        };
      }

      if (snapped) {
        persistentCursor3D.x = snapped.x;
        persistentCursor3D.y = snapped.y;
        persistentCursor3D.z = snapped.z;
      } else {
        if (qIdx === 0) {
          // ALONG_X: horizontal is Y, vertical is Z
          persistentCursor3D.y = u;
          persistentCursor3D.z = v;
        } else if (qIdx === 1) {
          // ALONG_Y: horizontal is X, vertical is Z
          persistentCursor3D.x = u;
          persistentCursor3D.z = v;
        } else if (qIdx === 2) {
          // ALONG_Z: horizontal is X, vertical is Y
          persistentCursor3D.x = u;
          persistentCursor3D.y = v;
        } else if (qIdx === 3) {
          // CUSTOM_3D: unproject from orbit camera with depth = 0
          const az = orbitCamera.azimuth, el = orbitCamera.elevation;
          const cosT = Math.cos(az), sinT = Math.sin(az);
          const cosP = Math.cos(el), sinP = Math.sin(el);
          let tx = 0, ty = 0, tz = 0;
          if (orbitCamera && orbitCamera.isLocked) {
            tx = orbitCamera.targetX || 0;
            ty = orbitCamera.targetY || 0;
            tz = orbitCamera.targetZ || 0;
          }
          const dx = u * cosT - v * sinT * sinP;
          const dy = u * sinT + v * cosT * sinP;
          const dz = v * cosP;
          persistentCursor3D.x = tx + dx;
          persistentCursor3D.y = ty + dy;
          persistentCursor3D.z = tz + dz;
        }
      }

      hoveredPanelPointer = {
        qIdx: qIdx,
        screenX: px,
        screenY: py,
        u: u,
        v: v,
        x: persistentCursor3D.x,
        y: persistentCursor3D.y,
        z: persistentCursor3D.z
      };

      requestHoverDraw();
    }
    window.updatePanelPointer = updatePanelPointer;

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
      const isReconImgDown = (typeof isRecon4PanelView !== 'undefined' &&
        isRecon4PanelView && typeof isReconstructionImageMode === 'function' &&
        isReconstructionImageMode());
      if (dataMode === 'image' || isReconImgDown) {
        if (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null) {
          if (py <= 28 && px >= W - 180) {
            maximizedQuad = null;
            if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
            draw();
            return;
          }
        } else {
          if (px >= qRect.x + qRect.w - 32 && px <= qRect.x + qRect.w - 4 &&
              py >= qRect.y && py <= qRect.y + 26) {
            maximizedQuad = qIdx;
            if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
            draw();
            return;
          }
        }

        // Check if clicking or dragging vertical slider in image view panels
        const targetQ = (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null)
          ? maximizedQuad : qIdx;
        if (typeof getPanelSliderRect === 'function') {
          const slider = getPanelSliderRect(targetQ, W, H);
          if (slider) {
            const hitMargin = 5;
            if (px >= slider.trackX - hitMargin &&
                px <= slider.trackX + slider.trackW + hitMargin &&
                py >= slider.trackY &&
                py <= slider.trackY + slider.trackH) {
              const scrollFraction = Math.max(
                0, Math.min(1, (slider.scrollY || 0) / slider.maxScroll)
              );
              const thumbY = slider.trackY + scrollFraction * (slider.trackH - slider.thumbH);

              if (py >= thumbY && py <= thumbY + slider.thumbH) {
                if (typeof startPanelSliderDrag === 'function') {
                  startPanelSliderDrag(
                    slider.effectiveQuad,
                    slider.viewMode,
                    slider.trackY,
                    slider.trackH,
                    slider.thumbH,
                    slider.maxScroll,
                    py - thumbY
                  );
                }
              } else {
                const offset = slider.thumbH / 2;
                if (typeof startPanelSliderDrag === 'function') {
                  startPanelSliderDrag(
                    slider.effectiveQuad,
                    slider.viewMode,
                    slider.trackY,
                    slider.trackH,
                    slider.thumbH,
                    slider.maxScroll,
                    offset
                  );
                }
                if (typeof updatePanelSliderDrag === 'function') {
                  updatePanelSliderDrag(py);
                }
              }
              draw();
              return;
            }
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

      // Check if grabbing 3D view projection vector handle (in panels 0, 1, 2)
      if (typeof hoveredViewVectorHandle !== 'undefined' && hoveredViewVectorHandle !== null) {
        isDragging = true;
        dragMode = 'viewVector';
        draggingViewVectorHandle = { ...hoveredViewVectorHandle };
        activeDragQuad = hoveredViewVectorHandle.qIdx;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        canvas.classList.add('grabbing');
        canvas.style.cursor = 'grabbing';
        if (typeof hoveredPanelPointer !== 'undefined') hoveredPanelPointer = null;
        draw();
        return;
      }

      // Normal Navigation / Drag Mode
      isDragging = true;
      if (typeof hoveredPanelPointer !== 'undefined') hoveredPanelPointer = null;
      activeDragQuad = qIdx;
      dragStartX = e.clientX;
      dragStartY = e.clientY;

      const isRecon = (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView);
      if (isRecon) {
        if (typeof isReconstructionImageMode === 'function' && isReconstructionImageMode()) {
          dragMode = 'pan';
          canvas.classList.add('grabbing');
        } else {
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
        }
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
      if (typeof getPanelSliderState === 'function') {
        const sState = getPanelSliderState();
        if (sState && sState.isDragging) {
          const rect = canvas.getBoundingClientRect();
          const py = e.clientY - rect.top;
          if (typeof updatePanelSliderDrag === 'function') {
            updatePanelSliderDrag(py);
          }
          draw();
          return;
        }
      }

      if (!isDragging || isAddPointMode) return;

      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      dragStartX = e.clientX;
      dragStartY = e.clientY;

      // Dragging 3D View Projection Vector in Panels 0, 1, 2
      if (dragMode === 'viewVector' && typeof draggingViewVectorHandle !== 'undefined' &&
          draggingViewVectorHandle) {
        const qIdx = draggingViewVectorHandle.qIdx;
        const end = draggingViewVectorHandle.end;
        const hInfo = (typeof viewVectorHandles !== 'undefined' && viewVectorHandles)
          ? viewVectorHandles[qIdx] : null;

        if (hInfo) {
          const rect = canvas.getBoundingClientRect();
          const px = e.clientX - rect.left;
          const py = e.clientY - rect.top;
          const qRect = getQuadRect(qIdx, rect.width, rect.height);
          const m = mapQuadToMetric(px, py, qIdx, qRect);

          const uT = hInfo.uT || 0;
          const vT = hInfo.vT || 0;
          const dU = m.u - uT;
          const dV = m.v - vT;

          const L = (end === 'cam') ? (hInfo.L_cam || 0.85) : (hInfo.L_fwd || 0.40);
          const inU = (end === 'cam') ? (-dU / L) : (dU / L);
          const inV = (end === 'cam') ? (-dV / L) : (dV / L);

          const curAz = orbitCamera.azimuth;
          const curEl = orbitCamera.elevation;
          const curCosT = Math.cos(curAz), curSinT = Math.sin(curAz);
          const curCosP = Math.cos(curEl), curSinP = Math.sin(curEl);
          const curVx = -curSinT * curCosP;
          const curVy = curCosT * curCosP;
          const curVz = -curSinP;

          let vxNew = 0, vyNew = 0, vzNew = 0;

          if (qIdx === 0) {
            // Panel X: inU is vy, inV is vz, perp is vx
            vyNew = inU;
            vzNew = inV;
            const r2 = vyNew * vyNew + vzNew * vzNew;
            if (r2 > 0.998 * 0.998) {
              const s = 0.998 / Math.sqrt(r2);
              vyNew *= s;
              vzNew *= s;
            }
            const r2Clamped = Math.min(0.998 * 0.998, vyNew * vyNew + vzNew * vzNew);
            const signVx = (curVx >= 0) ? 1 : -1;
            vxNew = signVx * Math.sqrt(Math.max(0, 1 - r2Clamped));
          } else if (qIdx === 1) {
            // Panel Y: inU is vx, inV is vz, perp is vy
            vxNew = inU;
            vzNew = inV;
            const r2 = vxNew * vxNew + vzNew * vzNew;
            if (r2 > 0.998 * 0.998) {
              const s = 0.998 / Math.sqrt(r2);
              vxNew *= s;
              vzNew *= s;
            }
            const r2Clamped = Math.min(0.998 * 0.998, vxNew * vxNew + vzNew * vzNew);
            const signVy = (curVy >= 0) ? 1 : -1;
            vyNew = signVy * Math.sqrt(Math.max(0, 1 - r2Clamped));
          } else if (qIdx === 2) {
            // Panel Z: inU is vx, inV is vy, perp is vz
            vxNew = inU;
            vyNew = inV;
            const r2 = vxNew * vxNew + vyNew * vyNew;
            if (r2 > 0.998 * 0.998) {
              const s = 0.998 / Math.sqrt(r2);
              vxNew *= s;
              vyNew *= s;
            }
            const r2Clamped = Math.min(0.998 * 0.998, vxNew * vxNew + vyNew * vyNew);
            const signVz = (curVz >= 0) ? 1 : -1;
            vzNew = signVz * Math.sqrt(Math.max(0, 1 - r2Clamped));
          }

          const newEl = Math.max(
            -1.52,
            Math.min(1.52, -Math.asin(Math.max(-0.999, Math.min(0.999, vzNew))))
          );
          const rawAz = Math.atan2(-vxNew, vyNew);
          const dAz = Math.atan2(Math.sin(rawAz - curAz), Math.cos(rawAz - curAz));

          orbitCamera.elevation = newEl;
          orbitCamera.azimuth = curAz + dAz;
          draw();
        }
        return;
      }

      if (dataMode === 'image') {
        const targetQ = (maximizedQuad !== null) ? maximizedQuad : activeDragQuad;
        const viewMode = (typeof getImagePanelViewMode === 'function')
          ? getImagePanelViewMode(targetQ)
          : (targetQ === 2
            ? (imageQ2ViewMode || 'members')
            : (targetQ === 3 ? 'clusters' : ''));

        if (viewMode === 'members') {
          imageMembersScrollY = Math.max(0, (imageMembersScrollY || 0) - dy);
          draw();
        } else if (viewMode === 'knn') {
          imageKnnScrollY = Math.max(0, (imageKnnScrollY || 0) - dy);
          draw();
        } else if (viewMode === 'clusters') {
          imageClustersScrollY = Math.max(0, (imageClustersScrollY || 0) - dy);
          draw();
        }
        return;
      }

      // --- 4-Panel Reconstruction View Drag Handling ---
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (typeof isReconstructionImageMode === 'function' && isReconstructionImageMode()) {
          return;
        }
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
      if (typeof stopPanelSliderDrag === 'function' && stopPanelSliderDrag()) {
        draw();
        return;
      }

      if (dragMode === 'viewVector') {
        isDragging = false;
        dragMode = null;
        draggingViewVectorHandle = null;
        canvas.classList.remove('grabbing');
        canvas.style.cursor = hoveredViewVectorHandle ? 'grab' : '';
        draw();
        return;
      }

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
          const isReconImgClick = (typeof isRecon4PanelView !== 'undefined' &&
            isRecon4PanelView && typeof isReconstructionImageMode === 'function' &&
            isReconstructionImageMode());
          if (dataMode === 'image' || isReconImgClick) {
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

      if (dataMode === 'image') {
        const rect = canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;
        const W = rect.width;
        const H = rect.height;
        const qIdx = getQuadrantAt(e.clientX, e.clientY);
        const isMax = (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null);
        const targetQ = isMax ? maximizedQuad : qIdx;

        let onSlider = false;
        if (typeof getPanelSliderRect === 'function') {
          const slider = getPanelSliderRect(targetQ, W, H);
          if (slider) {
            const hitMargin = 5;
            if (px >= slider.trackX - hitMargin &&
                px <= slider.trackX + slider.trackW + hitMargin &&
                py >= slider.trackY &&
                py <= slider.trackY + slider.trackH) {
              onSlider = true;
              const sState = (typeof getPanelSliderState === 'function')
                ? getPanelSliderState() : null;
              if (sState && sState.hoveredQuad !== slider.effectiveQuad) {
                if (typeof setHoveredSliderQuad === 'function') {
                  setHoveredSliderQuad(slider.effectiveQuad);
                }
                canvas.style.cursor = 'pointer';
                draw();
              }
            }
          }
        }
        if (!onSlider) {
          const sState = (typeof getPanelSliderState === 'function')
            ? getPanelSliderState() : null;
          if (sState && sState.hoveredQuad !== -1) {
            if (typeof setHoveredSliderQuad === 'function') {
              setHoveredSliderQuad(-1);
            }
            canvas.style.cursor = '';
            draw();
          }
        }
        return;
      }

      // --- 4-Panel Reconstruction View Hover Handling ---
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (typeof isReconstructionImageMode === 'function' && isReconstructionImageMode()) {
          const rect = canvas.getBoundingClientRect();
          const px = e.clientX - rect.left;
          const py = e.clientY - rect.top;
          const W = rect.width;
          const H = rect.height;
          const qIdx = getQuadrantAt(e.clientX, e.clientY);

          // Slider hover
          let onSlider = false;
          if (typeof getPanelSliderRect === 'function') {
            const slider = getPanelSliderRect(qIdx, W, H);
            if (slider) {
              const hitMargin = 5;
              if (px >= slider.trackX - hitMargin &&
                  px <= slider.trackX + slider.trackW + hitMargin &&
                  py >= slider.trackY &&
                  py <= slider.trackY + slider.trackH) {
                onSlider = true;
                if (typeof setHoveredSliderQuad === 'function') {
                  setHoveredSliderQuad(slider.effectiveQuad);
                }
                canvas.style.cursor = 'pointer';
                draw();
                return;
              }
            }
          }
          if (!onSlider && typeof getPanelSliderState === 'function') {
            const sState = getPanelSliderState();
            if (sState && sState.hoveredQuad !== -1) {
              if (typeof setHoveredSliderQuad === 'function') {
                setHoveredSliderQuad(-1);
              }
              canvas.style.cursor = '';
              draw();
            }
          }

          // Thumbnail or Header hover in Quad 0 or Quad 1
          if (qIdx === 0 || qIdx === 1) {
            const headerH = 28;
            const qRect = (typeof getImageQuadRect === 'function')
              ? getImageQuadRect(qIdx, W, H)
              : { x: (qIdx === 1 ? W / 2 : 0), y: 0, w: W / 2, h: H / 2 };

            // Hovering over header bar or mode toggle button
            if (py >= qRect.y && py <= qRect.y + headerH) {
              if (reconHoveredTrainingIdx !== -1) {
                reconHoveredTrainingIdx = -1;
                draw();
              }
              canvas.style.cursor = 'pointer';
              return;
            }

            const isSingle = (qIdx === 0 && typeof reconPanelAMode !== 'undefined' &&
                              reconPanelAMode === 'single') ||
                             (qIdx === 1 && typeof reconPanelBMode !== 'undefined' &&
                              reconPanelBMode === 'single');
            if (isSingle) {
              if (reconHoveredTrainingIdx !== -1) {
                reconHoveredTrainingIdx = -1;
                draw();
              }
              canvas.style.cursor = 'pointer';
              return;
            }

            let curQ = 0;
            if (typeof reconLockedQueryIdx !== 'undefined' && reconLockedQueryIdx >= 0) {
              curQ = reconLockedQueryIdx;
            } else if (typeof inspectedImageFrameIdx === 'number' &&
                       inspectedImageFrameIdx >= 0) {
              curQ = inspectedImageFrameIdx;
            }

            const neighbors = (typeof getReconstructionKnnNeighbors === 'function')
              ? getReconstructionKnnNeighbors(curQ) : null;
            let hoveredIdx = -1;
            if (neighbors && neighbors.length > 0) {
              const pad = 8;
              const contentX = qRect.x + pad;
              const contentY = qRect.y + headerH + pad;
              const contentW = qRect.w - pad * 2;
              const thumbSize = (typeof getImageThumbSize === 'function')
                ? getImageThumbSize() : 64;
              const style = (typeof getThumbCardStyle === 'function')
                ? getThumbCardStyle(thumbSize, true) : { infoH: 26 };
              const cardW = thumbSize;
              const cardH = thumbSize + style.infoH;
              const gap = 8;
              const availableW = contentW - 16;
              const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));
              const scrollY = (typeof imageReconKnnScrollY !== 'undefined')
                ? imageReconKnnScrollY : 0;

              for (let r = 0; r < neighbors.length; r++) {
                const col = r % cols;
                const row = Math.floor(r / cols);
                const tx = contentX + col * (cardW + gap);
                const ty = contentY + row * (cardH + gap) - scrollY;

                if (px >= tx && px <= tx + cardW && py >= ty && py <= ty + cardH) {
                  hoveredIdx = neighbors[r].id;
                  break;
                }
              }
            }

            if (hoveredIdx !== reconHoveredTrainingIdx) {
              reconHoveredTrainingIdx = hoveredIdx;
              reconHoveredTrainingSlot = (qIdx === 0) ? 'A' : 'B';
              canvas.style.cursor = (hoveredIdx >= 0) ? 'pointer' : '';
              draw();
            }
            return;
          } else {
            if (reconHoveredTrainingIdx !== -1) {
              reconHoveredTrainingIdx = -1;
              canvas.style.cursor = '';
              draw();
            }
          }
          return;
        }
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
          const pts = (typeof getSlotPoints === 'function')
            ? getSlotPoints(targetSlot)
            : (targetSlot && targetSlot.benchmarkDataset && targetSlot.benchmarkDataset.length > 0
                ? targetSlot.benchmarkDataset : (targetSlot ? targetSlot.pastSamples : null));
          if (!pts || pts.length === 0) return { index: -1, distSq: Infinity };

          const is3D = (targetSlot && targetSlot.currentDim >= 3);
          const scale = (Math.min(qRect.w, qRect.h) / 2.35) * (activeView.zoom || 1.0);
          const cx = qRect.x + qRect.w / 2;
          const cy = qRect.y + qRect.h / 2;

          let bestIdx = -1;
          let bestDistSq = Infinity;
          const checkN = Math.min(pts.length, 1000000);
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
        if (typeof updatePanelPointer === 'function') {
          updatePanelPointer(e.clientX, e.clientY);
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
        if (typeof hoveredViewVectorHandle !== 'undefined' &&
            hoveredViewVectorHandle !== null) {
          hoveredViewVectorHandle = null;
          canvas.style.cursor = '';
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
        if (typeof hoveredViewVectorHandle !== 'undefined' &&
            hoveredViewVectorHandle !== null) {
          hoveredViewVectorHandle = null;
          canvas.style.cursor = '';
          draw();
        }
        return;
      }

      // Check for hover over 3D View Axis handles (in orthogonal panels 0, 1, 2)
      let hitVectorHandle = null;
      if (currentDim >= 3 && maximizedQuad === null &&
          (typeof isRecon4PanelView === 'undefined' || !isRecon4PanelView) &&
          (typeof dataMode === 'undefined' || dataMode !== 'image') &&
          (typeof showCameraViewAxis === 'undefined' || showCameraViewAxis) &&
          (qIdx === 0 || qIdx === 1 || qIdx === 2)) {
        const hInfo = (typeof viewVectorHandles !== 'undefined' && viewVectorHandles)
          ? viewVectorHandles[qIdx] : null;
        if (hInfo) {
          if (hInfo.isPerp) {
            const dTarget = Math.hypot(px - hInfo.targetPos.px, py - hInfo.targetPos.py);
            if (dTarget <= 14) {
              hitVectorHandle = { qIdx, end: 'perp' };
            }
          } else {
            const dFwd = Math.hypot(px - hInfo.pFwd.px, py - hInfo.pFwd.py);
            const dCam = Math.hypot(px - hInfo.pCam.px, py - hInfo.pCam.py);
            if (dFwd <= 15) {
              hitVectorHandle = { qIdx, end: 'fwd' };
            } else if (dCam <= 15) {
              hitVectorHandle = { qIdx, end: 'cam' };
            }
          }
        }
      }

      const prevHandle = hoveredViewVectorHandle;
      if (hitVectorHandle) {
        hoveredViewVectorHandle = hitVectorHandle;
        canvas.style.cursor = 'grab';
        if (!prevHandle || prevHandle.qIdx !== hitVectorHandle.qIdx ||
            prevHandle.end !== hitVectorHandle.end) {
          draw();
        }
        if (hoveredClosestSample !== null) {
          hoveredClosestSample = null;
          draw();
        }
        if (typeof updatePanelPointer === 'function') {
          updatePanelPointer(e.clientX, e.clientY);
        }
        return;
      } else if (prevHandle !== null) {
        hoveredViewVectorHandle = null;
        canvas.style.cursor = '';
        draw();
      }

      if (!highlightClosestSample) {
        if (hoveredClosestSample !== null) {
          hoveredClosestSample = null;
          draw();
        }
        if (typeof updatePanelPointer === 'function') {
          updatePanelPointer(e.clientX, e.clientY);
        }
        return;
      }

      const numPast = pastSamples.length;
      if (numPast === 0) {
        if (hoveredClosestSample !== null) {
          hoveredClosestSample = null;
          draw();
        }
        if (typeof updatePanelPointer === 'function') {
          updatePanelPointer(e.clientX, e.clientY);
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

      const maxCheck = Math.min(numPast, 1000000);
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

      if (typeof updatePanelPointer === 'function') {
        updatePanelPointer(e.clientX, e.clientY);
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
      if (typeof hoveredPanelPointer !== 'undefined' && hoveredPanelPointer !== null) {
        hoveredPanelPointer = null;
        draw();
      }
      if (typeof hoveredViewVectorHandle !== 'undefined' &&
          hoveredViewVectorHandle !== null) {
        hoveredViewVectorHandle = null;
        canvas.style.cursor = '';
        draw();
      }
      if (lockedClosestSample !== null) {
        return;
      }
      if (hoveredClosestSample !== null) {
        hoveredClosestSample = null;
        draw();
      }

      if (typeof setHoveredSliderQuad === 'function') {
        const sState = (typeof getPanelSliderState === 'function')
          ? getPanelSliderState() : null;
        if (sState && sState.hoveredQuad !== -1) {
          setHoveredSliderQuad(-1);
          canvas.style.cursor = '';
          draw();
        }
      }
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const qIdx = getQuadrantAt(e.clientX, e.clientY);

      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (typeof isReconstructionImageMode === 'function' && isReconstructionImageMode()) {
          if (qIdx === 0 || qIdx === 1) {
            const isSingle = (qIdx === 0 && typeof reconPanelAMode !== 'undefined' &&
                              reconPanelAMode === 'single') ||
                             (qIdx === 1 && typeof reconPanelBMode !== 'undefined' &&
                              reconPanelBMode === 'single');
            if (!isSingle) {
              if (e.ctrlKey || e.altKey) {
                const delta = e.deltaY < 0 ? 12 : -12;
                const cur = (typeof imageThumbSize !== 'undefined') ? imageThumbSize : 64;
                const next = Math.max(36, Math.min(220, cur + delta));
                imageThumbSize = next;
                const lbl = document.getElementById('lblImgThumbSize');
                if (lbl) lbl.textContent = `${next}px`;
                draw();
                return;
              }
              const slider = (typeof getPanelSliderRect === 'function')
                ? getPanelSliderRect(qIdx, canvas.width, canvas.height) : null;
              const maxS = slider ? slider.maxScroll : 2000;
              const step = e.shiftKey ? 90 : 30;
              imageReconKnnScrollY = Math.max(
                0,
                Math.min(maxS, (imageReconKnnScrollY || 0) + (e.deltaY > 0 ? step : -step))
              );
              draw();
              return;
            }
          }

          const slotC = (typeof datasetSlots !== 'undefined') ? datasetSlots['C'] : null;
          const totalQ = (slotC && slotC.benchmarkDataset)
            ? slotC.benchmarkDataset.length : 0;
          if (totalQ > 0) {
            const step = e.shiftKey ? 10 : (e.altKey ? 50 : 1);
            const delta = e.deltaY > 0 ? step : -step;
            const curIdx = (typeof inspectedImageFrameIdx === 'number' &&
                            inspectedImageFrameIdx >= 0)
              ? inspectedImageFrameIdx : 0;
            const newIdx = Math.max(0, Math.min(totalQ - 1, curIdx + delta));
            if (typeof selectImageFrame === 'function') {
              selectImageFrame(newIdx);
            } else {
              inspectedImageFrameIdx = newIdx;
              if (typeof updateUI === 'function') updateUI();
              draw();
            }
          }
          return;
        }
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
        const targetQ = (maximizedQuad !== null) ? maximizedQuad : qIdx;
        const viewMode = (typeof getImagePanelViewMode === 'function')
          ? getImagePanelViewMode(targetQ)
          : (targetQ === 2
            ? (imageQ2ViewMode || 'members')
            : (targetQ === 3 ? 'clusters' : ''));

        if ((e.ctrlKey || e.altKey) &&
            (viewMode === 'members' || viewMode === 'knn' || viewMode === 'clusters')) {
          const delta = e.deltaY < 0 ? 12 : -12;
          const cur = (typeof imageThumbSize !== 'undefined') ? imageThumbSize : 64;
          const next = Math.max(36, Math.min(220, cur + delta));
          imageThumbSize = next;
          const lbl = document.getElementById('lblImgThumbSize');
          if (lbl) lbl.textContent = `${next}px`;
          draw();
          return;
        }

        if (viewMode === 'members') {
          imageMembersScrollY = Math.max(
            0, (imageMembersScrollY || 0) + (e.deltaY > 0 ? 30 : -30)
          );
          draw();
        } else if (viewMode === 'knn') {
          imageKnnScrollY = Math.max(
            0, (imageKnnScrollY || 0) + (e.deltaY > 0 ? 30 : -30)
          );
          draw();
        } else if (viewMode === 'clusters') {
          imageClustersScrollY = Math.max(
            0, (imageClustersScrollY || 0) + (e.deltaY > 0 ? 30 : -30)
          );
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
      if (typeof updatePanelPointer === 'function' &&
          typeof hoveredPanelPointer !== 'undefined' && hoveredPanelPointer !== null) {
        updatePanelPointer(e.clientX, e.clientY);
      }
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
          const targetQ = (maximizedQuad !== null) ? maximizedQuad : activeTouchQuad;
          const viewMode = (typeof getImagePanelViewMode === 'function')
            ? getImagePanelViewMode(targetQ)
            : (targetQ === 2
              ? (imageQ2ViewMode || 'members')
              : (targetQ === 3 ? 'clusters' : ''));

          if (viewMode === 'members') {
            imageMembersScrollY = Math.max(0, (imageMembersScrollY || 0) - dy);
            draw();
          } else if (viewMode === 'knn') {
            imageKnnScrollY = Math.max(0, (imageKnnScrollY || 0) - dy);
            draw();
          } else if (viewMode === 'clusters') {
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

    async function handleGenButtonClick(slotId = null) {
      const targetSlot = (slotId && DATASET_SLOTS.includes(slotId))
        ? slotId : activeDatasetSlot;

      if (activeDatasetSlot !== targetSlot) {
        switchDatasetSlot(targetSlot);
      }

      if (typeof pauseSimulation === 'function' && isRunning) {
        pauseSimulation();
      }

      if (typeof updateSlotGenState === 'function') {
        updateSlotGenState(targetSlot, 'generating');
      }

      await new Promise(r => setTimeout(r, 35));

      stageDataset(null, targetSlot);

      if (typeof updateSlotGenState === 'function') {
        updateSlotGenState(targetSlot, 'ready');
      }

      const slotObj = datasetSlots[targetSlot];
      const bKey = slotObj ? slotObj.benchmarkKey : currentBenchmark;
      const count = slotObj && slotObj.benchmarkDataset
        ? slotObj.benchmarkDataset.length
        : (benchmarkDataset ? benchmarkDataset.length : 0);
      if (typeof showToast === 'function') {
        showToast(
          `🎲 Generated & loaded dataset [${targetSlot}] "${bKey}" ` +
          `(${count.toLocaleString()} pts)`
        );
      }
    }
    window.handleGenButtonClick = handleGenButtonClick;

    async function runPredefinedTest(testKey) {
      if (!testKey) return;

      if (testKey === 'asteroid-recon' || testKey === 'asteroid-recon-10k') {
        if (typeof setupAsteroidReconTest === 'function') {
          await setupAsteroidReconTest(10000, 0.80);
        }
        return;
      }

      if (testKey === 'asteroid-recon-2k') {
        if (typeof setupAsteroidReconTest === 'function') {
          await setupAsteroidReconTest(2000, 0.80);
        }
        return;
      }

      // If in 4-Panel ABCD View, exit to single panel view
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (typeof setRecon4PanelView === 'function') {
          setRecon4PanelView(false);
        }
      }

      // Single slot benchmark tests
      const targetSlot = activeDatasetSlot || 'A';
      const slot = (typeof datasetSlots !== 'undefined') ? datasetSlots[targetSlot] : null;
      if (slot) {
        slot.benchmarkKey = testKey;
      }
      currentBenchmark = testKey;

      const selBench = document.getElementById('selectBenchmark');
      if (selBench) {
        selBench.value = testKey;
      }
      const selBenchSide = document.getElementById('selectBenchmarkSide');
      if (selBenchSide) {
        selBenchSide.value = testKey;
      }

      // Adjust parameters based on benchmark type
      if (testKey.startsWith('32D') || testKey.startsWith('128D') || testKey.startsWith('512D')) {
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
      } else if (testKey.startsWith('img-asteroid')) {
        if (typeof setClusteringRlim === 'function') {
          setClusteringRlim(2.98, false);
        } else {
          rlim = 2.98;
        }
      } else if (testKey.startsWith('img-ball')) {
        const imgRlim = (testKey === 'img-ball-3') ? 11.0 : 8.0;
        if (typeof setClusteringRlim === 'function') {
          setClusteringRlim(imgRlim, false);
        } else {
          rlim = imgRlim;
        }
      }

      // Switch 2D / 3D view mode if needed
      if (typeof is3DBenchmark === 'function' && is3DBenchmark(testKey)) {
        if (typeof setViewMode3D === 'function' && !is3DView) {
          setViewMode3D(true);
        }
      }

      // Auto-generate and stage
      if (typeof handleGenButtonClick === 'function') {
        await handleGenButtonClick(targetSlot);
      } else if (typeof stageDataset === 'function') {
        stageDataset(testKey, targetSlot);
      }
    }
    window.runPredefinedTest = runPredefinedTest;

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
      const selBench = document.getElementById(`selectBenchmark_${sId}`) ||
        (sId === 'A' ? document.getElementById('selectBenchmark') : null);
      if (selBench) {
        selBench.addEventListener('change', (e) => {
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
          const newBench = e.target.value;
          const slot = datasetSlots[sId];
          if (slot) {
            slot.benchmarkKey = newBench;
          }
          currentBenchmark = newBench;

          if (newBench.startsWith('32D') ||
              newBench.startsWith('128D') ||
              newBench.startsWith('512D')) {
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
          } else if (newBench.startsWith('img-asteroid')) {
            if (typeof setClusteringRlim === 'function') {
              setClusteringRlim(2.98, false);
            } else {
              rlim = 2.98;
            }
          } else if (newBench.startsWith('img-ball')) {
            const imgRlim = (newBench === 'img-ball-3') ? 11.0 : 8.0;
            if (typeof setClusteringRlim === 'function') {
              setClusteringRlim(imgRlim, false);
            } else {
              rlim = imgRlim;
            }
          }

          const descEl = document.getElementById('benchmarkDesc');
          if (descEl && typeof BENCHMARK_DESCS !== 'undefined') {
            descEl.innerHTML = BENCHMARK_DESCS[newBench] || `<b>${newBench}</b>`;
          }

          const selSide = document.getElementById('selectBenchmarkSide');
          if (selSide) selSide.value = newBench;

          if (typeof stageDataset === 'function') {
            stageDataset(newBench, sId);
          } else if (typeof updateSlotGenState === 'function') {
            updateSlotGenState(sId, 'pending');
          }
          if (typeof updateDatasetStatusBadge === 'function') {
            updateDatasetStatusBadge();
          }
          if (typeof renderReconstructionDashboard === 'function') {
            renderReconstructionDashboard();
          }
          if (typeof draw === 'function') {
            draw();
          }
        });
      }

      // Loop count dropdown per slot
      const selLoop = document.getElementById(`selectLoop_${sId}`) ||
        (sId === 'A' ? document.getElementById('selectLoop') : null);
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
          handleGenButtonClick(sId);
        });
      }

      // Shuffle button per slot
      const btnShuffle = document.getElementById(`btnShuffle_${sId}`);
      if (btnShuffle) {
        btnShuffle.addEventListener('click', (e) => {
          e.stopPropagation();
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
          pauseSimulation();
          shuffleDataset(sId);
          if (typeof showToast === 'function') {
            showToast(`🔀 Shuffled frame order for Dataset [${sId}]`);
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

      // Probe button per slot
      const btnProbeSlot = document.getElementById(`btnProbeDataset_${sId}`);
      if (btnProbeSlot) {
        btnProbeSlot.addEventListener('click', (e) => {
          e.stopPropagation();
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
          triggerDatasetProbe(sId);
        });
      }

      // Clustering status button: switch to slot (if needed) and (re)run clustering
      const btnClustPill = document.getElementById(`datasetClusteredPill_${sId}`);
      if (btnClustPill) {
        btnClustPill.addEventListener('click', (e) => {
          e.stopPropagation();
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
          if (engineMode === 'cli') {
            if (isCliRunning) {
              killNativeCli();
            } else {
              runNativeCli();
            }
          } else {
            const btnPlayEl = document.getElementById('btnPlay');
            if (btnPlayEl) {
              btnPlayEl.click();
            }
          }
        });
      }

      // k-NN status button: switch to slot (if needed) and (re)run k-NN
      const btnKnnPill = document.getElementById(`datasetKnnPill_${sId}`);
      if (btnKnnPill) {
        btnKnnPill.addEventListener('click', (e) => {
          e.stopPropagation();
          if (activeDatasetSlot !== sId) {
            switchDatasetSlot(sId);
          }
          const btnRunKnn = document.getElementById('btnRunKnn');
          if (btnRunKnn) {
            btnRunKnn.click();
          }
        });
      }
    });

    // Active Dataset Slot Badge in CTRL row: click to cycle slot
    const ctrlSlotBadge = document.getElementById('ctrlSelectedSlotBadge');
    if (ctrlSlotBadge) {
      ctrlSlotBadge.addEventListener('click', () => {
        if (!multiDatasetEnabled && typeof setMultiDatasetEnabled === 'function') {
          setMultiDatasetEnabled(true);
        }
        const slots = DATASET_SLOTS;
        const idx = slots.indexOf(activeDatasetSlot || 'A');
        const nextSlot = slots[(idx + 1) % slots.length];
        switchDatasetSlot(nextSlot);
      });
    }

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

    const selectPredefinedTests = document.getElementById('selectPredefinedTests');
    if (selectPredefinedTests) {
      selectPredefinedTests.addEventListener('change', async (e) => {
        const val = e.target.value;
        if (!val) return;
        try {
          await runPredefinedTest(val);
        } finally {
          e.target.value = '';
        }
      });
    }

    const selectBenchmarkLegacy = document.getElementById('selectBenchmark');
    if (selectBenchmarkLegacy) {
      selectBenchmarkLegacy.addEventListener('change', async (e) => {
        const newBench = e.target.value;
        if (newBench === 'asteroid-recon') {
          if (typeof setupAsteroidReconTest === 'function') {
            await setupAsteroidReconTest(10000, 0.80);
          }
          return;
        }

        const slot = datasetSlots[activeDatasetSlot];
        if (slot) {
          slot.benchmarkKey = newBench;
        }
        currentBenchmark = newBench;

        if (newBench.startsWith('32D')) {
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
        } else if (newBench.startsWith('img-asteroid')) {
          if (typeof setClusteringRlim === 'function') {
            setClusteringRlim(2.98, false);
          } else {
            rlim = 2.98;
          }
        } else if (newBench.startsWith('img-ball')) {
          const imgRlim = (newBench === 'img-ball-3') ? 11.0 : 8.0;
          if (typeof setClusteringRlim === 'function') {
            setClusteringRlim(imgRlim, false);
          } else {
            rlim = imgRlim;
          }
        }
        if (typeof updateSlotGenState === 'function') {
          updateSlotGenState(activeDatasetSlot, 'pending');
        }
        await tryAutoLoadCompanionProfile(newBench, activeDatasetSlot);
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

    const btnPlayEl = document.getElementById('btnPlay');
    if (btnPlayEl) {
      btnPlayEl.addEventListener('click', () => {
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
        if (isRunning) {
          pauseSimulation();
          return;
        }

        // If clustering is finished, re-cluster from beginning
        let nClust = 0;
        if (typeof useTiles !== 'undefined' && useTiles) {
          nClust = (typeof jointTuplesMap !== 'undefined' && jointTuplesMap)
            ? jointTuplesMap.size
            : (typeof tileEngineX !== 'undefined' && tileEngineX.clusters
                ? tileEngineX.clusters.length : 0);
        } else if (typeof clusters !== 'undefined' && clusters && clusters.length > 0) {
          nClust = clusters.length;
        } else if (typeof GricWasm !== 'undefined' &&
                   GricWasm.isLoaded() &&
                   GricWasm.isReady() &&
                   wasmSessionActive) {
          nClust = GricWasm.getNumClusters();
        }

        if (nClust > 0 && typeof hasMoreFrames === 'function' && !hasMoreFrames()) {
          resetClustering(true);
          currentFrameIdx = 0;
          if (typeof showToast === 'function') {
            showToast('↺ Re-clustering active dataset from frame 0...');
          }
        }

        startSimulation();
      });
    }

    const btnStopEl = document.getElementById('btnStop');
    if (btnStopEl) {
      btnStopEl.addEventListener('click', () => {
        stopClustering();
      });
    }

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
        const isClusteringRunning = (typeof isRunning !== 'undefined' && isRunning) ||
          (typeof isComputeAllRunning !== 'undefined' && isComputeAllRunning) ||
          (typeof isCliRunning !== 'undefined' && isCliRunning);
        if (isClusteringRunning) {
          e.preventDefault();
          stopClustering();
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
        handleGenButtonClick(activeDatasetSlot);
      });
    }

    const btnStageDatasetSide = document.getElementById('btnStageDatasetSide');
    if (btnStageDatasetSide) {
      btnStageDatasetSide.addEventListener('click', () => {
        handleGenButtonClick(activeDatasetSlot);
      });
    }

    const btnClearDatasetSide = document.getElementById('btnClearDatasetSide');
    if (btnClearDatasetSide) {
      btnClearDatasetSide.addEventListener('click', () => {
        clearDatasetSlot(activeDatasetSlot);
      });
    }

    // Random Ball Seed Checkbox & Re-roll Button
    const chkRandomBallSeed = document.getElementById('chkRandomBallSeed');
    if (chkRandomBallSeed) {
      chkRandomBallSeed.addEventListener('change', (e) => {
        randomBallSeed = e.target.checked;
        if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
          datasetSlots[activeDatasetSlot].randomBallSeed = randomBallSeed;
        }
        if (typeof isImageBenchmark === 'function' && isImageBenchmark(currentBenchmark)) {
          pauseSimulation();
          stageDataset();
          if (typeof showToast === 'function') {
            showToast(randomBallSeed
              ? '🎲 Random seed enabled for bouncing balls'
              : '🔒 Default seed restored for bouncing balls');
          }
        }
      });
    }

    const btnNewBallSeed = document.getElementById('btnNewBallSeed');
    if (btnNewBallSeed) {
      btnNewBallSeed.addEventListener('click', () => {
        randomBallSeed = true;
        ballSeed = Math.floor(Math.random() * 0x7FFFFFFF);
        if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
          datasetSlots[activeDatasetSlot].randomBallSeed = true;
          datasetSlots[activeDatasetSlot].ballSeed = ballSeed;
        }
        if (chkRandomBallSeed) chkRandomBallSeed.checked = true;
        pauseSimulation();
        stageDataset();
        if (typeof showToast === 'function') {
          showToast(`🎲 Generated new bouncing ball seed: ${ballSeed}`);
        }
      });
    }

    // Shuffle Checkbox & Shuffle Now Button
    const chkShuffleFrames = document.getElementById('chkShuffleFrames');
    if (chkShuffleFrames) {
      chkShuffleFrames.addEventListener('change', (e) => {
        shuffleFrames = e.target.checked;
        if (typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot]) {
          datasetSlots[activeDatasetSlot].shuffleFrames = shuffleFrames;
        }
        if (shuffleFrames && !isShuffled &&
            benchmarkDataset && benchmarkDataset.length > 1) {
          pauseSimulation();
          shuffleDataset(activeDatasetSlot);
          if (typeof showToast === 'function') {
            showToast(`🔀 Shuffled frame sequence for [${activeDatasetSlot}]`);
          }
        } else if (!shuffleFrames && isShuffled) {
          pauseSimulation();
          stageDataset();
          if (typeof showToast === 'function') {
            showToast(`Restored original frame sequence for [${activeDatasetSlot}]`);
          }
        }
      });
    }

    const btnShuffleNow = document.getElementById('btnShuffleNow');
    if (btnShuffleNow) {
      btnShuffleNow.addEventListener('click', () => {
        pauseSimulation();
        shuffleDataset(activeDatasetSlot);
        if (typeof showToast === 'function') {
          showToast(`🔀 Permuted frame order for [${activeDatasetSlot}]`);
        }
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

    const btnReset = document.getElementById('btnReset');
    if (btnReset) {
      btnReset.addEventListener('click', () => {
        pauseSimulation();
        resetSimulation();
        currentFrameIdx = 0;
        updateUI();
        draw();
        if (typeof showToast === 'function') {
          showToast('⟲ Simulator reset completely to clean blank canvas.');
        }
      });
    }

    const btnResetViewEl = document.getElementById('btnResetView');
    if (btnResetViewEl) {
      btnResetViewEl.addEventListener('click', resetView);
    }

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
    setupSingleToggle('optToggleCameraViewAxis', () => showCameraViewAxis,
                      v => showCameraViewAxis = v);
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
      syncBtn('optToggleCameraViewAxis', showCameraViewAxis);
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
        syncControlDependencies();
        updateCliCommand();
      });
    });

    const optSq8El = document.getElementById('optSq8');
    const optSq16El = document.getElementById('optSq16');
    const optEq16El = document.getElementById('optEq16');
    const optEq16AdcEl = document.getElementById('optEq16Adc');
    const badgeQuantRatioEl = document.getElementById('badgeQuantRatio');

    function updateClusteringQuantToggles() {
      if (optSq8El) optSq8El.classList.toggle('active', clusterUseSq8);
      if (optSq16El) optSq16El.classList.toggle('active', clusterUseSq16);
      if (optEq16El) optEq16El.classList.toggle('active', clusterUseEq16);
      if (optEq16AdcEl) {
        optEq16AdcEl.classList.toggle('active', clusterUseEq16 && clusterUseEq16Adc);
        optEq16AdcEl.style.display = clusterUseEq16 ? 'inline-flex' : 'none';
      }
      if (badgeQuantRatioEl) {
        badgeQuantRatioEl.textContent = clusterUseEq16 ? '--eq16-ratio' : '--sq16-ratio';
      }
    }

    if (optSq8El) {
      optSq8El.addEventListener('click', () => {
        clusterUseSq8 = !clusterUseSq8;
        if (clusterUseSq8) {
          clusterUseSq16 = false;
          clusterUseEq16 = false;
        }
        updateClusteringQuantToggles();
        syncControlDependencies();
        updateCliCommand();
        draw();
      });
    }

    if (optSq16El) {
      optSq16El.addEventListener('click', () => {
        clusterUseSq16 = !clusterUseSq16;
        if (clusterUseSq16) {
          clusterUseSq8 = false;
          clusterUseEq16 = false;
        }
        updateClusteringQuantToggles();
        syncControlDependencies();
        updateCliCommand();
        draw();
      });
    }

    if (optEq16El) {
      optEq16El.addEventListener('click', () => {
        clusterUseEq16 = !clusterUseEq16;
        if (clusterUseEq16) {
          clusterUseSq8 = false;
          clusterUseSq16 = false;
        }
        updateClusteringQuantToggles();
        syncControlDependencies();
        updateCliCommand();
        draw();
      });
    }

    if (optEq16AdcEl) {
      optEq16AdcEl.addEventListener('click', () => {
        clusterUseEq16Adc = !clusterUseEq16Adc;
        optEq16AdcEl.classList.toggle('active', clusterUseEq16 && clusterUseEq16Adc);
        updateCliCommand();
        draw();
      });
    }

    const optMemoEl = document.getElementById('optMemo');
    if (optMemoEl) {
      optMemoEl.addEventListener('click', () => {
        clusterUseMemo = !clusterUseMemo;
        optMemoEl.classList.toggle('active', clusterUseMemo);
        updateCliCommand();
        draw();
      });
    }

    const inputSq16RatioEl = document.getElementById('inputSq16Ratio');
    const sliderSq16RatioEl = document.getElementById('sliderSq16Ratio');
    if (inputSq16RatioEl && sliderSq16RatioEl) {
      inputSq16RatioEl.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (!isNaN(val) && val >= 0.005 && val <= 0.50) {
          clusterSq16Ratio = val;
          clusterEq16Ratio = val;
          sliderSq16RatioEl.value = val;
          updateCliCommand();
        }
      });
      sliderSq16RatioEl.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        clusterSq16Ratio = val;
        clusterEq16Ratio = val;
        inputSq16RatioEl.value = val.toFixed(3);
        updateCliCommand();
      });
    }

    const optBatchDistEl = document.getElementById('optBatchDist');
    if (optBatchDistEl) {
      optBatchDistEl.addEventListener('click', () => {
        clusterUseBatchDist = !clusterUseBatchDist;
        optBatchDistEl.classList.toggle('active', clusterUseBatchDist);
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
      selBenchSide.addEventListener('change', async (e) => {
        const newBench = e.target.value;
        const slot = datasetSlots[activeDatasetSlot];
        if (slot) {
          slot.benchmarkKey = newBench;
        }
        currentBenchmark = newBench;

        if (newBench.startsWith('32D')) {
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
        } else if (newBench.startsWith('img-asteroid')) {
          if (typeof setClusteringRlim === 'function') {
            setClusteringRlim(2.98, false);
          } else {
            rlim = 2.98;
          }
        } else if (newBench.startsWith('img-ball')) {
          const imgRlim = (newBench === 'img-ball-3') ? 11.0 : 8.0;
          if (typeof setClusteringRlim === 'function') {
            setClusteringRlim(imgRlim, false);
          } else {
            rlim = imgRlim;
          }
        }

        const descEl = document.getElementById('benchmarkDesc');
        if (descEl && typeof BENCHMARK_DESCS !== 'undefined') {
          descEl.innerHTML = BENCHMARK_DESCS[newBench] || `<b>${newBench}</b>`;
        }

        const selSlot = document.getElementById(`selectBenchmark_${activeDatasetSlot}`) ||
          (activeDatasetSlot === 'A' ? document.getElementById('selectBenchmark') : null);
        if (selSlot) selSlot.value = newBench;

        if (typeof updateSlotGenState === 'function') {
          updateSlotGenState(activeDatasetSlot, 'pending');
        }
        await tryAutoLoadCompanionProfile(newBench, activeDatasetSlot);
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

    // Dataset Probe & Calibration (gric-probe)
    const btnProbeEl = document.getElementById('btnProbeDataset');
    if (btnProbeEl) {
      btnProbeEl.addEventListener('click', () => {
        triggerDatasetProbe(activeDatasetSlot || 'A');
      });
    }

    const pillP01 = document.getElementById('pillPresetP01');
    if (pillP01) {
      pillP01.addEventListener('click', () => applyProbePreset('p01'));
    }
    const pillP03 = document.getElementById('pillPresetP03');
    if (pillP03) {
      pillP03.addEventListener('click', () => applyProbePreset('p03'));
    }
    const pillFine = document.getElementById('pillPresetFine');
    if (pillFine) {
      pillFine.addEventListener('click', () => applyProbePreset('fine'));
    }
    const pillBalanced = document.getElementById('pillPresetBalanced');
    if (pillBalanced) {
      pillBalanced.addEventListener('click', () => applyProbePreset('balanced'));
    }
    const pillCoarse = document.getElementById('pillPresetCoarse');
    if (pillCoarse) {
      pillCoarse.addEventListener('click', () => applyProbePreset('coarse'));
    }

    function normalizeGricProfile(raw) {
      if (!raw) return null;
      const norm = { ...raw };

      // 1. Clustering & Presets
      if (raw.clustering) {
        if (typeof raw.clustering.rlim_recommended === 'number') {
          norm.rlim_recommended = raw.clustering.rlim_recommended;
        }
        if (raw.clustering.rlim_presets) {
          if (typeof raw.clustering.rlim_presets.p01 === 'number') {
            norm.rlim_p01 = raw.clustering.rlim_presets.p01;
          }
          if (typeof raw.clustering.rlim_presets.p03 === 'number') {
            norm.rlim_p03 = raw.clustering.rlim_presets.p03;
          }
          if (typeof raw.clustering.rlim_presets.fine === 'number') {
            norm.rlim_fine = raw.clustering.rlim_presets.fine;
          }
          if (typeof raw.clustering.rlim_presets.balanced === 'number') {
            norm.rlim_balanced = raw.clustering.rlim_presets.balanced;
          }
          if (typeof raw.clustering.rlim_presets.coarse === 'number') {
            norm.rlim_coarse = raw.clustering.rlim_presets.coarse;
          }
        }
        if (typeof raw.clustering.recommended_maxcl === 'number') {
          norm.recommended_maxcl = raw.clustering.recommended_maxcl;
        }
        if (raw.clustering.tiles_x !== undefined) norm.tiles_x = raw.clustering.tiles_x;
        if (raw.clustering.tiles_y !== undefined) norm.tiles_y = raw.clustering.tiles_y;
      }

      // 2. Prediction
      if (raw.prediction) {
        if (raw.prediction.enabled !== undefined) {
          norm.pred_enabled = (raw.prediction.enabled === true ||
                               raw.prediction.enabled === 1) ? 1 : 0;
        }
        if (typeof raw.prediction.pred_len === 'number') {
          norm.pred_len = raw.prediction.pred_len;
        }
        if (typeof raw.prediction.pred_h === 'number') {
          norm.pred_h = raw.prediction.pred_h;
        }
        if (typeof raw.prediction.continuity_ratio === 'number') {
          norm.continuity_ratio = raw.prediction.continuity_ratio;
        }
        if (typeof raw.prediction.tm_mixing_coeff === 'number') {
          norm.tm_mixing_coeff = raw.prediction.tm_mixing_coeff;
        }
      }

      // 3. Acceleration & Advanced Clustering
      if (raw.acceleration) {
        if (raw.acceleration.use_sq8 !== undefined) {
          norm.use_sq8 = (raw.acceleration.use_sq8 === true ||
                          raw.acceleration.use_sq8 === 1) ? 1 : 0;
        }
        if (raw.acceleration.use_sq16 !== undefined) {
          norm.use_sq16 = (raw.acceleration.use_sq16 === true ||
                           raw.acceleration.use_sq16 === 1) ? 1 : 0;
        }
        if (raw.acceleration.te4 !== undefined) {
          norm.te4_enabled = (raw.acceleration.te4 === true ||
                              raw.acceleration.te4 === 1) ? 1 : 0;
        }
        if (raw.acceleration.te5 !== undefined) {
          norm.te5_enabled = (raw.acceleration.te5 === true ||
                              raw.acceleration.te5 === 1) ? 1 : 0;
        }
        if (raw.acceleration.sparse_dcc !== undefined) {
          norm.sparse_dcc_enabled = (raw.acceleration.sparse_dcc === true ||
                                     raw.acceleration.sparse_dcc === 1) ? 1 : 0;
        }
        if (raw.acceleration.entropy !== undefined) {
          norm.entropy_enabled = (raw.acceleration.entropy === true ||
                                  raw.acceleration.entropy === 1) ? 1 : 0;
        }
        if (typeof raw.acceleration.entropy_gate === 'number') {
          norm.entropy_gate = raw.acceleration.entropy_gate;
        }
        if (raw.acceleration.soft_bayesian !== undefined) {
          norm.soft_bayesian_enabled = (raw.acceleration.soft_bayesian === true ||
                                        raw.acceleration.soft_bayesian === 1) ? 1 : 0;
        }
        if (typeof raw.acceleration.soft_sigma_coeff === 'number') {
          norm.soft_bayesian_sigma_coeff = raw.acceleration.soft_sigma_coeff;
        }
        if (raw.acceleration.recommend_double !== undefined) {
          norm.recommend_double = (raw.acceleration.recommend_double === true ||
                                   raw.acceleration.recommend_double === 1) ? 1 : 0;
        }
        if (typeof raw.acceleration.noise_floor === 'number') {
          norm.noise_floor_est = raw.acceleration.noise_floor;
        }
        if (raw.acceleration.sq8_scale !== undefined) {
          norm.sq8_params = {
            min_val: raw.acceleration.sq8_min,
            max_val: raw.acceleration.sq8_max,
            scale: raw.acceleration.sq8_scale
          };
        }
        if (raw.acceleration.sq16_scale !== undefined) {
          norm.sq16_params = {
            min_val: raw.acceleration.sq16_min,
            max_val: raw.acceleration.sq16_max,
            scale: raw.acceleration.sq16_scale
          };
        }
      }

      // 4. Pruning
      if (raw.pruning) {
        if (typeof raw.pruning.te3_prune_rate === 'number') {
          norm.te3_prune_rate = raw.pruning.te3_prune_rate;
        }
        if (typeof raw.pruning.te4_marginal_rate === 'number') {
          norm.te4_marginal_rate = raw.pruning.te4_marginal_rate;
        }
        if (typeof raw.pruning.te5_marginal_rate === 'number') {
          norm.te5_marginal_rate = raw.pruning.te5_marginal_rate;
        }
        if (raw.pruning.recommended_mode) {
          norm.recommended_prune_mode = String(raw.pruning.recommended_mode).toUpperCase();
        }
      }

      // 5. Spectral Variance
      if (raw.spectral) {
        if (Array.isArray(raw.spectral.variance_ordering)) {
          norm.perm_dim = raw.spectral.variance_ordering;
        }
        if (Array.isArray(raw.spectral.residual_tail)) {
          norm.residual_tail = raw.spectral.residual_tail;
        }
      }

      // Fallbacks
      if (typeof norm.rlim_recommended === 'number' && typeof norm.rlim_balanced !== 'number') {
        norm.rlim_balanced = norm.rlim_recommended;
      }
      if (typeof norm.rlim_balanced === 'number') {
        if (typeof norm.rlim_fine !== 'number') norm.rlim_fine = norm.rlim_balanced * 0.5;
        if (typeof norm.rlim_p01 !== 'number') norm.rlim_p01 = norm.rlim_fine * 0.4;
        if (typeof norm.rlim_p03 !== 'number') norm.rlim_p03 = norm.rlim_fine * 0.7;
        if (typeof norm.rlim_coarse !== 'number') norm.rlim_coarse = norm.rlim_balanced * 2.0;
        if (typeof norm.rlim_recommended !== 'number') norm.rlim_recommended = norm.rlim_balanced;
      }

      return norm;
    }

    function updateProbePresetPills(activePresetName = 'balanced') {
      const activeProf = window._activeGricProfile ||
        ((typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot])
          ? datasetSlots[activeDatasetSlot].gricProfile : null);
      const prof = normalizeGricProfile(activeProf);
      const p01 = document.getElementById('pillPresetP01');
      const p03 = document.getElementById('pillPresetP03');
      const pFine = document.getElementById('pillPresetFine');
      const pBal = document.getElementById('pillPresetBalanced');
      const pCoarse = document.getElementById('pillPresetCoarse');

      if (prof) {
        if (p01 && typeof prof.rlim_p01 === 'number') {
          p01.title = `Ultra-Fine (D1%): rlim = ${prof.rlim_p01.toFixed(3)}`;
        }
        if (p03 && typeof prof.rlim_p03 === 'number') {
          p03.title = `Very Fine (D3%): rlim = ${prof.rlim_p03.toFixed(3)}`;
        }
        if (pFine && typeof prof.rlim_fine === 'number') {
          pFine.title = `Fine (D5%): rlim = ${prof.rlim_fine.toFixed(3)}`;
        }
        if (pBal && typeof prof.rlim_balanced === 'number') {
          pBal.title = `Balanced (D10%): rlim = ${prof.rlim_balanced.toFixed(3)}`;
        }
        if (pCoarse && typeof prof.rlim_coarse === 'number') {
          pCoarse.title = `Coarse (D25%): rlim = ${prof.rlim_coarse.toFixed(3)}`;
        }
      }

      [p01, p03, pFine, pBal, pCoarse].forEach(p => {
        if (p) p.classList.remove('active');
      });
      if (activePresetName) {
        let pillId = null;
        if (activePresetName === 'p01' || activePresetName === '1%' ||
            activePresetName === 'ultrafine') {
          pillId = 'pillPresetP01';
        } else if (activePresetName === 'p03' || activePresetName === '3%' ||
                   activePresetName === 'vfine') {
          pillId = 'pillPresetP03';
        } else if (activePresetName === 'fine' || activePresetName === 'p05' ||
                   activePresetName === '5%') {
          pillId = 'pillPresetFine';
        } else if (activePresetName === 'coarse' || activePresetName === 'p25' ||
                   activePresetName === '25%') {
          pillId = 'pillPresetCoarse';
        } else {
          pillId = 'pillPresetBalanced';
        }
        const activePill = document.getElementById(pillId);
        if (activePill) activePill.classList.add('active');
      }
    }
    window.updateProbePresetPills = updateProbePresetPills;

    function applyProbePreset(presetName) {
      const activeProf = window._activeGricProfile ||
        ((typeof datasetSlots !== 'undefined' && datasetSlots[activeDatasetSlot])
          ? datasetSlots[activeDatasetSlot].gricProfile : null);

      const prof = normalizeGricProfile(activeProf);
      let targetR = null;

      if (prof) {
        if ((presetName === 'p01' || presetName === '1%' || presetName === 'ultrafine') &&
            typeof prof.rlim_p01 === 'number') {
          targetR = prof.rlim_p01;
        } else if ((presetName === 'p03' || presetName === '3%' || presetName === 'vfine') &&
                   typeof prof.rlim_p03 === 'number') {
          targetR = prof.rlim_p03;
        } else if ((presetName === 'fine' || presetName === 'p05' || presetName === '5%') &&
                   typeof prof.rlim_fine === 'number') {
          targetR = prof.rlim_fine;
        } else if ((presetName === 'coarse' || presetName === 'p25' || presetName === '25%') &&
                   typeof prof.rlim_coarse === 'number') {
          targetR = prof.rlim_coarse;
        } else if (typeof prof.rlim_balanced === 'number') {
          targetR = prof.rlim_balanced;
        }
      } else {
        const baseR = (typeof rlim === 'number' && rlim > 0) ? rlim : 0.100;
        if (presetName === 'p01' || presetName === '1%' || presetName === 'ultrafine') {
          targetR = baseR * 0.2;
        } else if (presetName === 'p03' || presetName === '3%' || presetName === 'vfine') {
          targetR = baseR * 0.35;
        } else if (presetName === 'fine' || presetName === 'p05' || presetName === '5%') {
          targetR = baseR * 0.5;
        } else if (presetName === 'coarse' || presetName === 'p25' || presetName === '25%') {
          targetR = baseR * 2.0;
        } else {
          targetR = baseR;
        }
      }

      if (typeof targetR === 'number' && !isNaN(targetR) && targetR > 0) {
        if (typeof setClusteringRlim === 'function') {
          setClusteringRlim(targetR, false);
        } else if (typeof window.setClusteringRlim === 'function') {
          window.setClusteringRlim(targetR, false);
        } else {
          rlim = parseFloat(targetR.toFixed(3));
          const slR = document.getElementById('sliderRlim');
          if (slR) slR.value = rlim;
          const inpR = document.getElementById('inputRlim');
          if (inpR) inpR.value = rlim.toFixed(3);
        }
      }

      updateProbePresetPills(presetName);

      const capPreset = presetName.charAt(0).toUpperCase() + presetName.slice(1);
      if (typeof showToast === 'function') {
        const est = prof ? '' : ' (pre-probe estimate)';
        showToast(`🎯 Applied ${capPreset} preset: rlim = ${Number(targetR).toFixed(3)}${est}`);
      }
      if (typeof syncControlDependencies === 'function') syncControlDependencies();
      if (typeof updateCliCommand === 'function') updateCliCommand();
      if (typeof updateUI === 'function') updateUI();
      if (typeof draw === 'function') draw();
    }

    async function runJsInMemoryProbe(points, dim, onProgress = null) {
      const N = points.length;
      if (N === 0) return null;

      if (onProgress) {
        onProgress(15, 'Computing coordinate statistics');
        await new Promise(r => setTimeout(r, 15));
      }

      const minVal = new Float64Array(dim).fill(Infinity);
      const maxVal = new Float64Array(dim).fill(-Infinity);
      const mean = new Float64Array(dim);
      const M2 = new Float64Array(dim);

      for (let i = 0; i < N; i++) {
        const pt = points[i];
        for (let d = 0; d < dim; d++) {
          let v = 0;
          if (dim <= 3) {
            v = (d === 0) ? pt.x : ((d === 1) ? pt.y : (pt.z || 0));
          } else {
            v = (pt.coords) ? (pt.coords[d] || 0) : ((d === 0) ? pt.x : 0);
          }
          if (v < minVal[d]) minVal[d] = v;
          if (v > maxVal[d]) maxVal[d] = v;
          const count = i + 1;
          const delta = v - mean[d];
          mean[d] += delta / count;
          const delta2 = v - mean[d];
          M2[d] += delta * delta2;
        }
      }

      if (onProgress) {
        onProgress(45, 'Spectral variance sorting');
        await new Promise(r => setTimeout(r, 15));
      }

      const variances = [];
      for (let d = 0; d < dim; d++) {
        const variance = (N > 1) ? (M2[d] / (N - 1)) : 0;
        variances.push({ dim: d, variance: variance, min: minVal[d], max: maxVal[d] });
      }
      variances.sort((a, b) => b.variance - a.variance);
      const perm_dim = variances.map(v => v.dim);

      if (onProgress) {
        onProgress(70, 'Sampling pairwise distance spectrum');
        await new Promise(r => setTimeout(r, 15));
      }

      const nPairs = Math.min(1000, Math.floor((N * (N - 1)) / 2));
      const dists = [];
      for (let p = 0; p < nPairs; p++) {
        const i = Math.floor(Math.random() * N);
        let j = Math.floor(Math.random() * N);
        if (i === j) j = (j + 1) % N;
        const ptA = points[i];
        const ptB = points[j];
        let sq = 0;
        for (let d = 0; d < dim; d++) {
          const vA = (dim <= 3) ? ((d === 0) ? ptA.x : ((d === 1) ? ptA.y : (ptA.z || 0)))
                                : ((ptA.coords) ? (ptA.coords[d] || 0) : 0);
          const vB = (dim <= 3) ? ((d === 0) ? ptB.x : ((d === 1) ? ptB.y : (ptB.z || 0)))
                                : ((ptB.coords) ? (ptB.coords[d] || 0) : 0);
          const diff = vA - vB;
          sq += diff * diff;
        }
        dists.push(Math.sqrt(sq));
      }
      dists.sort((a, b) => a - b);

      const d1 = dists[Math.floor(dists.length * 0.01)] || (dists[0] || 0.02);
      const d3 = dists[Math.floor(dists.length * 0.03)] || (dists[0] || 0.035);
      const d5 = dists[Math.floor(dists.length * 0.05)] || 0.05;
      const d10 = dists[Math.floor(dists.length * 0.10)] || 0.10;
      const d25 = dists[Math.floor(dists.length * 0.25)] || 0.20;
      const median = dists[Math.floor(dists.length * 0.50)] || 0.25;

      if (onProgress) {
        onProgress(90, 'Evaluating temporal continuity');
        await new Promise(r => setTimeout(r, 15));
      }

      let seqSum = 0;
      const seqN = Math.min(N - 1, 300);
      for (let i = 0; i < seqN; i++) {
        const ptA = points[i];
        const ptB = points[i + 1];
        let sq = 0;
        for (let d = 0; d < dim; d++) {
          const vA = (dim <= 3) ? ((d === 0) ? ptA.x : ((d === 1) ? ptA.y : (ptA.z || 0)))
                                : ((ptA.coords) ? (ptA.coords[d] || 0) : 0);
          const vB = (dim <= 3) ? ((d === 0) ? ptB.x : ((d === 1) ? ptB.y : (ptB.z || 0)))
                                : ((ptB.coords) ? (ptB.coords[d] || 0) : 0);
          const diff = vA - vB;
          sq += diff * diff;
        }
        seqSum += Math.sqrt(sq);
      }
      const meanSeq = (seqN > 0) ? (seqSum / seqN) : median;
      const continuityRatio = (median > 1e-6) ? (meanSeq / median) : 1.0;

      let gMin = Infinity;
      let gMax = -Infinity;
      for (let d = 0; d < dim; d++) {
        if (minVal[d] < gMin) gMin = minVal[d];
        if (maxVal[d] > gMax) gMax = maxVal[d];
      }

      function getCoord(pt, d) {
        if (dim <= 3) {
          return (d === 0) ? pt.x : ((d === 1) ? pt.y : (pt.z || 0));
        }
        return (pt.coords) ? (pt.coords[d] || 0) : ((d === 0) ? pt.x : 0);
      }
      function ptDist(ptA, ptB) {
        let sum = 0;
        for (let d = 0; d < dim; d++) {
          const diff = getCoord(ptA, d) - getCoord(ptB, d);
          sum += diff * diff;
        }
        return Math.sqrt(sum);
      }

      /* Empirical Pruning Evaluation */
      let te3Rate = 0;
      let te4Marg = 0;
      let te5Marg = 0;
      let te4_enabled = 0;
      let te5_enabled = 0;
      let recommended_prune_mode = '3P';

      if (N >= 6) {
        const kAnchors = Math.min(32, Math.max(4, Math.floor(N / 3)));
        const anchorIndices = [];
        for (let k = 0; k < kAnchors; k++) {
          const frac = (kAnchors > 1) ? k / (kAnchors - 1) : 0;
          anchorIndices.push(Math.floor(frac * (N - 1)));
        }

        const dcc = new Float64Array(kAnchors * kAnchors);
        for (let i = 0; i < kAnchors; i++) {
          const pA = points[anchorIndices[i]];
          for (let j = i + 1; j < kAnchors; j++) {
            const pB = points[anchorIndices[j]];
            const dist = ptDist(pA, pB);
            dcc[i * kAnchors + j] = dist;
            dcc[j * kAnchors + i] = dist;
          }
        }

        const mQueries = Math.min(64, N);
        let totalCandidates = 0;
        let te3Pruned = 0;
        let te4Pruned = 0;
        let te5Pruned = 0;
        let greedyEvals = 0;
        let entropyEvals = 0;
        const dfc = new Float64Array(kAnchors);

        for (let m = 0; m < mQueries; m++) {
          const frac = (mQueries > 1) ? m / (mQueries - 1) : 0;
          const qPt = points[Math.floor(frac * (N - 1))];

          for (let k = 0; k < kAnchors; k++) {
            dfc[k] = ptDist(qPt, points[anchorIndices[k]]);
          }

          let a1 = 0;
          let minDfc = dfc[0];
          for (let k = 1; k < kAnchors; k++) {
            if (dfc[k] < minDfc) {
              minDfc = dfc[k];
              a1 = k;
            }
          }

          const unpruned = [];
          for (let k = 0; k < kAnchors; k++) {
            if (k === a1) continue;
            totalCandidates++;
            const te3Bound = Math.abs(dfc[a1] - dcc[a1 * kAnchors + k]);
            if (te3Bound > d10) {
              te3Pruned++;
            } else {
              unpruned.push(k);
            }
          }

          if (unpruned.length >= 3 && dim >= 8) {
            const candLb = unpruned.map(k => Math.abs(dfc[a1] - dcc[a1 * kAnchors + k]));
            const orderG = unpruned.map((_, idx) => idx).sort((a, b) => candLb[a] - candLb[b]);
            for (let u = 0; u < orderG.length; u++) {
              greedyEvals++;
              if (dfc[unpruned[orderG[u]]] <= d10) break;
            }
            const medLb = candLb[orderG[Math.floor(orderG.length / 2)]];
            const orderE = unpruned.map((_, idx) => idx).sort((a, b) => {
              return Math.abs(candLb[a] - medLb) - Math.abs(candLb[b] - medLb);
            });
            for (let u = 0; u < orderE.length; u++) {
              entropyEvals++;
              if (dfc[unpruned[orderE[u]]] <= d10) break;
            }
          }

          if (unpruned.length > 1) {
            const a2 = unpruned[0];
            const remaining = [];
            for (let u = 1; u < unpruned.length; u++) {
              const k = unpruned[u];
              const te3Bound2 = Math.abs(dfc[a2] - dcc[a2 * kAnchors + k]);
              if (te3Bound2 > d10) continue;

              if (typeof calc_min_dist_4pt === 'function') {
                const minD4 = calc_min_dist_4pt(
                  dfc[a1], dfc[a2], dcc[a1 * kAnchors + a2],
                  dcc[a1 * kAnchors + k], dcc[a2 * kAnchors + k]
                );
                if (minD4 > d10) {
                  te4Pruned++;
                  continue;
                }
              }
              remaining.push(k);
            }

            if (remaining.length > 1 && dim >= 3 &&
                typeof calc_min_dist_5pt === 'function') {
              const a3 = remaining[0];
              for (let u = 1; u < remaining.length; u++) {
                const k = remaining[u];
                const te3Bound3 = Math.abs(dfc[a3] - dcc[a3 * kAnchors + k]);
                if (te3Bound3 > d10) continue;
                const minD4_3 = calc_min_dist_4pt(
                  dfc[a1], dfc[a3], dcc[a1 * kAnchors + a3],
                  dcc[a1 * kAnchors + k], dcc[a3 * kAnchors + k]
                );
                if (minD4_3 > d10) continue;

                const minD5 = calc_min_dist_5pt(
                  dfc[a1], dfc[a2], dfc[a3],
                  dcc[k * kAnchors + a1], dcc[k * kAnchors + a2], dcc[k * kAnchors + a3],
                  dcc[a1 * kAnchors + a2], dcc[a1 * kAnchors + a3],
                  dcc[a2 * kAnchors + a3]
                );
                if (minD5 > d10) {
                  te5Pruned++;
                }
              }
            }
          }
        }

        te3Rate = (totalCandidates > 0) ? (te3Pruned / totalCandidates) : 0;
        te4Marg = (totalCandidates > 0) ? (te4Pruned / totalCandidates) : 0;
        te5Marg = (totalCandidates > 0) ? (te5Pruned / totalCandidates) : 0;

        const costDist = 2.0 * dim;
        const benefitTe4 = te4Marg * costDist - 40.0;
        const benefitTe5 = te5Marg * costDist - 120.0;

        if (dim >= 8 && benefitTe5 > 0 && te5Marg >= 0.03) {
          te4_enabled = 1;
          te5_enabled = 1;
          recommended_prune_mode = '5P';
        } else if (dim >= 4 && benefitTe4 > 0 && te4Marg >= 0.03) {
          te4_enabled = 1;
          te5_enabled = 0;
          recommended_prune_mode = '4P';
        } else {
          te4_enabled = 0;
          te5_enabled = 0;
          recommended_prune_mode = '3P';
        }

        let entropy_enabled = 0;
        if (dim >= 8 && greedyEvals > 20 && entropyEvals < greedyEvals * 0.95) {
          entropy_enabled = 1;
        }
      }

      let tmMixingCoeff = 0.0;
      if (continuityRatio < 0.20) {
        tmMixingCoeff = 0.35;
      } else if (continuityRatio < 0.50) {
        tmMixingCoeff = 0.15;
      }

      const noiseFloorEst = meanSeq / Math.sqrt(2.0 * dim);
      const eta = (d10 > 0) ? (noiseFloorEst / d10) : 0;
      const softBayesianEnabled = (eta > 0.25) ? 1 : 0;
      const softBayesianSigmaCoeff = softBayesianEnabled
        ? Math.min(1.5, 1.0 + (eta - 0.25))
        : 1.0;

      const recMaxcl = Math.min(10000, Math.max(100, Math.floor(N * 0.15)));
      const sparseDccEnabled = (recMaxcl >= 2000) ? 1 : 0;
      const recommendDouble = ((gMax - gMin) > 1e7) ? 1 : 0;

      if (onProgress) {
        onProgress(100, 'Profile calibration complete');
      }

      return {
        num_frames: N,
        dim: dim,
        rlim_p01: parseFloat(d1.toFixed(4)),
        rlim_p03: parseFloat(d3.toFixed(4)),
        rlim_fine: parseFloat(d5.toFixed(4)),
        rlim_balanced: parseFloat(d10.toFixed(4)),
        rlim_coarse: parseFloat(d25.toFixed(4)),
        rlim_recommended: parseFloat(d10.toFixed(4)),
        preset_name: 'balanced',
        tiles_x: 1,
        tiles_y: 1,
        pred_enabled: (continuityRatio < 0.6) ? 1 : 0,
        pred_len: 2,
        continuity_ratio: parseFloat(continuityRatio.toFixed(3)),
        tm_mixing_coeff: tmMixingCoeff,
        use_sq8: 0,
        sq8_params: {
          min_val: gMin,
          max_val: gMax,
          scale: (gMax > gMin) ? (gMax - gMin) / 255.0 : 1.0 / 255.0
        },
        use_sq16: (dim >= 32) ? 1 : 0,
        sq16_params: {
          min_val: gMin,
          max_val: gMax,
          scale: (gMax > gMin) ? (gMax - gMin) / 32767.0 : 1.0 / 32767.0
        },
        te4_enabled: te4_enabled,
        te5_enabled: te5_enabled,
        sparse_dcc_enabled: sparseDccEnabled,
        entropy_enabled: (typeof entropy_enabled === 'number') ? entropy_enabled : 0,
        entropy_gate: 0.20,
        soft_bayesian_enabled: softBayesianEnabled,
        soft_bayesian_sigma_coeff: parseFloat(softBayesianSigmaCoeff.toFixed(2)),
        recommend_double: recommendDouble,
        noise_floor_est: parseFloat(noiseFloorEst.toFixed(6)),
        te3_prune_rate: parseFloat(te3Rate.toFixed(4)),
        te4_marginal_rate: parseFloat(te4Marg.toFixed(4)),
        te5_marginal_rate: parseFloat(te5Marg.toFixed(4)),
        recommended_prune_mode: recommended_prune_mode,
        perm_dim: perm_dim
      };
    }

    function handleAutoConfigureFromProbe(rawProfile, slotId) {
      if (!rawProfile) return;
      const profile = normalizeGricProfile(rawProfile);
      if (!profile) return;
      window._activeGricProfile = profile;
      const sId = slotId || (typeof activeDatasetSlot !== 'undefined' ? activeDatasetSlot : 'A');
      if (typeof datasetSlots !== 'undefined' && datasetSlots[sId]) {
        datasetSlots[sId].gricProfile = profile;
      }

      window._probeUndoState = {
        rlim: rlim,
        pruneMode: pruneMode,
        usePred: usePred,
        predHorizon: (typeof predHorizon === 'number') ? predHorizon : 2,
        maxcl: (typeof maxcl === 'number') ? maxcl : 2000,
        clusterUseSq8: clusterUseSq8,
        clusterUseSq16: clusterUseSq16,
        clusterUseBatchDist: clusterUseBatchDist,
        targetMode: targetMode,
        entropyGate: (typeof entropyGate === 'number') ? entropyGate : 0.75,
        useSoftBayesian: useSoftBayesian,
        softBayesianSigmaCoeff: (typeof softBayesianSigmaCoeff === 'number')
          ? softBayesianSigmaCoeff : 1.0,
        useSparseDcc: useSparseDcc,
        useTM: useTM,
        tmMixingCoeff: (typeof tmMixingCoeff === 'number') ? tmMixingCoeff : 0.50,
        useTiles: (typeof useTiles !== 'undefined') ? useTiles : false,
        plotDimX: (typeof plotDimX === 'number') ? plotDimX : 0,
        plotDimY: (typeof plotDimY === 'number') ? plotDimY : 1,
        plotDimZ: (typeof plotDimZ === 'number') ? plotDimZ : 2
      };

      const recR = (typeof profile.rlim_recommended === 'number' && profile.rlim_recommended > 0)
        ? profile.rlim_recommended
        : (typeof profile.rlim_balanced === 'number' ? profile.rlim_balanced : null);

      if (typeof recR === 'number') {
        if (typeof setClusteringRlim === 'function') {
          setClusteringRlim(recR, false);
        } else if (typeof window.setClusteringRlim === 'function') {
          window.setClusteringRlim(recR, false);
        } else {
          rlim = parseFloat(recR.toFixed(3));
          const slR = document.getElementById('sliderRlim');
          if (slR) slR.value = rlim;
          const inpR = document.getElementById('inputRlim');
          if (inpR) inpR.value = rlim.toFixed(3);
        }
      }

      updateProbePresetPills('balanced');

      if (profile.pred_enabled !== undefined) {
        usePred = (profile.pred_enabled === 1 || profile.pred_enabled === true);
        const optPred = document.getElementById('optPred');
        if (optPred) optPred.classList.toggle('active', usePred);
        if (profile.pred_len || profile.pred_h) {
          const phVal = profile.pred_h || profile.pred_len;
          predHorizon = phVal;
          const sl = document.getElementById('sliderPredHorizon');
          const inp = document.getElementById('inputPredHorizon');
          if (sl) sl.value = Math.max(1, Math.min(5, phVal));
          if (inp) inp.value = phVal;
        }
      }

      if (typeof profile.recommended_maxcl === 'number' && profile.recommended_maxcl >= 0) {
        maxcl = profile.recommended_maxcl;
        const inpMaxcl = document.getElementById('inputMaxcl');
        const slMaxcl = document.getElementById('sliderMaxcl');
        if (inpMaxcl) inpMaxcl.value = maxcl;
        const idx = (maxcl === 0) ? 0 :
          Math.min(17, Math.max(1, Math.round(Math.log2(maxcl)) + 1));
        if (slMaxcl) slMaxcl.value = idx;
        if (typeof updateMaxclUnit === 'function') updateMaxclUnit(maxcl);
      }

      if (profile.use_sq16 !== undefined && (profile.use_sq16 === 1 || profile.use_sq16 === true)) {
        clusterUseSq16 = true;
        clusterUseSq8 = false;
      } else if (profile.use_sq8 !== undefined) {
        clusterUseSq8 = (profile.use_sq8 === 1 || profile.use_sq8 === true);
        clusterUseSq16 = false;
      }
      const optSq8 = document.getElementById('optSq8');
      if (optSq8) optSq8.classList.toggle('active', clusterUseSq8);
      const optSq16 = document.getElementById('optSq16');
      if (optSq16) optSq16.classList.toggle('active', clusterUseSq16);

      const targetPrune = profile.recommended_prune_mode ||
        (profile.te5_enabled ? '5P' : (profile.te4_enabled ? '4P' : '3P'));
      pruneMode = targetPrune;
      ['3P', '4P', '5P'].forEach(other => {
        const el = document.getElementById(`prune${other}`);
        if (el) el.classList.toggle('active', other === targetPrune);
      });

      if (profile.entropy_enabled !== undefined) {
        targetMode = (profile.entropy_enabled === 1 || profile.entropy_enabled === true)
          ? 'entropy' : 'greedy';
        const btnG = document.getElementById('modeGreedy');
        const btnE = document.getElementById('modeEntropy');
        if (btnG) btnG.classList.toggle('active', targetMode === 'greedy');
        if (btnE) btnE.classList.toggle('active', targetMode === 'entropy');
        if (typeof profile.entropy_gate === 'number') {
          entropyGate = profile.entropy_gate;
          const slEG = document.getElementById('sliderEntropyGate');
          const inpEG = document.getElementById('inputEntropyGate');
          if (slEG) slEG.value = entropyGate;
          if (inpEG) inpEG.value = entropyGate.toFixed(2);
        }
      }

      if (profile.soft_bayesian_enabled !== undefined) {
        useSoftBayesian = (profile.soft_bayesian_enabled === 1 ||
                           profile.soft_bayesian_enabled === true);
        const optSB = document.getElementById('optSoftBayesian');
        if (optSB) optSB.classList.toggle('active', useSoftBayesian);
        if (typeof profile.soft_bayesian_sigma_coeff === 'number') {
          softBayesianSigmaCoeff = profile.soft_bayesian_sigma_coeff;
          const slBS = document.getElementById('sliderBayesSigma');
          const inpBS = document.getElementById('inputBayesSigma');
          if (slBS) slBS.value = softBayesianSigmaCoeff;
          if (inpBS) inpBS.value = softBayesianSigmaCoeff.toFixed(1);
        }
      }

      if (profile.sparse_dcc_enabled !== undefined) {
        useSparseDcc = (profile.sparse_dcc_enabled === 1 ||
                        profile.sparse_dcc_enabled === true);
        const optSD = document.getElementById('optSparseDcc');
        if (optSD) optSD.classList.toggle('active', useSparseDcc);
      }

      if (profile.tm_mixing_coeff !== undefined) {
        useTM = (profile.tm_mixing_coeff > 0.0);
        tmMixingCoeff = profile.tm_mixing_coeff;
        const optTM = document.getElementById('optTM');
        if (optTM) optTM.classList.toggle('active', useTM);
        const slTM = document.getElementById('sliderTmMix');
        const inpTM = document.getElementById('inputTmMix');
        if (slTM) slTM.value = tmMixingCoeff;
        if (inpTM) inpTM.value = tmMixingCoeff.toFixed(2);
      }

      if (profile.tiles_x > 1 || profile.tiles_y > 1) {
        useTiles = true;
      } else if (profile.tiles_x !== undefined || profile.tiles_y !== undefined) {
        useTiles = false;
      }
      const optTiles = document.getElementById('optTiles');
      if (optTiles) optTiles.classList.toggle('active', useTiles);

      if (profile.perm_dim && profile.perm_dim.length >= 3 && currentDim > 3) {
        if (typeof setPlottingDimensions === 'function') {
          setPlottingDimensions(profile.perm_dim[0], profile.perm_dim[1], profile.perm_dim[2]);
        }
      }

      const lbl = document.getElementById('lblProbeSummary');
      if (lbl) {
        const piStr = (profile.perm_dim && profile.perm_dim.length >= 3) ?
          `[D${profile.perm_dim[0]},D${profile.perm_dim[1]},D${profile.perm_dim[2]}]` : '';
        const cRatio = (typeof profile.continuity_ratio === 'number') ?
          `R_cont=${profile.continuity_ratio.toFixed(2)}` : '';
        lbl.innerText = `${piStr} ${cRatio}`.trim();
        lbl.title = `Variance dims: ${profile.perm_dim ? profile.perm_dim.join(',') : ''}`;
      }

      if (typeof syncControlDependencies === 'function') syncControlDependencies();

      if (typeof showToast === 'function') {
        const rVal = (typeof rlim === 'number') ? rlim.toFixed(3) : '0.100';
        const predStr = usePred ? 'Pred ON' : 'Pred OFF';
        const sq8Str = clusterUseSq16 ? 'SQ16 ON' : (clusterUseSq8 ? 'SQ8 ON' : 'SQ OFF');
        const entStr = (targetMode === 'entropy') ? ' | Entropy ON' : '';
        const sbStr = useSoftBayesian ? ' | Bayes ON' : '';
        const spStr = useSparseDcc ? ' | SparseDCC ON' : '';
        const tmStr = (useTM && tmMixingCoeff > 0) ? ` | TM=${tmMixingCoeff.toFixed(2)}` : '';
        showToast(
          `🔍 Probed [${sId}]: rlim=${rVal} (${pruneMode}) | ${predStr} | ` +
          `${sq8Str}${entStr}${sbStr}${spStr}${tmStr}`
        );
      }
      if (typeof updateCliCommand === 'function') updateCliCommand();
      if (typeof updateUI === 'function') updateUI();
      if (typeof draw === 'function') draw();
    }

    function undoProbeConfiguration() {
      if (!window._probeUndoState) return;
      const prev = window._probeUndoState;
      setClusteringRlim(prev.rlim, false);
      usePred = prev.usePred;
      const optPred = document.getElementById('optPred');
      if (optPred) optPred.classList.toggle('active', usePred);
      if (typeof prev.predHorizon === 'number') {
        predHorizon = prev.predHorizon;
        const sl = document.getElementById('sliderPredHorizon');
        const inp = document.getElementById('inputPredHorizon');
        if (sl) sl.value = Math.max(1, Math.min(5, predHorizon));
        if (inp) inp.value = predHorizon;
      }
      if (typeof prev.maxcl === 'number') {
        maxcl = prev.maxcl;
        const inpMaxcl = document.getElementById('inputMaxcl');
        const slMaxcl = document.getElementById('sliderMaxcl');
        if (inpMaxcl) inpMaxcl.value = maxcl;
        const idx = (maxcl === 0) ? 0 :
          Math.min(17, Math.max(1, Math.round(Math.log2(maxcl)) + 1));
        if (slMaxcl) slMaxcl.value = idx;
        if (typeof updateMaxclUnit === 'function') updateMaxclUnit(maxcl);
      }
      clusterUseSq8 = prev.clusterUseSq8;
      clusterUseSq16 = prev.clusterUseSq16 !== undefined ? prev.clusterUseSq16 : false;
      const optSq8 = document.getElementById('optSq8');
      if (optSq8) optSq8.classList.toggle('active', clusterUseSq8);
      const optSq16 = document.getElementById('optSq16');
      if (optSq16) optSq16.classList.toggle('active', clusterUseSq16);
      clusterUseBatchDist = prev.clusterUseBatchDist !== undefined ? prev.clusterUseBatchDist : true;
      const optBatchDist = document.getElementById('optBatchDist');
      if (optBatchDist) optBatchDist.classList.toggle('active', clusterUseBatchDist);
      pruneMode = prev.pruneMode;
      ['3P', '4P', '5P'].forEach(other => {
        const el = document.getElementById(`prune${other}`);
        if (el) el.classList.remove('active');
      });
      const elPrune = document.getElementById(`prune${pruneMode}`);
      if (elPrune) elPrune.classList.add('active');

      if (prev.targetMode) {
        targetMode = prev.targetMode;
        const btnG = document.getElementById('modeGreedy');
        const btnE = document.getElementById('modeEntropy');
        if (btnG) btnG.classList.toggle('active', targetMode === 'greedy');
        if (btnE) btnE.classList.toggle('active', targetMode === 'entropy');
      }
      if (typeof prev.entropyGate === 'number') {
        entropyGate = prev.entropyGate;
        const slEG = document.getElementById('sliderEntropyGate');
        const inpEG = document.getElementById('inputEntropyGate');
        if (slEG) slEG.value = entropyGate;
        if (inpEG) inpEG.value = entropyGate.toFixed(2);
      }
      if (prev.useSoftBayesian !== undefined) {
        useSoftBayesian = prev.useSoftBayesian;
        const optSB = document.getElementById('optSoftBayesian');
        if (optSB) optSB.classList.toggle('active', useSoftBayesian);
      }
      if (typeof prev.softBayesianSigmaCoeff === 'number') {
        softBayesianSigmaCoeff = prev.softBayesianSigmaCoeff;
        const slBS = document.getElementById('sliderBayesSigma');
        const inpBS = document.getElementById('inputBayesSigma');
        if (slBS) slBS.value = softBayesianSigmaCoeff;
        if (inpBS) inpBS.value = softBayesianSigmaCoeff.toFixed(1);
      }
      if (prev.useSparseDcc !== undefined) {
        useSparseDcc = prev.useSparseDcc;
        const optSD = document.getElementById('optSparseDcc');
        if (optSD) optSD.classList.toggle('active', useSparseDcc);
      }
      if (prev.useTM !== undefined) {
        useTM = prev.useTM;
        const optTM = document.getElementById('optTM');
        if (optTM) optTM.classList.toggle('active', useTM);
      }
      if (typeof prev.tmMixingCoeff === 'number') {
        tmMixingCoeff = prev.tmMixingCoeff;
        const slTM = document.getElementById('sliderTmMix');
        const inpTM = document.getElementById('inputTmMix');
        if (slTM) slTM.value = tmMixingCoeff;
        if (inpTM) inpTM.value = tmMixingCoeff.toFixed(2);
      }
      if (prev.useTiles !== undefined) {
        useTiles = prev.useTiles;
        const optTiles = document.getElementById('optTiles');
        if (optTiles) optTiles.classList.toggle('active', useTiles);
      }

      if (typeof setPlottingDimensions === 'function') {
        setPlottingDimensions(prev.plotDimX, prev.plotDimY, prev.plotDimZ);
      }
      window._probeUndoState = null;
      if (typeof showToast === 'function') {
        showToast('↩️ Reverted probe configuration.');
      }
      if (typeof syncControlDependencies === 'function') syncControlDependencies();
      updateCliCommand();
      updateUI();
      draw();
    }
    window.undoProbeConfiguration = undoProbeConfiguration;

    async function tryAutoLoadCompanionProfile(datasetName, slotId) {
      if (!datasetName) return false;
      if (typeof datasetName !== 'string' || datasetName.startsWith('shm:')) return false;

      const sId = slotId || (typeof activeDatasetSlot !== 'undefined' ? activeDatasetSlot : 'A');
      const baseStem = datasetName.replace(/\.(bin|txt|csv|fits|dat|mp4|fits\.fz)$/i, '');
      const candidates = [
        `${datasetName}.gricprof`,
        `${baseStem}.gricprof`,
        `${baseStem}.bin.gricprof`,
        `${baseStem}.txt.gricprof`
      ];
      const uniqueCandidates = [...new Set(candidates)];

      for (const cand of uniqueCandidates) {
        try {
          let rawText = null;
          if (isDesktopBackend && DesktopBridge.isAvailable()) {
            rawText = await DesktopBridge.readFile(cand);
          } else if (typeof WebFs !== 'undefined' && WebFs.isOpen()) {
            rawText = await WebFs.readFile(cand);
          }
          if (rawText && typeof rawText === 'string' && rawText.trim().startsWith('{')) {
            const parsed = JSON.parse(rawText);
            if (parsed && (parsed.clustering || parsed.acceleration ||
                           parsed.rlim_recommended !== undefined ||
                           parsed.dataset !== undefined)) {
              console.log(`[AutoProfile] Loaded companion profile: ${cand}`);
              handleAutoConfigureFromProbe(parsed, sId);
              return true;
            }
          }
        } catch (e) {
          // File not found or unparseable; try next candidate
        }
      }
      return false;
    }
    window.tryAutoLoadCompanionProfile = tryAutoLoadCompanionProfile;

    function setProbeButtonState(state, slotId, pct = null, phaseText = '') {
      const sId = slotId || activeDatasetSlot || 'A';
      const btnAlgo = document.getElementById('btnProbeDataset');
      const btnSlot = document.getElementById(`btnProbeDataset_${sId}`);
      const lblSummary = document.getElementById('lblProbeSummary');
      const progressWrap = document.getElementById('probeProgressBarWrap');
      const progressBar = document.getElementById('probeProgressBar');

      if (state === 'running') {
        const hasPct = (typeof pct === 'number' && !isNaN(pct));
        const pctStr = hasPct ? `${pct}%` : '';
        const spin = '<span class="gen-spin-icon">🔄</span>';

        if (btnAlgo) {
          btnAlgo.classList.add('btn-probe-running');
          btnAlgo.classList.remove('btn-probe-done');
          btnAlgo.innerHTML = hasPct ? `${spin} ${pctStr}` : `${spin} Probing...`;
          btnAlgo.disabled = true;
          btnAlgo.title = hasPct ? `Probing: ${pctStr} complete` : 'Running gric-probe...';
        }
        if (btnSlot) {
          btnSlot.classList.add('btn-probe-running');
          btnSlot.classList.remove('btn-probe-done');
          btnSlot.innerHTML = hasPct ? `${spin} ${pctStr}` : spin;
          btnSlot.disabled = true;
          btnSlot.title = `Probing Dataset ${sId}: ${pctStr}`;
        }
        if (lblSummary) {
          const detail = phaseText ? `${pctStr} ${phaseText}`.trim() : (pctStr || 'Probing...');
          lblSummary.innerHTML = `${spin} ${detail}`;
          lblSummary.style.color = '#fbbf24';
        }
        if (progressWrap) {
          progressWrap.style.display = 'block';
        }
        if (progressBar && hasPct) {
          progressBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
        }
      } else if (state === 'done') {
        if (btnAlgo) {
          btnAlgo.classList.remove('btn-probe-running');
          btnAlgo.classList.add('btn-probe-done');
          btnAlgo.innerHTML = '<span class="gen-check-icon">✓</span> Probed';
          btnAlgo.disabled = false;
          btnAlgo.title = 'Dataset probed & calibrated! Click to re-run.';
        }
        if (btnSlot) {
          btnSlot.classList.remove('btn-probe-running');
          btnSlot.classList.add('btn-probe-done');
          btnSlot.innerHTML = '<span class="gen-check-icon">✓</span>';
          btnSlot.disabled = false;
          btnSlot.title = `Dataset ${sId} probed & calibrated!`;
        }
        if (lblSummary) {
          lblSummary.style.color = '';
        }
        if (progressBar) {
          progressBar.style.width = '100%';
        }
        setTimeout(() => {
          if (progressWrap) {
            progressWrap.style.display = 'none';
          }
          if (progressBar) {
            progressBar.style.width = '0%';
          }
          if (btnAlgo && btnAlgo.classList.contains('btn-probe-done')) {
            btnAlgo.classList.remove('btn-probe-done');
            btnAlgo.innerHTML = '🔍 Probe';
            btnAlgo.title = 'Probe Dataset (gric-probe)';
          }
          if (btnSlot && btnSlot.classList.contains('btn-probe-done')) {
            btnSlot.classList.remove('btn-probe-done');
            btnSlot.innerHTML = '🔍';
            btnSlot.title = `Probe Dataset ${sId} (gric-probe)`;
          }
        }, 2200);
      } else if (state === 'error') {
        if (btnAlgo) {
          btnAlgo.classList.remove('btn-probe-running');
          btnAlgo.innerHTML = '⚠️ Error';
          btnAlgo.disabled = false;
          btnAlgo.title = 'Probe failed';
        }
        if (btnSlot) {
          btnSlot.classList.remove('btn-probe-running');
          btnSlot.innerHTML = '⚠️';
          btnSlot.disabled = false;
          btnSlot.title = `Probe failed for Dataset ${sId}`;
        }
        if (lblSummary) {
          lblSummary.style.color = '';
        }
        if (progressWrap) {
          progressWrap.style.display = 'none';
        }
        setTimeout(() => {
          if (btnAlgo) {
            btnAlgo.innerHTML = '🔍 Probe';
            btnAlgo.title = 'Probe Dataset (gric-probe)';
          }
          if (btnSlot) {
            btnSlot.innerHTML = '🔍';
            btnSlot.title = `Probe Dataset ${sId} (gric-probe)`;
          }
        }, 2500);
      } else {
        if (btnAlgo) {
          btnAlgo.classList.remove('btn-probe-running', 'btn-probe-done');
          btnAlgo.innerHTML = '🔍 Probe';
          btnAlgo.disabled = false;
          btnAlgo.title = 'Probe Dataset (gric-probe)';
        }
        if (btnSlot) {
          btnSlot.classList.remove('btn-probe-running', 'btn-probe-done');
          btnSlot.innerHTML = '🔍';
          btnSlot.disabled = false;
          btnSlot.title = `Probe Dataset ${sId} (gric-probe)`;
        }
        if (progressWrap) {
          progressWrap.style.display = 'none';
        }
      }
    }

    async function triggerDatasetProbe(slotId) {
      if (window._isProbingActive) return;
      const sId = slotId || activeDatasetSlot || 'A';
      const slot = (typeof datasetSlots !== 'undefined') ? datasetSlots[sId] : null;
      let pts = (slot && slot.benchmarkDataset && slot.benchmarkDataset.length > 0) ?
        slot.benchmarkDataset : benchmarkDataset;

      if (!pts || pts.length === 0) {
        if (typeof generateBenchmark === 'function') {
          rawBenchmarkDataset = generateBenchmark(currentBenchmark, 250);
          applyNoiseToDataset();
          pts = benchmarkDataset;
        }
      }

      if (!pts || pts.length === 0) {
        if (typeof showToast === 'function') {
          showToast(`⚠️ Dataset [${sId}] has no points to probe.`);
        }
        return;
      }

      window._isProbingActive = true;
      setProbeButtonState('running', sId, 0, 'Initializing probe');
      const startTime = Date.now();

      if (typeof showToast === 'function') {
        showToast(`🔍 Probing Dataset [${sId}]...`);
      }

      try {
        let profile = null;
        if (typeof DesktopBridge !== 'undefined' && DesktopBridge.isAvailable()) {
          try {
            const rawName = (slot && slot.stagedDatasetInfo && slot.stagedDatasetInfo.name)
              ? slot.stagedDatasetInfo.name
              : (slot && slot.benchmarkKey ? slot.benchmarkKey
                                           : (currentBenchmark || `dataset_${sId}`));
            const cleanBase = rawName.replace(/\.(txt|csv|fits|dat|mp4|fits\.fz)$/i, '')
                                     .replace(/[^a-zA-Z0-9_.-]/g, '_');
            const stageFile = await DesktopBridge.stageDatasetFile(
              cleanBase,
              pts,
              currentDim,
              (pct, phase) => {
                const overallPct = Math.floor(pct * 0.25);
                setProbeButtonState('running', sId, overallPct, phase);
              }
            );
            profile = await DesktopBridge.runDatasetProbe(stageFile, {
              onProgress: (pct, phase) => {
                const overallPct = 25 + Math.floor(pct * 0.75);
                setProbeButtonState('running', sId, overallPct, phase);
              }
            });
          } catch (err) {
            console.warn('[Probe] Desktop bridge probe error, using in-memory:', err);
          }
        }

        if (!profile) {
          profile = await runJsInMemoryProbe(pts, currentDim, (pct, phase) => {
            setProbeButtonState('running', sId, pct, phase);
          });
        }

        const elapsed = Date.now() - startTime;
        if (elapsed < 300) {
          await new Promise(r => setTimeout(r, 300 - elapsed));
        }

        if (profile) {
          handleAutoConfigureFromProbe(profile, sId);
          setProbeButtonState('done', sId);
        } else {
          setProbeButtonState('error', sId);
        }
      } catch (err) {
        console.error('[Probe] Error during dataset probe:', err);
        setProbeButtonState('error', sId);
      } finally {
        window._isProbingActive = false;
      }
    }

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

    // Note: k-NN Controls moved to sim_knn.js
    // Note: Dim & Density Controls moved to sim_dimdensity.js
    // Note: Reconstruction Engine moved to sim_reconstruction.js

    // Note: Sidebar panel resizer & collapse controller moved to panel_controller.js

    // Note: Workspace, Dual-Mode and Command Palette moved to sim_workspace.js


    initTimelineScrubber();
    initImageScrubber();
    initCommandPalette();
    switchSidebarMode('clustering');
    if (typeof updateReconstructionButtonState === 'function') {
      updateReconstructionButtonState();
    }

    setTimeout(() => { updateTMCanvasDimensions(); resizeCanvas(); }, 50);
    setTimeout(() => { updateTMCanvasDimensions(); resizeCanvas(); }, 250);
