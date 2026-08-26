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

//  ASYNC PRODUCER-CONSUMER DISPLAY LOOP & COMPUTE ENGINE
    // =========================================================================
    let displayLoopId = null;

    function startDisplayLoop() {
      if (displayLoopId !== null) return;
      function displayTick() {
        if (!isRunning && displayLoopId === null) return;
        // Sync WASM state to JS before rendering (deferred from batch frames)
        if (useWasm && wasmSessionActive && GricWasm.isReady()) {
          const snapshot = GricWasm.syncState();
          if (snapshot) {
            GricWasm.applyToJsState(snapshot);
          }
        }
        updateUI();
        draw();
        if (isRunning) {
          displayLoopId = requestAnimationFrame(displayTick);
        } else {
          displayLoopId = null;
        }
      }
      displayLoopId = requestAnimationFrame(displayTick);
    }

    function stopDisplayLoop() {
      if (displayLoopId !== null) {
        cancelAnimationFrame(displayLoopId);
        displayLoopId = null;
      }
    }

    function startSimulation() {
      if (isAddPointMode) setAddPointMode(false);

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

      // Start independent asynchronous 60 FPS display render loop
      startDisplayLoop();

      if (playSpeed <= 0) {
        // "⚡ As fast as possible" Mode:
        // Compute loop runs in ~12ms time-slices,
        // yielding to the display loop between slices.
        function pumpCompute() {
          if (!isRunning) return;
          const sliceStart = performance.now();
          while (hasMoreFrames()
            && (performance.now() - sliceStart < 12))
          {
            stepNextFrame(true);
          }
          if (!hasMoreFrames()) {
            pauseSimulation();
            return;
          }
          playTimer = setTimeout(pumpCompute, 0);
        }
        playTimer = setTimeout(pumpCompute, 0);
        return;
      }

      // Paced playback modes (150ms, 50ms, 15ms, 5ms):
      let lastPacedTime = performance.now();
      let frameAccumulator = 0;
      function pacedCompute() {
        if (!isRunning) return;
        const now = performance.now();
        const delta = now - lastPacedTime;
        lastPacedTime = now;
        frameAccumulator += delta;

        const interval = playSpeed;
        let framesToProcess = Math.floor(frameAccumulator / interval);
        if (framesToProcess > 0) {
          frameAccumulator -= framesToProcess * interval;
          framesToProcess = Math.min(framesToProcess, 100);
          for (let f = 0; f < framesToProcess; f++) {
            if (!hasMoreFrames()) {
              pauseSimulation();
              return;
            }
            stepNextFrame(true);
          }
        }
        playTimer = setTimeout(pacedCompute, Math.max(0, Math.min(interval, 4)));
      }
      playTimer = setTimeout(pacedCompute, 0);
    }

    function pauseSimulation() {
      isRunning = false;
      if (sessionIsActive) {
        sessionElapsedMs = Math.max(0.0001, performance.now() - sessionStartTime);
        const framesClustered = totalFrames - sessionStartFrames;
        sessionAvgFps = sessionElapsedMs > 0.001 ? (framesClustered / (sessionElapsedMs / 1000.0)) : 0.0;
        sessionIsActive = false;
      }

      const btn = document.getElementById('btnPlay');
      btn.innerText = "► Cluster";
      btn.classList.remove('danger');
      btn.classList.add('primary');
      if (playTimer) {
        clearTimeout(playTimer);
        clearInterval(playTimer);
        if (typeof cancelAnimationFrame !== 'undefined') {
          cancelAnimationFrame(playTimer);
        }
        playTimer = null;
      }
      if (typeof GricWasmWorker !== 'undefined' && GricWasmWorker.isBusy()) {
        GricWasmWorker.pauseBatch();
      }
      stopDisplayLoop();
      // Final WASM sync — batch frames since last display tick
      if (useWasm && wasmSessionActive && GricWasm.isReady()) {
        const snapshot = GricWasm.syncState();
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
        btn.classList.add('toggle-active');
        setTab('narrative');
        // Enable C-side trace buffer for WASM explain
        if (typeof GricWasm !== 'undefined' && GricWasm.isReady()) {
          GricWasm.setTrace(true);
        }
      } else {
        btn.classList.remove('toggle-active');
        currentExplanation = [];
        // Disable C-side trace buffer
        if (typeof GricWasm !== 'undefined' && GricWasm.isReady()) {
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

      for (let line of lines) {
        line = line.trim();
        if (!line || line.startsWith('#')) continue;

        const tokens = line.split(/[,\s\t]+/).filter(t => t.length > 0);
        if (tokens.length >= 3) {
          const x = parseFloat(tokens[0]);
          const y = parseFloat(tokens[1]);
          const z = parseFloat(tokens[2]);
          if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            rawPoints.push({ x, y, z });
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
        alert("Error: No valid 2D or 3D coordinates found in file. Expected 'x y [z]' or 'x,y[,z]' per line.");
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

      rawBenchmarkDataset = rawPoints.map(p => ({
        x: ((p.x - midX) / maxSpan) * 1.76,
        y: ((p.y - midY) / maxSpan) * 1.76,
        z: detected3D ? (((p.z - midZ) / maxSpan) * 1.76) : 0.0
      }));
      applyNoiseToDataset();

      currentDim = detected3D ? 3 : 2;
      if (currentDim === 2) {
        maximizedQuad = null;
      }
      BENCHMARK_DESCS["custom"] = `<b>Custom File (${filename})</b>: ${benchmarkDataset.length} ${detected3D ? '3D' : '2D'} frames loaded from upload.`;
      document.getElementById('selectBenchmark').value = 'custom';
      document.getElementById('benchmarkDesc').innerHTML = BENCHMARK_DESCS["custom"];

      resetSimulation();
      resetView();
    }

    const fileInput = document.getElementById('fileUpload');
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => parseCoordinateFile(evt.target.result, file.name);
      reader.readAsText(file);
      fileInput.value = '';
    });

    document.getElementById('btnUpload').addEventListener('click', () => fileInput.click());

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
      if (currentDim === 2) {
        return 2;
      }
      if (maximizedQuad !== null && currentDim === 3) {
        return maximizedQuad;
      }

      const rect = canvas.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      const W = rect.width;
      const H = rect.height;

      if (px < W / 2 && py < H / 2) return 0; // Along X
      if (px >= W / 2 && py < H / 2) return 1; // Along Y
      if (px < W / 2 && py >= H / 2) return 2; // Along Z
      return 3; // Custom 3D
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

      // Check if clicking Maximize / Restore Icon in top-right of quadrant
      if (currentDim === 3 && px >= qRect.x + qRect.w - 30 && px <= qRect.x + qRect.w - 4 &&
          py >= qRect.y && py <= qRect.y + 24) {
        maximizedQuad = (maximizedQuad === qIdx) ? null : qIdx;
        draw();
        return;
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
        if (qIdx === 3 && currentDim === 3 && typeof orbitCamera !== 'undefined') {
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

      const is3DTarget = (qIdx === 3 || maximizedQuad === 3) && currentDim === 3;
      if (is3DTarget) {
        // Shift + Drag in 3D: pan the 3D viewport
        // Normal Drag in 3D: orbit the 3D camera
        dragMode = e.shiftKey ? 'pan' : 'orbit';
        canvas.classList.add('grabbing');
      } else {
        dragMode = 'pan';
        canvas.classList.add('grabbing');
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging || isAddPointMode) return;

      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      dragStartX = e.clientX;
      dragStartY = e.clientY;

      if (dataMode === 'image') {
        if (activeDragQuad === 3 || maximizedQuad === 3) {
          imageGalleryScrollY = Math.max(0, (imageGalleryScrollY || 0) - dy);
          draw();
        }
        return;
      }

      const is3DTarget = (activeDragQuad === 3 || maximizedQuad === 3) && currentDim === 3;

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
              if (enableKnn && knnResults) {
                selectedKnnQuerySample = lockedClosestSample.index;
                if (typeof renderKnnTrace === 'function') renderKnnTrace();
              }
              if (typeof showToast === 'function') {
                showToast(`🔒 Locked Sample #${lockedClosestSample.index} (Click or Esc to unlock)`);
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
        if (currentDim === 2 || qIdx === 2) return { u: p.x, v: p.y, depth: p.z || 0 };
        if (qIdx === 0) return { u: p.y, v: p.z, depth: p.x };
        if (qIdx === 1) return { u: p.x, v: p.z, depth: p.y };
        // CUSTOM_3D (qIdx === 3)
        return project3D(p.x, p.y, p.z, orbitCamera.azimuth, orbitCamera.elevation);
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
        } else if (typeof clustersAssigned !== 'undefined' && clustersAssigned && bestIdx < clustersAssigned.length) {
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
          if (enableKnn && knnResults) {
            selectedKnnQuerySample = bestIdx;
            if (typeof renderKnnTrace === 'function') renderKnnTrace();
          }
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
      if (lockedClosestSample !== null) return;
      if (hoveredClosestSample !== null) {
        hoveredClosestSample = null;
        draw();
      }
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const qIdx = getQuadrantAt(e.clientX, e.clientY);

      if (dataMode === 'image') {
        if (qIdx === 3 || maximizedQuad === 3) {
          imageGalleryScrollY = Math.max(0, (imageGalleryScrollY || 0) + (e.deltaY > 0 ? 30 : -30));
          draw();
        }
        return;
      }

      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;

      const v = quadViews[qIdx];
      if (v) {
        v.zoom = Math.max(0.05, (v.zoom || 1.0) * zoomFactor);
        if (qIdx === 3 && currentDim === 3) {
          orbitCamera.zoom = v.zoom;
        }
      }

      updateZoomBadge();
      draw();
    }, { passive: false });

    canvas.addEventListener('dblclick', (e) => {
      if (isAddPointMode) return;
      const qIdx = getQuadrantAt(e.clientX, e.clientY);
      if (dataMode === 'image') {
        maximizedQuad = (maximizedQuad === qIdx) ? null : qIdx;
        draw();
        return;
      }
      if (qIdx === 3 && currentDim === 3) {
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
          if (qIdx === 3 && currentDim === 3) {
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
        if (currentDim === 3 && px >= qRect.x + qRect.w - 36 && px <= qRect.x + qRect.w &&
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
          if (qIdx === 3 && currentDim === 3 && typeof orbitCamera !== 'undefined') {
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
            if (targetQuad === 3 && currentDim === 3) orbitCamera.zoom = v.zoom;
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
          if (activeTouchQuad === 3 || maximizedQuad === 3) {
            imageGalleryScrollY = Math.max(0, (imageGalleryScrollY || 0) - dy);
            draw();
          }
          return;
        }

        const is3DTarget = (activeTouchQuad === 3 || maximizedQuad === 3) && currentDim === 3;
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
      orbitCamera.azimuth = -35 * (Math.PI / 180);
      orbitCamera.elevation = 25 * (Math.PI / 180);
      draw();
    });

    document.getElementById('presetFront').addEventListener('click', () => {
      orbitCamera.azimuth = 0;
      orbitCamera.elevation = 0;
      draw();
    });

    document.getElementById('presetTop').addEventListener('click', () => {
      orbitCamera.azimuth = 0;
      orbitCamera.elevation = 89 * (Math.PI / 180);
      draw();
    });

    document.getElementById('presetSide').addEventListener('click', () => {
      orbitCamera.azimuth = 90 * (Math.PI / 180);
      orbitCamera.elevation = 0;
      draw();
    });

    document.getElementById('presetReset3D').addEventListener('click', () => {
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

    // Select Benchmark Handler
    document.getElementById('selectBenchmark').addEventListener('change', () => {
      loadSelectedBenchmark();
      resetView();
    });

    document.getElementById('selectSpeed').addEventListener('change', (e) => {
      playSpeed = parseInt(e.target.value);
      const side = document.getElementById('selectSpeedSide');
      if (side) side.value = e.target.value;
      if (isRunning) {
        pauseSimulation();
        startSimulation();
      }
    });

    const selectLoop = document.getElementById('selectLoop');
    if (selectLoop) {
      selectLoop.addEventListener('change', (e) => {
        loopCount = parseInt(e.target.value);
        const side = document.getElementById('selectLoopSide');
        if (side) side.value = e.target.value;
        stageDataset();
      });
    }

    document.getElementById('btnPlay').addEventListener('click', () => {
      if (engineMode === 'cli') {
        runNativeCli();
        return;
      }
      if (isRunning) pauseSimulation();
      else startSimulation();
    });

    document.getElementById('btnStep').addEventListener('click', () => {
      if (engineMode === 'cli') {
        showToast('Step inspection is only available in WASM Interactive Simulation mode');
        return;
      }
      if (isRunning) pauseSimulation();
      stepNextFrame(false);
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

    document.getElementById('tabNarrative').addEventListener('click', () => setTab('narrative'));
    document.getElementById('tabCandidates').addEventListener('click', () => setTab('candidates'));
    const tabTMEl = document.getElementById('tabTM');
    if (tabTMEl) tabTMEl.addEventListener('click', () => setTab('tm'));
    const tabEntropyTraceEl = document.getElementById('tabEntropyTrace');
    if (tabEntropyTraceEl) tabEntropyTraceEl.addEventListener('click', () => setTab('entropy'));

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
      } else if (e.key === 'h' || e.key === 'H' || e.key === 't' || e.key === 'T') {
        const btnToggleTooltips = document.getElementById('btnToggleTooltips');
        if (btnToggleTooltips) btnToggleTooltips.click();
      } else if (e.key === 'z' || e.key === 'Z') {
        const btnResetView = document.getElementById('btnResetView');
        if (btnResetView) btnResetView.click();
      } else if (e.key === '[' || e.key === 'ArrowLeft') {
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
          if (enableKnn && knnResults) {
            selectedKnnQuerySample = lockedClosestSample.index;
            if (typeof renderKnnTrace === 'function') renderKnnTrace();
          }
          if (typeof showToast === 'function') {
            showToast(`🔒 Locked Sample #${lockedClosestSample.index} (Click or Esc to unlock)`);
          }
          draw();
        }
      } else if (e.key === 'Escape') {
        if (lockedClosestSample !== null) {
          lockedClosestSample = null;
          hoveredClosestSample = null;
          if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
          draw();
        }
      } else if (e.key === 'l' || e.key === 'L') {
        returnToLiveStream();
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

    const btnResetClusters = document.getElementById('btnResetClusters');
    if (btnResetClusters) {
      btnResetClusters.addEventListener('click', () => {
        pauseSimulation();
        resetClustering(true);
        if (typeof showToast === 'function') {
          showToast('↺ Cluster models reset. Staged points preserved for next run.');
        }
      });
    }

    document.getElementById('btnReset').addEventListener('click', () => {
      pauseSimulation();
      resetSimulation();
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
        draw();
      });
    }
    if (inputRlim) {
      inputRlim.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0) {
          rlim = v;
          if (sliderRlim) sliderRlim.value = Math.max(0.02, Math.min(0.30, v));
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
    setupSingleToggle('optToggleHighlightClosest', () => highlightClosestSample, v => {
      highlightClosestSample = v;
      if (!v) hoveredClosestSample = null;
    });

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
        document.getElementById('selectBenchmark').value = e.target.value;
        loadSelectedBenchmark();
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

    function updateKnnButtonUI(computing, statusText) {
      const btnTop = document.getElementById('btnRunKnn');
      const btnSide = document.getElementById('btnRunKnnSide');
      const badgeTop = document.getElementById('knnStatusBadgeTop');

      if (computing) {
        if (btnTop) {
          btnTop.innerHTML = '⏹ Stop k-NN';
          btnTop.style.background = 'rgba(239, 68, 68, 0.25)';
          btnTop.style.color = '#f87171';
          btnTop.style.borderColor = 'rgba(239, 68, 68, 0.6)';
          btnTop.title = 'Click to stop / cancel k-NN computation';
        }
        if (btnSide) {
          btnSide.innerHTML = '⏹ Stop k-NN';
          btnSide.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
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
          btnTop.style.background = 'rgba(34, 197, 94, 0.2)';
          btnTop.style.color = '#4ade80';
          btnTop.style.borderColor = 'rgba(34, 197, 94, 0.5)';
          btnTop.title = 'Compute k-Nearest Neighbors';
        }
        if (btnSide) {
          btnSide.innerHTML = '▶ Compute k-NN';
          btnSide.style.background = 'linear-gradient(135deg, #10b981, #059669)';
          btnSide.title = 'Run Out-of-Core k-NN Solver';
        }
        if (badgeTop) {
          badgeTop.textContent = `k=${knnK} • ${knnDirection} • dt≥${knnDtmin}`;
          badgeTop.style.color = '#c084fc';
          badgeTop.style.borderColor = 'rgba(192, 132, 252, 0.3)';
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

      const pts = (typeof benchmarkDataset !== 'undefined' && benchmarkDataset && benchmarkDataset.length > 0) ?
                  benchmarkDataset : (typeof pastSamples !== 'undefined' ? pastSamples : []);
      if (!pts || pts.length === 0) {
        showToast('⚠️ No dataset staged to run k-NN. Stage or generate a dataset first.');
        return;
      }

      enableKnn = true;
      showKnnLines = true;
      syncControlDependencies();
      updateCliCommand();

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

      try {
        if (engineMode === 'cli' && DesktopBridge.isNativeSupported()) {
          const selCli = document.getElementById('selectCliDataset');
          let datasetName = selCli ? selCli.value : '';
          if (!datasetName) {
            datasetName = `${currentBenchmark}.txt`;
          }
          const datasetBase = datasetName.replace(/\.(txt|csv|fits|dat|mp4|fits\.fz)$/i, '');
          const clusterDir = `${datasetBase}.clusterdat`;

          // 1. Ensure staged coordinates exist in workspace file
          const pts = (typeof benchmarkDataset !== 'undefined' && benchmarkDataset && benchmarkDataset.length > 0) ?
                      benchmarkDataset : (typeof pastSamples !== 'undefined' ? pastSamples : []);
          if (pts && pts.length > 0) {
            await DesktopBridge.stageDatasetFile(datasetBase, pts).catch(() => {});
          }

          // 2. Check if clusterdat exists; if not and we have in-memory clusters, export them
          const filesInWorkspace = await DesktopBridge.listFiles().catch(() => []);
          const hasClusterDir = filesInWorkspace.some(f => (f.name === clusterDir || f.name === `${datasetName}.clusterdat`) && f.is_dir);
          if (!hasClusterDir && typeof clusters !== 'undefined' && clusters.length > 0) {
            let centroidsText = `# GRIC Cluster Centroids\n# ID X Y Z MEMBERS\n`;
            clusters.forEach(c => {
              centroidsText += `${c.id} ${c.x.toFixed(6)} ${c.y.toFixed(6)} ${(c.z || 0).toFixed(6)} ${c.members || 1}\n`;
            });
            let dccText = `# GRIC Cluster-to-Cluster Distance Matrix D_cc\n`;
            if (typeof dcc !== 'undefined' && dcc && dcc.length > 0) {
              dcc.forEach(row => {
                dccText += row.map(v => Number(v).toFixed(6)).join(' ') + '\n';
              });
            }
            let memText = `# Frame Membership Assignments\n`;
            if (typeof pastSamples !== 'undefined' && pastSamples && pastSamples.length > 0) {
              for (let i = 0; i < pastSamples.length; i++) {
                const p = pastSamples[i];
                const cId = p.clusterId >= 0 ? p.clusterId : (typeof clustersAssigned !== 'undefined' && clustersAssigned ? clustersAssigned[i] || 0 : 0);
                memText += `${i} ${cId} 0.000000\n`;
              }
            }
            let exportFiles = {
              'anchors.txt': centroidsText,
              'dcc.txt': dccText,
              'frame_membership.txt': memText
            };
            if (typeof dccMin !== 'undefined' && dccMin && dccMin.length > 0) {
              let dccMinText = `# GRIC Cluster-to-Cluster Lower Bounds Matrix D_cc,min\n`;
              for (let i = 0; i < clusters.length; i++) {
                for (let j = 0; j < clusters.length; j++) {
                  const val = (dccMin[i] && dccMin[i][j]) ? dccMin[i][j] : 0.0;
                  if (val > 0.0) {
                    dccMinText += `${i} ${j} ${val.toFixed(6)}\n`;
                  }
                }
              }
              exportFiles['dccmin.txt'] = dccMinText;
            }
            await DesktopBridge.exportClusterDat(datasetBase, exportFiles).catch(() => {});
          }

          const args = [
            datasetName,
            clusterDir,
            '-k', String(knnK),
            '-dtmin', String(knnDtmin)
          ];
          if (knnDirection === 'past') args.push('-past');
          if (knnDirection === 'future') args.push('-future');
          if (knnEpsilon > 0) args.push('-eps', String(knnEpsilon));
          if (knnRlim > 0) args.push('-rlim', String(knnRlim));
          if (typeof knnMvp !== 'undefined' && knnMvp) args.push('-multipivot');
          args.push('-progress');
          args.push('-txt');

          const consoleEl = document.getElementById('cliConsoleLog');
          if (consoleEl) {
            consoleEl.textContent = `🚀 Dispatched in tmux session: gric_cli\n` +
              `🖥️ Attach live: tmux attach -t gric_cli\n` +
              `📄 Log stream: /tmp/gric_latest.log\n` +
              `⚙️ Command: gric-knn ${args.join(' ')}\n` +
              `─────────────────────────────────────────────────────────────\n`;
          }

          showToast(`💻 Running native gric-knn (k=${knnK})...`);

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
              },
              onTelemetry: () => {},
              onFinish: (res) => {
                finalExitCode = res?.exitCode ?? 0;
                if (finalExitCode === 0) {
                  showToast(`✅ Native gric-knn finished successfully!`);
                } else {
                  showToast(`⚠️ Native gric-knn finished (Exit: ${finalExitCode})`);
                }
                resolve();
              }
            }).catch((err) => {
              showToast(`⚠️ Failed to start gric-knn: ${err.message}`);
              if (consoleEl) {
                consoleEl.textContent += `\n❌ Error: ${err.message}\n`;
              }
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
          if (!nativeData) {
            const N = parsedTelem.totalQueries || pts.length;
            nativeData = {
              totalFrames: N,
              k: knnK,
              indices: new Int32Array(N * knnK).fill(-1),
              distances: new Float64Array(N * knnK).fill(0.0)
            };
          }
          nativeData.telemetry = parsedTelem;

          knnResults = nativeData;
          if (typeof renderKnnTrace === 'function') {
            renderKnnTrace();
          }
          draw();
        } else {
          // WASM Execution
          if (typeof GricWasm === 'undefined' || !GricWasm.isReady()) {
            showToast('⚠️ WASM engine not ready for k-NN');
            isKnnComputing = false;
            updateKnnButtonUI(false);
            return;
          }

          const config = {
            k: knnK,
            dtmin: knnDtmin,
            direction: knnDirection,
            epsilon: knnEpsilon,
            rlim: knnRlim,
            multiPivot: (typeof knnMvp !== 'undefined' && knnMvp)
          };

          const t0 = performance.now();
          const results = GricWasm.runKnn(config, pts);
          const elapsed = (performance.now() - t0).toFixed(1);

          if (!knnAbortRequested && results) {
            knnResults = results;
            if (typeof renderKnnTrace === 'function') {
              renderKnnTrace();
            }
            draw();
            showToast(`⚡ k-NN computed in ${elapsed} ms (k=${knnK}, ${pts.length.toLocaleString()} points)`);
          }
        }
      } catch (err) {
        console.error('[k-NN] Computation error:', err);
        showToast('⚠️ k-NN computation failed');
      } finally {
        isKnnComputing = false;
        knnAbortRequested = false;
        updateKnnButtonUI(false);
      }
    }

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

    const sliderKnnK = document.getElementById('sliderKnnK');
    const inputKnnK = document.getElementById('inputKnnK');
    if (sliderKnnK) {
      sliderKnnK.addEventListener('input', (e) => {
        knnK = parseInt(e.target.value, 10);
        if (inputKnnK) inputKnnK.value = knnK;
        updateCliCommand();
        draw();
      });
    }
    if (inputKnnK) {
      inputKnnK.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v >= 1) {
          knnK = v;
          if (sliderKnnK) sliderKnnK.value = Math.min(50, v);
          updateCliCommand();
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
        if (enableKnn) executeKnnComputation();
      });
    }


    // =========================================================================
    //  SIDEBAR PANELS RESIZING & COLLAPSE CONTROLLER (9 PANELS & 8 RESIZERS)
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
      { id: 'cardKnnTrace', btnId: 'btnCollapseKnnTrace', defaultFlex: 1.0, savedFlex: 1.0 }
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
      const is3D = (currentDim === 3);
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

      listEl.innerHTML = '';
      structures.forEach(s => {
        const item = document.createElement('div');
        item.className = 'data-struct-item';

        const sizeStr = (s.size > 1048576)
          ? `${(s.size / 1048576).toFixed(2)} MB`
          : `${(s.size / 1024).toFixed(1)} KB`;

        const badgeBg = s.ready ? 'rgba(56, 189, 248, 0.15)' : 'rgba(148, 163, 184, 0.1)';
        const badgeColor = s.ready ? '#38bdf8' : '#94a3b8';

        item.innerHTML = `
          <div class="data-struct-info">
            <div class="data-struct-title">
              <span>${s.icon}</span>
              <span style="color: ${s.ready ? '#f1f5f9' : '#64748b'};">${s.filename}</span>
              <span class="badge-pill"
                    style="background: ${badgeBg}; color: ${badgeColor};
                           font-size: 0.65rem; padding: 1px 6px;">
                ${s.badge}
              </span>
              <span style="font-size: 0.65rem; color: var(--text-muted); margin-left: auto;">
                ${sizeStr}
              </span>
            </div>
            <div class="data-struct-desc" title="${s.desc}">${s.desc}</div>
          </div>
          <div class="data-struct-actions">
            <button class="btn-action btn-view-struct"
                    style="padding: 2px 6px; font-size: 0.68rem;"
                    title="View file content" ${s.ready ? '' : 'disabled'}>👁️</button>
            <button class="btn-action btn-dl-struct"
                    style="padding: 2px 6px; font-size: 0.68rem;"
                    title="Download file to disk" ${s.ready ? '' : 'disabled'}>💾</button>
            <button class="btn-action btn-copy-struct"
                    style="padding: 2px 6px; font-size: 0.68rem;"
                    title="Copy file text to clipboard" ${s.ready ? '' : 'disabled'}>📋</button>
          </div>
        `;

        const btnView = item.querySelector('.btn-view-struct');
        if (btnView) {
          btnView.addEventListener('click', (e) => {
            e.stopPropagation();
            openFileViewerModal(s.filename, s.category, s.badge, s.size, s.content);
          });
        }
        const btnDl = item.querySelector('.btn-dl-struct');
        if (btnDl) {
          btnDl.addEventListener('click', (e) => {
            e.stopPropagation();
            DataManager.downloadTextFile(s.filename, s.content);
            showToast(`💾 Downloaded ${s.filename}`);
          });
        }
        const btnCopy = item.querySelector('.btn-copy-struct');
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

        listEl.appendChild(item);
      });
    }
    window.renderDataStructuresUI = renderDataStructuresUI;

    function renderWorkspaceFilesTree() {
      const treeEl = document.getElementById('workspaceFilesTree');
      const headerEl = document.getElementById('lblWorkspaceTreeHeader');
      if (!treeEl) return;

      treeEl.innerHTML = '';

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
            <div style="font-size: 1.1rem; margin-bottom: 4px;">⚡ Web Browser Sandbox</div>
            <div>Working in client-side memory. Click <b>"Save All"</b> or <b>"ZIP"</b>
                 to export results to disk.</div>
            <div style="margin-top: 6px; font-size: 0.68rem; color: #38bdf8;">
              Tip: Click <b>"Open"</b> in the top Workspace bar to link a local folder directly.
            </div>
          </div>
        `;
        return;
      }

      if (workspaceFiles.length === 0) {
        treeEl.innerHTML = `
          <div style="padding: 8px; color: var(--text-muted); text-align: center;">
            (Directory is empty)
          </div>
        `;
        return;
      }

      workspaceFiles.forEach(f => {
        const item = document.createElement('div');
        const isDir = f.is_dir || f.isDir;
        item.className = `ws-tree-item ${isDir ? 'is-dir' : ''}`;

        const icon = isDir ? '📁' : '📄';
        const sizeStr = !isDir ? `(${(f.size / 1024).toFixed(1)} KB)` : '';

        item.innerHTML = `
          <div style="display: flex; align-items: center; gap: 6px; overflow: hidden;
                      text-overflow: ellipsis; white-space: nowrap; flex: 1;">
            <span>${icon}</span>
            <span style="font-weight: ${isDir ? '700' : '400'};">${f.name}</span>
            <span style="font-size: 0.65rem; color: var(--text-muted);">${sizeStr}</span>
          </div>
          <div style="display: flex; gap: 4px; flex-shrink: 0;">
            ${isDir && f.name.includes('cluster')
              ? `<button class="btn-action btn-load-cluster"
                         style="padding: 1px 5px; font-size: 0.65rem; color: #4ade80;">
                   Load
                 </button>`
              : ''}
            ${!isDir
              ? `<button class="btn-action btn-view-file"
                         style="padding: 1px 5px; font-size: 0.65rem;">
                   View
                 </button>`
              : ''}
          </div>
        `;

        const btnLoad = item.querySelector('.btn-load-cluster');
        if (btnLoad) {
          btnLoad.addEventListener('click', async (e) => {
            e.stopPropagation();
            await loadClusterResults(f.name);
          });
        }

        const btnView = item.querySelector('.btn-view-file');
        if (btnView) {
          btnView.addEventListener('click', async (e) => {
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
        }

        treeEl.appendChild(item);
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
        if (btnOpenFolder) btnOpenFolder.style.display = 'none';
        if (cliNotice) cliNotice.style.display = 'none';
        if (cliControls) cliControls.style.display = 'flex';

        await refreshWorkspaceFiles();
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

    async function setEngineMode(mode) {
      if (mode === 'cli' && !DesktopBridge.isNativeSupported()) {
        if (DesktopBridge.isMobileDevice()) {
          showToast('📱 Cell Phone: Native CLI is disabled (In-Browser WASM active)');
        } else {
          showToast('🌐 Web Mode: Native CLI requires a local desktop gric-server');
        }
        mode = 'wasm';
      }

      engineMode = mode;
      updateEngineModeUI();

      if (mode === 'cli') {
        const cardCli = document.getElementById('cardCli');
        if (cardCli && cardCli.classList.contains('collapsed')) {
          togglePanelCollapse('cardCli');
        }
        updateCliCommand();

        if (DesktopBridge.isAvailable()) {
          await DesktopBridge.initCliSession();
        }
        showToast('💻 Native CLI mode active (tmux session "gric_cli" ready)');
      } else {
        if (DesktopBridge.isAvailable()) {
          await DesktopBridge.stopCliSession();
        }
        showToast('⚡ Switched to In-Browser WebAssembly (WASM)');
      }
    }

    function setupCliRunnerListeners() {
      const btnRun = document.getElementById('btnRunCli');
      const btnKill = document.getElementById('btnKillCli');
      const btnClear = document.getElementById('btnClearCliConsole');
      const btnLoadManual = document.getElementById('btnLoadClusterDatManual');
      const chkAutoLoad = document.getElementById('chkAutoLoadResults');

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

      if (dataset.startsWith('shm:')) {
        const streamName = dataset.substring(4);
        isStreamInput = true;
        args = [rlim.toFixed(3), streamName, '-stream'];
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
            if (Array.isArray(pt)) {
              content += pt.map(v => Number(v).toFixed(6)).join(' ') + '\n';
            } else if (pt && typeof pt === 'object') {
              if (currentDim === 3) {
                content += `${Number(pt.x || 0).toFixed(6)} ${Number(pt.y || 0).toFixed(6)} ${Number(pt.z || 0).toFixed(6)}\n`;
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
      const btnKill = document.getElementById('btnKillCli');
      const btnPlay = document.getElementById('btnPlay');
      const badgeStatus = document.getElementById('badgeCliStatus');
      const consoleEl = document.getElementById('cliConsoleLog');

      if (btnRun) btnRun.disabled = true;
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
        consoleEl.textContent = `🚀 Dispatched in tmux session: gric_cli\n` +
          `🖥️ Attach live: tmux attach -t gric_cli\n` +
          `📄 Log stream: /tmp/gric_latest.log\n` +
          `⚙️ Command: gric-cluster ${args.join(' ')}\n` +
          `─────────────────────────────────────────────────────────────\n`;
      }
      showToast('🚀 Native CLI launched in tmux session "gric_cli" (tmux attach -t gric_cli)');

      const tStart = performance.now();
      isCliRunning = true;

      try {
        await DesktopBridge.runCliJob({
          cmd: 'gric-cluster',
          args: args,
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
            if (btnKill) btnKill.disabled = true;
            if (btnPlay) {
              btnPlay.innerHTML = '▶ Run gric-cluster';
              btnPlay.disabled = false;
            }

            if (res.exitCode === 0) {
              if (badgeStatus) {
                badgeStatus.textContent = `Done (${elapsed}s)`;
                badgeStatus.style.background = 'rgba(56, 189, 248, 0.2)';
                badgeStatus.style.color = '#38bdf8';
              }
              showToast(`✅ gric-cluster completed in ${elapsed}s`);

              if (!isStreamInput) {
                const baseName = dataset.replace(/\.[^/.]+$/, '');
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
            x: Array.isArray(p) ? p[0] : (p.x || 0),
            y: Array.isArray(p) ? p[1] : (p.y || 0),
            z: currentDim === 3 ? (Array.isArray(p) ? (p[2] || 0) : (p.z || 0)) : 0,
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
                  pts.push({
                    x: parseFloat(tokens[0]),
                    y: parseFloat(tokens[1]),
                    z: tokens.length >= 3 ? parseFloat(tokens[2]) : 0,
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

        updateUI();
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
    loadSelectedBenchmark();
    updateZoomBadge();
    setExplainMode(false);
    updateCliCommand();
    initWorkspaceAndEngine();

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
      }
    });


    setTimeout(() => { updateTMCanvasDimensions(); resizeCanvas(); }, 50);
    setTimeout(() => { updateTMCanvasDimensions(); resizeCanvas(); }, 250);
