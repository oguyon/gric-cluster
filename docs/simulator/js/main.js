/**
 * GRIC Simulator - main.js
 * Part of the GRIC Interactive Algorithm Simulator
 */

//  WASM ENGINE TOGGLE
    // =========================================================================

    /**
     * Returns true if the WASM engine is effectively active
     * for the current frame pipeline (preference ON, module
     * loaded, and no overriding mode like Explain or Tiles).
     */
    function isWasmEffective() {
      return useWasm
        && wasmSessionActive
        && GricWasm.isReady()
        && !isExplainMode
        && !useTiles;
    }

    /**
     * Update all WASM-related UI elements to reflect the
     * effective code path, including overrides from Explain
     * mode and Tile mode.
     */
    function updateWasmBadge() {
      const btn = document.getElementById('btnWasm');
      const badge = document.getElementById('badgeWasmStatus');
      const label = document.getElementById('statEngineBackend');
      const effective = isWasmEffective();
      const loaded = GricWasm.isLoaded();

      // Toolbar button
      if (btn) {
        if (effective) {
          btn.classList.add('toggle-active');
          btn.innerText = '⚡ WASM';
        } else if (useWasm && loaded && (isExplainMode || useTiles)) {
          btn.classList.remove('toggle-active');
          btn.innerText = '⚡ WASM (overridden)';
        } else {
          btn.classList.remove('toggle-active');
          btn.innerText = '⚡ WASM';
        }
      }

      // Resource tracker badge
      if (badge) {
        badge.innerText = effective ? 'WASM' : 'JS';
        badge.style.background = effective
          ? 'rgba(74, 222, 128, 0.15)'
          : 'rgba(100,100,100,0.2)';
        badge.style.color = effective ? '#4ade80' : '#888';
      }

      // Backend label
      if (label) {
        if (effective) {
          label.innerText = 'C/WebAssembly (SIMD)';
        } else if (useWasm && (isExplainMode || useTiles)) {
          const reason = isExplainMode ? 'Explain' : 'Tiles';
          label.innerText = 'JS (WASM paused: ' + reason + ')';
        } else {
          label.innerText = 'JavaScript';
        }
      }
    }

    function toggleWasmEngine() {
      if (!GricWasm.isLoaded()) {
        showToast('⚠️ WASM module not loaded');
        return;
      }
      useWasm = !useWasm;
      if (useWasm) {
        if (isExplainMode) {
          isExplainMode = false;
          const btnExp = document.getElementById('btnExplain');
          if (btnExp) btnExp.classList.remove('toggle-active');
        }
        const params = GricWasm.buildParamsFromState();
        wasmSessionActive = GricWasm.init(params);
        if (typeof GricWasmWorker !== 'undefined') {
          GricWasmWorker.startSession(params);
        }
        resetSimulation();
        showToast('⚡ Engine: C/WASM');
      } else {
        GricWasm.destroy();
        if (typeof GricWasmWorker !== 'undefined') {
          GricWasmWorker.reset();
        }
        wasmSessionActive = false;
        resetSimulation();
        showToast('⚡ Engine: JavaScript');
      }
      updateWasmBadge();
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

    function runBatchInstant() {
      if (currentBenchmark === "stream" || currentBenchmark === "3Dlorenz") {
        for (let i = 0; i < 1000; i++) stepNextFrame(true);
        // Sync WASM state after batch
        if (useWasm && wasmSessionActive && GricWasm.isReady()) {
          const snapshot = GricWasm.syncState();
          if (snapshot) GricWasm.applyToJsState(snapshot);
        }
        updateUI();
        draw();
        pauseSimulation();
        return;
      }
      while (hasMoreFrames()) {
        if (currentFrameIdx >= benchmarkDataset.length) {
          if (loopCount === 0 || currentLoop < loopCount) {
            currentLoop++;
            currentFrameIdx = 0;
          } else {
            break;
          }
        }
        if (loopCount === 0 && totalFrames >= 100000) {
          showToast("Infinite batch capped at 100,000 frames");
          break;
        }
        const rawPt = benchmarkDataset[currentFrameIdx++];
        const pt = applyNoiseToPoint(rawPt.x, rawPt.y, rawPt.z || 0.0);
        clusterFrame(pt.x, pt.y, pt.z || 0.0, true);
      }
      // Sync WASM state after batch
      if (useWasm && wasmSessionActive && GricWasm.isReady()) {
        const snapshot = GricWasm.syncState();
        if (snapshot) GricWasm.applyToJsState(snapshot);
      }
      updateUI();
      draw();
      pauseSimulation();
    }

    function startSimulation() {
      if (isAddPointMode) setAddPointMode(false);

      isRunning = true;
      sessionStartTime = performance.now();
      sessionStartFrames = totalFrames;
      sessionElapsedMs = 0;
      sessionIsActive = true;
      sessionAvgFps = 0.0;

      const btn = document.getElementById('btnPlay');
      btn.innerText = "❚❚ Pause";
      btn.classList.add('danger');
      btn.classList.remove('primary');

      // Start independent asynchronous 60 FPS display render loop
      startDisplayLoop();

      if (playSpeed === 0 && benchmarkDataset.length > 0) {
        // Instant Batch Mode: Compute as fast as possible in non-blocking time-slices
        function batchChunk() {
          if (!isRunning) return;
          const sliceStart = performance.now();
          while (hasMoreFrames() && (performance.now() - sliceStart < 14)) {
            if (currentFrameIdx >= benchmarkDataset.length) {
              if (loopCount === 0 || currentLoop < loopCount) {
                currentLoop++;
                currentFrameIdx = 0;
              } else {
                break;
              }
            }
            if (loopCount === 0 && totalFrames >= 100000) {
              showToast("Infinite batch capped at 100,000 frames");
              break;
            }
            const rawPt = benchmarkDataset[currentFrameIdx++];
            const pt = applyNoiseToPoint(rawPt.x, rawPt.y, rawPt.z || 0.0);
            clusterFrame(pt.x, pt.y, pt.z || 0.0, true);
          }
          if (!hasMoreFrames() || (loopCount > 0 && currentLoop >= loopCount && currentFrameIdx >= benchmarkDataset.length)) {
            pauseSimulation();
            return;
          }
          playTimer = setTimeout(batchChunk, 0);
        }
        playTimer = setTimeout(batchChunk, 0);
        return;
      }

      if (playSpeed <= 0) {
        // "⚡ As fast as possible" Mode:
        // Pure mathematical compute loop runs continuously without waiting for display
        function pumpCompute() {
          if (!isRunning) return;
          const sliceStart = performance.now();
          while (hasMoreFrames() && (performance.now() - sliceStart < 12)) {
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
      btn.innerText = "► Run / Play";
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
        // Explain mode requires the JS narrative pipeline -> explicitly disable WASM
        if (useWasm) {
          useWasm = false;
          wasmSessionActive = false;
          if (typeof GricWasm !== 'undefined') {
            GricWasm.destroy();
          }
          if (typeof GricWasmWorker !== 'undefined') {
            GricWasmWorker.reset();
          }
          showToast('💬 Explain active: switched to JavaScript engine');
        }
      } else {
        btn.classList.remove('toggle-active');
        currentExplanation = [];
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

      benchmarkDataset = rawPoints.map(p => ({
        x: ((p.x - midX) / maxSpan) * 1.76,
        y: ((p.y - midY) / maxSpan) * 1.76,
        z: detected3D ? (((p.z - midZ) / maxSpan) * 1.76) : 0.0
      }));

      currentDim = detected3D ? 3 : 2;
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
      const rect = canvas.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      const W = rect.width;
      const H = rect.height;

      if (currentDim === 2 || maximizedQuad !== null) {
        return maximizedQuad !== null ? maximizedQuad : 2;
      }

      if (px < W / 2 && py < H / 2) return 0; // Along X
      if (px >= W / 2 && py < H / 2) return 1; // Along Y
      if (px < W / 2 && py >= H / 2) return 2; // Along Z
      return 3; // Custom 3D
    }

    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const W = rect.width;
      const H = rect.height;
      const qIdx = getQuadrantAt(e.clientX, e.clientY);
      const qRect = getQuadRect(qIdx, W, H);

      // Check if clicking Maximize / Restore Icon in top-right of quadrant
      if (currentDim === 3 && px >= qRect.x + qRect.w - 30 && px <= qRect.x + qRect.w - 4 && py >= qRect.y && py <= qRect.y + 24) {
        maximizedQuad = (maximizedQuad === qIdx) ? null : qIdx;
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
        const rect = canvas.getBoundingClientRect();
        const qRect = getQuadRect(activeDragQuad, rect.width, rect.height);
        const scale = getQuadScale(activeDragQuad, qRect);

        const v = quadViews[activeDragQuad];
        v.panX -= dx / scale;
        v.panY += dy / scale;
        draw();
      }
    });

    window.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        dragMode = null;
        canvas.classList.remove('grabbing');
      }
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const qIdx = getQuadrantAt(e.clientX, e.clientY);
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;

      const v = quadViews[qIdx];
      if (v) {
        v.zoom = Math.max(0.2, Math.min(25.0, (v.zoom || 1.0) * zoomFactor));
        if (qIdx === 3) {
          orbitCamera.zoom = v.zoom;
        }
      }

      updateZoomBadge();
      draw();
    }, { passive: false });

    canvas.addEventListener('dblclick', (e) => {
      if (isAddPointMode) return;
      const qIdx = getQuadrantAt(e.clientX, e.clientY);
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
        updateUI();
      });
    }

    document.getElementById('btnPlay').addEventListener('click', () => {
      if (isRunning) pauseSimulation();
      else startSimulation();
    });

    document.getElementById('btnStep').addEventListener('click', () => {
      if (isRunning) pauseSimulation();
      stepNextFrame(true);
      updateUI();
      draw();
    });

    document.getElementById('btnAddPoint').addEventListener('click', () => {
      setAddPointMode(!isAddPointMode);
    });

    document.getElementById('btnExplain').addEventListener('click', () => {
      setExplainMode(!isExplainMode);
    });

    document.getElementById('btnWasm').addEventListener('click', () => {
      toggleWasmEngine();
    });

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

      if (e.key === '[' || e.key === 'ArrowLeft') {
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
      } else if (e.key === 'l' || e.key === 'L') {
        returnToLiveStream();
      }
    });

    document.getElementById('btnReset').addEventListener('click', () => {
      pauseSimulation();
      loadSelectedBenchmark();
    });

    document.getElementById('btnResetView').addEventListener('click', resetView);

    const sliderRlim = document.getElementById('sliderRlim');
    sliderRlim.addEventListener('input', (e) => {
      rlim = parseFloat(e.target.value);
      document.getElementById('lblRlim').innerText = rlim.toFixed(3);
      draw();
    });

    const sliderFocus = document.getElementById('sliderFocus');
    sliderFocus.addEventListener('input', (e) => {
      visualFocus = parseInt(e.target.value);
      const lbl = document.getElementById('lblFocus');
      if (lbl) {
        if (visualFocus === 0) {
          lbl.innerText = "Points Only (0%)";
        } else if (visualFocus < 45) {
          lbl.innerText = `Points Emphasis (${visualFocus}%)`;
        } else if (visualFocus <= 55) {
          lbl.innerText = `Balanced (${visualFocus}%)`;
        } else if (visualFocus === 100) {
          lbl.innerText = "Clusters Only (100%)";
        } else {
          lbl.innerText = `Clusters Emphasis (${visualFocus}%)`;
        }
      }
      draw();
    });

    const sliderPointSize = document.getElementById('sliderPointSize');
    if (sliderPointSize) {
      sliderPointSize.addEventListener('input', (e) => {
        samplePointSize = parseFloat(e.target.value);
        const lbl = document.getElementById('lblPointSize');
        if (lbl) {
          lbl.innerText = `${samplePointSize.toFixed(2).replace(/\.?0+$/, '')} px`;
        }
        draw();
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
    sliderNoiseSigma.addEventListener('input', (e) => {
      noiseSigma = parseFloat(e.target.value);
      const lbl = document.getElementById('lblNoiseSigma');
      if (lbl) {
        lbl.innerText = noiseSigma <= 1e-6 ? "0.000 (Off)" : noiseSigma.toFixed(3);
      }
      syncControlDependencies();
    });

    const sliderNoiseTrunc = document.getElementById('sliderNoiseTrunc');
    sliderNoiseTrunc.addEventListener('input', (e) => {
      noiseTruncLimit = parseFloat(e.target.value);
      const lbl = document.getElementById('lblNoiseTrunc');
      if (lbl) {
        lbl.innerText = noiseTruncLimit.toFixed(3);
      }
    });

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
      resetSimulation();
      if (currentBenchmark !== "stream" && currentBenchmark !== "3Dlorenz" && currentBenchmark !== "custom") {
        benchmarkDataset = generateBenchmark(currentBenchmark, 1000);
      }
      updateWasmBadge();
      updateUI();
      draw();
    });

    // Simulator Settings 3-Tab Switching
    const configTabs = [
      { id: 'tabCfgAlgo', panel: 'cfgAlgoPanel' },
      { id: 'tabCfgInput', panel: 'cfgInputPanel' },
      { id: 'tabCfgDisplay', panel: 'cfgDisplayPanel' }
    ];

    configTabs.forEach(t => {
      const tabBtn = document.getElementById(t.id);
      if (tabBtn) {
        tabBtn.addEventListener('click', () => {
          configTabs.forEach(other => {
            const btnEl = document.getElementById(other.id);
            const panelEl = document.getElementById(other.panel);
            if (btnEl) btnEl.classList.toggle('active', other.id === t.id);
            if (panelEl) panelEl.style.display = (other.id === t.id) ? 'block' : 'none';
          });
        });
      }
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

    // Sync Side Panel Input Stream Selectors
    const selBenchSide = document.getElementById('selectBenchmarkSide');
    if (selBenchSide) {
      selBenchSide.addEventListener('change', (e) => {
        document.getElementById('selectBenchmark').value = e.target.value;
        loadSelectedBenchmark();
        resetView();
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
        updateUI();
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
    sliderMaxcl.addEventListener('input', (e) => {
      const idx = parseInt(e.target.value, 10);
      maxcl = idx === 0 ? 0 : (1 << (idx - 1));
      if (maxcl === 0) {
        document.getElementById('lblMaxcl').innerText = "Unlimited (0)";
      } else if (maxcl === 1) {
        document.getElementById('lblMaxcl').innerText = "1 cluster";
      } else if (maxcl >= 1024) {
        const k = maxcl / 1024;
        document.getElementById('lblMaxcl').innerText = `${maxcl.toLocaleString()} (${k}k)`;
      } else {
        document.getElementById('lblMaxcl').innerText = `${maxcl} clusters`;
      }
      syncControlDependencies();
    });

    ['stratStop', 'stratDiscard', 'stratMerge'].forEach(id => {
      document.getElementById(id).addEventListener('click', () => {
        ['stratStop', 'stratDiscard', 'stratMerge'].forEach(other => document.getElementById(other).classList.remove('active'));
        document.getElementById(id).classList.add('active');
        maxclStrategy = id.replace('strat', '').toLowerCase();
        syncControlDependencies();
      });
    });

    const sliderDiscardFrac = document.getElementById('sliderDiscardFrac');
    sliderDiscardFrac.addEventListener('input', (e) => {
      discardFraction = parseFloat(e.target.value);
      document.getElementById('lblDiscardFrac').innerText = discardFraction.toFixed(2);
    });

    // Prior & Acceleration Tuning (-tm, -pred, -maxvis)
    const sliderTmMix = document.getElementById('sliderTmMix');
    sliderTmMix.addEventListener('input', (e) => {
      tmMixingCoeff = parseFloat(e.target.value);
      document.getElementById('lblTmMix').innerText = tmMixingCoeff.toFixed(2);
    });

    const sliderPredHorizon = document.getElementById('sliderPredHorizon');
    sliderPredHorizon.addEventListener('input', (e) => {
      predHorizon = parseInt(e.target.value);
      document.getElementById('lblPredHorizon').innerText = `${predHorizon} frames`;
    });

    const sliderMaxVis = document.getElementById('sliderMaxVis');
    sliderMaxVis.addEventListener('input', (e) => {
      maxVisitors = parseInt(e.target.value);
      document.getElementById('lblMaxVis').innerText = `${maxVisitors} frames`;
    });

    // Entropy & Soft Bayesian Likelihood (-entropy_first_gate, -entropy_gate, -entropy_fast, -soft_bayesian)
    const sliderEntropyFirstGate = document.getElementById('sliderEntropyFirstGate');
    sliderEntropyFirstGate.addEventListener('input', (e) => {
      entropyFirstGate = parseFloat(e.target.value);
      document.getElementById('lblEntropyFirstGate').innerText = `${entropyFirstGate.toFixed(2)} bits`;
    });

    const sliderEntropyGate = document.getElementById('sliderEntropyGate');
    sliderEntropyGate.addEventListener('input', (e) => {
      entropyGate = parseFloat(e.target.value);
      document.getElementById('lblEntropyGate').innerText = `${entropyGate.toFixed(2)} bits`;
    });

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
    sliderLeaderCutoff.addEventListener('input', (e) => {
      entropyLeaderCutoff = parseFloat(e.target.value);
      document.getElementById('lblLeaderCutoff').innerText = entropyLeaderCutoff.toFixed(2);
    });

    document.getElementById('optSoftBayesian').addEventListener('click', () => {
      useSoftBayesian = !useSoftBayesian;
      document.getElementById('optSoftBayesian').classList.toggle('active', useSoftBayesian);
      syncControlDependencies();
    });

    const sliderBayesSigma = document.getElementById('sliderBayesSigma');
    sliderBayesSigma.addEventListener('input', (e) => {
      softBayesianSigmaCoeff = parseFloat(e.target.value);
      document.getElementById('lblBayesSigma').innerText = `${softBayesianSigmaCoeff.toFixed(1)} × rlim`;
    });

    // Cross-Tile Subspace Prior Transfer (-xtile, -xtile_decay)
    document.getElementById('optXTile').addEventListener('click', () => {
      useXTile = !useXTile;
      document.getElementById('optXTile').classList.toggle('active', useXTile);
      syncControlDependencies();
    });

    const sliderXTileDecay = document.getElementById('sliderXTileDecay');
    sliderXTileDecay.addEventListener('input', (e) => {
      xtileDecay = parseFloat(e.target.value);
      document.getElementById('lblXTileDecay').innerText = xtileDecay.toFixed(2);
    });

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
      if (colTmMix && sliderTmMixEl) {
        colTmMix.classList.toggle('disabled', !useTM);
        sliderTmMixEl.disabled = !useTM;
      }

      const colPredHorizon = document.getElementById('colPredHorizon');
      const sliderPredHorizonEl = document.getElementById('sliderPredHorizon');
      if (colPredHorizon && sliderPredHorizonEl) {
        colPredHorizon.classList.toggle('disabled', !usePred);
        sliderPredHorizonEl.disabled = !usePred;
      }

      const colMaxVis = document.getElementById('colMaxVis');
      const sliderMaxVisEl = document.getElementById('sliderMaxVis');
      if (colMaxVis && sliderMaxVisEl) {
        colMaxVis.classList.toggle('disabled', !useGprob);
        sliderMaxVisEl.disabled = !useGprob;
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
      if (rowNoiseTrunc && sliderNoiseTruncEl) {
        rowNoiseTrunc.classList.toggle('disabled', !hasNoise);
        sliderNoiseTruncEl.disabled = !hasNoise;
      }

      // 5. 3D Mode Camera Presets
      const is3D = (currentDim === 3);
      const card3DPresetsSide = document.getElementById('card3DPresetsSide');
      if (card3DPresetsSide) {
        card3DPresetsSide.classList.toggle('disabled', !is3D);
      }
    }

    window.syncControlDependencies = syncControlDependencies;
    syncControlDependencies();

    // =========================================================================
    //  SIDEBAR PANELS RESIZING & COLLAPSE CONTROLLER
    // =========================================================================
    const panelConfigs = [
      { id: 'cardSettings', btnId: 'btnCollapseSettings', defaultFlex: 1.1, savedFlex: 1.1 },
      { id: 'cardResources', btnId: 'btnCollapseResources', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardTrace', btnId: 'btnCollapseTrace', defaultFlex: 1.2, savedFlex: 1.2 }
    ];

    function updateResizersVisibility() {
      const isSettingsExp = !document.getElementById('cardSettings').classList.contains('collapsed');
      const isResourcesExp = !document.getElementById('cardResources').classList.contains('collapsed');
      const isTraceExp = !document.getElementById('cardTrace').classList.contains('collapsed');

      const resizer1 = document.getElementById('resizer1');
      const resizer2 = document.getElementById('resizer2');

      // Resizer 1 is between Settings and whatever is expanded below it
      if (resizer1) {
        const show1 = isSettingsExp && (isResourcesExp || isTraceExp);
        resizer1.classList.toggle('hidden', !show1);
      }

      // Resizer 2 is between Resources (or Settings if Resources is collapsed) and Trace
      if (resizer2) {
        const show2 = isTraceExp && (isResourcesExp || isSettingsExp);
        resizer2.classList.toggle('hidden', !show2);
      }
    }

    function togglePanelCollapse(cardId) {
      const card = document.getElementById(cardId);
      if (!card) return;

      const isCurrentlyCollapsed = card.classList.contains('collapsed');
      const shouldCollapse = !isCurrentlyCollapsed;

      card.classList.toggle('collapsed', shouldCollapse);
      card.classList.toggle('expanded', !shouldCollapse);

      if (shouldCollapse) {
        card.style.flex = '0 0 auto';
      } else {
        const cfg = panelConfigs.find(p => p.id === cardId);
        const flexVal = (cfg && cfg.savedFlex) ? cfg.savedFlex : (cfg ? cfg.defaultFlex : 1.0);
        card.style.flex = `${flexVal} 1 0px`;
      }

      updateResizersVisibility();
    }

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

      setupResizer('resizer1',
        () => document.getElementById('cardSettings'),
        () => {
          const res = document.getElementById('cardResources');
          if (!res.classList.contains('collapsed')) return res;
          const trace = document.getElementById('cardTrace');
          if (!trace.classList.contains('collapsed')) return trace;
          return null;
        }
      );

      setupResizer('resizer2',
        () => {
          const res = document.getElementById('cardResources');
          if (!res.classList.contains('collapsed')) return res;
          const set = document.getElementById('cardSettings');
          if (!set.classList.contains('collapsed')) return set;
          return null;
        },
        () => document.getElementById('cardTrace')
      );

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

    // Transition Matrix Heatmap Hover Handlers
    function setupTMCanvasListeners() {
      const cvs = document.getElementById('tmHeatmapCanvas');
      if (!cvs) return;

      cvs.addEventListener('mousemove', (e) => {
        const layout = cvs._tmLayout;
        if (!layout || layout.K === 0 || transitionCounts.length === 0) return;
        const rect = cvs.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) * (cvs.width / rect.width);
        const mouseY = (e.clientY - rect.top) * (cvs.height / rect.height);

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
      });

      cvs.addEventListener('mouseleave', () => {
        if (hoveredTMCell !== null) {
          hoveredTMCell = null;
          const tipEl = document.getElementById('tmCellTooltip');
          if (tipEl) tipEl.innerText = 'Hover over any cell to inspect transition details';
          drawTransitionMatrix('tmHeatmapCanvas', false);
          draw();
        }
      });
    }

    setupTMCanvasListeners();

    // Initial Startup
    initSidebarResizers();
    initLayoutResizer();
    updateTMCanvasDimensions();
    resizeCanvas();
    loadSelectedBenchmark();
    updateZoomBadge();
    setExplainMode(false);
    setTimeout(() => { updateTMCanvasDimensions(); resizeCanvas(); }, 50);
    setTimeout(() => { updateTMCanvasDimensions(); resizeCanvas(); }, 250);
