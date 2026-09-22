/**
 * GRIC Simulator - renderer.js
 * Part of the GRIC Interactive Algorithm Simulator
 */

//  6. QUAD-SCREEN VIEWPORT MANAGER & 3D RENDERING
    // =========================================================================

    // Viridis color ramp (16-stop approximation)
    const VIRIDIS_STOPS = [
      [68,1,84],[72,35,116],[64,67,135],
      [57,86,140],[49,104,142],[42,120,142],
      [35,137,142],[31,154,138],[53,170,120],
      [94,186,97],[144,201,67],[194,210,35],
      [227,220,25],[240,229,30],[248,238,35],
      [253,231,37]
    ];

    // Inferno color ramp (16-stop approximation)
    const INFERNO_STOPS = [
      [0,0,4],[14,11,53],[40,11,84],
      [73,10,119],[101,0,167],[137,34,141],
      [165,53,112],[187,72,86],[208,92,60],
      [224,116,38],[239,143,18],[249,170,10],
      [252,195,19],[250,219,67],[244,240,136],
      [252,255,164]
    ];

    // Quality color ramp: green → yellow → red (good → poor)
    const QUALITY_STOPS = [
      [34,197,94], [52,211,106], [74,222,128],
      [110,231,150], [153,240,176], [200,247,200],
      [254,249,195], [254,240,138], [253,224,71],
      [250,204,21], [245,158,11], [239,115,22],
      [239,68,68], [220,38,38], [185,28,28],
      [153,27,27]
    ];

    /**
     * Map a scalar value to an RGB color string.
     * @param {number} val - Data value
     * @param {number} mn  - Data minimum
     * @param {number} mx  - Data maximum
     * @param {Array}  stops - Color ramp array
     * @returns {string} CSS rgb() string
     */
    function getColorFromRamp(val, mn, mx, stops) {
      if (!stops || stops.length === 0) {
        return 'rgb(148, 163, 184)';
      }
      if (typeof val !== 'number' || isNaN(val) || !isFinite(val)) {
        val = (typeof mn === 'number' && isFinite(mn)) ? mn : 0;
      }
      if (typeof mn !== 'number' || isNaN(mn) || !isFinite(mn)) {
        mn = 0;
      }
      if (typeof mx !== 'number' || isNaN(mx) || !isFinite(mx) || mx <= mn) {
        mx = mn + 1.0;
      }
      const t = Math.max(0, Math.min(1, (val - mn) / (mx - mn)));
      const n = stops.length - 1;
      const fi = Math.max(0, Math.min(n, t * n));
      const lo = Math.floor(fi);
      const hi = Math.min(lo + 1, n);
      const f = fi - lo;
      const c0 = stops[lo] || stops[0];
      const c1 = stops[hi] || stops[n];
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
      return `rgb(${r},${g},${b})`;
    }

    if (typeof window !== 'undefined')
    {
      window.getColorFromRamp = getColorFromRamp;
      window.QUALITY_STOPS = QUALITY_STOPS;
    }

    /**
     * Get quality coloring info for the active slot.
     * @returns {Object|null} {arr, min, max, label} or null
     */
    function getReconQualityInfo()
    {
      if (!reconQualityColoringEnabled)
      {
        return null;
      }
      const slot = datasetSlots[activeDatasetSlot];
      if (slot && activeDatasetSlot === 'C' && slot.reconKthDist)
      {
        return {
          arr: slot.reconKthDist,
          min: slot.reconKthDistMin,
          max: slot.reconKthDistMax,
          label: 'k-th NN Dist'
        };
      }
      if (slot && activeDatasetSlot === 'D' && slot.reconVariance)
      {
        return {
          arr: slot.reconVariance,
          min: slot.reconVarianceMin,
          max: slot.reconVarianceMax,
          label: 'Recon Variance'
        };
      }
      if (datasetSlots['D'] && datasetSlots['D'].reconVariance)
      {
        const slotD = datasetSlots['D'];
        return {
          arr: slotD.reconVariance,
          min: slotD.reconVarianceMin,
          max: slotD.reconVarianceMax,
          label: 'Recon Variance'
        };
      }
      if (datasetSlots['C'] && datasetSlots['C'].reconKthDist)
      {
        const slotC = datasetSlots['C'];
        return {
          arr: slotC.reconKthDist,
          min: slotC.reconKthDistMin,
          max: slotC.reconKthDistMax,
          label: 'k-th NN Dist'
        };
      }
      return null;
    }


    let lastRenderedSquareSize = -1;
    let lastRenderedDpr = -1;

    function resizeCanvas(force) {
      const canvasWrapper = document.getElementById('canvasWrapper');
      if (!canvasWrapper) return;
      const availableW = canvasWrapper.clientWidth;
      const availableH = canvasWrapper.clientHeight;
      if (availableW <= 0 || availableH <= 0) return;

      const squareSize = Math.max(100, Math.floor(Math.min(availableW, availableH)));
      const dpr = window.devicePixelRatio || 1;

      if (!force && squareSize === lastRenderedSquareSize && dpr === lastRenderedDpr) {
        return;
      }
      lastRenderedSquareSize = squareSize;
      lastRenderedDpr = dpr;

      canvas.style.width = `${squareSize}px`;
      canvas.style.height = `${squareSize}px`;

      canvas.width = Math.round(squareSize * dpr);
      canvas.height = Math.round(squareSize * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();

      updateTMCanvasDimensions();
      if (typeof currentActiveTab !== 'undefined' && currentActiveTab === 'tm') {
        drawTransitionMatrix('tmHeatmapCanvas', false);
      }
    }

    function updateTMCanvasDimensions() {
      const heatWrap = document.getElementById('tmHeatmapWrapper');
      if (heatWrap && heatWrap.clientWidth > 0) {
        const wrapW = Math.max(260, Math.min(480, Math.floor(heatWrap.clientWidth)));
        tmCanvasDimensions.tmHeatmapCanvas = {
          w: wrapW,
          h: 260
        };
      }
    }
    window.addEventListener('resize', () => {
      resizeCanvas(true);
      if (typeof updateReconQualityBar === 'function') {
        updateReconQualityBar();
      } else if (typeof window.updateReconQualityBar === 'function') {
        window.updateReconQualityBar();
      }
    });
    window.addEventListener('load', () => resizeCanvas(true));
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(() => resizeCanvas(false));
      const cWrap = document.getElementById('canvasWrapper');
      if (cWrap) ro.observe(cWrap);
    }

    // Coordinate Transforms for Sub-Viewports
    function project3DVector(dx, dy, dz, az, el) {
      const cosT = Math.cos(az), sinT = Math.sin(az);
      const cosP = Math.cos(el), sinP = Math.sin(el);
      
      const u = dx * cosT + dy * sinT;
      const v = -dx * sinT * sinP + dy * cosT * sinP + dz * cosP;
      const depth = -dx * sinT * cosP + dy * cosT * cosP - dz * sinP;
      return { u, v, depth };
    }

    function project3D(x, y, z, az, el) {
      let tx = 0, ty = 0, tz = 0;
      if (typeof orbitCamera !== 'undefined' && orbitCamera && orbitCamera.isLocked) {
        tx = orbitCamera.targetX || 0;
        ty = orbitCamera.targetY || 0;
        tz = orbitCamera.targetZ || 0;
      }
      return project3DVector(x - tx, y - ty, z - tz, az, el);
    }

    function getQuadRect(qIdx, W, H) {
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        const halfW = W / 2;
        if (typeof reconOverlayMode !== 'undefined' && reconOverlayMode) {
          if (qIdx === 0 || qIdx === 2) {
            return { x: 0, y: 0, w: halfW, h: H };
          }
          return { x: halfW, y: 0, w: halfW, h: H };
        }
        const halfH = H / 2;
        switch (qIdx) {
          case 0: return { x: 0, y: 0, w: halfW, h: halfH };
          case 1: return { x: halfW, y: 0, w: halfW, h: halfH };
          case 2: return { x: 0, y: halfH, w: halfW, h: halfH };
          case 3: return { x: halfW, y: halfH, w: halfW, h: halfH };
          default: return { x: 0, y: 0, w: W, h: H };
        }
      }
      if (maximizedQuad !== null) {
        return { x: 0, y: 0, w: W, h: H };
      }
      if (currentDim === 2) {
        return { x: 0, y: 0, w: W, h: H };
      }
      const halfW = W / 2;
      const halfH = H / 2;
      switch (qIdx) {
        case 0: return { x: 0, y: 0, w: halfW, h: halfH };        // Top-Left: Along X
        case 1: return { x: halfW, y: 0, w: halfW, h: halfH };    // Top-Right: Along Y
        case 2: return { x: 0, y: halfH, w: halfW, h: halfH };    // Bottom-Left: Along Z
        case 3: return { x: halfW, y: halfH, w: halfW, h: halfH };// Bottom-Right: Custom 3D
        default: return { x: 0, y: 0, w: W, h: H };
      }
    }

    function getQuadScale(qIdx, rect) {
      const view = quadViews[qIdx];
      const baseScale = Math.min(rect.w, rect.h) / 2.35;
      return baseScale * (view.zoom || 1.0);
    }

    function mapMetricToQuad(u, v, qIdx, rect) {
      const view = quadViews[qIdx];
      const scale = getQuadScale(qIdx, rect);
      const cx = rect.x + rect.w / 2;
      const cy = rect.y + rect.h / 2;
      return {
        px: cx + (u - view.panX) * scale,
        py: cy - (v - view.panY) * scale
      };
    }

    function mapQuadToMetric(px, py, qIdx, rect) {
      const view = quadViews[qIdx];
      const scale = getQuadScale(qIdx, rect);
      const cx = rect.x + rect.w / 2;
      const cy = rect.y + rect.h / 2;
      return {
        u: (px - cx) / scale + view.panX,
        v: (cy - py) / scale + view.panY
      };
    }

    window.mapMetricToQuad = mapMetricToQuad;
    window.mapQuadToMetric = mapQuadToMetric;

    /**
     * Helper to render a stylized coordinate pill badge with colored axis labels.
     */
    function drawColoredCoordBadge(ctx, x, y, parts, isHovered, clampRect) {
      ctx.font = 'bold 9px monospace';
      let totalW = 0;
      for (let i = 0; i < parts.length; i++) {
        totalW += ctx.measureText(parts[i].text).width;
      }
      const pillW = totalW + 12;
      const pillH = 16;

      let drawX = x;
      let drawY = y;
      if (clampRect) {
        if (drawX + pillW > clampRect.x + clampRect.w - 6) {
          drawX = clampRect.x + clampRect.w - pillW - 6;
        }
        if (drawX < clampRect.x + 6) drawX = clampRect.x + 6;
        if (drawY < clampRect.y + 28) drawY = clampRect.y + 28;
        if (drawY + pillH > clampRect.y + clampRect.h - 6) {
          drawY = clampRect.y + clampRect.h - pillH - 6;
        }
      }

      ctx.fillStyle = isHovered ? 'rgba(15, 23, 42, 0.92)' : 'rgba(15, 23, 42, 0.88)';
      ctx.strokeStyle = isHovered ? 'rgba(56, 189, 248, 0.60)' : 'rgba(56, 189, 248, 0.35)';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.roundRect(drawX, drawY, pillW, pillH, 3);
      ctx.fill();
      ctx.stroke();

      let curX = drawX + 6;
      const midY = drawY + pillH / 2 + 0.5;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      for (let i = 0; i < parts.length; i++) {
        ctx.fillStyle = parts[i].color;
        ctx.fillText(parts[i].text, curX, midY);
        curX += ctx.measureText(parts[i].text).width;
      }
    }

    function updateViewPresetBarPosition() {
      const bar = document.getElementById('viewPresetBar');
      if (!bar) return;

      const isImage = (typeof dataMode !== 'undefined' && dataMode === 'image');
      if (isImage) {
        bar.style.display = 'none';
        return;
      }

      bar.style.display = 'flex';
      if (typeof updatePlottingDimSelectorsUI === 'function') {
        updatePlottingDimSelectorsUI();
      }

      const btnResetView = document.getElementById('btnResetView');
      const btnIso = document.getElementById('presetIso');
      const btnFront = document.getElementById('presetFront');
      const btnTop = document.getElementById('presetTop');
      const btnSide = document.getElementById('presetSide');
      const btnReset3D = document.getElementById('presetReset3D');
      const btnRecon4P = document.getElementById('btnPresetRecon4Panel');

      const isRecon = (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView);
      let anySlotIs3D = false;
      if (isRecon && typeof datasetSlots !== 'undefined') {
        anySlotIs3D = Object.values(datasetSlots).some(s => s && s.currentDim >= 3);
      }
      const is3D = isRecon ? anySlotIs3D : (currentDim >= 3);
      const is3DOrbitVisible = is3D && (maximizedQuad === null || maximizedQuad === 3 || isRecon);

      if (btnIso) btnIso.style.display = is3DOrbitVisible ? '' : 'none';
      if (btnFront) btnFront.style.display = is3DOrbitVisible ? '' : 'none';
      if (btnTop) btnTop.style.display = is3DOrbitVisible ? '' : 'none';
      if (btnSide) btnSide.style.display = is3DOrbitVisible ? '' : 'none';
      if (btnReset3D) btnReset3D.style.display = is3DOrbitVisible ? '' : 'none';
      if (btnRecon4P) {
        btnRecon4P.classList.toggle('active', !!isRecon);
        const isOverlay = (typeof reconOverlayMode !== 'undefined' && reconOverlayMode);
        btnRecon4P.textContent = (isRecon && isOverlay) ? '⧉ A+C | B+D' : '⊞ ABCD';
      }

      if (btnResetView) {
        btnResetView.style.display = '';
        btnResetView.textContent = is3D ? '🔍 1:1' : '🔍 1:1 Reset View';
      }

      const canvasWrapper = document.getElementById('canvasWrapper');
      if (!canvasWrapper || !canvas) return;

      if (typeof updateImageQuadDropdowns === 'function') {
        updateImageQuadDropdowns();
      }

      const wrapRect = canvasWrapper.getBoundingClientRect();
      const cRect = canvas.getBoundingClientRect();

      let targetLeft, targetTop, targetWidth, targetHeight;
      if (isRecon || !is3D || maximizedQuad !== null) {
        targetLeft = cRect.left - wrapRect.left;
        targetTop = cRect.top - wrapRect.top;
        targetWidth = cRect.width;
        targetHeight = cRect.height;
      } else {
        // Quad 3 (Bottom-Right: 3D Orbit view)
        targetLeft = (cRect.left - wrapRect.left) + cRect.width / 2;
        targetTop = (cRect.top - wrapRect.top) + cRect.height / 2;
        targetWidth = cRect.width / 2;
        targetHeight = cRect.height / 2;
      }

      const barRight = wrapRect.width - (targetLeft + targetWidth) + 8;
      const barBottom = wrapRect.height - (targetTop + targetHeight) + 8;

      bar.style.position = 'absolute';
      bar.style.right = `${Math.max(6, Math.round(barRight))}px`;
      bar.style.bottom = `${Math.max(6, Math.round(barBottom))}px`;
      bar.style.top = 'auto';
      bar.style.left = 'auto';

      // Update active preset button highlight
      if (is3DOrbitVisible && typeof orbitCamera !== 'undefined') {
        const azDeg = Math.round(orbitCamera.azimuth * 180 / Math.PI);
        const elDeg = Math.round(orbitCamera.elevation * 180 / Math.PI);

        if (btnIso) {
          btnIso.classList.toggle(
            'active', Math.abs(azDeg - (-35)) <= 2 && Math.abs(elDeg - 25) <= 2
          );
        }
        if (btnFront) {
          btnFront.classList.toggle(
            'active', Math.abs(azDeg - 0) <= 2 && Math.abs(elDeg - 0) <= 2
          );
        }
        if (btnTop) {
          btnTop.classList.toggle(
            'active', Math.abs(azDeg - 0) <= 2 && Math.abs(elDeg - 89) <= 2
          );
        }
        if (btnSide) {
          btnSide.classList.toggle(
            'active', Math.abs(azDeg - 90) <= 2 && Math.abs(elDeg - 0) <= 2
          );
        }

        const btnLock = document.getElementById('btnLockCenter3D');
        if (btnLock) {
          btnLock.classList.toggle('active', !!orbitCamera.isLocked);
          btnLock.classList.toggle('toggle-active', !!orbitCamera.isLocked);
          if (orbitCamera.isLocked) {
            btnLock.style.background = 'rgba(56, 189, 248, 0.25)';
            btnLock.style.borderColor = '#38bdf8';
            btnLock.style.color = '#38bdf8';
          } else {
            btnLock.style.background = '';
            btnLock.style.borderColor = '';
            btnLock.style.color = '';
          }
        }
      }
    }
    window.updateViewPresetBarPosition = updateViewPresetBarPosition;

    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.width / dpr;
      const H = canvas.height / dpr;

      if (typeof viewportZoomBoxRects !== 'undefined') {
        viewportZoomBoxRects = [null, null, null, null];
      }

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#0b1120';
      ctx.fillRect(0, 0, W, H);

      if (typeof updateImageQuadDropdowns === 'function') {
        updateImageQuadDropdowns();
      }

      // --- RECONSTRUCTION 4-PANEL VIEW MODE (A: TL, B: TR, C: BL, D: BR) ---
      if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView) {
        if (typeof isReconstructionImageMode === 'function' && isReconstructionImageMode()) {
          if (typeof drawReconImage4PanelView === 'function') {
            drawReconImage4PanelView(ctx, W, H);
          }
        } else {
          drawRecon4PanelView(ctx, W, H);
        }
        updateViewPresetBarPosition();
        return;
      }

      // --- IMAGE MODE: 4-Quadrant Raster / Centroid Gallery ---
      if (typeof dataMode !== 'undefined' && dataMode === 'image') {
        if (typeof drawImageMode === 'function') {
          drawImageMode(ctx, W, H);
        }
        updateViewPresetBarPosition();
        return;
      }

      // --- 2D MODE: Single Screen ---
      if (currentDim === 2) {
        renderSubViewport(2, "2D", { x: 0, y: 0, w: W, h: H });
        updateViewPresetBarPosition();
        return;
      }

      // --- 3D MODE: Quad-Split or Maximized View ---
      if (typeof viewVectorHandles !== 'undefined') {
        viewVectorHandles = [null, null, null];
      }
      if (maximizedQuad !== null) {
        const types = ["ALONG_X", "ALONG_Y", "ALONG_Z", "CUSTOM_3D"];
        renderSubViewport(maximizedQuad, types[maximizedQuad], { x: 0, y: 0, w: W, h: H });
        updateViewPresetBarPosition();
        return;
      }

      // Render 4 Quadrants
      for (let qi = 0; qi < 4; qi++)
      {
        const types = [
          "ALONG_X", "ALONG_Y", "ALONG_Z", "CUSTOM_3D"
        ];
        try
        {
          renderSubViewport(
            qi, types[qi], getQuadRect(qi, W, H)
          );
        }
        catch (err)
        {
          console.error(
            `[Renderer] Viewport ${qi} error:`, err
          );
        }
      }

      // Draw Viewport Divider Lines
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1.5;
      
      ctx.beginPath();
      ctx.moveTo(W / 2, 0);
      ctx.lineTo(W / 2, H);
      ctx.moveTo(0, H / 2);
      ctx.lineTo(W, H / 2);
      ctx.stroke();

      updateViewPresetBarPosition();
    }

    function renderSubViewport(qIdx, viewType, rect) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(rect.x, rect.y, rect.w, rect.h);
      ctx.clip();

      // Check if Quad 2 is in 32D Parallel Coordinates (PCP) mode
      if (qIdx === 2 && currentDim > 3 && typeof quad2Mode !== 'undefined' && quad2Mode === 'pcp') {
        if (typeof HighDEngine !== 'undefined' && HighDEngine.renderParallelCoordinates) {
          const sampleData = (typeof pastSamples !== 'undefined' && pastSamples.length > 0)
            ? pastSamples : (typeof benchmarkDataset !== 'undefined' ? benchmarkDataset : null);
          HighDEngine.renderParallelCoordinates(
            ctx, rect, clusters, sampleData, hoveredClusterId, selectedClusterId, currentDim
          );
          ctx.restore();
          return;
        }
      }

      const scale = getQuadScale(qIdx, rect);
      const is3DCustom = (viewType === "CUSTOM_3D");

      // Projection point helper
      function getProjectedCoord(p) {
        if (currentDim > 3 && typeof decoupledQuads !== 'undefined' &&
            decoupledQuads && qIdx < 3) {
          const pair = (typeof quadAxisPairs !== 'undefined' && quadAxisPairs[qIdx])
            ? quadAxisPairs[qIdx] : { x: qIdx * 2, y: qIdx * 2 + 1 };
          const u = (typeof getPointDim === 'function')
            ? getPointDim(p, pair.x)
            : (p.coords ? (p.coords[pair.x] || 0.0) : 0.0);
          const v = (typeof getPointDim === 'function')
            ? getPointDim(p, pair.y)
            : (p.coords ? (p.coords[pair.y] || 0.0) : 0.0);
          return { u, v, depth: 0 };
        }
        const pt = (typeof getPlotCoords === 'function') ? getPlotCoords(p) : p;
        if (viewType === "ALONG_X") return { u: pt.y, v: pt.z, depth: pt.x };
        if (viewType === "ALONG_Y") return { u: pt.x, v: pt.z, depth: pt.y };
        if (viewType === "ALONG_Z" || viewType === "2D") {
          return { u: pt.x, v: pt.y, depth: pt.z };
        }
        // CUSTOM_3D
        return project3D(pt.x, pt.y, pt.z, orbitCamera.azimuth, orbitCamera.elevation);
      }

      function resolvePointClusterId(p, i) {
        if (!p) return -1;
        if (typeof p.clusterId === 'number' && p.clusterId >= 0) {
          return p.clusterId;
        }
        if (typeof assignmentHistory !== 'undefined' && assignmentHistory &&
            typeof assignmentHistory[i] === 'number' && assignmentHistory[i] >= 0) {
          p.clusterId = assignmentHistory[i];
          return p.clusterId;
        }
        if (typeof benchmarkDataset !== 'undefined' && benchmarkDataset &&
            benchmarkDataset[i] && typeof benchmarkDataset[i].clusterId === 'number' &&
            benchmarkDataset[i].clusterId >= 0) {
          p.clusterId = benchmarkDataset[i].clusterId;
          return p.clusterId;
        }
        if (typeof clusters !== 'undefined' && clusters && clusters.length > 0) {
          let bestK = -1;
          let bestD2 = Infinity;
          const px = p.x, py = p.y, pz = p.z || 0;
          for (let k = 0; k < clusters.length; k++) {
            const c = clusters[k];
            const dx = px - c.x, dy = py - c.y, dz = pz - (c.z || 0);
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < bestD2) {
              bestD2 = d2;
              bestK = k;
            }
          }
          if (bestK >= 0) {
            p.clusterId = bestK;
            return bestK;
          }
        }
        return -1;
      }

      // 1. Grid & Axes
      if (showGridAxes) {
        if (!is3DCustom) {
          // Orthogonal 2D Projection Grids
          const center = mapMetricToQuad(0, 0, qIdx, rect);
          
          ctx.strokeStyle = '#1e293b';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(rect.x, center.py);
          ctx.lineTo(rect.x + rect.w, center.py);
          ctx.moveTo(center.px, rect.y);
          ctx.lineTo(center.px, rect.y + rect.h);
          ctx.stroke();

          // Reference Circles (radius 0.5, 0.85)
          ctx.strokeStyle = '#172554';
          ctx.setLineDash([2, 4]);
          [0.5, 0.85].forEach(rad => {
            ctx.beginPath();
            ctx.arc(center.px, center.py, rad * scale, 0, Math.PI * 2);
            ctx.stroke();
          });
          ctx.setLineDash([]);
        } else {
          // 3D Custom Projection: Bounding Cube & Grid Floor
          const az = orbitCamera.azimuth, el = orbitCamera.elevation;
          
          // 3D Bounding Box [-0.85, 0.85]^3
          const b = 0.85;
          const boxCorners = [
            {x:-b, y:-b, z:-b}, {x: b, y:-b, z:-b}, {x: b, y: b, z:-b}, {x:-b, y: b, z:-b},
            {x:-b, y:-b, z: b}, {x: b, y:-b, z: b}, {x: b, y: b, z: b}, {x:-b, y: b, z: b}
          ];
          const boxPx = boxCorners.map(pt => {
            const pr = project3D(pt.x, pt.y, pt.z, az, el);
            return mapMetricToQuad(pr.u, pr.v, qIdx, rect);
          });

          // Floor Grid at z = -b
          ctx.strokeStyle = 'rgba(30, 41, 59, 0.6)';
          ctx.lineWidth = 1;
          [-0.85, -0.425, 0, 0.425, 0.85].forEach(val => {
            const p1 = mapMetricToQuad(project3D(val, -b, -b, az, el).u,
                                       project3D(val, -b, -b, az, el).v, qIdx, rect);
            const p2 = mapMetricToQuad(project3D(val,  b, -b, az, el).u,
                                       project3D(val,  b, -b, az, el).v, qIdx, rect);
            ctx.beginPath(); ctx.moveTo(p1.px, p1.py); ctx.lineTo(p2.px, p2.py); ctx.stroke();

            const p3 = mapMetricToQuad(project3D(-b, val, -b, az, el).u,
                                       project3D(-b, val, -b, az, el).v, qIdx, rect);
            const p4 = mapMetricToQuad(project3D( b, val, -b, az, el).u,
                                       project3D( b, val, -b, az, el).v, qIdx, rect);
            ctx.beginPath(); ctx.moveTo(p3.px, p3.py); ctx.lineTo(p4.px, p4.py); ctx.stroke();
          });

          // Bounding Box Edges
          const edges = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
          ];
          ctx.strokeStyle = 'rgba(51, 65, 85, 0.5)';
          ctx.setLineDash([2, 3]);
          edges.forEach(([i, j]) => {
            ctx.beginPath();
            ctx.moveTo(boxPx[i].px, boxPx[i].py);
            ctx.lineTo(boxPx[j].px, boxPx[j].py);
            ctx.stroke();
          });
          ctx.setLineDash([]);

          // 3D Locked Rotation Center Reticle
          if (orbitCamera && orbitCamera.isLocked) {
            const centerPr = project3D(orbitCamera.targetX, orbitCamera.targetY, orbitCamera.targetZ, az, el);
            const centerPos = mapMetricToQuad(centerPr.u, centerPr.v, qIdx, rect);

            ctx.save();
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)';
            ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
            ctx.lineWidth = 1.5;

            // Target focal ring
            ctx.beginPath();
            ctx.arc(centerPos.px, centerPos.py, 10, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // Reticle crosshair ticks
            ctx.beginPath();
            ctx.moveTo(centerPos.px - 16, centerPos.py);
            ctx.lineTo(centerPos.px - 6, centerPos.py);
            ctx.moveTo(centerPos.px + 6, centerPos.py);
            ctx.lineTo(centerPos.px + 16, centerPos.py);
            ctx.moveTo(centerPos.px, centerPos.py - 16);
            ctx.lineTo(centerPos.px, centerPos.py - 6);
            ctx.moveTo(centerPos.px, centerPos.py + 6);
            ctx.lineTo(centerPos.px, centerPos.py + 16);
            ctx.stroke();

            // Center focal dot
            ctx.beginPath();
            ctx.arc(centerPos.px, centerPos.py, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = '#38bdf8';
            ctx.fill();

            // Label tag below reticle
            const lbl = orbitCamera.targetLabel ||
              `(${orbitCamera.targetX.toFixed(2)}, ${orbitCamera.targetY.toFixed(2)}, ${orbitCamera.targetZ.toFixed(2)})`;
            ctx.fillStyle = '#38bdf8';
            ctx.font = 'bold 9px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`🎯 ${lbl}`, centerPos.px, centerPos.py + 22);

            ctx.restore();
          }
        }
      }

      // Visual Focus / Opacity calculations
      const tFocus = visualFocus / 100.0;
      let pointAlpha = 0.38;
      let ptSizeScale = 1.0;
      if (tFocus < 0.5) {
        const ratio = (0.5 - tFocus) / 0.5; // 0.0 at 0.5, 1.0 at 0.0
        pointAlpha = 0.38 + ratio * 0.57; // 0.38 -> 0.95
        ptSizeScale = 1.0 + ratio * 0.6; // 1.0 -> 1.6
      } else {
        const ratio = (tFocus - 0.5) / 0.5; // 0.0 at 0.5, 1.0 at 1.0
        pointAlpha = 0.38 * (1.0 - ratio); // 0.38 -> 0.0
        ptSizeScale = 1.0;
      }

      let clusterAlpha = 1.0;
      if (tFocus < 0.5) {
        clusterAlpha = tFocus / 0.5; // 0.0 at t=0.0, 1.0 at t=0.5
      }

      // 2. Sample Point Cloud (Viewport-Aware Dynamic Density & Subsampling)
      let drawnPointsCount = 0;
      let totalVisiblePoints = 0;
      let isPointsTruncated = false;
      const numPast = pastSamples.length;

      if (showPastSamples && numPast > 0 && pointAlpha > 0.001) {
        const basePtRad = samplePointSize;
        const view = quadViews[qIdx] || { panX: 0, panY: 0, zoom: 1.0 };
        const cx = rect.x + rect.w / 2;
        const cy = rect.y + rect.h / 2;

        // Determine visible metric bounds with margin
        const marginPx = 6;
        const halfW_metric = (rect.w / 2 + marginPx) / scale;
        const halfH_metric = (rect.h / 2 + marginPx) / scale;
        const uMin = view.panX - halfW_metric;
        const uMax = view.panX + halfW_metric;
        const vMin = view.panY - halfH_metric;
        const vMax = view.panY + halfH_metric;

        // Ensure reusable index buffer is allocated
        if (!visibleIndicesBuffer || visibleIndicesBuffer.length < numPast) {
          visibleIndicesBuffer = new Int32Array(Math.max(numPast + 50000, 500000));
        }

        const slotObj = (typeof datasetSlots !== 'undefined')
          ? datasetSlots[activeDatasetSlot] : null;
        const qFovMask = (typeof reconQualityThreshold !== 'undefined' &&
          reconQualityThreshold < 1.0)
          ? ((slotObj && slotObj.reconQualityMask) ||
             (activeDatasetSlot === 'C' || activeDatasetSlot === 'D'
               ? reconQualityMask : null)) : null;

        // Collect indices of all points that fall within visible FOV
        const isHighD = (currentDim > 3);
        const isDecoupled = (isHighD && typeof decoupledQuads !== 'undefined' &&
          decoupledQuads && qIdx < 3);
        const pair = isDecoupled
          ? ((typeof quadAxisPairs !== 'undefined' && quadAxisPairs[qIdx])
              ? quadAxisPairs[qIdx] : { x: qIdx * 2, y: qIdx * 2 + 1 })
          : null;
        const pairX = pair ? pair.x : 0;
        const pairY = pair ? pair.y : 1;

        const hasSlice = (isHighD && typeof sliceDim === 'number' &&
          sliceDim >= 0 && typeof HighDEngine !== 'undefined');
        for (let i = 0; i < numPast; i++) {
          if (qFovMask && i < qFovMask.length && !qFovMask[i]) continue;
          const pt = pastSamples[i];
          if (hasSlice) {
            const inSlice = HighDEngine.isPointInSlice(
              pt, sliceDim, sliceCenter, sliceThickness
            );
            if (!inSlice.inside) continue;
          }
          let pu = 0.0, pv = 0.0;
          if (isDecoupled) {
            const coords = pt.coords;
            pu = coords ? (coords[pairX] || 0.0)
              : (pairX === 0 ? pt.x : (pairX === 1 ? pt.y : (pt.z || 0.0)));
            pv = coords ? (coords[pairY] || 0.0)
              : (pairY === 0 ? pt.x : (pairY === 1 ? pt.y : (pt.z || 0.0)));
          } else if (viewType === "ALONG_Z" || viewType === "2D") {
            pu = pt.x;
            pv = pt.y;
          } else if (viewType === "ALONG_X") {
            pu = pt.y;
            pv = (typeof pt.z === 'number') ? pt.z : 0.0;
          } else if (viewType === "ALONG_Y") {
            pu = pt.x;
            pv = (typeof pt.z === 'number') ? pt.z : 0.0;
          } else {
            const pr = getProjectedCoord(pt);
            pu = pr.u;
            pv = pr.v;
          }
          if (pu >= uMin && pu <= uMax && pv >= vMin && pv <= vMax) {
            visibleIndicesBuffer[totalVisiblePoints++] = i;
          }
        }

        const maxDraw = maxDrawPoints;
        const drawCount = Math.min(totalVisiblePoints, maxDraw);
        isPointsTruncated = (totalVisiblePoints > maxDraw);
        const step = totalVisiblePoints > 0 ? (totalVisiblePoints / drawCount) : 1;

        // Distinct colors for unprocessed vs processed sample points
        const unprocColor = `rgba(100, 116, 139, ${(pointAlpha * 0.40).toFixed(3)})`;
        const procDefaultColor = `rgba(148, 163, 184, ${pointAlpha.toFixed(3)})`;

        // Cache cluster colors to maximize rendering throughput
        const clusterColorCache = {};
        const getCachedColor = (cid) => {
          if (cid === undefined || cid === null || cid < 0) return procDefaultColor;
          if (!clusterColorCache[cid]) {
            const hues = [199, 142, 270, 38, 340, 180, 48, 220, 110, 300, 15, 160, 205, 80, 320];
            const hue = hues[cid % hues.length];
            clusterColorCache[cid] = `hsla(${hue}, 85%, 60%, ${pointAlpha.toFixed(3)})`;
          }
          return clusterColorCache[cid];
        };

        // Quality coloring lookup (once per viewport)
        const qInfo = getReconQualityInfo();
        const qArr = qInfo ? qInfo.arr : null;
        const qMin = qInfo ? qInfo.min : 0;
        const qMax = qInfo ? qInfo.max : 1;
        const qMask = reconQualityMask;

        if (!is3DCustom) {
          const ptRad = Math.max(0.5, basePtRad * ptSizeScale);
          const d = ptRad * 2;

          // Pass 1: Unprocessed Points (staged, not yet ingested)
          ctx.fillStyle = unprocColor;
          for (let k = 0; k < drawCount; k++) {
            const idx = visibleIndicesBuffer[Math.floor(k * step)];
            const pt = pastSamples[idx];
            const cId = resolvePointClusterId(pt, idx);
            const isProcessed = (cId >= 0) ||
                                (pt.frameIndex !== undefined && pt.frameIndex < currentFrameIdx) ||
                                (idx < currentFrameIdx) ||
                                (clusters && clusters.length > 0 && totalFrames > 0 &&
                                 currentFrameIdx >= totalFrames);
            if (isProcessed) continue;
            if (qMask && idx < qMask.length && !qMask[idx]) continue;

            const pr = getProjectedCoord(pt);
            const px = cx + (pr.u - view.panX) * scale;
            const py = cy - (pr.v - view.panY) * scale;

            // Color override: quality > dim/density
            if (qArr && idx < qArr.length) {
              ctx.fillStyle = getColorFromRamp(
                qArr[idx], qMin, qMax, QUALITY_STOPS
              );
            } else if (typeof pointColorMode !== 'undefined'
                && pointColorMode !== 'cluster'
                && dimDensityResults
                && dimDensitySummary
                && idx < dimDensityResults.totalFrames
            ) {
              let fc;
              if (pointColorMode === 'dimension') {
                const s =
                  dimDensitySummary
                    .intrinsic_dimension;
                fc = getColorFromRamp(
                  dimDensityResults.localDim[idx],
                  s.min, s.max, VIRIDIS_STOPS
                );
              } else {
                const s =
                  dimDensitySummary.log_density;
                fc = getColorFromRamp(
                  dimDensityResults.logDensity[idx],
                  s.min, s.max, INFERNO_STOPS
                );
              }
              ctx.fillStyle = fc;
            }

            ctx.fillRect(px - ptRad, py - ptRad, d, d);
            drawnPointsCount++;
          }

          // Pass 2: Processed Points (clustered / ingested)
          let lastFill = null;
          for (let k = 0; k < drawCount; k++) {
            const idx = visibleIndicesBuffer[Math.floor(k * step)];
            const pt = pastSamples[idx];
            const cId = resolvePointClusterId(pt, idx);
            const isProcessed = (cId >= 0) ||
                                (pt.frameIndex !== undefined && pt.frameIndex < currentFrameIdx) ||
                                (idx < currentFrameIdx) ||
                                (clusters && clusters.length > 0 && totalFrames > 0 &&
                                 currentFrameIdx >= totalFrames);
            if (!isProcessed) continue;
            if (qMask && idx < qMask.length && !qMask[idx]) continue;

            const pr = getProjectedCoord(pt);
            const px = cx + (pr.u - view.panX) * scale;
            const py = cy - (pr.v - view.panY) * scale;

            let fillColor;
            if (qArr && idx < qArr.length) {
              fillColor = getColorFromRamp(
                qArr[idx], qMin, qMax, QUALITY_STOPS
              );
            } else if (typeof pointColorMode !== 'undefined'
                && pointColorMode !== 'cluster'
                && dimDensityResults
                && dimDensitySummary
                && idx < dimDensityResults.totalFrames
            ) {
              if (pointColorMode === 'dimension') {
                const s =
                  dimDensitySummary
                    .intrinsic_dimension;
                fillColor = getColorFromRamp(
                  dimDensityResults.localDim[idx],
                  s.min, s.max, VIRIDIS_STOPS
                );
              } else {
                const s =
                  dimDensitySummary.log_density;
                fillColor = getColorFromRamp(
                  dimDensityResults.logDensity[idx],
                  s.min, s.max, INFERNO_STOPS
                );
              }
            } else {
              fillColor = (showColorPerCluster && cId >= 0)
                ? getCachedColor(cId)
                : procDefaultColor;
            }
            if (fillColor !== lastFill) {
              ctx.fillStyle = fillColor;
              lastFill = fillColor;
            }
            ctx.fillRect(
              px - ptRad, py - ptRad, d, d
            );
            drawnPointsCount++;
          }
        } else {
          // 3D Perspective Orbit View
          const az = orbitCamera.azimuth;
          const el = orbitCamera.elevation;

          // Pass 1: Unprocessed Points
          ctx.fillStyle = unprocColor;
          for (let k = 0; k < drawCount; k++) {
            const idx = visibleIndicesBuffer[Math.floor(k * step)];
            const pt = pastSamples[idx];
            const cId = resolvePointClusterId(pt, idx);
            const isProcessed = (cId >= 0) ||
                                (pt.frameIndex !== undefined && pt.frameIndex < currentFrameIdx) ||
                                (idx < currentFrameIdx) ||
                                (clusters && clusters.length > 0 && totalFrames > 0 &&
                                 currentFrameIdx >= totalFrames);
            if (isProcessed) continue;
            if (qMask && idx < qMask.length && !qMask[idx]) continue;
            const pr = getProjectedCoord(pt);
            const px = cx + (pr.u - view.panX) * scale;
            const py = cy - (pr.v - view.panY) * scale;
            const depthFactor = Math.max(
              0.4, Math.min(2.2, 1.0 + pr.depth * 0.5)
            );
            const ptRad = Math.max(
              0.4, basePtRad * depthFactor * ptSizeScale
            );

            // Color override: quality > dim/density
            if (qArr && idx < qArr.length) {
              ctx.fillStyle = getColorFromRamp(
                qArr[idx], qMin, qMax, QUALITY_STOPS
              );
            } else if (typeof pointColorMode !== 'undefined'
                && pointColorMode !== 'cluster'
                && dimDensityResults
                && dimDensitySummary
                && idx < dimDensityResults.totalFrames
            ) {
              let fc;
              if (pointColorMode === 'dimension') {
                const s =
                  dimDensitySummary
                    .intrinsic_dimension;
                fc = getColorFromRamp(
                  dimDensityResults.localDim[idx],
                  s.min, s.max, VIRIDIS_STOPS
                );
              } else {
                const s =
                  dimDensitySummary.log_density;
                fc = getColorFromRamp(
                  dimDensityResults.logDensity[idx],
                  s.min, s.max, INFERNO_STOPS
                );
              }
              ctx.fillStyle = fc;
            }

            ctx.fillRect(
              px - ptRad, py - ptRad, ptRad * 2, ptRad * 2
            );
            drawnPointsCount++;
          }

          // Pass 2: Processed Points
          let lastFill = null;
          for (let k = 0; k < drawCount; k++) {
            const idx = visibleIndicesBuffer[Math.floor(k * step)];
            const pt = pastSamples[idx];
            const cId = resolvePointClusterId(pt, idx);
            const isProcessed = (cId >= 0) ||
                                (pt.frameIndex !== undefined && pt.frameIndex < currentFrameIdx) ||
                                (idx < currentFrameIdx) ||
                                (clusters && clusters.length > 0 && totalFrames > 0 &&
                                 currentFrameIdx >= totalFrames);
            if (!isProcessed) continue;
            if (qMask && idx < qMask.length && !qMask[idx]) continue;

            const pr = getProjectedCoord(pt);
            const px = cx + (pr.u - view.panX) * scale;
            const py = cy - (pr.v - view.panY) * scale;
            const depthFactor = Math.max(
              0.4, Math.min(2.2, 1.0 + pr.depth * 0.5)
            );
            const ptRad = Math.max(
              0.4, basePtRad * depthFactor * ptSizeScale
            );

            let fillColor;
            if (qArr && idx < qArr.length) {
              fillColor = getColorFromRamp(
                qArr[idx], qMin, qMax, QUALITY_STOPS
              );
            } else if (typeof pointColorMode !== 'undefined'
                && pointColorMode !== 'cluster'
                && dimDensityResults
                && dimDensitySummary
                && idx < dimDensityResults.totalFrames
            ) {
              if (pointColorMode === 'dimension') {
                const s =
                  dimDensitySummary
                    .intrinsic_dimension;
                fillColor = getColorFromRamp(
                  dimDensityResults.localDim[idx],
                  s.min, s.max, VIRIDIS_STOPS
                );
              } else {
                const s =
                  dimDensitySummary.log_density;
                fillColor = getColorFromRamp(
                  dimDensityResults.logDensity[idx],
                  s.min, s.max, INFERNO_STOPS
                );
              }
            } else {
              fillColor = (showColorPerCluster && cId >= 0)
                ? getCachedColor(cId)
                : procDefaultColor;
            }
            if (fillColor !== lastFill) {
              ctx.fillStyle = fillColor;
              lastFill = fillColor;
            }
            ctx.fillRect(
              px - ptRad, py - ptRad,
              ptRad * 2, ptRad * 2
            );
            drawnPointsCount++;
          }
        }
      }

      // Quality color bar legend
      {
        const qLegend = getReconQualityInfo();
        if (qLegend)
        {
          const barW = 12, barH = 80;
          const barX = rect.x + 10;
          const barY = rect.y + rect.h - barH - 30;

          for (let row = 0; row < barH; row++)
          {
            const t = row / (barH - 1);
            const c = getColorFromRamp(
              qLegend.min + t * (qLegend.max - qLegend.min),
              qLegend.min, qLegend.max, QUALITY_STOPS
            );
            ctx.fillStyle = c;
            ctx.fillRect(barX, barY + barH - 1 - row, barW, 1);
          }

          ctx.strokeStyle = 'rgba(255,255,255,0.3)';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(barX, barY, barW, barH);

          ctx.font = '9px monospace';
          ctx.fillStyle = 'rgba(255,255,255,0.7)';
          ctx.textAlign = 'left';
          ctx.fillText(
            qLegend.max.toFixed(4),
            barX + barW + 3, barY + 8
          );
          ctx.fillText(
            qLegend.min.toFixed(4),
            barX + barW + 3, barY + barH
          );
          ctx.fillText(qLegend.label, barX, barY - 4);
        }
      }

      if (typeof viewportPointStats !== 'undefined' && viewportPointStats[qIdx]) {
        viewportPointStats[qIdx] = {
          drawn: drawnPointsCount,
          visible: totalVisiblePoints,
          truncated: isPointsTruncated
        };
      }

      // Dynamic Stream Trajectory Motion Tail (off by default, toggleable)
      if (showMotionTail && typeof pastSamples !== 'undefined' &&
          pastSamples && pastSamples.length > 1) {
        const currIdx = (typeof currentFrameIdx === 'number' && currentFrameIdx > 0)
          ? Math.min(currentFrameIdx, pastSamples.length)
          : pastSamples.length;
        const N_tail = Math.min(12, currIdx);
        const startTail = currIdx - N_tail;
        if (N_tail > 1) {
          ctx.save();
          for (let t = startTail; t < currIdx - 1; t++) {
            const ptA = pastSamples[t];
            const ptB = pastSamples[t + 1];
            if (!ptA || !ptB) continue;
            const prA = getProjectedCoord(ptA);
            const prB = getProjectedCoord(ptB);
            const posA = mapMetricToQuad(prA.u, prA.v, qIdx, rect);
            const posB = mapMetricToQuad(prB.u, prB.v, qIdx, rect);
            const tRatio = (t - startTail + 1) / N_tail;
            ctx.strokeStyle = `rgba(56, 189, 248, ${tRatio * 0.75})`;
            ctx.lineWidth = 1.0 + tRatio * 2.0;
            ctx.beginPath();
            ctx.moveTo(posA.px, posA.py);
            ctx.lineTo(posB.px, posB.py);
            ctx.stroke();
          }
          ctx.restore();
        }
      }

      // Cluster Spawn Expanding Shockwave Ripple Animation
      if (typeof clusterSpawnRipples !== 'undefined' && clusterSpawnRipples.length > 0) {
        const nowTime = performance.now();
        let anyRippleActive = false;
        clusterSpawnRipples.forEach(rip => {
          const elapsed = nowTime - rip.startTime;
          if (elapsed < rip.duration) {
            anyRippleActive = true;
            const progress = elapsed / rip.duration;
            const ripRad = (6 + progress * 28);
            const pr = getProjectedCoord(rip);
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);

            ctx.save();
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, ripRad, 0, Math.PI * 2);
            ctx.strokeStyle = rip.color;
            ctx.globalAlpha = (1.0 - progress) * 0.85;
            ctx.lineWidth = 2.2 * (1.0 - progress);
            ctx.stroke();
            ctx.restore();
          }
        });
        clusterSpawnRipples = clusterSpawnRipples.filter(r => (nowTime - r.startTime) < r.duration);
        if (anyRippleActive) {
          requestAnimationFrame(draw);
        }
      }

      // 3. Multi-Tile Mode Rendering
      if (useTiles) {
        if (clusterAlpha > 0.001) {
          ctx.save();
          ctx.globalAlpha = clusterAlpha;
          if (viewType === "ALONG_X") {
            // Horizontal Y, Vertical Z
            tileEngineY.clusters.forEach(cy => {
              const p = mapMetricToQuad(cy.val, 0, qIdx, rect);
              ctx.fillStyle = 'rgba(45, 212, 191, 0.03)';
              ctx.fillRect(p.px - rlim * scale, rect.y, 2 * rlim * scale, rect.h);
            });
            tileEngineZ.clusters.forEach(cz => {
              const p = mapMetricToQuad(0, cz.val, qIdx, rect);
              ctx.fillStyle = 'rgba(192, 132, 252, 0.03)';
              ctx.fillRect(rect.x, p.py - rlim * scale, rect.w, 2 * rlim * scale);
            });
          } else if (viewType === "ALONG_Y") {
            // Horizontal X, Vertical Z
            tileEngineX.clusters.forEach(cx => {
              const p = mapMetricToQuad(cx.val, 0, qIdx, rect);
              ctx.fillStyle = 'rgba(56, 189, 248, 0.03)';
              ctx.fillRect(p.px - rlim * scale, rect.y, 2 * rlim * scale, rect.h);
            });
            tileEngineZ.clusters.forEach(cz => {
              const p = mapMetricToQuad(0, cz.val, qIdx, rect);
              ctx.fillStyle = 'rgba(192, 132, 252, 0.03)';
              ctx.fillRect(rect.x, p.py - rlim * scale, rect.w, 2 * rlim * scale);
            });
          } else if (viewType === "ALONG_Z" || viewType === "2D") {
            // Horizontal X, Vertical Y
            tileEngineX.clusters.forEach(cx => {
              const p = mapMetricToQuad(cx.val, 0, qIdx, rect);
              ctx.fillStyle = 'rgba(56, 189, 248, 0.03)';
              ctx.fillRect(p.px - rlim * scale, rect.y, 2 * rlim * scale, rect.h);
            });
            tileEngineY.clusters.forEach(cy => {
              const p = mapMetricToQuad(0, cy.val, qIdx, rect);
              ctx.fillStyle = 'rgba(45, 212, 191, 0.03)';
              ctx.fillRect(rect.x, p.py - rlim * scale, rect.w, 2 * rlim * scale);
            });
          }

          // Draw Joint Tuples
          let maxTupleCount = 1;
          let maxTupleSCDists = 1;
          if (showCircleMembers || showCircleSCDists) {
            jointTuplesMap.forEach(entry => {
              if (entry.count > maxTupleCount) maxTupleCount = entry.count;
              const sc = entry.scDists || entry.count;
              if (sc > maxTupleSCDists) maxTupleSCDists = sc;
            });
          }

          jointTuplesMap.forEach((entry, key) => {
            const clX = tileEngineX.clusters[entry.cx];
            const clY = tileEngineY.clusters[entry.cy];
            const clZ = currentDim >= 3 ? tileEngineZ.clusters[entry.cz] : { val: 0 };
            if (!clX || !clY || (currentDim >= 3 && !clZ)) return;

            const pr = getProjectedCoord({ x: clX.val, y: clY.val, z: clZ.val });
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);

            // 1. Proportional Circle: Member Points (Green)
            if (showCircleMembers && entry.count > 0) {
              const rMetric = rlim * Math.sqrt(entry.count / maxTupleCount);
              const rPx = Math.max(2, rMetric * scale);
              ctx.save();
              ctx.beginPath();
              ctx.arc(pos.px, pos.py, rPx, 0, Math.PI * 2);
              ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
              ctx.fill();
              ctx.strokeStyle = '#10b981';
              ctx.lineWidth = 1.6;
              ctx.stroke();
              ctx.restore();
            }

            // 2. Proportional Circle: #SC Distances (Amber)
            const scCount = entry.scDists || entry.count;
            if (showCircleSCDists && scCount > 0) {
              const rMetric = rlim * Math.sqrt(scCount / maxTupleSCDists);
              const rPx = Math.max(2, rMetric * scale);
              ctx.save();
              ctx.beginPath();
              ctx.arc(pos.px, pos.py, rPx, 0, Math.PI * 2);
              ctx.fillStyle = 'rgba(245, 158, 11, 0.08)';
              ctx.fill();
              ctx.strokeStyle = '#f59e0b';
              ctx.lineWidth = 1.6;
              ctx.setLineDash([4, 3]);
              ctx.stroke();
              ctx.restore();
            }

            ctx.beginPath();
            ctx.arc(pos.px, pos.py, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#c084fc';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1;
            ctx.stroke();
          });
          ctx.restore();
        }

        // Highlight Selected Joint Tuple
        if (selectedTupleKey && jointTuplesMap.has(selectedTupleKey)) {
          const entry = jointTuplesMap.get(selectedTupleKey);
          const clX = tileEngineX.clusters[entry.cx];
          const clY = tileEngineY.clusters[entry.cy];
          const clZ = currentDim >= 3 ? tileEngineZ.clusters[entry.cz] : { val: 0 };
          if (clX && clY && (currentDim === 2 || clZ)) {
            const pr = getProjectedCoord({ x: clX.val, y: clY.val, z: clZ.val });
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);

            ctx.save();
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, 10, 0, Math.PI * 2);
            ctx.strokeStyle = '#facc15';
            ctx.lineWidth = 2.2;
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(pos.px, pos.py, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#facc15';
            ctx.fill();
            ctx.strokeStyle = '#0f172a';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            const czPart = currentDim >= 3 ? `, ${entry.cz}` : '';
            const tagText = `Tuple (${entry.cx}, ${entry.cy}${czPart}) [${entry.count} frames]`;
            ctx.font = 'bold 11px monospace';
            const textWidth = ctx.measureText(tagText).width;
            const tagX = Math.min(pos.px + 12, rect.x + rect.w - textWidth - 14);
            const tagY = Math.max(pos.py - 12, rect.y + 20);

            ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
            ctx.strokeStyle = '#facc15';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.roundRect(tagX - 4, tagY - 12, textWidth + 8, 16, 4);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = '#facc15';
            ctx.fillText(tagText, tagX, tagY);
            ctx.restore();
          }
        }
      } else {
        // 4. Monolithic Mode: Cluster Spheres & Anchors
        if (clusterAlpha > 0.001) {
          ctx.save();
          ctx.globalAlpha = clusterAlpha;

          let maxClusterMembers = 1;
          let maxClusterSCDists = 1;
          let maxInfoGain = 0.001;
          if (showCircleMembers || showCircleSCDists || showEntropyMap) {
            clusters.forEach(cl => {
              if (cl.members > maxClusterMembers) maxClusterMembers = cl.members;
              const sc = cl.scDists || 0;
              if (sc > maxClusterSCDists) maxClusterSCDists = sc;
              if (cl.infoGain && cl.infoGain > maxInfoGain) maxInfoGain = cl.infoGain;
            });
          }

          clusters.forEach(c => {
            const pr = getProjectedCoord(c);
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);
            const rPx = rlim * scale;

            // 3. Spatial Information Gain Heatmap (Entropy Reduction Map)
            if (showEntropyMap && c.infoGain !== undefined && c.infoGain > 0) {
              const gainNorm = Math.min(1.0, Math.max(0.05, c.infoGain / maxInfoGain));
              const rPxGain = Math.max(8, rlim * scale * (0.6 + 0.8 * gainNorm));
              ctx.save();

              // Radial gradient halo: violet -> cyan -> emerald -> gold
              const grad = ctx.createRadialGradient(pos.px, pos.py, 2, pos.px, pos.py, rPxGain);
              if (gainNorm > 0.75) {
                grad.addColorStop(0.0, 'rgba(250, 204, 21, 0.45)');
                grad.addColorStop(0.5, 'rgba(74, 222, 128, 0.25)');
                grad.addColorStop(1.0, 'rgba(15, 23, 42, 0.0)');
              } else if (gainNorm > 0.40) {
                grad.addColorStop(0.0, 'rgba(74, 222, 128, 0.35)');
                grad.addColorStop(0.6, 'rgba(56, 189, 248, 0.18)');
                grad.addColorStop(1.0, 'rgba(15, 23, 42, 0.0)');
              } else {
                grad.addColorStop(0.0, 'rgba(56, 189, 248, 0.30)');
                grad.addColorStop(0.7, 'rgba(192, 132, 252, 0.12)');
                grad.addColorStop(1.0, 'rgba(15, 23, 42, 0.0)');
              }
              ctx.beginPath();
              ctx.arc(pos.px, pos.py, rPxGain, 0, Math.PI * 2);
              ctx.fillStyle = grad;
              ctx.fill();

              ctx.strokeStyle = (gainNorm > 0.75) ? '#facc15' : ((gainNorm > 0.4) ? '#4ade80' : '#38bdf8');
              ctx.lineWidth = 1.4;
              ctx.setLineDash([2, 3]);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.restore();

              // Mini Info Gain tag
              ctx.save();
              ctx.font = 'bold 9px monospace';
              ctx.fillStyle = (gainNorm > 0.75) ? '#facc15' : '#38bdf8';
              ctx.fillText(`+${c.infoGain.toFixed(2)}b`, pos.px - 14, pos.py + 16);
              ctx.restore();
            }

            if (showClusterRadii && !showCircleMembers && !showCircleSCDists) {
              if (!is3DCustom) {
                // 2D Circle outline for receptive field
                ctx.beginPath();
                ctx.arc(pos.px, pos.py, rPx, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(56, 189, 248, 0.04)';
                ctx.fill();
                ctx.strokeStyle = c.color;
                ctx.lineWidth = 1.2;
                ctx.setLineDash([3, 3]);
                ctx.stroke();
                ctx.setLineDash([]);
              } else {
                // 3D Custom: 3 Orthogonal Wireframe Rings (perpendicular to Z, Y, X axes)
                const az = orbitCamera.azimuth, el = orbitCamera.elevation;
                const p0 = (typeof getPlotCoords === 'function') ? getPlotCoords(c) : c;
                const cx = p0.x, cy = p0.y, cz = p0.z;
                
                ctx.save();
                ctx.strokeStyle = c.color;
                ctx.globalAlpha = clusterAlpha * 0.35; // Fainter, subtle wireframe
                ctx.lineWidth = 0.75;

                // 1. Circle perpendicular to Z-axis (XY plane)
                ctx.beginPath();
                for (let s = 0; s <= CIRCLE_LUT_STEPS; s++) {
                  const rx = cx + rlim * CIRCLE_COS[s];
                  const ry = cy + rlim * CIRCLE_SIN[s];
                  const rz = cz;
                  const rp = project3D(rx, ry, rz, az, el);
                  const ppos = mapMetricToQuad(rp.u, rp.v, qIdx, rect);
                  if (s === 0) ctx.moveTo(ppos.px, ppos.py);
                  else ctx.lineTo(ppos.px, ppos.py);
                }
                ctx.stroke();

                // 2. Circle perpendicular to Y-axis (XZ plane)
                ctx.beginPath();
                for (let s = 0; s <= CIRCLE_LUT_STEPS; s++) {
                  const rx = cx + rlim * CIRCLE_COS[s];
                  const ry = cy;
                  const rz = cz + rlim * CIRCLE_SIN[s];
                  const rp = project3D(rx, ry, rz, az, el);
                  const ppos = mapMetricToQuad(rp.u, rp.v, qIdx, rect);
                  if (s === 0) ctx.moveTo(ppos.px, ppos.py);
                  else ctx.lineTo(ppos.px, ppos.py);
                }
                ctx.stroke();

                // 3. Circle perpendicular to X-axis (YZ plane)
                ctx.beginPath();
                for (let s = 0; s <= CIRCLE_LUT_STEPS; s++) {
                  const rx = cx;
                  const ry = cy + rlim * CIRCLE_COS[s];
                  const rz = cz + rlim * CIRCLE_SIN[s];
                  const rp = project3D(rx, ry, rz, az, el);
                  const ppos = mapMetricToQuad(rp.u, rp.v, qIdx, rect);
                  if (s === 0) ctx.moveTo(ppos.px, ppos.py);
                  else ctx.lineTo(ppos.px, ppos.py);
                }
                ctx.stroke();
                ctx.restore();
              }
            }

            // 1. Proportional Area Circle: Points in Cluster (Emerald Green #10b981)
            // Area within circle is strictly proportional to c.members
            if (showCircleMembers && c.members > 0) {
              const rMetricMem = rlim * Math.sqrt(c.members / maxClusterMembers);
              const rPxMem = Math.max(2, rMetricMem * scale);

              ctx.save();
              ctx.beginPath();
              ctx.arc(pos.px, pos.py, rPxMem, 0, Math.PI * 2);
              ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
              ctx.fill();
              ctx.strokeStyle = '#10b981';
              ctx.lineWidth = 1.6;
              ctx.setLineDash([]);
              ctx.stroke();
              ctx.restore();
            }

            // 2. Proportional Area Circle: #SC Distances (Amber #f59e0b)
            // Area within circle is strictly proportional to c.scDists
            const scCount = c.scDists || 0;
            if (showCircleSCDists && scCount > 0) {
              const rMetricDist = rlim * Math.sqrt(scCount / maxClusterSCDists);
              const rPxDist = Math.max(2, rMetricDist * scale);

              ctx.save();
              ctx.beginPath();
              ctx.arc(pos.px, pos.py, rPxDist, 0, Math.PI * 2);
              ctx.fillStyle = 'rgba(245, 158, 11, 0.08)';
              ctx.fill();
              ctx.strokeStyle = '#f59e0b';
              ctx.lineWidth = 1.6;
              ctx.setLineDash([4, 3]);
              ctx.stroke();
              ctx.restore();
            }
          });

          // Draw Learned Markov Transition Vector Arcs (-tm) (Optimized Batched)
          if (showTransitionLines && useTM && transitionCounts.length > 0 && clusters.length > 1) {
            ctx.save();
            const K = clusters.length;

            // 1. Draw Active Frame Transition (Gold Beam)
            if (lastTransitionFrom >= 0 && lastTransitionTo >= 0 && 
                lastTransitionFrom < K && lastTransitionTo < K && 
                lastTransitionFrom !== lastTransitionTo && 
                clusters[lastTransitionFrom] && clusters[lastTransitionTo]) {
              ctx.save();
              if (typeof currentEvaluationsAlpha === 'number') {
                ctx.globalAlpha = Math.max(0.0, Math.min(1.0, currentEvaluationsAlpha));
              }
              const prA = getProjectedCoord(clusters[lastTransitionFrom]);
              const prB = getProjectedCoord(clusters[lastTransitionTo]);
              const posA = mapMetricToQuad(prA.u, prA.v, qIdx, rect);
              const posB = mapMetricToQuad(prB.u, prB.v, qIdx, rect);

              ctx.beginPath();
              ctx.moveTo(posA.px, posA.py);
              ctx.lineTo(posB.px, posB.py);
              ctx.strokeStyle = '#fbbf24';
              ctx.lineWidth = 2.5;
              ctx.stroke();
              ctx.restore();
            }

            // 2. Batched Background Learned Paths (Top paths only, single stroke)
            if (topLearnedPathsCache && topLearnedPathsCache.length > 0) {
              ctx.beginPath();
              const maxPaths = Math.min(8, topLearnedPathsCache.length);
              for (let pIdx = 0; pIdx < maxPaths; pIdx++) {
                const p = topLearnedPathsCache[pIdx];
                if (p.from === p.to || p.from >= K || p.to >= K) continue;
                if (p.from === lastTransitionFrom && p.to === lastTransitionTo) continue;
                const prA = getProjectedCoord(clusters[p.from]);
                const prB = getProjectedCoord(clusters[p.to]);
                const posA = mapMetricToQuad(prA.u, prA.v, qIdx, rect);
                const posB = mapMetricToQuad(prB.u, prB.v, qIdx, rect);
                ctx.moveTo(posA.px, posA.py);
                ctx.lineTo(posB.px, posB.py);
              }
              ctx.strokeStyle = 'rgba(192, 132, 252, 0.35)';
              ctx.lineWidth = 1.2;
              ctx.stroke();
            }
            ctx.restore();
          }

          clusters.forEach(c => {
            const pr = getProjectedCoord(c);
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);

            // Radius scales slightly with cluster membership points
            const memberCount = (typeof c.members === 'number')
              ? c.members : (c.members ? c.members.length : 1);
            const nodeRadius = Math.min(8.0, 5.0 + Math.sqrt(Math.max(0, memberCount)) * 0.2);

            // Radiant Bloom / Glow aura around anchor
            ctx.save();
            ctx.shadowColor = c.color;
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, nodeRadius, 0, Math.PI * 2);
            ctx.fillStyle = c.color;
            ctx.fill();
            ctx.restore();

            // Anchor Core Node Outline
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, nodeRadius, 0, Math.PI * 2);
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.3;
            ctx.stroke();

            if (showClusterLabels) {
              ctx.fillStyle = '#f8fafc';
              ctx.font = 'bold 10px sans-serif';
              ctx.fillText(`C${c.id}`, pos.px + nodeRadius + 3, pos.py - 5);
            }

            if (typeof showAnchorSparklines !== 'undefined' &&
                showAnchorSparklines && currentDim > 3) {
              if (typeof HighDEngine !== 'undefined' && HighDEngine.renderAnchorSparkline) {
                HighDEngine.renderAnchorSparkline(ctx, pos.px, pos.py, c, c.color, currentDim);
              }
            }
          });

          // Pruned Crosshairs
          if (showPrunedMarks && currentPruned && currentPruned.length > 0) {
            ctx.save();
            if (typeof currentEvaluationsAlpha === 'number') {
              ctx.globalAlpha = Math.max(0.0, Math.min(1.0, currentEvaluationsAlpha));
            }
            currentPruned.forEach(p => {
              const pr = getProjectedCoord(p.cluster);
              const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);

              ctx.strokeStyle = '#ef4444';
              ctx.lineWidth = 1.8;
              ctx.beginPath();
              ctx.moveTo(pos.px - 6, pos.py - 6);
              ctx.lineTo(pos.px + 6, pos.py + 6);
              ctx.moveTo(pos.px + 6, pos.py - 6);
              ctx.lineTo(pos.px - 6, pos.py + 6);
              ctx.stroke();
            });
            ctx.restore();
          }
          ctx.restore();
        }

        // Helper to draw highlighted cluster (selected in gold #facc15, hovered in cyan #38bdf8)
        function renderClusterHighlight(clObj, color, isSelected) {
          const pr = getProjectedCoord(clObj);
          const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);
          const rPx = rlim * scale;

          ctx.save();
          if (!is3DCustom) {
            // Glowing receptive circle
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, rPx, 0, Math.PI * 2);
            ctx.fillStyle = isSelected ? 'rgba(250, 204, 21, 0.16)' : 'rgba(56, 189, 248, 0.16)';
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = isSelected ? 2.4 : 2.0;
            ctx.stroke();

            // Outer dashed accent ring
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, rPx + 4, 0, Math.PI * 2);
            ctx.strokeStyle = isSelected ? 'rgba(250, 204, 21, 0.6)' : 'rgba(56, 189, 248, 0.6)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
          } else {
            // 3D Custom: 3 Highlighted Equatorial Rings & halo
            const az = orbitCamera.azimuth, el = orbitCamera.elevation;
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, rPx, 0, Math.PI * 2);
            ctx.fillStyle = isSelected ? 'rgba(250, 204, 21, 0.08)' : 'rgba(56, 189, 248, 0.08)';
            ctx.fill();

            // 1. Highlight XY ring (perpendicular to Z)
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            for (let a = 0; a <= Math.PI * 2 + 0.05; a += Math.PI / 16) {
              const rx = clObj.x + rlim * Math.cos(a);
              const ry = clObj.y + rlim * Math.sin(a);
              const rz = clObj.z;
              const rp = project3D(rx, ry, rz, az, el);
              const ppos = mapMetricToQuad(rp.u, rp.v, qIdx, rect);
              if (a === 0) ctx.moveTo(ppos.px, ppos.py);
              else ctx.lineTo(ppos.px, ppos.py);
            }
            ctx.stroke();

            // 2. Highlight XZ ring (perpendicular to Y)
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            for (let a = 0; a <= Math.PI * 2 + 0.05; a += Math.PI / 16) {
              const rx = clObj.x + rlim * Math.cos(a);
              const ry = clObj.y;
              const rz = clObj.z + rlim * Math.sin(a);
              const rp = project3D(rx, ry, rz, az, el);
              const ppos = mapMetricToQuad(rp.u, rp.v, qIdx, rect);
              if (a === 0) ctx.moveTo(ppos.px, ppos.py);
              else ctx.lineTo(ppos.px, ppos.py);
            }
            ctx.stroke();

            // 3. Highlight YZ ring (perpendicular to X)
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            for (let a = 0; a <= Math.PI * 2 + 0.05; a += Math.PI / 16) {
              const rx = clObj.x;
              const ry = clObj.y + rlim * Math.cos(a);
              const rz = clObj.z + rlim * Math.sin(a);
              const rp = project3D(rx, ry, rz, az, el);
              const ppos = mapMetricToQuad(rp.u, rp.v, qIdx, rect);
              if (a === 0) ctx.moveTo(ppos.px, ppos.py);
              else ctx.lineTo(ppos.px, ppos.py);
            }
            ctx.stroke();
          }

          // Highlight Anchor Node with Target Reticle
          ctx.beginPath();
          ctx.arc(pos.px, pos.py, 10, 0, Math.PI * 2);
          ctx.strokeStyle = color;
          ctx.lineWidth = 2.0;
          ctx.stroke();

          // Crosshair tick marks
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.moveTo(pos.px - 14, pos.py); ctx.lineTo(pos.px - 9, pos.py);
          ctx.moveTo(pos.px + 9, pos.py); ctx.lineTo(pos.px + 14, pos.py);
          ctx.moveTo(pos.px, pos.py - 14); ctx.lineTo(pos.px, pos.py - 9);
          ctx.moveTo(pos.px, pos.py + 9); ctx.lineTo(pos.px, pos.py + 14);
          ctx.stroke();

          // Center core
          ctx.beginPath();
          ctx.arc(pos.px, pos.py, 6, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 1.5;
          ctx.stroke();

          // Callout Badge with short name C0, coordinates and member count
          const zCoord = currentDim >= 3 ? `, ${clObj.z.toFixed(2)}` : '';
          const tagText =
            `C${clObj.id} [${clObj.members}f] ` +
            `(${clObj.x.toFixed(2)}, ${clObj.y.toFixed(2)}${zCoord})`;
          ctx.font = 'bold 11px monospace';
          const textWidth = ctx.measureText(tagText).width;
          const tagX = Math.min(pos.px + 12, rect.x + rect.w - textWidth - 14);
          const tagY = Math.max(pos.py - 14, rect.y + 20);

          ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.roundRect(tagX - 4, tagY - 12, textWidth + 8, 16, 4);
          ctx.fill();
          ctx.stroke();

          ctx.fillStyle = color;
          ctx.fillText(tagText, tagX, tagY);
          ctx.restore();
        }

        // Draw Hovered Cluster (Cyan)
        if (hoveredClusterId !== -1 && hoveredClusterId !== selectedClusterId) {
          const hovC = clusters.find(c => c.id === hoveredClusterId);
          if (hovC) renderClusterHighlight(hovC, '#38bdf8', false);
        }

        // Draw Selected Cluster (Gold)
        if (selectedClusterId !== -1) {
          const selC = clusters.find(c => c.id === selectedClusterId);
          if (selC) renderClusterHighlight(selC, '#facc15', true);
        }

        // Draw Inspected / Hovered Past Sample Point
        const targetSample = hoveredSampleTracePoint || 
          (selectedSampleTraceIndex >= 0 ? sampleTraceLog.find(e => e.frameIndex === selectedSampleTraceIndex) : null);
        if (targetSample) {
          const ptObj = targetSample.point || targetSample;
          const prP = getProjectedCoord(ptObj);
          const posP = mapMetricToQuad(prP.u, prP.v, qIdx, rect);

          ctx.save();
          // Outer glowing ring
          ctx.beginPath();
          ctx.arc(posP.px, posP.py, 7, 0, Math.PI * 2);
          ctx.strokeStyle = '#facc15';
          ctx.lineWidth = 2.0;
          ctx.stroke();

          // Center solid point
          ctx.beginPath();
          ctx.arc(posP.px, posP.py, 3, 0, Math.PI * 2);
          ctx.fillStyle = '#facc15';
          ctx.fill();

          // Crosshairs
          ctx.beginPath();
          ctx.moveTo(posP.px - 10, posP.py); ctx.lineTo(posP.px + 10, posP.py);
          ctx.moveTo(posP.px, posP.py - 10); ctx.lineTo(posP.px, posP.py + 10);
          ctx.strokeStyle = 'rgba(250, 204, 21, 0.85)';
          ctx.lineWidth = 1.2;
          ctx.stroke();

          // Label
          if (showDistLabels) {
            ctx.font = 'bold 9px monospace';
            ctx.fillStyle = '#facc15';
            const fIdx = targetSample.frameIndex || 0;
            ctx.fillText(`Point #${fIdx}`, posP.px + 9, posP.py - 5);
          }
          ctx.restore();
        }

        // Draw Hovered / Locked Closest Sample Point Highlight
        const activeSampleHighlight = (typeof lockedClosestSample !== 'undefined' && lockedClosestSample)
          ? lockedClosestSample
          : (highlightClosestSample ? hoveredClosestSample : null);

        if (activeSampleHighlight && activeSampleHighlight.point) {
          const pt = activeSampleHighlight.point;
          const pr = getProjectedCoord(pt);
          const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);

          if (pos.px >= rect.x - 20 && pos.px <= rect.x + rect.w + 20 &&
              pos.py >= rect.y - 20 && pos.py <= rect.y + rect.h + 20) {
            ctx.save();

            const isLocked = (typeof lockedClosestSample !== 'undefined' && lockedClosestSample && lockedClosestSample.index === activeSampleHighlight.index);
            const cId = activeSampleHighlight.clusterId;
            const sampleIdx = activeSampleHighlight.index;

            // Frame point lookup helper (ensures no lines to non-existing points)
            function getFramePoint(idx) {
              if (typeof pastSamples !== 'undefined' && pastSamples && pastSamples.length > 0) {
                if (idx >= 0 && idx < pastSamples.length && pastSamples[idx] && pastSamples[idx].frameIndex === idx) {
                  return pastSamples[idx];
                }
                const found = pastSamples.find(p => p.frameIndex === idx);
                if (found) return found;
                if (idx >= 0 && idx < pastSamples.length) return pastSamples[idx];
              }
              if (typeof benchmarkDataset !== 'undefined' && benchmarkDataset && idx >= 0 && idx < benchmarkDataset.length) {
                return benchmarkDataset[idx];
              }
              return null;
            }

            // Helper to format ordinal numbers (1st, 2nd, 3rd, 4th...)
            function getOrdinalSuffix(num) {
              const j = num % 10, k = num % 100;
              if (j === 1 && k !== 11) return num + "st";
              if (j === 2 && k !== 12) return num + "nd";
              if (j === 3 && k !== 13) return num + "rd";
              return num + "th";
            }

            // Helper to draw pill labels
            function drawDistPill(px, py, text, color, isHighlight) {
              ctx.font = 'bold 9px monospace';
              const pillW = ctx.measureText(text).width + 8;
              const pillH = 14;
              const clPx = Math.max(rect.x + 4, Math.min(rect.x + rect.w - pillW - 4, px - pillW / 2));
              const clPy = Math.max(rect.y + 12, Math.min(rect.y + rect.h - 4, py - pillH / 2));

              ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
              ctx.strokeStyle = color;
              ctx.lineWidth = isHighlight ? 1.4 : 1.0;
              ctx.beginPath();
              ctx.roundRect(clPx, clPy, pillW, pillH, 3);
              ctx.fill();
              ctx.stroke();

              ctx.fillStyle = color;
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(text, clPx + pillW / 2, clPy + pillH / 2 + 0.5);
            }

            // -----------------------------------------------------------------
            // 1. Distance Computations that were part of SOLVING FOR THIS POINT
            //    (Thick Green for Matches d <= rlim, Thick Red for Mismatches d > rlim)
            //    Displaying explicit chronological order: 1st, 2nd, 3rd...
            // -----------------------------------------------------------------

            // A. Sample-to-Cluster evaluations recorded when solving this point
            const evaluatedClusters = [];
            if (typeof frameEvaluationsLog !== 'undefined' && frameEvaluationsLog && frameEvaluationsLog[sampleIdx]) {
              frameEvaluationsLog[sampleIdx].forEach(ev => {
                const targetCId = typeof ev.clusterId === 'number' ? ev.clusterId : (ev.target && ev.target.id);
                if (targetCId >= 0 && targetCId < clusters.length && clusters[targetCId]) {
                  evaluatedClusters.push({
                    cluster: clusters[targetCId],
                    dist: ev.dist,
                    match: ev.match
                  });
                }
              });
            } else {
              const traceEntry = (typeof sampleTraceLog !== 'undefined' && sampleTraceLog)
                ? sampleTraceLog.find(e => e.frameIndex === sampleIdx + 1 || e.frameIndex === sampleIdx)
                : null;
              if (traceEntry && traceEntry.evaluations && traceEntry.evaluations.length > 0) {
                traceEntry.evaluations.forEach(ev => {
                  const targetCId = typeof ev.clusterId === 'number' ? ev.clusterId : (ev.target && ev.target.id);
                  if (targetCId >= 0 && targetCId < clusters.length && clusters[targetCId]) {
                    evaluatedClusters.push({
                      cluster: clusters[targetCId],
                      dist: ev.dist,
                      match: ev.match
                    });
                  }
                });
              }
            }

            // If no evaluation list in logs but assigned cluster exists, include assigned match
            const hasAssigned = (evaluatedClusters.length === 0 && cId >= 0 &&
              clusters && cId < clusters.length && clusters[cId]);
            if (hasAssigned) {
              const clObj = clusters[cId];
              const dx = pt.x - clObj.x;
              const dy = pt.y - clObj.y;
              const dz = currentDim >= 3 ? (pt.z - clObj.z) : 0;
              const dVal = Math.sqrt(dx * dx + dy * dy + dz * dz);
              evaluatedClusters.push({
                cluster: clObj,
                dist: dVal,
                match: dVal <= (rlim || 0.1)
              });
            }

            // Draw Sample-to-Cluster solving distance lines with chronological sequence order
            evaluatedClusters.forEach((ev, evIdx) => {
              const clObj = ev.cluster;
              const prC = getProjectedCoord(clObj);
              const posC = mapMetricToQuad(prC.u, prC.v, qIdx, rect);
              const lineColor = ev.match ? '#4ade80' : '#ef4444';
              const orderSuffix = getOrdinalSuffix(evIdx + 1);

              if (showDistLines) {
                ctx.beginPath();
                ctx.moveTo(pos.px, pos.py);
                ctx.lineTo(posC.px, posC.py);
                ctx.strokeStyle = lineColor;
                ctx.lineWidth = ev.match ? 2.4 : 1.6;
                if (!ev.match) ctx.setLineDash([4, 3]);
                ctx.stroke();
                ctx.setLineDash([]);

                // Highlight cluster anchor node
                ctx.beginPath();
                ctx.arc(posC.px, posC.py, ev.match ? 7 : 5, 0, Math.PI * 2);
                ctx.strokeStyle = lineColor;
                ctx.lineWidth = 1.6;
                ctx.stroke();
              }

              if (showDistLabels) {
                // Step computation order badge on cluster anchor (e.g. 1, 2, 3)
                ctx.beginPath();
                ctx.arc(posC.px + 9, posC.py - 9, 6.5, 0, Math.PI * 2);
                ctx.fillStyle = lineColor;
                ctx.fill();
                ctx.strokeStyle = '#0f172a';
                ctx.lineWidth = 1.0;
                ctx.stroke();

                ctx.fillStyle = '#0f172a';
                ctx.font = 'bold 8.5px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(`${evIdx + 1}`, posC.px + 9, posC.py - 8.5);

                // Distance pill along line (at 52%-72% towards cluster, staggered if multiple)
                const staggerRatio = 0.52 + Math.min(0.20, evIdx * 0.08);
                const midX = pos.px * (1 - staggerRatio) + posC.px * staggerRatio;
                const midY = pos.py * (1 - staggerRatio) + posC.py * staggerRatio;
                const mark = ev.match ? '✓' : '✗';
                const pillText = `${orderSuffix}: C${clObj.id} (d=${ev.dist.toFixed(3)}) ${mark}`;
                drawDistPill(midX, midY, pillText, lineColor, ev.match);
              }
            });

            // B. Sample-to-Sample nearest neighbor distances computed when solving FOR this point
            // (Distinct Vibrant Violet / Magenta palette #c084fc / #e879f9)
            if (showKnnLines &&
                typeof knnResults !== 'undefined' && knnResults && knnResults.indices) {
              const N = knnResults.totalFrames;
              const k = knnResults.k;
              let targetLookupIdx = sampleIdx;
              if (targetLookupIdx >= N && sampleIdx > 0 && (sampleIdx - 1) < N) {
                targetLookupIdx = sampleIdx - 1;
              }
              if (targetLookupIdx >= 0 && targetLookupIdx < N) {
                for (let r = 0; r < k; r++) {
                  const nId = knnResults.indices[targetLookupIdx * k + r];
                  const dist = knnResults.distances[targetLookupIdx * k + r];
                  if (nId >= 0 && nId !== targetLookupIdx) {
                    const nPt = getFramePoint(nId);
                    if (!nPt) continue; // Ensure no lines to non-existing points

                    const prN = getProjectedCoord(nPt);
                    const posN = mapMetricToQuad(prN.u, prN.v, qIdx, rect);
                    const isRank1 = (r === 0);
                    const nnColor = isRank1 ? '#e879f9' : '#c084fc';
                    // Glowing dual-tone laser gradient ray
                    const grad = ctx.createLinearGradient(pos.px, pos.py, posN.px, posN.py);
                    grad.addColorStop(0, '#38bdf8');
                    grad.addColorStop(1, nnColor);

                    ctx.save();
                    ctx.shadowColor = nnColor;
                    ctx.shadowBlur = isRank1 ? 8 : 4;
                    ctx.beginPath();
                    ctx.moveTo(pos.px, pos.py);
                    ctx.lineTo(posN.px, posN.py);
                    ctx.strokeStyle = grad;
                    ctx.lineWidth = isRank1 ? 2.6 : 1.8;
                    ctx.stroke();
                    ctx.restore();

                    // Highlight neighbor point node
                    ctx.beginPath();
                    ctx.arc(posN.px, posN.py, isRank1 ? 5.0 : 4.0, 0, Math.PI * 2);
                    ctx.fillStyle = nnColor;
                    ctx.fill();
                    ctx.strokeStyle = '#0f172a';
                    ctx.lineWidth = 1.2;
                    ctx.stroke();

                    if (showDistLabels) {
                      // Distance pill along ray (at 50% midpoint)
                      const midX = (pos.px + posN.px) / 2;
                      const midY = (pos.py + posN.py) / 2;
                      const pillText = `NN: #${nId} (d=${dist.toFixed(3)})`;
                      drawDistPill(midX, midY, pillText, nnColor, isRank1);
                    }
                  }
                }
              }
            }

            // -----------------------------------------------------------------
            // 2. OTHER Distance Computations INVOLVING THIS POINT
            //    (Thinner dashed lines in distinct amber/orange)
            // -----------------------------------------------------------------
            if (showKnnLines &&
                typeof knnResults !== 'undefined' && knnResults && knnResults.indices) {
              const N = knnResults.totalFrames;
              const k = knnResults.k;
              const maxIncoming = 12; // Cap display of incoming queries for visual clarity
              let incomingCount = 0;

              for (let m = 0; m < N && incomingCount < maxIncoming; m++) {
                if (m === sampleIdx) continue;
                for (let r = 0; r < k; r++) {
                  if (knnResults.indices[m * k + r] === sampleIdx) {
                    const dist = knnResults.distances[m * k + r];
                    const mPt = getFramePoint(m);
                    if (!mPt) continue;

                    const prM = getProjectedCoord(mPt);
                    const posM = mapMetricToQuad(prM.u, prM.v, qIdx, rect);

                    // Thinner line in distinct amber / orange color
                    const otherColor = (incomingCount % 2 === 0) ? '#f59e0b' : '#fb923c';

                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(pos.px, pos.py);
                    ctx.lineTo(posM.px, posM.py);
                    ctx.strokeStyle = otherColor;
                    ctx.lineWidth = 1.2;
                    ctx.globalAlpha = 0.75;
                    ctx.setLineDash([3, 3]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.globalAlpha = 1.0;

                    // Highlight other query node
                    ctx.beginPath();
                    ctx.arc(posM.px, posM.py, 3.2, 0, Math.PI * 2);
                    ctx.fillStyle = otherColor;
                    ctx.fill();
                    ctx.strokeStyle = '#0f172a';
                    ctx.lineWidth = 0.8;
                    ctx.stroke();

                    if (showDistLabels) {
                      // Mini distance pill at 35% towards source query
                      const midX = pos.px * 0.65 + posM.px * 0.35;
                      const midY = pos.py * 0.65 + posM.py * 0.35;
                      const pillText = `#${m}→#${sampleIdx}: d=${dist.toFixed(3)}`;
                      drawDistPill(midX, midY, pillText, otherColor, false);
                    }
                    ctx.restore();

                    incomingCount++;
                    break;
                  }
                }
              }
            }

            const reticleColor = isLocked ? '#fbbf24' : '#38bdf8';
            const reticleGlow = isLocked ? 'rgba(251, 191, 36, 0.28)' : 'rgba(56, 189, 248, 0.18)';

            // 3. Outer glowing reticle ring
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, isLocked ? 10 : 8, 0, Math.PI * 2);
            ctx.fillStyle = reticleGlow;
            ctx.fill();
            ctx.strokeStyle = reticleColor;
            ctx.lineWidth = isLocked ? 2.2 : 2.0;
            ctx.stroke();

            // 4. Inner solid center core
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, 3.5, 0, Math.PI * 2);
            ctx.fillStyle = isLocked ? '#fef08a' : '#f8fafc';
            ctx.fill();
            ctx.strokeStyle = '#0f172a';
            ctx.lineWidth = 1.2;
            ctx.stroke();

            // 5. Crosshair reticle ticks
            const tickR = isLocked ? 16 : 13;
            const tickIn = isLocked ? 11 : 9;
            ctx.beginPath();
            ctx.moveTo(pos.px - tickR, pos.py); ctx.lineTo(pos.px - tickIn, pos.py);
            ctx.moveTo(pos.px + tickIn, pos.py);  ctx.lineTo(pos.px + tickR, pos.py);
            ctx.moveTo(pos.px, pos.py - tickR); ctx.lineTo(pos.px, pos.py - tickIn);
            ctx.moveTo(pos.px, pos.py + tickIn);  ctx.lineTo(pos.px, pos.py + tickR);
            ctx.strokeStyle = reticleColor;
            ctx.lineWidth = isLocked ? 2.0 : 1.6;
            ctx.stroke();

            // 32-Bar Equalizer HUD for Hovered / Locked Sample
            // (Render once at top-left to avoid toolbar collision)
            const shouldDrawEqualizer = (maximizedQuad !== null)
              ? (qIdx === maximizedQuad)
              : (qIdx === 0);
            if (shouldDrawEqualizer && currentDim > 3 &&
                typeof HighDEngine !== 'undefined' && HighDEngine.renderEqualizerHUD) {
              const hudX = rect.x + 10;
              const hudY = rect.y + 26;
              const label = isLocked ? `Sample #${sampleIdx} (Locked)` : `Sample #${sampleIdx}`;
              HighDEngine.renderEqualizerHUD(
                ctx, hudX, hudY, pt, label, reticleColor, currentDim
              );
            }

            ctx.restore();
          }
        }

        // 32-Bar Equalizer HUD for Hovered Cluster (render once at top-left)
        const shouldDrawClusterEqualizer = (maximizedQuad !== null)
          ? (qIdx === maximizedQuad)
          : (qIdx === 0);
        if (shouldDrawClusterEqualizer && currentDim > 3 && hoveredClusterId !== -1 &&
            (!activeSampleHighlight || !activeSampleHighlight.point)) {
          const hovCl = clusters.find(c => c.id === hoveredClusterId);
          if (hovCl && typeof HighDEngine !== 'undefined' && HighDEngine.renderEqualizerHUD) {
            const hudX = rect.x + 10;
            const hudY = rect.y + 26;
            const cnt = typeof hovCl.members === 'number'
              ? hovCl.members
              : (hovCl.members ? hovCl.members.length : 1);
            const clLabel = `Cluster C${hovCl.id} (N=${cnt})`;
            HighDEngine.renderEqualizerHUD(
              ctx, hudX, hudY, hovCl, clLabel, hovCl.color || '#38bdf8', currentDim
            );
          }
        }

        // Draw Hovered Transition Matrix Vector Arrow (from C_i -> C_j)
        if (hoveredTMCell && hoveredTMCell.i >= 0 && hoveredTMCell.j >= 0 &&
            hoveredTMCell.i < clusters.length && hoveredTMCell.j < clusters.length &&
            clusters[hoveredTMCell.i] && clusters[hoveredTMCell.j]) {
          const fromId = hoveredTMCell.i;
          const toId = hoveredTMCell.j;
          const clA = clusters[fromId];
          const clB = clusters[toId];
          const prA = getProjectedCoord(clA);
          const prB = getProjectedCoord(clB);
          const posA = mapMetricToQuad(prA.u, prA.v, qIdx, rect);
          const posB = mapMetricToQuad(prB.u, prB.v, qIdx, rect);

          ctx.save();

          // Highlight source C_i node
          ctx.beginPath();
          ctx.arc(posA.px, posA.py, 10, 0, Math.PI * 2);
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 2.0;
          ctx.setLineDash([3, 3]);
          ctx.stroke();

          // Highlight target C_j node
          ctx.beginPath();
          ctx.arc(posB.px, posB.py, 11, 0, Math.PI * 2);
          ctx.strokeStyle = '#facc15';
          ctx.lineWidth = 2.5;
          ctx.setLineDash([]);
          ctx.stroke();

          // Probability & count stats
          let rowSum = 0;
          if (transitionCounts[fromId]) {
            for (let k = 0; k < clusters.length; k++) rowSum += transitionCounts[fromId][k] || 0;
          }
          const cnt = (transitionCounts[fromId] && transitionCounts[fromId][toId]) || 0;
          const prob = rowSum > 0 ? (cnt / rowSum) : 0.0;

          if (fromId === toId) {
            // Self-loop transition: draw smooth circular return loop
            const loopR = 18;
            const loopCx = posA.px;
            const loopCy = posA.py - loopR - 6;

            if (showTransitionLines) {
              ctx.beginPath();
              ctx.arc(loopCx, loopCy, loopR, 0.25 * Math.PI, 1.75 * Math.PI, false);
              ctx.strokeStyle = '#facc15';
              ctx.lineWidth = 3.0;
              ctx.shadowColor = '#facc15';
              ctx.shadowBlur = 8;
              ctx.stroke();

              // Arrowhead on loop
              const headX = loopCx + loopR * Math.cos(0.25 * Math.PI);
              const headY = loopCy + loopR * Math.sin(0.25 * Math.PI);
              const angle = 0.25 * Math.PI + Math.PI / 2;
              const headLen = 10;
              ctx.fillStyle = '#facc15';
              ctx.beginPath();
              ctx.moveTo(headX, headY);
              ctx.lineTo(headX - headLen * Math.cos(angle - Math.PI / 6),
                         headY - headLen * Math.sin(angle - Math.PI / 6));
              ctx.lineTo(headX - headLen * Math.cos(angle + Math.PI / 6),
                         headY - headLen * Math.sin(angle + Math.PI / 6));
              ctx.closePath();
              ctx.fill();
            }

            if (showDistLabels) {
              // Label badge
              const badgeText = `C${fromId} ↺ C${toId} [${(prob * 100).toFixed(1)}%, N=${cnt}]`;
              ctx.font = 'bold 10px monospace';
              const tw = ctx.measureText(badgeText).width;
              const badgeX = Math.max(rect.x + 10,
                                      Math.min(rect.x + rect.w - tw - 10, loopCx - tw / 2));
              const badgeY = Math.max(rect.y + 16, loopCy - loopR - 6);

              ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
              ctx.strokeStyle = '#facc15';
              ctx.lineWidth = 1.2;
              ctx.beginPath();
              ctx.roundRect(badgeX - 4, badgeY - 11, tw + 8, 15, 3);
              ctx.fill();
              ctx.stroke();

              ctx.fillStyle = '#facc15';
              ctx.textAlign = 'left';
              ctx.fillText(badgeText, badgeX, badgeY);
            }
          } else {
            // Directed straight vector arrow
            const dx = posB.px - posA.px;
            const dy = posB.py - posA.py;
            const dist = Math.hypot(dx, dy);
            const angle = Math.atan2(dy, dx);

            const startOffset = Math.min(8, dist * 0.15);
            const endOffset = Math.min(10, dist * 0.2);
            const startX = posA.px + startOffset * Math.cos(angle);
            const startY = posA.py + startOffset * Math.sin(angle);
            const endX = posB.px - endOffset * Math.cos(angle);
            const endY = posB.py - endOffset * Math.sin(angle);

            if (showTransitionLines) {
              // Arrow shaft with glowing outline
              ctx.strokeStyle = '#facc15';
              ctx.lineWidth = 3.2;
              ctx.shadowColor = '#facc15';
              ctx.shadowBlur = 10;
              ctx.beginPath();
              ctx.moveTo(startX, startY);
              ctx.lineTo(endX, endY);
              ctx.stroke();

              // Arrowhead at target C_j
              const headLen = Math.min(15, Math.max(9, dist * 0.25));
              ctx.fillStyle = '#facc15';
              ctx.beginPath();
              ctx.moveTo(endX, endY);
              ctx.lineTo(endX - headLen * Math.cos(angle - Math.PI / 6.5),
                         endY - headLen * Math.sin(angle - Math.PI / 6.5));
              ctx.lineTo(endX - (headLen * 0.6) * Math.cos(angle),
                         endY - (headLen * 0.6) * Math.sin(angle));
              ctx.lineTo(endX - headLen * Math.cos(angle + Math.PI / 6.5),
                         endY - headLen * Math.sin(angle + Math.PI / 6.5));
              ctx.closePath();
              ctx.fill();
            }

            if (showDistLabels) {
              // Label badge along arrow
              const badgeText = `C${fromId} → C${toId} [${(prob * 100).toFixed(1)}%, N=${cnt}]`;
              ctx.font = 'bold 10px monospace';
              const tw = ctx.measureText(badgeText).width;
              const midX = (startX + endX) / 2;
              const midY = (startY + endY) / 2 - 10;

              const badgeX = Math.max(rect.x + 10,
                                      Math.min(rect.x + rect.w - tw - 10, midX - tw / 2));
              const badgeY = Math.max(rect.y + 16, Math.min(rect.y + rect.h - 10, midY));

              ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
              ctx.strokeStyle = '#facc15';
              ctx.lineWidth = 1.2;
              ctx.beginPath();
              ctx.roundRect(badgeX - 4, badgeY - 11, tw + 8, 15, 3);
              ctx.fill();
              ctx.stroke();

              ctx.fillStyle = '#facc15';
              ctx.textAlign = 'left';
              ctx.fillText(badgeText, badgeX, badgeY);
            }
          }
          ctx.restore();
        }

        // Active Query Frame Node (f_i) & Evaluated Distance Rays
        if (currentFrame) {
          ctx.save();
          const prFrame = getProjectedCoord(currentFrame);
          const posFrame = mapMetricToQuad(prFrame.u, prFrame.v, qIdx, rect);

          // Current Evaluated Rays
          if (showDistLines && currentEvaluations && currentEvaluations.length > 0) {
            if (typeof currentEvaluationsAlpha === 'number') {
              ctx.globalAlpha = Math.max(0.0, Math.min(1.0, currentEvaluationsAlpha));
            }
            currentEvaluations.forEach(ev => {
              const prTarget = getProjectedCoord(ev.target);
              const posTarget = mapMetricToQuad(prTarget.u, prTarget.v, qIdx, rect);

              ctx.beginPath();
              ctx.moveTo(posFrame.px, posFrame.py);
              ctx.lineTo(posTarget.px, posTarget.py);
              ctx.strokeStyle = ev.match ? '#4ade80' : '#ef4444';
              ctx.lineWidth = ev.match ? 2.2 : 1.2;
              ctx.stroke();
            });
            ctx.globalAlpha = 1.0;
          }

          // Active Query Frame Node (f_i) with pulsing outer ring
          ctx.beginPath();
          ctx.arc(posFrame.px, posFrame.py, 8.0, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(250, 204, 21, 0.4)';
          ctx.lineWidth = 3.0;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(posFrame.px, posFrame.py, 5.5, 0, Math.PI * 2);
          ctx.fillStyle = '#facc15';
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.8;
          ctx.stroke();

          if (showDistLabels) {
            ctx.fillStyle = '#facc15';
            ctx.font = 'bold 10px sans-serif';
            ctx.fillText("fi", posFrame.px + 8, posFrame.py + 10);
          }
          ctx.restore();
        }

        // 4b. Draw k-NN Graph Overlay (Query Point -> Top-k Nearest Neighbors)
        if (showKnnLines && !activeSampleHighlight &&
            typeof knnResults !== 'undefined' && knnResults && knnResults.indices) {
          const N = knnResults.totalFrames;
          const k = knnResults.k;
          const pts = (typeof benchmarkDataset !== 'undefined' &&
                       benchmarkDataset && benchmarkDataset.length > 0) ?
                      benchmarkDataset : (typeof pastSamples !== 'undefined' ? pastSamples : []);

          let qId = (typeof selectedKnnQuerySample !== 'undefined')
            ? selectedKnnQuerySample
            : -1;
          if (qId < 0 || qId >= N) {
            qId = (typeof currentFrameIdx !== 'undefined' && currentFrameIdx > 0 &&
                   currentFrameIdx <= N) ? currentFrameIdx - 1 : 0;
          }

          if (qId >= 0 && qId < pts.length && qId < N) {
            const queryPt = pts[qId];
            const prQ = getProjectedCoord(queryPt);
            const posQ = mapMetricToQuad(prQ.u, prQ.v, qIdx, rect);

            ctx.save();

            // Draw vector rays from query to each of its k nearest neighbors
            for (let r = 0; r < k; r++) {
              const nId = knnResults.indices[qId * k + r];
              if (nId < 0 || nId >= pts.length) continue;

              const nPt = pts[nId];
              if (!nPt) continue;
              const prN = getProjectedCoord(nPt);
              const posN = mapMetricToQuad(prN.u, prN.v, qIdx, rect);

              const isRank1 = (r === 0);
              const isHovered = (typeof hoveredKnnNeighborId !== 'undefined' &&
                                 hoveredKnnNeighborId === nId);

              const alphaStr = Math.max(0.20, 0.75 - r * 0.06).toFixed(2);
              const rayColor = isHovered ? '#e879f9' :
                                (isRank1 ? '#e879f9' : `rgba(192, 132, 252, ${alphaStr})`);
              const grad = ctx.createLinearGradient(posQ.px, posQ.py, posN.px, posN.py);
              grad.addColorStop(0, '#38bdf8');
              grad.addColorStop(1, rayColor);

              ctx.save();
              ctx.shadowColor = isHovered ? '#facc15' : (isRank1 ? '#e879f9' : '#c084fc');
              ctx.shadowBlur = isHovered ? 8 : 4;
              ctx.beginPath();
              ctx.moveTo(posQ.px, posQ.py);
              ctx.lineTo(posN.px, posN.py);
              ctx.strokeStyle = grad;
              ctx.lineWidth = isHovered ? 2.6 : (isRank1 ? 2.2 : 1.4);
              ctx.stroke();
              ctx.restore();

              // Highlight neighbor node
              ctx.beginPath();
              ctx.arc(posN.px, posN.py, isHovered ? 6.0 : (isRank1 ? 4.8 : 3.5), 0, Math.PI * 2);
              ctx.fillStyle = isHovered ? '#e879f9' : (isRank1 ? '#c084fc' : '#a855f7');
              ctx.fill();
              ctx.strokeStyle = '#f8fafc';
              ctx.lineWidth = 1;
              ctx.stroke();

              // Rank label for top neighbors
              if (showDistLabels && (r < 3 || isHovered)) {
                ctx.font = 'bold 8.5px monospace';
                ctx.fillStyle = isHovered ? '#e879f9' : (isRank1 ? '#c084fc' : '#d8b4fe');
                ctx.fillText(`#${r + 1}`, posN.px + 5, posN.py - 3);
              }
            }

            // Highlight query sample node with pulsing violet ring
            ctx.beginPath();
            ctx.arc(posQ.px, posQ.py, 8.5, 0, Math.PI * 2);
            ctx.strokeStyle = '#e879f9';
            ctx.lineWidth = 2.0;
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(posQ.px, posQ.py, 3.5, 0, Math.PI * 2);
            ctx.fillStyle = '#f8fafc';
            ctx.fill();

            ctx.fillStyle = '#e879f9';
            ctx.font = 'bold 9px monospace';
            ctx.fillText(`Query #${qId}`, posQ.px + 10, posQ.py - 6);
            ctx.restore();
          }
        }
      }

      // 5. 3D Coordinate RGB Triad Gizmo (for Custom 3D Viewport)
      if (is3DCustom && showGridAxes) {
        const gizmoOrigin = { px: rect.x + 36, py: rect.y + rect.h - 36 };
        const gLen = 24;
        const az = orbitCamera.azimuth, el = orbitCamera.elevation;

        const gx = project3DVector(1, 0, 0, az, el);
        const gy = project3DVector(0, 1, 0, az, el);
        const gz = project3DVector(0, 0, 1, az, el);

        const dXG = (typeof plotDimX === 'number') ? plotDimX : 0;
        const dYG = (typeof plotDimY === 'number') ? plotDimY : 1;
        const dZG = (typeof plotDimZ === 'number') ? plotDimZ : 2;
        const isCustomGizmo = (currentDim > 3 || dXG !== 0 || dYG !== 1 || dZG !== 2);

        const axes = [
          { name: isCustomGizmo ? `D${dXG}` : 'X', u: gx.u, v: gx.v, color: '#f87171' },
          { name: isCustomGizmo ? `D${dYG}` : 'Y', u: gy.u, v: gy.v, color: '#4ade80' },
          { name: isCustomGizmo ? `D${dZG}` : 'Z', u: gz.u, v: gz.v, color: '#38bdf8' }
        ];

        axes.forEach(ax => {
          const endPx = gizmoOrigin.px + ax.u * gLen;
          const endPy = gizmoOrigin.py - ax.v * gLen;

          ctx.strokeStyle = ax.color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(gizmoOrigin.px, gizmoOrigin.py);
          ctx.lineTo(endPx, endPy);
          ctx.stroke();

          ctx.fillStyle = ax.color;
          ctx.font = 'bold 9px sans-serif';
          ctx.fillText(ax.name, endPx + 3, endPy + 3);
        });

        // Center hub
        ctx.beginPath();
        ctx.arc(gizmoOrigin.px, gizmoOrigin.py, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }

      // 5C. Axis of View of the 3D View (in orthogonal panels X, Y, and Z)
      if (!is3DCustom && currentDim >= 3 && maximizedQuad === null &&
          (typeof isRecon4PanelView === 'undefined' || !isRecon4PanelView) &&
          (typeof dataMode === 'undefined' || dataMode !== 'image') &&
          (typeof showCameraViewAxis === 'undefined' || showCameraViewAxis)) {

        let tx = 0, ty = 0, tz = 0;
        if (typeof orbitCamera !== 'undefined' && orbitCamera && orbitCamera.isLocked) {
          tx = orbitCamera.targetX || 0;
          ty = orbitCamera.targetY || 0;
          tz = orbitCamera.targetZ || 0;
        }

        const az = (typeof orbitCamera !== 'undefined')
          ? orbitCamera.azimuth : -35 * (Math.PI / 180);
        const el = (typeof orbitCamera !== 'undefined')
          ? orbitCamera.elevation : 25 * (Math.PI / 180);
        const cosT = Math.cos(az), sinT = Math.sin(az);
        const cosP = Math.cos(el), sinP = Math.sin(el);

        // Line-of-sight unit vector looking from camera through target:
        const vx = -sinT * cosP;
        const vy = cosT * cosP;
        const vz = -sinP;

        let uT = 0, vT = 0;
        let du = 0, dv = 0;
        let perpComp = 0;

        if (qIdx === 0) {
          // Panel X: ALONG_X (Y-Z plane)
          uT = ty;
          vT = tz;
          du = vy;
          dv = vz;
          perpComp = vx;
        } else if (qIdx === 1) {
          // Panel Y: ALONG_Y (X-Z plane)
          uT = tx;
          vT = tz;
          du = vx;
          dv = vz;
          perpComp = vy;
        } else if (qIdx === 2) {
          // Panel Z: ALONG_Z (X-Y plane)
          uT = tx;
          vT = ty;
          du = vx;
          dv = vy;
          perpComp = vz;
        }

        const targetPos = mapMetricToQuad(uT, vT, qIdx, rect);
        const dirLen2D = Math.hypot(du, dv);

        const z3 = (typeof quadViews !== 'undefined' && quadViews[3] && quadViews[3].zoom)
          ? quadViews[3].zoom
          : ((typeof orbitCamera !== 'undefined' && orbitCamera.zoom) ? orbitCamera.zoom : 1.0);
        const spanFactor = 1.0 / Math.max(0.05, Math.min(20.0, z3));

        ctx.save();

        if (dirLen2D < 0.035) {
          if (typeof viewVectorHandles !== 'undefined') {
            viewVectorHandles[qIdx] = {
              qIdx: qIdx,
              isPerp: true,
              targetPos: { px: targetPos.px, py: targetPos.py },
              uT: uT,
              vT: vT,
              du: du,
              dv: dv,
              L_fwd: 0.40 * spanFactor,
              L_cam: 0.85 * spanFactor,
              perpComp: perpComp
            };
          }

          const isHovered = (typeof hoveredViewVectorHandle !== 'undefined' &&
                             hoveredViewVectorHandle &&
                             hoveredViewVectorHandle.qIdx === qIdx);
          const isDraggingThis = (typeof draggingViewVectorHandle !== 'undefined' &&
                                  draggingViewVectorHandle &&
                                  draggingViewVectorHandle.qIdx === qIdx);

          if (isHovered || isDraggingThis) {
            ctx.save();
            ctx.shadowColor = '#38bdf8';
            ctx.shadowBlur = 9;
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2.0;
            ctx.beginPath();
            ctx.arc(targetPos.px, targetPos.py, 11, 0, Math.PI * 2);
            ctx.stroke();
            ctx.fillStyle = isDraggingThis
              ? 'rgba(56, 189, 248, 0.45)'
              : 'rgba(56, 189, 248, 0.22)';
            ctx.fill();
            ctx.restore();
          }

          // Viewing axis is perpendicular to this panel (into or out of screen)
          ctx.strokeStyle = '#f59e0b';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(targetPos.px, targetPos.py, 7, 0, Math.PI * 2);
          ctx.stroke();

          if (perpComp > 0) {
            // Pointing into screen: X cross
            const r7 = 4.5;
            ctx.beginPath();
            ctx.moveTo(targetPos.px - r7, targetPos.py - r7);
            ctx.lineTo(targetPos.px + r7, targetPos.py + r7);
            ctx.moveTo(targetPos.px + r7, targetPos.py - r7);
            ctx.lineTo(targetPos.px - r7, targetPos.py + r7);
            ctx.stroke();
          } else {
            // Pointing out of screen: center dot
            ctx.beginPath();
            ctx.arc(targetPos.px, targetPos.py, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = '#f59e0b';
            ctx.fill();
          }

          // Small perpendicular view badge with live angles
          const degAz = Math.round(az * 180 / Math.PI);
          const degEl = Math.round(el * 180 / Math.PI);
          const dragSuffix = isDraggingThis ? ' (rotating)' : ' (⟂)';
          ctx.font = 'bold 8.5px monospace';
          const pText = `📷 3D View [θ:${degAz}°, φ:${degEl}°]${dragSuffix}`;
          const bW = ctx.measureText(pText).width + 12;
          const bH = 16;
          const bX = Math.max(rect.x + 6,
            Math.min(rect.x + rect.w - bW - 6, targetPos.px + 10));
          const bY = Math.max(rect.y + 28,
            Math.min(rect.y + rect.h - bH - 6, targetPos.py - 8));

          ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.50)';
          ctx.lineWidth = 1.0;
          ctx.beginPath();
          ctx.roundRect(bX, bY, bW, bH, 3);
          ctx.fill();
          ctx.stroke();

          ctx.fillStyle = '#fbbf24';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(pText, bX + 6, bY + bH / 2 + 0.5);

        } else {
          // Scale back to camera and forward along line of sight scaled to 3D panel span
          const L_cam = 0.85 * spanFactor;
          const L_fwd = 0.40 * spanFactor;

          // du and dv are the un-normalized projected components of the 3D unit viewing vector
          const camU = uT - du * L_cam;
          const camV = vT - dv * L_cam;
          const fwdU = uT + du * L_fwd;
          const fwdV = vT + dv * L_fwd;

          const pCam = mapMetricToQuad(camU, camV, qIdx, rect);
          const pFwd = mapMetricToQuad(fwdU, fwdV, qIdx, rect);
          const pTarget = targetPos;

          if (typeof viewVectorHandles !== 'undefined') {
            viewVectorHandles[qIdx] = {
              qIdx: qIdx,
              isPerp: false,
              targetPos: { px: targetPos.px, py: targetPos.py },
              pFwd: { px: pFwd.px, py: pFwd.py },
              pCam: { px: pCam.px, py: pCam.py },
              uT: uT,
              vT: vT,
              du: du,
              dv: dv,
              L_fwd: L_fwd,
              L_cam: L_cam,
              perpComp: perpComp
            };
          }

          const isHovered = (typeof hoveredViewVectorHandle !== 'undefined' &&
                             hoveredViewVectorHandle &&
                             hoveredViewVectorHandle.qIdx === qIdx);
          const isDraggingThis = (typeof draggingViewVectorHandle !== 'undefined' &&
                                  draggingViewVectorHandle &&
                                  draggingViewVectorHandle.qIdx === qIdx);
          const activeEnd = isDraggingThis
            ? draggingViewVectorHandle.end
            : (isHovered ? hoveredViewVectorHandle.end : null);

          const spDx = pFwd.px - pCam.px;
          const spDy = pFwd.py - pCam.py;
          const spLen = Math.hypot(spDx, spDy);

          const distCam = Math.hypot(pTarget.px - pCam.px, pTarget.py - pCam.py);
          const distFwd = Math.hypot(pFwd.px - pTarget.px, pFwd.py - pTarget.py);

          if (spLen > 5) {
            const sUx = spDx / spLen;
            const sUy = spDy / spLen;

            // 1. Subtle glow halo behind line
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.18)';
            ctx.lineWidth = 4.0;
            ctx.beginPath();
            ctx.moveTo(pCam.px, pCam.py);
            ctx.lineTo(pFwd.px, pFwd.py);
            ctx.stroke();

            // 2. Main dashed viewing axis line
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([5, 3]);
            ctx.beginPath();
            ctx.moveTo(pCam.px, pCam.py);
            ctx.lineTo(pFwd.px, pFwd.py);
            ctx.stroke();
            ctx.setLineDash([]);

            // 3. Arrowhead at forward tip pointing in look direction
            const aLen = Math.min(9.0, Math.max(3.5, distFwd * 0.8));
            const aWid = aLen * 0.5;
            const perpX = -sUy * aWid;
            const perpY = sUx * aWid;

            ctx.fillStyle = '#f59e0b';
            ctx.beginPath();
            ctx.moveTo(pFwd.px, pFwd.py);
            ctx.lineTo(pFwd.px - sUx * aLen + perpX, pFwd.py - sUy * aLen + perpY);
            ctx.lineTo(pFwd.px - sUx * aLen - perpX, pFwd.py - sUy * aLen - perpY);
            ctx.closePath();
            ctx.fill();

            // Draggable handle highlight or grip ring at forward tip
            if (activeEnd === 'fwd') {
              ctx.save();
              ctx.shadowColor = '#38bdf8';
              ctx.shadowBlur = 9;
              ctx.strokeStyle = '#38bdf8';
              ctx.lineWidth = 2.0;
              ctx.beginPath();
              ctx.arc(pFwd.px, pFwd.py, 9.5, 0, Math.PI * 2);
              ctx.stroke();
              ctx.fillStyle = isDraggingThis
                ? 'rgba(56, 189, 248, 0.45)'
                : 'rgba(56, 189, 248, 0.22)';
              ctx.fill();
              ctx.restore();
            } else {
              ctx.save();
              ctx.strokeStyle = 'rgba(251, 191, 36, 0.6)';
              ctx.lineWidth = 1.0;
              ctx.beginPath();
              ctx.arc(pFwd.px, pFwd.py, 5.0, 0, Math.PI * 2);
              ctx.stroke();
              ctx.restore();
            }

            // 4. Camera icon & FOV frustum cone at pCam
            if (distCam >= 14) {
              const cBoxW = 9;
              const cBoxH = 7;
              ctx.fillStyle = 'rgba(217, 119, 6, 0.9)';
              ctx.strokeStyle = '#fbbf24';
              ctx.lineWidth = 1.0;
              ctx.beginPath();
              ctx.roundRect(pCam.px - cBoxW / 2, pCam.py - cBoxH / 2, cBoxW, cBoxH, 2);
              ctx.fill();
              ctx.stroke();

              // Lens circle
              ctx.beginPath();
              ctx.arc(
                pCam.px + sUx * (cBoxW / 2 + 1),
                pCam.py + sUy * (cBoxH / 2 + 1),
                2, 0, Math.PI * 2
              );
              ctx.fillStyle = '#fbbf24';
              ctx.fill();

              // Diverging FOV frustum lines
              const fovLen = Math.min(16.0, distCam * 0.8);
              const fovAngle = 20 * (Math.PI / 180);
              const cosA = Math.cos(fovAngle), sinA = Math.sin(fovAngle);

              const r1x = sUx * cosA - sUy * sinA;
              const r1y = sUx * sinA + sUy * cosA;
              const r2x = sUx * cosA + sUy * sinA;
              const r2y = -sUx * sinA + sUy * cosA;

              ctx.strokeStyle = 'rgba(251, 191, 36, 0.55)';
              ctx.lineWidth = 1.0;
              ctx.setLineDash([2, 3]);
              ctx.beginPath();
              ctx.moveTo(pCam.px, pCam.py);
              ctx.lineTo(pCam.px + r1x * fovLen, pCam.py + r1y * fovLen);
              ctx.moveTo(pCam.px, pCam.py);
              ctx.lineTo(pCam.px + r2x * fovLen, pCam.py + r2y * fovLen);
              ctx.stroke();
              ctx.setLineDash([]);
            } else if (distCam >= 5) {
              // Compact camera circle dot when foreshortened
              ctx.fillStyle = '#fbbf24';
              ctx.strokeStyle = '#0f172a';
              ctx.lineWidth = 1.0;
              ctx.beginPath();
              ctx.arc(pCam.px, pCam.py, 2.5, 0, Math.PI * 2);
              ctx.fill();
              ctx.stroke();
            }

            // Draggable handle highlight or grip ring at camera position
            if (activeEnd === 'cam') {
              ctx.save();
              ctx.shadowColor = '#38bdf8';
              ctx.shadowBlur = 9;
              ctx.strokeStyle = '#38bdf8';
              ctx.lineWidth = 2.0;
              ctx.beginPath();
              ctx.arc(pCam.px, pCam.py, 10.5, 0, Math.PI * 2);
              ctx.stroke();
              ctx.fillStyle = isDraggingThis
                ? 'rgba(56, 189, 248, 0.45)'
                : 'rgba(56, 189, 248, 0.22)';
              ctx.fill();
              ctx.restore();
            } else {
              ctx.save();
              ctx.strokeStyle = 'rgba(251, 191, 36, 0.6)';
              ctx.lineWidth = 1.0;
              ctx.beginPath();
              ctx.arc(pCam.px, pCam.py, 6.0, 0, Math.PI * 2);
              ctx.stroke();
              ctx.restore();
            }

            // 5. Look-at target pivot dot
            ctx.beginPath();
            ctx.arc(pTarget.px, pTarget.py, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = '#fbbf24';
            ctx.fill();
            ctx.strokeStyle = '#0f172a';
            ctx.lineWidth = 1.0;
            ctx.stroke();

            // 6. Camera View Angle Badge
            const degAz = Math.round(az * 180 / Math.PI);
            const degEl = Math.round(el * 180 / Math.PI);
            const dragSuffix = isDraggingThis ? ' (rotating)' : '';
            const badgeText = `📷 3D View [θ:${degAz}°, φ:${degEl}°]${dragSuffix}`;

            ctx.font = 'bold 8.5px monospace';
            const bW = ctx.measureText(badgeText).width + 12;
            const bH = 16;

            let bX, bY;
            if (distCam >= 25) {
              bX = pCam.px + 10;
              bY = pCam.py - bH - 4;
              if (bX + bW > rect.x + rect.w - 6) bX = pCam.px - bW - 10;
            } else {
              bX = pTarget.px + 12;
              bY = pTarget.py - bH - 6;
              if (bX + bW > rect.x + rect.w - 6) bX = pTarget.px - bW - 12;
            }
            if (bX < rect.x + 6) bX = rect.x + 6;
            if (bY < rect.y + 28) bY = pTarget.py + 12;
            if (bY + bH > rect.y + rect.h - 6) bY = rect.y + rect.h - bH - 6;

            ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.50)';
            ctx.lineWidth = 1.0;
            ctx.beginPath();
            ctx.roundRect(bX, bY, bW, bH, 3);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = '#fbbf24';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(badgeText, bX + 6, bY + bH / 2 + 0.5);
          }
        }

        ctx.restore();
      }

      // 5B. 3D Biplot Basis Rays (for High-D)
      if (is3DCustom && typeof showBiplotRays !== 'undefined' && showBiplotRays && currentDim > 3) {
        if (typeof HighDEngine !== 'undefined' && HighDEngine.getBiplotRays) {
          let projMatrix = null;
          if (highDProjMode === 'tour' && grandTour) {
            projMatrix = grandTour.frame;
          } else if (highDProjMode === 'pca' && cachedPCAResult && cachedPCAResult.components) {
            const i0 = pcaComponentIndices[0] || 0;
            const i1 = pcaComponentIndices[1] || 1;
            const i2 = pcaComponentIndices[2] || 2;
            projMatrix = [
              cachedPCAResult.components[i0],
              cachedPCAResult.components[i1],
              cachedPCAResult.components[i2]
            ];
          } else if (highDProjMode === 'lda' && cachedLDAResult && cachedLDAResult.components) {
            projMatrix = cachedLDAResult.components;
          } else {
            projMatrix = [
              new Float64Array(currentDim),
              new Float64Array(currentDim),
              new Float64Array(currentDim)
            ];
            if (plotDimX < currentDim) projMatrix[0][plotDimX] = 1.0;
            if (plotDimY < currentDim) projMatrix[1][plotDimY] = 1.0;
            if (plotDimZ < currentDim) projMatrix[2][plotDimZ] = 1.0;
          }

          if (projMatrix) {
            const rays = HighDEngine.getBiplotRays(projMatrix, currentDim, 0.75);
            const center = mapMetricToQuad(0, 0, qIdx, rect);
            const az = orbitCamera.azimuth, el = orbitCamera.elevation;

            ctx.save();
            rays.forEach(ray => {
              if (ray.mag < 0.05) return;
              const pr = project3D(ray.x, ray.y, ray.z, az, el);
              const px = center.px + pr.u * scale;
              const py = center.py - pr.v * scale;

              const hue = Math.floor((ray.dim * 360) / currentDim);
              ctx.strokeStyle = `hsla(${hue}, 85%, 65%, 0.75)`;
              ctx.lineWidth = 1.5;

              ctx.beginPath();
              ctx.moveTo(center.px, center.py);
              ctx.lineTo(px, py);
              ctx.stroke();

              ctx.beginPath();
              ctx.arc(px, py, 2.5, 0, Math.PI * 2);
              ctx.fillStyle = `hsl(${hue}, 85%, 65%)`;
              ctx.fill();

              ctx.fillStyle = '#f8fafc';
              ctx.font = 'bold 9px monospace';
              ctx.fillText(`d${ray.dim}`, px + 4, py + 2);
            });
            ctx.restore();
          }
        }
      }

      // Color-bar legend for dim/density modes
      if (typeof pointColorMode !== 'undefined'
          && pointColorMode !== 'cluster'
          && dimDensitySummary
      ) {
        const barH = 12;
        const margin = 16;
        const barY = rect.y + rect.h - margin - barH - 8;
        const barX = rect.x + margin + 70;
        const barW = Math.max(120, rect.w - (margin * 2 + 140));

        // Background card
        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.fillRect(
          barX - 8, barY - 18,
          barW + 16, barH + 26
        );

        let stops, mn, mx, label;
        if (pointColorMode === 'dimension') {
          stops = VIRIDIS_STOPS;
          const s =
            dimDensitySummary.intrinsic_dimension;
          mn = s ? s.min : 0;
          mx = s ? s.max : 1;
          label = 'Local Intrinsic Dimension (Viridis)';
        } else {
          stops = INFERNO_STOPS;
          const s = dimDensitySummary.log_density;
          mn = s ? s.min : 0;
          mx = s ? s.max : 1;
          label = 'Log-Density ln(f) (Inferno)';
        }

        // Draw gradient bar
        const segW = barW / stops.length;
        for (let s = 0; s < stops.length; s++) {
          const [r, g, b] = stops[s];
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(
            barX + s * segW, barY,
            segW + 1, barH
          );
        }

        // Border
        ctx.strokeStyle =
          'rgba(255, 255, 255, 0.35)';
        ctx.lineWidth = 0.8;
        ctx.strokeRect(
          barX, barY, barW, barH
        );

        // Labels
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 9.5px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(
          mn.toFixed(2), barX, barY - 4
        );
        ctx.textAlign = 'right';
        ctx.fillText(
          mx.toFixed(2), barX + barW, barY - 4
        );
        ctx.textAlign = 'center';
        ctx.fillText(
          label, barX + barW / 2, barY - 4
        );
      } // color-bar legend

      // 5.9 Pointer Crosshair & Multi-Panel Coordinate Hover Overlay
      if (typeof hoveredPanelPointer !== 'undefined' && hoveredPanelPointer !== null &&
          currentDim >= 3 && maximizedQuad === null &&
          (typeof isRecon4PanelView === 'undefined' || !isRecon4PanelView) &&
          (typeof dataMode === 'undefined' || dataMode !== 'image')) {

        const isHoveredQuad = (qIdx === hoveredPanelPointer.qIdx);
        const isHighD = (currentDim > 3);
        const dX = (typeof plotDimX === 'number') ? plotDimX : 0;
        const dY = (typeof plotDimY === 'number') ? plotDimY : 1;
        const dZ = (typeof plotDimZ === 'number') ? plotDimZ : 2;
        const isCustomAxes = (isHighD || dX !== 0 || dY !== 1 || dZ !== 2);

        let nameX = isCustomAxes ? `D${dX}` : 'X';
        let nameY = isCustomAxes ? `D${dY}` : 'Y';
        let nameZ = isCustomAxes ? `D${dZ}` : 'Z';
        if (isHighD && typeof highDProjMode !== 'undefined' && highDProjMode === 'pca') {
          const i0 = (typeof pcaComponentIndices !== 'undefined' &&
            pcaComponentIndices[0] !== undefined) ? pcaComponentIndices[0] + 1 : 1;
          const i1 = (typeof pcaComponentIndices !== 'undefined' &&
            pcaComponentIndices[1] !== undefined) ? pcaComponentIndices[1] + 1 : 2;
          const i2 = (typeof pcaComponentIndices !== 'undefined' &&
            pcaComponentIndices[2] !== undefined) ? pcaComponentIndices[2] + 1 : 3;
          nameX = `PC${i0}`;
          nameY = `PC${i1}`;
          nameZ = `PC${i2}`;
        }

        function formatCoordVal(val) {
          if (typeof val !== 'number' || isNaN(val)) return '0.000';
          const sign = val >= 0 ? '+' : '';
          if (Math.abs(val) >= 100) return sign + val.toFixed(1);
          if (Math.abs(val) >= 10) return sign + val.toFixed(2);
          return sign + val.toFixed(3);
        }

        ctx.save();

        if (isHoveredQuad) {
          const px = hoveredPanelPointer.screenX;
          const py = hoveredPanelPointer.screenY;

          // Faint crosshair across hovered quadrant
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.30)';
          ctx.lineWidth = 1.0;
          ctx.setLineDash([3, 4]);

          ctx.beginPath();
          ctx.moveTo(rect.x, py);
          ctx.lineTo(rect.x + rect.w, py);
          ctx.moveTo(px, rect.y);
          ctx.lineTo(px, rect.y + rect.h);
          ctx.stroke();

          // Subtle pointer reticle
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.arc(px, py, 4.0, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.70)';
          ctx.lineWidth = 1.2;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(px, py, 1.5, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(56, 189, 248, 0.90)';
          ctx.fill();

          // Coordinate badge for hovered panel
          let parts = [];
          if (qIdx === 0) {
            parts = [
              { text: `${nameY}: `, color: '#c084fc' },
              { text: formatCoordVal(hoveredPanelPointer.y) + '  ', color: '#f8fafc' },
              { text: `${nameZ}: `, color: '#4ade80' },
              { text: formatCoordVal(hoveredPanelPointer.z), color: '#f8fafc' }
            ];
          } else if (qIdx === 1) {
            parts = [
              { text: `${nameX}: `, color: '#38bdf8' },
              { text: formatCoordVal(hoveredPanelPointer.x) + '  ', color: '#f8fafc' },
              { text: `${nameZ}: `, color: '#4ade80' },
              { text: formatCoordVal(hoveredPanelPointer.z), color: '#f8fafc' }
            ];
          } else if (qIdx === 2) {
            parts = [
              { text: `${nameX}: `, color: '#38bdf8' },
              { text: formatCoordVal(hoveredPanelPointer.x) + '  ', color: '#f8fafc' },
              { text: `${nameY}: `, color: '#c084fc' },
              { text: formatCoordVal(hoveredPanelPointer.y), color: '#f8fafc' }
            ];
          } else {
            parts = [
              { text: `${nameX}: `, color: '#38bdf8' },
              { text: formatCoordVal(hoveredPanelPointer.x) + ' ', color: '#f8fafc' },
              { text: `${nameY}: `, color: '#c084fc' },
              { text: formatCoordVal(hoveredPanelPointer.y) + ' ', color: '#f8fafc' },
              { text: `${nameZ}: `, color: '#4ade80' },
              { text: formatCoordVal(hoveredPanelPointer.z), color: '#f8fafc' }
            ];
          }

          drawColoredCoordBadge(ctx, px + 10, py - 20, parts, true, rect);

        } else {
          let u = 0, v = 0;
          let parts = [];

          if (qIdx === 0) {
            // ALONG_X (Y-Z plane)
            u = hoveredPanelPointer.y;
            v = hoveredPanelPointer.z;
            parts = [
              { text: `${nameY}: `, color: '#c084fc' },
              { text: formatCoordVal(u) + '  ', color: '#f8fafc' },
              { text: `${nameZ}: `, color: '#4ade80' },
              { text: formatCoordVal(v), color: '#f8fafc' }
            ];
          } else if (qIdx === 1) {
            // ALONG_Y (X-Z plane)
            u = hoveredPanelPointer.x;
            v = hoveredPanelPointer.z;
            parts = [
              { text: `${nameX}: `, color: '#38bdf8' },
              { text: formatCoordVal(u) + '  ', color: '#f8fafc' },
              { text: `${nameZ}: `, color: '#4ade80' },
              { text: formatCoordVal(v), color: '#f8fafc' }
            ];
          } else if (qIdx === 2) {
            // ALONG_Z (X-Y plane)
            u = hoveredPanelPointer.x;
            v = hoveredPanelPointer.y;
            parts = [
              { text: `${nameX}: `, color: '#38bdf8' },
              { text: formatCoordVal(u) + '  ', color: '#f8fafc' },
              { text: `${nameY}: `, color: '#c084fc' },
              { text: formatCoordVal(v), color: '#f8fafc' }
            ];
          } else if (qIdx === 3) {
            // CUSTOM_3D (3D Orbit)
            const pr = project3D(
              hoveredPanelPointer.x, hoveredPanelPointer.y, hoveredPanelPointer.z,
              orbitCamera.azimuth, orbitCamera.elevation
            );
            u = pr.u;
            v = pr.v;
            parts = [
              { text: `${nameX}: `, color: '#38bdf8' },
              { text: formatCoordVal(hoveredPanelPointer.x) + ' ', color: '#f8fafc' },
              { text: `${nameY}: `, color: '#c084fc' },
              { text: formatCoordVal(hoveredPanelPointer.y) + ' ', color: '#f8fafc' },
              { text: `${nameZ}: `, color: '#4ade80' },
              { text: formatCoordVal(hoveredPanelPointer.z), color: '#f8fafc' }
            ];
          }

          const pos = mapMetricToQuad(u, v, qIdx, rect);
          const px = pos.px;
          const py = pos.py;

          const inX = (px >= rect.x && px <= rect.x + rect.w);
          const inY = (py >= rect.y && py <= rect.y + rect.h);

          ctx.strokeStyle = 'rgba(56, 189, 248, 0.28)';
          ctx.lineWidth = 1.0;
          ctx.setLineDash([2, 4]);

          ctx.beginPath();
          if (inY) {
            ctx.moveTo(rect.x, py);
            ctx.lineTo(rect.x + rect.w, py);
          }
          if (inX) {
            ctx.moveTo(px, rect.y);
            ctx.lineTo(px, rect.y + rect.h);
          }
          ctx.stroke();

          // Tracking reticle in other panel
          if (inX && inY) {
            ctx.setLineDash([]);
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.65)';
            ctx.lineWidth = 1.1;
            ctx.beginPath();
            ctx.arc(px, py, 3.5, 0, Math.PI * 2);
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(px, py, 1.2, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(56, 189, 248, 0.85)';
            ctx.fill();
          }

          // Coordinate badge positioned near crosshair and clamped within viewport
          ctx.setLineDash([]);
          const badgeX = inX ? (px + 8) : (px < rect.x ? rect.x + 6 : rect.x + rect.w - 120);
          const badgeY = inY ? (py - 20) : (py < rect.y ? rect.y + 28 : rect.y + rect.h - 22);
          drawColoredCoordBadge(ctx, badgeX, badgeY, parts, false, rect);
        }

        ctx.restore();
      }

      if (showViewportHUD) {
        // 6. Viewport Header Overlay & Maximize Button
        let title = "";
        let subtitle = "";
        const isHighD = (currentDim > 3);
        const dX = (typeof plotDimX === 'number') ? plotDimX : 0;
        const dY = (typeof plotDimY === 'number') ? plotDimY : 1;
        const dZ = (typeof plotDimZ === 'number') ? plotDimZ : 2;

        const isCustomAxes = (isHighD || dX !== 0 || dY !== 1 || dZ !== 2);

        if (isHighD && typeof decoupledQuads !== 'undefined' && decoupledQuads && qIdx < 3) {
          const pair = (typeof quadAxisPairs !== 'undefined' && quadAxisPairs[qIdx])
            ? quadAxisPairs[qIdx] : { x: qIdx * 2, y: qIdx * 2 + 1 };
          title = `📐 Subspace Q${qIdx + 1} (D${pair.x} vs D${pair.y})`;
          subtitle = `H: +D${pair.x} ➔ | V: +D${pair.y} ⬆`;
        } else if (isHighD && typeof highDProjMode !== 'undefined' &&
                   highDProjMode === 'pca' && !is3DCustom) {
          const i0 = (typeof pcaComponentIndices !== 'undefined' &&
                      pcaComponentIndices[0] !== undefined) ? pcaComponentIndices[0] : 0;
          const i1 = (typeof pcaComponentIndices !== 'undefined' &&
                      pcaComponentIndices[1] !== undefined) ? pcaComponentIndices[1] : 1;
          const i2 = (typeof pcaComponentIndices !== 'undefined' &&
                      pcaComponentIndices[2] !== undefined) ? pcaComponentIndices[2] : 2;
          if (viewType === "ALONG_X") {
            title = `📐 PCA: Along PC${i0 + 1} (PC${i1 + 1}-PC${i2 + 1})`;
            subtitle = `H: +PC${i1 + 1} ➔ | V: +PC${i2 + 1} ⬆`;
          } else if (viewType === "ALONG_Y") {
            title = `📐 PCA: Along PC${i1 + 1} (PC${i0 + 1}-PC${i2 + 1})`;
            subtitle = `H: +PC${i0 + 1} ➔ | V: +PC${i2 + 1} ⬆`;
          } else {
            title = `📐 PCA: Along PC${i2 + 1} (PC${i0 + 1}-PC${i1 + 1})`;
            subtitle = `H: +PC${i0 + 1} ➔ | V: +PC${i1 + 1} ⬆`;
          }
        } else if (viewType === "ALONG_X") {
          title = isCustomAxes
            ? `📐 Along D${dX} (D${dY}-D${dZ} Plane)`
            : "📐 Along X (Y-Z Plane)";
          subtitle = isCustomAxes
            ? `H: +D${dY} ➔ | V: +D${dZ} ⬆`
            : "H: +Y ➔ | V: +Z ⬆";
        } else if (viewType === "ALONG_Y") {
          title = isCustomAxes
            ? `📐 Along D${dY} (D${dX}-D${dZ} Plane)`
            : "📐 Along Y (X-Z Plane)";
          subtitle = isCustomAxes
            ? `H: +D${dX} ➔ | V: +D${dZ} ⬆`
            : "H: +X ➔ | V: +Z ⬆";
        } else if (viewType === "ALONG_Z") {
          title = isCustomAxes
            ? `📐 Along D${dZ} (D${dX}-D${dY} Plane)`
            : "📐 Along Z (X-Y Plane)";
          subtitle = isCustomAxes
            ? `H: +D${dX} ➔ | V: +D${dY} ⬆`
            : "H: +X ➔ | V: +Y ⬆";
        } else if (viewType === "2D") {
          title = "📐 2D Standard View (X-Y Plane)";
          subtitle = "H: +X ➔ | V: +Y ⬆";
        } else if (viewType === "CUSTOM_3D") {
          const degAz = Math.round(orbitCamera.azimuth * 180 / Math.PI);
          const degEl = Math.round(orbitCamera.elevation * 180 / Math.PI);
          const cCenter = (orbitCamera && orbitCamera.isLocked)
            ? (orbitCamera.targetLabel ||
               `(${orbitCamera.targetX.toFixed(2)},${orbitCamera.targetY.toFixed(2)},` +
               `${orbitCamera.targetZ.toFixed(2)})`)
            : '';
          const lockNote = (orbitCamera && orbitCamera.isLocked)
            ? ` [🎯 Center: ${cCenter}]`
            : '';
          let modeTag = '';
          if (isCustomAxes) {
            if (isHighD && highDProjMode === 'pca') modeTag = ' [PCA 3D]';
            else if (isHighD && highDProjMode === 'tour') modeTag = ' [Grand Tour 32D]';
            else if (isHighD && highDProjMode === 'lda') modeTag = ' [Fisher LDA]';
            else modeTag = ` [D${dX}, D${dY}, D${dZ}]`;
          }
          title = `🌐 3D Orbit View [θ: ${degAz}°, φ: ${degEl}°]${modeTag}${lockNote}`;
          subtitle = (orbitCamera && orbitCamera.isLocked)
            ? "Rotating around Locked Center"
            : (highDProjMode === 'tour'
                ? "Continuous 60 FPS Geodesic Rotation"
                : "Drag to Rotate Camera");
        }

        // Header background
        ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
        ctx.fillRect(rect.x, rect.y, rect.w, 24);
        ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
        ctx.beginPath();
        ctx.moveTo(rect.x, rect.y + 24);
        ctx.lineTo(rect.x + rect.w, rect.y + 24);
        ctx.stroke();

        // Title text
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 11px sans-serif';
        const titleW = ctx.measureText(title).width;
        ctx.fillText(title, rect.x + 8, rect.y + 16);

        // Subtitle text (guaranteed no overlap with title)
        ctx.font = '10px monospace';
        const subW = ctx.measureText(subtitle).width;
        const iconMargin = (currentDim >= 3) ? 32 : 12;
        const subX = rect.x + rect.w - subW - iconMargin;
        if (subX > rect.x + titleW + 24) {
          ctx.fillStyle = '#94a3b8';
          ctx.fillText(subtitle, subX, rect.y + 16);
        }

        // Maximize / Restore Icon
        if (currentDim >= 3) {
          ctx.fillStyle = maximizedQuad === qIdx ? '#38bdf8' : '#94a3b8';
          ctx.font = '12px sans-serif';
          ctx.fillText(maximizedQuad === qIdx ? '🗗' : '⛶', rect.x + rect.w - 20, rect.y + 16);
        }

        // 7. Small Stats Overlay Box (Number of Samples & Clusters Displayed)
        let drawnClustersCount = 0;
        if (useTiles) {
          jointTuplesMap.forEach(entry => {
            const clX = tileEngineX.clusters[entry.cx];
            const clY = tileEngineY.clusters[entry.cy];
            const clZ = currentDim >= 3 ? tileEngineZ.clusters[entry.cz] : { val: 0 };
            if (!clX || !clY || (currentDim >= 3 && !clZ)) return;
            const pr = getProjectedCoord({ x: clX.val, y: clY.val, z: clZ.val });
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);
            if (pos.px >= rect.x - 20 && pos.px <= rect.x + rect.w + 20 &&
                pos.py >= rect.y - 20 && pos.py <= rect.y + rect.h + 20) {
              drawnClustersCount++;
            }
          });
        } else {
          clusters.forEach(c => {
            const pr = getProjectedCoord(c);
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);
            if (pos.px >= rect.x - 20 && pos.px <= rect.x + rect.w + 20 &&
                pos.py >= rect.y - 20 && pos.py <= rect.y + rect.h + 20) {
              drawnClustersCount++;
            }
          });
        }

        let labelPts = "";
        if (isPointsTruncated) {
          const pct = Math.round((drawnPointsCount / totalVisiblePoints) * 100);
          const strDrn = drawnPointsCount.toLocaleString();
          const strVis = totalVisiblePoints.toLocaleString();
          labelPts = `⚠ ${strDrn}/${strVis} pts (${pct}%)`;
        } else if (totalVisiblePoints > 0) {
          labelPts = `${drawnPointsCount.toLocaleString()} pts`;
        } else {
          labelPts = `0 pts`;
        }

        const labelClust = `${drawnClustersCount.toLocaleString()} cl`;
        const fullText = `${labelPts}  •  ${labelClust}`;

        ctx.save();
        ctx.font = 'bold 9.5px monospace';
        const textW = ctx.measureText(fullText).width;
        const boxW = textW + 16;
        const boxH = 20;
        const boxX = rect.x + rect.w - boxW - 8;
        const boxY = rect.y + 28;

        if (isPointsTruncated) {
          ctx.fillStyle = 'rgba(28, 18, 8, 0.94)';
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
          ctx.lineWidth = 1.2;
        } else {
          ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
          ctx.lineWidth = 1.0;
        }
        ctx.beginPath();
        ctx.roundRect(boxX, boxY, boxW, boxH, 4);
        ctx.fill();
        ctx.stroke();

        const midBoxY = boxY + boxH / 2;
        let curX = boxX + 8;

        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';

        if (isPointsTruncated) {
          ctx.fillStyle = '#fbbf24';
        } else {
          ctx.fillStyle = '#cbd5e1';
        }
        ctx.fillText(labelPts, curX, midBoxY);
        curX += ctx.measureText(labelPts).width;

        ctx.fillStyle = 'rgba(100, 116, 139, 0.7)';
        ctx.fillText('  •  ', curX, midBoxY);
        curX += ctx.measureText('  •  ').width;

        ctx.fillStyle = '#38bdf8';
        ctx.fillText(labelClust, curX, midBoxY);
        ctx.restore();

        // 8. Dedicated Corner Zoom Box (One per Viewport)
        const view = quadViews[qIdx] || { zoom: 1.0 };
        const zoomPct = Math.round((view.zoom || 1.0) * 100);
        const zoomText = `Zoom: ${zoomPct}%`;

        ctx.save();
        ctx.font = 'bold 9.5px monospace';
        const zoomTextW = ctx.measureText(zoomText).width;
        const zoomBoxW = zoomTextW + 14;
        const zoomBoxH = 20;
        const zoomBoxX = boxX - zoomBoxW - 6;
        const zoomBoxY = rect.y + 28;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.strokeStyle = (zoomPct !== 100)
          ? 'rgba(74, 222, 128, 0.6)'
          : 'rgba(56, 189, 248, 0.35)';
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.roundRect(zoomBoxX, zoomBoxY, zoomBoxW, zoomBoxH, 4);
        ctx.fill();
        ctx.stroke();

        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = (zoomPct !== 100) ? '#4ade80' : '#94a3b8';
        ctx.fillText(zoomText, zoomBoxX + zoomBoxW / 2, zoomBoxY + zoomBoxH / 2);
        ctx.restore();

        if (typeof viewportZoomBoxRects !== 'undefined') {
          viewportZoomBoxRects[qIdx] = {
            x: zoomBoxX,
            y: zoomBoxY,
            w: zoomBoxW,
            h: zoomBoxH
          };
        }
      }

      ctx.restore();
    }

    // Note: 4-Panel Synchronized Reconstruction View moved to recon_renderer.js

