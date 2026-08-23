/**
 * GRIC Simulator - renderer.js
 * Part of the GRIC Interactive Algorithm Simulator
 */

//  6. QUAD-SCREEN VIEWPORT MANAGER & 3D RENDERING
    // =========================================================================

    function resizeCanvas() {
      const canvasWrapper = document.getElementById('canvasWrapper');
      if (!canvasWrapper) return;
      const availableW = canvasWrapper.clientWidth;
      const availableH = canvasWrapper.clientHeight;
      if (availableW <= 0 || availableH <= 0) return;

      const squareSize = Math.max(100, Math.floor(Math.min(availableW, availableH)));

      canvas.style.width = `${squareSize}px`;
      canvas.style.height = `${squareSize}px`;

      const dpr = window.devicePixelRatio || 1;
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
    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('load', resizeCanvas);
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(() => resizeCanvas());
      const cWrap = document.getElementById('canvasWrapper');
      if (cWrap) ro.observe(cWrap);
    }

    // Coordinate Transforms for Sub-Viewports
    function project3D(x, y, z, az, el) {
      const cosT = Math.cos(az), sinT = Math.sin(az);
      const cosP = Math.cos(el), sinP = Math.sin(el);
      
      const u = x * cosT + y * sinT;
      const v = -x * sinT * sinP + y * cosT * sinP + z * cosP;
      const depth = -x * sinT * cosP + y * cosT * cosP - z * sinP;
      return { u, v, depth };
    }

    function getQuadRect(qIdx, W, H) {
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

    function updateViewPresetBarPosition() {
      const bar = document.getElementById('viewPresetBar');
      if (!bar) return;

      const isImage = (typeof dataMode !== 'undefined' && dataMode === 'image');
      if (isImage) {
        bar.style.display = 'none';
        return;
      }

      bar.style.display = 'flex';

      const btnResetView = document.getElementById('btnResetView');
      const btnIso = document.getElementById('presetIso');
      const btnFront = document.getElementById('presetFront');
      const btnTop = document.getElementById('presetTop');
      const btnSide = document.getElementById('presetSide');
      const btnReset3D = document.getElementById('presetReset3D');

      const is3D = (currentDim === 3);
      const is3DOrbitVisible = is3D && (maximizedQuad === null || maximizedQuad === 3);

      if (btnIso) btnIso.style.display = is3DOrbitVisible ? '' : 'none';
      if (btnFront) btnFront.style.display = is3DOrbitVisible ? '' : 'none';
      if (btnTop) btnTop.style.display = is3DOrbitVisible ? '' : 'none';
      if (btnSide) btnSide.style.display = is3DOrbitVisible ? '' : 'none';
      if (btnReset3D) btnReset3D.style.display = is3DOrbitVisible ? '' : 'none';
      if (btnResetView) {
        btnResetView.style.display = '';
        btnResetView.textContent = is3D ? '🔍 1:1' : '🔍 1:1 Reset View';
      }

      const canvasWrapper = document.getElementById('canvasWrapper');
      if (!canvasWrapper || !canvas) return;

      const wrapRect = canvasWrapper.getBoundingClientRect();
      const cRect = canvas.getBoundingClientRect();

      let targetLeft, targetTop, targetWidth, targetHeight;
      if (!is3D || maximizedQuad !== null) {
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

        if (btnIso) btnIso.classList.toggle('active', Math.abs(azDeg - (-35)) <= 2 && Math.abs(elDeg - 25) <= 2);
        if (btnFront) btnFront.classList.toggle('active', Math.abs(azDeg - 0) <= 2 && Math.abs(elDeg - 0) <= 2);
        if (btnTop) btnTop.classList.toggle('active', Math.abs(azDeg - 0) <= 2 && Math.abs(elDeg - 89) <= 2);
        if (btnSide) btnSide.classList.toggle('active', Math.abs(azDeg - 90) <= 2 && Math.abs(elDeg - 0) <= 2);
      }
    }
    window.updateViewPresetBarPosition = updateViewPresetBarPosition;

    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.width / dpr;
      const H = canvas.height / dpr;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#0b1120';
      ctx.fillRect(0, 0, W, H);

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
      if (maximizedQuad !== null) {
        const types = ["ALONG_X", "ALONG_Y", "ALONG_Z", "CUSTOM_3D"];
        renderSubViewport(maximizedQuad, types[maximizedQuad], { x: 0, y: 0, w: W, h: H });
        updateViewPresetBarPosition();
        return;
      }

      // Render 4 Quadrants
      renderSubViewport(0, "ALONG_X", getQuadRect(0, W, H));
      renderSubViewport(1, "ALONG_Y", getQuadRect(1, W, H));
      renderSubViewport(2, "ALONG_Z", getQuadRect(2, W, H));
      renderSubViewport(3, "CUSTOM_3D", getQuadRect(3, W, H));

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

      const scale = getQuadScale(qIdx, rect);
      const is3DCustom = (viewType === "CUSTOM_3D");

      // Projection point helper
      function getProjectedCoord(p) {
        if (viewType === "ALONG_X") return { u: p.y, v: p.z, depth: p.x };
        if (viewType === "ALONG_Y") return { u: p.x, v: p.z, depth: p.y };
        if (viewType === "ALONG_Z" || viewType === "2D") return { u: p.x, v: p.y, depth: p.z };
        // CUSTOM_3D
        return project3D(p.x, p.y, p.z, orbitCamera.azimuth, orbitCamera.elevation);
      }

      // 1. Grid & Axes
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
          const p1 = mapMetricToQuad(project3D(val, -b, -b, az, el).u, project3D(val, -b, -b, az, el).v, qIdx, rect);
          const p2 = mapMetricToQuad(project3D(val,  b, -b, az, el).u, project3D(val,  b, -b, az, el).v, qIdx, rect);
          ctx.beginPath(); ctx.moveTo(p1.px, p1.py); ctx.lineTo(p2.px, p2.py); ctx.stroke();

          const p3 = mapMetricToQuad(project3D(-b, val, -b, az, el).u, project3D(-b, val, -b, az, el).v, qIdx, rect);
          const p4 = mapMetricToQuad(project3D( b, val, -b, az, el).u, project3D( b, val, -b, az, el).v, qIdx, rect);
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

      // 2. Sample Point Cloud (Processed vs Unprocessed Differentiation)
      let drawnPointsCount = 0;
      const numPast = pastSamples.length;
      if (showPastSamples && numPast > 0 && pointAlpha > 0.001) {
        const basePtRad = samplePointSize;
        const maxDraw = maxDrawPoints;
        const stride = numPast > maxDraw ? Math.ceil(numPast / maxDraw) : 1;

        // Distinct colors for unprocessed vs processed sample points
        const unprocColor = `rgba(100, 116, 139, ${(pointAlpha * 0.40).toFixed(3)})`;
        const procDefaultColor = `rgba(56, 189, 248, ${pointAlpha.toFixed(3)})`;

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

        if (!is3DCustom) {
          const ptRad = Math.max(0.5, basePtRad * ptSizeScale);
          const d = ptRad * 2;

          // Pass 1: Unprocessed Points (staged, not yet ingested)
          ctx.fillStyle = unprocColor;
          for (let i = 0; i < numPast; i += stride) {
            const pt = pastSamples[i];
            const isProcessed = (pt.clusterId !== undefined && pt.clusterId >= 0) ||
                                (pt.frameIndex !== undefined && pt.frameIndex < currentFrameIdx) ||
                                (i < currentFrameIdx);
            if (isProcessed) continue;

            const pr = getProjectedCoord(pt);
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);
            if (pos.px < rect.x - 5 || pos.px > rect.x + rect.w + 5 || 
                pos.py < rect.y - 5 || pos.py > rect.y + rect.h + 5) continue;
            ctx.fillRect(pos.px - ptRad, pos.py - ptRad, d, d);
            drawnPointsCount++;
          }

          // Pass 2: Processed Points (clustered / ingested)
          let lastFill = null;
          for (let i = 0; i < numPast; i += stride) {
            const pt = pastSamples[i];
            const isProcessed = (pt.clusterId !== undefined && pt.clusterId >= 0) ||
                                (pt.frameIndex !== undefined && pt.frameIndex < currentFrameIdx) ||
                                (i < currentFrameIdx);
            if (!isProcessed) continue;

            const pr = getProjectedCoord(pt);
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);
            if (pos.px < rect.x - 5 || pos.px > rect.x + rect.w + 5 || 
                pos.py < rect.y - 5 || pos.py > rect.y + rect.h + 5) continue;

            const fillColor = (pt.clusterId !== undefined && pt.clusterId >= 0) 
                              ? getCachedColor(pt.clusterId) 
                              : procDefaultColor;
            if (fillColor !== lastFill) {
              ctx.fillStyle = fillColor;
              lastFill = fillColor;
            }
            ctx.fillRect(pos.px - ptRad, pos.py - ptRad, d, d);
            drawnPointsCount++;
          }
        } else {
          // 3D Perspective Orbit View
          // Pass 1: Unprocessed Points
          ctx.fillStyle = unprocColor;
          for (let i = 0; i < numPast; i += stride) {
            const pt = pastSamples[i];
            const isProcessed = (pt.clusterId !== undefined && pt.clusterId >= 0) ||
                                (pt.frameIndex !== undefined && pt.frameIndex < currentFrameIdx) ||
                                (i < currentFrameIdx);
            if (isProcessed) continue;

            const pr = getProjectedCoord(pt);
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);
            if (pos.px < rect.x - 5 || pos.px > rect.x + rect.w + 5 || 
                pos.py < rect.y - 5 || pos.py > rect.y + rect.h + 5) continue;
            const depthFactor = Math.max(0.4, Math.min(2.2, 1.0 + pr.depth * 0.5));
            const ptRad = Math.max(0.4, basePtRad * depthFactor * ptSizeScale);
            ctx.fillRect(pos.px - ptRad, pos.py - ptRad, ptRad * 2, ptRad * 2);
            drawnPointsCount++;
          }

          // Pass 2: Processed Points
          let lastFill = null;
          for (let i = 0; i < numPast; i += stride) {
            const pt = pastSamples[i];
            const isProcessed = (pt.clusterId !== undefined && pt.clusterId >= 0) ||
                                (pt.frameIndex !== undefined && pt.frameIndex < currentFrameIdx) ||
                                (i < currentFrameIdx);
            if (!isProcessed) continue;

            const pr = getProjectedCoord(pt);
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);
            if (pos.px < rect.x - 5 || pos.px > rect.x + rect.w + 5 || 
                pos.py < rect.y - 5 || pos.py > rect.y + rect.h + 5) continue;
            const depthFactor = Math.max(0.4, Math.min(2.2, 1.0 + pr.depth * 0.5));
            const ptRad = Math.max(0.4, basePtRad * depthFactor * ptSizeScale);

            const fillColor = (pt.clusterId !== undefined && pt.clusterId >= 0) 
                              ? getCachedColor(pt.clusterId) 
                              : procDefaultColor;
            if (fillColor !== lastFill) {
              ctx.fillStyle = fillColor;
              lastFill = fillColor;
            }
            ctx.fillRect(pos.px - ptRad, pos.py - ptRad, ptRad * 2, ptRad * 2);
            drawnPointsCount++;
          }
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
            const clZ = currentDim === 3 ? tileEngineZ.clusters[entry.cz] : { val: 0 };
            if (!clX || !clY || (currentDim === 3 && !clZ)) return;

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
          const clZ = currentDim === 3 ? tileEngineZ.clusters[entry.cz] : { val: 0 };
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

            const tagText = `Tuple (${entry.cx}, ${entry.cy}${currentDim === 3 ? `, ${entry.cz}` : ''}) [${entry.count} frames]`;
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

            if (!showCircleMembers && !showCircleSCDists) {
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
                
                ctx.save();
                ctx.strokeStyle = c.color;
                ctx.globalAlpha = clusterAlpha * 0.35; // Fainter, subtle wireframe
                ctx.lineWidth = 0.75;

                // 1. Circle perpendicular to Z-axis (XY plane)
                ctx.beginPath();
                for (let s = 0; s <= CIRCLE_LUT_STEPS; s++) {
                  const rx = c.x + rlim * CIRCLE_COS[s];
                  const ry = c.y + rlim * CIRCLE_SIN[s];
                  const rz = c.z;
                  const rp = project3D(rx, ry, rz, az, el);
                  const ppos = mapMetricToQuad(rp.u, rp.v, qIdx, rect);
                  if (s === 0) ctx.moveTo(ppos.px, ppos.py);
                  else ctx.lineTo(ppos.px, ppos.py);
                }
                ctx.stroke();

                // 2. Circle perpendicular to Y-axis (XZ plane)
                ctx.beginPath();
                for (let s = 0; s <= CIRCLE_LUT_STEPS; s++) {
                  const rx = c.x + rlim * CIRCLE_COS[s];
                  const ry = c.y;
                  const rz = c.z + rlim * CIRCLE_SIN[s];
                  const rp = project3D(rx, ry, rz, az, el);
                  const ppos = mapMetricToQuad(rp.u, rp.v, qIdx, rect);
                  if (s === 0) ctx.moveTo(ppos.px, ppos.py);
                  else ctx.lineTo(ppos.px, ppos.py);
                }
                ctx.stroke();

                // 3. Circle perpendicular to X-axis (YZ plane)
                ctx.beginPath();
                for (let s = 0; s <= CIRCLE_LUT_STEPS; s++) {
                  const rx = c.x;
                  const ry = c.y + rlim * CIRCLE_COS[s];
                  const rz = c.z + rlim * CIRCLE_SIN[s];
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
            // Area within circle is strictly proportional to c.members => Radius is proportional to sqrt(c.members)
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
            // Area within circle is strictly proportional to c.scDists => Radius is proportional to sqrt(c.scDists)
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
          if (useTM && transitionCounts.length > 0 && clusters.length > 1) {
            ctx.save();
            const K = clusters.length;

            // 1. Draw Active Frame Transition (Gold Beam)
            if (lastTransitionFrom >= 0 && lastTransitionTo >= 0 && 
                lastTransitionFrom < K && lastTransitionTo < K && 
                lastTransitionFrom !== lastTransitionTo && 
                clusters[lastTransitionFrom] && clusters[lastTransitionTo]) {
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

            // Anchor Core Node
            ctx.beginPath();
            ctx.arc(pos.px, pos.py, 5.5, 0, Math.PI * 2);
            ctx.fillStyle = c.color;
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.2;
            ctx.stroke();

            ctx.fillStyle = '#f8fafc';
            ctx.font = '10px sans-serif';
            ctx.fillText(`C${c.id}`, pos.px + 7, pos.py - 5);
          });

          // Pruned Crosshairs
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
          const tagText = `C${clObj.id} [${clObj.members}f] (${clObj.x.toFixed(2)}, ${clObj.y.toFixed(2)}${currentDim === 3 ? `, ${clObj.z.toFixed(2)}` : ''})`;
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
          ctx.font = 'bold 9px monospace';
          ctx.fillStyle = '#facc15';
          const fIdx = targetSample.frameIndex || 0;
          ctx.fillText(`Point #${fIdx}`, posP.px + 9, posP.py - 5);
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

            // If no evaluation list in logs but assigned cluster exists, include assigned cluster match
            if (evaluatedClusters.length === 0 && cId >= 0 && clusters && cId < clusters.length && clusters[cId]) {
              const clObj = clusters[cId];
              const dx = pt.x - clObj.x;
              const dy = pt.y - clObj.y;
              const dz = currentDim === 3 ? (pt.z - clObj.z) : 0;
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
              const pillText = `${orderSuffix}: C${clObj.id} (d=${ev.dist.toFixed(3)}) ${ev.match ? '✓' : '✗'}`;
              drawDistPill(midX, midY, pillText, lineColor, ev.match);
            });

            // B. Sample-to-Sample nearest neighbor distances computed when solving FOR this query point in k-NN
            // (Distinct Vibrant Violet / Magenta palette #c084fc / #e879f9 to clearly distinguish from cluster lines)
            if (typeof knnResults !== 'undefined' && knnResults && knnResults.indices) {
              const N = knnResults.totalFrames;
              const k = knnResults.k;
              if (sampleIdx >= 0 && sampleIdx < N) {
                for (let r = 0; r < k; r++) {
                  const nId = knnResults.indices[sampleIdx * k + r];
                  const dist = knnResults.distances[sampleIdx * k + r];
                  if (nId >= 0 && nId !== sampleIdx) {
                    const nPt = getFramePoint(nId);
                    if (!nPt) continue; // Ensure no lines to non-existing points

                    const prN = getProjectedCoord(nPt);
                    const posN = mapMetricToQuad(prN.u, prN.v, qIdx, rect);
                    const isRank1 = (r === 0);
                    const nnColor = isRank1 ? '#e879f9' : '#c084fc';
                    const orderSuffix = getOrdinalSuffix(r + 1);

                    ctx.beginPath();
                    ctx.moveTo(pos.px, pos.py);
                    ctx.lineTo(posN.px, posN.py);
                    ctx.strokeStyle = nnColor;
                    ctx.lineWidth = isRank1 ? 2.4 : 1.8;
                    ctx.stroke();

                    // Highlight neighbor point node
                    ctx.beginPath();
                    ctx.arc(posN.px, posN.py, isRank1 ? 5.0 : 4.0, 0, Math.PI * 2);
                    ctx.fillStyle = nnColor;
                    ctx.fill();
                    ctx.strokeStyle = '#0f172a';
                    ctx.lineWidth = 1.2;
                    ctx.stroke();

                    // Distance pill along ray (at 50% midpoint)
                    const midX = (pos.px + posN.px) / 2;
                    const midY = (pos.py + posN.py) / 2;
                    const pillText = `${orderSuffix} NN: #${nId} (d=${dist.toFixed(3)})`;
                    drawDistPill(midX, midY, pillText, nnColor, isRank1);
                  }
                }
              }
            }

            // -----------------------------------------------------------------
            // 2. OTHER Distance Computations INVOLVING THIS POINT
            //    (Thinner dashed lines in distinct amber/orange, e.g. when other points queried this point)
            // -----------------------------------------------------------------
            if (typeof knnResults !== 'undefined' && knnResults && knnResults.indices) {
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

                    // Mini distance pill at 35% towards source query
                    const midX = pos.px * 0.65 + posM.px * 0.35;
                    const midY = pos.py * 0.65 + posM.py * 0.35;
                    const pillText = `#${m}→#${sampleIdx}: d=${dist.toFixed(3)}`;
                    drawDistPill(midX, midY, pillText, otherColor, false);
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

            // 6. Detailed Info Callout Badge (in the hovered quadrant or 2D screen)
            if (qIdx === activeSampleHighlight.qIdx || currentDim === 2 || maximizedQuad !== null) {
              const fIdx = activeSampleHighlight.index;
              const coordText = currentDim === 3 
                ? `(${pt.x.toFixed(2)}, ${pt.y.toFixed(2)}, ${pt.z.toFixed(2)})`
                : `(${pt.x.toFixed(2)}, ${pt.y.toFixed(2)})`;

              const lockTag = isLocked ? '🔒 LOCKED #' : '#';
              const titleLine = `${lockTag}${fIdx}: ${coordText} ${cId >= 0 ? `→ C${cId}` : '(new cluster)'}`;
              const calloutLines = [titleLine];

              if (evaluatedClusters.length > 1) {
                const seqParts = evaluatedClusters.map((ev, idx) => 
                  `${getOrdinalSuffix(idx + 1)}: C${ev.cluster.id} (d=${ev.dist.toFixed(3)}${ev.match ? ' ✓' : ' ✗'})`
                );
                calloutLines.push(`Solving Sequence: ${seqParts.join(' → ')}`);
              } else if (evaluatedClusters.length === 1) {
                const ev = evaluatedClusters[0];
                calloutLines.push(`1st eval: C${ev.cluster.id} (d=${ev.dist.toFixed(3)}${ev.match ? ' ✓' : ' ✗'})`);
              }

              if (isLocked) {
                calloutLines.push('Click point/canvas or press Esc to unlock');
              }

              ctx.font = 'bold 9.5px monospace';
              let maxLineW = 0;
              calloutLines.forEach(ln => {
                const w = ctx.measureText(ln).width;
                if (w > maxLineW) maxLineW = w;
              });

              const lineH = 14;
              const boxW = Math.min(rect.w - 16, maxLineW + 16);
              const boxH = calloutLines.length * lineH + 8;
              const badgeX = rect.x + 8;
              const badgeY = rect.y + 28;

              ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
              ctx.strokeStyle = isLocked ? '#f59e0b' : '#38bdf8';
              ctx.lineWidth = isLocked ? 1.5 : 1.2;
              ctx.beginPath();
              ctx.roundRect(badgeX, badgeY, boxW, boxH, 4);
              ctx.fill();
              ctx.stroke();

              ctx.textAlign = 'left';
              ctx.textBaseline = 'middle';
              calloutLines.forEach((ln, lIdx) => {
                if (lIdx === 0) {
                  ctx.fillStyle = isLocked ? '#fbbf24' : '#38bdf8';
                } else if (isLocked && lIdx === calloutLines.length - 1) {
                  ctx.fillStyle = '#94a3b8';
                } else {
                  ctx.fillStyle = '#e2e8f0';
                }
                ctx.fillText(ln, badgeX + 8, badgeY + 6 + lIdx * lineH + lineH / 2);
              });
            }

            ctx.restore();
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
            ctx.lineTo(headX - headLen * Math.cos(angle - Math.PI / 6), headY - headLen * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(headX - headLen * Math.cos(angle + Math.PI / 6), headY - headLen * Math.sin(angle + Math.PI / 6));
            ctx.closePath();
            ctx.fill();

            // Label badge
            const badgeText = `C${fromId} ↺ C${toId} [${(prob * 100).toFixed(1)}%, N=${cnt}]`;
            ctx.font = 'bold 10px monospace';
            const tw = ctx.measureText(badgeText).width;
            const badgeX = Math.max(rect.x + 10, Math.min(rect.x + rect.w - tw - 10, loopCx - tw / 2));
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
            ctx.lineTo(endX - headLen * Math.cos(angle - Math.PI / 6.5), endY - headLen * Math.sin(angle - Math.PI / 6.5));
            ctx.lineTo(endX - (headLen * 0.6) * Math.cos(angle), endY - (headLen * 0.6) * Math.sin(angle));
            ctx.lineTo(endX - headLen * Math.cos(angle + Math.PI / 6.5), endY - headLen * Math.sin(angle + Math.PI / 6.5));
            ctx.closePath();
            ctx.fill();

            // Label badge along arrow
            const badgeText = `C${fromId} → C${toId} [${(prob * 100).toFixed(1)}%, N=${cnt}]`;
            ctx.font = 'bold 10px monospace';
            const tw = ctx.measureText(badgeText).width;
            const midX = (startX + endX) / 2;
            const midY = (startY + endY) / 2 - 10;

            const badgeX = Math.max(rect.x + 10, Math.min(rect.x + rect.w - tw - 10, midX - tw / 2));
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
          ctx.restore();
        }

        // Current Evaluated Rays
        if (currentFrame) {
          const prFrame = getProjectedCoord(currentFrame);
          const posFrame = mapMetricToQuad(prFrame.u, prFrame.v, qIdx, rect);

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

          // Active Query Frame Node (f_i)
          ctx.beginPath();
          ctx.arc(posFrame.px, posFrame.py, 6.5, 0, Math.PI * 2);
          ctx.fillStyle = '#facc15';
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.8;
          ctx.stroke();

          ctx.fillStyle = '#facc15';
          ctx.font = 'bold 10px sans-serif';
          ctx.fillText("fi", posFrame.px + 8, posFrame.py + 10);
        }

        // 4b. Draw k-NN Graph Overlay (Query Point -> Top-k Nearest Neighbors)
        if (typeof enableKnn !== 'undefined' && enableKnn &&
            typeof showKnnLines !== 'undefined' && showKnnLines &&
            typeof knnResults !== 'undefined' && knnResults && knnResults.indices) {
          const N = knnResults.totalFrames;
          const k = knnResults.k;
          const pts = (typeof benchmarkDataset !== 'undefined' && benchmarkDataset && benchmarkDataset.length > 0) ?
                      benchmarkDataset : (typeof pastSamples !== 'undefined' ? pastSamples : []);

          let qId = (typeof selectedKnnQuerySample !== 'undefined') ? selectedKnnQuerySample : -1;
          if (qId < 0 || qId >= N) {
            qId = (typeof currentFrameIdx !== 'undefined' && currentFrameIdx > 0 && currentFrameIdx <= N) ?
                  currentFrameIdx - 1 : 0;
          }

          if (qId >= 0 && qId < pts.length) {
            const queryPt = pts[qId];
            const prQ = getProjectedCoord(queryPt);
            const posQ = mapMetricToQuad(prQ.u, prQ.v, qIdx, rect);

            ctx.save();

            // Draw vector rays from query to each of its k nearest neighbors
            for (let r = 0; r < k; r++) {
              const nId = knnResults.indices[qId * k + r];
              if (nId < 0 || nId >= pts.length) continue;

              const nPt = pts[nId];
              const prN = getProjectedCoord(nPt);
              const posN = mapMetricToQuad(prN.u, prN.v, qIdx, rect);

              const isRank1 = (r === 0);
              const isHovered = (typeof hoveredKnnNeighborId !== 'undefined' && hoveredKnnNeighborId === nId);

              ctx.beginPath();
              ctx.moveTo(posQ.px, posQ.py);
              ctx.lineTo(posN.px, posN.py);
              ctx.strokeStyle = isHovered ? '#e879f9' :
                                (isRank1 ? 'rgba(232, 121, 249, 0.95)' :
                                 `rgba(192, 132, 252, ${Math.max(0.20, 0.75 - r * 0.06).toFixed(2)})`);
              ctx.lineWidth = isHovered ? 2.4 : (isRank1 ? 2.0 : 1.2);
              ctx.stroke();

              // Highlight neighbor node
              ctx.beginPath();
              ctx.arc(posN.px, posN.py, isHovered ? 6.0 : (isRank1 ? 4.8 : 3.5), 0, Math.PI * 2);
              ctx.fillStyle = isHovered ? '#e879f9' : (isRank1 ? '#c084fc' : '#a855f7');
              ctx.fill();
              ctx.strokeStyle = '#f8fafc';
              ctx.lineWidth = 1;
              ctx.stroke();

              // Rank label for top neighbors
              if (r < 3 || isHovered) {
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
            ctx.fillStyle = '#e879f9';
            ctx.fill();

            ctx.font = 'bold 9px monospace';
            ctx.fillStyle = '#e879f9';
            ctx.fillText(`Query #${qId}`, posQ.px + 10, posQ.py - 6);

            ctx.restore();
          }
        }
      }

      // 5. 3D Coordinate RGB Triad Gizmo (for Custom 3D Viewport)
      if (is3DCustom) {
        const gizmoOrigin = { px: rect.x + 36, py: rect.y + rect.h - 36 };
        const gLen = 24;
        const az = orbitCamera.azimuth, el = orbitCamera.elevation;

        const gx = project3D(1, 0, 0, az, el);
        const gy = project3D(0, 1, 0, az, el);
        const gz = project3D(0, 0, 1, az, el);

        const axes = [
          { name: 'X', u: gx.u, v: gx.v, color: '#f87171' },
          { name: 'Y', u: gy.u, v: gy.v, color: '#4ade80' },
          { name: 'Z', u: gz.u, v: gz.v, color: '#38bdf8' }
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

      // 6. Viewport Header Overlay & Maximize Button
      let title = "";
      let subtitle = "";
      if (viewType === "ALONG_X") {
        title = "📐 Along X (Y-Z Plane)";
        subtitle = "H: +Y ➔ | V: +Z ⬆";
      } else if (viewType === "ALONG_Y") {
        title = "📐 Along Y (X-Z Plane)";
        subtitle = "H: +X ➔ | V: +Z ⬆";
      } else if (viewType === "ALONG_Z") {
        title = "📐 Along Z (X-Y Plane)";
        subtitle = "H: +X ➔ | V: +Y ⬆";
      } else if (viewType === "2D") {
        title = "📐 2D Standard View (X-Y Plane)";
        subtitle = "H: +X ➔ | V: +Y ⬆";
      } else if (viewType === "CUSTOM_3D") {
        const degAz = Math.round(orbitCamera.azimuth * 180 / Math.PI);
        const degEl = Math.round(orbitCamera.elevation * 180 / Math.PI);
        title = `🌐 3D Orbit View [θ: ${degAz}°, φ: ${degEl}°]`;
        subtitle = "Drag to Rotate Camera";
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
      ctx.fillText(title, rect.x + 8, rect.y + 16);

      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px monospace';
      ctx.fillText(subtitle, rect.x + rect.w - 180, rect.y + 16);

      // Maximize / Restore Icon
      if (currentDim === 3) {
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
          const clZ = currentDim === 3 ? tileEngineZ.clusters[entry.cz] : { val: 0 };
          if (!clX || !clY || (currentDim === 3 && !clZ)) return;
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

      const labelPts = `${drawnPointsCount.toLocaleString()} pts`;
      const labelClust = `${drawnClustersCount.toLocaleString()} cl`;
      const fullText = `${labelPts}  •  ${labelClust}`;

      ctx.save();
      ctx.font = 'bold 9.5px monospace';
      const textW = ctx.measureText(fullText).width;
      const boxW = textW + 14;
      const boxH = 18;
      const boxX = rect.x + rect.w - boxW - 8;
      const boxY = rect.y + 28;

      ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.roundRect(boxX, boxY, boxW, boxH, 4);
      ctx.fill();
      ctx.stroke();

      const midBoxY = boxY + boxH / 2;
      let curX = boxX + 7;

      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#cbd5e1';
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
      const zoomBoxH = 18;
      const zoomBoxX = boxX - zoomBoxW - 6;
      const zoomBoxY = rect.y + 28;

      ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
      ctx.strokeStyle = (zoomPct !== 100) ? 'rgba(74, 222, 128, 0.6)' : 'rgba(56, 189, 248, 0.35)';
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

      ctx.restore();
    }

    // =========================================================================
