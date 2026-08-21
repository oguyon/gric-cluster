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
      if (heatWrap && heatWrap.clientWidth > 0 && heatWrap.clientHeight > 0) {
        tmCanvasDimensions.tmHeatmapCanvas = {
          w: Math.floor(heatWrap.clientWidth),
          h: Math.floor(heatWrap.clientHeight)
        };
      }
    }
    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('load', resizeCanvas);
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(() => resizeCanvas());
      const cWrap = document.getElementById('canvasWrapper');
      if (cWrap) ro.observe(cWrap);
      const tmHeatWrap = document.getElementById('tmHeatmapWrapper');
      if (tmHeatWrap) ro.observe(tmHeatWrap);
      if (document.body) ro.observe(document.body);
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

    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.width / dpr;
      const H = canvas.height / dpr;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#0b1120';
      ctx.fillRect(0, 0, W, H);

      // --- 2D MODE: Single Screen ---
      if (currentDim === 2) {
        renderSubViewport(2, "2D", { x: 0, y: 0, w: W, h: H });
        return;
      }

      // --- 3D MODE: Quad-Split or Maximized View ---
      if (maximizedQuad !== null) {
        const types = ["ALONG_X", "ALONG_Y", "ALONG_Z", "CUSTOM_3D"];
        renderSubViewport(maximizedQuad, types[maximizedQuad], { x: 0, y: 0, w: W, h: H });
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

      // 2. Past Samples Point Cloud (Optimized Batch & Stride Subsampling)
      const numPast = pastSamples.length;
      if (showPastSamples && numPast > 0 && pointAlpha > 0.001) {
        ctx.fillStyle = `rgba(148, 163, 184, ${pointAlpha.toFixed(3)})`;
        const basePtRad = samplePointSize;
        const maxDraw = maxDrawPoints;
        const stride = numPast > maxDraw ? Math.ceil(numPast / maxDraw) : 1;

        if (!is3DCustom) {
          const ptRad = Math.max(0.5, basePtRad * ptSizeScale);
          const d = ptRad * 2;
          ctx.beginPath();
          for (let i = 0; i < numPast; i += stride) {
            const pr = getProjectedCoord(pastSamples[i]);
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);
            if (pos.px < rect.x - 5 || pos.px > rect.x + rect.w + 5 || 
                pos.py < rect.y - 5 || pos.py > rect.y + rect.h + 5) continue;
            ctx.rect(pos.px - ptRad, pos.py - ptRad, d, d);
          }
          ctx.fill();
        } else {
          for (let i = 0; i < numPast; i += stride) {
            const pr = getProjectedCoord(pastSamples[i]);
            const pos = mapMetricToQuad(pr.u, pr.v, qIdx, rect);
            if (pos.px < rect.x - 5 || pos.px > rect.x + rect.w + 5 || 
                pos.py < rect.y - 5 || pos.py > rect.y + rect.h + 5) continue;
            const depthFactor = Math.max(0.4, Math.min(2.2, 1.0 + pr.depth * 0.5));
            const ptRad = Math.max(0.4, basePtRad * depthFactor * ptSizeScale);
            ctx.fillRect(pos.px - ptRad, pos.py - ptRad, ptRad * 2, ptRad * 2);
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

      ctx.restore();
    }

    // =========================================================================
