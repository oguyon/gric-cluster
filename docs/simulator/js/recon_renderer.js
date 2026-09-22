/**
 * GRIC Simulator - recon_renderer.js
 * 4-Panel Synchronized Reconstruction View (A, B, C, D) and neighbor visualizations.
 */

    // =========================================================================
    //  4-PANEL SYNCHRONIZED RECONSTRUCTION VIEW (A, B, C, D)
    // =========================================================================

    /**
     * Compute or retrieve k-NN neighbors for query C[activeQueryIdx]
     */
    function getOrComputeKnnNeighbors(activeQueryIdx, k = 10) {
      const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;
      if (slotD && slotD.reconstructionSourceNeighbors &&
          slotD.reconstructionSourceNeighbors[activeQueryIdx]) {
        return slotD.reconstructionSourceNeighbors[activeQueryIdx];
      }
      const slotA = (typeof datasetSlots !== 'undefined') ? datasetSlots['A'] : null;
      const slotC = (typeof datasetSlots !== 'undefined') ? datasetSlots['C'] : null;
      const ptsA = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotA)
        : (slotA && slotA.benchmarkDataset && slotA.benchmarkDataset.length > 0
            ? slotA.benchmarkDataset : (slotA ? slotA.pastSamples : null));
      const ptsC = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotC)
        : (slotC && slotC.benchmarkDataset && slotC.benchmarkDataset.length > 0
            ? slotC.benchmarkDataset : (slotC ? slotC.pastSamples : null));
      if (!ptsA || !ptsC || activeQueryIdx < 0 || activeQueryIdx >= ptsC.length) return [];

      if (!slotC._onDemandKnnCache) slotC._onDemandKnnCache = new Map();
      if (slotC._onDemandKnnCache.has(activeQueryIdx)) {
        return slotC._onDemandKnnCache.get(activeQueryIdx);
      }

      const qc = ptsC[activeQueryIdx];
      const qx = qc.x, qy = qc.y, qz = (typeof qc.z === 'number') ? qc.z : 0.0;
      const dimA = slotA.currentDim || 2;
      const numCandidates = ptsA.length;
      const effK = Math.min(k, numCandidates);

      const dists = new Float64Array(numCandidates);
      const indices = new Int32Array(numCandidates);
      for (let j = 0; j < numCandidates; j++) {
        const pa = ptsA[j];
        const dx = qx - pa.x;
        const dy = qy - pa.y;
        const dz = (dimA >= 3 && typeof pa.z === 'number') ? (qz - pa.z) : 0.0;
        dists[j] = Math.sqrt(dx * dx + dy * dy + dz * dz);
        indices[j] = j;
      }

      for (let p = 0; p < effK; p++) {
        let minIdx = p;
        for (let j = p + 1; j < numCandidates; j++) {
          if (dists[j] < dists[minIdx]) minIdx = j;
        }
        const tmpD = dists[p]; dists[p] = dists[minIdx]; dists[minIdx] = tmpD;
        const tmpI = indices[p]; indices[p] = indices[minIdx]; indices[minIdx] = tmpI;
      }

      const weightMode = (slotD && slotD.reconstructionInfo)
        ? slotD.reconstructionInfo.weightMode : 'uniform';

      const result = [];
      if (weightMode === 'uniform') {
        const normW = 1.0 / effK;
        for (let p = 0; p < effK; p++) {
          result.push({
            id: indices[p],
            dist: dists[p],
            weight: normW
          });
        }
      } else {
        let sumW = 0.0;
        const weights = new Float64Array(effK);
        for (let p = 0; p < effK; p++) {
          const d = dists[p];
          const w = 1.0 / Math.max(d, 1e-7);
          weights[p] = w;
          sumW += w;
        }
        for (let p = 0; p < effK; p++) {
          result.push({
            id: indices[p],
            dist: dists[p],
            weight: sumW > 0 ? (weights[p] / sumW) : (1.0 / effK)
          });
        }
      }

      if (slotC._onDemandKnnCache.size > 200) {
        slotC._onDemandKnnCache.clear();
      }
      slotC._onDemandKnnCache.set(activeQueryIdx, result);
      return result;
    }

    /**
     * Helper to find all queries C_i that include targetTrainingIdx in their k-NN neighbor set.
     * Uses precomputed mapping from slotD with lazy inverted index.
     * If D has not been reconstructed, returns [] immediately to prevent UI thread freeze.
     */
    function getOrComputeReverseKnnNeighbors(targetTrainingIdx, k = 10) {
      const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;
      const slotA = (typeof datasetSlots !== 'undefined') ? datasetSlots['A'] : null;
      const slotC = (typeof datasetSlots !== 'undefined') ? datasetSlots['C'] : null;
      const ptsA = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotA)
        : (slotA && slotA.benchmarkDataset && slotA.benchmarkDataset.length > 0
            ? slotA.benchmarkDataset : (slotA ? slotA.pastSamples : null));
      const ptsC = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotC)
        : (slotC && slotC.benchmarkDataset && slotC.benchmarkDataset.length > 0
            ? slotC.benchmarkDataset : (slotC ? slotC.pastSamples : null));

      if (!ptsA || !ptsC || targetTrainingIdx < 0 || targetTrainingIdx >= ptsA.length) {
        return [];
      }

      const mapping = (slotD && slotD.reconstructionSourceNeighbors)
        ? slotD.reconstructionSourceNeighbors : null;

      // If D has not been reconstructed, no reverse mapping exists yet
      if (!mapping || mapping.length !== ptsC.length) {
        return [];
      }

      // Fast inverted index: build once in O(k * N_c), subsequent lookups are O(1)
      if (!slotD._reverseNeighborsIndex) {
        const revIdx = {};
        for (let i = 0; i < mapping.length; i++) {
          const nbs = mapping[i];
          if (!nbs) continue;
          for (let r = 0; r < nbs.length; r++) {
            const trId = nbs[r].id;
            if (!revIdx[trId]) revIdx[trId] = [];
            revIdx[trId].push({
              queryIdx: i,
              rank: r + 1,
              dist: nbs[r].dist,
              weight: nbs[r].weight
            });
          }
        }
        slotD._reverseNeighborsIndex = revIdx;
      }

      return slotD._reverseNeighborsIndex[targetTrainingIdx] || [];
    }

    /**
     * Retrieves computed k-NN neighbors for a given sample in dataset slot A or B.
     */
    function getSlotKnnNeighbors(slotId, pointIdx) {
      const slot = datasetSlots[slotId];
      const knn = (slot && slot.knnResults) ? slot.knnResults
        : (activeDatasetSlot === slotId && typeof knnResults !== 'undefined' ? knnResults : null);
      if (!knn || pointIdx < 0) return null;
      const k = knn.k || (slot ? slot.knnK : 10) || (typeof knnK !== 'undefined' ? knnK : 10);

      if (knn.indices && knn.indices.length >= (pointIdx + 1) * k) {
        const neighbors = [];
        for (let r = 0; r < k; r++) {
          const idx = knn.indices[pointIdx * k + r];
          const dist = (knn.distances && knn.distances.length > pointIdx * k + r)
            ? knn.distances[pointIdx * k + r] : 0.0;
          if (idx >= 0) {
            neighbors.push({ index: idx, dist: dist, rank: r + 1 });
          }
        }
        return neighbors;
      }
      if (knn.queries && knn.queries[pointIdx] && knn.queries[pointIdx].neighbors) {
        return knn.queries[pointIdx].neighbors.map((n, r) => ({
          index: (typeof n === 'number') ? n : n.index,
          dist: (typeof n === 'number') ? 0.0 : (n.dist || 0.0),
          rank: r + 1
        }));
      }
      if (Array.isArray(knn) && knn[pointIdx] && knn[pointIdx].neighbors) {
        return knn[pointIdx].neighbors.map((n, r) => ({
          index: (typeof n === 'number') ? n : n.index,
          dist: (typeof n === 'number') ? 0.0 : (n.dist || 0.0),
          rank: r + 1
        }));
      }
      return null;
    }

    /**
     * Renders the 4-Panel Synchronized Reconstruction View:
     * Panel 0 (Top-Left): Slot A (Training Input)
     * Panel 1 (Top-Right): Slot B (Training Output)
     * Panel 2 (Bottom-Left): Slot C (Query Input)
     * Panel 3 (Bottom-Right): Slot D (Reconstructed Output)
     */
    /**
     * Compute set of active highlight neighbor indices for a slot in ABCD view
     */
    function getActiveNeighborSetForSlot(
      slotId, activeTrainingIdx, activeTrainingSlot, activeQueryIdx, k
    ) {
      const isKnnActive = (typeof showReconKnn === 'undefined' || showReconKnn);
      if (!isKnnActive) return null;

      if (activeTrainingIdx >= 0) {
        if (slotId === 'A' || slotId === 'B') {
          const activeNeighborSet = new Set();
          activeNeighborSet.add(activeTrainingIdx);
          const nbs = getSlotKnnNeighbors(activeTrainingSlot, activeTrainingIdx);
          if (nbs) {
            for (let r = 0; r < nbs.length; r++) {
              activeNeighborSet.add(nbs[r].index);
            }
          }
          return activeNeighborSet;
        } else if (slotId === 'C' || slotId === 'D') {
          const revQueries = getOrComputeReverseKnnNeighbors(activeTrainingIdx, k);
          if (revQueries && revQueries.length > 0) {
            const activeNeighborSet = new Set();
            for (let r = 0; r < revQueries.length; r++) {
              activeNeighborSet.add(revQueries[r].queryIdx);
            }
            return activeNeighborSet;
          }
        }
      } else if (activeQueryIdx >= 0) {
        if (slotId === 'C' || slotId === 'D') {
          const activeNeighborSet = new Set();
          activeNeighborSet.add(activeQueryIdx);
          return activeNeighborSet;
        } else if (slotId === 'A' || slotId === 'B') {
          const nbs = getOrComputeKnnNeighbors(activeQueryIdx, k);
          if (nbs) {
            const activeNeighborSet = new Set();
            for (let r = 0; r < nbs.length; r++) {
              activeNeighborSet.add(nbs[r].id);
            }
            return activeNeighborSet;
          }
        }
      }
      return null;
    }

    /**
     * Render grid and axes for an ABCD panel
     */
    function drawPanelGridAxes(ctx, rect, is3D, scale, mapToScreen, projectPt) {
      if (!showGridAxes) return;
      if (!is3D) {
        const center = mapToScreen({ u: 0, v: 0 });
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(rect.x, center.py); ctx.lineTo(rect.x + rect.w, center.py);
        ctx.moveTo(center.px, rect.y); ctx.lineTo(center.px, rect.y + rect.h);
        ctx.stroke();

        ctx.strokeStyle = '#172554';
        ctx.setLineDash([2, 4]);
        [0.5, 0.85].forEach(rad => {
          ctx.beginPath();
          ctx.arc(center.px, center.py, rad * scale, 0, Math.PI * 2);
          ctx.stroke();
        });
        ctx.setLineDash([]);
      } else {
        const b = 0.85;
        const boxCorners = [
          {x:-b, y:-b, z:-b}, {x: b, y:-b, z:-b}, {x: b, y: b, z:-b}, {x:-b, y: b, z:-b},
          {x:-b, y:-b, z: b}, {x: b, y:-b, z: b}, {x: b, y: b, z: b}, {x:-b, y: b, z: b}
        ];
        const boxPx = boxCorners.map(pt => mapToScreen(projectPt(pt)));

        ctx.strokeStyle = 'rgba(30, 41, 59, 0.6)';
        ctx.lineWidth = 1;
        [-0.85, -0.425, 0, 0.425, 0.85].forEach(val => {
          const p1 = mapToScreen(projectPt({ x: val, y: -b, z: -b }));
          const p2 = mapToScreen(projectPt({ x: val, y:  b, z: -b }));
          ctx.beginPath(); ctx.moveTo(p1.px, p1.py); ctx.lineTo(p2.px, p2.py); ctx.stroke();
          const p3 = mapToScreen(projectPt({ x: -b, y: val, z: -b }));
          const p4 = mapToScreen(projectPt({ x:  b, y: val, z: -b }));
          ctx.beginPath(); ctx.moveTo(p3.px, p3.py); ctx.lineTo(p4.px, p4.py); ctx.stroke();
        });

        const edges = [
          [0,1],[1,2],[2,3],[3,0],
          [4,5],[5,6],[6,7],[7,4],
          [0,4],[1,5],[2,6],[3,7]
        ];
        ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
        ctx.setLineDash([2, 3]);
        edges.forEach(([i, j]) => {
          ctx.beginPath();
          ctx.moveTo(boxPx[i].px, boxPx[i].py);
          ctx.lineTo(boxPx[j].px, boxPx[j].py);
          ctx.stroke();
        });
        ctx.setLineDash([]);
      }
    }

    /**
     * Render point cloud for a specific slot in ABCD view
     */
    function drawSlotPoints(
      ctx, slot, slotId, defaultColor, rect, projectPt, mapToScreen,
      activeNeighborSet, isFocusMode, primaryFocusedIdx
    ) {
      const pts = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slot)
        : (slot && slot.benchmarkDataset && slot.benchmarkDataset.length > 0
            ? slot.benchmarkDataset : (slot && slot.pastSamples ? slot.pastSamples : []));
      if (!pts || pts.length === 0) return;

      const numPts = pts.length;
      const ptRad = Math.max(1.0, samplePointSize * 0.9);

      const qualArr = (slotId === 'C' && slot.reconKthDist) ? slot.reconKthDist
                    : (slotId === 'D' && slot.reconVariance) ? slot.reconVariance : null;
      const qualMin = (slotId === 'C') ? slot.reconKthDistMin
                    : (slotId === 'D') ? slot.reconVarianceMin : 0;
      const qualMax = (slotId === 'C') ? slot.reconKthDistMax
                    : (slotId === 'D') ? slot.reconVarianceMax : 1;
      const qMask = (slotId === 'C' || slotId === 'D')
        ? (slot.reconQualityMask || (slotId === 'C' ? datasetSlots['C']?.reconQualityMask
                                                    : datasetSlots['D']?.reconQualityMask)
                                 || reconQualityMask)
        : null;
      const qIndices = (slotId === 'C' || slotId === 'D')
        ? (slot.reconQualityIndices || (slotId === 'C' ? datasetSlots['C']?.reconQualityIndices
                                                       : datasetSlots['D']?.reconQualityIndices)
                                    || reconQualityIndices)
        : null;

      const isQualityFiltering = (reconQualityThreshold < 1.0 && qMask);
      const maxDraw = (typeof maxDrawPoints !== 'undefined') ? maxDrawPoints : 10000;

      let drawPoolLength = numPts;
      let getPointIndex = (idx) => idx;

      if (isQualityFiltering && qIndices && qIndices.length > 0)
      {
        drawPoolLength = qIndices.length;
        getPointIndex = (idx) => qIndices[idx];
      }

      const drawCount = Math.min(drawPoolLength, maxDraw);
      const step = drawPoolLength > drawCount ? (drawPoolLength / drawCount) : 1;

      // Pass 1: Render background / non-neighbor points
      for (let k = 0; k < drawCount; k++) {
        const i = getPointIndex(Math.floor(k * step));
        if (i < 0 || i >= numPts) continue;

        if (isQualityFiltering && !qIndices && qMask && i < qMask.length && !qMask[i]) {
          continue;
        }

        const isNeighbor = activeNeighborSet && activeNeighborSet.has(i);
        if (isFocusMode && isNeighbor) {
          continue;
        }

        const p = pts[i];
        if (!p) continue;
        const pr = projectPt(p);
        const pos = mapToScreen(pr);

        if (pos.px < rect.x - 5 || pos.px > rect.x + rect.w + 5 ||
            pos.py < rect.y - 5 || pos.py > rect.y + rect.h + 5) {
          continue;
        }

        let pCol = defaultColor;
        let pAlpha = 0.55;

        if (reconQualityColoringEnabled && qualArr && i < qualArr.length) {
          pCol = getColorFromRamp(qualArr[i], qualMin, qualMax, QUALITY_STOPS);
          pAlpha = isFocusMode ? 0.50 : 0.85;
        } else if (isFocusMode) {
          pCol = (slotId === 'C' || slotId === 'D') ? defaultColor : '#64748b';
          pAlpha = (slotId === 'C' || slotId === 'D') ? 0.45 : 0.18;
        }

        ctx.fillStyle = pCol;
        ctx.globalAlpha = pAlpha;
        ctx.beginPath();
        ctx.arc(pos.px, pos.py, ptRad, 0, Math.PI * 2);
        ctx.fill();
      }

      // Pass 2: Render active k-NN neighbor points bright on top
      if (isFocusMode && activeNeighborSet && activeNeighborSet.size > 0) {
        for (const i of activeNeighborSet) {
          if (i < 0 || i >= numPts) continue;
          if (isQualityFiltering && qMask && i < qMask.length && !qMask[i]) {
            continue;
          }

          const p = pts[i];
          if (!p) continue;
          const pr = projectPt(p);
          const pos = mapToScreen(pr);

          if (pos.px < rect.x - 5 || pos.px > rect.x + rect.w + 5 ||
              pos.py < rect.y - 5 || pos.py > rect.y + rect.h + 5) {
            continue;
          }

          const isAnchor = (i === primaryFocusedIdx);
          let pCol = defaultColor;
          if (reconQualityColoringEnabled && qualArr && i < qualArr.length) {
            pCol = getColorFromRamp(qualArr[i], qualMin, qualMax, QUALITY_STOPS);
          } else {
            if (slotId === 'A') pCol = isAnchor ? '#38bdf8' : '#7dd3fc';
            else if (slotId === 'B') pCol = isAnchor ? '#4ade80' : '#86efac';
            else if (slotId === 'C') pCol = isAnchor ? '#fbbf24' : '#fde68a';
            else if (slotId === 'D') pCol = isAnchor ? '#c084fc' : '#e9d5ff';
          }

          const rad = isAnchor ? ptRad * 1.8 : ptRad * 1.35;
          ctx.fillStyle = pCol;
          ctx.globalAlpha = 1.0;
          ctx.beginPath();
          ctx.arc(pos.px, pos.py, rad, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = pCol;
          ctx.lineWidth = 1.0;
          ctx.beginPath();
          ctx.arc(pos.px, pos.py, rad + 1.5, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1.0;
    }

    /**
     * Renders the 4-Panel Synchronized Reconstruction View:
     * - 4-Panel mode: A (Top-Left), B (Top-Right), C (Bottom-Left), D (Bottom-Right)
     * - Overlay mode: A+C in Left View (Input), B+D in Right View (Output)
     */
    function drawRecon4PanelView(ctx, W, H) {
      const halfW = W / 2;
      const halfH = H / 2;
      const isOverlay = (typeof reconOverlayMode !== 'undefined' && reconOverlayMode);
      const quadRects = isOverlay ? [
        { x: 0, y: 0, w: halfW, h: H, slotId: 'A', name: 'Training Input [A]',
          color: '#38bdf8' },
        { x: halfW, y: 0, w: halfW, h: H, slotId: 'B', name: 'Training Output [B]',
          color: '#4ade80' },
        { x: 0, y: 0, w: halfW, h: H, slotId: 'C', name: 'Query Input [C]',
          color: '#fbbf24' },
        { x: halfW, y: 0, w: halfW, h: H, slotId: 'D', name: 'Reconstructed Output [D]',
          color: '#c084fc' }
      ] : [
        { x: 0, y: 0, w: halfW, h: halfH, slotId: 'A', name: 'Training Input [A]',
          color: '#38bdf8' },
        { x: halfW, y: 0, w: halfW, h: halfH, slotId: 'B', name: 'Training Output [B]',
          color: '#4ade80' },
        { x: 0, y: halfH, w: halfW, h: halfH, slotId: 'C', name: 'Query Input [C]',
          color: '#fbbf24' },
        { x: halfW, y: halfH, w: halfW, h: halfH, slotId: 'D', name: 'Reconstructed Output [D]',
          color: '#c084fc' }
      ];

      const activeTrainingIdx = (typeof reconLockedTrainingIdx !== 'undefined' &&
        reconLockedTrainingIdx >= 0) ? reconLockedTrainingIdx
        : (typeof reconHoveredTrainingIdx !== 'undefined' ? reconHoveredTrainingIdx : -1);

      const activeTrainingSlot = (typeof reconLockedTrainingIdx !== 'undefined' &&
        reconLockedTrainingIdx >= 0)
        ? (typeof reconLockedTrainingSlot !== 'undefined' ? reconLockedTrainingSlot : 'A')
        : (typeof reconHoveredTrainingSlot !== 'undefined' ? reconHoveredTrainingSlot : 'A');

      const activeQueryIdx = (typeof reconLockedQueryIdx !== 'undefined' &&
        reconLockedQueryIdx >= 0) ? reconLockedQueryIdx
        : (typeof reconHoveredQueryIdx !== 'undefined' ? reconHoveredQueryIdx : -1);

      const slotA = datasetSlots['A'];
      const slotB = datasetSlots['B'];
      const slotC = datasetSlots['C'];
      const slotD = datasetSlots['D'];
      const k = (slotD && slotD.reconstructionInfo)
        ? slotD.reconstructionInfo.k : (typeof knnK !== 'undefined' ? knnK : 10);

      function projectPtCustom(p, is3D, activeCam, az, el) {
        const pt = (typeof getPlotCoords === 'function') ? getPlotCoords(p) : p;
        if (is3D) {
          let tx = 0, ty = 0, tz = 0;
          if (activeCam && activeCam.isLocked) {
            tx = activeCam.targetX || 0;
            ty = activeCam.targetY || 0;
            tz = activeCam.targetZ || 0;
          }
          return project3DVector(pt.x - tx, pt.y - ty, (pt.z || 0.0) - tz, az, el);
        }
        return { u: pt.x, v: pt.y, depth: pt.z || 0.0 };
      }

      function mapToScreenCustom(pr, rect, activePanX, activePanY, scale) {
        const cx = rect.x + rect.w / 2;
        const cy = rect.y + rect.h / 2;
        return {
          px: cx + (pr.u - activePanX) * scale,
          py: cy - (pr.v - activePanY) * scale
        };
      }

      if (isOverlay) {
        // =====================================================================
        // OVERLAY MODE: Left Panel (A + C) & Right Panel (B + D)
        // =====================================================================

        // --- 1. Left Panel (Input Space: A + C) ---
        {
          const rect = { x: 0, y: 0, w: halfW, h: H };
          const activeView = quadViews[0] || { panX: 0, panY: 0, zoom: 1.0 };
          const activePanX = activeView.panX || 0;
          const activePanY = activeView.panY || 0;
          const activeZoom = activeView.zoom || 1.0;
          const activeCam = (typeof reconInputCamera !== 'undefined')
            ? reconInputCamera : orbitCamera;
          const az = activeCam.azimuth;
          const el = activeCam.elevation;
          const scale = (Math.min(rect.w, rect.h) / 2.35) * activeZoom;
          const is3D = (slotA && slotA.currentDim >= 3) || (slotC && slotC.currentDim >= 3);

          const projectPt = (p) => projectPtCustom(p, is3D, activeCam, az, el);
          const mapToScreen = (pr) => mapToScreenCustom(pr, rect, activePanX, activePanY, scale);

          ctx.save();
          ctx.beginPath();
          ctx.rect(rect.x, rect.y, rect.w, rect.h);
          ctx.clip();

          // A. Grid & Axes
          drawPanelGridAxes(ctx, rect, is3D, scale, mapToScreen, projectPt);

          // B. Point Cloud A (Training Input)
          const isFocus = (activeTrainingIdx >= 0 || activeQueryIdx >= 0);
          const nbSetA = getActiveNeighborSetForSlot(
            'A', activeTrainingIdx, activeTrainingSlot, activeQueryIdx, k
          );
          drawSlotPoints(
            ctx, slotA, 'A', '#38bdf8', rect, projectPt, mapToScreen,
            nbSetA, isFocus, activeTrainingIdx
          );

          // C. Point Cloud C (Query Input)
          const nbSetC = getActiveNeighborSetForSlot(
            'C', activeTrainingIdx, activeTrainingSlot, activeQueryIdx, k
          );
          drawSlotPoints(
            ctx, slotC, 'C', '#fbbf24', rect, projectPt, mapToScreen,
            nbSetC, isFocus, activeQueryIdx
          );

          // D. Left Panel HUD Header
          ctx.save();
          const ptsA = (typeof getSlotPoints === 'function')
            ? getSlotPoints(slotA)
            : (slotA && slotA.benchmarkDataset && slotA.benchmarkDataset.length > 0
                ? slotA.benchmarkDataset : (slotA && slotA.pastSamples ? slotA.pastSamples : []));
          const ptsC = (typeof getSlotPoints === 'function')
            ? getSlotPoints(slotC)
            : (slotC && slotC.benchmarkDataset && slotC.benchmarkDataset.length > 0
                ? slotC.benchmarkDataset : (slotC && slotC.pastSamples ? slotC.pastSamples : []));
          const effDim = Math.max(
            slotA ? (slotA.currentDim || 0) : 0,
            slotC ? (slotC.currentDim || 0) : 0
          );
          const dimStr = effDim > 0 ? `${effDim}D` : (is3D ? '3D' : '2D');

          // Slot A Pill
          ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.roundRect(rect.x + 8, rect.y + 8, 20, 16, 3);
          ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#38bdf8';
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText('A', rect.x + 18, rect.y + 16);

          // Slot C Pill
          ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.roundRect(rect.x + 32, rect.y + 8, 20, 16, 3);
          ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#fbbf24';
          ctx.font = 'bold 10px monospace';
          ctx.fillText('C', rect.x + 42, rect.y + 16);

          // Header Text
          ctx.textAlign = 'left';
          ctx.font = 'bold 10px -apple-system, BlinkMacSystemFont, sans-serif';
          ctx.fillStyle = '#f8fafc';
          ctx.fillText('Input Domain: Training [A] + Query [C]', rect.x + 58, rect.y + 15);

          ctx.font = '9px monospace';
          ctx.fillStyle = 'var(--text-muted, #94a3b8)';
          const countAStr = `${ptsA.length.toLocaleString()} [A]`;
          let countCStr = `${ptsC.length.toLocaleString()} [C] (${dimStr})`;
          if (reconQualityThreshold < 1.0 &&
              (slotC?.reconQualityIndices || slotC?.reconQualityMask)) {
            const visC = slotC.reconQualityIndices ? slotC.reconQualityIndices.length : 0;
            if (visC > 0) {
              countCStr =
                `${visC.toLocaleString()}/${ptsC.length.toLocaleString()} [C] (${dimStr})`;
            }
          }
          ctx.fillText(`${countAStr} • ${countCStr}`, rect.x + 58, rect.y + 26);

          // Top-right status / hint
          ctx.textAlign = 'right';
          if (activeTrainingIdx >= 0) {
            const isLocked = (typeof reconLockedTrainingIdx !== 'undefined' &&
              reconLockedTrainingIdx >= 0);
            const knnA = getSlotKnnNeighbors('A', activeTrainingIdx);
            ctx.fillStyle = isLocked ? '#38bdf8' : '#94a3b8';
            let hText = isLocked ? `🔒 PINNED #${activeTrainingIdx}`
              : `Sample #${activeTrainingIdx}`;
            if (activeTrainingSlot === 'A' && knnA && knnA.length > 0) {
              hText = `${isLocked ? '🔒 ' : ''}k-NN #${activeTrainingIdx} (${knnA.length} NNs in A)`;
            } else if (activeTrainingSlot === 'B') {
              hText = `Mapped #${activeTrainingIdx} from [B]`;
            }
            ctx.fillText(hText, rect.x + rect.w - 10, rect.y + 15);
          } else if (activeQueryIdx >= 0) {
            const isLocked = (typeof reconLockedQueryIdx !== 'undefined' &&
              reconLockedQueryIdx >= 0);
            ctx.fillStyle = isLocked ? '#fbbf24' : '#94a3b8';
            ctx.font = isLocked ? 'bold 9px monospace' : '9px -apple-system, sans-serif';
            ctx.fillText(
              `${isLocked ? '🔒 ' : ''}Query #${activeQueryIdx}`,
              rect.x + rect.w - 10, rect.y + 15
            );
          } else {
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px -apple-system, sans-serif';
            ctx.fillText('⧉ Overlay A+C • Input Space', rect.x + rect.w - 10, rect.y + 15);
          }

          ctx.restore();
          ctx.restore(); // restore clip
        }

        // --- 2. Right Panel (Output Space: B + D) ---
        {
          const rect = { x: halfW, y: 0, w: halfW, h: H };
          const activeView = quadViews[1] || { panX: 0, panY: 0, zoom: 1.0 };
          const activePanX = activeView.panX || 0;
          const activePanY = activeView.panY || 0;
          const activeZoom = activeView.zoom || 1.0;
          const activeCam = (typeof reconOutputCamera !== 'undefined')
            ? reconOutputCamera : orbitCamera;
          const az = activeCam.azimuth;
          const el = activeCam.elevation;
          const scale = (Math.min(rect.w, rect.h) / 2.35) * activeZoom;
          const is3D = (slotB && slotB.currentDim >= 3) || (slotD && slotD.currentDim >= 3);

          const projectPt = (p) => projectPtCustom(p, is3D, activeCam, az, el);
          const mapToScreen = (pr) => mapToScreenCustom(pr, rect, activePanX, activePanY, scale);

          ctx.save();
          ctx.beginPath();
          ctx.rect(rect.x, rect.y, rect.w, rect.h);
          ctx.clip();

          // A. Grid & Axes
          drawPanelGridAxes(ctx, rect, is3D, scale, mapToScreen, projectPt);

          // B. Point Cloud B (Training Output)
          const isFocus = (activeTrainingIdx >= 0 || activeQueryIdx >= 0);
          const nbSetB = getActiveNeighborSetForSlot(
            'B', activeTrainingIdx, activeTrainingSlot, activeQueryIdx, k
          );
          drawSlotPoints(
            ctx, slotB, 'B', '#4ade80', rect, projectPt, mapToScreen,
            nbSetB, isFocus, activeTrainingIdx
          );

          // C. Point Cloud D (Reconstructed Output)
          const nbSetD = getActiveNeighborSetForSlot(
            'D', activeTrainingIdx, activeTrainingSlot, activeQueryIdx, k
          );
          drawSlotPoints(
            ctx, slotD, 'D', '#c084fc', rect, projectPt, mapToScreen,
            nbSetD, isFocus, activeQueryIdx
          );

          // D. Right Panel HUD Header
          ctx.save();
          const ptsB = (typeof getSlotPoints === 'function')
            ? getSlotPoints(slotB)
            : (slotB && slotB.benchmarkDataset && slotB.benchmarkDataset.length > 0
                ? slotB.benchmarkDataset : (slotB && slotB.pastSamples ? slotB.pastSamples : []));
          const ptsD = (typeof getSlotPoints === 'function')
            ? getSlotPoints(slotD)
            : (slotD && slotD.benchmarkDataset && slotD.benchmarkDataset.length > 0
                ? slotD.benchmarkDataset : (slotD && slotD.pastSamples ? slotD.pastSamples : []));
          const effDim = Math.max(
            slotB ? (slotB.currentDim || 0) : 0,
            slotD ? (slotD.currentDim || 0) : 0
          );
          const dimStr = effDim > 0 ? `${effDim}D` : (is3D ? '3D' : '2D');

          // Slot B Pill
          ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
          ctx.strokeStyle = '#4ade80';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.roundRect(rect.x + 8, rect.y + 8, 20, 16, 3);
          ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#4ade80';
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText('B', rect.x + 18, rect.y + 16);

          // Slot D Pill
          ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
          ctx.strokeStyle = '#c084fc';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.roundRect(rect.x + 32, rect.y + 8, 20, 16, 3);
          ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#c084fc';
          ctx.font = 'bold 10px monospace';
          ctx.fillText('D', rect.x + 42, rect.y + 16);

          // Header Text
          ctx.textAlign = 'left';
          ctx.font = 'bold 10px -apple-system, BlinkMacSystemFont, sans-serif';
          ctx.fillStyle = '#f8fafc';
          ctx.fillText('Output Domain: Training [B] + Recon [D]', rect.x + 58, rect.y + 15);

          ctx.font = '9px monospace';
          ctx.fillStyle = 'var(--text-muted, #94a3b8)';
          const countBStr = `${ptsB.length.toLocaleString()} [B]`;
          let countDStr = `${ptsD.length.toLocaleString()} [D] (${dimStr})`;
          if (reconQualityThreshold < 1.0 &&
              (slotD?.reconQualityIndices || slotD?.reconQualityMask)) {
            const visD = slotD.reconQualityIndices ? slotD.reconQualityIndices.length : 0;
            if (visD > 0) {
              countDStr =
                `${visD.toLocaleString()}/${ptsD.length.toLocaleString()} [D] (${dimStr})`;
            }
          }
          ctx.fillText(`${countBStr} • ${countDStr}`, rect.x + 58, rect.y + 26);

          // Top-right status / hint
          ctx.textAlign = 'right';
          const isRecon = slotD && slotD.reconstructionSourceNeighbors &&
            slotD.reconstructionSourceNeighbors.length > 0;
          if (activeTrainingIdx >= 0) {
            const revQueries = getOrComputeReverseKnnNeighbors(activeTrainingIdx, k);
            ctx.fillStyle = '#c084fc';
            ctx.font = '9px monospace';
            const inflText = isRecon
              ? `Influences ${revQueries.length} outputs`
              : 'Recon [D] to inspect';
            ctx.fillText(inflText, rect.x + rect.w - 10, rect.y + 15);
          } else if (activeQueryIdx >= 0) {
            const info = slotD ? slotD.reconstructionInfo : null;
            ctx.fillStyle = info ? '#c084fc' : '#94a3b8';
            ctx.font = '9px monospace';
            const dText = info
              ? `k=${info.k} • ${info.weightMode}`
              : `Output #${activeQueryIdx}`;
            ctx.fillText(dText, rect.x + rect.w - 10, rect.y + 15);
          } else {
            ctx.fillStyle = '#c084fc';
            ctx.font = '9px -apple-system, sans-serif';
            ctx.fillText('⧉ Overlay B+D • Output Space', rect.x + rect.w - 10, rect.y + 15);
          }

          ctx.restore();
          ctx.restore(); // restore clip
        }
      } else {
        // =====================================================================
        // 4-PANEL QUADRANT MODE (A, B, C, D separate)
        // =====================================================================
        for (let q = 0; q < 4; q++) {
          const qConfig = quadRects[q];
          const rect = { x: qConfig.x, y: qConfig.y, w: qConfig.w, h: qConfig.h };
          const slot = datasetSlots[qConfig.slotId];
          const pts = (typeof getSlotPoints === 'function')
            ? getSlotPoints(slot)
            : (slot && slot.benchmarkDataset && slot.benchmarkDataset.length > 0
                ? slot.benchmarkDataset : (slot && slot.pastSamples ? slot.pastSamples : []));
          const is3D = (slot && slot.currentDim >= 3);

          const isInputSpace = (q === 0 || q === 2);
          const activeView = isInputSpace ? quadViews[0] : quadViews[1];
          const activePanX = activeView ? (activeView.panX || 0) : 0;
          const activePanY = activeView ? (activeView.panY || 0) : 0;
          const activeZoom = activeView ? (activeView.zoom || 1.0) : 1.0;
          const activeCam = isInputSpace
            ? (typeof reconInputCamera !== 'undefined' ? reconInputCamera : orbitCamera)
            : (typeof reconOutputCamera !== 'undefined' ? reconOutputCamera : orbitCamera);

          const az = activeCam.azimuth;
          const el = activeCam.elevation;
          const scale = (Math.min(rect.w, rect.h) / 2.35) * activeZoom;

          const projectPt = (p) => projectPtCustom(p, is3D, activeCam, az, el);
          const mapToScreen = (pr) => mapToScreenCustom(pr, rect, activePanX, activePanY, scale);

          ctx.save();
          ctx.beginPath();
          ctx.rect(rect.x, rect.y, rect.w, rect.h);
          ctx.clip();

          // A. Grid & Axes
          drawPanelGridAxes(ctx, rect, is3D, scale, mapToScreen, projectPt);

          // B. Point Cloud Drawing
          const isFocus = (activeTrainingIdx >= 0 || activeQueryIdx >= 0);
          const nbSet = getActiveNeighborSetForSlot(
            qConfig.slotId, activeTrainingIdx, activeTrainingSlot, activeQueryIdx, k
          );
          const primIdx = (qConfig.slotId === 'A' || qConfig.slotId === 'B')
            ? activeTrainingIdx : activeQueryIdx;
          drawSlotPoints(
            ctx, slot, qConfig.slotId, qConfig.color, rect, projectPt, mapToScreen,
            nbSet, isFocus, primIdx
          );

          // C. Quadrant Header HUD
          ctx.save();
          const ptCount = pts ? pts.length : 0;
          const dimVal = (slot && slot.currentDim) ? slot.currentDim : (is3D ? 3 : 2);
          const dimStr = `${dimVal}D`;
          const benchName = (slot && slot.stagedDatasetInfo && slot.stagedDatasetInfo.name)
            ? slot.stagedDatasetInfo.name
            : (slot ? (slot.benchmarkKey || 'None') : 'None');

          const qHudMask = (q === 2 || q === 3)
            ? (slot.reconQualityMask || (q === 2 ? datasetSlots['C'].reconQualityMask
                                                 : datasetSlots['D'].reconQualityMask)
                                     || reconQualityMask)
            : null;
          let countText = `${ptCount.toLocaleString()} pts (${dimStr})`;
          if (ptCount > 0 && (q === 2 || q === 3) && reconQualityThreshold < 1.0 && qHudMask) {
            let visibleCount = (slot && slot.reconQualityIndices)
              ? slot.reconQualityIndices.length : 0;
            if (visibleCount === 0 && qHudMask) {
              for (let i = 0; i < qHudMask.length; i++) {
                if (qHudMask[i]) visibleCount++;
              }
            }
            const visStr = visibleCount.toLocaleString();
            const totStr = ptCount.toLocaleString();
            countText = `${visStr}/${totStr} pts (${dimStr})`;
          }

          // Top-left slot identifier pill
          const slotPillW = 20;
          const slotPillH = 16;
          ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
          ctx.strokeStyle = qConfig.color;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.roundRect(rect.x + 8, rect.y + 8, slotPillW, slotPillH, 3);
          ctx.fill();
          ctx.stroke();

          ctx.fillStyle = qConfig.color;
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(qConfig.slotId, rect.x + 8 + slotPillW / 2, rect.y + 8 + slotPillH / 2);

          // Title and stats
          ctx.textAlign = 'left';
          ctx.font = 'bold 10px -apple-system, BlinkMacSystemFont, sans-serif';
          ctx.fillStyle = '#f8fafc';
          ctx.fillText(qConfig.name, rect.x + 34, rect.y + 15);

          ctx.font = '9px monospace';
          ctx.fillStyle = 'var(--text-muted, #94a3b8)';
          ctx.fillText(`${benchName} • ${countText}`, rect.x + 34, rect.y + 26);

          // Top-right status / hint
          ctx.textAlign = 'right';
          if (q === 0) {
            if (activeTrainingIdx >= 0) {
              const isLocked = (typeof reconLockedTrainingIdx !== 'undefined' &&
                reconLockedTrainingIdx >= 0);
              const knnA = getSlotKnnNeighbors('A', activeTrainingIdx);
              const knnB = (activeTrainingSlot === 'B')
                ? getSlotKnnNeighbors('B', activeTrainingIdx) : null;
              ctx.fillStyle = isLocked ? '#38bdf8' : '#94a3b8';
              let hText = isLocked ? `🔒 PINNED #${activeTrainingIdx}`
                : `Sample #${activeTrainingIdx}`;
              if (activeTrainingSlot === 'A' && knnA && knnA.length > 0) {
                const lockPrefix = isLocked ? '🔒 ' : '';
                hText = `${lockPrefix}k-NN #${activeTrainingIdx} (${knnA.length} NNs in A)`;
              } else if (activeTrainingSlot === 'B' && knnB && knnB.length > 0) {
                hText = `Mapped #${activeTrainingIdx} & ${knnB.length} NNs from [B]`;
              }
              ctx.fillText(hText, rect.x + rect.w - 10, rect.y + 15);
            } else {
              ctx.fillStyle = '#38bdf8';
              ctx.font = '9px -apple-system, sans-serif';
              ctx.fillText('k-NN Neighbors', rect.x + rect.w - 10, rect.y + 15);
            }
          } else if (q === 1) {
            if (activeTrainingIdx >= 0) {
              const isLocked = (typeof reconLockedTrainingIdx !== 'undefined' &&
                reconLockedTrainingIdx >= 0);
              const knnB = getSlotKnnNeighbors('B', activeTrainingIdx);
              const knnA = (activeTrainingSlot === 'A')
                ? getSlotKnnNeighbors('A', activeTrainingIdx) : null;
              ctx.fillStyle = isLocked ? '#4ade80' : '#94a3b8';
              let hText = isLocked ? `🔒 PINNED #${activeTrainingIdx}`
                : `Counterpart #${activeTrainingIdx}`;
              if (activeTrainingSlot === 'B' && knnB && knnB.length > 0) {
                const lockPrefix = isLocked ? '🔒 ' : '';
                hText = `${lockPrefix}k-NN #${activeTrainingIdx} (${knnB.length} NNs in B)`;
              } else if (activeTrainingSlot === 'A' && knnA && knnA.length > 0) {
                hText = `Mapped #${activeTrainingIdx} & ${knnA.length} NNs from [A]`;
              }
              ctx.fillText(hText, rect.x + rect.w - 10, rect.y + 15);
            } else {
              ctx.fillStyle = '#4ade80';
              ctx.font = '9px -apple-system, sans-serif';
              ctx.fillText('Weighted Contribution', rect.x + rect.w - 10, rect.y + 15);
            }
          } else if (q === 2) {
            const isRecon = datasetSlots.D && datasetSlots.D.reconstructionSourceNeighbors &&
              datasetSlots.D.reconstructionSourceNeighbors.length > 0;
            if (activeTrainingIdx >= 0) {
              const revQueries = getOrComputeReverseKnnNeighbors(activeTrainingIdx, k);
              ctx.fillStyle = '#fbbf24';
              ctx.font = 'bold 9px monospace';
              const revText = isRecon
                ? `${revQueries.length} queries use A[${activeTrainingIdx}]`
                : 'Recon [D] to inspect influences';
              ctx.fillText(revText, rect.x + rect.w - 10, rect.y + 15);
            } else {
              const isLocked = (activeQueryIdx >= 0 &&
                typeof reconLockedQueryIdx !== 'undefined' &&
                reconLockedQueryIdx >= 0);
              ctx.fillStyle = isLocked ? '#fbbf24' : '#94a3b8';
              ctx.font = isLocked ? 'bold 9px monospace' : '9px -apple-system, sans-serif';
              const hintText = isLocked
                ? `🔒 PINNED #${activeQueryIdx}` : '🔍 Hover query to inspect';
              ctx.fillText(hintText, rect.x + rect.w - 10, rect.y + 15);
            }
          } else if (q === 3) {
            const isRecon = datasetSlots.D && datasetSlots.D.reconstructionSourceNeighbors &&
              datasetSlots.D.reconstructionSourceNeighbors.length > 0;
            if (activeTrainingIdx >= 0) {
              const revQueries = getOrComputeReverseKnnNeighbors(activeTrainingIdx, k);
              ctx.fillStyle = '#c084fc';
              ctx.font = '9px monospace';
              const inflText = isRecon
                ? `Influences ${revQueries.length} outputs`
                : 'Not reconstructed';
              ctx.fillText(inflText, rect.x + rect.w - 10, rect.y + 15);
            } else {
              const info = datasetSlots.D ? datasetSlots.D.reconstructionInfo : null;
              ctx.fillStyle = info ? '#c084fc' : '#94a3b8';
              ctx.font = '9px monospace';
              const dText = info
                ? `k=${info.k} • ${info.weightMode}`
                : (isRecon ? 'Slot D' : 'Slot D (not reconstructed)');
              ctx.fillText(dText, rect.x + rect.w - 10, rect.y + 15);
            }
          }

          ctx.restore();
          ctx.restore(); // restore clip
        }
      }

      // 2. Synchronized 4-Panel Highlight Overlays
      if (activeTrainingIdx >= 0 || activeQueryIdx >= 0) {
        renderRecon4PanelHighlights(
          ctx, W, H, quadRects, activeQueryIdx, activeTrainingIdx, activeTrainingSlot
        );
      }

      // 3. Viewport Divider Lines
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(halfW, 0); ctx.lineTo(halfW, H);
      if (!isOverlay) {
        ctx.moveTo(0, halfH); ctx.lineTo(W, halfH);
      }
      ctx.stroke();
    }

    /**
     * Cross-quadrant synchronized highlighting for k-NN reconstruction
     */
    function renderRecon4PanelHighlights(
      ctx, W, H, quadRects, activeQueryIdx, activeTrainingIdx = -1, activeTrainingSlot = 'A'
    ) {
      if (typeof showReconKnn !== 'undefined' && !showReconKnn) return;
      const isOverlay = (typeof reconOverlayMode !== 'undefined' && reconOverlayMode);

      const slotA = datasetSlots['A'];
      const slotB = datasetSlots['B'];
      const slotC = datasetSlots['C'];
      const slotD = datasetSlots['D'];

      const ptsA = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotA)
        : (slotA && slotA.benchmarkDataset && slotA.benchmarkDataset.length > 0
            ? slotA.benchmarkDataset : (slotA ? slotA.pastSamples : null));
      const ptsB = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotB)
        : (slotB && slotB.benchmarkDataset && slotB.benchmarkDataset.length > 0
            ? slotB.benchmarkDataset : (slotB ? slotB.pastSamples : null));
      const ptsC = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotC)
        : (slotC && slotC.benchmarkDataset && slotC.benchmarkDataset.length > 0
            ? slotC.benchmarkDataset : (slotC ? slotC.pastSamples : null));
      const ptsD = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotD)
        : (slotD && slotD.benchmarkDataset && slotD.benchmarkDataset.length > 0
            ? slotD.benchmarkDataset : (slotD ? slotD.pastSamples : null));

      const dimA = slotA ? (slotA.currentDim || 2) : 2;
      const dimB = slotB ? (slotB.currentDim || 2) : 2;
      const dimC = slotC ? (slotC.currentDim || 2) : 2;
      const dimD = slotD ? (slotD.currentDim || 2) : 2;

      const k = (slotD && slotD.reconstructionInfo)
        ? slotD.reconstructionInfo.k : (typeof knnK !== 'undefined' ? knnK : 10);

      function mapPt(p, rect, is3D, isInputSpace) {
        const activeView = isInputSpace ? quadViews[0] : quadViews[1];
        const activePanX = activeView ? (activeView.panX || 0) : 0;
        const activePanY = activeView ? (activeView.panY || 0) : 0;
        const activeZoom = activeView ? (activeView.zoom || 1.0) : 1.0;
        const activeCam = isInputSpace
          ? (typeof reconInputCamera !== 'undefined' ? reconInputCamera : orbitCamera)
          : (typeof reconOutputCamera !== 'undefined' ? reconOutputCamera : orbitCamera);

        const scale = (Math.min(rect.w, rect.h) / 2.35) * activeZoom;
        const cx = rect.x + rect.w / 2;
        const cy = rect.y + rect.h / 2;
        let u = p.x, v = p.y;
        if (is3D) {
          let tx = 0, ty = 0, tz = 0;
          if (activeCam && activeCam.isLocked) {
            tx = activeCam.targetX || 0;
            ty = activeCam.targetY || 0;
            tz = activeCam.targetZ || 0;
          }
          const pr = project3DVector(
            p.x - tx, p.y - ty, (p.z || 0.0) - tz, activeCam.azimuth, activeCam.elevation
          );
          u = pr.u;
          v = pr.v;
        }
        return {
          px: cx + (u - activePanX) * scale,
          py: cy - (v - activePanY) * scale
        };
      }

      function drawPillBadge(px, py, text, color, rect, offsetY = -16) {
        ctx.font = 'bold 9px monospace';
        const tw = ctx.measureText(text).width;
        const pw = tw + 8;
        const ph = 14;
        const clX = Math.max(rect.x + 4, Math.min(rect.x + rect.w - pw - 4, px - pw / 2));
        const clY = Math.max(rect.y + 4, Math.min(rect.y + rect.h - ph - 4, py + offsetY));

        ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.roundRect(clX, clY, pw, ph, 3);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, clX + pw / 2, clY + ph / 2 + 0.5);
      }

      // =======================================================================
      // MODE A: Training Point Hovered / Locked (Panel A / B) -> k-NN in A/B
      // =======================================================================
      if (activeTrainingIdx >= 0) {
        const isFromA = (activeTrainingSlot === 'A');
        const primSlotId = isFromA ? 'A' : 'B';
        const otherSlotId = isFromA ? 'B' : 'A';
        const primRect = isFromA ? quadRects[0] : quadRects[1];
        const otherRect = isFromA ? quadRects[1] : quadRects[0];
        const primPts = isFromA ? ptsA : ptsB;
        const otherPts = isFromA ? ptsB : ptsA;
        const primDim = isFromA ? dimA : dimB;
        const otherDim = isFromA ? dimB : dimA;
        const primColor = isFromA ? '#38bdf8' : '#4ade80';
        const otherColor = isFromA ? '#4ade80' : '#38bdf8';
        const isPrimInput = isFromA;
        const isOtherInput = !isFromA;

        if (primPts && activeTrainingIdx < primPts.length) {
          const pPrim = primPts[activeTrainingIdx];
          const pOther = (otherPts && activeTrainingIdx < otherPts.length)
            ? otherPts[activeTrainingIdx] : null;
          const knnNbs = getSlotKnnNeighbors(primSlotId, activeTrainingIdx);
          const revQueries = isFromA
            ? getOrComputeReverseKnnNeighbors(activeTrainingIdx, k)
            : [];

          // --- 1. Primary Panel Highlight (A if hovered A, B if hovered B) ---
          ctx.save();
          ctx.beginPath();
          ctx.rect(primRect.x, primRect.y, primRect.w, primRect.h);
          ctx.clip();

          const posPrim = mapPt(pPrim, primRect, primDim >= 3, isPrimInput);

          // Pulse & Outer target circle
          ctx.beginPath();
          ctx.arc(posPrim.px, posPrim.py, 14, 0, Math.PI * 2);
          ctx.fillStyle = isFromA ? 'rgba(56, 189, 248, 0.25)' : 'rgba(74, 222, 128, 0.25)';
          ctx.fill();

          ctx.beginPath();
          ctx.arc(posPrim.px, posPrim.py, 8, 0, Math.PI * 2);
          ctx.strokeStyle = primColor;
          ctx.lineWidth = 2.0;
          ctx.stroke();

          // Crosshairs
          ctx.beginPath();
          ctx.moveTo(posPrim.px - 14, posPrim.py); ctx.lineTo(posPrim.px + 14, posPrim.py);
          ctx.moveTo(posPrim.px, posPrim.py - 14); ctx.lineTo(posPrim.px, posPrim.py + 14);
          ctx.strokeStyle = isFromA ? 'rgba(56, 189, 248, 0.85)' : 'rgba(74, 222, 128, 0.85)';
          ctx.lineWidth = 1.2;
          ctx.stroke();

          // Center solid point
          ctx.beginPath();
          ctx.arc(posPrim.px, posPrim.py, 3.5, 0, Math.PI * 2);
          ctx.fillStyle = primColor;
          ctx.fill();

          // Draw k-NN neighbor rays & markers if dataset has been k-NNed
          if (knnNbs && knnNbs.length > 0) {
            for (let r = 0; r < knnNbs.length; r++) {
              const nb = knnNbs[r];
              const nbIdx = nb.index;
              if (nbIdx < 0 || nbIdx >= primPts.length) continue;
              const pNb = primPts[nbIdx];
              const posNb = mapPt(pNb, primRect, primDim >= 3, isPrimInput);
              const isTop1 = (nb.rank === 1);

              // Connecting Ray
              ctx.beginPath();
              ctx.moveTo(posPrim.px, posPrim.py);
              ctx.lineTo(posNb.px, posNb.py);
              ctx.strokeStyle = isTop1 ? '#4ade80' : (isFromA ? 'rgba(56, 189, 248, 0.75)'
                                                             : 'rgba(192, 132, 252, 0.75)');
              ctx.lineWidth = isTop1 ? 2.0 : 1.2;
              if (!isTop1) ctx.setLineDash([3, 3]);
              ctx.stroke();
              ctx.setLineDash([]);

              // Neighbor ring
              ctx.beginPath();
              ctx.arc(posNb.px, posNb.py, isTop1 ? 7.5 : 5.0, 0, Math.PI * 2);
              ctx.strokeStyle = isTop1 ? '#4ade80' : primColor;
              ctx.lineWidth = isTop1 ? 2.0 : 1.4;
              ctx.stroke();

              // Neighbor center dot
              ctx.beginPath();
              ctx.arc(posNb.px, posNb.py, 2.5, 0, Math.PI * 2);
              ctx.fillStyle = isTop1 ? '#4ade80' : primColor;
              ctx.fill();
            }
          }

          const coordStrPrim = (primDim >= 3 && typeof pPrim.z === 'number')
            ? `(${pPrim.x.toFixed(2)}, ${pPrim.y.toFixed(2)}, ${pPrim.z.toFixed(2)})`
            : `(${pPrim.x.toFixed(2)}, ${pPrim.y.toFixed(2)})`;
          const primTag = (knnNbs && knnNbs.length > 0)
            ? `[${primSlotId}] #${activeTrainingIdx} (k=${knnNbs.length})`
            : `[${primSlotId}] #${activeTrainingIdx}`;
          drawPillBadge(
            posPrim.px, posPrim.py, `${primTag} ${coordStrPrim}`,
            primColor, primRect, -20
          );
          ctx.restore();

          // --- 2. Counterpart Panel Highlight (B if hovered A, A if hovered B) ---
          if (pOther) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(otherRect.x, otherRect.y, otherRect.w, otherRect.h);
            ctx.clip();

            const posOther = mapPt(pOther, otherRect, otherDim >= 3, isOtherInput);

            ctx.beginPath();
            ctx.arc(posOther.px, posOther.py, 14, 0, Math.PI * 2);
            ctx.fillStyle = isFromA ? 'rgba(74, 222, 128, 0.25)' : 'rgba(56, 189, 248, 0.25)';
            ctx.fill();

            ctx.beginPath();
            ctx.arc(posOther.px, posOther.py, 8, 0, Math.PI * 2);
            ctx.strokeStyle = otherColor;
            ctx.lineWidth = 2.0;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(posOther.px - 14, posOther.py); ctx.lineTo(posOther.px + 14, posOther.py);
            ctx.moveTo(posOther.px, posOther.py - 14); ctx.lineTo(posOther.px, posOther.py + 14);
            ctx.strokeStyle = isFromA ? 'rgba(74, 222, 128, 0.85)' : 'rgba(56, 189, 248, 0.85)';
            ctx.lineWidth = 1.2;
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(posOther.px, posOther.py, 3.5, 0, Math.PI * 2);
            ctx.fillStyle = otherColor;
            ctx.fill();

            // Also show corresponding k-NN neighbor points in the other panel!
            if (knnNbs && knnNbs.length > 0 && otherPts) {
              for (let r = 0; r < knnNbs.length; r++) {
                const nb = knnNbs[r];
                const nbIdx = nb.index;
                if (nbIdx < 0 || nbIdx >= otherPts.length) continue;
                const pOtherNb = otherPts[nbIdx];
                const posOtherNb = mapPt(pOtherNb, otherRect, otherDim >= 3, isOtherInput);
                const isTop1 = (nb.rank === 1);

                // Connecting Ray in Counterpart Panel
                ctx.beginPath();
                ctx.moveTo(posOther.px, posOther.py);
                ctx.lineTo(posOtherNb.px, posOtherNb.py);
                ctx.strokeStyle = isTop1 ? '#4ade80' : (isFromA ? 'rgba(74, 222, 128, 0.75)'
                                                               : 'rgba(56, 189, 248, 0.75)');
                ctx.lineWidth = isTop1 ? 2.0 : 1.2;
                if (!isTop1) ctx.setLineDash([3, 3]);
                ctx.stroke();
                ctx.setLineDash([]);

                // Counterpart neighbor ring
                ctx.beginPath();
                ctx.arc(posOtherNb.px, posOtherNb.py, isTop1 ? 7.5 : 5.0, 0, Math.PI * 2);
                ctx.strokeStyle = isTop1 ? '#4ade80' : otherColor;
                ctx.lineWidth = isTop1 ? 2.0 : 1.4;
                ctx.stroke();

                // Counterpart neighbor center dot
                ctx.beginPath();
                ctx.arc(posOtherNb.px, posOtherNb.py, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = isTop1 ? '#4ade80' : otherColor;
                ctx.fill();
              }
            }

            const coordStrOther = (otherDim >= 3 && typeof pOther.z === 'number')
              ? `(${pOther.x.toFixed(2)}, ${pOther.y.toFixed(2)}, ${pOther.z.toFixed(2)})`
              : `(${pOther.x.toFixed(2)}, ${pOther.y.toFixed(2)})`;
            const otherTag = (knnNbs && knnNbs.length > 0)
              ? `Mapped [${otherSlotId}] #${activeTrainingIdx}`
              : `Counterpart #${activeTrainingIdx}`;
            drawPillBadge(
              posOther.px, posOther.py, `${otherTag} ${coordStrOther}`,
              otherColor, otherRect, -20
            );
            ctx.restore();
          }

          // --- 3. Panel C (Bottom-Left): Reverse k-NN Queries C_i (if hovering A) ---
          const rectC = quadRects[2];
          if (isFromA && ptsC && revQueries.length > 0) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(rectC.x, rectC.y, rectC.w, rectC.h);
            ctx.clip();

            // Draw Ghost A[j] in Panel C
            const posA_in_C = mapPt(pPrim, rectC, dimC >= 3, true);

            if (!isOverlay) {
              ctx.beginPath();
              ctx.arc(posA_in_C.px, posA_in_C.py, 7, 0, Math.PI * 2);
              ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)';
              ctx.setLineDash([3, 3]);
              ctx.lineWidth = 1.4;
              ctx.stroke();
              ctx.setLineDash([]);

              ctx.beginPath();
              ctx.moveTo(posA_in_C.px - 10, posA_in_C.py);
              ctx.lineTo(posA_in_C.px + 10, posA_in_C.py);
              ctx.moveTo(posA_in_C.px, posA_in_C.py - 10);
              ctx.lineTo(posA_in_C.px, posA_in_C.py + 10);
              ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
              ctx.lineWidth = 1.0;
              ctx.stroke();

              ctx.beginPath();
              ctx.arc(posA_in_C.px, posA_in_C.py, 2.5, 0, Math.PI * 2);
              ctx.fillStyle = '#38bdf8';
              ctx.fill();
            }

            // Connect each reverse query C_i
            for (let p = 0; p < revQueries.length; p++) {
              const item = revQueries[p];
              const qIdx = item.queryIdx;
              if (qIdx < 0 || qIdx >= ptsC.length) continue;

              const qc = ptsC[qIdx];
              const posQc = mapPt(qc, rectC, dimC >= 3, true);
              const isTop1 = (item.rank === 1);

              ctx.beginPath();
              ctx.moveTo(posA_in_C.px, posA_in_C.py);
              ctx.lineTo(posQc.px, posQc.py);
              ctx.strokeStyle = isTop1 ? '#4ade80' : 'rgba(251, 191, 36, 0.7)';
              ctx.lineWidth = isTop1 ? 2.0 : 1.1;
              if (!isTop1) ctx.setLineDash([3, 3]);
              ctx.stroke();
              ctx.setLineDash([]);

              ctx.beginPath();
              ctx.arc(posQc.px, posQc.py, isTop1 ? 7.5 : 5.0, 0, Math.PI * 2);
              ctx.strokeStyle = isTop1 ? '#4ade80' : '#fbbf24';
              ctx.lineWidth = isTop1 ? 2.0 : 1.4;
              ctx.stroke();

              ctx.beginPath();
              ctx.arc(posQc.px, posQc.py, 2.5, 0, Math.PI * 2);
              ctx.fillStyle = isTop1 ? '#4ade80' : '#fbbf24';
              ctx.fill();
            }
            ctx.restore();
          }

          // --- 4. Panel D (Bottom-Right): Reconstructed Outputs D_i Influenced by B[j] ---
          const rectD = quadRects[3];
          if (isFromA && ptsD && pOther && revQueries.length > 0) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(rectD.x, rectD.y, rectD.w, rectD.h);
            ctx.clip();

            // Draw Ghost B[j] in Panel D
            const posB_in_D = mapPt(pOther, rectD, dimD >= 3, false);

            if (!isOverlay) {
              ctx.beginPath();
              ctx.arc(posB_in_D.px, posB_in_D.py, 7, 0, Math.PI * 2);
              ctx.strokeStyle = 'rgba(74, 222, 128, 0.85)';
              ctx.setLineDash([3, 3]);
              ctx.lineWidth = 1.4;
              ctx.stroke();
              ctx.setLineDash([]);

              ctx.beginPath();
              ctx.moveTo(posB_in_D.px - 10, posB_in_D.py);
              ctx.lineTo(posB_in_D.px + 10, posB_in_D.py);
              ctx.moveTo(posB_in_D.px, posB_in_D.py - 10);
              ctx.lineTo(posB_in_D.px, posB_in_D.py + 10);
              ctx.strokeStyle = 'rgba(74, 222, 128, 0.6)';
              ctx.lineWidth = 1.0;
              ctx.stroke();

              ctx.beginPath();
              ctx.arc(posB_in_D.px, posB_in_D.py, 2.5, 0, Math.PI * 2);
              ctx.fillStyle = '#4ade80';
              ctx.fill();
            }

            for (let p = 0; p < revQueries.length; p++) {
              const item = revQueries[p];
              const qIdx = item.queryIdx;
              if (qIdx < 0 || qIdx >= ptsD.length) continue;

              const qd = ptsD[qIdx];
              const posQd = mapPt(qd, rectD, dimD >= 3, false);
              const w = item.weight;
              const isTop1 = (item.rank === 1);

              ctx.beginPath();
              ctx.moveTo(posB_in_D.px, posB_in_D.py);
              ctx.lineTo(posQd.px, posQd.py);
              ctx.strokeStyle =
                `rgba(192, 132, 252, ${Math.max(0.35, Math.min(0.95, w * 2.5))})`;
              ctx.lineWidth = Math.max(1.0, Math.min(3.5, w * 6.0));
              ctx.stroke();

              const ringRad = Math.max(4.5, Math.min(10.0, 4.0 + w * 12.0));
              ctx.beginPath();
              ctx.arc(posQd.px, posQd.py, ringRad, 0, Math.PI * 2);
              ctx.strokeStyle = isTop1 ? '#4ade80' : 'rgba(192, 132, 252, 0.85)';
              ctx.lineWidth = isTop1 ? 2.0 : 1.4;
              ctx.stroke();

              ctx.beginPath();
              ctx.arc(posQd.px, posQd.py, 2.5, 0, Math.PI * 2);
              ctx.fillStyle = '#c084fc';
              ctx.fill();
            }
            ctx.restore();
          }
        }
        return;
      }

      // =======================================================================
      // MODE B: Query Point Hovered / Locked (Panel C / D) -> Forward k-NN
      // =======================================================================
      if (!ptsC || activeQueryIdx < 0 || activeQueryIdx >= ptsC.length) return;

      const qc = ptsC[activeQueryIdx];
      const qd = (ptsD && activeQueryIdx < ptsD.length) ? ptsD[activeQueryIdx] : null;
      const neighbors = getOrComputeKnnNeighbors(activeQueryIdx, k);

      // --- 1. Panel C (Bottom-Left): Query Point Highlight ---
      const rectC = quadRects[2];
      ctx.save();
      ctx.beginPath();
      ctx.rect(rectC.x, rectC.y, rectC.w, rectC.h);
      ctx.clip();

      const posC = mapPt(qc, rectC, dimC >= 3, true);

      ctx.beginPath();
      ctx.arc(posC.px, posC.py, 13, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(251, 191, 36, 0.22)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(posC.px, posC.py, 8, 0, Math.PI * 2);
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2.0;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(posC.px - 14, posC.py); ctx.lineTo(posC.px + 14, posC.py);
      ctx.moveTo(posC.px, posC.py - 14); ctx.lineTo(posC.px, posC.py + 14);
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.85)';
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(posC.px, posC.py, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#fbbf24';
      ctx.fill();

      const coordStrC = (dimC >= 3 && typeof qc.z === 'number')
        ? `(${qc.x.toFixed(2)}, ${qc.y.toFixed(2)}, ${qc.z.toFixed(2)})`
        : `(${qc.x.toFixed(2)}, ${qc.y.toFixed(2)})`;
      drawPillBadge(
        posC.px, posC.py, `Query #${activeQueryIdx} ${coordStrC}`, '#fbbf24', rectC, -20
      );
      ctx.restore();

      // --- 2. Panel D (Bottom-Right): Reconstructed Output Highlight ---
      const rectD = quadRects[3];
      if (qd) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(rectD.x, rectD.y, rectD.w, rectD.h);
        ctx.clip();

        const posD = mapPt(qd, rectD, dimD >= 3, false);

        ctx.beginPath();
        ctx.arc(posD.px, posD.py, 13, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(192, 132, 252, 0.22)';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(posD.px, posD.py, 8, 0, Math.PI * 2);
        ctx.strokeStyle = '#c084fc';
        ctx.lineWidth = 2.0;
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(posD.px - 14, posD.py); ctx.lineTo(posD.px + 14, posD.py);
        ctx.moveTo(posD.px, posD.py - 14); ctx.lineTo(posD.px, posD.py + 14);
        ctx.strokeStyle = 'rgba(192, 132, 252, 0.85)';
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(posD.px, posD.py, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = '#c084fc';
        ctx.fill();

        const coordStrD = (dimD >= 3 && typeof qd.z === 'number')
          ? `(${qd.x.toFixed(2)}, ${qd.y.toFixed(2)}, ${qd.z.toFixed(2)})`
          : `(${qd.x.toFixed(2)}, ${qd.y.toFixed(2)})`;
        const varStr = (slotD.reconVariance && activeQueryIdx < slotD.reconVariance.length)
          ? ` • Var:${slotD.reconVariance[activeQueryIdx].toFixed(4)}` : '';
        drawPillBadge(
          posD.px, posD.py, `Output #${activeQueryIdx} ${coordStrD}${varStr}`,
          '#c084fc', rectD, -20
        );
        ctx.restore();
      }

      // --- 3. Panel A (Top-Left): Ghost Query C_i and k-NN Distance Vectors ---
      const rectA = quadRects[0];
      if (ptsA && neighbors && neighbors.length > 0) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(rectA.x, rectA.y, rectA.w, rectA.h);
        ctx.clip();

        const posQ_in_A = mapPt(qc, rectA, dimA >= 3, true);

        // Draw Ghost Query reticle in A
        if (!isOverlay) {
          ctx.beginPath();
          ctx.arc(posQ_in_A.px, posQ_in_A.py, 7, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(251, 191, 36, 0.8)';
          ctx.setLineDash([3, 3]);
          ctx.lineWidth = 1.4;
          ctx.stroke();
          ctx.setLineDash([]);

          ctx.beginPath();
          ctx.moveTo(posQ_in_A.px - 10, posQ_in_A.py);
          ctx.lineTo(posQ_in_A.px + 10, posQ_in_A.py);
          ctx.moveTo(posQ_in_A.px, posQ_in_A.py - 10);
          ctx.lineTo(posQ_in_A.px, posQ_in_A.py + 10);
          ctx.strokeStyle = 'rgba(251, 191, 36, 0.6)';
          ctx.lineWidth = 1.0;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(posQ_in_A.px, posQ_in_A.py, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = '#fbbf24';
          ctx.fill();
        }

        // Draw lines and nodes for all k neighbors in A
        for (let p = 0; p < neighbors.length; p++) {
          const item = neighbors[p];
          const nId = item.id;
          if (nId < 0 || nId >= ptsA.length) continue;

          const pa = ptsA[nId];
          const posNa = mapPt(pa, rectA, dimA >= 3, true);
          const isTop1 = (p === 0);

          // Vector distance line from ghost query to neighbor
          ctx.beginPath();
          ctx.moveTo(posQ_in_A.px, posQ_in_A.py);
          ctx.lineTo(posNa.px, posNa.py);
          ctx.strokeStyle = isTop1 ? '#4ade80' : 'rgba(56, 189, 248, 0.7)';
          ctx.lineWidth = isTop1 ? 2.0 : 1.1;
          if (!isTop1) ctx.setLineDash([3, 3]);
          ctx.stroke();
          ctx.setLineDash([]);

          // Neighbor node highlight
          ctx.beginPath();
          ctx.arc(posNa.px, posNa.py, isTop1 ? 7.5 : 5.0, 0, Math.PI * 2);
          ctx.strokeStyle = isTop1 ? '#4ade80' : '#38bdf8';
          ctx.lineWidth = isTop1 ? 2.0 : 1.4;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(posNa.px, posNa.py, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = isTop1 ? '#4ade80' : '#38bdf8';
          ctx.fill();
        }
        ctx.restore();
      }

      // --- 4. Panel B (Top-Right): Ghost Recon Output D_i and Weighted Vectors ---
      const rectB = quadRects[1];
      if (ptsB && neighbors && neighbors.length > 0) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(rectB.x, rectB.y, rectB.w, rectB.h);
        ctx.clip();

        let posD_in_B = null;
        if (qd) {
          posD_in_B = mapPt(qd, rectB, dimB >= 3, false);

          // Draw Ghost Reconstructed Output reticle in B
          if (!isOverlay) {
            ctx.beginPath();
            ctx.arc(posD_in_B.px, posD_in_B.py, 7, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(192, 132, 252, 0.8)';
            ctx.setLineDash([3, 3]);
            ctx.lineWidth = 1.4;
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.beginPath();
            ctx.moveTo(posD_in_B.px - 10, posD_in_B.py);
            ctx.lineTo(posD_in_B.px + 10, posD_in_B.py);
            ctx.moveTo(posD_in_B.px, posD_in_B.py - 10);
            ctx.lineTo(posD_in_B.px, posD_in_B.py + 10);
            ctx.strokeStyle = 'rgba(192, 132, 252, 0.6)';
            ctx.lineWidth = 1.0;
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(posD_in_B.px, posD_in_B.py, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = '#c084fc';
            ctx.fill();
          }
        }

        // Draw weighted contribution vectors and nodes in B
        for (let p = 0; p < neighbors.length; p++) {
          const item = neighbors[p];
          const nId = item.id;
          if (nId < 0 || nId >= ptsB.length) continue;

          const pb = ptsB[nId];
          const posNb = mapPt(pb, rectB, dimB >= 3, false);
          const w = item.weight;
          const isTop1 = (p === 0);

          if (posD_in_B) {
            ctx.beginPath();
            ctx.moveTo(posNb.px, posNb.py);
            ctx.lineTo(posD_in_B.px, posD_in_B.py);
            ctx.strokeStyle = `rgba(74, 222, 128, ${Math.max(0.35, Math.min(0.95, w * 2.5))})`;
            ctx.lineWidth = Math.max(1.0, Math.min(3.5, w * 6.0));
            ctx.stroke();
          }

          const ringRad = Math.max(4.5, Math.min(10.0, 4.0 + w * 12.0));
          ctx.beginPath();
          ctx.arc(posNb.px, posNb.py, ringRad, 0, Math.PI * 2);
          ctx.strokeStyle = isTop1 ? '#4ade80' : 'rgba(74, 222, 128, 0.85)';
          ctx.lineWidth = isTop1 ? 2.0 : 1.4;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(posNb.px, posNb.py, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = '#4ade80';
          ctx.fill();
        }
        ctx.restore();
      }
    }

    window.drawRecon4PanelView = drawRecon4PanelView;
    window.getOrComputeKnnNeighbors = getOrComputeKnnNeighbors;
    window.getOrComputeReverseKnnNeighbors = getOrComputeReverseKnnNeighbors;
