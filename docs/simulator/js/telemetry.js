/**
 * GRIC Simulator - telemetry.js
 * Resource Trackers, Performance Sparklines, Distance Curves & Markov Visualizations
 */

// =========================================================================
    //  RESOURCE & PERFORMANCE TRACKER STATE & TELEMETRY
    // =========================================================================
    let distSampleCluster = 0;      // Total Sample-to-Cluster evaluations d(f, c)
    let distClusterCluster = 0;     // Total Cluster-to-Cluster evaluations d(ci, cj)
    let distSampleClusterLast = 0;  // Last frame SC evaluations
    let distClusterClusterLast = 0; // Last frame CC evaluations
    let dccPopulated = 0;           // Populated exact inter-cluster distance pairs
    let dccPairsTotal = 0;          // Total possible inter-cluster pairs K*(K-1)/2
    let pruneCount3P = 0;           // Pruned candidates via 3-Point
    let pruneCount4P = 0;           // Pruned candidates via 4-Point
    let pruneCount5P = 0;           // Pruned candidates via 5-Point
    let predHitCount = 0;           // Successful sequence/TM predictor hits

    // High-Resolution Execution Profiling
    let totalComputeTimeMs = 0.0;
    let lastComputeTimeMs = 0.0;
    let avgComputeTimeMs = 0.0;
    let sparklineHistory = new Array(60).fill(0.0);
    let distHistoryDFC = [];
    let distHistoryDCC = [];
    let hoverDistIndex = null;      // Frame index hovered on distCurvesCanvas
    let hoverDistAvgIndex = null;   // Frame index hovered on distAvgCurvesCanvas
    let rollingHistory = [];        // [{ time, computeMs, frames, dists }] in rolling 1-sec window
    let currentCpuLoadPct = 0.0;
    let currentFps = 0.0;
    let currentDistRate = 0.0;

    // Session Clock Timing & Live Average FPS
    let sessionStartTime = 0;
    let sessionStartFrames = 0;
    let sessionElapsedMs = 0;
    let sessionIsActive = false;
    let sessionAvgFps = 0.0;

    function formatClockTime(ms) {
      if (ms <= 0 || isNaN(ms)) return "0.00 s";
      const sec = ms / 1000.0;
      if (sec < 1.0) return `${sec.toFixed(3)} s`;
      if (sec < 60.0) return `${sec.toFixed(2)} s`;
      const m = Math.floor(sec / 60.0);
      const s = (sec % 60.0).toFixed(1);
      return `${m}m ${s}s`;
    }

    function formatNumber(num) {
      if (num === null || num === undefined || isNaN(num)) return "0";
      return Math.round(num).toLocaleString();
    }

    function formatBytes(bytes) {
      if (bytes === 0 || isNaN(bytes)) return "0 B";
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }

    function getMemoryStats() {
      let heapUsedMB = null;
      let heapTotalMB = null;
      let heapLimitMB = null;

      if (typeof performance !== 'undefined' && performance.memory) {
        heapUsedMB = performance.memory.usedJSHeapSize / (1024 * 1024);
        heapTotalMB = performance.memory.totalJSHeapSize / (1024 * 1024);
        heapLimitMB = performance.memory.jsHeapSizeLimit / (1024 * 1024);
      }

      const K = useTiles 
        ? (tileEngineX.clusters.length + tileEngineY.clusters.length + (currentDim === 3 ? tileEngineZ.clusters.length : 0)) 
        : clusters.length;

      // Centroids & metadata: ~64 bytes per cluster centroid
      const centroidsBytes = K * 64;

      // Inter-cluster distance matrix Dcc: Float64 (8 bytes) per cell
      let dccCells = 0;
      if (useTiles) {
        dccCells = (tileEngineX.clusters.length ** 2) + 
                   (tileEngineY.clusters.length ** 2) + 
                   (currentDim === 3 ? (tileEngineZ.clusters.length ** 2) : 0);
      } else {
        dccCells = clusters.length * clusters.length;
      }
      const dccBytes = dccCells * 8;

      // Markov Transition Matrix: 4 bytes per cell
      const tmBytes = (clusters.length * clusters.length) * 4;

      // Past samples point cloud: 3 doubles (24 bytes) per coordinate
      const pastSamplesBytes = pastSamples.length * 24;

      // History buffers: sequence history (4B) + frame telemetry structures (~32B)
      const historyBytes = (assignmentHistory.length * 4) + (frameHistory.length * 32);

      // Subspace Tiles & Joint Tuple Map entries: ~48B per entry
      const tuplesBytes = jointTuplesMap.size * 48;

      const totalModelBytes = centroidsBytes + dccBytes + tmBytes + pastSamplesBytes + historyBytes + tuplesBytes;

      return {
        heapUsedMB,
        heapTotalMB,
        heapLimitMB,
        centroidsBytes,
        dccBytes,
        dccCells,
        tmBytes,
        pastSamplesBytes,
        historyBytes,
        tuplesBytes,
        totalModelBytes
      };
    }

    function updateResourceMetrics() {
      const now = performance.now();
      const cutoff = now - 1000;
      let pruneIdx = 0;
      while (pruneIdx < rollingHistory.length && rollingHistory[pruneIdx].time < cutoff) {
        pruneIdx++;
      }
      if (pruneIdx > 0) {
        rollingHistory = rollingHistory.slice(pruneIdx);
      }

      if (rollingHistory.length > 0) {
        let windowComputeMs = 0;
        let windowFrames = 0;
        let windowDists = 0;
        for (let i = 0; i < rollingHistory.length; i++) {
          windowComputeMs += rollingHistory[i].computeMs;
          windowFrames += rollingHistory[i].frames;
          windowDists += rollingHistory[i].dists;
        }
        const wallSpanMs = Math.max(16, now - rollingHistory[0].time);
        const wallSpanSec = wallSpanMs / 1000.0;

        currentCpuLoadPct = Math.min(100.0, (windowComputeMs / wallSpanMs) * 100.0);
        currentFps = isRunning ? (windowFrames / wallSpanSec) : (sessionAvgFps || currentFps);
        currentDistRate = windowDists / wallSpanSec;
      } else if (!isRunning && sessionAvgFps > 0) {
        currentFps = sessionAvgFps;
      } else if (!isRunning && sessionElapsedMs <= 0) {
        currentCpuLoadPct = 0.0;
        currentFps = 0.0;
        currentDistRate = 0.0;
      }
    }

    function drawSparkline() {
      const spCanvas = document.getElementById('sparklineCanvas');
      if (!spCanvas) return;
      const spCtx = spCanvas.getContext('2d');
      const w = spCanvas.width;
      const h = spCanvas.height;

      spCtx.clearRect(0, 0, w, h);
      spCtx.fillStyle = '#0f172a';
      spCtx.fillRect(0, 0, w, h);

      const maxVal = Math.max(0.05, ...sparklineHistory);
      const n = sparklineHistory.length;
      if (n < 2) return;

      // Draw middle guideline
      spCtx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
      spCtx.lineWidth = 1;
      spCtx.beginPath();
      spCtx.moveTo(0, h / 2);
      spCtx.lineTo(w, h / 2);
      spCtx.stroke();

      // Sparkline fill gradient
      spCtx.beginPath();
      spCtx.moveTo(0, h);
      for (let i = 0; i < n; i++) {
        const x = (i / (n - 1)) * w;
        const y = h - (sparklineHistory[i] / maxVal) * (h - 4) - 2;
        spCtx.lineTo(x, y);
      }
      spCtx.lineTo(w, h);
      spCtx.closePath();
      spCtx.fillStyle = 'rgba(56, 189, 248, 0.18)';
      spCtx.fill();

      // Sparkline stroke
      spCtx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = (i / (n - 1)) * w;
        const y = h - (sparklineHistory[i] / maxVal) * (h - 4) - 2;
        if (i === 0) spCtx.moveTo(x, y);
        else spCtx.lineTo(x, y);
      }
      spCtx.strokeStyle = '#38bdf8';
      spCtx.lineWidth = 1.5;
      spCtx.stroke();

      // Latest value indicator dot
      const lastY = h - (sparklineHistory[n - 1] / maxVal) * (h - 4) - 2;
      spCtx.fillStyle = '#fbbf24';
      spCtx.beginPath();
      spCtx.arc(w - 2, lastY, 2.5, 0, Math.PI * 2);
      spCtx.fill();
    }

    function recordFrameTelemetry(frameComputeMs, batchFrames = 1, batchDists = 0) {
      lastComputeTimeMs = frameComputeMs;
      totalComputeTimeMs += frameComputeMs;
      avgComputeTimeMs = totalFrames > 0 ? (totalComputeTimeMs / totalFrames) : 0.0;

      sparklineHistory.push(frameComputeMs);
      if (sparklineHistory.length > 80) {
        sparklineHistory = sparklineHistory.slice(-60);
      }

      distHistoryDFC.push(distSampleClusterLast);
      distHistoryDCC.push(distClusterClusterLast);
      if (distHistoryDFC.length > 6000) {
        distHistoryDFC = distHistoryDFC.slice(-5000);
        distHistoryDCC = distHistoryDCC.slice(-5000);
      }

      const nowTime = performance.now();
      rollingHistory.push({
        time: nowTime,
        computeMs: frameComputeMs,
        frames: batchFrames,
        dists: batchDists || (distSampleClusterLast + distClusterClusterLast)
      });
      if (rollingHistory.length > 1000) {
        rollingHistory = rollingHistory.slice(-500);
      }
    }

    function drawDistCurves() {
      const cvs = document.getElementById('distCurvesCanvas');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      const w = cvs.width;
      const h = cvs.height;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, w, h);

      const n = distHistoryDFC.length;

      const lblFrames = document.getElementById('lblDistCurveFrames');
      const lblDFC = document.getElementById('lblLastDFC');
      const lblDCC = document.getElementById('lblLastDCC');

      if (n === 0) {
        if (lblFrames) lblFrames.innerText = 'Frame 0 → 0';
        if (lblDFC) lblDFC.innerText = '0';
        if (lblDCC) lblDCC.innerText = '0';
        ctx.fillStyle = '#64748b';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Awaiting frame stream data...', w / 2, h / 2 + 3);
        return;
      }

      // Max value for Y scaling (computed with safe loop, no spread operator)
      let maxVal = 4;
      for (let i = 0; i < n; i++) {
        if (distHistoryDFC[i] > maxVal) maxVal = distHistoryDFC[i];
        if (distHistoryDCC[i] > maxVal) maxVal = distHistoryDCC[i];
      }
      maxVal = Math.ceil(maxVal * 1.15);

      // Grid horizontal lines (50%)
      ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5);
      ctx.stroke();

      // Scale indicator labels
      ctx.fillStyle = '#64748b';
      ctx.font = '8px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`max: ${maxVal}`, 3, 9);
      ctx.fillText(`f=0`, 3, h - 2);
      ctx.textAlign = 'right';
      ctx.fillText(`f=${n - 1}`, w - 3, h - 2);

      if (n === 1) {
        // Single frame point
        const y_dfc = h - (distHistoryDFC[0] / maxVal) * (h - 6) - 3;
        const y_dcc = h - (distHistoryDCC[0] / maxVal) * (h - 6) - 3;
        
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc(10, y_dfc, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#c084fc';
        ctx.beginPath();
        ctx.arc(10, y_dcc, 3, 0, Math.PI * 2);
        ctx.fill();

        if (lblFrames) lblFrames.innerText = 'Frame 0';
        if (lblDFC) lblDFC.innerText = `${distHistoryDFC[0]}`;
        if (lblDCC) lblDCC.innerText = `${distHistoryDCC[0]}`;
        return;
      }

      // Subsample points if n > w * 1.5 to prevent overdraw
      const maxPts = Math.min(n, Math.max(100, Math.floor(w * 1.5)));
      const step = (n - 1) / (maxPts - 1);

      // 1. Draw d(f,c) Shaded Area and Stroke (Cyan #38bdf8)
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let p = 0; p < maxPts; p++) {
        const i = Math.min(n - 1, Math.round(p * step));
        const x = (i / (n - 1)) * w;
        const y = h - (distHistoryDFC[i] / maxVal) * (h - 6) - 3;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
      ctx.fill();

      ctx.beginPath();
      for (let p = 0; p < maxPts; p++) {
        const i = Math.min(n - 1, Math.round(p * step));
        const x = (i / (n - 1)) * w;
        const y = h - (distHistoryDFC[i] / maxVal) * (h - 6) - 3;
        if (p === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.6;
      ctx.stroke();

      const lastY_dfc = h - (distHistoryDFC[n - 1] / maxVal) * (h - 6) - 3;
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(w - 2, lastY_dfc, 2.5, 0, Math.PI * 2);
      ctx.fill();

      // 2. Draw d(c,c) Shaded Area and Stroke (Purple #c084fc)
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let p = 0; p < maxPts; p++) {
        const i = Math.min(n - 1, Math.round(p * step));
        const x = (i / (n - 1)) * w;
        const y = h - (distHistoryDCC[i] / maxVal) * (h - 6) - 3;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fillStyle = 'rgba(192, 132, 252, 0.12)';
      ctx.fill();

      ctx.beginPath();
      for (let p = 0; p < maxPts; p++) {
        const i = Math.min(n - 1, Math.round(p * step));
        const x = (i / (n - 1)) * w;
        const y = h - (distHistoryDCC[i] / maxVal) * (h - 6) - 3;
        if (p === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#c084fc';
      ctx.lineWidth = 1.6;
      ctx.stroke();

      const lastY_dcc = h - (distHistoryDCC[n - 1] / maxVal) * (h - 6) - 3;
      ctx.fillStyle = '#c084fc';
      ctx.beginPath();
      ctx.arc(w - 2, lastY_dcc, 2.5, 0, Math.PI * 2);
      ctx.fill();

      // 3. Hover Guide & Exact Value Tooltip
      if (hoverDistIndex !== null && hoverDistIndex >= 0 && hoverDistIndex < n) {
        const hX = (hoverDistIndex / (n - 1)) * w;
        const hValDFC = distHistoryDFC[hoverDistIndex];
        const hValDCC = distHistoryDCC[hoverDistIndex];
        const hValTot = hValDFC + hValDCC;
        const hY_dfc = h - (hValDFC / maxVal) * (h - 6) - 3;
        const hY_dcc = h - (hValDCC / maxVal) * (h - 6) - 3;

        // Vertical Dashed Crosshair
        ctx.save();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(hX, 0);
        ctx.lineTo(hX, h);
        ctx.stroke();
        ctx.restore();

        // Highlight Dots with Halos
        ctx.fillStyle = '#ffffff';
        ctx.beginPath(); ctx.arc(hX, hY_dfc, 4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath(); ctx.arc(hX, hY_dfc, 2.5, 0, Math.PI * 2); ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.beginPath(); ctx.arc(hX, hY_dcc, 4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#c084fc';
        ctx.beginPath(); ctx.arc(hX, hY_dcc, 2.5, 0, Math.PI * 2); ctx.fill();

        // In-Canvas Floating Tooltip Tag
        const tagText = `f=${hoverDistIndex}: d(f,c)=${hValDFC} d(c,c)=${hValDCC} (Σ=${hValTot})`;
        ctx.font = 'bold 9px monospace';
        const tw = ctx.measureText(tagText).width;
        const boxW = tw + 8;
        const boxH = 14;
        let boxX = hX + 6;
        if (boxX + boxW > w - 2) boxX = hX - boxW - 6;
        if (boxX < 2) boxX = 2;
        const boxY = 3;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(boxX, boxY, boxW, boxH, 3);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#f8fafc';
        ctx.textAlign = 'left';
        ctx.fillText(tagText, boxX + 4, boxY + 10);

        if (lblFrames) lblFrames.innerText = `Frame ${hoverDistIndex} (Hovered)`;
        if (lblDFC) lblDFC.innerText = `${hValDFC}`;
        if (lblDCC) lblDCC.innerText = `${hValDCC}`;
      } else {
        if (lblFrames) lblFrames.innerText = `Frame 0 → ${n - 1} (${n} frames)`;
        if (lblDFC) lblDFC.innerText = `${distHistoryDFC[n - 1]}`;
        if (lblDCC) lblDCC.innerText = `${distHistoryDCC[n - 1]}`;
      }
    }

    function drawDistAvgCurves() {
      const cvs = document.getElementById('distAvgCurvesCanvas');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      const w = cvs.width;
      const h = cvs.height;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, w, h);

      const n = distHistoryDFC.length;

      const lblFrames = document.getElementById('lblDistAvgCurveFrames');
      const lblMeanDFC = document.getElementById('lblLiveMeanDFC');
      const lblMeanDCC = document.getElementById('lblLiveMeanDCC');

      if (n === 0) {
        if (lblFrames) lblFrames.innerText = 'Since Sample 0';
        if (lblMeanDFC) lblMeanDFC.innerText = '0.00';
        if (lblMeanDCC) lblMeanDCC.innerText = '0.00';
        ctx.fillStyle = '#64748b';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Awaiting frame stream data...', w / 2, h / 2 + 3);
        return;
      }

      // Subsample points if n > w * 1.5 to prevent overdraw
      const maxPts = Math.min(n, Math.max(100, Math.floor(w * 1.5)));
      const step = (n - 1) / (maxPts - 1);

      const sampledAvgDFC = new Float64Array(maxPts);
      const sampledAvgDCC = new Float64Array(maxPts);
      const sampleIndices = new Int32Array(maxPts);
      for (let p = 0; p < maxPts; p++) {
        sampleIndices[p] = Math.min(n - 1, Math.round(p * step));
      }

      let sumDFC = 0, sumDCC = 0;
      let maxVal = 1.0;
      let samplePtr = 0;

      for (let i = 0; i < n; i++) {
        sumDFC += distHistoryDFC[i];
        sumDCC += distHistoryDCC[i];
        while (samplePtr < maxPts && sampleIndices[samplePtr] === i) {
          const aDFC = sumDFC / (i + 1);
          const aDCC = sumDCC / (i + 1);
          sampledAvgDFC[samplePtr] = aDFC;
          sampledAvgDCC[samplePtr] = aDCC;
          if (aDFC > maxVal) maxVal = aDFC;
          if (aDCC > maxVal) maxVal = aDCC;
          samplePtr++;
        }
      }

      const finalAvgDFC = sumDFC / n;
      const finalAvgDCC = sumDCC / n;
      maxVal = Math.ceil(maxVal * 1.2 * 10) / 10;

      // Grid horizontal lines (50%)
      ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5);
      ctx.stroke();

      // Scale indicator labels
      ctx.fillStyle = '#64748b';
      ctx.font = '8px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`max: ${maxVal.toFixed(1)}`, 3, 9);
      ctx.fillText(`f=0`, 3, h - 2);
      ctx.textAlign = 'right';
      ctx.fillText(`f=${n - 1}`, w - 3, h - 2);

      if (n === 1) {
        // Single frame point
        const y_dfc = h - (sampledAvgDFC[0] / maxVal) * (h - 6) - 3;
        const y_dcc = h - (sampledAvgDCC[0] / maxVal) * (h - 6) - 3;
        
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc(10, y_dfc, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#c084fc';
        ctx.beginPath();
        ctx.arc(10, y_dcc, 3, 0, Math.PI * 2);
        ctx.fill();

        if (lblFrames) lblFrames.innerText = 'f=0 (1 frame)';
        if (lblMeanDFC) lblMeanDFC.innerText = finalAvgDFC.toFixed(2);
        if (lblMeanDCC) lblMeanDCC.innerText = finalAvgDCC.toFixed(2);
        return;
      }

      // 1. Draw Running Avg d(f,c) Shaded Area and Stroke (Cyan #38bdf8)
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let p = 0; p < maxPts; p++) {
        const i = sampleIndices[p];
        const x = (i / (n - 1)) * w;
        const y = h - (sampledAvgDFC[p] / maxVal) * (h - 6) - 3;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
      ctx.fill();

      ctx.beginPath();
      for (let p = 0; p < maxPts; p++) {
        const i = sampleIndices[p];
        const x = (i / (n - 1)) * w;
        const y = h - (sampledAvgDFC[p] / maxVal) * (h - 6) - 3;
        if (p === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.6;
      ctx.stroke();

      const lastY_dfc = h - (finalAvgDFC / maxVal) * (h - 6) - 3;
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(w - 2, lastY_dfc, 2.5, 0, Math.PI * 2);
      ctx.fill();

      // 2. Draw Running Avg d(c,c) Shaded Area and Stroke (Purple #c084fc)
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let p = 0; p < maxPts; p++) {
        const i = sampleIndices[p];
        const x = (i / (n - 1)) * w;
        const y = h - (sampledAvgDCC[p] / maxVal) * (h - 6) - 3;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fillStyle = 'rgba(192, 132, 252, 0.12)';
      ctx.fill();

      ctx.beginPath();
      for (let p = 0; p < maxPts; p++) {
        const i = sampleIndices[p];
        const x = (i / (n - 1)) * w;
        const y = h - (sampledAvgDCC[p] / maxVal) * (h - 6) - 3;
        if (p === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#c084fc';
      ctx.lineWidth = 1.6;
      ctx.stroke();

      const lastY_dcc = h - (finalAvgDCC / maxVal) * (h - 6) - 3;
      ctx.fillStyle = '#c084fc';
      ctx.beginPath();
      ctx.arc(w - 2, lastY_dcc, 2.5, 0, Math.PI * 2);
      ctx.fill();

      // 3. Hover Guide & Exact Value Tooltip
      if (hoverDistAvgIndex !== null && hoverDistAvgIndex >= 0 && hoverDistAvgIndex < n) {
        const hX = (hoverDistAvgIndex / (n - 1)) * w;
        let sDFC = 0, sDCC = 0;
        for (let i = 0; i <= hoverDistAvgIndex; i++) {
          sDFC += distHistoryDFC[i];
          sDCC += distHistoryDCC[i];
        }
        const hAvgDFC = sDFC / (hoverDistAvgIndex + 1);
        const hAvgDCC = sDCC / (hoverDistAvgIndex + 1);
        const hAvgTot = hAvgDFC + hAvgDCC;
        const hY_dfc = h - (hAvgDFC / maxVal) * (h - 6) - 3;
        const hY_dcc = h - (hAvgDCC / maxVal) * (h - 6) - 3;

        // Vertical Dashed Crosshair
        ctx.save();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(hX, 0);
        ctx.lineTo(hX, h);
        ctx.stroke();
        ctx.restore();

        // Highlight Dots with Halos
        ctx.fillStyle = '#ffffff';
        ctx.beginPath(); ctx.arc(hX, hY_dfc, 4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath(); ctx.arc(hX, hY_dfc, 2.5, 0, Math.PI * 2); ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.beginPath(); ctx.arc(hX, hY_dcc, 4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#c084fc';
        ctx.beginPath(); ctx.arc(hX, hY_dcc, 2.5, 0, Math.PI * 2); ctx.fill();

        // In-Canvas Floating Tooltip Tag
        const tagText = `f=${hoverDistAvgIndex}: avg d(f,c)=${hAvgDFC.toFixed(2)} avg d(c,c)=${hAvgDCC.toFixed(2)} (Σ=${hAvgTot.toFixed(2)})`;
        ctx.font = 'bold 9px monospace';
        const tw = ctx.measureText(tagText).width;
        const boxW = tw + 8;
        const boxH = 14;
        let boxX = hX + 6;
        if (boxX + boxW > w - 2) boxX = hX - boxW - 6;
        if (boxX < 2) boxX = 2;
        const boxY = 3;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
        ctx.strokeStyle = 'rgba(192, 132, 252, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(boxX, boxY, boxW, boxH, 3);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#f8fafc';
        ctx.textAlign = 'left';
        ctx.fillText(tagText, boxX + 4, boxY + 10);

        if (lblFrames) lblFrames.innerText = `f=${hoverDistAvgIndex} (Cumul. Avg)`;
        if (lblMeanDFC) lblMeanDFC.innerText = hAvgDFC.toFixed(2);
        if (lblMeanDCC) lblMeanDCC.innerText = hAvgDCC.toFixed(2);
      } else {
        if (lblFrames) lblFrames.innerText = `f=0 → ${n - 1} (${n} frames)`;
        if (lblMeanDFC) lblMeanDFC.innerText = finalAvgDFC.toFixed(2);
        if (lblMeanDCC) lblMeanDCC.innerText = finalAvgDCC.toFixed(2);
      }
    }

    function computeTopLearnedPaths() {
      const K = clusters.length;
      if (K === 0 || transitionCounts.length === 0) {
        topLearnedPathsCache = [];
        return;
      }
      const paths = [];
      for (let i = 0; i < K; i++) {
        if (!transitionCounts[i]) continue;
        let sum = 0;
        for (let j = 0; j < K; j++) sum += transitionCounts[i][j] || 0;
        if (sum === 0) continue;
        for (let j = 0; j < K; j++) {
          if (i === j) continue;
          const cnt = transitionCounts[i][j] || 0;
          if (cnt > 0) {
            paths.push({ from: i, to: j, count: cnt, prob: cnt / sum });
          }
        }
      }
      paths.sort((a, b) => b.count - a.count);
      topLearnedPathsCache = paths.slice(0, 10);
    }

    function drawTransitionMatrix(canvasId, isMini) {
      const cvs = document.getElementById(canvasId);
      if (!cvs) return;

      // Visibility guard: skip drawing if canvas or tab is not visible
      if (!isMini && typeof currentActiveTab !== 'undefined' && currentActiveTab !== 'tm') return;
      if (isMini && !useTM) return;

      const ctx = cvs.getContext('2d');
      const dims = tmCanvasDimensions[canvasId] || { w: isMini ? 340 : 420, h: isMini ? 210 : 320 };
      const w = Math.max(isMini ? 240 : 280, dims.w);
      const h = Math.max(isMini ? 150 : 200, dims.h);

      if (cvs.width !== w || cvs.height !== h) {
        cvs.width = w;
        cvs.height = h;
      }

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#0b1120';
      ctx.fillRect(0, 0, w, h);

      const K = clusters.length;
      if (K === 0 || transitionCounts.length === 0) {
        ctx.fillStyle = '#64748b';
        ctx.font = isMini ? '10px monospace' : '12px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Awaiting clusters & transitions...', w / 2, h / 2 + 4);
        return;
      }

      // Layout Margins
      const ml = isMini ? 28 : 36; // Left margin for row headers
      const mt = isMini ? 24 : 30; // Top margin for col headers
      const mb = isMini ? 8 : 12;  // Bottom margin

      // Colorbar space on the right
      const cbarW = isMini ? 10 : 13;
      const cbarGap = isMini ? 10 : 14;
      const cbarLblW = isMini ? 28 : 34;
      const mr = cbarGap + cbarW + cbarLblW + 6;

      // Available space for the square heatmap grid
      const maxGridW = Math.max(20, w - ml - mr);
      const maxGridH = Math.max(20, h - mt - mb);

      // Strict 1:1 Aspect Ratio: S x S square grid
      const S = Math.min(maxGridW, maxGridH);
      const cw = S / K;
      const ch = S / K;

      // Centering offset for 1:1 grid within available space
      const gridX = ml + Math.max(0, Math.floor((maxGridW - S) / 2));
      const gridY = mt + Math.max(0, Math.floor((maxGridH - S) / 2));

      // Row totals with cached TypedArray reuse
      if (!cachedTMRowTotals || cachedTMRowTotals.length < K) {
        cachedTMRowTotals = new Float64Array(Math.max(64, K * 2));
      }
      let totalTransCount = 0;
      let nonZeroCount = 0;

      for (let i = 0; i < K; i++) {
        let sum = 0;
        if (transitionCounts[i]) {
          for (let j = 0; j < K; j++) {
            const cnt = transitionCounts[i][j] || 0;
            sum += cnt;
            totalTransCount += cnt;
            if (cnt > 0) nonZeroCount++;
          }
        }
        cachedTMRowTotals[i] = sum;
      }

      // Helper: Map probability [0, 1] to multi-hue RGBA color
      function getHeatmapColor(prob) {
        if (prob === 0) return '#0f172a';
        if (prob < 0.25) {
          const t = prob / 0.25;
          const r = Math.round(15 + t * 124);
          const g = Math.round(23 + t * 69);
          const b = Math.round(42 + t * 204);
          return `rgba(${r},${g},${b},${(0.30 + t * 0.25).toFixed(2)})`;
        } else if (prob < 0.60) {
          const t = (prob - 0.25) / 0.35;
          const r = Math.round(139 - t * 83);
          const g = Math.round(92 + t * 97);
          return `rgba(${r},${g},248,${(0.55 + t * 0.20).toFixed(2)})`;
        } else {
          const t = (prob - 0.60) / 0.40;
          const r = Math.round(56 + t * 189);
          const g = Math.round(189 - t * 31);
          const b = Math.round(248 - t * 237);
          return `rgba(${r},${g},${b},${(0.75 + t * 0.23).toFixed(2)})`;
        }
      }

      // 1. Draw 1:1 Aspect Ratio Grid Cells
      const drawText = (!isRunning && !isMini && cw >= 28 && ch >= 18 && K <= 16);
      if (drawText) {
        ctx.font = `bold ${Math.min(10, Math.floor(ch * 0.55))}px monospace`;
        ctx.textAlign = 'center';
      }

      for (let i = 0; i < K; i++) {
        const sum = cachedTMRowTotals[i];
        for (let j = 0; j < K; j++) {
          const count = (transitionCounts[i] && transitionCounts[i][j]) ? transitionCounts[i][j] : 0;
          const prob = sum > 0 ? (count / sum) : 0.0;
          const x = gridX + j * cw;
          const y = gridY + i * ch;

          ctx.fillStyle = getHeatmapColor(prob);
          ctx.fillRect(x, y, cw, ch);

          // Cell Border
          ctx.strokeStyle = 'rgba(51, 65, 85, 0.45)';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x, y, cw, ch);

          // Value inside cell (only when paused/stepping with small K)
          if (drawText && prob > 0) {
            ctx.fillStyle = prob > 0.4 ? '#ffffff' : '#cbd5e1';
            ctx.fillText(`${(prob * 100).toFixed(0)}%`, x + cw / 2, y + ch / 2 + 3);
          }

          // Active Last Transition Glow (Golden Outline)
          if (lastTransitionFrom === i && lastTransitionTo === j) {
            ctx.save();
            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = Math.min(3.0, Math.max(1.5, cw * 0.15));
            ctx.shadowColor = '#fbbf24';
            ctx.shadowBlur = 8;
            ctx.strokeRect(x + 1, y + 1, cw - 2, ch - 2);
            ctx.restore();
          }

          // Hover Highlight (White Outline)
          if (hoveredTMCell && hoveredTMCell.i === i && hoveredTMCell.j === j) {
            ctx.save();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2.0;
            ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
            ctx.restore();
          }
        }
      }

      // Outer Heatmap Border
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
      ctx.lineWidth = 1;
      ctx.strokeRect(gridX, gridY, S, S);

      // 2. Draw Column Headers (Target Clusters c_t)
      for (let j = 0; j < K; j++) {
        const x = gridX + j * cw + cw / 2;
        const color = getClusterColor(j);
        
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, gridY - (isMini ? 8 : 11), isMini ? 2.5 : 3.5, 0, Math.PI * 2);
        ctx.fill();

        if (!isMini || K <= 10) {
          ctx.fillStyle = '#94a3b8';
          ctx.font = `${isMini ? 7 : 8}px monospace`;
          ctx.textAlign = 'center';
          ctx.fillText(`c${j}`, x, gridY - (isMini ? 2 : 3));
        }
      }

      // 3. Draw Row Headers (Source Clusters c_{t-1})
      for (let i = 0; i < K; i++) {
        const y = gridY + i * ch + ch / 2;
        const color = getClusterColor(i);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(gridX - (isMini ? 8 : 12), y, isMini ? 2.5 : 3.5, 0, Math.PI * 2);
        ctx.fill();

        if (!isMini || K <= 10) {
          ctx.fillStyle = '#94a3b8';
          ctx.font = `${isMini ? 7 : 8}px monospace`;
          ctx.textAlign = 'right';
          ctx.fillText(`c${i}`, gridX - (isMini ? 13 : 18), y + 3);
        }
      }

      // 4. Axis Labels
      ctx.fillStyle = '#64748b';
      ctx.font = '7px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('c(t-1)↓', Math.max(2, gridX - ml), gridY - 2);
      ctx.textAlign = 'center';
      ctx.fillText('Target c(t) →', gridX + S / 2, Math.max(8, gridY - (isMini ? 14 : 18)));

      // 5. Draw Colorbar Next to Heatmap
      const cbarX = gridX + S + cbarGap;
      const cbarY = gridY;
      const cbarH = S;

      const cbarGrad = ctx.createLinearGradient(0, cbarY + cbarH, 0, cbarY);
      cbarGrad.addColorStop(0.00, '#0f172a');
      cbarGrad.addColorStop(0.25, 'rgba(139, 92, 246, 0.50)');
      cbarGrad.addColorStop(0.50, 'rgba(56, 189, 248, 0.70)');
      cbarGrad.addColorStop(0.75, 'rgba(16, 185, 129, 0.85)');
      cbarGrad.addColorStop(1.00, 'rgba(245, 158, 11, 0.95)');
      ctx.fillStyle = cbarGrad;
      ctx.fillRect(cbarX, cbarY, cbarW, cbarH);

      ctx.strokeStyle = 'rgba(71, 85, 105, 0.7)';
      ctx.lineWidth = 0.8;
      ctx.strokeRect(cbarX, cbarY, cbarW, cbarH);

      // Colorbar Ticks and Labels
      const ticks = isMini ? [0.0, 0.5, 1.0] : [0.0, 0.25, 0.5, 0.75, 1.0];
      ticks.forEach(t => {
        const ty = cbarY + cbarH - t * cbarH;
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(cbarX + cbarW, ty);
        ctx.lineTo(cbarX + cbarW + 3, ty);
        ctx.stroke();

        ctx.fillStyle = '#94a3b8';
        ctx.font = `${isMini ? 7 : 8}px monospace`;
        ctx.textAlign = 'left';
        ctx.fillText(`${(t * 100).toFixed(0)}%`, cbarX + cbarW + 5, ty + 2.5);
      });

      // Colorbar Title
      ctx.fillStyle = '#c084fc';
      ctx.font = `bold ${isMini ? 7 : 8}px monospace`;
      ctx.textAlign = 'center';
      ctx.fillText('P(j|i)', cbarX + cbarW / 2, cbarY - 6);

      // Hover Pointer on Colorbar
      if (hoveredTMCell && hoveredTMCell.i >= 0 && hoveredTMCell.i < K && hoveredTMCell.j >= 0 && hoveredTMCell.j < K) {
        const hSum = cachedTMRowTotals[hoveredTMCell.i];
        const hCnt = (transitionCounts[hoveredTMCell.i] && transitionCounts[hoveredTMCell.i][hoveredTMCell.j]) || 0;
        const hProb = hSum > 0 ? (hCnt / hSum) : 0.0;
        const pointerY = cbarY + cbarH - hProb * cbarH;

        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.moveTo(cbarX - 1, pointerY);
        ctx.lineTo(cbarX - 5, pointerY - 3.5);
        ctx.lineTo(cbarX - 5, pointerY + 3.5);
        ctx.closePath();
        ctx.fill();
      }

      // 6. Update Status & Badges
      const densityPct = K > 0 ? ((nonZeroCount / (K * K)) * 100).toFixed(0) : '0';
      const badgeLastTrans = document.getElementById('badgeLastTrans');
      const lastStr = (lastTransitionFrom >= 0 && lastTransitionTo >= 0) 
        ? `Last: c${lastTransitionFrom} → c${lastTransitionTo}` 
        : `Last: None`;
      if (badgeLastTrans) badgeLastTrans.innerText = lastStr;

      const lblStats = document.getElementById('lblTMStats');
      if (lblStats) lblStats.innerText = `${totalTransCount} trans (${densityPct}% dense)`;

      // Save geometry for mouse events
      cvs._tmLayout = { gridX, gridY, S, cw, ch, K };
    }

    function updateTMTopPaths() {
      const cont = document.getElementById('tmTopPathsList');
      if (!cont) return;

      const K = clusters.length;
      if (K === 0 || transitionCounts.length === 0 || !topLearnedPathsCache || topLearnedPathsCache.length === 0) {
        cont.innerHTML = `<div style="color: var(--text-muted); text-align: center; padding: 6px 0;">Awaiting transitions...</div>`;
        return;
      }

      let html = '';
      topLearnedPathsCache.slice(0, 8).forEach(p => {
        const colFrom = getClusterColor(p.from);
        const colTo = getClusterColor(p.to);
        const isCurrent = (lastTransitionFrom === p.from && lastTransitionTo === p.to);
        html += `
          <div onmouseenter="setHoveredTMCell(${p.from}, ${p.to})" onmouseleave="setHoveredTMCell(-1, -1)" style="display: flex; justify-content: space-between; align-items: center; background: ${isCurrent ? 'rgba(251, 191, 36, 0.12)' : '#172033'}; border: 1px solid ${isCurrent ? '#fbbf24' : 'var(--card-border)'}; border-radius: 4px; padding: 3px 6px; cursor: pointer; transition: all 0.15s ease;">
            <span style="display: flex; align-items: center; gap: 4px;">
              <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: ${colFrom};"></span>
              <b>C${p.from}</b>
              <span style="color: var(--text-muted);">→</span>
              <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: ${colTo};"></span>
              <b>C${p.to}</b>
            </span>
            <span style="font-family: monospace; color: ${isCurrent ? '#fbbf24' : '#38bdf8'}; font-weight: 700;">
              ${p.count} (${(p.prob * 100).toFixed(1)}%)
            </span>
          </div>
        `;
      });
      cont.innerHTML = html;
    }

    function setHoveredTMCell(from, to) {
      if (from >= 0 && to >= 0) {
        hoveredTMCell = { i: from, j: to };
        const tipEl = document.getElementById('tmCellTooltip');
        if (tipEl && transitionCounts[from]) {
          let sum = 0;
          for (let k = 0; k < clusters.length; k++) sum += transitionCounts[from][k] || 0;
          const cnt = transitionCounts[from][to] || 0;
          const prob = sum > 0 ? (cnt / sum) : 0.0;
          tipEl.innerText = `C${from} → C${to}: Count=${cnt}, P=${(prob * 100).toFixed(1)}% (Row total: ${sum})`;
        }
      } else {
        hoveredTMCell = null;
        const tipEl = document.getElementById('tmCellTooltip');
        if (tipEl) tipEl.innerText = 'Hover over any cell to inspect transition details';
      }
      if (typeof currentActiveTab !== 'undefined' && currentActiveTab === 'tm') {
        drawTransitionMatrix('tmHeatmapCanvas', false);
      }
      draw();
    }

    // Multi-Tile 1D Subspace Engines
    let tileEngineX = { axisName: "X", clusters: [], dcc: [], totalEvals: 0, naiveEvals: 0 };
    let tileEngineY = { axisName: "Y", clusters: [], dcc: [], totalEvals: 0, naiveEvals: 0 };
    let tileEngineZ = { axisName: "Z", clusters: [], dcc: [], totalEvals: 0, naiveEvals: 0 };
    let jointTuplesMap = new Map();
    let currentJointTuple = null;
    let tileTraceX = null;
    let tileTraceY = null;
    let tileTraceZ = null;

    function getClusterColor(id) {
      const hues = [199, 142, 270, 38, 340, 180, 48, 220, 110, 300, 15, 160, 205, 80, 320];
      const hue = hues[id % hues.length];
      return `hsl(${hue}, 85%, 58%)`;
    }

    function showToast(msg) {
      const toast = document.getElementById('toastFeedback');
      if (toast) {
        toast.innerText = msg;
        toast.classList.add('show');
        clearTimeout(toast._timer);
        toast._timer = setTimeout(() => toast.classList.remove('show'), 2200);
      }
    }

    function logExplainStep(step) {
      if (isExplainMode) {
        currentExplanation.push(step);
      }
    }

    function countActive(mask, K) {
      let c = 0;
      const limit = typeof K === 'number' ? K : mask.length;
      for (let i = 0; i < limit; i++) if (mask[i]) c++;
      return c;
    }

    function getArgMaxP(probs, activeMask, K) {
      let maxP = -1;
      let argMax = -1;
      const limit = typeof K === 'number' ? K : probs.length;
      for (let i = 0; i < limit; i++) {
        if (activeMask[i] && probs[i] > maxP) {
          maxP = probs[i];
          argMax = i;
        }
      }
      return argMax;
    }

    // =========================================================================

//  7. UI UPDATES & TELEMETRY
    // =========================================================================

    function updateZoomBadge() {
      const badge = document.getElementById('zoomBadge');
      if (badge) {
        const activeIdx = (currentDim === 2) ? 2 : (maximizedQuad !== null ? maximizedQuad : 0);
        const view = quadViews[activeIdx] || quadViews[0];
        badge.innerText = `${Math.round((view.zoom || 1.0) * 100)}%`;
      }
    }

    function resetView() {
      maximizedQuad = null;
      quadViews.forEach(v => { v.panX = 0; v.panY = 0; v.zoom = 1.0; });
      orbitCamera.azimuth = -35 * (Math.PI / 180);
      orbitCamera.elevation = 25 * (Math.PI / 180);
      orbitCamera.panX = 0;
      orbitCamera.panY = 0;
      orbitCamera.zoom = 1.0;
      if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
      updateZoomBadge();
      draw();
    }

    function getActiveSampleTrace() {
      if (selectedSampleTraceIndex >= 0) {
        const found = sampleTraceLog.find(e => e.frameIndex === selectedSampleTraceIndex);
        if (found) return found;
      }
      if (sampleTraceLog.length > 0) {
        return sampleTraceLog[sampleTraceLog.length - 1];
      }
      return null;
    }

    function hoverSampleHistoryPoint(frameIndex) {
      const entry = sampleTraceLog.find(e => e.frameIndex === frameIndex);
      if (entry) {
        hoveredSampleTracePoint = {
          x: entry.point.x,
          y: entry.point.y,
          z: entry.point.z || 0.0,
          frameIndex: entry.frameIndex,
          clusterId: entry.assignedCluster
        };
        hoveredClusterId = entry.assignedCluster;
        draw();
      }
    }

    function clearSampleHistoryHover() {
      hoveredSampleTracePoint = null;
      hoveredClusterId = -1;
      draw();
    }

    function selectPastSample(frameIndex) {
      selectedSampleTraceIndex = frameIndex;
      const entry = sampleTraceLog.find(e => e.frameIndex === frameIndex);
      if (entry) {
        hoveredSampleTracePoint = {
          x: entry.point.x,
          y: entry.point.y,
          z: entry.point.z || 0.0,
          frameIndex: entry.frameIndex,
          clusterId: entry.assignedCluster
        };
      }
      updateUI();
      draw();
    }

    function returnToLiveStream() {
      selectedSampleTraceIndex = -1;
      hoveredSampleTracePoint = null;
      updateUI();
      draw();
    }

    function renderSampleHistoryUI() {
      const selectEl = document.getElementById('selectSampleHistory');
      const countEl = document.getElementById('lblSampleHistoryCount');
      const btnPrev = document.getElementById('btnPrevSample');
      const btnNext = document.getElementById('btnNextSample');
      const btnLive = document.getElementById('btnLiveSample');
      const stripEl = document.getElementById('sampleHistoryStrip');

      const totalLogged = sampleTraceLog.length;
      if (countEl) {
        countEl.innerText = `${totalLogged}/${totalFrames}`;
      }

      const isLive = (selectedSampleTraceIndex === -1);

      if (btnLive) {
        btnLive.style.opacity = isLive ? '1.0' : '0.6';
        btnLive.style.background = isLive ? 'rgba(74, 222, 128, 0.2)' : 'rgba(100, 116, 139, 0.2)';
        btnLive.style.borderColor = isLive ? 'rgba(74, 222, 128, 0.4)' : 'rgba(100, 116, 139, 0.3)';
        btnLive.style.color = isLive ? '#4ade80' : '#94a3b8';
      }

      // Populate Select Dropdown
      if (selectEl) {
        let activeVal = isLive ? -1 : selectedSampleTraceIndex;
        let html = `<option value="-1" ${isLive ? 'selected' : ''}>● Live: Point #${totalFrames || 0}</option>`;
        for (let i = sampleTraceLog.length - 1; i >= 0; i--) {
          const entry = sampleTraceLog[i];
          const isSel = (entry.frameIndex === selectedSampleTraceIndex);
          const clName = `C${entry.assignedCluster}`;
          html += `<option value="${entry.frameIndex}" ${isSel ? 'selected' : ''}>Point #${entry.frameIndex} (${clName}, ${entry.distSC}ev)</option>`;
        }
        selectEl.innerHTML = html;
        selectEl.value = String(activeVal);
      }

      // Button states
      let currentPos = -1;
      if (!isLive) {
        currentPos = sampleTraceLog.findIndex(e => e.frameIndex === selectedSampleTraceIndex);
      } else {
        currentPos = sampleTraceLog.length - 1;
      }

      if (btnPrev) btnPrev.disabled = (totalLogged === 0 || currentPos <= 0);
      if (btnNext) btnNext.disabled = (totalLogged === 0 || isLive || currentPos >= totalLogged - 1);

      // Render timeline strip
      if (stripEl) {
        if (sampleTraceLog.length === 0) {
          stripEl.innerHTML = `<span style="color: var(--text-muted); font-size: 0.65rem; padding: 2px 4px;">No samples logged yet. Click Step or Play to ingest frames.</span>`;
        } else {
          const recentToShow = sampleTraceLog.slice(-25); // show last 25 points in strip
          stripEl.innerHTML = recentToShow.map(entry => {
            const isSel = (!isLive && entry.frameIndex === selectedSampleTraceIndex) || (isLive && entry.frameIndex === totalFrames);
            const clColor = (clusters[entry.assignedCluster] && clusters[entry.assignedCluster].color) || '#38bdf8';
            const bg = isSel ? 'rgba(250, 204, 21, 0.25)' : '#172033';
            const border = isSel ? '#facc15' : 'rgba(51, 65, 85, 0.6)';
            return `
              <div class="sample-chip ${isSel ? 'active' : ''}"
                   style="display: inline-flex; align-items: center; gap: 3px; background: ${bg}; border: 1px solid ${border}; border-radius: 4px; padding: 1px 5px; font-size: 0.65rem; font-family: monospace; cursor: pointer; white-space: nowrap; transition: all 0.15s ease;"
                   onmouseenter="hoverSampleHistoryPoint(${entry.frameIndex})"
                   onmouseleave="clearSampleHistoryHover()"
                   onclick="selectPastSample(${entry.frameIndex})"
                   data-tooltip-title="Sample #${entry.frameIndex}"
                   data-tooltip-badge="Assigned C${entry.assignedCluster}"
                   data-tooltip-color="cyan"
                   data-tooltip-desc="Coordinates: (${entry.point.x.toFixed(2)}, ${entry.point.y.toFixed(2)}${currentDim === 3 ? `, ${entry.point.z.toFixed(2)}` : ''}). Distance Evaluations: ${entry.distSC}. Click to inspect decision trace.">
                <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: ${clColor};"></span>
                <span style="color: ${isSel ? '#facc15' : '#f8fafc'}; font-weight: ${isSel ? '700' : 'normal'};">#${entry.frameIndex}</span>
              </div>
            `;
          }).join('');
        }
      }
    }

    function renderEntropyTrace() {
      const listEl = document.getElementById('entropyRankingsList');
      const powerListEl = document.getElementById('entropyClusterPowerList');
      const badgeStatus = document.getElementById('badgeEntropyStatus');

      const activeTrace = getActiveSampleTrace();
      const rankings = (activeTrace && activeTrace.entropyRankings && activeTrace.entropyRankings.length > 0)
        ? activeTrace.entropyRankings
        : lastEntropyRankings;

      if (badgeStatus) {
        if (selectedSampleTraceIndex >= 0) {
          badgeStatus.innerText = `Sample #${selectedSampleTraceIndex}`;
        } else {
          badgeStatus.innerText = targetMode === 'entropy' ? (entropyFastMode ? '-entropy_fast' : '-entropy') : 'Greedy Mode';
        }
      }

      if (listEl) {
        if (!rankings || rankings.length === 0) {
          listEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.76rem; text-align: center; padding: 12px 0;">
            ${targetMode === 'entropy' ? 'Step through or pause simulation to inspect candidate rankings.' : 'Switch Target Mode to <b>Entropy</b> to inspect target candidate rankings.'}
          </div>`;
        } else {
          let maxH = 0.001;
          rankings.forEach(r => { if (r.expectedH > maxH) maxH = r.expectedH; });
          listEl.innerHTML = rankings.slice(0, 10).map(r => {
            const barW = Math.min(100, Math.max(5, (r.expectedH / maxH) * 100));
            const barColor = r.isChosen ? '#4ade80' : '#38bdf8';
            const isHovered = (hoveredClusterId === r.id);
            const isSelected = (selectedClusterId === r.id);
            const bg = isSelected 
              ? 'rgba(250, 204, 21, 0.15)' 
              : (isHovered ? 'rgba(56, 189, 248, 0.15)' : (r.isChosen ? 'rgba(74, 222, 128, 0.08)' : '#172033'));
            const border = isSelected 
              ? 'rgba(250, 204, 21, 0.6)' 
              : (isHovered ? 'rgba(56, 189, 248, 0.6)' : (r.isChosen ? 'rgba(74, 222, 128, 0.4)' : 'rgba(56, 189, 248, 0.2)'));

            return `
              <div class="entropy-rank-item ${isSelected ? 'selected' : ''} ${isHovered ? 'hovered' : ''}"
                   data-cluster-id="${r.id}"
                   style="background: ${bg}; border: 1px solid ${border}; border-radius: 4px; padding: 4px 6px; margin-bottom: 3px; cursor: pointer; transition: all 0.15s ease;"
                   onmouseenter="setHoveredCluster(${r.id})"
                   onmouseleave="setHoveredCluster(-1)"
                   onclick="toggleSelectCluster(${r.id})"
                   data-tooltip-title="Target Candidate C${r.id}"
                   data-tooltip-badge="${r.isChosen ? 'Selected Target (★)' : 'Candidate Anchor'}"
                   data-tooltip-color="${r.isChosen ? 'green' : 'cyan'}"
                   data-tooltip-desc="Prior probability: P=${r.p.toFixed(4)}. Expected residual entropy: E[H]=${r.expectedH.toFixed(3)} bits. Expected info gain: ΔH=+${r.infoGain.toFixed(3)} bits. Click to pin cluster highlight.">
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.72rem; margin-bottom: 2px;">
                  <span style="font-weight: 700; color: ${isSelected ? '#facc15' : (isHovered ? '#38bdf8' : (r.isChosen ? '#4ade80' : '#f8fafc'))};">
                    ${r.isChosen ? '★ ' : ''}Candidate C${r.id} <span style="color: var(--text-muted); font-size: 0.68rem;">(P=${r.p.toFixed(3)})</span>
                  </span>
                  <span style="font-family: monospace; font-size: 0.70rem; color: ${r.isChosen ? '#4ade80' : '#38bdf8'};">
                    ${r.isSupport ? `Support: ${r.expectedH.toFixed(1)} cl` : `E[H]=${r.expectedH.toFixed(2)} b`} <span style="color: #fbbf24;">(+${r.infoGain.toFixed(2)}b gain)</span>
                  </span>
                </div>
                <div style="height: 4px; width: 100%; background: #0f172a; border-radius: 2px; overflow: hidden;">
                  <div style="height: 100%; width: ${barW}%; background: ${barColor};"></div>
                </div>
              </div>
            `;
          }).join('');
        }
      }

      if (powerListEl) {
        if (clusters.length === 0) {
          powerListEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.76rem; text-align: center; padding: 12px 0;">No active clusters.</div>`;
        } else {
          const sortedCl = [...clusters].filter(c => c && c.infoGain !== undefined).sort((a, b) => (b.infoGain || 0) - (a.infoGain || 0));
          if (sortedCl.length === 0) {
            powerListEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.76rem; text-align: center; padding: 12px 0;">Run simulation in Entropy mode to compute anchor information gain.</div>`;
          } else {
            powerListEl.innerHTML = sortedCl.slice(0, 8).map(c => {
              const isHovered = (hoveredClusterId === c.id);
              const isSelected = (selectedClusterId === c.id);
              const bg = isSelected 
                ? 'rgba(250, 204, 21, 0.15)' 
                : (isHovered ? 'rgba(56, 189, 248, 0.15)' : '#172033');
              const border = isSelected 
                ? 'rgba(250, 204, 21, 0.6)' 
                : (isHovered ? 'rgba(56, 189, 248, 0.6)' : 'var(--card-border)');

              return `
                <div class="entropy-power-item ${isSelected ? 'selected' : ''} ${isHovered ? 'hovered' : ''}"
                     data-cluster-id="${c.id}"
                     style="display: flex; justify-content: space-between; align-items: center; background: ${bg}; border: 1px solid ${border}; border-radius: 4px; padding: 4px 6px; font-size: 0.71rem; cursor: pointer; transition: all 0.15s ease;"
                     onmouseenter="setHoveredCluster(${c.id})"
                     onmouseleave="setHoveredCluster(-1)"
                     onclick="toggleSelectCluster(${c.id})"
                     data-tooltip-title="Cluster C${c.id} Pruning Power"
                     data-tooltip-badge="${c.members} Members"
                     data-tooltip-color="cyan"
                     data-tooltip-desc="Information reduction power: ΔH=+${(c.infoGain || 0).toFixed(3)} bits. Centroid: (${c.x.toFixed(2)}, ${c.y.toFixed(2)}${currentDim === 3 ? `, ${c.z.toFixed(2)}` : ''}). Click to pin highlight.">
                  <span style="color: ${isSelected ? '#facc15' : (isHovered ? '#38bdf8' : c.color)}; font-weight: 700;">
                    Cluster C${c.id} <span style="color: var(--text-muted); font-size: 0.67rem;">(${c.members} members)</span>
                  </span>
                  <span style="font-family: monospace; color: #38bdf8; font-weight: 600;">ΔH = +${(c.infoGain || 0).toFixed(2)} bits</span>
                </div>
              `;
            }).join('');
          }
        }
      }
    }

    function setTab(tab) {
      currentActiveTab = tab;
      if (tab === 'narrative' || tab === 'tm' || tab === 'entropy') {
        setHoveredCluster(-1);
      }
      const tabNarrative = document.getElementById('tabNarrative');
      const tabCandidates = document.getElementById('tabCandidates');
      const tabTM = document.getElementById('tabTM');
      const tabEntropyTrace = document.getElementById('tabEntropyTrace');
      const contNarrative = document.getElementById('narrativeContainer');
      const contCandidates = document.getElementById('candidateContainer');
      const contTM = document.getElementById('tmContainer');
      const contEntropyTrace = document.getElementById('entropyTraceContainer');

      [tabNarrative, tabCandidates, tabTM, tabEntropyTrace].forEach(t => t && t.classList.remove('active'));
      if (contNarrative) contNarrative.style.display = 'none';
      if (contCandidates) contCandidates.style.display = 'none';
      if (contTM) contTM.style.display = 'none';
      if (contEntropyTrace) contEntropyTrace.style.display = 'none';

      if (tab === 'narrative') {
        if (tabNarrative) tabNarrative.classList.add('active');
        if (contNarrative) contNarrative.style.display = 'flex';
      } else if (tab === 'candidates') {
        if (tabCandidates) tabCandidates.classList.add('active');
        if (contCandidates) contCandidates.style.display = 'flex';
      } else if (tab === 'tm') {
        if (tabTM) tabTM.classList.add('active');
        if (contTM) contTM.style.display = 'flex';
        updateTMCanvasDimensions();
        computeTopLearnedPaths();
        drawTransitionMatrix('tmHeatmapCanvas', false);
        updateTMTopPaths();
      } else if (tab === 'entropy') {
        if (tabEntropyTrace) tabEntropyTrace.classList.add('active');
        if (contEntropyTrace) contEntropyTrace.style.display = 'flex';
        renderEntropyTrace();
      }
    }

    function renderKnnTrace() {
      const tbody = document.getElementById('knnNeighborsBody');
      const effVal = document.getElementById('knnPruneEffVal');
      const speedupVal = document.getElementById('knnSpeedupVal');
      const distVal = document.getElementById('knnDistCallsVal');
      const distPerQVal = document.getElementById('knnDistPerQueryVal');
      const latVal = document.getElementById('knnLatencyVal');
      const totalTimeVal = document.getElementById('knnTotalTimeVal');
      const qpsVal = document.getElementById('knnQpsVal');
      const queriesVal = document.getElementById('knnQueriesVal');
      const kParamVal = document.getElementById('knnKParamVal');
      const speedBadge = document.getElementById('knnSpeedBadge');
      const statusBadge = document.getElementById('knnStatusBadge');
      const titleEl = document.getElementById('knnNeighborsTitle');
      const kBadge = document.getElementById('knnKBadge');

      // Pruning tab elements
      const l1Val = document.getElementById('knnL1PrunedVal');
      const l1PctVal = document.getElementById('knnL1PctVal');
      const l2Val = document.getElementById('knnL2PrunedVal');
      const l2PctVal = document.getElementById('knnL2PctVal');
      const l3Val = document.getElementById('knnL3PrunedVal');
      const l3PctVal = document.getElementById('knnL3PctVal');
      const tempVal = document.getElementById('knnTemporalPrunedVal');

      // Progress / Stacked bars
      const pruneSumTxt = document.getElementById('knnPruneSummaryTxt');
      const barPruned = document.getElementById('barKnnPruned');
      const barEval = document.getElementById('barKnnEvaluated');
      const lblPrunedCount = document.getElementById('lblKnnPrunedCount');
      const lblEvalCount = document.getElementById('lblKnnEvalCount');

      const barL1 = document.getElementById('barKnnL1');
      const barL2 = document.getElementById('barKnnL2');
      const barL3 = document.getElementById('barKnnL3');
      const barTemp = document.getElementById('barKnnTemp');
      const barExact = document.getElementById('barKnnExact');

      // Memory tab elements
      const memAnchorsEl = document.getElementById('knnMemAnchorsVal');
      const memClustersCountEl = document.getElementById('knnMemClustersCount');
      const memDccEl = document.getElementById('knnMemDccVal');
      const memMembersEl = document.getElementById('knnMemMembersVal');
      const memResultsEl = document.getElementById('knnMemResultsVal');
      const memTotalEl = document.getElementById('knnMemTotalVal');
      const memPerFrameEl = document.getElementById('knnMemPerFrameVal');

      // Scaling tab elements
      const evalRateEl = document.getElementById('knnEvalRateVal');
      const perQueryLatEl = document.getElementById('knnPerQueryLatVal');
      const timingComputeEl = document.getElementById('knnTimingCompute');
      const timingIoEl = document.getElementById('knnTimingIo');
      const timingTotalEl = document.getElementById('knnTimingTotal');
      const diagBruteEl = document.getElementById('knnDiagBruteOps');
      const diagMetricEl = document.getElementById('knnDiagMetricOps');
      const diagEpsEl = document.getElementById('knnDiagEpsilon');
      const diagTempEl = document.getElementById('knnDiagTemporal');

      if (!knnResults || !knnResults.telemetry) {
        if (tbody) {
          tbody.innerHTML = '<tr><td colspan="5" style="padding: 12px; text-align: center; color: var(--text-muted);">Enable "Run gric-knn" and run/step simulation to compute k-NN graph</td></tr>';
        }
        if (effVal) effVal.textContent = '--%';
        if (speedupVal) speedupVal.textContent = '--x';
        if (distVal) distVal.textContent = '0';
        if (distPerQVal) distPerQVal.textContent = '0/q';
        if (latVal) latVal.textContent = '0.00 ms';
        if (totalTimeVal) totalTimeVal.textContent = '0.00 ms tot';
        if (qpsVal) qpsVal.textContent = '0 QPS';
        if (queriesVal) queriesVal.textContent = '0';
        if (kParamVal) kParamVal.textContent = `k=${knnK || 10}`;
        if (speedBadge) speedBadge.textContent = '0 FPS';
        if (statusBadge) {
          statusBadge.textContent = 'Ready';
          statusBadge.style.color = 'var(--text-muted)';
        }
        if (l1Val) l1Val.textContent = '0';
        if (l1PctVal) l1PctVal.textContent = '0%';
        if (l2Val) l2Val.textContent = '0';
        if (l2PctVal) l2PctVal.textContent = '0%';
        if (l3Val) l3Val.textContent = '0';
        if (l3PctVal) l3PctVal.textContent = '0%';
        if (tempVal) tempVal.textContent = '0';

        if (pruneSumTxt) pruneSumTxt.textContent = '0% Pruned';
        if (barPruned) barPruned.style.width = '0%';
        if (barEval) barEval.style.width = '100%';
        if (lblPrunedCount) lblPrunedCount.textContent = '0';
        if (lblEvalCount) lblEvalCount.textContent = '0';

        if (barL1) barL1.style.width = '0%';
        if (barL2) barL2.style.width = '0%';
        if (barL3) barL3.style.width = '0%';
        if (barTemp) barTemp.style.width = '0%';
        if (barExact) barExact.style.width = '100%';

        if (memAnchorsEl) memAnchorsEl.textContent = '0 B';
        if (memClustersCountEl) memClustersCountEl.textContent = '0 cl';
        if (memDccEl) memDccEl.textContent = '0 B';
        if (memMembersEl) memMembersEl.textContent = '0 B';
        if (memResultsEl) memResultsEl.textContent = '0 B';
        if (memTotalEl) memTotalEl.textContent = '0 KB';
        if (memPerFrameEl) memPerFrameEl.textContent = '0 bytes / frame';

        if (evalRateEl) evalRateEl.textContent = '0 M/s';
        if (perQueryLatEl) perQueryLatEl.textContent = '0.0 µs';
        if (timingComputeEl) timingComputeEl.textContent = '0.00 ms';
        if (timingIoEl) timingIoEl.textContent = '0.00 ms';
        if (timingTotalEl) timingTotalEl.textContent = '0.00 ms';
        if (diagBruteEl) diagBruteEl.textContent = '0';
        if (diagMetricEl) diagMetricEl.textContent = '0';
        if (diagEpsEl) diagEpsEl.textContent = 'ε = 0.00 (Exact)';
        if (diagTempEl) diagTempEl.textContent = `|i-j| ≥ ${knnDtmin || 1}`;
        return;
      }

      const telem = knnResults.telemetry;
      const N = knnResults.totalFrames;
      const k = knnResults.k;
      const bruteForce = N * N;
      const dists = telem.framedistCalls;
      const pruned = Math.max(0, bruteForce - dists);
      const pruneEffNum = bruteForce > 0 ? (1.0 - dists / bruteForce) * 100.0 : 100.0;
      const pruneEff = pruneEffNum.toFixed(3);
      const speedup = dists > 0 ? (bruteForce / dists).toFixed(1) : '∞';
      const timeCompute = (typeof telem.timeComputeMs === 'number')
        ? telem.timeComputeMs
        : (telem.timeSearchMs || 0.0);
      const timeTotal = (typeof telem.timeTotalMs === 'number')
        ? telem.timeTotalMs
        : timeCompute;
      const timeIo = (typeof telem.timeIoMs === 'number')
        ? telem.timeIoMs
        : Math.max(0, timeTotal - timeCompute);

      const fps = timeCompute > 0 ? Math.round(N / (timeCompute / 1000.0)).toLocaleString() : '∞';
      const distsPerQuery = N > 0 ? (dists / N).toFixed(1) : '0';

      // 1. Overview Tab
      if (effVal) {
        effVal.textContent = `${pruneEff}%`;
      }
      if (speedupVal) speedupVal.textContent = `${speedup}x`;
      if (distVal) distVal.textContent = `${dists.toLocaleString()}`;
      if (distPerQVal) distPerQVal.textContent = `${distsPerQuery}/q`;
      if (latVal) latVal.textContent = `${timeCompute.toFixed(2)} ms`;
      if (totalTimeVal) totalTimeVal.textContent = `${timeTotal.toFixed(2)} ms tot`;
      if (qpsVal) qpsVal.textContent = `${fps} QPS`;
      if (queriesVal) queriesVal.textContent = `${N.toLocaleString()}`;
      if (kParamVal) kParamVal.textContent = `k=${k}`;
      if (speedBadge) speedBadge.textContent = `${fps} QPS`;
      if (statusBadge) {
        statusBadge.textContent = 'Active';
        statusBadge.style.color = '#4ade80';
      }

      // Funnel bar
      if (pruneSumTxt) pruneSumTxt.textContent = `${pruneEff}% Pruned`;
      if (barPruned) barPruned.style.width = `${Math.min(100, Math.max(0, pruneEffNum))}%`;
      if (barEval) barEval.style.width = `${Math.min(100, Math.max(0, 100 - pruneEffNum))}%`;
      if (lblPrunedCount) lblPrunedCount.textContent = pruned.toLocaleString();
      if (lblEvalCount) lblEvalCount.textContent = dists.toLocaleString();

      // 2. Pruning Tab
      const l1 = telem.level1ClustersPruned || 0;
      const l2 = telem.level2AnchorsPruned || 0;
      const l3 = telem.level3AnnularPruned || 0;
      const temp = telem.temporalPruned || 0;

      if (l1Val) l1Val.textContent = l1.toLocaleString();
      if (l1PctVal) l1PctVal.textContent = `${bruteForce > 0 ? ((l1 / bruteForce) * 100).toFixed(3) : 0}%`;
      if (l2Val) l2Val.textContent = l2.toLocaleString();
      if (l2PctVal) l2PctVal.textContent = `${bruteForce > 0 ? ((l2 / bruteForce) * 100).toFixed(3) : 0}%`;
      if (l3Val) l3Val.textContent = l3.toLocaleString();
      if (l3PctVal) l3PctVal.textContent = `${bruteForce > 0 ? ((l3 / bruteForce) * 100).toFixed(3) : 0}%`;
      if (tempVal) tempVal.textContent = temp.toLocaleString();

      const totalHierarchy = Math.max(1, l1 + l2 + l3 + temp + dists);
      if (barL1) barL1.style.width = `${((l1 / totalHierarchy) * 100).toFixed(1)}%`;
      if (barL2) barL2.style.width = `${((l2 / totalHierarchy) * 100).toFixed(1)}%`;
      if (barL3) barL3.style.width = `${((l3 / totalHierarchy) * 100).toFixed(1)}%`;
      if (barTemp) barTemp.style.width = `${((temp / totalHierarchy) * 100).toFixed(1)}%`;
      if (barExact) barExact.style.width = `${((dists / totalHierarchy) * 100).toFixed(1)}%`;

      // 3. Memory Tab
      const M = (typeof clusters !== 'undefined' && clusters) ? clusters.length : 0;
      const D = (typeof currentDim !== 'undefined') ? currentDim : 2;
      const memAnchors = M * D * 8;
      const memDcc = M * M * 8;
      const memMembers = N * 8;
      const memResults = N * k * 12;
      const memTotal = memAnchors + memDcc + memMembers + memResults;

      if (memAnchorsEl) memAnchorsEl.textContent = formatBytes(memAnchors);
      if (memClustersCountEl) memClustersCountEl.textContent = `${M} cl`;
      if (memDccEl) memDccEl.textContent = formatBytes(memDcc);
      if (memMembersEl) memMembersEl.textContent = formatBytes(memMembers);
      if (memResultsEl) memResultsEl.textContent = formatBytes(memResults);
      if (memTotalEl) memTotalEl.textContent = formatBytes(memTotal);
      if (memPerFrameEl) memPerFrameEl.textContent = `${(memTotal / Math.max(1, N)).toFixed(1)} bytes / frame`;

      // 4. Scaling Tab
      const evalRate = timeCompute > 0
        ? (dists / (timeCompute * 1000.0)).toFixed(2)
        : '0.00';
      const perQueryLat = (N > 0 && timeCompute > 0)
        ? ((timeCompute * 1000.0) / N).toFixed(1)
        : '0.0';

      if (evalRateEl) evalRateEl.textContent = `${evalRate} M/s`;
      if (perQueryLatEl) perQueryLatEl.textContent = `${perQueryLat} µs`;
      if (timingComputeEl) timingComputeEl.textContent = `${timeCompute.toFixed(2)} ms`;
      if (timingIoEl) timingIoEl.textContent = `${timeIo.toFixed(2)} ms`;
      if (timingTotalEl) timingTotalEl.textContent = `${timeTotal.toFixed(2)} ms`;
      if (diagBruteEl) diagBruteEl.textContent = `${bruteForce.toLocaleString()} evals`;
      if (diagMetricEl) diagMetricEl.textContent = `${dists.toLocaleString()} evals`;
      if (diagEpsEl) {
        diagEpsEl.textContent = (typeof knnEpsilon !== 'undefined' && knnEpsilon > 0)
          ? `ε = ${knnEpsilon.toFixed(2)} (${(knnEpsilon * 100).toFixed(0)}% ANN)`
          : 'ε = 0.00 (Exact Search)';
      }
      if (diagTempEl) {
        const dir = (typeof knnDirection !== 'undefined') ? knnDirection : 'all';
        diagTempEl.textContent = `|i-j| ≥ ${typeof knnDtmin !== 'undefined' ? knnDtmin : 1} (${dir})`;
      }

      if (kBadge) kBadge.textContent = `k = ${k}`;

      // Selected query sample
      let qId = selectedKnnQuerySample;
      if (qId < 0 || qId >= N) {
        qId = (currentFrameIdx > 0 && currentFrameIdx <= N) ? currentFrameIdx - 1 : 0;
      }
      if (titleEl) {
        titleEl.textContent = `Top-${k} Nearest Neighbors (Query Sample #${qId})`;
      }

      if (!tbody) return;

      let html = '';
      for (let r = 0; r < k; r++) {
        const nId = knnResults.indices[qId * k + r];
        const dist = knnResults.distances[qId * k + r];
        if (nId < 0) continue;

        const dt = nId - qId;
        const dtStr = dt > 0 ? `+${dt}` : `${dt}`;
        let cId = '-';
        if (typeof clustersAssigned !== 'undefined' && clustersAssigned && nId < clustersAssigned.length) {
          cId = `C${clustersAssigned[nId]}`;
        }

        const isHovered = (hoveredKnnNeighborId === nId);
        const bgStyle = isHovered ? 'background: rgba(56, 189, 248, 0.2);' : (r % 2 === 1 ? 'background: rgba(15, 23, 42, 0.4);' : '');

        html += `
          <tr style="${bgStyle} cursor: pointer; transition: background 0.15s;"
              onmouseenter="hoveredKnnNeighborId=${nId}; draw();"
              onmouseleave="hoveredKnnNeighborId=-1; draw();"
              onclick="selectedKnnQuerySample=${nId}; renderKnnTrace(); draw();">
            <td style="padding: 4px 6px; font-weight: 700; color: ${r === 0 ? '#4ade80' : '#94a3b8'};">#${r + 1}</td>
            <td style="padding: 4px 6px; color: #38bdf8; font-weight: 600;">#${nId}</td>
            <td style="padding: 4px 6px; color: #f8fafc;">${dist.toFixed(4)}</td>
            <td style="padding: 4px 6px; color: var(--accent-purple);">${cId}</td>
            <td style="padding: 4px 6px; color: ${dt < 0 ? '#fbbf24' : '#a78bfa'}; font-family: monospace;">${dtStr}</td>
          </tr>
        `;
      }

      tbody.innerHTML = html || '<tr><td colspan="5" style="padding: 8px; text-align: center; color: var(--text-muted);">No neighbors found</td></tr>';
    }

    window.renderKnnTrace = renderKnnTrace;

    function updateUI() {
      if (window.syncControlDependencies) {
        window.syncControlDependencies();
      }
      if (window.updateDisplayTogglesUI) {
        window.updateDisplayTogglesUI();
      }
      if (window.renderDataStructuresUI) {
        window.renderDataStructuresUI();
      }

      const presetBar = document.getElementById('viewPresetBar');
      if (presetBar) {
        presetBar.style.display =
          (typeof dataMode !== 'undefined' && dataMode === 'image') || currentDim !== 3
            ? 'none'
            : 'flex';
      }

      const imgScrubberBar = document.getElementById('imageScrubberBar');
      if (imgScrubberBar) {
        const isImg = (typeof dataMode !== 'undefined' && dataMode === 'image');
        imgScrubberBar.style.display = isImg ? 'flex' : 'none';
        if (isImg) {
          const slider = document.getElementById('sliderImgFrame');
          const lbl = document.getElementById('lblImgFrameStatus');
          const btnLive = document.getElementById('btnImgLiveStream');
          const btnPrev = document.getElementById('btnImgPrevFrame');
          const btnNext = document.getElementById('btnImgNextFrame');

          const total = (benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset.length
            : totalFrames;
          const isInspecting = (typeof inspectedImageFrameIdx !== 'undefined' &&
            inspectedImageFrameIdx >= 0);
          const curVal = isInspecting ? inspectedImageFrameIdx : Math.max(0, totalFrames - 1);

          if (slider) {
            slider.min = 0;
            slider.max = Math.max(0, total - 1);
            slider.value = curVal;
            slider.disabled = (total === 0);
          }
          if (lbl) {
            if (total === 0) {
              lbl.innerText = 'No frames';
            } else if (isInspecting) {
              lbl.innerText = `Frame #${curVal + 1} / ${total} (Inspecting)`;
            } else {
              lbl.innerText = `Frame #${totalFrames} / ${total} (Live)`;
            }
          }
          if (btnLive) {
            btnLive.classList.toggle('active', !isInspecting);
          }
          if (btnPrev) btnPrev.disabled = (curVal <= 0 || total === 0);
          if (btnNext) btnNext.disabled = (curVal >= total - 1 || total === 0);
        }
      }

      const legendHint = document.getElementById('legendHint');
      if (legendHint) {
        if (typeof dataMode !== 'undefined' && dataMode === 'image') {
          legendHint.innerText =
            "[Image Mode: TL=Current Frame • TR=Anchor/Residual (Click ⇄) • BL=Cluster Members • BR=All Clusters]";
        } else {
          legendHint.innerText = currentDim === 3
            ? "[3D Mode: Drag Quad 4 to Orbit • Shift+Drag to Pan • Wheel to Zoom • ⛶ to Maximize]"
            : "[2D Mode: Drag to Pan • Scroll to Zoom • ＋ Add Point: Inject]";
        }
      }

      const legRlim = document.getElementById('legendRlim');
      if (legRlim) legRlim.style.display = (showCircleMembers || showCircleSCDists) ? 'none' : 'inline-flex';
      const legMem = document.getElementById('legendCircleMembers');
      if (legMem) legMem.style.display = showCircleMembers ? 'inline-flex' : 'none';
      const legSC = document.getElementById('legendCircleSCDists');
      if (legSC) legSC.style.display = showCircleSCDists ? 'inline-flex' : 'none';
      const legEntropy = document.getElementById('legendEntropyMap');
      if (legEntropy) legEntropy.style.display = showEntropyMap ? 'inline-flex' : 'none';

      const txtPointDensityMode = document.getElementById('txtPointDensityMode');
      if (txtPointDensityMode) {
        let isAnyTruncated = false;
        let maxVis = 0;
        let maxDrn = 0;
        if (typeof viewportPointStats !== 'undefined') {
          viewportPointStats.forEach(st => {
            if (st && st.truncated) isAnyTruncated = true;
            if (st && st.visible > maxVis) maxVis = st.visible;
            if (st && st.drawn > maxDrn) maxDrn = st.drawn;
          });
        }
        if (isAnyTruncated) {
          txtPointDensityMode.innerText =
            `⚠ Subsampled (Zoom in to reveal up to ${maxVis.toLocaleString()} pts)`;
          txtPointDensityMode.style.color = '#fbbf24';
        } else if (pastSamples.length > 0) {
          txtPointDensityMode.innerText =
            `✓ 100% Fidelity (${maxDrn.toLocaleString()} pts in FOV)`;
          txtPointDensityMode.style.color = '#4ade80';
        } else {
          txtPointDensityMode.innerText = `Adaptive FOV (Zoom in for detail)`;
          txtPointDensityMode.style.color = '#38bdf8';
        }
      }

      // 1. Refresh Resource Metrics, Heap Memory, and Latency Sparkline
      updateResourceMetrics();
      const memStats = getMemoryStats();
      drawSparkline();
      drawDistCurves();
      drawDistAvgCurves();

      // 2. Refresh Live Transition Matrix (-tm) or Entropy Trace
      if (useTM && typeof currentActiveTab !== 'undefined' && currentActiveTab === 'tm') {
        const now = performance.now();
        const shouldRenderTM = !isRunning || (totalFrames % 4 === 0) || (now - lastTMRenderTimestamp > 80);
        if (shouldRenderTM) {
          lastTMRenderTimestamp = now;
          drawTransitionMatrix('tmHeatmapCanvas', false);
          updateTMTopPaths();
        }
      }
      if (typeof currentActiveTab !== 'undefined' && currentActiveTab === 'entropy') {
        renderEntropyTrace();
      }

      // Top Badges & Clusters Count
      const clusterBadge = document.getElementById('clusterBadge');
      const statClusters = document.getElementById('statClusters');
      const lblStatClusters = document.getElementById('lblStatClusters');
      const statFrames = document.getElementById('statFrames');
      const statDistTotalClusters = document.getElementById('statDistTotalClusters');

      if (useTiles) {
        const tupleCount = jointTuplesMap.size;
        if (clusterBadge) clusterBadge.innerText = `${tupleCount} tuples`;
        if (statClusters) statClusters.innerText = `${tupleCount}`;
        if (statDistTotalClusters) statDistTotalClusters.innerText = `${tupleCount} joint tuples`;
        if (lblStatClusters) {
          lblStatClusters.innerText = currentDim === 3 
            ? `3D Tuples (Kx=${tileEngineX.clusters.length}, Ky=${tileEngineY.clusters.length}, Kz=${tileEngineZ.clusters.length})`
            : `2D Tuples (Kx=${tileEngineX.clusters.length}, Ky=${tileEngineY.clusters.length})`;
        }
      } else {
        const kCount = clusters.length;
        if (clusterBadge) clusterBadge.innerText = `${kCount} clusters`;
        if (statClusters) statClusters.innerText = `${kCount}`;
        if (statDistTotalClusters) statDistTotalClusters.innerText = `${kCount} clusters`;
        if (lblStatClusters) lblStatClusters.innerText = `Clusters Formed`;
      }

      if (statFrames) statFrames.innerText = formatNumber(totalFrames);

      const cpuBadge = document.getElementById('cpuBadge');
      if (cpuBadge) {
        cpuBadge.innerText = `CPU: ${currentCpuLoadPct.toFixed(1)}%`;
      }

      if (useTiles) {
        document.getElementById('measCountBadge').innerText = currentJointTuple 
          ? `Tuple: ${currentDim === 3 ? `(${currentJointTuple.cx},${currentJointTuple.cy},${currentJointTuple.cz})` : `(${currentJointTuple.cx},${currentJointTuple.cy})`}` 
          : `0 tuples`;
        document.getElementById('entropyBadge').innerText = `Tiles: ${currentDim} Subspaces`;
      } else {
        const evalsCount = (currentEvaluations && currentEvaluations.length > 0)
          ? currentEvaluations.length
          : (distSampleClusterLast || 0);
        document.getElementById('measCountBadge').innerText = `${evalsCount} evals`;
        document.getElementById('entropyBadge').innerText = `H: ${currentEntropyBits.toFixed(2)} bits`;
      }

      // 2. Resource Tracker - Tab 1: Overview
      let currentElapsedMs = sessionElapsedMs;
      let currentFramesClustered = totalFrames - sessionStartFrames;
      if (sessionIsActive && isRunning) {
        currentElapsedMs = Math.max(0.0001, performance.now() - sessionStartTime);
        currentFramesClustered = totalFrames - sessionStartFrames;
      }

      let currentAvgFps = sessionAvgFps;
      if (sessionIsActive && isRunning) {
        currentAvgFps = currentElapsedMs > 0.001
          ? (currentFramesClustered / (currentElapsedMs / 1000.0))
          : 0.0;
      } else if (currentAvgFps <= 0 && sessionElapsedMs > 0 && totalFrames > 0) {
        currentAvgFps = (totalFrames / (sessionElapsedMs / 1000.0));
      }

      const statTotalTime = document.getElementById('statTotalTime');
      if (statTotalTime) {
        statTotalTime.innerText = formatClockTime(currentElapsedMs);
      }

      // Update Mobile Live Telemetry HUD Overlay
      const hudSamples = document.getElementById('hudSamples');
      if (hudSamples) {
        hudSamples.innerText = formatNumber(totalFrames);
      }
      const hudClusters = document.getElementById('hudClusters');
      if (hudClusters) {
        const kCount = useTiles ? jointTuplesMap.size : clusters.length;
        hudClusters.innerText = formatNumber(kCount);
      }
      const hudTime = document.getElementById('hudTime');
      if (hudTime) {
        hudTime.innerText = formatClockTime(currentElapsedMs);
      }

      const statAvgFps = document.getElementById('statAvgFps');
      if (statAvgFps) {
        if (currentAvgFps >= 1000) {
          statAvgFps.innerText = `${formatNumber(Math.round(currentAvgFps))} fps`;
        } else if (currentAvgFps > 0) {
          statAvgFps.innerText = `${currentAvgFps.toFixed(1)} fps`;
        } else {
          statAvgFps.innerText = `0.0 fps`;
        }
      }

      const statCpuLoad = document.getElementById('statCpuLoad');
      if (statCpuLoad) statCpuLoad.innerText = `${currentCpuLoadPct.toFixed(1)}%`;
      const statComputeMs = document.getElementById('statComputeMs');
      if (statComputeMs) statComputeMs.innerText = `${avgComputeTimeMs.toFixed(2)} ms`;

      const statTotalDists = document.getElementById('statTotalDists');
      if (statTotalDists) statTotalDists.innerText = formatNumber(distSampleCluster + distClusterCluster);

      const statDistRatio = document.getElementById('statDistRatio');
      if (statDistRatio) statDistRatio.innerText = `${formatNumber(distSampleCluster)} / ${formatNumber(distClusterCluster)}`;
      const statDccPopBadge = document.getElementById('statDccPopBadge');
      if (statDccPopBadge) {
        const K = clusters.length;
        const maxPairs = dccPairsTotal || (K > 1 ? (K * (K - 1) / 2) : 0);
        statDccPopBadge.innerText = maxPairs > 0
          ? `[${formatNumber(dccPopulated)}/${formatNumber(maxPairs)} pop]`
          : `[${formatNumber(dccPopulated)} pop]`;
      }

      const statMemoryTotal = document.getElementById('statMemoryTotal');
      if (statMemoryTotal) statMemoryTotal.innerText = formatBytes(memStats.totalModelBytes);

      // Entropy Telemetry Stat Cards in Overview Panel
      const avgInfoGainRate = distSampleCluster > 0 ? (totalEntropyReducedBits / distSampleCluster) : 0.0;
      const avgInitialH = totalFrames > 0 ? (totalInitialEntropyBits / totalFrames) : 0.0;
      const totalEntropyGatedAndEval = totalEntropyGated + totalEntropyEvals;
      const gateRatioPct = totalEntropyGatedAndEval > 0 ? ((totalEntropyGated / totalEntropyGatedAndEval) * 100.0) : 0.0;

      const statInfoGainRate = document.getElementById('statInfoGainRate');
      if (statInfoGainRate) statInfoGainRate.innerText = `${avgInfoGainRate.toFixed(2)} b/ev`;
      const statLastInfoGain = document.getElementById('statLastInfoGain');
      if (statLastInfoGain) statLastInfoGain.innerText = lastInfoGainRate.toFixed(2);

      const statInitialH = document.getElementById('statInitialH');
      if (statInitialH) statInitialH.innerText = `${lastInitialEntropy.toFixed(2)} b`;
      const statAvgInitialH = document.getElementById('statAvgInitialH');
      if (statAvgInitialH) statAvgInitialH.innerText = avgInitialH.toFixed(2);

      const statTotalEntropy = document.getElementById('statTotalEntropy');
      if (statTotalEntropy) {
        if (totalEntropyReducedBits >= 1000) {
          statTotalEntropy.innerText = `${(totalEntropyReducedBits / 1000.0).toFixed(2)} kb`;
        } else {
          statTotalEntropy.innerText = `${totalEntropyReducedBits.toFixed(1)} b`;
        }
      }

      const statGateRatio = document.getElementById('statGateRatio');
      if (statGateRatio) statGateRatio.innerText = `${gateRatioPct.toFixed(1)}%`;
      const statGatedCount = document.getElementById('statGatedCount');
      if (statGatedCount) statGatedCount.innerText = formatNumber(totalEntropyGated);

      const sparklineVal = document.getElementById('sparklineVal');
      if (sparklineVal) sparklineVal.innerText = `${lastComputeTimeMs.toFixed(2)} ms`;

      // 3. Resource Tracker - Tab 2: Distance Breakdown
      const avgDFC = totalFrames > 0 ? (distSampleCluster / totalFrames) : 0.0;
      const avgDCC = totalFrames > 0 ? (distClusterCluster / totalFrames) : 0.0;
      const avgTotal = avgDFC + avgDCC;

      const statLiveAvgDFC = document.getElementById('statLiveAvgDFC');
      if (statLiveAvgDFC) statLiveAvgDFC.innerText = avgDFC.toFixed(2);
      const statLiveAvgDCC = document.getElementById('statLiveAvgDCC');
      if (statLiveAvgDCC) statLiveAvgDCC.innerText = avgDCC.toFixed(2);
      const statLiveAvgTotal = document.getElementById('statLiveAvgTotal');
      if (statLiveAvgTotal) statLiveAvgTotal.innerText = avgTotal.toFixed(2);

      const lblLastDFC = document.getElementById('lblLastDFC');
      if (lblLastDFC) lblLastDFC.innerText = distSampleClusterLast;
      const lblLastDCC = document.getElementById('lblLastDCC');
      if (lblLastDCC) lblLastDCC.innerText = distClusterClusterLast;

      const lblLiveMeanDFC = document.getElementById('lblLiveMeanDFC');
      if (lblLiveMeanDFC) lblLiveMeanDFC.innerText = avgDFC.toFixed(2);
      const lblLiveMeanDCC = document.getElementById('lblLiveMeanDCC');
      if (lblLiveMeanDCC) lblLiveMeanDCC.innerText = avgDCC.toFixed(2);

      const statDistSC = document.getElementById('statDistSC');
      if (statDistSC) statDistSC.innerText = formatNumber(distSampleCluster);
      const statAvgSCEvals = document.getElementById('statAvgSCEvals');
      if (statAvgSCEvals) statAvgSCEvals.innerText = avgDFC.toFixed(2);

      const savedPct = naiveEvals > 0 ? (((naiveEvals - distSampleCluster) / naiveEvals) * 100).toFixed(0) : '0';
      const statSavedPct = document.getElementById('statSavedPct');
      if (statSavedPct) statSavedPct.innerText = `${Math.max(0, savedPct)}%`;

      const statNaiveEvals = document.getElementById('statNaiveEvals');
      if (statNaiveEvals) statNaiveEvals.innerText = formatNumber(naiveEvals);

      const statDistCC = document.getElementById('statDistCC');
      if (statDistCC) statDistCC.innerText = formatNumber(distClusterCluster);
      const statDccDim = document.getElementById('statDccDim');
      if (statDccDim) {
        const K = clusters.length;
        const maxPairs = dccPairsTotal || (K > 1 ? (K * (K - 1) / 2) : 0);
        const popPct = maxPairs > 0 ? ((dccPopulated / maxPairs) * 100).toFixed(1) : "100.0";
        if (useTiles) {
          statDccDim.innerText = `1D Tiles (${formatNumber(memStats.dccCells)} cells, ${formatNumber(dccPopulated)} pop)`;
        } else {
          statDccDim.innerText = `${clusters.length} × ${clusters.length} (${formatNumber(dccPopulated)}/${formatNumber(maxPairs)} pairs, ${popPct}% pop)`;
        }
      }
      const statAvgCCEvals = document.getElementById('statAvgCCEvals');
      if (statAvgCCEvals) statAvgCCEvals.innerText = totalFrames > 0 ? (distClusterCluster / totalFrames).toFixed(2) : "0.00";

      const statTotalClustersCount = document.getElementById('statTotalClustersCount');
      if (statTotalClustersCount) {
        statTotalClustersCount.innerText = useTiles 
          ? `${jointTuplesMap.size} tuples` 
          : `${clusters.length} anchors`;
      }

      const statActiveClusters = document.getElementById('statActiveClusters');
      if (statActiveClusters) {
        statActiveClusters.innerText = useTiles 
          ? `${jointTuplesMap.size} tuples (Kx=${tileEngineX.clusters.length}, Ky=${tileEngineY.clusters.length}${currentDim===3 ? `, Kz=${tileEngineZ.clusters.length}` : ''})`
          : `${clusters.length} anchors`;
      }

      const statDistThroughput = document.getElementById('statDistThroughput');
      if (statDistThroughput) statDistThroughput.innerText = `${formatNumber(Math.round(currentDistRate))} dist/s`;
      const statDistGrandTotal = document.getElementById('statDistGrandTotal');
      if (statDistGrandTotal) statDistGrandTotal.innerText = formatNumber(distSampleCluster + distClusterCluster);

      // 4. Resource Tracker - Tab 3: Memory
      const statModelTotalBytes = document.getElementById('statModelTotalBytes');
      if (statModelTotalBytes) statModelTotalBytes.innerText = formatBytes(memStats.totalModelBytes);
      const statCentroidsMem = document.getElementById('statCentroidsMem');
      if (statCentroidsMem) statCentroidsMem.innerText = formatBytes(memStats.centroidsBytes);
      const statDccMem = document.getElementById('statDccMem');
      if (statDccMem) statDccMem.innerText = formatBytes(memStats.dccBytes);
      const statTmMem = document.getElementById('statTmMem');
      if (statTmMem) statTmMem.innerText = formatBytes(memStats.tmBytes);
      const statPastSamplesMem = document.getElementById('statPastSamplesMem');
      if (statPastSamplesMem) statPastSamplesMem.innerText = formatBytes(memStats.pastSamplesBytes + memStats.historyBytes);
      const statTuplesMem = document.getElementById('statTuplesMem');
      if (statTuplesMem) statTuplesMem.innerText = formatBytes(memStats.tuplesBytes);

      const statHeapUsed = document.getElementById('statHeapUsed');
      if (statHeapUsed) statHeapUsed.innerText = memStats.heapUsedMB !== null ? `${memStats.heapUsedMB.toFixed(2)} MB` : "N/A (Chromium)";
      const statHeapTotal = document.getElementById('statHeapTotal');
      if (statHeapTotal) statHeapTotal.innerText = memStats.heapTotalMB !== null ? `${memStats.heapTotalMB.toFixed(2)} MB` : "N/A";
      const statHeapLimit = document.getElementById('statHeapLimit');
      if (statHeapLimit) statHeapLimit.innerText = memStats.heapLimitMB !== null ? `${memStats.heapLimitMB.toFixed(0)} MB` : "N/A";

      // 5. Resource Tracker - Tab 4: Compute & Pruning
      const statComputeThroughput = document.getElementById('statComputeThroughput');
      if (statComputeThroughput) statComputeThroughput.innerText = `${currentFps.toFixed(1)} FPS`;
      const statInstComputeMs = document.getElementById('statInstComputeMs');
      if (statInstComputeMs) statInstComputeMs.innerText = `${lastComputeTimeMs.toFixed(3)} ms`;
      const statAvgComputeMs = document.getElementById('statAvgComputeMs');
      if (statAvgComputeMs) statAvgComputeMs.innerText = `${avgComputeTimeMs.toFixed(3)} ms`;
      const statTotalComputeMs = document.getElementById('statTotalComputeMs');
      if (statTotalComputeMs) statTotalComputeMs.innerText = `${(totalComputeTimeMs / 1000).toFixed(3)} s`;

      const statPrune3P = document.getElementById('statPrune3P');
      if (statPrune3P) statPrune3P.innerText = formatNumber(pruneCount3P);
      const statPrune4P = document.getElementById('statPrune4P');
      if (statPrune4P) statPrune4P.innerText = formatNumber(pruneCount4P);
      const statPrune5P = document.getElementById('statPrune5P');
      if (statPrune5P) statPrune5P.innerText = formatNumber(pruneCount5P);
      const statPredHits = document.getElementById('statPredHits');
      if (statPredHits) statPredHits.innerText = formatNumber(predHitCount);

      // Shannon Entropy Diagnostics in Compute Panel
      const statDiagAvgInfoGain = document.getElementById('statDiagAvgInfoGain');
      if (statDiagAvgInfoGain) {
        const avgHPerFrame = totalFrames > 0 ? (totalEntropyReducedBits / totalFrames) : 0.0;
        statDiagAvgInfoGain.innerText = `${avgHPerFrame.toFixed(2)} bits/fr`;
      }
      const statDiagInfoPower = document.getElementById('statDiagInfoPower');
      if (statDiagInfoPower) statDiagInfoPower.innerText = `${avgInfoGainRate.toFixed(2)} bits/eval`;
      const statDiagMaxInitialH = document.getElementById('statDiagMaxInitialH');
      if (statDiagMaxInitialH) statDiagMaxInitialH.innerText = `${maxInitialEntropyObserved.toFixed(2)} bits`;
      const statDiagGatedCount = document.getElementById('statDiagGatedCount');
      if (statDiagGatedCount) statDiagGatedCount.innerText = `${formatNumber(totalEntropyGated)} (${gateRatioPct.toFixed(1)}%)`;
      const statDiagEvalCount = document.getElementById('statDiagEvalCount');
      if (statDiagEvalCount) statDiagEvalCount.innerText = formatNumber(totalEntropyEvals);

      // Progress bar and frame counter (matches dataset row sample count, Nloop removed)
      const frameCounterEl = document.getElementById('frameCounter');
      const progressFillEl = document.getElementById('progressFill');
      if (typeof dataMode !== 'undefined' && dataMode === 'image' &&
          typeof inspectedImageFrameIdx !== 'undefined' && inspectedImageFrameIdx >= 0) {
        const total = (benchmarkDataset && benchmarkDataset.length > 0)
          ? benchmarkDataset.length
          : totalFrames;
        const current = inspectedImageFrameIdx + 1;
        const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;
        if (progressFillEl) progressFillEl.style.width = `${pct}%`;
        if (frameCounterEl) {
          const countStr = `${formatNumber(current)} / ${formatNumber(total)}`;
          frameCounterEl.innerText = `${countStr} (Inspecting)`;
        }
      } else if (benchmarkDataset && benchmarkDataset.length > 0) {
        const total = benchmarkDataset.length;
        const current = Math.min(currentFrameIdx, total);
        const pct = Math.min(100, (current / total) * 100);
        if (progressFillEl) progressFillEl.style.width = `${pct}%`;
        if (frameCounterEl) frameCounterEl.innerText = `${formatNumber(current)} / ${formatNumber(total)}`;
      } else {
        const pct = totalFrames > 0 ? 100 : 0;
        if (progressFillEl) progressFillEl.style.width = `${pct}%`;
        if (frameCounterEl) {
          frameCounterEl.innerText = totalFrames > 0
            ? `${formatNumber(totalFrames)} live`
            : '0 / 0';
        }
      }

      // Render Sample History Toolbar
      renderSampleHistoryUI();

      // Render Narrative Log
      const contNarrative = document.getElementById('narrativeContainer');
      const activeTrace = getActiveSampleTrace();
      const isPastInspected = (selectedSampleTraceIndex >= 0 && activeTrace);

      if (!isExplainMode && !isPastInspected) {
        contNarrative.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 30px 12px; line-height: 1.6;">
          <b>💬 Explain Mode is OFF</b><br>
          Click <b>💬 Explain</b> in the top toolbar to enable real-time step-by-step decision tracking, or inspect recent samples above.
        </div>`;
      } else if (!activeTrace || !activeTrace.steps || activeTrace.steps.length === 0) {
        contNarrative.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 30px 12px; line-height: 1.5;">
          No decisions logged for this frame yet.<br>Click <b>➔ Step</b> or <b>＋ Add Point</b> to inspect algorithm decisions.
        </div>`;
      } else {
        let bannerHtml = '';
        if (isPastInspected) {
          const framesAgo = totalFrames - activeTrace.frameIndex;
          bannerHtml = `
            <div style="background: rgba(250, 204, 21, 0.10); border: 1px solid rgba(250, 204, 21, 0.4); border-radius: 4px; padding: 6px 8px; margin-bottom: 8px; font-size: 0.72rem; display: flex; justify-content: space-between; align-items: center;">
              <div>
                <span style="color: #facc15; font-weight: 700;">🕒 Inspecting Sample #${activeTrace.frameIndex}</span>
                <span style="color: var(--text-muted); font-size: 0.68rem; margin-left: 4px;">(${framesAgo === 0 ? 'Latest' : `${framesAgo} frames ago`})</span>
              </div>
              <button onclick="returnToLiveStream()" class="btn-micro" style="background: rgba(74, 222, 128, 0.2); color: #4ade80; border-color: rgba(74, 222, 128, 0.4); font-size: 0.68rem; padding: 2px 6px;">● Return to Live</button>
            </div>
          `;
        }

        contNarrative.innerHTML = bannerHtml + activeTrace.steps.map(step => `
          <div class="narrative-step ${step.type}">
            <div class="narrative-title">
              <span>${step.title}</span>
            </div>
            <div class="narrative-text">${step.text}</div>
            ${step.entropyRankings && step.entropyRankings.length > 0 ? `
              <div style="margin-top: 6px; background: #0f172a; border: 1px solid rgba(74, 222, 128, 0.25); border-radius: 4px; padding: 6px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.68rem; font-weight: 700; color: var(--accent-green); margin-bottom: 4px;">
                  <span>Evaluated Targets Ranking</span>
                  <span>Uncertainty E[H] / Info Gain</span>
                </div>
                ${step.entropyRankings.slice(0, 5).map(r => {
                  const isHovered = (hoveredClusterId === r.id);
                  const isSelected = (selectedClusterId === r.id);
                  const bg = isSelected 
                    ? 'rgba(250, 204, 21, 0.15)' 
                    : (isHovered ? 'rgba(56, 189, 248, 0.15)' : 'transparent');
                  return `
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.68rem; padding: 2px 4px; border-top: 1px solid rgba(51, 65, 85, 0.4); background: ${bg}; cursor: pointer; border-radius: 3px; transition: background 0.15s ease;"
                         onmouseenter="setHoveredCluster(${r.id})"
                         onmouseleave="setHoveredCluster(-1)"
                         onclick="toggleSelectCluster(${r.id})">
                      <span style="color: ${isSelected ? '#facc15' : (isHovered ? '#38bdf8' : (r.isChosen ? '#4ade80' : '#cbd5e1'))}; font-weight: ${r.isChosen || isHovered ? '700' : 'normal'};">
                        ${r.isChosen ? '★ ' : ''}C${r.id} <span style="color: var(--text-muted); font-size: 0.64rem;">(P=${r.p.toFixed(3)})</span>
                      </span>
                      <span style="font-family: monospace; color: ${r.isChosen ? '#4ade80' : '#38bdf8'}; font-size: 0.66rem;">
                        ${r.isSupport ? `${r.expectedH.toFixed(1)} cl` : `E[H]=${r.expectedH.toFixed(2)}b`} <span style="color: #fbbf24;">(+${r.infoGain.toFixed(2)}b)</span>
                      </span>
                    </div>
                  `;
                }).join('')}
              </div>
            ` : ''}
          </div>
        `).join('');
      }

      // Render Cluster Table
      const contCandidates = document.getElementById('candidateContainer');
      
      function renderTableToolbar() {
        const isModified = clusterSortKey !== 'cluster' || clusterSortDir !== 'asc' || Object.values(clusterTableCols).some(c => !c.visible);
        return `
          <div class="table-toolbar">
            <div style="display:flex; align-items:center; gap:5px;">
              <span style="font-size:0.68rem; color:var(--text-muted); font-weight:700;">Columns:</span>
              <div class="table-col-chips">
                <span class="col-chip ${clusterTableCols.cluster.visible ? 'active' : ''}" onclick="toggleClusterColumn('cluster')" data-tooltip-title="Toggle Cluster ID Column" data-tooltip-badge="${clusterTableCols.cluster.visible ? 'Visible' : 'Minimized'}" data-tooltip-color="cyan" data-tooltip-desc="Click to ${clusterTableCols.cluster.visible ? 'minimize (hide)' : 'restore (show)'} the Cluster ID column.">${clusterTableCols.cluster.visible ? '✓' : '+'} Cluster</span>
                <span class="col-chip ${clusterTableCols.centroid.visible ? 'active' : ''}" onclick="toggleClusterColumn('centroid')" data-tooltip-title="Toggle Centroid Column" data-tooltip-badge="${clusterTableCols.centroid.visible ? 'Visible' : 'Minimized'}" data-tooltip-color="cyan" data-tooltip-desc="Click to ${clusterTableCols.centroid.visible ? 'minimize (hide)' : 'restore (show)'} the Centroid Coordinates column.">${clusterTableCols.centroid.visible ? '✓' : '+'} Centroid</span>
                <span class="col-chip ${clusterTableCols.frames.visible ? 'active' : ''}" onclick="toggleClusterColumn('frames')" data-tooltip-title="Toggle Frames Column" data-tooltip-badge="${clusterTableCols.frames.visible ? 'Visible' : 'Minimized'}" data-tooltip-color="cyan" data-tooltip-desc="Click to ${clusterTableCols.frames.visible ? 'minimize (hide)' : 'restore (show)'} the Ingested Frames Count column.">${clusterTableCols.frames.visible ? '✓' : '+'} Frames</span>
                <span class="col-chip ${clusterTableCols.scDists.visible ? 'active' : ''}" onclick="toggleClusterColumn('scDists')" data-tooltip-title="Toggle #SC Dists Column" data-tooltip-badge="${clusterTableCols.scDists.visible ? 'Visible' : 'Minimized'}" data-tooltip-color="cyan" data-tooltip-desc="Click to ${clusterTableCols.scDists.visible ? 'minimize (hide)' : 'restore (show)'} the Sample-Cluster Distance Evaluation Count column.">${clusterTableCols.scDists.visible ? '✓' : '+'} #SC dists</span>
                <span class="col-chip ${clusterTableCols.dist.visible ? 'active' : ''}" onclick="toggleClusterColumn('dist')" data-tooltip-title="Toggle Distance Column" data-tooltip-badge="${clusterTableCols.dist.visible ? 'Visible' : 'Minimized'}" data-tooltip-color="cyan" data-tooltip-desc="Click to ${clusterTableCols.dist.visible ? 'minimize (hide)' : 'restore (show)'} the Real-time Query Distance column.">${clusterTableCols.dist.visible ? '✓' : '+'} Dist</span>
                <span class="col-chip ${clusterTableCols.status.visible ? 'active' : ''}" onclick="toggleClusterColumn('status')" data-tooltip-title="Toggle Status Column" data-tooltip-badge="${clusterTableCols.status.visible ? 'Visible' : 'Minimized'}" data-tooltip-color="cyan" data-tooltip-desc="Click to ${clusterTableCols.status.visible ? 'minimize (hide)' : 'restore (show)'} the Decision Status / Prior badge column.">${clusterTableCols.status.visible ? '✓' : '+'} Status</span>
              </div>
            </div>
            ${isModified ? `
              <span style="font-size:0.67rem; color:var(--accent-blue); cursor:pointer; text-decoration:underline; font-weight:600;" 
                    onclick="resetClusterTableColumns()" 
                    data-tooltip-title="Reset Columns &amp; Sort" 
                    data-tooltip-badge="Reset Table" 
                    data-tooltip-color="cyan" 
                    data-tooltip-desc="Restores all minimized columns to visible and resets sorting back to Cluster ID ascending.">↺ Reset</span>
            ` : ''}
          </div>
        `;
      }

      function renderColHeader(colKey, label, isNum = false) {
        if (!clusterTableCols[colKey] || !clusterTableCols[colKey].visible) return '';
        const isSorted = clusterSortKey === colKey;
        const sortIcon = isSorted ? (clusterSortDir === 'asc' ? '▲' : '▼') : '<span class="sort-icon">⇅</span>';
        const sortBadge = isSorted ? (clusterSortDir === 'asc' ? 'Ascending (▲)' : 'Descending (▼)') : 'Click to Sort';
        const sortDesc = isSorted 
          ? `Currently sorted ${clusterSortDir === 'asc' ? 'Ascending' : 'Descending'}. Click to toggle sort direction. Click '−' to minimize column.` 
          : `Click to sort table by ${label} ascending. Click '−' to minimize column.`;

        return `
          <th class="${isSorted ? 'sorted' : ''} ${isNum ? 'num-col' : ''}" 
              onclick="setClusterSort('${colKey}')" 
              data-tooltip-title="Sort by ${label}" 
              data-tooltip-badge="${sortBadge}" 
              data-tooltip-color="cyan" 
              data-tooltip-desc="${sortDesc}">
            <div class="th-content">
              <span>${label} ${sortIcon}</span>
              <span class="btn-min-col" 
                    onclick="event.stopPropagation(); toggleClusterColumn('${colKey}');" 
                    title="Minimize ${label} column">−</span>
            </div>
          </th>
        `;
      }

      if (typeof dataMode !== 'undefined' && dataMode === 'image' &&
          typeof inspectedClusterId !== 'undefined' && inspectedClusterId >= 0 &&
          clusters[inspectedClusterId]) {
        const c = clusters[inspectedClusterId];
        const members = (imageClusterMembers && imageClusterMembers[inspectedClusterId]) || [];
        const memberPct = totalFrames > 0
          ? ((members.length / totalFrames) * 100).toFixed(1)
          : '0.0';

        let html = `
          <div style="display:flex; flex-direction:column; gap:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; ` +
              `padding:6px 10px; background:#0f172a; border:1px solid var(--card-border); ` +
              `border-radius:6px;">
              <div style="display:flex; align-items:center; gap:6px;">
                <span style="display:inline-block; width:9px; height:9px; border-radius:50%; ` +
                  `background:${c.color || '#38bdf8'};"></span>
                <span style="font-weight:700; color:#f8fafc; font-size:0.78rem;">` +
                  `Cluster C${c.id} Member Frames</span>
                <span class="badge-pill" style="background:rgba(56,189,248,0.15); ` +
                  `color:#38bdf8; font-size:0.68rem;">` +
                  `${members.length} frames (${memberPct}%)</span>
              </div>
              <button class="btn-action" style="padding:2px 8px; font-size:0.68rem;" ` +
                `onclick="clearImageClusterInspection()">← All Clusters</button>
            </div>
        `;

        if (members.length === 0) {
          html += `<div style="color:var(--text-muted); font-size:0.75rem; text-align:center; ` +
            `padding:20px 0;">No frames assigned to this cluster yet.</div>`;
        } else {
          html += `
            <div style="display:grid; ` +
              `grid-template-columns:repeat(auto-fill, minmax(72px, 1fr)); ` +
              `gap:6px; max-height:360px; overflow-y:auto; padding:4px;">
          `;
          members.forEach(fIdx => {
            const isSelected = (inspectedImageFrameIdx === fIdx);
            const dist = (imageFrameDists && imageFrameDists[fIdx] !== undefined)
              ? imageFrameDists[fIdx].toFixed(3)
              : '—';
            const cardBorder = isSelected ? '2px solid #facc15' : '1px solid #1e293b';
            html += `
              <div class="image-member-card ${isSelected ? 'selected' : ''}" 
                   style="display:flex; flex-direction:column; align-items:center; ` +
                     `background:#0b1120; border:${cardBorder}; ` +
                     `border-radius:6px; padding:4px; cursor:pointer;"
                   onclick="selectImageFrame(${fIdx})"
                   title="Frame #${fIdx + 1} • d=${dist}">
                <canvas class="member-thumb-canvas" data-frame-idx="${fIdx}" ` +
                  `width="32" height="32" ` +
                  `style="width:52px; height:52px; border-radius:3px; background:#020617; ` +
                    `image-rendering:pixelated; border:1px solid rgba(255,255,255,0.08);"></canvas>
                <div style="display:flex; justify-content:space-between; width:100%; ` +
                  `margin-top:3px; font-size:0.62rem; font-family:monospace;">
                  <span style="color:${isSelected ? '#facc15' : '#38bdf8'}; font-weight:700;">` +
                    `#${fIdx + 1}</span>
                  <span style="color:#94a3b8;">d:${dist}</span>
                </div>
              </div>
            `;
          });
          html += `</div>`;
        }
        html += `</div>`;
        contCandidates.innerHTML = html;

        requestAnimationFrame(() => {
          const canvases = contCandidates.querySelectorAll('.member-thumb-canvas');
          canvases.forEach(cv => {
            const fIdx = parseInt(cv.dataset.frameIdx, 10);
            if (benchmarkDataset && benchmarkDataset[fIdx]) {
              const cCtx = cv.getContext('2d');
              if (typeof drawRasterBuffer === 'function') {
                drawRasterBuffer(
                  cCtx, benchmarkDataset[fIdx], imageWidth, imageHeight,
                  0, 0, cv.width, cv.height, 1.0
                );
              }
            }
          });
        });
        return;
      }

      if (useTiles) {
        if (jointTuplesMap.size === 0) {
          contCandidates.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 20px 0;">No tiles processed yet.</div>`;
        } else {
          let entries = Array.from(jointTuplesMap.entries());

          // Sort entries
          entries.sort(([keyA, itemA], [keyB, itemB]) => {
            let res = 0;
            if (clusterSortKey === 'cluster') {
              res = keyA.localeCompare(keyB);
            } else if (clusterSortKey === 'centroid') {
              res = (itemA.cx - itemB.cx) || (itemA.cy - itemB.cy) || ((itemA.cz || 0) - (itemB.cz || 0));
            } else if (clusterSortKey === 'frames') {
              res = itemA.count - itemB.count;
            } else if (clusterSortKey === 'scDists') {
              res = (itemA.scDists || itemA.count) - (itemB.scDists || itemB.count);
            } else if (clusterSortKey === 'status') {
              const isCurrA = currentJointTuple && currentJointTuple.cx === itemA.cx && currentJointTuple.cy === itemA.cy && (currentDim === 2 || currentJointTuple.cz === itemA.cz);
              const isCurrB = currentJointTuple && currentJointTuple.cx === itemB.cx && currentJointTuple.cy === itemB.cy && (currentDim === 2 || currentJointTuple.cz === itemB.cz);
              res = (isCurrA ? 1 : 0) - (isCurrB ? 1 : 0);
            }
            if (res === 0) res = keyA.localeCompare(keyB);
            return clusterSortDir === 'asc' ? res : -res;
          });

          let html = renderTableToolbar() + `
            <table class="cluster-table">
              <thead>
                <tr>
                  ${renderColHeader('cluster', 'Joint Tuple', false)}
                  ${renderColHeader('centroid', 'Indices', false)}
                  ${renderColHeader('frames', 'Frames', true)}
                  ${renderColHeader('scDists', '#SC dists', true)}
                  ${renderColHeader('status', 'Status', false)}
                </tr>
              </thead>
              <tbody>
          `;
          entries.forEach(([key, item]) => {
            const isCurrent = currentJointTuple && 
              currentJointTuple.cx === item.cx && 
              currentJointTuple.cy === item.cy && 
              (currentDim === 2 || currentJointTuple.cz === item.cz);
            const isSelected = selectedTupleKey === key;
            const statusClass = isCurrent ? 'matched' : '';
            const badgeText = isCurrent ? 'ACTIVE' : 'IDLE';

            html += `
              <tr class="cluster-row ${statusClass} ${isSelected ? 'selected' : ''}" onclick="toggleSelectTuple('${key}')">
                ${clusterTableCols.cluster.visible ? `
                  <td>
                    <div style="display:flex; align-items:center; gap:5px;">
                      <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#c084fc;"></span>
                      <span style="font-weight:700; color:${isSelected ? '#facc15' : '#f8fafc'};">T_${item.cx}_${item.cy}${currentDim === 3 ? `_${item.cz}` : ''}</span>
                    </div>
                  </td>
                ` : ''}
                ${clusterTableCols.centroid.visible ? `
                  <td style="color:#94a3b8;">(${item.cx}, ${item.cy}${currentDim === 3 ? `, ${item.cz}` : ''})</td>
                ` : ''}
                ${clusterTableCols.frames.visible ? `
                  <td class="num-col"><span style="color:#f8fafc; font-weight:600;">${item.count}</span> <span style="color:#64748b; font-size:0.68rem;">(${((item.count / Math.max(1, totalFrames)) * 100).toFixed(1)}%)</span></td>
                ` : ''}
                ${clusterTableCols.scDists.visible ? `
                  <td class="num-col" style="color:var(--accent-blue); font-weight:600;">${item.scDists || item.count}</td>
                ` : ''}
                ${clusterTableCols.status.visible ? `
                  <td><span class="badge-pill" style="font-size:0.65rem; ${isCurrent ? 'background:rgba(74,222,128,0.2); color:var(--accent-green);' : ''}">${badgeText}</span></td>
                ` : ''}
              </tr>
            `;
          });
          html += `</tbody></table>`;
          contCandidates.innerHTML = html;
        }
      } else {
        if (clusters.length === 0) {
          contCandidates.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 20px 0;">No clusters formed yet.</div>`;
        } else {
          // Sort clusters copy according to clusterSortKey and clusterSortDir
          let displayClusters = clusters.slice();

          displayClusters.sort((a, b) => {
            let res = 0;
            if (clusterSortKey === 'cluster') {
              res = a.id - b.id;
            } else if (clusterSortKey === 'centroid') {
              const normA = Math.sqrt(a.x * a.x + a.y * a.y + (currentDim === 3 ? a.z * a.z : 0));
              const normB = Math.sqrt(b.x * b.x + b.y * b.y + (currentDim === 3 ? b.z * b.z : 0));
              res = normA - normB;
            } else if (clusterSortKey === 'frames') {
              res = a.members - b.members;
            } else if (clusterSortKey === 'scDists') {
              res = (a.scDists || 0) - (b.scDists || 0);
            } else if (clusterSortKey === 'dist') {
              const evA = currentEvaluations.find(ev => ev.target.id === a.id);
              const evB = currentEvaluations.find(ev => ev.target.id === b.id);
              const distA = evA ? evA.dist : (currentPruned.find(p => p.cluster.id === a.id) ? 9998 : 9999);
              const distB = evB ? evB.dist : (currentPruned.find(p => p.cluster.id === b.id) ? 9998 : 9999);
              res = distA - distB;
            } else if (clusterSortKey === 'status') {
              function getStatusRank(cObj) {
                const ev = currentEvaluations.find(e => e.target.id === cObj.id);
                if (ev && ev.match) return 5;
                if (ev && !ev.match) return 3;
                if (currentPredicted.includes(cObj.id)) return 4;
                if (currentPruned.find(p => p.cluster.id === cObj.id)) return 1;
                return 0; // IDLE
              }
              res = getStatusRank(a) - getStatusRank(b);
            }
            if (res === 0) res = a.id - b.id;
            return clusterSortDir === 'asc' ? res : -res;
          });

          let html = renderTableToolbar() + `
            <table class="cluster-table">
              <thead>
                <tr>
                  ${renderColHeader('cluster', 'Cluster', false)}
                  ${renderColHeader('centroid', 'Centroid', false)}
                  ${renderColHeader('frames', 'Frames', true)}
                  ${renderColHeader('scDists', '#SC dists', true)}
                  ${renderColHeader('dist', 'Dist d(f,c)', true)}
                  ${renderColHeader('status', 'Status', false)}
                </tr>
              </thead>
              <tbody>
          `;

          displayClusters.forEach(c => {
            const isEval = currentEvaluations.find(ev => ev.target.id === c.id);
            const isPruned = currentPruned.find(p => p.cluster.id === c.id);
            const isPred = currentPredicted.includes(c.id);
            const isSelected = selectedClusterId === c.id;
            const isHovered = hoveredClusterId === c.id;

            let statusClass = '';
            let badgeText = 'IDLE';
            let badgeStyle = 'color: #94a3b8;';
            let distText = '—';
            let distColor = '#64748b';

            if (isEval) {
              distText = isEval.dist.toFixed(3);
              if (isEval.match) {
                statusClass = 'matched';
                badgeText = isPred ? 'PRED MATCH' : 'ASSIGNED';
                badgeStyle = 'background: rgba(74,222,128,0.2); color: var(--accent-green); font-weight:700;';
                distColor = 'var(--accent-green)';
              } else {
                badgeText = isPred ? 'PRED MISMATCH' : 'MISMATCH';
                badgeStyle = 'background: rgba(248,113,113,0.15); color: var(--accent-red);';
                distColor = 'var(--accent-red)';
              }
            } else if (isPruned) {
              statusClass = 'pruned';
              badgeText = `PRUNED (${isPruned.reason})`;
              badgeStyle = 'background: rgba(192,132,252,0.15); color: var(--accent-purple);';
              distText = 'Pruned';
              distColor = 'var(--accent-purple)';
            } else if (isPred) {
              badgeText = 'PREDICTED';
              badgeStyle = 'background: rgba(56,189,248,0.15); color: var(--accent-blue);';
            }

            const coordStr = (typeof dataMode !== 'undefined' && dataMode === 'image')
              ? `[32×32 Image]`
              : (currentDim === 3
                ? `(${c.x.toFixed(2)}, ${c.y.toFixed(2)}, ${c.z.toFixed(2)})`
                : `(${c.x.toFixed(2)}, ${c.y.toFixed(2)})`);

            const framePct = totalFrames > 0 ? ((c.members / totalFrames) * 100).toFixed(1) : '100.0';
            const rowClick = (typeof dataMode !== 'undefined' && dataMode === 'image')
              ? `inspectClusterMembers(${c.id})`
              : `toggleSelectCluster(${c.id})`;

            html += `
              <tr class="cluster-row ${statusClass} ${isSelected ? 'selected' : ''} ${isHovered ? 'hovered' : ''}" 
                  data-cluster-id="${c.id}" 
                  onmouseenter="setHoveredCluster(${c.id})" 
                  onmouseleave="setHoveredCluster(-1)" 
                  onclick="${rowClick}">
                ${clusterTableCols.cluster.visible ? `
                  <td>
                    <div style="display:flex; align-items:center; gap:5px;">
                      <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:${c.color};"></span>
                      <span style="font-weight:700; color:${isSelected ? '#facc15' : (isHovered ? '#38bdf8' : '#f8fafc')};">C${c.id}</span>
                    </div>
                  </td>
                ` : ''}
                ${clusterTableCols.centroid.visible ? `
                  <td style="color:#cbd5e1; font-size:0.70rem;">${coordStr}</td>
                ` : ''}
                ${clusterTableCols.frames.visible ? `
                  <td class="num-col"><span style="color:#f8fafc; font-weight:600;">${c.members}</span> <span style="color:#64748b; font-size:0.66rem;">(${framePct}%)</span></td>
                ` : ''}
                ${clusterTableCols.scDists.visible ? `
                  <td class="num-col" style="color:var(--accent-blue); font-weight:600;">
                    ${c.scDists || c.members || 0}
                  </td>
                ` : ''}
                ${clusterTableCols.dist.visible ? `
                  <td class="num-col" style="color:${distColor}; font-weight:600;">${distText}</td>
                ` : ''}
                ${clusterTableCols.status.visible ? `
                  <td><span class="badge-pill" style="font-size:0.65rem; ${badgeStyle}">${badgeText}</span></td>
                ` : ''}
              </tr>
            `;
          });

          html += `</tbody></table>`;
          contCandidates.innerHTML = html;
        }
      }
    }

    function resetClustering(keepDataset = true) {
      if (typeof pauseSimulation === 'function') {
        pauseSimulation();
      }
      if (typeof clearActiveFrameEvaluations === 'function') {
        clearActiveFrameEvaluations(true);
      }
      selectedClusterId = -1;
      hoveredClusterId = -1;
      selectedTupleKey = null;
      clusters = [];
      dcc = [];
      transitionCounts = [];
      prevAssignedCluster = -1;
      assignmentHistory = [];
      frameHistory = [];
      totalFrames = 0;
      totalEvals = 0;
      naiveEvals = 0;
      currentFrame = null;
      currentImageFrame = null;
      imageGalleryScrollY = 0;
      inspectedImageFrameIdx = -1;
      inspectedClusterId = -1;
      imageFrameAssignments = [];
      imageFrameDists = [];
      imageClusterMembers = {};
      currentEvaluations = [];
      currentPruned = [];
      currentPredicted = [];
      currentEntropyBits = 0;
      currentFrameIdx = 0;
      currentLoop = 1;
      currentExplanation = [];
      sampleTraceLog = [];
      selectedSampleTraceIndex = -1;
      hoveredSampleTracePoint = null;
      hoveredClosestSample = null;
      frameEvaluationsLog = [];

      // Resource Tracker reset
      distSampleCluster = 0;
      distClusterCluster = 0;
      distSampleClusterLast = 0;
      distClusterClusterLast = 0;
      dccPopulated = 0;
      dccPairsTotal = 0;
      pruneCount3P = 0;
      pruneCount4P = 0;
      pruneCount5P = 0;
      predHitCount = 0;
      totalComputeTimeMs = 0.0;
      lastComputeTimeMs = 0.0;
      avgComputeTimeMs = 0.0;
      sparklineHistory = new Array(60).fill(0.0);
      distHistoryDFC = [];
      distHistoryDCC = [];
      hoverDistIndex = null;
      hoverDistAvgIndex = null;
      lastTransitionFrom = -1;
      lastTransitionTo = -1;
      hoveredTMCell = null;
      topLearnedPathsCache = [];
      cachedTMRowTotals = null;
      lastTMRenderTimestamp = 0;
      rollingHistory = [];
      currentCpuLoadPct = 0.0;
      currentFps = 0.0;
      currentDistRate = 0.0;
      sessionStartTime = 0;
      sessionStartFrames = 0;
      sessionElapsedMs = 0;
      sessionIsActive = false;
      sessionAvgFps = 0.0;

      // Entropy Telemetry reset
      totalInitialEntropyBits = 0.0;
      totalEntropyReducedBits = 0.0;
      totalEntropyEvals = 0;
      totalEntropyGated = 0;
      lastInitialEntropy = 0.0;
      lastEntropyReduced = 0.0;
      lastInfoGainRate = 0.0;
      maxInitialEntropyObserved = 0.0;
      lastEntropyRankings = [];

      // k-NN Solver & Hover Selection reset
      knnResults = null;
      selectedKnnQuerySample = -1;
      hoveredKnnNeighborId = -1;
      hoveredClosestSample = null;
      lockedClosestSample = null;
      if (typeof renderKnnTrace === 'function') {
        renderKnnTrace();
      }

      // Reset tile engine plain objects
      tileEngineX.clusters = [];
      tileEngineX.dcc = [];
      tileEngineX.totalEvals = 0;
      tileEngineX.naiveEvals = 0;
      tileEngineY.clusters = [];
      tileEngineY.dcc = [];
      tileEngineY.totalEvals = 0;
      tileEngineY.naiveEvals = 0;
      tileEngineZ.clusters = [];
      tileEngineZ.dcc = [];
      tileEngineZ.totalEvals = 0;
      tileEngineZ.naiveEvals = 0;
      jointTuplesMap.clear();
      currentJointTuple = null;
      tileTraceX = null;
      tileTraceY = null;
      tileTraceZ = null;

      // WASM session init (re-create handle)
      if (GricWasm.isLoaded()) {
        const params = GricWasm.buildParamsFromState();
        if (useTiles && GricWasm.initMultiTile) {
          GricWasm.destroy();
          GricWasm.initMultiTile(params);
          wasmSessionActive = true;
        } else {
          if (GricWasm.destroyMultiTile) {
            GricWasm.destroyMultiTile();
          }
          wasmSessionActive = GricWasm.init(params);
        }
        if (typeof GricWasmWorker !== 'undefined') {
          GricWasmWorker.startSession(params);
        }
      } else {
        wasmSessionActive = false;
      }

      if (!keepDataset) {
        pastSamples = [];
        benchmarkDataset = [];
        rawBenchmarkDataset = [];
        isDatasetStaged = false;
      } else if (pastSamples && pastSamples.length > 0) {
        // Reset all sample cluster assignments to -1 (neutral unclustered preview)
        for (let i = 0; i < pastSamples.length; i++) {
          pastSamples[i].clusterId = -1;
        }
      }

      clusterMilestoneFrames = [];
      clusterSpawnRipples = [];
      if (typeof renderTimelineMilestones === 'function') {
        renderTimelineMilestones();
      }

      if (typeof updateWasmBadge === 'function') {
        updateWasmBadge();
      }

      updateDatasetStatusBadge();
      updateUI();
      if (typeof draw === 'function') {
        draw();
      }
    }

    function resetSimulation() {
      resetClustering(false);
    }

    function updateDatasetStatusBadge() {
      const pill = document.getElementById('datasetStatusPill');
      if (!pill) return;

      if (!isDatasetStaged || !benchmarkDataset || benchmarkDataset.length === 0) {
        pill.textContent = '⚪ No dataset staged';
        pill.style.background = 'rgba(100, 116, 139, 0.2)';
        pill.style.color = '#94a3b8';
        pill.style.borderColor = 'rgba(100, 116, 139, 0.4)';
        return;
      }

      const countStr = benchmarkDataset.length.toLocaleString();
      const dimStr = dataMode === 'image' ? '1024D Img' : (currentDim === 3 ? '3D' : '2D');
      pill.textContent = `📦 Staged: ${countStr} pts (${dimStr}, ${currentBenchmark})`;
      pill.style.background = 'rgba(56, 189, 248, 0.15)';
      pill.style.color = '#38bdf8';
      pill.style.borderColor = 'rgba(56, 189, 248, 0.35)';

      const dimBadge = document.getElementById('inputDataDimBadge');
      if (dimBadge) {
        dimBadge.textContent = `${dimStr} • ${countStr} pts`;
      }
    }

    function hasMoreFrames() {
      if (!benchmarkDataset || benchmarkDataset.length === 0) return false;
      return currentFrameIdx < benchmarkDataset.length;
    }

    function stageDataset(benchmarkKey = null) {
      if (benchmarkKey) {
        currentBenchmark = benchmarkKey;
      } else {
        const selMain = document.getElementById('selectBenchmark');
        const selSide = document.getElementById('selectBenchmarkSide');
        if (selMain && selMain.value) {
          currentBenchmark = selMain.value;
        } else if (selSide && selSide.value) {
          currentBenchmark = selSide.value;
        }
      }

      const selMain = document.getElementById('selectBenchmark');
      if (selMain) {
        const opt = selMain.querySelector(`option[value="${currentBenchmark}"]`);
        if (opt) selMain.value = currentBenchmark;
      }

      const selSide = document.getElementById('selectBenchmarkSide');
      if (selSide) {
        const optSide = selSide.querySelector(`option[value="${currentBenchmark}"]`);
        if (optSide) selSide.value = currentBenchmark;
      }

      // Reset clustering while keeping dataset staging pipeline clean
      resetClustering(false);

      const descEl = document.getElementById('benchmarkDesc');
      if (descEl) {
        descEl.innerHTML = BENCHMARK_DESCS[currentBenchmark] || `<b>${currentBenchmark}</b>`;
      }

      if (typeof isImageBenchmark === 'function' && isImageBenchmark(currentBenchmark)) {
        dataMode = 'image';
        imageWidth = 32;
        imageHeight = 32;
        imageDim = 1024;
        currentDim = 1024;
        rawBenchmarkDataset = generateImageBenchmark(currentBenchmark, sampleCount);
        applyNoiseToDataset();
      } else {
        dataMode = 'coord';
        currentDim = is3DBenchmark(currentBenchmark) ? 3 : 2;
        if (currentDim === 2) {
          maximizedQuad = null;
        }
        if (currentBenchmark === "custom") {
          const fileInput = document.getElementById('fileUpload');
          if (fileInput) fileInput.click();
          return;
        } else {
          rawBenchmarkDataset = generateBenchmark(
            currentBenchmark, sampleCount
          );
          applyNoiseToDataset();
        }
      }

      // Populate pastSamples for immediate point cloud preview in viewports
      pastSamples = [];
      if (dataMode === 'coord' && benchmarkDataset && benchmarkDataset.length > 0) {
        const maxStagedPreview = 100000;
        const stride = benchmarkDataset.length > maxStagedPreview ? Math.ceil(benchmarkDataset.length / maxStagedPreview) : 1;
        for (let i = 0; i < benchmarkDataset.length; i += stride) {
          const pt = benchmarkDataset[i];
          pastSamples.push({
            x: pt.x,
            y: pt.y,
            z: pt.z || 0.0,
            clusterId: -1,
            frameIndex: i
          });
        }
      }

      isDatasetStaged = true;
      stagedDatasetInfo = {
        name: currentBenchmark,
        count: benchmarkDataset.length,
        dim: currentDim,
        passes: loopCount || 1,
        noise: noiseSigma
      };

      // Hide/Show 3D Viewport Preset Bar in image mode
      const presetBar = document.getElementById('viewPresetBar');
      if (presetBar) {
        presetBar.style.display =
          (dataMode === 'image' || currentDim === 2) ? 'none' : 'flex';
      }

      // Re-initialize WASM session for new ndim / benchmark
      if (useWasm && GricWasm.isLoaded()) {
        const params = GricWasm.buildParamsFromState();
        wasmSessionActive = GricWasm.init(params);
        updateWasmBadge();
      }

      // Auto-export staged dataset to native desktop workspace if connected
      if (typeof DesktopBridge !== 'undefined' && DesktopBridge.isAvailable() &&
          dataMode === 'coord' && benchmarkDataset.length > 0) {
        DesktopBridge.stageDatasetFile(currentBenchmark, benchmarkDataset)
          .catch(err => console.warn('[DesktopBridge] Auto-stage export failed:', err));
      }

      updateDatasetStatusBadge();
      updateUI();
      if (typeof draw === 'function') {
        draw();
      }
    }

    function loadSelectedBenchmark() {
      const selMain = document.getElementById('selectBenchmark');
      const selSide = document.getElementById('selectBenchmarkSide');
      const key = (selMain && selMain.value) || (selSide && selSide.value) || currentBenchmark;
      stageDataset(key);
    }

    function stepNextFrame(skipRender = false) {
      if (!benchmarkDataset || benchmarkDataset.length === 0) return;

      if (currentFrameIdx >= benchmarkDataset.length) {
        pauseSimulation();
        return;
      }

      if (dataMode === 'image') {
        const frameData = benchmarkDataset[currentFrameIdx++];
        currentImageFrame = frameData;
        clusterImageFrame(frameData, skipRender);
        return;
      }

      const pt = benchmarkDataset[currentFrameIdx++];
      clusterFrame(pt.x, pt.y, pt.z || 0.0, skipRender);
    }

    // =========================================================================

