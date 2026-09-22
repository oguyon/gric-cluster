/**
 * GRIC Simulator - sim_dimdensity.js
 * Intrinsic Dimension & Density Estimation UI and visualizations.
 */

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
        datasetName = `${currentBenchmark}.bin`;
      }
      const dsBase = datasetName.replace(
        /\.(bin|txt|csv|fits|dat|mp4|fits\.fz)$/i, ''
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

