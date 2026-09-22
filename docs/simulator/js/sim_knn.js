/**
 * GRIC Simulator - sim_knn.js
 * k-Nearest Neighbors (gric-knn) UI and query runner.
 */

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

      const activeKnnObj = (typeof knnResults !== 'undefined') ? knnResults : null;
      const slotKnnObj = slot ? slot.knnResults : null;
      const hasActiveKnn = Boolean(
        activeKnnObj && (
          activeKnnObj.indices ||
          activeKnnObj.queries ||
          (Array.isArray(activeKnnObj) && activeKnnObj.length > 0) ||
          (typeof activeKnnObj.totalFrames === 'number' && activeKnnObj.totalFrames > 0)
        )
      ) || (typeof reconstructionInfo !== 'undefined' && Boolean(reconstructionInfo))
        || (typeof reconstructionSourceNeighbors !== 'undefined' &&
            Boolean(reconstructionSourceNeighbors));
      const hasSlotKnn = Boolean(
        slotKnnObj && (
          slotKnnObj.indices ||
          slotKnnObj.queries ||
          (Array.isArray(slotKnnObj) && slotKnnObj.length > 0) ||
          (typeof slotKnnObj.totalFrames === 'number' && slotKnnObj.totalFrames > 0)
        )
      ) || (slot && Boolean(slot.reconstructionInfo))
        || (slot && Boolean(slot.reconstructionSourceNeighbors));
      const hasKnn = hasActiveKnn || hasSlotKnn;
      const kVal = (activeKnnObj?.k ||
                    (typeof reconstructionInfo !== 'undefined' && reconstructionInfo?.k) ||
                    knnK || 10);

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
          btnTop.classList.remove('btn-knn-computed');
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
          if (!hasPoints) {
            btnTop.innerHTML = '⚡ Run k-NN';
            btnTop.disabled = true;
            btnTop.style.background = 'rgba(71, 85, 105, 0.2)';
            btnTop.style.color = '#94a3b8';
            btnTop.style.borderColor = 'rgba(71, 85, 105, 0.4)';
            btnTop.style.opacity = '0.5';
            btnTop.style.cursor = 'not-allowed';
            btnTop.title = 'No dataset staged: Stage or generate a dataset first';
            btnTop.classList.remove('btn-knn-computed');
          } else if (hasKnn) {
            btnTop.innerHTML = `🟢 k-NN (k=${kVal}) • Re-run`;
            btnTop.disabled = false;
            btnTop.style.background = 'rgba(56, 189, 248, 0.2)';
            btnTop.style.color = '#38bdf8';
            btnTop.style.borderColor = 'rgba(56, 189, 248, 0.55)';
            btnTop.style.opacity = '1.0';
            btnTop.style.cursor = 'pointer';
            btnTop.title = `k-NN graph computed (k=${kVal}). Click to re-run k-NN solver.`;
            btnTop.classList.add('btn-knn-computed');
          } else {
            btnTop.innerHTML = '⚡ Run k-NN';
            btnTop.disabled = false;
            btnTop.style.background = 'rgba(14, 165, 233, 0.18)';
            btnTop.style.color = '#38bdf8';
            btnTop.style.borderColor = 'rgba(14, 165, 233, 0.45)';
            btnTop.style.opacity = '1.0';
            btnTop.style.cursor = 'pointer';
            btnTop.title = 'Compute k-Nearest Neighbors graph';
            btnTop.classList.remove('btn-knn-computed');
          }
        }
        if (btnSide) {
          if (!hasPoints) {
            btnSide.innerHTML = '⚡ Run k-NN';
            btnSide.disabled = true;
            btnSide.style.background = 'linear-gradient(135deg, #475569, #334155)';
            btnSide.style.opacity = '0.5';
            btnSide.style.cursor = 'not-allowed';
            btnSide.title = 'No dataset staged: Stage or generate a dataset first';
          } else if (hasKnn) {
            btnSide.innerHTML = `🟢 k-NN (k=${kVal}) • Re-run`;
            btnSide.disabled = false;
            btnSide.style.background = 'linear-gradient(135deg, #0284c7, #0369a1)';
            btnSide.style.opacity = '1.0';
            btnSide.style.cursor = 'pointer';
            btnSide.title = `k-NN graph computed (k=${kVal}). Click to re-run out-of-core solver.`;
          } else {
            btnSide.innerHTML = '⚡ Run k-NN';
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
    window.updateKnnButtonUI = updateKnnButtonUI;

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
            rawDatasetName = selCli ? selCli.value : `${currentBenchmark}.bin`;
          }
          const reExt = /\.(bin|txt|csv|fits|dat|mp4|fits\.fz)$/i;
          const datasetBase = rawDatasetName.replace(reExt, '').replace(/[^a-zA-Z0-9_.-]/g, '_');
          const datasetFile = `${datasetBase}.bin`;
          const clusterDir = `${datasetBase}.clusterdat`;

          // 1. Check workspace files
          const filesInWorkspace = await DesktopBridge.listFiles().catch(() => []);
          const hasDatasetFile = filesInWorkspace.some(
            f => (f.name === datasetFile || f.name === `${datasetBase}.bin`) && f.size > 0
          );
          if (!hasDatasetFile && pts && pts.length > 0) {
            await DesktopBridge.stageDatasetFile(datasetBase, pts).catch(() => {});
          }

          // 2. Check if clusterdat exists; if not, create clusters first
          let hasClusterDir = filesInWorkspace.some(
            f => (f.name === clusterDir || f.name === `${datasetBase}.clusterdat`) &&
                 (f.is_dir || f.isDir)
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
              const clusterArgs = ['-outdir', clusterDir, '0.15', datasetFile];
              if (useGpu) {
                clusterArgs.push('--gpu');
                const bSize = document.getElementById('selectCliGpuBatchSize')?.value;
                if (bSize) clusterArgs.push('--gpu-batch-size', bSize);
              }
              await new Promise((resolve) => {
                DesktopBridge.runCliJob({
                  cmd: 'gric-cluster',
                  args: clusterArgs,
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
          if (typeof knnUseRq8 !== 'undefined' && knnUseRq8) {
            args.push('-rq8');
          } else if (typeof knnUseEq16 !== 'undefined' && knnUseEq16) {
            args.push('-eq16');
            if (typeof knnUseEq16Adc !== 'undefined' && knnUseEq16Adc) {
              args.push('-eq16-adc');
            } else {
              args.push('-no-eq16-adc');
            }
            if (typeof knnSq16Ratio !== 'undefined' && knnSq16Ratio !== 0.05) {
              args.push('-eq16-ratio', String(knnSq16Ratio));
            }
            args.push('-no-sq16', '-no-sq8', '-no-rq8');
          } else if (typeof knnUseSq16 !== 'undefined' && knnUseSq16) {
            args.push('-sq16');
            if (typeof knnUseSq16Sparse !== 'undefined' && knnUseSq16Sparse) {
              args.push('-sq16-sparse');
            }
            if (typeof knnSq16Ratio !== 'undefined') {
              args.push('-sq16-ratio', String(knnSq16Ratio));
            }
            if (typeof knnUseMemo !== 'undefined') {
              args.push(knnUseMemo ? '-memo' : '-no-memo');
            }
          } else if (typeof knnUseSq8 !== 'undefined' && knnUseSq8) {
            args.push('-sq8');
          } else {
            args.push('-no-rq8', '-no-sq16', '-no-sq8', '-no-eq16');
          }
          if (typeof knnUseBatchDist !== 'undefined' && !knnUseBatchDist) {
            args.push('-no-batch-dist');
          }
          if (typeof knnUseClusterGraph !== 'undefined') {
            if (knnUseClusterGraph) {
              args.push('-cluster-graph');
              if (typeof knnEfCluster !== 'undefined' && knnEfCluster >= 0) {
                args.push('-ef-cluster', String(knnEfCluster));
              }
            } else {
              args.push('-no-cluster-graph');
            }
          }
          if (useGpu) {
            args.push('--gpu');
            const bSize = document.getElementById('selectCliGpuBatchSize')?.value;
            if (bSize) args.push('--gpu-batch-size', bSize);
          }
          args.push('-progress');
          args.push('-no-txt');
          args.push('-no-mutual');

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
              useRq8: (typeof knnUseRq8 !== 'undefined' && knnUseRq8),
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
          const timeLoad = parsedTelem.timeLoadMs || 0.0;
          const timeWrite = parsedTelem.timeWriteMs || 0.0;
          const timeNative = (timeCompute > 0)
            ? (timeLoad + timeCompute + timeWrite)
            : 0.0;
          parsedTelem.timeComputeMs = timeCompute;
          parsedTelem.timeLoadMs = timeLoad;
          parsedTelem.timeWriteMs = timeWrite;
          parsedTelem.timeNativeMs = timeNative;
          parsedTelem.timeTotalMs = totalWallMs;
          parsedTelem.timeGuiOverheadMs = Math.max(0, totalWallMs - timeNative);
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
          const nativeNote = timeNative > 0
            ? ` (Native ELF: ${timeNative.toFixed(1)} ms | GUI: ${totalWallMs.toFixed(1)} ms)`
            : ` (GUI Wall: ${totalWallMs.toFixed(1)} ms)`;
          showToast(
            `✅ Native k-NN: Non-UI Compute ${timeCompute.toFixed(1)} ms${nativeNote}`
          );
        } else {
          // WASM / JS Execution
          const config = {
            k: knnK,
            dtmin: knnDtmin,
            direction: knnDirection,
            epsilon: knnEpsilon,
            rlim: knnRlim,
            useRq8: (typeof knnUseRq8 !== 'undefined' && knnUseRq8),
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
            results.telemetry.timeNativeMs = timeCompute;
            results.telemetry.timeTotalMs = totalWallMs;
            results.telemetry.timeGuiOverheadMs = Math.max(0, totalWallMs - timeCompute);
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
              `⚡ k-NN computed: Non-UI Compute ${timeCompute.toFixed(1)} ms ` +
              `(GUI Wall: ${totalWallMs.toFixed(1)} ms)`
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
    const btnKnnSq16 = document.getElementById('btnKnnSq16');
    const btnKnnRq8 = document.getElementById('btnKnnRq8');
    const btnKnnEq16 = document.getElementById('btnKnnEq16');
    const btnKnnEq16Adc = document.getElementById('btnKnnEq16Adc');
    const btnKnnSparse = document.getElementById('btnKnnSparse');

    function updateKnnQuantToggles() {
      if (btnKnnRq8) {
        btnKnnRq8.classList.toggle('toggle-active', knnUseRq8);
        btnKnnRq8.classList.toggle('toggle-cyan', knnUseRq8);
        btnKnnRq8.classList.toggle('active', knnUseRq8);
      }
      if (btnKnnSq16) {
        btnKnnSq16.classList.toggle('toggle-active', knnUseSq16);
        btnKnnSq16.classList.toggle('toggle-cyan', knnUseSq16);
        btnKnnSq16.classList.toggle('active', knnUseSq16);
      }
      if (btnKnnSq8) {
        btnKnnSq8.classList.toggle('toggle-active', knnUseSq8);
        btnKnnSq8.classList.toggle('toggle-cyan', knnUseSq8);
        btnKnnSq8.classList.toggle('active', knnUseSq8);
      }
      if (btnKnnEq16) {
        btnKnnEq16.classList.toggle('toggle-active', knnUseEq16);
        btnKnnEq16.classList.toggle('toggle-cyan', knnUseEq16);
        btnKnnEq16.classList.toggle('active', knnUseEq16);
      }
      if (btnKnnEq16Adc) {
        btnKnnEq16Adc.classList.toggle('toggle-active', knnUseEq16 && knnUseEq16Adc);
        btnKnnEq16Adc.classList.toggle('toggle-cyan', knnUseEq16 && knnUseEq16Adc);
        btnKnnEq16Adc.classList.toggle('active', knnUseEq16 && knnUseEq16Adc);
        btnKnnEq16Adc.style.display = knnUseEq16 ? 'inline-flex' : 'none';
      }
      if (btnKnnSparse) {
        btnKnnSparse.classList.toggle('toggle-active', knnUseSq16Sparse);
        btnKnnSparse.classList.toggle('toggle-cyan', knnUseSq16Sparse);
        btnKnnSparse.classList.toggle('active', knnUseSq16Sparse);
      }
    }

    if (btnKnnRq8) {
      btnKnnRq8.addEventListener('click', () => {
        knnUseRq8 = !knnUseRq8;
        if (knnUseRq8) {
          knnUseSq16 = false;
          knnUseSq8 = false;
          knnUseEq16 = false;
        }
        updateKnnQuantToggles();
        syncControlDependencies();
        updateCliCommand();
        draw();
      });
    }

    if (btnKnnSq8) {
      btnKnnSq8.addEventListener('click', () => {
        knnUseSq8 = !knnUseSq8;
        if (knnUseSq8) {
          knnUseRq8 = false;
          knnUseSq16 = false;
          knnUseEq16 = false;
        }
        updateKnnQuantToggles();
        syncControlDependencies();
        updateCliCommand();
        draw();
      });
    }

    if (btnKnnSq16) {
      btnKnnSq16.addEventListener('click', () => {
        knnUseSq16 = !knnUseSq16;
        if (knnUseSq16) {
          knnUseRq8 = false;
          knnUseSq8 = false;
          knnUseEq16 = false;
        }
        updateKnnQuantToggles();
        syncControlDependencies();
        updateCliCommand();
        draw();
      });
    }

    if (btnKnnEq16) {
      btnKnnEq16.addEventListener('click', () => {
        knnUseEq16 = !knnUseEq16;
        if (knnUseEq16) {
          knnUseRq8 = false;
          knnUseSq8 = false;
          knnUseSq16 = false;
        }
        updateKnnQuantToggles();
        syncControlDependencies();
        updateCliCommand();
        draw();
      });
    }

    if (btnKnnEq16Adc) {
      btnKnnEq16Adc.addEventListener('click', () => {
        knnUseEq16Adc = !knnUseEq16Adc;
        updateKnnQuantToggles();
        updateCliCommand();
        draw();
      });
    }

    updateKnnQuantToggles();

    const btnKnnMemo = document.getElementById('btnKnnMemo');
    if (btnKnnMemo) {
      btnKnnMemo.addEventListener('click', () => {
        knnUseMemo = !knnUseMemo;
        btnKnnMemo.classList.toggle('toggle-active', knnUseMemo);
        btnKnnMemo.classList.toggle('toggle-cyan', knnUseMemo);
        btnKnnMemo.classList.toggle('active', knnUseMemo);
        updateCliCommand();
        draw();
      });
    }

    if (btnKnnSparse) {
      btnKnnSparse.addEventListener('click', () => {
        knnUseSq16Sparse = !knnUseSq16Sparse;
        btnKnnSparse.classList.toggle('toggle-active', knnUseSq16Sparse);
        btnKnnSparse.classList.toggle('toggle-cyan', knnUseSq16Sparse);
        btnKnnSparse.classList.toggle('active', knnUseSq16Sparse);
        updateCliCommand();
        draw();
      });
    }

    const inputKnnSq16RatioEl = document.getElementById('inputKnnSq16Ratio');
    const sliderKnnSq16RatioEl = document.getElementById('sliderKnnSq16Ratio');
    if (inputKnnSq16RatioEl && sliderKnnSq16RatioEl) {
      inputKnnSq16RatioEl.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (!isNaN(val) && val >= 0.005 && val <= 0.50) {
          knnSq16Ratio = val;
          sliderKnnSq16RatioEl.value = val;
          updateCliCommand();
        }
      });
      sliderKnnSq16RatioEl.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        knnSq16Ratio = val;
        inputKnnSq16RatioEl.value = val.toFixed(3);
        updateCliCommand();
      });
    }

    const btnKnnBatchDist = document.getElementById('btnKnnBatchDist');
    if (btnKnnBatchDist) {
      btnKnnBatchDist.addEventListener('click', () => {
        knnUseBatchDist = !knnUseBatchDist;
        btnKnnBatchDist.classList.toggle('toggle-active', knnUseBatchDist);
        btnKnnBatchDist.classList.toggle('toggle-cyan', knnUseBatchDist);
        btnKnnBatchDist.classList.toggle('active', knnUseBatchDist);
        updateCliCommand();
        draw();
      });
    }

    const btnKnnClusterGraph = document.getElementById('btnKnnClusterGraph');
    const rowKnnEfCluster = document.getElementById('rowKnnEfCluster');
    const sliderKnnEfCluster = document.getElementById('sliderKnnEfCluster');
    const inputKnnEfCluster = document.getElementById('inputKnnEfCluster');

    if (btnKnnClusterGraph) {
      btnKnnClusterGraph.addEventListener('click', () => {
        knnUseClusterGraph = !knnUseClusterGraph;
        btnKnnClusterGraph.classList.toggle('toggle-active', knnUseClusterGraph);
        btnKnnClusterGraph.classList.toggle('toggle-cyan', knnUseClusterGraph);
        btnKnnClusterGraph.classList.toggle('active', knnUseClusterGraph);
        if (rowKnnEfCluster) {
          rowKnnEfCluster.style.display = knnUseClusterGraph ? 'flex' : 'none';
        }
        updateCliCommand();
        draw();
      });
    }

    const updateEfClusterDisplay = () => {
      const unitKnnEf = document.getElementById('unitKnnEfCluster');
      if (unitKnnEf) {
        unitKnnEf.textContent = (knnEfCluster === 0) ? 'Auto' : 'ef';
        unitKnnEf.style.color = (knnEfCluster === 0) ? '#34d399' : '';
      }
    };

    if (sliderKnnEfCluster) {
      sliderKnnEfCluster.addEventListener('input', (e) => {
        knnEfCluster = parseInt(e.target.value, 10) || 0;
        if (inputKnnEfCluster) inputKnnEfCluster.value = knnEfCluster;
        updateEfClusterDisplay();
        updateCliCommand();
        draw();
      });
    }

    if (inputKnnEfCluster) {
      inputKnnEfCluster.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        if (!isNaN(v) && v >= 0) {
          knnEfCluster = v;
          if (sliderKnnEfCluster) sliderKnnEfCluster.value = Math.min(250, v);
          updateEfClusterDisplay();
          updateCliCommand();
          draw();
        }
      });
    }

