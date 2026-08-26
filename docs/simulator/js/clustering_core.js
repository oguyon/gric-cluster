/**
 * GRIC Simulator - clustering_core.js
 * Part of the GRIC Interactive Algorithm Simulator
 */

//  5. CORE 2D / 3D CLUSTERING ENGINE
    // =========================================================================

    function clusterFrame(x, y, z = 0.0, skipRender = false) {
      const tComputeStart = performance.now();
      distSampleClusterLast = 0;
      distClusterClusterLast = 0;

      totalFrames++;
      currentFrame = { x, y, z };
      const currentIdx = totalFrames - 1;
      if (!skipRender || (totalFrames % batchThinRate === 0)) {
        if (currentIdx < pastSamples.length) {
          pastSamples[currentIdx].x = x;
          pastSamples[currentIdx].y = y;
          pastSamples[currentIdx].z = z;
        } else if (pastSamples.length < sampleBufferCap) {
          pastSamples.push({ x, y, z, frameIndex: currentIdx, clusterId: -1 });
        }
      }
      currentExplanation = [];

      const coordStr = currentDim === 3 
        ? `(x=${x.toFixed(3)}, y=${y.toFixed(3)}, z=${z.toFixed(3)})`
        : `(x=${x.toFixed(3)}, y=${y.toFixed(3)})`;

      if (useTiles) {
        if (isExplainMode) {
          logExplainStep({
            type: 'target',
            title: `📍 Ingesting Frame #${totalFrames}`,
            text: `Coordinates ${coordStr} partitioned into ${currentDim === 3 ? '3' : '2'} 1D Subspaces.`
          });
        }

        GricWasm.processFrameMultiTile(x, y, z);

        if (!skipRender || isExplainMode) {
          const snapshot = GricWasm.syncMultiTileState();
          if (snapshot) {
            GricWasm.applyMultiTileToJsState(snapshot);
            
            let evalsThis = 0;
            let naiveThis = 0;
            if (snapshot.tiles) {
              snapshot.tiles.forEach(t => {
                evalsThis += t.evals || 0;
                naiveThis += t.clusters ? t.clusters.length : 0;
              });
            }
            totalEvals += evalsThis;
            naiveEvals += naiveThis;

            if (snapshot.lastTuple) {
              const cx = snapshot.lastTuple[0] || 0;
              const cy = snapshot.lastTuple[1] || 0;
              const cz = currentDim === 3 ? (snapshot.lastTuple[2] || 0) : 0;
              currentJointTuple = { cx, cy, cz };
              const tupleKey = currentDim === 3 ? `${cx}_${cy}_${cz}` : `${cx}_${cy}`;
              assignmentHistory.push(tupleKey);
              if (assignmentHistory.length > 5000) assignmentHistory.shift();
            }
            
            if (isExplainMode) {
              if (snapshot.traceSteps) {
                currentExplanation.push(...snapshot.traceSteps);
              }
            }
          }

          const tComputeEnd = performance.now();
          const frameComputeMs = Math.max(0.0001, tComputeEnd - tComputeStart);
          recordFrameTelemetry(frameComputeMs);

          if (!isRunning) {
            updateUI();
            draw();
            if (typeof scheduleActiveFrameCleanup === 'function') {
              scheduleActiveFrameCleanup(800);
            }
          }
        }
        return;
      }

      // WASM engine — always active
      if (wasmSessionActive && GricWasm.isReady()) {
        const assigned = GricWasm.processFrame(x, y, z);

        // MAXCL_STOP: C engine returned -2 → stop
        if (assigned === -2) {
          if (typeof pauseSimulation === 'function') {
            pauseSimulation();
          }
          showToast('🛑 Max cluster limit reached (stop)');
          updateUI();
          draw();
          return;
        }

        // Always capture distance evaluations for every frame (for hover inspector)
        const frameEvals = GricWasm.getFrameEvaluations();
        if (frameEvals && frameEvals.length > 0) {
          frameEvaluationsLog[currentIdx] = frameEvals;
        } else if (assigned >= 0 && clusters[assigned]) {
          const cl = clusters[assigned];
          const dx = x - cl.x;
          const dy = y - cl.y;
          const dz = (currentDim === 3) ? (z - cl.z) : 0;
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
          frameEvaluationsLog[currentIdx] = [{
            clusterId: assigned,
            dist: d,
            match: d <= (rlim || 0.1)
          }];
        }

        if (assigned >= 0) {
          if (currentIdx < pastSamples.length) {
            pastSamples[currentIdx].clusterId = assigned;
          }
          if (prevAssignedCluster >= 0 && transitionCounts[prevAssignedCluster]) {
            transitionCounts[prevAssignedCluster][assigned] =
              (transitionCounts[prevAssignedCluster][assigned] || 0) + 1;
            lastTransitionFrom = prevAssignedCluster;
            lastTransitionTo = assigned;
          } else {
            lastTransitionFrom = -1;
            lastTransitionTo = -1;
          }
          prevAssignedCluster = assigned;

          assignmentHistory.push(assigned);
          if (assignmentHistory.length > 6000) {
            assignmentHistory = assignmentHistory.slice(-5000);
          }
        }

        if (!skipRender || isExplainMode) {
          const snapshot = GricWasm.syncState();
          if (snapshot) {
            GricWasm.applyToJsState(snapshot);
            if (snapshot.numClusters > 0) {
              naiveEvals += snapshot.numClusters;
            }
          }

          if (assigned >= 0) {
            frameHistory.push({
              indices: [assigned],
              dists: [snapshot ? snapshot.telemetry.lastAssignmentDist : 0],
              assignment: assigned
            });
            if (frameHistory.length > 600) {
              frameHistory = frameHistory.slice(-500);
            }

            let traceSteps;
            let traceRankings = [];
            if (isExplainMode) {
              traceSteps = GricWasm.getTrace();
              currentExplanation = [...traceSteps];
              for (let si = traceSteps.length - 1; si >= 0; si--) {
                if (traceSteps[si].entropyRankings) {
                  traceRankings = traceSteps[si].entropyRankings;
                  break;
                }
              }
            } else {
              traceSteps = [{
                type: 'target',
                title: `📍 Sample #${totalFrames} (WASM)`,
                text: `${coordStr} → C${assigned}.`
              }];
            }

            const evList = (snapshot && snapshot.evaluations) ? snapshot.evaluations.map(ev => ({
              clusterId: (ev.target && ev.target.id !== undefined) ? ev.target.id : ev.target,
              dist: ev.dist,
              match: ev.match
            })) : [];

            if (evList.length > 0) {
              frameEvaluationsLog[currentIdx] = evList;
            } else if (assigned >= 0 && clusters[assigned]) {
              const cl = clusters[assigned];
              const dx = x - cl.x;
              const dy = y - cl.y;
              const dz = (currentDim === 3) ? (z - cl.z) : 0;
              const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
              frameEvaluationsLog[currentIdx] = [{
                clusterId: assigned,
                dist: d,
                match: d <= (rlim || 0.1)
              }];
            }

            const sampleEntry = {
              frameIndex: totalFrames,
              timestamp: performance.now(),
              point: { x, y, z },
              assignedCluster: assigned,
              isNewCluster: false,
              distSC: distSampleClusterLast,
              distCC: distClusterClusterLast,
              evals: snapshot ? snapshot.telemetry.lastFrameDists : 0,
              evaluations: evList,
              initialEntropy: 0,
              entropyReduced: 0,
              steps: traceSteps,
              entropyRankings: traceRankings
            };
            sampleTraceLog.push(sampleEntry);
            if (sampleTraceLog.length > MAX_SAMPLE_TRACE_HISTORY) {
              sampleTraceLog = sampleTraceLog.slice(-MAX_SAMPLE_TRACE_HISTORY);
            }
          }

          const tComputeEnd = performance.now();
          const frameComputeMs = Math.max(0.0001, tComputeEnd - tComputeStart);
          recordFrameTelemetry(frameComputeMs);

          if (!isRunning) {
            updateUI();
            draw();
            if (typeof scheduleActiveFrameCleanup === 'function') {
              scheduleActiveFrameCleanup(800);
            }
          }
        }
        return;
      }
    }

    /**
     * Process a raster image frame vector through the WASM clustering engine.
     * @param {ArrayLike<number>} pixels - Image pixel buffer (W * H)
     * @param {boolean} skipRender - If true, skip UI update / rendering
     */
    function clusterImageFrame(pixels, skipRender = false) {
      const tComputeStart = performance.now();
      distSampleClusterLast = 0;
      distClusterClusterLast = 0;
      currentExplanation = [];

      if (useWasm && wasmSessionActive && GricWasm.isReady()) {
        const assigned = GricWasm.processFrameVector(pixels);

        if (assigned === -2) {
          if (typeof pauseSimulation === 'function') {
            pauseSimulation();
          }
          showToast('🛑 Max cluster limit reached (stop)');
          updateUI();
          draw();
          return;
        }

        if (assigned >= 0) {
          if (prevAssignedCluster >= 0 && transitionCounts[prevAssignedCluster]) {
            transitionCounts[prevAssignedCluster][assigned] =
              (transitionCounts[prevAssignedCluster][assigned] || 0) + 1;
            lastTransitionFrom = prevAssignedCluster;
            lastTransitionTo = assigned;
          } else {
            lastTransitionFrom = -1;
            lastTransitionTo = -1;
          }
          prevAssignedCluster = assigned;

          assignmentHistory.push(assigned);
          if (assignmentHistory.length > 6000) {
            assignmentHistory = assignmentHistory.slice(-5000);
          }
        }

        if (!skipRender) {
          const snapshot = GricWasm.syncState();
          if (snapshot) {
            GricWasm.applyToJsState(snapshot);
            if (snapshot.numClusters > 0) {
              naiveEvals += snapshot.numClusters;
            }
          }

          if (assigned >= 0) {
            frameHistory.push({
              indices: [assigned],
              dists: [snapshot ? snapshot.telemetry.lastAssignmentDist : 0],
              assignment: assigned
            });
            if (frameHistory.length > 600) {
              frameHistory = frameHistory.slice(-500);
            }
          }

          const tComputeEnd = performance.now();
          const frameComputeMs = Math.max(0.0001, tComputeEnd - tComputeStart);
          recordFrameTelemetry(frameComputeMs);

          if (!isRunning) {
            updateUI();
            draw();
          }
        }
      }
    }

    /**
     * Reassign all points in pastSamples / active dataset to their closest cluster anchor.
     * Uses triangle inequality bounds against the DCC matrix where available.
     */
    function runSecondPassClustering() {
      if (!clusters || clusters.length <= 1) {
        showToast('ℹ️ Need at least 2 clusters to run 2nd pass reassignment');
        return;
      }
      if (!pastSamples || pastSamples.length === 0) {
        showToast('ℹ️ No points available to reassign');
        return;
      }

      const tStart = performance.now();
      const K = clusters.length;
      const N = pastSamples.length;
      let reassignedCount = 0;
      let distEvals = 0;
      let distPruned = 0;

      // Reset cluster member counts
      clusters.forEach(c => { c.members = 0; });

      // Reallocate each sample
      for (let i = 0; i < N; i++) {
        const pt = pastSamples[i];
        let bestK = pt.clusterId;
        let dBest = 1e30;

        if (bestK >= 0 && bestK < K) {
          const anc = clusters[bestK];
          const dx = pt.x - anc.x;
          const dy = pt.y - anc.y;
          const dz = (pt.z || 0) - (anc.z || 0);
          dBest = Math.sqrt(dx * dx + dy * dy + dz * dz);
          distEvals++;
        }

        for (let k = 0; k < K; k++) {
          if (k === bestK) continue;

          // Triangle inequality lower bound check if D_cc available
          if (dcc && dcc[bestK] && dcc[bestK][k] !== undefined && dcc[bestK][k] >= 0) {
            const dccDist = dcc[bestK][k];
            const lb = Math.abs(dBest - dccDist);
            if (lb >= dBest) {
              distPruned++;
              continue;
            }
          }

          const anc = clusters[k];
          const dx = pt.x - anc.x;
          const dy = pt.y - anc.y;
          const dz = (pt.z || 0) - (anc.z || 0);
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
          distEvals++;

          if (d < dBest) {
            dBest = d;
            bestK = k;
          }
        }

        if (bestK !== pt.clusterId) {
          reassignedCount++;
          pt.clusterId = bestK;
        }

        if (bestK >= 0 && bestK < K) {
          clusters[bestK].members++;
        }
      }

      const tElapsed = performance.now() - tStart;
      const pct = N > 0 ? ((reassignedCount / N) * 100).toFixed(1) : 0;
      showToast(`🔄 2nd Pass: ${reassignedCount} / ${N} pts reassigned (${pct}%) in ${tElapsed.toFixed(1)}ms`);

      updateUI();
      draw();
    }

    // =========================================================================
