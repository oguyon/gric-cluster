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
      if (!skipRender || (totalFrames % batchThinRate === 0)) {
        pastSamples.push({ x, y, z });
        if (pastSamples.length > sampleBufferCap) {
          const trimTo = Math.floor(sampleBufferCap * 0.67);
          pastSamples = pastSamples.slice(-trimTo);
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

            const sampleEntry = {
              frameIndex: totalFrames,
              timestamp: performance.now(),
              point: { x, y, z },
              assignedCluster: assigned,
              isNewCluster: false,
              distSC: distSampleClusterLast,
              distCC: distClusterClusterLast,
              evals: snapshot ? snapshot.telemetry.lastFrameDists : 0,
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
          }
        }
        return;
      }
    }

    // =========================================================================
