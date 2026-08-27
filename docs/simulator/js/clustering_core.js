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

      if (useWasm && GricWasm.isLoaded()) {
        const params = GricWasm.buildParamsFromState();
        if (!wasmSessionActive || !GricWasm.isReady() ||
            (GricWasm.isConfigChanged && GricWasm.isConfigChanged(params))) {
          wasmSessionActive = GricWasm.init(params);
          if (typeof updateWasmBadge === 'function') {
            updateWasmBadge();
          }
        }
      }

      // WASM engine
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

        // Fast cluster ID computation
        const actualClusterId = (assigned >= 0)
          ? assigned
          : Math.max(0, GricWasm.getNumClusters() - 1);

        // ALWAYS update sample assignment so points are colored per cluster!
        if (actualClusterId >= 0) {
          if (currentIdx < pastSamples.length) {
            pastSamples[currentIdx].clusterId = actualClusterId;
          } else if (pastSamples.length < sampleBufferCap) {
            pastSamples.push({
              x, y, z,
              frameIndex: currentIdx,
              clusterId: actualClusterId
            });
          }

          if (typeof benchmarkDataset !== 'undefined' &&
              benchmarkDataset && currentIdx < benchmarkDataset.length) {
            benchmarkDataset[currentIdx].clusterId = actualClusterId;
          }

          if (prevAssignedCluster >= 0) {
            if (!transitionCounts[prevAssignedCluster]) {
              transitionCounts[prevAssignedCluster] = [];
            }
            transitionCounts[prevAssignedCluster][actualClusterId] =
              (transitionCounts[prevAssignedCluster][actualClusterId] || 0) + 1;
            lastTransitionFrom = prevAssignedCluster;
            lastTransitionTo = actualClusterId;
          } else {
            lastTransitionFrom = -1;
            lastTransitionTo = -1;
          }
          prevAssignedCluster = actualClusterId;
        }

        // Full state sync and trace only when rendering frame or in explain mode
        if (!skipRender || isExplainMode) {
          const snapshot = GricWasm.syncState();
          if (snapshot) {
            GricWasm.applyToJsState(snapshot);
            if (snapshot.numClusters > 0) {
              naiveEvals += snapshot.numClusters;
            }
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
              text: (assigned >= 0)
                ? `${coordStr} → C${assigned}.`
                : `${coordStr} → ✨ Spawned Cluster C${actualClusterId}.`
            }];
          }

          const evList = (snapshot && snapshot.evaluations) ? snapshot.evaluations.map(ev => ({
            clusterId: (ev.target && ev.target.id !== undefined) ? ev.target.id : ev.target,
            dist: ev.dist,
            match: ev.match
          })) : [];

          if (evList.length > 0) {
            frameEvaluationsLog[currentIdx] = evList;
          }

          const sampleEntry = {
            frameIndex: totalFrames,
            timestamp: performance.now(),
            point: { x, y, z },
            assignedCluster: actualClusterId,
            isNewCluster: (assigned < 0),
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
      } else {
        // Pure JS 2D/3D clustering fallback
        let bestDist = Infinity;
        let bestCluster = -1;
        const rlimSq = (rlim || 0.1) * (rlim || 0.1);
        for (let k = 0; k < clusters.length; k++) {
          const c = clusters[k];
          const dx = x - c.x;
          const dy = y - c.y;
          const dz = (currentDim === 3) ? (z - c.z) : 0.0;
          const dSq = dx * dx + dy * dy + dz * dz;
          if (dSq < bestDist) {
            bestDist = dSq;
            bestCluster = k;
          }
        }
        const actualDist = Math.sqrt(bestDist < Infinity ? bestDist : 0.0);
        let actualClusterId = -1;
        if (bestCluster >= 0 && actualDist <= (rlim || 0.1)) {
          actualClusterId = bestCluster;
          clusters[actualClusterId].members++;
          clusters[actualClusterId].lastActive = totalFrames;
        } else {
          actualClusterId = clusters.length;
          const col = (typeof getClusterColor === 'function')
            ? getClusterColor(actualClusterId)
            : '#38bdf8';
          clusters.push({
            id: actualClusterId,
            x, y, z,
            members: 1,
            color: col,
            lastActive: totalFrames
          });
        }
        if (actualClusterId >= 0) {
          if (currentIdx < pastSamples.length) {
            pastSamples[currentIdx].clusterId = actualClusterId;
          } else if (pastSamples.length < sampleBufferCap) {
            pastSamples.push({ x, y, z, frameIndex: currentIdx, clusterId: actualClusterId });
          }
          prevAssignedCluster = actualClusterId;
        }
        distSampleClusterLast = actualDist;
        if (!skipRender && !isRunning) {
          updateUI();
          draw();
        }
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

      totalFrames++;
      currentImageFrame = pixels;
      const currentIdx = totalFrames - 1;

      if (useWasm && GricWasm.isLoaded()) {
        const params = GricWasm.buildParamsFromState();
        if (!wasmSessionActive || !GricWasm.isReady() ||
            (GricWasm.isConfigChanged && GricWasm.isConfigChanged(params))) {
          wasmSessionActive = GricWasm.init(params);
          if (typeof updateWasmBadge === 'function') {
            updateWasmBadge();
          }
        }
      }

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

        const actualClusterId = (assigned >= 0)
          ? assigned
          : Math.max(0, GricWasm.getNumClusters() - 1);

        imageFrameAssignments[currentIdx] = actualClusterId;
        if (!imageClusterMembers[actualClusterId]) {
          imageClusterMembers[actualClusterId] = [];
        }
        imageClusterMembers[actualClusterId].push(currentIdx);

        if (typeof benchmarkDataset !== 'undefined' &&
            benchmarkDataset && currentIdx < benchmarkDataset.length) {
          if (benchmarkDataset[currentIdx]) {
            benchmarkDataset[currentIdx].clusterId = actualClusterId;
          }
        }

        if (actualClusterId >= 0) {
          if (prevAssignedCluster >= 0) {
            if (!transitionCounts[prevAssignedCluster]) {
              transitionCounts[prevAssignedCluster] = [];
            }
            transitionCounts[prevAssignedCluster][actualClusterId] =
              (transitionCounts[prevAssignedCluster][actualClusterId] || 0) + 1;
            lastTransitionFrom = prevAssignedCluster;
            lastTransitionTo = actualClusterId;
          } else {
            lastTransitionFrom = -1;
            lastTransitionTo = -1;
          }
          prevAssignedCluster = actualClusterId;

          assignmentHistory.push(actualClusterId);
          if (assignmentHistory.length > 6000) {
            assignmentHistory = assignmentHistory.slice(-5000);
          }
        }

        // Compute distance to assigned anchor
        let frameDist = 0.0;
        if (clusters[actualClusterId] && clusters[actualClusterId].anchor) {
          const anch = clusters[actualClusterId].anchor;
          let sumSq = 0.0;
          const len = Math.min(pixels.length, anch.length);
          for (let p = 0; p < len; p++) {
            const diff = pixels[p] - anch[p];
            sumSq += diff * diff;
          }
          frameDist = Math.sqrt(sumSq);
        }
        imageFrameDists[currentIdx] = frameDist;
        distSampleClusterLast = frameDist;

        if (!skipRender) {
          const snapshot = GricWasm.syncState();
          if (snapshot) {
            GricWasm.applyToJsState(snapshot);
            if (snapshot.numClusters > 0) {
              naiveEvals += snapshot.numClusters;
            }
          }

          if (actualClusterId >= 0) {
            frameHistory.push({
              indices: [actualClusterId],
              dists: [frameDist || (snapshot ? snapshot.telemetry.lastAssignmentDist : 0)],
              assignment: actualClusterId
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
      } else {
        // Pure JS Image Clustering fallback:
        let bestDist = Infinity;
        let bestCluster = -1;
        const rlimSq = (rlim || 0.1) * (rlim || 0.1);
        const len = pixels.length;

        for (let k = 0; k < clusters.length; k++) {
          const anch = clusters[k].anchor;
          if (!anch) continue;
          let sumSq = 0.0;
          for (let p = 0; p < len; p++) {
            const diff = pixels[p] - anch[p];
            sumSq += diff * diff;
            if (sumSq > rlimSq && bestDist < Infinity) break;
          }
          if (sumSq < bestDist) {
            bestDist = sumSq;
            bestCluster = k;
          }
        }

        const actualDist = Math.sqrt(bestDist < Infinity ? bestDist : 0.0);
        let actualClusterId = -1;

        if (bestCluster >= 0 && actualDist <= (rlim || 0.1)) {
          actualClusterId = bestCluster;
          clusters[actualClusterId].members++;
          clusters[actualClusterId].lastActive = totalFrames;
        } else {
          // Spawn new cluster
          actualClusterId = clusters.length;
          const col = (typeof getClusterColor === 'function')
            ? getClusterColor(actualClusterId)
            : '#38bdf8';
          const anchCopy = new Float64Array(len);
          for (let p = 0; p < len; p++) anchCopy[p] = pixels[p];
          clusters.push({
            id: actualClusterId,
            anchor: anchCopy,
            members: 1,
            color: col,
            lastActive: totalFrames
          });
        }

        imageFrameAssignments[currentIdx] = actualClusterId;
        if (!imageClusterMembers[actualClusterId]) {
          imageClusterMembers[actualClusterId] = [];
        }
        imageClusterMembers[actualClusterId].push(currentIdx);
        imageFrameDists[currentIdx] = actualDist;
        distSampleClusterLast = actualDist;
        prevAssignedCluster = actualClusterId;
        if (!skipRender && !isRunning) {
          updateUI();
          draw();
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
