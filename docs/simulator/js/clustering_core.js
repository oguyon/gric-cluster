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

      // MULTI-TILE SUBSPACE MODE
      if (useTiles) {
        logExplainStep({
          type: 'target',
          title: `📍 Ingesting Frame #${totalFrames}`,
          text: `Coordinates ${coordStr} partitioned into ${currentDim === 3 ? '3' : '2'} 1D Subspaces.`
        });

        tileTraceX = tileEngineX.cluster1D(x, rlim, isExplainMode);
        tileTraceY = tileEngineY.cluster1D(y, rlim, isExplainMode);
        if (currentDim === 3) {
          tileTraceZ = tileEngineZ.cluster1D(z, rlim, isExplainMode);
        }

        const evalsThis = tileTraceX.evals + tileTraceY.evals + (currentDim === 3 ? tileTraceZ.evals : 0);
        const naiveThis = tileEngineX.clusters.length + tileEngineY.clusters.length + (currentDim === 3 ? tileEngineZ.clusters.length : 0);

        totalEvals += evalsThis;
        naiveEvals += naiveThis;

        if (isExplainMode) {
          if (tileTraceX.steps) currentExplanation.push(...tileTraceX.steps);
          if (tileTraceY.steps) currentExplanation.push(...tileTraceY.steps);
          if (currentDim === 3 && tileTraceZ.steps) currentExplanation.push(...tileTraceZ.steps);
        }

        const cx = tileTraceX.assigned;
        const cy = tileTraceY.assigned;
        const cz = currentDim === 3 ? tileTraceZ.assigned : 0;
        currentJointTuple = { cx, cy, cz };

        const tupleKey = currentDim === 3 ? `${cx}_${cy}_${cz}` : `${cx}_${cy}`;
        if (!jointTuplesMap.has(tupleKey)) {
          jointTuplesMap.set(tupleKey, { cx, cy, cz, count: 1, lastActive: totalFrames });
        } else {
          const entry = jointTuplesMap.get(tupleKey);
          entry.count++;
          entry.lastActive = totalFrames;
        }

        logExplainStep({
          type: 'match',
          title: `🎯 Joint Tuple Assigned: ${currentDim === 3 ? `(${cx}, ${cy}, ${cz})` : `(${cx}, ${cy})`}`,
          text: `Combined 1D assignments into Joint State. Total unique joint states: ${jointTuplesMap.size}.`
        });

        assignmentHistory.push(tupleKey);
        if (assignmentHistory.length > 5000) assignmentHistory.shift();

        const tComputeEnd = performance.now();
        const frameComputeMs = Math.max(0.0001, tComputeEnd - tComputeStart);
        recordFrameTelemetry(frameComputeMs);

        if (!skipRender && !isRunning) {
          updateUI();
          draw();
        }
        return;
      }

      // WASM FAST PATH — route through C/WASM engine
      // Conditions: WASM loaded, session active, not tile mode
      if (useWasm && wasmSessionActive && GricWasm.isReady()) {
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

        // Minimal JS bookkeeping (cheap — no WASM heap reads)
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

        // Full state sync only when rendering (expensive: O(K²) getValue calls)
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

            // Build trace steps for sample history
            let traceSteps;
            let traceRankings = [];
            if (isExplainMode) {
              traceSteps = GricWasm.getTrace();
              currentExplanation = [...traceSteps];
              // Extract entropy rankings from last
              // TARGET_SELECTED step if present
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

      // MONOLITHIC JS PIPELINE (2D or 3D) — fallback when WASM is not active
      currentEvaluations = [];
      currentPruned = [];
      currentPredicted = [];

      const K = clusters.length;

      // Base Case: First Frame
      if (K === 0) {
        const c0 = {
          id: 0,
          x, y, z,
          members: 1,
          prob: 1.0,
          scDists: 1,
          lastActive: totalFrames,
          color: getClusterColor(0)
        };
        clusters.push(c0);
        dcc = [[0.0]];
        transitionCounts = [[0]];
        prevAssignedCluster = 0;
        assignmentHistory.push(0);
        frameHistory.push({ indices: [0], dists: [0.0], assignment: 0 });

        logExplainStep({
          type: 'new-cluster',
          title: '✨ Initial Cluster Anchor Created',
          text: `First frame ingested at ${coordStr}. Initialized cluster C0 with anchor.`
        });

        const tComputeEnd = performance.now();
        const frameComputeMs = Math.max(0.0001, tComputeEnd - tComputeStart);
        recordFrameTelemetry(frameComputeMs);

        if (!skipRender && !isRunning) {
          updateUI();
          draw();
        }
        return;
      }

      naiveEvals += K;
      ensureScratchCapacity(K);
      scratchClMembFlag.fill(1, 0, K);
      const clmembflag = scratchClMembFlag;

      // Step 2: Trajectory prediction candidates (-pred)
      let predCandidates = [];
      if (usePred && assignmentHistory.length >= 5) {
        const predLen = 4;
        const pattern = assignmentHistory.slice(-predLen);
        const matchCounts = new Array(K).fill(0);

        for (let i = 0; i <= assignmentHistory.length - predLen - 1; i++) {
          let matched = true;
          for (let j = 0; j < predLen; j++) {
            if (assignmentHistory[i + j] !== pattern[j]) {
              matched = false;
              break;
            }
          }
          if (matched) {
            for (let h = 1; h <= predHorizon; h++) {
              const nextCl = assignmentHistory[i + predLen + h - 1];
              if (nextCl >= 0 && nextCl < K) matchCounts[nextCl] += (predHorizon - h + 1);
            }
          }
        }

        const candList = [];
        for (let i = 0; i < K; i++) {
          if (matchCounts[i] > 0 && clmembflag[i]) {
            candList.push({ id: i, count: matchCounts[i] });
          }
        }
        candList.sort((a, b) => b.count - a.count);
        predCandidates = candList.map(c => c.id).slice(0, Math.min(3, predHorizon + 1));
      }

      // Step 3a: Prior Probability Calculation (matches C compute_priors_and_mixing.c)
      let sumProb = 0;
      for (let i = 0; i < K; i++) sumProb += (clusters[i].prob || 1.0);
      if (sumProb > 0) {
        for (let i = 0; i < K; i++) {
          clusters[i].prob = (clusters[i].prob || 1.0) / sumProb;
        }
      }

      const pBase = scratchPBase;
      for (let i = 0; i < K; i++) pBase[i] = clusters[i].prob;

      // Transition matrix mixing (-tm <val>)
      if (useTM && prevAssignedCluster >= 0 && transitionCounts[prevAssignedCluster]) {
        let tSum = 0;
        for (let i = 0; i < K; i++) tSum += transitionCounts[prevAssignedCluster][i] || 0;
        if (tSum > 0) {
          const alpha = tmMixingCoeff;
          for (let i = 0; i < K; i++) {
            const pTrans = (transitionCounts[prevAssignedCluster][i] || 0) / tSum;
            pBase[i] = (1.0 - alpha) * pBase[i] + alpha * pTrans;
          }
        }
      }

      const currentGprobs = scratchCurrentGprobs;
      currentGprobs.fill(1.0, 0, K);
      const pCurrent = scratchPCurrent;
      for (let i = 0; i < K; i++) pCurrent[i] = pBase[i];
      const measuredIndices = [];
      const measuredDists = [];

      let assignedCluster = -1;
      let evalsThisFrame = 0;
      let measDepth = 0;

      logExplainStep({
        type: 'target',
        title: `📍 Ingesting Frame #${totalFrames}`,
        text: `Query frame at ${coordStr} against ${K} candidate clusters (rlim=${rlim.toFixed(3)}).`
      });

      while (true) {
        let activeSum = 0;
        let activeCount = 0;
        for (let i = 0; i < K; i++) {
          if (clmembflag[i]) {
            pCurrent[i] = pBase[i] * (useGprob ? currentGprobs[i] : 1.0);
            activeSum += pCurrent[i];
            activeCount++;
          } else {
            pCurrent[i] = 0;
          }
        }

        if (activeCount === 0) break;

        if (activeSum > 0) {
          for (let i = 0; i < K; i++) {
            if (clmembflag[i]) pCurrent[i] /= activeSum;
          }
        }

        let H = 0.0;
        for (let i = 0; i < K; i++) {
          if (clmembflag[i] && pCurrent[i] > 1e-12) {
            H -= pCurrent[i] * Math.log2(pCurrent[i]);
          }
        }
        if (measDepth === 0) {
          currentEntropyBits = H;
          lastInitialEntropy = H;
          totalInitialEntropyBits += H;
          if (H > maxInitialEntropyObserved) maxInitialEntropyObserved = H;
        }

        let chosenTarget = -1;
        let selectionReason = "";
        let stepEntropyRankings = null;

        // Prediction shortcuts first
        while (predCandidates.length > 0) {
          const cand = predCandidates.shift();
          if (clmembflag[cand]) {
            chosenTarget = cand;
            currentPredicted.push(cand);
            selectionReason = `Trajectory pattern predictor (-pred): matched recent temporal cluster transition history`;
            break;
          }
        }

        // Entropy or Greedy Selection
        if (chosenTarget === -1) {
          const activeIndices = [];
          for (let i = 0; i < K; i++) {
            if (clmembflag[i]) activeIndices.push(i);
          }

          const currentGate = (measDepth === 0) ? entropyFirstGate : entropyGate;

          if (targetMode === 'entropy' && activeIndices.length > 1 && H > currentGate) {
            let maxP = -1;
            let argMaxP = -1;
            for (let i of activeIndices) {
              if (pCurrent[i] > maxP) {
                maxP = pCurrent[i];
                argMaxP = i;
              }
            }

            // Dominant Leader Short-Circuit Option (e.g. P > 0.50)
            if (entropyLeaderShortcut && maxP >= entropyLeaderCutoff) {
              chosenTarget = argMaxP;
              selectionReason = `Dominant leader shortcut (P(C${argMaxP}) = ${maxP.toFixed(3)} ≥ ${entropyLeaderCutoff.toFixed(2)}): bypassed entropy calculation to measure clear favorite directly`;
              if (measDepth === 0) totalEntropyGated++;
            } else {
              if (measDepth === 0) totalEntropyEvals++;
              let bestCi = -1;
              let minExpectedH = 1e30;
              const twoRlim = 2.0 * rlim;
              const dynamicMinProb = Math.max(1e-5, maxP * 0.01);

              // Precompute p * log2(p) for active candidates
              const plogp = new Float64Array(K);
              for (let i of activeIndices) {
                const p = pCurrent[i];
                plogp[i] = (p > 1e-12) ? p * Math.log2(p) : 0.0;
              }

              // Top candidate targets to evaluate (in C: top M based on probability and limit)
              let targetsToEval = activeIndices;
              if (activeIndices.length > 24) {
                const sorted = [...activeIndices].sort((a, b) => pCurrent[b] - pCurrent[a]);
                targetsToEval = sorted.slice(0, 24);
              }

              const numA = activeIndices.length;
              let sampledHypotheses = activeIndices;
              if (numA > 32) {
                sampledHypotheses = [];
                const step = numA / 32.0;
                for (let s = 0; s < 32; s++) {
                  sampledHypotheses.push(activeIndices[Math.floor(s * step)]);
                }
              }
              const numH = sampledHypotheses.length;
              const evalRankings = [];

              if (entropyFastMode) {
                // Popcount surrogate mode (-entropy_fast in C)
                let minExpectedSupport = 1e30;
                for (let tc of targetsToEval) {
                  const dcc_tc = dcc[tc];
                  let expectedSupport = 0.0;
                  for (let h = 0; h < numH; h++) {
                    const hj = sampledHypotheses[h];
                    const p_hj = pCurrent[hj];
                    if (p_hj < dynamicMinProb) continue;
                    const d_ij = dcc_tc[hj];
                    const minD = d_ij - twoRlim;
                    const maxD = d_ij + twoRlim;

                    let survivingCount = 0;
                    for (let k = 0; k < numA; k++) {
                      const hk = activeIndices[k];
                      const d_ik = dcc_tc[hk];
                      if (d_ik >= minD && d_ik <= maxD) {
                        survivingCount++;
                      }
                    }
                    expectedSupport += p_hj * survivingCount;
                  }
                  evalRankings.push({
                    id: tc,
                    p: pCurrent[tc],
                    expectedH: expectedSupport,
                    infoGain: Math.max(0, numA - expectedSupport),
                    isSupport: true
                  });
                  if (clusters[tc]) {
                    clusters[tc].infoGain = Math.max(0, numA - expectedSupport);
                  }
                  if (expectedSupport < minExpectedSupport) {
                    minExpectedSupport = expectedSupport;
                    bestCi = tc;
                  }
                }
                chosenTarget = (bestCi !== -1) ? bestCi : getArgMaxP(pCurrent, clmembflag, K);
                selectionReason = `Popcount surrogate (-entropy_fast): candidate C${chosenTarget} minimizes expected surviving candidate support to ${minExpectedSupport.toFixed(1)} clusters (current H=${H.toFixed(2)} > gate ${currentGate.toFixed(1)})`;
              } else {
                // Full Expected Shannon Entropy Minimization (C Reference)
                for (let tc of targetsToEval) {
                  const dcc_tc = dcc[tc];
                  let expectedH = 0.0;

                  for (let h = 0; h < numH; h++) {
                    const hj = sampledHypotheses[h];
                    const p_hj = pCurrent[hj];
                    if (p_hj < dynamicMinProb) continue;

                    const d_ij = dcc_tc[hj];
                    const minD = d_ij - twoRlim;
                    const maxD = d_ij + twoRlim;

                    let hypoSum = 0.0;
                    let plogpSum = 0.0;

                    for (let k = 0; k < numA; k++) {
                      const hk = activeIndices[k];
                      const d_ik = dcc_tc[hk];
                      if (d_ik >= minD && d_ik <= maxD) {
                        hypoSum += pCurrent[hk];
                        plogpSum += plogp[hk];
                      }
                    }

                    if (hypoSum > 0.0) {
                      expectedH += p_hj * (Math.log2(hypoSum) - (plogpSum / hypoSum));
                    }
                  }

                  evalRankings.push({
                    id: tc,
                    p: pCurrent[tc],
                    expectedH: expectedH,
                    infoGain: Math.max(0, H - expectedH),
                    isSupport: false
                  });
                  if (clusters[tc]) {
                    clusters[tc].infoGain = Math.max(0, H - expectedH);
                  }

                  if (expectedH < minExpectedH) {
                    minExpectedH = expectedH;
                    bestCi = tc;
                  }
                }
                chosenTarget = (bestCi !== -1) ? bestCi : getArgMaxP(pCurrent, clmembflag, K);
                selectionReason = `Shannon entropy minimization (-entropy): candidate C${chosenTarget} provides lowest expected residual uncertainty E[H]=${minExpectedH.toFixed(2)} bits (current H=${H.toFixed(2)} > gate ${currentGate.toFixed(1)}) to maximize anticipated geometric pruning`;
              }
              evalRankings.forEach(r => { r.isChosen = (r.id === chosenTarget); });
              evalRankings.sort((a, b) => a.expectedH - b.expectedH);
              stepEntropyRankings = evalRankings;
              lastEntropyRankings = evalRankings;
            }
          } else {
            // Greedy Prior Selection (argmax P) or Gated
            if (targetMode === 'entropy' && measDepth === 0) totalEntropyGated++;
            chosenTarget = getArgMaxP(pCurrent, clmembflag, K);
            const tCount = (prevAssignedCluster >= 0 && transitionCounts[prevAssignedCluster]) ? (transitionCounts[prevAssignedCluster][chosenTarget] || 0) : 0;
            if (useTM && tCount > 0) {
              selectionReason = `Markov transition prior (-tm): transition probability P(C${chosenTarget} | C${prevAssignedCluster}) = ${(pCurrent[chosenTarget] || 0).toFixed(3)} (${tCount} observed transitions from previous anchor C${prevAssignedCluster})`;
            } else if (targetMode === 'entropy') {
              selectionReason = `Entropy gating: candidate distribution uncertainty (H=${H.toFixed(2)} bits) ≤ gate (${currentGate.toFixed(1)} bits) — prioritized highest likelihood candidate C${chosenTarget} (P=${(pCurrent[chosenTarget] || 0).toFixed(3)})`;
            } else {
              selectionReason = `Greedy prior target selection: highest active likelihood P=${(pCurrent[chosenTarget] || 0).toFixed(3)} (${clusters[chosenTarget].members} historical members)`;
            }
          }
        }

        if (chosenTarget === -1) break;

        evalsThisFrame++;
        measDepth++;
        distSampleCluster++;
        distSampleClusterLast++;

        const targetCl = clusters[chosenTarget];
        targetCl.scDists = (targetCl.scDists || 0) + 1;
        
        // Exact Euclidean distance in 2D or 3D (squared distance check)
        const dx = x - targetCl.x;
        const dy = y - targetCl.y;
        const dz = currentDim === 3 ? (z - targetCl.z) : 0.0;
        const dfcSq = dx * dx + dy * dy + dz * dz;
        const isMatch = dfcSq <= rlim * rlim;
        const dfc = Math.sqrt(dfcSq);

        measuredIndices.push(chosenTarget);
        measuredDists.push(dfc);
        currentEvaluations.push({ target: targetCl, dist: dfc, match: isMatch });

        logExplainStep({
          type: 'target',
          title: `🔍 Measuring Distance: Anchor C${chosenTarget}`,
          text: `<b>Reason for choice:</b> ${selectionReason}.<br>Anchor coordinates: (${targetCl.x.toFixed(3)}, ${targetCl.y.toFixed(3)}${currentDim === 3 ? `, ${targetCl.z.toFixed(3)}` : ''}).`,
          entropyRankings: stepEntropyRankings,
          currentH: H
        });

        if (isMatch) {
          assignedCluster = chosenTarget;
          targetCl.members++;
          targetCl.lastActive = totalFrames;

          if (currentPredicted.includes(chosenTarget) || (useTM && prevAssignedCluster >= 0 && (transitionCounts[prevAssignedCluster][chosenTarget] || 0) > 0)) {
            predHitCount++;
          }

          logExplainStep({
            type: 'match',
            title: `🎯 Match Found on Anchor C${chosenTarget}`,
            text: `Distance to anchor C${chosenTarget} is d = ${dfc.toFixed(4)} <= rlim (${rlim.toFixed(3)}). Query frame assigned to cluster C${chosenTarget}.<br><b>Anchor choice reason:</b> ${selectionReason}.<br>Resolved in ${evalsThisFrame} distance calculation(s), saving ${K - evalsThisFrame} distance calls.`
          });
          break;
        }

        logExplainStep({
          type: 'mismatch',
          title: `❌ Mismatch on Anchor C${chosenTarget}`,
          text: `Distance to anchor C${chosenTarget} is d = ${dfc.toFixed(4)} > rlim (${rlim.toFixed(3)}). Anchor C${chosenTarget} excluded.<br><b>Anchor choice reason:</b> ${selectionReason}.`
        });

        clmembflag[chosenTarget] = false;

        // Soft Bayesian continuous likelihood decay (-soft_bayesian)
        if (useSoftBayesian) {
          const sigmaBayes = softBayesianSigmaCoeff * rlim;
          for (let cl = 0; cl < K; cl++) {
            if (!clmembflag[cl]) continue;
            const d_cc = dcc[chosenTarget][cl];
            const dr = Math.abs(dfc - d_cc);
            if (dr > rlim) {
              const decay = Math.exp(-((dr - rlim) ** 2) / (2.0 * sigmaBayes * sigmaBayes));
              pCurrent[cl] *= decay;
            }
          }
        }

        // 1. Triangle Inequality (3-Point Pruning)
        const before3P = countActive(clmembflag, K);
        for (let cl = 0; cl < K; cl++) {
          if (!clmembflag[cl]) continue;
          const d_cc = dcc[chosenTarget][cl];
          if (Math.abs(dfc - d_cc) > rlim) {
            clmembflag[cl] = false;
            currentPruned.push({ cluster: clusters[cl], reason: '3P' });
          }
        }
        const after3P = countActive(clmembflag, K);
        const pruned3P = before3P - after3P;
        if (pruned3P > 0) pruneCount3P += pruned3P;

        logExplainStep({
          type: 'prune',
          title: '📐 3-Point Triangle Inequality Pruning',
          text: `3-pt pruning: ${after3P} clusters kept, ${pruned3P} excluded (${((pruned3P / (before3P || 1)) * 100).toFixed(0)}% pruned).`
        });

        // 2. 4-Point Pruning (-te4)
        if (pruneMode === '4P' || pruneMode === '5P') {
          const mCount = measuredIndices.length;
          if (mCount >= 2) {
            const cPrev = measuredIndices[mCount - 2];
            const dPrev = measuredDists[mCount - 2];
            const d12 = dcc[cPrev][chosenTarget];
            const before4P = countActive(clmembflag, K);

            for (let cl = 0; cl < K; cl++) {
              if (!clmembflag[cl]) continue;
              const d1k = dcc[cPrev][cl];
              const d2k = dcc[chosenTarget][cl];
              const minD4 = calc_min_dist_4pt(dPrev, dfc, d12, d1k, d2k);
              if (minD4 > rlim) {
                clmembflag[cl] = false;
                currentPruned.push({ cluster: clusters[cl], reason: '4P' });
              }
            }
            const after4P = countActive(clmembflag, K);
            const pruned4P = before4P - after4P;
            if (pruned4P > 0) {
              pruneCount4P += pruned4P;
              logExplainStep({
                type: 'prune',
                title: '📐 4-Point Pruning (-te4)',
                text: `4-pt pruning: ${after4P} clusters kept, ${pruned4P} additional excluded via triangulation bounds.`
              });
            }
          }
        }

        // 3. 5-Point Pruning (-te5)
        if (pruneMode === '5P') {
          const mCount = measuredIndices.length;
          if (mCount >= 3) {
            const c1 = measuredIndices[mCount - 3];
            const d1 = measuredDists[mCount - 3];
            const c2 = measuredIndices[mCount - 2];
            const d2 = measuredDists[mCount - 2];
            const c3 = chosenTarget;
            const d3 = dfc;

            const d12 = dcc[c1][c2];
            const d13 = dcc[c1][c3];
            const d23 = dcc[c2][c3];
            const before5P = countActive(clmembflag, K);

            for (let cl = 0; cl < K; cl++) {
              if (!clmembflag[cl]) continue;
              const dk1 = dcc[cl][c1];
              const dk2 = dcc[cl][c2];
              const dk3 = dcc[cl][c3];
              const minD5 = calc_min_dist_5pt(d1, d2, d3, dk1, dk2, dk3, d12, d13, d23);
              if (minD5 > rlim) {
                clmembflag[cl] = false;
                currentPruned.push({ cluster: clusters[cl], reason: '5P' });
              }
            }
            const after5P = countActive(clmembflag, K);
            const pruned5P = before5P - after5P;
            if (pruned5P > 0) {
              pruneCount5P += pruned5P;
              logExplainStep({
                type: 'prune',
                title: '📐 5-Point Pruning (-te5)',
                text: `5-pt pruning: ${after5P} clusters kept, ${pruned5P} additional excluded via 3D simplex height bounds.`
              });
            }
          }
        }

        // 4. Geometric Probabilities (-gprob, -maxvis)
        if (useGprob) {
          const startF = Math.max(0, frameHistory.length - maxVisitors);
          for (let f = startF; f < frameHistory.length; f++) {
            const fh = frameHistory[f];
            const dIdx = fh.indices.indexOf(chosenTarget);
            if (dIdx !== -1 && clmembflag[fh.assignment]) {
              const distK = fh.dists[dIdx];
              const dr = Math.abs(dfc - distK) / rlim;
              currentGprobs[fh.assignment] *= fmatch(dr);
            }
          }
        }
      }

      totalEvals += evalsThisFrame;

      // Handle New Cluster Creation & Eviction (-maxcl, -maxcl_strategy)
      if (assignedCluster === -1) {
        if (maxcl > 0 && clusters.length >= maxcl) {
          if (maxclStrategy === 'stop') {
            logExplainStep({
              type: 'mismatch',
              title: '🛑 Max Clusters Limit Reached (-maxcl stop)',
              text: `Cluster budget (${maxcl}) reached. Frame at ${coordStr} cannot form a new cluster and remains unassigned.`
            });
            if (typeof pauseSimulation === 'function') {
              pauseSimulation();
            }
            showToast('🛑 Max cluster limit reached (stop)');
            updateUI();
            draw();
            return;
          } else if (maxclStrategy === 'discard') {
            let victimIdx = 0;
            let minMembers = Infinity;
            for (let i = 0; i < clusters.length; i++) {
              if (clusters[i].members < minMembers) {
                minMembers = clusters[i].members;
                victimIdx = i;
              }
            }
            logExplainStep({
              type: 'mismatch',
              title: '♻️ Cluster Eviction (-maxcl discard)',
              text: `Cluster budget (${maxcl}) reached. Evicted lowest-frequency cluster C${clusters[victimIdx].id} (${minMembers} frames) to allocate new anchor.`
            });
            clusters[victimIdx] = {
              id: clusters[victimIdx].id,
              x, y, z,
              members: 1,
              prob: 1.0,
              scDists: 0,
              lastActive: totalFrames,
              color: getClusterColor(clusters[victimIdx].id)
            };
            for (let i = 0; i < clusters.length; i++) {
              if (i === victimIdx) {
                dcc[victimIdx][i] = 0.0;
              } else {
                const cO = clusters[i];
                const d = Math.sqrt((x - cO.x)**2 + (y - cO.y)**2 + (z - cO.z)**2);
                distClusterCluster++;
                distClusterClusterLast++;
                dcc[victimIdx][i] = d;
                dcc[i][victimIdx] = d;
              }
            }
            assignedCluster = victimIdx;
          } else if (maxclStrategy === 'merge') {
            let minD = Infinity, mergeI = 0, mergeJ = 1;
            for (let i = 0; i < clusters.length; i++) {
              for (let j = i + 1; j < clusters.length; j++) {
                if (dcc[i][j] < minD) {
                  minD = dcc[i][j];
                  mergeI = i;
                  mergeJ = j;
                }
              }
            }
            logExplainStep({
              type: 'mismatch',
              title: '🤝 Cluster Merge (-maxcl merge)',
              text: `Cluster budget (${maxcl}) reached. Merged closest pair (C${clusters[mergeI].id}, C${clusters[mergeJ].id}, d=${minD.toFixed(3)}) and reused slot.`
            });
            clusters[mergeI].members += clusters[mergeJ].members;
            clusters[mergeJ] = {
              id: clusters[mergeJ].id,
              x, y, z,
              members: 1,
              prob: 1.0,
              scDists: 0,
              lastActive: totalFrames,
              color: getClusterColor(clusters[mergeJ].id)
            };
            for (let i = 0; i < clusters.length; i++) {
              if (i === mergeJ) {
                dcc[mergeJ][i] = 0.0;
              } else {
                const cO = clusters[i];
                const d = Math.sqrt((x - cO.x)**2 + (y - cO.y)**2 + (z - cO.z)**2);
                distClusterCluster++;
                distClusterClusterLast++;
                dcc[mergeJ][i] = d;
                dcc[i][mergeJ] = d;
              }
            }
            assignedCluster = mergeJ;
          }
        } else {
          assignedCluster = clusters.length;
          const newAnchor = {
            id: assignedCluster,
            x, y, z,
            members: 1,
            prob: 1.0,
            scDists: 0,
            lastActive: totalFrames,
            color: getClusterColor(assignedCluster)
          };

          const newDccRow = [];
          for (let i = 0; i < clusters.length; i++) {
            const cOther = clusters[i];
            const d_inter = Math.sqrt(
              (x - cOther.x) ** 2 +
              (y - cOther.y) ** 2 +
              (z - cOther.z) ** 2
            );
            distClusterCluster++;
            distClusterClusterLast++;
            dcc[i].push(d_inter);
            newDccRow.push(d_inter);
          }
          newDccRow.push(0.0);
          dcc.push(newDccRow);

          for (let i = 0; i < clusters.length; i++) transitionCounts[i].push(0);
          transitionCounts.push(new Array(clusters.length + 1).fill(0));

          clusters.push(newAnchor);

          logExplainStep({
            type: 'new-cluster',
            title: `✨ New Cluster Created: C${assignedCluster}`,
            text: `Frame fell outside all ${K} candidates (distance > rlim). Spawned new anchor C${assignedCluster} at ${coordStr} and computed ${K} inter-cluster distances.`
          });
        }
      }

      if (prevAssignedCluster >= 0 && transitionCounts[prevAssignedCluster]) {
        transitionCounts[prevAssignedCluster][assignedCluster] =
          (transitionCounts[prevAssignedCluster][assignedCluster] || 0) + 1;
        lastTransitionFrom = prevAssignedCluster;
        lastTransitionTo = assignedCluster;
      } else {
        lastTransitionFrom = -1;
        lastTransitionTo = -1;
      }
      prevAssignedCluster = assignedCluster;
      assignmentHistory.push(assignedCluster);
      if (assignmentHistory.length > 5000) assignmentHistory.shift();

      // Prediction probability dynamics (matches C record_step_assignment.c)
      if (usePred && assignedCluster >= 0 && assignedCluster < clusters.length) {
        clusters[assignedCluster].prob = (clusters[assignedCluster].prob || 0.0) + 0.3;
        let sumP = 0.0;
        for (let i = 0; i < clusters.length; i++) sumP += (clusters[i].prob || 0.0);
        if (sumP > 0.0) {
          for (let i = 0; i < clusters.length; i++) clusters[i].prob /= sumP;
        }
        const floorVal = 0.2 / clusters.length;
        for (let i = 0; i < clusters.length; i++) clusters[i].prob += floorVal;
        sumP = 0.0;
        for (let i = 0; i < clusters.length; i++) sumP += clusters[i].prob;
        if (sumP > 0.0) {
          for (let i = 0; i < clusters.length; i++) clusters[i].prob /= sumP;
        }
      }

      frameHistory.push({
        indices: [...measuredIndices],
        dists: [...measuredDists],
        assignment: assignedCluster
      });
      if (frameHistory.length > 500) frameHistory.shift();

      if (evalsThisFrame > 0) {
        lastEntropyReduced = currentEntropyBits;
        totalEntropyReducedBits += currentEntropyBits;
        lastInfoGainRate = currentEntropyBits / evalsThisFrame;
      }

      // Record Sample Trace in History Log
      const sampleEntry = {
        frameIndex: totalFrames,
        timestamp: performance.now(),
        point: { x, y, z },
        assignedCluster: assignedCluster,
        isNewCluster: (assignedCluster === clusters.length - 1 && !useTiles),
        distSC: distSampleClusterLast,
        distCC: distClusterClusterLast,
        evals: evalsThisFrame,
        initialEntropy: lastInitialEntropy,
        entropyReduced: currentEntropyBits,
        steps: currentExplanation.length > 0 ? [...currentExplanation] : [
          {
            type: 'target',
            title: `📍 Sample #${totalFrames} Ingested`,
            text: `Coordinates ${coordStr} assigned to Cluster C${assignedCluster} with ${distSampleClusterLast} sample-cluster distance evaluations.`
          }
        ],
        entropyRankings: lastEntropyRankings ? [...lastEntropyRankings] : []
      };
      sampleTraceLog.push(sampleEntry);
      if (sampleTraceLog.length > MAX_SAMPLE_TRACE_HISTORY) {
        sampleTraceLog.shift();
      }

      const tComputeEnd = performance.now();
      const frameComputeMs = Math.max(0.0001, tComputeEnd - tComputeStart);
      recordFrameTelemetry(frameComputeMs);

      if (!skipRender && !isRunning) {
        updateUI();
        draw();
      }
    }

    // =========================================================================
