/**
 * GRIC Simulator - sim_reconstruction.js
 * Multi-dataset reconstruction engine and query tracker.
 */

    // =========================================================================
    //  k-NN RECONSTRUCTION ENGINE & MULTI-DATASET INTERPOLATOR
    // =========================================================================

    /**
     * Executes non-parametric k-NN reconstruction:
     * Mapping Input A -> Output B, evaluating queries C to reconstruct D.
     */
    async function executeDatasetReconstruction() {
      if (typeof checkReconstructionConditions === 'function') {
        const cond = checkReconstructionConditions();
        if (!cond.ready) {
          showToast(`⚠️ Reconstruction Error: ${cond.reason}`);
          return;
        }
      }

      // 1. Validate Slots
      const slotA = datasetSlots['A'];
      const slotB = datasetSlots['B'];
      const slotC = datasetSlots['C'];
      const slotD = datasetSlots['D'];

      let ptsA = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotA)
        : (slotA && slotA.benchmarkDataset && slotA.benchmarkDataset.length > 0
            ? slotA.benchmarkDataset : (slotA ? slotA.pastSamples : null));
      let ptsB = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotB)
        : (slotB && slotB.benchmarkDataset && slotB.benchmarkDataset.length > 0
            ? slotB.benchmarkDataset : (slotB ? slotB.pastSamples : null));
      let ptsC = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotC)
        : (slotC && slotC.benchmarkDataset && slotC.benchmarkDataset.length > 0
            ? slotC.benchmarkDataset : (slotC ? slotC.pastSamples : null));

      if (slotA && (!slotA.benchmarkDataset || slotA.benchmarkDataset.length === 0) && ptsA) {
        slotA.benchmarkDataset = ptsA;
      }
      if (slotB && (!slotB.benchmarkDataset || slotB.benchmarkDataset.length === 0) && ptsB) {
        slotB.benchmarkDataset = ptsB;
      }
      if (slotC && (!slotC.benchmarkDataset || slotC.benchmarkDataset.length === 0) && ptsC) {
        slotC.benchmarkDataset = ptsC;
      }

      if (!ptsA || ptsA.length === 0) {
        showToast('⚠️ Reconstruction Error: Training Input Dataset [A] is empty or not staged.');
        return;
      }
      if (!ptsC || ptsC.length === 0) {
        showToast('⚠️ Reconstruction Error: Query Input Dataset [C] is empty or not staged.');
        return;
      }

      // Auto-stage or auto-align Slot B to match Slot A if needed
      if (slotB && (!ptsB || ptsB.length === 0 || (ptsA && ptsB.length !== ptsA.length))) {
        if (slotB.benchmarkKey && slotB.benchmarkKey !== 'custom' && slotB.dataMode !== 'image') {
          slotB.sampleCount = slotA.sampleCount ||
            Math.round(ptsA.length / (slotA.loopCount || 1));
          slotB.loopCount = slotA.loopCount || 1;
          if (typeof stageDataset === 'function') {
            stageDataset(slotB.benchmarkKey, 'B');
            ptsB = (typeof getSlotPoints === 'function')
              ? getSlotPoints(slotB)
              : (slotB.benchmarkDataset || slotB.pastSamples);
            if (slotB && ptsB) slotB.benchmarkDataset = ptsB;
          }
        }
      }

      if (!ptsB || ptsB.length === 0) {
        showToast('⚠️ Reconstruction Error: Training Output Dataset [B] is empty or not staged.');
        return;
      }
      if (ptsA.length !== ptsB.length) {
        showToast(
          `⚠️ Reconstruction Error: Sample count mismatch between [A] ` +
          `(${ptsA.length.toLocaleString()}) and [B] (${ptsB.length.toLocaleString()}). ` +
          `A and B must have identical sample counts.`
        );
        return;
      }

      const isImageOutput = (slotB && slotB.dataMode === 'image') ||
                            (ptsB && ptsB.length > 0 &&
                             (ptsB[0] instanceof Float32Array ||
                              ptsB[0] instanceof Float64Array));
      const isImageInput = (slotA && slotA.dataMode === 'image') ||
                           (ptsA && ptsA.length > 0 &&
                            (ptsA[0] instanceof Float32Array ||
                             ptsA[0] instanceof Float64Array));

      function detectSlotDim(slot, pts) {
        if (slot && slot.dataMode === 'image') {
          return slot.imageDim || 1024;
        }
        if (pts && pts.length > 0 &&
            (pts[0] instanceof Float32Array || pts[0] instanceof Float64Array)) {
          return pts[0].length;
        }
        if (slot && slot.benchmarkKey && typeof getBenchmarkDim === 'function') {
          return getBenchmarkDim(slot.benchmarkKey);
        }
        if (slot && slot.currentDim) {
          return slot.currentDim;
        }
        if (pts && pts.length > 0) {
          if (pts[0].coords && pts[0].coords.length > 0) {
            return pts[0].coords.length;
          }
          for (let i = 0; i < Math.min(200, pts.length); i++) {
            if (pts[i].z !== undefined && Math.abs(pts[i].z) > 1e-6) return 3;
          }
        }
        return 2;
      }

      const dimA = detectSlotDim(slotA, ptsA);
      const dimB = detectSlotDim(slotB, ptsB);
      const dimC = detectSlotDim(slotC, ptsC);

      if (dimA !== dimC) {
        showToast(
          `⚠️ Reconstruction Error: Coordinate dimension mismatch between Input [A] ` +
          `(${dimA}D) and Query [C] (${dimC}D). They must match.`
        );
        return;
      }

      // 2. Read Parameters
      const inputK = document.getElementById('inputReconK');
      const k = inputK
        ? Math.max(1, Math.min(parseInt(inputK.value, 10) || 30, ptsA.length)) : 30;
      const selectWeight = document.getElementById('selectReconWeight');
      const weightMode = selectWeight ? selectWeight.value : 'uniform';
      const inputAlpha = document.getElementById('inputReconAlpha');
      const alpha = inputAlpha
        ? Math.max(0.1, parseFloat(inputAlpha.value) || 1.0) : 1.0;

      const numQueries = ptsC.length;
      const numCandidates = ptsA.length;

      // 3. In Native C mode, automatically run the compiled native C gric-knn -query C
      if (DesktopBridge.isNativeSupported()) {
        const hasValidNativeKnn = !!(
          slotC && slotC.knnResults && slotC.knnResultsForQuery &&
          slotC.knnResults.indices &&
          slotC.knnResults.totalFrames === numQueries &&
          slotC.knnResults.k >= k
        );

        if (!hasValidNativeKnn) {
          showToast(`⚡ Running native compiled C gric-knn -query C (k=${k})...`);
          await runNativeReconQueryKnn({ autoReconstruct: false });
        }
      }

      const activeSlotC = datasetSlots['C'] || slotC;
      const nativeKnn = (activeSlotC && activeSlotC.knnResults &&
                         activeSlotC.knnResultsForQuery &&
                         activeSlotC.knnResults.indices &&
                         activeSlotC.knnResults.totalFrames === numQueries)
        ? activeSlotC.knnResults : null;

      if (DesktopBridge.isNativeSupported() && !nativeKnn) {
        showToast('⚠️ Native k-NN search could not load result indices.');
        return;
      }

      showToast(
        `⚡ Running k-NN Reconstruction (A: ${ptsA.length} pts in ${dimA}D, ` +
        `B: ${dimB}D target, C: ${ptsC.length} queries, k=${k})...`
      );

      const tStart = performance.now();
      const reconstructedPoints = new Array(numQueries);
      const sourceNeighbors = new Array(numQueries);
      let totalDistSum = 0.0;
      let totalDistCount = 0;

      if (nativeKnn) {
        /* Fast path: use pre-computed C-in-A indices from gric-knn -query */
        console.log(`[Reconstruction] Using pre-computed native k-NN results ` +
                    `(k=${nativeKnn.k}, N=${numQueries})`);
        const kUsed = Math.min(k, nativeKnn.k);
        for (let i = 0; i < numQueries; i++) {
          const offset = i * nativeKnn.k;
          const topNeighbors = [];

          /* Filter and collect valid neighbor indices and distances */
          const validIdx = [];
          const validDists = [];
          for (let p = 0; p < kUsed; p++) {
            const idx = nativeKnn.indices[offset + p];
            const d = nativeKnn.distances[offset + p];
            if (idx >= 0 && idx < ptsB.length && ptsB[idx] && d >= 0) {
              validIdx.push(idx);
              validDists.push(d);
            }
          }
          const numValid = validIdx.length;

          if (isImageOutput) {
            const reconBuf = new Float32Array(dimB);
            if (numValid > 0) {
              if (weightMode === 'idw') {
                let exactMatchIdx = -1;
                for (let p = 0; p < numValid; p++) {
                  if (validDists[p] < 1e-9) { exactMatchIdx = p; break; }
                }
                if (exactMatchIdx >= 0) {
                  const pb = ptsB[validIdx[exactMatchIdx]];
                  for (let d = 0; d < dimB; d++) {
                    reconBuf[d] = pb[d];
                  }
                  for (let p = 0; p < numValid; p++) {
                    topNeighbors.push({
                      id: validIdx[p], dist: validDists[p],
                      weight: (p === exactMatchIdx) ? 1.0 : 0.0
                    });
                    totalDistSum += validDists[p];
                    totalDistCount++;
                  }
                } else {
                  let sumW = 0.0;
                  const weights = new Float64Array(numValid);
                  for (let p = 0; p < numValid; p++) {
                    const d = validDists[p];
                    const w = 1.0 / Math.pow(Math.max(d, 1e-7), alpha);
                    weights[p] = w; sumW += w;
                    totalDistSum += d; totalDistCount++;
                  }
                  const invSumW = sumW > 0 ? (1.0 / sumW) : 0.0;
                  for (let p = 0; p < numValid; p++) {
                    const normW = weights[p] * invSumW;
                    const pb = ptsB[validIdx[p]];
                    for (let d = 0; d < dimB; d++) {
                      reconBuf[d] += normW * pb[d];
                    }
                    topNeighbors.push({ id: validIdx[p], dist: validDists[p], weight: normW });
                  }
                }
              } else {
                const normW = 1.0 / numValid;
                for (let p = 0; p < numValid; p++) {
                  const pb = ptsB[validIdx[p]];
                  for (let d = 0; d < dimB; d++) {
                    reconBuf[d] += normW * pb[d];
                  }
                  topNeighbors.push({ id: validIdx[p], dist: validDists[p], weight: normW });
                  totalDistSum += validDists[p]; totalDistCount++;
                }
              }
            }
            reconstructedPoints[i] = reconBuf;
            sourceNeighbors[i]     = topNeighbors;
          } else {
            let avgX = 0.0, avgY = 0.0, avgZ = 0.0;
            const avgCoords = (dimB > 3) ? new Float64Array(dimB) : null;
            if (numValid > 0) {
              if (weightMode === 'idw') {
                let exactMatchIdx = -1;
                for (let p = 0; p < numValid; p++) {
                  if (validDists[p] < 1e-9) { exactMatchIdx = p; break; }
                }
                if (exactMatchIdx >= 0) {
                  const pb = ptsB[validIdx[exactMatchIdx]];
                  avgX = pb.x; avgY = pb.y;
                  avgZ = (dimB >= 3 && typeof pb.z === 'number') ? pb.z : 0.0;
                  if (avgCoords && pb.coords) {
                    for (let d = 0; d < dimB; d++) avgCoords[d] = pb.coords[d];
                  }
                  for (let p = 0; p < numValid; p++) {
                    topNeighbors.push({
                      id: validIdx[p], dist: validDists[p],
                      weight: (p === exactMatchIdx) ? 1.0 : 0.0
                    });
                    totalDistSum += validDists[p];
                    totalDistCount++;
                  }
                } else {
                  let sumW = 0.0;
                  const weights = new Float64Array(numValid);
                  for (let p = 0; p < numValid; p++) {
                    const d = validDists[p];
                    const w = 1.0 / Math.pow(Math.max(d, 1e-7), alpha);
                    weights[p] = w; sumW += w;
                    totalDistSum += d; totalDistCount++;
                  }
                  const invSumW = sumW > 0 ? (1.0 / sumW) : 0.0;
                  for (let p = 0; p < numValid; p++) {
                    const normW = weights[p] * invSumW;
                    const pb = ptsB[validIdx[p]];
                    avgX += normW * pb.x; avgY += normW * pb.y;
                    if (dimB >= 3 && typeof pb.z === 'number') { avgZ += normW * pb.z; }
                    if (avgCoords && pb.coords) {
                      for (let d = 0; d < dimB; d++) avgCoords[d] += normW * pb.coords[d];
                    }
                    topNeighbors.push({ id: validIdx[p], dist: validDists[p], weight: normW });
                  }
                }
              } else {
                const normW = 1.0 / numValid;
                for (let p = 0; p < numValid; p++) {
                  const pb = ptsB[validIdx[p]];
                  avgX += normW * pb.x; avgY += normW * pb.y;
                  if (dimB >= 3 && typeof pb.z === 'number') { avgZ += normW * pb.z; }
                  if (avgCoords && pb.coords) {
                    for (let d = 0; d < dimB; d++) avgCoords[d] += normW * pb.coords[d];
                  }
                  topNeighbors.push({ id: validIdx[p], dist: validDists[p], weight: normW });
                  totalDistSum += validDists[p]; totalDistCount++;
                }
              }
            }
            const reconPt = { x: avgX, y: avgY };
            if (dimB >= 3) { reconPt.z = avgZ; }
            if (avgCoords) {
              reconPt.coords = avgCoords;
              reconPt.x = avgCoords[0];
              reconPt.y = avgCoords[1];
              reconPt.z = avgCoords[2];
            }
            reconstructedPoints[i] = reconPt;
            sourceNeighbors[i]     = topNeighbors;
          }
        } // for each query (fast path)

      } else {
        /* Brute-force path: O(N_C × N_A) for each query in C */
        if (nativeKnn === null && slotC && slotC.knnResults) {
          console.log('[Reconstruction] Native k-NN results exist but dimensions mismatch; ' +
                      'falling back to brute-force.');
        }
        for (let i = 0; i < numQueries; i++) {
          if (i > 0 && i % 25 === 0) {
            if (typeof showToast === 'function' && numQueries > 50) {
              showToast(
                `⚡ Reconstructing Frame ${i}/${numQueries} ` +
                `(${Math.round((i / numQueries) * 100)}%)...`
              );
            }
            await new Promise(r => setTimeout(r, 0));
          }
          const qc = ptsC[i];
          const dists   = new Float64Array(numCandidates);
          const indices = new Int32Array(numCandidates);

          if (isImageInput) {
            for (let j = 0; j < numCandidates; j++) {
              const pa = ptsA[j];
              let sumSq = 0.0;
              for (let d = 0; d < dimA; d++) {
                const diff = qc[d] - pa[d];
                sumSq += diff * diff;
              }
              dists[j]   = Math.sqrt(sumSq);
              indices[j] = j;
            }
          } else {
            const qx = qc.x;
            const qy = qc.y;
            const qz = (dimA >= 3 && typeof qc.z === 'number') ? qc.z : 0.0;
            for (let j = 0; j < numCandidates; j++) {
              const pa = ptsA[j];
              const dx = qx - pa.x;
              const dy = qy - pa.y;
              const dz = (dimA >= 3 && typeof pa.z === 'number') ? (qz - pa.z) : 0.0;
              dists[j]   = Math.sqrt(dx * dx + dy * dy + dz * dz);
              indices[j] = j;
            }
          }

          /* Partial sort top-k smallest distances */
          for (let p = 0; p < k; p++) {
            let minIdx = p;
            for (let j = p + 1; j < numCandidates; j++) {
              if (dists[j] < dists[minIdx]) { minIdx = j; }
            }
            const tmpD = dists[p]; dists[p] = dists[minIdx]; dists[minIdx] = tmpD;
            const tmpI = indices[p]; indices[p] = indices[minIdx]; indices[minIdx] = tmpI;
          }

          /* Calculate weights and average corresponding B samples */
          const topNeighbors = [];
          if (isImageOutput) {
            const reconBuf = new Float32Array(dimB);
            if (weightMode === 'idw') {
              let exactMatchIdx = -1;
              for (let p = 0; p < k; p++) {
                if (dists[p] < 1e-9) { exactMatchIdx = p; break; }
              }
              if (exactMatchIdx >= 0) {
                const matchedSampleId = indices[exactMatchIdx];
                const pb = ptsB[matchedSampleId];
                for (let d = 0; d < dimB; d++) {
                  reconBuf[d] = pb[d];
                }
                for (let p = 0; p < k; p++) {
                  topNeighbors.push({
                    id: indices[p], dist: dists[p],
                    weight: (p === exactMatchIdx) ? 1.0 : 0.0
                  });
                  totalDistSum += dists[p]; totalDistCount++;
                }
              } else {
                let sumW = 0.0;
                const weights = new Float64Array(k);
                for (let p = 0; p < k; p++) {
                  const d = dists[p];
                  const w = 1.0 / Math.pow(Math.max(d, 1e-7), alpha);
                  weights[p] = w; sumW += w;
                  totalDistSum += d; totalDistCount++;
                }
                for (let p = 0; p < k; p++) {
                  const normW = weights[p] / sumW;
                  const pb = ptsB[indices[p]];
                  for (let d = 0; d < dimB; d++) {
                    reconBuf[d] += normW * pb[d];
                  }
                  topNeighbors.push({ id: indices[p], dist: dists[p], weight: normW });
                }
              }
            } else {
              const normW = 1.0 / k;
              for (let p = 0; p < k; p++) {
                const pb = ptsB[indices[p]];
                for (let d = 0; d < dimB; d++) {
                  reconBuf[d] += normW * pb[d];
                }
                topNeighbors.push({ id: indices[p], dist: dists[p], weight: normW });
                totalDistSum += dists[p]; totalDistCount++;
              }
            }
            reconstructedPoints[i] = reconBuf;
            sourceNeighbors[i]     = topNeighbors;
          } else {
            let avgX = 0.0, avgY = 0.0, avgZ = 0.0;
            const avgCoords = (dimB > 3) ? new Float64Array(dimB) : null;
            if (weightMode === 'idw') {
              let exactMatchIdx = -1;
              for (let p = 0; p < k; p++) {
                if (dists[p] < 1e-9) { exactMatchIdx = p; break; }
              }
              if (exactMatchIdx >= 0) {
                const matchedSampleId = indices[exactMatchIdx];
                const pb = ptsB[matchedSampleId];
                avgX = pb.x; avgY = pb.y;
                avgZ = (dimB >= 3 && typeof pb.z === 'number') ? pb.z : 0.0;
                if (avgCoords && pb.coords) {
                  for (let d = 0; d < dimB; d++) avgCoords[d] = pb.coords[d] || 0.0;
                }
                for (let p = 0; p < k; p++) {
                  topNeighbors.push({
                    id: indices[p], dist: dists[p],
                    weight: (p === exactMatchIdx) ? 1.0 : 0.0
                  });
                  totalDistSum += dists[p]; totalDistCount++;
                }
              } else {
                let sumW = 0.0;
                const weights = new Float64Array(k);
                for (let p = 0; p < k; p++) {
                  const d = dists[p];
                  const w = 1.0 / Math.pow(Math.max(d, 1e-7), alpha);
                  weights[p] = w; sumW += w;
                  totalDistSum += d; totalDistCount++;
                }
                for (let p = 0; p < k; p++) {
                  const normW = weights[p] / sumW;
                  const pb = ptsB[indices[p]];
                  avgX += normW * pb.x; avgY += normW * pb.y;
                  if (dimB >= 3 && typeof pb.z === 'number') { avgZ += normW * pb.z; }
                  if (avgCoords && pb.coords) {
                    for (let d = 0; d < dimB; d++) avgCoords[d] += normW * (pb.coords[d] || 0.0);
                  }
                  topNeighbors.push({ id: indices[p], dist: dists[p], weight: normW });
                }
              }
            } else {
              const normW = 1.0 / k;
              for (let p = 0; p < k; p++) {
                const pb = ptsB[indices[p]];
                avgX += normW * pb.x; avgY += normW * pb.y;
                if (dimB >= 3 && typeof pb.z === 'number') { avgZ += normW * pb.z; }
                if (avgCoords && pb.coords) {
                  for (let d = 0; d < dimB; d++) avgCoords[d] += normW * (pb.coords[d] || 0.0);
                }
                topNeighbors.push({ id: indices[p], dist: dists[p], weight: normW });
                totalDistSum += dists[p]; totalDistCount++;
              }
            }

            const reconPt = { x: avgX, y: avgY };
            if (dimB >= 3) { reconPt.z = avgZ; }
            if (avgCoords) {
              reconPt.coords = avgCoords;
              reconPt.x = avgCoords[0];
              reconPt.y = avgCoords[1];
              reconPt.z = avgCoords[2];
            }
            reconstructedPoints[i] = reconPt;
            sourceNeighbors[i]     = topNeighbors;
          }
        } // for each query (brute-force)
      } // if nativeKnn

      const elapsedMs = performance.now() - tStart;
      const avgNeighborDist =
        totalDistCount > 0 ? (totalDistSum / totalDistCount) : 0.0;

      // Compute quality metrics
      const reconKthDist = new Float64Array(numQueries);
      const reconVariance = new Float64Array(numQueries);
      let kthDistMin = Infinity, kthDistMax = -Infinity;
      let varMin = Infinity, varMax = -Infinity;

      for (let i = 0; i < numQueries; i++)
      {
        const neighbors = sourceNeighbors[i];
        if (!neighbors || neighbors.length === 0)
        {
          continue;
        }

        // k-th NN distance = farthest neighbor dist
        const kthD = neighbors[neighbors.length - 1].dist;
        reconKthDist[i] = kthD;
        if (kthD < kthDistMin) { kthDistMin = kthD; }
        if (kthD > kthDistMax) { kthDistMax = kthD; }

        // Weighted variance of B-samples around mean D[i]
        const rp = reconstructedPoints[i];
        if (!rp) continue;
        let wvar = 0.0;
        for (let p = 0; p < neighbors.length; p++)
        {
          const nb = neighbors[p];
          if (!nb || nb.id < 0 || nb.id >= ptsB.length) continue;
          const pb = ptsB[nb.id];
          if (!pb) continue;
          let dxSq = 0.0;
          if (isImageOutput) {
            for (let d = 0; d < dimB; d++) {
              const diff = pb[d] - rp[d];
              dxSq += diff * diff;
            }
          } else {
            dxSq = (pb.x - rp.x) * (pb.x - rp.x)
                 + (pb.y - rp.y) * (pb.y - rp.y);
            if (dimB >= 3)
            {
              const pz = (typeof pb.z === 'number') ? pb.z : 0.0;
              const rz = (typeof rp.z === 'number') ? rp.z : 0.0;
              dxSq += (pz - rz) * (pz - rz);
            }
          }
          const weightVal = (typeof nb.weight === 'number' && !isNaN(nb.weight))
            ? nb.weight : (1.0 / neighbors.length);
          wvar += weightVal * dxSq;
        }
        reconVariance[i] = wvar;
        if (wvar < varMin) { varMin = wvar; }
        if (wvar > varMax) { varMax = wvar; }
      } // for quality metrics

      if (kthDistMin === Infinity) { kthDistMin = 0; }
      if (kthDistMax === -Infinity) { kthDistMax = 0; }
      if (varMin === Infinity) { varMin = 0; }
      if (varMax === -Infinity) { varMax = 0; }

      // 4. Save into Slot D
      slotD.benchmarkDataset = reconstructedPoints;
      slotD.rawBenchmarkDataset = reconstructedPoints;
      slotD.currentDim = dimB;
      slotD.benchmarkKey = 'reconstructed';
      slotD.sampleCount = numQueries;
      slotD.isDatasetStaged = true;

      if (isImageOutput) {
        slotD.dataMode = 'image';
        slotD.imageWidth = (slotB && slotB.imageWidth) || 32;
        slotD.imageHeight = (slotB && slotB.imageHeight) || 32;
        slotD.imageDim = dimB;
        slotD.pastSamples = [];
        slotD.totalFrames = numQueries;
        slotD.currentImageFrame = reconstructedPoints.length > 0
          ? reconstructedPoints[0] : null;
        slotD.stagedDatasetInfo = {
          name: 'Reconstructed Images (from A, B, C)',
          count: numQueries,
          dim: dimB,
          passes: 1,
          noise: 0
        };
      } else {
        slotD.dataMode = 'coord';
        slotD.pastSamples = reconstructedPoints.map((p, idx) => {
          const pt = {
            x: p.x,
            y: p.y,
            z: (dimB >= 3 && typeof p.z === 'number') ? p.z : 0.0,
            clusterId: -1,
            frameIndex: idx
          };
          if (p.coords) { pt.coords = p.coords; }
          return pt;
        });
        slotD.stagedDatasetInfo = {
          name: 'Reconstructed (from A, B, C)',
          count: numQueries,
          dim: dimB,
          passes: 1,
          noise: 0
        };
      }

      slotD.reconstructionInfo = {
        queryCount: numQueries,
        outputDim: dimB,
        k: k,
        weightMode: weightMode,
        alpha: alpha,
        computeTimeMs: elapsedMs,
        avgNeighborDist: avgNeighborDist
      };
      slotD.reconstructionSourceNeighbors = sourceNeighbors;
      slotD._reverseNeighborsIndex = null;
      if (slotC && slotC._onDemandKnnCache) slotC._onDemandKnnCache.clear();

      // Store quality metrics on slots C and D
      slotC.reconKthDist = reconKthDist;
      slotC.reconKthDistMin = kthDistMin;
      slotC.reconKthDistMax = kthDistMax;
      slotC.reconVariance = reconVariance;
      slotC.reconVarianceMin = varMin;
      slotC.reconVarianceMax = varMax;
      slotD.reconKthDist = reconKthDist;
      slotD.reconKthDistMin = kthDistMin;
      slotD.reconKthDistMax = kthDistMax;
      slotD.reconVariance = reconVariance;
      slotD.reconVarianceMin = varMin;
      slotD.reconVarianceMax = varMax;

      // Update Query k-NN Resource Tracker card with query execution metrics
      if (!nativeKnn) {
        const bruteDistCalls = numQueries * numCandidates;
        updateReconQueryTracker({
          framedistCalls: bruteDistCalls,
          timeSearchMs: elapsedMs,
          timeLoadMs: 0,
          level1ClustersPruned: 0,
          level2AnchorsPruned: 0,
          level3AnnularPruned: 0,
          temporalPruned: 0
        }, numQueries, numCandidates, k);
        const pBadge = document.getElementById('reconQueryPruneEffBadge');
        if (pBadge) pBadge.textContent = '0% (WASM Brute)';
      }

      // Update Toolbar status pill for D
      const pillD = document.getElementById('datasetStatusPill_D');
      if (pillD) {
        const countFormatted = numQueries >= 1000
          ? `${(numQueries / 1000).toFixed(numQueries % 1000 === 0 ? 0 : 1)}k`
          : numQueries;
        pillD.textContent = isImageOutput
          ? `📦 ${countFormatted} frames (${slotD.imageWidth}×${slotD.imageHeight})`
          : `📦 ${countFormatted} pts (${dimB}D)`;
        pillD.title = `Reconstructed (${numQueries.toLocaleString()} points, ${dimB}D, k=${k})`;
        pillD.style.background = 'rgba(168, 85, 247, 0.2)';
        pillD.style.color = '#c084fc';
        pillD.style.borderColor = 'rgba(168, 85, 247, 0.5)';
      }

      // 5. Automatically enable 4-panel view and switch active slot to D
      if (!multiDatasetEnabled) {
        setMultiDatasetEnabled(true);
      }
      setRecon4PanelView(true);

      if (activeDatasetSlot !== 'D') {
        switchDatasetSlot('D');
      } else {
        loadSlotState('D');
        if (typeof updateDatasetStatusBadge === 'function') updateDatasetStatusBadge();
        if (typeof updateUI === 'function') updateUI();
        if (typeof renderReconstructionDashboard === 'function') {
          renderReconstructionDashboard();
        }
      }

      if (typeof resetView === 'function') {
        resetView();
      }
      if (typeof draw === 'function') {
        draw();
      }
      if (typeof updateReconQualityBar === 'function') {
        updateReconQualityBar();
      }
      if (typeof updateReconstructionButtonState === 'function') {
        updateReconstructionButtonState();
      }

      showToast(
        `✅ Reconstruction Complete! Dataset [D] contains ` +
        `${numQueries.toLocaleString()} ` +
        `${isImageOutput ? 'frames (image)' : `points (${dimB}D)`} ` +
        `in ${elapsedMs.toFixed(1)} ms.`
      );
    }

    // =========================================================================
    //  NATIVE gric-knn -query C RUNNER
    // =========================================================================

    /**
     * Populate the Query k-NN Resource Tracker card with parsed telemetry.
     *
     * @param {object} telem    Result of DesktopBridge.parseKnnTelemetryLog()
     * @param {number} numQ     Number of query (C) frames
     * @param {number} numCand  Number of candidate (A) frames
     * @param {number} k        k neighbors
     */
    function updateReconQueryTracker(telem, numQ, numCand, k)
    {
      const bruteForce = numQ * numCand;
      const pruned = Math.max(0, bruteForce - telem.framedistCalls);
      const pruneEff = bruteForce > 0
        ? (100.0 * pruned / bruteForce) : 0.0;
      const speedup = telem.framedistCalls > 0
        ? (bruteForce / telem.framedistCalls) : 0.0;
      const qps = telem.timeSearchMs > 0
        ? (numQ / (telem.timeSearchMs / 1000.0)) : 0.0;
      const distPerQuery = numQ > 0
        ? Math.round(telem.framedistCalls / numQ) : 0;

      /* Compute per-level percentages against total candidate slots */
      const totalSlots = numQ * numCand;
      function pct(val) {
        return totalSlots > 0
          ? `${(100.0 * val / totalSlots).toFixed(1)}%`
          : '0%';
      }
      const l1 = (telem.level0SuperClustersPruned || 0) + (telem.level1ClustersPruned || 0);
      const l3 = telem.level3AnnularPruned || 0;
      const gPruned = telem.graphEdgesPruned || 0;
      const angular = telem.angularPruned || 0;
      const multiPivot = telem.multiPivotPruned || 0;
      const gSeeds = telem.graphSeedsEvaluated || 0;
      const isRq8 = Boolean(
        telem.rq8Evaluations || telem.rq8MembersPruned || telem.rq8GraphPruned
      );
      const isEq16 = Boolean(
        telem.eq16Evaluations || telem.eq16MembersPruned || telem.eq16GraphPruned
      );
      const isSq16 = Boolean(
        telem.sq16Evaluations || telem.sq16MembersPruned || telem.sq16GraphPruned
      );
      const sqType = isRq8 ? 'RQ8' : (isEq16 ? 'EQ16' : (isSq16 ? 'SQ16' : 'SQ8'));
      const sqEvals = (telem.rq8Evaluations || 0) +
                      (telem.eq16Evaluations || 0) +
                      (telem.sq16Evaluations || 0) +
                      (telem.sq8Evaluations || 0);
      const sqMembers = (telem.rq8MembersPruned || 0) +
                        (telem.eq16MembersPruned || 0) +
                        (telem.sq16MembersPruned || 0) +
                        (telem.sq8MembersPruned || 0);
      const sqGraph = (telem.rq8GraphPruned || 0) +
                      (telem.eq16GraphPruned || 0) +
                      (telem.sq16GraphPruned || 0) +
                      (telem.sq8GraphPruned || 0);
      const hasTotalPruned = (telem.rq8TotalPruned || telem.eq16TotalPruned ||
                              telem.sq16TotalPruned || telem.sq8TotalPruned !== undefined);
      const sqPruned = hasTotalPruned
        ? ((telem.rq8TotalPruned || 0) + (telem.eq16TotalPruned || 0) +
           (telem.sq16TotalPruned || 0) + (telem.sq8TotalPruned || 0))
        : (sqMembers + sqGraph);
      const exact = Math.max(0, telem.framedistCalls || 0);
      const totalAll = l1 + l3 + gPruned + angular + multiPivot + gSeeds + sqPruned + exact;

      function barPct(val) {
        return totalAll > 0
          ? `${(100.0 * val / totalAll).toFixed(2)}%`
          : '0%';
      }

      /* Helper to safely set textContent by id */
      function set(id, text) {
        const el = document.getElementById(id);
        if (el) { el.textContent = text; }
      }

      /* -- Overview tab -- */
      set('reconQueryPruneEffVal',     `${pruneEff.toFixed(1)}%`);
      set('reconQuerySpeedupVal',      `${speedup.toFixed(1)}×`);
      set('reconQueryDistCallsVal',    telem.framedistCalls.toLocaleString());
      set('reconQueryDistPerQueryVal', `${distPerQuery.toLocaleString()}/q`);
      set('lblReconQuerySqPrecType',   sqType);
      set('reconQuerySq8EvalsVal',     sqEvals > 0 ? sqEvals.toLocaleString() : '--');
      set('reconQueryFullPrecCallsVal', exact.toLocaleString());
      set('reconQuerySearchTimeVal',   `${telem.timeSearchMs.toFixed(1)} ms`);
      set('reconQueryLoadTimeVal',     telem.timeLoadMs != null
                                         ? `load ${telem.timeLoadMs.toFixed(0)} ms`
                                         : 'load --');
      set('reconQueryQpsVal',          `${Math.round(qps).toLocaleString()} QPS`);
      set('reconQueryTotalQueriesVal', numQ.toLocaleString());
      set('reconQueryKParamVal',       `k=${k}`);

      /* Funnel bar */
      set('reconQueryPruneSummaryTxt', `${pruneEff.toFixed(1)}% Pruned`);
      const barPruned = document.getElementById('barReconQueryPruned');
      const barEval   = document.getElementById('barReconQueryEvaluated');
      if (barPruned) { barPruned.style.width = `${pruneEff.toFixed(2)}%`; }
      if (barEval)   { barEval.style.width   = `${(100 - pruneEff).toFixed(2)}%`; }
      set('lblReconQueryPrunedCount',  pruned.toLocaleString());
      set('lblReconQueryEvalCount',    exact.toLocaleString());

      /* -- Pruning tab -- */
      set('reconQueryL1PrunedVal',          l1.toLocaleString());
      set('reconQueryL1PctVal',             pct(l1));
      set('reconQueryL3PrunedVal',          l3.toLocaleString());
      set('reconQueryL3PctVal',             pct(l3));
      set('reconQueryGraphPrunedVal',       gPruned.toLocaleString());
      set('reconQueryGraphPrunedPctVal',    pct(gPruned));
      set('reconQueryGraphSeedsVal',        gSeeds.toLocaleString());
      set('reconQueryGraphSeedsPctVal',     pct(gSeeds));

      set('reconQueryAngularVal',           angular.toLocaleString());
      set('reconQueryAngularPctVal',        pct(angular));

      set('reconQueryMultiPivotVal',        multiPivot.toLocaleString());
      set('reconQueryMultiPivotPctVal',     pct(multiPivot));

      set('lblReconQuerySqType',            sqType);
      set('reconQuerySq8PrunedVal',         sqPruned.toLocaleString());
      set('reconQuerySq8PctVal',            pct(sqPruned));
      set('reconQuerySq8BreakdownVal',
          `Members: ${sqMembers.toLocaleString()} | Graph: ${sqGraph.toLocaleString()}`);

      const clustersGraph = telem.clustersGraphEvaluated || 0;
      set('reconQueryClustersGraphVal', clustersGraph.toLocaleString());

      const containmentHits = telem.globalContainmentHits || 0;
      const containmentPct = numQ > 0 ? (100.0 * containmentHits / numQ).toFixed(1) : '0.0';
      set('reconQueryContainmentVal',       containmentHits.toLocaleString());
      set('reconQueryContainmentPctVal',    `${containmentPct}% of queries`);

      /* Hierarchy stacked bar */
      const bL1         = document.getElementById('barReconQueryL1');
      const bL3         = document.getElementById('barReconQueryL3');
      const bGPruned    = document.getElementById('barReconQueryGraphPruned');
      const bAngular    = document.getElementById('barReconQueryAngular');
      const bMultiPivot = document.getElementById('barReconQueryMultiPivot');
      const bGSeeds     = document.getElementById('barReconQueryGraphSeeds');
      const bSq8        = document.getElementById('barReconQuerySq8');
      const bExact      = document.getElementById('barReconQueryExact');
      if (bL1)         { bL1.style.width         = barPct(l1); }
      if (bL3)         { bL3.style.width         = barPct(l3); }
      if (bGPruned)    { bGPruned.style.width    = barPct(gPruned); }
      if (bAngular)    { bAngular.style.width    = barPct(angular); }
      if (bMultiPivot) { bMultiPivot.style.width = barPct(multiPivot); }
      if (bGSeeds)     { bGSeeds.style.width     = barPct(gSeeds); }
      if (bSq8)        { bSq8.style.width        = barPct(sqPruned); }
      if (bExact)      { bExact.style.width      = barPct(exact); }

      /* Header badges */
      set('reconQueryPruneEffBadge', `${pruneEff.toFixed(1)}% pruned`);
      set('reconQueryQpsBadge',      `${Math.round(qps).toLocaleString()} QPS`);

      /* Show and expand the tracker card */
      const card = document.getElementById('cardReconQueryResources');
      if (card) {
        card.style.display = '';
        if (card.classList.contains('collapsed')) {
          togglePanelCollapse('cardReconQueryResources');
        }
      }
    }

    /**
     * Run native gric-knn -query C via gric-server, showing a live progress
     * bar and populating the Query k-NN Resource Tracker on completion.
     */
    async function runNativeReconQueryKnn(options = {})
    {
      if (!DesktopBridge.isNativeSupported()) {
        showToast('⚠️ Native runner not available in Web Mode.');
        return;
      }

      const isEvent = options && typeof options.preventDefault === 'function';
      const autoReconstruct = isEvent ? true :
        (options && options.autoReconstruct !== undefined ? options.autoReconstruct : true);

      const slotA = datasetSlots['A'];
      const slotB = datasetSlots['B'];
      const slotC = datasetSlots['C'];

      let ptsA = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotA)
        : (slotA ? (slotA.benchmarkDataset || slotA.pastSamples) : null);
      let ptsB = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotB)
        : (slotB ? (slotB.benchmarkDataset || slotB.pastSamples) : null);
      let ptsC = (typeof getSlotPoints === 'function')
        ? getSlotPoints(slotC)
        : (slotC ? (slotC.benchmarkDataset || slotC.pastSamples) : null);

      if (slotA && (!slotA.benchmarkDataset || slotA.benchmarkDataset.length === 0) && ptsA) {
        slotA.benchmarkDataset = ptsA;
      }
      if (slotB && (!slotB.benchmarkDataset || slotB.benchmarkDataset.length === 0) && ptsB) {
        slotB.benchmarkDataset = ptsB;
      }
      if (slotC && (!slotC.benchmarkDataset || slotC.benchmarkDataset.length === 0) && ptsC) {
        slotC.benchmarkDataset = ptsC;
      }

      if (!ptsA || ptsA.length === 0) {
        showToast('⚠️ Dataset A (Training Input) must be staged first.');
        return;
      }
      if (!ptsC || ptsC.length === 0) {
        showToast('⚠️ Dataset C (Query Input) must be staged first.');
        return;
      }

      // Auto-align or stage Slot B to match Slot A if needed
      if (slotB && (!ptsB || ptsB.length === 0 || (ptsA && ptsB.length !== ptsA.length))) {
        if (slotB.benchmarkKey && slotB.benchmarkKey !== 'custom' && slotB.dataMode !== 'image') {
          slotB.sampleCount = slotA.sampleCount ||
            Math.round(ptsA.length / (slotA.loopCount || 1));
          slotB.loopCount = slotA.loopCount || 1;
          if (typeof stageDataset === 'function') {
            stageDataset(slotB.benchmarkKey, 'B');
            ptsB = (typeof getSlotPoints === 'function')
              ? getSlotPoints(slotB)
              : (slotB.benchmarkDataset || slotB.pastSamples);
            if (slotB && ptsB) slotB.benchmarkDataset = ptsB;
          }
        }
      }

      /* Resolve clean dataset base names and full filenames */
      let rawNameA = (slotA.stagedDatasetInfo && slotA.stagedDatasetInfo.name)
        ? slotA.stagedDatasetInfo.name
        : (slotA.benchmarkKey || 'dataset_A');
      let rawNameC = (slotC.stagedDatasetInfo && slotC.stagedDatasetInfo.name)
        ? slotC.stagedDatasetInfo.name
        : (slotC.benchmarkKey || 'dataset_C');

      const reExt = /\.(bin|txt|csv|fits|dat|mp4|fits\.fz)$/i;
      const baseA = rawNameA.replace(reExt, '').replace(/[^a-zA-Z0-9_.-]/g, '_');
      const baseC = rawNameC.replace(reExt, '').replace(/[^a-zA-Z0-9_.-]/g, '_');

      const stagedNameC = (baseA === baseC) ? `${baseC}_query_C` : baseC;
      const datasetFileA = `${baseA}.bin`;
      const datasetFileC = `${stagedNameC}.bin`;
      const clusterDir   = `${baseA}.clusterdat`;

      const btnRun  = document.getElementById('btnRunNativeReconQuery');
      const btnKill = document.getElementById('btnKillReconQuery');
      const boxProg = document.getElementById('reconQueryProgressBox');
      const elFill  = document.getElementById('reconQueryProgressFill');
      const elPct   = document.getElementById('reconQueryPct');
      const elFrames= document.getElementById('reconQueryFrames');
      const elEta   = document.getElementById('reconQueryEta');
      const elSpeed = document.getElementById('reconQuerySpeed');
      const elElapsed = document.getElementById('reconQueryElapsed');
      const consoleEl = document.getElementById('cliConsoleLog');

      if (btnRun)  { btnRun.disabled  = true; }
      if (btnKill) { btnKill.disabled = false; }
      if (boxProg) { boxProg.style.display = ''; }
      if (elFill)  { elFill.style.width = '0%'; }
      if (elPct)   { elPct.textContent   = '0%'; }
      if (elFrames){ elFrames.textContent = `0 / ${ptsC.length.toLocaleString()}`; }
      if (elEta)   { elEta.textContent    = 'ETA --'; }
      if (elSpeed) { elSpeed.textContent  = '0 fr/s'; }
      if (elElapsed){ elElapsed.textContent = '0.0 s'; }

      try {
        /* 1. Ensure staged coordinates exist on disk in workspace */
        function detectLocalDim(slot, pts) {
          if (slot && slot.dataMode === 'image') return slot.imageDim || 1024;
          if (pts && pts.length > 0 &&
              (pts[0] instanceof Float32Array || pts[0] instanceof Float64Array)) {
            return pts[0].length;
          }
          if (slot && slot.benchmarkKey && typeof getBenchmarkDim === 'function') {
            return getBenchmarkDim(slot.benchmarkKey);
          }
          if (slot && slot.currentDim) return slot.currentDim;
          if (pts && pts.length > 0) {
            if (pts[0].coords && pts[0].coords.length > 0) return pts[0].coords.length;
            for (let i = 0; i < Math.min(200, pts.length); i++) {
              if (pts[i].z !== undefined && Math.abs(pts[i].z) > 1e-6) return 3;
            }
          }
          return 2;
        }
        const dimA = detectLocalDim(slotA, ptsA);
        const dimC = detectLocalDim(slotC, ptsC);
        if (ptsA && ptsA.length > 0) {
          await DesktopBridge.stageDatasetFile(baseA, ptsA, dimA).catch(() => {});
        }
        if (ptsC && ptsC.length > 0) {
          await DesktopBridge.stageDatasetFile(stagedNameC, ptsC, dimC).catch(() => {});
        }

        /* 2. Check if cluster directory exists on disk; if not, export clusters of A */
        const filesInWorkspace = await DesktopBridge.listFiles().catch(() => []);
        let hasClusterDir = filesInWorkspace.some(
          f => (f.name === clusterDir || f.name === `${baseA}.clusterdat`) && (f.is_dir || f.isDir)
        );

        if (!hasClusterDir) {
          const memClusters = (slotA && slotA.clusters && slotA.clusters.length > 0)
            ? slotA.clusters : (typeof clusters !== 'undefined' ? clusters : []);
          if (memClusters && memClusters.length > 0) {
            let centroidsText = `# GRIC Cluster Centroids\n# ID X Y Z MEMBERS\n`;
            memClusters.forEach(c => {
              centroidsText += `${c.id} ${Number(c.x || 0).toFixed(6)} ${Number(c.y || 0).toFixed(6)} ` +
                               `${Number(c.z || 0).toFixed(6)} ${c.members || 1}\n`;
            });
            let dccText = `# GRIC Cluster-to-Cluster Distance Matrix D_cc\n`;
            const memDcc = (slotA && slotA.dcc && slotA.dcc.length > 0)
              ? slotA.dcc : (typeof dcc !== 'undefined' ? dcc : []);
            if (memDcc && memDcc.length > 0) {
              memDcc.forEach(row => {
                dccText += row.map(v => Number(v).toFixed(6)).join(' ') + '\n';
              });
            }
            let memText = `# Frame Membership Assignments\n`;
            const memPast = (slotA && slotA.pastSamples && slotA.pastSamples.length > 0)
              ? slotA.pastSamples : (typeof pastSamples !== 'undefined' ? pastSamples : []);
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
            await DesktopBridge.exportClusterDat(baseA, exportFiles).catch(() => {});
          } else {
            showToast(`⚡ Clustering Dataset A (${datasetFileA}) for metric bounds...`);
            const clusterArgs = ['-outdir', clusterDir, '0.15', datasetFileA];
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

        /* 3. Read k from the reconstruction k input */
        const inputK = document.getElementById('inputReconK');
        const k = inputK ? Math.max(1, Math.min(parseInt(inputK.value, 10) || 30, ptsA.length)) : 30;

        /* Distinct output path so we don't overwrite the A-vs-A knn results */
        const queryOutPrefix = `${clusterDir}/knn_query_C`;

        const args = [
          datasetFileA,
          clusterDir,
          '-query', datasetFileC,
          '-k', String(k),
          '--all-queries',
          '-progress',
          '-no-txt',
          '-o', queryOutPrefix
        ];

        const chkApprox = document.getElementById('chkReconQueryApprox');
        if (chkApprox && chkApprox.checked) {
          args.push('--approx');
        }

        const chkTrajectory = document.getElementById('chkReconQueryTrajectory');
        if (chkTrajectory && chkTrajectory.checked) {
          args.push('--trajectory');
        } else {
          args.push('--no-trajectory');
        }

        const radRq8 = document.getElementById('radReconQueryRq8');
        const radEq16 = document.getElementById('radReconQueryEq16');
        const radSq16 = document.getElementById('radReconQuerySq16');
        const radSq8 = document.getElementById('radReconQuerySq8');
        const chkSq8 = document.getElementById('chkReconQuerySq8');
        if (radRq8 && radRq8.checked) {
          args.push('-rq8');
        } else if (radEq16 && radEq16.checked) {
          args.push('-eq16', '-eq16-adc');
          if (typeof knnSq16Ratio !== 'undefined' && knnSq16Ratio !== 0.05) {
            args.push('-eq16-ratio', String(knnSq16Ratio));
          }
          args.push('-no-sq16', '-no-sq8', '-no-rq8');
        } else if (radSq16 && radSq16.checked) {
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
        } else if (radSq8 && radSq8.checked) {
          args.push('-sq8');
        } else if (chkSq8 && chkSq8.checked) {
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

        if (consoleEl) {
          consoleEl.textContent +=
            `\n🔍 [Reconstruction] Dispatching native gric-knn -query C (k=${k})...\n` +
            `🗂️  Dataset A:    ${datasetFileA}\n` +
            `🗂️  Query C:      ${datasetFileC}\n` +
            `📁  Cluster Dir:  ${clusterDir}\n` +
            `📄  Output Base:  ${queryOutPrefix}\n` +
            `─────────────────────────────────────────────────────────\n`;
          consoleEl.scrollTop = consoleEl.scrollHeight;
        }

        showToast(`🔍 Running native gric-knn -query C (k=${k})...`);

        const tKnnStart = performance.now();
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

              /* Parse live progress: "Searching k-NN: [...] XX.X% (N / M frames)" */
              const lines = chunk.split(/\r|\n/);
              for (const line of lines) {
                const m = line.match(
                  /Searching k-NN:\s*\[.*?\]\s*([\d.]+)%\s*\(\s*(\d+)\s*\/\s*(\d+)\s*frames\)/
                );
                if (!m) { continue; }

                const pct      = parseFloat(m[1]);
                const processed = parseInt(m[2], 10);
                const total    = parseInt(m[3], 10);
                const now      = performance.now();
                const elapsedSec = Math.max(0.01, (now - tKnnStart) / 1000.0);
                const speed    = processed > 0 ? (processed / elapsedSec) : 0;
                const remaining = Math.max(0, total - processed);
                const etaSec   = speed > 0 ? (remaining / speed) : 0;

                if (elFill)   { elFill.style.width     = `${pct.toFixed(1)}%`; }
                if (elPct)    { elPct.textContent       = `${pct.toFixed(1)}%`; }
                if (elFrames) { elFrames.textContent    = `${processed.toLocaleString()} / ${total.toLocaleString()}`; }
                if (elSpeed)  { elSpeed.textContent     = `${Math.round(speed).toLocaleString()} fr/s`; }
                if (elElapsed){ elElapsed.textContent   = `${elapsedSec.toFixed(1)} s`; }
                if (elEta) {
                  elEta.textContent = etaSec > 1
                    ? `ETA ${etaSec.toFixed(0)} s`
                    : 'ETA < 1 s';
                }
              } // for lines
            },
            onTelemetry: () => {},
            onFinish: (res) => {
              finalExitCode = res?.exitCode ?? 0;
              if (finalExitCode === 0) {
                showToast('✅ gric-knn -query C completed successfully!');
                if (elFill) { elFill.style.width = '100%'; }
                if (elPct)  { elPct.textContent   = '100%'; }
                if (elEta)  { elEta.textContent   = 'Done'; }
              } else {
                showToast(`⚠️ gric-knn -query C finished with exit code ${finalExitCode}`);
              }
              resolve();
            }
          }).catch((err) => {
            showToast(`⚠️ Failed to execute gric-knn: ${err.message}`);
            if (consoleEl) {
              consoleEl.textContent += `\n❌ Error: ${err.message}\n`;
              consoleEl.scrollTop = consoleEl.scrollHeight;
            }
            resolve();
          });
        });

        if (finalExitCode === 0) {
          /* Parse final telemetry from stdout */
          const parsedTelem = DesktopBridge.parseKnnTelemetryLog(rawCliOutput);

          /* Populate tracker card */
          updateReconQueryTracker(parsedTelem, ptsC.length, ptsA.length, k);

          /* Load the query k-NN results from distinct output binary files */
          let queryData = await DesktopBridge.readKnnQueryResults(queryOutPrefix, k).catch(() => null);
          if (queryData && queryData.indices && queryData.totalFrames > 0) {
            slotC.knnResults = queryData;
            slotC.knnResultsForQuery = true;
            showToast(
              `📊 Loaded ${queryData.totalFrames.toLocaleString()} query k-NN results ` +
              `(k=${queryData.k}). Auto-reconstructing D...`
            );
            /* Auto-reconstruct Dataset D with the loaded query k-NN neighbors */
            if (autoReconstruct && typeof executeDatasetReconstruction === 'function') {
              await executeDatasetReconstruction();
            }
          } else {
            showToast('⚠️ Query search completed but result file could not be read.');
          }
        }
      } catch (err) {
        showToast(`⚠️ Error in gric-knn -query: ${err.message}`);
        console.error('[runNativeReconQueryKnn]', err);
      } finally {
        if (btnRun)  { btnRun.disabled  = false; }
        if (btnKill) { btnKill.disabled = true;  }
        if (boxProg) {
          setTimeout(() => {
            boxProg.style.display = 'none';
          }, 800);
        }
      }
    } // runNativeReconQueryKnn


    // =========================================================================
    //  Query k-NN resource tracker tab switching
    // =========================================================================

    {
      const tabOverview = document.getElementById('tabReconQueryOverview');
      const tabPruning  = document.getElementById('tabReconQueryPruning');
      const panelOverview = document.getElementById('reconQueryOverviewPanel');
      const panelPruning  = document.getElementById('reconQueryPruningPanel');

      if (tabOverview && tabPruning) {
        tabOverview.addEventListener('click', () => {
          tabOverview.classList.add('active');
          tabPruning.classList.remove('active');
          if (panelOverview) { panelOverview.style.display = ''; }
          if (panelPruning)  { panelPruning.style.display  = 'none'; }
        });
        tabPruning.addEventListener('click', () => {
          tabPruning.classList.add('active');
          tabOverview.classList.remove('active');
          if (panelPruning)  { panelPruning.style.display  = ''; }
          if (panelOverview) { panelOverview.style.display = 'none'; }
        });
      }
    }

    // Reconstruction UI Listeners
    const btnReconSide = document.getElementById('btnRunReconstructionSide');
    if (btnReconSide) {
      btnReconSide.addEventListener('click', executeDatasetReconstruction);
    }
    const btnReconTop = document.getElementById('btnReconstructDTop');
    if (btnReconTop) {
      btnReconTop.addEventListener('click', executeDatasetReconstruction);
    }
    const btnReconArrow = document.getElementById('btnReconstructArrow');
    if (btnReconArrow) {
      btnReconArrow.addEventListener('click', (e) => {
        e.stopPropagation();
        executeDatasetReconstruction();
      });
    }
    const btnAsteroidReconTop = document.getElementById('btnPresetAsteroidReconTop');
    if (btnAsteroidReconTop) {
      btnAsteroidReconTop.addEventListener('click', async () => {
        if (typeof setupAsteroidReconTest === 'function') {
          await setupAsteroidReconTest(10000, 0.80);
        }
      });
    }
    const btnAsteroidReconSide = document.getElementById('btnPresetAsteroidReconSide');
    if (btnAsteroidReconSide) {
      btnAsteroidReconSide.addEventListener('click', async () => {
        if (typeof setupAsteroidReconTest === 'function') {
          await setupAsteroidReconTest(10000, 0.80);
        }
      });
    }

    /* Native query button — only active in desktop mode */
    const btnNativeQuery = document.getElementById('btnRunNativeReconQuery');
    if (btnNativeQuery) {
      btnNativeQuery.addEventListener('click', runNativeReconQueryKnn);
    }
    const btnKillReconQuery = document.getElementById('btnKillReconQuery');
    if (btnKillReconQuery) {
      btnKillReconQuery.addEventListener('click', () => {
        DesktopBridge.killActiveJob().catch(() => {});
        btnKillReconQuery.disabled = true;
        const btnNQ = document.getElementById('btnRunNativeReconQuery');
        if (btnNQ) { btnNQ.disabled = false; }
      });
    }

    const btnToggleRecon4P = document.getElementById('btnToggleRecon4PanelView');
    if (btnToggleRecon4P) {
      btnToggleRecon4P.addEventListener('click', () => {
        toggleRecon4PanelView();
      });
    }
    const btnToggleRecon4PTop = document.getElementById('btnToggleRecon4PanelTop');
    if (btnToggleRecon4PTop) {
      btnToggleRecon4PTop.addEventListener('click', () => {
        toggleRecon4PanelView();
      });
    }
    const btnPresetRecon4P = document.getElementById('btnPresetRecon4Panel');
    if (btnPresetRecon4P) {
      btnPresetRecon4P.addEventListener('click', () => {
        toggleRecon4PanelView();
      });
    }
    const btnToggleReconKnnTop = document.getElementById('btnToggleReconKnnTop');
    if (btnToggleReconKnnTop) {
      btnToggleReconKnnTop.addEventListener('click', () => {
        setReconKnn(!showReconKnn);
      });
    }
    const btnToggleReconKnnSide = document.getElementById('btnToggleReconKnnSide');
    if (btnToggleReconKnnSide) {
      btnToggleReconKnnSide.addEventListener('click', () => {
        setReconKnn(!showReconKnn);
      });
    }
    const btnToggleReconOverlayTop = document.getElementById('btnToggleReconOverlayTop');
    if (btnToggleReconOverlayTop) {
      btnToggleReconOverlayTop.addEventListener('click', () => {
        setReconOverlayMode();
      });
    }
    const btnToggleReconOverlaySide = document.getElementById('btnToggleReconOverlaySide');
    if (btnToggleReconOverlaySide) {
      btnToggleReconOverlaySide.addEventListener('click', () => {
        setReconOverlayMode();
      });
    }
    const selA = document.getElementById('selectReconPanelAMode');
    if (selA) {
      selA.addEventListener('change', (e) => {
        setReconPanelAMode(e.target.value);
      });
    }
    const selASide = document.getElementById('selectReconPanelAModeSide');
    if (selASide) {
      selASide.addEventListener('change', (e) => {
        setReconPanelAMode(e.target.value);
      });
    }
    const selB = document.getElementById('selectReconPanelBMode');
    if (selB) {
      selB.addEventListener('change', (e) => {
        setReconPanelBMode(e.target.value);
      });
    }
    const selBSide = document.getElementById('selectReconPanelBModeSide');
    if (selBSide) {
      selBSide.addEventListener('change', (e) => {
        setReconPanelBMode(e.target.value);
      });
    }
    const btnInspectD = document.getElementById('btnInspectDatasetDSide');
    if (btnInspectD) {
      btnInspectD.addEventListener('click', () => {
        if (!multiDatasetEnabled) setMultiDatasetEnabled(true);
        switchDatasetSlot('D');
      });
    }
    const btnClearDSide = document.getElementById('btnClearDatasetDSide');
    if (btnClearDSide) {
      btnClearDSide.addEventListener('click', () => {
        clearDatasetSlot('D');
      });
    }

    // Quality coloring & filtering helper: compute mask from percentile
    function updateReconQualityMask()
    {
      const slotC = (typeof datasetSlots !== 'undefined') ? datasetSlots['C'] : null;
      const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;

      // 1. Compute for Slot C (k-th NN distance)
      const arrC = (slotC && slotC.reconKthDist) ? slotC.reconKthDist
                 : (slotD && slotD.reconKthDist) ? slotD.reconKthDist : null;
      let visCountC = 0;
      if (arrC && arrC.length > 0 && reconQualityThreshold < 1.0)
      {
        if (!slotC || !slotC.sortedReconKthDist ||
            slotC.sortedReconKthDist.length !== arrC.length)
        {
          if (slotC) { slotC.sortedReconKthDist = Float64Array.from(arrC).sort(); }
        }
        const sortedC = (slotC && slotC.sortedReconKthDist)
          ? slotC.sortedReconKthDist : Float64Array.from(arrC).sort();
        const pctIdxC = Math.min(
          sortedC.length - 1,
          Math.floor(reconQualityThreshold * sortedC.length)
        );
        const threshC = sortedC[pctIdxC];
        const maskC = new Uint8Array(arrC.length);
        const indicesC = [];
        for (let i = 0; i < arrC.length; i++)
        {
          if (arrC[i] <= threshC)
          {
            maskC[i] = 1;
            indicesC.push(i);
            visCountC++;
          }
        }
        if (slotC)
        {
          slotC.reconQualityMask = maskC;
          slotC.reconQualityIndices = new Int32Array(indicesC);
        }
      }
      else
      {
        if (slotC)
        {
          slotC.reconQualityMask = null;
          slotC.reconQualityIndices = null;
        }
        if (arrC) { visCountC = arrC.length; }
      }

      // 2. Compute for Slot D (Recon Variance, fallback to k-th NN dist)
      const arrD = (slotD && slotD.reconVariance) ? slotD.reconVariance
                 : (slotD && slotD.reconKthDist) ? slotD.reconKthDist : null;
      let visCountD = 0;
      if (arrD && arrD.length > 0 && reconQualityThreshold < 1.0)
      {
        if (!slotD || !slotD.sortedReconVariance ||
            slotD.sortedReconVariance.length !== arrD.length)
        {
          if (slotD) { slotD.sortedReconVariance = Float64Array.from(arrD).sort(); }
        }
        const sortedD = (slotD && slotD.sortedReconVariance)
          ? slotD.sortedReconVariance : Float64Array.from(arrD).sort();
        const pctIdxD = Math.min(
          sortedD.length - 1,
          Math.floor(reconQualityThreshold * sortedD.length)
        );
        const threshD = sortedD[pctIdxD];
        const maskD = new Uint8Array(arrD.length);
        const indicesD = [];
        for (let i = 0; i < arrD.length; i++)
        {
          if (arrD[i] <= threshD)
          {
            maskD[i] = 1;
            indicesD.push(i);
            visCountD++;
          }
        }
        if (slotD)
        {
          slotD.reconQualityMask = maskD;
          slotD.reconQualityIndices = new Int32Array(indicesD);
        }
      }
      else
      {
        if (slotD)
        {
          slotD.reconQualityMask = null;
          slotD.reconQualityIndices = null;
        }
        if (arrD) { visCountD = arrD.length; }
      }

      // 3. Set global reconQualityMask & reconQualityIndices according to activeDatasetSlot
      if (activeDatasetSlot === 'C' && slotC)
      {
        reconQualityMask = slotC.reconQualityMask;
        reconQualityIndices = slotC.reconQualityIndices;
      }
      else if (activeDatasetSlot === 'D' && slotD)
      {
        reconQualityMask = slotD.reconQualityMask;
        reconQualityIndices = slotD.reconQualityIndices;
      }
      else if (slotD && slotD.reconQualityMask)
      {
        reconQualityMask = slotD.reconQualityMask;
        reconQualityIndices = slotD.reconQualityIndices;
      }
      else if (slotC && slotC.reconQualityMask)
      {
        reconQualityMask = slotC.reconQualityMask;
        reconQualityIndices = slotC.reconQualityIndices;
      }
      else
      {
        reconQualityMask = null;
        reconQualityIndices = null;
      }

      // 4. Update UI labels
      const targetArr = (activeDatasetSlot === 'C') ? arrC : (arrD || arrC);
      const targetVis = (activeDatasetSlot === 'C') ? visCountC : (visCountD || visCountC);
      updateReconQualityLabels(targetArr, targetVis);

      // 5. Update Sample Pool Action Button State
      const btnUpdatePool = document.getElementById('btnUpdatePoolToPruned');
      if (btnUpdatePool)
      {
        const activeIdxs = (activeDatasetSlot === 'C')
          ? (slotC && slotC.reconQualityIndices)
          : ((slotD && slotD.reconQualityIndices) || (slotC && slotC.reconQualityIndices));
        if (reconQualityThreshold < 1.0 && activeIdxs && activeIdxs.length > 0)
        {
          btnUpdatePool.disabled = false;
          btnUpdatePool.style.opacity = '1.0';
          btnUpdatePool.style.cursor = 'pointer';
          btnUpdatePool.innerHTML =
            `<span>⚡</span> Update Pool to Pruned (${activeIdxs.length.toLocaleString()} pts)`;
        }
        else
        {
          btnUpdatePool.disabled = true;
          btnUpdatePool.style.opacity = '0.45';
          btnUpdatePool.style.cursor = 'not-allowed';
          btnUpdatePool.innerHTML = '<span>⚡</span> Update Pool to Pruned';
        }
      }
    }

    function updateReconQualityLabels(arr, visCount)
    {
      const lblVis = document.getElementById('lblReconQualityVisible');
      const lblTot = document.getElementById('lblReconQualityTotal');
      const lblMetric = document.getElementById('lblReconQualityMetric');

      if (lblTot && arr)
      {
        lblTot.textContent = arr.length.toLocaleString();
      }
      if (lblVis)
      {
        if (visCount !== undefined)
        {
          lblVis.textContent = visCount.toLocaleString();
        }
        else if (arr)
        {
          lblVis.textContent = arr.length.toLocaleString();
        }
        else
        {
          lblVis.textContent = '--';
        }
      }
      if (lblMetric)
      {
        if (activeDatasetSlot === 'C')
        {
          lblMetric.textContent = 'k-th NN Dist (C)';
        }
        else if (activeDatasetSlot === 'D')
        {
          lblMetric.textContent = 'Recon Variance (D)';
        }
        else if (datasetSlots['D'] && datasetSlots['D'].reconVariance)
        {
          lblMetric.textContent = 'Recon Variance (D)';
        }
        else if (datasetSlots['C'] && datasetSlots['C'].reconKthDist)
        {
          lblMetric.textContent = 'k-th NN Dist (C)';
        }
        else
        {
          lblMetric.textContent = '--';
        }
      }
    }

    // Quality checkbox listener
    const chkQuality = document.getElementById('chkReconQuality');
    if (chkQuality)
    {
      chkQuality.addEventListener('change', () => {
        reconQualityColoringEnabled = chkQuality.checked;
        updateReconQualityMask();
        if (typeof draw === 'function') { draw(); }
      });
    }

    // Quality slider listener
    const sliderQuality = document.getElementById('sliderReconQuality');
    if (sliderQuality)
    {
      sliderQuality.addEventListener('input', () => {
        const pct = parseInt(sliderQuality.value, 10);
        reconQualityThreshold = pct / 100.0;
        const lblPct = document.getElementById('lblReconQualityPct');
        if (lblPct) { lblPct.textContent = pct + '%'; }
        updateReconQualityMask();
        if (typeof draw === 'function') { draw(); }
      });
    }

    window.updateReconQualityMask = updateReconQualityMask;

    function updatePoolToPruned()
    {
      const slotC = datasetSlots['C'];
      const slotD = datasetSlots['D'];
      const indices = (slotD && slotD.reconQualityIndices) ? slotD.reconQualityIndices
                    : (slotC && slotC.reconQualityIndices) ? slotC.reconQualityIndices : null;

      if (!indices || indices.length === 0)
      {
        return;
      }

      // Save backups if not already saved
      if (slotC && !slotC._unprunedBackup)
      {
        slotC._unprunedBackup = {
          benchmarkDataset: slotC.benchmarkDataset,
          rawBenchmarkDataset: slotC.rawBenchmarkDataset,
          pastSamples: slotC.pastSamples,
          sampleCount: slotC.sampleCount,
          stagedDatasetInfo: slotC.stagedDatasetInfo ? { ...slotC.stagedDatasetInfo } : null,
          reconKthDist: slotC.reconKthDist,
          reconKthDistMin: slotC.reconKthDistMin,
          reconKthDistMax: slotC.reconKthDistMax,
          sortedReconKthDist: slotC.sortedReconKthDist
        };
      }
      if (slotD && !slotD._unprunedBackup)
      {
        slotD._unprunedBackup = {
          benchmarkDataset: slotD.benchmarkDataset,
          rawBenchmarkDataset: slotD.rawBenchmarkDataset,
          pastSamples: slotD.pastSamples,
          sampleCount: slotD.sampleCount,
          stagedDatasetInfo: slotD.stagedDatasetInfo ? { ...slotD.stagedDatasetInfo } : null,
          reconstructionInfo: slotD.reconstructionInfo ? { ...slotD.reconstructionInfo } : null,
          reconstructionSourceNeighbors: slotD.reconstructionSourceNeighbors,
          reconVariance: slotD.reconVariance,
          reconVarianceMin: slotD.reconVarianceMin,
          reconVarianceMax: slotD.reconVarianceMax,
          reconKthDist: slotD.reconKthDist,
          reconKthDistMin: slotD.reconKthDistMin,
          reconKthDistMax: slotD.reconKthDistMax,
          sortedReconVariance: slotD.sortedReconVariance,
          sortedReconKthDist: slotD.sortedReconKthDist
        };
      }

      const newCount = indices.length;
      const dimB = slotD ? slotD.currentDim : 3;

      if (slotC && slotC.benchmarkDataset)
      {
        const newPtsC = new Array(newCount);
        const newPastC = new Array(newCount);
        const newKthDistC = slotC.reconKthDist ? new Float64Array(newCount) : null;
        let kthMin = Infinity, kthMax = -Infinity;

        for (let j = 0; j < newCount; j++)
        {
          const origIdx = indices[j];
          const pc = slotC.benchmarkDataset[origIdx];
          if (pc)
          {
            newPtsC[j] = { x: pc.x, y: pc.y, z: pc.z };
            newPastC[j] = {
              x: pc.x,
              y: pc.y,
              z: pc.z || 0.0,
              clusterId: -1,
              frameIndex: j
            };
          }
          if (newKthDistC && slotC.reconKthDist)
          {
            const kd = slotC.reconKthDist[origIdx];
            newKthDistC[j] = kd;
            if (kd < kthMin) kthMin = kd;
            if (kd > kthMax) kthMax = kd;
          }
        }
        slotC.benchmarkDataset = newPtsC;
        slotC.rawBenchmarkDataset = newPtsC;
        slotC.pastSamples = newPastC;
        slotC.sampleCount = newCount;
        if (slotC.stagedDatasetInfo)
        {
          slotC.stagedDatasetInfo.count = newCount;
          if (!slotC.stagedDatasetInfo.name.includes('(pruned)'))
          {
            slotC.stagedDatasetInfo.name += ' (pruned)';
          }
        }
        slotC.reconKthDist = newKthDistC;
        slotC.reconKthDistMin = (kthMin === Infinity) ? 0 : kthMin;
        slotC.reconKthDistMax = (kthMax === -Infinity) ? 1 : kthMax;
        slotC.sortedReconKthDist = newKthDistC ? Float64Array.from(newKthDistC).sort() : null;
      }

      if (slotD && slotD.benchmarkDataset)
      {
        const newPtsD = new Array(newCount);
        const newPastD = new Array(newCount);
        const newVarD = slotD.reconVariance ? new Float64Array(newCount) : null;
        const newKthDistD = slotD.reconKthDist ? new Float64Array(newCount) : null;
        const newNeighborsD = slotD.reconstructionSourceNeighbors ? new Array(newCount) : null;
        let varMin = Infinity, varMax = -Infinity;

        for (let j = 0; j < newCount; j++)
        {
          const origIdx = indices[j];
          const pd = slotD.benchmarkDataset[origIdx];
          if (pd)
          {
            newPtsD[j] = { x: pd.x, y: pd.y, z: pd.z };
            newPastD[j] = {
              x: pd.x,
              y: pd.y,
              z: pd.z || 0.0,
              clusterId: -1,
              frameIndex: j
            };
          }
          if (newVarD && slotD.reconVariance)
          {
            const v = slotD.reconVariance[origIdx];
            newVarD[j] = v;
            if (v < varMin) varMin = v;
            if (v > varMax) varMax = v;
          }
          if (newKthDistD && slotD.reconKthDist)
          {
            newKthDistD[j] = slotD.reconKthDist[origIdx];
          }
          if (newNeighborsD && slotD.reconstructionSourceNeighbors)
          {
            newNeighborsD[j] = slotD.reconstructionSourceNeighbors[origIdx];
          }
        }
        slotD.benchmarkDataset = newPtsD;
        slotD.rawBenchmarkDataset = newPtsD;
        slotD.pastSamples = newPastD;
        slotD.sampleCount = newCount;
        if (slotD.stagedDatasetInfo)
        {
          slotD.stagedDatasetInfo.count = newCount;
          if (!slotD.stagedDatasetInfo.name.includes('(pruned)'))
          {
            slotD.stagedDatasetInfo.name += ' (pruned)';
          }
        }
        if (slotD.reconstructionInfo)
        {
          slotD.reconstructionInfo.queryCount = newCount;
        }
        slotD.reconstructionSourceNeighbors = newNeighborsD;
        slotD._reverseNeighborsIndex = null;
        if (slotC && slotC._onDemandKnnCache) slotC._onDemandKnnCache.clear();
        slotD.reconVariance = newVarD;
        slotD.reconVarianceMin = (varMin === Infinity) ? 0 : varMin;
        slotD.reconVarianceMax = (varMax === -Infinity) ? 1 : varMax;
        slotD.sortedReconVariance = newVarD ? Float64Array.from(newVarD).sort() : null;
        if (slotC)
        {
          slotD.reconKthDist = slotC.reconKthDist;
          slotD.reconKthDistMin = slotC.reconKthDistMin;
          slotD.reconKthDistMax = slotC.reconKthDistMax;
          slotD.sortedReconKthDist = slotC.sortedReconKthDist;
        }
      }

      // Reset quality filter slider to 100%
      reconQualityThreshold = 1.0;
      if (sliderQuality) sliderQuality.value = 100;
      const lblPct = document.getElementById('lblReconQualityPct');
      if (lblPct) lblPct.textContent = '100%';

      if (slotC)
      {
        slotC.reconQualityMask = null;
        slotC.reconQualityIndices = null;
      }
      if (slotD)
      {
        slotD.reconQualityMask = null;
        slotD.reconQualityIndices = null;
      }
      reconQualityMask = null;
      reconQualityIndices = null;

      updateReconQualityMask();

      const btnRestore = document.getElementById('btnRestoreOriginalPool');
      if (btnRestore) btnRestore.style.display = 'inline-flex';

      if (typeof updateDatasetStatusBadge === 'function') updateDatasetStatusBadge();
      const pillD = document.getElementById('datasetStatusPill_D');
      if (pillD && slotD)
      {
        const countD = newCount >= 1000
          ? `${(newCount / 1000).toFixed(newCount % 1000 === 0 ? 0 : 1)}k`
          : newCount;
        pillD.textContent = `📦 ${countD} pts (${dimB}D)`;
      }
      const pillC = document.getElementById('datasetStatusPill_C');
      if (pillC && slotC)
      {
        const countC = newCount >= 1000
          ? `${(newCount / 1000).toFixed(newCount % 1000 === 0 ? 0 : 1)}k`
          : newCount;
        pillC.textContent = `📦 ${countC} pts (${slotC.currentDim || 2}D)`;
      }
      const statPts = document.getElementById('reconStatsPoints');
      if (statPts) statPts.textContent = newCount.toLocaleString();

      if (typeof draw === 'function') draw();
    }

    function restoreOriginalPool()
    {
      const slotC = datasetSlots['C'];
      const slotD = datasetSlots['D'];
      if (slotC && slotC._unprunedBackup)
      {
        Object.assign(slotC, slotC._unprunedBackup);
        delete slotC._unprunedBackup;
      }
      if (slotD && slotD._unprunedBackup)
      {
        Object.assign(slotD, slotD._unprunedBackup);
        delete slotD._unprunedBackup;
      }
      if (slotD) slotD._reverseNeighborsIndex = null;
      if (slotC && slotC._onDemandKnnCache) slotC._onDemandKnnCache.clear();

      const btnRestore = document.getElementById('btnRestoreOriginalPool');
      if (btnRestore) btnRestore.style.display = 'none';

      reconQualityThreshold = 1.0;
      if (sliderQuality) sliderQuality.value = 100;
      const lblPct = document.getElementById('lblReconQualityPct');
      if (lblPct) lblPct.textContent = '100%';

      updateReconQualityMask();
      if (typeof updateDatasetStatusBadge === 'function') updateDatasetStatusBadge();
      const pillD = document.getElementById('datasetStatusPill_D');
      if (pillD && slotD)
      {
        const countD = slotD.sampleCount >= 1000
          ? `${(slotD.sampleCount / 1000).toFixed(slotD.sampleCount % 1000 === 0 ? 0 : 1)}k`
          : slotD.sampleCount;
        pillD.textContent = `📦 ${countD} pts (${slotD.currentDim}D)`;
      }
      const pillC = document.getElementById('datasetStatusPill_C');
      if (pillC && slotC)
      {
        const countC = slotC.sampleCount >= 1000
          ? `${(slotC.sampleCount / 1000).toFixed(slotC.sampleCount % 1000 === 0 ? 0 : 1)}k`
          : slotC.sampleCount;
        pillC.textContent = `📦 ${countC} pts (${slotC.currentDim || 2}D)`;
      }
      const statPts = document.getElementById('reconStatsPoints');
      if (statPts && slotD) statPts.textContent = slotD.sampleCount.toLocaleString();

      if (typeof draw === 'function') draw();
    }

    const btnUpdatePool = document.getElementById('btnUpdatePoolToPruned');
    if (btnUpdatePool)
    {
      btnUpdatePool.addEventListener('click', updatePoolToPruned);
    }
    const btnRestorePool = document.getElementById('btnRestoreOriginalPool');
    if (btnRestorePool)
    {
      btnRestorePool.addEventListener('click', restoreOriginalPool);
    }
    window.updatePoolToPruned = updatePoolToPruned;
    window.restoreOriginalPool = restoreOriginalPool;

    const inputReconK = document.getElementById('inputReconK');
    const sliderReconK = document.getElementById('sliderReconK');
    if (inputReconK && sliderReconK) {
      inputReconK.addEventListener('input', () => {
        sliderReconK.value = inputReconK.value;
      });
      sliderReconK.addEventListener('input', () => {
        inputReconK.value = sliderReconK.value;
      });
    }

    const selectReconWeight = document.getElementById('selectReconWeight');
    const reconIdwRow = document.getElementById('reconIdwRow');
    if (selectReconWeight && reconIdwRow) {
      selectReconWeight.addEventListener('change', () => {
        reconIdwRow.style.display = (selectReconWeight.value === 'idw') ? 'flex' : 'none';
      });
    }

