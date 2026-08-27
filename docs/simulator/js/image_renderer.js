/**
 * GRIC Simulator - image_renderer.js
 * 4-Quadrant Raster Image Viewport & Cluster Navigation System
 *
 * Quad 0 (Top-Left):     Current Query / Inspected Frame
 * Quad 1 (Top-Right):    Cluster Anchor (Click to toggle Residual heatmap)
 * Quad 2 (Bottom-Left):  Members of Current Cluster (with current frame highlighted)
 * Quad 3 (Bottom-Right): Full Set of Clusters (with current cluster highlighted)
 */

/* eslint-disable no-unused-vars */

// Dedicated offscreen canvas cache by slot to prevent canvas buffer collision
const _offscreenCanvases = {};

function _getOffscreenCanvas(slot, w, h)
{
  let entry = _offscreenCanvases[slot];
  if (!entry || entry.canvas.width !== w || entry.canvas.height !== h)
  {
    const canvas = (typeof OffscreenCanvas !== 'undefined')
      ? new OffscreenCanvas(w, h)
      : document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const data = ctx.createImageData(w, h);
    entry = { canvas, ctx, data };
    _offscreenCanvases[slot] = entry;
  }
  return entry;
}

/**
 * Retrieve member frame indices for a cluster.
 */
function getClusterMembersList(clusterId)
{
  if (clusterId < 0) return [];
  if (typeof imageClusterMembers !== 'undefined' &&
      imageClusterMembers &&
      imageClusterMembers[clusterId] &&
      imageClusterMembers[clusterId].length > 0)
  {
    return imageClusterMembers[clusterId];
  }
  if (typeof assignmentHistory !== 'undefined' &&
      assignmentHistory &&
      assignmentHistory.length > 0)
  {
    const list = [];
    for (let i = 0; i < assignmentHistory.length; i++)
    {
      if (assignmentHistory[i] === clusterId)
      {
        list.push(i);
      }
    }
    return list;
  }
  if (typeof imageFrameAssignments !== 'undefined' &&
      imageFrameAssignments &&
      imageFrameAssignments.length > 0)
  {
    const list = [];
    for (let i = 0; i < imageFrameAssignments.length; i++)
    {
      if (imageFrameAssignments[i] === clusterId)
      {
        list.push(i);
      }
    }
    return list;
  }
  return [];
}

/**
 * Return array of cluster indices sorted by the active sort mode.
 */
function getSortedClusterIndices()
{
  if (!clusters || clusters.length === 0) return [];
  const indices = clusters.map((_, i) => i);
  if (typeof imageClustersSortMode === 'undefined' || imageClustersSortMode === 'id')
  {
    return indices;
  }
  if (imageClustersSortMode === 'size_desc')
  {
    indices.sort((a, b) => {
      const countA = (clusters[a] && clusters[a].members) || getClusterMembersList(a).length || 0;
      const countB = (clusters[b] && clusters[b].members) || getClusterMembersList(b).length || 0;
      if (countB !== countA) return countB - countA;
      return a - b;
    });
  }
  else if (imageClustersSortMode === 'size_asc')
  {
    indices.sort((a, b) => {
      const countA = (clusters[a] && clusters[a].members) || getClusterMembersList(a).length || 0;
      const countB = (clusters[b] && clusters[b].members) || getClusterMembersList(b).length || 0;
      if (countA !== countB) return countA - countB;
      return a - b;
    });
  }
  return indices;
}

/**
 * Cycle cluster sorting mode in image mode: id -> size_desc -> size_asc -> id.
 */
function cycleImageClusterSortMode()
{
  if (typeof imageClustersSortMode === 'undefined' || imageClustersSortMode === 'id')
  {
    imageClustersSortMode = 'size_desc';
  }
  else if (imageClustersSortMode === 'size_desc')
  {
    imageClustersSortMode = 'size_asc';
  }
  else
  {
    imageClustersSortMode = 'id';
  }

  const sel = document.getElementById('selectImgClusterSort');
  if (sel)
  {
    sel.value = imageClustersSortMode;
  }

  const label = (imageClustersSortMode === 'size_desc')
    ? '📊 Sorted by Cluster Size (Descending: Largest first)'
    : (imageClustersSortMode === 'size_asc')
      ? '📉 Sorted by Cluster Size (Ascending: Smallest first)'
      : '🔢 Sorted by Creation ID (Default)';

  if (typeof showToast === 'function')
  {
    showToast(label);
  }
  if (typeof draw === 'function') draw();
}

/**
 * Convert a float pixel array into ImageData and draw onto target canvas.
 * @param {CanvasRenderingContext2D} targetCtx - Destination 2D context
 * @param {ArrayLike<number>} pixels - Float pixel buffer of size W * H
 * @param {number} imgW - Image width (e.g. 32)
 * @param {number} imgH - Image height (e.g. 32)
 * @param {number} dstX - Target destination X
 * @param {number} dstY - Target destination Y
 * @param {number} dstW - Target destination width
 * @param {number} dstH - Target destination height
 * @param {number} maxVal - Max normalization value (default: 1.0)
 * @param {string} slot - Dedicated canvas slot name
 */
function drawRasterBuffer(
  targetCtx,
  pixels,
  imgW,
  imgH,
  dstX,
  dstY,
  dstW,
  dstH,
  maxVal = 1.0,
  slot = 'default'
)
{
  if (!pixels || pixels.length < imgW * imgH) return;

  const { canvas, ctx, data } = _getOffscreenCanvas(slot, imgW, imgH);
  const numPix = imgW * imgH;
  const scale = maxVal > 0 ? 255.0 / maxVal : 255.0;
  const d = data.data;

  for (let i = 0; i < numPix; i++)
  {
    const val = pixels[i];
    const lum = Math.max(0, Math.min(255, Math.round(val * scale)));
    const idx = i * 4;
    d[idx] = lum;     // R
    d[idx + 1] = lum; // G
    d[idx + 2] = lum; // B
    d[idx + 3] = 255; // A
  }

  ctx.putImageData(data, 0, 0);

  targetCtx.imageSmoothingEnabled = false;
  targetCtx.drawImage(
    canvas,
    0, 0, imgW, imgH,
    Math.round(dstX), Math.round(dstY), Math.round(dstW), Math.round(dstH)
  );
}

/**
 * Retrieve list of k-NN nearest neighbors for activeFrameIdx.
 */
function getActiveImageKnnNeighbors(activeFrameIdx)
{
  if (typeof knnResults === 'undefined' || !knnResults || !knnResults.indices)
  {
    return [];
  }
  const k = knnResults.k || (typeof knnK !== 'undefined' ? knnK : 10);
  const totalQ = knnResults.totalFrames || Math.floor(knnResults.indices.length / k);
  if (activeFrameIdx < 0 || activeFrameIdx >= totalQ)
  {
    return [];
  }
  const list = [];
  for (let r = 0; r < k; r++)
  {
    const nIdx = knnResults.indices[activeFrameIdx * k + r];
    let dist = (knnResults.distances && knnResults.distances[activeFrameIdx * k + r] !== undefined)
      ? Number(knnResults.distances[activeFrameIdx * k + r])
      : NaN;

    // Robust Fallback: If distance is NaN / not finite, compute Euclidean metric directly
    if (isNaN(dist) || !isFinite(dist))
    {
      if (typeof benchmarkDataset !== 'undefined' && benchmarkDataset &&
          benchmarkDataset[activeFrameIdx] && benchmarkDataset[nIdx])
      {
        const fA = benchmarkDataset[activeFrameIdx];
        const fB = benchmarkDataset[nIdx];
        let sumSq = 0.0;
        if (Array.isArray(fA) || ArrayBuffer.isView(fA))
        {
          const len = Math.min(fA.length, fB.length);
          for (let p = 0; p < len; p++)
          {
            const diff = fA[p] - fB[p];
            sumSq += diff * diff;
          }
        }
        else if (typeof fA === 'object')
        {
          const dx = (fA.x || 0) - (fB.x || 0);
          const dy = (fA.y || 0) - (fB.y || 0);
          const dz = (fA.z || 0) - (fB.z || 0);
          sumSq = dx * dx + dy * dy + dz * dz;
        }
        dist = Math.sqrt(sumSq);
      }
      else
      {
        dist = 0.0;
      }
    }

    if (nIdx >= 0)
    {
      const cId = (imageFrameAssignments && imageFrameAssignments[nIdx] !== undefined)
        ? imageFrameAssignments[nIdx]
        : -1;
      list.push({
        rank: r + 1,
        frameIdx: nIdx,
        dist: dist,
        clusterId: cId
      });
    }
  }
  return list;
}

/**
 * Get current thumbnail size in pixels for image galleries.
 */
function getImageThumbSize()
{
  if (typeof imageThumbSize !== 'undefined' && imageThumbSize >= 32)
  {
    return imageThumbSize;
  }
  return 64;
}

/**
 * Get card layout & badge typography with info box placed below the image.
 * @param {number} thumbSize - Current thumbnail size in pixels.
 * @param {boolean} [isTwoLine=false] - True for 2-line cards (e.g. k-NN).
 */
function getThumbCardStyle(thumbSize, isTwoLine)
{
  if (isTwoLine)
  {
    if (thumbSize <= 50)
    {
      return { infoH: 22, font: '7.5px monospace', fontBold: 'bold 7.5px monospace' };
    }
    if (thumbSize <= 75)
    {
      return { infoH: 26, font: '8.5px monospace', fontBold: 'bold 8.5px monospace' };
    }
    if (thumbSize <= 110)
    {
      return { infoH: 30, font: '10px monospace', fontBold: 'bold 10px monospace' };
    }
    if (thumbSize <= 160)
    {
      return { infoH: 36, font: '11.5px monospace', fontBold: 'bold 11.5px monospace' };
    }
    return { infoH: 42, font: '13px monospace', fontBold: 'bold 13px monospace' };
  }

  if (thumbSize <= 50)
  {
    return { infoH: 16, font: '8px monospace', fontBold: 'bold 8px monospace' };
  }
  if (thumbSize <= 75)
  {
    return { infoH: 20, font: '9.5px monospace', fontBold: 'bold 9.5px monospace' };
  }
  if (thumbSize <= 110)
  {
    return { infoH: 24, font: '11px monospace', fontBold: 'bold 11px monospace' };
  }
  if (thumbSize <= 160)
  {
    return { infoH: 28, font: '12.5px monospace', fontBold: 'bold 12.5px monospace' };
  }
  return { infoH: 32, font: '14px monospace', fontBold: 'bold 14px monospace' };
}

/**
 * Get quadrant rectangle for quad index in image mode.
 */
function getImageQuadRect(qIdx, W, H)
{
  if (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null)
  {
    return { x: 0, y: 0, w: W, h: H };
  }
  const halfW = W / 2;
  const halfH = H / 2;
  switch (qIdx)
  {
    case 0: return { x: 0, y: 0, w: halfW, h: halfH };        // Top-Left: Current Frame
    case 1: return { x: halfW, y: 0, w: halfW, h: halfH };    // Top-Right: Anchor / Residual
    case 2: return { x: 0, y: halfH, w: halfW, h: halfH };    // Bottom-Left: Members
    case 3: return { x: halfW, y: halfH, w: halfW, h: halfH };// Bottom-Right: All Clusters
    default: return { x: 0, y: 0, w: W, h: H };
  }
}

/**
 * Render image-mode 4-quadrant viewport layout on the main canvas.
 * @param {CanvasRenderingContext2D} ctx - Main canvas context
 * @param {number} W - Canvas width in CSS pixels
 * @param {number} H - Canvas height in CSS pixels
 */
function drawImageMode(ctx, W, H)
{
  ctx.save();

  if (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null)
  {
    renderImageSubViewport(ctx, maximizedQuad, { x: 0, y: 0, w: W, h: H });
    drawRestoreBanner(ctx, W, H, maximizedQuad);
    ctx.restore();
    return;
  }

  // Render 4 Quadrants
  renderImageSubViewport(ctx, 0, getImageQuadRect(0, W, H));
  renderImageSubViewport(ctx, 1, getImageQuadRect(1, W, H));
  renderImageSubViewport(ctx, 2, getImageQuadRect(2, W, H));
  renderImageSubViewport(ctx, 3, getImageQuadRect(3, W, H));

  // Draw Divider Grid Lines
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(W / 2, 0);
  ctx.lineTo(W / 2, H);
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();

  ctx.restore();
}

/**
 * Render an individual sub-viewport in image mode.
 */
function renderImageSubViewport(ctx, qIdx, rect)
{
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();

  const pad = 12;
  const headerH = 26;
  const contentX = rect.x + pad;
  const contentY = rect.y + headerH + 6;
  const contentW = rect.w - pad * 2;
  const contentH = rect.h - headerH - pad - 6;

  // Header background bar
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(rect.x, rect.y, rect.w, headerH);
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(rect.x, rect.y + headerH);
  ctx.lineTo(rect.x + rect.w, rect.y + headerH);
  ctx.stroke();

  // Determine active frame and assigned cluster
  const isRetro = (typeof inspectedImageFrameIdx !== 'undefined' &&
    inspectedImageFrameIdx >= 0 &&
    benchmarkDataset && benchmarkDataset[inspectedImageFrameIdx]);
  const activeFrameIdx = isRetro
    ? inspectedImageFrameIdx
    : (totalFrames > 0
      ? totalFrames - 1
      : (benchmarkDataset && benchmarkDataset.length > 0 ? 0 : -1));
  const frameBuf = isRetro
    ? benchmarkDataset[inspectedImageFrameIdx]
    : (currentImageFrame ||
      (benchmarkDataset && benchmarkDataset.length > 0
        ? benchmarkDataset[Math.min(currentFrameIdx, benchmarkDataset.length - 1)]
        : null));

  let assignedId = -1;
  let lastDist = 0;
  if (isRetro)
  {
    const hasAssigned = imageFrameAssignments &&
      imageFrameAssignments[inspectedImageFrameIdx] !== undefined;
    assignedId = hasAssigned ? imageFrameAssignments[inspectedImageFrameIdx] : -1;
    lastDist = (imageFrameDists && imageFrameDists[inspectedImageFrameIdx] !== undefined)
      ? imageFrameDists[inspectedImageFrameIdx]
      : 0;
  }
  else
  {
    assignedId = typeof prevAssignedCluster !== 'undefined' ? prevAssignedCluster : -1;
    lastDist = typeof distSampleClusterLast !== 'undefined' ? distSampleClusterLast : 0;
  }

  // Fallback assignedId from clusters if needed
  if (assignedId < 0 && clusters && clusters.length > 0)
  {
    assignedId = (typeof selectedClusterId !== 'undefined' && selectedClusterId >= 0)
      ? selectedClusterId : 0;
  }

  const assignedCluster = (assignedId >= 0 && clusters[assignedId]) ? clusters[assignedId] : null;

  // -------------------------------------------------------------
  // Q0: Top-Left - Current Frame
  // -------------------------------------------------------------
  if (qIdx === 0)
  {
    const title = isRetro
      ? `🔍 Inspected Frame #${activeFrameIdx + 1}`
      : '📸 Current Frame';
    drawHeader(ctx, rect.x + 8, rect.y + 17, title, isRetro ? '#facc15' : '#38bdf8');

    if (frameBuf)
    {
      const size = Math.max(32, Math.min(contentW, contentH) - 16);
      const imgX = contentX + (contentW - size) / 2;
      const imgY = contentY + (contentH - size) / 2;

      ctx.fillStyle = '#020617';
      ctx.fillRect(imgX - 2, imgY - 2, size + 4, size + 4);
      ctx.strokeStyle = isRetro ? '#facc15' : '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(imgX - 2, imgY - 2, size + 4, size + 4);

      drawRasterBuffer(
        ctx,
        frameBuf,
        imageWidth,
        imageHeight,
        imgX,
        imgY,
        size,
        size,
        1.0,
        'quad0'
      );

      const totalDatasetCount = (benchmarkDataset && benchmarkDataset.length > 0)
        ? benchmarkDataset.length
        : totalFrames;
      const frameNum = activeFrameIdx >= 0
        ? activeFrameIdx + 1
        : (totalFrames > 0 ? totalFrames : 1);
      drawBadge(
        ctx,
        rect.x + 8,
        rect.y + rect.h - 8,
        `Frame #${frameNum}/${totalDatasetCount} | ${imageWidth}×${imageHeight} (D=${imageDim})`,
        isRetro ? '#facc15' : '#94a3b8'
      );
    }
    else
    {
      drawEmptyMessage(ctx, rect, 'Press Play or Step to ingest frames');
    }
  }

  // -------------------------------------------------------------
  // Q1: Top-Right - Cluster Anchor / Residual / k-NN #1 Comparison
  // -------------------------------------------------------------
  else if (qIdx === 1)
  {
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    const hasKnn = (knnList.length > 0);
    const nn1 = hasKnn ? knnList[0] : null;

    let mode = (typeof imageTopRightMode !== 'undefined') ? imageTopRightMode : 'anchor';
    if ((mode === 'nn1' || mode === 'nn1_diff') && !hasKnn)
    {
      mode = 'anchor';
    }

    const maxOffset = (typeof maximizedQuad === 'undefined' || maximizedQuad === null) ? 28 : 0;

    if (mode === 'anchor')
    {
      // --- ANCHOR VIEW ---
      const title = assignedCluster
        ? `🎯 Cluster Anchor (C${assignedCluster.id})`
        : '🎯 Cluster Anchor';
      const titleColor = assignedCluster ? (assignedCluster.color || '#4ade80') : '#94a3b8';
      drawHeader(ctx, rect.x + 8, rect.y + 17, title, titleColor);

      const nextLabel = hasKnn ? '⇄ Residual (1/4)' : '⇄ Residual (1/2)';
      drawToggleChip(ctx, rect.x + rect.w - 120 - maxOffset, rect.y + 4, 112, 18, nextLabel);

      if (assignedCluster && assignedCluster.anchor)
      {
        const size = Math.max(32, Math.min(contentW, contentH) - 16);
        const imgX = contentX + (contentW - size) / 2;
        const imgY = contentY + (contentH - size) / 2;

        ctx.fillStyle = '#020617';
        ctx.fillRect(imgX - 2, imgY - 2, size + 4, size + 4);
        ctx.strokeStyle = assignedCluster.color || '#4ade80';
        ctx.lineWidth = 2.0;
        ctx.strokeRect(imgX - 2, imgY - 2, size + 4, size + 4);

        drawRasterBuffer(
          ctx,
          assignedCluster.anchor,
          imageWidth,
          imageHeight,
          imgX,
          imgY,
          size,
          size,
          1.0,
          'quad1_anchor'
        );

        const isMatch = lastDist <= (rlim || 0.1);
        drawBadge(
          ctx,
          rect.x + 8,
          rect.y + rect.h - 8,
          `Members: ${assignedCluster.members} | d(f,c): ${lastDist.toFixed(3)} | Cycle ⇄`,
          isMatch ? '#4ade80' : '#facc15'
        );
      }
      else
      {
        drawEmptyMessage(ctx, rect, 'No anchor matched yet');
      }
    }
    else if (mode === 'residual')
    {
      // --- RESIDUAL VIEW ---
      const title = assignedCluster
        ? `⚡ Residual |f - C${assignedCluster.id}|`
        : '⚡ Residual |f - C_anchor|';
      drawHeader(ctx, rect.x + 8, rect.y + 17, title, '#f87171');

      const nextLabel = hasKnn ? '⇄ NN #1 (2/4)' : '⇄ Anchor (2/2)';
      drawToggleChip(ctx, rect.x + rect.w - 120 - maxOffset, rect.y + 4, 112, 18, nextLabel);

      if (frameBuf && assignedCluster && assignedCluster.anchor)
      {
        const size = Math.max(32, Math.min(contentW, contentH) - 16);
        const imgX = contentX + (contentW - size) / 2;
        const imgY = contentY + (contentH - size) / 2;

        const numPix = imageWidth * imageHeight;
        const diffBuf = new Float32Array(numPix);
        let sumSq = 0.0;
        let maxDiff = 0.0;

        for (let p = 0; p < numPix; p++)
        {
          const diff = Math.abs(frameBuf[p] - assignedCluster.anchor[p]);
          diffBuf[p] = diff;
          sumSq += diff * diff;
          if (diff > maxDiff) maxDiff = diff;
        }

        const rms = Math.sqrt(sumSq / numPix);
        const l2 = Math.sqrt(sumSq);

        ctx.fillStyle = '#020617';
        ctx.fillRect(imgX - 2, imgY - 2, size + 4, size + 4);
        ctx.strokeStyle = '#f87171';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(imgX - 2, imgY - 2, size + 4, size + 4);

        drawRasterBuffer(
          ctx,
          diffBuf,
          imageWidth,
          imageHeight,
          imgX,
          imgY,
          size,
          size,
          Math.max(0.5, maxDiff),
          'quad1_residual'
        );

        drawBadge(
          ctx,
          rect.x + 8,
          rect.y + rect.h - 8,
          `L2: ${l2.toFixed(3)} | RMS: ${rms.toFixed(4)} | Max: ${maxDiff.toFixed(3)} | Cycle ⇄`,
          '#94a3b8'
        );
      }
      else
      {
        drawEmptyMessage(ctx, rect, 'Awaiting frame and anchor evaluation');
      }
    }
    else if (mode === 'nn1')
    {
      // --- NEAREST NEIGHBOR #1 VIEW ---
      const nnBuf = (nn1 && benchmarkDataset) ? benchmarkDataset[nn1.frameIdx] : null;
      const title = nn1
        ? `🥇 NN #1 (Frame #${nn1.frameIdx + 1}, C${nn1.clusterId >= 0 ? nn1.clusterId : '?'})`
        : '🥇 Nearest Neighbor #1';
      drawHeader(ctx, rect.x + 8, rect.y + 17, title, '#facc15');

      drawToggleChip(
        ctx, rect.x + rect.w - 120 - maxOffset, rect.y + 4, 112, 18, '⇄ NN Diff (3/4)'
      );

      if (nnBuf && nn1)
      {
        const size = Math.max(32, Math.min(contentW, contentH) - 16);
        const imgX = contentX + (contentW - size) / 2;
        const imgY = contentY + (contentH - size) / 2;
        const dt = nn1.frameIdx - activeFrameIdx;
        const dtStr = (dt >= 0) ? `+${dt}` : `${dt}`;

        ctx.fillStyle = '#020617';
        ctx.fillRect(imgX - 2, imgY - 2, size + 4, size + 4);
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2.0;
        ctx.strokeRect(imgX - 2, imgY - 2, size + 4, size + 4);

        drawRasterBuffer(
          ctx,
          nnBuf,
          imageWidth,
          imageHeight,
          imgX,
          imgY,
          size,
          size,
          1.0,
          'quad1_nn1'
        );

        drawBadge(
          ctx,
          rect.x + 8,
          rect.y + rect.h - 8,
          `F#${nn1.frameIdx + 1} | Δt: ${dtStr} f | d: ${nn1.dist.toFixed(4)} | Cycle ⇄`,
          '#facc15'
        );
      }
      else
      {
        drawEmptyMessage(ctx, rect, 'No k-NN neighbor #1 available');
      }
    }
    else if (mode === 'nn1_diff')
    {
      // --- NEAREST NEIGHBOR #1 DIFFERENCE VIEW ---
      const nnBuf = (nn1 && benchmarkDataset) ? benchmarkDataset[nn1.frameIdx] : null;
      const title = nn1
        ? `⚡ NN #1 Diff |f - F#${nn1.frameIdx + 1}|`
        : '⚡ NN #1 Difference';
      drawHeader(ctx, rect.x + 8, rect.y + 17, title, '#c084fc');

      drawToggleChip(
        ctx, rect.x + rect.w - 120 - maxOffset, rect.y + 4, 112, 18, '⇄ Anchor (4/4)'
      );

      if (frameBuf && nnBuf && nn1)
      {
        const size = Math.max(32, Math.min(contentW, contentH) - 16);
        const imgX = contentX + (contentW - size) / 2;
        const imgY = contentY + (contentH - size) / 2;

        const numPix = imageWidth * imageHeight;
        const diffBuf = new Float32Array(numPix);
        let sumSq = 0.0;
        let maxDiff = 0.0;

        for (let p = 0; p < numPix; p++)
        {
          const diff = Math.abs(frameBuf[p] - nnBuf[p]);
          diffBuf[p] = diff;
          sumSq += diff * diff;
          if (diff > maxDiff) maxDiff = diff;
        }

        const rms = Math.sqrt(sumSq / numPix);
        const l2 = Math.sqrt(sumSq);

        ctx.fillStyle = '#020617';
        ctx.fillRect(imgX - 2, imgY - 2, size + 4, size + 4);
        ctx.strokeStyle = '#c084fc';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(imgX - 2, imgY - 2, size + 4, size + 4);

        drawRasterBuffer(
          ctx,
          diffBuf,
          imageWidth,
          imageHeight,
          imgX,
          imgY,
          size,
          size,
          Math.max(0.5, maxDiff),
          'quad1_nn1_diff'
        );

        drawBadge(
          ctx,
          rect.x + 8,
          rect.y + rect.h - 8,
          `L2: ${l2.toFixed(3)} | RMS: ${rms.toFixed(4)} | d: ${nn1.dist.toFixed(4)} | Cycle ⇄`,
          '#c084fc'
        );
      }
      else
      {
        drawEmptyMessage(ctx, rect, 'No k-NN neighbor #1 available');
      }
    }
  }

  // -------------------------------------------------------------
  // Q2: Bottom-Left - Members of Current Cluster OR k-NN Neighbors
  // -------------------------------------------------------------
  else if (qIdx === 2)
  {
    const isKnnView = (typeof imageQ2ViewMode !== 'undefined' && imageQ2ViewMode === 'knn');
    const members = (assignedId >= 0) ? getClusterMembersList(assignedId) : [];
    const mCount = members.length;
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    const kCount = knnList.length;
    const maxOffset = (typeof maximizedQuad === 'undefined' || maximizedQuad === null) ? 28 : 0;

    if (!isKnnView)
    {
      // --- CLUSTER MEMBERS VIEW ---
      const clColor = assignedCluster ? (assignedCluster.color || '#a78bfa') : '#94a3b8';
      const title = assignedCluster
        ? `👥 C${assignedId} Members (${mCount} frames)`
        : '👥 Current Cluster Members';
      drawHeader(ctx, rect.x + 8, rect.y + 17, title, clColor);

      const knnLabel = (kCount > 0) ? `⇄ k-NN (${kCount})` : '⇄ k-NN';
      drawToggleChip(ctx, rect.x + rect.w - 100 - maxOffset, rect.y + 4, 92, 18, knnLabel);

      if (mCount > 0)
      {
        const thumbSize = getImageThumbSize();
        const style = getThumbCardStyle(thumbSize);
        const cardW = thumbSize;
        const cardH = thumbSize + style.infoH;
        const gap = 8;
        const cols = Math.max(1, Math.floor(contentW / (cardW + gap)));
        const rows = Math.ceil(mCount / cols);
        const totalH = rows * (cardH + gap);

        const maxScroll = Math.max(0, totalH - contentH);
        imageMembersScrollY = Math.max(0, Math.min(maxScroll, imageMembersScrollY || 0));

        ctx.save();
        ctx.beginPath();
        ctx.rect(contentX, contentY, contentW, contentH);
        ctx.clip();

        for (let i = 0; i < mCount; i++)
        {
          const col = i % cols;
          const row = Math.floor(i / cols);
          const tx = contentX + col * (cardW + gap);
          const ty = contentY + row * (cardH + gap) - (imageMembersScrollY || 0);

          if (ty + cardH < contentY - 10 || ty > contentY + contentH + 10)
          {
            continue;
          }

          const memberFrameIdx = members[i];
          const rasterData = benchmarkDataset ? benchmarkDataset[memberFrameIdx] : null;
          const isCurrent = (memberFrameIdx === activeFrameIdx);
          const thumbLabel = `#${memberFrameIdx + 1}`;
          const badgeColor = isCurrent ? '#facc15' : '#cbd5e1';

          if (!rasterData) continue;

          // 1. Fully displayed image raster (unobstructed)
          ctx.fillStyle = '#020617';
          ctx.fillRect(tx, ty, thumbSize, thumbSize);
          ctx.strokeStyle = isCurrent ? '#facc15' : '#334155';
          ctx.lineWidth = isCurrent ? 2.5 : 1.0;
          ctx.strokeRect(tx, ty, thumbSize, thumbSize);

          drawRasterBuffer(
            ctx,
            rasterData,
            imageWidth,
            imageHeight,
            tx,
            ty,
            thumbSize,
            thumbSize,
            1.0,
            'q2_thumb_' + i
          );

          // 2. Info text box below the image
          const infoY = ty + thumbSize;
          ctx.fillStyle = isCurrent ? 'rgba(250, 204, 21, 0.2)' : 'rgba(15, 23, 42, 0.95)';
          ctx.fillRect(tx, infoY, thumbSize, style.infoH);
          ctx.strokeStyle = isCurrent ? '#facc15' : '#334155';
          ctx.lineWidth = 1.0;
          ctx.strokeRect(tx, infoY, thumbSize, style.infoH);

          ctx.fillStyle = badgeColor;
          ctx.font = isCurrent ? style.fontBold : style.font;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            isCurrent ? `${thumbLabel} ★` : thumbLabel,
            tx + thumbSize / 2,
            infoY + style.infoH / 2
          );
        }

        if (maxScroll > 0)
        {
          const scrollbarH = Math.max(16, (contentH / totalH) * contentH);
          const scrollbarY = contentY + (imageMembersScrollY / maxScroll) * (contentH - scrollbarH);
          ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
          ctx.fillRect(rect.x + rect.w - 6, scrollbarY, 4, scrollbarH);
        }

        ctx.restore();
      }
      else
      {
        drawEmptyMessage(ctx, rect, 'No member frames in this cluster yet');
      }
    }
    else
    {
      // --- k-NN NEAREST NEIGHBORS VIEW ---
      drawHeader(ctx, rect.x + 8, rect.y + 17, `⚡ k-NN Neighbors (k=${kCount})`, '#c084fc');

      const memLabel = `⇄ Members (${mCount})`;
      drawToggleChip(ctx, rect.x + rect.w - 116 - maxOffset, rect.y + 4, 108, 18, memLabel);

      if (kCount > 0)
      {
        const thumbSize = getImageThumbSize();
        const style = getThumbCardStyle(thumbSize, true);
        const cardW = thumbSize;
        const cardH = thumbSize + style.infoH;
        const gap = 8;
        const cols = Math.max(1, Math.floor(contentW / (cardW + gap)));
        const rows = Math.ceil(kCount / cols);
        const totalH = rows * (cardH + gap);

        const maxScroll = Math.max(0, totalH - contentH);
        imageKnnScrollY = Math.max(0, Math.min(maxScroll, imageKnnScrollY || 0));

        ctx.save();
        ctx.beginPath();
        ctx.rect(contentX, contentY, contentW, contentH);
        ctx.clip();

        for (let r = 0; r < kCount; r++)
        {
          const item = knnList[r];
          const col = r % cols;
          const row = Math.floor(r / cols);
          const tx = contentX + col * (cardW + gap);
          const ty = contentY + row * (cardH + gap) - (imageKnnScrollY || 0);

          if (ty + cardH < contentY - 10 || ty > contentY + contentH + 10)
          {
            continue;
          }

          const rasterData = benchmarkDataset ? benchmarkDataset[item.frameIdx] : null;
          const isSelected = (item.frameIdx === activeFrameIdx);
          const rankColor = (r === 0)
            ? '#facc15'
            : (r === 1 ? '#cbd5e1' : (r === 2 ? '#fb923c' : '#38bdf8'));
          const cl = (item.clusterId >= 0 && clusters && clusters[item.clusterId])
            ? clusters[item.clusterId] : null;
          const clColor = cl ? (cl.color || '#a78bfa') : '#64748b';

          if (!rasterData) continue;

          // 1. Fully displayed image raster (unobstructed)
          ctx.fillStyle = '#020617';
          ctx.fillRect(tx, ty, thumbSize, thumbSize);
          ctx.strokeStyle = isSelected ? '#facc15' : (r === 0 ? '#facc15' : clColor);
          ctx.lineWidth = isSelected ? 2.5 : 1.5;
          ctx.strokeRect(tx, ty, thumbSize, thumbSize);

          drawRasterBuffer(
            ctx,
            rasterData,
            imageWidth,
            imageHeight,
            tx,
            ty,
            thumbSize,
            thumbSize,
            1.0,
            'q2_knn_thumb_' + r
          );

          // 2. Info text box below the image (2 distinct lines)
          const infoY = ty + thumbSize;
          ctx.fillStyle = isSelected ? 'rgba(250, 204, 21, 0.2)' : 'rgba(15, 23, 42, 0.95)';
          ctx.fillRect(tx, infoY, thumbSize, style.infoH);
          ctx.strokeStyle = isSelected ? '#facc15' : (r === 0 ? '#facc15' : clColor);
          ctx.lineWidth = 1.0;
          ctx.strokeRect(tx, infoY, thumbSize, style.infoH);

          const distVal = (typeof item.dist === 'number' && !isNaN(item.dist) &&
                           isFinite(item.dist))
            ? item.dist
            : 0.0;
          const distStr = (thumbSize >= 90) ? distVal.toFixed(4) : distVal.toFixed(3);
          const clTag = (item.clusterId >= 0) ? `C${item.clusterId}` : '';

          // Line 1: Rank, Frame #, Cluster #
          const line1Text = clTag
            ? `#${item.rank} F#${item.frameIdx + 1} ${clTag}`
            : `#${item.rank} F#${item.frameIdx + 1}`;

          const line1Y = infoY + style.infoH * 0.31;
          const line2Y = infoY + style.infoH * 0.74;

          ctx.fillStyle = rankColor;
          ctx.font = style.fontBold;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(line1Text, tx + thumbSize / 2, line1Y);

          // Line 2: Distance value
          ctx.fillStyle = '#38bdf8';
          ctx.font = style.font;
          ctx.fillText(`d=${distStr}`, tx + thumbSize / 2, line2Y);
        }

        if (maxScroll > 0)
        {
          const scrollbarH = Math.max(16, (contentH / totalH) * contentH);
          const scrollbarY = contentY + (imageKnnScrollY / maxScroll) * (contentH - scrollbarH);
          ctx.fillStyle = 'rgba(192, 132, 252, 0.4)';
          ctx.fillRect(rect.x + rect.w - 6, scrollbarY, 4, scrollbarH);
        }

        ctx.restore();
      }
      else
      {
        drawEmptyMessage(
          ctx, rect, 'No k-NN results computed yet. Click "▶ Compute k-NN" to run solver.'
        );
      }
    }
  }

  // -------------------------------------------------------------
  // Q3: Bottom-Right - Full Set of Clusters (Current cluster highlighted)
  // -------------------------------------------------------------
  else if (qIdx === 3)
  {
    const kCount = clusters ? clusters.length : 0;
    const sortedIndices = getSortedClusterIndices();
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);

    let sortChipLabel = '🔢 ID';
    if (typeof imageClustersSortMode !== 'undefined')
    {
      if (imageClustersSortMode === 'size_desc') sortChipLabel = '📊 Size ↓';
      else if (imageClustersSortMode === 'size_asc') sortChipLabel = '📉 Size ↑';
    }

    const maxOffset = (typeof maximizedQuad === 'undefined' || maximizedQuad === null) ? 28 : 0;
    const sortChipW = 76;
    const sortChipH = 18;
    const sortChipX = rect.x + rect.w - sortChipW - 8 - maxOffset;
    const sortChipY = rect.y + 4;

    drawHeader(ctx, rect.x + 8, rect.y + 17, `📚 All Clusters (${kCount} anchors)`, '#38bdf8');
    drawToggleChip(ctx, sortChipX, sortChipY, sortChipW, sortChipH, sortChipLabel);

    if (kCount > 0)
    {
      const thumbSize = getImageThumbSize();
      const style = getThumbCardStyle(thumbSize);
      const cardW = thumbSize;
      const cardH = thumbSize + style.infoH;
      const gap = 8;
      const cols = Math.max(1, Math.floor(contentW / (cardW + gap)));
      const rows = Math.ceil(kCount / cols);
      const totalH = rows * (cardH + gap);

      const maxScroll = Math.max(0, totalH - contentH);
      imageClustersScrollY = Math.max(0, Math.min(maxScroll, imageClustersScrollY || 0));

      ctx.save();
      ctx.beginPath();
      ctx.rect(contentX, contentY, contentW, contentH);
      ctx.clip();

      for (let slot = 0; slot < kCount; slot++)
      {
        const i = sortedIndices[slot];
        const col = slot % cols;
        const row = Math.floor(slot / cols);
        const tx = contentX + col * (cardW + gap);
        const ty = contentY + row * (cardH + gap) - (imageClustersScrollY || 0);

        if (ty + cardH < contentY - 10 || ty > contentY + contentH + 10)
        {
          continue;
        }

        const cl = clusters[i];
        if (!cl || !cl.anchor) continue;

        const memCount = (cl && cl.members) || getClusterMembersList(i).length || 0;
        const isCurrentCluster = (i === assignedId);
        const thumbLabel = `C${i}`;
        const badgeColor = isCurrentCluster ? '#facc15' : (cl.color || '#cbd5e1');

        // Check if any of the active frame's k-NN nearest neighbors belong to cluster i
        const knnHitCount = knnList.filter(n => n.clusterId === i).length;

        // 1. Fully displayed image raster (unobstructed)
        ctx.fillStyle = '#020617';
        ctx.fillRect(tx, ty, thumbSize, thumbSize);
        ctx.strokeStyle = isCurrentCluster ? '#facc15' : (knnHitCount > 0 ? '#c084fc' : '#334155');
        ctx.lineWidth = isCurrentCluster ? 2.5 : (knnHitCount > 0 ? 2.0 : 1.0);
        ctx.strokeRect(tx, ty, thumbSize, thumbSize);

        drawRasterBuffer(
          ctx,
          cl.anchor,
          imageWidth,
          imageHeight,
          tx,
          ty,
          thumbSize,
          thumbSize,
          1.0,
          'q3_thumb_' + i
        );

        // 2. Info text box below the image
        const infoY = ty + thumbSize;
        ctx.fillStyle = isCurrentCluster
          ? 'rgba(250, 204, 21, 0.2)'
          : (knnHitCount > 0 ? 'rgba(192, 132, 252, 0.2)' : 'rgba(15, 23, 42, 0.95)');
        ctx.fillRect(tx, infoY, thumbSize, style.infoH);
        ctx.strokeStyle = isCurrentCluster ? '#facc15' : (knnHitCount > 0 ? '#c084fc' : '#334155');
        ctx.lineWidth = 1.0;
        ctx.strokeRect(tx, infoY, thumbSize, style.infoH);

        let label = `${thumbLabel} (${memCount})`;
        if (knnHitCount > 0) label += ` ★${knnHitCount}`;
        else if (isCurrentCluster) label = `${thumbLabel}★ (${memCount})`;

        ctx.fillStyle = badgeColor;
        ctx.font = isCurrentCluster ? style.fontBold : style.font;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          label,
          tx + thumbSize / 2,
          infoY + style.infoH / 2
        );
      }

      if (maxScroll > 0)
      {
        const scrollbarH = Math.max(16, (contentH / totalH) * contentH);
        const scrollbarY = contentY + (imageClustersScrollY / maxScroll) * (contentH - scrollbarH);
        ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
        ctx.fillRect(rect.x + rect.w - 6, scrollbarY, 4, scrollbarH);
      }

      ctx.restore();
    }
    else
    {
      drawEmptyMessage(ctx, rect, 'No clusters created yet');
    }
  }

  // Viewport Stats Chip Box (top-right of sub-viewport)
  let imgPtsCount = 0;
  let imgClustCount = 0;
  if (qIdx === 0)
  {
    imgPtsCount = frameBuf ? 1 : 0;
    imgClustCount = 0;
  }
  else if (qIdx === 1)
  {
    imgPtsCount = 0;
    imgClustCount = assignedCluster ? 1 : 0;
  }
  else if (qIdx === 2)
  {
    const members = (assignedId >= 0) ? getClusterMembersList(assignedId) : [];
    imgPtsCount = members.length;
    imgClustCount = assignedCluster ? 1 : 0;
  }
  else if (qIdx === 3)
  {
    imgPtsCount = 0;
    imgClustCount = clusters ? clusters.length : 0;
  }

  // Maximize Icon in Header for 4-panel view
  if (typeof maximizedQuad === 'undefined' || maximizedQuad === null)
  {
    const maxBtnX = rect.x + rect.w - 24;
    const maxBtnY = rect.y + 4;
    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(maxBtnX, maxBtnY, 18, 18, 3);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#38bdf8';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('⛶', maxBtnX + 9, maxBtnY + 9);
    ctx.restore();
  }

  // For Q1 and Q3, toggle chip buttons occupy header right; skip stats box for clarity
  if (qIdx !== 1 && qIdx !== 3)
  {
    const labelPts = `${imgPtsCount} pts`;
    const labelClust = `${imgClustCount} cl`;
    const fullText = `${labelPts}  •  ${labelClust}`;

    ctx.save();
    ctx.font = 'bold 9.5px monospace';
    const textW = ctx.measureText(fullText).width;
    const boxW = textW + 14;
    const boxH = 18;
    const maxW = (typeof maximizedQuad === 'undefined' || maximizedQuad === null) ? 28 : 0;
    const boxX = rect.x + rect.w - boxW - 8 - maxW;
    const boxY = rect.y + 4;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 4);
    ctx.fill();
    ctx.stroke();

    const midBoxY = boxY + boxH / 2;
    let curX = boxX + 7;

    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(labelPts, curX, midBoxY);
    curX += ctx.measureText(labelPts).width;

    ctx.fillStyle = 'rgba(100, 116, 139, 0.7)';
    ctx.fillText('  •  ', curX, midBoxY);
    curX += ctx.measureText('  •  ').width;

    ctx.fillStyle = '#38bdf8';
    ctx.fillText(labelClust, curX, midBoxY);
    ctx.restore();
  }

  ctx.restore();
}

/**
 * Draw prominent floating restore button when in single panel mode.
 */
function drawRestoreBanner(ctx, W, H, qIdx)
{
  ctx.save();
  const qNames = ['Current Frame', 'Anchor / Residual', 'Cluster Members', 'All Clusters'];
  const name = qNames[qIdx] || `Quadrant ${qIdx}`;
  const label = '⊞ Show All Panels (Esc)';

  const btnW = 164;
  const btnH = 22;
  const btnX = W - btnW - 8;
  const btnY = 2;

  ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(btnX, btnY, btnW, btnH, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#38bdf8';
  ctx.font = 'bold 10px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, btnX + btnW / 2, btnY + btnH / 2);

  // Left chip showing current panel name
  const nameLabel = `🔍 View: Q${qIdx} ${name}`;
  ctx.font = '9.5px sans-serif';
  const nameW = ctx.measureText(nameLabel).width + 12;
  const nameX = btnX - nameW - 6;
  if (nameX > 160)
  {
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(nameX, btnY, nameW, btnH, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.fillText(nameLabel, nameX + nameW / 2, btnY + btnH / 2);
  }

  ctx.restore();
}

function drawToggleChip(ctx, x, y, w, h, text)
{
  ctx.save();
  ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#38bdf8';
  ctx.font = 'bold 10px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + w / 2, y + h / 2);
  ctx.restore();
}

/**
 * Handle mouse click inside an image sub-viewport.
 * @param {number} px - Canvas X coordinate
 * @param {number} py - Canvas Y coordinate
 * @param {number} qIdx - Quadrant index (0..3)
 * @param {number} W - Canvas width
 * @param {number} H - Canvas height
 * @returns {boolean} True if click was handled
 */
function handleImageModeClick(px, py, qIdx, W, H)
{
  // 1. Check if clicking floating Restore Banner when in single panel mode
  if (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null)
  {
    if (py <= 28 && px >= W - 180)
    {
      maximizedQuad = null;
      if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
      if (typeof draw === 'function') draw();
      if (typeof showToast === 'function') showToast('⊞ Restored All 4 View Panels');
      return true;
    }
  }

  const rect = getImageQuadRect(qIdx, W, H);

  // 2. Check if clicking Maximize Icon in quadrant header (4-panel mode)
  if (typeof maximizedQuad === 'undefined' || maximizedQuad === null)
  {
    if (py >= rect.y && py <= rect.y + 26 &&
        px >= rect.x + rect.w - 32 && px <= rect.x + rect.w - 4)
    {
      maximizedQuad = qIdx;
      if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
      if (typeof draw === 'function') draw();
      const qNames = ['Current Frame', 'Anchor / Residual', 'Cluster Members', 'All Clusters'];
      if (typeof showToast === 'function')
      {
        showToast(`🔍 Maximized Q${qIdx}: ${qNames[qIdx]} (Click 'All 4 Panels' or Esc to restore)`);
      }
      return true;
    }
  }

  const isRetro = (typeof inspectedImageFrameIdx !== 'undefined' &&
    inspectedImageFrameIdx >= 0 &&
    benchmarkDataset && benchmarkDataset[inspectedImageFrameIdx]);
  const activeFrameIdx = isRetro
    ? inspectedImageFrameIdx
    : (totalFrames > 0
      ? totalFrames - 1
      : (benchmarkDataset && benchmarkDataset.length > 0 ? 0 : -1));

  // Q1: Click in Top-Right quadrant cycles between Anchor, Residual, NN1, NN1 Diff
  if (qIdx === 1)
  {
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    const hasKnn = (knnList.length > 0);

    let cur = (typeof imageTopRightMode !== 'undefined') ? imageTopRightMode : 'anchor';
    if (cur === 'anchor') imageTopRightMode = 'residual';
    else if (cur === 'residual') imageTopRightMode = hasKnn ? 'nn1' : 'anchor';
    else if (cur === 'nn1') imageTopRightMode = 'nn1_diff';
    else imageTopRightMode = 'anchor';

    if (typeof draw === 'function') draw();
    return true;
  }

  // Q2: Bottom-Left - Members of Current Cluster OR k-NN Nearest Neighbors
  if (qIdx === 2)
  {
    const rect = getImageQuadRect(2, W, H);
    const pad = 12;
    const headerH = 26;
    const contentX = rect.x + pad;
    const contentY = rect.y + headerH + 6;
    const contentW = rect.w - pad * 2;
    const contentH = rect.h - headerH - pad - 6;
    const maxOffset = (typeof maximizedQuad === 'undefined' || maximizedQuad === null) ? 28 : 0;

    // Check if clicking Q2 header toggle chip: ⇄ k-NN / ⇄ Members
    if (py >= rect.y && py <= rect.y + 26 &&
        px >= rect.x + rect.w - 130 - maxOffset && px <= rect.x + rect.w - maxOffset)
    {
      imageQ2ViewMode = (imageQ2ViewMode === 'knn') ? 'members' : 'knn';
      if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
      if (typeof draw === 'function') draw();
      if (typeof showToast === 'function')
      {
        showToast(imageQ2ViewMode === 'knn'
          ? '⚡ Showing k-NN Nearest Neighbors in Q2'
          : '👥 Showing Cluster Members in Q2');
      }
      return true;
    }

    const isKnnView = (typeof imageQ2ViewMode !== 'undefined' && imageQ2ViewMode === 'knn');

    if (isKnnView)
    {
      const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
      if (knnList.length === 0) return false;

      const thumbSize = getImageThumbSize();
      const style = getThumbCardStyle(thumbSize, true);
      const cardW = thumbSize;
      const cardH = thumbSize + style.infoH;
      const gap = 8;
      const cols = Math.max(1, Math.floor(contentW / (cardW + gap)));

      const relX = px - contentX;
      const relY = py - contentY + (imageKnnScrollY || 0);

      const col = Math.floor(relX / (cardW + gap));
      const row = Math.floor(relY / (cardH + gap));

      if (col < 0 || col >= cols) return false;

      const inThumbX = relX - col * (cardW + gap);
      const inThumbY = relY - row * (cardH + gap);
      if (inThumbX > cardW || inThumbY > cardH) return false;

      const clickedIdx = row * cols + col;
      if (clickedIdx >= 0 && clickedIdx < knnList.length)
      {
        const neighbor = knnList[clickedIdx];
        if (typeof selectImageFrame === 'function')
        {
          selectImageFrame(neighbor.frameIdx);
        }
        if (typeof showToast === 'function')
        {
          const distStr = (typeof neighbor.dist === 'number' && !isNaN(neighbor.dist))
            ? neighbor.dist.toFixed(3)
            : '0.000';
          const fNum = neighbor.frameIdx + 1;
          showToast(`🔍 Selected k-NN #${neighbor.rank} (Frame #${fNum}, d=${distStr})`);
        }
        return true;
      }
      return false;
    }

    let assignedId = -1;
    if (typeof inspectedImageFrameIdx !== 'undefined' && inspectedImageFrameIdx >= 0)
    {
      const fMap = imageFrameAssignments;
      assignedId = (fMap && fMap[inspectedImageFrameIdx] !== undefined)
        ? fMap[inspectedImageFrameIdx] : -1;
    }
    else
    {
      assignedId = typeof prevAssignedCluster !== 'undefined' ? prevAssignedCluster : -1;
    }
    if (assignedId < 0 && clusters && clusters.length > 0)
    {
      assignedId = (typeof selectedClusterId !== 'undefined' && selectedClusterId >= 0)
        ? selectedClusterId : 0;
    }

    const members = (assignedId >= 0) ? getClusterMembersList(assignedId) : [];
    if (members.length === 0) return false;

    const thumbSize = getImageThumbSize();
    const style = getThumbCardStyle(thumbSize);
    const cardW = thumbSize;
    const cardH = thumbSize + style.infoH;
    const gap = 8;
    const cols = Math.max(1, Math.floor(contentW / (cardW + gap)));

    const relX = px - contentX;
    const relY = py - contentY + (imageMembersScrollY || 0);

    const col = Math.floor(relX / (cardW + gap));
    const row = Math.floor(relY / (cardH + gap));

    if (col < 0 || col >= cols) return false;

    const inThumbX = relX - col * (cardW + gap);
    const inThumbY = relY - row * (cardH + gap);
    if (inThumbX > cardW || inThumbY > cardH) return false;

    const clickedIdx = row * cols + col;
    if (clickedIdx >= 0 && clickedIdx < members.length)
    {
      const frameIdx = members[clickedIdx];
      if (typeof selectImageFrame === 'function')
      {
        selectImageFrame(frameIdx);
      }
      if (typeof showToast === 'function')
      {
        showToast(`🔍 Inspected Member Frame #${frameIdx + 1}`);
      }
      return true;
    }
    return false;
  }

  // Q3: Bottom-Right - Full Set of Clusters
  if (qIdx === 3)
  {
    const rect = getImageQuadRect(3, W, H);
    const pad = 12;
    const headerH = 26;
    const contentX = rect.x + pad;
    const contentY = rect.y + headerH + 6;
    const contentW = rect.w - pad * 2;
    const contentH = rect.h - headerH - pad - 6;

    // Check if clicked inside sort chip button at top-right of Q3
    const sortChipW = 76;
    const sortChipH = 18;
    const sortChipX = rect.x + rect.w - sortChipW - 8;
    const sortChipY = rect.y + 4;
    if (px >= sortChipX && px <= sortChipX + sortChipW &&
        py >= sortChipY && py <= sortChipY + sortChipH)
    {
      cycleImageClusterSortMode();
      return true;
    }

    const kCount = clusters ? clusters.length : 0;
    if (kCount === 0) return false;

    const sortedIndices = getSortedClusterIndices();
    const thumbSize = getImageThumbSize();
    const style = getThumbCardStyle(thumbSize);
    const cardW = thumbSize;
    const cardH = thumbSize + style.infoH;
    const gap = 8;
    const cols = Math.max(1, Math.floor(contentW / (cardW + gap)));

    const relX = px - contentX;
    const relY = py - contentY + (imageClustersScrollY || 0);

    const col = Math.floor(relX / (cardW + gap));
    const row = Math.floor(relY / (cardH + gap));

    if (col < 0 || col >= cols) return false;

    const inThumbX = relX - col * (cardW + gap);
    const inThumbY = relY - row * (cardH + gap);
    if (inThumbX > cardW || inThumbY > cardH) return false;

    const clickedSlot = row * cols + col;
    if (clickedSlot >= 0 && clickedSlot < sortedIndices.length)
    {
      const clIdx = sortedIndices[clickedSlot];
      const members = getClusterMembersList(clIdx);
      if (members && members.length > 0)
      {
        if (typeof selectImageFrame === 'function')
        {
          selectImageFrame(members[0]);
        }
      }
      else
      {
        selectedClusterId = clIdx;
        inspectedClusterId = clIdx;
        if (typeof updateUI === 'function') updateUI();
        if (typeof draw === 'function') draw();
      }
      return true;
    }
    return false;
  }

  return false;
}

function drawHeader(ctx, x, y, title, color)
{
  ctx.fillStyle = color;
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(title, x, y);
}

function drawBadge(ctx, x, y, text, color)
{
  ctx.fillStyle = color;
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(text, x, y);
}

function drawEmptyMessage(ctx, rect, msg)
{
  ctx.fillStyle = '#64748b';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(msg, rect.x + rect.w / 2, rect.y + rect.h / 2);
}
