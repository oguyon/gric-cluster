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
    case 2: return { x: 0, y: halfH, w: halfW, h: halfH };    // Bottom-Left: Current Cluster Members
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
  // Q1: Top-Right - Cluster Anchor (Click to toggle Residual)
  // -------------------------------------------------------------
  else if (qIdx === 1)
  {
    const isResidualView = (typeof imageTopRightMode !== 'undefined' &&
      imageTopRightMode === 'residual');

    if (!isResidualView)
    {
      // --- ANCHOR VIEW ---
      const title = assignedCluster
        ? `🎯 Cluster Anchor (C${assignedCluster.id})`
        : '🎯 Cluster Anchor';
      const titleColor = assignedCluster ? (assignedCluster.color || '#4ade80') : '#94a3b8';
      drawHeader(ctx, rect.x + 8, rect.y + 17, title, titleColor);

      // Toggle Chip Button in Header
      drawToggleChip(ctx, rect.x + rect.w - 110, rect.y + 4, 102, 18, '⇄ Residual');

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
        const matchStatus = isMatch ? 'MATCH' : 'NEW ANCHOR';
        drawBadge(
          ctx,
          rect.x + 8,
          rect.y + rect.h - 8,
          `Members: ${assignedCluster.members} | d(f,c): ${lastDist.toFixed(3)} | Click to toggle Residual ⇄`,
          isMatch ? '#4ade80' : '#facc15'
        );
      }
      else
      {
        drawEmptyMessage(ctx, rect, 'No anchor matched yet');
      }
    }
    else
    {
      // --- RESIDUAL VIEW ---
      const title = assignedCluster
        ? `⚡ Residual |f - C${assignedCluster.id}|`
        : '⚡ Residual |f - C_anchor|';
      drawHeader(ctx, rect.x + 8, rect.y + 17, title, '#f87171');

      // Toggle Chip Button in Header
      drawToggleChip(ctx, rect.x + rect.w - 96, rect.y + 4, 88, 18, '⇄ Anchor');

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
          `L2: ${l2.toFixed(3)} | RMS: ${rms.toFixed(4)} | Max: ${maxDiff.toFixed(3)} | Click to toggle Anchor ⇄`,
          '#94a3b8'
        );
      }
      else
      {
        drawEmptyMessage(ctx, rect, 'Awaiting frame and anchor evaluation');
      }
    }
  }

  // -------------------------------------------------------------
  // Q2: Bottom-Left - Members of Current Cluster (Current frame highlighted)
  // -------------------------------------------------------------
  else if (qIdx === 2)
  {
    const members = (assignedId >= 0) ? getClusterMembersList(assignedId) : [];
    const mCount = members.length;
    const clColor = assignedCluster ? (assignedCluster.color || '#a78bfa') : '#94a3b8';

    const title = assignedCluster
      ? `👥 C${assignedId} Members (${mCount} frames)`
      : '👥 Current Cluster Members';
    drawHeader(ctx, rect.x + 8, rect.y + 17, title, clColor);

    if (mCount > 0)
    {
      const thumbSize = 44;
      const gap = 8;
      const cols = Math.max(1, Math.floor(contentW / (thumbSize + gap)));
      const rows = Math.ceil(mCount / cols);
      const totalH = rows * (thumbSize + gap);

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
        const tx = contentX + col * (thumbSize + gap);
        const ty = contentY + row * (thumbSize + gap) - (imageMembersScrollY || 0);

        if (ty + thumbSize < contentY - 10 || ty > contentY + contentH + 10)
        {
          continue;
        }

        const memberFrameIdx = members[i];
        const rasterData = benchmarkDataset ? benchmarkDataset[memberFrameIdx] : null;
        const isCurrent = (memberFrameIdx === activeFrameIdx);
        const thumbLabel = `#${memberFrameIdx + 1}`;
        const badgeColor = isCurrent ? '#facc15' : '#cbd5e1';

        if (!rasterData) continue;

        // Background & border (Highlighted if current frame)
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

        // Thumbnail badge
        ctx.fillStyle = isCurrent ? 'rgba(250, 204, 21, 0.25)' : 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(tx, ty + thumbSize - 12, thumbSize, 12);
        ctx.fillStyle = badgeColor;
        ctx.font = isCurrent ? 'bold 9px monospace' : '9px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(
          isCurrent ? `${thumbLabel} ★` : thumbLabel,
          tx + thumbSize / 2,
          ty + thumbSize - 3
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

  // -------------------------------------------------------------
  // Q3: Bottom-Right - Full Set of Clusters (Current cluster highlighted)
  // -------------------------------------------------------------
  else if (qIdx === 3)
  {
    const kCount = clusters ? clusters.length : 0;
    const sortedIndices = getSortedClusterIndices();

    let sortChipLabel = '🔢 ID';
    if (typeof imageClustersSortMode !== 'undefined')
    {
      if (imageClustersSortMode === 'size_desc') sortChipLabel = '📊 Size ↓';
      else if (imageClustersSortMode === 'size_asc') sortChipLabel = '📉 Size ↑';
    }

    const sortChipW = 76;
    const sortChipH = 18;
    const sortChipX = rect.x + rect.w - sortChipW - 8;
    const sortChipY = rect.y + 4;

    drawHeader(ctx, rect.x + 8, rect.y + 17, `📚 All Clusters (${kCount} anchors)`, '#38bdf8');
    drawToggleChip(ctx, sortChipX, sortChipY, sortChipW, sortChipH, sortChipLabel);

    if (kCount > 0)
    {
      const thumbSize = 44;
      const gap = 8;
      const cols = Math.max(1, Math.floor(contentW / (thumbSize + gap)));
      const rows = Math.ceil(kCount / cols);
      const totalH = rows * (thumbSize + gap);

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
        const tx = contentX + col * (thumbSize + gap);
        const ty = contentY + row * (thumbSize + gap) - (imageClustersScrollY || 0);

        if (ty + thumbSize < contentY - 10 || ty > contentY + contentH + 10)
        {
          continue;
        }

        const cl = clusters[i];
        if (!cl || !cl.anchor) continue;

        const memCount = (cl && cl.members) || getClusterMembersList(i).length || 0;
        const isCurrentCluster = (i === assignedId);
        const thumbLabel = `C${i}`;
        const badgeColor = isCurrentCluster ? '#facc15' : (cl.color || '#cbd5e1');

        ctx.fillStyle = '#020617';
        ctx.fillRect(tx, ty, thumbSize, thumbSize);
        ctx.strokeStyle = isCurrentCluster ? '#facc15' : '#334155';
        ctx.lineWidth = isCurrentCluster ? 2.5 : 1.0;
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

        // Thumbnail badge with cluster index and member count
        ctx.fillStyle = isCurrentCluster ? 'rgba(250, 204, 21, 0.25)' : 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(tx, ty + thumbSize - 12, thumbSize, 12);
        ctx.fillStyle = badgeColor;
        ctx.font = isCurrentCluster ? 'bold 8.5px monospace' : '8.5px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(
          isCurrentCluster ? `${thumbLabel}★ (${memCount})` : `${thumbLabel} (${memCount})`,
          tx + thumbSize / 2,
          ty + thumbSize - 3
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
    const boxX = rect.x + rect.w - boxW - 8;
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
  // Q1: Click in Top-Right quadrant toggles between Anchor and Residual
  if (qIdx === 1)
  {
    imageTopRightMode = (imageTopRightMode === 'anchor') ? 'residual' : 'anchor';
    if (typeof draw === 'function') draw();
    return true;
  }

  // Q2: Bottom-Left - Members of Current Cluster
  if (qIdx === 2)
  {
    const rect = getImageQuadRect(2, W, H);
    const pad = 12;
    const headerH = 26;
    const contentX = rect.x + pad;
    const contentY = rect.y + headerH + 6;
    const contentW = rect.w - pad * 2;
    const contentH = rect.h - headerH - pad - 6;

    let assignedId = -1;
    if (typeof inspectedImageFrameIdx !== 'undefined' && inspectedImageFrameIdx >= 0)
    {
      assignedId = (imageFrameAssignments && imageFrameAssignments[inspectedImageFrameIdx] !== undefined)
        ? imageFrameAssignments[inspectedImageFrameIdx] : -1;
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

    const thumbSize = 44;
    const gap = 8;
    const cols = Math.max(1, Math.floor(contentW / (thumbSize + gap)));

    const relX = px - contentX;
    const relY = py - contentY + (imageMembersScrollY || 0);

    const col = Math.floor(relX / (thumbSize + gap));
    const row = Math.floor(relY / (thumbSize + gap));

    if (col < 0 || col >= cols) return false;

    const inThumbX = relX - col * (thumbSize + gap);
    const inThumbY = relY - row * (thumbSize + gap);
    if (inThumbX > thumbSize || inThumbY > thumbSize) return false;

    const clickedIdx = row * cols + col;
    if (clickedIdx >= 0 && clickedIdx < members.length)
    {
      const frameIdx = members[clickedIdx];
      if (typeof selectImageFrame === 'function')
      {
        selectImageFrame(frameIdx);
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
    const thumbSize = 44;
    const gap = 8;
    const cols = Math.max(1, Math.floor(contentW / (thumbSize + gap)));

    const relX = px - contentX;
    const relY = py - contentY + (imageClustersScrollY || 0);

    const col = Math.floor(relX / (thumbSize + gap));
    const row = Math.floor(relY / (thumbSize + gap));

    if (col < 0 || col >= cols) return false;

    const inThumbX = relX - col * (thumbSize + gap);
    const inThumbY = relY - row * (thumbSize + gap);
    if (inThumbX > thumbSize || inThumbY > thumbSize) return false;

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
