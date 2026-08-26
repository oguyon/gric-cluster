/**
 * GRIC Simulator - image_renderer.js
 * 4-Quadrant Raster Image Viewport & Scrollable Centroid Gallery
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
    case 0: return { x: 0, y: 0, w: halfW, h: halfH };        // Top-Left: Query Frame
    case 1: return { x: halfW, y: 0, w: halfW, h: halfH };    // Top-Right: Assigned Anchor
    case 2: return { x: 0, y: halfH, w: halfW, h: halfH };    // Bottom-Left: Difference / Residual
    case 3: return { x: halfW, y: halfH, w: halfW, h: halfH };// Bottom-Right: Centroid Gallery
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

  // Determine if we are in retro-inspection mode or live stream mode
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

  const assignedCluster = (assignedId >= 0 && clusters[assignedId]) ? clusters[assignedId] : null;

  // Q0: Query Frame (Live or Inspected)
  if (qIdx === 0)
  {
    const title = isRetro
      ? `🔍 Inspected Frame #${activeFrameIdx + 1}`
      : '📸 Active Query Frame';
    drawHeader(ctx, rect.x + 8, rect.y + 17, title, isRetro ? '#facc15' : '#38bdf8');

    if (frameBuf)
    {
      const size = Math.max(32, Math.min(contentW, contentH) - 16);
      const imgX = contentX + (contentW - size) / 2;
      const imgY = contentY + (contentH - size) / 2;

      // Outer bounding frame
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

      // Info badge
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

  // Q1: Assigned Cluster Anchor
  else if (qIdx === 1)
  {
    const title = assignedCluster
      ? `🎯 Assigned Anchor (C${assignedCluster.id})`
      : '🎯 Assigned Anchor';
    const titleColor = assignedCluster ? (assignedCluster.color || '#4ade80') : '#94a3b8';
    drawHeader(ctx, rect.x + 8, rect.y + 17, title, titleColor);

    if (assignedCluster && assignedCluster.anchor)
    {
      const size = Math.max(32, Math.min(contentW, contentH) - 16);
      const imgX = contentX + (contentW - size) / 2;
      const imgY = contentY + (contentH - size) / 2;

      // Outer bounding frame
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
        'quad1'
      );

      const isMatch = lastDist <= (rlim || 0.1);
      const matchStatus = isMatch ? 'MATCH' : 'NEW ANCHOR';
      drawBadge(
        ctx,
        rect.x + 8,
        rect.y + rect.h - 8,
        `Members: ${assignedCluster.members} | d(f,c): ${lastDist.toFixed(3)} | ${matchStatus}`,
        isMatch ? '#4ade80' : '#facc15'
      );
    }
    else
    {
      drawEmptyMessage(ctx, rect, 'No anchor matched yet');
    }
  }

  // Q2: Difference / Residual Heatmap
  else if (qIdx === 2)
  {
    drawHeader(ctx, rect.x + 8, rect.y + 17, '⚡ Residual |f - C_assigned|', '#f87171');

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

      // Outer bounding frame
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
        'quad2'
      );

      drawBadge(
        ctx,
        rect.x + 8,
        rect.y + rect.h - 8,
        `L2 Norm: ${l2.toFixed(3)} | RMS: ${rms.toFixed(4)} | Max Diff: ${maxDiff.toFixed(3)}`,
        '#94a3b8'
      );
    }
    else
    {
      drawEmptyMessage(ctx, rect, 'Awaiting frame and anchor evaluation');
    }
  }

  // Q3: Scrollable Centroid Gallery OR Cluster Member Gallery
  else if (qIdx === 3)
  {
    const isClusterInspection = (typeof inspectedClusterId !== 'undefined' &&
      inspectedClusterId >= 0 &&
      clusters[inspectedClusterId]);
    const members = isClusterInspection
      ? ((imageClusterMembers && imageClusterMembers[inspectedClusterId]) || [])
      : [];
    const kCount = isClusterInspection ? members.length : (clusters ? clusters.length : 0);

    if (isClusterInspection)
    {
      const c = clusters[inspectedClusterId];
      drawHeader(
        ctx,
        rect.x + 8,
        rect.y + 17,
        `👥 C${inspectedClusterId} Members (${kCount} frames)`,
        c.color || '#a78bfa'
      );

      // Back chip button in header
      const btnBackW = 74;
      const btnBackH = 18;
      const btnBackX = rect.x + rect.w - btnBackW - 8;
      const btnBackY = rect.y + 4;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(btnBackX, btnBackY, btnBackW, btnBackH, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#cbd5e1';
      ctx.font = 'bold 9.5px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('← All Clusters', btnBackX + btnBackW / 2, btnBackY + btnBackH / 2);
    }
    else
    {
      drawHeader(ctx, rect.x + 8, rect.y + 17, `📚 Centroid Gallery (${kCount})`, '#a78bfa');
    }

    if (kCount > 0)
    {
      const thumbSize = 44;
      const gap = 8;
      const cols = Math.max(1, Math.floor((contentW) / (thumbSize + gap)));
      const rows = Math.ceil(kCount / cols);
      const totalH = rows * (thumbSize + gap);

      // Clamp scroll
      const maxScroll = Math.max(0, totalH - contentH);
      imageGalleryScrollY = Math.max(0, Math.min(maxScroll, imageGalleryScrollY || 0));

      ctx.save();
      ctx.beginPath();
      ctx.rect(contentX, contentY, contentW, contentH);
      ctx.clip();

      for (let i = 0; i < kCount; i++)
      {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const tx = contentX + col * (thumbSize + gap);
        const ty = contentY + row * (thumbSize + gap) - imageGalleryScrollY;

        if (ty + thumbSize < contentY - 10 || ty > contentY + contentH + 10)
        {
          continue; // Cull off-screen thumbnails
        }

        let rasterData = null;
        let isSelected = false;
        let thumbLabel = '';
        let badgeColor = '#cbd5e1';

        if (isClusterInspection)
        {
          const memberFrameIdx = members[i];
          rasterData = benchmarkDataset ? benchmarkDataset[memberFrameIdx] : null;
          isSelected = (memberFrameIdx === activeFrameIdx);
          thumbLabel = `#${memberFrameIdx + 1}`;
          badgeColor = isSelected ? '#facc15' : '#cbd5e1';
        }
        else
        {
          const cl = clusters[i];
          if (!cl || !cl.anchor) continue;
          rasterData = cl.anchor;
          isSelected = (i === assignedId);
          thumbLabel = `C${i}`;
          badgeColor = isSelected ? '#4ade80' : '#cbd5e1';
        }

        if (!rasterData) continue;

        // Thumbnail background & border
        ctx.fillStyle = '#020617';
        ctx.fillRect(tx, ty, thumbSize, thumbSize);
        ctx.strokeStyle = isSelected ? '#facc15' : '#334155';
        ctx.lineWidth = isSelected ? 2.5 : 1.0;
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
          'thumb_' + i
        );

        // Thumbnail badge
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(tx, ty + thumbSize - 12, thumbSize, 12);
        ctx.fillStyle = badgeColor;
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(thumbLabel, tx + thumbSize / 2, ty + thumbSize - 3);
      }

      // Scrollbar indicator if scrollable
      if (maxScroll > 0)
      {
        const scrollbarH = Math.max(16, (contentH / totalH) * contentH);
        const scrollbarY = contentY + (imageGalleryScrollY / maxScroll) * (contentH - scrollbarH);
        ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
        ctx.fillRect(rect.x + rect.w - 6, scrollbarY, 4, scrollbarH);
      }

      ctx.restore();
    }
    else
    {
      drawEmptyMessage(
        ctx,
        rect,
        isClusterInspection ? 'No member frames in this cluster' : 'No clusters created yet'
      );
    }
  }

  // Viewport Stats Box (Samples & Clusters currently displayed in this view)
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
    imgPtsCount = frameBuf ? 1 : 0;
    imgClustCount = assignedCluster ? 1 : 0;
  }
  else if (qIdx === 3)
  {
    const isClusterInspection = (typeof inspectedClusterId !== 'undefined' &&
      inspectedClusterId >= 0 &&
      clusters[inspectedClusterId]);
    if (isClusterInspection)
    {
      imgPtsCount = (imageClusterMembers && imageClusterMembers[inspectedClusterId])
        ? imageClusterMembers[inspectedClusterId].length
        : 0;
      imgClustCount = 1;
    }
    else
    {
      imgPtsCount = 0;
      imgClustCount = clusters ? clusters.length : 0;
    }
  }

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
  if (qIdx !== 3 && (typeof maximizedQuad === 'undefined' || maximizedQuad !== 3))
  {
    return false;
  }

  const rect = getImageQuadRect(3, W, H);
  const pad = 12;
  const headerH = 26;
  const contentX = rect.x + pad;
  const contentY = rect.y + headerH + 6;
  const contentW = rect.w - pad * 2;
  const contentH = rect.h - headerH - pad - 6;

  const isClusterInspection = (typeof inspectedClusterId !== 'undefined' &&
    inspectedClusterId >= 0 &&
    clusters[inspectedClusterId]);

  // Check if click was on the "← All Clusters" back button
  if (isClusterInspection)
  {
    const btnBackW = 74;
    const btnBackH = 18;
    const btnBackX = rect.x + rect.w - btnBackW - 8;
    const btnBackY = rect.y + 4;
    if (px >= btnBackX && px <= btnBackX + btnBackW &&
        py >= btnBackY && py <= btnBackY + btnBackH)
    {
      if (typeof clearImageClusterInspection === 'function')
      {
        clearImageClusterInspection();
      }
      return true;
    }
  }

  // Check if click is inside the thumbnail grid area
  if (px < contentX || px > contentX + contentW || py < contentY || py > contentY + contentH)
  {
    return false;
  }

  const members = isClusterInspection
    ? ((imageClusterMembers && imageClusterMembers[inspectedClusterId]) || [])
    : [];
  const kCount = isClusterInspection ? members.length : (clusters ? clusters.length : 0);
  if (kCount === 0) return false;

  const thumbSize = 44;
  const gap = 8;
  const cols = Math.max(1, Math.floor(contentW / (thumbSize + gap)));

  const relX = px - contentX;
  const relY = py - contentY + (imageGalleryScrollY || 0);

  const col = Math.floor(relX / (thumbSize + gap));
  const row = Math.floor(relY / (thumbSize + gap));

  if (col < 0 || col >= cols) return false;

  const inThumbX = relX - col * (thumbSize + gap);
  const inThumbY = relY - row * (thumbSize + gap);
  if (inThumbX > thumbSize || inThumbY > thumbSize) return false;

  const clickedIdx = row * cols + col;
  if (clickedIdx >= 0 && clickedIdx < kCount)
  {
    if (isClusterInspection)
    {
      const frameIdx = members[clickedIdx];
      if (typeof selectImageFrame === 'function')
      {
        selectImageFrame(frameIdx);
      }
    }
    else
    {
      if (typeof inspectClusterMembers === 'function')
      {
        inspectClusterMembers(clickedIdx);
      }
    }
    return true;
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
