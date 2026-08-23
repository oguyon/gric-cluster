/**
 * GRIC Simulator - image_renderer.js
 * 4-Quadrant Raster Image Viewport & Scrollable Centroid Gallery
 */

/* eslint-disable no-unused-vars */

// Shared reusable off-screen canvas for raster conversions
let _imgOffCanvas = null;
let _imgOffCtx = null;
let _imgOffData = null;

function _ensureOffscreenCanvas(w, h)
{
  if (!_imgOffCanvas || _imgOffCanvas.width !== w || _imgOffCanvas.height !== h)
  {
    _imgOffCanvas = document.createElement('canvas');
    _imgOffCanvas.width = w;
    _imgOffCanvas.height = h;
    _imgOffCtx = _imgOffCanvas.getContext('2d');
    _imgOffData = _imgOffCtx.createImageData(w, h);
  }
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
  maxVal = 1.0
)
{
  if (!pixels || pixels.length < imgW * imgH) return;

  _ensureOffscreenCanvas(imgW, imgH);
  const data = _imgOffData.data;
  const numPix = imgW * imgH;
  const scale = maxVal > 0 ? 255.0 / maxVal : 255.0;

  for (let i = 0; i < numPix; i++)
  {
    const val = pixels[i];
    const lum = Math.max(0, Math.min(255, Math.round(val * scale)));
    const idx = i * 4;
    data[idx] = lum;     // R
    data[idx + 1] = lum; // G
    data[idx + 2] = lum; // B
    data[idx + 3] = 255; // A
  }

  _imgOffCtx.putImageData(_imgOffData, 0, 0);

  targetCtx.imageSmoothingEnabled = false;
  targetCtx.drawImage(
    _imgOffCanvas,
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

  // Determine active cluster and frame
  const assignedId = typeof prevAssignedCluster !== 'undefined' ? prevAssignedCluster : -1;
  const assignedCluster = (assignedId >= 0 && clusters[assignedId]) ? clusters[assignedId] : null;
  const frameBuf = currentImageFrame;

  // Q0: Current Query Frame
  if (qIdx === 0)
  {
    drawHeader(ctx, rect.x + 8, rect.y + 17, '📸 Active Query Frame', '#38bdf8');

    if (frameBuf)
    {
      const size = Math.max(32, Math.min(contentW, contentH) - 16);
      const imgX = contentX + (contentW - size) / 2;
      const imgY = contentY + (contentH - size) / 2;

      // Outer bounding frame
      ctx.fillStyle = '#020617';
      ctx.fillRect(imgX - 2, imgY - 2, size + 4, size + 4);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(imgX - 2, imgY - 2, size + 4, size + 4);

      drawRasterBuffer(ctx, frameBuf, imageWidth, imageHeight, imgX, imgY, size, size, 1.0);

      // Info badge
      drawBadge(
        ctx,
        rect.x + 8,
        rect.y + rect.h - 8,
        `Frame #${totalFrames} | ${imageWidth}×${imageHeight} (D=${imageDim})`,
        '#94a3b8'
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
        1.0
      );

      const lastDist = typeof distSampleClusterLast !== 'undefined'
        ? distSampleClusterLast
        : 0;
      drawBadge(
        ctx,
        rect.x + 8,
        rect.y + rect.h - 8,
        `Members: ${assignedCluster.members} | d(f,c): ${lastDist.toFixed(3)} | rlim: ${rlim.toFixed(3)}`,
        '#94a3b8'
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
        Math.max(0.5, maxDiff)
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

  // Q3: Scrollable Centroid Gallery
  else if (qIdx === 3)
  {
    const kCount = clusters ? clusters.length : 0;
    drawHeader(ctx, rect.x + 8, rect.y + 17, `📚 Centroid Gallery (${kCount})`, '#a78bfa');

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

      for (let k = 0; k < kCount; k++)
      {
        const cl = clusters[k];
        if (!cl || !cl.anchor) continue;

        const col = k % cols;
        const row = Math.floor(k / cols);
        const tx = contentX + col * (thumbSize + gap);
        const ty = contentY + row * (thumbSize + gap) - imageGalleryScrollY;

        if (ty + thumbSize < contentY - 10 || ty > contentY + contentH + 10)
        {
          continue; // Cull off-screen thumbnails
        }

        const isAssigned = (k === assignedId);

        // Thumbnail background & border
        ctx.fillStyle = '#020617';
        ctx.fillRect(tx, ty, thumbSize, thumbSize);
        ctx.strokeStyle = isAssigned ? '#22c55e' : (cl.color || '#334155');
        ctx.lineWidth = isAssigned ? 2.5 : 1.0;
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
          1.0
        );

        // Thumbnail badge
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(tx, ty + thumbSize - 12, thumbSize, 12);
        ctx.fillStyle = isAssigned ? '#4ade80' : '#cbd5e1';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`C${k}`, tx + thumbSize / 2, ty + thumbSize - 3);
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
      drawEmptyMessage(ctx, rect, 'No clusters created yet');
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
    imgPtsCount = 0;
    imgClustCount = clusters ? clusters.length : 0;
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
