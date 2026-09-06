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
window.getClusterMembersList = getClusterMembersList;

/**
 * Auto-scroll Q2 Cluster Members gallery so member card is visible.
 */
function scrollImageMemberIntoView(memberIdx)
{
  if (memberIdx < 0 || typeof canvas === 'undefined' || !canvas) return;
  const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
  const rect = (canvas.getBoundingClientRect) ? canvas.getBoundingClientRect() : null;
  const cssW = (rect && rect.width) ? rect.width : (canvas.width / dpr);
  const cssH = (rect && rect.height) ? rect.height : (canvas.height / dpr);
  const isMax = (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null);
  const quadW = isMax ? cssW : (cssW / 2);
  const quadH = isMax ? cssH : (cssH / 2);
  const pad = 12;
  const headerH = 26;
  const contentW = Math.max(100, quadW - pad * 2);
  const contentH = Math.max(100, quadH - headerH - pad - 6);

  const thumbSize = getImageThumbSize();
  const style = getThumbCardStyle(thumbSize);
  const cardH = thumbSize + style.infoH;
  const gap = 8;
  const availableW = contentW - 16;
  const cols = Math.max(1, Math.floor(availableW / (thumbSize + gap)));
  const row = Math.floor(memberIdx / cols);
  const itemTop = row * (cardH + gap);
  const itemBottom = itemTop + cardH;

  if (itemTop < imageMembersScrollY)
  {
    imageMembersScrollY = itemTop;
  }
  else if (itemBottom > imageMembersScrollY + contentH)
  {
    imageMembersScrollY = itemBottom - contentH;
  }
}
window.scrollImageMemberIntoView = scrollImageMemberIntoView;

/**
 * Auto-scroll Q3 All Clusters gallery so cluster card is visible.
 */
function scrollImageClusterIntoView(clusterId)
{
  if (clusterId < 0 || typeof canvas === 'undefined' || !canvas) return;
  const sortedIndices = (typeof getSortedClusterIndices === 'function')
    ? getSortedClusterIndices() : [];
  const slot = sortedIndices.indexOf(clusterId);
  if (slot < 0) return;

  const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
  const rect = (canvas.getBoundingClientRect) ? canvas.getBoundingClientRect() : null;
  const cssW = (rect && rect.width) ? rect.width : (canvas.width / dpr);
  const cssH = (rect && rect.height) ? rect.height : (canvas.height / dpr);
  const isMax = (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null);
  const quadW = isMax ? cssW : (cssW / 2);
  const quadH = isMax ? cssH : (cssH / 2);
  const pad = 12;
  const headerH = 26;
  const contentW = Math.max(100, quadW - pad * 2);
  const contentH = Math.max(100, quadH - headerH - pad - 6);

  const thumbSize = getImageThumbSize();
  const style = getThumbCardStyle(thumbSize);
  const cardH = thumbSize + style.infoH;
  const gap = 8;
  const availableW = contentW - 16;
  const cols = Math.max(1, Math.floor(availableW / (thumbSize + gap)));
  const row = Math.floor(slot / cols);
  const itemTop = row * (cardH + gap);
  const itemBottom = itemTop + cardH;

  if (itemTop < imageClustersScrollY)
  {
    imageClustersScrollY = itemTop;
  }
  else if (itemBottom > imageClustersScrollY + contentH)
  {
    imageClustersScrollY = itemBottom - contentH;
  }
}
window.scrollImageClusterIntoView = scrollImageClusterIntoView;

let isPanelSliderDragging = false;
let activeSliderQuad = -1;
let activeSliderViewMode = null;
let activeSliderDragOffset = 0;
let activeSliderTrackY = 0;
let activeSliderTrackH = 0;
let activeSliderThumbH = 0;
let activeSliderMaxScroll = 0;
let hoveredSliderQuad = -1;

/**
 * Calculate vertical slider geometry and bounds for view panels with thumbnails.
 * @param {number} qIdx - Quadrant index (0..3)
 * @param {number} W - Canvas width in CSS pixels
 * @param {number} H - Canvas height in CSS pixels
 * @param {object} [optRect] - Optional pre-computed quadrant rect
 * @param {string} [optViewMode] - Optional pre-computed view mode
 * @returns {object|null} Slider metrics or null if no slider needed
 */
function getPanelSliderRect(qIdx, W, H, optRect, optViewMode)
{
  if (qIdx < 0 || qIdx > 3) return null;
  const isMax = (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null);
  const effectiveQuad = isMax ? maximizedQuad : qIdx;
  const rect = optRect ||
    (isMax ? { x: 0, y: 0, w: W, h: H } : getImageQuadRect(effectiveQuad, W, H));
  const viewMode = optViewMode || ((typeof getImagePanelViewMode === 'function')
    ? getImagePanelViewMode(effectiveQuad) : 'current_frame');

  const isReconImg = (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView &&
    typeof isReconstructionImageMode === 'function' && isReconstructionImageMode());

  if (isReconImg && (effectiveQuad === 0 || effectiveQuad === 1 ||
      optViewMode === 'recon_knn_a' || optViewMode === 'recon_knn_b'))
  {
    let queryIdx = 0;
    if (typeof reconLockedQueryIdx !== 'undefined' && reconLockedQueryIdx >= 0)
    {
      queryIdx = reconLockedQueryIdx;
    }
    else if (typeof inspectedImageFrameIdx === 'number' && inspectedImageFrameIdx >= 0)
    {
      queryIdx = inspectedImageFrameIdx;
    }
    const neighbors = (typeof getReconstructionKnnNeighbors === 'function')
      ? getReconstructionKnnNeighbors(queryIdx) : null;
    const itemCount = neighbors ? neighbors.length : 0;
    if (itemCount <= 0) return null;

    const pad = 8;
    const headerH = 28;
    const contentX = rect.x + pad;
    const contentY = rect.y + headerH + pad;
    const contentW = rect.w - pad * 2;
    const contentH = rect.h - headerH - pad * 2;

    const thumbSize = getImageThumbSize();
    const style = getThumbCardStyle(thumbSize, true);
    const cardW = thumbSize;
    const cardH = thumbSize + style.infoH;
    const gap = 8;
    const availableW = contentW - 16;
    const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));
    const rows = Math.ceil(itemCount / cols);
    const totalH = rows * (cardH + gap);
    const maxScroll = Math.max(0, totalH - contentH);
    if (maxScroll <= 0) return null;

    const trackW = 10;
    const trackX = rect.x + rect.w - trackW - 5;
    const trackY = contentY;
    const trackH = contentH;
    const thumbMinH = 26;
    const thumbH = Math.max(thumbMinH, Math.min(trackH - 4, (contentH / totalH) * trackH));
    const scrollY = (typeof imageReconKnnScrollY !== 'undefined') ? imageReconKnnScrollY : 0;

    return {
      effectiveQuad,
      viewMode: (effectiveQuad === 0) ? 'recon_knn_a' : 'recon_knn_b',
      rect,
      contentX,
      contentY,
      contentW,
      contentH,
      trackX,
      trackY,
      trackW,
      trackH,
      thumbH,
      maxScroll,
      totalH,
      scrollY
    };
  }

  if (viewMode !== 'members' && viewMode !== 'knn' && viewMode !== 'clusters')
  {
    return null;
  }

  const pad = 12;
  const headerH = 26;
  const contentX = rect.x + pad;
  const contentY = rect.y + headerH + 6;
  const contentW = rect.w - pad * 2;
  const contentH = rect.h - headerH - pad - 6;

  let itemCount = 0;
  if (viewMode === 'members')
  {
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
    if (assignedId < 0 && typeof inspectedClusterId !== 'undefined' && inspectedClusterId >= 0)
    {
      assignedId = inspectedClusterId;
    }
    else if (assignedId < 0 && clusters && clusters.length > 0)
    {
      assignedId = (typeof selectedClusterId !== 'undefined' && selectedClusterId >= 0)
        ? selectedClusterId : 0;
    }
    const memList = (assignedId >= 0 && typeof getClusterMembersList === 'function')
      ? getClusterMembersList(assignedId) : [];
    itemCount = memList ? memList.length : 0;
  }
  else if (viewMode === 'knn')
  {
    const isRetro = (typeof inspectedImageFrameIdx !== 'undefined' &&
      inspectedImageFrameIdx >= 0 &&
      benchmarkDataset && benchmarkDataset[inspectedImageFrameIdx]);
    const hasBench = (typeof benchmarkDataset !== 'undefined' &&
      benchmarkDataset && benchmarkDataset.length > 0);
    const hasTotal = (typeof totalFrames !== 'undefined' && totalFrames > 0);
    const activeFrameIdx = isRetro
      ? inspectedImageFrameIdx
      : (hasTotal ? totalFrames - 1 : (hasBench ? 0 : -1));
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    itemCount = knnList ? knnList.length : 0;
  }
  else if (viewMode === 'clusters')
  {
    itemCount = clusters ? clusters.length : 0;
  }

  if (itemCount <= 0) return null;

  const thumbSize = getImageThumbSize();
  const style = getThumbCardStyle(thumbSize, viewMode === 'knn');
  const cardW = thumbSize;
  const cardH = thumbSize + style.infoH;
  const gap = 8;
  const availableW = contentW - 16;
  const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));
  const rows = Math.ceil(itemCount / cols);
  const totalH = rows * (cardH + gap);
  const maxScroll = Math.max(0, totalH - contentH);

  if (maxScroll <= 0) return null;

  const trackW = 10;
  const trackX = rect.x + rect.w - trackW - 5;
  const trackY = contentY;
  const trackH = contentH;

  const thumbMinH = 26;
  const thumbH = Math.max(thumbMinH, Math.min(trackH - 4, (contentH / totalH) * trackH));

  const scrollY = (viewMode === 'members')
    ? (imageMembersScrollY || 0)
    : (viewMode === 'knn' ? (imageKnnScrollY || 0) : (imageClustersScrollY || 0));

  return {
    effectiveQuad,
    viewMode,
    rect,
    contentX,
    contentY,
    contentW,
    contentH,
    trackX,
    trackY,
    trackW,
    trackH,
    thumbH,
    maxScroll,
    totalH,
    scrollY
  };
}
window.getPanelSliderRect = getPanelSliderRect;

/**
 * Render a vertical navigation slider inside a view panel.
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {object} slider - Slider metrics from getPanelSliderRect
 */
function drawPanelSlider(ctx, slider)
{
  if (!slider || slider.maxScroll <= 0) return;

  const { trackX, trackY, trackW, trackH, thumbH, maxScroll, scrollY, effectiveQuad } = slider;

  ctx.save();

  // 1. Slider Track Background
  ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
  ctx.strokeStyle = 'rgba(51, 65, 85, 0.8)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(trackX, trackY, trackW, trackH, 5);
  ctx.fill();
  ctx.stroke();

  // 2. Subtle center groove line
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(trackX + trackW / 2, trackY + 5);
  ctx.lineTo(trackX + trackW / 2, trackY + trackH - 5);
  ctx.stroke();

  // 3. Slider Thumb Handle
  const scrollFraction = Math.max(0, Math.min(1, (scrollY || 0) / maxScroll));
  const thumbY = trackY + scrollFraction * (trackH - thumbH);
  const thumbW = trackW - 2;
  const thumbX = trackX + 1;

  const isDraggingThis = (isPanelSliderDragging && activeSliderQuad === effectiveQuad);
  const isHovered = (hoveredSliderQuad === effectiveQuad);

  if (isDraggingThis)
  {
    ctx.fillStyle = '#38bdf8';
    ctx.strokeStyle = '#bae6fd';
    ctx.lineWidth = 1.5;
  }
  else if (isHovered)
  {
    ctx.fillStyle = '#0284c7';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.2;
  }
  else
  {
    ctx.fillStyle = '#475569';
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
    ctx.lineWidth = 1;
  }

  ctx.beginPath();
  ctx.roundRect(thumbX, thumbY, thumbW, thumbH, 4);
  ctx.fill();
  ctx.stroke();

  // 4. Handle grip notches in the center of thumb
  ctx.strokeStyle = isDraggingThis ? '#0369a1' : (isHovered ? '#ffffff' : '#94a3b8');
  ctx.lineWidth = 1;
  const midY = thumbY + thumbH / 2;
  [-3, 0, 3].forEach(offset => {
    ctx.beginPath();
    ctx.moveTo(thumbX + 2, midY + offset);
    ctx.lineTo(thumbX + thumbW - 2, midY + offset);
    ctx.stroke();
  });

  ctx.restore();
}
window.drawPanelSlider = drawPanelSlider;

function setPanelScroll(viewMode, scrollY)
{
  if (viewMode === 'members')
  {
    imageMembersScrollY = scrollY;
  }
  else if (viewMode === 'knn')
  {
    imageKnnScrollY = scrollY;
  }
  else if (viewMode === 'clusters')
  {
    imageClustersScrollY = scrollY;
  }
  else if (viewMode === 'recon_knn_a' ||
           viewMode === 'recon_knn_b' ||
           viewMode === 'recon_knn')
  {
    imageReconKnnScrollY = scrollY;
  }
}
window.setPanelScroll = setPanelScroll;

function getPanelSliderState()
{
  return {
    isDragging: isPanelSliderDragging,
    activeQuad: activeSliderQuad,
    activeViewMode: activeSliderViewMode,
    dragOffset: activeSliderDragOffset,
    trackY: activeSliderTrackY,
    trackH: activeSliderTrackH,
    thumbH: activeSliderThumbH,
    maxScroll: activeSliderMaxScroll,
    hoveredQuad: hoveredSliderQuad
  };
}
window.getPanelSliderState = getPanelSliderState;

function startPanelSliderDrag(quad, viewMode, trackY, trackH, thumbH, maxScroll, dragOffset)
{
  isPanelSliderDragging = true;
  activeSliderQuad = quad;
  activeSliderViewMode = viewMode;
  activeSliderTrackY = trackY;
  activeSliderTrackH = trackH;
  activeSliderThumbH = thumbH;
  activeSliderMaxScroll = maxScroll;
  activeSliderDragOffset = (typeof dragOffset === 'number') ? dragOffset : 0;
}
window.startPanelSliderDrag = startPanelSliderDrag;

function updatePanelSliderDrag(clientRelY)
{
  if (!isPanelSliderDragging) return;
  const targetThumbY = clientRelY - (activeSliderDragOffset || 0);
  const travel = activeSliderTrackH - activeSliderThumbH;
  const newFrac = (travel > 0)
    ? Math.max(0, Math.min(1, (targetThumbY - activeSliderTrackY) / travel))
    : 0;
  const newScroll = Math.round(newFrac * activeSliderMaxScroll);
  setPanelScroll(activeSliderViewMode, newScroll);
}
window.updatePanelSliderDrag = updatePanelSliderDrag;

function stopPanelSliderDrag()
{
  if (isPanelSliderDragging)
  {
    isPanelSliderDragging = false;
    activeSliderQuad = -1;
    return true;
  }
  return false;
}
window.stopPanelSliderDrag = stopPanelSliderDrag;

function setHoveredSliderQuad(quad)
{
  hoveredSliderQuad = quad;
}
window.setHoveredSliderQuad = setHoveredSliderQuad;

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
      const mA = getClusterMembersList(a);
      const countA = (mA && mA.length > 0)
        ? mA.length
        : ((clusters[a] && clusters[a].members) || 0);
      const mB = getClusterMembersList(b);
      const countB = (mB && mB.length > 0)
        ? mB.length
        : ((clusters[b] && clusters[b].members) || 0);
      if (countB !== countA) return countB - countA;
      return a - b;
    });
  }
  else if (imageClustersSortMode === 'size_asc')
  {
    indices.sort((a, b) => {
      const mA = getClusterMembersList(a);
      const countA = (mA && mA.length > 0)
        ? mA.length
        : ((clusters[a] && clusters[a].members) || 0);
      const mB = getClusterMembersList(b);
      const countB = (mB && mB.length > 0)
        ? mB.length
        : ((clusters[b] && clusters[b].members) || 0);
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
    if (typeof updateImageQuadDropdowns === 'function')
    {
      updateImageQuadDropdowns();
    }
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

  if (typeof updateImageQuadDropdowns === 'function')
  {
    updateImageQuadDropdowns();
  }
}

/**
 * Get human-readable title for a panel view mode.
 */
function getImageViewTitle(mode)
{
  switch (mode)
  {
    case 'current_frame': return 'Current Frame';
    case 'anchor': return 'Cluster Anchor';
    case 'residual': return 'Residual (|f - C|)';
    case 'nn1': return 'Nearest Neighbor #1';
    case 'nn1_diff': return 'NN #1 Difference';
    case 'members': return 'Cluster Members';
    case 'knn': return 'k-NN Neighbors';
    case 'clusters': return 'All Clusters';
    default: return 'Image View';
  }
}

/**
 * Draw compact context pill in quadrant header.
 */
function drawContextPill(ctx, x, y, text, color)
{
  ctx.save();
  ctx.font = 'bold 9px monospace';
  const tw = ctx.measureText(text).width;
  const pw = tw + 10;
  const ph = 18;
  ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
  ctx.strokeStyle = color || '#38bdf8';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, pw, ph, 3);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = color || '#38bdf8';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + pw / 2, y + ph / 2);
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

  // Fallback assignedId from inspected or selected cluster if needed
  if (assignedId < 0 && typeof inspectedClusterId !== 'undefined' && inspectedClusterId >= 0)
  {
    assignedId = inspectedClusterId;
  }
  else if (assignedId < 0 && clusters && clusters.length > 0)
  {
    assignedId = (typeof selectedClusterId !== 'undefined' && selectedClusterId >= 0)
      ? selectedClusterId : 0;
  }

  const assignedCluster = (assignedId >= 0 && clusters[assignedId]) ? clusters[assignedId] : null;

  const viewMode = (typeof getImagePanelViewMode === 'function')
    ? getImagePanelViewMode(qIdx)
    : (qIdx === 0 ? 'current_frame'
      : (qIdx === 1 ? 'anchor' : (qIdx === 2 ? 'members' : 'clusters')));

  const maxOffset = (typeof maximizedQuad === 'undefined' || maximizedQuad === null) ? 28 : 0;

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

  // -------------------------------------------------------------
  // View: Current Frame
  // -------------------------------------------------------------
  if (viewMode === 'current_frame')
  {
    if (rect.w >= 340)
    {
      drawContextPill(
        ctx,
        rect.x + 192,
        rect.y + 4,
        isRetro ? `F#${activeFrameIdx + 1} (Retro)` : 'Live',
        isRetro ? '#facc15' : '#38bdf8'
      );
    }

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
        `quad_${qIdx}_frame`
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
  // View: Cluster Anchor
  // -------------------------------------------------------------
  else if (viewMode === 'anchor')
  {
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    const hasKnn = (knnList.length > 0);

    if (rect.w >= 340 && assignedCluster)
    {
      drawContextPill(
        ctx,
        rect.x + 192,
        rect.y + 4,
        `C${assignedCluster.id}`,
        assignedCluster.color || '#4ade80'
      );
    }

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
        `quad_${qIdx}_anchor`
      );

      const memList = (assignedId >= 0) ? getClusterMembersList(assignedId) : [];
      const memCount = (memList && memList.length > 0)
        ? memList.length
        : (assignedCluster ? (assignedCluster.members || 0) : 0);
      const isMatch = lastDist <= (rlim || 0.1);
      drawBadge(
        ctx,
        rect.x + 8,
        rect.y + rect.h - 8,
        `Members: ${memCount} | d(f,c): ${lastDist.toFixed(3)} | Cycle ⇄`,
        isMatch ? '#4ade80' : '#facc15'
      );
    }
    else
    {
      drawEmptyMessage(ctx, rect, 'No anchor matched yet');
    }
  }

  // -------------------------------------------------------------
  // View: Residual (|f - C|)
  // -------------------------------------------------------------
  else if (viewMode === 'residual')
  {
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    const hasKnn = (knnList.length > 0);

    if (rect.w >= 340 && assignedCluster)
    {
      drawContextPill(
        ctx,
        rect.x + 192,
        rect.y + 4,
        `|f - C${assignedCluster.id}|`,
        '#f87171'
      );
    }

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
        `quad_${qIdx}_residual`
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

  // -------------------------------------------------------------
  // View: Nearest Neighbor #1
  // -------------------------------------------------------------
  else if (viewMode === 'nn1')
  {
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    const nn1 = (knnList.length > 0) ? knnList[0] : null;

    if (rect.w >= 340 && nn1)
    {
      drawContextPill(
        ctx,
        rect.x + 192,
        rect.y + 4,
        `F#${nn1.frameIdx + 1}`,
        '#facc15'
      );
    }

    drawToggleChip(
      ctx, rect.x + rect.w - 120 - maxOffset, rect.y + 4, 112, 18, '⇄ NN Diff (3/4)'
    );

    const nnBuf = (nn1 && benchmarkDataset) ? benchmarkDataset[nn1.frameIdx] : null;
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
        `quad_${qIdx}_nn1`
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

  // -------------------------------------------------------------
  // View: Nearest Neighbor #1 Difference
  // -------------------------------------------------------------
  else if (viewMode === 'nn1_diff')
  {
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    const nn1 = (knnList.length > 0) ? knnList[0] : null;

    if (rect.w >= 340 && nn1)
    {
      drawContextPill(
        ctx,
        rect.x + 192,
        rect.y + 4,
        '|f - NN1|',
        '#c084fc'
      );
    }

    drawToggleChip(
      ctx, rect.x + rect.w - 120 - maxOffset, rect.y + 4, 112, 18, '⇄ Anchor (4/4)'
    );

    const nnBuf = (nn1 && benchmarkDataset) ? benchmarkDataset[nn1.frameIdx] : null;
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
        `quad_${qIdx}_nn1_diff`
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

  // -------------------------------------------------------------
  // View: Cluster Members
  // -------------------------------------------------------------
  else if (viewMode === 'members')
  {
    const members = (assignedId >= 0) ? getClusterMembersList(assignedId) : [];
    const mCount = members.length;
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    const kCount = knnList.length;

    if (rect.w >= 340)
    {
      drawContextPill(
        ctx,
        rect.x + 192,
        rect.y + 4,
        `${mCount} frames`,
        '#a78bfa'
      );
    }

    const knnLabel = (kCount > 0) ? `⇄ k-NN (${kCount})` : '⇄ k-NN';
    drawToggleChip(ctx, rect.x + rect.w - 100 - maxOffset, rect.y + 4, 92, 18, knnLabel);

    if (mCount > 0)
    {
      const thumbSize = getImageThumbSize();
      const style = getThumbCardStyle(thumbSize);
      const cardW = thumbSize;
      const cardH = thumbSize + style.infoH;
      const gap = 8;
      const availableW = contentW - 16;
      const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));
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
          `quad_${qIdx}_mthumb_${i}`
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

      ctx.restore();

      const slider = getPanelSliderRect(qIdx, 0, 0, rect, 'members');
      if (slider)
      {
        drawPanelSlider(ctx, slider);
      }
    }
    else
    {
      drawEmptyMessage(ctx, rect, 'No member frames in this cluster yet');
    }
  }

  // -------------------------------------------------------------
  // View: k-NN Neighbors
  // -------------------------------------------------------------
  else if (viewMode === 'knn')
  {
    const members = (assignedId >= 0) ? getClusterMembersList(assignedId) : [];
    const mCount = members.length;
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    const kCount = knnList.length;

    if (rect.w >= 340)
    {
      drawContextPill(
        ctx,
        rect.x + 192,
        rect.y + 4,
        `k=${kCount}`,
        '#c084fc'
      );
    }

    const memLabel = `⇄ Members (${mCount})`;
    drawToggleChip(ctx, rect.x + rect.w - 116 - maxOffset, rect.y + 4, 108, 18, memLabel);

    if (kCount > 0)
    {
      const thumbSize = getImageThumbSize();
      const style = getThumbCardStyle(thumbSize, true);
      const cardW = thumbSize;
      const cardH = thumbSize + style.infoH;
      const gap = 8;
      const availableW = contentW - 16;
      const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));
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
          `quad_${qIdx}_knn_thumb_${r}`
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

      ctx.restore();

      const slider = getPanelSliderRect(qIdx, 0, 0, rect, 'knn');
      if (slider)
      {
        drawPanelSlider(ctx, slider);
      }
    }
    else
    {
      drawEmptyMessage(
        ctx, rect, 'No k-NN results computed yet. Click "▶ Compute k-NN" to run solver.'
      );
    }
  }

  // -------------------------------------------------------------
  // View: All Clusters
  // -------------------------------------------------------------
  else if (viewMode === 'clusters')
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

    const sortChipW = 76;
    const sortChipH = 18;
    const sortChipX = rect.x + rect.w - sortChipW - 8 - maxOffset;
    const sortChipY = rect.y + 4;

    if (rect.w >= 340)
    {
      drawContextPill(
        ctx,
        rect.x + 192,
        rect.y + 4,
        `${kCount} anchors`,
        '#38bdf8'
      );
    }
    drawToggleChip(ctx, sortChipX, sortChipY, sortChipW, sortChipH, sortChipLabel);

    if (kCount > 0)
    {
      const thumbSize = getImageThumbSize();
      const style = getThumbCardStyle(thumbSize);
      const cardW = thumbSize;
      const cardH = thumbSize + style.infoH;
      const gap = 8;
      const availableW = contentW - 16;
      const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));
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

        const memList = getClusterMembersList(i);
        const memCount = (memList && memList.length > 0)
          ? memList.length
          : (cl ? (cl.members || 0) : 0);
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
          `quad_${qIdx}_cl_thumb_${i}`
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

      ctx.restore();

      const slider = getPanelSliderRect(qIdx, 0, 0, rect, 'clusters');
      if (slider)
      {
        drawPanelSlider(ctx, slider);
      }
    }
    else
    {
      drawEmptyMessage(ctx, rect, 'No clusters discovered yet');
    }
  }

  // Current Frame stats box if room
  if (viewMode === 'current_frame')
  {
    const labelPts = `${frameBuf ? 1 : 0} pts`;
    const labelClust = `${clusters ? clusters.length : 0} cl`;
    const fullText = `${labelPts}  •  ${labelClust}`;

    ctx.save();
    ctx.font = 'bold 9.5px monospace';
    const textW = ctx.measureText(fullText).width;
    const boxW = textW + 14;
    const boxH = 18;
    const maxW = (typeof maximizedQuad === 'undefined' || maximizedQuad === null) ? 28 : 0;
    const boxX = rect.x + rect.w - boxW - 8 - maxW;
    const boxY = rect.y + 4;

    if (boxX > rect.x + 220)
    {
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
    }
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
  const viewMode = (typeof getImagePanelViewMode === 'function')
    ? getImagePanelViewMode(qIdx)
    : 'current_frame';
  const name = getImageViewTitle(viewMode);
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
  const nameLabel = `🔍 Q${qIdx}: ${name}`;
  ctx.font = '9.5px sans-serif';
  const nameW = ctx.measureText(nameLabel).width + 12;
  const nameX = btnX - nameW - 6;
  if (nameX > 220)
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
  // 0. ABCD Reconstruction Image View Click / Lock Handling
  if (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView &&
      typeof isReconstructionImageMode === 'function' && isReconstructionImageMode())
  {
    const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;
    const slotC = (typeof datasetSlots !== 'undefined') ? datasetSlots['C'] : null;
    const totalQ = (slotC && slotC.benchmarkDataset) ? slotC.benchmarkDataset.length : 0;
    const curQ = (typeof inspectedImageFrameIdx === 'number' && inspectedImageFrameIdx >= 0)
      ? inspectedImageFrameIdx : 0;

    const neighbors = (slotD && slotD.reconstructionSourceNeighbors)
      ? slotD.reconstructionSourceNeighbors[curQ] : null;
    const topNeighborId = (neighbors && neighbors.length > 0) ? neighbors[0].id : curQ;

    if (qIdx === 0 || qIdx === 1)
    {
      const headerH = 28;
      const qRect = getImageQuadRect(qIdx, W, H);

      // Check click in header: toggles mode between Gallery and Single Frame
      if (py >= qRect.y && py <= qRect.y + headerH)
      {
        if (qIdx === 0)
        {
          const curMode = (typeof reconPanelAMode !== 'undefined') ? reconPanelAMode : 'knn';
          const next = (curMode === 'single') ? 'knn' : 'single';
          if (typeof setReconPanelAMode === 'function')
          {
            setReconPanelAMode(next);
          }
          else
          {
            reconPanelAMode = next;
          }
          if (typeof showToast === 'function')
          {
            showToast(next === 'single'
              ? 'Panel [A]: Single Frame in Dataset A'
              : 'Panel [A]: Training Input [A] (k-NN)');
          }
        }
        else
        {
          const curMode = (typeof reconPanelBMode !== 'undefined') ? reconPanelBMode : 'targets';
          const next = (curMode === 'single') ? 'targets' : 'single';
          if (typeof setReconPanelBMode === 'function')
          {
            setReconPanelBMode(next);
          }
          else
          {
            reconPanelBMode = next;
          }
          if (typeof showToast === 'function')
          {
            showToast(next === 'single'
              ? 'Panel [B]: Single Frame in Dataset B'
              : 'Panel [B]: Training Output [B] (Targets)');
          }
        }
        if (typeof draw === 'function') draw();
        return true;
      }

      // If in Single Frame mode, clicking the body toggles query frame lock
      const isSingleMode = (qIdx === 0 && typeof reconPanelAMode !== 'undefined' &&
                            reconPanelAMode === 'single') ||
                           (qIdx === 1 && typeof reconPanelBMode !== 'undefined' &&
                            reconPanelBMode === 'single');
      if (isSingleMode)
      {
        if (typeof reconLockedQueryIdx !== 'undefined' && reconLockedQueryIdx === curQ)
        {
          reconLockedQueryIdx = -1;
          reconHoveredQueryIdx = -1;
          if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
        }
        else
        {
          reconLockedQueryIdx = curQ;
          reconLockedTrainingIdx = -1;
          if (typeof showToast === 'function')
          {
            showToast(`🔒 Locked Frame #${curQ + 1}`);
          }
        }
        if (typeof draw === 'function') draw();
        return true;
      }

      // Otherwise in thumbnail gallery mode: check clicked thumbnail
      const neighbors = (typeof getReconstructionKnnNeighbors === 'function')
        ? getReconstructionKnnNeighbors(curQ) : null;
      if (neighbors && neighbors.length > 0)
      {
        const pad = 8;
        const contentX = qRect.x + pad;
        const contentY = qRect.y + headerH + pad;
        const contentW = qRect.w - pad * 2;
        const thumbSize = getImageThumbSize();
        const style = getThumbCardStyle(thumbSize, true);
        const cardW = thumbSize;
        const cardH = thumbSize + style.infoH;
        const gap = 8;
        const availableW = contentW - 16;
        const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));
        const scrollY = (typeof imageReconKnnScrollY !== 'undefined')
          ? imageReconKnnScrollY : 0;

        let clickedTrainIdx = -1;
        for (let r = 0; r < neighbors.length; r++)
        {
          const col = r % cols;
          const row = Math.floor(r / cols);
          const tx = contentX + col * (cardW + gap);
          const ty = contentY + row * (cardH + gap) - scrollY;

          if (px >= tx && px <= tx + cardW && py >= ty && py <= ty + cardH)
          {
            clickedTrainIdx = neighbors[r].id;
            break;
          }
        }

        if (clickedTrainIdx >= 0)
        {
          if (typeof reconLockedTrainingIdx !== 'undefined' &&
              reconLockedTrainingIdx === clickedTrainIdx)
          {
            reconLockedTrainingIdx = -1;
            reconHoveredTrainingIdx = -1;
            if (typeof showToast === 'function')
            {
              showToast('🔓 Selection Unlocked');
            }
          }
          else
          {
            reconLockedTrainingIdx = clickedTrainIdx;
            reconHoveredTrainingIdx = clickedTrainIdx;
            reconLockedTrainingSlot = (qIdx === 0) ? 'A' : 'B';
            reconLockedQueryIdx = -1;
            if (typeof showToast === 'function')
            {
              showToast(`🔒 Locked Training Pair #${clickedTrainIdx + 1}`);
            }
          }
          if (typeof draw === 'function') draw();
          return true;
        }
      }
      return false;
    }
    else if (qIdx === 2 || qIdx === 3)
    {
      if (typeof reconLockedQueryIdx !== 'undefined' && reconLockedQueryIdx === curQ)
      {
        reconLockedQueryIdx = -1;
        reconHoveredQueryIdx = -1;
        if (typeof showToast === 'function') showToast('🔓 Selection Unlocked');
      }
      else
      {
        reconLockedQueryIdx = curQ;
        reconLockedTrainingIdx = -1;
        if (typeof showToast === 'function')
        {
          showToast(`🔒 Locked Query Frame #${curQ + 1}`);
        }
      }
      if (typeof draw === 'function') draw();
      return true;
    }
    return false;
  }

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

  const isMax = (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null);
  const effectiveQuad = isMax ? maximizedQuad : qIdx;
  const rect = isMax ? { x: 0, y: 0, w: W, h: H } : getImageQuadRect(effectiveQuad, W, H);

  // 2. Check if clicking Maximize Icon in quadrant header (4-panel mode)
  if (!isMax)
  {
    if (py >= rect.y && py <= rect.y + 26 &&
        px >= rect.x + rect.w - 32 && px <= rect.x + rect.w - 4)
    {
      maximizedQuad = qIdx;
      if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
      if (typeof draw === 'function') draw();
      const viewMode = (typeof getImagePanelViewMode === 'function')
        ? getImagePanelViewMode(qIdx) : 'current_frame';
      const name = getImageViewTitle(viewMode);
      if (typeof showToast === 'function')
      {
        showToast(`🔍 Maximized Q${qIdx}: ${name} (Click 'All 4 Panels' or Esc to restore)`);
      }
      return true;
    }
  }

  const isRetro = (typeof inspectedImageFrameIdx !== 'undefined' &&
    inspectedImageFrameIdx >= 0 &&
    benchmarkDataset && benchmarkDataset[inspectedImageFrameIdx]);
  const hasBench = (typeof benchmarkDataset !== 'undefined' &&
    benchmarkDataset && benchmarkDataset.length > 0);
  const hasTotal = (typeof totalFrames !== 'undefined' && totalFrames > 0);
  const activeFrameIdx = isRetro
    ? inspectedImageFrameIdx
    : (hasTotal ? totalFrames - 1 : (hasBench ? 0 : -1));

  const viewMode = (typeof getImagePanelViewMode === 'function')
    ? getImagePanelViewMode(effectiveQuad)
    : 'current_frame';

  const maxOffset = isMax ? 0 : 28;
  const pad = 12;
  const headerH = 26;
  const contentX = rect.x + pad;
  const contentY = rect.y + headerH + 6;
  const contentW = rect.w - pad * 2;
  const contentH = rect.h - headerH - pad - 6;

  // Views with toggle chip: anchor, residual, nn1, nn1_diff
  if (viewMode === 'anchor' || viewMode === 'residual' ||
      viewMode === 'nn1' || viewMode === 'nn1_diff')
  {
    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    const hasKnn = (knnList.length > 0);

    let nextMode = 'anchor';
    if (viewMode === 'anchor') nextMode = 'residual';
    else if (viewMode === 'residual') nextMode = hasKnn ? 'nn1' : 'anchor';
    else if (viewMode === 'nn1') nextMode = 'nn1_diff';
    else nextMode = 'anchor';

    if (typeof setImagePanelViewMode === 'function')
    {
      setImagePanelViewMode(effectiveQuad, nextMode);
    }
    else
    {
      imageTopRightMode = nextMode;
      if (typeof draw === 'function') draw();
    }
    return true;
  }

  // View: members
  if (viewMode === 'members')
  {
    // Check if clicking header toggle chip: ⇄ k-NN
    if (py >= rect.y && py <= rect.y + 26 &&
        px >= rect.x + rect.w - 130 - maxOffset && px <= rect.x + rect.w - maxOffset)
    {
      if (typeof setImagePanelViewMode === 'function')
      {
        setImagePanelViewMode(effectiveQuad, 'knn');
      }
      else
      {
        imageQ2ViewMode = 'knn';
        if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
        if (typeof draw === 'function') draw();
      }
      if (typeof showToast === 'function')
      {
        showToast(`⚡ Q${effectiveQuad}: Showing k-NN Nearest Neighbors`);
      }
      return true;
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
    const availableW = contentW - 16;
    const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));

    if (px >= rect.x + rect.w - 18) return false;

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
      if (typeof selectImageClusterMember === 'function')
      {
        selectImageClusterMember(clickedIdx, false);
      }
      else if (typeof selectImageFrame === 'function')
      {
        selectImageFrame(frameIdx);
      }
      if (typeof showToast === 'function')
      {
        showToast(`🔍 Inspected Member #${clickedIdx + 1} (Frame #${frameIdx + 1})`);
      }
      return true;
    }
    return false;
  }

  // View: knn
  if (viewMode === 'knn')
  {
    // Check if clicking header toggle chip: ⇄ Members
    if (py >= rect.y && py <= rect.y + 26 &&
        px >= rect.x + rect.w - 130 - maxOffset && px <= rect.x + rect.w - maxOffset)
    {
      if (typeof setImagePanelViewMode === 'function')
      {
        setImagePanelViewMode(effectiveQuad, 'members');
      }
      else
      {
        imageQ2ViewMode = 'members';
        if (typeof syncImageQuadUI === 'function') syncImageQuadUI();
        if (typeof draw === 'function') draw();
      }
      if (typeof showToast === 'function')
      {
        showToast(`👥 Q${effectiveQuad}: Showing Cluster Members`);
      }
      return true;
    }

    const knnList = getActiveImageKnnNeighbors(activeFrameIdx);
    if (knnList.length === 0) return false;

    const thumbSize = getImageThumbSize();
    const style = getThumbCardStyle(thumbSize, true);
    const cardW = thumbSize;
    const cardH = thumbSize + style.infoH;
    const gap = 8;
    const availableW = contentW - 16;
    const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));

    if (px >= rect.x + rect.w - 18) return false;

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

  // View: clusters
  if (viewMode === 'clusters')
  {
    // Check if clicked inside sort chip button at top-right of panel
    const sortChipW = 76;
    const sortChipH = 18;
    const sortChipX = rect.x + rect.w - sortChipW - 8 - maxOffset;
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
    const availableW = contentW - 16;
    const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));

    if (px >= rect.x + rect.w - 18) return false;

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
      if (typeof selectImageCluster === 'function')
      {
        selectImageCluster(clIdx, false);
      }
      else
      {
        const members = getClusterMembersList(clIdx);
        if (members && members.length > 0 && typeof selectImageFrame === 'function')
        {
          selectImageFrame(members[0]);
        }
        else
        {
          selectedClusterId = clIdx;
          inspectedClusterId = clIdx;
          if (typeof updateUI === 'function') updateUI();
          if (typeof draw === 'function') draw();
        }
      }
      if (typeof showToast === 'function')
      {
        showToast(`🗂️ Inspected Cluster C${clIdx}`);
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

/**
 * Synchronize and position the 4 quadrant dropdown overlay elements.
 */
function updateImageQuadDropdowns()
{
  const container = document.getElementById('imageQuadDropdowns');
  if (!container) return;

  if (typeof dataMode === 'undefined' || dataMode !== 'image' ||
      (typeof isRecon4PanelView !== 'undefined' && isRecon4PanelView))
  {
    container.style.display = 'none';
    return;
  }

  const canvas = document.getElementById('simCanvas');
  const canvasWrapper = document.getElementById('canvasWrapper');
  if (!canvas || !canvasWrapper) return;

  const cRect = canvas.getBoundingClientRect();
  const wRect = canvasWrapper.getBoundingClientRect();
  const offsetX = cRect.left - wRect.left;
  const offsetY = cRect.top - wRect.top;
  const cW = cRect.width;
  const cH = cRect.height;

  container.style.display = 'block';

  const isMaximized = (typeof maximizedQuad !== 'undefined' && maximizedQuad !== null);

  for (let q = 0; q < 4; q++)
  {
    const wrap = document.getElementById(`imageQuadDropdown${q}`);
    const sel = document.getElementById(`selectImgQuad${q}`);
    if (!wrap) continue;

    if (isMaximized)
    {
      if (q === maximizedQuad)
      {
        wrap.style.display = 'inline-flex';
        wrap.style.left = `${Math.round(offsetX + 8)}px`;
        wrap.style.top = `${Math.round(offsetY + 2)}px`;
      }
      else
      {
        wrap.style.display = 'none';
      }
    }
    else
    {
      wrap.style.display = 'inline-flex';
      const qX = (q % 2 === 1) ? (cW / 2) : 0;
      const qY = (q >= 2) ? (cH / 2) : 0;

      wrap.style.left = `${Math.round(offsetX + qX + 8)}px`;
      wrap.style.top = `${Math.round(offsetY + qY + 2)}px`;
    }

    if (sel && typeof getImagePanelViewMode === 'function')
    {
      const currentVal = getImagePanelViewMode(q);
      if (sel.value !== currentVal)
      {
        sel.value = currentVal;
      }
    }
  } // for (let q = 0; q < 4; q++)
}

/**
 * Check if the active reconstruction configuration involves image datasets.
 *
 * @returns {boolean} True if slot A, B, C, or D is in image dataMode.
 */
function isReconstructionImageMode()
{
  if (typeof isRecon4PanelView === 'undefined' || !isRecon4PanelView)
  {
    return false;
  }
  const sA = (typeof datasetSlots !== 'undefined') ? datasetSlots['A'] : null;
  const sB = (typeof datasetSlots !== 'undefined') ? datasetSlots['B'] : null;
  const sC = (typeof datasetSlots !== 'undefined') ? datasetSlots['C'] : null;
  const sD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;

  return Boolean(
    (sD && sD.dataMode === 'image') ||
    (sB && sB.dataMode === 'image') ||
    (sA && sA.dataMode === 'image') ||
    (sC && sC.dataMode === 'image') ||
    (typeof dataMode !== 'undefined' && dataMode === 'image')
  );
}

/**
 * Compute or retrieve k-NN neighbors for a query frame in ABCD reconstruction mode.
 *
 * @param {number} queryIdx - Query index in Dataset C / D.
 * @param {number} [k=10] - Number of neighbors if computed on-demand.
 * @returns {Array<{id: number, dist: number, weight: number}>} Array of neighbor objects.
 */
function getReconstructionKnnNeighbors(queryIdx, k = 10)
{
  const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;
  if (slotD && slotD.reconstructionSourceNeighbors &&
      slotD.reconstructionSourceNeighbors[queryIdx])
  {
    return slotD.reconstructionSourceNeighbors[queryIdx];
  }

  const slotA = (typeof datasetSlots !== 'undefined') ? datasetSlots['A'] : null;
  const slotC = (typeof datasetSlots !== 'undefined') ? datasetSlots['C'] : null;
  const ptsA = (slotA && slotA.benchmarkDataset) ? slotA.benchmarkDataset : null;
  const ptsC = (slotC && slotC.benchmarkDataset) ? slotC.benchmarkDataset : null;

  if (!ptsA || !ptsC || queryIdx < 0 || queryIdx >= ptsC.length || ptsA.length === 0)
  {
    return [];
  }

  if (!slotC._onDemandImageKnnCache)
  {
    slotC._onDemandImageKnnCache = new Map();
  }
  if (slotC._onDemandImageKnnCache.has(queryIdx))
  {
    return slotC._onDemandImageKnnCache.get(queryIdx);
  }

  const effK = Math.min(
    (slotD && slotD.reconstructionInfo && slotD.reconstructionInfo.k)
      ? slotD.reconstructionInfo.k : k,
    ptsA.length
  );
  const qc = ptsC[queryIdx];
  const numCandidates = ptsA.length;
  const dists = new Float64Array(numCandidates);
  const indices = new Int32Array(numCandidates);

  for (let j = 0; j < numCandidates; j++)
  {
    const pa = ptsA[j];
    let sumSq = 0.0;
    const len = Math.min(qc.length, pa.length);
    for (let d = 0; d < len; d++)
    {
      const diff = qc[d] - pa[d];
      sumSq += diff * diff;
    }
    dists[j] = Math.sqrt(sumSq);
    indices[j] = j;
  }

  for (let p = 0; p < effK; p++)
  {
    let minIdx = p;
    for (let j = p + 1; j < numCandidates; j++)
    {
      if (dists[j] < dists[minIdx])
      {
        minIdx = j;
      }
    }
    const tmpD = dists[p]; dists[p] = dists[minIdx]; dists[minIdx] = tmpD;
    const tmpI = indices[p]; indices[p] = indices[minIdx]; indices[minIdx] = tmpI;
  }

  const normW = 1.0 / effK;
  const result = [];
  for (let p = 0; p < effK; p++)
  {
    result.push({
      id: indices[p],
      dist: dists[p],
      weight: normW
    });
  }

  slotC._onDemandImageKnnCache.set(queryIdx, result);
  return result;
}

/**
 * Render a gallery of k-NN thumbnails for Panel A or Panel B in ABCD reconstruction view.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas 2D context
 * @param {number} qX - Quadrant X
 * @param {number} qY - Quadrant Y
 * @param {number} qW - Quadrant width
 * @param {number} qH - Quadrant height
 * @param {string} slotId - Slot ID ('A' or 'B')
 * @param {string} title - Panel title
 * @param {string} pillText - Text for pill badge
 * @param {string} pillColor - Color for pill badge
 * @param {Array<Float32Array|Array<number>>} pts - Dataset frames (ptsA or ptsB)
 * @param {Array<{id: number, dist: number, weight: number}>} neighbors - k-NN neighbors
 * @param {number} imgW - Image raster width
 * @param {number} imgH - Image raster height
 * @param {number} scrollY - Vertical scroll offset
 */
function renderReconThumbnailGallery(
  ctx,
  qX, qY, qW, qH,
  slotId,
  title,
  pillText,
  pillColor,
  pts,
  neighbors,
  imgW, imgH,
  scrollY
)
{
  ctx.save();
  ctx.beginPath();
  ctx.rect(qX, qY, qW, qH);
  ctx.clip();

  // Background
  ctx.fillStyle = '#0b1120';
  ctx.fillRect(qX, qY, qW, qH);

  // Header bar
  const headerH = 28;
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(qX, qY, qW, headerH);
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(qX, qY + headerH);
  ctx.lineTo(qX + qW, qY + headerH);
  ctx.stroke();

  // Pill badge on left
  drawContextPill(ctx, qX + 8, qY + 5, pillText, pillColor);

  // Title next to pill
  ctx.font = 'bold 9px monospace';
  const pillW = ctx.measureText(pillText).width + 10;
  const textStartX = qX + 8 + pillW + 8;
  ctx.font = 'bold 11px system-ui, sans-serif';
  ctx.fillStyle = '#f8fafc';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(title, textStartX, qY + 14);

  // Mode Toggle Button Pill
  if (slotId === 'A' || slotId === 'B')
  {
    const titleW = ctx.measureText(title).width;
    const btnX = textStartX + titleW + 10;
    const btnW = 96;
    const btnH = 18;
    const btnY = qY + 5;
    ctx.fillStyle = (slotId === 'A')
      ? 'rgba(56, 189, 248, 0.15)' : 'rgba(74, 222, 128, 0.15)';
    ctx.strokeStyle = (slotId === 'A')
      ? 'rgba(56, 189, 248, 0.5)' : 'rgba(74, 222, 128, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(btnX, btnY, btnW, btnH, 3);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'bold 9px monospace';
    ctx.fillStyle = (slotId === 'A') ? '#38bdf8' : '#4ade80';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('⇄ Single Frame', btnX + btnW / 2, btnY + btnH / 2);
  }

  const kCount = (neighbors && pts) ? neighbors.length : 0;

  // Subtext / info on right
  const subText = (slotId === 'A')
    ? (kCount > 0 ? `k=${kCount} Selected Neighbors` : 'No Neighbors')
    : (kCount > 0 ? `k=${kCount} Paired Targets` : 'No Targets');

  ctx.font = '10px monospace';
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'right';
  ctx.fillText(subText, qX + qW - 10, qY + 14);

  // Content area
  const pad = 8;
  const contentX = qX + pad;
  const contentY = qY + headerH + pad;
  const contentW = qW - pad * 2;
  const contentH = qH - headerH - pad * 2;

  if (kCount > 0 && pts && pts.length > 0)
  {
    const thumbSize = getImageThumbSize();
    const style = getThumbCardStyle(thumbSize, true);
    const cardW = thumbSize;
    const cardH = thumbSize + style.infoH;
    const gap = 8;
    const availableW = contentW - 16;
    const cols = Math.max(1, Math.floor(availableW / (cardW + gap)));
    const rows = Math.ceil(kCount / cols);
    const totalH = rows * (cardH + gap);
    const maxScroll = Math.max(0, totalH - contentH);
    const clampedScroll = Math.max(0, Math.min(maxScroll, scrollY || 0));

    ctx.save();
    ctx.beginPath();
    ctx.rect(contentX, contentY, contentW, contentH);
    ctx.clip();

    for (let r = 0; r < kCount; r++)
    {
      const item = neighbors[r];
      const trainIdx = item.id;
      const col = r % cols;
      const row = Math.floor(r / cols);
      const tx = contentX + col * (cardW + gap);
      const ty = contentY + row * (cardH + gap) - clampedScroll;

      if (ty + cardH < contentY - 10 || ty > contentY + contentH + 10)
      {
        continue;
      }

      const rasterData = (trainIdx >= 0 && trainIdx < pts.length)
        ? pts[trainIdx] : null;
      if (!rasterData)
      {
        continue;
      }

      const isLocked = (typeof reconLockedTrainingIdx !== 'undefined' &&
                        reconLockedTrainingIdx === trainIdx);
      const isHovered = (typeof reconHoveredTrainingIdx !== 'undefined' &&
                         reconHoveredTrainingIdx === trainIdx);
      const isSelected = isLocked || isHovered;

      const strokeColor = isSelected
        ? '#facc15'
        : (r === 0 ? '#facc15' : (pillColor || '#38bdf8'));
      const lineWidth = isSelected ? 2.5 : (r === 0 ? 2.0 : 1.2);

      // Thumbnail Image raster
      ctx.fillStyle = '#020617';
      ctx.fillRect(tx, ty, thumbSize, thumbSize);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(tx, ty, thumbSize, thumbSize);

      drawRasterBuffer(
        ctx,
        rasterData,
        imgW,
        imgH,
        tx,
        ty,
        thumbSize,
        thumbSize,
        1.0,
        `recon_${slotId}_thumb_${trainIdx}`
      );

      // Info box below image
      const infoY = ty + thumbSize;
      ctx.fillStyle = isSelected
        ? 'rgba(250, 204, 21, 0.22)'
        : 'rgba(15, 23, 42, 0.95)';
      ctx.fillRect(tx, infoY, thumbSize, style.infoH);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1.0;
      ctx.strokeRect(tx, infoY, thumbSize, style.infoH);

      // Line 1: Rank and Frame #
      const line1Text = `#${r + 1} ${slotId}#${trainIdx + 1}`;
      ctx.fillStyle = isSelected ? '#facc15' : (r === 0 ? '#facc15' : '#f8fafc');
      ctx.font = style.fontBold;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(line1Text, tx + thumbSize / 2, infoY + style.infoH * 0.31);

      // Line 2: Dist / Weight
      const distVal = Number(item.dist) || 0.0;
      const weightVal = (Number(item.weight) || 0.0) * 100;
      let line2Text = '';
      if (slotId === 'A')
      {
        line2Text = (thumbSize >= 90)
          ? `d=${distVal.toFixed(3)} (${weightVal.toFixed(1)}%)`
          : `d=${distVal.toFixed(2)}`;
      }
      else
      {
        line2Text = (thumbSize >= 90)
          ? `w=${weightVal.toFixed(1)}%`
          : `w=${weightVal.toFixed(0)}%`;
      }

      ctx.fillStyle = pillColor || '#38bdf8';
      ctx.font = style.font;
      ctx.fillText(line2Text, tx + thumbSize / 2, infoY + style.infoH * 0.74);
    } // for r

    ctx.restore();

    // Vertical slider if needed
    const qEffective = (slotId === 'A') ? 0 : 1;
    const slider = getPanelSliderRect(
      qEffective,
      0,
      0,
      { x: qX, y: qY, w: qW, h: qH },
      (slotId === 'A') ? 'recon_knn_a' : 'recon_knn_b'
    );
    if (slider)
    {
      drawPanelSlider(ctx, slider);
    }
  }
  else
  {
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No training frames available', qX + qW / 2, qY + qH / 2);
  }

  ctx.restore();
}

/**
 * Renders the 4-panel ABCD reconstruction comparison view for raster image datasets.
 *
 * Quad 0 (Top-Left):     [A] Training Input (k-NN thumbnails to active query)
 * Quad 1 (Top-Right):    [B] Training Output (paired targets corresponding to neighbors in A)
 * Quad 2 (Bottom-Left):  [C] Query Input (query image frame)
 * Quad 3 (Bottom-Right): [D] Reconstructed Output (weighted combination of B targets)
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas 2D context
 * @param {number} W - Canvas width
 * @param {number} H - Canvas height
 */
function drawReconImage4PanelView(ctx, W, H)
{
  ctx.save();

  const slotA = (typeof datasetSlots !== 'undefined') ? datasetSlots['A'] : null;
  const slotB = (typeof datasetSlots !== 'undefined') ? datasetSlots['B'] : null;
  const slotC = (typeof datasetSlots !== 'undefined') ? datasetSlots['C'] : null;
  const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;

  const ptsA = (slotA && slotA.benchmarkDataset) ? slotA.benchmarkDataset : null;
  const ptsB = (slotB && slotB.benchmarkDataset) ? slotB.benchmarkDataset : null;
  const ptsC = (slotC && slotC.benchmarkDataset) ? slotC.benchmarkDataset : null;
  const ptsD = (slotD && slotD.benchmarkDataset) ? slotD.benchmarkDataset : null;

  const totalQ = ptsC ? ptsC.length : (ptsD ? ptsD.length : 0);

  // Active query index
  let queryIdx = 0;
  if (typeof reconLockedQueryIdx !== 'undefined' && reconLockedQueryIdx >= 0)
  {
    queryIdx = reconLockedQueryIdx;
  }
  else if (typeof reconHoveredQueryIdx !== 'undefined' && reconHoveredQueryIdx >= 0)
  {
    queryIdx = reconHoveredQueryIdx;
  }
  else if (typeof inspectedImageFrameIdx === 'number' && inspectedImageFrameIdx >= 0)
  {
    queryIdx = inspectedImageFrameIdx;
  }
  else if (typeof selectedKnnQuerySample !== 'undefined' && selectedKnnQuerySample >= 0)
  {
    queryIdx = selectedKnnQuerySample;
  }
  if (totalQ > 0)
  {
    queryIdx = Math.max(0, Math.min(totalQ - 1, queryIdx));
  }

  // Neighbor lookup in training datasets A and B
  const neighborsList = getReconstructionKnnNeighbors(queryIdx);
  let topNeighborId = -1;
  let topNeighborDist = 0.0;
  let topNeighborWeight = 1.0;

  if (neighborsList && neighborsList.length > 0)
  {
    topNeighborId = neighborsList[0].id;
    topNeighborDist = Number(neighborsList[0].dist) || 0.0;
    topNeighborWeight = Number(neighborsList[0].weight) || (1.0 / neighborsList.length);
  }
  else if (ptsA && queryIdx < ptsA.length)
  {
    topNeighborId = queryIdx;
  }

  if (typeof reconLockedTrainingIdx !== 'undefined' && reconLockedTrainingIdx >= 0)
  {
    topNeighborId = reconLockedTrainingIdx;
  }
  else if (typeof reconHoveredTrainingIdx !== 'undefined' && reconHoveredTrainingIdx >= 0)
  {
    topNeighborId = reconHoveredTrainingIdx;
  }

  const frameA = (ptsA && topNeighborId >= 0 && topNeighborId < ptsA.length)
    ? ptsA[topNeighborId] : null;
  const frameB = (ptsB && topNeighborId >= 0 && topNeighborId < ptsB.length)
    ? ptsB[topNeighborId] : null;
  const frameC = (ptsC && queryIdx >= 0 && queryIdx < ptsC.length)
    ? ptsC[queryIdx] : null;
  const frameD = (ptsD && queryIdx >= 0 && queryIdx < ptsD.length)
    ? ptsD[queryIdx] : null;

  const imgWA = (slotA && slotA.imageWidth) || 32;
  const imgHA = (slotA && slotA.imageHeight) || 32;
  const imgWB = (slotB && slotB.imageWidth) || 32;
  const imgHB = (slotB && slotB.imageHeight) || 32;
  const imgWC = (slotC && slotC.imageWidth) || 32;
  const imgHC = (slotC && slotC.imageHeight) || 32;
  const imgWD = (slotD && slotD.imageWidth) || 32;
  const imgHD = (slotD && slotD.imageHeight) || 32;

  const isOverlay = (typeof reconOverlayMode !== 'undefined' && reconOverlayMode);
  const halfW = W / 2;
  const halfH = H / 2;

  function renderSingleReconQuad(
    qX, qY, qW, qH,
    slotId, title, subText,
    pillText, pillColor,
    pixels, imgW, imgH,
    metricText = '',
    footerText = ''
  )
  {
    ctx.save();
    ctx.beginPath();
    ctx.rect(qX, qY, qW, qH);
    ctx.clip();

    // Background
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(qX, qY, qW, qH);

    // Header bar
    const headerH = 28;
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(qX, qY, qW, headerH);
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(qX, qY + headerH);
    ctx.lineTo(qX + qW, qY + headerH);
    ctx.stroke();

    // Pill badge on left
    drawContextPill(ctx, qX + 8, qY + 5, pillText, pillColor);

    // Title next to pill
    ctx.font = 'bold 9px monospace';
    const pillW = ctx.measureText(pillText).width + 10;
    const textStartX = qX + 8 + pillW + 8;
    ctx.font = 'bold 11px system-ui, sans-serif';
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(title, textStartX, qY + 14);

    // Mode Toggle Button Pill
    if (slotId === 'A' || slotId === 'B')
    {
      const titleW = ctx.measureText(title).width;
      const btnX = textStartX + titleW + 10;
      const btnLabel = (slotId === 'A') ? '⇄ k-NN Gallery' : '⇄ Targets Gallery';
      ctx.font = 'bold 9px monospace';
      const btnW = ctx.measureText(btnLabel).width + 12;
      const btnH = 18;
      const btnY = qY + 5;
      ctx.fillStyle = (slotId === 'A')
        ? 'rgba(56, 189, 248, 0.15)' : 'rgba(74, 222, 128, 0.15)';
      ctx.strokeStyle = (slotId === 'A')
        ? 'rgba(56, 189, 248, 0.5)' : 'rgba(74, 222, 128, 0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(btnX, btnY, btnW, btnH, 3);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = (slotId === 'A') ? '#38bdf8' : '#4ade80';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(btnLabel, btnX + btnW / 2, btnY + btnH / 2);
    }

    // Subtext / info on right
    if (subText)
    {
      ctx.font = '10px monospace';
      ctx.fillStyle = '#94a3b8';
      ctx.textAlign = 'right';
      ctx.fillText(subText, qX + qW - 10, qY + 14);
    }

    // Image area
    const availPad = 8;
    const availX = qX + availPad;
    const availY = qY + headerH + availPad;
    const availW = qW - availPad * 2;
    const availH = qH - headerH - availPad * 2 - (footerText ? 16 : 0);

    if (pixels && pixels.length >= imgW * imgH && availW > 10 && availH > 10)
    {
      const aspect = imgW / imgH;
      let dstW = availW;
      let dstH = dstW / aspect;
      if (dstH > availH)
      {
        dstH = availH;
        dstW = dstH * aspect;
      }
      const dstX = availX + (availW - dstW) / 2;
      const dstY = availY + (availH - dstH) / 2;

      // Draw image
      drawRasterBuffer(
        ctx, pixels, imgW, imgH,
        dstX, dstY, dstW, dstH,
        1.0, slotId
      );

      // Border around image
      ctx.strokeStyle = pillColor ? `${pillColor}88` : 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;
      ctx.strokeRect(
        Math.round(dstX) - 0.5,
        Math.round(dstY) - 0.5,
        Math.round(dstW) + 1,
        Math.round(dstH) + 1
      );

      // Pixel dimensions tag on bottom-right of image
      ctx.font = '9px monospace';
      ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
      const dimLabel = `${imgW}×${imgH}`;
      const dimW = ctx.measureText(dimLabel).width + 6;
      ctx.fillRect(dstX + dstW - dimW - 2, dstY + dstH - 15, dimW + 2, 14);
      ctx.fillStyle = '#94a3b8';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(dimLabel, dstX + dstW - 3, dstY + dstH - 8);

      // Metric badge on top-right of image (e.g. Var, kth-dist, RMS)
      if (metricText)
      {
        ctx.font = 'bold 9px monospace';
        const mW = ctx.measureText(metricText).width + 8;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.strokeStyle = pillColor || '#c084fc';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(dstX + dstW - mW - 4, dstY + 4, mW + 4, 16, 3);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = pillColor || '#c084fc';
        ctx.textAlign = 'center';
        ctx.fillText(metricText, dstX + dstW - mW / 2 - 2, dstY + 12);
      }
    }
    else
    {
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillStyle = '#64748b';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No image data available', qX + qW / 2, qY + qH / 2);
    }

    // Optional footer hint
    if (footerText)
    {
      ctx.font = '9px system-ui, sans-serif';
      ctx.fillStyle = '#475569';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(footerText, qX + qW / 2, qY + qH - 4);
    }

    ctx.restore();
  }

  const isSingleA = (typeof reconPanelAMode !== 'undefined' && reconPanelAMode === 'single');
  const isSingleB = (typeof reconPanelBMode !== 'undefined' && reconPanelBMode === 'single');
  const totalA = ptsA ? ptsA.length : 0;
  const totalB = ptsB ? ptsB.length : 0;
  const singleFrameA = (ptsA && queryIdx >= 0 && queryIdx < ptsA.length)
    ? ptsA[queryIdx] : null;
  const singleFrameB = (ptsB && queryIdx >= 0 && queryIdx < ptsB.length)
    ? ptsB[queryIdx] : null;

  // Single Frame Metric for A (RMS vs C)
  let metricA = '';
  if (isSingleA && singleFrameA && frameC && singleFrameA.length === frameC.length)
  {
    let diffSumSq = 0.0;
    for (let d = 0; d < singleFrameA.length; d++)
    {
      const diff = singleFrameA[d] - frameC[d];
      diffSumSq += diff * diff;
    }
    const rmsAC = Math.sqrt(diffSumSq / singleFrameA.length);
    metricA = `RMS vs [C]: ${rmsAC.toFixed(4)}`;
  }

  // Single Frame Metric for B (RMS vs D)
  let metricB = '';
  let rmsBD = -1.0;
  if (singleFrameB && frameD && singleFrameB.length === frameD.length)
  {
    let diffSumSq = 0.0;
    for (let d = 0; d < singleFrameB.length; d++)
    {
      const diff = singleFrameB[d] - frameD[d];
      diffSumSq += diff * diff;
    }
    rmsBD = Math.sqrt(diffSumSq / singleFrameB.length);
    if (isSingleB)
    {
      metricB = `RMS vs [D]: ${rmsBD.toFixed(4)}`;
    }
  }

  if (!isOverlay)
  {
    // Normal 4-Quadrant View
    // Quad 0: [A] Training Input (k-NN thumbnails or Single Frame)
    if (isSingleA)
    {
      renderSingleReconQuad(
        0, 0, halfW, halfH,
        'A', 'Single Frame in Dataset A',
        `Frame #${queryIdx + 1} of ${totalA}`,
        '[A]', '#38bdf8',
        singleFrameA, imgWA, imgHA,
        metricA,
        'Wheel to scrub frames (Shift: ×10, Alt: ×50)'
      );
    }
    else
    {
      renderReconThumbnailGallery(
        ctx,
        0, 0, halfW, halfH,
        'A', 'Training Input [A] (k-NN)',
        '[A]', '#38bdf8',
        ptsA, neighborsList,
        imgWA, imgHA,
        imageReconKnnScrollY
      );
    }

    // Quad 1: [B] Training Output (Paired targets thumbnails or Single Frame)
    if (isSingleB)
    {
      renderSingleReconQuad(
        halfW, 0, halfW, halfH,
        'B', 'Single Frame in Dataset B',
        `Target #${queryIdx + 1} of ${totalB}`,
        '[B]', '#4ade80',
        singleFrameB, imgWB, imgHB,
        metricB,
        'Wheel to scrub frames (Shift: ×10, Alt: ×50)'
      );
    }
    else
    {
      renderReconThumbnailGallery(
        ctx,
        halfW, 0, halfW, halfH,
        'B', 'Training Output [B] (Targets)',
        '[B]', '#4ade80',
        ptsB, neighborsList,
        imgWB, imgHB,
        imageReconKnnScrollY
      );
    }

    // Quad 2: [C] Query Input
    const cSub = `Query #${queryIdx + 1} of ${totalQ}`;
    const scrubHint = 'Wheel to scrub queries (Shift: ×10, Alt: ×50)';
    renderSingleReconQuad(
      0, halfH, halfW, halfH,
      'C', 'Query Input', cSub,
      '[C]', '#fbbf24',
      frameC, imgWC, imgHC,
      '',
      scrubHint
    );

    // Quad 3: [D] Reconstructed Output
    const kVal = (slotD && slotD.reconstructionInfo)
      ? slotD.reconstructionInfo.k : (neighborsList ? neighborsList.length : 10);
    const dSub = `Reconstructed #${queryIdx + 1} (k=${kVal})`;
    let qualStr = '';
    if (slotD && slotD.reconVariance && queryIdx < slotD.reconVariance.length)
    {
      const varVal = slotD.reconVariance[queryIdx];
      qualStr = `Var: ${varVal.toFixed(4)}`;
    }
    if (isSingleB && rmsBD >= 0.0)
    {
      qualStr = qualStr ? `${qualStr} | RMS: ${rmsBD.toFixed(4)}`
                        : `RMS vs [B]: ${rmsBD.toFixed(4)}`;
    }
    renderSingleReconQuad(
      halfW, halfH, halfW, halfH,
      'D', 'Reconstructed Output', dSub,
      '[D]', '#c084fc',
      frameD, imgWD, imgHD,
      qualStr
    );

    // Divider Grid Lines
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(halfW, 0);
    ctx.lineTo(halfW, H);
    ctx.moveTo(0, halfH);
    ctx.lineTo(W, halfH);
    ctx.stroke();
  }
  else
  {
    // Overlay Mode: Left Column (Input Space A & C), Right Column (Output Space B & D)
    // Left: Input Space
    if (isSingleA)
    {
      renderSingleReconQuad(
        0, 0, halfW, halfH,
        'A', 'Single Frame in Dataset A',
        `Frame #${queryIdx + 1} / ${totalA}`,
        '[A]', '#38bdf8',
        singleFrameA, imgWA, imgHA,
        metricA,
        'Wheel to scrub frames'
      );
    }
    else
    {
      renderReconThumbnailGallery(
        ctx,
        0, 0, halfW, halfH,
        'A', 'Training Input [A] (k-NN)',
        '[A]', '#38bdf8',
        ptsA, neighborsList,
        imgWA, imgHA,
        imageReconKnnScrollY
      );
    }

    renderSingleReconQuad(
      0, halfH, halfW, halfH,
      'C', 'Query Input [C]',
      `Query #${queryIdx + 1} / ${totalQ}`,
      '[C]', '#fbbf24',
      frameC, imgWC, imgHC,
      '',
      'Wheel to scrub queries'
    );

    // Right: Output Space
    if (isSingleB)
    {
      renderSingleReconQuad(
        halfW, 0, halfW, halfH,
        'B', 'Single Frame in Dataset B',
        `Target #${queryIdx + 1} / ${totalB}`,
        '[B]', '#4ade80',
        singleFrameB, imgWB, imgHB,
        metricB,
        'Wheel to scrub frames'
      );
    }
    else
    {
      renderReconThumbnailGallery(
        ctx,
        halfW, 0, halfW, halfH,
        'B', 'Training Output Target [B]',
        '[B]', '#4ade80',
        ptsB, neighborsList,
        imgWB, imgHB,
        imageReconKnnScrollY
      );
    }

    let rmsStr = '';
    if (isSingleB && rmsBD >= 0.0)
    {
      rmsStr = `RMS vs [B]: ${rmsBD.toFixed(4)}`;
    }
    else if (frameB && frameD && frameB.length === frameD.length)
    {
      let diffSumSq = 0.0;
      for (let d = 0; d < frameB.length; d++)
      {
        const diff = frameB[d] - frameD[d];
        diffSumSq += diff * diff;
      }
      const rms = Math.sqrt(diffSumSq / frameB.length);
      rmsStr = `RMS Diff: ${rms.toFixed(4)}`;
    }

    renderSingleReconQuad(
      halfW, halfH, halfW, halfH,
      'D', 'Reconstructed Output [D]',
      `Reconstructed #${queryIdx + 1}`,
      '[D]', '#c084fc',
      frameD, imgWD, imgHD,
      rmsStr
    );

    // Vertical Divider
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.moveTo(halfW, 0);
    ctx.lineTo(halfW, H);
    ctx.stroke();
  }

  ctx.restore();
}

const QUALITY_STOPS_FALLBACK = [
  [34, 197, 94],  [52, 211, 106], [74, 222, 128],
  [110, 231, 150],[153, 240, 176],[200, 247, 200],
  [254, 249, 195],[254, 240, 138],[253, 224, 71],
  [250, 204, 21], [245, 158, 11], [239, 115, 22],
  [239, 68, 68],  [220, 38, 38],  [185, 28, 28],
  [153, 27, 27]
];

/**
 * Evaluate quality color from ramp for a given variance value.
 *
 * @param {number} val - Metric variance value
 * @param {number} mn  - Minimum metric value
 * @param {number} mx  - Maximum metric value
 * @returns {string} CSS rgb(...) color string
 */
function evalQualityRamp(val, mn, mx)
{
  if (typeof window !== 'undefined' && typeof window.getColorFromRamp === 'function')
  {
    const stops = window.QUALITY_STOPS || QUALITY_STOPS_FALLBACK;
    return window.getColorFromRamp(val, mn, mx, stops);
  }
  if (typeof getColorFromRamp === 'function')
  {
    const stops = (typeof QUALITY_STOPS !== 'undefined')
      ? QUALITY_STOPS : QUALITY_STOPS_FALLBACK;
    return getColorFromRamp(val, mn, mx, stops);
  }

  if (typeof val !== 'number' || isNaN(val) || !isFinite(val)) val = mn || 0;
  if (typeof mn !== 'number' || isNaN(mn) || !isFinite(mn)) mn = 0;
  if (typeof mx !== 'number' || isNaN(mx) || !isFinite(mx) || mx <= mn) mx = mn + 1.0;

  const t = Math.max(0, Math.min(1, (val - mn) / (mx - mn)));
  const stops = QUALITY_STOPS_FALLBACK;
  const n = stops.length - 1;
  const fi = Math.max(0, Math.min(n, t * n));
  const lo = Math.floor(fi);
  const hi = Math.min(lo + 1, n);
  const f = fi - lo;
  const c0 = stops[lo];
  const c1 = stops[hi];
  const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
  const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
  const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
  return `rgb(${r},${g},${b})`;
}

/**
 * Retrieve reconstruction variance or frame quality dataset for coloring.
 *
 * @returns {Object|null} Object containing {arr, min, max, total, metricName} or null.
 */
function getReconVarianceData()
{
  const activeSlot = (typeof datasetSlots !== 'undefined' &&
                      typeof activeDatasetSlot !== 'undefined')
    ? datasetSlots[activeDatasetSlot] : null;

  // 1. Direct reconstruction variance on active slot
  if (activeSlot && activeSlot.reconVariance && activeSlot.reconVariance.length > 0)
  {
    return {
      arr: activeSlot.reconVariance,
      min: activeSlot.reconVarianceMin,
      max: activeSlot.reconVarianceMax,
      metricName: 'Recon Variance',
      total: activeSlot.reconVariance.length
    };
  }

  // 2. Output Slot D reconstruction variance
  const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;
  if (slotD && slotD.reconVariance && slotD.reconVariance.length > 0)
  {
    return {
      arr: slotD.reconVariance,
      min: slotD.reconVarianceMin,
      max: slotD.reconVarianceMax,
      metricName: 'Recon Variance',
      total: slotD.reconVariance.length
    };
  }

  // 3. Query Slot C reconstruction variance
  const slotC = (typeof datasetSlots !== 'undefined') ? datasetSlots['C'] : null;
  if (slotC && slotC.reconVariance && slotC.reconVariance.length > 0)
  {
    return {
      arr: slotC.reconVariance,
      min: slotC.reconVarianceMin,
      max: slotC.reconVarianceMax,
      metricName: 'Recon Variance',
      total: slotC.reconVariance.length
    };
  }

  // 4. Fallback: k-NN neighbor distance
  if (activeSlot && activeSlot.reconKthDist && activeSlot.reconKthDist.length > 0)
  {
    return {
      arr: activeSlot.reconKthDist,
      min: activeSlot.reconKthDistMin,
      max: activeSlot.reconKthDistMax,
      metricName: 'k-NN Dist',
      total: activeSlot.reconKthDist.length
    };
  }
  if (slotD && slotD.reconKthDist && slotD.reconKthDist.length > 0)
  {
    return {
      arr: slotD.reconKthDist,
      min: slotD.reconKthDistMin,
      max: slotD.reconKthDistMax,
      metricName: 'k-NN Dist',
      total: slotD.reconKthDist.length
    };
  }
  if (slotC && slotC.reconKthDist && slotC.reconKthDist.length > 0)
  {
    return {
      arr: slotC.reconKthDist,
      min: slotC.reconKthDistMin,
      max: slotC.reconKthDistMax,
      metricName: 'k-NN Dist',
      total: slotC.reconKthDist.length
    };
  }

  // 5. Fallback: Per-frame intrinsic image variance for loaded dataset frames
  const frames = (activeSlot && activeSlot.benchmarkDataset &&
                  activeSlot.benchmarkDataset.length > 0)
    ? activeSlot.benchmarkDataset
    : (typeof benchmarkDataset !== 'undefined' && benchmarkDataset &&
       benchmarkDataset.length > 0) ? benchmarkDataset : null;

  if (frames && frames.length > 0 && frames[0] &&
      (frames[0].length > 0 || typeof frames[0] === 'object'))
  {
    const targetObj = activeSlot || window;
    if (!targetObj._cachedFrameVar || targetObj._cachedFrameVar.length !== frames.length)
    {
      const N = frames.length;
      const dim = frames[0].length || 1024;
      const varArr = new Float64Array(N);
      let vMin = Infinity, vMax = -Infinity;
      for (let i = 0; i < N; i++)
      {
        const fr = frames[i];
        if (!fr || fr.length === 0) continue;
        let sum = 0.0, sumSq = 0.0;
        for (let d = 0; d < dim; d++)
        {
          const v = fr[d] || 0.0;
          sum += v;
          sumSq += v * v;
        }
        const mean = sum / dim;
        const v = Math.max(0.0, (sumSq / dim) - (mean * mean));
        varArr[i] = v;
        if (v < vMin) vMin = v;
        if (v > vMax) vMax = v;
      }
      targetObj._cachedFrameVar = varArr;
      targetObj._cachedFrameVarMin = (vMin === Infinity) ? 0.0 : vMin;
      targetObj._cachedFrameVarMax = (vMax === -Infinity) ? 1.0 : vMax;
    }
    return {
      arr: targetObj._cachedFrameVar,
      min: targetObj._cachedFrameVarMin,
      max: targetObj._cachedFrameVarMax,
      metricName: 'Frame Var',
      total: targetObj._cachedFrameVar.length
    };
  }

  return null;
}

/**
 * Update the reconstruction quality colored bar and current frame single color box.
 *
 * @param {number} [curFrameIdx] - Current frame index (0-based)
 * @param {number} [totalFramesCount] - Total count of frames in dataset
 */
function updateReconQualityBar(curFrameIdx, totalFramesCount)
{
  const canvas = document.getElementById('canvasImgReconQualityBar');
  const box = document.getElementById('boxImgFrameReconQuality');
  const slider = document.getElementById('sliderImgFrame');
  if (!canvas || !box)
  {
    return;
  }

  const total = (typeof totalFramesCount === 'number' && totalFramesCount > 0)
    ? totalFramesCount
    : (benchmarkDataset && benchmarkDataset.length > 0)
      ? benchmarkDataset.length
      : (typeof totalFrames !== 'undefined' ? totalFrames : 0);

  const curIdx = (typeof curFrameIdx === 'number' && curFrameIdx >= 0)
    ? curFrameIdx
    : (typeof inspectedImageFrameIdx !== 'undefined' && inspectedImageFrameIdx >= 0)
      ? inspectedImageFrameIdx
      : Math.max(0, total - 1);

  const qualData = getReconVarianceData();

  // Attach interactive clicking and scrubbing on quality bar
  if (!canvas._hasQualityScrubListeners)
  {
    canvas._hasQualityScrubListeners = true;
    let isScrubbing = false;

    const scrubHandler = (e) => {
      const cRect = canvas.getBoundingClientRect();
      const numFrames = (typeof totalFrames !== 'undefined' && totalFrames > 0)
        ? totalFrames
        : (benchmarkDataset ? benchmarkDataset.length : 0);
      if (numFrames <= 0 || cRect.width <= 0) return;

      const clickX = Math.max(0, Math.min(cRect.width, e.clientX - cRect.left));
      const targetIdx = Math.max(0, Math.min(numFrames - 1,
        Math.round((clickX / cRect.width) * (numFrames - 1))));

      if (typeof inspectedImageFrameIdx !== 'undefined')
      {
        inspectedImageFrameIdx = targetIdx;
      }
      const s = document.getElementById('sliderImgFrame');
      if (s) s.value = targetIdx;
      const inp = document.getElementById('inputImgFrame');
      if (inp) inp.value = targetIdx + 1;

      if (typeof updateTelemetryUI === 'function')
      {
        updateTelemetryUI();
      }
      if (typeof draw === 'function')
      {
        draw();
      }
    };

    canvas.addEventListener('mousedown', (e) => {
      isScrubbing = true;
      scrubHandler(e);
    });
    window.addEventListener('mousemove', (e) => {
      if (isScrubbing) scrubHandler(e);
    });
    window.addEventListener('mouseup', () => {
      isScrubbing = false;
    });
  }

  // Exact vertical alignment with slider width
  const sliderRect = slider ? slider.getBoundingClientRect() : null;
  const canvasRect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const displayW = Math.max(10, Math.floor(
    (sliderRect && sliderRect.width > 0) ? sliderRect.width : (canvasRect.width || 300)
  ));
  const displayH = 6;
  const canvasW = Math.floor(displayW * dpr);
  const canvasH = Math.floor(displayH * dpr);

  if (canvas.width !== canvasW || canvas.height !== canvasH)
  {
    canvas.width = canvasW;
    canvas.height = canvasH;
  }
  canvas.style.width = '100%';
  canvas.style.height = `${displayH}px`;

  const ctx = canvas.getContext('2d');
  if (!ctx)
  {
    return;
  }

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, displayW, displayH);

  if (!qualData || !qualData.arr || qualData.arr.length === 0 || total <= 0)
  {
    // Draw subtle gradient guide rather than empty gray
    const grad = ctx.createLinearGradient(0, 0, displayW, 0);
    grad.addColorStop(0, '#22c55e');
    grad.addColorStop(0.5, '#eab308');
    grad.addColorStop(1, '#ef4444');
    ctx.fillStyle = grad;
    ctx.globalAlpha = 0.25;
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function')
    {
      ctx.roundRect(0, 0, displayW, displayH, 3);
    }
    else
    {
      ctx.rect(0, 0, displayW, displayH);
    }
    ctx.fill();
    ctx.globalAlpha = 1.0;

    box.style.backgroundColor = '#334155';
    box.style.borderColor = '#64748b';
    box.title = 'Reconstruction quality (variance) not computed yet';
    canvas.title = 'Reconstruction quality (Green = Best, Red = Poor) - Not computed yet';
    ctx.restore();
    return;
  }

  const arr = qualData.arr;
  let mn = qualData.min;
  let mx = qualData.max;
  const isUniform = (mx <= mn + 1e-9);
  if (isUniform)
  {
    mx = mn + 1.0;
  }
  const N = arr.length;
  const metricName = qualData.metricName || 'Variance';

  ctx.beginPath();
  if (typeof ctx.roundRect === 'function')
  {
    ctx.roundRect(0, 0, displayW, displayH, 3);
  }
  else
  {
    ctx.rect(0, 0, displayW, displayH);
  }
  ctx.clip();

  // Render colored columns across frames
  for (let px = 0; px < displayW; px++)
  {
    const fIdx = Math.min(N - 1, Math.floor((px / displayW) * N));
    const val = arr[fIdx];
    const col = (isUniform && mn === 0.0)
      ? 'rgb(34,197,94)'
      : evalQualityRamp(val, mn, mx);
    ctx.fillStyle = col;
    ctx.fillRect(px, 0, 1, displayH);
  }

  // Draw current frame cursor needle
  if (curIdx >= 0 && curIdx < total)
  {
    const cursorX = Math.round((curIdx / Math.max(1, total - 1)) * (displayW - 1));

    // Dark outline
    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
    ctx.fillRect(Math.max(0, cursorX - 1), 0, 3, displayH);

    // Bright white center tick
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(cursorX, 0, 1, displayH);
  }

  ctx.restore();

  // Update single color box for current frame
  const curVal = (curIdx >= 0 && curIdx < N) ? arr[curIdx] : null;
  if (typeof curVal === 'number' && !isNaN(curVal))
  {
    const curColor = (isUniform && mn === 0.0)
      ? 'rgb(34,197,94)'
      : evalQualityRamp(curVal, mn, mx);
    box.style.backgroundColor = curColor;
    box.style.borderColor = '#ffffff';

    const t = isUniform ? 0.0 : Math.max(0, Math.min(1, (curVal - mn) / (mx - mn)));
    const rating = (curVal === 0.0)
      ? 'Perfect (Zero Var)'
      : (t < 0.33)
        ? 'Good (Low Var)'
        : (t < 0.66)
          ? 'Moderate'
          : 'Poor (High Var)';
    box.title = `Frame #${curIdx + 1} Quality (${metricName}): ` +
                `${curVal.toFixed(4)} [${rating}]`;
    canvas.title = `Reconstruction Quality Bar (${metricName})\n` +
                   `Green = Best (${mn.toFixed(4)}), Red = Poor (${mx.toFixed(4)})\n` +
                   `Current Frame #${curIdx + 1}: ${curVal.toFixed(4)} (${rating}) ` +
                   `— Click or drag to jump`;
  }
  else
  {
    box.style.backgroundColor = '#334155';
    box.style.borderColor = '#475569';
    box.title = `Frame #${curIdx + 1}: Quality metric not available`;
  }
}

if (typeof window !== 'undefined')
{
  window.updateImageQuadDropdowns = updateImageQuadDropdowns;
  window.getImageViewTitle = getImageViewTitle;
  window.isReconstructionImageMode = isReconstructionImageMode;
  window.getReconstructionKnnNeighbors = getReconstructionKnnNeighbors;
  window.renderReconThumbnailGallery = renderReconThumbnailGallery;
  window.handleImageModeClick = handleImageModeClick;
  window.drawReconImage4PanelView = drawReconImage4PanelView;
  window.getReconVarianceData = getReconVarianceData;
  window.updateReconQualityBar = updateReconQualityBar;
}
