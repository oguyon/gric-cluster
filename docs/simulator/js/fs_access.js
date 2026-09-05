/**
 * @file fs_access.js
 * @brief W3C File System Access API abstraction, in-browser ZIP generator,
 *        and unified Data Structure tracking for the GRIC Interactive Simulator.
 */

// eslint-disable-next-line no-unused-vars
const WebFs = (function () {
  'use strict';

  let _dirHandle = null;
  let _dirName = '';

  function isSupported() {
    return typeof window !== 'undefined' &&
           typeof window.showDirectoryPicker === 'function';
  }

  async function openDirectory() {
    if (!isSupported()) {
      alert(
        'File System Access API is not supported in this browser.\n' +
        'Please use Chrome, Edge, Chromium, or the Desktop application (gric-gui).'
      );
      return null;
    }

    try {
      _dirHandle = await window.showDirectoryPicker({
        mode: 'readwrite',
        startIn: 'documents'
      });

      _dirName = _dirHandle.name;
      return {
        name: _dirName,
        handle: _dirHandle
      };
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('[WebFs] Error selecting directory:', err);
      }
      return null;
    }
  }

  function isOpen() {
    return _dirHandle !== null;
  }

  function getDirectoryName() {
    return _dirName || '';
  }

  async function listFiles(subDirHandle = null) {
    const targetHandle = subDirHandle || _dirHandle;
    if (!targetHandle) return [];

    const files = [];
    try {
      // eslint-disable-next-line no-undef
      for await (const [name, handle] of targetHandle.entries()) {
        const isDir = handle.kind === 'directory';
        const dot = name.lastIndexOf('.');
        const ext = dot >= 0 ? name.slice(dot + 1).toLowerCase() : '';

        let isRelevant = false;
        if (isDir) {
          isRelevant = true;
        } else if (['txt', 'csv', 'dat', 'fits', 'log', 'json'].includes(ext)) {
          isRelevant = true;
        }

        if (isRelevant) {
          let size = 0;
          if (!isDir) {
            try {
              const f = await handle.getFile();
              size = f.size;
            } catch (e) {
              /* ignore */
            }
          }
          files.push({
            name: name,
            is_dir: isDir,
            isDir: isDir,
            ext: ext,
            size: size,
            handle: handle
          });
        }
      }
    } catch (err) {
      console.error('[WebFs] Error listing directory entries:', err);
    }

    files.sort((a, b) => {
      if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    return files;
  }

  async function readFile(fileName) {
    if (!_dirHandle) {
      throw new Error('No local workspace directory opened.');
    }
    const fileHandle = await _dirHandle.getFileHandle(fileName);
    const file = await fileHandle.getFile();
    return await file.text();
  }

  async function writeFile(fileName, content) {
    if (!_dirHandle) {
      throw new Error('No local workspace directory opened.');
    }
    const fileHandle = await _dirHandle.getFileHandle(fileName, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(content);
    await writable.close();
  }

  async function exportClusterDat(datasetName, artifacts) {
    if (!_dirHandle) {
      throw new Error('No local workspace directory opened.');
    }

    const safeName = datasetName.replace(/[^a-zA-Z0-9_-]/g, '_');
    const folderName = `${safeName}.clusterdat`;

    const subDirHandle = await _dirHandle.getDirectoryHandle(folderName, { create: true });

    for (const [fname, content] of Object.entries(artifacts)) {
      if (typeof content === 'string') {
        const fh = await subDirHandle.getFileHandle(fname, { create: true });
        const wr = await fh.createWritable();
        await wr.write(content);
        await wr.close();
      }
    }

    return folderName;
  }

  return {
    isSupported,
    openDirectory,
    isOpen,
    getDirectoryName,
    listFiles,
    readFile,
    writeFile,
    exportClusterDat
  };
})();

// eslint-disable-next-line no-unused-vars
const DataManager = (function () {
  'use strict';

  const crcTable = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    crcTable[i] = c;
  }

  function crc32(bytes) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < bytes.length; i++) {
      c = crcTable[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
    }
    return (c ^ 0xFFFFFFFF) >>> 0;
  }

  function buildZip(files) {
    const enc = new TextEncoder();
    const localChunks = [];
    const cdChunks = [];
    let offset = 0;

    for (const [filename, content] of Object.entries(files)) {
      const data = typeof content === 'string' ? enc.encode(content) : content;
      const nameBytes = enc.encode(filename);
      const crc = crc32(data);

      const lh = new Uint8Array(30 + nameBytes.length + data.length);
      const lv = new DataView(lh.buffer);
      lv.setUint32(0, 0x04034b50, true);
      lv.setUint16(4, 20, true);
      lv.setUint16(6, 0x0800, true); // UTF-8
      lv.setUint16(8, 0, true);      // Store
      lv.setUint16(10, 0, true);
      lv.setUint16(12, 0, true);
      lv.setUint32(14, crc, true);
      lv.setUint32(18, data.length, true);
      lv.setUint32(22, data.length, true);
      lv.setUint16(26, nameBytes.length, true);
      lv.setUint16(28, 0, true);
      lh.set(nameBytes, 30);
      lh.set(data, 30 + nameBytes.length);
      localChunks.push(lh);

      const cd = new Uint8Array(46 + nameBytes.length);
      const cv = new DataView(cd.buffer);
      cv.setUint32(0, 0x02014b50, true);
      cv.setUint16(4, 20, true);
      cv.setUint16(6, 20, true);
      cv.setUint16(8, 0x0800, true);
      cv.setUint16(10, 0, true);
      cv.setUint16(12, 0, true);
      cv.setUint16(14, 0, true);
      cv.setUint32(16, crc, true);
      cv.setUint32(20, data.length, true);
      cv.setUint32(24, data.length, true);
      cv.setUint16(28, nameBytes.length, true);
      cv.setUint16(30, 0, true);
      cv.setUint16(32, 0, true);
      cv.setUint16(34, 0, true);
      cv.setUint16(36, 0, true);
      cv.setUint32(38, 0, true);
      cv.setUint32(42, offset, true);
      cd.set(nameBytes, 46);
      cdChunks.push(cd);

      offset += lh.length;
    }

    const cdOffset = offset;
    let cdSize = 0;
    for (const c of cdChunks) cdSize += c.length;

    const numEntries = Object.keys(files).length;
    const eocd = new Uint8Array(22);
    const ev = new DataView(eocd.buffer);
    ev.setUint32(0, 0x06054b50, true);
    ev.setUint16(4, 0, true);
    ev.setUint16(6, 0, true);
    ev.setUint16(8, numEntries, true);
    ev.setUint16(10, numEntries, true);
    ev.setUint32(12, cdSize, true);
    ev.setUint32(16, cdOffset, true);
    ev.setUint16(20, 0, true);

    const totalLen = cdOffset + cdSize + 22;
    const out = new Uint8Array(totalLen);
    let pos = 0;
    for (const c of localChunks) { out.set(c, pos); pos += c.length; }
    for (const c of cdChunks) { out.set(c, pos); pos += c.length; }
    out.set(eocd, pos);
    return out;
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function downloadTextFile(filename, content) {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    downloadBlob(blob, filename);
  }

  function downloadBinaryFile(filename, uint8Bytes) {
    const blob = new Blob([uint8Bytes], { type: 'application/octet-stream' });
    downloadBlob(blob, filename);
  }

  /**
   * Constructs a self-describing 64-byte GRIC binary buffer with payload.
   */
  function createGricBinaryFile(fileType, dataType, dims, typedArray, comment = '') {
    const enc = new TextEncoder();
    const commentBytes = comment ? enc.encode(comment) : new Uint8Array(0);
    const headerBytes = 64 + commentBytes.length;
    const dataBytes = typedArray.byteLength;
    const totalBytes = headerBytes + dataBytes;

    const buffer = new ArrayBuffer(totalBytes);
    const view = new DataView(buffer);
    const u8 = new Uint8Array(buffer);

    // Magic "GRIC"
    u8[0] = 0x47; u8[1] = 0x52; u8[2] = 0x49; u8[3] = 0x43;
    // Version 1
    view.setUint8(4, 1);
    // File Type (1=ANCHORS, 2=DCC, 3=MEMBERSHIP, 4=COUNTS, 5=EVALS, 6=COORDS, 0=GENERIC)
    view.setUint8(5, fileType);
    // Data Type (1=FLOAT32, 2=FLOAT64, 3=UINT32, 4=INT32)
    view.setUint8(6, dataType);
    // Endianness (1 = Little-Endian)
    view.setUint8(7, 1);
    // Header bytes (uint16)
    view.setUint16(8, headerBytes, true);
    // ndim (uint16)
    view.setUint16(10, dims.length, true);
    // flags (uint32) - Row-Major (0x0001)
    view.setUint32(12, 0x0001, true);
    // num_elements (uint64)
    view.setBigUint64(16, BigInt(typedArray.length), true);
    // data_bytes (uint64)
    view.setBigUint64(24, BigInt(dataBytes), true);
    // dims[4] (4 x uint64)
    for (let d = 0; d < 4; d++) {
      const dimVal = (d < dims.length) ? BigInt(dims[d]) : 0n;
      view.setBigUint64(32 + d * 8, dimVal, true);
    }
    // Comment
    if (commentBytes.length > 0) {
      u8.set(commentBytes, 64);
    }
    // Payload Data
    u8.set(
      new Uint8Array(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength),
      headerBytes
    );

    return u8;
  }

  function generateCurrentDataStructures() {
    const dsName = (typeof currentBenchmark !== 'undefined' && currentBenchmark)
      ? currentBenchmark : 'custom_dataset';

    const numClust = (typeof clusters !== 'undefined') ? clusters.length : 0;
    const numPast = (typeof pastSamples !== 'undefined') ? pastSamples.length : 0;
    const dim = (typeof currentDim !== 'undefined') ? currentDim : 2;
    const radius = (typeof rlim !== 'undefined') ? rlim : 0.1;

    // 1. anchors.bin (FLOAT32 [numClust, dim])
    const f32Anchors = new Float32Array(numClust * dim);
    let anchorsPreview = `# GRIC Binary File: anchors.bin [FLOAT32, ${numClust} x ${dim}]\n` +
                         `# Format: 64-byte self-describing GRIC header\n` +
                         `# ID X Y ${dim >= 3 ? 'Z ' : ''}MEMBERS RADIUS STATUS\n`;
    if (typeof clusters !== 'undefined' && clusters.length > 0) {
      clusters.forEach((c, idx) => {
        if (c.anchor && (c.anchor.length === dim || dim > 3)) {
          for (let d = 0; d < dim; d++) {
            f32Anchors[idx * dim + d] = Number(c.anchor[d] || 0);
          }
        } else {
          f32Anchors[idx * dim + 0] = Number(c.x || 0);
          f32Anchors[idx * dim + 1] = Number(c.y || 0);
          if (dim >= 3) {
            f32Anchors[idx * dim + 2] = Number(c.z || 0);
          }
        }
        const x = Number(c.x || 0).toFixed(6);
        const y = Number(c.y || 0).toFixed(6);
        const z = Number(c.z || 0).toFixed(6);
        const m = c.members || 0;
        const r = Number(c.radius || radius).toFixed(6);
        const st = c.pruned ? 'PRUNED' : 'ACTIVE';
        anchorsPreview += `${c.id} ${x} ${y} ${dim >= 3 ? z + ' ' : ''}${m} ${r} ${st}\n`;
      });
    }
    const binAnchors = createGricBinaryFile(
      1, 1, [numClust, dim], f32Anchors, 'Cluster centroid coordinates'
    );

    // 2. dcc.bin (FLOAT32 [numClust, numClust])
    const f32Dcc = new Float32Array(numClust * numClust);
    let dccPreview = `# GRIC Binary File: dcc.bin [FLOAT32, ${numClust} x ${numClust}]\n` +
                     `# Format: 64-byte self-describing GRIC header\n` +
                     `# Cluster-to-Cluster Metric Distance Matrix D_cc\n`;
    if (typeof dcc !== 'undefined' && dcc.length > 0) {
      for (let i = 0; i < numClust; i++) {
        for (let j = 0; j < numClust; j++) {
          const val = (dcc[i] && dcc[i][j] !== undefined) ? Number(dcc[i][j]) : 0;
          f32Dcc[i * numClust + j] = val;
        }
        if (dcc[i]) {
          dccPreview += dcc[i].map(v => Number(v).toFixed(6)).join(' ') + '\n';
        }
      }
    }
    const binDcc = createGricBinaryFile(
      2, 1, [numClust, numClust], f32Dcc, 'Cluster distance matrix D_cc'
    );

    // 3. frame_membership.bin (UINT32 [numPast])
    const u32Mem = new Uint32Array(numPast);
    let memPreview = `# GRIC Binary File: frame_membership.bin [UINT32, ${numPast} frames]\n` +
                     `# Format: 64-byte self-describing GRIC header\n` +
                     `# FrameIdx -> ClusterID\n`;
    if (typeof assignmentHistory !== 'undefined' && assignmentHistory.length > 0) {
      assignmentHistory.forEach((cid, idx) => {
        u32Mem[idx] = cid >= 0 ? cid : 0;
        memPreview += `${idx} ${cid}\n`;
      });
    } else if (typeof pastSamples !== 'undefined' && pastSamples.length > 0) {
      pastSamples.forEach((p, idx) => {
        const cid = p.clusterId !== undefined ? p.clusterId : -1;
        u32Mem[idx] = cid >= 0 ? cid : 0;
        memPreview += `${idx} ${cid}\n`;
      });
    }
    const binMem = createGricBinaryFile(
      3, 3, [numPast], u32Mem, 'Frame membership assignments'
    );

    // 4. cluster_counts.bin (UINT32 [numClust])
    const u32Counts = new Uint32Array(numClust);
    let countsPreview = `# GRIC Binary File: cluster_counts.bin [UINT32, ${numClust} clusters]\n` +
                        `# Format: 64-byte self-describing GRIC header\n` +
                        `# ClusterID -> Member Count\n`;
    if (typeof clusters !== 'undefined' && clusters.length > 0) {
      clusters.forEach((c, idx) => {
        u32Counts[idx] = c.members || 0;
        countsPreview += `${c.id} ${c.members || 0}\n`;
      });
    }
    const binCounts = createGricBinaryFile(
      4, 3, [numClust], u32Counts, 'Cluster member counts'
    );

    // 5. transition_matrix.txt
    let tmText = `# GRIC Markov State Transition Matrix\n`;
    if (typeof transitionCounts !== 'undefined' && transitionCounts.length > 0) {
      transitionCounts.forEach(row => {
        tmText += row.join(' ') + '\n';
      });
    }

    // 6. input_samples.bin (FLOAT32 [numPast, dim])
    const f32Samples = new Float32Array(numPast * dim);
    let samplesPreview = `# GRIC Binary File: input_samples.bin [FLOAT32, ${numPast} x ${dim}]\n` +
                         `# Format: 64-byte self-describing GRIC header\n` +
                         `# X Y ${dim >= 3 ? 'Z' : ''}\n`;
    const sourceSamples = (typeof benchmarkDataset !== 'undefined' && benchmarkDataset.length > 0)
      ? benchmarkDataset : (typeof pastSamples !== 'undefined' ? pastSamples : []);

    if (sourceSamples && sourceSamples.length > 0) {
      sourceSamples.forEach((p, idx) => {
        if (Array.isArray(p) || ArrayBuffer.isView(p)) {
          for (let d = 0; d < dim; d++) {
            f32Samples[idx * dim + d] = Number(p[d] || 0);
          }
          if (idx < 50) {
            samplesPreview += Array.from(p.slice(0, Math.min(dim, 3)))
              .map(v => Number(v).toFixed(6)).join(' ') + (dim > 3 ? ' ...\n' : '\n');
          }
        } else if (p.coords && p.coords.length >= dim) {
          for (let d = 0; d < dim; d++) {
            f32Samples[idx * dim + d] = Number(p.coords[d] || 0);
          }
          if (idx < 50) {
            samplesPreview += Array.from(p.coords.slice(0, Math.min(dim, 3)))
              .map(v => Number(v).toFixed(6)).join(' ') + (dim > 3 ? ' ...\n' : '\n');
          }
        } else {
          f32Samples[idx * dim + 0] = Number(p.x || 0);
          f32Samples[idx * dim + 1] = Number(p.y || 0);
          if (dim >= 3) {
            f32Samples[idx * dim + 2] = Number(p.z || 0);
            samplesPreview += `${Number(p.x||0).toFixed(6)} ${Number(p.y||0).toFixed(6)} ` +
                              `${Number(p.z||0).toFixed(6)}\n`;
          } else {
            samplesPreview += `${Number(p.x||0).toFixed(6)} ${Number(p.y||0).toFixed(6)}\n`;
          }
        }
      });
    }
    const binSamples = createGricBinaryFile(
      6, 1, [numPast, dim], f32Samples, 'Input dataset coordinates'
    );

    // 7. knn_results.txt (k-NN search graph and metric distances)
    let knnText = `# GRIC k-Nearest Neighbor (k-NN) Graph & Diagnostics\n`;
    let isKnnReady = false;
    let knnSummaryBadge = 'Pending (Run k-NN)';

    if (typeof knnResults !== 'undefined' && knnResults) {
      const k = knnResults.k || (typeof knnK !== 'undefined' ? knnK : 8);
      const totalQueries = knnResults.totalFrames ||
        (knnResults.indices ? Math.floor(knnResults.indices.length / k) : 0) ||
        (knnResults.queries ? knnResults.queries.length : 0);

      if (totalQueries > 0) {
        isKnnReady = true;
        knnSummaryBadge = `${totalQueries} queries (k=${k})`;
        const telem = knnResults.telemetry || {};
        const eff = (typeof telem.pruneEfficiencyPct === 'number')
          ? telem.pruneEfficiencyPct.toFixed(1)
          : (typeof telem.efficiencyPct === 'number'
             ? telem.efficiencyPct.toFixed(1)
             : '0.0');
        const spd = (typeof telem.speedup === 'number')
          ? telem.speedup.toFixed(2)
          : '1.00';

        knnText += `# Target k: ${k}\n`;
        knnText += `# Total Queries: ${totalQueries}\n`;
        knnText += `# Pruning Efficiency: ${eff}%\n`;
        knnText += `# Speedup Factor: ${spd}x\n`;
        knnText += `# Format: QueryFrameIdx NeighborIdx:Distance NeighborIdx:Distance ...\n\n`;

        if (knnResults.indices && knnResults.distances) {
          const maxRows = Math.min(totalQueries, 5000);
          for (let q = 0; q < maxRows; q++) {
            let rowStr = `${q}`;
            for (let r = 0; r < k; r++) {
              const nIdx = knnResults.indices[q * k + r];
              const dist = knnResults.distances[q * k + r];
              if (nIdx !== undefined && nIdx >= 0) {
                rowStr += ` ${nIdx}:${Number(dist || 0).toFixed(4)}`;
              }
            }
            knnText += `${rowStr}\n`;
          }
          if (totalQueries > maxRows) {
            knnText += `# ... [${totalQueries - maxRows} additional queries omitted from preview]\n`;
          }
        } else if (knnResults.queries) {
          knnResults.queries.forEach((q, qIdx) => {
            const nList = q.neighbors
              ? q.neighbors.map(n => `${n.index}:${n.dist.toFixed(4)}`).join(' ')
              : '';
            knnText += `Query ${qIdx} (distCalls=${q.distCalls || 0}): ${nList}\n`;
          });
        }
      }
    }

    let binKnnIdx = null;
    let binKnnDst = null;
    if (isKnnReady && knnResults.indices && knnResults.distances) {
      const kVal = knnResults.k || (typeof knnK !== 'undefined' ? knnK : 8);
      const totalQ = knnResults.totalFrames || Math.floor(knnResults.indices.length / kVal);
      const totalElems = totalQ * kVal;
      const u32Idx = new Uint32Array(totalElems);
      const f32Dst = new Float32Array(totalElems);
      for (let i = 0; i < totalElems; i++) {
        u32Idx[i] = (knnResults.indices[i] >= 0) ? knnResults.indices[i] : 0;
        f32Dst[i] = (knnResults.distances[i] !== undefined)
          ? Number(knnResults.distances[i]) : 0;
      }
      binKnnIdx = createGricBinaryFile(
        0, 3, [totalQ, kVal], u32Idx, 'k-NN neighbor indices [N x k]'
      );
      binKnnDst = createGricBinaryFile(
        0, 1, [totalQ, kVal], f32Dst, 'k-NN metric distances [N x k]'
      );
    }

    if (!isKnnReady) {
      knnText += `# Status: k-NN graph not yet computed for this session.\n` +
                 `# Run Pass 2 (k-NN) to generate nearest-neighbor graph & diagnostics.\n`;
    }

    // 8. cluster_run.log
    const nowIso = new Date().toISOString();
    const modeStr = typeof targetMode !== 'undefined' ? targetMode : 'greedy';
    const pruneStr = typeof pruneMode !== 'undefined' ? pruneMode : '4P';
    const framesCnt = typeof totalFrames !== 'undefined' ? totalFrames : numPast;
    const dfcCalls = typeof distSampleCluster !== 'undefined' ? distSampleCluster : 0;
    const dccCalls = typeof distClusterCluster !== 'undefined' ? distClusterCluster : 0;
    const prunedCnt = typeof pruneCount4P !== 'undefined' ? pruneCount4P : 0;
    const predHits = typeof predHitCount !== 'undefined' ? predHitCount : 0;
    const engStr = typeof engineMode !== 'undefined' ? engineMode : 'wasm';

    let logText = `=================================================================\n` +
                  `GRIC CLUSTER EXECUTION SUMMARY\n` +
                  `=================================================================\n` +
                  `Timestamp:           ${nowIso}\n` +
                  `Dataset:             ${dsName}\n` +
                  `Dimensions:          ${dim}D\n` +
                  `Receptive Radius:    ${radius.toFixed(4)}\n` +
                  `Clustering Mode:     ${modeStr}\n` +
                  `Pruning Strategy:    ${pruneStr}\n` +
                  `Total Clusters:      ${numClust}\n` +
                  `Total Frames:        ${framesCnt}\n` +
                  `Distance D_FC Calls: ${dfcCalls}\n` +
                  `Distance D_CC Calls: ${dccCalls}\n` +
                  `Pruned by Metric:    ${prunedCnt}\n` +
                  `Predicted Hits:      ${predHits}\n` +
                  `Engine Mode:         ${engStr}\n` +
                  `=================================================================\n`;

    // 9. metadata.json
    const metadataObj = {
      timestamp: nowIso,
      dataset: dsName,
      dimensions: dim,
      rlim: radius,
      clusters_count: numClust,
      frames_count: (typeof totalFrames !== 'undefined') ? totalFrames : numPast,
      clustering_mode: (typeof targetMode !== 'undefined') ? targetMode : 'greedy',
      prune_mode: (typeof pruneMode !== 'undefined') ? pruneMode : '4P',
      engine: (typeof engineMode !== 'undefined') ? engineMode : 'wasm',
      metrics: {
        dist_sample_cluster: (typeof distSampleCluster !== 'undefined') ? distSampleCluster : 0,
        dist_cluster_cluster: (typeof distClusterCluster !== 'undefined') ? distClusterCluster : 0,
        prune_count_4p: (typeof pruneCount4P !== 'undefined') ? pruneCount4P : 0,
        pred_hit_count: (typeof predHitCount !== 'undefined') ? predHitCount : 0
      }
    };
    const metadataJson = JSON.stringify(metadataObj, null, 2);

    const enc = new TextEncoder();
    const list = [
      {
        id: 'anchors',
        filename: 'anchors.bin',
        category: 'Cluster Anchors',
        icon: '📊',
        badge: `${numClust} anchors`,
        desc: 'Centroid coordinates in 64-byte self-describing GRIC float32 binary format.',
        content: anchorsPreview,
        binaryBytes: binAnchors,
        size: binAnchors.length,
        ready: numClust > 0
      },
      {
        id: 'dcc',
        filename: 'dcc.bin',
        category: 'Distance Matrix',
        icon: '📐',
        badge: `${numClust}×${numClust} matrix`,
        desc: 'Symmetric cluster-to-cluster distance matrix D_cc in GRIC float32 binary format.',
        content: dccPreview,
        binaryBytes: binDcc,
        size: binDcc.length,
        ready: numClust > 0
      },
      {
        id: 'membership',
        filename: 'frame_membership.bin',
        category: 'Assignments',
        icon: '🔗',
        badge: `${numPast} frames`,
        desc: 'Frame-to-cluster assignment array in uint32 binary format.',
        content: memPreview,
        binaryBytes: binMem,
        size: binMem.length,
        ready: numPast > 0
      },
      {
        id: 'counts',
        filename: 'cluster_counts.bin',
        category: 'Cluster Counts',
        icon: '📈',
        badge: `${numClust} clusters`,
        desc: 'Cluster member counts in uint32 binary format.',
        content: countsPreview,
        binaryBytes: binCounts,
        size: binCounts.length,
        ready: numClust > 0
      },
      {
        id: 'transitions',
        filename: 'transition_matrix.txt',
        category: 'Markov Transitions',
        icon: '🔁',
        badge: `${numClust}×${numClust} transitions`,
        desc: 'State-to-state temporal transition counts and probability graph.',
        content: tmText,
        size: enc.encode(tmText).length,
        ready: numClust > 0
      },
      {
        id: 'samples',
        filename: 'input_samples.bin',
        category: 'Input Coordinates',
        icon: '📥',
        badge: `${numPast} points`,
        desc: 'Raw floating-point point coordinates in float32 binary format.',
        content: samplesPreview,
        binaryBytes: binSamples,
        size: binSamples.length,
        ready: numPast > 0
      },
      {
        id: 'knn_indices',
        filename: 'knn_indices.bin',
        category: 'k-NN Indices',
        icon: '🎯',
        badge: knnSummaryBadge,
        desc: 'k-nearest neighbor index matrix in uint32 binary format.',
        content: knnText,
        binaryBytes: binKnnIdx,
        size: binKnnIdx ? binKnnIdx.length : 0,
        ready: isKnnReady
      },
      {
        id: 'knn_distances',
        filename: 'knn_distances.bin',
        category: 'k-NN Distances',
        icon: '📏',
        badge: knnSummaryBadge,
        desc: 'k-nearest neighbor metric distances in float32 binary format.',
        content: knnText,
        binaryBytes: binKnnDst,
        size: binKnnDst ? binKnnDst.length : 0,
        ready: isKnnReady
      },
      {
        id: 'knn_txt',
        filename: 'knn_results.txt',
        category: 'k-NN Table',
        icon: '📄',
        badge: knnSummaryBadge,
        desc: 'Human-readable ASCII table of query frame neighbor indices and distances.',
        content: knnText,
        size: enc.encode(knnText).length,
        ready: isKnnReady
      },
      {
        id: 'knn_quality',
        filename: 'knn_quality.bin',
        category: 'k-NN Quality',
        icon: '🎨',
        badge: 'Quality Metrics',
        desc: 'Per-query quality metrics: k-th NN dist '
            + 'and reconstruction variance [N x 2] '
            + 'float32.',
        content: '',
        size: 0,
        ready: false
      },
      {
        id: 'knn_recon',
        filename: 'knn_recon.bin',
        category: 'Reconstruction',
        icon: '🔮',
        badge: 'Recon Output',
        desc: 'Reconstructed output dataset D as [N x D]'
            + ' float32 binary.',
        content: '',
        size: 0,
        ready: false
      },
      {
        id: 'log',
        filename: 'cluster_run.log',
        category: 'Execution Log',
        icon: '📋',
        badge: 'Summary',
        desc: 'Clustering hyperparameters, speedup factors, and run diagnostics.',
        content: logText,
        size: enc.encode(logText).length,
        ready: true
      },
      {
        id: 'metadata',
        filename: 'metadata.json',
        category: 'JSON Metadata',
        icon: '⚙️',
        badge: 'Schema 1.0',
        desc: 'Machine-readable JSON specification of clustering parameters & state.',
        content: metadataJson,
        size: enc.encode(metadataJson).length,
        ready: true
      }
    ];

    const artifactsMap = {};
    list.forEach(item => {
      if (item.ready) {
        artifactsMap[item.filename] = item.binaryBytes || item.content;
      }
    });

    return {
      datasetName: dsName,
      structures: list,
      artifacts: artifactsMap
    };
  }

  async function saveAllToDisk() {
    const data = generateCurrentDataStructures();
    const dsName = data.datasetName;
    const artifacts = data.artifacts;

    if (Object.keys(artifacts).length === 0) {
      if (typeof showToast === 'function') showToast('No clustering data in memory to export');
      return;
    }

    if (typeof isDesktopBackend !== 'undefined' && isDesktopBackend) {
      try {
        const folder = await DesktopBridge.exportClusterDat(dsName, artifacts);
        if (typeof refreshWorkspaceFiles === 'function') await refreshWorkspaceFiles();
        if (typeof showToast === 'function') {
          showToast(`💾 Saved ${folder}/ to native workspace`);
        }
      } catch (err) {
        console.error('[DataManager] Desktop export error:', err);
        if (typeof showToast === 'function') showToast(`Export failed: ${err.message}`);
      }
    } else if (WebFs.isSupported() && WebFs.isOpen()) {
      try {
        const folder = await WebFs.exportClusterDat(dsName, artifacts);
        if (typeof refreshWorkspaceFiles === 'function') await refreshWorkspaceFiles();
        if (typeof showToast === 'function') {
          showToast(`💾 Saved ${folder}/ directly to local disk folder`);
        }
      } catch (err) {
        console.error('[DataManager] WebFs export error:', err);
        downloadZipBundle(dsName);
      }
    } else {
      downloadZipBundle(dsName);
    }
  }

  function downloadZipBundle(datasetName) {
    const data = generateCurrentDataStructures();
    const dsName = datasetName || data.datasetName;
    const artifacts = data.artifacts;

    const zipFiles = {};
    for (const [fname, content] of Object.entries(artifacts)) {
      zipFiles[`${dsName}.clusterdat/${fname}`] = content;
    }

    const zipBytes = buildZip(zipFiles);
    const zipBlob = new Blob([zipBytes], { type: 'application/zip' });
    downloadBlob(zipBlob, `${dsName}.clusterdat.zip`);
    if (typeof showToast === 'function') {
      showToast(`📦 Downloaded ${dsName}.clusterdat.zip`);
    }
  }

  return {
    crc32,
    buildZip,
    downloadBlob,
    downloadTextFile,
    downloadBinaryFile,
    createGricBinaryFile,
    generateCurrentDataStructures,
    saveAllToDisk,
    downloadZipBundle
  };
})();
