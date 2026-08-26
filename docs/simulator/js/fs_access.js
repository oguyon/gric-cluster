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

  function generateCurrentDataStructures() {
    const dsName = (typeof currentBenchmark !== 'undefined' && currentBenchmark)
      ? currentBenchmark : 'custom_dataset';

    const numClust = (typeof clusters !== 'undefined') ? clusters.length : 0;
    const numPast = (typeof pastSamples !== 'undefined') ? pastSamples.length : 0;
    const dim = (typeof currentDim !== 'undefined') ? currentDim : 2;
    const radius = (typeof rlim !== 'undefined') ? rlim : 0.1;

    // 1. centroids.txt
    let centroidsText = `# GRIC Cluster Centroids\n# ID X Y Z MEMBERS RADIUS STATUS\n`;
    if (typeof clusters !== 'undefined' && clusters.length > 0) {
      clusters.forEach(c => {
        const x = Number(c.x || 0).toFixed(6);
        const y = Number(c.y || 0).toFixed(6);
        const z = Number(c.z || 0).toFixed(6);
        const m = c.members || 0;
        const r = Number(c.radius || radius).toFixed(6);
        const st = c.pruned ? 'PRUNED' : 'ACTIVE';
        centroidsText += `${c.id} ${x} ${y} ${z} ${m} ${r} ${st}\n`;
      });
    }

    // 2. dcc.txt
    let dccText = `# GRIC Cluster-to-Cluster Distance Matrix D_cc\n`;
    if (typeof dcc !== 'undefined' && dcc.length > 0) {
      dcc.forEach(row => {
        dccText += row.map(v => Number(v).toFixed(6)).join(' ') + '\n';
      });
    }

    // 3. frame_membership.txt
    let memText = `# Frame Membership Assignments\n# FrameIdx -> ClusterID\n`;
    if (typeof assignmentHistory !== 'undefined' && assignmentHistory.length > 0) {
      assignmentHistory.forEach((cid, idx) => {
        memText += `${idx} ${cid}\n`;
      });
    } else if (typeof pastSamples !== 'undefined' && pastSamples.length > 0) {
      pastSamples.forEach((p, idx) => {
        memText += `${idx} ${p.clusterId !== undefined ? p.clusterId : -1}\n`;
      });
    }

    // 4. transition_matrix.txt
    let tmText = `# GRIC Markov State Transition Matrix\n`;
    if (typeof transitionCounts !== 'undefined' && transitionCounts.length > 0) {
      transitionCounts.forEach(row => {
        tmText += row.join(' ') + '\n';
      });
    }

    // 5. input_samples.txt
    let samplesText = `# GRIC Input Dataset Coordinates\n# X Y Z\n`;
    if (typeof pastSamples !== 'undefined' && pastSamples.length > 0) {
      pastSamples.forEach(p => {
        if (dim === 3) {
          samplesText += `${Number(p.x||0).toFixed(6)} ${Number(p.y||0).toFixed(6)} ` +
                         `${Number(p.z||0).toFixed(6)}\n`;
        } else {
          samplesText += `${Number(p.x||0).toFixed(6)} ${Number(p.y||0).toFixed(6)}\n`;
        }
      });
    }

    // 6. knn_results.txt (if k-NN graph is available)
    let knnText = `# GRIC k-Nearest Neighbor (k-NN) Graph & Diagnostics\n`;
    if (typeof knnResults !== 'undefined' && knnResults && knnResults.queries) {
      knnText += `# Target k: ${typeof knnK !== 'undefined' ? knnK : 8}\n`;
      knnText += `# Total Queries: ${knnResults.queries.length}\n`;
      knnText += `# Pruning Efficiency: ${knnResults.efficiencyPct || 0}%\n`;
      knnText += `# Speedup Factor: ${knnResults.speedup || 1.0}x\n\n`;
      knnResults.queries.forEach((q, qIdx) => {
        const nList = q.neighbors
          ? q.neighbors.map(n => `${n.index}:${n.dist.toFixed(4)}`).join(' ')
          : '';
        knnText += `Query ${qIdx} (distCalls=${q.distCalls || 0}): ${nList}\n`;
      });
    }

    // 7. cluster_run.log
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

    // 8. metadata.json
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
        id: 'centroids',
        filename: 'centroids.txt',
        category: 'Cluster Anchors',
        icon: '📊',
        badge: `${numClust} anchors`,
        desc: 'Centroid coordinates, member counts, radius, and state status.',
        content: centroidsText,
        size: enc.encode(centroidsText).length,
        ready: numClust > 0
      },
      {
        id: 'dcc',
        filename: 'dcc.txt',
        category: 'Distance Matrix',
        icon: '📐',
        badge: `${numClust}×${numClust} matrix`,
        desc: 'Symmetric cluster-to-cluster metric distance matrix D_cc.',
        content: dccText,
        size: enc.encode(dccText).length,
        ready: numClust > 0
      },
      {
        id: 'membership',
        filename: 'frame_membership.txt',
        category: 'Assignments',
        icon: '🔗',
        badge: `${numPast} frames`,
        desc: 'Sequence index to cluster ID assignment mapping.',
        content: memText,
        size: enc.encode(memText).length,
        ready: numPast > 0
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
        filename: 'input_samples.txt',
        category: 'Input Coordinates',
        icon: '📥',
        badge: `${numPast} points`,
        desc: 'Full raw floating-point coordinates for all ingested points.',
        content: samplesText,
        size: enc.encode(samplesText).length,
        ready: numPast > 0
      },
      {
        id: 'knn',
        filename: 'knn_results.txt',
        category: 'k-NN Search Graph',
        icon: '🎯',
        badge: (typeof knnResults !== 'undefined' && knnResults) ? 'Graph Ready' : 'Idle',
        desc: 'k-nearest neighbor query indices, distances, and speedup diagnostics.',
        content: knnText,
        size: enc.encode(knnText).length,
        ready: typeof knnResults !== 'undefined' && !!knnResults
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
        artifactsMap[item.filename] = item.content;
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
    generateCurrentDataStructures,
    saveAllToDisk,
    downloadZipBundle
  };
})();
