/**
 * @file desktop_bridge.js
 * @brief Bridge communication layer between the GRIC simulator frontend and
 *        the native C gric-server desktop backend.
 */

// eslint-disable-next-line no-unused-vars
const DesktopBridge = (function () {
  'use strict';

  let _isDesktop = false;
  let _serverInfo = null;
  let _activeJobId = null;
  let _pollTimer = null;

  let _baseUrl = '';

  function _getUrls(endpoint) {
    if (_baseUrl) return [`${_baseUrl}${endpoint}`];
    if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {
      return [endpoint, `http://127.0.0.1:8080${endpoint}`, `http://localhost:8080${endpoint}`];
    }
    return [`http://127.0.0.1:8080${endpoint}`, `http://localhost:8080${endpoint}`];
  }

  async function _fetchApi(endpoint, options = {}) {
    const urls = _getUrls(endpoint);
    let lastErr = null;
    for (const u of urls) {
      try {
        const resp = await fetch(u, options);
        if (resp.ok || resp.status === 400 || resp.status === 500) {
          if (u.startsWith('http') && !_baseUrl) {
            const parsed = new URL(u);
            _baseUrl = `${parsed.protocol}//${parsed.host}`;
          }
          return resp;
        }
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr || new Error(`Failed to fetch ${endpoint}`);
  }

  /**
   * Probe if the native C gric-server is active on the current host.
   *
   * @returns {Promise<Object|null>} Server info or null if purely static web mode.
   */
  async function probe() {
    try {
      const resp = await _fetchApi('/api/info', {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        cache: 'no-store'
      });

      if (resp && resp.ok) {
        const data = await resp.json();
        if (data && data.status === 'ok' && data.mode === 'desktop') {
          _isDesktop = true;
          _serverInfo = data;
          console.log('[DesktopBridge] Native C gric-server connected. Workspace:', data.cwd);
          return _serverInfo;
        }
      }
    } catch (err) {
      /* Not running against gric-server; purely static web mode */
    }

    _isDesktop = false;
    _serverInfo = null;
    return null;
  }

  /**
   * Check if the client is running on a mobile phone / handheld device.
   * @returns {boolean}
   */
  function isMobileDevice() {
    if (typeof navigator === 'undefined') return false;
    const ua = navigator.userAgent || navigator.vendor || (window.opera || '');
    const mobileRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile/i;
    if (mobileRegex.test(ua)) return true;
    if (navigator.userAgentData && navigator.userAgentData.mobile) return true;
    if (typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(max-width: 768px) and (pointer: coarse)').matches) {
      return true;
    }
    return false;
  }

  /**
   * Check if Native CLI mode is supported on the current device and backend.
   * @returns {boolean} True only if on a desktop device connected to gric-server with binaries.
   */
  function isNativeSupported() {
    if (isMobileDevice()) return false;
    if (!_isDesktop || !_serverInfo) return false;
    if (_serverInfo.binaries && _serverInfo.binaries['gric-cluster'] === false) return false;
    return true;
  }

  function isAvailable() {
    return _isDesktop;
  }

  function getServerInfo() {
    return _serverInfo;
  }

  function getWorkspaceDir() {
    return _serverInfo ? _serverInfo.cwd : '';
  }

  /**
   * List files in the active desktop workspace directory.
   */
  async function listFiles() {
    if (!_isDesktop) return [];

    try {
      const resp = await _fetchApi('/api/files', { cache: 'no-store' });
      if (resp.ok) {
        const data = await resp.json();
        return (data && data.files) ? data.files : [];
      }
    } catch (err) {
      console.error('[DesktopBridge] Error listing workspace files:', err);
    }
    return [];
  }

  /**
   * Read file content from the desktop workspace.
   */
  async function readFile(relPath) {
    if (!_isDesktop) throw new Error('Desktop backend not connected.');

    const url = `/api/file/read?path=${encodeURIComponent(relPath)}`;
    const resp = await _fetchApi(url, { cache: 'no-store' });
    if (!resp.ok) {
      throw new Error(`Failed to read file ${relPath} (HTTP ${resp.status})`);
    }
    return await resp.text();
  }

  /**
   * Write text content to a file in the desktop workspace.
   */
  async function writeFile(relPath, content) {
    if (!_isDesktop) throw new Error('Desktop backend not connected.');

    const resp = await _fetchApi('/api/file/write', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        path: relPath,
        content: content
      })
    });

    if (!resp.ok) {
      throw new Error(`Failed to write file ${relPath} (HTTP ${resp.status})`);
    }
    return await resp.json();
  }

  /**
   * Run a native CLI process (gric-cluster or gric-knn) via gric-server.
   */
  async function runCliJob(options) {
    if (!_isDesktop) throw new Error('Desktop backend not connected.');

    let opts = options;
    if (typeof options === 'string') {
      opts = {
        cmd: options,
        args: arguments[1] || [],
        onOutput: arguments[2],
        onFinish: arguments[3]
      };
    } else if (!opts) {
      opts = {};
    }

    const cmd = opts.cmd || 'gric-cluster';
    const args = opts.args || [];
    const onOutput = opts.onOutput || function () {};
    const onTelemetry = opts.onTelemetry || function () {};
    const onFinish = opts.onFinish || function () {};

    if (_activeJobId) {
      await killActiveJob();
    }

    const runResp = await _fetchApi('/api/cli/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cmd: cmd,
        args: args
      })
    });

    if (!runResp.ok) {
      const errData = await runResp.json().catch(() => ({}));
      throw new Error(errData.error || `Failed to start CLI process (HTTP ${runResp.status})`);
    }

    const runData = await runResp.json();
    _activeJobId = runData.job_id;

    let currentOffset = 0;

    async function pollStatus() {
      if (!_activeJobId) return;

      try {
        const statUrl = `/api/cli/status?job_id=${encodeURIComponent(_activeJobId)}&offset=${currentOffset}`;
        const statResp = await _fetchApi(statUrl, { cache: 'no-store' });
        if (statResp.ok) {
          const statData = await statResp.json();

          if (statData.output && statData.output.length > 0) {
            onOutput(statData.output);
            currentOffset = statData.offset || (currentOffset + statData.output.length);
          }

          if (statData.telemetry) {
            onTelemetry(statData.telemetry);
          }

          if (!statData.active) {
            _activeJobId = null;
            if (_pollTimer) {
              clearInterval(_pollTimer);
              _pollTimer = null;
            }
            onFinish({
              exitCode: statData.exit_code,
              status: statData.status
            });
            return;
          }
        }
      } catch (err) {
        console.error('[DesktopBridge] Poll error:', err);
      }
    }

    _pollTimer = setInterval(pollStatus, 150);
    pollStatus();
  }

  /**
   * Abort / kill the currently running CLI job.
   */
  async function killActiveJob() {
    if (!_activeJobId) return;

    try {
      await _fetchApi('/api/cli/kill', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: _activeJobId })
      });
    } catch (err) {
      console.warn('[DesktopBridge] Kill request failed:', err);
    }

    _activeJobId = null;
    if (_pollTimer) {
      clearInterval(_pollTimer);
      _pollTimer = null;
    }
  }

  /**
   * List live ImageStreamIO shared memory streams (/dev/shm/*.im.shm)
   */
  async function listShmStreams() {
    if (!_isDesktop) return [];
    try {
      const resp = await _fetchApi('/api/shm/list', { cache: 'no-store' });
      if (resp.ok) {
        const data = await resp.json();
        return (data && data.streams) ? data.streams : [];
      }
    } catch (err) {
      console.error('[DesktopBridge] Error listing SHM streams:', err);
    }
    return [];
  }

  /**
   * Fetch live telemetry from a shared memory file
   */
  async function getShmTelemetry(path) {
    if (!_isDesktop) return null;
    try {
      const resp = await _fetchApi(`/api/shm/telemetry?path=${encodeURIComponent(path)}`, {
        cache: 'no-store'
      });
      if (resp.ok) {
        const data = await resp.json();
        return (data && data.telemetry) ? data.telemetry : null;
      }
    } catch (err) {
      console.error('[DesktopBridge] Error reading SHM telemetry:', err);
    }
    return null;
  }

  /**
   * Stage dataset coordinates to a workspace file on disk.
   */
  async function stageDatasetFile(datasetName, dataset) {
    const safeName = datasetName.replace(/[^a-zA-Z0-9_-]/g, '_');
    const fileName = `${safeName}.txt`;
    let content = '';
    for (let i = 0; i < dataset.length; i++) {
      const pt = dataset[i];
      if (Array.isArray(pt)) {
        content += pt.map(v => Number(v).toFixed(6)).join(' ') + '\n';
      } else if (pt && typeof pt === 'object') {
        if (typeof pt.z === 'number' && !isNaN(pt.z)) {
          content += `${Number(pt.x || 0).toFixed(6)} ${Number(pt.y || 0).toFixed(6)} ${Number(pt.z || 0).toFixed(6)}\n`;
        } else {
          content += `${Number(pt.x || 0).toFixed(6)} ${Number(pt.y || 0).toFixed(6)}\n`;
        }
      }
    }
    await writeFile(fileName, content);
    return fileName;
  }

  /**
   * Export in-memory WASM clustering state to the desktop workspace.
   */
  async function exportClusterDat(datasetName, artifacts) {
    const safeName = datasetName.replace(/[^a-zA-Z0-9_-]/g, '_');
    const folderName = `${safeName}.clusterdat`;

    for (const [fname, content] of Object.entries(artifacts)) {
      if (typeof content === 'string') {
        await writeFile(`${folderName}/${fname}`, content);
      }
    }
    return folderName;
  }

  /**
   * Load and parse a completed .clusterdat/ directory from workspace into visualizer format.
   *
   * @param {string} clusterDir Relative path to clusterdat directory.
   */
  async function parseClusterDatDir(clusterDir) {
    const cleanDir = clusterDir.replace(/\/+$/, '');
    const results = {
      anchors: [],
      dcc: [],
      membership: [],
      logText: ''
    };

    // 1. Try reading centroids.txt or anchors.txt
    try {
      let text = '';
      try {
        text = await readFile(`${cleanDir}/anchors.txt`);
      } catch (e) {
        text = await readFile(`${cleanDir}/centroids.txt`);
      }

      const lines = text.split(/\r?\n/);
      let cId = 0;
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const tokens = trimmed.split(/[,\s\t]+/).filter(t => t.length > 0);
        if (tokens.length >= 2) {
          const x = parseFloat(tokens[0]);
          const y = parseFloat(tokens[1]);
          const z = tokens.length >= 3 ? parseFloat(tokens[2]) : 0.0;
          if (!isNaN(x) && !isNaN(y)) {
            results.anchors.push({
              id: cId++,
              x: x,
              y: y,
              z: isNaN(z) ? 0.0 : z,
              members: 0
            });
          }
        }
      }
    } catch (err) {
      console.warn('[DesktopBridge] Could not read centroids/anchors:', err);
    }

    // 2. Try reading dcc.txt
    try {
      const dccText = await readFile(`${cleanDir}/dcc.txt`);
      const lines = dccText.split(/\r?\n/);
      const mat = [];
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const row = trimmed.split(/[,\s\t]+/)
          .filter(t => t.length > 0)
          .map(t => parseFloat(t))
          .filter(v => !isNaN(v));
        if (row.length > 0) {
          mat.push(row);
        }
      }
      results.dcc = mat;
    } catch (err) {
      /* DCC optional */
    }

    // 3. Try reading cluster_counts.txt (or fallback to frame_membership.txt)
    try {
      const countsText = await readFile(`${cleanDir}/cluster_counts.txt`);
      const lines = countsText.split(/\r?\n/);
      let cIdx = 0;
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const match = trimmed.match(/Cluster\s+(\d+):\s*(\d+)\s*frames/i) || trimmed.match(/(\d+)\s+(\d+)/);
        if (match) {
          const idx = parseInt(match[1], 10);
          const count = parseInt(match[2], 10);
          if (idx >= 0 && idx < results.anchors.length) {
            results.anchors[idx].members = count;
          }
        } else {
          const count = parseInt(trimmed, 10);
          if (!isNaN(count) && cIdx < results.anchors.length) {
            results.anchors[cIdx++].members = count;
          }
        }
      }
    } catch (e) {
      /* Fallback to frame_membership.txt if cluster_counts.txt is not available */
      try {
        const memText = await readFile(`${cleanDir}/frame_membership.txt`);
        const lines = memText.split(/\r?\n/);
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith('#')) continue;
          const val = parseInt(trimmed, 10);
          if (!isNaN(val)) {
            results.membership.push(val);
            if (val >= 0 && val < results.anchors.length) {
              results.anchors[val].members = (results.anchors[val].members || 0) + 1;
            }
          }
        }
      } catch (err) {
        /* Membership optional */
      }
    }

    // 4. Try reading cluster_run.log and extract stats
    results.stats = {
      frames: 0,
      clusters: 0,
      timeMs: 0,
      totalDists: 0,
      sampleDists: 0,
      interclusterDists: 0,
      pruned: 0,
      rssKb: 0,
      predHits: 0,
      predAttempts: 0
    };
    try {
      results.logText = await readFile(`${cleanDir}/cluster_run.log`);
      if (results.logText) {
        const logLines = results.logText.split(/\r?\n/);
        for (const l of logLines) {
          const trimmed = l.trim();
          if (trimmed.startsWith('STATS_FRAMES:')) {
            results.stats.frames = parseInt(trimmed.substring(13).trim(), 10) || 0;
          } else if (trimmed.startsWith('TIME_CLUSTERING_MS:')) {
            results.stats.timeMs = parseFloat(trimmed.substring(19).trim()) || 0;
          } else if (trimmed.startsWith('STATS_CLUSTERS:')) {
            results.stats.clusters = parseInt(trimmed.substring(15).trim(), 10) || 0;
          } else if (trimmed.startsWith('STATS_DISTS:')) {
            results.stats.totalDists = parseInt(trimmed.substring(12).trim(), 10) || 0;
          } else if (trimmed.startsWith('STATS_DISTS_SAMPLE:')) {
            results.stats.sampleDists = parseInt(trimmed.substring(19).trim(), 10) || 0;
          } else if (trimmed.startsWith('STATS_DISTS_INTERCLUSTER:')) {
            results.stats.interclusterDists = parseInt(trimmed.substring(25).trim(), 10) || 0;
          } else if (trimmed.startsWith('STATS_PRUNED:')) {
            results.stats.pruned = parseInt(trimmed.substring(13).trim(), 10) || 0;
          } else if (trimmed.startsWith('STATS_MAX_RSS_KB:')) {
            results.stats.rssKb = parseInt(trimmed.substring(17).trim(), 10) || 0;
          } else if (trimmed.startsWith('STATS_PRED_HITS:')) {
            results.stats.predHits = parseInt(trimmed.substring(16).trim(), 10) || 0;
          } else if (trimmed.startsWith('STATS_PRED_ATTEMPTS:')) {
            results.stats.predAttempts = parseInt(trimmed.substring(20).trim(), 10) || 0;
          }
        }
      }
    } catch (err) {
      /* Log optional */
    }

    // 5. Try reading frame_evals.txt (per-frame distance evaluations)
    results.evals = [];
    results.assignments = [];
    try {
      const evalsText = await readFile(`${cleanDir}/frame_evals.txt`);
      if (evalsText) {
        const lines = evalsText.split(/\r?\n/);
        for (let i = 0; i < lines.length; i++) {
          const l = lines[i].trim();
          if (!l || l.startsWith('#')) continue;
          const tokens = l.split(/[,\s\t]+/).filter(t => t.length > 0);
          if (tokens.length >= 4) {
            const fIdx = parseInt(tokens[0], 10);
            const cId = parseInt(tokens[1], 10);
            const dist = parseFloat(tokens[2]);
            const match = parseInt(tokens[3], 10) === 1;

            if (!isNaN(fIdx) && !isNaN(cId) && !isNaN(dist)) {
              if (!results.evals[fIdx]) {
                results.evals[fIdx] = [];
              }
              results.evals[fIdx].push({
                clusterId: cId,
                dist: dist,
                match: match
              });
              if (match) {
                results.assignments[fIdx] = cId;
              }
            }
          }
        }
      }
    } catch (err) {
      /* frame_evals optional */
    }

    // 6. Fallback reading frame_membership.txt for cluster assignments if evals missing
    if (results.assignments.length === 0) {
      try {
        const memText = await readFile(`${cleanDir}/frame_membership.txt`);
        if (memText) {
          const lines = memText.split(/\r?\n/);
          for (let i = 0; i < lines.length; i++) {
            const l = lines[i].trim();
            if (!l || l.startsWith('#')) continue;
            const tokens = l.split(/[,\s\t]+/).filter(t => t.length > 0);
            if (tokens.length >= 2) {
              const fIdx = parseInt(tokens[0], 10);
              const cId = parseInt(tokens[1], 10);
              const dist = tokens.length >= 3 ? parseFloat(tokens[2]) : 0.0;
              if (!isNaN(fIdx) && !isNaN(cId)) {
                results.assignments[fIdx] = cId;
                if (!results.evals[fIdx]) {
                  results.evals[fIdx] = [{
                    clusterId: cId,
                    dist: isNaN(dist) ? 0.0 : dist,
                    match: true
                  }];
                }
              }
            }
          }
        }
      } catch (err) {
        /* frame_membership optional */
      }
    }

    return results;
  }

  /**
   * Initialize and ensure the persistent gric_cli tmux session is active.
   */
  async function initCliSession() {
    if (!_isDesktop) return null;
    try {
      const resp = await _fetchApi('/api/cli/session/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      if (resp.ok) {
        return await resp.json();
      }
    } catch (err) {
      console.warn('[DesktopBridge] initCliSession failed:', err);
    }
    return null;
  }

  /**
   * Terminate the persistent gric_cli tmux session.
   */
  async function stopCliSession() {
    if (!_isDesktop) return null;
    try {
      const resp = await _fetchApi('/api/cli/session/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      if (resp.ok) {
        return await resp.json();
      }
    } catch (err) {
      console.warn('[DesktopBridge] stopCliSession failed:', err);
    }
    return null;
  }

  /**
   * Parse k-NN search telemetry from gric-knn stdout log text.
   *
   * @param {string} logText Raw or ANSI-colored log output from gric-knn.
   * @return {Object} Structured telemetry object.
   */
  function parseKnnTelemetryLog(logText) {
    const clean = (logText || '').replace(/\x1b\[[0-9;]*m/g, '');
    const telem = {
      totalQueries: 0,
      framedistCalls: 0,
      level0SuperClustersPruned: 0,
      level1ClustersPruned: 0,
      level2AnchorsPruned: 0,
      level3AnnularPruned: 0,
      temporalPruned: 0,
      totalCandidatesConsidered: 0,
      timeSearchMs: 0.0
    };

    const mTotal = clean.match(/Total Query Frames:\s+(\d+)/);
    if (mTotal) telem.totalQueries = parseInt(mTotal[1], 10);

    const mCalls = clean.match(/Framedist Computations:\s+(\d+)/);
    if (mCalls) telem.framedistCalls = parseInt(mCalls[1], 10);

    const mL0 = clean.match(/Level 0 Super-Clusters:\s+(\d+)/);
    if (mL0) telem.level0SuperClustersPruned = parseInt(mL0[1], 10);

    const mL1 = clean.match(/Level 1 Clusters Pruned:\s+(\d+)/);
    if (mL1) telem.level1ClustersPruned = parseInt(mL1[1], 10);

    const mL2 = clean.match(/Level 2 Anchors Pruned:\s+(\d+)/);
    if (mL2) telem.level2AnchorsPruned = parseInt(mL2[1], 10);

    const mL3 = clean.match(/Level 3 Annular Pruned:\s+(\d+)/);
    if (mL3) telem.level3AnnularPruned = parseInt(mL3[1], 10);

    const mTemp = clean.match(/Temporal Exclusions:\s+(\d+)/);
    if (mTemp) telem.temporalPruned = parseInt(mTemp[1], 10);

    const mTime = clean.match(/Search Wall Time:\s+([\d.]+)\s*ms/);
    if (mTime) telem.timeSearchMs = parseFloat(mTime[1]);

    return telem;
  }

  /**
   * Load and parse knn_results.txt from workspace into visualizer format.
   *
   * @param {string} clusterDir Relative path to clusterdat directory.
   * @param {number} k Number of nearest neighbors per frame.
   */
  async function readKnnResults(clusterDir, k) {
    const cleanDir = clusterDir.replace(/\/+$/, '');
    try {
      const text = await readFile(`${cleanDir}/knn_results.txt`);
      if (!text) return null;

      const lines = text.split(/\r?\n/);
      let N = 0;
      const indices = [];
      const distances = [];

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line || line.startsWith('#')) continue;
        const parts = line.split(/\s+/);
        if (parts.length < 1 + 2 * k) continue;
        N++;
        for (let r = 0; r < k; r++) {
          indices.push(parseInt(parts[1 + 2 * r], 10));
          distances.push(parseFloat(parts[1 + 2 * r + 1]));
        }
      }

      if (N === 0) return null;

      return {
        totalFrames: N,
        k: k,
        indices: new Int32Array(indices),
        distances: new Float64Array(distances)
      };
    } catch (err) {
      console.warn('[DesktopBridge] Could not read knn_results.txt:', err);
      return null;
    }
  }

  return {
    probe,
    isAvailable,
    isMobileDevice,
    isNativeSupported,
    getServerInfo,
    getWorkspaceDir,
    listFiles,
    readFile,
    writeFile,
    initCliSession,
    stopCliSession,
    getCliSessionStatus,
    runCliJob,
    killActiveJob,
    listShmStreams,
    getShmTelemetry,
    stageDatasetFile,
    exportClusterDat,
    parseClusterDatDir,
    parseKnnTelemetryLog,
    readKnnResults
  };
})();
