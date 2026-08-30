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
          _startHeartbeat();
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

  let _heartbeatTimer = null;

  /**
   * Start keepalive heartbeats and unload beacon listeners.
   */
  function _startHeartbeat() {
    if (_heartbeatTimer) return;
    _sendHeartbeat();
    _heartbeatTimer = setInterval(_sendHeartbeat, 2500);

    if (typeof window !== 'undefined') {
      window.addEventListener('pagehide', _sendLeaveBeacon);
      window.addEventListener('beforeunload', _sendLeaveBeacon);
    }
  }

  function _sendHeartbeat() {
    if (!_isDesktop) return;
    _fetchApi('/api/heartbeat', {
      method: 'POST',
      keepalive: true,
      headers: { 'Content-Type': 'application/json' }
    }).catch(() => {});
  }

  function _sendLeaveBeacon() {
    if (!_isDesktop || typeof navigator === 'undefined' || !navigator.sendBeacon) return;
    try {
      const url = (_baseUrl ? _baseUrl : '') + '/api/heartbeat/leave';
      navigator.sendBeacon(url, '');
    } catch (_) {}
  }

  /**
   * Request graceful remote shutdown of the native gric-server.
   */
  async function shutdownServer() {
    if (!_isDesktop) return false;
    try {
      if (_heartbeatTimer) {
        clearInterval(_heartbeatTimer);
        _heartbeatTimer = null;
      }
      await _fetchApi('/api/shutdown', { method: 'POST' });
      _isDesktop = false;
      _serverInfo = null;
      return true;
    } catch (_) {
      return false;
    }
  }

  /**
   * Check if the client is running on a mobile phone / handheld device.
   * @returns {boolean}
   */
  function isMobileDevice() {
    if (typeof navigator === 'undefined') return false;
    const ua = navigator.userAgent || navigator.vendor || (window.opera || '');
    const mobileRegex = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
    if (mobileRegex.test(ua)) return true;
    if (navigator.userAgentData && navigator.userAgentData.mobile) return true;
    return false;
  }

  /**
   * Check if Native CLI mode is supported on the current device and backend.
   * @returns {boolean} True only if connected to native gric-server with binaries.
   */
  function isNativeSupported() {
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
   * List files in the active desktop workspace directory or a subdirectory.
   */
  async function listFiles(subDir = '') {
    if (!_isDesktop) return [];

    try {
      const endpoint = subDir ? `/api/files?dir=${encodeURIComponent(subDir)}` : '/api/files';
      const resp = await _fetchApi(endpoint, { cache: 'no-store' });
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
   * Read binary ArrayBuffer from the desktop workspace.
   */
  async function readBinaryFile(relPath) {
    if (!_isDesktop) throw new Error('Desktop backend not connected.');

    const url = `/api/file/read?path=${encodeURIComponent(relPath)}`;
    const resp = await _fetchApi(url, { cache: 'no-store' });
    if (!resp.ok) {
      throw new Error(`Failed to read binary file ${relPath} (HTTP ${resp.status})`);
    }
    return await resp.arrayBuffer();
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

    const payload = {
      cmd: cmd,
      args: args
    };
    if (opts.streamFile) payload.stream_file = opts.streamFile;
    if (opts.streamName) payload.stream_name = opts.streamName;
    if (typeof opts.streamFps === 'number') payload.stream_fps = opts.streamFps;
    if (typeof opts.streamLoop === 'boolean') payload.stream_loop = opts.streamLoop;
    if (typeof opts.streamCnt2sync === 'boolean') payload.stream_cnt2sync = opts.streamCnt2sync;

    const runResp = await _fetchApi('/api/cli/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
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
        const encId = encodeURIComponent(_activeJobId);
        const statUrl = `/api/cli/status?job_id=${encId}&offset=${currentOffset}`;
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
      if (Array.isArray(pt) || ArrayBuffer.isView(pt) || (pt && typeof pt.length === 'number')) {
        content += Array.from(pt).map(v => Number(v).toFixed(6)).join(' ') + '\n';
      } else if (pt && typeof pt === 'object') {
        const px = Number(pt.x || 0).toFixed(6);
        const py = Number(pt.y || 0).toFixed(6);
        if (typeof pt.z === 'number' && !isNaN(pt.z)) {
          const pz = Number(pt.z).toFixed(6);
          content += `${px} ${py} ${pz}\n`;
        } else {
          content += `${px} ${py}\n`;
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
      } else if (content instanceof Uint8Array || ArrayBuffer.isView(content)) {
        await writeBinaryFile(
          `${folderName}/${fname}`,
          content.buffer || content,
          content.byteOffset || 0,
          content.byteLength || content.length
        );
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

    // 1. Try reading anchors.bin first, then fallback to anchors.txt / centroids.txt
    try {
      const buf = await readBinaryFile(`${cleanDir}/anchors.bin`);
      const view = new DataView(buf);
      const magic = String.fromCharCode(
        view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
      );
      if (magic === 'GRIC') {
        const headerSize = view.getUint16(8, true);
        const ndim = view.getUint16(10, true);
        const nclusters = Number(view.getBigUint64(32, true));
        const dims = ndim > 1 ? Number(view.getBigUint64(40, true)) : 1;
        const floats = new Float32Array(buf, headerSize, nclusters * dims);
        for (let i = 0; i < nclusters; i++) {
          const x = floats[i * dims + 0] || 0.0;
          const y = dims > 1 ? (floats[i * dims + 1] || 0.0) : 0.0;
          const z = dims > 2 ? (floats[i * dims + 2] || 0.0) : 0.0;
          const anchorBuf = new Float32Array(dims);
          for (let d = 0; d < dims; d++) {
            anchorBuf[d] = floats[i * dims + d];
          }
          results.anchors.push({
            id: i,
            x: x,
            y: y,
            z: isNaN(z) ? 0.0 : z,
            anchor: anchorBuf,
            members: 0
          });
        }
      }
    } catch (binErr) {
      /* Fallback to text anchors */
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
          if (tokens.length >= 1) {
            const x = parseFloat(tokens[0]);
            const y = tokens.length > 1 ? parseFloat(tokens[1]) : 0.0;
            const z = tokens.length > 2 ? parseFloat(tokens[2]) : 0.0;
            const anchorBuf = new Float32Array(tokens.length);
            for (let d = 0; d < tokens.length; d++) {
              anchorBuf[d] = parseFloat(tokens[d]);
            }
            if (!isNaN(x)) {
              results.anchors.push({
                id: cId++,
                x: x,
                y: isNaN(y) ? 0.0 : y,
                z: isNaN(z) ? 0.0 : z,
                anchor: anchorBuf,
                members: 0
              });
            }
          }
        }
      } catch (err) {
        console.warn('[DesktopBridge] Could not read centroids/anchors:', err);
      }
    }

    // 2. Try reading dcc.bin first, then fallback to dcc.txt
    try {
      const buf = await readBinaryFile(`${cleanDir}/dcc.bin`);
      const view = new DataView(buf);
      const magic = String.fromCharCode(
        view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
      );
      if (magic === 'GRIC') {
        const headerSize = view.getUint16(8, true);
        const rows = Number(view.getBigUint64(32, true));
        const cols = Number(view.getBigUint64(40, true));
        const floats = new Float32Array(buf, headerSize, rows * cols);
        const mat = [];
        for (let r = 0; r < rows; r++) {
          const row = [];
          for (let c = 0; c < cols; c++) {
            row.push(floats[r * cols + c]);
          }
          mat.push(row);
        }
        results.dcc = mat;
      }
    } catch (binErr) {
      /* Fallback to text dcc.txt */
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
    }

    // 3. Try reading cluster_counts.bin / cluster_counts.txt
    try {
      const buf = await readBinaryFile(`${cleanDir}/cluster_counts.bin`);
      const view = new DataView(buf);
      const magic = String.fromCharCode(
        view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
      );
      if (magic === 'GRIC') {
        const headerSize = view.getUint16(8, true);
        const nclusters = Number(view.getBigUint64(32, true));
        const counts = new Uint32Array(buf, headerSize, nclusters);
        for (let i = 0; i < nclusters && i < results.anchors.length; i++) {
          results.anchors[i].members = counts[i];
        }
      }
    } catch (binErr) {
      try {
        const countsText = await readFile(`${cleanDir}/cluster_counts.txt`);
        const lines = countsText.split(/\r?\n/);
        let cIdx = 0;
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith('#')) continue;
          const match = trimmed.match(/Cluster\s+(\d+):\s*(\d+)\s*frames/i) ||
                        trimmed.match(/(\d+)\s+(\d+)/);
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
        /* Fallback to frame_membership */
        try {
          const memBuf = await readBinaryFile(`${cleanDir}/frame_membership.bin`);
          const view = new DataView(memBuf);
          const magic = String.fromCharCode(
            view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
          );
          if (magic === 'GRIC') {
            const headerSize = view.getUint16(8, true);
            const nframes = Number(view.getBigUint64(32, true));
            const mems = new Uint32Array(memBuf, headerSize, nframes);
            for (let f = 0; f < nframes; f++) {
              const val = mems[f];
              results.membership.push(val);
              if (val < results.anchors.length) {
                results.anchors[val].members = (results.anchors[val].members || 0) + 1;
              }
            }
          }
        } catch (memErr) {
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
      dccPopulated: 0,
      dccPairsTotal: 0,
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
          } else if (trimmed.startsWith('STATS_DCC_POPULATED:')) {
            results.stats.dccPopulated = parseInt(trimmed.substring(20).trim(), 10) || 0;
          } else if (trimmed.startsWith('STATS_DCC_PAIRS_TOTAL:')) {
            results.stats.dccPairsTotal = parseInt(trimmed.substring(22).trim(), 10) || 0;
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
   * Check status of the persistent gric_cli tmux session.
   */
  async function getCliSessionStatus() {
    if (!_isDesktop) return { exists: false };
    try {
      const resp = await _fetchApi('/api/cli/session/status', { cache: 'no-store' });
      if (resp.ok) {
        return await resp.json();
      }
    } catch (err) {
      /* ignore */
    }
    return { exists: false };
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
      reciprocalReused: 0,
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

    const mRecip = clean.match(/Reciprocal Reused:\s+(\d+)/);
    if (mRecip) telem.reciprocalReused = parseInt(mRecip[1], 10);

    const mTemp = clean.match(/Temporal Exclusions:\s+(\d+)/);
    if (mTemp) telem.temporalPruned = parseInt(mTemp[1], 10);

    const mTime = clean.match(/Search Wall Time:\s+([\d.]+)\s*ms/);
    if (mTime) telem.timeSearchMs = parseFloat(mTime[1]);
    telem.timeComputeMs = telem.timeSearchMs;

    const mLoad = clean.match(/Loaded Pass 1 Model in\s+([\d.]+)\s*ms/);
    if (mLoad) telem.timeLoadMs = parseFloat(mLoad[1]);

    const mWrite = clean.match(/Output Write Time:\s+([\d.]+)\s*ms/);
    if (mWrite) telem.timeWriteMs = parseFloat(mWrite[1]);

    return telem;
  }

  /**
   * Load and parse k-NN results (binary or ASCII) from workspace into visualizer format.
   *
   * @param {string} clusterDir Relative path to clusterdat directory.
   * @param {number} k Number of nearest neighbors per frame.
   */
  async function readKnnResults(clusterDir, k) {
    const cleanDir = clusterDir.replace(/\/+$/, '');

    // 1. Try reading binary files first: knn_indices.bin & knn_distances.bin
    try {
      const idxBuf = await readBinaryFile(`${cleanDir}/knn_indices.bin`);
      const dstBuf = await readBinaryFile(`${cleanDir}/knn_distances.bin`);
      if (idxBuf && dstBuf) {
        const viewIdx = new DataView(idxBuf);
        const viewDst = new DataView(dstBuf);
        const magicIdx = String.fromCharCode(
          viewIdx.getUint8(0), viewIdx.getUint8(1), viewIdx.getUint8(2), viewIdx.getUint8(3)
        );
        const magicDst = String.fromCharCode(
          viewDst.getUint8(0), viewDst.getUint8(1), viewDst.getUint8(2), viewDst.getUint8(3)
        );

        if (magicIdx === 'GRIC' && magicDst === 'GRIC') {
          const hdrBytesIdx = viewIdx.getUint16(8, true);
          const hdrBytesDst = viewDst.getUint16(8, true);
          const N = Number(viewIdx.getBigUint64(32, true));
          const binK = Number(viewIdx.getBigUint64(40, true)) || k;
          const totalElems = N * binK;
          const rawIndices = new Uint32Array(idxBuf, hdrBytesIdx, totalElems);
          const rawDistances = new Float32Array(dstBuf, hdrBytesDst, totalElems);

          const indices = new Int32Array(totalElems);
          const distances = new Float64Array(totalElems);
          for (let i = 0; i < totalElems; i++) {
            indices[i] = rawIndices[i];
            distances[i] = rawDistances[i];
          }

          return {
            totalFrames: N,
            k: binK,
            indices: indices,
            distances: distances
          };
        }
      }
    } catch (e) {
      /* Fallback to text format */
    }

    // 2. Fallback to ASCII knn_results.txt
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
      console.warn('[DesktopBridge] Could not read knn results:', err);
      return null;
    }
  }

  /**
   * Parse gric-dimdensity JSON summary from stdout.
   *
   * @param {string} logText Raw or ANSI-colored output.
   * @return {Object|null} Parsed JSON summary or null.
   */
  function parseDimDensityLog(logText) {
    const clean = (logText || '')
      .replace(/\x1b\[[0-9;]*m/g, '');

    // Find the outermost JSON object block
    const start = clean.indexOf('{');
    if (start < 0) return null;

    let depth = 0;
    let end = -1;
    for (let i = start; i < clean.length; i++) {
      if (clean[i] === '{') depth++;
      else if (clean[i] === '}') depth--;
      if (depth === 0) {
        end = i + 1;
        break;
      }
    }
    if (end <= start) return null;

    try {
      return JSON.parse(clean.substring(start, end));
    } catch (e) {
      console.warn(
        '[DesktopBridge] parseDimDensityLog: ' +
        'JSON parse failed:', e
      );
      return null;
    }
  }

  /**
   * Read per-sample dim/density results from binary
   * dimdensity.bin in a clusterdat directory.
   *
   * @param {string} clusterDir Relative clusterdat path.
   * @return {Object|null} Per-sample result arrays.
   */
  async function readDimDensityResults(clusterDir) {
    const dir = clusterDir.replace(/\/+$/, '');

    // Try binary file first
    try {
      const buf = await readBinaryFile(
        `${dir}/dimdensity.bin`
      );
      if (!buf) return null;

      const view = new DataView(buf);
      const magic = String.fromCharCode(
        view.getUint8(0), view.getUint8(1),
        view.getUint8(2), view.getUint8(3)
      );

      if (magic === 'GRIC') {
        const hdrBytes = view.getUint16(8, true);
        const N = Number(
          view.getBigUint64(32, true)
        );
        const cols = Number(
          view.getBigUint64(40, true)
        ) || 4;

        const totalElems = N * cols;
        let raw;
        if (hdrBytes % 4 === 0) {
          raw = new Float32Array(
            buf, hdrBytes, totalElems
          );
        } else {
          raw = new Float32Array(
            buf.slice(
              hdrBytes, hdrBytes + totalElems * 4
            )
          );
        }

        const localDim = new Float32Array(N);
        const density = new Float32Array(N);
        const logDensity = new Float32Array(N);
        const rkDist = new Float32Array(N);

        for (let i = 0; i < N; i++) {
          localDim[i] = raw[i * cols + 0];
          density[i] = raw[i * cols + 1];
          logDensity[i] = raw[i * cols + 2];
          rkDist[i] = raw[i * cols + 3];
        }

        return {
          totalFrames: N,
          localDim: localDim,
          density: density,
          logDensity: logDensity,
          rkDist: rkDist
        };
      }
    } catch (e) {
      /* fallback to ASCII */
    }

    // Fallback: parse ASCII dimdensity.txt
    try {
      const text = await readFile(
        `${dir}/dimdensity.txt`
      );
      if (!text) return null;

      const lines = text.split(/\r?\n/);
      const dims = [];
      const dens = [];
      const logd = [];
      const rks = [];

      for (const line of lines) {
        const t = line.trim();
        if (!t || t.startsWith('#')) continue;
        const parts = t.split(/\s+/);
        if (parts.length < 5) continue;
        dims.push(parseFloat(parts[1]));
        dens.push(parseFloat(parts[2]));
        logd.push(parseFloat(parts[3]));
        rks.push(parseFloat(parts[4]));
      }

      if (dims.length === 0) return null;

      return {
        totalFrames: dims.length,
        localDim: new Float32Array(dims),
        density: new Float32Array(dens),
        logDensity: new Float32Array(logd),
        rkDist: new Float32Array(rks)
      };
    } catch (e) {
      console.warn(
        '[DesktopBridge] readDimDensityResults:',
        e
      );
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
    readBinaryFile,
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
    readKnnResults,
    parseDimDensityLog,
    readDimDensityResults,
    shutdownServer
  };
})();
