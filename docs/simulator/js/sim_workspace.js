/**
 * GRIC Simulator - sim_workspace.js
 * Workspace, Dual-Mode Controller, and Command Palette.
 */

    // =========================================================================
    //  WORKSPACE & DUAL-MODE CONTROLLER (WASM vs Native CLI)
    // =========================================================================

    let currentViewingFile = null;
    let currentDataFilesTab = 'structures';

    function updateStorageModeBanner() {
      const iconMode = document.getElementById('iconStorageMode');
      const titleMode = document.getElementById('txtStorageModeTitle');
      const subMode = document.getElementById('txtStorageModeSubtitle');
      if (!iconMode || !titleMode || !subMode) return;

      if (isDesktopBackend) {
        iconMode.textContent = '💻';
        titleMode.textContent = 'Native Desktop Workspace';
        subMode.textContent = workspacePath || 'Local Filesystem Active';
      } else if (WebFs.isOpen()) {
        iconMode.textContent = '📁';
        titleMode.textContent = 'Linked Local Folder';
        subMode.textContent = `Local Folder: ${WebFs.getDirectoryName()}`;
      } else {
        iconMode.textContent = '⚡';
        titleMode.textContent = 'WebAssembly Sandbox';
        subMode.textContent = 'In-Browser Memory (Save/Download to Disk)';
      }
    }

    function switchDataFilesTab(tab) {
      currentDataFilesTab = tab;
      const tabNavStruct = document.getElementById('tabNavDataStructures');
      const tabNavWs = document.getElementById('tabNavWorkspaceFiles');
      const paneStruct = document.getElementById('tabPaneDataStructures');
      const paneWs = document.getElementById('tabPaneWorkspaceFiles');

      if (tab === 'workspace') {
        if (tabNavStruct) tabNavStruct.classList.remove('active');
        if (tabNavWs) tabNavWs.classList.add('active');
        if (paneStruct) paneStruct.style.display = 'none';
        if (paneWs) paneWs.style.display = 'block';
        refreshWorkspaceFiles();
      } else {
        if (tabNavWs) tabNavWs.classList.remove('active');
        if (tabNavStruct) tabNavStruct.classList.add('active');
        if (paneWs) paneWs.style.display = 'none';
        if (paneStruct) paneStruct.style.display = 'block';
        renderDataStructuresUI();
      }
    }
    window.switchDataFilesTab = switchDataFilesTab;

    function openFileViewerModal(title, category, badge, sizeBytes, content) {
      const modal = document.getElementById('modalFileViewer');
      const lblTitle = document.getElementById('fileViewerTitle');
      const lblSub = document.getElementById('fileViewerSubtitle');
      const preContent = document.getElementById('fileViewerContent');

      currentViewingFile = { title, category, badge, sizeBytes, content };

      if (lblTitle) lblTitle.textContent = title;
      if (lblSub) {
        const sizeStr = (sizeBytes > 1048576)
          ? `${(sizeBytes / 1048576).toFixed(2)} MB`
          : `${(sizeBytes / 1024).toFixed(1)} KB`;
        lblSub.textContent = `${category} • ${badge} • ${sizeStr}`;
      }
      if (preContent) preContent.textContent = content || '(Empty file)';

      if (modal) modal.style.display = 'flex';
    }
    window.openFileViewerModal = openFileViewerModal;

    function closeFileViewerModal() {
      const modal = document.getElementById('modalFileViewer');
      if (modal) modal.style.display = 'none';
    }
    window.closeFileViewerModal = closeFileViewerModal;

    function renderDataStructuresUI() {
      const listEl = document.getElementById('dataStructuresList');
      const badgeCount = document.getElementById('badgeDataFilesCount');
      if (!listEl || typeof DataManager === 'undefined') return;

      const data = DataManager.generateCurrentDataStructures();
      const structures = data.structures;

      let readyCount = 0;
      structures.forEach(s => { if (s.ready) readyCount++; });

      if (badgeCount) {
        badgeCount.textContent = `${readyCount}/${structures.length} ready`;
      }

      let html = `
        <table class="data-files-table">
          <thead>
            <tr>
              <th style="width: 32%;">File</th>
              <th style="width: 22%;">Role</th>
              <th style="width: 20%;">Summary</th>
              <th style="width: 12%; text-align: right;">Size</th>
              <th style="width: 14%; text-align: center;">Actions</th>
            </tr>
          </thead>
          <tbody>
      `;

      structures.forEach((s, idx) => {
        const sizeStr = (s.size > 1048576)
          ? `${(s.size / 1048576).toFixed(2)} MB`
          : `${(s.size / 1024).toFixed(1)} KB`;

        const badgeBg = s.ready ? 'rgba(56, 189, 248, 0.15)' : 'rgba(148, 163, 184, 0.1)';
        const badgeColor = s.ready ? '#38bdf8' : '#94a3b8';
        const statusColor = s.ready ? '#4ade80' : '#64748b';

        html += `
          <tr class="df-row" data-idx="${idx}">
            <td>
              <span class="df-filename btn-view-struct" title="${s.desc}">
                <span>${s.icon}</span>
                <span style="color: ${s.ready ? '#f1f5f9' : '#64748b'};">${s.filename}</span>
              </span>
            </td>
            <td>
              <span class="badge-pill"
                    style="background: ${badgeBg}; color: ${badgeColor};
                           font-size: 0.62rem; padding: 1px 5px;">
                ${s.category}
              </span>
            </td>
            <td>
              <span style="font-size: 0.64rem; color: ${statusColor};" title="${s.badge}">
                ${s.badge}
              </span>
            </td>
            <td style="text-align: right;">
              <span class="df-size">${sizeStr}</span>
            </td>
            <td>
              <div class="df-actions">
                <button class="df-btn-action btn-view-struct"
                        title="View file content" ${s.ready ? '' : 'disabled'}>👁️</button>
                <button class="df-btn-action btn-dl-struct"
                        title="Download file" ${s.ready ? '' : 'disabled'}>💾</button>
                <button class="df-btn-action btn-copy-struct"
                        title="Copy text" ${s.ready ? '' : 'disabled'}>📋</button>
              </div>
            </td>
          </tr>
        `;
      });

      html += `</tbody></table>`;
      listEl.innerHTML = html;

      // Attach event handlers
      listEl.querySelectorAll('.df-row').forEach((row) => {
        const idx = parseInt(row.getAttribute('data-idx'), 10);
        const s = structures[idx];
        if (!s) return;

        row.querySelectorAll('.btn-view-struct').forEach(btn => {
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (s.ready) {
              openFileViewerModal(s.filename, s.category, s.badge, s.size, s.content);
            }
          });
        });

        const btnDl = row.querySelector('.btn-dl-struct');
        if (btnDl) {
          btnDl.addEventListener('click', (e) => {
            e.stopPropagation();
            if (s.binaryBytes) {
              DataManager.downloadBinaryFile(s.filename, s.binaryBytes);
            } else {
              DataManager.downloadTextFile(s.filename, s.content);
            }
            showToast(`💾 Downloaded ${s.filename}`);
          });
        }

        const btnCopy = row.querySelector('.btn-copy-struct');
        if (btnCopy) {
          btnCopy.addEventListener('click', (e) => {
            e.stopPropagation();
            if (navigator.clipboard) {
              navigator.clipboard.writeText(s.content).then(() => {
                showToast(`📋 Copied ${s.filename} to clipboard`);
              }).catch(() => {
                showToast(`📋 Copied ${s.filename}`);
              });
            } else {
              showToast(`📋 Copied ${s.filename}`);
            }
          });
        }
      });
    }
    window.renderDataStructuresUI = renderDataStructuresUI;

    function renderWorkspaceFilesTree() {
      const treeEl = document.getElementById('workspaceFilesTree');
      const headerEl = document.getElementById('lblWorkspaceTreeHeader');
      if (!treeEl) return;

      if (headerEl) {
        if (isDesktopBackend) {
          headerEl.textContent = `💻 Native Workspace (${workspaceFiles.length} items):`;
        } else if (WebFs.isOpen()) {
          headerEl.textContent = `📁 Local Folder (${WebFs.getDirectoryName()}):`;
        } else {
          headerEl.textContent = '⚡ In-Browser Sandbox (No local folder linked):';
        }
      }

      if (!isDesktopBackend && !WebFs.isOpen()) {
        treeEl.innerHTML = `
          <div style="padding: 10px; color: var(--text-muted); text-align: center;
                      line-height: 1.4;">
            <div style="font-size: 1.05rem; margin-bottom: 4px;">⚡ Web Browser Sandbox</div>
            <div style="font-size: 0.70rem;">Working in client-side memory. Click <b>"Save All"</b> or <b>"ZIP"</b>
                 to export results to disk.</div>
            <div style="margin-top: 5px; font-size: 0.66rem; color: #38bdf8;">
              Tip: Click <b>"Open"</b> in the top Workspace bar to link a local folder directly.
            </div>
          </div>
        `;
        return;
      }

      if (workspaceFiles.length === 0) {
        treeEl.innerHTML = `
          <div style="padding: 8px; color: var(--text-muted); text-align: center; font-size: 0.70rem;">
            (Directory is empty)
          </div>
        `;
        return;
      }

      let html = `
        <table class="data-files-table">
          <thead>
            <tr>
              <th style="width: 46%;">Name</th>
              <th style="width: 24%;">Type</th>
              <th style="width: 16%; text-align: right;">Size</th>
              <th style="width: 14%; text-align: center;">Action</th>
            </tr>
          </thead>
          <tbody>
      `;

      workspaceFiles.forEach((f, idx) => {
        const isDir = f.is_dir || f.isDir;
        const icon = isDir
          ? '📁'
          : (f.name.endsWith('.bin') ? '⚡' : (f.name.endsWith('.fits') ? '🌌' : '📄'));
        const sizeStr = !isDir ? `${(f.size / 1024).toFixed(1)} KB` : '—';
        const typeStr = isDir
          ? (f.name.includes('cluster') ? 'Cluster Dir' : 'Directory')
          : (f.name.endsWith('.bin')
             ? 'GRIC Binary'
             : (f.name.endsWith('.fits') ? 'FITS' : 'ASCII Text'));
        const typeBg = isDir
          ? 'rgba(250, 204, 21, 0.12)'
          : (f.name.endsWith('.bin') ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255, 255, 255, 0.06)');
        const typeColor = isDir
          ? '#facc15'
          : (f.name.endsWith('.bin') ? '#38bdf8' : '#94a3b8');

        html += `
          <tr class="df-row" data-ws-idx="${idx}">
            <td>
              <span class="df-filename ${!isDir ? 'btn-view-file' : ''}"
                    style="cursor: ${isDir ? 'default' : 'pointer'};">
                <span>${icon}</span>
                <span style="color: ${isDir ? '#facc15' : '#e2e8f0'};
                             font-weight: ${isDir ? '700' : '400'};">
                  ${f.name}
                </span>
              </span>
            </td>
            <td>
              <span class="badge-pill"
                    style="background: ${typeBg}; color: ${typeColor};
                           font-size: 0.60rem; padding: 1px 4px;">
                ${typeStr}
              </span>
            </td>
            <td style="text-align: right;">
              <span class="df-size">${sizeStr}</span>
            </td>
            <td style="text-align: center;">
              ${isDir && f.name.includes('cluster')
                ? `<button class="df-btn-action btn-load-cluster"
                           style="color: #4ade80; border-color: rgba(74, 222, 128, 0.4);
                                  font-weight: 700;">Load</button>`
                : (!isDir ? `<button class="df-btn-action btn-view-file"
                                     title="View file">👁️</button>` : '')}
            </td>
          </tr>
        `;
      });

      html += `</tbody></table>`;
      treeEl.innerHTML = html;

      // Attach event listeners
      treeEl.querySelectorAll('.df-row').forEach((row) => {
        const idx = parseInt(row.getAttribute('data-ws-idx'), 10);
        const f = workspaceFiles[idx];
        if (!f) return;

        const btnLoad = row.querySelector('.btn-load-cluster');
        if (btnLoad) {
          btnLoad.addEventListener('click', async (e) => {
            e.stopPropagation();
            await loadClusterResults(f.name);
          });
        }

        row.querySelectorAll('.btn-view-file').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
              let text = '';
              if (isDesktopBackend) {
                text = await DesktopBridge.readFile(f.name);
              } else if (WebFs.isOpen()) {
                text = await WebFs.readFile(f.name);
              }
              openFileViewerModal(f.name, 'Workspace File', 'Local Disk', f.size, text);
            } catch (err) {
              showToast(`Failed to read ${f.name}: ${err.message}`);
            }
          });
        });
      });
    }
    window.renderWorkspaceFilesTree = renderWorkspaceFilesTree;

    async function initWorkspaceAndEngine() {
      const lblPath = document.getElementById('lblWorkspacePath');
      const btnCliMode = document.getElementById('btnEngineCli');
      const btnWasmMode = document.getElementById('btnEngineWasm');
      const btnOpenFolder = document.getElementById('btnOpenLocalFolder');
      const btnRefresh = document.getElementById('btnRefreshWorkspace');
      const btnSaveWorkspace = document.getElementById('btnSaveToWorkspace');
      const btnSaveAllData = document.getElementById('btnSaveAllDataStructures');
      const btnDownloadZip = document.getElementById('btnDownloadAllZip');
      const btnRefreshTree = document.getElementById('btnRefreshWorkspaceTree');
      const cliNotice = document.getElementById('cliWebNotice');
      const cliControls = document.getElementById('cliDesktopControls');

      // Probe native C gric-server
      const serverInfo = await DesktopBridge.probe();
      if (serverInfo && !DesktopBridge.isMobileDevice()) {
        isDesktopBackend = true;
        workspacePath = serverInfo.cwd;
        if (lblPath) {
          lblPath.textContent = workspacePath;
          lblPath.title = workspacePath;
        }
        await refreshWorkspaceFiles();

        // Detect GPU availability from native server
        if (DesktopBridge.hasGpu()) {
          gpuAvailable = true;
          gpuInfo = DesktopBridge.getGpuInfo();
          const btnGpu = document.getElementById('btnToggleGpu');
          const cliGpuRow = document.getElementById('cliGpuRow');
          const lblGpuName = document.getElementById('lblGpuDeviceName');
          if (btnGpu) {
            btnGpu.style.display = 'inline-flex';
            const vramGb = (gpuInfo.total_memory_mb / 1024).toFixed(1);
            btnGpu.setAttribute('data-tooltip-desc',
              `Toggle CUDA GPU acceleration (${gpuInfo.name}, ${vramGb} GB VRAM) for native clustering and k-NN.`);
          }
          if (cliGpuRow) cliGpuRow.style.display = 'flex';
          if (lblGpuName && gpuInfo.name) {
            const vramGb = (gpuInfo.total_memory_mb / 1024).toFixed(1);
            lblGpuName.textContent = `${gpuInfo.name} (${vramGb} GB VRAM)`;
          }
        }

        // Default to Native C Engine mode when desktop backend is available
        await setEngineMode('cli', true);
      } else {
        isDesktopBackend = false;
        if (lblPath) {
          lblPath.textContent = DesktopBridge.isMobileDevice()
            ? '📱 Cell Phone Client (In-Browser Sandbox)'
            : 'Web Browser Sandbox (Client-Side Storage)';
        }
        if (btnOpenFolder && WebFs.isSupported() && !DesktopBridge.isMobileDevice()) {
          btnOpenFolder.style.display = 'inline-block';
        } else if (btnOpenFolder) {
          btnOpenFolder.style.display = 'none';
        }
        if (cliNotice) {
          cliNotice.style.display = 'block';
          if (DesktopBridge.isMobileDevice()) {
            cliNotice.innerHTML = '📱 <b>Mobile Phone Mode:</b> In-Browser WebAssembly (WASM) ' +
              'is active with SIMD hardware acceleration. Native CLI is disabled on mobile.';
          }
        }
        if (cliControls) cliControls.style.display = 'none';

        await setEngineMode('wasm', true);
      }

      updateStorageModeBanner();
      updateEngineModeUI();
      renderDataStructuresUI();

      // Bind Engine Switcher Toggle-Slider
      const engineToggleSlider = document.getElementById('engineToggleSlider');
      if (engineToggleSlider) {
        engineToggleSlider.addEventListener('click', (e) => {
          if (!DesktopBridge.isNativeSupported()) {
            if (DesktopBridge.isMobileDevice()) {
              showToast('📱 Cell Phone: Native CLI is disabled (In-Browser WASM active)');
            } else {
              showToast('🌐 Web Mode: Native CLI requires a local desktop gric-server');
            }
            setEngineMode('wasm');
            return;
          }
          if (e.target.id === 'btnEngineWasm') {
            setEngineMode('wasm');
          } else if (e.target.id === 'btnEngineCli') {
            setEngineMode('cli');
          } else {
            setEngineMode(engineMode === 'wasm' ? 'cli' : 'wasm');
          }
        });
      }
      if (btnWasmMode) {
        btnWasmMode.addEventListener('click', (e) => {
          e.stopPropagation();
          setEngineMode('wasm');
        });
      }
      if (btnCliMode) {
        btnCliMode.addEventListener('click', (e) => {
          e.stopPropagation();
          if (!DesktopBridge.isNativeSupported()) {
            if (DesktopBridge.isMobileDevice()) {
              showToast('📱 Cell Phone: Native CLI is disabled (In-Browser WASM active)');
            } else {
              showToast('🌐 Web Mode: Native CLI requires a local desktop gric-server');
            }
            return;
          }
          setEngineMode('cli');
        });
      }

      // GPU Hardware Acceleration Controller
      function setGpuState(enabled) {
        useGpu = Boolean(enabled);
        const btnGpu = document.getElementById('btnToggleGpu');
        const btnGpuSide = document.getElementById('btnToggleGpuSide');
        const btns = [btnGpu, btnGpuSide].filter(Boolean);

        for (const btn of btns) {
          if (useGpu) {
            btn.classList.add('active');
            btn.textContent = '🚀 GPU: ON';
            btn.setAttribute('data-tooltip-badge', 'GPU: ON (CUDA)');
            btn.setAttribute('data-tooltip-color', 'green');
          } else {
            btn.classList.remove('active');
            btn.textContent = '🚀 GPU: OFF';
            btn.setAttribute('data-tooltip-badge', 'GPU: OFF (CPU)');
            btn.setAttribute('data-tooltip-color', 'purple');
          }
        }
        const cliGpuBatchRow = document.getElementById('cliGpuBatchRow');
        if (cliGpuBatchRow) {
          cliGpuBatchRow.style.display = useGpu ? 'flex' : 'none';
        }
        const toolbarGpuBatchWrap = document.getElementById('toolbarGpuBatchWrap');
        if (toolbarGpuBatchWrap) {
          toolbarGpuBatchWrap.style.display = useGpu ? 'inline-flex' : 'none';
        }
        updateEngineModeUI();
        if (typeof updateCliCommand === 'function') {
          updateCliCommand();
        }
      }
      window.setGpuState = setGpuState;

      const btnToggleGpu = document.getElementById('btnToggleGpu');
      const btnToggleGpuSide = document.getElementById('btnToggleGpuSide');
      function onGpuToggleClick() {
        setGpuState(!useGpu);
        if (useGpu) {
          const vramStr = (gpuInfo && gpuInfo.total_memory_mb)
            ? ` (${(gpuInfo.total_memory_mb / 1024).toFixed(1)} GB VRAM)`
            : '';
          showToast(`🚀 NVIDIA CUDA GPU acceleration enabled${vramStr}`);
        } else {
          showToast('💻 Switched to host CPU multi-threading (OpenMP + AVX)');
        }
      }
      if (btnToggleGpu) btnToggleGpu.addEventListener('click', onGpuToggleClick);
      if (btnToggleGpuSide) btnToggleGpuSide.addEventListener('click', onGpuToggleClick);

      const selectCliGpuBatchSize = document.getElementById('selectCliGpuBatchSize');
      const selectToolbarGpuBatchSize = document.getElementById('selectToolbarGpuBatchSize');

      function syncGpuBatchSize(val) {
        if (selectCliGpuBatchSize && selectCliGpuBatchSize.value !== val) {
          selectCliGpuBatchSize.value = val;
        }
        if (selectToolbarGpuBatchSize && selectToolbarGpuBatchSize.value !== val) {
          selectToolbarGpuBatchSize.value = val;
          selectToolbarGpuBatchSize.setAttribute('data-tooltip-badge', `B: ${val}`);
        }
        if (typeof updateCliCommand === 'function') {
          updateCliCommand();
        }
      }

      if (selectCliGpuBatchSize) {
        selectCliGpuBatchSize.addEventListener('change', (e) => syncGpuBatchSize(e.target.value));
      }
      if (selectToolbarGpuBatchSize) {
        selectToolbarGpuBatchSize.addEventListener('change', (e) => syncGpuBatchSize(e.target.value));
      }

      // Bind Workspace buttons
      if (btnRefresh) {
        btnRefresh.addEventListener('click', async () => {
          await refreshWorkspaceFiles();
          showToast('🔄 Workspace refreshed');
        });
      }

      if (btnRefreshTree) {
        btnRefreshTree.addEventListener('click', async () => {
          await refreshWorkspaceFiles();
          showToast('🔄 Workspace refreshed');
        });
      }

      if (btnSaveWorkspace) {
        btnSaveWorkspace.addEventListener('click', async () => {
          await DataManager.saveAllToDisk();
        });
      }

      if (btnSaveAllData) {
        btnSaveAllData.addEventListener('click', async () => {
          await DataManager.saveAllToDisk();
        });
      }

      if (btnDownloadZip) {
        btnDownloadZip.addEventListener('click', () => {
          DataManager.downloadZipBundle();
        });
      }

      if (btnOpenFolder) {
        btnOpenFolder.addEventListener('click', async () => {
          const dir = await WebFs.openDirectory();
          if (dir) {
            if (lblPath) {
              lblPath.textContent = `📁 ${dir.name} (Client Local)`;
            }
            updateStorageModeBanner();
            await refreshWorkspaceFiles();
            showToast(`📁 Opened local folder: ${dir.name}`);
          }
        });
      }

      // Modal File Viewer Buttons
      const btnCopyModal = document.getElementById('btnCopyFileViewer');
      if (btnCopyModal) {
        btnCopyModal.addEventListener('click', () => {
          if (currentViewingFile && currentViewingFile.content) {
            if (navigator.clipboard) {
              navigator.clipboard.writeText(currentViewingFile.content).then(() => {
                showToast(`📋 Copied ${currentViewingFile.title} to clipboard`);
              }).catch(() => {
                showToast(`📋 Copied ${currentViewingFile.title}`);
              });
            } else {
              showToast(`📋 Copied ${currentViewingFile.title}`);
            }
          }
        });
      }

      const btnDlModal = document.getElementById('btnDownloadFileViewer');
      if (btnDlModal) {
        btnDlModal.addEventListener('click', () => {
          if (currentViewingFile && currentViewingFile.content) {
            DataManager.downloadTextFile(currentViewingFile.title, currentViewingFile.content);
            showToast(`💾 Downloaded ${currentViewingFile.title}`);
          }
        });
      }

      const btnCloseModal = document.getElementById('btnCloseFileViewer');
      if (btnCloseModal) {
        btnCloseModal.addEventListener('click', closeFileViewerModal);
      }

      // Bind Banner toggle
      const btnBannerToggle = document.getElementById('btnBannerToggleEngine');
      if (btnBannerToggle) {
        btnBannerToggle.addEventListener('click', () => {
          setEngineMode(engineMode === 'wasm' ? 'cli' : 'wasm');
        });
      }

      updateEngineModeUI();
      setupCliRunnerListeners();
    }

    async function refreshWorkspaceFiles() {
      let files = [];
      let shmStreams = [];
      if (isDesktopBackend) {
        files = await DesktopBridge.listFiles();
        shmStreams = await DesktopBridge.listShmStreams();
      } else if (WebFs.isOpen()) {
        files = await WebFs.listFiles();
      }
      workspaceFiles = files;

      updateStorageModeBanner();
      renderWorkspaceFilesTree();

      const selCli = document.getElementById('selectCliDataset');
      if (selCli) {
        const prevVal = selCli.value;
        selCli.innerHTML = '<option value="">(Select dataset from workspace or SHM...)</option>';

        if (shmStreams.length > 0) {
          const shmGroup = document.createElement('optgroup');
          shmGroup.label = '📡 Live Shared Memory Streams (ImageStreamIO)';
          shmStreams.forEach(s => {
            const opt = document.createElement('option');
            opt.value = `shm:${s.name}`;
            opt.textContent = `📡 ${s.name} (${(s.size / (1024 * 1024)).toFixed(1)} MB)`;
            shmGroup.appendChild(opt);
          });
          selCli.appendChild(shmGroup);
        }

        const fileGroup = document.createElement('optgroup');
        fileGroup.label = '📁 Local Workspace Files (.bin default)';
        const sortedFiles = [...files].sort((a, b) => {
          const aBin = a.name.endsWith('.bin') ? 0 : 1;
          const bBin = b.name.endsWith('.bin') ? 0 : 1;
          return aBin - bBin || a.name.localeCompare(b.name);
        });
        sortedFiles.forEach(f => {
          if (!f.isDir) {
            const opt = document.createElement('option');
            opt.value = f.name;
            const icon = f.name.endsWith('.bin') ? '📦 ' : '';
            opt.textContent = `${icon}${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
            fileGroup.appendChild(opt);
          }
        });
        selCli.appendChild(fileGroup);

        if (prevVal) selCli.value = prevVal;
      }
    }

    async function setEngineMode(mode, silent = false) {
      if (mode === 'cli' && !DesktopBridge.isNativeSupported()) {
        if (!silent) {
          if (DesktopBridge.isMobileDevice()) {
            showToast('📱 Cell Phone: Native CLI is disabled (In-Browser WASM active)');
          } else {
            showToast('🌐 Web Mode: Native CLI requires a local desktop gric-server');
          }
        }
        mode = 'wasm';
      }

      engineMode = mode;
      useWasm = (mode === 'wasm');
      useJSFallback = (mode === 'js');
      updateEngineModeUI();

      if (mode === 'cli') {
        updateCliCommand();

        if (DesktopBridge.isAvailable()) {
          await DesktopBridge.initCliSession();
        }
        if (!silent) {
          showToast('💻 Native CLI mode active (tmux session "gric_cli" ready)');
        }
      } else {
        if (DesktopBridge.isAvailable()) {
          await DesktopBridge.stopCliSession();
        }
        if (typeof GricWasm !== 'undefined' && GricWasm.isLoaded()) {
          const params = GricWasm.buildParamsFromState();
          if (!wasmSessionActive || !GricWasm.isReady() || (GricWasm.isConfigChanged && GricWasm.isConfigChanged(params))) {
            wasmSessionActive = GricWasm.init(params);
            if (typeof updateWasmBadge === 'function') updateWasmBadge();
          }
        }
        if (!silent) {
          showToast('⚡ Switched to In-Browser WebAssembly (WASM)');
        }
      }
    }

    function setupCliRunnerListeners() {
      const btnRun = document.getElementById('btnRunCli');
      const btnKill = document.getElementById('btnKillCli');
      const btnClear = document.getElementById('btnClearCliConsole');
      const btnLoadManual = document.getElementById('btnLoadClusterDatManual');
      const chkAutoLoad = document.getElementById('chkAutoLoadResults');
      const selSideInputMode = document.getElementById('selectInputModeSide');
      const pnlSideStreamConfig = document.getElementById('sideStreamConfigPanel');
      const selSideFps = document.getElementById('selectSideStreamFps');
      const chkSideLoop = document.getElementById('chkSideStreamLoop');
      const chkSideCnt2 = document.getElementById('chkSideStreamCnt2Sync');
      const badgeIngestion = document.getElementById('cliIngestionSourceBadge');

      const updateIngestionStatus = () => {
        const isStream = (selSideInputMode && selSideInputMode.value === 'stream');
        if (pnlSideStreamConfig) {
          pnlSideStreamConfig.style.display = isStream ? 'flex' : 'none';
        }
        if (badgeIngestion) {
          if (isStream) {
            const fpsVal = selSideFps ? selSideFps.value : '0';
            const fpsLabel = (fpsVal === '0') ? 'Lockstep (cnt2sync)' : `${fpsVal} FPS`;
            badgeIngestion.textContent = `⚡ ImageStreamIO (${fpsLabel})`;
            badgeIngestion.style.color = '#38bdf8';
          } else {
            badgeIngestion.textContent = '📁 Direct File / Memory';
            badgeIngestion.style.color = '#4ade80';
          }
        }
        if (isStream && engineMode !== 'cli') {
          setEngineMode('cli');
        }
      };

      if (selSideInputMode) {
        selSideInputMode.addEventListener('change', updateIngestionStatus);
      }
      if (selSideFps) {
        selSideFps.addEventListener('change', updateIngestionStatus);
      }
      updateIngestionStatus();

      const selCliDataset = document.getElementById('selectCliDataset');
      if (selCliDataset) {
        selCliDataset.addEventListener('change', async (e) => {
          const ds = e.target.value;
          updateCliCommand();
          if (ds) {
            await tryAutoLoadCompanionProfile(ds, activeDatasetSlot);
          }
        });
      }

      if (chkAutoLoad) {
        chkAutoLoad.addEventListener('change', (e) => {
          autoLoadCliResults = e.target.checked;
        });
      }

      if (btnClear) {
        btnClear.addEventListener('click', () => {
          const consoleEl = document.getElementById('cliConsoleLog');
          if (consoleEl) consoleEl.textContent = '[Console cleared]';
        });
      }

      if (btnRun) {
        btnRun.addEventListener('click', async () => {
          await runNativeCli();
        });
      }

      const btnRunCliKnn = document.getElementById('btnRunCliKnn');
      if (btnRunCliKnn) {
        btnRunCliKnn.addEventListener('click', async () => {
          await executeKnnComputation();
        });
      }

      if (btnKill) {
        btnKill.addEventListener('click', async () => {
          await killNativeCli();
        });
      }

      if (btnLoadManual) {
        btnLoadManual.addEventListener('click', async () => {
          const clusterDirs = workspaceFiles.filter(f => f.isDir && f.name.includes('cluster'));
          if (clusterDirs.length === 0) {
            showToast('No .clusterdat folders found in workspace');
            return;
          }
          const choice = clusterDirs[0].name;
          await loadClusterResults(choice);
        });
      }

      // Tmux attach copy buttons
      const copyTmuxAttachCmd = () => {
        if (navigator.clipboard) {
          navigator.clipboard.writeText('tmux attach -t gric_cli').then(() => {
            showToast('📋 Copied: tmux attach -t gric_cli');
          }).catch(() => {
            showToast('tmux attach -t gric_cli');
          });
        } else {
          showToast('tmux attach -t gric_cli');
        }
      };

      const badgeNativeTmux = document.getElementById('badgeNativeTmux');
      if (badgeNativeTmux) {
        badgeNativeTmux.addEventListener('click', copyTmuxAttachCmd);
      }

      const btnCopyTmux = document.getElementById('btnCopyTmuxCmd');
      if (btnCopyTmux) {
        btnCopyTmux.addEventListener('click', () => {
          copyTmuxAttachCmd();
          btnCopyTmux.textContent = '✅ Copied';
          setTimeout(() => {
            btnCopyTmux.textContent = '📋 Copy Attach';
          }, 1500);
        });
      }
    }

    async function runNativeCli() {
      if (!isDesktopBackend || !DesktopBridge.isAvailable()) {
        const probed = await DesktopBridge.probe();
        if (!probed) {
          showToast(
            '❌ Cannot connect to gric-server. Launch via ./tools/gric-gui or start gric-server'
          );
          return;
        }
        isDesktopBackend = true;
      }

      const selCli = document.getElementById('selectCliDataset');
      let dataset = selCli ? selCli.value : '';
      if (!dataset) {
        dataset = `${currentBenchmark}.bin`;
      }

      let args = [];
      let isStreamInput = false;
      const inputMode = document.getElementById('selectInputModeSide')?.value || 'file';
      const isStreamingMode = (inputMode === 'stream');
      const streamFps = parseFloat(
        document.getElementById('selectSideStreamFps')?.value || '0'
      );
      const streamLoop =
        document.getElementById('chkSideStreamLoop')?.checked || false;
      const streamName = `gric_sim_${Date.now() % 100000}`;
      let streamJobOpts = null;

      if (isStreamingMode) {
        isStreamInput = true;
        const isSynthetic = !selCli || !selCli.value ||
          dataset === `${currentBenchmark}.bin` || dataset === `${currentBenchmark}.txt` ||
          (typeof BENCHMARK_DESCS !== 'undefined' &&
           BENCHMARK_DESCS[dataset.replace(/\.[^/.]+$/, '')]);

        const existingFile = workspaceFiles &&
          workspaceFiles.find(f => f.name === dataset);

        let needStage = false;
        if (isSynthetic || !dataset) {
          if (!existingFile || existingFile.size === 0) {
            needStage = true;
          } else if (benchmarkDataset && benchmarkDataset.length > 0) {
            const stagedCount = DesktopBridge.getStagedDatasetCount(dataset);
            const expectedMin = benchmarkDataset.length * Math.max(2, (currentDim || 2) * 4);
            if (stagedCount > 0 && stagedCount !== benchmarkDataset.length) {
              needStage = true;
            } else if (existingFile.size < expectedMin) {
              needStage = true;
            }
          }
        }

        if (needStage) {
          dataset = `${currentBenchmark}.bin`;
          if (!benchmarkDataset || benchmarkDataset.length === 0) {
            stageDataset();
          }
          try {
            const countStr = benchmarkDataset.length.toLocaleString();
            showToast(`📦 Staging ${countStr} pts for streaming...`);
            await DesktopBridge.stageDatasetFile(
              currentBenchmark,
              benchmarkDataset,
              currentDim
            );
            await refreshWorkspaceFiles();
          } catch (err) {
            console.warn('[CLI] Could not stage stream dataset file:', err);
          }
        }

        const baseName = dataset.replace(/\.[^/.]+$/, '');
        const clusterDir = `${baseName}.clusterdat`;
        const streamCnt2sync = (streamFps === 0) ||
          (document.getElementById('chkSideStreamCnt2Sync')?.checked ?? false);

        args = [rlim.toFixed(3), streamName, '-stream', '-outdir', clusterDir];
        if (streamCnt2sync) {
          args.push('-cnt2sync');
        }
        if (!streamLoop && benchmarkDataset && benchmarkDataset.length > 0) {
          args.push('-maxim', String(benchmarkDataset.length));
        }

        streamJobOpts = {
          streamFile: dataset,
          streamName: streamName,
          streamFps: streamFps,
          streamLoop: streamLoop,
          streamCnt2sync: streamCnt2sync
        };
      } else if (dataset.startsWith('shm:')) {
        const customStreamName = dataset.substring(4);
        isStreamInput = true;
        args = [
          rlim.toFixed(3),
          customStreamName,
          '-stream',
          '-outdir',
          `${customStreamName}.clusterdat`
        ];
      } else {
        // If running active synthetic benchmark, ensure file exists in workspace
        const isSynthetic = !selCli || !selCli.value ||
          dataset === `${currentBenchmark}.bin` || dataset === `${currentBenchmark}.txt` ||
          (typeof BENCHMARK_DESCS !== 'undefined' &&
           BENCHMARK_DESCS[dataset.replace(/\.[^/.]+$/, '')]);

        const existingFile = workspaceFiles &&
          workspaceFiles.find(f => f.name === dataset);

        let needStage = false;
        if (isSynthetic) {
          if (!existingFile || existingFile.size === 0) {
            needStage = true;
          } else if (benchmarkDataset && benchmarkDataset.length > 0) {
            const stagedCount = DesktopBridge.getStagedDatasetCount(dataset);
            const expectedMin = benchmarkDataset.length * Math.max(2, (currentDim || 2) * 4);
            if (stagedCount > 0 && stagedCount !== benchmarkDataset.length) {
              needStage = true;
            } else if (existingFile.size < expectedMin) {
              needStage = true;
            }
          }
        }

        if (needStage) {
          dataset = `${currentBenchmark}.bin`;
          if (!benchmarkDataset || benchmarkDataset.length === 0) {
            stageDataset();
          }
          try {
            const countStr = benchmarkDataset.length.toLocaleString();
            showToast(`📦 Staging ${countStr} pts to workspace (.bin)...`);
            await DesktopBridge.stageDatasetFile(
              currentBenchmark,
              benchmarkDataset,
              currentDim
            );
            await refreshWorkspaceFiles();
          } catch (err) {
            console.warn('[CLI] Could not stage benchmark dataset file:', err);
            showToast(`❌ Failed to stage dataset: ${err.message}`);
          }
        }
        args = [rlim.toFixed(3), dataset];
        if (benchmarkDataset && benchmarkDataset.length > 0) {
          args.push('-maxim', String(benchmarkDataset.length));
        }
      }

      if (pruneMode === '3P') {
        args.push('-no-te4', '-no-te5');
      } else if (pruneMode === '4P') {
        args.push('-te4', '-no-te5');
      } else if (pruneMode === '5P') {
        args.push('-te4', '-te5');
      }

      if (targetMode === 'entropy') {
        args.push('-entropy');
        if (entropyGate !== 2.0) args.push('-entropy_gate', entropyGate.toFixed(2));
        if (entropyFirstGate !== 4.0) {
          args.push('-entropy_first_gate', entropyFirstGate.toFixed(2));
        }
        if (entropyFastMode) args.push('-entropy_fast');
      } else {
        args.push('-no-entropy');
      }

      if (useTM && tmMixingCoeff > 0) {
        args.push('-tm', tmMixingCoeff.toFixed(2));
      } else {
        args.push('-no-tm');
      }

      if (usePred) {
        if (predHorizon !== 2) {
          args.push('-pred[,,' + predHorizon + ']');
        } else {
          args.push('-pred');
        }
      } else {
        args.push('-no-pred');
      }

      if (useGprob) {
        args.push('-gprob');
        if (maxVisitors !== 1000) args.push('-maxvis', maxVisitors.toString());
      }

      if (useSoftBayesian) {
        args.push('-soft_bayesian');
        if (softBayesianSigmaCoeff !== 1.0) {
          args.push('-soft_bayesian_sigma', softBayesianSigmaCoeff.toFixed(2));
        }
      } else {
        args.push('-no-soft-bayesian');
      }

      if (typeof useTiles !== 'undefined' && useTiles) {
        args.push('-tiles');
        if (useXTile) args.push('-xtile');
      } else {
        args.push('-no-tiles');
      }

      if (useSparseDcc) {
        args.push('-sparse_dcc');
        if (sparseDccExtraEvals > 0) {
          args.push('-sparse_dcc_extra_evals', sparseDccExtraEvals.toString());
        }
      } else {
        args.push('-no-sparse-dcc');
      }

      if (typeof clusterUseEq16 === 'boolean' && clusterUseEq16) {
        args.push('-eq16');
        if (typeof clusterUseEq16Adc === 'boolean' && clusterUseEq16Adc) {
          args.push('-eq16-adc');
        } else {
          args.push('-no-eq16-adc');
        }
        if (typeof clusterSq16Ratio !== 'undefined' && clusterSq16Ratio !== 0.05) {
          args.push('-eq16-ratio', String(clusterSq16Ratio));
        }
        args.push('-no-sq16', '-no-sq8');
      } else if (typeof clusterUseSq16 === 'boolean' && clusterUseSq16) {
        args.push('-sq16');
        if (typeof clusterSq16Ratio !== 'undefined' && clusterSq16Ratio !== 0.05) {
          args.push('-sq16-ratio', String(clusterSq16Ratio));
        }
        if (typeof clusterUseMemo !== 'undefined' && !clusterUseMemo) {
          args.push('-no-memo');
        }
        args.push('-no-sq8', '-no-eq16');
      } else if (typeof clusterUseSq8 === 'boolean' && clusterUseSq8) {
        args.push('-sq8', '-no-sq16', '-no-eq16');
      } else {
        args.push('-no-sq16', '-no-sq8', '-no-eq16');
      }

      if (typeof clusterUseBatchDist === 'boolean' && !clusterUseBatchDist) {
        args.push('-no-batch-dist');
      }
      if (maxcl > 0) {
        args.push('-maxcl', maxcl.toString());
      } else {
        // In GUI, 0 = Unlimited: allocate large capacity headroom for native mode
        args.push('-maxcl', '10000');
      }
      if (maxclStrategy !== 'stop') {
        args.push('-maxcl_strategy', maxclStrategy);
      }
      if (maxclStrategy === 'discard' && discardFraction !== 0.10) {
        args.push('-discard_frac', discardFraction.toFixed(2));
      }
      args.push('-evals');
      if (useGpu) {
        args.push('--gpu');
        const bSize = document.getElementById('selectCliGpuBatchSize')?.value;
        if (bSize) args.push('--gpu-batch-size', bSize);
      }

      const btnRun = document.getElementById('btnRunCli');
      const btnRunKnn = document.getElementById('btnRunCliKnn');
      const btnKill = document.getElementById('btnKillCli');
      const btnPlay = document.getElementById('btnPlay');
      const btnStop = document.getElementById('btnStop');
      const badgeStatus = document.getElementById('badgeCliStatus');
      const consoleEl = document.getElementById('cliConsoleLog');

      if (btnRun) btnRun.disabled = true;
      if (btnRunKnn) btnRunKnn.disabled = true;
      if (btnKill) btnKill.disabled = false;
      if (btnStop) btnStop.disabled = false;
      if (btnPlay) {
        btnPlay.innerHTML = '⏳ Stop gric-cluster';
        btnPlay.title = 'Native gric-cluster is running. Click to terminate.';
        btnPlay.disabled = false;
        btnPlay.classList.add('danger');
      }
      if (badgeStatus) {
        badgeStatus.textContent = '● tmux: gric_cli';
        badgeStatus.style.background = 'rgba(74, 222, 128, 0.2)';
        badgeStatus.style.color = '#4ade80';
      }
      if (consoleEl) {
        const streamNote = isStreamingMode
          ? `📡 Streamer: gric-txt2stream ${dataset} ${streamName} -fps ${streamFps}${streamLoop ? ' -loop' : ''}\n`
          : '';
        const bSize = document.getElementById('selectCliGpuBatchSize')?.value || '64';
        const gpuNote = useGpu
          ? `🚀 Hardware Engine: NVIDIA CUDA Acceleration (--gpu --gpu-batch-size ${bSize})\n`
          : `💻 Hardware Engine: Host CPU Multi-threading (OpenMP + AVX)\n`;
        consoleEl.textContent = `🚀 Dispatched in tmux session: gric_cli\n` +
          `🖥️ Attach live: tmux attach -t gric_cli\n` +
          `📄 Log stream: /tmp/gric_latest.log\n` +
          gpuNote +
          streamNote +
          `⚙️ Command: gric-cluster ${args.join(' ')}\n` +
          `─────────────────────────────────────────────────────────────\n`;
      }
      showToast(useGpu
        ? '🚀 Native CLI launched with NVIDIA CUDA GPU acceleration'
        : '🚀 Native CLI launched in tmux session "gric_cli"');

      const tStart = performance.now();
      isCliRunning = true;
      knnResults = null;

      try {
        await DesktopBridge.runCliJob({
          cmd: 'gric-cluster',
          args: args,
          ...(streamJobOpts || {}),
          onOutput: (chunk) => {
            if (consoleEl) {
              consoleEl.textContent += chunk;
              consoleEl.scrollTop = consoleEl.scrollHeight;
            }
          },
          onTelemetry: (t) => {
            if (!t) return;

            // Live progress & frame counters from SHM
            if (t.total_frames > 0) {
              const pct = Math.min(100, Math.max(0, (t.processed_frames / t.total_frames) * 100));
              const fill = document.getElementById('progressFill');
              if (fill) fill.style.width = `${pct.toFixed(1)}%`;
              const fc = document.getElementById('frameCounter');
              if (fc) fc.textContent = `${t.processed_frames} / ${t.total_frames} (${pct.toFixed(1)}%)`;
            } else if (t.processed_frames > 0) {
              const fc = document.getElementById('frameCounter');
              if (fc) fc.textContent = `${t.processed_frames} frames (streaming)`;
            }

            const cb = document.getElementById('clusterBadge');
            if (cb) cb.textContent = `${t.num_clusters} clusters`;

            const fpsBadge = document.getElementById('fpsBadge');
            if (fpsBadge && t.elapsed_ms > 0) {
              const fps = Math.round(t.processed_frames / (t.elapsed_ms / 1000.0));
              fpsBadge.textContent = `${fps} fps`;
            }

            if (t.elapsed_ms > 0 && t.processed_frames > 0) {
              const liveFps = t.processed_frames / (t.elapsed_ms / 1000.0);
              sessionAvgFps = liveFps;
              avgComputeTimeMs = t.elapsed_ms / t.processed_frames;
              currentCpuLoadPct = 100.0;

              const statAvgFpsEl = document.getElementById('statAvgFps');
              if (statAvgFpsEl) {
                statAvgFpsEl.textContent = liveFps >= 1000
                  ? `${Math.round(liveFps).toLocaleString()} fps`
                  : `${liveFps.toFixed(1)} fps`;
              }
              const statCpuLoadEl = document.getElementById('statCpuLoad');
              if (statCpuLoadEl) {
                statCpuLoadEl.textContent = `${currentCpuLoadPct.toFixed(1)}%`;
              }
              const statComputeMsEl = document.getElementById('statComputeMs');
              if (statComputeMsEl) {
                statComputeMsEl.textContent = `${avgComputeTimeMs.toFixed(2)} ms`;
              }
            }

            // Sync resource tracker panel
            const statFramesEl = document.getElementById('statFrames');
            if (statFramesEl) statFramesEl.textContent = t.processed_frames;

            const statClustersEl = document.getElementById('statClusters');
            if (statClustersEl) statClustersEl.textContent = t.num_clusters;

            const statTotalTimeEl = document.getElementById('statTotalTime');
            if (statTotalTimeEl) statTotalTimeEl.textContent = formatClockTime(t.elapsed_ms);

            const statTotalDistsEl = document.getElementById('statTotalDists');
            if (statTotalDistsEl) statTotalDistsEl.textContent = (t.framedist_calls || 0).toLocaleString();

            const statDistSampleEl = document.getElementById('statDistSample');
            if (statDistSampleEl) {
              statDistSampleEl.textContent =
                (t.framedist_sample || 0).toLocaleString();
            }

            const statDistClusterEl = document.getElementById('statDistCluster');
            if (statDistClusterEl) {
              statDistClusterEl.textContent =
                (t.framedist_intercluster || 0).toLocaleString();
            }

            const statDistRatioEl = document.getElementById('statDistRatio');
            if (statDistRatioEl) {
              statDistRatioEl.textContent =
                `${(t.framedist_sample || 0).toLocaleString()} / ` +
                `${(t.framedist_intercluster || 0).toLocaleString()}`;
            }

            const statDccPopBadge = document.getElementById('statDccPopBadge');
            if (statDccPopBadge && t.dcc_entries_populated !== undefined) {
              const pop = t.dcc_entries_populated || 0;
              const total = t.dcc_pairs_total || 0;
              const formatCompact = typeof formatCompactNumber === 'function'
                ? formatCompactNumber
                : n => (n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' :
                        (n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : n));

              if (total > 0) {
                const pct = ((pop / total) * 100).toFixed(total > 100000 ? 2 : 1);
                statDccPopBadge.textContent = `${formatCompact(pop)} (${pct}%)`;
                statDccPopBadge.title =
                  `${pop.toLocaleString()} / ${total.toLocaleString()} pairs (${pct}%)`;
              } else {
                statDccPopBadge.textContent = pop.toLocaleString();
                statDccPopBadge.title = `${pop.toLocaleString()} pairs populated`;
              }
            }

            const statMemoryTotalEl = document.getElementById('statMemoryTotal');
            if (statMemoryTotalEl && t.memory_rss_kb > 0) {
              statMemoryTotalEl.textContent = formatBytes(t.memory_rss_kb * 1024);
            }

            const statPrune3PEl = document.getElementById('statPrune3P');
            if (statPrune3PEl) statPrune3PEl.textContent = (t.clusters_pruned || 0).toLocaleString();

            const statPrune4PEl = document.getElementById('statPrune4P');
            if (statPrune4PEl) statPrune4PEl.textContent = (t.prune_4p_count || 0).toLocaleString();

            const statPrune5PEl = document.getElementById('statPrune5P');
            if (statPrune5PEl) statPrune5PEl.textContent = (t.prune_5p_count || 0).toLocaleString();

            const statPredHitsEl = document.getElementById('statPredHits');
            if (statPredHitsEl) statPredHitsEl.textContent = (t.pred_hits || 0).toLocaleString();

            const statPredAttemptsEl = document.getElementById('statPredAttempts');
            if (statPredAttemptsEl) statPredAttemptsEl.textContent = (t.pred_attempts || 0).toLocaleString();

            const statPredRateEl = document.getElementById('statPredRate');
            if (statPredRateEl && t.pred_attempts > 0) {
              const rate = ((t.pred_hits / t.pred_attempts) * 100).toFixed(1);
              statPredRateEl.textContent = `${rate}%`;
            }

            const statMemoHitsEl = document.getElementById('statMemoHits');
            const statMemoHitsOverview = document.getElementById('statMemoHitsOverview');
            if (t.memo_hits !== undefined) {
              if (typeof memoHits !== 'undefined') memoHits = t.memo_hits || 0;
              if (typeof memoLookups !== 'undefined') memoLookups = t.memo_lookups || 0;
              const hitsStr = (t.memo_hits || 0).toLocaleString();
              if (statMemoHitsEl) statMemoHitsEl.textContent = hitsStr;
              if (statMemoHitsOverview) {
                const pct = t.memo_lookups > 0
                  ? ((t.memo_hits / t.memo_lookups) * 100).toFixed(1)
                  : '0.0';
                statMemoHitsOverview.textContent = `${hitsStr} (${pct}%)`;
              }
            }

            const lblMemoHitsEl = document.getElementById('lblMemoHits');
            if (lblMemoHitsEl && t.memo_lookups > 0) {
              const pct = ((t.memo_hits / t.memo_lookups) * 100).toFixed(1);
              lblMemoHitsEl.textContent = `Memo Hits (${pct}%)`;
            }

            const memHashCacheEl = document.getElementById('memHashCache');
            const lblHashCacheEl = document.getElementById('lblHashCache');
            const statHashCacheOverview = document.getElementById('statHashCacheOverview');
            if (t.memo_cache_capacity > 0) {
              if (typeof memoCacheCapacity !== 'undefined') memoCacheCapacity = t.memo_cache_capacity || 0;
              const cacheStr = formatBytes(t.memo_cache_capacity * 32);
              if (memHashCacheEl) memHashCacheEl.textContent = cacheStr;
              if (statHashCacheOverview) statHashCacheOverview.textContent = `Cache: ${cacheStr}`;
            }
            if (lblHashCacheEl && t.memo_cache_entries !== undefined) {
              if (typeof memoCacheEntries !== 'undefined') memoCacheEntries = t.memo_cache_entries || 0;
              lblHashCacheEl.textContent = `Hash Cache (${t.memo_cache_entries.toLocaleString()} entries)`;
            }

            // Mobile HUD overlay
            const hudSamples = document.getElementById('hudSamples');
            if (hudSamples) hudSamples.textContent = t.processed_frames;
            const hudClusters = document.getElementById('hudClusters');
            if (hudClusters) hudClusters.textContent = t.num_clusters;
            const hudTime = document.getElementById('hudTime');
            if (hudTime) hudTime.textContent = formatClockTime(t.elapsed_ms);
          },
          onFinish: async (res) => {
            isCliRunning = false;
            updateClusteringButtonUI();
            const elapsed = ((performance.now() - tStart) / 1000).toFixed(2);
            if (btnRun) btnRun.disabled = false;
            if (btnRunKnn) btnRunKnn.disabled = false;
            if (btnKill) btnKill.disabled = true;
            if (btnStop) btnStop.disabled = true;
            if (btnPlay) {
              btnPlay.innerHTML = '▶ Run gric-cluster';
              btnPlay.disabled = false;
            }

            if (res.exitCode === 0 || res.exitCode === 137) {
              if (badgeStatus) {
                badgeStatus.textContent = `Done (${elapsed}s)`;
                badgeStatus.style.background = 'rgba(56, 189, 248, 0.2)';
                badgeStatus.style.color = '#38bdf8';
              }
              showToast(`✅ gric-cluster completed in ${elapsed}s`);

              if (autoLoadCliResults) {
                const baseName = dataset.startsWith('shm:')
                  ? dataset.substring(4)
                  : dataset.replace(/\.[^/.]+$/, '');
                const clusterDir = `${baseName}.clusterdat`;
                await loadClusterResults(clusterDir);
              }
            } else {
              if (badgeStatus) {
                badgeStatus.textContent = `Failed (exit ${res.exitCode})`;
                badgeStatus.style.background = 'rgba(248, 113, 113, 0.2)';
                badgeStatus.style.color = '#f87171';
              }
              showToast(`❌ CLI execution failed (exit ${res.exitCode})`);
            }

            await refreshWorkspaceFiles();
          }
        });
      } catch (err) {
        isCliRunning = false;
        updateClusteringButtonUI();
        if (btnRun) btnRun.disabled = false;
        if (btnKill) btnKill.disabled = true;
        if (btnStop) btnStop.disabled = true;
        if (btnPlay) {
          btnPlay.innerHTML = '▶ Run gric-cluster';
          btnPlay.disabled = false;
        }
        if (consoleEl) consoleEl.textContent += `\n[Error]: ${err.message}\n`;
        showToast(`Error: ${err.message}`);
      }
    }

    async function loadClusterResults(clusterDir) {
      try {
        const data = await DesktopBridge.parseClusterDatDir(clusterDir);
        if (!data.anchors || data.anchors.length === 0) {
          showToast(`No cluster centroids found in ${clusterDir}`);
          return;
        }

        if (isRunning) pauseSimulation();

        clusters = data.anchors.map((a, i) => ({
          id: i,
          x: a.x,
          y: a.y,
          z: a.z,
          anchor: a.anchor,
          members: a.members || 0,
          prob: 0,
          scDists: 0,
          lastActive: 0,
          color: getClusterColor(i)
        }));

        if (data.dcc && data.dcc.length > 0) {
          dcc = data.dcc;
        }

        // Populate pastSamples so points appear across the viewports
        if (benchmarkDataset && benchmarkDataset.length > 0) {
          pastSamples = benchmarkDataset.map((p, idx) => ({
            x: (Array.isArray(p) || ArrayBuffer.isView(p)) ? p[0] : (p.x || 0),
            y: (Array.isArray(p) || ArrayBuffer.isView(p)) ? p[1] : (p.y || 0),
            z: currentDim >= 3
              ? ((Array.isArray(p) || ArrayBuffer.isView(p)) ? (p[2] || 0) : (p.z || 0)) : 0,
            coords: p.coords || (Array.isArray(p) || ArrayBuffer.isView(p) ? p : null),
            frameIndex: idx
          }));
        } else {
          try {
            const baseName = clusterDir.replace(/\.clusterdat\/?$/, '');
            const txtContent = await DesktopBridge.readFile(`${baseName}.txt`);
            if (txtContent) {
              const lines = txtContent.split(/\r?\n/);
              const pts = [];
              for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line || line.startsWith('#')) continue;
                const tokens = line.split(/[,\s\t]+/).filter(t => t.length > 0);
                if (tokens.length >= 2) {
                  const coords = new Float64Array(tokens.length);
                  for (let d = 0; d < tokens.length; d++) {
                    coords[d] = parseFloat(tokens[d]) || 0;
                  }
                  pts.push({
                    x: coords[0],
                    y: coords[1],
                    z: coords.length >= 3 ? coords[2] : 0,
                    coords: coords,
                    frameIndex: pts.length
                  });
                }
                if (pts.length >= 1000000) break;
              }
              if (pts.length > 0) {
                pastSamples = pts;
                benchmarkDataset = pts;
                rawBenchmarkDataset = pts;
              }
            }
          } catch (e) {
            /* ignore */
          }
        }

        if (data.anchors && data.anchors.length > 0 && data.anchors[0].anchor) {
          currentDim = data.anchors[0].anchor.length;
        } else if (pastSamples && pastSamples.length > 0 && pastSamples[0].coords) {
          currentDim = pastSamples[0].coords.length;
        }

        if (data.evals && data.evals.length > 0) {
          frameEvaluationsLog = data.evals;
        }
        if (data.assignments && data.assignments.length > 0) {
          assignmentHistory = data.assignments;
          imageFrameAssignments = data.assignments;
          imageClusterMembers = {};
          data.assignments.forEach((cId, fIdx) => {
            if (!imageClusterMembers[cId]) {
              imageClusterMembers[cId] = [];
            }
            imageClusterMembers[cId].push(fIdx);
            if (pastSamples && fIdx < pastSamples.length) {
              pastSamples[fIdx].clusterId = cId;
            }
            if (benchmarkDataset && fIdx < benchmarkDataset.length) {
              benchmarkDataset[fIdx].clusterId = cId;
            }
          });
        } else if (clusters && clusters.length > 0 && pastSamples && pastSamples.length > 0) {
          pastSamples.forEach((pt, idx) => {
            if (pt.clusterId === undefined || pt.clusterId < 0) {
              let bestK = 0, bestD2 = Infinity;
              for (let k = 0; k < clusters.length; k++) {
                const c = clusters[k];
                const dx = pt.x - c.x, dy = pt.y - c.y, dz = (pt.z || 0) - (c.z || 0);
                const d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < bestD2) { bestD2 = d2; bestK = k; }
              }
              pt.clusterId = bestK;
              if (benchmarkDataset && idx < benchmarkDataset.length) {
                benchmarkDataset[idx].clusterId = bestK;
              }
            }
          });
        }

        // Apply native execution stats to global telemetry & Resource Tracker
        let sumMembers = 0;
        clusters.forEach(c => { sumMembers += c.members; });

        if (data.stats && data.stats.frames > 0) {
          totalFrames = data.stats.frames;
          currentFrameIdx = totalFrames;
          sessionStartFrames = 0;
          distSampleCluster = data.stats.sampleDists;
          distClusterCluster = data.stats.interclusterDists;
          distSampleClusterTotal = data.stats.sampleDists;
          distClusterClusterTotal = data.stats.interclusterDists;
          dccPopulated = data.stats.dccPopulated || 0;
          dccPairsTotal = data.stats.dccPairsTotal ||
            (clusters.length > 1 ? (clusters.length * (clusters.length - 1) / 2) : 0);
          totalEvals = data.stats.sampleDists;
          naiveEvals = data.stats.sampleDists + data.stats.pruned;
          sessionElapsedMs = data.stats.timeMs;
          sessionAvgFps = data.stats.timeMs > 0 ? (totalFrames / (data.stats.timeMs / 1000.0)) : 0.0;
          avgComputeTimeMs = totalFrames > 0 ? (data.stats.timeMs / totalFrames) : 0.0;
          currentCpuLoadPct = 100.0;
          sessionIsActive = false;

          if (data.stats.rssKb > 0) {
            const statMemoryTotalEl = document.getElementById('statMemoryTotal');
            if (statMemoryTotalEl) {
              statMemoryTotalEl.textContent = formatBytes(data.stats.rssKb * 1024);
            }
          }

          if (data.stats.memoHits !== undefined) {
            if (typeof memoHits !== 'undefined') memoHits = data.stats.memoHits || 0;
            if (typeof memoLookups !== 'undefined') memoLookups = data.stats.memoLookups || 0;
            const statMemoHitsEl = document.getElementById('statMemoHits');
            const statMemoHitsOverview = document.getElementById('statMemoHitsOverview');
            const hitsStr = data.stats.memoHits.toLocaleString();
            if (statMemoHitsEl) {
              statMemoHitsEl.textContent = hitsStr;
            }
            if (statMemoHitsOverview) {
              const pct = data.stats.memoLookups > 0
                ? ((data.stats.memoHits / data.stats.memoLookups) * 100).toFixed(1)
                : '0.0';
              statMemoHitsOverview.textContent = `${hitsStr} (${pct}%)`;
            }
            const lblMemoHitsEl = document.getElementById('lblMemoHits');
            if (lblMemoHitsEl && data.stats.memoLookups > 0) {
              const pct = ((data.stats.memoHits / data.stats.memoLookups) * 100).toFixed(1);
              lblMemoHitsEl.textContent = `Memo Hits (${pct}%)`;
            }
          }
          if (data.stats.memoCacheCapacity !== undefined) {
            if (typeof memoCacheCapacity !== 'undefined') memoCacheCapacity = data.stats.memoCacheCapacity || 0;
            if (typeof memoCacheEntries !== 'undefined') memoCacheEntries = data.stats.memoCacheEntries || 0;
            const cacheStr = formatBytes(data.stats.memoCacheCapacity * 32);
            const memHashCacheEl = document.getElementById('memHashCache');
            if (memHashCacheEl) {
              memHashCacheEl.textContent = cacheStr;
            }
            const statHashCacheOverview = document.getElementById('statHashCacheOverview');
            if (statHashCacheOverview) {
              statHashCacheOverview.textContent = `Cache: ${cacheStr}`;
            }
            const lblHashCacheEl = document.getElementById('lblHashCache');
            if (lblHashCacheEl && data.stats.memoCacheEntries !== undefined) {
              lblHashCacheEl.textContent = `Hash Cache (${data.stats.memoCacheEntries.toLocaleString()} entries)`;
            }
          }
        } else if (sumMembers > 0) {
          totalFrames = sumMembers;
          currentFrameIdx = totalFrames;
          sessionStartFrames = 0;
          sessionIsActive = false;
        } else if (pastSamples && pastSamples.length > 0) {
          totalFrames = pastSamples.length;
          currentFrameIdx = totalFrames;
          sessionStartFrames = 0;
          sessionIsActive = false;
        }

        // Compute Shannon entropy over cluster member distribution
        if (totalFrames > 0 && clusters.length > 0) {
          let hBits = 0.0;
          for (let i = 0; i < clusters.length; i++) {
            const p = clusters[i].members / totalFrames;
            clusters[i].prob = p;
            if (p > 0.0) {
              hBits -= p * Math.log2(p);
            }
          }
          currentEntropyBits = hBits;
        }

        // Update progress bar and frame counter HUD
        const progressFill = document.getElementById('progressFill');
        if (progressFill) progressFill.style.width = '100%';
        const frameCounter = document.getElementById('frameCounter');
        if (frameCounter) frameCounter.textContent = `${totalFrames} / ${totalFrames} (100.0%)`;

        // Reset k-NN post-processing results for newly computed/loaded clusters
        knnResults = null;
        if (typeof renderKnnTrace === 'function') {
          renderKnnTrace();
        }

        if (typeof saveSlotState === 'function') {
          saveSlotState(activeDatasetSlot);
        }
        if (typeof updateDatasetStatusBadge === 'function') {
          updateDatasetStatusBadge();
        }
        if (typeof renderReconstructionDashboard === 'function') {
          renderReconstructionDashboard();
        }
        updateUI();
        if (typeof renderDataStructuresUI === 'function') {
          renderDataStructuresUI();
        }
        resizeCanvas();
        draw();
        requestAnimationFrame(() => {
          resizeCanvas();
          draw();
        });
        setTimeout(() => {
          resizeCanvas();
          draw();
        }, 80);

        showToast(`📊 Loaded ${clusters.length} clusters (${totalFrames.toLocaleString()} frames) from ${clusterDir}`);
      } catch (err) {
        console.error('[LoadResults] Error loading cluster results:', err);
        showToast(`Failed to load ${clusterDir}: ${err.message}`);
      }
    }

    async function exportRunToWorkspace() {
      await DataManager.saveAllToDisk();
    }

    // Initial Startup
    initSidebarResizers();
    initLayoutResizer();
    updateTMCanvasDimensions();
    resizeCanvas();
    if (typeof populateDefaultMultiDatasets === 'function') {
      populateDefaultMultiDatasets();
    } else {
      stageDataset('3Dspiral', 'B');
      stageDataset('3Drand', 'C');
      clearDatasetSlot('D');
    }
    activeDatasetSlot = 'A';
    loadSelectedBenchmark();
    updateZoomBadge();
    setExplainMode(false);
    updateCliCommand();
    initWorkspaceAndEngine();
    updateDatasetStatusBadge();
    if (typeof updateMultiDatasetUI === 'function') {
      updateMultiDatasetUI();
    }
    collapseAllControlPanels();

    // Note: Help & Documentation modal & Presets moved to help_modal.js

    const commandPaletteCommands = [
      // Actions
      { id: 'act-play', group: 'Actions', icon: '▶',
        name: 'Cluster Active Dataset (Run or Re-cluster)',
        hint: 'Space', action: () => document.getElementById('btnPlay')?.click() },
      { id: 'act-compute-all', group: 'Actions', icon: '⚡',
        name: 'Instant Cluster / Re-cluster to Completion',
        hint: 'Batch', action: () => document.getElementById('btnPlay')?.click() },
      { id: 'act-stop', group: 'Actions', icon: '⏹',
        name: 'Stop Clustering',
        hint: 'Esc', action: () => document.getElementById('btnStop')?.click() },
      { id: 'act-step', group: 'Actions', icon: '⏭', name: 'Step Single Frame',
        hint: 'S', action: () => document.getElementById('btnStep')?.click() },
      { id: 'act-reset-clusters', group: 'Actions', icon: '↺',
        name: 'Reset Clusters (Keep Dataset)', hint: 'Reset Model',
        action: () => document.getElementById('btnResetClusters')?.click() },
      { id: 'act-reset-all', group: 'Actions', icon: '⟲',
        name: 'Clear Active Dataset & Clusters', hint: 'Clear',
        action: () => document.getElementById('btnClearDataset_A')?.click() },
      { id: 'act-pass2', group: 'Actions', icon: '🔄',
        name: '2nd Pass Nearest Anchor Reallocation', hint: '2 / P',
        action: () => document.getElementById('btnPass2Nearest')?.click() },
      { id: 'act-explain', group: 'Actions', icon: '💬', name: 'Toggle Decision Explain Mode',
        hint: 'E', action: () => document.getElementById('btnExplain')?.click() },
      { id: 'act-knn-run', group: 'Actions', icon: '⚡',
        name: 'Compute / Re-run k-Nearest Neighbors',
        hint: 'k-NN', action: () => document.getElementById('btnRunKnn')?.click() },
      { id: 'act-multi-dataset-toggle', group: 'Actions', icon: '🗂️', name: 'Toggle Multi-Dataset Mode (A, B, C)',
        hint: 'Multi-DS', action: () => {
          if (typeof setMultiDatasetEnabled === 'function') {
            setMultiDatasetEnabled(!multiDatasetEnabled);
          }
        } },
      { id: 'act-shuffle-dataset', group: 'Actions', icon: '🔀',
        name: 'Shuffle Frame Order (Active Dataset)', hint: 'Shuffle',
        action: () => document.getElementById('btnShuffleNow')?.click() },
      { id: 'act-reroll-ball-seed', group: 'Actions', icon: '🎲',
        name: 'Re-roll Random Ball Seed', hint: 'Re-roll',
        action: () => document.getElementById('btnNewBallSeed')?.click() },
      { id: 'act-img-show-all', group: 'Actions', icon: '⊞', name: 'Show All View Panels (4-Split Grid)',
        hint: 'Esc', action: () => {
          maximizedQuad = null;
          syncImageQuadUI();
          if (typeof draw === 'function') draw();
          if (typeof showToast === 'function') showToast('⊞ Restored All 4 View Panels');
        } },
      { id: 'act-img-sort-desc', group: 'Actions', icon: '📊', name: 'Image Mode: Sort Clusters by Size (Descending)',
        hint: 'Sort Desc', action: () => {
          imageClustersSortMode = 'size_desc';
          const sel = document.getElementById('selectImgClusterSort');
          if (sel) sel.value = 'size_desc';
          if (typeof draw === 'function') draw();
          if (typeof showToast === 'function') showToast('📊 Sorted by Cluster Size (Descending)');
        } },
      { id: 'act-img-sort-asc', group: 'Actions', icon: '📉', name: 'Image Mode: Sort Clusters by Size (Ascending)',
        hint: 'Sort Asc', action: () => {
          imageClustersSortMode = 'size_asc';
          const sel = document.getElementById('selectImgClusterSort');
          if (sel) sel.value = 'size_asc';
          if (typeof draw === 'function') draw();
          if (typeof showToast === 'function') showToast('📉 Sorted by Cluster Size (Ascending)');
        } },
      { id: 'act-img-sort-id', group: 'Actions', icon: '🔢', name: 'Image Mode: Sort Clusters by ID (Default)',
        hint: 'Sort ID', action: () => {
          imageClustersSortMode = 'id';
          const sel = document.getElementById('selectImgClusterSort');
          if (sel) sel.value = 'id';
          if (typeof draw === 'function') draw();
          if (typeof showToast === 'function') showToast('🔢 Sorted by Creation ID (Default)');
        } },
      { id: 'act-motion-tail', group: 'Actions', icon: '〰️',
        name: 'Toggle Recent Points Motion Trail', hint: 'Trail',
        action: () => document.getElementById('btnToggleMotionTail')?.click() },
      { id: 'act-color-per-cluster', group: 'Actions', icon: '🎨',
        name: 'Toggle Per-Cluster Point Colors', hint: 'Colors',
        action: () => document.getElementById('btnToggleColorPerCluster')?.click() },
      { id: 'act-export', group: 'Actions', icon: '💾', name: 'Export Run to Local Workspace',
        hint: '.clusterdat', action: () => document.getElementById('btnSaveToWorkspace')?.click() },

      // Camera Views
      { id: 'cam-11', group: 'Camera', icon: '🔍', name: '1:1 Reset Pan & Zoom',
        hint: 'Z', action: () => document.getElementById('btnResetView')?.click() },
      { id: 'cam-iso', group: 'Camera', icon: '📐', name: 'Isometric 3D Perspective',
        hint: '3D Orbit', action: () => document.getElementById('presetIso')?.click() },
      { id: 'cam-front', group: 'Camera', icon: '🔲', name: 'Front View (XZ Plane)',
        hint: 'Front', action: () => document.getElementById('presetFront')?.click() },
      { id: 'cam-top', group: 'Camera', icon: '🔝', name: 'Top View (XY Plane)',
        hint: 'Top', action: () => document.getElementById('presetTop')?.click() },
      { id: 'cam-side', group: 'Camera', icon: '🔳', name: 'Side View (YZ Plane)',
        hint: 'Side', action: () => document.getElementById('presetSide')?.click() },
      { id: 'cam-reset3d', group: 'Camera', icon: '↺', name: 'Reset 3D Orbit Camera',
        hint: 'Reset 3D', action: () => document.getElementById('presetReset3D')?.click() },
      { id: 'cam-lock-center', group: 'Camera', icon: '🎯', name: 'Lock / Unlock 3D Center of Rotation',
        hint: 'C', action: () => toggleLockCenter3D() },

      // Datasets
      { id: 'ds-slot-a', group: 'Datasets', icon: '🅰️',
        name: 'Switch to Dataset A (workspace/A)', hint: 'Slot A',
        action: () => switchDatasetSlot('A') },
      { id: 'ds-slot-b', group: 'Datasets', icon: '🅱️',
        name: 'Switch to Dataset B (workspace/B)', hint: 'Slot B',
        action: () => switchDatasetSlot('B') },
      { id: 'ds-slot-c', group: 'Datasets', icon: '🅲',
        name: 'Switch to Dataset C (workspace/C)', hint: 'Slot C',
        action: () => switchDatasetSlot('C') },
      { id: 'ds-3dtorus', group: 'Datasets', icon: '🍩',
        name: 'Load 3Dtorus (3D Torus Manifold Knot)', hint: '3D Dataset',
        action: () => loadDatasetByKey('3Dtorus') },
      { id: 'ds-2dspiral', group: 'Datasets', icon: '🌀',
        name: 'Load 2Dspiral (Archimedean Spiral)', hint: '2D Dataset',
        action: () => loadDatasetByKey('2Dspiral') },
      { id: 'ds-3dsphere', group: 'Datasets', icon: '🌐',
        name: 'Load 3Dsphere (Spherical Shell S²)', hint: '3D Dataset',
        action: () => loadDatasetByKey('3Dsphere') },
      { id: 'ds-3dstar', group: 'Datasets', icon: '✨',
        name: 'Load 3Dstar (Multi-Spoke Radial Star)', hint: '3D Dataset',
        action: () => loadDatasetByKey('3Dstar') },
      { id: 'ds-3dlorenz', group: 'Datasets', icon: '🦋',
        name: 'Load 3Dlorenz (Lorenz Strange Attractor)', hint: '3D Dataset',
        action: () => loadDatasetByKey('3Dlorenz') },
      { id: 'ds-2dcircle', group: 'Datasets', icon: '⭕',
        name: 'Load 2Dcircle-shuffle (Circle Shuffled)', hint: '2D Dataset',
        action: () => loadDatasetByKey('2Dcircle-shuffle') },
      { id: 'ds-img-ball1', group: 'Datasets', icon: '⚽',
        name: 'Load Single Bouncing Ball (32×32 Image)', hint: 'Image Dataset',
        action: () => loadDatasetByKey('img-ball-1') },

      // Panels & Modes
      { id: 'nav-clustering', group: 'Navigation', icon: '📊', name: 'Switch to Clustering Mode',
        hint: 'Sidebar', action: () => switchSidebarMode('clustering') },
      { id: 'nav-knn', group: 'Navigation', icon: '⚡', name: 'Switch to k-NN & Topology Mode',
        hint: 'Sidebar', action: () => {
          if (!enableKnn && typeof toggleKnnModule === 'function') toggleKnnModule(true);
          switchSidebarMode('knn');
        } },
      { id: 'nav-recon', group: 'Navigation', icon: '⚡', name: 'Switch to Reconstruction Mode',
        hint: 'Sidebar', action: () => switchSidebarMode('recon') },
      { id: 'nav-files', group: 'Navigation', icon: '📂', name: 'Switch to Files & CLI Mode',
        hint: 'Sidebar', action: () => switchSidebarMode('files') },
      { id: 'nav-all', group: 'Navigation', icon: '📑', name: 'Show All Panels',
        hint: 'Sidebar', action: () => switchSidebarMode('all') }
    ];

    function loadDatasetByKey(key) {
      const sel = document.getElementById('selectBenchmark');
      if (sel) {
        sel.value = key;
        sel.dispatchEvent(new Event('change'));
      }
    }

    let cmdSelectedIdx = 0;
    let cmdFilteredList = [];

    function openCommandPalette() {
      const modal = document.getElementById('modalCommandPalette');
      const input = document.getElementById('commandPaletteInput');
      if (!modal || !input) return;
      modal.style.display = 'flex';
      input.value = '';
      renderCommandPaletteResults('');
      setTimeout(() => input.focus(), 20);
    }

    function closeCommandPalette() {
      const modal = document.getElementById('modalCommandPalette');
      if (modal) modal.style.display = 'none';
    }

    function toggleCommandPalette() {
      const modal = document.getElementById('modalCommandPalette');
      if (modal && modal.style.display !== 'none') {
        closeCommandPalette();
      } else {
        openCommandPalette();
      }
    }
    window.openCommandPalette = openCommandPalette;
    window.closeCommandPalette = closeCommandPalette;
    window.toggleCommandPalette = toggleCommandPalette;

    function renderCommandPaletteResults(query) {
      const resultsEl = document.getElementById('commandPaletteResults');
      if (!resultsEl) return;
      const q = query.trim().toLowerCase();

      cmdFilteredList = commandPaletteCommands.filter(c => {
        if (!q) return true;
        return c.name.toLowerCase().includes(q) ||
               c.group.toLowerCase().includes(q) ||
               (c.hint && c.hint.toLowerCase().includes(q));
      });

      cmdSelectedIdx = Math.max(0, Math.min(cmdSelectedIdx, cmdFilteredList.length - 1));

      if (cmdFilteredList.length === 0) {
        resultsEl.innerHTML = `
          <div style="padding: 24px; text-align: center; color: var(--text-muted);
                      font-size: 0.8rem;">
            No matching commands found
          </div>`;
        return;
      }

      const groups = {};
      cmdFilteredList.forEach((cmd, idx) => {
        if (!groups[cmd.group]) groups[cmd.group] = [];
        groups[cmd.group].push({ cmd, flatIdx: idx });
      });

      let html = '';
      Object.entries(groups).forEach(([groupName, items]) => {
        html += `<div class="command-palette-group-title">${groupName}</div>`;
        items.forEach(({ cmd, flatIdx }) => {
          const isActive = (flatIdx === cmdSelectedIdx);
          const badgeHtml = cmd.hint
            ? `<span class="command-palette-item-badge">${cmd.hint}</span>`
            : '';
          html += `
            <div class="command-palette-item ${isActive ? 'active' : ''}"
                 data-idx="${flatIdx}" onclick="executeCommand(${flatIdx})">
              <div class="command-palette-item-left">
                <span class="command-palette-item-icon">${cmd.icon}</span>
                <span class="command-palette-item-text">${cmd.name}</span>
              </div>
              ${badgeHtml}
            </div>
          `;
        });
      });

      resultsEl.innerHTML = html;

      const activeEl = resultsEl.querySelector('.command-palette-item.active');
      if (activeEl) {
        activeEl.scrollIntoView({ block: 'nearest' });
      }
    }

    function executeCommand(idx) {
      if (idx >= 0 && idx < cmdFilteredList.length) {
        const cmd = cmdFilteredList[idx];
        closeCommandPalette();
        if (cmd && typeof cmd.action === 'function') {
          cmd.action();
        }
      }
    }
    window.executeCommand = executeCommand;

    function initCommandPalette() {
      const btn = document.getElementById('btnOpenCommandPalette');
      if (btn) btn.addEventListener('click', openCommandPalette);

      const modal = document.getElementById('modalCommandPalette');
      if (modal) {
        modal.addEventListener('click', (e) => {
          if (e.target === modal) closeCommandPalette();
        });
      }

      const input = document.getElementById('commandPaletteInput');
      if (input) {
        input.addEventListener('input', (e) => {
          cmdSelectedIdx = 0;
          renderCommandPaletteResults(e.target.value);
        });

        input.addEventListener('keydown', (e) => {
          if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (cmdFilteredList.length > 0) {
              cmdSelectedIdx = (cmdSelectedIdx + 1) % cmdFilteredList.length;
              renderCommandPaletteResults(input.value);
            }
          } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (cmdFilteredList.length > 0) {
              const len = cmdFilteredList.length;
              cmdSelectedIdx = (cmdSelectedIdx - 1 + len) % len;
              renderCommandPaletteResults(input.value);
            }
          } else if (e.key === 'Enter') {
            e.preventDefault();
            executeCommand(cmdSelectedIdx);
          } else if (e.key === 'Escape') {
            e.preventDefault();
            closeCommandPalette();
          }
        });
      }
    }
