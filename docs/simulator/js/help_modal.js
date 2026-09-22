/**
 * GRIC Simulator - help_modal.js
 * Documentation center, presets configuration, and interactive help modal.
 */

    // =========================================================================
    //  HELP & DOCUMENTATION CENTER (MULTI-TOPIC MODAL & PRESETS)
    // =========================================================================

    const PRESETS = {
      'preset-basic-spiral': {
        name: '2D Spiral Baseline',
        benchmark: '2Dspiral',
        rlim: 0.100,
        targetMode: 'greedy',
        pruneMode: '3P',
        useTM: false,
        usePred: false,
        useTiles: false,
        speed: 50,
        loopCount: 1,
        isExplainMode: false,
        useWasm: true,
        autoPlay: true,
      },
      'preset-entropy-target': {
        name: 'Shannon Entropy Scheduling',
        benchmark: '2Dspiral',
        rlim: 0.100,
        targetMode: 'entropy',
        entropyGate: 0.75,
        entropyFirstGate: 1.50,
        pruneMode: '3P',
        useTM: false,
        usePred: false,
        useTiles: false,
        speed: 50,
        loopCount: 1,
        isExplainMode: false,
        useWasm: true,
        autoPlay: true,
      },
      'preset-pruning-5p': {
        name: '5P Metric Geometric Pruning (3D Torus)',
        benchmark: '3Dtorus',
        rlim: 0.100,
        targetMode: 'greedy',
        pruneMode: '5P',
        useTM: false,
        usePred: false,
        useTiles: false,
        speed: 50,
        loopCount: 1,
        isExplainMode: false,
        useWasm: true,
        autoPlay: true,
      },
      'preset-markov-pred': {
        name: 'Markov Transitions & Sequence Prediction',
        benchmark: '2DcircleP10n',
        rlim: 0.120,
        targetMode: 'greedy',
        pruneMode: '3P',
        useTM: true,
        tmMixingCoeff: 0.70,
        usePred: true,
        predHorizon: 2,
        useTiles: false,
        speed: 50,
        loopCount: 5,
        isExplainMode: false,
        useWasm: true,
        autoPlay: true,
      },
      'preset-explain-walkthrough': {
        name: 'Explain Mode Step-by-Step Walkthrough',
        benchmark: '2Dspiral',
        rlim: 0.100,
        targetMode: 'entropy',
        entropyGate: 0.75,
        pruneMode: '3P',
        useTM: false,
        usePred: false,
        useTiles: false,
        speed: 150,
        loopCount: 1,
        isExplainMode: true,
        useWasm: true,
        autoPlay: false,
      },
      'preset-max-throughput': {
        name: 'Maximum Performance C/WASM Engine',
        benchmark: '3Dspiral',
        rlim: 0.080,
        targetMode: 'entropy',
        pruneMode: '4P',
        useTM: false,
        usePred: true,
        predHorizon: 2,
        useTiles: false,
        speed: 0,
        loopCount: 1,
        isExplainMode: false,
        useWasm: true,
        autoPlay: true,
      }
    };

    function applySimulatorPreset(presetKey) {
      const preset = PRESETS[presetKey];
      if (!preset) return;

      if (isRunning) pauseSimulation();

      // Benchmark
      if (preset.benchmark) {
        currentBenchmark = preset.benchmark;
        const selBench = document.getElementById('selectBenchmark');
        if (selBench) selBench.value = preset.benchmark;
        const selBenchSide = document.getElementById('selectBenchmarkSide');
        if (selBenchSide) selBenchSide.value = preset.benchmark;
        const descEl = document.getElementById('benchmarkDesc');
        if (descEl && BENCHMARK_DESCS[preset.benchmark]) {
          descEl.innerHTML = BENCHMARK_DESCS[preset.benchmark];
        }
      }

      // Radius
      if (preset.rlim !== undefined) {
        rlim = preset.rlim;
        const slRlim = document.getElementById('sliderRlim');
        if (slRlim) slRlim.value = rlim;
        const inpRlim = document.getElementById('inputRlim');
        if (inpRlim) inpRlim.value = rlim.toFixed(3);
      }

      // Target mode
      if (preset.targetMode) {
        targetMode = preset.targetMode;
        const btnGreedy = document.getElementById('modeGreedy');
        const btnEntropy = document.getElementById('modeEntropy');
        if (btnGreedy) btnGreedy.classList.toggle('active', targetMode === 'greedy');
        if (btnEntropy) btnEntropy.classList.toggle('active', targetMode === 'entropy');
      }

      // Pruning mode
      if (preset.pruneMode) {
        pruneMode = preset.pruneMode;
        ['3P', '4P', '5P'].forEach(p => {
          const btn = document.getElementById(`prune${p}`);
          if (btn) btn.classList.toggle('active', p === pruneMode);
        });
      }

      // TM mixing
      if (preset.useTM !== undefined) {
        useTM = preset.useTM;
        const optTM = document.getElementById('optTM');
        if (optTM) optTM.classList.toggle('active', useTM);
        if (preset.tmMixingCoeff !== undefined) {
          tmMixingCoeff = preset.tmMixingCoeff;
          const sl = document.getElementById('sliderTmMix');
          if (sl) sl.value = tmMixingCoeff;
          const inp = document.getElementById('inputTmMix');
          if (inp) inp.value = tmMixingCoeff.toFixed(2);
        }
      }

      // Sequence prediction
      if (preset.usePred !== undefined) {
        usePred = preset.usePred;
        const optPred = document.getElementById('optPred');
        if (optPred) optPred.classList.toggle('active', usePred);
        if (preset.predHorizon !== undefined) {
          predHorizon = preset.predHorizon;
          const sl = document.getElementById('sliderPredHorizon');
          if (sl) sl.value = predHorizon;
          const inp = document.getElementById('inputPredHorizon');
          if (inp) inp.value = predHorizon;
        }
      }

      // Tiling
      if (preset.useTiles !== undefined) {
        useTiles = preset.useTiles;
        const optTiles = document.getElementById('optTiles');
        if (optTiles) optTiles.classList.toggle('active', useTiles);
      }

      // Entropy gates
      if (preset.entropyGate !== undefined) {
        entropyGate = preset.entropyGate;
        const sl = document.getElementById('sliderEntropyGate');
        if (sl) sl.value = entropyGate;
        const inp = document.getElementById('inputEntropyGate');
        if (inp) inp.value = entropyGate.toFixed(2);
      }
      if (preset.entropyFirstGate !== undefined) {
        entropyFirstGate = preset.entropyFirstGate;
        const sl = document.getElementById('sliderEntropyFirstGate');
        if (sl) sl.value = entropyFirstGate;
        const inp = document.getElementById('inputEntropyFirstGate');
        if (inp) inp.value = entropyFirstGate.toFixed(2);
      }

      // Speed
      if (preset.speed !== undefined) {
        playSpeed = preset.speed;
        const selSpeed = document.getElementById('selectSpeed');
        if (selSpeed) selSpeed.value = preset.speed.toString();
        const selSpeedSide = document.getElementById('selectSpeedSide');
        if (selSpeedSide) selSpeedSide.value = preset.speed.toString();
      }

      // Loops
      if (preset.loopCount !== undefined) {
        loopCount = preset.loopCount;
        const selLoop = document.getElementById('selectLoop');
        if (selLoop) selLoop.value = preset.loopCount.toString();
        const selLoopSide = document.getElementById('selectLoopSide');
        if (selLoopSide) selLoopSide.value = preset.loopCount.toString();
      }

      // Explain mode
      if (preset.isExplainMode !== undefined) {
        setExplainMode(preset.isExplainMode);
      }

      // WASM preference
      if (preset.useWasm !== undefined && GricWasm.isLoaded()) {
        useWasm = preset.useWasm;
      }

      syncControlDependencies();
      updateWasmBadge();
      updateCliCommand();

      // Reset and reload dataset
      loadSelectedBenchmark();
      resetView();
      updateUI();
      draw();

      // Close modal
      const modal = document.getElementById('helpModal');
      if (modal) modal.style.display = 'none';

      showToast(`⚡ Loaded: ${preset.name}`);

      if (preset.autoPlay) {
        setTimeout(() => {
          startSimulation();
        }, 250);
      }
    }

    // Help modal lifecycle & topic switching
    const helpModal = document.getElementById('helpModal');
    const toggleHelp = () => {
      if (!helpModal) return;
      helpModal.style.display =
        helpModal.style.display === 'none' ? 'flex' : 'none';
    };

    const btnHelp = document.getElementById('btnHelp');
    if (btnHelp) btnHelp.addEventListener('click', toggleHelp);

    const btnHelpClose = document.getElementById('btnHelpClose');
    if (btnHelpClose) {
      btnHelpClose.addEventListener('click', () => {
        if (helpModal) helpModal.style.display = 'none';
      });
    }

    if (helpModal) {
      helpModal.addEventListener('click', (e) => {
        if (e.target === helpModal) {
          helpModal.style.display = 'none';
        }
      });
    }

    // Help Topic Sidebar Switching
    const topicButtons = document.querySelectorAll('.help-topic-btn');
    topicButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const topic = btn.getAttribute('data-topic');
        topicButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        document.querySelectorAll('.help-topic-content').forEach(pane => {
          pane.classList.remove('active');
        });

        const targetPane = document.getElementById(`topic-${topic}`);
        if (targetPane) {
          targetPane.classList.add('active');
        }
      });
    });

    // Preset Run Buttons inside Help Modal
    document.querySelectorAll('.help-preset-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const key = btn.getAttribute('data-preset');
        if (key) {
          applySimulatorPreset(key);
        }
      });
    });

    const fileModalEl = document.getElementById('modalFileViewer');
    if (fileModalEl) {
      fileModalEl.addEventListener('click', (e) => {
        if (e.target === fileModalEl) {
          fileModalEl.style.display = 'none';
        }
      });
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === '?' && !e.target.matches('input, textarea, select')) {
        e.preventDefault();
        toggleHelp();
      }
      if (e.key === 'Escape') {
        if (helpModal && helpModal.style.display !== 'none') {
          helpModal.style.display = 'none';
        }
        if (fileModalEl && fileModalEl.style.display !== 'none') {
          fileModalEl.style.display = 'none';
        }
        const cmdModal = document.getElementById('modalCommandPalette');
        if (cmdModal && cmdModal.style.display !== 'none') {
          cmdModal.style.display = 'none';
        }
      }
    });

    // -------------------------------------------------------------------------
    // Sidebar Master Mode Switcher
    // -------------------------------------------------------------------------
    function switchSidebarMode(mode) {
      activeSidebarMode = mode;
      document.querySelectorAll('#sidebarModeNav .sidebar-mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
      });

      if (mode === 'knn' && !enableKnn) {
        if (typeof toggleKnnModule === 'function') {
          toggleKnnModule(true);
        }
      }

      const cards = {
        cardInputData: ['all', 'clustering', 'knn', 'recon'],
        cardSettings: ['all', 'clustering'],
        cardDisplay: ['all', 'clustering', 'knn', 'recon'],
        cardDataFiles: ['all', 'files'],
        cardCli: ['all', 'files'],
        cardResources: ['all', 'clustering', 'telemetry'],
        cardTrace: ['all', 'clustering', 'telemetry'],
        cardKnnSettings: ['all', 'knn'],
        cardKnnResources: ['all', 'knn', 'telemetry'],
        cardKnnTrace: ['all', 'knn', 'telemetry'],
        cardDimDensity: ['all', 'knn', 'telemetry'],
        cardReconstruction: ['all', 'recon']
      };

      Object.entries(cards).forEach(([cardId, allowedModes]) => {
        const el = document.getElementById(cardId);
        if (!el) return;
        const isKnnCard = cardId.startsWith('cardKnn') || cardId === 'cardDimDensity';
        const allowed = allowedModes.includes(mode);

        if (isKnnCard && !enableKnn && mode !== 'knn') {
          el.style.display = 'none';
        } else {
          el.style.display = allowed ? '' : 'none';
        }
      });

      if (typeof updateResizersVisibility === 'function') {
        updateResizersVisibility();
      }

      const reconActions = document.getElementById('reconRowDShortcutActions');
      if (reconActions) {
        reconActions.style.display =
          (mode === 'recon' || activeDatasetSlot === 'D') ? 'inline-flex' : 'none';
      }
    }
    window.switchSidebarMode = switchSidebarMode;

    // -------------------------------------------------------------------------
    // Interactive Timeline Scrubber & Milestones
    // -------------------------------------------------------------------------
    function addClusterMilestone(frameIdx) {
      if (!clusterMilestoneFrames.includes(frameIdx)) {
        clusterMilestoneFrames.push(frameIdx);
      }
      renderTimelineMilestones();
    }
    window.addClusterMilestone = addClusterMilestone;

    function renderTimelineMilestones() {
      const container = document.getElementById('timelineMilestones');
      if (!container) return;
      const total = (benchmarkDataset && benchmarkDataset.length > 0) ? benchmarkDataset.length : 1;
      container.innerHTML = '';
      clusterMilestoneFrames.forEach(f => {
        const tick = document.createElement('div');
        tick.className = 'timeline-milestone-tick';
        tick.style.left = `${(f / total) * 100}%`;
        container.appendChild(tick);
      });
    }
    window.renderTimelineMilestones = renderTimelineMilestones;

    function initTimelineScrubber() {
      const scrubber = document.getElementById('timelineScrubber');
      const tooltip = document.getElementById('timelineScrubTooltip');
      if (!scrubber) return;

      let isScrubbing = false;

      function scrubToMouse(e) {
        if (!benchmarkDataset || benchmarkDataset.length === 0) return;
        const rect = scrubber.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const total = benchmarkDataset.length;
        const targetFrame = Math.floor(ratio * (total - 1));

        if (typeof dataMode !== 'undefined' && dataMode === 'image') {
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(targetFrame);
          }
        } else {
          selectPastSample(targetFrame);
        }

        if (tooltip) {
          tooltip.style.left = `${ratio * 100}%`;
          tooltip.textContent = `Frame ${targetFrame + 1} / ${total}`;
          tooltip.style.display = 'block';
        }
      }

      scrubber.addEventListener('mousedown', (e) => {
        isScrubbing = true;
        scrubToMouse(e);
      });

      window.addEventListener('mousemove', (e) => {
        if (isScrubbing) {
          scrubToMouse(e);
        }
      });

      window.addEventListener('mouseup', () => {
        if (isScrubbing) {
          isScrubbing = false;
          if (tooltip) tooltip.style.display = 'none';
        }
      });

      scrubber.addEventListener('mousemove', (e) => {
        if (!benchmarkDataset || benchmarkDataset.length === 0) return;
        const rect = scrubber.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const total = benchmarkDataset.length;
        const targetFrame = Math.floor(ratio * (total - 1));

        if (tooltip && !isScrubbing) {
          tooltip.style.left = `${ratio * 100}%`;
          tooltip.textContent = `Frame ${targetFrame + 1} / ${total}`;
          tooltip.style.display = 'block';
        }
      });

      scrubber.addEventListener('mouseleave', () => {
        if (!isScrubbing && tooltip) {
          tooltip.style.display = 'none';
        }
      });
    }

    function initImageScrubber() {
      const slider = document.getElementById('sliderImgFrame');
      const inputFrame = document.getElementById('inputImgFrame');
      const btnPrev = document.getElementById('btnImgPrevFrame');
      const btnNext = document.getElementById('btnImgNextFrame');
      const btnLive = document.getElementById('btnImgLiveStream');

      window.isDraggingImageSlider = false;

      if (slider) {
        const onSliderUpdate = (e) => {
          const val = parseInt(e.target.value, 10);
          if (!isNaN(val) && typeof selectImageFrame === 'function') {
            selectImageFrame(val);
          }
        };

        slider.addEventListener('pointerdown', () => { window.isDraggingImageSlider = true; });
        slider.addEventListener('mousedown', () => { window.isDraggingImageSlider = true; });
        window.addEventListener('pointerup', () => { window.isDraggingImageSlider = false; });
        window.addEventListener('mouseup', () => { window.isDraggingImageSlider = false; });

        slider.addEventListener('input', onSliderUpdate);
        slider.addEventListener('change', onSliderUpdate);
      }

      if (inputFrame) {
        const commitFrameInput = () => {
          const total = (benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset.length
            : (typeof totalFrames !== 'undefined' ? totalFrames : 0);
          if (total === 0) return;
          let val = parseInt(inputFrame.value, 10);
          if (isNaN(val)) {
            const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
              inspectedImageFrameIdx >= 0)
              ? inspectedImageFrameIdx
              : Math.max(0, (typeof totalFrames !== 'undefined' ? totalFrames : 1) - 1);
            inputFrame.value = cur + 1;
            return;
          }
          val = Math.max(1, Math.min(total, val));
          inputFrame.value = val;
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(val - 1);
          }
        };

        inputFrame.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commitFrameInput();
            inputFrame.blur();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
              inspectedImageFrameIdx >= 0)
              ? inspectedImageFrameIdx
              : Math.max(0, (typeof totalFrames !== 'undefined' ? totalFrames : 1) - 1);
            inputFrame.value = cur + 1;
            inputFrame.blur();
          }
        });
        inputFrame.addEventListener('change', commitFrameInput);
      }

      if (btnPrev) {
        btnPrev.addEventListener('click', () => {
          const total = (benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset.length
            : totalFrames;
          if (total === 0) return;
          const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
            inspectedImageFrameIdx >= 0)
            ? inspectedImageFrameIdx
            : totalFrames - 1;
          const target = Math.max(0, cur - 1);
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(target);
          }
        });
      }

      if (btnNext) {
        btnNext.addEventListener('click', () => {
          const total = (benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset.length
            : totalFrames;
          if (total === 0) return;
          const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
            inspectedImageFrameIdx >= 0)
            ? inspectedImageFrameIdx
            : totalFrames - 1;
          const target = Math.min(total - 1, cur + 1);
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(target);
          }
        });
      }

      if (btnLive) {
        btnLive.addEventListener('click', () => {
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(-1);
          }
        });
      }

      // -------------------------------------------------------------
      // Reconstruction Quality Bar & Box Event Handlers
      // -------------------------------------------------------------
      const qualityCanvas = document.getElementById('canvasImgReconQualityBar');
      const qualityBox = document.getElementById('boxImgFrameReconQuality');
      if (qualityCanvas) {
        let isScrubbingQualityBar = false;

        const scrubFromEvent = (e) => {
          const rect = qualityCanvas.getBoundingClientRect();
          if (rect.width <= 0) return;
          const total = (benchmarkDataset && benchmarkDataset.length > 0)
            ? benchmarkDataset.length
            : (typeof totalFrames !== 'undefined' ? totalFrames : 0);
          if (total <= 0) return;

          const clientX = (e.touches && e.touches.length > 0)
            ? e.touches[0].clientX : e.clientX;
          const relX = Math.max(0, Math.min(rect.width, clientX - rect.left));
          const ratio = relX / rect.width;
          const target = Math.min(total - 1, Math.floor(ratio * total));
          if (typeof selectImageFrame === 'function') {
            selectImageFrame(target);
          }
        };

        qualityCanvas.addEventListener('pointerdown', (e) => {
          isScrubbingQualityBar = true;
          try { qualityCanvas.setPointerCapture(e.pointerId); } catch (_) {}
          scrubFromEvent(e);
        });

        qualityCanvas.addEventListener('pointermove', (e) => {
          if (isScrubbingQualityBar) {
            scrubFromEvent(e);
          }
        });

        const stopQualityScrub = (e) => {
          if (isScrubbingQualityBar) {
            isScrubbingQualityBar = false;
            try { qualityCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
          }
        };

        qualityCanvas.addEventListener('pointerup', stopQualityScrub);
        qualityCanvas.addEventListener('pointercancel', stopQualityScrub);

        qualityCanvas.addEventListener('mousemove', (e) => {
          if (isScrubbingQualityBar) return;
          const qualData = (typeof getReconVarianceData === 'function')
            ? getReconVarianceData()
            : (typeof window.getReconVarianceData === 'function')
              ? window.getReconVarianceData() : null;
          if (!qualData || !qualData.arr || qualData.arr.length === 0) return;

          const rect = qualityCanvas.getBoundingClientRect();
          if (rect.width <= 0) return;
          const relX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
          const ratio = relX / rect.width;
          const target = Math.min(qualData.arr.length - 1,
                                  Math.floor(ratio * qualData.arr.length));
          const val = qualData.arr[target];
          qualityCanvas.title = `Recon Quality: Frame #${target + 1} ` +
            `Var = ${val.toFixed(4)} (Click to jump)`;
        });
      }

      if (qualityBox) {
        qualityBox.addEventListener('click', () => {
          const cur = (typeof inspectedImageFrameIdx !== 'undefined' &&
                       inspectedImageFrameIdx >= 0)
            ? inspectedImageFrameIdx : 0;
          const qualData = (typeof getReconVarianceData === 'function')
            ? getReconVarianceData()
            : (typeof window.getReconVarianceData === 'function')
              ? window.getReconVarianceData() : null;
          if (qualData && qualData.arr && cur < qualData.arr.length) {
            const val = qualData.arr[cur];
            if (typeof showToast === 'function') {
              showToast(`📊 Frame #${cur + 1} Recon Variance: ${val.toFixed(4)}`);
            }
          }
        });
      }

      // -------------------------------------------------------------
      // Cluster Inspector Event Handlers
      // -------------------------------------------------------------
      const sliderCluster = document.getElementById('sliderImgCluster');
      const inputCluster = document.getElementById('inputImgCluster');
      const btnPrevCluster = document.getElementById('btnImgPrevCluster');
      const btnNextCluster = document.getElementById('btnImgNextCluster');
      const btnAutoCluster = document.getElementById('btnImgAutoCluster');

      window.isDraggingImageClusterSlider = false;

      if (sliderCluster) {
        const onClusterSliderUpdate = (e) => {
          const val = parseInt(e.target.value, 10);
          if (!isNaN(val) && typeof selectImageCluster === 'function') {
            selectImageCluster(val);
          }
        };

        sliderCluster.addEventListener('pointerdown', () => {
          window.isDraggingImageClusterSlider = true;
        });
        sliderCluster.addEventListener('mousedown', () => {
          window.isDraggingImageClusterSlider = true;
        });
        window.addEventListener('pointerup', () => {
          window.isDraggingImageClusterSlider = false;
        });
        window.addEventListener('mouseup', () => {
          window.isDraggingImageClusterSlider = false;
        });

        sliderCluster.addEventListener('input', onClusterSliderUpdate);
        sliderCluster.addEventListener('change', onClusterSliderUpdate);
      }

      if (inputCluster) {
        const commitClusterInput = () => {
          const totalCls = (clusters && clusters.length > 0) ? clusters.length : 0;
          if (totalCls === 0) return;
          const raw = inputCluster.value.trim().replace(/^[cC#]/, '');
          let val = parseInt(raw, 10);
          if (isNaN(val)) {
            const cur = (typeof inspectedClusterId !== 'undefined' &&
              inspectedClusterId >= 0)
              ? inspectedClusterId
              : 0;
            inputCluster.value = cur;
            return;
          }
          val = Math.max(0, Math.min(totalCls - 1, val));
          inputCluster.value = val;
          if (typeof selectImageCluster === 'function') {
            selectImageCluster(val);
          }
        };

        inputCluster.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commitClusterInput();
            inputCluster.blur();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            const cur = (typeof inspectedClusterId !== 'undefined' &&
              inspectedClusterId >= 0)
              ? inspectedClusterId
              : 0;
            inputCluster.value = cur;
            inputCluster.blur();
          }
        });
        inputCluster.addEventListener('change', commitClusterInput);
      }

      if (btnPrevCluster) {
        btnPrevCluster.addEventListener('click', () => {
          const totalCls = (clusters && clusters.length > 0) ? clusters.length : 0;
          if (totalCls === 0) return;
          const cur = (typeof inspectedClusterId !== 'undefined' &&
            inspectedClusterId >= 0)
            ? inspectedClusterId
            : 0;
          const target = Math.max(0, cur - 1);
          if (typeof selectImageCluster === 'function') {
            selectImageCluster(target);
          }
        });
      }

      if (btnNextCluster) {
        btnNextCluster.addEventListener('click', () => {
          const totalCls = (clusters && clusters.length > 0) ? clusters.length : 0;
          if (totalCls === 0) return;
          const cur = (typeof inspectedClusterId !== 'undefined' &&
            inspectedClusterId >= 0)
            ? inspectedClusterId
            : 0;
          const target = Math.min(totalCls - 1, cur + 1);
          if (typeof selectImageCluster === 'function') {
            selectImageCluster(target);
          }
        });
      }

      if (btnAutoCluster) {
        btnAutoCluster.addEventListener('click', () => {
          autoClusterFollow = !autoClusterFollow;
          if (autoClusterFollow) {
            const total = (benchmarkDataset && benchmarkDataset.length > 0)
              ? benchmarkDataset.length
              : (typeof totalFrames !== 'undefined' ? totalFrames : 0);
            const effFrame = (typeof inspectedImageFrameIdx !== 'undefined' &&
              inspectedImageFrameIdx >= 0)
              ? inspectedImageFrameIdx
              : Math.max(0, total - 1);
            if (effFrame >= 0 && imageFrameAssignments &&
                imageFrameAssignments[effFrame] !== undefined &&
                imageFrameAssignments[effFrame] >= 0) {
              if (typeof selectImageCluster === 'function') {
                selectImageCluster(imageFrameAssignments[effFrame]);
              }
            } else if (typeof updateUI === 'function') {
              updateUI();
            }
          } else if (typeof updateUI === 'function') {
            updateUI();
          }
        });
      }

      // -------------------------------------------------------------
      // Member Inspector Event Handlers
      // -------------------------------------------------------------
      const sliderMember = document.getElementById('sliderImgMember');
      const inputMember = document.getElementById('inputImgMember');
      const btnPrevMember = document.getElementById('btnImgPrevMember');
      const btnNextMember = document.getElementById('btnImgNextMember');
      const btnAnchorMember = document.getElementById('btnImgAnchorMember');

      window.isDraggingImageMemberSlider = false;

      if (sliderMember) {
        const onMemberSliderUpdate = (e) => {
          const val = parseInt(e.target.value, 10);
          if (!isNaN(val) && val >= 1 &&
              typeof selectImageClusterMember === 'function') {
            selectImageClusterMember(val - 1);
          }
        };

        sliderMember.addEventListener('pointerdown', () => {
          window.isDraggingImageMemberSlider = true;
        });
        sliderMember.addEventListener('mousedown', () => {
          window.isDraggingImageMemberSlider = true;
        });
        window.addEventListener('pointerup', () => {
          window.isDraggingImageMemberSlider = false;
        });
        window.addEventListener('mouseup', () => {
          window.isDraggingImageMemberSlider = false;
        });

        sliderMember.addEventListener('input', onMemberSliderUpdate);
        sliderMember.addEventListener('change', onMemberSliderUpdate);
      }

      if (inputMember) {
        const commitMemberInput = () => {
          const curCls = (typeof inspectedClusterId !== 'undefined' &&
            inspectedClusterId >= 0)
            ? inspectedClusterId
            : 0;
          const curMembers = (typeof getClusterMembersList === 'function')
            ? getClusterMembersList(curCls)
            : ((imageClusterMembers && imageClusterMembers[curCls]) || []);
          const totalM = curMembers ? curMembers.length : 0;
          if (totalM === 0) return;

          const raw = inputMember.value.trim().replace(/^#/, '');
          let val = parseInt(raw, 10);
          if (isNaN(val)) {
            const cur = (typeof inspectedImageMemberIdx !== 'undefined')
              ? inspectedImageMemberIdx
              : 0;
            inputMember.value = cur + 1;
            return;
          }
          val = Math.max(1, Math.min(totalM, val));
          inputMember.value = val;
          if (typeof selectImageClusterMember === 'function') {
            selectImageClusterMember(val - 1);
          }
        };

        inputMember.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commitMemberInput();
            inputMember.blur();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            const cur = (typeof inspectedImageMemberIdx !== 'undefined')
              ? inspectedImageMemberIdx
              : 0;
            inputMember.value = cur + 1;
            inputMember.blur();
          }
        });
        inputMember.addEventListener('change', commitMemberInput);
      }

      if (btnPrevMember) {
        btnPrevMember.addEventListener('click', () => {
          const cur = (typeof inspectedImageMemberIdx !== 'undefined')
            ? inspectedImageMemberIdx
            : 0;
          const target = Math.max(0, cur - 1);
          if (typeof selectImageClusterMember === 'function') {
            selectImageClusterMember(target);
          }
        });
      }

      if (btnNextMember) {
        btnNextMember.addEventListener('click', () => {
          const curCls = (typeof inspectedClusterId !== 'undefined' &&
            inspectedClusterId >= 0)
            ? inspectedClusterId
            : 0;
          const curMembers = (typeof getClusterMembersList === 'function')
            ? getClusterMembersList(curCls)
            : ((imageClusterMembers && imageClusterMembers[curCls]) || []);
          const totalM = curMembers ? curMembers.length : 0;
          if (totalM === 0) return;
          const cur = (typeof inspectedImageMemberIdx !== 'undefined')
            ? inspectedImageMemberIdx
            : 0;
          const target = Math.min(totalM - 1, cur + 1);
          if (typeof selectImageClusterMember === 'function') {
            selectImageClusterMember(target);
          }
        });
      }

      if (btnAnchorMember) {
        btnAnchorMember.addEventListener('click', () => {
          if (typeof selectImageClusterMember === 'function') {
            selectImageClusterMember(0);
          }
        });
      }

      const selectSort = document.getElementById('selectImgClusterSort');
      if (selectSort) {
        selectSort.addEventListener('change', (e) => {
          imageClustersSortMode = e.target.value;
          if (typeof draw === 'function') draw();
          const label = (imageClustersSortMode === 'size_desc')
            ? '📊 Sorted by Cluster Size (Descending: Largest first)'
            : (imageClustersSortMode === 'size_asc')
              ? '📉 Sorted by Cluster Size (Ascending: Smallest first)'
              : '🔢 Sorted by Creation ID (Default)';
          if (typeof showToast === 'function') showToast(label);
        });
      }

      for (let q = 0; q < 4; q++) {
        const sel = document.getElementById(`selectImgQuad${q}`);
        if (sel) {
          sel.addEventListener('change', (e) => {
            const newMode = e.target.value;
            if (typeof setImagePanelViewMode === 'function') {
              setImagePanelViewMode(q, newMode);
            }
            const title = (typeof getImageViewTitle === 'function')
              ? getImageViewTitle(newMode)
              : newMode;
            if (typeof showToast === 'function') {
              showToast(`Q${q}: ${title}`);
            }
          });
        }
      }

      const btnShowAll = document.getElementById('btnImgShowAllPanels');
      if (btnShowAll) {
        btnShowAll.addEventListener('click', () => {
          maximizedQuad = null;
          syncImageQuadUI();
          if (typeof draw === 'function') draw();
          if (typeof showToast === 'function') showToast('⊞ Restored All 4 View Panels');
        });
      }

      const selectView = document.getElementById('selectImgViewPanel');
      if (selectView) {
        selectView.addEventListener('change', (e) => {
          const val = e.target.value;
          if (val === 'all') {
            maximizedQuad = null;
            if (typeof showToast === 'function') showToast('⊞ All 4 Panels (Split Grid)');
          } else if (val === '2_knn') {
            maximizedQuad = 2;
            if (typeof setImagePanelViewMode === 'function') {
              setImagePanelViewMode(2, 'knn');
            } else {
              imageQ2ViewMode = 'knn';
            }
            if (typeof showToast === 'function') showToast('⚡ Maximized Q2: k-NN Neighbors');
          } else {
            maximizedQuad = parseInt(val, 10);
            const title = (typeof getImagePanelViewMode === 'function' &&
                           typeof getImageViewTitle === 'function')
              ? getImageViewTitle(getImagePanelViewMode(maximizedQuad))
              : `Panel ${maximizedQuad}`;
            if (typeof showToast === 'function') {
              showToast(`🔍 Maximized Q${maximizedQuad}: ${title}`);
            }
          }
          syncImageQuadUI();
          if (typeof draw === 'function') draw();
        });
      }

      function setImageThumbSize(size) {
        const clamped = Math.max(36, Math.min(220, Math.round(size)));
        imageThumbSize = clamped;
        const lblSize = document.getElementById('lblImgThumbSize');
        if (lblSize) lblSize.textContent = `${clamped}px`;
        if (typeof draw === 'function') draw();
      }

      const btnThumbSmaller = document.getElementById('btnImgThumbSmaller');
      if (btnThumbSmaller) {
        btnThumbSmaller.addEventListener('click', () => {
          const cur = (typeof imageThumbSize !== 'undefined') ? imageThumbSize : 64;
          setImageThumbSize(cur - 16);
          if (typeof showToast === 'function') showToast(`🔍 Thumbnail Size: ${imageThumbSize}px`);
        });
      }

      const btnThumbBigger = document.getElementById('btnImgThumbBigger');
      if (btnThumbBigger) {
        btnThumbBigger.addEventListener('click', () => {
          const cur = (typeof imageThumbSize !== 'undefined') ? imageThumbSize : 64;
          setImageThumbSize(cur + 16);
          if (typeof showToast === 'function') showToast(`🔍 Thumbnail Size: ${imageThumbSize}px`);
        });
      }
    }

    function syncImageQuadUI() {
      const selectView = document.getElementById('selectImgViewPanel');
      const btnShowAll = document.getElementById('btnImgShowAllPanels');
      if (selectView) {
        if (maximizedQuad === null) {
          selectView.value = 'all';
        } else if (maximizedQuad === 2 &&
                   (typeof getImagePanelViewMode === 'function'
                     ? getImagePanelViewMode(2) === 'knn'
                     : imageQ2ViewMode === 'knn')) {
          selectView.value = '2_knn';
        } else {
          selectView.value = String(maximizedQuad);
        }

        // Update option labels with dynamic view titles
        if (typeof getImagePanelViewMode === 'function' &&
            typeof getImageViewTitle === 'function') {
          for (let q = 0; q < 4; q++) {
            const opt = selectView.querySelector(`option[value="${q}"]`);
            if (opt) {
              opt.textContent = `Q${q}: ${getImageViewTitle(getImagePanelViewMode(q))}`;
            }
          }
        }
      }
      if (btnShowAll) {
        if (maximizedQuad === null) {
          btnShowAll.style.background = 'rgba(56, 189, 248, 0.2)';
          btnShowAll.style.color = '#38bdf8';
          btnShowAll.style.borderColor = 'rgba(56, 189, 248, 0.4)';
          btnShowAll.textContent = '⊞ All 4 Panels';
        } else {
          btnShowAll.style.background = 'rgba(34, 197, 94, 0.25)';
          btnShowAll.style.color = '#4ade80';
          btnShowAll.style.borderColor = 'rgba(34, 197, 94, 0.6)';
          btnShowAll.textContent = '⊞ Show All Panels';
        }
      }

      // Sync the 4 quadrant dropdown select elements & update layout
      for (let q = 0; q < 4; q++) {
        const sel = document.getElementById(`selectImgQuad${q}`);
        if (sel && typeof getImagePanelViewMode === 'function') {
          const cur = getImagePanelViewMode(q);
          if (sel.value !== cur) {
            sel.value = cur;
          }
        }
      }
      if (typeof updateImageQuadDropdowns === 'function') {
        updateImageQuadDropdowns();
      }
    }
    window.syncImageQuadUI = syncImageQuadUI;

    // -------------------------------------------------------------------------
    // Quick Command Palette (Ctrl+K / Cmd+K)
    // -------------------------------------------------------------------------
