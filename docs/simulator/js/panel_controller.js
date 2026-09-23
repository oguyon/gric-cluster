/**
 * GRIC Simulator - panel_controller.js
 * Sidebar panel resizing, collapse, and expand layout controller.
 */

    // =========================================================================
    //  SIDEBAR PANELS RESIZING & COLLAPSE CONTROLLER (10 PANELS & 9 RESIZERS)
    // =========================================================================
    const panelConfigs = [
      { id: 'cardInputData', btnId: 'btnCollapseInputData', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardSettings', btnId: 'btnCollapseSettings', defaultFlex: 1.1, savedFlex: 1.1 },
      { id: 'cardDisplay', btnId: 'btnCollapseDisplay', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardDataFiles', btnId: 'btnCollapseDataFiles', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardCli', btnId: 'btnCollapseCli', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardResources', btnId: 'btnCollapseResources', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardTrace', btnId: 'btnCollapseTrace', defaultFlex: 1.1, savedFlex: 1.1 },
      { id: 'cardKnnSettings', btnId: 'btnCollapseKnnSettings', defaultFlex: 0.9, savedFlex: 0.9 },
      { id: 'cardKnnResources', btnId: 'btnCollapseKnnResources', defaultFlex: 0.9, savedFlex: 0.9 },
      { id: 'cardKnnTrace', btnId: 'btnCollapseKnnTrace', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardDimDensity', btnId: 'btnCollapseDimDensity', defaultFlex: 1.0, savedFlex: 1.0 },
      { id: 'cardReconstruction', btnId: 'btnCollapseReconstruction', defaultFlex: 1.1, savedFlex: 1.1 }
    ];

    function getExpandedCardAbove(cardIndex) {
      for (let i = cardIndex; i >= 0; i--) {
        const card = document.getElementById(panelConfigs[i].id);
        if (card && card.style.display !== 'none' && !card.classList.contains('collapsed')) return card;
      }
      return null;
    }

    function getExpandedCardBelow(cardIndex) {
      for (let i = cardIndex; i < panelConfigs.length; i++) {
        const card = document.getElementById(panelConfigs[i].id);
        if (card && card.style.display !== 'none' && !card.classList.contains('collapsed')) return card;
      }
      return null;
    }

    function updateResizersVisibility() {
      for (let i = 0; i < panelConfigs.length - 1; i++) {
        const resizer = document.getElementById(`resizer${i + 1}`);
        if (!resizer) continue;
        const topCard = getExpandedCardAbove(i);
        const botCard = getExpandedCardBelow(i + 1);
        const isVisible = (topCard !== null && botCard !== null);
        resizer.classList.toggle('hidden', !isVisible);
      }
    }

    function togglePanelCollapse(cardId) {
      const card = document.getElementById(cardId);
      if (!card) return;

      const isCurrentlyCollapsed = card.classList.contains('collapsed');
      const shouldCollapse = !isCurrentlyCollapsed;

      card.classList.toggle('collapsed', shouldCollapse);
      card.classList.toggle('expanded', !shouldCollapse);

      card.style.flex = '0 0 auto';
      card.style.height = 'auto';

      updateResizersVisibility();
    }
    window.togglePanelCollapse = togglePanelCollapse;

    function collapseAllControlPanels() {
      const cards = document.querySelectorAll('.side-panel .card');
      cards.forEach(card => {
        card.classList.add('collapsed');
        card.classList.remove('expanded');
        card.style.flex = '0 0 auto';
        card.style.height = 'auto';
      });
      updateResizersVisibility();
    }
    window.collapseAllControlPanels = collapseAllControlPanels;

    // Central Control Enablement & Dependency Synchronization
    function syncControlDependencies() {
      // 1. Target Selection Mode -> Shannon Entropy controls & Display Heatmap
      const isEntropy = (targetMode === 'entropy');
      const cardEntropy = document.getElementById('cardEntropySection');
      if (cardEntropy) {
        cardEntropy.classList.toggle('disabled', !isEntropy);
      }
      const optEntropyMap = document.getElementById('optEntropyMap');
      if (optEntropyMap) {
        optEntropyMap.classList.toggle('disabled', !isEntropy);
      }

      // 2. Prior & Subspace Acceleration Sliders
      const colTmMix = document.getElementById('colTmMix');
      const sliderTmMixEl = document.getElementById('sliderTmMix');
      const inputTmMixEl = document.getElementById('inputTmMix');
      if (colTmMix && sliderTmMixEl) {
        colTmMix.classList.toggle('disabled', !useTM);
        sliderTmMixEl.disabled = !useTM;
        if (inputTmMixEl) inputTmMixEl.disabled = !useTM;
      }

      const colPredHorizon = document.getElementById('colPredHorizon');
      const sliderPredHorizonEl = document.getElementById('sliderPredHorizon');
      const inputPredHorizonEl = document.getElementById('inputPredHorizon');
      if (colPredHorizon && sliderPredHorizonEl) {
        colPredHorizon.classList.toggle('disabled', !usePred);
        sliderPredHorizonEl.disabled = !usePred;
        if (inputPredHorizonEl) inputPredHorizonEl.disabled = !usePred;
      }

      const colMaxVis = document.getElementById('colMaxVis');
      const sliderMaxVisEl = document.getElementById('sliderMaxVis');
      const inputMaxVisEl = document.getElementById('inputMaxVis');
      if (colMaxVis && sliderMaxVisEl) {
        colMaxVis.classList.toggle('disabled', !useGprob);
        sliderMaxVisEl.disabled = !useGprob;
        if (inputMaxVisEl) inputMaxVisEl.disabled = !useGprob;
      }

      // -tiles -> -xtile toggle
      const optXTile = document.getElementById('optXTile');
      if (optXTile) {
        optXTile.classList.toggle('disabled', !useTiles);
      }
      const rowXTileDecay = document.getElementById('rowXTileDecay');
      if (rowXTileDecay) {
        rowXTileDecay.style.display = (useTiles && useXTile) ? 'flex' : 'none';
      }
      const rowSparseDccExtra = document.getElementById(
        'rowSparseDccExtra'
      );
      if (rowSparseDccExtra) {
        rowSparseDccExtra.style.display =
          useSparseDcc ? 'flex' : 'none';
      }

      // Leader shortcut & Bayes Sigma conditional rows
      const rowLeaderCutoff = document.getElementById('rowLeaderCutoff');
      if (rowLeaderCutoff) {
        rowLeaderCutoff.style.display = (isEntropy && entropyLeaderShortcut) ? 'flex' : 'none';
      }
      const rowBayesSigma = document.getElementById('rowBayesSigma');
      if (rowBayesSigma) {
        rowBayesSigma.style.display = useSoftBayesian ? 'flex' : 'none';
      }

      // 3. Cluster Budget & Eviction (-maxcl)
      const isBudgeted = (maxcl > 0);
      const sectionEviction = document.getElementById('sectionEvictionStrategy');
      if (sectionEviction) {
        sectionEviction.classList.toggle('disabled', !isBudgeted);
      }
      const rowDiscardFrac = document.getElementById('rowDiscardFrac');
      if (rowDiscardFrac) {
        rowDiscardFrac.style.display = (isBudgeted && maxclStrategy === 'discard') ? 'flex' : 'none';
      }

      // 4. Sample Truncated Gaussian Noise
      const hasNoise = (noiseSigma > 1e-6);
      const rowNoiseTrunc = document.getElementById('rowNoiseTrunc');
      const sliderNoiseTruncEl = document.getElementById('sliderNoiseTrunc');
      const inputNoiseTruncEl = document.getElementById('inputNoiseTrunc');
      if (rowNoiseTrunc && sliderNoiseTruncEl) {
        rowNoiseTrunc.classList.toggle('disabled', !hasNoise);
        sliderNoiseTruncEl.disabled = !hasNoise;
        if (inputNoiseTruncEl) inputNoiseTruncEl.disabled = !hasNoise;
      }

      // 5. 3D Mode Camera Presets
      const is3D = (currentDim >= 3);
      const card3DPresetsSide = document.getElementById('card3DPresetsSide');
      if (card3DPresetsSide) {
        card3DPresetsSide.classList.toggle('disabled', !is3D);
      }

      // 6. k-NN Post-Processing Controls & Sidebar Panels Visibility
      const btnToggleKnnModule = document.getElementById('btnToggleKnnModule');
      if (btnToggleKnnModule) {
        btnToggleKnnModule.classList.toggle('active', enableKnn);
        btnToggleKnnModule.classList.toggle('toggle-active', enableKnn);
        btnToggleKnnModule.innerHTML = enableKnn ? '✓ k-NN Enabled' : '⚡ Enable k-NN';
      }

      const btnToggleKnn = document.getElementById('btnToggleKnn');
      if (btnToggleKnn) {
        btnToggleKnn.classList.toggle('active', enableKnn);
        btnToggleKnn.classList.toggle('toggle-active', enableKnn);
      }

      const knnExpandedGroup = document.getElementById('knnExpandedGroup');
      if (knnExpandedGroup) {
        knnExpandedGroup.style.display = enableKnn ? 'inline-flex' : 'none';
      }

      const knnStatusBadgeTop = document.getElementById('knnStatusBadgeTop');
      if (knnStatusBadgeTop) {
        if (typeof isKnnComputing === 'undefined' || !isKnnComputing) {
          knnStatusBadgeTop.textContent = `k=${knnK} • ${knnDirection} • dt≥${knnDtmin}`;
        }
      }

      const optEnableKnnEl = document.getElementById('optEnableKnn');
      const knnControlsContainer = document.getElementById('knnControlsContainer');
      if (optEnableKnnEl) {
        optEnableKnnEl.classList.toggle('active', enableKnn);
      }
      if (knnControlsContainer) {
        knnControlsContainer.style.display = enableKnn ? 'flex' : 'none';
      }

      // Show or hide sidebar panels according to active sidebar mode
      if (typeof switchSidebarMode === 'function') {
        switchSidebarMode(activeSidebarMode || 'clustering');
      }

      if (typeof updateResizersVisibility === 'function') {
        updateResizersVisibility();
      }

      if (typeof updateKnnButtonUI === 'function') {
        updateKnnButtonUI(typeof isKnnComputing !== 'undefined' ? isKnnComputing : false);
      } else if (typeof window !== 'undefined' && typeof window.updateKnnButtonUI === 'function') {
        window.updateKnnButtonUI(typeof isKnnComputing !== 'undefined' ? isKnnComputing : false);
      }

      if (typeof updateClusteringButtonUI === 'function') {
        updateClusteringButtonUI();
      } else if (typeof window !== 'undefined' && typeof window.updateClusteringButtonUI === 'function') {
        window.updateClusteringButtonUI();
      }

      // 7. Ball seed & shuffle options sync
      const rowBallSeedOptions = document.getElementById('rowBallSeedOptions');
      if (rowBallSeedOptions) {
        const isBallBench = typeof isImageBenchmark === 'function' &&
                            isImageBenchmark(currentBenchmark);
        rowBallSeedOptions.style.display = isBallBench ? 'flex' : 'none';
      }
      const chkBallSeedEl = document.getElementById('chkRandomBallSeed');
      if (chkBallSeedEl && typeof randomBallSeed !== 'undefined') {
        chkBallSeedEl.checked = !!randomBallSeed;
      }
      const chkShuffleEl = document.getElementById('chkShuffleFrames');
      if (chkShuffleEl && typeof shuffleFrames !== 'undefined') {
        chkShuffleEl.checked = !!shuffleFrames;
      }

      // 8. SQ16 / EQ16 Ratio & Memoization controls
      const rowSq16Ratio = document.getElementById('rowSq16Ratio');
      if (rowSq16Ratio) {
        rowSq16Ratio.style.display = ((typeof clusterUseSq16 === 'boolean' && clusterUseSq16) ||
                                      (typeof clusterUseEq16 === 'boolean' && clusterUseEq16)) ? 'flex' : 'none';
      }
      const optSq8 = document.getElementById('optSq8');
      if (optSq8 && typeof clusterUseSq8 === 'boolean') {
        optSq8.classList.toggle('active', clusterUseSq8);
      }
      const optSq16 = document.getElementById('optSq16');
      if (optSq16 && typeof clusterUseSq16 === 'boolean') {
        optSq16.classList.toggle('active', clusterUseSq16);
      }
      const optEq16 = document.getElementById('optEq16');
      if (optEq16 && typeof clusterUseEq16 === 'boolean') {
        optEq16.classList.toggle('active', clusterUseEq16);
      }
      const optEq16Adc = document.getElementById('optEq16Adc');
      if (optEq16Adc && typeof clusterUseEq16Adc === 'boolean') {
        optEq16Adc.classList.toggle('active', clusterUseEq16 && clusterUseEq16Adc);
        optEq16Adc.style.display = (typeof clusterUseEq16 === 'boolean' && clusterUseEq16) ? 'inline-flex' : 'none';
      }
      const badgeQuantRatio = document.getElementById('badgeQuantRatio');
      if (badgeQuantRatio) {
        badgeQuantRatio.textContent = (typeof clusterUseEq16 === 'boolean' && clusterUseEq16) ? '--eq16-ratio' : '--sq16-ratio';
      }
      const optMemo = document.getElementById('optMemo');
      if (optMemo && typeof clusterUseMemo === 'boolean') {
        optMemo.classList.toggle('active', clusterUseMemo);
      }
      const inputSq16Ratio = document.getElementById('inputSq16Ratio');
      const sliderSq16Ratio = document.getElementById('sliderSq16Ratio');
      const activeRatio = (typeof clusterUseEq16 === 'boolean' && clusterUseEq16)
        ? clusterEq16Ratio : clusterSq16Ratio;
      if (inputSq16Ratio && typeof activeRatio === 'number') {
        inputSq16Ratio.value = activeRatio.toFixed(3);
      }
      if (sliderSq16Ratio && typeof activeRatio === 'number') {
        sliderSq16Ratio.value = activeRatio;
      }

      const rowKnnSq16Ratio = document.getElementById('rowKnnSq16Ratio');
      if (rowKnnSq16Ratio) {
        rowKnnSq16Ratio.style.display = (enableKnn &&
          ((typeof knnUseSq16 === 'boolean' && knnUseSq16) ||
           (typeof knnUseEq16 === 'boolean' && knnUseEq16))) ? 'flex' : 'none';
      }
      const btnKnnEq16 = document.getElementById('btnKnnEq16');
      if (btnKnnEq16 && typeof knnUseEq16 === 'boolean') {
        btnKnnEq16.classList.toggle('toggle-active', knnUseEq16);
        btnKnnEq16.classList.toggle('toggle-cyan', knnUseEq16);
        btnKnnEq16.classList.toggle('active', knnUseEq16);
      }
      const btnKnnEq16Adc = document.getElementById('btnKnnEq16Adc');
      if (btnKnnEq16Adc && typeof knnUseEq16Adc === 'boolean') {
        btnKnnEq16Adc.classList.toggle('toggle-active', knnUseEq16 && knnUseEq16Adc);
        btnKnnEq16Adc.classList.toggle('toggle-cyan', knnUseEq16 && knnUseEq16Adc);
        btnKnnEq16Adc.classList.toggle('active', knnUseEq16 && knnUseEq16Adc);
        btnKnnEq16Adc.style.display = (typeof knnUseEq16 === 'boolean' && knnUseEq16) ? 'inline-flex' : 'none';
      }
      const badgeKnnQuantRatio = document.getElementById('badgeKnnQuantRatio');
      if (badgeKnnQuantRatio) {
        badgeKnnQuantRatio.textContent = (typeof knnUseEq16 === 'boolean' && knnUseEq16) ? '--eq16-ratio' : '--sq16-ratio';
      }
      const btnKnnMemo = document.getElementById('btnKnnMemo');
      if (btnKnnMemo && typeof knnUseMemo === 'boolean') {
        btnKnnMemo.classList.toggle('toggle-active', knnUseMemo);
        btnKnnMemo.classList.toggle('toggle-cyan', knnUseMemo);
        btnKnnMemo.classList.toggle('active', knnUseMemo);
      }
      const inputKnnSq16Ratio = document.getElementById('inputKnnSq16Ratio');
      const sliderKnnSq16Ratio = document.getElementById('sliderKnnSq16Ratio');
      const activeKnnRatio = (typeof knnUseEq16 === 'boolean' && knnUseEq16)
        ? (typeof knnEq16Ratio === 'number' ? knnEq16Ratio : 0.05)
        : (typeof knnSq16Ratio === 'number' ? knnSq16Ratio : 0.05);
      if (inputKnnSq16Ratio) {
        inputKnnSq16Ratio.value = activeKnnRatio.toFixed(3);
      }
      if (sliderKnnSq16Ratio) {
        sliderKnnSq16Ratio.value = activeKnnRatio;
      }

      if (typeof updateCliCommand === 'function') {
        updateCliCommand();
      }
    }

    window.syncControlDependencies = syncControlDependencies;
    syncControlDependencies();

    function initSidebarResizers() {
      function setupResizer(resizerId, getTopCard, getBottomCard) {
        const resizer = document.getElementById(resizerId);
        if (!resizer) return;

        resizer.addEventListener('mousedown', (e) => {
          e.preventDefault();
          const topCard = getTopCard();
          const bottomCard = getBottomCard();
          if (!topCard || !bottomCard) return;
          if (topCard.classList.contains('collapsed') || bottomCard.classList.contains('collapsed')) return;

          resizer.classList.add('dragging');
          document.body.style.cursor = 'row-resize';
          document.body.style.userSelect = 'none';

          const startY = e.clientY;
          const topRect = topCard.getBoundingClientRect();
          const bottomRect = bottomCard.getBoundingClientRect();
          const totalHeight = topRect.height + bottomRect.height;
          const minH = 65;

          const topCfg = panelConfigs.find(p => p.id === topCard.id);
          const botCfg = panelConfigs.find(p => p.id === bottomCard.id);
          const initialCombinedFlex = (topCfg ? topCfg.savedFlex : 1.0) + (botCfg ? botCfg.savedFlex : 1.0);

          function onMouseMove(moveEvent) {
            const deltaY = moveEvent.clientY - startY;
            let newTopH = topRect.height + deltaY;
            let newBottomH = bottomRect.height - deltaY;

            if (newTopH < minH) {
              newTopH = minH;
              newBottomH = totalHeight - minH;
            } else if (newBottomH < minH) {
              newBottomH = minH;
              newTopH = totalHeight - minH;
            }

            const topRatio = newTopH / totalHeight;
            const bottomRatio = newBottomH / totalHeight;

            const flexTop = (topRatio * initialCombinedFlex).toFixed(3);
            const flexBottom = (bottomRatio * initialCombinedFlex).toFixed(3);

            topCard.style.flex = `${flexTop} 1 0px`;
            bottomCard.style.flex = `${flexBottom} 1 0px`;

            if (topCfg) topCfg.savedFlex = parseFloat(flexTop);
            if (botCfg) botCfg.savedFlex = parseFloat(flexBottom);
          }

          function onMouseUp() {
            resizer.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
          }

          window.addEventListener('mousemove', onMouseMove);
          window.addEventListener('mouseup', onMouseUp);
        });
      }

      for (let i = 0; i < panelConfigs.length - 1; i++) {
        const resizerId = `resizer${i + 1}`;
        const topIdx = i;
        const botIdx = i + 1;
        setupResizer(
          resizerId,
          () => getExpandedCardAbove(topIdx),
          () => getExpandedCardBelow(botIdx)
        );
      }

      updateResizersVisibility();
    }

    function initLayoutResizer() {
      const layoutResizer = document.getElementById('layoutResizer');
      const sidePanel = document.getElementById('sidePanel');
      const mainLayout = document.getElementById('mainLayout');
      if (!layoutResizer || !sidePanel || !mainLayout) return;

      let isDragging = false;
      let startX = 0;
      let startWidth = 0;

      layoutResizer.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startWidth = sidePanel.getBoundingClientRect().width;
        layoutResizer.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();

        function onMouseMove(moveEvent) {
          if (!isDragging) return;
          const deltaX = startX - moveEvent.clientX; // Moving left expands sidebar
          const layoutRect = mainLayout.getBoundingClientRect();
          const minW = 280;
          const maxW = Math.max(minW, layoutRect.width - 320);
          const newWidth = Math.round(Math.max(minW, Math.min(maxW, startWidth + deltaX)));

          sidePanel.style.width = `${newWidth}px`;
          sidePanel.style.minWidth = `${newWidth}px`;
          resizeCanvas();
        }

        function onMouseUp() {
          if (isDragging) {
            isDragging = false;
            layoutResizer.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
            resizeCanvas();
          }
        }

        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
      });
    }

    // Ensure hovered cluster highlight is strictly active only when mouse pointer is over the cluster table
    const candidateContainerEl = document.getElementById('candidateContainer');
    if (candidateContainerEl) {
      candidateContainerEl.addEventListener('mouseleave', () => {
        setHoveredCluster(-1);
      });
    }

    document.addEventListener('mousemove', (e) => {
      if (hoveredClusterId !== -1) {
        const inClusterRow = e.target && e.target.closest && e.target.closest('#candidateContainer .cluster-row');
        if (!inClusterRow) {
          setHoveredCluster(-1);
        }
      }
    });

    // Distance Trace Curve Hover Handlers
    const distCanvas = document.getElementById('distCurvesCanvas');
    if (distCanvas) {
      distCanvas.addEventListener('mousemove', (e) => {
        const n = distHistoryDFC.length;
        if (n === 0) return;
        const rect = distCanvas.getBoundingClientRect();
        const mouseX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
        hoverDistIndex = n > 1 ? Math.max(0, Math.min(n - 1, Math.round((mouseX / rect.width) * (n - 1)))) : 0;
        drawDistCurves();
      });
      distCanvas.addEventListener('mouseleave', () => {
        hoverDistIndex = null;
        drawDistCurves();
      });
    }

    const distAvgCanvas = document.getElementById('distAvgCurvesCanvas');
    if (distAvgCanvas) {
      distAvgCanvas.addEventListener('mousemove', (e) => {
        const n = distHistoryDFC.length;
        if (n === 0) return;
        const rect = distAvgCanvas.getBoundingClientRect();
        const mouseX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
        hoverDistAvgIndex = n > 1 ? Math.max(0, Math.min(n - 1, Math.round((mouseX / rect.width) * (n - 1)))) : 0;
        drawDistAvgCurves();
      });
      distAvgCanvas.addEventListener('mouseleave', () => {
        hoverDistAvgIndex = null;
        drawDistAvgCurves();
      });
    }

    // Transition Matrix Heatmap Hover & Touch Handlers
    function setupTMCanvasListeners() {
      const cvs = document.getElementById('tmHeatmapCanvas');
      if (!cvs) return;

      function handleTMInteraction(clientX, clientY) {
        const layout = cvs._tmLayout;
        if (!layout || layout.K === 0 || transitionCounts.length === 0) return;
        const rect = cvs.getBoundingClientRect();
        const mouseX = (clientX - rect.left) * (cvs.width / rect.width);
        const mouseY = (clientY - rect.top) * (cvs.height / rect.height);

        const { gridX, gridY, S, cw, ch, K } = layout;

        if (mouseX >= gridX && mouseX < gridX + S && mouseY >= gridY && mouseY < gridY + S) {
          const col = Math.max(0, Math.min(K - 1, Math.floor((mouseX - gridX) / cw)));
          const row = Math.max(0, Math.min(K - 1, Math.floor((mouseY - gridY) / ch)));

          hoveredTMCell = { i: row, j: col };
          let sum = 0;
          if (transitionCounts[row]) {
            for (let k = 0; k < K; k++) sum += transitionCounts[row][k] || 0;
          }
          const cnt = (transitionCounts[row] && transitionCounts[row][col]) ? transitionCounts[row][col] : 0;
          const prob = sum > 0 ? (cnt / sum) : 0.0;

          const desc = `C${row} → C${col}: Count=${cnt}, P=${(prob * 100).toFixed(1)}% (Row total: ${sum})`;
          const tipEl = document.getElementById('tmCellTooltip');
          if (tipEl) tipEl.innerText = desc;
          drawTransitionMatrix('tmHeatmapCanvas', false);
          draw();
        } else {
          if (hoveredTMCell !== null) {
            hoveredTMCell = null;
            const tipEl = document.getElementById('tmCellTooltip');
            if (tipEl) tipEl.innerText = 'Hover over any cell to inspect transition details';
            drawTransitionMatrix('tmHeatmapCanvas', false);
            draw();
          }
        }
      }

      function clearTMInteraction() {
        if (hoveredTMCell !== null) {
          hoveredTMCell = null;
          const tipEl = document.getElementById('tmCellTooltip');
          if (tipEl) tipEl.innerText = 'Hover over any cell to inspect transition details';
          drawTransitionMatrix('tmHeatmapCanvas', false);
          draw();
        }
      }

      cvs.addEventListener('mousemove', (e) => handleTMInteraction(e.clientX, e.clientY));
      cvs.addEventListener('mouseleave', clearTMInteraction);

      cvs.addEventListener('touchstart', (e) => {
        if (e.touches.length === 1) {
          handleTMInteraction(e.touches[0].clientX, e.touches[0].clientY);
        }
      }, { passive: true });

      cvs.addEventListener('touchmove', (e) => {
        if (e.touches.length === 1) {
          handleTMInteraction(e.touches[0].clientX, e.touches[0].clientY);
        }
      }, { passive: true });

      cvs.addEventListener('touchend', clearTMInteraction);
      cvs.addEventListener('touchcancel', clearTMInteraction);
    }

    setupTMCanvasListeners();

    // CLI Command Display
    function updateCliCommand() {
      const el = document.getElementById('cliCommandOutput');
      if (!el) return;
      const cmd = buildCliCommand();
      el.textContent = cmd;
    }

    // Update CLI and WASM build info
    updateCliCommand();
    const hashEl = document.getElementById('wasmBuildHash');
    if (hashEl && typeof GricWasm !== 'undefined' && GricWasm.isReady && GricWasm.isReady()) {
      hashEl.textContent = GricWasm.getVersion();
    }

    // Copy CLI command to clipboard
    const btnCopy = document.getElementById('btnCopyCli');
    if (btnCopy) {
      btnCopy.addEventListener('click', () => {
        const el = document.getElementById('cliCommandOutput');
        if (!el) return;
        navigator.clipboard.writeText(el.textContent)
          .then(() => {
            btnCopy.textContent = '✅ Copied';
            setTimeout(() => {
              btnCopy.textContent = '📋 Copy';
            }, 1500);
          });
      });
    }

