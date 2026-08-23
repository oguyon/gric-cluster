/**
 * GRIC Simulator - tooltip.js
 * Part of the GRIC Interactive Algorithm Simulator
 */

//  0. RICH HOVER TOOLTIP HELP POPUP CONTROLLER
    // =========================================================================
    const richTooltipEl = document.createElement('div');
    richTooltipEl.id = 'richTooltip';
    richTooltipEl.className = 'rich-tooltip';
    document.body.appendChild(richTooltipEl);

    let activeTooltipTarget = null;
    let tooltipTimeout = null;
    let tooltipsEnabled = true;

    function setTooltipsEnabled(enabled) {
      tooltipsEnabled = !!enabled;
      if (!tooltipsEnabled) {
        hideRichTooltip();
      }
      const btn = document.getElementById('btnToggleTooltips');
      if (btn) {
        btn.classList.toggle('active', tooltipsEnabled);
        btn.classList.toggle('toggle-active', tooltipsEnabled);
      }
      const optOn = document.getElementById('optTooltipsOn');
      const optOff = document.getElementById('optTooltipsOff');
      if (optOn) optOn.classList.toggle('toggle-active', tooltipsEnabled);
      if (optOff) optOff.classList.toggle('toggle-active', !tooltipsEnabled);
    }

    function toggleTooltips() {
      setTooltipsEnabled(!tooltipsEnabled);
      if (typeof showToast === 'function') {
        showToast(tooltipsEnabled ? '💡 Help Hover Tooltips: ON' : '💡 Help Hover Tooltips: OFF');
      }
    }

    window.tooltipsEnabled = tooltipsEnabled;
    window.setTooltipsEnabled = setTooltipsEnabled;
    window.toggleTooltips = toggleTooltips;

    function showRichTooltip(target, e) {
      if (!tooltipsEnabled) return;

      const title = target.getAttribute('data-tooltip-title') || target.getAttribute('title');
      const desc = target.getAttribute('data-tooltip-desc');
      const badge = target.getAttribute('data-tooltip-badge');
      const color = target.getAttribute('data-tooltip-color') || 'cyan';

      if (!title && !desc) return;

      activeTooltipTarget = target;
      if (tooltipTimeout) clearTimeout(tooltipTimeout);

      richTooltipEl.innerHTML = `
        <div class="rich-tooltip-header">
          <span class="rich-tooltip-title ${color}">${title || ''}</span>
          ${badge ? `<span class="rich-tooltip-badge">${badge}</span>` : ''}
        </div>
        ${desc ? `<div class="rich-tooltip-desc">${desc}</div>` : ''}
      `;

      richTooltipEl.style.display = 'block';
      positionRichTooltip(e);
      requestAnimationFrame(() => {
        richTooltipEl.classList.add('visible');
      });
    }

    function positionRichTooltip(e) {
      if (!richTooltipEl || richTooltipEl.style.display === 'none') return;

      const offset = 14;
      const tooltipRect = richTooltipEl.getBoundingClientRect();
      const winW = window.innerWidth;
      const winH = window.innerHeight;

      let left = e.clientX + offset;
      let top = e.clientY + offset;

      // Flip left if overflowing right edge
      if (left + tooltipRect.width > winW - 12) {
        left = e.clientX - tooltipRect.width - offset;
      }

      // Flip up if overflowing bottom edge
      if (top + tooltipRect.height > winH - 12) {
        top = e.clientY - tooltipRect.height - offset;
      }

      // Safe screen boundary clamps
      if (left < 10) left = 10;
      if (top < 10) top = 10;

      richTooltipEl.style.left = `${left}px`;
      richTooltipEl.style.top = `${top}px`;
    }

    function hideRichTooltip() {
      if (tooltipTimeout) clearTimeout(tooltipTimeout);
      activeTooltipTarget = null;
      richTooltipEl.classList.remove('visible');
      tooltipTimeout = setTimeout(() => {
        if (!activeTooltipTarget) {
          richTooltipEl.style.display = 'none';
        }
      }, 120);
    }

    document.addEventListener('mouseover', (e) => {
      const target = e.target.closest('[data-tooltip-title], [data-tooltip-desc]');
      if (target) {
        showRichTooltip(target, e);
      }
    });

    document.addEventListener('mousemove', (e) => {
      if (activeTooltipTarget) {
        positionRichTooltip(e);
      }
    });

    document.addEventListener('mouseout', (e) => {
      if (activeTooltipTarget) {
        const target = e.target.closest('[data-tooltip-title], [data-tooltip-desc]');
        if (!target || target !== activeTooltipTarget) {
          hideRichTooltip();
        }
      }
    });

    document.addEventListener('click', () => {
      hideRichTooltip();
    });

    document.addEventListener('touchstart', () => {
      hideRichTooltip();
    }, { passive: true });

    // =========================================================================
