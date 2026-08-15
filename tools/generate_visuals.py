#!/usr/bin/env python3
"""
GRIC Visual Generator
Generates publication-quality SVG diagrams, PNG exports, and animated MP4 video
explaining the GRIC clustering algorithm and its configuration options.
"""

import os
import sys
import math
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Master Pipeline SVG
# -----------------------------------------------------------------------------
def generate_master_pipeline_svg():
    svg_path = os.path.join(OUTPUT_DIR, "gric_master_pipeline.svg")
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 860" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <linearGradient id="gradHeader" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#334155"/>
    </linearGradient>
    <linearGradient id="gradStep1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#eff6ff"/>
      <stop offset="100%" stop-color="#dbeafe"/>
    </linearGradient>
    <linearGradient id="gradStep2" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fdf4ff"/>
      <stop offset="100%" stop-color="#fae8ff"/>
    </linearGradient>
    <linearGradient id="gradStep3" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fff7ed"/>
      <stop offset="100%" stop-color="#ffedd5"/>
    </linearGradient>
    <linearGradient id="gradStep4" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f0fdf4"/>
      <stop offset="100%" stop-color="#dcfce7"/>
    </linearGradient>
    <linearGradient id="gradStep5" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fef2f2"/>
      <stop offset="100%" stop-color="#fee2e2"/>
    </linearGradient>
    
    <filter id="cardShadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>

    <marker id="arr" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#475569"/>
    </marker>
    <marker id="arr-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#16a34a"/>
    </marker>
    <marker id="arr-red" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#dc2626"/>
    </marker>
    <marker id="arr-blue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#0284c7"/>
    </marker>
  </defs>

  <!-- Canvas Background -->
  <rect width="1200" height="860" fill="#f8fafc"/>

  <!-- Title Header Bar -->
  <rect x="50" y="24" width="1100" height="72" rx="12" fill="url(#gradHeader)" filter="url(#cardShadow)"/>
  <text x="80" y="54" fill="#ffffff" font-size="21" font-weight="700" letter-spacing="0.4">GRIC Sequential Frame Ingestion &amp; Clustering Pipeline</text>
  <text x="80" y="76" fill="#94a3b8" font-size="13" font-weight="400">High-speed distance-based geometric clustering with active pruning and target scheduling</text>
  <rect x="990" y="44" width="135" height="32" rx="16" fill="#3b82f6"/>
  <text x="1057" y="65" fill="#ffffff" font-size="12" font-weight="600" text-anchor="middle">5-Stage Pipeline</text>

  <!-- Left Column: Sequential Stages 1 to 4 -->

  <!-- STAGE 1: Incoming Frame & Priors -->
  <g transform="translate(60, 120)">
    <rect width="470" height="112" rx="10" fill="url(#gradStep1)" stroke="#bfdbfe" stroke-width="1.5" filter="url(#cardShadow)"/>
    <rect x="14" y="14" width="28" height="28" rx="6" fill="#2563eb"/>
    <text x="28" y="33" fill="#ffffff" font-size="14" font-weight="700" text-anchor="middle">1</text>
    <text x="52" y="33" fill="#1e3a8a" font-size="16" font-weight="700">Frame Ingestion &amp; Prior Modeling</text>
    <text x="52" y="55" fill="#334155" font-size="12.5">Ingests frame <tspan font-family="monospace" font-weight="600">f_i</tspan>. Normalizes recency frequency score.</text>
    <text x="52" y="73" fill="#334155" font-size="12.5">Blends Markov transitions and multi-step trajectory sequences.</text>
    
    <!-- Option Badges -->
    <g transform="translate(52, 84)">
      <rect width="68" height="18" rx="9" fill="#2563eb"/>
      <text x="34" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-dprob</text>
      
      <rect x="74" width="48" height="18" rx="9" fill="#7c3aed"/>
      <text x="98" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-tm</text>
      
      <rect x="128" width="55" height="18" rx="9" fill="#9333ea"/>
      <text x="155.5" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-pred</text>
      
      <rect x="189" width="70" height="18" rx="9" fill="#0284c7"/>
      <text x="224" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-stream</text>
    </g>
  </g>

  <!-- Flow 1 -> 2 -->
  <line x1="295" y1="232" x2="295" y2="258" stroke="#475569" stroke-width="2" marker-end="url(#arr)"/>

  <!-- STAGE 2: Target Selection (Greedy vs Entropy) -->
  <g transform="translate(60, 258)">
    <rect width="470" height="114" rx="10" fill="url(#gradStep2)" stroke="#f5d0fe" stroke-width="1.5" filter="url(#cardShadow)"/>
    <rect x="14" y="14" width="28" height="28" rx="6" fill="#a855f7"/>
    <text x="28" y="33" fill="#ffffff" font-size="14" font-weight="700" text-anchor="middle">2</text>
    <text x="52" y="33" fill="#581c87" font-size="16" font-weight="700">Target Candidate Selection</text>
    <text x="52" y="55" fill="#334155" font-size="12.5"><tspan font-weight="600">Greedy:</tspan> Select highest posterior probability <tspan font-family="monospace">P(c_j)</tspan>.</text>
    <text x="52" y="73" fill="#334155" font-size="12.5"><tspan font-weight="600">Entropy:</tspan> Minimize expected posterior Shannon entropy <tspan font-family="monospace">H(X)</tspan>.</text>
    
    <!-- Option Badges -->
    <g transform="translate(52, 85)">
      <rect width="70" height="18" rx="9" fill="#9333ea"/>
      <text x="35" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-entropy</text>
      
      <rect x="76" width="95" height="18" rx="9" fill="#a855f7"/>
      <text x="123.5" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-entropy_fast</text>
      
      <rect x="177" width="95" height="18" rx="9" fill="#c026d3"/>
      <text x="224.5" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-entropy_gate</text>
      
      <rect x="278" width="95" height="18" rx="9" fill="#7c3aed"/>
      <text x="325.5" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-entropy_max</text>
    </g>
  </g>

  <!-- Flow 2 -> 3 -->
  <line x1="295" y1="372" x2="295" y2="398" stroke="#475569" stroke-width="2" marker-end="url(#arr)"/>

  <!-- STAGE 3: Distance Metric Computation -->
  <g transform="translate(60, 398)">
    <rect width="470" height="112" rx="10" fill="url(#gradStep3)" stroke="#fed7aa" stroke-width="1.5" filter="url(#cardShadow)"/>
    <rect x="14" y="14" width="28" height="28" rx="6" fill="#ea580c"/>
    <text x="28" y="33" fill="#ffffff" font-size="14" font-weight="700" text-anchor="middle">3</text>
    <text x="52" y="33" fill="#7c2d12" font-size="16" font-weight="700">Distance Metric Computation</text>
    <text x="52" y="55" fill="#334155" font-size="12.5">Compute Euclidean distance <tspan font-family="monospace" font-weight="600">d = ||f_i - anchor(c_j)||_2</tspan>.</text>
    <text x="52" y="73" fill="#334155" font-size="12.5">Optimized with AVX2 SIMD intrinsics &amp; OpenMP threads.</text>
    
    <!-- Option Badges -->
    <g transform="translate(52, 84)">
      <rect width="52" height="18" rx="9" fill="#ea580c"/>
      <text x="26" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">rlim</text>
      
      <rect x="58" width="80" height="18" rx="9" fill="#f97316"/>
      <text x="98" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-auto_rlim</text>
      
      <rect x="144" width="60" height="18" rx="9" fill="#c2410c"/>
      <text x="174" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-ncpu</text>
      
      <rect x="210" width="75" height="18" rx="9" fill="#d97706"/>
      <text x="247.5" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-scandist</text>
    </g>
  </g>

  <!-- Flow 3 -> 4 Decision Junction -->
  <line x1="295" y1="510" x2="295" y2="538" stroke="#475569" stroke-width="2" marker-end="url(#arr)"/>

  <!-- STAGE 4: Match / Prune Decision & Geometry -->
  <g transform="translate(60, 538)">
    <rect width="470" height="135" rx="10" fill="url(#gradStep4)" stroke="#bbf7d0" stroke-width="1.5" filter="url(#cardShadow)"/>
    <rect x="14" y="14" width="28" height="28" rx="6" fill="#16a34a"/>
    <text x="28" y="33" fill="#ffffff" font-size="14" font-weight="700" text-anchor="middle">4</text>
    <text x="52" y="33" fill="#14532d" font-size="16" font-weight="700">Distance Check &amp; Geometric Pruning</text>
    <text x="52" y="56" fill="#334155" font-size="12.5">If <tspan font-family="monospace" font-weight="700" fill="#16a34a">d &lt;= rlim</tspan>: <tspan font-weight="600">Match Found!</tspan> Stop search &amp; assign to cluster.</text>
    <text x="52" y="74" fill="#334155" font-size="12.5">If <tspan font-family="monospace" font-weight="700" fill="#dc2626">d &gt; rlim</tspan>: Prune remaining candidates via triangle inequality</text>
    <text x="52" y="92" fill="#334155" font-size="12.5">(3-point, 4-point, 5-point, and Sparse DCC bounds).</text>
    
    <!-- Option Badges -->
    <g transform="translate(52, 104)">
      <rect width="48" height="18" rx="9" fill="#16a34a"/>
      <text x="24" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-te4</text>
      
      <rect x="54" width="48" height="18" rx="9" fill="#15803d"/>
      <text x="78" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-te5</text>
      
      <rect x="108" width="92" height="18" rx="9" fill="#047857"/>
      <text x="154" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-sparse_dcc</text>
      
      <rect x="206" width="58" height="18" rx="9" fill="#0d9488"/>
      <text x="235" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-gprob</text>

      <rect x="270" width="105" height="18" rx="9" fill="#0891b2"/>
      <text x="322.5" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-soft_bayesian</text>
    </g>
  </g>

  <!-- Left Loop-Back Arrow: Mismatch -> Re-schedule Candidate -->
  <path d="M 60 610 L 25 610 L 25 315 L 60 315" fill="none" stroke="#ea580c" stroke-width="2.5" stroke-dasharray="6,4" marker-end="url(#arr)"/>
  <rect x="8" y="445" width="34" height="60" rx="4" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="25" y="475" fill="#ea580c" font-size="10.5" font-weight="700" text-anchor="middle" transform="rotate(-90 25,475)">Prune &amp; Next</text>

  <!-- Right Column: Stage 5 (Top) & Multi-Tile JTF (Bottom) -->

  <!-- STAGE 5: Assignment or New Anchor (Right Column Top) -->
  <g transform="translate(660, 120)">
    <rect width="490" height="255" rx="10" fill="url(#gradStep5)" stroke="#fecaca" stroke-width="1.5" filter="url(#cardShadow)"/>
    <rect x="14" y="14" width="28" height="28" rx="6" fill="#dc2626"/>
    <text x="28" y="33" fill="#ffffff" font-size="14" font-weight="700" text-anchor="middle">5</text>
    <text x="52" y="33" fill="#7f1d1d" font-size="16" font-weight="700">Cluster Assignment &amp; Topology Updates</text>
    
    <!-- Sub-branch A: Match -->
    <rect x="52" y="48" width="422" height="78" rx="8" fill="#ffffff" stroke="#86efac" stroke-width="1.2"/>
    <circle cx="70" cy="70" r="8" fill="#16a34a"/>
    <text x="70" y="74" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">A</text>
    <text x="88" y="70" fill="#14532d" font-size="13.5" font-weight="700">Match Confirmed: Assign to Cluster c_j</text>
    <text x="88" y="90" fill="#475569" font-size="11.5">Increment frequency score (+dprob), update Markov matrix (+tm),</text>
    <text x="88" y="108" fill="#475569" font-size="11.5">and record frame visitor measurements for spatial learning (-gprob).</text>

    <!-- Sub-branch B: New Anchor -->
    <rect x="52" y="136" width="422" height="102" rx="8" fill="#ffffff" stroke="#fca5a5" stroke-width="1.2"/>
    <circle cx="70" cy="158" r="8" fill="#dc2626"/>
    <text x="70" y="162" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">B</text>
    <text x="88" y="158" fill="#7f1d1d" font-size="13.5" font-weight="700">All Candidates Excluded: Spawn Anchor c_{K+1}</text>
    <text x="88" y="178" fill="#475569" font-size="11.5">Current frame becomes the exemplar anchor for a new cluster.</text>
    <text x="88" y="196" fill="#475569" font-size="11.5">Compute row in DCC matrix (or sparse bounds).</text>
    
    <!-- Capacity Options Inside Card B -->
    <g transform="translate(88, 208)">
      <rect width="60" height="18" rx="9" fill="#dc2626"/>
      <text x="30" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-maxcl</text>
      
      <rect x="66" width="90" height="18" rx="9" fill="#b91c1c"/>
      <text x="111" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-discard_frac</text>
      <text x="165" y="13" fill="#64748b" font-size="10.5">(Prunes oldest inactive clusters)</text>
    </g>
  </g>

  <!-- Multi-Tile & JTF Sub-Architecture Card (Right Column Bottom) -->
  <g transform="translate(660, 400)">
    <rect width="490" height="273" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#cardShadow)"/>
    <rect x="14" y="14" width="28" height="28" rx="6" fill="#0284c7"/>
    <text x="28" y="33" fill="#ffffff" font-size="14" font-weight="700" text-anchor="middle">T</text>
    <text x="52" y="33" fill="#0369a1" font-size="16" font-weight="700">Multi-Tile Architecture &amp; JTF (Pass 2)</text>
    
    <text x="52" y="56" fill="#334155" font-size="12.5"><tspan font-weight="600">Pass 1 (ISC):</tspan> Scatter frame into <tspan font-family="monospace">NxM</tspan> sub-tiles. Cluster independently.</text>
    <text x="52" y="74" fill="#334155" font-size="12.5"><tspan font-weight="600">Pass 2 (JTF):</tspan> Joint Trajectory Fusion eliminates boundary flicker</text>
    <text x="52" y="92" fill="#334155" font-size="12.5">by matching multi-tile tuple keys against recent history.</text>

    <!-- Mini diagram of tiles -->
    <g transform="translate(52, 106)">
      <!-- 2x2 Grid -->
      <rect x="0" y="0" width="46" height="46" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="2,2"/>
      <rect x="46" y="0" width="46" height="46" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="2,2"/>
      <rect x="0" y="46" width="46" height="46" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="2,2"/>
      <rect x="46" y="46" width="46" height="46" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="2,2"/>
      <text x="23" y="28" font-size="10.5" fill="#64748b" text-anchor="middle">Tile 0</text>
      <text x="69" y="28" font-size="10.5" fill="#64748b" text-anchor="middle">Tile 1</text>
      <text x="23" y="74" font-size="10.5" fill="#64748b" text-anchor="middle">Tile 2</text>
      <text x="69" y="74" font-size="10.5" fill="#64748b" text-anchor="middle">Tile 3</text>

      <!-- JTF Arrow -->
      <line x1="108" y1="46" x2="148" y2="46" stroke="#0284c7" stroke-width="2.5" marker-end="url(#arr-blue)"/>

      <!-- Fusion Box -->
      <rect x="158" y="10" width="248" height="74" rx="6" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1.2"/>
      <text x="170" y="32" fill="#0369a1" font-size="12" font-weight="700">Joint Trajectory Tuple</text>
      <text x="170" y="50" fill="#334155" font-size="11.5">Raw: (0, 3, <tspan fill="#dc2626" font-weight="700">5</tspan>, 1) &#x2192; Flickered Seam</text>
      <text x="170" y="68" fill="#15803d" font-size="11.5" font-weight="600">JTF: (0, 3, 2, 1) &#x2714; Corrected</text>
    </g>

    <!-- Option Badges -->
    <g transform="translate(52, 236)">
      <rect width="52" height="18" rx="9" fill="#0284c7"/>
      <text x="26" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-tiles</text>
      
      <rect x="58" width="45" height="18" rx="9" fill="#0369a1"/>
      <text x="80.5" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-jtf</text>
      
      <rect x="109" width="115" height="18" rx="9" fill="#0284c7"/>
      <text x="166.5" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-retrieval_window</text>
      
      <rect x="230" width="55" height="18" rx="9" fill="#075985"/>
      <text x="257.5" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-xtile</text>
      
      <rect x="291" width="48" height="18" rx="9" fill="#0c4a6e"/>
      <text x="315" y="13" fill="#ffffff" font-size="10.5" font-weight="600" text-anchor="middle">-cpt</text>
    </g>
  </g>

  <!-- Clear Connector Channels in Central Gutter (x=530 to 660) -->
  <!-- MATCH -> Step 5A (Green Channel) -->
  <path d="M 530 580 L 570 580 L 570 200 L 660 200" fill="none" stroke="#16a34a" stroke-width="2.5" stroke-dasharray="6,4" marker-end="url(#arr-green)"/>
  <rect x="542" y="360" width="102" height="24" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="1.2"/>
  <text x="593" y="376" fill="#15803d" font-size="11" font-weight="700" text-anchor="middle">d &lt;= rlim (Match)</text>

  <!-- EXHAUSTED -> Step 5B (Red Channel) -->
  <path d="M 530 635 L 610 635 L 610 300 L 660 300" fill="none" stroke="#dc2626" stroke-width="2.5" marker-end="url(#arr-red)"/>
  <rect x="548" y="475" width="115" height="24" rx="12" fill="#fee2e2" stroke="#dc2626" stroke-width="1.2"/>
  <text x="605.5" y="491" fill="#991b1b" font-size="11" font-weight="700" text-anchor="middle">All Candidates Pruned</text>

  <!-- Bottom Key Takeaway Bar -->
  <g transform="translate(50, 792)">
    <rect width="1100" height="44" rx="8" fill="#ffffff" stroke="#cbd5e1" filter="url(#cardShadow)"/>
    <text x="24" y="27" fill="#475569" font-size="13" font-weight="700">Core GRIC Invariant:</text>
    <text x="175" y="27" fill="#1e293b" font-size="13">Triangle inequality pruning + active target scheduling reduces distance calls from <tspan font-family="monospace" font-weight="700" fill="#dc2626">O(K)</tspan> down to <tspan font-family="monospace" font-weight="700" fill="#16a34a">O(1 ~ 3)</tspan> per frame on high-D streams.</text>
  </g>
</svg>
"""
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg.strip())
    print(f"Generated: {svg_path}")



# -----------------------------------------------------------------------------
# 2. Pruning & Distance Geometry SVG
# -----------------------------------------------------------------------------
def generate_pruning_geometry_svg():
    svg_path = os.path.join(OUTPUT_DIR, "gric_pruning_geometry.svg")
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 880" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <linearGradient id="gradHeadPrune" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#1e293b"/>
    </linearGradient>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>

    <marker id="m-blue" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#2563eb"/>
    </marker>
    <marker id="m-red" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#dc2626"/>
    </marker>
    <marker id="m-gray" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#64748b"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1200" height="880" fill="#f8fafc"/>

  <!-- Header -->
  <rect x="40" y="20" width="1120" height="72" rx="12" fill="url(#gradHeadPrune)" filter="url(#shadow)"/>
  <text x="70" y="52" fill="#ffffff" font-size="22" font-weight="700">GRIC Multi-Point Geometric Pruning Architecture</text>
  <text x="70" y="74" fill="#94a3b8" font-size="13">Exploiting metric space properties to eliminate impossible candidate clusters without computing distances</text>
  <rect x="990" y="40" width="145" height="32" rx="16" fill="#10b981"/>
  <text x="1062" y="61" fill="#ffffff" font-size="12" font-weight="600" text-anchor="middle">Distance Geometry</text>

  <!-- Quadrant 1: 3-Point Triangle Inequality & 1D Baseline Projection -->
  <g transform="translate(40, 110)">
    <rect width="540" height="355" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="16" y="16" width="30" height="30" rx="6" fill="#2563eb"/>
    <text x="31" y="36" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">3P</text>
    <text x="56" y="36" fill="#1e3a8a" font-size="16" font-weight="700">3-Point Pruning: 1D Baseline Projection</text>
    
    <!-- Legend Status Bar -->
    <g transform="translate(20, 52)">
      <rect width="105" height="20" rx="10" fill="#dbeafe" stroke="#bfdbfe"/>
      <circle cx="10" cy="10" r="4.5" fill="#2563eb"/>
      <text x="20" y="14" fill="#1e40af" font-size="10" font-weight="700">MEASURED</text>

      <rect x="115" width="85" height="20" rx="10" fill="#f1f5f9" stroke="#cbd5e1"/>
      <circle cx="125" cy="10" r="4.5" fill="#64748b"/>
      <text x="135" y="14" fill="#334155" font-size="10" font-weight="700">KNOWN</text>

      <rect x="210" width="95" height="20" rx="10" fill="#f3e8ff" stroke="#d8b4fe"/>
      <circle cx="220" cy="10" r="4.5" fill="#9333ea"/>
      <text x="230" y="14" fill="#6b21a8" font-size="10" font-weight="700">UNKNOWN</text>

      <rect x="315" width="185" height="20" rx="10" fill="#fee2e2" stroke="#fca5a5"/>
      <circle cx="325" cy="10" r="4.5" fill="#dc2626"/>
      <text x="335" y="14" fill="#991b1b" font-size="10" font-weight="700">DERIVED MIN DISTANCE</text>
    </g>

    <!-- Geometry Canvas -->
    <g transform="translate(20, 85)">
      <!-- Baseline Line connecting cA and cX -->
      <line x1="50" y1="105" x2="450" y2="105" stroke="#cbd5e1" stroke-width="2"/>
      
      <!-- Known Distance Bracket: d(cA, cX) -->
      <line x1="50" y1="88" x2="450" y2="88" stroke="#64748b" stroke-width="1.5"/>
      <line x1="50" y1="82" x2="50" y2="94" stroke="#64748b" stroke-width="1.5"/>
      <line x1="450" y1="82" x2="450" y2="94" stroke="#64748b" stroke-width="1.5"/>
      <text x="250" y="80" font-size="11.5" font-weight="700" fill="#475569" text-anchor="middle">d(cA, cX) = 5.40  [KNOWN in DCC Matrix]</text>

      <!-- Frame f -->
      <circle cx="130" cy="25" r="7" fill="#facc15" stroke="#0f172a" stroke-width="2"/>
      <text x="130" y="12" font-size="12.5" font-weight="700" fill="#0f172a" text-anchor="middle">f (Incoming Frame)</text>

      <!-- Anchor cA (Pivot) -->
      <circle cx="50" cy="105" r="7" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>
      <text x="50" y="127" font-size="12" font-weight="700" fill="#1e40af" text-anchor="middle">cA (Pivot)</text>

      <!-- Anchor cX (Candidate) -->
      <circle cx="450" cy="105" r="32" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.7"/>
      <circle cx="450" cy="105" r="7" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>
      <text x="450" y="127" font-size="12" font-weight="700" fill="#991b1b" text-anchor="middle">cX (Candidate)</text>
      <text x="450" y="148" font-size="10" fill="#ef4444" font-weight="600" text-anchor="middle">r_lim sphere</text>

      <!-- Measured Distance Edge: d(f, cA) -->
      <line x1="50" y1="105" x2="130" y2="25" stroke="#2563eb" stroke-width="3"/>
      <text x="65" y="52" font-size="11.5" font-weight="700" fill="#1d4ed8" transform="rotate(-47 65,52)">d(f, cA) = 2.20 [MEASURED]</text>

      <!-- Arc projecting d(f, cA) down to the baseline cA -> cX -->
      <path d="M 130 25 A 113.1 113.1 0 0 1 163.1 105" fill="none" stroke="#2563eb" stroke-width="1.5" stroke-dasharray="4,4"/>

      <!-- Projected Measured Segment on Baseline (cA to Proj) -->
      <line x1="50" y1="105" x2="163.1" y2="105" stroke="#2563eb" stroke-width="5" stroke-linecap="round"/>
      <circle cx="163.1" cy="105" r="4" fill="#2563eb"/>
      <text x="106" y="124" font-size="10" font-weight="700" fill="#2563eb" text-anchor="middle">d(f, cA) (Proj)</text>

      <!-- Remaining Difference Segment (Proj to cX) = MIN POSSIBLE DISTANCE -->
      <line x1="163.1" y1="105" x2="450" y2="105" stroke="#dc2626" stroke-width="5" stroke-linecap="round"/>
      
      <!-- Difference Bracket -->
      <line x1="163.1" y1="150" x2="450" y2="150" stroke="#dc2626" stroke-width="2"/>
      <line x1="163.1" y1="144" x2="163.1" y2="156" stroke="#dc2626" stroke-width="2"/>
      <line x1="450" y1="144" x2="450" y2="156" stroke="#dc2626" stroke-width="2"/>
      <text x="306" y="167" font-size="11.5" font-weight="700" fill="#dc2626" text-anchor="middle">d_min = d(cA, cX) - d(f, cA) = 3.20 &gt; r_lim</text>

      <!-- Unknown Distance d(f, cX) -->
      <line x1="130" y1="25" x2="450" y2="105" stroke="#9333ea" stroke-width="2" stroke-dasharray="5,4"/>
      <text x="295" y="48" font-size="11" font-weight="700" fill="#9333ea" transform="rotate(14 295,48)">d(f, cX) [UNKNOWN] &gt;= d_min</text>

      <!-- Pruned Badge Stamp -->
      <rect x="380" y="15" width="115" height="24" rx="12" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"/>
      <text x="437.5" y="31" font-size="11" font-weight="700" fill="#991b1b" text-anchor="middle">&#x2718; PRUNED (0 Evals)</text>
    </g>

    <!-- Formula Box -->
    <rect x="16" y="295" width="508" height="46" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="26" y="314" font-size="11.5" font-weight="600" fill="#334155">3-Point Pruning Rule:</text>
    <text x="160" y="314" font-family="monospace" font-size="12" font-weight="700" fill="#dc2626">d(f, cX) &gt;= | d(cA, cX) - d(f, cA) | &gt; r_lim</text>
    <text x="26" y="331" font-size="11" fill="#64748b">1D projection along baseline cA &#x2192; cX eliminates impossible candidate cX without metric evaluations.</text>
  </g>

  <!-- Quadrant 2: 4-Point Pruning (-te4: 2 Measured Anchors, 2D Triangulation Projection) -->
  <g transform="translate(620, 110)">
    <rect width="540" height="355" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="16" y="16" width="30" height="30" rx="6" fill="#7c3aed"/>
    <text x="31" y="36" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">4P</text>
    <text x="56" y="36" fill="#581c87" font-size="16" font-weight="700">4-Point Pruning: 2D Triangulation Projection</text>
    
    <!-- Legend Status Bar -->
    <g transform="translate(20, 52)">
      <rect width="140" height="20" rx="10" fill="#f3e8ff" stroke="#d8b4fe"/>
      <circle cx="10" cy="10" r="4.5" fill="#7c3aed"/>
      <text x="20" y="14" fill="#6b21a8" font-size="10" font-weight="700">2 MEASURED (cA, cB)</text>

      <rect x="150" width="130" height="20" rx="10" fill="#dbeafe" stroke="#bfdbfe"/>
      <circle cx="160" cy="10" r="4.5" fill="#0284c7"/>
      <text x="170" y="14" fill="#0369a1" font-size="10" font-weight="700">ORTHOGONAL h_f</text>

      <rect x="290" width="200" height="20" rx="10" fill="#fee2e2" stroke="#fca5a5"/>
      <circle cx="300" cy="10" r="4.5" fill="#dc2626"/>
      <text x="310" y="14" fill="#991b1b" font-size="10" font-weight="700">TIGHTER 2D BOUND</text>
    </g>

    <!-- Geometry Canvas -->
    <g transform="translate(20, 85)">
      <!-- Baseline Line connecting cA, cB and extending to cX -->
      <line x1="40" y1="105" x2="450" y2="105" stroke="#cbd5e1" stroke-width="2"/>

      <!-- Frame f -->
      <circle cx="140" cy="25" r="7" fill="#facc15" stroke="#0f172a" stroke-width="2"/>
      <text x="140" y="12" font-size="12.5" font-weight="700" fill="#0f172a" text-anchor="middle">f (Incoming Frame)</text>

      <!-- Measured Anchors cA and cB -->
      <circle cx="40" cy="105" r="7" fill="#7c3aed" stroke="#ffffff" stroke-width="2"/>
      <text x="40" y="127" font-size="12" font-weight="700" fill="#581c87" text-anchor="middle">cA (Meas)</text>

      <circle cx="210" cy="105" r="7" fill="#7c3aed" stroke="#ffffff" stroke-width="2"/>
      <text x="210" y="127" font-size="12" font-weight="700" fill="#581c87" text-anchor="middle">cB (Meas)</text>

      <!-- Candidate Anchor cX -->
      <circle cx="450" cy="105" r="32" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.7"/>
      <circle cx="450" cy="105" r="7" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>
      <text x="450" y="127" font-size="12" font-weight="700" fill="#991b1b" text-anchor="middle">cX (Candidate)</text>

      <!-- Measured Distance Edges: d(f, cA) & d(f, cB) -->
      <line x1="40" y1="105" x2="140" y2="25" stroke="#7c3aed" stroke-width="2.5"/>
      <text x="70" y="55" font-size="10.5" font-weight="700" fill="#7c3aed">d(f, cA)</text>

      <line x1="210" y1="105" x2="140" y2="25" stroke="#7c3aed" stroke-width="2.5"/>
      <text x="185" y="55" font-size="10.5" font-weight="700" fill="#7c3aed">d(f, cB)</text>

      <!-- Orthogonal Height Projection h_f down to Baseline at (140, 105) -->
      <line x1="140" y1="25" x2="140" y2="105" stroke="#0284c7" stroke-width="2" stroke-dasharray="4,3"/>
      <circle cx="140" cy="105" r="4" fill="#0284c7"/>
      <!-- Right angle marker -->
      <rect x="140" y="95" width="10" height="10" fill="none" stroke="#0284c7" stroke-width="1.2"/>
      <text x="125" y="70" font-size="11" font-weight="700" fill="#0284c7" text-anchor="end">h_f (Height)</text>

      <!-- Longitudinal Distance along baseline from projection (140) to cX (450) -->
      <line x1="140" y1="148" x2="450" y2="148" stroke="#64748b" stroke-width="1.5"/>
      <line x1="140" y1="142" x2="140" y2="154" stroke="#64748b" stroke-width="1.5"/>
      <line x1="450" y1="142" x2="450" y2="154" stroke="#64748b" stroke-width="1.5"/>
      <text x="295" y="165" font-size="11" font-weight="700" fill="#475569" text-anchor="middle">&#x394;x = | x_cX - x_f |</text>

      <!-- 2D Triangulated Hypotenuse Bound to cX -->
      <line x1="140" y1="25" x2="450" y2="105" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5,4"/>
      <text x="315" y="55" font-size="11" font-weight="700" fill="#dc2626" transform="rotate(14 315,55)">d_min = &#x221A;(h_f&#178; + &#x394;x&#178;) &gt; r_lim</text>

      <!-- Pruned Badge Stamp -->
      <rect x="375" y="15" width="125" height="24" rx="12" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"/>
      <text x="437.5" y="31" font-size="11" font-weight="700" fill="#991b1b" text-anchor="middle">&#x2718; PRUNED by -te4</text>
    </g>

    <!-- Formula Box -->
    <rect x="16" y="295" width="508" height="46" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="26" y="314" font-size="11.5" font-weight="600" fill="#334155">4-Point Triangulation Rule:</text>
    <text x="180" y="314" font-family="monospace" font-size="11.5" font-weight="700" fill="#7c3aed">d_min = sqrt( h_f^2 + (x_cX - x_f)^2 ) &gt; r_lim</text>
    <text x="26" y="331" font-size="11" fill="#64748b">Triangulating between 2 measured pivots computes orthogonal height h_f, strictly tightening the bound.</text>
  </g>

  <!-- Quadrant 3: 5-Point Pruning (-te5: 3 Measured Anchors, 3D Simplex Height Projection) -->
  <g transform="translate(40, 490)">
    <rect width="540" height="355" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="16" y="16" width="30" height="30" rx="6" fill="#059669"/>
    <text x="31" y="36" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">5P</text>
    <text x="56" y="36" fill="#064e3b" font-size="16" font-weight="700">5-Point Pruning: 3D Simplex Height Projection</text>
    
    <!-- Legend Status Bar -->
    <g transform="translate(20, 52)">
      <rect width="160" height="20" rx="10" fill="#ecfdf5" stroke="#a7f3d0"/>
      <circle cx="10" cy="10" r="4.5" fill="#059669"/>
      <text x="20" y="14" fill="#065f46" font-size="10" font-weight="700">3 MEASURED (cA, cB, cC)</text>

      <rect x="190" width="135" height="20" rx="10" fill="#dbeafe" stroke="#bfdbfe"/>
      <circle cx="200" cy="10" r="4.5" fill="#0284c7"/>
      <text x="210" y="14" fill="#0369a1" font-size="10" font-weight="700">3D HEIGHT h_3D</text>

      <rect x="335" width="175" height="20" rx="10" fill="#fee2e2" stroke="#fca5a5"/>
      <circle cx="345" cy="10" r="4.5" fill="#dc2626"/>
      <text x="355" y="14" fill="#991b1b" font-size="10" font-weight="700">3D PROJECTION BOUND</text>
    </g>

    <!-- Geometry Canvas -->
    <g transform="translate(20, 85)">
      <!-- 3D Simplex Base Plane (Triangle cA-cB-cC) -->
      <polygon points="50,135 190,155 125,85" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1.5"/>
      <text x="120" y="130" font-size="10.5" fill="#059669" font-weight="600" text-anchor="middle">Simplex Base</text>

      <!-- Frame f elevated in 3D -->
      <circle cx="140" cy="22" r="7" fill="#facc15" stroke="#0f172a" stroke-width="2"/>
      <text x="140" y="10" font-size="12.5" font-weight="700" fill="#0f172a" text-anchor="middle">f (Incoming Frame)</text>

      <!-- 3 Measured Anchors -->
      <circle cx="50" cy="135" r="6" fill="#059669"/>
      <text x="35" y="145" font-size="11" font-weight="700" fill="#065f46">cA</text>

      <circle cx="190" cy="155" r="6" fill="#059669"/>
      <text x="200" y="165" font-size="11" font-weight="700" fill="#065f46">cB</text>

      <circle cx="125" cy="85" r="6" fill="#059669"/>
      <text x="125" y="75" font-size="11" font-weight="700" fill="#065f46">cC</text>

      <!-- Orthogonal projection height h_3D down to plane at (135, 125) -->
      <line x1="140" y1="22" x2="135" y2="125" stroke="#0284c7" stroke-width="2" stroke-dasharray="4,3"/>
      <circle cx="135" cy="125" r="4" fill="#0284c7"/>
      <text x="130" y="65" font-size="11" font-weight="700" fill="#0284c7" text-anchor="end">h_3D</text>

      <!-- Connection Lines from f to cA, cB, cC -->
      <line x1="140" y1="22" x2="50" y2="135" stroke="#10b981" stroke-width="1.5"/>
      <line x1="140" y1="22" x2="190" y2="155" stroke="#10b981" stroke-width="1.5"/>
      <line x1="140" y1="22" x2="125" y2="85" stroke="#10b981" stroke-width="1.5"/>

      <!-- Candidate cX -->
      <circle cx="450" cy="100" r="32" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.7"/>
      <circle cx="450" cy="100" r="7" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>
      <text x="450" y="122" font-size="12" font-weight="700" fill="#991b1b" text-anchor="middle">cX (Candidate)</text>

      <!-- Planar offset from projection (135, 125) to cX (450, 100) -->
      <line x1="135" y1="125" x2="450" y2="100" stroke="#64748b" stroke-width="1.5" stroke-dasharray="2,2"/>
      <text x="290" y="130" font-size="11" font-weight="600" fill="#475569">&#x394;r_plane</text>

      <!-- Hypotenuse Bound to cX -->
      <line x1="140" y1="22" x2="450" y2="100" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5,4"/>
      <text x="310" y="50" font-size="11" font-weight="700" fill="#dc2626" transform="rotate(14 310,50)">d_min = &#x221A;(h_3D&#178; + &#x394;r&#178;) &gt; r_lim</text>

      <!-- Pruned Badge Stamp -->
      <rect x="375" y="15" width="125" height="24" rx="12" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"/>
      <text x="437.5" y="31" font-size="11" font-weight="700" fill="#991b1b" text-anchor="middle">&#x2718; PRUNED by -te5</text>
    </g>

    <!-- Formula Box -->
    <rect x="16" y="295" width="508" height="46" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="26" y="314" font-size="11.5" font-weight="600" fill="#334155">5-Point Simplex Rule:</text>
    <text x="160" y="314" font-family="monospace" font-size="11.5" font-weight="700" fill="#059669">d_min = sqrt( h_3D^2 + ||proj(f) - proj(cX)||^2 ) &gt; r_lim</text>
    <text x="26" y="331" font-size="11" fill="#64748b">3D orthogonal projection yields the tightest mathematical metric bound on curved manifolds.</text>
  </g>

  <!-- Quadrant 4: Sparse DCC Bounds (-sparse_dcc) -->
  <g transform="translate(620, 490)">
    <rect width="540" height="355" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="16" y="16" width="30" height="30" rx="6" fill="#d97706"/>
    <text x="31" y="36" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">SD</text>
    <text x="56" y="36" fill="#78350f" font-size="16" font-weight="700">Sparse DCC Distance Bounding (-sparse_dcc)</text>
    
    <!-- Geometry Canvas -->
    <g transform="translate(20, 55)">
      <!-- Dense vs Sparse comparison cards -->
      <rect x="15" y="10" width="220" height="90" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.2"/>
      <text x="25" y="32" font-size="12" font-weight="700" fill="#991b1b">Dense DCC Matrix:</text>
      <text x="25" y="52" font-size="11.5" fill="#7f1d1d">O(K^2) memory &amp; evals</text>
      <text x="25" y="70" font-size="11.5" fill="#7f1d1d">10,000 clusters = 50M pairs</text>
      <text x="25" y="88" font-size="11" fill="#b91c1c" font-weight="600">&#x26A0; High initialization &amp; RAM cost</text>

      <rect x="255" y="10" width="230" height="90" rx="8" fill="#f0fdf4" stroke="#86efac" stroke-width="1.2"/>
      <text x="265" y="32" font-size="12" font-weight="700" fill="#14532d">Sparse DCC Matrix:</text>
      <text x="265" y="52" font-size="11.5" fill="#14532d">O(K) active bound intervals</text>
      <text x="265" y="70" font-size="11.5" font-family="monospace" font-weight="600" fill="#15803d">[dcc_min(cA,cX), dcc_max(cA,cX)]</text>
      <text x="265" y="88" font-size="11" fill="#16a34a" font-weight="600">&#x2714; Scales seamlessly to 100k+ clusters</text>

      <!-- Bound Interval Line -->
      <g transform="translate(15, 120)">
        <line x1="20" y1="35" x2="450" y2="35" stroke="#cbd5e1" stroke-width="4" stroke-linecap="round"/>
        <line x1="120" y1="35" x2="340" y2="35" stroke="#f59e0b" stroke-width="7" stroke-linecap="round"/>
        
        <!-- Markers -->
        <circle cx="120" cy="35" r="5" fill="#d97706"/>
        <text x="120" y="18" font-size="11" font-weight="700" fill="#b45309" text-anchor="middle">dcc_min(cA, cX)</text>

        <circle cx="340" cy="35" r="5" fill="#d97706"/>
        <text x="340" y="18" font-size="11" font-weight="700" fill="#b45309" text-anchor="middle">dcc_max(cA, cX)</text>

        <text x="230" y="62" font-size="11.5" font-weight="700" fill="#334155" text-anchor="middle">Dynamic Interval Bound Range</text>
      </g>
    </g>

    <!-- Formula Box -->
    <rect x="16" y="295" width="508" height="46" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="26" y="314" font-size="11.5" font-weight="600" fill="#334155">Sparse Pruning Rule:</text>
    <text x="165" y="314" font-family="monospace" font-size="11.5" font-weight="700" fill="#d97706">d(f, cA) - dcc_max(cA, cX) &gt; r_lim</text>
    <text x="26" y="331" font-size="11" fill="#64748b">Tightened on demand with -sparse_dcc_extra_evals &lt;N&gt; without dense O(K^2) memory.</text>
  </g>
</svg>
"""
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg.strip())
    print(f"Generated: {svg_path}")


# -----------------------------------------------------------------------------
# 3. Target Selection & Entropy SVG
# -----------------------------------------------------------------------------
def generate_target_selection_svg():
    svg_path = os.path.join(OUTPUT_DIR, "gric_target_selection_entropy.svg")
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 780" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <linearGradient id="gradHeadEntropy" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#4a044e"/>
      <stop offset="100%" stop-color="#701a75"/>
    </linearGradient>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
  </defs>

  <rect width="1100" height="780" fill="#f8fafc"/>

  <!-- Header -->
  <rect x="30" y="20" width="1040" height="70" rx="12" fill="url(#gradHeadEntropy)" filter="url(#shadow)"/>
  <text x="60" y="52" fill="#ffffff" font-size="22" font-weight="700">Target Candidate Selection: Greedy vs Shannon Entropy</text>
  <text x="60" y="74" fill="#e879f9" font-size="13">Active decision-theoretic scheduling to minimize search ambiguity per distance measurement</text>
  <rect x="890" y="40" width="155" height="30" rx="15" fill="#c026d3"/>
  <text x="967.5" y="60" fill="#ffffff" font-size="12" font-weight="600" text-anchor="middle">Information Theory</text>

  <!-- Left Card: Greedy Selection -->
  <g transform="translate(40, 110)">
    <rect width="490" height="630" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="15" y="15" width="28" height="28" rx="6" fill="#3b82f6"/>
    <text x="29" y="34" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">G</text>
    <text x="52" y="34" fill="#1e3a8a" font-size="17" font-weight="700">Greedy Mode (Default)</text>
    <text x="52" y="54" fill="#64748b" font-size="12">Tests candidate with highest posterior likelihood first</text>

    <!-- Visual Bars of Probability -->
    <g transform="translate(30, 80)">
      <text x="0" y="15" font-size="12" font-weight="600" fill="#334155">Posterior Probabilities P(c_j):</text>
      
      <!-- Cluster 1 (0.35) -->
      <text x="0" y="45" font-size="12" font-family="monospace" fill="#334155">c_1 (35%)</text>
      <rect x="80" y="32" width="180" height="18" rx="4" fill="#3b82f6"/>
      <rect x="270" y="32" width="75" height="18" rx="9" fill="#dbeafe"/>
      <text x="307" y="45" font-size="11" font-weight="700" fill="#1d4ed8" text-anchor="middle">1st Pick &#x2794;</text>

      <!-- Cluster 2 (0.25) -->
      <text x="0" y="75" font-size="12" font-family="monospace" fill="#334155">c_2 (25%)</text>
      <rect x="80" y="62" width="130" height="18" rx="4" fill="#93c5fd"/>
      <text x="270" y="75" font-size="11" fill="#64748b">2nd Pick</text>

      <!-- Cluster 3 (0.20) -->
      <text x="0" y="105" font-size="12" font-family="monospace" fill="#334155">c_3 (20%)</text>
      <rect x="80" y="92" width="105" height="18" rx="4" fill="#bfdbfe"/>
      <text x="270" y="105" font-size="11" fill="#64748b">3rd Pick</text>

      <!-- Cluster 4 (0.12) -->
      <text x="0" y="135" font-size="12" font-family="monospace" fill="#334155">c_4 (12%)</text>
      <rect x="80" y="122" width="65" height="18" rx="4" fill="#dbeafe"/>

      <!-- Cluster 5 (0.08) -->
      <text x="0" y="165" font-size="12" font-family="monospace" fill="#334155">c_5 ( 8%)</text>
      <rect x="80" y="152" width="40" height="18" rx="4" fill="#eff6ff"/>
    </g>

    <!-- Strategy Breakdown Box -->
    <rect x="20" y="280" width="450" height="175" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="35" y="305" font-size="13" font-weight="700" fill="#1e293b">Mechanism &amp; Properties:</text>
    <text x="35" y="330" font-size="12" fill="#334155">&#x2022; <tspan font-weight="600">Decision rule:</tspan> <tspan font-family="monospace">target = argmax P(c_j)</tspan></text>
    <text x="35" y="355" font-size="12" fill="#334155">&#x2022; <tspan font-weight="600">Computation Overhead:</tspan> Zero overhead (simple sort/heap).</text>
    <text x="35" y="380" font-size="12" fill="#334155">&#x2022; <tspan font-weight="600">Best for:</tspan> High-frame-rate video or continuous streams</text>
    <text x="47" y="398" font-size="12" fill="#334155">with extreme temporal persistence (<tspan font-family="monospace">P(c_last) &gt; 0.7</tspan>).</text>
    <text x="35" y="425" font-size="12" fill="#dc2626">&#x2022; <tspan font-weight="600">Weakness:</tspan> Suboptimal in high-entropy / noisy distributions;</text>
    <text x="47" y="443" font-size="12" fill="#dc2626">may test 10+ candidates before finding match.</text>

    <!-- Evaluation Box -->
    <rect x="20" y="475" width="450" height="135" rx="8" fill="#eff6ff" stroke="#bfdbfe"/>
    <text x="35" y="500" font-size="13" font-weight="700" fill="#1e3a8a">Greedy In Action:</text>
    <text x="35" y="525" font-size="12" fill="#1e3a8a">Frame arrives &#x2192; Checks c_1 (Mismatch!) &#x2192; Checks c_2 (Mismatch!) &#x2192;</text>
    <text x="35" y="545" font-size="12" fill="#1e3a8a">Checks c_3 (Mismatch!) &#x2192; Checks c_4 (Match!).</text>
    <text x="35" y="575" font-size="12.5" font-weight="700" fill="#b91c1c">Total Cost: 4 Expensive Metric Computations</text>
    <text x="35" y="595" font-size="11" fill="#64748b">Each mismatch only eliminated 1 cluster because c_1, c_2 were uninformative.</text>
  </g>

  <!-- Right Card: Entropy Minimization Mode -->
  <g transform="translate(570, 110)">
    <rect width="490" height="630" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="15" y="15" width="28" height="28" rx="6" fill="#a855f7"/>
    <text x="29" y="34" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">E</text>
    <text x="52" y="34" fill="#581c87" font-size="17" font-weight="700">Entropy Optimization (-entropy)</text>
    <text x="52" y="54" fill="#64748b" font-size="12">Information-Theoretic Active Search Scheduling</text>

    <!-- Information Gain Concept Graphic -->
    <g transform="translate(30, 80)">
      <rect width="430" height="90" rx="6" fill="#faf5ff" stroke="#e9d5ff"/>
      <text x="15" y="22" font-size="12" font-weight="700" fill="#6b21a8">Expected Shannon Posterior Entropy:</text>
      <text x="15" y="44" font-family="monospace" font-size="12" font-weight="700" fill="#7e22ce">H(X | measure c_j) = P_m * 0 + (1 - P_m) * H(X | mismatch)</text>
      <text x="15" y="65" font-size="11.5" fill="#4c1d95"><tspan font-weight="600">Goal:</tspan> Select target anchor that bisects / prunes the largest</text>
      <text x="15" y="80" font-size="11.5" fill="#4c1d95">fraction of candidate probability mass upon mismatch.</text>
    </g>

    <!-- Option Subsystem Grid -->
    <g transform="translate(20, 280)">
      <rect width="450" height="175" rx="8" fill="#fdf4ff" stroke="#f0abfc"/>
      <text x="15" y="22" font-size="13" font-weight="700" fill="#701a75">Entropy Engine Parameters:</text>

      <text x="15" y="48" font-size="12" font-weight="600" fill="#86198f">-entropy_fast</text>
      <text x="120" y="48" font-size="11.5" fill="#4a044e">Replaces float entropy math with popcount bitmask.</text>

      <text x="15" y="75" font-size="12" font-weight="600" fill="#86198f">-entropy_gate &lt;t&gt;</text>
      <text x="120" y="75" font-size="11.5" fill="#4a044e">Bypasses entropy calculation if max prob &gt; threshold.</text>

      <text x="15" y="102" font-size="12" font-weight="600" fill="#86198f">-entropy_first_gate</text>
      <text x="140" y="102" font-size="11.5" fill="#4a044e">Enforces greedy probe on 1st step, entropy thereafter.</text>

      <text x="15" y="129" font-size="12" font-weight="600" fill="#86198f">-entropy_max_targets</text>
      <text x="155" y="129" font-size="11.5" fill="#4a044e">Limits evaluation to top N candidate anchors (default 15).</text>

      <text x="15" y="156" font-size="12" font-weight="600" fill="#86198f">-entropy_min_prob</text>
      <text x="140" y="156" font-size="11.5" fill="#4a044e">Ignores clusters with negligible probability (&lt; 0.001).</text>
    </g>

    <!-- Entropy in Action Box -->
    <rect x="20" y="475" width="450" height="135" rx="8" fill="#f0fdf4" stroke="#86efac"/>
    <text x="35" y="500" font-size="13" font-weight="700" fill="#14532d">Entropy Optimization In Action:</text>
    <text x="35" y="525" font-size="12" fill="#14532d">Picks central pivot anchor c_pivot (Mismatch!) &#x2192; Geometric bounds</text>
    <text x="35" y="545" font-size="12" fill="#14532d">immediately prune c_1, c_2, c_3 and c_5 at once &#x2192; Checks c_4 (Match!).</text>
    <text x="35" y="575" font-size="12.5" font-weight="700" fill="#15803d">Total Cost: 2 Distance Computations (50% Speedup!)</text>
    <text x="35" y="595" font-size="11" fill="#166534">Maximal Information Gain: 1 test resolved entire search space.</text>
  </g>
</svg>
"""
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg.strip())
    print(f"Generated: {svg_path}")


# -----------------------------------------------------------------------------
# 4. Priors, Predictions & Spatial Graph SVG
# -----------------------------------------------------------------------------
def generate_priors_prediction_svg():
    svg_path = os.path.join(OUTPUT_DIR, "gric_priors_prediction.svg")
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 800" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <linearGradient id="gradHeadPrior" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#1e1b4b"/>
      <stop offset="100%" stop-color="#312e81"/>
    </linearGradient>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
  </defs>

  <rect width="1100" height="800" fill="#f8fafc"/>

  <!-- Header -->
  <rect x="30" y="20" width="1040" height="70" rx="12" fill="url(#gradHeadPrior)" filter="url(#shadow)"/>
  <text x="60" y="52" fill="#ffffff" font-size="22" font-weight="700">GRIC Probability, Prediction &amp; Spatial Learning Layers</text>
  <text x="60" y="74" fill="#a5b4fc" font-size="13">Fusing temporal Markov transitions, sequence forecasting, visitor geometry and Bayesian fading</text>
  <rect x="910" y="40" width="135" height="30" rx="15" fill="#6366f1"/>
  <text x="977" y="60" fill="#ffffff" font-size="12" font-weight="600" text-anchor="middle">Prior Modeling</text>

  <!-- Panel 1: Transition Matrix (-tm) -->
  <g transform="translate(40, 110)">
    <rect width="490" height="310" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="15" y="15" width="28" height="28" rx="6" fill="#4f46e5"/>
    <text x="29" y="34" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">TM</text>
    <text x="52" y="34" fill="#312e81" font-size="16" font-weight="700">Markov Transition Matrix (-tm &lt;coeff&gt;)</text>
    
    <!-- Matrix Heatmap & Graph Graphic -->
    <g transform="translate(30, 60)">
      <!-- Matrix grid -->
      <rect x="0" y="0" width="150" height="150" rx="6" fill="#f8fafc" stroke="#cbd5e1"/>
      <!-- Grid cells -->
      <rect x="10" y="10" width="38" height="38" fill="#e0e7ff"/><text x="29" y="33" font-size="10" font-family="monospace" text-anchor="middle">0.05</text>
      <rect x="56" y="10" width="38" height="38" fill="#4f46e5"/><text x="75" y="33" font-size="10" font-family="monospace" fill="#fff" font-weight="700" text-anchor="middle">0.85</text>
      <rect x="102" y="10" width="38" height="38" fill="#e0e7ff"/><text x="121" y="33" font-size="10" font-family="monospace" text-anchor="middle">0.10</text>
      
      <rect x="10" y="56" width="38" height="38" fill="#e0e7ff"/><text x="29" y="79" font-size="10" font-family="monospace" text-anchor="middle">0.10</text>
      <rect x="56" y="56" width="38" height="38" fill="#e0e7ff"/><text x="75" y="79" font-size="10" font-family="monospace" text-anchor="middle">0.10</text>
      <rect x="102" y="56" width="38" height="38" fill="#6366f1"/><text x="121" y="79" font-size="10" font-family="monospace" fill="#fff" font-weight="700" text-anchor="middle">0.80</text>

      <rect x="10" y="102" width="38" height="38" fill="#4338ca"/><text x="29" y="125" font-size="10" font-family="monospace" fill="#fff" font-weight="700" text-anchor="middle">0.75</text>
      <rect x="56" y="102" width="38" height="38" fill="#e0e7ff"/><text x="75" y="125" font-size="10" font-family="monospace" text-anchor="middle">0.15</text>
      <rect x="102" y="102" width="38" height="38" fill="#e0e7ff"/><text x="121" y="125" font-size="10" font-family="monospace" text-anchor="middle">0.10</text>

      <!-- Graph Arrows on the right -->
      <g transform="translate(180, 20)">
        <circle cx="40" cy="20" r="16" fill="#4f46e5"/>
        <text x="40" y="24" fill="#fff" font-size="11" font-weight="700" text-anchor="middle">c_1</text>

        <circle cx="140" cy="20" r="16" fill="#6366f1"/>
        <text x="140" y="24" fill="#fff" font-size="11" font-weight="700" text-anchor="middle">c_2</text>

        <circle cx="90" cy="90" r="16" fill="#4338ca"/>
        <text x="90" y="94" fill="#fff" font-size="11" font-weight="700" text-anchor="middle">c_3</text>

        <!-- Directed arrows -->
        <path d="M 58 15 Q 90 -5 120 15" fill="none" stroke="#4f46e5" stroke-width="2"/>
        <text x="90" y="5" font-size="10" font-weight="700" fill="#4338ca">85%</text>

        <path d="M 135 38 Q 125 70 108 85" fill="none" stroke="#6366f1" stroke-width="2"/>
        <text x="130" y="65" font-size="10" font-weight="700" fill="#4338ca">80%</text>

        <path d="M 75 85 Q 55 60 45 38" fill="none" stroke="#4338ca" stroke-width="2"/>
        <text x="42" y="65" font-size="10" font-weight="700" fill="#4338ca">75%</text>
      </g>
    </g>

    <!-- Formula Box -->
    <rect x="15" y="245" width="460" height="50" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="25" y="266" font-size="11.5" font-weight="600" fill="#334155">Mixing Equation:</text>
    <text x="135" y="266" font-family="monospace" font-size="11.5" font-weight="700" fill="#4f46e5">P_mix(c_j) = (1 - c)*P_freq(c_j) + c*T(prev, c_j)</text>
    <text x="25" y="284" font-size="11" fill="#64748b">Learns cyclic paths (e.g. rotating machinery, satellite orbits, periodic loops).</text>
  </g>

  <!-- Panel 2: Sequence Pattern Predictor (-pred) -->
  <g transform="translate(570, 110)">
    <rect width="490" height="310" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="15" y="15" width="28" height="28" rx="6" fill="#9333ea"/>
    <text x="29" y="34" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">PR</text>
    <text x="52" y="34" fill="#581c87" font-size="16" font-weight="700">Trajectory Predictor (-pred [len,h,n])</text>
    
    <!-- Time Sequence Graphic -->
    <g transform="translate(30, 65)">
      <text x="0" y="15" font-size="12" font-weight="600" fill="#334155">Recent History Window (len=4):</text>
      <!-- History blocks -->
      <g transform="translate(0, 25)">
        <rect x="0" y="0" width="42" height="32" rx="4" fill="#e9d5ff"/><text x="21" y="20" font-family="monospace" font-size="12" font-weight="700" fill="#581c87" text-anchor="middle">c_4</text>
        <rect x="48" y="0" width="42" height="32" rx="4" fill="#e9d5ff"/><text x="69" y="20" font-family="monospace" font-size="12" font-weight="700" fill="#581c87" text-anchor="middle">c_1</text>
        <rect x="96" y="0" width="42" height="32" rx="4" fill="#e9d5ff"/><text x="117" y="20" font-family="monospace" font-size="12" font-weight="700" fill="#581c87" text-anchor="middle">c_7</text>
        <rect x="144" y="0" width="42" height="32" rx="4" fill="#e9d5ff"/><text x="165" y="20" font-family="monospace" font-size="12" font-weight="700" fill="#581c87" text-anchor="middle">c_2</text>
        
        <!-- Extrapolation Arrow -->
        <line x1="195" y1="16" x2="230" y2="16" stroke="#9333ea" stroke-width="2.5"/>
        <polygon points="230 11, 240 16, 230 21" fill="#9333ea"/>

        <rect x="250" y="0" width="55" height="32" rx="4" fill="#9333ea"/><text x="277.5" y="20" font-family="monospace" font-size="12" font-weight="700" fill="#fff" text-anchor="middle">c_9 ?</text>
        <text x="315" y="20" font-size="11" font-weight="700" fill="#16a34a">&#x2714; 92% Conf</text>
      </g>

      <!-- Subsequence Match Scan -->
      <g transform="translate(0, 85)">
        <rect width="430" height="50" rx="6" fill="#faf5ff" stroke="#e9d5ff"/>
        <text x="15" y="20" font-size="11.5" font-weight="700" fill="#6b21a8">Historical Scan in Buffer (lookback h=1000):</text>
        <text x="15" y="38" font-size="11" fill="#581c87">Found 8 occurrences of sequence [4 &#x2192; 1 &#x2192; 7 &#x2192; 2]. 7 transitioned to c_9.</text>
      </g>
    </g>

    <!-- Formula Box -->
    <rect x="15" y="245" width="460" height="50" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="25" y="266" font-size="11.5" font-weight="600" fill="#334155">Bypass Priority:</text>
    <text x="135" y="266" font-family="monospace" font-size="11.5" font-weight="700" fill="#9333ea">Top n predictions tested first, bypassing search order</text>
    <text x="25" y="284" font-size="11" fill="#64748b">Directly hits matching cluster in 1 measurement for predictable physics / walks.</text>
  </g>

  <!-- Panel 3: Geometric Match Visitor Probability (-gprob) -->
  <g transform="translate(40, 445)">
    <rect width="490" height="330" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="15" y="15" width="28" height="28" rx="6" fill="#0d9488"/>
    <text x="29" y="34" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">GP</text>
    <text x="52" y="34" fill="#115e59" font-size="16" font-weight="700">Visitor Geometric Probability (-gprob)</text>
    
    <!-- Co-measurement correlation graphic -->
    <g transform="translate(30, 60)">
      <rect width="430" height="100" rx="6" fill="#f0fdfa" stroke="#99f6e4"/>
      <text x="15" y="22" font-size="12" font-weight="700" fill="#0f766e">Visitor Co-Measurement Matching:</text>
      <text x="15" y="42" font-size="11.5" fill="#134e4a">When frame <tspan font-family="monospace">m</tspan> measures distance to cluster <tspan font-family="monospace">c</tspan>, compare against</text>
      <text x="15" y="58" font-size="11.5" fill="#134e4a">historical frames <tspan font-family="monospace">k</tspan> that also measured cluster <tspan font-family="monospace">c</tspan>.</text>
      <text x="15" y="80" font-family="monospace" font-size="11" font-weight="700" fill="#0d9488">dr = |dist(m, c) - dist(k, c)| / rlim &#x2192; fmatch(dr)</text>
    </g>

    <!-- Formula Box -->
    <rect x="15" y="260" width="460" height="55" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="25" y="280" font-size="11.5" font-weight="600" fill="#334155">Posterior Update:</text>
    <text x="135" y="280" font-family="monospace" font-size="11.5" font-weight="700" fill="#0d9488">gprob(m, target[k]) *= fmatcha - (fmatcha-fmatchb)*dr/2</text>
    <text x="25" y="300" font-size="11" fill="#64748b">Learns high-dimensional manifold topology without explicit coordinates.</text>
  </g>

  <!-- Panel 4: Soft Bayesian Likelihood Fading (-soft_bayesian) -->
  <g transform="translate(570, 445)">
    <rect width="490" height="330" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="15" y="15" width="28" height="28" rx="6" fill="#0284c7"/>
    <text x="29" y="34" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">SB</text>
    <text x="52" y="34" fill="#075985" font-size="16" font-weight="700">Soft Bayesian Likelihood Fading (-soft_bayesian)</text>
    
    <!-- Gaussian Falloff Curve -->
    <g transform="translate(30, 60)">
      <!-- Smooth Gaussian curve vs hard threshold step -->
      <path d="M 20 100 L 150 100 L 150 20 L 380 20" fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="4,4"/>
      <text x="260" y="35" font-size="11" font-weight="700" fill="#dc2626">Hard Pruning Step (|d - dcc| &gt; rlim)</text>

      <path d="M 20 100 C 120 100, 160 20, 260 20 C 320 20, 360 20, 380 20" fill="none" stroke="#0284c7" stroke-width="2.5"/>
      <text x="140" y="85" font-size="11" font-weight="700" fill="#0369a1">Soft Gaussian Likelihood Decay</text>

      <text x="20" y="125" font-size="11" fill="#64748b">Lower Bound Distance Difference &#x2192;</text>
    </g>

    <!-- Formula Box -->
    <rect x="15" y="260" width="460" height="55" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="25" y="280" font-size="11.5" font-weight="600" fill="#334155">Gaussian Fading:</text>
    <text x="135" y="280" font-family="monospace" font-size="11.5" font-weight="700" fill="#0284c7">L(c_l) = exp( - (d_lower - rlim)^2 / (2 * sigma^2) )</text>
    <text x="25" y="300" font-size="11" fill="#64748b">Prevents false-negative exclusions caused by measurement noise or jitter.</text>
  </g>
</svg>
"""
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg.strip())
    print(f"Generated: {svg_path}")


# -----------------------------------------------------------------------------
# 5. Multi-Tile & Joint Trajectory Fusion SVG
# -----------------------------------------------------------------------------
def generate_tiling_jtf_svg():
    svg_path = os.path.join(OUTPUT_DIR, "gric_tiling_jtf.svg")
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 800" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <linearGradient id="gradHeadTile" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#064e3b"/>
      <stop offset="100%" stop-color="#0f766e"/>
    </linearGradient>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
  </defs>

  <rect width="1100" height="800" fill="#f8fafc"/>

  <!-- Header -->
  <rect x="30" y="20" width="1040" height="70" rx="12" fill="url(#gradHeadTile)" filter="url(#shadow)"/>
  <text x="60" y="52" fill="#ffffff" font-size="22" font-weight="700">GRIC Spatial Tiling &amp; Joint Trajectory Fusion (JTF)</text>
  <text x="60" y="74" fill="#99f6e4" font-size="13">High-speed sub-image parallelization with Pass-2 joint trajectory boundary noise correction</text>
  <rect x="910" y="40" width="135" height="30" rx="15" fill="#0d9488"/>
  <text x="977" y="60" fill="#ffffff" font-size="12" font-weight="600" text-anchor="middle">Multi-Tile &amp; JTF</text>

  <!-- Left Column: Spatial Tiling & Boundary Noise Problem -->
  <g transform="translate(40, 110)">
    <rect width="490" height="650" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="15" y="15" width="28" height="28" rx="6" fill="#0f766e"/>
    <text x="29" y="34" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">P1</text>
    <text x="52" y="34" fill="#134e4a" font-size="17" font-weight="700">Pass 1: Independent Spatial Clustering</text>
    <text x="52" y="54" fill="#64748b" font-size="12">Parallel per-tile clustering &amp; boundary flicker</text>

    <!-- Tiling Split Graphic -->
    <g transform="translate(35, 75)">
      <!-- Big image frame -->
      <rect width="180" height="180" rx="6" fill="#f1f5f9" stroke="#0f766e" stroke-width="2"/>
      
      <!-- Seam lines -->
      <line x1="90" y1="0" x2="90" y2="180" stroke="#0f766e" stroke-width="2" stroke-dasharray="4,4"/>
      <line x1="0" y1="90" x2="180" y2="90" stroke="#0f766e" stroke-width="2" stroke-dasharray="4,4"/>

      <!-- Moving object on boundary -->
      <circle cx="92" cy="85" r="22" fill="#ef4444" opacity="0.8"/>
      <text x="92" y="90" font-size="10" font-weight="700" fill="#fff" text-anchor="middle">Object</text>

      <!-- Tile Labels -->
      <text x="45" y="45" font-size="12" font-weight="700" fill="#334155" text-anchor="middle">Tile 0</text>
      <text x="135" y="45" font-size="12" font-weight="700" fill="#334155" text-anchor="middle">Tile 1</text>
      <text x="45" y="135" font-size="12" font-weight="700" fill="#334155" text-anchor="middle">Tile 2</text>
      <text x="135" y="135" font-size="12" font-weight="700" fill="#334155" text-anchor="middle">Tile 3</text>

      <!-- Parallel Execution Callout -->
      <g transform="translate(210, 20)">
        <rect width="200" height="140" rx="6" fill="#f0fdfa" stroke="#ccfbf1"/>
        <text x="15" y="25" font-size="12" font-weight="700" fill="#115e59">3 Major Benefits:</text>
        <text x="15" y="50" font-size="11.5" fill="#134e4a">1. <tspan font-weight="600">4x Math Speedup</tspan> (2x2)</text>
        <text x="15" y="75" font-size="11.5" fill="#134e4a">2. <tspan font-weight="600">OpenMP Parallelism</tspan></text>
        <text x="15" y="100" font-size="11.5" fill="#134e4a">3. <tspan font-weight="600">100x Memory Drop</tspan></text>
        <text x="15" y="122" font-size="10.5" fill="#047857">(K_tile &lt;&lt; K_global)</text>
      </g>
    </g>

    <!-- The Problem: Boundary Noise Box -->
    <rect x="20" y="290" width="450" height="155" rx="8" fill="#fef2f2" stroke="#fca5a5"/>
    <text x="35" y="315" font-size="13" font-weight="700" fill="#991b1b">&#x26A0; The Tile-Boundary Noise Problem:</text>
    <text x="35" y="340" font-size="12" fill="#7f1d1d">When an event straddles a boundary seam, slight numerical noise</text>
    <text x="35" y="358" font-size="12" fill="#7f1d1d">causes one tile to switch cluster while others stay in place.</text>
    <text x="35" y="382" font-size="12" font-weight="600" fill="#b91c1c">Result: Spurious flickered joint state (0, 3, <tspan text-decoration="underline" font-weight="700">5</tspan>, 1)</text>
    <text x="35" y="402" font-size="11.5" fill="#7f1d1d">Inflates total global state combinations artificially.</text>
    <text x="35" y="425" font-size="11" font-weight="600" fill="#dc2626">&#x27A8; Requires JTF Pass 2 to restore global consistency!</text>

    <!-- Options Box -->
    <rect x="20" y="465" width="450" height="160" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="35" y="490" font-size="13" font-weight="700" fill="#1e293b">Tiling Options:</text>
    <text x="35" y="515" font-size="12" fill="#334155"><tspan font-weight="600" font-family="monospace">-tiles &lt;NxM&gt;</tspan>: Grid configuration (e.g. -tiles 2x2, -tiles 4x4).</text>
    <text x="35" y="540" font-size="12" fill="#334155"><tspan font-weight="600" font-family="monospace">-tilemap &lt;file.fits&gt;</tspan>: Arbitrary custom region masks.</text>
    <text x="35" y="565" font-size="12" fill="#334155"><tspan font-weight="600" font-family="monospace">-tileconf &lt;file&gt;</tspan>: Per-tile rlim and maxcl overrides.</text>
    <text x="35" y="590" font-size="12" fill="#334155"><tspan font-weight="600" font-family="monospace">-xtile [mode]</tspan>: Live cross-tile conditional probability injection.</text>
  </g>

  <!-- Right Column: Pass 2 Joint Trajectory Fusion (JTF) -->
  <g transform="translate(570, 110)">
    <rect width="490" height="650" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="15" y="15" width="28" height="28" rx="6" fill="#0284c7"/>
    <text x="29" y="34" fill="#ffffff" font-size="13" font-weight="700" text-anchor="middle">JTF</text>
    <text x="52" y="34" fill="#0369a1" font-size="17" font-weight="700">Pass 2: Joint Trajectory Fusion (-jtf)</text>
    <text x="52" y="54" fill="#64748b" font-size="12">Trajectory lookup &amp; boundary smoothing engine</text>

    <!-- JTF Step-by-Step Graphic -->
    <g transform="translate(30, 80)">
      <!-- Step 1 -->
      <rect width="430" height="75" rx="6" fill="#f0f9ff" stroke="#bae6fd"/>
      <text x="15" y="22" font-size="12" font-weight="700" fill="#0369a1">1. Build Spatial &amp; Temporal Query Keys</text>
      <text x="15" y="42" font-size="11.5" fill="#0c4a6e">Spatial Key: Tile 0, 1, 3 current assignments = (0, 3, -, 1)</text>
      <text x="15" y="60" font-size="11.5" fill="#0c4a6e">Temporal Key: Previous frame tuple = (0, 3, 2, 1)</text>

      <!-- Step 2 -->
      <g transform="translate(0, 88)">
        <rect width="430" height="75" rx="6" fill="#f0f9ff" stroke="#bae6fd"/>
        <text x="15" y="22" font-size="12" font-weight="700" fill="#0369a1">2. Lookback Scan in History (-retrieval_window)</text>
        <text x="15" y="42" font-size="11.5" fill="#0c4a6e">Scans last 1,000 tuples for identical spatial+temporal contexts.</text>
        <text x="15" y="60" font-size="11.5" font-weight="700" fill="#0284c7">History match weight: cluster 2 = 0.95, cluster 5 = 0.01</text>
      </g>

      <!-- Step 3 -->
      <g transform="translate(0, 176)">
        <rect width="430" height="80" rx="6" fill="#f0fdf4" stroke="#86efac"/>
        <text x="15" y="22" font-size="12" font-weight="700" fill="#14532d">3. Posterior Multiplication &amp; Distance Verification</text>
        <text x="15" y="42" font-size="11.5" fill="#14532d">Fused posterior: P_fused[2] (0.285) &gt;&gt; P_fused[5] (0.006)</text>
        <text x="15" y="60" font-size="11.5" font-weight="700" fill="#15803d">&#x2714; Distance check: d(frame, anchor_2) &lt;= rlim &#x2192; Override accepted!</text>
      </g>
    </g>

    <!-- Corrected Result Box -->
    <rect x="20" y="375" width="450" height="110" rx="8" fill="#ecfdf5" stroke="#a7f3d0"/>
    <text x="35" y="405" font-size="13" font-weight="700" fill="#065f46">Resulting Correction:</text>
    <text x="35" y="430" font-size="12" fill="#047857">Tile 2 is overridden from transient flicker <tspan fill="#dc2626" font-weight="700">c_5</tspan> to true physical state <tspan fill="#16a34a" font-weight="700">c_2</tspan>.</text>
    <text x="35" y="455" font-size="12" font-weight="700" fill="#064e3b">Global Trajectory restored to valid manifold state (0, 3, 2, 1).</text>

    <!-- Safeguard Note -->
    <rect x="20" y="505" width="450" height="120" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>
    <text x="35" y="530" font-size="13" font-weight="700" fill="#1e293b">JTF Safety Invariants:</text>
    <text x="35" y="555" font-size="12" fill="#334155">&#x2022; <tspan font-weight="600">Never overrides new anchors:</tspan> Truly novel features are preserved.</text>
    <text x="35" y="578" font-size="12" fill="#334155">&#x2022; <tspan font-weight="600">Hard distance limit enforced:</tspan> Override must strictly satisfy <tspan font-family="monospace">d &lt;= rlim</tspan>.</text>
    <text x="35" y="601" font-size="12" fill="#334155">&#x2022; <tspan font-weight="600">Zero false overrides:</tspan> Ensures physical realism.</text>
  </g>
</svg>
"""
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg.strip())
    print(f"Generated: {svg_path}")


# -----------------------------------------------------------------------------
# 6. Options Map & Decision Cheatsheet SVG
# -----------------------------------------------------------------------------
def generate_options_map_svg():
    svg_path = os.path.join(OUTPUT_DIR, "gric_options_map.svg")
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 820" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <linearGradient id="gradHeadOpt" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#18181b"/>
      <stop offset="100%" stop-color="#27272a"/>
    </linearGradient>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
  </defs>

  <rect width="1100" height="820" fill="#f8fafc"/>

  <!-- Header -->
  <rect x="30" y="20" width="1040" height="70" rx="12" fill="url(#gradHeadOpt)" filter="url(#shadow)"/>
  <text x="60" y="52" fill="#ffffff" font-size="22" font-weight="700">GRIC CLI Options &amp; Tuning Matrix Cheatsheet</text>
  <text x="60" y="74" fill="#a1a1aa" font-size="13">Complete reference map organized by functional domain and recommended presets</text>
  <rect x="910" y="40" width="135" height="30" rx="15" fill="#3b82f6"/>
  <text x="977" y="60" fill="#ffffff" font-size="12" font-weight="600" text-anchor="middle">CLI Reference</text>

  <!-- 6 Grid Category Boxes -->

  <!-- 1. Core Clustering & Thresholds -->
  <g transform="translate(40, 110)">
    <rect width="325" height="305" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="12" y="12" width="24" height="24" rx="4" fill="#ea580c"/>
    <text x="24" y="28" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">C</text>
    <text x="44" y="29" fill="#9a3412" font-size="14" font-weight="700">Core &amp; Thresholds</text>

    <g transform="translate(15, 50)">
      <text x="0" y="15" font-size="12" font-family="monospace" font-weight="700" fill="#ea580c">rlim &lt;val&gt;</text>
      <text x="0" y="30" font-size="11" fill="#475569">Max Euclidean radius threshold for cluster match.</text>

      <text x="0" y="55" font-size="12" font-family="monospace" font-weight="700" fill="#ea580c">a&lt;factor&gt; / -auto_rlim</text>
      <text x="0" y="70" font-size="11" fill="#475569">Auto-scales rlim from NN sample scan.</text>

      <text x="0" y="95" font-size="12" font-family="monospace" font-weight="700" fill="#ea580c">-maxcl &lt;N&gt;</text>
      <text x="0" y="110" font-size="11" fill="#475569">Maximum cluster capacity limit (default 10,000).</text>

      <text x="0" y="135" font-size="12" font-family="monospace" font-weight="700" fill="#ea580c">-discard_frac &lt;f&gt;</text>
      <text x="0" y="150" font-size="11" fill="#475569">Fraction of inactive clusters to discard on limit.</text>

      <text x="0" y="175" font-size="12" font-family="monospace" font-weight="700" fill="#ea580c">-ncpu &lt;N&gt;</text>
      <text x="0" y="190" font-size="11" fill="#475569">Worker threads for OpenMP parallelism.</text>

      <text x="0" y="215" font-size="12" font-family="monospace" font-weight="700" fill="#ea580c">-scandist &lt;N&gt;</text>
      <text x="0" y="230" font-size="11" fill="#475569">Pre-scan first N frames for distance statistics.</text>
    </g>
  </g>

  <!-- 2. Pruning & Distance Geometry -->
  <g transform="translate(385, 110)">
    <rect width="325" height="305" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="12" y="12" width="24" height="24" rx="4" fill="#059669"/>
    <text x="24" y="28" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">P</text>
    <text x="44" y="29" fill="#064e3b" font-size="14" font-weight="700">Pruning &amp; Geometry</text>

    <g transform="translate(15, 50)">
      <text x="0" y="15" font-size="12" font-family="monospace" font-weight="700" fill="#059669">-te4</text>
      <text x="0" y="30" font-size="11" fill="#475569">4-point pruning using 2 measured anchors.</text>

      <text x="0" y="55" font-size="12" font-family="monospace" font-weight="700" fill="#059669">-te5</text>
      <text x="0" y="70" font-size="11" fill="#475569">5-point pruning using 3D simplex bounds.</text>

      <text x="0" y="95" font-size="12" font-family="monospace" font-weight="700" fill="#059669">-sparse_dcc</text>
      <text x="0" y="110" font-size="11" fill="#475569">Maintains interval bounds instead of dense DCC.</text>

      <text x="0" y="135" font-size="12" font-family="monospace" font-weight="700" fill="#059669">-sparse_dcc_extra_evals &lt;N&gt;</text>
      <text x="0" y="150" font-size="11" fill="#475569">Tightens sparse DCC bounds on demand.</text>

      <text x="0" y="175" font-size="12" font-family="monospace" font-weight="700" fill="#059669">-no_dcc</text>
      <text x="0" y="190" font-size="11" fill="#475569">Disables inter-cluster distance matrix storage.</text>

      <text x="0" y="215" font-size="12" font-family="monospace" font-weight="700" fill="#059669">-dcc &lt;file&gt;</text>
      <text x="0" y="230" font-size="11" fill="#475569">Exports complete pairwise cluster distance matrix.</text>
    </g>
  </g>

  <!-- 3. Entropy Scheduling -->
  <g transform="translate(730, 110)">
    <rect width="325" height="305" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="12" y="12" width="24" height="24" rx="4" fill="#9333ea"/>
    <text x="24" y="28" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">E</text>
    <text x="44" y="29" fill="#581c87" font-size="14" font-weight="700">Entropy &amp; Gating</text>

    <g transform="translate(15, 50)">
      <text x="0" y="15" font-size="12" font-family="monospace" font-weight="700" fill="#9333ea">-entropy</text>
      <text x="0" y="30" font-size="11" fill="#475569">Shannon entropy target selection scheduling.</text>

      <text x="0" y="55" font-size="12" font-family="monospace" font-weight="700" fill="#9333ea">-entropy_fast</text>
      <text x="0" y="70" font-size="11" fill="#475569">Fast popcount bitmask entropy surrogate.</text>

      <text x="0" y="95" font-size="12" font-family="monospace" font-weight="700" fill="#9333ea">-entropy_gate &lt;t&gt;</text>
      <text x="0" y="110" font-size="11" fill="#475569">Bypasses entropy if max candidate prob &gt; t.</text>

      <text x="0" y="135" font-size="12" font-family="monospace" font-weight="700" fill="#9333ea">-entropy_first_gate</text>
      <text x="0" y="150" font-size="11" fill="#475569">Forces greedy probe on 1st step of each frame.</text>

      <text x="0" y="175" font-size="12" font-family="monospace" font-weight="700" fill="#9333ea">-entropy_max_targets &lt;N&gt;</text>
      <text x="0" y="190" font-size="11" fill="#475569">Limits entropy evaluation to top N candidates.</text>

      <text x="0" y="215" font-size="12" font-family="monospace" font-weight="700" fill="#9333ea">-entropy_min_prob &lt;p&gt;</text>
      <text x="0" y="230" font-size="11" fill="#475569">Ignores candidate clusters with prob &lt; p.</text>
    </g>
  </g>

  <!-- 4. Priors & Prediction -->
  <g transform="translate(40, 435)">
    <rect width="325" height="340" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="12" y="12" width="24" height="24" rx="4" fill="#2563eb"/>
    <text x="24" y="28" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">L</text>
    <text x="44" y="29" fill="#1e3a8a" font-size="14" font-weight="700">Priors &amp; Prediction</text>

    <g transform="translate(15, 50)">
      <text x="0" y="15" font-size="12" font-family="monospace" font-weight="700" fill="#2563eb">-gprob</text>
      <text x="0" y="30" font-size="11" fill="#475569">Learns geometric probabilities from visitors.</text>

      <text x="0" y="55" font-size="12" font-family="monospace" font-weight="700" fill="#2563eb">-tm &lt;coeff&gt;</text>
      <text x="0" y="70" font-size="11" fill="#475569">Markov transition matrix mixing weight.</text>

      <text x="0" y="95" font-size="12" font-family="monospace" font-weight="700" fill="#2563eb">-pred [len,h,n]</text>
      <text x="0" y="110" font-size="11" fill="#475569">Multi-step trajectory sequence predictor.</text>

      <text x="0" y="135" font-size="12" font-family="monospace" font-weight="700" fill="#2563eb">-dprob &lt;val&gt;</text>
      <text x="0" y="150" font-size="11" fill="#475569">Recency score increment upon cluster match.</text>

      <text x="0" y="175" font-size="12" font-family="monospace" font-weight="700" fill="#2563eb">-soft_bayesian</text>
      <text x="0" y="190" font-size="11" fill="#475569">Gaussian likelihood fading instead of hard cut.</text>

      <text x="0" y="215" font-size="12" font-family="monospace" font-weight="700" fill="#2563eb">-soft_bayesian_sigma &lt;s&gt;</text>
      <text x="0" y="230" font-size="11" fill="#475569">Standard deviation for soft likelihood decay.</text>
    </g>
  </g>

  <!-- 5. Spatial Tiling & JTF -->
  <g transform="translate(385, 435)">
    <rect width="325" height="340" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="12" y="12" width="24" height="24" rx="4" fill="#0d9488"/>
    <text x="24" y="28" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">T</text>
    <text x="44" y="29" fill="#115e59" font-size="14" font-weight="700">Tiling &amp; Multi-Tile</text>

    <g transform="translate(15, 50)">
      <text x="0" y="15" font-size="12" font-family="monospace" font-weight="700" fill="#0d9488">-tiles &lt;NxM&gt;</text>
      <text x="0" y="30" font-size="11" fill="#475569">Partitions image into NxM independent sub-tiles.</text>

      <text x="0" y="55" font-size="12" font-family="monospace" font-weight="700" fill="#0d9488">-tilemap &lt;file.fits&gt;</text>
      <text x="0" y="70" font-size="11" fill="#475569">Custom integer mask for non-rectangular tiles.</text>

      <text x="0" y="95" font-size="12" font-family="monospace" font-weight="700" fill="#0d9488">-tileconf &lt;file&gt;</text>
      <text x="0" y="110" font-size="11" fill="#475569">Per-tile configuration overrides (rlim, maxcl).</text>

      <text x="0" y="135" font-size="12" font-family="monospace" font-weight="700" fill="#0d9488">-jtf</text>
      <text x="0" y="150" font-size="11" fill="#475569">Joint Trajectory Fusion Pass 2 for boundary noise.</text>

      <text x="0" y="175" font-size="12" font-family="monospace" font-weight="700" fill="#0d9488">-retrieval_window &lt;N&gt;</text>
      <text x="0" y="190" font-size="11" fill="#475569">Lookback window size for JTF matching (1000).</text>

      <text x="0" y="215" font-size="12" font-family="monospace" font-weight="700" fill="#0d9488">-xtile / -cpt</text>
      <text x="0" y="230" font-size="11" fill="#475569">Live cross-tile prior conditional probability table.</text>
    </g>
  </g>

  <!-- 6. Recommended Preset Recipes -->
  <g transform="translate(730, 435)">
    <rect width="325" height="340" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="12" y="12" width="24" height="24" rx="4" fill="#475569"/>
    <text x="24" y="28" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">R</text>
    <text x="44" y="29" fill="#1e293b" font-size="14" font-weight="700">Recommended Presets</text>

    <g transform="translate(15, 50)">
      <!-- Preset 1 -->
      <rect width="295" height="65" rx="6" fill="#eff6ff" stroke="#bfdbfe"/>
      <text x="10" y="18" font-size="11.5" font-weight="700" fill="#1e3a8a">1. Smooth Video / High FPS:</text>
      <text x="10" y="35" font-family="monospace" font-size="10.5" fill="#2563eb">./gric-cluster a1.5 -tm 0.8 -pred</text>
      <text x="10" y="52" font-size="10.5" fill="#475569">Leverages temporal continuity &amp; greedy bypass.</text>

      <!-- Preset 2 -->
      <g transform="translate(0, 75)">
        <rect width="295" height="65" rx="6" fill="#faf5ff" stroke="#e9d5ff"/>
        <text x="10" y="18" font-size="11.5" font-weight="700" fill="#581c87">2. Complex / High-Dim Manifolds:</text>
        <text x="10" y="35" font-family="monospace" font-size="10.5" fill="#7e22ce">-entropy -gprob -te5 -sparse_dcc</text>
        <text x="10" y="52" font-size="10.5" fill="#475569">Maximizes pruning &amp; information gain.</text>
      </g>

      <!-- Preset 3 -->
      <g transform="translate(0, 150)">
        <rect width="295" height="65" rx="6" fill="#f0fdf4" stroke="#86efac"/>
        <text x="10" y="18" font-size="11.5" font-weight="700" fill="#14532d">3. Large Images (512x512+):</text>
        <text x="10" y="35" font-family="monospace" font-size="10.5" fill="#15803d">-tiles 2x2 -jtf -ncpu 8</text>
        <text x="10" y="52" font-size="10.5" fill="#475569">Spatial sub-division + fusion for speed &amp; low RAM.</text>
      </g>
    </g>
  </g>
</svg>
"""
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg.strip())
    print(f"Generated: {svg_path}")


if __name__ == "__main__":
    generate_master_pipeline_svg()
    generate_pruning_geometry_svg()
    generate_target_selection_svg()
    generate_priors_prediction_svg()
    generate_tiling_jtf_svg()
    generate_options_map_svg()
    print("All 6 SVG diagrams generated successfully.")
