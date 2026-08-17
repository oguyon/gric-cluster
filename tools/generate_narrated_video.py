#!/usr/bin/env python3
"""
GRIC Full Narrated Video Explainer Generator (Enhanced Natural Neural Narration)
Features:
- High-quality natural neural voice synthesis (edge-tts) at energetic, dynamic pace (+18% rate)
- Detailed animated visual breakdown with step-by-step frame ingestion
- Explicit Scene 2 demonstrating anchor creation, frame wandering within rlim, boundary crossing,
  and sequential anchor spawning.
"""

import os
import sys
import json
import asyncio
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Ellipse, Polygon
from matplotlib.gridspec import GridSpec
import edge_tts

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
TEMP_DIR = os.path.join(OUTPUT_DIR, "video_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

FINAL_MP4 = os.path.join(OUTPUT_DIR, "gric_explainer.mp4")
FINAL_GIF = os.path.join(OUTPUT_DIR, "gric_explainer.gif")

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = '#0b1120'

FPS = 25
VOICE = "en-US-ChristopherNeural"
RATE = "+18%"

SCENES = [
    {
        "id": "scene1_intro",
        "title": "1. Introduction: High-Speed Sequential Stream Clustering",
        "subtitle": "Sequential streaming on high-D datasets: Minimizing distance measurements per frame",
        "text": "Welcome to Gric, Geometric Real-Time Image Clustering. Gric is optimized to cluster data frames sequentially as they arrive from high-speed streams using distance measurements. When processing high-dimensional datasets, computing Euclidean distance between high-dimensional frames is the primary computational bottleneck. Therefore, the core objective of Gric is to strictly minimize the number of distance measurements evaluated per frame. As incoming frames travel along continuous paths, like our benchmark 2D spiral, every frame is strictly situated within an active cluster sphere under distance threshold R-lim, and the instant a frame leaves a cluster boundary, a new anchor is spawned immediately.",
        "subtitles": [
            (0.00, 0.14, "Welcome to GRIC: Geometric Real-Time Image Clustering."),
            (0.14, 0.32, "GRIC clusters frames sequentially from high-speed streams using distance metrics."),
            (0.32, 0.50, "In high dimensions, evaluating Euclidean distance is the primary computational bottleneck."),
            (0.50, 0.68, "Therefore, GRIC's central objective is strictly minimizing distance calls per frame."),
            (0.68, 0.85, "Along continuous paths (e.g. 2D spiral), frames are situated inside active cluster spheres."),
            (0.85, 1.00, "The instant a frame leaves a cluster boundary (d > rlim), a new anchor is spawned!")
        ]
    },
    {
        "id": "scene2_anchors",
        "title": "2. Anchor Formation & Distance Threshold (rlim)",
        "subtitle": "Testing existing clusters before spawning new anchors on boundaries",
        "text": "Here is how Gric forms clusters. When the very first frame arrives, it establishes Anchor C-zero, surrounded by its R-lim radius. When subsequent incoming frames arrive, Gric will always first try to allocate each new frame to existing clusters by checking if its distance is within R-lim. If a match is found, the frame is assigned to that cluster without creating anything new. Only if the frame does not belong to any existing cluster, having crossed beyond all existing R-lim boundaries, does Gric create a new cluster anchor on the boundary. This leads to the critical question: how can Gric test existing clusters quickly without computing distances to every single one?",
        "subtitles": [
            (0.00, 0.16, "When the first frame arrives, it establishes Anchor c0 with radius rlim."),
            (0.16, 0.36, "For subsequent frames, GRIC ALWAYS tests existing clusters first (checking d <= rlim)."),
            (0.36, 0.54, "If matched, the frame is assigned to that cluster with 0 new clusters created!"),
            (0.54, 0.78, "Only when crossing outside all existing boundaries does GRIC spawn a new anchor on the boundary."),
            (0.78, 1.00, "Key question: How to test existing clusters without calculating distances to all of them?")
        ]
    },
    {
        "id": "scene3_pruning",
        "title": "3. Geometric Pruning via the Triangle Inequality",
        "subtitle": "Eliminating impossible candidate clusters without computing distances",
        "text": "The answer is geometric pruning using the triangle inequality. Consider three points: an incoming query frame f, a pivot Anchor A whose distance we measure, and a candidate Anchor B whose distance to Anchor A is precomputed. These three points form a triangle. By the triangle inequality, the distance from our frame to Anchor B must be at least the known distance between A and B, minus the measured distance to Anchor A. If this calculated lower bound exceeds R-lim, the frame cannot belong to Anchor B, and Anchor B is pruned with zero distance computations. A single measurement can thus eliminate dozens of candidate clusters simultaneously across the dataset. Furthermore, Gric extends this distance geometry from sets of three points to higher-order bounds with sets of four and five points.",
        "subtitles": [
            (0.00, 0.16, "The answer is geometric pruning using the metric triangle inequality."),
            (0.16, 0.34, "Consider 3 points: query frame f, measured pivot cA, and candidate cB with known distance d(cA, cB)."),
            (0.34, 0.54, "By triangle inequality: lower bound d(f, cB) >= | d(cA, cB) - d(f, cA) |."),
            (0.54, 0.74, "If lower bound > rlim, candidate cB is pruned with 0 distance computations!"),
            (0.74, 0.88, "1 pivot measurement simultaneously eliminates dozens of candidates across the dataset."),
            (0.88, 1.00, "GRIC extends this geometry from 3 points to 4-point (-te4) and 5-point (-te5) simplex bounds.")
        ]
    },
    {
        "id": "scene4_entropy",
        "title": "4. Target Selection: Greedy vs Shannon Entropy (-entropy)",
        "subtitle": "Information-theoretic active search scheduling to maximize information gain",
        "text": "Target selection determines which cluster anchor to evaluate next. In default Greedy mode, Gric hopes for early success by testing the most likely candidates first. In contrast, Entropy mode seeks to maximize information gain. Consider our 2D spiral manifold. Measuring the distance to the center of the spiral maximizes information gain, because the measured radius unambiguously resolves where on the spiral the incoming frame is located with a single measurement. Entropy mode computes the expected Shannon entropy of the candidate distribution to schedule such highly informative pivots. To support this, Gric maintains and dynamically updates a probability distribution in memory based on recent patterns and frame allocations.",
        "subtitles": [
            (0.00, 0.15, "Target selection determines which cluster anchor to evaluate next."),
            (0.15, 0.32, "Greedy mode tests the most likely candidates first, hoping for early success."),
            (0.32, 0.48, "In contrast, Entropy mode (-entropy) seeks to maximize Shannon information gain."),
            (0.48, 0.72, "On a spiral, measuring distance to center (radius) unambiguously resolves position in 1 call!"),
            (0.72, 0.88, "Entropy mode computes expected posterior entropy to schedule highly informative pivots."),
            (0.88, 1.00, "To support this, GRIC dynamically updates an in-memory probability distribution.")
        ]
    },
    {
        "id": "scene5_priors",
        "title": "5. Prior Modeling & Topological Learning (-tm, -pred, -gprob)",
        "subtitle": "Fusing Markov transitions, sequence forecasting, and visitor geometry",
        "text": "Gric layers multiple prior probability models to accelerate search order. The transition matrix option, T M, learns Markov transition probabilities between clusters over time. The sequence predictor, pred, scans historical assignment logs to forecast multi-step trajectories. Simultaneously, G-prob dynamically updates spatial probabilities by comparing visitor measurement history, while soft-bayesian prevents false exclusions in noisy streams with smooth Gaussian likelihood decay.",
        "subtitles": [
            (0.00, 0.18, "GRIC layers multiple prior probability models to accelerate search order."),
            (0.18, 0.38, "Transition matrix (-tm) learns Markov transition probabilities between clusters over time."),
            (0.38, 0.58, "Sequence predictor (-pred) scans historical assignment logs to forecast multi-step paths."),
            (0.58, 0.80, "Visitor geometry (-gprob) dynamically updates spatial probabilities from visitor history."),
            (0.80, 1.00, "Soft Bayesian (-soft_bayesian) handles noisy data with smooth Gaussian likelihood decay.")
        ]
    },
    {
        "id": "scene6_tiling",
        "title": "6. Spatial Tiling & Joint Trajectory Fusion (-tiles, -jtf)",
        "subtitle": "Sub-image OpenMP parallelism with Pass 2 boundary noise correction",
        "text": "For high-dimension images, Gric provides a multi-tile architecture. Partitioning the frame into an N by M grid of sub-tiles provides major benefits: it drastically accelerates compute speed through OpenMP parallelism, maximizes CPU cache locality, and unlocks cross-entropy information across spatial regions. Instead of assigning a single global cluster index, tiling produces a joint cluster tuple, such as tuple 0, 3, 2, 1, representing the simultaneous state of all sub-regions. This tuple is vastly richer in information than a single scalar cluster index. In Pass 2, Joint Trajectory Fusion leverages these rich tuple keys against recent history to eliminate boundary noise and seam flickering, while strictly preserving the hard distance threshold.",
        "subtitles": [
            (0.00, 0.16, "For high-dimension images, GRIC provides a multi-tile architecture (-tiles NxM)."),
            (0.16, 0.36, "Tiling accelerates OpenMP threads, maximizes CPU L1/L2 cache locality, and captures cross-entropy."),
            (0.36, 0.56, "Instead of 1 scalar index, tiling produces a rich joint tuple (e.g. 0, 3, 2, 1) across all tiles."),
            (0.56, 0.76, "This joint tuple is vastly richer in spatial-temporal information than a single scalar index."),
            (0.76, 1.00, "Pass 2 Joint Trajectory Fusion (-jtf) eliminates seam flicker while verifying d <= rlim.")
        ]
    },
    {
        "id": "scene7_summary",
        "title": "7. Summary & CLI Tuning Matrix",
        "subtitle": "Ultra-fast geometric clustering optimized for real-time sensor streams",
        "text": "From adaptive geometric pruning to information-theoretic entropy scheduling and multi-tile trajectory fusion, Gric delivers ultra-fast, robust, distance-based clustering for high-throughput scientific data. Check out the documentation and interactive simulator to tune the optimal parameters for your application.",
        "subtitles": [
            (0.00, 0.30, "From geometric pruning to entropy scheduling and multi-tile trajectory fusion..."),
            (0.30, 0.65, "...GRIC delivers ultra-fast, robust distance-based clustering for high-throughput streams."),
            (0.65, 1.00, "Explore the documentation and interactive simulator to tune parameters for your application!")
        ]
    }
]

def get_audio_duration(file_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(res.stdout.strip())

async def generate_single_narration(scene):
    mp3_path = os.path.join(TEMP_DIR, f"{scene['id']}.mp3")
    wav_path = os.path.join(TEMP_DIR, f"{scene['id']}.wav")
    
    # Generate neural voice MP3 via edge-tts
    communicate = edge_tts.Communicate(scene["text"], VOICE, rate=RATE)
    await communicate.save(mp3_path)
    
    # Convert to 44.1kHz stereo WAV with 0.8s padding at end
    subprocess.run([
        "ffmpeg", "-y", "-i", mp3_path,
        "-af", "apad=pad_dur=0.8",
        "-ar", "44100", "-ac", "2", wav_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    dur = get_audio_duration(wav_path)
    return wav_path, dur

async def generate_all_narrations_async():
    print(f"Generating dynamic neural voiceovers using {VOICE} ({RATE})...")
    audio_files = []
    durations = []
    for idx, scene in enumerate(SCENES):
        wav_path, dur = await generate_single_narration(scene)
        audio_files.append(wav_path)
        durations.append(dur)
        print(f"  Scene {idx+1} ({scene['id']}): {dur:.2f}s generated.")
    return audio_files, durations

def generate_all_narrations():
    return asyncio.run(generate_all_narrations_async())

def render_scene_video(scene_idx, scene, duration, audio_wav, mode='mp4'):
    is_gif = (mode == 'gif')
    num_frames = int(duration * FPS)
    video_raw_path = os.path.join(TEMP_DIR, f"{scene['id']}_{mode}_video.mp4")
    scene_with_audio_path = os.path.join(TEMP_DIR, f"{scene['id']}_{mode}_muxed.mp4")
    
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor('#0f172a')
    
    gs = GridSpec(2, 2, width_ratios=[1.30, 1], height_ratios=[1, 1], figure=fig,
                  left=0.05, right=0.95, bottom=0.12, top=0.91, wspace=0.20, hspace=0.30)
    
    ax_main = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    # Dynamic styling scale based on mode
    fs_sup = 20 if is_gif else 17
    fs_sub = 14.5 if is_gif else 12.5
    fs_main_title = 14 if is_gif else 12
    fs_side_title = 13 if is_gif else 11
    fs_ticks = 10.5 if is_gif else 9
    fs_header = 13.5 if is_gif else 11
    fs_bullet = 11.5 if is_gif else 9.5
    fs_banner = 12.0 if is_gif else 10.5
    s_anchor = 320 if is_gif else 180
    s_frame = 360 if is_gif else 200
    s_x = 480 if is_gif else 260
    lw_thick = 3.5 if is_gif else 2.5
    lw_arrow = 3.5 if is_gif else 2.5

    # Bottom Subtitle Banner (Synchronized with narration for video and GIF)
    sub_artist = fig.text(0.50, 0.048, "", ha='center', va='center',
                          fontsize=fs_sub, fontweight='bold', color='#f8fafc',
                          bbox=dict(boxstyle="round,pad=0.55", facecolor="#020617", edgecolor="#38bdf8", lw=2.0, alpha=0.98))

    # Preset anchors for scenes 1, 3, 4, 5
    anchors = np.array([
        [2.5, 3.0],   # c0
        [7.2, 7.5],   # c1
        [3.0, 7.0],   # c2
        [7.5, 2.5],   # c3
        [5.0, 5.0]    # c4
    ])
    rlim_std = 1.35

    def draw_frame(frame_num):
        ax_main.clear()
        ax_top.clear()
        ax_bot.clear()

        prog = frame_num / float(max(1, num_frames - 1))

        # Base styling
        for ax in [ax_main, ax_top, ax_bot]:
            ax.set_facecolor('#1e293b')
            for spine in ax.spines.values():
                spine.set_color('#334155')
            ax.tick_params(colors='#94a3b8', labelsize=fs_ticks)

        # Supertitle
        fig.suptitle(f"GRIC: {scene['title']}", 
                     fontsize=fs_sup, fontweight='bold', color='#38bdf8', y=0.968)

        # Update Subtitle Banner
        current_sub = ""
        for t_start, t_end, txt in scene.get("subtitles", []):
            if t_start <= prog < t_end or (prog >= t_end and t_end == 1.0):
                current_sub = txt
                break
        if not current_sub and scene.get("subtitles"):
            current_sub = scene["subtitles"][-1][2]
        sub_artist.set_text(current_sub)

        # -------------------------------------------------------------
        # SCENE 1: Introduction & 2D Spiral Benchmark Sequential Stream
        # -------------------------------------------------------------
        if scene_idx == 0:
            ax_main.set_xlim(0, 10); ax_main.set_ylim(0, 10)
            ax_main.set_title("2D Spiral Benchmark Stream: Never Cluster-Less", color='#38bdf8', fontsize=fs_main_title, fontweight='bold')
            
            s1_rlim = 1.65
            # Precompute the exact 2D Archimedean spiral benchmark path
            N_steps = 1500
            t_grid = np.linspace(0, 1, N_steps)
            loops = 2.0
            r_max = 3.9
            r_grid = 0.35 + r_max * t_grid
            theta_grid = 2.0 * np.pi * loops * t_grid

            x_grid = 5.0 + r_grid * np.cos(theta_grid)
            y_grid = 5.0 + r_grid * np.sin(theta_grid)

            # Draw faint background spiral guide curve
            ax_main.plot(x_grid, y_grid, color='#475569', linestyle=':', linewidth=2.0 if is_gif else 1.5, alpha=0.6, zorder=2)

            # Build exact anchor spawn history online along spiral
            s1_anchors = [np.array([x_grid[0], y_grid[0]])]
            s1_anchor_times = [0.0]
            for step_i in range(1, N_steps):
                p_cur = np.array([x_grid[step_i], y_grid[step_i]])
                if np.hypot(p_cur[0] - s1_anchors[-1][0], p_cur[1] - s1_anchors[-1][1]) >= s1_rlim:
                    s1_anchors.append(p_cur)
                    s1_anchor_times.append(t_grid[step_i])

            cols = ['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#8b5cf6', '#14b8a6', '#f43f5e', '#6366f1']
            
            # Current frame position along spiral at prog
            cur_r = 0.35 + r_max * prog
            cur_theta = 2.0 * np.pi * loops * prog
            cur_x = 5.0 + cur_r * np.cos(cur_theta)
            cur_y = 5.0 + cur_r * np.sin(cur_theta)
            p_now = np.array([cur_x, cur_y])

            # Find active anchors spawned up to current prog
            active_anchors = [anc for anc, at in zip(s1_anchors, s1_anchor_times) if at <= prog]
            active_idx = len(active_anchors) - 1
            ca = active_anchors[active_idx]
            dist_to_cur = np.hypot(p_now[0] - ca[0], p_now[1] - ca[1])

            # Draw all spawned anchors along spiral
            for i, anc in enumerate(active_anchors):
                c_col = cols[i % len(cols)]
                ax_main.scatter(anc[0], anc[1], s=s_anchor, color=c_col, edgecolors='#fff', linewidth=2.0 if is_gif else 1.5, zorder=5)
                c_patch = Circle(anc, s1_rlim, fill=True, facecolor=c_col, alpha=0.15, edgecolor=c_col, linewidth=1.5, linestyle='--')
                ax_main.add_patch(c_patch)
                
                # Radial label offset away from center
                v_dx = anc[0] - 5.0
                v_dy = anc[1] - 5.0
                v_norm = max(0.2, np.hypot(v_dx, v_dy))
                lx = anc[0] + (v_dx / v_norm) * 0.38
                ly = anc[1] + (v_dy / v_norm) * 0.38
                ax_main.text(lx, ly, f"c_{i}", color=c_col, fontweight='bold', fontsize=12 if is_gif else 9.5, ha='center', va='center', zorder=6)

            # Draw EXACTLY ONE yellow incoming frame traveling along spiral
            ax_main.scatter(cur_x, cur_y, s=s_frame, color='#facc15', edgecolors='#ffffff', linewidth=2.8 if is_gif else 2.5, zorder=8)
            ax_main.text(cur_x, cur_y - 0.50, "f_i (Incoming Frame)", color='#facc15', fontweight='bold', fontsize=12.5 if is_gif else 10.5, ha='center', zorder=9)

            # Distance line to active matching anchor (strictly <= rlim)
            ax_main.plot([ca[0], cur_x], [ca[1], cur_y], color='#4ade80', linewidth=lw_thick, zorder=6)
            
            # Status banner
            ax_main.text(5.0, 0.7, f"Inside c_{active_idx} (d={dist_to_cur:.2f} <= rlim) — Manifold Coverage", 
                         color='#ffffff', fontweight='bold', fontsize=fs_banner, ha='center',
                         bbox=dict(boxstyle="round,pad=0.40", facecolor="#1e1b4b", edgecolor="#38bdf8", lw=1.8))

            # Top Right: Distance Evaluation Workload in High-D
            ax_top.set_title("Distance Calls per Frame (High-D Stream)", color='#94a3b8', fontsize=fs_side_title, fontweight='bold')
            k_vals = np.linspace(10, 1000, 50)
            naive_cost = k_vals
            gric_cost = 2.0 + np.log2(k_vals) * 0.4
            ax_top.plot(k_vals, naive_cost, color='#ef4444', linewidth=2.5, label="Naive Scan: K Calls (Heavy in High-D)")
            ax_top.plot(k_vals, gric_cost, color='#22c55e', linewidth=3.0, label="Gric: 1 ~ 3 Calls (Minimized)")
            ax_top.set_xlabel("Clusters (K)", color='#94a3b8', fontsize=10 if is_gif else 8.5, fontweight='bold')
            ax_top.set_ylabel("Distance Calls / Frame", color='#94a3b8', fontsize=10 if is_gif else 8.5, fontweight='bold')
            ax_top.legend(loc="upper left", facecolor='#0f172a', edgecolor='#334155', fontsize=9.5 if is_gif else 8.5, labelcolor='#f8fafc')

            # Bottom Right: High-D Streaming Objectives
            ax_bot.axis('off')
            ax_bot.text(0.05, 0.88, "HIGH-D STREAMING OBJECTIVES:", color='#38bdf8', fontweight='bold', fontsize=fs_header)
            ax_bot.text(0.05, 0.68, "• Online sequential clustering as frames arrive in stream.", color='#f8fafc', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.48, "• High-D Datasets: Distance calls are heavy bottleneck.", color='#facc15', fontweight='bold', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.28, "• Core Goal: Strictly MINIMIZE distance calls / frame.", color='#4ade80', fontweight='bold', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.08, "• C17 Engine: AVX2 SIMD intrinsics + OpenMP parallel.", color='#38bdf8', fontweight='bold', fontsize=fs_bullet)

        # -------------------------------------------------------------
        # SCENE 2: ANCHOR CREATION & DYNAMIC ALLOCATION STREAM
        # -------------------------------------------------------------
        elif scene_idx == 1:
            ax_main.set_xlim(0, 10); ax_main.set_ylim(0, 10)
            ax_main.set_title("Online Allocation Stream & Boundary Anchor Formation", color='#38bdf8', fontsize=fs_main_title, fontweight='bold')
            
            rlim_val = 1.8
            c_anchors = [
                np.array([1.8, 5.0]),  # c0 (covers 0.0 to 3.6)
                np.array([3.6, 5.0]),  # c1 (covers 1.8 to 5.4)
                np.array([5.4, 5.0]),  # c2 (covers 3.6 to 7.2)
                np.array([7.2, 5.0])   # c3 (covers 5.4 to 9.0)
            ]
            c_colors = ['#38bdf8', '#a855f7', '#10b981', '#f59e0b']

            # Continuous motion BACK AND FORTH across x
            if prog <= 0.50:
                is_return = False
                fwd_p = prog / 0.50
                cur_x = 1.8 + fwd_p * 6.2  # 1.8 to 8.0
                max_x = cur_x
            else:
                is_return = True
                ret_p = (prog - 0.50) / 0.50
                cur_x = 8.0 - ret_p * 6.2  # 8.0 back down to 1.8
                max_x = 8.0

            cur_y = 5.0

            # Determine active anchors formed so far
            c0_active = True
            c1_active = max_x >= 3.6
            c2_active = max_x >= 5.4
            c3_active = max_x >= 7.2

            # Determine active cluster assignment:
            if not is_return:
                if cur_x < 3.6:
                    active_idx = 0
                elif cur_x < 5.4:
                    active_idx = 1
                elif cur_x < 7.2:
                    active_idx = 2
                else:
                    active_idx = 3
            else:
                if cur_x >= 5.4:
                    active_idx = 3
                elif cur_x >= 3.6:
                    active_idx = 2
                elif cur_x >= 1.8:
                    active_idx = 1
                else:
                    active_idx = 0

            act_pos = c_anchors[active_idx]
            dist_to_act = abs(cur_x - act_pos[0])
            state_col = c_colors[active_idx]

            if not is_return:
                state_text = f"FORWARD: Inside c_{active_idx} (d={dist_to_act:.2f} <= rlim) -> Assigned c_{active_idx}"
            else:
                state_text = f"REVERSAL: Inside c_{active_idx} (d={dist_to_act:.2f} <= rlim) -> Retained until boundary"

            # Draw Anchors
            active_flags = [c0_active, c1_active, c2_active, c3_active]
            for i, is_act in enumerate(active_flags):
                if is_act:
                    pos = c_anchors[i]
                    col = c_colors[i]
                    is_cur_active = (i == active_idx)
                    lw_edge = (3.0 if is_gif else 2.5) if is_cur_active else (1.6 if is_gif else 1.5)
                    alpha_fill = 0.25 if is_cur_active else 0.10
                    ax_main.scatter(pos[0], pos[1], s=s_anchor, color=col, edgecolors='#fff', linewidth=2.2 if is_gif else 2.0, zorder=6)
                    c_patch = Circle(pos, rlim_val, fill=True, facecolor=col, alpha=alpha_fill, edgecolor=col, linewidth=lw_edge, linestyle='--' if not is_cur_active else '-')
                    ax_main.add_patch(c_patch)
                    ax_main.text(pos[0], pos[1] + 2.1, f"Anchor c_{i}", color=col, fontweight='bold', fontsize=13 if is_gif else 11.0, ha='center')

            # Staggered boundary callouts
            if c1_active:
                ax_main.plot([3.6, 3.6], [3.0, 6.8], color='#a855f7', linestyle=':', linewidth=1.8)
                ax_main.text(3.6, 2.5, "c_1 at boundary", color='#a855f7', fontsize=10.5 if is_gif else 9.0, ha='center', fontweight='bold')
            if c2_active:
                ax_main.plot([5.4, 5.4], [2.4, 6.8], color='#10b981', linestyle=':', linewidth=1.8)
                ax_main.text(5.4, 1.9, "c_2 at boundary", color='#10b981', fontsize=10.5 if is_gif else 9.0, ha='center', fontweight='bold')
            if c3_active:
                ax_main.plot([7.2, 7.2], [1.8, 6.8], color='#f59e0b', linestyle=':', linewidth=1.8)
                ax_main.text(7.2, 1.3, "c_3 at boundary", color='#f59e0b', fontsize=10.5 if is_gif else 9.0, ha='center', fontweight='bold')

            # Direction Arrow
            dir_x = 0.5 if not is_return else -0.5
            ax_main.annotate('', xy=(cur_x + dir_x, cur_y + 0.65), xytext=(cur_x, cur_y + 0.65),
                             arrowprops=dict(facecolor='#facc15', edgecolor='#ffffff', width=2.8 if is_gif else 2.0, headwidth=8 if is_gif else 6.0))
            ax_main.text(cur_x, cur_y + 0.95, "Trajectory Direction", color='#facc15', fontsize=10.5 if is_gif else 8.5, ha='center', fontweight='bold')

            # Draw Current Incoming Frame (Single Yellow Dot)
            ax_main.scatter(cur_x, cur_y, s=s_frame, color='#facc15', edgecolors='#ffffff', linewidth=2.8 if is_gif else 2.5, zorder=8)
            ax_main.text(cur_x, cur_y - 0.50, "f_i (Incoming Frame)", color='#facc15', fontweight='bold', fontsize=12.5 if is_gif else 10.5, ha='center')

            # Distance line to active anchor
            ax_main.plot([act_pos[0], cur_x], [act_pos[1], cur_y], color='#4ade80', linewidth=lw_thick, zorder=7)

            # Status Banner
            ax_main.text(5.0, 0.7, state_text, color='#ffffff', fontweight='bold', fontsize=fs_banner, ha='center',
                         bbox=dict(boxstyle="round,pad=0.40", facecolor="#1e1b4b", edgecolor=state_col, lw=1.8))

            # Dynamically Growing Member Count Bar Chart (Top Right)
            ax_top.set_title("Frames Ingested per Cluster (Growing)", color='#94a3b8', fontsize=fs_side_title, fontweight='bold')
            n_clusters = sum(active_flags)
            counts = [0.0, 0.0, 0.0, 0.0]
            if prog <= 0.50:
                counts[0] = min(1.0, fwd_p / (1.8/6.2)) * 25.0 if fwd_p > 0 else 0
                counts[1] = min(1.0, max(0.0, (cur_x - 3.6) / 1.8)) * 25.0 if cur_x >= 3.6 else 0
                counts[2] = min(1.0, max(0.0, (cur_x - 5.4) / 1.8)) * 25.0 if cur_x >= 5.4 else 0
                counts[3] = min(1.0, max(0.0, (cur_x - 7.2) / 0.8)) * 15.0 if cur_x >= 7.2 else 0
            else:
                counts[0] = 25.0 + (min(1.0, max(0.0, (1.8 - cur_x) / 1.8)) * 30.0 if cur_x < 1.8 else 0)
                counts[1] = 25.0 + (min(1.0, max(0.0, (3.6 - cur_x) / 1.8)) * 30.0 if cur_x < 3.6 else 0)
                counts[2] = 25.0 + (min(1.0, max(0.0, (5.4 - cur_x) / 1.8)) * 30.0 if cur_x < 5.4 else 0)
                counts[3] = 15.0 + min(1.0, (8.0 - cur_x) / 2.6) * 30.0
            
            bars = ax_top.bar([f"c_{i}" for i in range(n_clusters)], [int(counts[i]) for i in range(n_clusters)], 
                       color=c_colors[:n_clusters], width=0.55)
            ax_top.set_ylabel("Frames Assigned", color='#94a3b8', fontsize=10 if is_gif else 8.5, fontweight='bold')
            ax_top.set_ylim(0, 65)
            for b in bars:
                h = b.get_height()
                if h > 0:
                    ax_top.text(b.get_x() + b.get_width()/2.0, h + 1.5, f"{int(h)}", ha='center', color='#fff', fontweight='bold', fontsize=10.5 if is_gif else 9.0)

            # Bottom Right: Exact Boundary Ingestion & Revisit Rule
            ax_bot.axis('off')
            ax_bot.text(0.05, 0.88, "CLUSTER ALLOCATION & REVISIT RULE:", color='#38bdf8', fontweight='bold', fontsize=fs_header)
            ax_bot.text(0.05, 0.68, "1. Incoming frame tests existing clusters first (d <= rlim).", color='#4ade80', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.48, "2. Matched frames assigned directly to existing clusters.", color='#f8fafc', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.28, "3. Reversal: Point retained in active cluster until x < 5.4!", color='#fbbf24', fontweight='bold', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.08, "4. New anchors ONLY spawned when all existing tests fail.", color='#f87171', fontfamily='monospace', fontsize=fs_bullet)

        # -------------------------------------------------------------
        # SCENE 3: Geometric Pruning via the Triangle Inequality
        # -------------------------------------------------------------
        elif scene_idx == 2:
            ax_main.set_xlim(0, 10); ax_main.set_ylim(0, 10)
            ax_main.set_title("Triangle Inequality Metric Pruning (3-Point, -te4, -te5)", color='#10b981', fontsize=fs_main_title, fontweight='bold')
            
            s3_rlim = 1.50
            p_f = np.array([2.2, 3.2])    # Query frame f
            p_cA = np.array([3.2, 5.2])   # Pivot anchor cA
            p_cB = np.array([7.8, 8.0])   # Candidate anchor cB
            p_cC = np.array([8.2, 2.5])   # Candidate anchor cC

            d_meas = np.hypot(p_f[0] - p_cA[0], p_f[1] - p_cA[1])         # 2.24
            d_AB = np.hypot(p_cA[0] - p_cB[0], p_cA[1] - p_cB[1])         # 5.40
            d_AC = np.hypot(p_cA[0] - p_cC[0], p_cA[1] - p_cC[1])         # 5.69
            bound_B = d_AB - d_meas                                       # 3.16
            bound_C = d_AC - d_meas                                       # 3.45

            # 1. Pivot Anchor cA and Query Frame f (Always visible)
            ax_main.scatter(p_cA[0], p_cA[1], s=s_anchor, color='#38bdf8', edgecolors='#fff', linewidth=2.2 if is_gif else 2.0, zorder=6)
            ax_main.text(p_cA[0] - 0.4, p_cA[1] + 0.55, "Anchor cA (Pivot)", color='#38bdf8', fontweight='bold', fontsize=13 if is_gif else 11.0)
            c_patch_A = Circle(p_cA, s3_rlim, fill=True, facecolor='#38bdf8', alpha=0.12, edgecolor='#38bdf8', linestyle=':')
            ax_main.add_patch(c_patch_A)

            ax_main.scatter(p_f[0], p_f[1], s=s_frame, color='#facc15', edgecolors='#fff', linewidth=2.8 if is_gif else 2.5, zorder=8)
            ax_main.text(p_f[0] - 0.3, p_f[1] - 0.45, "f (Query Frame)", color='#facc15', fontweight='bold', fontsize=13 if is_gif else 11.0)

            # Measured distance edge (f to cA)
            ax_main.plot([p_f[0], p_cA[0]], [p_f[1], p_cA[1]], color='#facc15', linewidth=lw_thick, zorder=5)
            ax_main.text(1.4, 4.4, f"d(f, cA) = {d_meas:.2f}\n[MEASURED]", color='#facc15', fontweight='bold', fontsize=11.5 if is_gif else 10.0)

            # 2. Single Triangle (f, cA, cB) during main explanation (prog >= 0.20)
            if prog >= 0.20:
                ax_main.scatter(p_cB[0], p_cB[1], s=s_anchor, color='#a855f7', edgecolors='#fff', linewidth=2.2 if is_gif else 2.0, zorder=6)
                ax_main.text(p_cB[0] + 0.25, p_cB[1] + 0.2, "Anchor cB (Candidate)", color='#a855f7', fontweight='bold', fontsize=12.5 if is_gif else 10.5)
                c_patch_B = Circle(p_cB, s3_rlim, fill=True, facecolor='#a855f7', alpha=0.12, edgecolor='#a855f7', linestyle='--')
                ax_main.add_patch(c_patch_B)

                # Known DCC edge (cA to cB)
                ax_main.plot([p_cA[0], p_cB[0]], [p_cA[1], p_cB[1]], color='#38bdf8', linewidth=2.8 if is_gif else 2.5, linestyle='-', zorder=4)
                ax_main.text(5.2, 7.3, f"d(cA, cB) = {d_AB:.2f} (KNOWN)", color='#38bdf8', fontweight='bold', fontsize=11.5 if is_gif else 10.0)

                # Shaded Triangle (f, cA, cB)
                tri_B = Polygon([p_f, p_cA, p_cB], closed=True, facecolor='#818cf8', alpha=0.15, edgecolor='#818cf8', linestyle=':')
                ax_main.add_patch(tri_B)

                # Lower bound line (f to cB)
                ax_main.plot([p_f[0], p_cB[0]], [p_f[1], p_cB[1]], color='#ef4444', linewidth=2.5 if is_gif else 2.0, linestyle='--', zorder=4)
                ax_main.text(4.8, 4.8, f"Bound d(f, cB) >= {bound_B:.2f}", 
                             color='#f87171', fontweight='bold', fontsize=11.5 if is_gif else 9.5, rotation=41)

            # Pruning trigger for cB (prog >= 0.50)
            if prog >= 0.50:
                ax_main.scatter(p_cB[0], p_cB[1], s=s_x, color='#dc2626', marker='x', linewidth=4.5 if is_gif else 3.5, zorder=9)
                ax_main.text(p_cB[0], p_cB[1] + 0.85, f"PRUNED! ({bound_B:.2f} > {s3_rlim:.2f})", 
                             color='#f87171', fontweight='bold', fontsize=11.5 if is_gif else 10.5, ha='center',
                             bbox=dict(boxstyle="round,pad=0.25", facecolor="#450a0a", edgecolor="#ef4444", lw=1.5))

            # 3. Second Triangle (f, cA, cC) at end when showing dozens pruned (prog >= 0.75)
            if prog >= 0.75:
                ax_main.scatter(p_cC[0], p_cC[1], s=s_anchor, color='#10b981', edgecolors='#fff', linewidth=2.2 if is_gif else 2.0, zorder=6)
                ax_main.text(p_cC[0] + 0.25, p_cC[1] + 0.2, "Anchor cC (Candidate 2)", color='#10b981', fontweight='bold', fontsize=12.5 if is_gif else 10.5)
                c_patch_C = Circle(p_cC, s3_rlim, fill=True, facecolor='#10b981', alpha=0.12, edgecolor='#10b981', linestyle='--')
                ax_main.add_patch(c_patch_C)

                # Known DCC edge (cA to cC)
                ax_main.plot([p_cA[0], p_cC[0]], [p_cA[1], p_cC[1]], color='#38bdf8', linewidth=2.8 if is_gif else 2.5, linestyle='-', zorder=4)
                ax_main.text(5.5, 3.4, f"d(cA, cC) = {d_AC:.2f} (KNOWN)", color='#38bdf8', fontweight='bold', fontsize=11.5 if is_gif else 10.0)

                # Shaded Triangle (f, cA, cC)
                tri_C = Polygon([p_f, p_cA, p_cC], closed=True, facecolor='#34d399', alpha=0.15, edgecolor='#34d399', linestyle=':')
                ax_main.add_patch(tri_C)

                # Pruning trigger for cC
                ax_main.scatter(p_cC[0], p_cC[1], s=s_x, color='#dc2626', marker='x', linewidth=4.5 if is_gif else 3.5, zorder=9)
                ax_main.text(p_cC[0], p_cC[1] - 0.85, f"PRUNED! ({bound_C:.2f} > {s3_rlim:.2f})", 
                             color='#f87171', fontweight='bold', fontsize=11.5 if is_gif else 10.5, ha='center',
                             bbox=dict(boxstyle="round,pad=0.25", facecolor="#450a0a", edgecolor="#ef4444", lw=1.5))

            # Bottom Status Banner
            if prog >= 0.75:
                status_prune = "1 Pivot Call Pruned Both Candidates cB & cC (0 Extra Calls)"
            elif prog >= 0.50:
                status_prune = "Lower Bound (3.16) > rlim (1.50) -> Candidate cB Excluded!"
            else:
                status_prune = "Forming Triangle (f, cA, cB): Bound = | d(cA, cB) - d(f, cA) |"
            ax_main.text(5.0, 0.7, status_prune, color='#ffffff', fontweight='bold', fontsize=fs_banner, ha='center',
                         bbox=dict(boxstyle="round,pad=0.40", facecolor="#1e1b4b", edgecolor="#10b981", lw=1.8))

            # Top Right: Triangle Inequality Formulation
            ax_top.set_title("Triangle Inequality Formulation", color='#94a3b8', fontsize=fs_side_title, fontweight='bold')
            ax_top.axis('off')
            ax_top.text(0.05, 0.82, "TRIANGLE THEOREM:", color='#38bdf8', fontweight='bold', fontsize=fs_header)
            ax_top.text(0.05, 0.60, "d(f, cX) >= | d(cA, cX) - d(f, cA) |", color='#facc15', fontfamily='monospace', fontweight='bold', fontsize=11.5 if is_gif else 10.5)
            ax_top.text(0.05, 0.38, "• d(f, cA) = 2.24  [MEASURED from stream]", color='#4ade80', fontsize=fs_bullet)
            ax_top.text(0.05, 0.20, "• d(cA, cB) = 5.40  [PRECOMPUTED in DCC]", color='#38bdf8', fontsize=fs_bullet)
            ax_top.text(0.05, 0.02, "• Bound = 3.16 > rlim (1.50) -> PRUNED!", color='#f87171', fontweight='bold', fontsize=fs_bullet)

            # Bottom Right: Extension to 4-Point & 5-Point
            ax_bot.axis('off')
            ax_bot.text(0.05, 0.88, "MULTI-POINT EXTENSIONS:", color='#10b981', fontweight='bold', fontsize=fs_header)
            ax_bot.text(0.05, 0.68, "• 3-Point (Default): 1 pivot prunes candidate sets.", color='#f8fafc', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.48, "• 4-Point (-te4): 2 pivots triangulate orthogonal height.", color='#c084fc', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.08, "• -sparse_dcc: Dynamic bounds without dense O(K^2) RAM.", color='#fbbf24', fontfamily='monospace', fontsize=fs_bullet)

        # -------------------------------------------------------------
        # SCENE 4: Greedy vs Shannon Entropy on 2D Spiral
        # -------------------------------------------------------------
        elif scene_idx == 3:
            ax_main.set_xlim(0, 10); ax_main.set_ylim(0, 10)
            ax_main.set_title("Target Selection: Greedy vs Shannon Entropy (-entropy)", color='#c084fc', fontsize=fs_main_title, fontweight='bold')
            
            sp_theta = np.linspace(0.4, 4.2 * np.pi, 200)
            sp_r = 0.35 + 0.30 * sp_theta
            sp_x = 5.0 + sp_r * np.cos(sp_theta)
            sp_y = 5.0 + sp_r * np.sin(sp_theta)
            ax_main.plot(sp_x, sp_y, color='#334155', linewidth=3.0 if is_gif else 2.5, linestyle='--', zorder=2)
            
            # Spiral Center Pivot
            ax_main.scatter(5.0, 5.0, s=420 if is_gif else 220, color='#38bdf8', edgecolors='#ffffff', linewidth=2.5, zorder=7)
            ax_main.text(5.0, 4.3, "Spiral Center (c_center)\n[Max Info Gain]", color='#38bdf8', fontweight='bold', fontsize=11.5 if is_gif else 10.0, ha='center')

            # Spiral Cluster Anchors
            th_anchors = [1.2, 2.8, 4.6, 6.8, 9.2, 11.5]
            cl_anchors = []
            anchor_offsets = [
                (0.35, 0.25),   # c0
                (-0.45, 0.15),  # c1
                (0.35, -0.25),  # c2
                (0.35, 0.25),   # c3
                (-0.45, 0.15),  # c4
                (0.35, 0.25)    # c5
            ]
            for i, th in enumerate(th_anchors):
                r_a = 0.35 + 0.30 * th
                ax = 5.0 + r_a * np.cos(th)
                ay = 5.0 + r_a * np.sin(th)
                cl_anchors.append((ax, ay))
                ax_main.scatter(ax, ay, s=240 if is_gif else 140, color='#64748b', edgecolors='#ffffff', linewidth=1.8 if is_gif else 1.5, zorder=5)
                ox, oy = anchor_offsets[i]
                ax_main.text(ax + ox, ay + oy, f"c_{i}", color='#94a3b8', fontweight='bold', fontsize=11.5 if is_gif else 10.0, ha='center', va='center')

            # Query Frame f on outer winding
            f_th = 8.5
            f_r = 0.35 + 0.30 * f_th
            fx = 5.0 + f_r * np.cos(f_th)
            fy = 5.0 + f_r * np.sin(f_th)
            ax_main.scatter(fx, fy, s=s_frame, color='#facc15', edgecolors='#ffffff', linewidth=2.8 if is_gif else 2.5, zorder=9)
            ax_main.text(fx, fy + 0.55, "f (Incoming Frame)", color='#facc15', fontweight='bold', fontsize=13 if is_gif else 11.0, ha='center')

            # 1. Greedy Choice (prog < 0.45): Tests c_5 on outer winding
            if prog < 0.45:
                greedy_ax, greedy_ay = cl_anchors[5]
                ax_main.plot([fx, greedy_ax], [fy, greedy_ay], color='#ef4444', linewidth=2.8 if is_gif else 2.5, linestyle=':', zorder=6)
                ax_main.text(4.6, 1.4, "Greedy: Tests candidate c_5\nEliminates only 1 local winding", 
                             color='#f87171', fontweight='bold', fontsize=11.5 if is_gif else 10.0, ha='center',
                             bbox=dict(boxstyle="round,pad=0.30", facecolor="#450a0a", edgecolor="#ef4444", lw=1.5))
            else:
                # 2. Entropy Choice (prog >= 0.45): Measures distance to Spiral Center!
                ax_main.plot([fx, 5.0], [fy, 5.0], color='#22c55e', linewidth=lw_thick, zorder=8)
                ax_main.text(2.1, 6.8, f"Radius d={f_r:.2f}\nSolves spiral in 1 call!", 
                             color='#4ade80', fontweight='bold', fontsize=12.0 if is_gif else 10.5, ha='center',
                             bbox=dict(boxstyle="round,pad=0.30", facecolor="#064e3b", edgecolor="#22c55e", lw=1.5))
                
                # Highlight all wrong windings eliminated simultaneously
                for prune_id in [0, 1, 2, 5]:
                    ax_main.scatter(cl_anchors[prune_id][0], cl_anchors[prune_id][1], s=400 if is_gif else 200, color='#dc2626', marker='x', linewidth=3.8 if is_gif else 3.0, zorder=10)
                ax_main.text(5.0, 0.7, "Entropy Center Pivot Solves Manifold in 1 Measurement!", 
                             color='#ffffff', fontweight='bold', fontsize=fs_banner, ha='center',
                             bbox=dict(boxstyle="round,pad=0.40", facecolor="#1e1b4b", edgecolor="#c084fc", lw=1.8))

            # Top Right: Probability Distribution Collapse
            ax_top.set_title("In-Memory Probability Distribution", color='#94a3b8', fontsize=fs_side_title, fontweight='bold')
            if prog < 0.45:
                probs = [0.15, 0.20, 0.25, 0.18, 0.12, 0.10]
            else:
                probs = [0.0, 0.0, 0.0, 0.05, 0.95, 0.0]
            bars = ax_top.bar([f"c_{i}" for i in range(6)], probs, color=['#38bdf8' if p < 0.5 else '#22c55e' for p in probs], width=0.55)
            ax_top.set_ylabel("Posterior P(c_i)", color='#94a3b8', fontsize=10 if is_gif else 8.5, fontweight='bold')
            ax_top.set_ylim(0, 1.1)

            # Bottom Right: Entropy vs Greedy Logic
            ax_bot.axis('off')
            ax_bot.text(0.05, 0.88, "INFORMATION-THEORETIC SCHEDULING:", color='#c084fc', fontweight='bold', fontsize=fs_header)
            ax_bot.text(0.05, 0.68, "• Greedy Mode: Hopes for early match (highest prior).", color='#f87171', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.48, "• Entropy Mode: Maximizes Shannon info gain.", color='#4ade80', fontweight='bold', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.28, "• Center Pivot: 1 call solves spiral radius & cuts entropy.", color='#38bdf8', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.08, "• Priors: Dynamically updated from frame allocations.", color='#fbbf24', fontfamily='monospace', fontsize=fs_bullet)

        # -------------------------------------------------------------
        # SCENE 5: Priors & Topological Learning
        # -------------------------------------------------------------
        elif scene_idx == 4:
            ax_main.set_xlim(0, 10); ax_main.set_ylim(0, 10)
            ax_main.set_title("Priors, Markov Transitions & Topology Learning", color='#38bdf8', fontsize=fs_main_title, fontweight='bold')
            
            c_pos = [[2.5, 7.5], [7.5, 7.5], [5.0, 2.5]]
            c_labels = ["c_1", "c_2", "c_3"]
            for i in range(3):
                ax_main.scatter(c_pos[i][0], c_pos[i][1], s=480 if is_gif else 250, color='#4f46e5', edgecolors='#fff', linewidth=2.5 if is_gif else 2.0, zorder=5)
                ax_main.text(c_pos[i][0], c_pos[i][1], c_labels[i], color='#fff', fontweight='bold', fontsize=14 if is_gif else 12.0, ha='center', va='center', zorder=7)

            ax_main.annotate("", xy=(6.3, 7.5), xytext=(3.7, 7.5), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=lw_arrow))
            ax_main.text(5.0, 7.9, "T(1 -> 2) = 85% (-tm)", color='#38bdf8', fontweight='bold', fontsize=12.5 if is_gif else 11.0, ha='center')

            ax_main.annotate("", xy=(5.5, 3.5), xytext=(7.0, 6.5), arrowprops=dict(arrowstyle="->", color="#c084fc", lw=lw_arrow))
            ax_main.text(7.6, 5.0, "T(2 -> 3) = 80%", color='#c084fc', fontweight='bold', fontsize=12.0 if is_gif else 10.0, ha='center')

            ax_main.annotate("", xy=(3.0, 6.5), xytext=(4.5, 3.5), arrowprops=dict(arrowstyle="->", color="#10b981", lw=lw_arrow))
            ax_main.text(2.4, 5.0, "T(3 -> 1) = 75%", color='#10b981', fontweight='bold', fontsize=12.0 if is_gif else 10.0, ha='center')

            ax_main.text(5.0, 0.7, "Predictor: [4 -> 1 -> 7 -> 2] -> 92% Conf Next is c_9 (-pred)", 
                         color='#facc15', fontweight='bold', fontsize=fs_banner, ha='center',
                         bbox=dict(boxstyle="round,pad=0.40", facecolor="#1e1b4b", edgecolor="#9333ea", lw=1.8))

            # Top Right: Markov Mixing Equation
            ax_top.set_title("Probability Mixing Distribution", color='#94a3b8', fontsize=fs_side_title, fontweight='bold')
            ax_top.bar(["P_freq", "P_trans (-tm)", "P_mixed"], [0.25, 0.85, 0.73], color=['#38bdf8', '#4f46e5', '#22c55e'], width=0.55)
            ax_top.set_ylabel("Probability", color='#94a3b8', fontsize=10 if is_gif else 8.5, fontweight='bold')
            ax_top.set_ylim(0, 1.0)

            # Bottom Right: Prior Options
            ax_bot.axis('off')
            ax_bot.text(0.05, 0.88, "PRIOR & TOPOLOGY OPTIONS:", color='#38bdf8', fontweight='bold', fontsize=fs_header)
            ax_bot.text(0.05, 0.68, "• -tm <coeff>: Blends Markov transition history into prior.", color='#f8fafc', fontfamily='monospace', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.48, "• -pred [len,h,n]: Forecasts multi-step trajectories.", color='#f8fafc', fontfamily='monospace', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.28, "• -gprob: Learns spatial correlations from visitor history.", color='#f8fafc', fontfamily='monospace', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.08, "• -soft_bayesian: Gaussian likelihood decay for noise.", color='#4ade80', fontfamily='monospace', fontsize=fs_bullet)

        # -------------------------------------------------------------
        # SCENE 6: Spatial Tiling & Joint Trajectory Fusion
        # -------------------------------------------------------------
        elif scene_idx == 5:
            ax_main.set_xlim(0, 10); ax_main.set_ylim(0, 10)
            ax_main.set_title("Multi-Tile Architecture (-tiles 2x2) & Rich Joint Tuples (-jtf)", color='#0d9488', fontsize=fs_main_title, fontweight='bold')
            
            # Draw Tile Partition Grid
            ax_main.plot([5, 5], [0, 10], color='#0d9488', linewidth=2.8 if is_gif else 2.5, linestyle='--')
            ax_main.plot([0, 10], [5, 5], color='#0d9488', linewidth=2.8 if is_gif else 2.5, linestyle='--')

            # Sub-tile boxes and thread assignment
            t_data = [
                (2.5, 7.5, "Tile 0 (Thread 0)", "Sub-Cluster c_0", "#38bdf8"),
                (7.5, 7.5, "Tile 1 (Thread 1)", "Sub-Cluster c_3", "#a855f7"),
                (2.5, 2.5, "Tile 2 (Thread 2)", "Sub-Cluster c_2", "#10b981"),
                (7.5, 2.5, "Tile 3 (Thread 3)", "Sub-Cluster c_1", "#f59e0b"),
            ]
            for tx, ty, t_title, t_sub, t_col in t_data:
                ax_main.text(tx, ty + 0.65, t_title, color='#94a3b8', ha='center', va='center', fontsize=12.0 if is_gif else 11.0, fontweight='bold')
                ax_main.text(tx, ty - 0.40, t_sub, color=t_col, ha='center', va='center', fontsize=13.0 if is_gif else 11.5, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.30", facecolor="#0f172a", edgecolor=t_col, lw=1.5))

            # Seam crossing feature
            obj_x = 5.0 + 0.35 * np.sin(prog * 4 * np.pi)
            obj_y = 5.0 + 0.35 * np.cos(prog * 4 * np.pi)
            ax_main.scatter(obj_x, obj_y, s=400 if is_gif else 300, color='#f43f5e', edgecolors='#fff', linewidth=2.5 if is_gif else 2.0, zorder=6)
            ax_main.text(obj_x + 0.35, obj_y - 0.45, "Seam Crossing", color='#fb7185', fontweight='bold', fontsize=12.0 if is_gif else 10.5)

            # Joint Tuple Banner
            ax_main.text(5.0, 0.7, "Joint Tuple: (c_0, c_3, c_2, c_1) -> Key: (0, 3, 2, 1)", 
                         color='#ffffff', fontweight='bold', fontsize=fs_banner, ha='center',
                         bbox=dict(boxstyle="round,pad=0.40", facecolor="#134e4a", edgecolor="#2dd4bf", lw=1.8))

            # Top Right: Rich Tuple vs Scalar Index
            ax_top.set_title("Joint Tuple vs Scalar Cluster Index", color='#94a3b8', fontsize=fs_side_title, fontweight='bold')
            ax_top.axis('off')
            ax_top.text(0.05, 0.85, "• Scalar Index: c_7", color='#f87171', fontweight='bold', fontsize=12.5 if is_gif else 11.0)
            ax_top.text(0.08, 0.68, "  -> 1 index loses local spatial context", color='#94a3b8', fontsize=10.5 if is_gif else 9.5)
            ax_top.text(0.05, 0.45, "• Joint Tuple: ( 0, 3, 2, 1 )", color='#4ade80', fontweight='bold', fontsize=12.5 if is_gif else 11.0)
            ax_top.text(0.08, 0.28, "  -> Encodes simultaneous sub-region states", color='#f8fafc', fontsize=10.5 if is_gif else 9.5)
            ax_top.text(0.08, 0.10, "  -> Cross-entropy across spatial tiles", color='#38bdf8', fontsize=10.5 if is_gif else 9.5)

            # Bottom Right: Benefits of Tiling & JTF
            ax_bot.axis('off')
            ax_bot.text(0.05, 0.88, "TILING & JTF BENEFITS:", color='#0d9488', fontweight='bold', fontsize=fs_header)
            ax_bot.text(0.05, 0.68, "• Speed: Sub-tiles fit in L1/L2 cache; parallel OpenMP.", color='#f8fafc', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.48, "• Memory: Local cluster counts K_tile << K_global.", color='#f8fafc', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.28, "• Cross-Entropy: Tuples capture multi-tile correlation.", color='#38bdf8', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.08, "• Pass 2 JTF: Fixes seam flicker (verifies d <= rlim).", color='#4ade80', fontfamily='monospace', fontsize=fs_bullet)

        # -------------------------------------------------------------
        # SCENE 7: Architecture Summary & CLI Presets
        # -------------------------------------------------------------
        else:
            ax_main.set_xlim(0, 10); ax_main.set_ylim(0, 10)
            ax_main.set_title("GRIC Architectural Pipeline Summary", color='#38bdf8', fontsize=fs_main_title, fontweight='bold')
            
            boxes = [
                ("1. Priors", "-dprob, -tm, -pred", 8.3, "#2563eb"),
                ("2. Target Select", "-entropy, -entropy_gate", 6.5, "#9333ea"),
                ("3. SIMD Distance", "rlim, -auto_rlim, -ncpu", 4.7, "#ea580c"),
                ("4. Prune Geometry", "-te4, -te5, -sparse_dcc", 2.9, "#16a34a"),
                ("5. Assign / Anchor", "-maxcl, -discard_frac", 1.1, "#dc2626")
            ]
            for name, opts, y_pos, b_col in boxes:
                ax_main.text(5.0, y_pos, f"{name}:  {opts}", color='#ffffff', fontweight='bold', fontsize=12.5 if is_gif else 11.0, ha='center',
                             bbox=dict(boxstyle="round,pad=0.35", facecolor=b_col, alpha=0.85, edgecolor="#fff", lw=1.2))

            ax_top.set_title("Recommended Presets", color='#94a3b8', fontsize=fs_side_title, fontweight='bold')
            ax_top.axis('off')
            ax_top.text(0.05, 0.85, "1. Video Tracking:", color='#38bdf8', fontweight='bold', fontsize=12 if is_gif else 10.0)
            ax_top.text(0.05, 0.68, "./gric-cluster a1.5 -tm 0.8 -pred", color='#f8fafc', fontfamily='monospace', fontsize=11 if is_gif else 9.5)
            ax_top.text(0.05, 0.45, "2. High-Dim Manifolds:", color='#c084fc', fontweight='bold', fontsize=12 if is_gif else 10.0)
            ax_top.text(0.05, 0.28, "./gric-cluster 0.5 -entropy -te5 -gprob", color='#f8fafc', fontfamily='monospace', fontsize=11 if is_gif else 9.5)

            ax_bot.axis('off')
            ax_bot.text(0.05, 0.88, "GETTING STARTED:", color='#38bdf8', fontweight='bold', fontsize=fs_header)
            ax_bot.text(0.05, 0.68, "• Explore docs/algorithm/visual_guide.md", color='#f8fafc', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.48, "• Interactive simulator: docs/visual_simulator.html", color='#4ade80', fontsize=fs_bullet)
            ax_bot.text(0.05, 0.28, "• Check features: ./gric-info", color='#f8fafc', fontfamily='monospace', fontsize=fs_bullet)

    print(f"Rendering Scene {scene_idx+1}/{len(SCENES)} [{mode.upper()}]: {scene['title']} ({num_frames} frames)...")
    anim = animation.FuncAnimation(fig, draw_frame, frames=num_frames, interval=1000/FPS)
    
    writer = animation.FFMpegWriter(fps=FPS, bitrate=3000)
    anim.save(video_raw_path, writer=writer)
    plt.close(fig)
    
    if audio_wav:
        # Mux with neural audio
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_raw_path,
            "-i", audio_wav,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            scene_with_audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return scene_with_audio_path
    else:
        return video_raw_path

def main():
    audio_files, durations = generate_all_narrations()
    
    # 1. Render MP4 version with standard balanced text sizes
    print("=== Rendering MP4 video with standard typography ===")
    mp4_scene_videos = []
    for idx, scene in enumerate(SCENES):
        scene_mp4 = render_scene_video(idx, scene, durations[idx], audio_files[idx], mode='mp4')
        mp4_scene_videos.append(scene_mp4)

    concat_mp4_path = os.path.join(TEMP_DIR, "concat_mp4_list.txt")
    with open(concat_mp4_path, "w") as f:
        for sv in mp4_scene_videos:
            f.write(f"file '{sv}'\n")

    print("Concatenating MP4 scenes into final video...")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_mp4_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        FINAL_MP4
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Final Narrated Video created: {FINAL_MP4}")

    # 2. Render GIF version with enlarged text & symbols specifically for README
    print("=== Rendering GIF video stream with enlarged text & symbols ===")
    gif_scene_videos = []
    for idx, scene in enumerate(SCENES):
        scene_gif_video = render_scene_video(idx, scene, durations[idx], None, mode='gif')
        gif_scene_videos.append(scene_gif_video)

    concat_gif_path = os.path.join(TEMP_DIR, "concat_gif_list.txt")
    with open(concat_gif_path, "w") as f:
        for sv in gif_scene_videos:
            f.write(f"file '{sv}'\n")

    gif_raw_video = os.path.join(TEMP_DIR, "gif_raw_combined.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_gif_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        gif_raw_video
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Generating high-resolution animated GIF preview for README...")
    subprocess.run([
        "ffmpeg", "-y", "-i", gif_raw_video,
        "-vf", "fps=10,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
        FINAL_GIF
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Final GIF created: {FINAL_GIF}")

if __name__ == "__main__":
    main()
