#!/usr/bin/env python3
"""
GRIC Animated Video Explainer Generator
Generates an animated MP4 video and animated GIF illustrating:
1. Spatial Sequential Clustering & Anchor Spheres (rlim)
2. Triangle Inequality Geometric Pruning in 2D
3. Shannon Entropy Target Scheduling vs Greedy
4. Spatial Multi-Tile Partitioning & JTF Boundary Smoothing
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Ellipse, Polygon
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MP4_OUTPUT = os.path.join(OUTPUT_DIR, "gric_explainer.mp4")
GIF_OUTPUT = os.path.join(OUTPUT_DIR, "gric_explainer.gif")

# Set styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = '#0f172a' # Dark slate modern background

# Video properties
FPS = 25
TOTAL_FRAMES = 450 # 18 seconds total (4 distinct 4.5s phases)

def make_animation():
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor('#0f172a')

    # Create grid layout
    gs = GridSpec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 1], figure=fig,
                  left=0.06, right=0.94, bottom=0.08, top=0.90, wspace=0.25, hspace=0.32)

    ax_main = fig.add_subplot(gs[:, 0])      # Left large spatial canvas
    ax_prob = fig.add_subplot(gs[0, 1])      # Top-right: Probability / Entropy / Pruning stats
    ax_status = fig.add_subplot(gs[1, 1])    # Bottom-right: JTF / Option badges & state

    # Super title
    stitle = fig.suptitle("GRIC: Geometric Real-Time Image Clustering", 
                          fontsize=20, fontweight='bold', color='#f8fafc', y=0.96)

    # -------------------------------------------------------------
    # Synthetic Data Setup for 4 Phases
    # -------------------------------------------------------------
    np.random.seed(42)
    
    # Phase 1 & 2 Anchors
    anchors = np.array([
        [2.5, 3.0],   # c0
        [7.0, 7.5],   # c1
        [3.0, 7.0],   # c2
        [7.5, 2.5],   # c3
        [5.0, 5.0]    # c4 (central pivot)
    ])
    rlim = 1.3

    # Trajectory points for animation
    t_vals = np.linspace(0, 4 * np.pi, 200)
    traj1_x = 2.5 + 0.7 * np.cos(t_vals[:50])
    traj1_y = 3.0 + 0.7 * np.sin(t_vals[:50])

    def update(frame_idx):
        ax_main.clear()
        ax_prob.clear()
        ax_status.clear()

        # Style all axes
        for ax in [ax_main, ax_prob, ax_status]:
            ax.set_facecolor('#1e293b')
            for spine in ax.spines.values():
                spine.set_color('#334155')
            ax.tick_params(colors='#94a3b8', labelsize=9)

        # ---------------------------------------------------------
        # PHASE 1 (Frames 0 - 110): Stream Ingestion, Anchors & rlim
        # ---------------------------------------------------------
        if frame_idx < 110:
            p1_progress = frame_idx / 110.0
            ax_main.set_xlim(0, 10)
            ax_main.set_ylim(0, 10)
            ax_main.set_title("Phase 1: Sequential Ingestion & Anchor Creation (rlim)", 
                              fontsize=13, fontweight='bold', color='#38bdf8', pad=10)

            # Draw anchors formed so far
            n_anc = 1 if p1_progress < 0.35 else (2 if p1_progress < 0.7 else 3)
            colors = ['#38bdf8', '#a855f7', '#10b981']

            for i in range(n_anc):
                ax_main.scatter(anchors[i, 0], anchors[i, 1], s=160, color=colors[i], edgecolors='#ffffff', linewidth=2, zorder=5)
                # rlim sphere
                circle = Circle((anchors[i, 0], anchors[i, 1]), rlim, fill=True, facecolor=colors[i], alpha=0.2, edgecolor=colors[i], linewidth=2, linestyle='--')
                ax_main.add_patch(circle)
                ax_main.text(anchors[i, 0] + 0.2, anchors[i, 1] + 0.3, f"Anchor c_{i}", color=colors[i], fontweight='bold', fontsize=11)

            # Incoming frame position
            curr_pt_idx = int(p1_progress * 150) % 50
            if n_anc == 1:
                cur_x = traj1_x[curr_pt_idx]
                cur_y = traj1_y[curr_pt_idx]
                ax_main.scatter(cur_x, cur_y, s=120, color='#facc15', edgecolors='#ffffff', linewidth=2, zorder=6)
                ax_main.text(cur_x + 0.2, cur_y - 0.3, "f_i (dist < rlim: Assign to c_0)", color='#facc15', fontweight='bold', fontsize=10)
                # Distance line
                ax_main.plot([anchors[0,0], cur_x], [anchors[0,1], cur_y], color='#38bdf8', linewidth=2)
            else:
                cur_x = 7.0 + 0.6 * np.cos(frame_idx * 0.1)
                cur_y = 7.5 + 0.6 * np.sin(frame_idx * 0.1)
                ax_main.scatter(cur_x, cur_y, s=120, color='#facc15', edgecolors='#ffffff', linewidth=2, zorder=6)
                ax_main.text(cur_x + 0.2, cur_y - 0.3, "f_i (dist < rlim: Assign to c_1)", color='#facc15', fontweight='bold', fontsize=10)
                ax_main.plot([anchors[1,0], cur_x], [anchors[1,1], cur_y], color='#a855f7', linewidth=2)

            # Right Top: Cluster Stats
            ax_prob.set_title("Active Clusters & Frequency Priors", fontsize=11, fontweight='bold', color='#94a3b8')
            clust_names = [f"c_{i}" for i in range(n_anc)]
            counts = [int(30 + p1_progress * 50), int(15 + p1_progress * 30), 5][:n_anc]
            bars = ax_prob.bar(clust_names, counts, color=colors[:n_anc], width=0.5)
            ax_prob.set_ylabel("Member Count", color='#94a3b8', fontsize=9)
            ax_prob.set_ylim(0, 100)

            # Right Bottom: Pipeline Explainer Card
            ax_status.axis('off')
            ax_status.text(0.05, 0.85, "HOW CLUSTERING WORKS:", color='#38bdf8', fontweight='bold', fontsize=12)
            ax_status.text(0.05, 0.68, "• Anchor-based: Cluster center = first frame created.", color='#f8fafc', fontsize=10)
            ax_status.text(0.05, 0.50, "• No centroid recomputation: 0 extra overhead.", color='#f8fafc', fontsize=10)
            ax_status.text(0.05, 0.32, f"• Hard threshold: rlim = {rlim:.2f} (or -auto_rlim).", color='#f8fafc', fontsize=10)
            ax_status.text(0.05, 0.14, "• If distance <= rlim: Assign. Else: Create new anchor.", color='#4ade80', fontweight='bold', fontsize=10)

        # ---------------------------------------------------------
        # PHASE 2 (Frames 110 - 220): Geometric Pruning (3P, 4P, 5P)
        # ---------------------------------------------------------
        elif frame_idx < 220:
            p2_progress = (frame_idx - 110) / 110.0
            ax_main.set_xlim(0, 10)
            ax_main.set_ylim(0, 10)
            ax_main.set_title("Phase 2: Triangle Inequality Geometric Pruning (-te4, -te5)", 
                              fontsize=13, fontweight='bold', color='#10b981', pad=10)

            # 4 Anchors
            for i in range(4):
                ax_main.scatter(anchors[i, 0], anchors[i, 1], s=140, color='#64748b', edgecolors='#ffffff', linewidth=1.5, zorder=5)
                ax_main.text(anchors[i, 0] + 0.2, anchors[i, 1] + 0.3, f"c_{i}", color='#cbd5e1', fontweight='bold', fontsize=11)

            # Frame arrives near c_0
            fi_x, fi_y = 2.2, 2.7
            ax_main.scatter(fi_x, fi_y, s=140, color='#facc15', edgecolors='#ffffff', linewidth=2, zorder=7)
            ax_main.text(fi_x - 0.8, fi_y + 0.3, "f_i (Query Frame)", color='#facc15', fontweight='bold', fontsize=11)

            # Test c_2 first (Mismatch!)
            ax_main.plot([fi_x, anchors[2,0]], [fi_y, anchors[2,1]], color='#ef4444', linewidth=2.5, linestyle='-', zorder=4)
            d_meas = np.hypot(fi_x - anchors[2,0], fi_y - anchors[2,1])
            ax_main.text(2.8, 5.0, f"d(f_i, c_2) = {d_meas:.2f} > rlim (Mismatch)", color='#ef4444', fontweight='bold', fontsize=10)

            # Pruning exclusion arcs radiating from c_2
            r_prune = d_meas - rlim
            if p2_progress > 0.3:
                # Exclusion annulus for c_1
                prune_circle = Circle((anchors[2,0], anchors[2,1]), r_prune, fill=False, edgecolor='#ef4444', linewidth=2, linestyle=':')
                ax_main.add_patch(prune_circle)

                # Cross out pruned cluster c_1 and c_3
                ax_main.scatter(anchors[1,0], anchors[1,1], s=200, color='#dc2626', marker='x', linewidth=3, zorder=8)
                ax_main.text(anchors[1,0] - 1.2, anchors[1,1] - 0.5, "PRUNED! (3P bound > rlim)", color='#f87171', fontweight='bold', fontsize=10)

                if p2_progress > 0.6:
                    ax_main.scatter(anchors[3,0], anchors[3,1], s=200, color='#dc2626', marker='x', linewidth=3, zorder=8)
                    ax_main.text(anchors[3,0] - 1.2, anchors[3,1] - 0.5, "PRUNED! (4P bound > rlim)", color='#f87171', fontweight='bold', fontsize=10)

            # Match on c_0
            if p2_progress > 0.8:
                ax_main.plot([fi_x, anchors[0,0]], [fi_y, anchors[0,1]], color='#22c55e', linewidth=3, zorder=6)
                ax_main.text(anchors[0,0] + 0.2, anchors[0,1] - 0.5, "MATCH FOUND (d = 0.42 < rlim)", color='#4ade80', fontweight='bold', fontsize=11)

            # Right Top: Distance Computations Saved
            ax_prob.set_title("Metric Evals: Naive vs GRIC Pruning", fontsize=11, fontweight='bold', color='#94a3b8')
            bars = ax_prob.bar(["Naive Scan", "GRIC 3P", "GRIC 5P (-te5)"], [4, 2, 1], color=['#ef4444', '#38bdf8', '#10b981'], width=0.55)
            ax_prob.set_ylabel("Distance Calls", color='#94a3b8', fontsize=9)
            ax_prob.set_ylim(0, 5)
            for bar in bars:
                yval = bar.get_height()
                ax_prob.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{int(yval)} calls", ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=10)

            # Right Bottom: Pruning Logic Box
            ax_status.axis('off')
            ax_status.text(0.05, 0.85, "TRIANGLE INEQUALITY PRUNING:", color='#10b981', fontweight='bold', fontsize=12)
            ax_status.text(0.05, 0.68, "• Rule: | d(f_i, c_j) - dcc(c_j, c_l) | > rlim", color='#f8fafc', fontfamily='monospace', fontsize=10)
            ax_status.text(0.05, 0.50, "• 1 measurement eliminates entire clusters.", color='#f8fafc', fontsize=10)
            ax_status.text(0.05, 0.32, "• -te4: 2 measured anchors form 2D bounds.", color='#c084fc', fontsize=10)
            ax_status.text(0.05, 0.14, "• -te5: 3 anchors form 3D simplex bounds.", color='#4ade80', fontweight='bold', fontsize=10)

        # ---------------------------------------------------------
        # PHASE 3 (Frames 220 - 330): Entropy Target Scheduling
        # ---------------------------------------------------------
        elif frame_idx < 330:
            p3_progress = (frame_idx - 220) / 110.0
            ax_main.set_xlim(0, 10)
            ax_main.set_ylim(0, 10)
            ax_main.set_title("Phase 3: Shannon Entropy Target Scheduling (-entropy)", 
                              fontsize=13, fontweight='bold', color='#c084fc', pad=10)

            # Draw all 5 anchors (central pivot c4 in middle)
            for i in range(5):
                col = '#a855f7' if i == 4 else '#64748b'
                ax_main.scatter(anchors[i, 0], anchors[i, 1], s=150, color=col, edgecolors='#ffffff', linewidth=1.5, zorder=5)
                ax_main.text(anchors[i, 0] + 0.2, anchors[i, 1] + 0.3, f"c_{i}", color='#f8fafc', fontweight='bold', fontsize=11)

            # Pivot target measurement
            ax_main.scatter(anchors[4, 0], anchors[4, 1], s=250, facecolors='none', edgecolors='#c084fc', linewidth=2.5, linestyle='--', zorder=6)
            ax_main.text(anchors[4, 0] - 1.8, anchors[4, 1] - 0.5, "Optimal Entropy Pivot (c_4)", color='#c084fc', fontweight='bold', fontsize=10)

            # Query frame near c_3
            fi_x, fi_y = 7.3, 2.7
            ax_main.scatter(fi_x, fi_y, s=140, color='#facc15', edgecolors='#ffffff', linewidth=2, zorder=7)

            # Line to pivot c_4
            ax_main.plot([fi_x, anchors[4,0]], [fi_y, anchors[4,1]], color='#c084fc', linewidth=2, linestyle=':')
            
            # Massive simultaneous pruning wave
            if p3_progress > 0.4:
                for prune_id in [0, 2]:
                    ax_main.scatter(anchors[prune_id, 0], anchors[prune_id, 1], s=200, color='#dc2626', marker='x', linewidth=3, zorder=8)
                ax_main.text(1.5, 4.5, "c_0 & c_2 Pruned Simultaneously!", color='#f87171', fontweight='bold', fontsize=10)

            if p3_progress > 0.7:
                # Match on c_3
                ax_main.plot([fi_x, anchors[3,0]], [fi_y, anchors[3,1]], color='#22c55e', linewidth=3, zorder=6)
                ax_main.text(anchors[3,0] - 0.5, anchors[3,1] - 0.6, "MATCH c_3 (Step 2)", color='#4ade80', fontweight='bold', fontsize=11)

            # Right Top: Entropy Drop Graph
            ax_prob.set_title("Posterior Shannon Entropy H(X)", fontsize=11, fontweight='bold', color='#94a3b8')
            steps = ["Prior H0", "After c_4", "After c_3"]
            entropy_vals = [2.32, 0.95 if p3_progress > 0.4 else 2.32, 0.0 if p3_progress > 0.7 else (0.95 if p3_progress > 0.4 else 2.32)]
            bars = ax_prob.bar(steps, entropy_vals, color=['#38bdf8', '#c084fc', '#22c55e'], width=0.5)
            ax_prob.set_ylabel("Shannon Entropy (bits)", color='#94a3b8', fontsize=9)
            ax_prob.set_ylim(0, 3)

            # Right Bottom: Entropy Concept
            ax_status.axis('off')
            ax_status.text(0.05, 0.85, "INFORMATION GAIN SCHEDULER:", color='#c084fc', fontweight='bold', fontsize=12)
            ax_status.text(0.05, 0.68, "• Greedy: Tests highest P(c) first (often poor split).", color='#f8fafc', fontsize=10)
            ax_status.text(0.05, 0.50, "• Entropy: Tests pivot anchor with min expected H(X).", color='#c084fc', fontweight='bold', fontsize=10)
            ax_status.text(0.05, 0.32, "• Maximizes candidate exclusion on mismatch.", color='#f8fafc', fontsize=10)
            ax_status.text(0.05, 0.14, "• Options: -entropy, -entropy_fast, -entropy_gate", color='#38bdf8', fontfamily='monospace', fontsize=10)

        # ---------------------------------------------------------
        # PHASE 4 (Frames 330 - 450): Multi-Tile & JTF Smoothing
        # ---------------------------------------------------------
        else:
            p4_progress = (frame_idx - 330) / 120.0
            ax_main.set_xlim(0, 10)
            ax_main.set_ylim(0, 10)
            ax_main.set_title("Phase 4: Spatial Tiling (-tiles 2x2) & JTF Fusion (-jtf)", 
                              fontsize=13, fontweight='bold', color='#38bdf8', pad=10)

            # Draw 2x2 grid
            ax_main.plot([5, 5], [0, 10], color='#38bdf8', linewidth=2.5, linestyle='--')
            ax_main.plot([0, 10], [5, 5], color='#38bdf8', linewidth=2.5, linestyle='--')

            # Tile labels
            ax_main.text(2.5, 7.5, "Tile 0\n[Thread 0]", color='#94a3b8', ha='center', va='center', fontsize=12, fontweight='bold')
            ax_main.text(7.5, 7.5, "Tile 1\n[Thread 1]", color='#94a3b8', ha='center', va='center', fontsize=12, fontweight='bold')
            ax_main.text(2.5, 2.5, "Tile 2\n[Thread 2]", color='#94a3b8', ha='center', va='center', fontsize=12, fontweight='bold')
            ax_main.text(7.5, 2.5, "Tile 3\n[Thread 3]", color='#94a3b8', ha='center', va='center', fontsize=12, fontweight='bold')

            # Moving object straddling boundary seam
            obj_x = 5.0 + 0.3 * np.sin(p4_progress * 4 * np.pi)
            obj_y = 5.0 + 0.3 * np.cos(p4_progress * 4 * np.pi)
            ax_main.scatter(obj_x, obj_y, s=300, color='#f43f5e', edgecolors='#ffffff', linewidth=2, zorder=6)
            ax_main.text(obj_x + 0.4, obj_y + 0.4, "Boundary Feature", color='#fb7185', fontweight='bold', fontsize=11)

            # Right Top: Multi-Tile State Comparison
            ax_prob.set_title("Pass 1 (Raw) vs Pass 2 (JTF Corrected)", fontsize=11, fontweight='bold', color='#94a3b8')
            ax_prob.axis('off')
            
            raw_tuple = "(0, 3, 5, 1)  [Flicker!]"
            corr_tuple = "(0, 3, 2, 1)  [Smooth]"
            
            ax_prob.text(0.1, 0.70, f"Pass 1 Raw Tuple:\n{raw_tuple}", color='#f87171', fontweight='bold', fontsize=12)
            if p4_progress > 0.5:
                ax_prob.text(0.1, 0.25, f"Pass 2 JTF Fusion:\n{corr_tuple}", color='#4ade80', fontweight='bold', fontsize=12)

            # Right Bottom: JTF Logic
            ax_status.axis('off')
            ax_status.text(0.05, 0.85, "JOINT TRAJECTORY FUSION (JTF):", color='#38bdf8', fontweight='bold', fontsize=12)
            ax_status.text(0.05, 0.68, "• Pass 1: Independent Spatial Clustering per tile.", color='#f8fafc', fontsize=10)
            ax_status.text(0.05, 0.50, "• Pass 2: Scans lookback history for tuple matches.", color='#f8fafc', fontsize=10)
            ax_status.text(0.05, 0.32, "• Overrides spurious boundary noise (Tile 2: 5 -> 2).", color='#4ade80', fontweight='bold', fontsize=10)
            ax_status.text(0.05, 0.14, "• Invariant: Override strictly requires d <= rlim.", color='#f8fafc', fontsize=10)

    print("Rendering animated video frames...")
    anim = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000/FPS)

    # Save MP4
    writer_mp4 = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='GRIC Team'), bitrate=2500)
    anim.save(MP4_OUTPUT, writer=writer_mp4)
    print(f"Saved Video: {MP4_OUTPUT}")

    # Generate a lightweight GIF version for markdown preview (downsample frames to 60 frames)
    print("Generating animated GIF preview...")
    fig_gif = plt.figure(figsize=(10, 5.6), dpi=80)
    fig_gif.patch.set_facecolor('#0f172a')
    
    # We can use ffmpeg to directly convert the MP4 to a high-quality GIF
    os.system(f"ffmpeg -y -i {MP4_OUTPUT} -vf 'fps=12,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' {GIF_OUTPUT}")
    print(f"Saved GIF: {GIF_OUTPUT}")

if __name__ == "__main__":
    make_animation()
