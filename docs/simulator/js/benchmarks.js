/**
 * GRIC Simulator - benchmarks.js
 * 2D & 3D Synthetic Dataset Generators & Seedable PRNG
 */

//  2. 2D & 3D BENCHMARK DATASET GENERATORS
    // =========================================================================

    const BENCHMARK_DESCS = {
      // 2D Benchmarks
      "2Dspiral": "<b>2Dspiral</b>: 1,000 continuous 2D samples tracing an Archimedean spiral. Demonstrates ~1.0 eval/frame via sequential recency prior.",
      "2Dspiral-shuffle": "<b>2Dspiral-shuffle</b>: 1,000 samples on spiral with randomized temporal order. Stresses multi-point metric pruning on non-convex manifolds.",
      "2Dcircle-shuffle": "<b>2Dcircle-shuffle</b>: 1,000 samples on a 1D circle manifold in 2D with shuffled order. Tests pure distance geometry without temporal correlation.",
      "2DcircleP10n": "<b>2DcircleP10n</b>: 10 periodic circular cycles with additive Gaussian noise (sigma=0.04). Tests cyclic Markov transition matrix (-tm) and sequence predictor (-pred).",
      "2Drand": "<b>2Drand</b>: 1,000 uniform random points across the 2D plane. Stresses spatial coverage scaling.",
      "2Dwalk": "<b>2Dwalk</b>: 1,000 steps of bounded Brownian random walk in 2D. Tests localized drift and dynamic cluster reuse.",
      "stream": "<b>Dynamic 2D Stream</b>: Continuous Lissajous orbital trajectory with real-time parameter streaming.",
      
      // 3D Benchmarks
      "3Dspiral": "<b>3Dspiral</b>: 1,000 continuous 3D samples on a rotating helical manifold. Shows quad-split X/Y/Z projections &amp; custom 3D drag rotation.",
      "3Dspiral-shuffle": "<b>3Dspiral-shuffle</b>: 1,000 3D helical spiral points with shuffled temporal order. Stresses 3D metric pruning (3P/4P/5P) without temporal prior.",
      "3Dsphere": "<b>3Dsphere</b>: 1,000 points uniformly distributed on the surface of a 3D sphere manifold (S²). Tests non-Euclidean intrinsic 2D manifold embedded in 3D.",
      "3Dtorus": "<b>3Dtorus</b>: 1,000 points tracing a continuous knot trajectory around a 3D torus manifold (R=0.60, r=0.24).",
      "3Dstar": "<b>3Dstar</b>: 1,000 points distributed across 20 3D radial star branches extending from origin.",
      "3Drand": "<b>3Drand</b>: 1,000 uniform random points distributed throughout a 3D spherical volume.",
      "3Dwalk": "<b>3Dwalk</b>: 1,000 steps of bounded 3D Brownian random walk. Tests localized 3D spatial drift and anchor reuse.",
      "3Dlorenz": "<b>Dynamic 3D Lorenz</b>: Continuous dynamic integration of the chaotic Lorenz attractor (σ=10, ρ=28, β=8/3) with real-time 3D orbit streaming.",
      
      // Custom
      "custom": "<b>Custom Dataset</b>: User-uploaded 2D or 3D coordinate dataset."
    };

    function is3DBenchmark(type) {
      if (type.startsWith("3D") || type === "3Dlorenz") return true;
      return false;
    }

    function generateBenchmark(type, N = 1000) {
      const points = [];
      
      // --- 2D BENCHMARKS ---
      if (type === "2Dspiral") {
        const loops = 2.0;
        for (let i = 0; i < N; i++) {
          const t = i / N;
          const r = t * 0.92;
          const theta = 2.0 * Math.PI * loops * t;
          points.push({ x: r * Math.cos(theta), y: r * Math.sin(theta), z: 0.0 });
        }
      } else if (type === "2Dspiral-shuffle") {
        const pts = generateBenchmark("2Dspiral", N);
        for (let i = pts.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [pts[i], pts[j]] = [pts[j], pts[i]];
        }
        return pts;
      } else if (type === "2Dcircle-shuffle") {
        for (let i = 0; i < N; i++) {
          const theta = 2.0 * Math.PI * (i / N);
          points.push({ x: 0.85 * Math.cos(theta), y: 0.85 * Math.sin(theta), z: 0.0 });
        }
        for (let i = points.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [points[i], points[j]] = [points[j], points[i]];
        }
      } else if (type === "2DcircleP10n") {
        const periods = 10.0;
        for (let i = 0; i < N; i++) {
          const theta = 2.0 * Math.PI * periods * (i / N);
          const u1 = Math.max(1e-12, Math.random());
          const u2 = Math.random();
          const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * 0.04;
          const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2) * 0.04;
          points.push({
            x: 0.80 * Math.cos(theta) + z0,
            y: 0.80 * Math.sin(theta) + z1,
            z: 0.0
          });
        }
      } else if (type === "2Drand") {
        for (let i = 0; i < N; i++) {
          const r = Math.sqrt(Math.random()) * 0.92;
          const theta = 2.0 * Math.PI * Math.random();
          points.push({ x: r * Math.cos(theta), y: r * Math.sin(theta), z: 0.0 });
        }
      } else if (type === "2Dwalk") {
        let cx = 0, cy = 0;
        const step = 0.04;
        for (let i = 0; i < N; i++) {
          const theta = 2.0 * Math.PI * Math.random();
          cx = Math.max(-0.9, Math.min(0.9, cx + step * Math.cos(theta)));
          cy = Math.max(-0.9, Math.min(0.9, cy + step * Math.sin(theta)));
          points.push({ x: cx, y: cy, z: 0.0 });
        }
      } 
      
      // --- 3D BENCHMARKS ---
      else if (type === "3Dspiral") {
        const loops = 4.0;
        const cos45 = 0.70710678, sin45 = 0.70710678;
        const cos30 = 0.86602540, sin30 = 0.50000000;
        
        for (let i = 0; i < N; i++) {
          const t = (i + 1) / N;
          const raw_x = 0.68 * t * Math.cos(2.0 * Math.PI * loops * t);
          const raw_y = 0.68 * t * Math.sin(2.0 * Math.PI * loops * t);
          const raw_z = 1.68 * t - 0.84;
          
          // Rotation around Y (45 deg)
          const x1 = raw_x * cos45 + raw_z * sin45;
          const z1 = -raw_x * sin45 + raw_z * cos45;
          const y1 = raw_y;
          
          // Rotation around X (30 deg)
          const fx = x1;
          const fy = y1 * cos30 - z1 * sin30;
          const fz = y1 * sin30 + z1 * cos30;
          
          points.push({ x: fx, y: fy, z: fz });
        }
      } else if (type === "3Dspiral-shuffle") {
        const pts = generateBenchmark("3Dspiral", N);
        for (let i = pts.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [pts[i], pts[j]] = [pts[j], pts[i]];
        }
        return pts;
      } else if (type === "3Dsphere") {
        // Fibonacci spherical distribution on S2
        const R = 0.84;
        for (let i = 0; i < N; i++) {
          const y = 1.0 - (i / (N - 1.0 || 1)) * 2.0;
          const radius = Math.sqrt(Math.max(0, 1.0 - y * y)) * R;
          const theta = Math.PI * (1.0 + Math.sqrt(5.0)) * i;
          const x = Math.cos(theta) * radius;
          const z = Math.sin(theta) * radius;
          points.push({ x, y: y * R, z });
        }
      } else if (type === "3Dtorus") {
        const R = 0.58;
        const r = 0.24;
        for (let i = 0; i < N; i++) {
          const u = 2.0 * Math.PI * 3.0 * (i / N);
          const v = 2.0 * Math.PI * 8.0 * (i / N);
          const x = (R + r * Math.cos(v)) * Math.cos(u);
          const y = (R + r * Math.cos(v)) * Math.sin(u);
          const z = r * Math.sin(v);
          points.push({ x, y, z });
        }
      } else if (type === "3Dstar") {
        const numSpokes = 20;
        const ptsPerSpoke = Math.floor(N / numSpokes);
        const spokeDirs = [];
        for (let s = 0; s < numSpokes; s++) {
          const y = 1.0 - (s / (numSpokes - 1.0 || 1)) * 2.0;
          const r = Math.sqrt(Math.max(0, 1.0 - y * y));
          const theta = Math.PI * (1.0 + Math.sqrt(5.0)) * s;
          spokeDirs.push({ x: r * Math.cos(theta), y, z: r * Math.sin(theta) });
        }
        for (let i = 0; i < N; i++) {
          const s = i % numSpokes;
          const inSpokeIdx = Math.floor(i / numSpokes);
          const rad = 0.15 + 0.72 * (inSpokeIdx / ptsPerSpoke);
          const dir = spokeDirs[s];
          points.push({ x: dir.x * rad, y: dir.y * rad, z: dir.z * rad });
        }
      } else if (type === "3Drand") {
        for (let i = 0; i < N; i++) {
          const rad = 0.88 * Math.cbrt(Math.random());
          const costheta = 1.0 - 2.0 * Math.random();
          const sintheta = Math.sqrt(Math.max(0, 1.0 - costheta * costheta));
          const phi = 2.0 * Math.PI * Math.random();
          points.push({
            x: rad * sintheta * Math.cos(phi),
            y: rad * sintheta * Math.sin(phi),
            z: rad * costheta
          });
        }
      } else if (type === "3Dwalk") {
        let cx = 0, cy = 0, cz = 0;
        const step = 0.04;
        for (let i = 0; i < N; i++) {
          const costheta = 1.0 - 2.0 * Math.random();
          const sintheta = Math.sqrt(Math.max(0, 1.0 - costheta * costheta));
          const phi = 2.0 * Math.PI * Math.random();
          cx = Math.max(-0.85, Math.min(0.85, cx + step * sintheta * Math.cos(phi)));
          cy = Math.max(-0.85, Math.min(0.85, cy + step * sintheta * Math.sin(phi)));
          cz = Math.max(-0.85, Math.min(0.85, cz + step * costheta));
          points.push({ x: cx, y: cy, z: cz });
        }
      }

      return points;
    }

    // =========================================================================


