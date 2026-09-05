/**
 * GRIC Simulator - benchmarks.js
 * 2D & 3D Synthetic Dataset Generators & Seedable PRNG
 */

//  2. 2D & 3D BENCHMARK DATASET GENERATORS
    // =========================================================================

    const BENCHMARK_DESCS = {
      // 2D Benchmarks
      "2Dspiral": "<b>2Dspiral</b>: Continuous 2D samples tracing an Archimedean spiral. Demonstrates ~1.0 eval/frame via sequential recency prior.",
      "2Dspiral-shuffle": "<b>2Dspiral-shuffle</b>: Samples on spiral with randomized temporal order. Stresses multi-point metric pruning on non-convex manifolds.",
      "2Dcircle-shuffle": "<b>2Dcircle-shuffle</b>: Samples on a 1D circle manifold in 2D with shuffled order. Tests pure distance geometry without temporal correlation.",
      "2DcircleP10n": "<b>2DcircleP10n</b>: Periodic circular cycles with additive Gaussian noise (sigma=0.04). Tests cyclic Markov transition matrix (-tm) and sequence predictor (-pred).",
      "2Drand": "<b>2Drand</b>: Uniform random points across the 2D plane. Stresses spatial coverage scaling.",
      "2Dwalk": "<b>2Dwalk</b>: Steps of bounded Brownian random walk in 2D. Tests localized drift and dynamic cluster reuse.",
      "stream": "<b>stream</b>: Samples on a 2D Lissajous orbital trajectory. Tests smooth continuous motion clustering.",
      
      // 3D Benchmarks
      "3Dspiral": "<b>3Dspiral</b>: Continuous 3D samples on a rotating helical manifold. Shows quad-split X/Y/Z projections &amp; custom 3D drag rotation.",
      "3Dspiral-shuffle": "<b>3Dspiral-shuffle</b>: 3D helical spiral points with shuffled temporal order. Stresses 3D metric pruning (3P/4P/5P) without temporal prior.",
      "3Dsphere": "<b>3Dsphere</b>: Points uniformly distributed on the surface of a 3D sphere manifold (S²). Tests non-Euclidean intrinsic 2D manifold embedded in 3D.",
      "3Dtorus": "<b>3Dtorus</b>: Continuous knot trajectory around a 3D torus manifold (R=0.60, r=0.24).",
      "3Dstar": "<b>3Dstar</b>: Points distributed across 20 3D radial star branches extending from origin.",
      "3Drand": "<b>3Drand</b>: Uniform random points distributed throughout a 3D spherical volume.",
      "3Dwalk": "<b>3Dwalk</b>: Steps of bounded 3D Brownian random walk. Tests localized 3D spatial drift and anchor reuse.",
      "3Dlorenz": "<b>3Dlorenz</b>: Samples integrated along the chaotic Lorenz attractor (σ=10, ρ=28, β=8/3). Tests 3D chaotic trajectory clustering.",
      
      // High-Dimensional Benchmarks (50% 3D Variance)
      "32Dtorus": "<b>32Dtorus</b>: 32D torus knot " +
        "(dims 0..2: 50% variance, dims 3..31: 50% variance).",
      "32Dspiral": "<b>32Dspiral</b>: 32D spiral " +
        "(dims 0..2: 50% variance, dims 3..31: 50% variance).",
      "32Drand": "<b>32Drand</b>: 32D random " +
        "(dims 0..2: 50% variance, dims 3..31: 50% variance).",
      "128Dtorus": "<b>128Dtorus</b>: 128D torus knot " +
        "(dims 0..2: 50% variance, dims 3..127: 50% variance).",
      "128Dspiral": "<b>128Dspiral</b>: 128D spiral " +
        "(dims 0..2: 50% variance, dims 3..127: 50% variance).",
      "128Drand": "<b>128Drand</b>: 128D random " +
        "(dims 0..2: 50% variance, dims 3..127: 50% variance).",
      "512Dtorus": "<b>512Dtorus</b>: 512D torus knot " +
        "(dims 0..2: 50% variance, dims 3..511: 50% variance).",
      "512Dspiral": "<b>512Dspiral</b>: 512D spiral " +
        "(dims 0..2: 50% variance, dims 3..511: 50% variance).",
      "512Drand": "<b>512Drand</b>: 512D random " +
        "(dims 0..2: 50% variance, dims 3..511: 50% variance).",

      // Reconstructed
      "reconstructed": "<b>Reconstructed Dataset</b>: Non-parametric k-NN reconstruction evaluated from queries C mapped through training set (A → B).",

      // Custom
      "custom": "<b>Custom Dataset</b>: User-uploaded 2D or 3D coordinate dataset."
    };

    function is3DBenchmark(type) {
      if (!type) return false;
      if (type.startsWith("3D") || type === "3Dlorenz") return true;
      if (type.startsWith("32D") || type.startsWith("128D") || type.startsWith("512D")) return true;
      if (type === "reconstructed") {
        const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;
        if (slotD && (slotD.currentDim >= 3 ||
            (slotD.reconstructionInfo && slotD.reconstructionInfo.outputDim >= 3))) {
          return true;
        }
      }
      return false;
    }

    function getBenchmarkDim(type) {
      if (!type) return 2;
      if (typeof isImageBenchmark === 'function' && isImageBenchmark(type)) return 1024;
      if (type.startsWith("32D")) return 32;
      if (type.startsWith("128D")) return 128;
      if (type.startsWith("512D")) return 512;
      if (is3DBenchmark(type)) return 3;
      return 2;
    }

    function generateBenchmark(type, N = 1000) {
      const points = [];

      // If reconstructed slot has data, return it directly
      if (type === "reconstructed") {
        const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;
        if (slotD && slotD.benchmarkDataset && slotD.benchmarkDataset.length > 0) {
          return slotD.benchmarkDataset.map(p => ({ x: p.x, y: p.y, z: p.z || 0.0 }));
        }
        // Fallback default if not yet reconstructed
        return generateBenchmark("3Dtorus", N);
      }
      
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
      } else if (type === "stream") {
        // 2D Lissajous orbital trajectory
        for (let i = 0; i < N; i++) {
          const t = i * 0.15;
          const x = 0.70 * Math.cos(t)
            + 0.18 * Math.sin(t * 3.0);
          const y = 0.60 * Math.sin(t * 0.8)
            + 0.15 * Math.cos(t * 2.0);
          points.push({ x, y, z: 0.0 });
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
      } else if (type === "3Dlorenz") {
        // Lorenz attractor via Euler integration
        const dt = 0.009;
        const sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0;
        let lx = 0.1, ly = 0.0, lz = 0.0;
        for (let i = 0; i < N; i++) {
          const dx = sigma * (ly - lx);
          const dy = lx * (rho - lz) - ly;
          const dz = lx * ly - beta * lz;
          lx += dx * dt;
          ly += dy * dt;
          lz += dz * dt;
          points.push({
            x: (lx / 20.0) * 0.82,
            y: (ly / 28.0) * 0.82,
            z: ((lz - 25.0) / 25.0) * 0.82
          });
        }
      } else if (type.startsWith("32D") || type.startsWith("128D") || type.startsWith("512D")) {
        let D = 32;
        let baseType = "3Dtorus";
        if (type.startsWith("32D")) {
          D = 32;
          baseType = "3D" + type.slice(3);
        } else if (type.startsWith("128D")) {
          D = 128;
          baseType = "3D" + type.slice(4);
        } else if (type.startsWith("512D")) {
          D = 512;
          baseType = "3D" + type.slice(4);
        }

        // Generate base 3D coordinates
        const basePts = generateBenchmark(baseType, N);
        const remD = D - 3;

        let sumX = 0, sumY = 0, sumZ = 0;
        for (let i = 0; i < N; i++) {
          sumX += basePts[i].x;
          sumY += basePts[i].y;
          sumZ += basePts[i].z;
        }
        const meanX = sumX / N;
        const meanY = sumY / N;
        const meanZ = sumZ / N;

        let var3D = 0;
        for (let i = 0; i < N; i++) {
          const dx = basePts[i].x - meanX;
          const dy = basePts[i].y - meanY;
          const dz = basePts[i].z - meanZ;
          var3D += dx * dx + dy * dy + dz * dz;
        }
        var3D /= N;

        // Generate independent Gaussian components for remaining D - 3 dimensions
        const remValues = new Float64Array(N * remD);
        const remSums = new Float64Array(remD);
        for (let i = 0; i < N; i++) {
          for (let d = 0; d < remD; d += 2) {
            const u1 = Math.max(1e-12, Math.random());
            const u2 = Math.random();
            const mag = Math.sqrt(-2.0 * Math.log(u1));
            const g1 = mag * Math.cos(2.0 * Math.PI * u2);
            const g2 = mag * Math.sin(2.0 * Math.PI * u2);
            remValues[i * remD + d] = g1;
            remSums[d] += g1;
            if (d + 1 < remD) {
              remValues[i * remD + d + 1] = g2;
              remSums[d + 1] += g2;
            }
          }
        }

        // Center and compute sample variance of raw remaining values
        let rawRemVarSum = 0;
        for (let d = 0; d < remD; d++) {
          const m = remSums[d] / N;
          let v = 0;
          for (let i = 0; i < N; i++) {
            const diff = remValues[i * remD + d] - m;
            remValues[i * remD + d] = diff;
            v += diff * diff;
          }
          rawRemVarSum += v / N;
        }

        // Scale factor: empirical variance in dims 3..D-1 equals var3D exactly (50% ratio)
        const scale = (rawRemVarSum > 1e-12) ? Math.sqrt(var3D / rawRemVarSum) : 1.0;

        for (let i = 0; i < N; i++) {
          const coords = new Float64Array(D);
          coords[0] = basePts[i].x;
          coords[1] = basePts[i].y;
          coords[2] = basePts[i].z;
          for (let d = 0; d < remD; d++) {
            coords[3 + d] = remValues[i * remD + d] * scale;
          }
          points.push({
            x: coords[0],
            y: coords[1],
            z: coords[2],
            coords: coords
          });
        }
      }

      return points;
    }

    // =========================================================================


