/**
 * GRIC Simulator - image_benchmarks.js
 * Synthetic Image Time-Series Generators (Bouncing Balls)
 */

/* eslint-disable no-unused-vars */

// Descriptions for image benchmarks (added to global BENCHMARK_DESCS if defined)
if (typeof BENCHMARK_DESCS !== 'undefined')
{
  BENCHMARK_DESCS['img-ball-1'] =
    '<b>Single Bouncing Ball (32×32)</b>: 2D circular disk ' +
    '(radius=5.0) bouncing elastically in a 32×32 box (D=1024).';
  BENCHMARK_DESCS['img-ball-2'] =
    '<b>2 Colliding Balls (32×32)</b>: 2 circular disks with ' +
    'elastic inter-ball collisions and boundary bounces in a 32×32 box (D=1024).';
  BENCHMARK_DESCS['img-ball-3'] =
    '<b>3 Colliding Balls (32×32)</b>: 3 circular disks with ' +
    'elastic inter-ball collisions and boundary bounces in a 32×32 box (D=1024).';
  BENCHMARK_DESCS['img-asteroid-x'] =
    '<b>Asteroid View X (32×32)</b>: Synchronized lateral Y-Z projection of a 3D rotating ' +
    'non-spherical asteroid with modulated albedo crater features (D=1024).';
  BENCHMARK_DESCS['img-asteroid-y'] =
    '<b>Asteroid View Y (32×32)</b>: Synchronized lateral -X-Z projection of a 3D rotating ' +
    'non-spherical asteroid with modulated albedo crater features (D=1024).';
  BENCHMARK_DESCS['img-asteroid-z'] =
    '<b>Asteroid View Z (32×32)</b>: Synchronized polar X-Y projection of a 3D rotating ' +
    'non-spherical asteroid with modulated albedo crater features (D=1024).';
}

/**
 * Check if a benchmark identifier corresponds to an image-mode dataset.
 * @param {string} type - Benchmark key
 * @returns {boolean}
 */
function isImageBenchmark(type)
{
  if (typeof type === 'string' && type.startsWith('img-'))
  {
    return true;
  }
  if (type === 'reconstructed')
  {
    const slotD = (typeof datasetSlots !== 'undefined') ? datasetSlots['D'] : null;
    if (slotD && slotD.dataMode === 'image')
    {
      return true;
    }
  }
  return false;
}

/**
 * Simple seedable PRNG (Mulberry32) for reproducible benchmark sequences.
 * @param {number} seed
 * @returns {function(): number}
 */
function createSeededRandom(seed = 42)
{
  let a = (seed ^ 0x6d2b79f5) >>> 0;
  return function ()
  {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Initialize balls with random non-overlapping positions and random velocities.
 * Matches init_balls() in tools/gen_bouncing_balls.c.
 */
function initBalls(nballs, radius, W, H, rng)
{
  const balls = [];
  for (let i = 0; i < nballs; i++)
  {
    let attempts = 0;
    let overlap = false;
    let bx = 0;
    let by = 0;

    do
    {
      overlap = false;
      bx = radius + (W - 2.0 * radius) * rng();
      by = radius + (H - 2.0 * radius) * rng();

      for (let j = 0; j < i; j++)
      {
        const dx = bx - balls[j].x;
        const dy = by - balls[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < radius + balls[j].radius)
        {
          overlap = true;
          break;
        }
      }
      attempts++;
    } while (overlap && attempts < 1000);

    const speed = 0.5 + 1.5 * rng();
    const angle = 2.0 * Math.PI * rng();

    balls.push({
      x: bx,
      y: by,
      vx: speed * Math.cos(angle),
      vy: speed * Math.sin(angle),
      radius: radius
    });
  }
  return balls;
}

/**
 * Advance ball physics by one time step with wall reflections and optional
 * pairwise elastic collisions. Matches step_balls() in gen_bouncing_balls.c.
 */
function stepBalls(balls, nballs, W, H, ballCollisions)
{
  // 1. Move all balls by velocity
  for (let i = 0; i < nballs; i++)
  {
    balls[i].x += balls[i].vx;
    balls[i].y += balls[i].vy;
  }

  // 2. Resolve ball-to-ball collisions if enabled
  if (ballCollisions && nballs > 1)
  {
    for (let pass = 0; pass < 2; pass++)
    {
      for (let i = 0; i < nballs; i++)
      {
        for (let j = i + 1; j < nballs; j++)
        {
          let dx = balls[j].x - balls[i].x;
          let dy = balls[j].y - balls[i].y;
          let dist = Math.sqrt(dx * dx + dy * dy);
          const minDist = balls[i].radius + balls[j].radius;

          if (dist < minDist)
          {
            if (dist === 0.0)
            {
              dx = 0.1;
              dy = 0.0;
              dist = 0.1;
            }

            const nx = dx / dist;
            const ny = dy / dist;

            // Push apart to resolve overlap
            const overlap = minDist - dist;
            balls[i].x -= 0.5 * overlap * nx;
            balls[i].y -= 0.5 * overlap * ny;
            balls[j].x += 0.5 * overlap * nx;
            balls[j].y += 0.5 * overlap * ny;

            // Relative velocity along normal
            const rvx = balls[j].vx - balls[i].vx;
            const rvy = balls[j].vy - balls[i].vy;
            const velN = rvx * nx + rvy * ny;

            if (velN < 0.0)
            {
              // Equal mass elastic collision
              balls[i].vx += velN * nx;
              balls[i].vy += velN * ny;
              balls[j].vx -= velN * nx;
              balls[j].vy -= velN * ny;
            }
          }
        }
      }
    }
  }

  // 3. Bounce off walls and clamp to boundary
  for (let i = 0; i < nballs; i++)
  {
    const r = balls[i].radius;

    if (balls[i].x < r)
    {
      balls[i].x = r;
      balls[i].vx = -balls[i].vx;
    }
    else if (balls[i].x > W - r)
    {
      balls[i].x = W - r;
      balls[i].vx = -balls[i].vx;
    }

    if (balls[i].y < r)
    {
      balls[i].y = r;
      balls[i].vy = -balls[i].vy;
    }
    else if (balls[i].y > H - r)
    {
      balls[i].y = H - r;
      balls[i].vy = -balls[i].vy;
    }
  }
}

/**
 * Render all balls into a float pixel buffer [W * H].
 * Matches render_frame() in tools/gen_bouncing_balls.c.
 */
function renderFrameToBuffer(buf, balls, nballs, W, H)
{
  buf.fill(0.0);

  for (let b = 0; b < nballs; b++)
  {
    const cx = balls[b].x;
    const cy = balls[b].y;
    const r = balls[b].radius;
    const rCore = (2.0 * r) / 3.0;
    const rSoft = r / 3.0;

    let y0 = Math.floor(cy - r);
    let y1 = Math.ceil(cy + r);
    let x0 = Math.floor(cx - r);
    let x1 = Math.ceil(cx + r);

    if (y0 < 0) y0 = 0;
    if (y1 >= H) y1 = H - 1;
    if (x0 < 0) x0 = 0;
    if (x1 >= W) x1 = W - 1;

    for (let yy = y0; yy <= y1; yy++)
    {
      const dy = yy - cy;
      const rowOffset = yy * W;
      for (let xx = x0; xx <= x1; xx++)
      {
        const dx = xx - cx;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= rCore)
        {
          buf[rowOffset + xx] += 1.0;
        }
        else if (dist <= r)
        {
          const factor = 1.0 - (dist - rCore) / rSoft;
          buf[rowOffset + xx] += factor;
        }
      }
    }
  }
}

/**
 * Generate a sequence of bouncing balls frames.
 * @param {Object} options - Benchmark options
 * @param {number} numFrames - Total frames to generate
 * @returns {Array<Float32Array>}
 */
function generateBouncingBalls(options = {}, numFrames = 1000)
{
  const W = options.width || 32;
  const H = options.height || 32;
  const nballs = options.nballs || 1;
  const radius = options.radius || 5.0;
  const collisions = options.collisions || false;
  let seed = 42;
  if (options.seed !== undefined && options.seed !== null)
  {
    seed = options.seed;
  }
  else if (options.randomSeed)
  {
    seed = Math.floor(Math.random() * 0x7FFFFFFF);
  }

  const rng = createSeededRandom(seed);
  const balls = initBalls(nballs, radius, W, H, rng);
  const frames = [];

  for (let f = 0; f < numFrames; f++)
  {
    const buf = new Float32Array(W * H);
    renderFrameToBuffer(buf, balls, nballs, W, H);
    frames.push(buf);
    stepBalls(balls, nballs, W, H, collisions);
  }

  frames.seed = seed;
  return frames;
}

/**
 * Fisher-Yates array shuffle. Permutes array elements in-place.
 * @param {Array} arr - Array to shuffle
 * @param {Function} [rng=Math.random] - Random number generator returning [0, 1)
 * @returns {Array} Shuffled array
 */
function shuffleArray(arr, rng = Math.random)
{
  if (!Array.isArray(arr) || arr.length <= 1)
  {
    return arr;
  }
  for (let i = arr.length - 1; i > 0; i--)
  {
    const j = Math.floor(rng() * (i + 1));
    const temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
  return arr;
}

/**
 * Generate benchmark dataset for image-based benchmark modes.
 * @param {string} type - Benchmark key ('img-ball-1', 'img-ball-2', 'img-ball-3')
 * @param {number} N - Number of frames
 * @param {Object} [options={}] - Custom options (seed, randomSeed, etc.)
 * @returns {Array<Float32Array>}
 */
function generateImageBenchmark(type, N = 1000, options = {})
{
  let seed = 42;
  if (options.seed !== undefined && options.seed !== null)
  {
    seed = options.seed;
  }
  else if (options.randomSeed)
  {
    seed = Math.floor(Math.random() * 0x7FFFFFFF);
  }

  const width = options.width || 32;
  const height = options.height || 32;
  const radius = options.radius || 5.0;

  let frames;
  if (type === 'img-ball-1')
  {
    frames = generateBouncingBalls(
      { width, height, nballs: 1, radius, collisions: false, seed },
      N
    );
  }
  else if (type === 'img-ball-2')
  {
    frames = generateBouncingBalls(
      { width, height, nballs: 2, radius, collisions: true, seed },
      N
    );
  }
  else if (type === 'img-ball-3')
  {
    frames = generateBouncingBalls(
      { width, height, nballs: 3, radius, collisions: true, seed },
      N
    );
  }
  else if (type === 'img-asteroid-x' || type === 'img-asteroid-y' || type === 'img-asteroid-z')
  {
    const view = (type === 'img-asteroid-y') ? 'Y' : ((type === 'img-asteroid-z') ? 'Z' : 'X');
    frames = generateAsteroidSequence(view, N, options);
  }
  else
  {
    frames = generateBouncingBalls({ width, height, nballs: 1, radius, seed }, N);
  }

  frames.seed = seed;
  return frames;
}

const ASTEROID_CRATERS = [
  { cx:  0.707, cy:  0.707, cz:  0.000, sigmaSq: 0.040, weightDep: -0.28, weightRim: 0.18 },
  { cx: -0.500, cy:  0.300, cz:  0.812, sigmaSq: 0.060, weightDep: -0.32, weightRim: 0.15 },
  { cx:  0.200, cy: -0.800, cz:  0.566, sigmaSq: 0.035, weightDep: -0.25, weightRim: 0.18 },
  { cx: -0.700, cy: -0.600, cz:  0.387, sigmaSq: 0.070, weightDep: -0.22, weightRim: 0.12 },
  { cx:  0.000, cy:  0.500, cz: -0.866, sigmaSq: 0.050, weightDep: -0.28, weightRim: 0.14 },
  { cx:  0.600, cy: -0.400, cz: -0.693, sigmaSq: 0.030, weightDep: -0.30, weightRim: 0.16 },
  { cx: -0.800, cy:  0.200, cz: -0.566, sigmaSq: 0.038, weightDep: -0.25, weightRim: 0.16 },
  { cx:  0.100, cy:  0.200, cz:  0.975, sigmaSq: 0.045, weightDep: -0.28, weightRim: 0.14 }
];

const COS_05 = 0.87758256;
const SIN_05 = 0.47942554;
const COS_10 = 0.54030230;
const SIN_10 = 0.84147098;

let _lastBumpH = 0.0;

/**
 * Fast zero-transcendental bump height perturbation using Chebyshev recurrences.
 */
function computeAsteroidBumpHeight(ux, uy, uz)
{
  const uz2 = uz * uz;
  const rxy2 = Math.max(0.0, 1.0 - uz2);
  if (rxy2 < 1e-8)
  {
    return 0.0;
  }
  const rxy = Math.sqrt(rxy2);
  const invRxy = 1.0 / rxy;

  const c1 = ux * invRxy;
  const s1 = uy * invRxy;

  const c2 = c1 * c1 - s1 * s1;
  const s2 = 2.0 * c1 * s1;

  const c3 = c2 * c1 - s2 * s1;
  const s3 = s2 * c1 + c2 * s1;

  const c5 = c3 * c2 - s3 * s2;
  const s5 = s3 * c2 + c3 * s2;

  const c7 = c5 * c2 - s5 * s2;
  const s7 = s5 * c2 + c5 * s2;

  const sin2th = 2.0 * rxy * uz;
  const cos3th = uz * (4.0 * uz2 - 3.0);
  const sin5th = rxy * (16.0 * uz2 * uz2 - 12.0 * uz2 + 1.0);

  const h1 = 0.16 * s2 * uz;
  const cos3phi05 = c3 * COS_05 - s3 * SIN_05;
  const h2 = 0.12 * cos3phi05 * sin2th;
  const h3 = 0.08 * s5 * cos3th;
  const cos7phiM1 = c7 * COS_10 + s7 * SIN_10;
  const h4 = 0.05 * cos7phiM1 * sin5th;

  let h = h1 + h2 + h3 + h4;

  for (let k = 0; k < 8; k++)
  {
    const c = ASTEROID_CRATERS[k];
    const dot = ux * c.cx + uy * c.cy + uz * c.cz;
    if (dot < 0.65)
    {
      continue;
    }
    const angSq = 1.0 - dot;
    const rSq = angSq / c.sigmaSq;
    h += c.weightDep * Math.exp(-rSq * 2.5) + c.weightRim * rSq * Math.exp(-rSq);
  }
  return h;
}

/**
 * Fast zero-allocation surface implicit equation F(p).
 */
function evalAsteroidSurfaceF(p, invA2, invB2, invC2)
{
  const px = (Array.isArray(p) || ArrayBuffer.isView(p)) ? p[0] : p;
  const py = (Array.isArray(p) || ArrayBuffer.isView(p)) ? p[1] : arguments[1];
  const pz = (Array.isArray(p) || ArrayBuffer.isView(p)) ? p[2] : arguments[2];
  const a2 = (Array.isArray(p) || ArrayBuffer.isView(p)) ? invA2 : arguments[3];
  const b2 = (Array.isArray(p) || ArrayBuffer.isView(p)) ? invB2 : arguments[4];
  const c2 = (Array.isArray(p) || ArrayBuffer.isView(p)) ? invC2 : arguments[5];

  const plen2 = px * px + py * py + pz * pz;
  if (plen2 < 1e-6)
  {
    _lastBumpH = 0.0;
    return { f: -1.0, h: 0.0 };
  }
  const invLen = 1.0 / Math.sqrt(plen2);
  _lastBumpH = computeAsteroidBumpHeight(px * invLen, py * invLen, pz * invLen);
  const onePlusH = 1.0 + _lastBumpH;
  const f = (px * px * a2 + py * py * b2 + pz * pz * c2) - (onePlusH * onePlusH);
  return { f, h: _lastBumpH };
}

/**
 * Inlined fast scalar surface evaluation without object creation.
 */
function evalAsteroidF_scalar(px, py, pz, invA2, invB2, invC2)
{
  const plen2 = px * px + py * py + pz * pz;
  if (plen2 < 1e-6)
  {
    _lastBumpH = 0.0;
    return -1.0;
  }
  const invLen = 1.0 / Math.sqrt(plen2);
  _lastBumpH = computeAsteroidBumpHeight(px * invLen, py * invLen, pz * invLen);
  const onePlusH = 1.0 + _lastBumpH;
  return (px * px * invA2 + py * py * invB2 + pz * pz * invC2) - (onePlusH * onePlusH);
}

function generateAsteroidSequence(view = 'X', numFrames = 10000, options = {})
{
  const W = options.width || 32;
  const H = options.height || 32;
  const omegaPhi   = (options.omegaPhi !== undefined) ? options.omegaPhi : 1.0;
  const omegaTheta = (options.omegaTheta !== undefined) ? options.omegaTheta : 0.236;
  const axisA = options.axisA || 14.0;
  const axisB = options.axisB || 9.8;
  const axisC = options.axisC || 6.4;
  const ambient = (options.ambient !== undefined) ? options.ambient : 0.25;
  const albedoStr = (options.albedo !== undefined) ? options.albedo : 0.40;

  const scaleOut = 1.35;
  const invA2 = 1.0 / (axisA * axisA);
  const invB2 = 1.0 / (axisB * axisB);
  const invC2 = 1.0 / (axisC * axisC);
  const invA2Out = 1.0 / ((axisA * scaleOut) * (axisA * scaleOut));
  const invB2Out = 1.0 / ((axisB * scaleOut) * (axisB * scaleOut));
  const invC2Out = 1.0 / ((axisC * scaleOut) * (axisC * scaleOut));

  const minInvOut = Math.min(invA2Out, Math.min(invB2Out, invC2Out));
  const maxR = (1.0 / Math.sqrt(minInvOut)) + 0.6;
  const maxR2 = maxR * maxR;

  const sunDir = [0.57735, 0.57735, 0.57735];
  const radConv = Math.PI / 180.0;

  const viewId = (view === 'Y' || view === 'y' || view === 1) ? 1
               : ((view === 'Z' || view === 'z' || view === 2) ? 2 : 0);

  const cx = (W - 1) * 0.5;
  const cy = (H - 1) * 0.5;
  const subOffsets = [-0.25, 0.25];
  const frames = [];

  for (let fi = 0; fi < numFrames; fi++)
  {
    const phiRad   = (omegaPhi * fi) * radConv;
    const thetaRad = (omegaTheta * fi) * radConv;

    const cp = Math.cos(phiRad);
    const sp = Math.sin(phiRad);
    const ct = Math.cos(thetaRad);
    const st = Math.sin(thetaRad);

    const R0 =  cp * ct, R1 = -sp, R2 =  cp * st;
    const R3 =  sp * ct, R4 =  cp, R5 =  sp * st;
    const R6 = -st,      R7 = 0.0, R8 =  ct;

    let dbx, dby, dbz;
    if (viewId === 0)
    {
      dbx = -R0; dby = -R1; dbz = -R2;
    }
    else if (viewId === 1)
    {
      dbx = -R3; dby = -R4; dbz = -R5;
    }
    else
    {
      dbx = -R6; dby = -R7; dbz = -R8;
    }

    const A = dbx * dbx * invA2Out + dby * dby * invB2Out + dbz * dbz * invC2Out;
    const inv2A = 0.5 / A;
    const fourA = 4.0 * A;

    const buf = new Float32Array(W * H);

    for (let y = 0; y < H; y++)
    {
      const dyCenter = y - cy;
      const dyCenter2 = dyCenter * dyCenter;

      for (let x = 0; x < W; x++)
      {
        const dxCenter = x - cx;
        if (dxCenter * dxCenter + dyCenter2 > maxR2)
        {
          continue;
        }

        let accum = 0.0;
        for (let sy = 0; sy < 2; sy++)
        {
          const py = dyCenter + subOffsets[sy];
          for (let sx = 0; sx < 2; sx++)
          {
            const px = dxCenter + subOffsets[sx];

            let obx, oby, obz;
            if (viewId === 0)
            {
              obx = R0 * 100.0 + R3 * px + R6 * py;
              oby = R1 * 100.0 + R4 * px + R7 * py;
              obz = R2 * 100.0 + R5 * px + R8 * py;
            }
            else if (viewId === 1)
            {
              obx = -R0 * px + R3 * 100.0 + R6 * py;
              oby = -R1 * px + R4 * 100.0 + R7 * py;
              obz = -R2 * px + R5 * 100.0 + R8 * py;
            }
            else
            {
              obx = R0 * px + R3 * py + R6 * 100.0;
              oby = R1 * px + R4 * py + R7 * 100.0;
              obz = R2 * px + R5 * py + R8 * 100.0;
            }

            const B = 2.0 * (obx * dbx * invA2Out + oby * dby * invB2Out +
                             obz * dbz * invC2Out);
            const C = obx * obx * invA2Out + oby * oby * invB2Out +
                      obz * obz * invC2Out - 1.0;
            const discr = B * B - fourA * C;
            if (discr < 0.0)
            {
              continue;
            }

            const sqrtD = Math.sqrt(discr);
            let sNear = (-B - sqrtD) * inv2A;
            const sFar = (-B + sqrtD) * inv2A;
            if (sFar <= 0.0)
            {
              continue;
            }
            if (sNear < 0.0)
            {
              sNear = 0.0;
            }

            const N_PROBE = 4;
            const ds = (sFar - sNear) * 0.25;
            let sPrev = sNear;
            const pTx = obx + sPrev * dbx;
            const pTy = oby + sPrev * dby;
            const pTz = obz + sPrev * dbz;
            const fPrev = evalAsteroidF_scalar(pTx, pTy, pTz, invA2, invB2, invC2);

            let sHit = -1.0;
            if (fPrev <= 0.0)
            {
              sHit = sNear;
            }
            else
            {
              for (let step = 1; step <= N_PROBE; step++)
              {
                const sCurr = sNear + step * ds;
                const pCx = obx + sCurr * dbx;
                const pCy = oby + sCurr * dby;
                const pCz = obz + sCurr * dbz;
                const fCurr = evalAsteroidF_scalar(pCx, pCy, pCz, invA2, invB2, invC2);
                if (fCurr <= 0.0)
                {
                  let sLo = sPrev;
                  let sHi = sCurr;
                  for (let bi = 0; bi < 3; bi++)
                  {
                    const sMid = 0.5 * (sLo + sHi);
                    const fMid = evalAsteroidF_scalar(
                      obx + sMid * dbx,
                      oby + sMid * dby,
                      obz + sMid * dbz,
                      invA2, invB2, invC2
                    );
                    if (fMid <= 0.0)
                    {
                      sHi = sMid;
                    }
                    else
                    {
                      sLo = sMid;
                    }
                  }
                  sHit = 0.5 * (sLo + sHi);
                  break;
                }
                sPrev = sCurr;
              }
            }

            if (sHit <= 0.0)
            {
              continue;
            }

            const pHitX = obx + sHit * dbx;
            const pHitY = oby + sHit * dby;
            const pHitZ = obz + sHit * dbz;
            const f0 = evalAsteroidF_scalar(pHitX, pHitY, pHitZ, invA2, invB2, invC2);
            const hitH = _lastBumpH;

            const eps = 0.02;
            const fx = evalAsteroidF_scalar(pHitX + eps, pHitY, pHitZ, invA2, invB2, invC2);
            const fy = evalAsteroidF_scalar(pHitX, pHitY + eps, pHitZ, invA2, invB2, invC2);
            const fz = evalAsteroidF_scalar(pHitX, pHitY, pHitZ + eps, invA2, invB2, invC2);

            let nbx = fx - f0;
            let nby = fy - f0;
            let nbz = fz - f0;
            const nLen = 1.0 / Math.sqrt(nbx * nbx + nby * nby + nbz * nbz + 1e-12);
            nbx *= nLen;
            nby *= nLen;
            nbz *= nLen;

            const nwx = R0 * nbx + R1 * nby + R2 * nbz;
            const nwy = R3 * nbx + R4 * nby + R5 * nbz;
            const nwz = R6 * nbx + R7 * nby + R8 * nbz;

            const nDotL = nwx * sunDir[0] + nwy * sunDir[1] + nwz * sunDir[2];
            const diffuse = Math.max(0.0, nDotL);
            const albedo = Math.max(0.12, Math.min(1.0, 0.60 + albedoStr * hitH));

            accum += albedo * (ambient + (1.0 - ambient) * diffuse);
          } // for sx
        } // for sy
        buf[y * W + x] = accum * 0.25;
      } // for x
    } // for y
    frames.push(buf);
  } // for fi

  return frames;
}

/**
 * 1-Click setup for Asteroid View X -> View Y Reconstruction Test.
 * Sets Dataset A: Asteroid View X (Train split: 8,000 frames)
 * Sets Dataset B: Asteroid View Y (Train split: 8,000 frames)
 * Sets Dataset C: Asteroid View X (Test split: 2,000 frames)
 * Activates ABCD 4-Panel view and runs reconstruction.
 */
async function setupAsteroidReconTest(numTotalFrames = 10000, split = 0.80)
{
  if (typeof showToast === 'function')
  {
    showToast(
      `🪐 Synthesizing ${numTotalFrames.toLocaleString()}-frame ` +
      `Asteroid X & Y sequences...`
    );
  }

  if (typeof updateSlotGenState === 'function')
  {
    updateSlotGenState('A', 'generating');
    updateSlotGenState('B', 'generating');
    updateSlotGenState('C', 'generating');
  }

  await new Promise(r => setTimeout(r, 40));

  const nTrain = Math.floor(numTotalFrames * split);
  const nTest  = numTotalFrames - nTrain;

  let trainX = null;
  let trainY = null;
  let testX  = null;
  let useNative = false;

  function parseTxtLines(txt, expectedDim = 1024)
  {
    const lines = txt.trim().split('\n');
    const outFrames = [];
    for (let i = 0; i < lines.length; i++)
    {
      const line = lines[i].trim();
      if (!line) continue;
      const parts = line.split(/\s+/);
      const arr = new Float32Array(expectedDim);
      for (let p = 0; p < expectedDim; p++)
      {
        arr[p] = parseFloat(parts[p]);
      }
      outFrames.push(arr);
    }
    return outFrames;
  }

  if (typeof DesktopBridge !== 'undefined' && DesktopBridge.isNativeSupported())
  {
    try
    {
      let needGenerate = true;
      try
      {
        const trnX = await DesktopBridge.readFile('datasets/asteroid/asteroid_train_X.txt');
        const trnY = await DesktopBridge.readFile('datasets/asteroid/asteroid_train_Y.txt');
        const tstX = await DesktopBridge.readFile('datasets/asteroid/asteroid_test_X.txt');
        if (trnX && trnY && tstX)
        {
          const parsedTrnX = parseTxtLines(trnX, 1024);
          const parsedTrnY = parseTxtLines(trnY, 1024);
          const parsedTstX = parseTxtLines(tstX, 1024);
          if (parsedTrnX.length >= nTrain &&
              parsedTrnY.length >= nTrain &&
              parsedTstX.length >= nTest)
          {
            trainX = parsedTrnX.slice(0, nTrain);
            trainY = parsedTrnY.slice(0, nTrain);
            testX  = parsedTstX.slice(0, nTest);
            needGenerate = false;
            useNative = true;
          }
        }
      }
      catch (_)
      {
        needGenerate = true;
      }

      if (needGenerate)
      {
        if (typeof showToast === 'function')
        {
          showToast(
            `⚡ Offloading to native OpenMP gric-gen-asteroid ` +
            `(${numTotalFrames.toLocaleString()} frames)...`
          );
        }
        const cliRes = await DesktopBridge.runCliJob({
          cmd: 'gric-gen-asteroid',
          args: [
            '-f', String(numTotalFrames),
            '-split', String(split),
            '-txt',
            '-o', 'datasets/asteroid'
          ]
        });

        if (cliRes && cliRes.exitCode !== 0)
        {
          throw new Error(`gric-gen-asteroid exited with code ${cliRes.exitCode}`);
        }

        const txtTrnX = await DesktopBridge.readFile(
          'datasets/asteroid/asteroid_train_X.txt'
        );
        const txtTrnY = await DesktopBridge.readFile(
          'datasets/asteroid/asteroid_train_Y.txt'
        );
        const txtTstX = await DesktopBridge.readFile(
          'datasets/asteroid/asteroid_test_X.txt'
        );

        trainX = parseTxtLines(txtTrnX, 1024).slice(0, nTrain);
        trainY = parseTxtLines(txtTrnY, 1024).slice(0, nTrain);
        testX  = parseTxtLines(txtTstX, 1024).slice(0, nTest);
        useNative = true;
      }
    }
    catch (err)
    {
      console.warn('[setupAsteroidReconTest] Native generator failed, falling back to JS:', err);
      useNative = false;
    }
  }

  if (!useNative)
  {
    const seqX = generateAsteroidSequence('X', numTotalFrames);
    const seqY = generateAsteroidSequence('Y', nTrain);

    trainX = seqX.slice(0, nTrain);
    trainY = seqY;
    testX  = seqX.slice(nTrain);
  }

  if (typeof setMultiDatasetEnabled === 'function')
  {
    setMultiDatasetEnabled(true);
  }

  function initSlot(slotId, bKey, frames)
  {
    if (typeof datasetSlots !== 'undefined' && datasetSlots[slotId])
    {
      const slot = datasetSlots[slotId];
      slot.benchmarkKey = bKey;
      slot.dataMode = 'image';
      slot.imageWidth = 32;
      slot.imageHeight = 32;
      slot.imageDim = 1024;
      slot.currentDim = 1024;
      slot.rlim = 2.98;
      slot.benchmarkDataset = frames;
      slot.rawBenchmarkDataset = frames;
      slot.sampleCount = frames.length;
      slot.totalFrames = 0;
      slot.currentFrameIdx = 0;
      slot.clusters = [];
      slot.assignmentHistory = [];
      slot.frameHistory = [];
      slot.totalEvals = 0;
      slot.naiveEvals = 0;
      slot.dcc = [];
      slot.dccMin = null;
      slot.transitionCounts = [];
      slot.prevAssignedCluster = -1;
      slot.lastTransitionFrom = -1;
      slot.lastTransitionTo = -1;
      slot.currentEvaluations = [];
      slot.currentPruned = [];
      slot.currentPredicted = [];

      const sel = document.getElementById(`selectBenchmark_${slotId}`);
      if (sel) sel.value = bKey;
      slot.genState = 'ready';

      if (typeof activeDatasetSlot !== 'undefined' && activeDatasetSlot === slotId)
      {
        totalFrames = 0;
        clusters = [];
        totalEvals = 0;
        naiveEvals = 0;
        assignmentHistory = [];
        frameHistory = [];
        dcc = [];
        dccMin = null;
        transitionCounts = [];
        prevAssignedCluster = -1;
        benchmarkDataset = frames;
        rawBenchmarkDataset = frames;
        currentDim = 1024;
        imageWidth = 32;
        imageHeight = 32;
        imageDim = 1024;
        dataMode = 'image';
        currentImageFrame = frames[0];
        isDatasetStaged = true;
      }
    }
  }

  initSlot('A', 'img-asteroid-x', trainX);
  initSlot('B', 'img-asteroid-y', trainY);
  initSlot('C', 'img-asteroid-x', testX);

  if (typeof datasetSlots !== 'undefined')
  {
    if (datasetSlots['A'])
    {
      datasetSlots['A'].knnResults = {
        k: 30,
        totalFrames: nTrain,
        indices: new Int32Array(1)
      };
    }
    if (datasetSlots['B'])
    {
      datasetSlots['B'].knnResults = {
        k: 30,
        totalFrames: nTrain,
        indices: new Int32Array(1)
      };
    }
  }

  if (typeof updateSlotGenState === 'function')
  {
    updateSlotGenState('A', 'ready');
    updateSlotGenState('B', 'ready');
    updateSlotGenState('C', 'ready');
  }

  if (typeof setClusteringRlim === 'function')
  {
    setClusteringRlim(2.98, false);
  }
  else if (typeof rlim !== 'undefined')
  {
    rlim = 1.60;
  }

  if (typeof datasetSlots !== 'undefined' && datasetSlots['D'])
  {
    const slotD = datasetSlots['D'];
    slotD.benchmarkKey = 'reconstructed';
    slotD.dataMode = 'image';
    slotD.imageWidth = 32;
    slotD.imageHeight = 32;
    slotD.imageDim = 1024;
    slotD.currentDim = 1024;
    slotD.benchmarkDataset = [];
    slotD.rawBenchmarkDataset = [];
    slotD.reconstructionInfo = null;
    slotD.reconstructionSourceNeighbors = null;
    const selD = document.getElementById('selectBenchmark_D');
    if (selD) selD.value = 'reconstructed';
  }

  if (typeof updateDatasetStatusBadge === 'function')
  {
    updateDatasetStatusBadge();
  }

  if (typeof setRecon4PanelView === 'function')
  {
    setRecon4PanelView(true);
  }

  if (typeof showToast === 'function')
  {
    showToast(
      `🪐 Staged: [A] Asteroid X Train (${nTrain.toLocaleString()}), ` +
      `[B] Asteroid Y Train (${nTrain.toLocaleString()}), ` +
      `[C] Asteroid X Test (${nTest.toLocaleString()}). Computing Recon...`
    );
  }

  if (typeof executeDatasetReconstruction === 'function')
  {
    await executeDatasetReconstruction();
    if (typeof datasetSlots !== 'undefined' && datasetSlots['D'])
    {
      datasetSlots['D'].genState = 'ready';
    }
    if (typeof updateSlotGenState === 'function')
    {
      updateSlotGenState('D', 'ready');
    }
  }
}

if (typeof window !== 'undefined')
{
  window.isImageBenchmark = isImageBenchmark;
  window.generateImageBenchmark = generateImageBenchmark;
  window.generateBouncingBalls = generateBouncingBalls;
  window.generateAsteroidSequence = generateAsteroidSequence;
  window.setupAsteroidReconTest = setupAsteroidReconTest;
  window.shuffleArray = shuffleArray;
}

