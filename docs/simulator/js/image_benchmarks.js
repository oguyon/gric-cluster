/**
 * GRIC Simulator - image_benchmarks.js
 * Synthetic Image Time-Series Generators (Bouncing Balls)
 */

/* eslint-disable no-unused-vars */

// Descriptions for image benchmarks (added to global BENCHMARK_DESCS if defined)
if (typeof BENCHMARK_DESCS !== 'undefined')
{
  BENCHMARK_DESCS['img-ball-1'] =
    '<b>Single Bouncing Ball (32×32)</b>: 10,000 frames of a 2D circular disk ' +
    '(radius=5.0) bouncing elastically in a 32×32 box (D=1024).';
  BENCHMARK_DESCS['img-ball-3'] =
    '<b>3 Colliding Balls (32×32)</b>: 10,000 frames of 3 circular disks with ' +
    'elastic inter-ball collisions and boundary bounces in a 32×32 box (D=1024).';
}

/**
 * Check if a benchmark identifier corresponds to an image-mode dataset.
 * @param {string} type - Benchmark key
 * @returns {boolean}
 */
function isImageBenchmark(type)
{
  return typeof type === 'string' && type.startsWith('img-');
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
  const seed = options.seed !== undefined ? options.seed : 42;

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

  return frames;
}

/**
 * Generate benchmark dataset for image-based benchmark modes.
 * @param {string} type - Benchmark key ('img-ball-1', 'img-ball-3')
 * @param {number} N - Number of frames
 * @returns {Array<Float32Array>}
 */
function generateImageBenchmark(type, N = 1000)
{
  if (type === 'img-ball-1')
  {
    return generateBouncingBalls(
      { width: 32, height: 32, nballs: 1, radius: 5.0, collisions: false, seed: 42 },
      N
    );
  }
  if (type === 'img-ball-3')
  {
    return generateBouncingBalls(
      { width: 32, height: 32, nballs: 3, radius: 5.0, collisions: true, seed: 42 },
      N
    );
  }
  return generateBouncingBalls({ width: 32, height: 32, nballs: 1, radius: 5.0 }, N);
}
