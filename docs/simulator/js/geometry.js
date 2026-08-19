/**
 * GRIC Simulator - geometry.js
 * Metric Pruning Geometry & Exact C-Algorithm Math Routines
 * Replicated from src/gric-cluster/math (exact 4-Point, 5-Point Cayley-Menger, fmatch)
 */

function calc_min_dist_4pt(d1f, d2f, d12, d1k, d2k) {
  if (d12 < 1e-9) return Math.abs(d1f - d1k);

  const xk = (d1k * d1k + d12 * d12 - d2k * d2k) / (2.0 * d12);
  const yk_sq = d1k * d1k - xk * xk;
  const yk = yk_sq > 0.0 ? Math.sqrt(yk_sq) : 0.0;

  const xf = (d1f * d1f + d12 * d12 - d2f * d2f) / (2.0 * d12);
  const yf_sq = d1f * d1f - xf * xf;
  const yf = yf_sq > 0.0 ? Math.sqrt(yf_sq) : 0.0;

  return Math.sqrt((xf - xk) * (xf - xk) + (yf - yk) * (yf - yk));
}

function calc_min_dist_5pt(d_f_c1, d_f_c2, d_f_c3, d_t_c1, d_t_c2, d_t_c3, d_c1_c2, d_c1_c3, d_c2_c3) {
  if (d_c1_c2 < 1e-9) return 0.0;

  const x3 = (d_c1_c3 * d_c1_c3 + d_c1_c2 * d_c1_c2 - d_c2_c3 * d_c2_c3) / (2.0 * d_c1_c2);
  const y3_sq = d_c1_c3 * d_c1_c3 - x3 * x3;
  if (y3_sq < 1e-9) return 0.0;
  const y3 = Math.sqrt(y3_sq);

  const xF = (d_f_c1 * d_f_c1 + d_c1_c2 * d_c1_c2 - d_f_c2 * d_f_c2) / (2.0 * d_c1_c2);
  const yF = (d_f_c1 * d_f_c1 + d_c1_c3 * d_c1_c3 - d_f_c3 * d_f_c3 - 2.0 * xF * x3) / (2.0 * y3);
  const zF_sq = d_f_c1 * d_f_c1 - xF * xF - yF * yF;
  const zF = zF_sq > 0.0 ? Math.sqrt(zF_sq) : 0.0;

  const xT = (d_t_c1 * d_t_c1 + d_c1_c2 * d_c1_c2 - d_t_c2 * d_t_c2) / (2.0 * d_c1_c2);
  const yT = (d_t_c1 * d_t_c1 + d_c1_c3 * d_t_c3 - d_t_c3 * d_t_c3 - 2.0 * xT * x3) / (2.0 * y3);
  const zT_sq = d_t_c1 * d_t_c1 - xT * xT - yT * yT;
  const zT = zT_sq > 0.0 ? Math.sqrt(zT_sq) : 0.0;

  return Math.sqrt((xF - xT) * (xF - xT) + (yF - yT) * (yF - yT) + (zF - zT) * (zF - zT));
}

function fmatch(dr, a = 2.0, b = 0.5) {
  if (dr > 2.0) return 0.0;
  return a - (a - b) * dr / 2.0;
}

