#!/usr/bin/env python3
"""
generate_4dataset.py - Generate coupled datasets A, B, C and ground-truth D.

Generates 4 pre-synchronized datasets stored on the filesystem for the Milk framework
4-dataset reconstruction test:
  - Dataset A: Training input vectors [N x D_A] (noisy observations)
  - Dataset B: Training target vectors [N x D_B] (coupled underlying state)
               * Enforced to have the exact same sample count N as A and synchronized
                 sample-by-sample (row i of A <-> row i of B at timestamp t_i).
  - Dataset C: Query input vectors [M x D_A] (query trajectory in observation space)
  - Dataset D_true: Ground truth target vectors [M x D_B] for quality validation
"""

import argparse
import os
import sys
import numpy as np


def generate_coupled_data(
    n_train=1000, n_query=200, dim_a=3, dim_b=3, noise_a=0.02, noise_b=0.1
):
    """Generate coupled manifold trajectories for training and inference."""
    np.random.seed(42)

    # Parametric trajectory parameter t in [0, 4*pi]
    t_train = np.linspace(0.0, 4.0 * np.pi, n_train)

    # Observation space A: 3D spiral with noise
    a_x = np.cos(t_train)
    a_y = np.sin(t_train)
    a_z = 0.25 * t_train
    raw_a = np.column_stack([a_x, a_y, a_z])
    if dim_a > 3:
        extra_a = np.random.randn(n_train, dim_a - 3) * 0.05
        raw_a = np.hstack([raw_a, extra_a])
    data_a = raw_a + np.random.randn(*raw_a.shape) * noise_a

    # State space B: Coupled higher harmonic trajectory with noise
    b_x = np.sin(2.0 * t_train)
    b_y = np.cos(2.0 * t_train)
    b_z = np.sin(0.5 * t_train)
    raw_b = np.column_stack([b_x, b_y, b_z])
    if dim_b > 3:
        extra_b = np.column_stack([np.cos(3.0 * t_train)])
        raw_b = np.hstack([raw_b, extra_b])
    data_b = raw_b + np.random.randn(*raw_b.shape) * noise_b

    # Query space C: Intermediate evaluation points along the same manifold
    t_query = np.linspace(0.1, 4.0 * np.pi - 0.1, n_query)
    c_x = np.cos(t_query)
    c_y = np.sin(t_query)
    c_z = 0.25 * t_query
    raw_c = np.column_stack([c_x, c_y, c_z])
    if dim_a > 3:
        extra_c = np.random.randn(n_query, dim_a - 3) * 0.05
        raw_c = np.hstack([raw_c, extra_c])
    data_c = raw_c + np.random.randn(*raw_c.shape) * (noise_a * 0.5)

    # Ground truth D for queries: exact noiseless state on manifold B
    d_true_x = np.sin(2.0 * t_query)
    d_true_y = np.cos(2.0 * t_query)
    d_true_z = np.sin(0.5 * t_query)
    raw_d = np.column_stack([d_true_x, d_true_y, d_true_z])
    if dim_b > 3:
        extra_d = np.column_stack([np.cos(3.0 * t_query)])
        raw_d = np.hstack([raw_d, extra_d])
    data_d_true = raw_d

    return (
        data_a.astype(np.float32),
        data_b.astype(np.float32),
        data_c.astype(np.float32),
        data_d_true.astype(np.float32),
    )


def write_ascii_file(path, data):
    """Write 2D float array to ASCII space-separated format."""
    np.savetxt(path, data, fmt="%.6f", delimiter=" ")


def main():
    parser = argparse.ArgumentParser(description="Generate 4 datasets for Milk reconstruction.")
    parser.add_argument("--outdir", default="data_4dataset", help="Output directory")
    parser.add_argument("--ntrain", type=int, default=1000, help="Training sample count N")
    parser.add_argument("--nquery", type=int, default=200, help="Query sample count M")
    parser.add_argument("--dima", type=int, default=3, help="Dimension of A and C")
    parser.add_argument("--dimb", type=int, default=3, help="Dimension of B and D")
    parser.add_argument("--noise-a", type=float, default=0.02, help="Noise sigma in A")
    parser.add_argument("--noise-b", type=float, default=0.1, help="Noise sigma in B")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data_a, data_b, data_c, data_d_true = generate_coupled_data(
        n_train=args.ntrain,
        n_query=args.nquery,
        dim_a=args.dima,
        dim_b=args.dimb,
        noise_a=args.noise_a,
        noise_b=args.noise_b,
    )

    path_a = os.path.join(args.outdir, "dataset_A.txt")
    path_b = os.path.join(args.outdir, "dataset_B.txt")
    path_c = os.path.join(args.outdir, "dataset_C.txt")
    path_d = os.path.join(args.outdir, "dataset_D_true.txt")

    write_ascii_file(path_a, data_a)
    write_ascii_file(path_b, data_b)
    write_ascii_file(path_c, data_c)
    write_ascii_file(path_d, data_d_true)

    print(f"Generated 4 datasets in '{args.outdir}':")
    print(f"  Dataset A:      {data_a.shape[0]} samples x {data_a.shape[1]}D -> {path_a}")
    print(f"  Dataset B:      {data_b.shape[0]} samples x {data_b.shape[1]}D -> {path_b}")
    print(f"  Dataset C:      {data_c.shape[0]} samples x {data_c.shape[1]}D -> {path_c}")
    print(f"  Dataset D_true: {data_d_true.shape[0]} samples x {data_d_true.shape[1]}D -> {path_d}")


if __name__ == "__main__":
    main()
