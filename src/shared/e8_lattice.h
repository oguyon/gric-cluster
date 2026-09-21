#ifndef E8_LATTICE_H
#define E8_LATTICE_H

/**
 * @file e8_lattice.h
 * @brief E8 root lattice algebra, Conway-Sloane fast quantizer, and 240-NN geometry.
 *
 * Implements the E8 root lattice in 8 dimensions, its 240 root vectors (the Gosset
 * 8-polytope 4_21), Conway-Sloane fast O(D) coset decoding, and Product-E8 block
 * quantization for high-dimensional metric lower bounding.
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Dimension of the fundamental E8 lattice */
#define E8_DIM 8

/** Number of minimal non-zero vectors (kissing number / root count) */
#define E8_NUM_ROOTS 240

/** Squared Euclidean norm of root vectors in standard scaling */
#define E8_ROOT_NORM_SQ 2.0

/** Covering radius in standard normalization (roots norm^2 = 2) */
#define E8_COVERING_RADIUS 1.0

/** Normalized second moment of E8 (lowest in 8D) */
#define E8_SECOND_MOMENT 0.071682

/** Ratio of E8 covering radius to cubic Z^8 covering radius: 1 / sqrt(2) */
#define E8_COVERING_RATIO 0.7071067811865475

/**
 * e8_init_root_table() - Initialize the precomputed 240 root vectors of E8.
 *
 * Thread-safe one-time initialization of root lookup tables.
 */
void e8_init_root_table(void);

/**
 * e8_get_roots_doubled_int8() - Get pointer to the 240 root vectors in doubled int8 format.
 *
 * Coordinates are multiplied by 2 so all entries are exact integers in {-2, -1, 0, 1, 2}.
 *
 * Return: Pointer to array of 240 vectors of 8 int8_t values.
 */
const int8_t (*e8_get_roots_doubled_int8(void))[E8_DIM];

/**
 * e8_get_roots_float() - Get pointer to the 240 root vectors in float format.
 *
 * Return: Pointer to array of 240 vectors of 8 float values.
 */
const float (*e8_get_roots_float(void))[E8_DIM];

/**
 * e8_quantize_point_float() - Find closest point in E8 lattice to an 8D float vector.
 * @in:  Input 8D coordinate vector.
 * @out: Output closest E8 lattice vector.
 *
 * Uses the Conway-Sloane fast coset decoding algorithm: E8 = D8 U (D8 + 1/2).
 */
void e8_quantize_point_float(
    const float *restrict in,
    float       *restrict out);

/**
 * e8_quantize_point_double() - Find closest point in E8 lattice to an 8D double vector.
 * @in:  Input 8D coordinate vector.
 * @out: Output closest E8 lattice vector.
 */
void e8_quantize_point_double(
    const double *restrict in,
    double       *restrict out);

/**
 * e8_quantize_block_float() - Block-wise E8 quantization for vectors of dimension D.
 * @in:    Input vector of length @dim.
 * @out:   Output quantized vector of length @dim.
 * @dim:   Total vector dimension.
 * @scale: Lattice scaling factor (cell step size).
 *
 * Chunks vector into 8D slices and quantizes each slice onto scale * E8. Any trailing
 * dimensions (dim % 8) are quantized onto scale * Z.
 */
void e8_quantize_block_float(
    const float *restrict in,
    float       *restrict out,
    long                  dim,
    float                 scale);

/**
 * e8_quantize_block_double() - Block-wise E8 quantization for double vectors of dimension D.
 * @in:    Input double vector of length @dim.
 * @out:   Output quantized double vector of length @dim.
 * @dim:   Total vector dimension.
 * @scale: Lattice scaling factor (cell step size).
 */
void e8_quantize_block_double(
    const double *restrict in,
    double       *restrict out,
    long                   dim,
    double                 scale);

/**
 * e8_covering_radius() - Compute guaranteed maximum covering error for dimension D.
 * @dim:   Vector dimension.
 * @scale: Lattice step scale.
 *
 * Return: Maximum Euclidean distance from any point in R^D to the nearest E8^k lattice point.
 */
float e8_covering_radius(
    long  dim,
    float scale);

/**
 * e8_find_nearest_root_float() - Find the root vector among 240 roots closest to direction dir.
 * @dir:            Input 8D direction vector.
 * @best_root_idx:  Output index of the nearest root [0, 239].
 *
 * Return: Cosine similarity (inner product normalized) to the best root.
 */
float e8_find_nearest_root_float(
    const float *restrict dir,
    int         *restrict best_root_idx);

/**
 * e8_find_nearest_root_double() - Find the root vector among 240 roots closest to direction dir.
 * @dir:            Input 8D double direction vector.
 * @best_root_idx:  Output index of the nearest root [0, 239].
 *
 * Return: Cosine similarity (inner product normalized) to the best root.
 */
double e8_find_nearest_root_double(
    const double *restrict dir,
    int          *restrict best_root_idx);

#ifdef __cplusplus
}
#endif

#endif // E8_LATTICE_H
