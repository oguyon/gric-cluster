#ifndef CLUSTER_MATH_H
#define CLUSTER_MATH_H

/**
 * @brief Evaluates match probability weight based on normalized distance.
 *
 * Implements a linear decay matching reward: returns a value interpolating
 * from 'a' (at dr=0) to 'b' (at dr=2). Beyond dr=2, returns 0.
 *
 * @param dr Normalized distance ratio.
 * @param a Start threshold weight.
 * @param b End threshold weight.
 * @return Probability factor in range [0.0, a].
 */
double fmatch(double dr, double a, double b);

/**
 * @brief Computes minimum distance between a point and a candidate point using 4-point configuration.
 *
 * Calculates the Euclidean distance from a new 4th point to a candidate 3rd point,
 * using known distances to two base points (1 and 2). This utilizes 2D triangulation.
 *
 * @param d14 Distance between point 1 and 4.
 * @param d24 Distance between point 2 and 4.
 * @param d12 Distance between point 1 and 2.
 * @param d13 Distance between point 1 and 3.
 * @param d23 Distance between point 2 and 3.
 * @return Min distance between point 3 and 4 in the reconstructed space.
 */
double calc_min_dist_4pt(double d14, double d24, double d12, double d13, double d23);

/**
 * @brief Computes minimum distance between a point and a candidate point using 5-point configuration.
 *
 * Solves 3D triangulation coordinates for a point F (current frame) and T (target candidate cluster anchor)
 * using their known Euclidean distances to 3 fixed anchors (C1, C2, C3). Reconstructs their coordinates
 * in a local 3D coordinate system and returns the Euclidean distance between F and T.
 *
 * @param d_f_c1 Distance from frame F to C1.
 * @param d_f_c2 Distance from frame F to C2.
 * @param d_f_c3 Distance from frame F to C3.
 * @param d_t_c1 Distance from target T to C1.
 * @param d_t_c2 Distance from target T to C2.
 * @param d_t_c3 Distance from target T to C3.
 * @param d_c1_c2 Distance between C1 and C2.
 * @param d_c1_c3 Distance between C1 and C3.
 * @param d_c2_c3 Distance between C2 and C3.
 * @return The computed distance between F and T in reconstructed 3D space, or 0.0 if singular geometry.
 */
double calc_min_dist_5pt(double d_f_c1, double d_f_c2, double d_f_c3,
                         double d_t_c1, double d_t_c2, double d_t_c3,
                         double d_c1_c2, double d_c1_c3, double d_c2_c3);

#endif // CLUSTER_MATH_H
