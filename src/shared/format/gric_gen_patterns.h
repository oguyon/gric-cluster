/**
 * @file gric_gen_patterns.h
 * @brief Reusable synthetic coordinate and geometric pattern generators.
 */

#ifndef GRIC_GEN_PATTERNS_H
#define GRIC_GEN_PATTERNS_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef enum
{
    GEN_RANDOM,
    GEN_CIRCLE,
    GEN_WALK,
    GEN_SPIRAL,
    GEN_SPHERE,
    GEN_STAR,
    GEN_CONCENTRIC,
    GEN_TREE,
    GEN_CONCENTRIC_DENSE,
    GEN_RAND_EXP
} GenType;

typedef struct
{
    GenType type;
    int     dim;
    double  param;
} GeneratorConfig;

/**
 * @brief Generate a uniform random double in [0, 1].
 */
double rand_double(
    void);

/**
 * @brief Generate a uniform random point within the unit ball.
 */
void gen_random_point(
    double *out,
    int     dim);

/**
 * @brief Generate an exponential random point.
 */
void gen_randexp_point(
    double *out,
    int     dim);

/**
 * @brief Generate a point uniformly distributed on the surface of the unit sphere.
 */
void gen_sphere_point(
    double *out,
    int     dim);

/**
 * @brief Generate a point along a circular trajectory.
 */
void gen_circle_point(
    double *out,
    long    index,
    double  period,
    int     dim);

/**
 * @brief Generate a point along a 2D or 3D spiral trajectory.
 */
void gen_spiral_point(
    double *out,
    long    index,
    long    total_points,
    double  loops,
    int     dim);

/**
 * @brief Generate a point in a star/spoke distribution.
 */
void gen_star_point(
    double *out,
    long    index,
    long    total_points,
    double  spokes,
    int     dim);

/**
 * @brief Generate a point in concentric geometric shells.
 */
void gen_concentric_point(
    double *out,
    long    index,
    long    total_points,
    double  shells,
    int     dim);

/**
 * @brief Generate a point on a hierarchical binary tree.
 */
void gen_tree_point(
    double *out,
    long    index,
    long    total_points,
    double  unused_param,
    int     dim);

/**
 * @brief Generate a point in dense concentric shells.
 */
void gen_concentric_dense_point(
    double *out,
    long    index,
    long    total_points,
    double  shells,
    int     dim);

/**
 * @brief Step a random walk within the unit ball.
 */
void gen_walk_point(
    double *current,
    double  step_size,
    int     dim);

#endif // GRIC_GEN_PATTERNS_H
