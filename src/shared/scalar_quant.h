#ifndef SCALAR_QUANT_H
#define SCALAR_QUANT_H

/**
 * @file scalar_quant.h
 * @brief 8-bit Scalar Quantization (SQ8) with metric lower-bounding.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/** Magic identifier for .sq8 sidecar files */
#define SQ8_FILE_MAGIC "SQ8_0001"

/**
 * @brief Parameters defining uniform 8-bit scalar quantization.
 */
typedef struct
{
    float min_val;    /**< Minimum value mapped to 0 */
    float max_val;    /**< Maximum value mapped to 255 */
    float scale;      /**< Step delta: (max_val - min_val) / 255.0f */
    float inv_scale;  /**< Reciprocal step: 255.0f / (max_val - min_val) */
    float err_radius; /**< Max single-vector Euclidean error: sqrt(dim) * scale * 0.5f */
    long  dim;        /**< Vector dimension (element count per frame) */
} SQ8Params;

/**
 * @brief Initialize SQ8 parameters from known min and max values.
 */
void sq8_init_params(
    SQ8Params *params,
    float      min_val,
    float      max_val,
    long       dim);

/**
 * @brief Calibrate SQ8 parameters by scanning a float array.
 */
void sq8_calibrate_float(
    SQ8Params   *params,
    const float *data,
    long         num_elements,
    long         dim);

/**
 * @brief Calibrate SQ8 parameters by scanning a double array.
 */
void sq8_calibrate_double(
    SQ8Params    *params,
    const double *data,
    long          num_elements,
    long          dim);

/**
 * @brief Quantize a single-precision float frame to 8-bit unsigned integers.
 */
void sq8_quantize_float(
    const float     *restrict src,
    uint8_t         *restrict dst,
    const SQ8Params *restrict params);

/**
 * @brief Quantize a double-precision frame to 8-bit unsigned integers.
 */
void sq8_quantize_double(
    const double    *restrict src,
    uint8_t         *restrict dst,
    const SQ8Params *restrict params);

/**
 * @brief Compute sum of squared differences between two uint8 vectors using SIMD.
 */
uint64_t sq8_dist_squared_u8(
    const uint8_t *restrict a,
    const uint8_t *restrict b,
    long                    dim);

/**
 * @brief Compute guaranteed metric lower bound between two quantized vectors.
 */
double sq8_compute_lower_bound(
    const uint8_t   *restrict a,
    const uint8_t   *restrict b,
    const SQ8Params *restrict params,
    double                    epsilon);

/**
 * @brief Save quantized dataset buffer and parameters to a binary .sq8 file.
 */
int sq8_save_sidecar(
    const char      *filepath,
    const SQ8Params *params,
    const uint8_t   *data,
    long             num_frames);

/**
 * @brief Load quantized dataset buffer and parameters from a binary .sq8 file.
 */
int sq8_load_sidecar(
    const char *filepath,
    SQ8Params  *params,
    uint8_t   **data,
    long       *num_frames);

/** Magic identifier for .sq16 sidecar files */
#define SQ16_FILE_MAGIC "SQ16_0001"

/**
 * @brief Parameters defining uniform 16-bit scalar quantization into [0, 32767].
 */
typedef struct
{
    float min_val;    /**< Minimum value mapped to 0 */
    float max_val;    /**< Maximum value mapped to 32767 */
    float scale;      /**< Step delta: (max_val - min_val) / 32767.0f */
    float inv_scale;  /**< Reciprocal step: 32767.0f / (max_val - min_val) */
    float err_radius; /**< Max single-vector Euclidean error: sqrt(dim) * scale * 0.5f */
    long  dim;        /**< Vector dimension (element count per frame) */
} SQ16Params;

/**
 * @brief Initialize SQ16 parameters from known min and max values.
 */
void sq16_init_params(
    SQ16Params *params,
    float       min_val,
    float       max_val,
    long        dim);

/**
 * @brief Calibrate SQ16 parameters by scanning a float array.
 */
void sq16_calibrate_float(
    SQ16Params  *params,
    const float *data,
    long         num_elements,
    long         dim);

/**
 * @brief Calibrate SQ16 parameters by scanning a double array.
 */
void sq16_calibrate_double(
    SQ16Params   *params,
    const double *data,
    long          num_elements,
    long          dim);

/**
 * @brief Quantize a single-precision float frame to 16-bit signed integers in [0, 32767].
 */
void sq16_quantize_float(
    const float      *restrict src,
    int16_t          *restrict dst,
    const SQ16Params *restrict params);

/**
 * @brief Quantize a double-precision frame to 16-bit signed integers in [0, 32767].
 */
void sq16_quantize_double(
    const double     *restrict src,
    int16_t          *restrict dst,
    const SQ16Params *restrict params);

/**
 * @brief Compute sum of squared differences between two int16 vectors using SIMD.
 */
uint64_t sq16_dist_squared_i16(
    const int16_t *restrict a,
    const int16_t *restrict b,
    long                    dim);

/**
 * @brief Compute guaranteed metric lower bound between two 16-bit quantized vectors.
 */
double sq16_compute_lower_bound(
    const int16_t    *restrict a,
    const int16_t    *restrict b,
    const SQ16Params *restrict params,
    double                     epsilon);

/**
 * @brief Save quantized dataset buffer and parameters to a binary .sq16 file.
 */
int sq16_save_sidecar(
    const char       *filepath,
    const SQ16Params *params,
    const int16_t    *data,
    long              num_frames);

/**
 * @brief Load quantized dataset buffer and parameters from a binary .sq16 file.
 */
int sq16_load_sidecar(
    const char  *filepath,
    SQ16Params  *params,
    int16_t    **data,
    long        *num_frames);

#endif // SCALAR_QUANT_H
