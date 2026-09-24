/**
 * @file gric_simd.h
 * @brief Runtime SIMD capability detection, target attributes, and dispatch control.
 */

#ifndef GRIC_SIMD_H
#define GRIC_SIMD_H

#include <stdbool.h>
#include "gric_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
#define GRIC_TARGET_AVX512      __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
#define GRIC_TARGET_AVX512_VNNI \
    __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,avx512vnni")))
#define GRIC_TARGET_AVX2        __attribute__((target("avx2,fma")))
#define GRIC_TARGET_AVX_VNNI    __attribute__((target("avx2,fma,avxvnni")))
#define GRIC_HAVE_AVX512_TARGET 1
#else
#define GRIC_TARGET_AVX512
#define GRIC_TARGET_AVX512_VNNI
#define GRIC_TARGET_AVX2
#define GRIC_TARGET_AVX_VNNI
#define GRIC_HAVE_AVX512_TARGET 0
#endif

/**
 * @enum GricSimdLevel
 * @brief SIMD instruction set capability levels.
 */
typedef enum
{
    GRIC_SIMD_SCALAR = 0,
    GRIC_SIMD_AVX2   = 1,
    GRIC_SIMD_AVX512 = 2
} GricSimdLevel;

/**
 * gric_get_simd_level() - Query effective SIMD capability level.
 *
 * Checks GRIC_SIMD_MODE environment variable, then hardware CPUID.
 *
 * Return: Selected GricSimdLevel.
 */
GricSimdLevel gric_get_simd_level(void);

/**
 * gric_set_simd_level() - Programmatically override SIMD capability level.
 * @level: Desired SIMD level (-1 to restore automatic detection).
 */
void gric_set_simd_level(
    int level);

/**
 * gric_simd_level_to_string() - Get human-readable description of SIMD level.
 * @level: SIMD level enum value.
 *
 * Return: Static string describing SIMD level.
 */
const char *gric_simd_level_to_string(
    GricSimdLevel level);

/**
 * gric_has_avx512_vnni() - Check if host CPU supports AVX-512 VNNI.
 *
 * Return: true if AVX-512 VNNI is supported, false otherwise.
 */
bool gric_has_avx512_vnni(void);

/**
 * gric_has_avx_vnni() - Check if host CPU supports AVX-VNNI.
 *
 * Return: true if AVX-VNNI is supported, false otherwise.
 */
bool gric_has_avx_vnni(void);

#ifdef __cplusplus
}
#endif

#endif /* GRIC_SIMD_H */
