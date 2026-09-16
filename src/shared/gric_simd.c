/**
 * @file gric_simd.c
 * @brief Implementation of runtime SIMD feature detection and dispatch control.
 */

#include "gric_simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_simd_override = -1;

/**
 * detect_hardware_simd() - Probe CPU architecture flags via compiler builtins.
 *
 * Return: Maximum hardware-supported GricSimdLevel.
 */
static GricSimdLevel detect_hardware_simd(void)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl"))
    {
        return GRIC_SIMD_AVX512;
    }

    if (__builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("fma"))
    {
        return GRIC_SIMD_AVX2;
    }
#endif
#endif
    return GRIC_SIMD_SCALAR;
}

/**
 * gric_get_simd_level() - Query effective SIMD capability level.
 *
 * Checks programmatic override, GRIC_SIMD_MODE environment variable,
 * and falls back to CPU feature probe.
 *
 * Return: Effective GricSimdLevel to use for compute kernels.
 */
GricSimdLevel gric_get_simd_level(void)
{
    if (g_simd_override >= 0)
    {
        return (GricSimdLevel)g_simd_override;
    }

    const char *env = getenv("GRIC_SIMD_MODE");
    if (env != NULL)
    {
        if (strcasecmp(env, "scalar") == 0 || strcmp(env, "0") == 0)
        {
            return GRIC_SIMD_SCALAR;
        }
        if (strcasecmp(env, "avx2") == 0 || strcmp(env, "1") == 0)
        {
            return GRIC_SIMD_AVX2;
        }
        if (strcasecmp(env, "avx512") == 0 || strcmp(env, "2") == 0)
        {
            return GRIC_SIMD_AVX512;
        }
    }

    return detect_hardware_simd();
}

/**
 * gric_set_simd_level() - Programmatically override SIMD capability level.
 * @level: Desired SIMD level (-1 to restore automatic detection).
 */
void gric_set_simd_level(
    int level)
{
    g_simd_override = level;
}

/**
 * gric_simd_level_to_string() - Get human-readable description of SIMD level.
 * @level: SIMD level enum value.
 *
 * Return: Static string describing SIMD level.
 */
const char *gric_simd_level_to_string(
    GricSimdLevel level)
{
    switch (level)
    {
        case GRIC_SIMD_AVX512:
            return "AVX-512 (512-bit ZMM, F/BW/DQ/VL)";
        case GRIC_SIMD_AVX2:
            return "AVX2 (256-bit YMM, FMA)";
        case GRIC_SIMD_SCALAR:
        default:
            return "Scalar / Generic Baseline";
    }
}
