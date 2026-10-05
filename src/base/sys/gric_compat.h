/**
 * @file gric_compat.h
 * @brief Zero-overhead C23 compatibility layer with C17 baseline support.
 *
 * Provides conditional C23 language features, optimization attributes, standardized
 * bit manipulation (<stdbit.h>), and checked integer arithmetic (<stdckdint.h>)
 * while falling back gracefully to compiler builtins or standard C17 on older toolchains.
 */

#ifndef GRIC_COMPAT_H
#define GRIC_COMPAT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Feature Detection Probes
 * ------------------------------------------------------------------------- */
#ifndef __has_include
#  define __has_include(x) 0
#endif

#ifndef __has_attribute
#  define __has_attribute(x) 0
#endif

#ifndef __has_c_attribute
#  define __has_c_attribute(x) 0
#endif

/* -------------------------------------------------------------------------
 * Optimization & Purity Attributes
 * ------------------------------------------------------------------------- */

/*
 * GRIC_ATTR_CONST: Declares a function whose return value depends exclusively on its
 * scalar arguments, reading no global or pointed-to state and causing no side effects.
 * Allows loop hoisting, dead-code elimination, and vectorization.
 */
#if defined(__GNUC__) || defined(__clang__)
#  define GRIC_ATTR_CONST __attribute__((const))
#else
#  define GRIC_ATTR_CONST
#endif

/*
 * GRIC_ATTR_PURE: Declares a function that only reads memory (e.g. via pointers)
 * and has no side effects. Enables Common Subexpression Elimination (CSE).
 */
#if defined(__GNUC__) || defined(__clang__)
#  define GRIC_ATTR_PURE __attribute__((pure))
#else
#  define GRIC_ATTR_PURE
#endif

/*
 * GRIC_ATTR_NODISCARD: Warns if the caller discards the return value.
 */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 202311L)
#  define GRIC_ATTR_NODISCARD [[nodiscard]]
#elif defined(__has_c_attribute) && __has_c_attribute(nodiscard)
#  define GRIC_ATTR_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#  define GRIC_ATTR_NODISCARD __attribute__((warn_unused_result))
#else
#  define GRIC_ATTR_NODISCARD
#endif

/*
 * GRIC_ATTR_MAYBE_UNUSED: Suppresses unused variable or function warnings.
 */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 202311L)
#  define GRIC_ATTR_MAYBE_UNUSED [[maybe_unused]]
#elif defined(__has_c_attribute) && __has_c_attribute(maybe_unused)
#  define GRIC_ATTR_MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__)
#  define GRIC_ATTR_MAYBE_UNUSED __attribute__((unused))
#else
#  define GRIC_ATTR_MAYBE_UNUSED
#endif

/*
 * Inlining and hot/cold code path hints.
 */
#if defined(__GNUC__) || defined(__clang__)
#  define GRIC_ATTR_ALWAYS_INLINE inline __attribute__((always_inline))
#  define GRIC_ATTR_HOT           __attribute__((hot))
#  define GRIC_ATTR_COLD          __attribute__((cold))
#else
#  define GRIC_ATTR_ALWAYS_INLINE inline
#  define GRIC_ATTR_HOT
#  define GRIC_ATTR_COLD
#endif

/*
 * Pointer restrict qualifier.
 */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#  define GRIC_RESTRICT restrict
#elif defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#  define GRIC_RESTRICT __restrict
#else
#  define GRIC_RESTRICT
#endif

/*
 * Data alignment.
 */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 202311L)
#  define GRIC_ALIGNED(n) alignas(n)
#elif defined(__GNUC__) || defined(__clang__)
#  define GRIC_ALIGNED(n) __attribute__((aligned(n)))
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#  define GRIC_ALIGNED(n) _Alignas(n)
#else
#  define GRIC_ALIGNED(n)
#endif

/*
 * Static assertion.
 */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 202311L)
#  define GRIC_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#  define GRIC_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#  define GRIC_STATIC_ASSERT(cond, msg)
#endif

/* -------------------------------------------------------------------------
 * Checked Integer Arithmetic (<stdckdint.h> vs GCC/Clang builtins)
 * ------------------------------------------------------------------------- */
#if defined(__has_include)
#  if __has_include(<stdckdint.h>)
#    include <stdckdint.h>
#    define GRIC_HAVE_STDCKDINT 1
#  endif
#endif

#if defined(GRIC_HAVE_STDCKDINT)
#  define gric_ckd_mul(res, a, b) ckd_mul((res), (a), (b))
#  define gric_ckd_add(res, a, b) ckd_add((res), (a), (b))
#  define gric_ckd_sub(res, a, b) ckd_sub((res), (a), (b))
#elif defined(__GNUC__) || defined(__clang__)
#  define gric_ckd_mul(res, a, b) __builtin_mul_overflow((a), (b), (res))
#  define gric_ckd_add(res, a, b) __builtin_add_overflow((a), (b), (res))
#  define gric_ckd_sub(res, a, b) __builtin_sub_overflow((a), (b), (res))
#endif

/* -------------------------------------------------------------------------
 * Standard Bit Manipulation (<stdbit.h> vs Builtins)
 * ------------------------------------------------------------------------- */
#if defined(__has_include)
#  if __has_include(<stdbit.h>)
#    include <stdbit.h>
#    define GRIC_HAVE_STDBIT 1
#  endif
#endif

/**
 * gric_popcount32() - Count set bits in a 32-bit unsigned integer.
 * @val: 32-bit input word.
 *
 * Return: Number of bits set to 1 [0..32].
 */
static GRIC_ATTR_ALWAYS_INLINE GRIC_ATTR_CONST int gric_popcount32(
    uint32_t val)
{
#if defined(GRIC_HAVE_STDBIT)
    return (int)stdc_count_ones(val);
#elif defined(_MSC_VER)
    return (int)__popcnt(val);
#else
    return __builtin_popcount(val);
#endif
}

/**
 * gric_popcount64() - Count set bits in a 64-bit unsigned integer.
 * @val: 64-bit input word.
 *
 * Return: Number of bits set to 1 [0..64].
 */
static GRIC_ATTR_ALWAYS_INLINE GRIC_ATTR_CONST int gric_popcount64(
    uint64_t val)
{
#if defined(GRIC_HAVE_STDBIT)
    return (int)stdc_count_ones(val);
#elif defined(_MSC_VER)
    return (int)__popcnt64(val);
#else
    return __builtin_popcountll((unsigned long long)val);
#endif
}

/**
 * gric_ctz32() - Count trailing zero bits in a 32-bit unsigned integer.
 * @val: 32-bit input word.
 *
 * Well-defined for zero: returns 32 when val == 0.
 *
 * Return: Number of trailing zeros [0..32].
 */
static GRIC_ATTR_ALWAYS_INLINE GRIC_ATTR_CONST int gric_ctz32(
    uint32_t val)
{
#if defined(GRIC_HAVE_STDBIT)
    return (int)stdc_trailing_zeros(val);
#else
    if (val == 0)
    {
        return 32;
    }
#  if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward(&index, val);
    return (int)index;
#  else
    return __builtin_ctz(val);
#  endif
#endif
}

/**
 * gric_ctz64() - Count trailing zero bits in a 64-bit unsigned integer.
 * @val: 64-bit input word.
 *
 * Well-defined for zero: returns 64 when val == 0.
 *
 * Return: Number of trailing zeros [0..64].
 */
static GRIC_ATTR_ALWAYS_INLINE GRIC_ATTR_CONST int gric_ctz64(
    uint64_t val)
{
#if defined(GRIC_HAVE_STDBIT)
    return (int)stdc_trailing_zeros(val);
#else
    if (val == 0)
    {
        return 64;
    }
#  if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward64(&index, val);
    return (int)index;
#  else
    return __builtin_ctzll((unsigned long long)val);
#  endif
#endif
}

/**
 * gric_clz32() - Count leading zero bits in a 32-bit unsigned integer.
 * @val: 32-bit input word.
 *
 * Well-defined for zero: returns 32 when val == 0.
 *
 * Return: Number of leading zeros [0..32].
 */
static GRIC_ATTR_ALWAYS_INLINE GRIC_ATTR_CONST int gric_clz32(
    uint32_t val)
{
#if defined(GRIC_HAVE_STDBIT)
    return (int)stdc_leading_zeros(val);
#else
    if (val == 0)
    {
        return 32;
    }
#  if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanReverse(&index, val);
    return (int)(31 - index);
#  else
    return __builtin_clz(val);
#  endif
#endif
}

/**
 * gric_clz64() - Count leading zero bits in a 64-bit unsigned integer.
 * @val: 64-bit input word.
 *
 * Well-defined for zero: returns 64 when val == 0.
 *
 * Return: Number of leading zeros [0..64].
 */
static GRIC_ATTR_ALWAYS_INLINE GRIC_ATTR_CONST int gric_clz64(
    uint64_t val)
{
#if defined(GRIC_HAVE_STDBIT)
    return (int)stdc_leading_zeros(val);
#else
    if (val == 0)
    {
        return 64;
    }
#  if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanReverse64(&index, val);
    return (int)(63 - index);
#  else
    return __builtin_clzll((unsigned long long)val);
#  endif
#endif
}

/**
 * gric_has_single_bit32() - Check if a 32-bit integer is a positive power of two.
 * @val: 32-bit input word.
 *
 * Return: true if val has exactly one bit set, false otherwise.
 */
static GRIC_ATTR_ALWAYS_INLINE GRIC_ATTR_CONST bool gric_has_single_bit32(
    uint32_t val)
{
#if defined(GRIC_HAVE_STDBIT)
    return stdc_has_single_bit(val);
#else
    return val != 0 && (val & (val - 1)) == 0;
#endif
}

/**
 * gric_has_single_bit64() - Check if a 64-bit integer is a positive power of two.
 * @val: 64-bit input word.
 *
 * Return: true if val has exactly one bit set, false otherwise.
 */
static GRIC_ATTR_ALWAYS_INLINE GRIC_ATTR_CONST bool gric_has_single_bit64(
    uint64_t val)
{
#if defined(GRIC_HAVE_STDBIT)
    return stdc_has_single_bit(val);
#else
    return val != 0 && (val & (val - 1)) == 0;
#endif
}

#ifdef __cplusplus
}
#endif

#endif /* GRIC_COMPAT_H */
