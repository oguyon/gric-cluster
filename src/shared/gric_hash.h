#ifndef GRIC_HASH_H
#define GRIC_HASH_H

/**
 * @file gric_hash.h
 * @brief Ultra-fast non-cryptographic 64-bit hash for quantized int16 vectors.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/** Prime constants for 64-bit bit-mixing */
#define GRIC_HASH_P1 0xbf58476d1ce4e5b9ULL
#define GRIC_HASH_P2 0x94d049bb133111ebULL
#define GRIC_HASH_P3 0x9e3779b97f4a7c15ULL
#define GRIC_HASH_SEED 0xa0761d6478bd642fULL

/**
 * @brief 64-bit multiplication and folding helper.
 *
 * @param a First 64-bit operand.
 * @param b Second 64-bit operand.
 * @return Folded 64-bit product.
 */
static inline uint64_t gric_hash_mum(
    uint64_t a,
    uint64_t b)
{
#if defined(__SIZEOF_INT128__)
    __uint128_t r = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t ha = a >> 32;
    uint64_t la = (uint32_t)a;
    uint64_t hb = b >> 32;
    uint64_t lb = (uint32_t)b;
    uint64_t rh = ha * hb;
    uint64_t rm0 = ha * lb;
    uint64_t rm1 = hb * la;
    uint64_t rl = la * lb;
    uint64_t t = rl + (rm0 << 32);
    uint64_t c = (t < rl);
    uint64_t lo = t + (rm1 << 32);
    c += (lo < t);
    uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
    return lo ^ hi;
#endif
}

/**
 * @brief Read unaligned 64-bit integer in native endianness.
 *
 * @param p Pointer to memory.
 * @return 64-bit value.
 */
static inline uint64_t gric_hash_read64(
    const void *p)
{
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/**
 * @brief Read unaligned 32-bit integer in native endianness.
 *
 * @param p Pointer to memory.
 * @return 32-bit value.
 */
static inline uint32_t gric_hash_read32(
    const void *p)
{
    uint32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/**
 * @brief Compute 64-bit hash of an arbitrary byte buffer.
 *
 * @param data Pointer to input bytes.
 * @param len  Length in bytes.
 * @param seed Optional 64-bit seed.
 * @return Computed 64-bit hash value.
 */
static inline uint64_t gric_hash_bytes(
    const void *data,
    size_t      len,
    uint64_t    seed)
{
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = seed ^ (len * GRIC_HASH_P1);

    while (len >= 32)
    {
        uint64_t v0 = gric_hash_read64(p);
        uint64_t v1 = gric_hash_read64(p + 8);
        uint64_t v2 = gric_hash_read64(p + 16);
        uint64_t v3 = gric_hash_read64(p + 24);

        h ^= gric_hash_mum(v0 ^ GRIC_HASH_P2, v1 ^ h);
        h ^= gric_hash_mum(v2 ^ GRIC_HASH_P3, v3 ^ GRIC_HASH_P1);

        p += 32;
        len -= 32;
    } // while (len >= 32)

    while (len >= 8)
    {
        uint64_t v = gric_hash_read64(p);
        h = gric_hash_mum(h ^ v, GRIC_HASH_P2);
        p += 8;
        len -= 8;
    } // while (len >= 8)

    if (len >= 4)
    {
        uint64_t v = gric_hash_read32(p);
        h = gric_hash_mum(h ^ v, GRIC_HASH_P3);
        p += 4;
        len -= 4;
    } // if (len >= 4)

    if (len > 0)
    {
        uint64_t tail = 0;
        for (size_t i = 0; i < len; i++)
        {
            tail |= ((uint64_t)p[i]) << (i * 8);
        }
        h = gric_hash_mum(h ^ tail, GRIC_HASH_P1);
    } // if (len > 0)

    return gric_hash_mum(h, h ^ GRIC_HASH_P2);
}

/**
 * @brief Compute 64-bit hash of an int16 vector.
 *
 * @param data Pointer to int16 vector.
 * @param dim  Number of int16 elements.
 * @return 64-bit hash.
 */
static inline uint64_t gric_hash_i16(
    const int16_t *data,
    long           dim)
{
    if (data == NULL || dim <= 0)
    {
        return 0;
    }

    size_t byte_len = (size_t)dim * sizeof(int16_t);
    return gric_hash_bytes(data, byte_len, GRIC_HASH_SEED);
}

#endif // GRIC_HASH_H
