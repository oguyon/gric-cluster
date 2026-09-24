/**
 * @file knn_heap.c
 * @brief SIMD Bitonic Top-k Candidate Tracker & Bounded Max-Heap Implementation.
 *
 * Implements bounded top-k neighbor tracking using dual execution paths: AVX2 vectorized
 * sorted arrays for low k (k <= 64) with SIMD duplicate detection and bitmask insertion,
 * and a standard binary max-heap with logarithmic sift-up, sift-down, and heapsort extraction
 * for arbitrary k.
 */

#include "knn_heap.h"
#include "gric_compat.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
/**
 * simd_bitonic_cas() - Conditional compare-and-swap of 8 distance and ID lanes
 * @dist_a: Pointer to first distance vector.
 * @dist_b: Pointer to second distance vector.
 * @id_a:   Pointer to first ID vector.
 * @id_b:   Pointer to second ID vector.
 *
 * Compares two 8-element floating-point distance registers (dist_a and dist_b).
 * Computes element-wise minimum into dist_a and element-wise maximum into dist_b.
 * Permutes corresponding integer neighbor IDs (id_a and id_b) using the exact same
 * comparison bitmask, maintaining invariant mapping between candidates and distances.
 */
static inline void simd_bitonic_cas(
    __m256  *restrict dist_a,
    __m256  *restrict dist_b,
    __m256i *restrict id_a,
    __m256i *restrict id_b)
{
    __m256 da = *dist_a;
    __m256 db = *dist_b;

    __m256 cmp = _mm256_cmp_ps(da, db, _CMP_GT_OQ);
    __m256 min_d = _mm256_blendv_ps(da, db, cmp);
    __m256 max_d = _mm256_blendv_ps(db, da, cmp);

    __m256 ia = _mm256_castsi256_ps(*id_a);
    __m256 ib = _mm256_castsi256_ps(*id_b);
    __m256 min_i = _mm256_blendv_ps(ia, ib, cmp);
    __m256 max_i = _mm256_blendv_ps(ib, ia, cmp);

    *dist_a = min_d;
    *dist_b = max_d;
    *id_a = _mm256_castps_si256(min_i);
    *id_b = _mm256_castps_si256(max_i);
}
#endif // __AVX2__

/**
 * knn_heap_init() - Allocate and initialize a bounded heap / SIMD candidate tracker
 * @heap: Pointer to the KnnMaxHeap structure to initialize.
 * @k:    Capacity of the heap (number of nearest neighbors requested).
 *
 * For small neighbor counts (k <= KNN_SIMD_HEAP_MAX, i.e., k <= 64), initializes a
 * contiguous SIMD array padded up to 16, 32, or 64 elements with sentinel distance
 * tau = 1e30, avoiding dynamic heap allocation. For larger k, dynamically allocates
 * an array of KnnNeighbor structures for a standard binary max-heap.
 *
 * Return: 0 on success, or -1 on allocation failure or invalid parameters.
 */
int knn_heap_init(
    KnnMaxHeap *heap,
    int         k)
{
    if (heap == NULL || k <= 0)
    {
        return -1;
    }

    heap->k = k;
    heap->count = 0;

    if (k <= KNN_SIMD_HEAP_MAX)
    {
        int cap = 16;
        if (k > 32)
        {
            cap = 64;
        }
        else if (k > 16)
        {
            cap = 32;
        }
        heap->capacity = cap;
        heap->tau = 1e30f;
        heap->data = NULL;

        for (int i = 0; i < cap; i++)
        {
            heap->simd_dist[i] = 1e30f;
            heap->simd_id[i] = -1;
        }

        return 0;
    }

    heap->capacity = 0;
    heap->tau = 1e30f;
    heap->data = (KnnNeighbor *)malloc((size_t)k * sizeof(KnnNeighbor));
    if (heap->data == NULL)
    {
        return -1;
    }

    return 0;
}

/**
 * knn_heap_free() - Free heap resources
 * @heap: Pointer to the KnnMaxHeap structure.
 *
 * Releases dynamically allocated binary heap buffer if present, and zeroes out
 * heap capacity, element count, and limit k.
 */
void knn_heap_free(
    KnnMaxHeap *heap)
{
    if (heap != NULL)
    {
        if (heap->data != NULL)
        {
            free(heap->data);
            heap->data = NULL;
        }
        heap->count = 0;
        heap->k = 0;
        heap->capacity = 0;
    }
}

/**
 * knn_heap_reset() - Reset count to 0 for reuse in next query
 * @heap: Pointer to the KnnMaxHeap structure.
 *
 * Resets candidate count to 0 and reinitializes SIMD registers with sentinel
 * values (-1 ID, 1e30 distance) and threshold tau to infinity. Enables per-query
 * reuse without heap reallocation.
 */
void knn_heap_reset(
    KnnMaxHeap *heap)
{
    if (heap == NULL)
    {
        return;
    }

    heap->count = 0;
    if (heap->capacity > 0)
    {
        heap->tau = 1e30f;
        for (int i = 0; i < heap->capacity; i++)
        {
            heap->simd_dist[i] = 1e30f;
            heap->simd_id[i] = -1;
        }
    }
}

/**
 * knn_heap_contains() - Check if a frame_id is already in the heap
 * @heap:     Pointer to the KnnMaxHeap structure.
 * @frame_id: Frame index to look for.
 *
 * When using SIMD storage, broadcasts frame_id into an AVX2 256-bit integer register
 * and performs vectorized equality comparison across all active slots. For binary
 * heap fallback, performs a linear scan over registered neighbors.
 *
 * Return: 1 if present, 0 otherwise.
 */
int knn_heap_contains(
    const KnnMaxHeap *heap,
    int               frame_id)
{
    if (heap == NULL || heap->count == 0)
    {
        return 0;
    }

    if (heap->capacity > 0)
    {
#if defined(__AVX2__)
        __m256i v_id = _mm256_set1_epi32(frame_id);
        __m256i match = _mm256_setzero_si256();
        const __m256i *ids = (const __m256i *)heap->simd_id;
        int num_regs = heap->capacity / 8;

        for (int r = 0; r < num_regs; r++)
        {
            __m256i eq = _mm256_cmpeq_epi32(_mm256_load_si256(&ids[r]), v_id);
            match = _mm256_or_si256(match, eq);
        }

        return !_mm256_testz_si256(match, match);
#else
        for (int i = 0; i < heap->count; i++)
        {
            if (heap->simd_id[i] == frame_id)
            {
                return 1;
            }
        }
        return 0;
#endif
    }

    for (int i = 0; i < heap->count; i++)
    {
        if (heap->data[i].frame_id == frame_id)
        {
            return 1;
        }
    }

    return 0;
}

/**
 * knn_heap_push() - Insert a neighbor candidate into the bounded heap / tracker
 * @heap:     Pointer to the KnnMaxHeap structure.
 * @frame_id: Candidate frame index.
 * @dist:     Computed distance between query and candidate.
 *
 * Evaluates candidate against current kth-neighbor distance threshold tau:
 * - If heap is full and dist >= tau, candidate is immediately pruned.
 * - In SIMD mode: rejects duplicates via vectorized equality check, finds sorted
 *   insertion index via AVX2 comparison and ctzll bitmask, shifts trailing elements
 *   using memmove, inserts candidate, and updates tau to simd_dist[k - 1].
 * - In binary max-heap mode: rejects duplicates, appends candidate with sift-up
 *   if count < k, or replaces root element (current maximum) and performs sift-down
 *   if dist < root->dist.
 */
void knn_heap_push(
    KnnMaxHeap *heap,
    int         frame_id,
    double      dist)
{
    if (heap == NULL || heap->k <= 0)
    {
        return;
    }

    if (heap->capacity > 0)
    {
        float fdist = (float)dist;
        if (heap->count >= heap->k && fdist >= heap->tau)
        {
            return;
        }

#if defined(__AVX2__)
        __m256i v_id = _mm256_set1_epi32(frame_id);
        __m256i match = _mm256_setzero_si256();
        const __m256i *ids = (const __m256i *)heap->simd_id;
        int num_regs = heap->capacity / 8;

        for (int r = 0; r < num_regs; r++)
        {
            __m256i eq = _mm256_cmpeq_epi32(_mm256_load_si256(&ids[r]), v_id);
            match = _mm256_or_si256(match, eq);
        }

        if (!_mm256_testz_si256(match, match))
        {
            return;
        }

        __m256 vd = _mm256_set1_ps(fdist);
        const __m256 *dists = (const __m256 *)heap->simd_dist;
        uint64_t full_mask = 0;

        for (int r = 0; r < num_regs; r++)
        {
            __m256 cmp = _mm256_cmp_ps(vd, _mm256_load_ps((const float *)&dists[r]), _CMP_LT_OQ);
            uint32_t m = (uint32_t)_mm256_movemask_ps(cmp);
            full_mask |= ((uint64_t)m << (r * 8));
        }

        int pos = full_mask ? gric_ctz64(full_mask) : (heap->capacity - 1);
#else
        for (int i = 0; i < heap->count; i++)
        {
            if (heap->simd_id[i] == frame_id)
            {
                return;
            }
        }

        int pos = heap->count;
        for (int i = 0; i < heap->count; i++)
        {
            if (fdist < heap->simd_dist[i])
            {
                pos = i;
                break;
            }
        }
#endif

        if (pos >= heap->k && heap->count >= heap->k)
        {
            return;
        }

        int cap = heap->capacity;
        memmove(&heap->simd_dist[pos + 1], &heap->simd_dist[pos],
                (size_t)(cap - 1 - pos) * sizeof(float));
        memmove(&heap->simd_id[pos + 1], &heap->simd_id[pos],
                (size_t)(cap - 1 - pos) * sizeof(int32_t));

        heap->simd_dist[pos] = fdist;
        heap->simd_id[pos] = frame_id;

        if (heap->count < heap->k)
        {
            heap->count++;
        }
        heap->tau = heap->simd_dist[heap->k - 1];
        return;
    }

    // Binary heap fallback for k > KNN_SIMD_HEAP_MAX
    for (int i = 0; i < heap->count; i++)
    {
        if (heap->data[i].frame_id == frame_id)
        {
            return;
        }
    }

    if (heap->count < heap->k)
    {
        int idx = heap->count;
        heap->data[idx].frame_id = frame_id;
        heap->data[idx].dist = dist;
        heap->count++;

        while (idx > 0)
        {
            int parent = (idx - 1) / 2;
            if (heap->data[idx].dist > heap->data[parent].dist)
            {
                KnnNeighbor tmp = heap->data[idx];
                heap->data[idx] = heap->data[parent];
                heap->data[parent] = tmp;
                idx = parent;
            }
            else
            {
                break;
            }
        } // while (idx > 0)
    }
    else
    {
        if (dist >= heap->data[0].dist)
        {
            return;
        }

        heap->data[0].frame_id = frame_id;
        heap->data[0].dist = dist;

        int idx = 0;
        int n = heap->k;
        while (1)
        {
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;
            int largest = idx;

            if (left < n && heap->data[left].dist > heap->data[largest].dist)
            {
                largest = left;
            }
            if (right < n && heap->data[right].dist > heap->data[largest].dist)
            {
                largest = right;
            }

            if (largest != idx)
            {
                KnnNeighbor tmp = heap->data[idx];
                heap->data[idx] = heap->data[largest];
                heap->data[largest] = tmp;
                idx = largest;
            }
            else
            {
                break;
            }
        } // while (1)
    }
}

/**
 * knn_heap_extract_sorted() - Extract elements in ascending distance order
 * @heap:          Pointer to the KnnMaxHeap structure.
 * @out_indices:   Output array for neighbor frame indices (size k).
 * @out_distances: Output array for neighbor distances (size k).
 * @k:             Requested neighbor capacity.
 *
 * Copies nearest neighbors sorted from closest to farthest into the caller's output
 * arrays. For SIMD storage, elements are already kept in sorted order. For binary
 * max-heap fallback, executes heapsort by repeatedly moving the root maximum to the end
 * and sifting down. Remaining unused slots (when count < k) are padded with -1 and -1.0.
 */
void knn_heap_extract_sorted(
    KnnMaxHeap *heap,
    int        *out_indices,
    double     *out_distances,
    int         k)
{
    if (heap == NULL || out_indices == NULL || out_distances == NULL || k <= 0)
    {
        return;
    }

    if (heap->capacity > 0)
    {
        int valid_count = (heap->count < k) ? heap->count : k;
        for (int i = 0; i < valid_count; i++)
        {
            out_indices[i] = heap->simd_id[i];
            out_distances[i] = (double)heap->simd_dist[i];
        }
        for (int i = valid_count; i < k; i++)
        {
            out_indices[i] = -1;
            out_distances[i] = -1.0;
        }
        return;
    }

    // Binary heap fallback heapsort
    while (heap->count > k)
    {
        heap->data[0] = heap->data[heap->count - 1];
        heap->count--;
        int idx = 0;
        int n = heap->count;
        while (1)
        {
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;
            int largest = idx;

            if (left < n && heap->data[left].dist > heap->data[largest].dist)
            {
                largest = left;
            }
            if (right < n && heap->data[right].dist > heap->data[largest].dist)
            {
                largest = right;
            }

            if (largest != idx)
            {
                KnnNeighbor tmp = heap->data[idx];
                heap->data[idx] = heap->data[largest];
                heap->data[largest] = tmp;
                idx = largest;
            }
            else
            {
                break;
            }
        } // while (1)
    } // while (heap->count > k)

    int count = heap->count;

    for (int i = count - 1; i >= 0; i--)
    {
        out_indices[i] = heap->data[0].frame_id;
        out_distances[i] = heap->data[0].dist;

        if (i > 0)
        {
            heap->data[0] = heap->data[i];
            int idx = 0;
            int n = i;
            while (1)
            {
                int left = 2 * idx + 1;
                int right = 2 * idx + 2;
                int largest = idx;

                if (left < n && heap->data[left].dist > heap->data[largest].dist)
                {
                    largest = left;
                }
                if (right < n && heap->data[right].dist > heap->data[largest].dist)
                {
                    largest = right;
                }

                if (largest != idx)
                {
                    KnnNeighbor tmp = heap->data[idx];
                    heap->data[idx] = heap->data[largest];
                    heap->data[largest] = tmp;
                    idx = largest;
                }
                else
                {
                    break;
                }
            } // while (1)
        }
    } // for (int i = count - 1; ...)

    for (int i = count; i < k; i++)
    {
        out_indices[i] = -1;
        out_distances[i] = -1.0;
    }
}
