/**
 * @file knn_heap.c
 * @brief Bounded binary max-heap implementation for k-nearest neighbors tracking.
 */

#include "knn_heap.h"
#include <math.h>
#include <stdlib.h>

/**
 * knn_heap_init() - Allocate and initialize a bounded max-heap of capacity k.
 * @heap: Pointer to the KnnMaxHeap structure.
 * @k:    Capacity of the heap (number of nearest neighbors).
 *
 * Return: 0 on success, -1 on allocation failure.
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
    heap->data = (KnnNeighbor *)malloc((size_t)k * sizeof(KnnNeighbor));
    if (heap->data == NULL)
    {
        return -1;
    }

    return 0;
}

/**
 * knn_heap_free() - Free heap resources.
 * @heap: Pointer to the KnnMaxHeap structure.
 */
void knn_heap_free(
    KnnMaxHeap *heap)
{
    if (heap != NULL && heap->data != NULL)
    {
        free(heap->data);
        heap->data = NULL;
        heap->count = 0;
        heap->k = 0;
    }
}

/**
 * knn_heap_reset() - Reset count to 0 for reuse in next query.
 * @heap: Pointer to the KnnMaxHeap structure.
 */
void knn_heap_reset(
    KnnMaxHeap *heap)
{
    if (heap != NULL)
    {
        heap->count = 0;
    }
}

/**
 * knn_heap_peek_max_dist() - Return maximum distance currently in heap.
 * @heap: Pointer to the KnnMaxHeap structure.
 *
 * Return: The largest distance in the heap if full, or 1e30 if not full.
 */
double knn_heap_peek_max_dist(
    const KnnMaxHeap *heap)
{
    if (heap == NULL || heap->count < heap->k)
    {
        return 1e30;
    }

    return heap->data[0].dist;
}

/**
 * knn_heap_contains() - Check if a frame_id is already in the heap.
 * @heap:     Pointer to the KnnMaxHeap structure.
 * @frame_id: Frame index to look for.
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
 * knn_heap_push() - Insert a neighbor candidate into the bounded max-heap.
 * @heap:     Pointer to the KnnMaxHeap structure.
 * @frame_id: Candidate frame index.
 * @dist:     Computed distance between query and candidate.
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

    // Check if frame_id is already in heap
    for (int i = 0; i < heap->count; i++)
    {
        if (heap->data[i].frame_id == frame_id)
        {
            // Already present: ignore duplicate
            return;
        }
    }

    if (heap->count < heap->k)
    {
        int idx = heap->count;
        heap->data[idx].frame_id = frame_id;
        heap->data[idx].dist = dist;
        heap->count++;

        // Sift-up
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
        // Heap is full: only insert if distance is smaller than current max
        if (dist >= heap->data[0].dist)
        {
            return;
        }

        heap->data[0].frame_id = frame_id;
        heap->data[0].dist = dist;

        // Sift-down
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
 * knn_heap_extract_sorted() - Extract elements in ascending distance order.
 * @heap:          Pointer to the KnnMaxHeap structure.
 * @out_indices:   Output array for neighbor frame indices (size k).
 * @out_distances: Output array for neighbor distances (size k).
 * @k:             Requested neighbor capacity.
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

    int count = heap->count;
    if (count > k)
    {
        count = k;
    }

    // Repeatedly extract max and place from back to front of valid count
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

    // Pad any remaining underfilled slots with -1 and NAN
    for (int i = count; i < k; i++)
    {
        out_indices[i] = -1;
        out_distances[i] = NAN;
    }
}
