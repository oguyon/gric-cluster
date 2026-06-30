---
name: imagestream-internals
description: Reference for ImageStreamIO shared memory layout and semaphore synchronization.
---

# ImageStreamIO Internals

`ImageStreamIO` provides low-latency shared-memory data mapping and semaphore signaling.

## 1. Memory Mapping (`/dev/shm/`)
Each stream is backed by a shared memory file at `/dev/shm/<name>.im.shm`. It contains:
- `IMAGE_METADATA`: Size dimensions, data type, write counters.
- Pixel array data.
- Read/Write semaphores (`sem_t`).

## 2. Semaphore Protocol
- The **Writer** updates pixel data, increments `cnt0`, and posts to all semaphores using `ImageStreamIO_sempost(img, -1)`.
- The **Reader** blocks on its assigned semaphore index using `ImageStreamIO_semwait(img, semindex)` until notified of a new frame.

## 3. Circular Buffers
When a stream has 3 axes (naxis = 3), the buffer behaves as a circular ring. The current slice index is written in `md->cnt1` modulo `size[2]`.
