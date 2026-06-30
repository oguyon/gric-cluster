---
description: Safety protocols when using ImageStreamIO shared memory streams.
---

# Shared Memory Safety

When using `ImageStreamIO` shared memory buffers for low-latency streaming:

## 1. Testing Cleanup
- Clean up test streams created under `/dev/shm/` immediately after tests finish:
  ```bash
  rm -f /dev/shm/test_*.im.shm
  ```

## 2. Stream Owner Validation
- Check ownership of `/dev/shm/*.im.shm` files to prevent overwriting other users' active streams.
- Ensure that you connect to streams using correct read/write permissions.

## 3. Synchronization
- Always use the semaphore protocol (`ImageStreamIO_sempost`, `ImageStreamIO_semwait`) to block until frames are ready, avoiding busy-waiting spin loops.
