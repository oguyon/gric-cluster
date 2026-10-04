---
name: milk_realtime_streaming
title: Real-Time Milk Stream Clustering
---
# Recipe: Real-Time Stream Clustering with Milk Framework

1. Inspect input stream status in shared memory:
   Call gric_probe_shm(stream_name="cam_wfs")

2. Launch standalone FPS clustering daemon:
   Call gric_fps_run(fps_name="gric01", in_name="cam_wfs", out_name="clust_wfs", rlim=0.35)

3. Monitor live loop rate and clustering telemetry:
   Call gric_fps_status(fps_name="gric01")
   Call gric_probe_fps_streams(out_name="clust_wfs_assign")

4. Dynamically adjust parameters without stopping the stream:
   Call gric_fps_set(fps_name="gric01", param=".rlim", value="0.42")
   Call gric_fps_set(fps_name="gric01", param=".reset_state", value="ON")

5. Stop stream processing cleanly:
   Call gric_fps_stop(fps_name="gric01")
