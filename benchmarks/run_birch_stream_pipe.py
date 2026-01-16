import sys
import time
import numpy as np
from sklearn.cluster import Birch

def run_birch_pipe(width, height, threshold=0.1):
    frame_size = width * height
    frame_bytes = frame_size * 8 # double is 8 bytes
    
    print(f"Running BIRCH (Pipe Mode) {width}x{height}, thresh={threshold}...")
    
    brc = Birch(threshold=threshold, branching_factor=50, n_clusters=None)
    
    start_time = time.time()
    count = 0
    
    try:
        while True:
            # Read one frame
            data = sys.stdin.buffer.read(frame_bytes)
            if not data or len(data) != frame_bytes:
                break
                
            # Convert to numpy array
            frame = np.frombuffer(data, dtype=np.float64)
            
            # Reshape for partial_fit (1 sample, n_features)
            # wait, frames are usually images. Birch treats samples as vectors.
            # If we are clustering *frames* (where each frame is a point in high-dim space),
            # we reshape to (1, width*height).
            # If we are clustering *pixels* (each pixel is a point), we reshape to (width*height, 1) or similar.
            # image-cluster clusters FRAMES.
            
            sample = frame.reshape(1, -1)
            brc.partial_fit(sample)
            count += 1
            
            if count % 100 == 0:
                sys.stderr.write(f"\rProcessed {count} frames")
                
    except KeyboardInterrupt:
        pass
        
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    
    n_clusters = len(brc.subcluster_centers_)
    
    print(f"\nBIRCH (Pipe) Result: Time={duration_ms:.2f}ms, Clusters={n_clusters}")
    print(f"Processed {count} frames.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_birch_stream_pipe.py <width> <height> [threshold]")
        sys.exit(1)

    w = int(sys.argv[1])
    h = int(sys.argv[2])
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    
    run_birch_pipe(w, h, thresh)
