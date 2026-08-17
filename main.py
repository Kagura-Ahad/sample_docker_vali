import python_vali as vali
import numpy as np
from ultralytics.models.yolo.model import YOLO
import torch
import time
import sys

# 1. CONFIGURATION
RTSP_URL = "IMG_4478.MOV"
GPU_ID = 0
RECONNECT_INTERVAL = 5  # Seconds to wait before retrying connection

def main():
    print(f"Initializing connection to: {RTSP_URL}")
    
    # 2. LOAD YOLO MODEL
    print("Loading YOLO-Pose model...")
    model = YOLO("yolo11s-pose_fineTuned1000.pt") 

    opts = {
        # Optional: You can specify RTSP options for FFmpeg here if needed, e.g.:
        # "rtsp_transport": "tcp",
    } 

    # 3. OUTER RECONNECT LOOP
    while True:
        print(f"\nInitializing connection to: {RTSP_URL}")
        
        try:
            nv_dec = vali.PyDecoder(RTSP_URL, opts, GPU_ID)
        except Exception as e:
            print(f"Failed to connect to stream: {e}")
            print(f"Retrying connection in {RECONNECT_INTERVAL} seconds...")
            time.sleep(RECONNECT_INTERVAL)
            continue

        width = nv_dec.Width
        height = nv_dec.Height
        print(f"Stream Resolution: {width}x{height}")

        # 4. INITIALIZE PY_SURFACE_UD (One-shot Resize + Convert)
        # We pass the decoder's stream so they share the same GPU queue
        py_ud = vali.PySurfaceUD(gpu_id=GPU_ID, stream=nv_dec.Stream)

        # 5. PREPARE MEMORY FOR THIS SESSION
        target_w = 640
        target_h = 384  # Matches aspect ratio logic

        # Source Surface (NV12 from Decoder)
        surface_nv12 = vali.Surface.Make(
            vali.PixelFormat.NV12, 
            width, 
            height, 
            GPU_ID
        )
        
        # Destination Surface (RGB_32F_PLANAR)
        # This is the "Magic" format:
        # - It is Resized (by virtue of being created with target_w/h)
        # - It is Float (32F), so it's ready for AI math
        # - It is Planar, so it matches YOLO [C, H, W] layout
        surface_target = vali.Surface.Make(
            vali.PixelFormat.RGB_32F_PLANAR, 
            target_w, 
            target_h, 
            GPU_ID
        )
        
        # Wrap CUDA stream for torch operations
        cuda_stream = torch.cuda.Stream()
        try:
            # Requires Torch 2.9+, but helps synchronization if available
            cuda_stream = torch.cuda.get_stream_from_external(nv_dec.Stream)
            print("Using shared CUDA stream (zero-copy)")
        except AttributeError:
            # Fallback for older Torch versions
            pass

        # 6. INFERENCE LOOP
        frame_count = 0
        start_time = time.time()

        print("Starting Inference Loop...")
        
        while True:
            # A. Decode on GPU
            success, info = nv_dec.DecodeSingleSurfaceAsync(surface_nv12)
            if not success:
                print(f"Frame decode finished or connection lost (Status: {info}).")
                break

            # B. One-Shot: Resize + Convert NV12 -> RGB_32F_PLANAR
            # This performs resize, color conversion, and planar rearrangement in one kernel
            success_ud, info_ud = py_ud.RunAsync(surface_nv12, surface_target)
            if not success_ud:
                print(f"Failed to process frame: {info_ud}")
                break

            # C. Zero-Copy Transfer to PyTorch
            # Note: If your torch version is < 2.9, manual synchronization might be implicitly handled 
            # by the GIL or overhead, but ideally, you'd use the stream context.
            with torch.cuda.stream(cuda_stream):
                # Create tensor sharing memory with VALI surface
                # Because format is RGB_32F_PLANAR, data is already Float 0.0-1.0 (usually) and Planar
                frame_tensor = torch.from_dlpack(surface_target)
                
                # Reshape directly to [Batch, Channels, Height, Width]
                # No permute() needed because it is already PLANAR!
                # No float() division needed because it is already 32F!
                frame_tensor = frame_tensor.reshape(1, 3, target_h, target_w)
                
                # Optional: Clamp to ensure precision safety (as seen in Roman's sample)
                frame_tensor = frame_tensor.clamp_(0.0, 1.0)
                
                # D. YOLO Inference
                results = model.predict(frame_tensor, verbose=False, imgsz=(target_h, target_w))
            
            # E. Print Results
            detections = len(results[0])
            frame_count += 1
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.2f} | Frame: {frame_count} | People Detected: {detections}")

        print(f"Stream interrupted. Attempting reconnect in {RECONNECT_INTERVAL} seconds...")
        time.sleep(RECONNECT_INTERVAL)

if __name__ == "__main__":
    main()
