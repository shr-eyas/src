import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
import os

# ---- Settings ----
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)
W, H, FPS = 640, 480, 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# ---- Discover devices ----
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) < 2:
    raise RuntimeError("Need at least 2 RealSense devices connected")

# Pick first two devices
serials = [d.get_info(rs.camera_info.serial_number) for d in devices[:2]]
print(serials)

# ---- Create pipelines ----
pipelines = []
for serial in serials:
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    pipelines.append((serial, pipe, profile))

# Recorders
writers = {s: None for s, _, _ in pipelines}
recording = True

try:
    while True:
        for serial, pipe, profile in pipelines:
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_img = np.asanyarray(depth_frame.get_data())
            color_img = np.array(color_frame.get_data(), dtype=np.uint8, copy=True)

            if color_frame.get_profile().format() == rs.format.rgb8:
                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            depth_map = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03),
                cv2.COLORMAP_JET
            )
            combined = np.hstack((color_img, depth_map))

            # Start writers on demand
            if recording and writers[serial] is None:
                h, w = combined.shape[:2]
                fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{serial}.mp4"
                fpath = os.path.join(OUTPUT_DIR, fname)
                writers[serial] = cv2.VideoWriter(fpath, fourcc, FPS, (w, h))

            # Write frames
            if recording and writers[serial] is not None:
                writers[serial].write(combined)
                cv2.putText(combined, f"REC {serial}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(f"Device {serial}", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('s'), ord('S')):
            recording = False
            for s in serials:
                if writers[s] is not None:
                    writers[s].release()
                    writers[s] = None

finally:
    for s, pipe, _ in pipelines:
        if writers[s] is not None:
            writers[s].release()
    for _, pipe, _ in pipelines:
        pipe.stop()
    cv2.destroyAllWindows()
