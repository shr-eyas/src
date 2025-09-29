import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.array(color_frame.get_data(), dtype=np.uint8, copy=True)
        color_image = np.ascontiguousarray(color_image)

        # Convert only if stream is RGB8. If you enabled BGR8, skip this.
        if color_frame.get_profile().format() == rs.format.rgb8:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Sanity checks to avoid the “Expected Ptr<cv::UMat>” error
        if color_image is None or color_image.size == 0:
            continue

        # Apply colormap on depth image (optional)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        # Stack both images side by side
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.imshow('RealSense', images)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
