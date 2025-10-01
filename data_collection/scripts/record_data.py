#!/usr/bin/env python3
# coding: utf-8
"""
Data recorder for FR3 (right arm) + two RealSense cameras.
ROS Noetic (Python 3). Logs at 30 Hz to HDF5.
"""

import os
import sys
import threading
import argparse
import time
from collections import deque

import rospy
import numpy as np
import h5py

from sensor_msgs.msg import Image, CameraInfo, JointState
from franka_msgs.msg import FrankaState
import tf2_ros
import tf
from tf.transformations import quaternion_from_matrix, euler_from_matrix, euler_from_quaternion

from cv_bridge import CvBridge

# --------------------
# Helpers
# --------------------

def ros_time_to_float(stamp):
    return stamp.secs + stamp.nsecs * 1e-9

def mat4_from_colmajor_16(O_T_EE):
    # Franka O_T_EE is column-major; reshape then transpose for row-major numpy
    M = np.array(O_T_EE, dtype=np.float64).reshape((4,4), order='F')
    return M

def pose_from_mat4(M):
    # Quaternion XYZW per ROS convention
    q_xyzw = quaternion_from_matrix(M)  # returns (x,y,z,w)
    t = M[0:3, 3]
    return t, q_xyzw

def rpy_from_mat4(M):
    # returns roll, pitch, yaw (radians), XYZ-fixed (ROS default)
    r, p, y = euler_from_matrix(M, axes='sxyz')
    return np.array([r, p, y], dtype=np.float64)

def wrap_angle_diff(a):
    # wrap to [-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

def delta_rpy(prev_rpy, curr_rpy):
    d = curr_rpy - prev_rpy
    return np.array([wrap_angle_diff(x) for x in d], dtype=np.float64)

def ensure_ds(hf, path, shape, dtype, chunks=True, maxshape=None, compression="lzf"):
    if maxshape is None:
        maxshape = (None,) + tuple(shape[1:])
    if path in hf:
        return hf[path]
    return hf.create_dataset(path, shape=shape, maxshape=maxshape, dtype=dtype,
                             chunks=chunks, compression=compression)

# --------------------
# Recorder Node
# --------------------

class Recorder(object):
    def __init__(self, out_path, rate_hz=30.0):
        self.rate = rate_hz
        self.dt = 1.0 / rate_hz
        self.out_path = out_path

        # Buffers
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        self.arm_js = None           # sensor_msgs/JointState (arm)
        self.gripper_js = None       # sensor_msgs/JointState (gripper)
        self.franka_state = None     # franka_msgs/FrankaState

        self.cam_infos = {
            "hand": {"color": None, "depth": None},
            "ground": {"color": None, "depth": None},
        }

        self.images = {
            "hand": {"color": None, "depth": None, "stamp": None},
            "ground": {"color": None, "depth": None, "stamp": None},
        }

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.base_frame = "fr3_right_link0"
        self.hand_cam_frame = "hand_cam_link"
        self.ground_cam_frame = "ground_cam_link"

        # HDF5
        self.hf = h5py.File(self.out_path, "w")
        meta = self.hf.create_group("meta")
        meta.attrs["ros_distro"] = "noetic"
        meta.attrs["start_time_unix"] = time.time()
        meta.attrs["base_frame"] = self.base_frame
        meta.attrs["quat_order"] = "xyzw"  # explicit

        # Will lazy-create extensible datasets on first sample when shapes known
        self.initialized_ds = False
        self.sample_count = 0

        # Previous pose for delta action
        self.prev_rpy = None
        self.prev_t = None

        # Subscribers
        # Right arm state topics
        rospy.Subscriber("/fr3_right/franka_state_controller/franka_states", FrankaState, self.cb_franka_state, queue_size=5)
        rospy.Subscriber("/fr3_right/franka_state_controller/joint_states", JointState, self.cb_arm_js, queue_size=10)
        rospy.Subscriber("/fr3_right/franka_gripper/joint_states", JointState, self.cb_gripper_js, queue_size=10)

        # Cameras: color + depth + camera_info for both
        # Hand cam
        rospy.Subscriber("/hand_cam/color/camera_info", CameraInfo, self.cb_hand_color_info, queue_size=1)
        rospy.Subscriber("/hand_cam/depth/camera_info", CameraInfo, self.cb_hand_depth_info, queue_size=1)
        rospy.Subscriber("/hand_cam/color/image_raw", Image, self.cb_hand_color_img, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/hand_cam/depth/image_rect_raw", Image, self.cb_hand_depth_img, queue_size=1, buff_size=2**24)

        # Ground cam
        rospy.Subscriber("/ground_cam/color/camera_info", CameraInfo, self.cb_ground_color_info, queue_size=1)
        rospy.Subscriber("/ground_cam/depth/camera_info", CameraInfo, self.cb_ground_depth_info, queue_size=1)
        rospy.Subscriber("/ground_cam/color/image_raw", Image, self.cb_ground_color_img, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/ground_cam/depth/image_rect_raw", Image, self.cb_ground_depth_img, queue_size=1, buff_size=2**24)

        # Timer for logging at fixed rate
        self.timer = rospy.Timer(rospy.Duration.from_sec(self.dt), self.tick)

        rospy.loginfo("Recorder initialized. Writing to: %s", self.out_path)

    # ---- Callbacks

    def cb_franka_state(self, msg: FrankaState):
        with self.lock:
            self.franka_state = msg

    def cb_arm_js(self, msg: JointState):
        with self.lock:
            self.arm_js = msg

    def cb_gripper_js(self, msg: JointState):
        with self.lock:
            self.gripper_js = msg

    def cb_hand_color_info(self, msg: CameraInfo):
        with self.lock:
            self.cam_infos["hand"]["color"] = msg

    def cb_hand_depth_info(self, msg: CameraInfo):
        with self.lock:
            self.cam_infos["hand"]["depth"] = msg

    def cb_ground_color_info(self, msg: CameraInfo):
        with self.lock:
            self.cam_infos["ground"]["color"] = msg

    def cb_ground_depth_info(self, msg: CameraInfo):
        with self.lock:
            self.cam_infos["ground"]["depth"] = msg

    def cb_hand_color_img(self, msg: Image):
        with self.lock:
            self.images["hand"]["color"] = msg
            self.images["hand"]["stamp"] = msg.header.stamp

    def cb_hand_depth_img(self, msg: Image):
        with self.lock:
            self.images["hand"]["depth"] = msg
            self.images["hand"]["stamp"] = msg.header.stamp

    def cb_ground_color_img(self, msg: Image):
        with self.lock:
            self.images["ground"]["color"] = msg
            self.images["ground"]["stamp"] = msg.header.stamp

    def cb_ground_depth_img(self, msg: Image):
        with self.lock:
            self.images["ground"]["depth"] = msg
            self.images["ground"]["stamp"] = msg.header.stamp

    # ---- HDF5 schema setup on first sample

    def init_datasets_if_needed(self, hand_color_shape, hand_depth_shape, ground_color_shape, ground_depth_shape):
        if self.initialized_ds:
            return

        # Static camera intrinsics
        cams_grp = self.hf.create_group("cameras")
        for cam in ["hand", "ground"]:
            g = cams_grp.create_group(cam)
            for stream in ["color", "depth"]:
                info = self.cam_infos[cam][stream]
                sg = g.create_group(stream)
                # Intrinsics
                sg.create_dataset("K", data=np.array(info.K, dtype=np.float64).reshape(3,3))
                sg.create_dataset("D", data=np.array(info.D, dtype=np.float64))
                sg.create_dataset("R", data=np.array(info.R, dtype=np.float64).reshape(3,3))
                sg.create_dataset("P", data=np.array(info.P, dtype=np.float64).reshape(3,4))
                sg.attrs["distortion_model"] = info.distortion_model
                sg.attrs["frame_id"] = info.header.frame_id

        # Samples group with extensible datasets
        smp = self.hf.create_group("samples")

        # time
        ensure_ds(self.hf, "samples/time", shape=(0,), dtype='f8')

        # actions
        ensure_ds(self.hf, "samples/actions/ee_delta_rpy", shape=(0,6), dtype='f8')  # [dx,dy,dz,dr,dp,dy]
        ensure_ds(self.hf, "samples/actions/gripper/position", shape=(0,), dtype='f8')
        ensure_ds(self.hf, "samples/actions/gripper/velocity", shape=(0,), dtype='f8')
        ensure_ds(self.hf, "samples/actions/gripper/effort", shape=(0,), dtype='f8')

        # observations: EEF pose + quat (xyzw)
        ensure_ds(self.hf, "samples/observations/ee/position", shape=(0,3), dtype='f8')
        ensure_ds(self.hf, "samples/observations/ee/quat_xyzw", shape=(0,4), dtype='f8')
        # raw O_T_EE (column-major 16)
        ensure_ds(self.hf, "samples/observations/ee/O_T_EE_colmajor16", shape=(0,16), dtype='f8')

        # observations: joints (assume 7 DOF arm)
        ensure_ds(self.hf, "samples/observations/joint/position", shape=(0,7), dtype='f8')
        ensure_ds(self.hf, "samples/observations/joint/velocity", shape=(0,7), dtype='f8')
        ensure_ds(self.hf, "samples/observations/joint/effort", shape=(0,7), dtype='f8')

        # observations: gripper
        ensure_ds(self.hf, "samples/observations/gripper/position", shape=(0,), dtype='f8')
        ensure_ds(self.hf, "samples/observations/gripper/velocity", shape=(0,), dtype='f8')
        ensure_ds(self.hf, "samples/observations/gripper/effort", shape=(0,), dtype='f8')

        # observations: images
        # Store the raw arrays (uint8 color HxWx3, uint16 depth HxW)
        ensure_ds(self.hf, "samples/observations/images/hand/color",
                  shape=(0,) + hand_color_shape, dtype='uint8')
        ensure_ds(self.hf, "samples/observations/images/hand/depth",
                  shape=(0,) + hand_depth_shape, dtype='uint16')
        ensure_ds(self.hf, "samples/observations/images/ground/color",
                  shape=(0,) + ground_color_shape, dtype='uint8')
        ensure_ds(self.hf, "samples/observations/images/ground/depth",
                  shape=(0,) + ground_depth_shape, dtype='uint16')

        # extrinsics per-sample: 4x4 matrices base->cam
        ensure_ds(self.hf, "samples/extrinsics/T_base_handcam", shape=(0,4,4), dtype='f8')
        ensure_ds(self.hf, "samples/extrinsics/T_base_groundcam", shape=(0,4,4), dtype='f8')

        self.initialized_ds = True
        self.hf.flush()

    # ---- Main tick

    def tick(self, event):
        with self.lock:
            arm_js = self.arm_js
            gripper_js = self.gripper_js
            frs = self.franka_state
            img_hand_color = self.images["hand"]["color"]
            img_hand_depth = self.images["hand"]["depth"]
            img_ground_color = self.images["ground"]["color"]
            img_ground_depth = self.images["ground"]["depth"]
            # ensure camera infos are received
            hand_ci_c = self.cam_infos["hand"]["color"]
            hand_ci_d = self.cam_infos["hand"]["depth"]
            ground_ci_c = self.cam_infos["ground"]["color"]
            ground_ci_d = self.cam_infos["ground"]["depth"]

        # Require minimal set before logging
        if not (arm_js and gripper_js and frs and img_hand_color and img_hand_depth and
                img_ground_color and img_ground_depth and hand_ci_c and hand_ci_d and ground_ci_c and ground_ci_d):
            return  # skip until everything is ready

        # Convert images
        try:
            hand_color = self.bridge.imgmsg_to_cv2(img_hand_color, desired_encoding="bgr8")
            # store as RGB for ML consistency
            hand_color = hand_color[:, :, ::-1].copy()  # BGR->RGB
            hand_depth = self.bridge.imgmsg_to_cv2(img_hand_depth, desired_encoding="passthrough").copy()

            ground_color = self.bridge.imgmsg_to_cv2(img_ground_color, desired_encoding="bgr8")
            ground_color = ground_color[:, :, ::-1].copy()
            ground_depth = self.bridge.imgmsg_to_cv2(img_ground_depth, desired_encoding="passthrough").copy()
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"cv_bridge conversion failed: {e}")
            return

        # Init datasets when shapes are known
        self.init_datasets_if_needed(hand_color.shape, hand_depth.shape, ground_color.shape, ground_depth.shape)

        # Extract EEF transforms, joints, actions
        # FrankaState: O_T_EE is column-major 16
        O_T_EE = np.array(frs.O_T_EE, dtype=np.float64)  # keep raw
        M_ee = mat4_from_colmajor_16(O_T_EE)
        ee_t, ee_q_xyzw = pose_from_mat4(M_ee)
        ee_rpy = rpy_from_mat4(M_ee)

        # Action: deltas in base frame
        if self.prev_t is None:
            d_xyz = np.zeros(3, dtype=np.float64)
            d_rpy = np.zeros(3, dtype=np.float64)
        else:
            d_xyz = ee_t - self.prev_t
            d_rpy = delta_rpy(self.prev_rpy, ee_rpy)
        ee_delta_rpy = np.concatenate([d_xyz, d_rpy], axis=0)

        # Update prev
        self.prev_t = ee_t.copy()
        self.prev_rpy = ee_rpy.copy()

        # Joints
        # Arm: expect 7 elements; store as-is, no post-process
        arm_pos = np.array(arm_js.position, dtype=np.float64)
        arm_vel = np.array(arm_js.velocity, dtype=np.float64) if arm_js.velocity else np.zeros_like(arm_pos)
        arm_eff = np.array(arm_js.effort, dtype=np.float64) if arm_js.effort else np.zeros_like(arm_pos)

        # Gripper single joint (width) typically 1 entry
        grip_pos = float(gripper_js.position[0]) if gripper_js.position else 0.0
        grip_vel = float(gripper_js.velocity[0]) if gripper_js.velocity else 0.0
        grip_eff = float(gripper_js.effort[0]) if gripper_js.effort else 0.0

        # Extrinsics base->cam each tick
        def lookup_T(base, child):
            try:
                tr = self.tf_buffer.lookup_transform(base, child, rospy.Time(0), rospy.Duration(0.05))
                t = tr.transform.translation
                q = tr.transform.rotation  # XYZW in ROS
                M = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
                M[0,3], M[1,3], M[2,3] = t.x, t.y, t.z
                return M
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"TF lookup {base}->{child} failed: {e}")
                return None

        T_base_hand = lookup_T(self.base_frame, self.hand_cam_frame)
        T_base_ground = lookup_T(self.base_frame, self.ground_cam_frame)
        if T_base_hand is None or T_base_ground is None:
            return  # skip until TF available

        # Timestamp
        t_ros = frs.header.stamp if frs.header.stamp else rospy.Time.now()
        t_float = ros_time_to_float(t_ros)

        # Append into HDF5
        idx = self.sample_count
        self.sample_count += 1

        def append(ds_path, val):
            ds = self.hf[ds_path]
            ds.resize((idx+1,) + ds.shape[1:])
            ds[idx] = val

        append("samples/time", t_float)
        append("samples/actions/ee_delta_rpy", ee_delta_rpy)
        append("samples/actions/gripper/position", grip_pos)
        append("samples/actions/gripper/velocity", grip_vel)
        append("samples/actions/gripper/effort", grip_eff)

        append("samples/observations/ee/position", ee_t)
        append("samples/observations/ee/quat_xyzw", ee_q_xyzw)
        append("samples/observations/ee/O_T_EE_colmajor16", O_T_EE)

        # arm joints sized to dataset; pad or trim to 7 if needed
        def fit7(x):
            if x.size == 7:
                return x
            y = np.zeros(7, dtype=np.float64)
            n = min(7, x.size)
            y[:n] = x[:n]
            return y

        append("samples/observations/joint/position", fit7(arm_pos))
        append("samples/observations/joint/velocity", fit7(arm_vel))
        append("samples/observations/joint/effort", fit7(arm_eff))

        append("samples/observations/gripper/position", grip_pos)
        append("samples/observations/gripper/velocity", grip_vel)
        append("samples/observations/gripper/effort", grip_eff)

        append("samples/observations/images/hand/color", hand_color)
        append("samples/observations/images/hand/depth", hand_depth.astype(np.uint16))
        append("samples/observations/images/ground/color", ground_color)
        append("samples/observations/images/ground/depth", ground_depth.astype(np.uint16))

        append("samples/extrinsics/T_base_handcam", T_base_hand)
        append("samples/extrinsics/T_base_groundcam", T_base_ground)

        # Periodic flush
        if idx % 30 == 0:
            self.hf.flush()

    def shutdown(self):
        try:
            self.hf["meta"].attrs["end_time_unix"] = time.time()
            self.hf.flush()
            self.hf.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=os.path.join(os.getcwd(), "fr3_right_log.hdf5"),
                        help="Output HDF5 path")
    parser.add_argument("--hz", type=float, default=30.0, help="Logging rate")
    args, _ = parser.parse_known_args()

    rospy.init_node("fr3_right_data_recorder", anonymous=False)
    rec = Recorder(args.out, rate_hz=args.hz)

    def on_shutdown():
        rospy.loginfo("Shutting down recorder")
        rec.shutdown()

    rospy.on_shutdown(on_shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()