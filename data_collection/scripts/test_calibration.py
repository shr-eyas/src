#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf2_ros
from tf2_ros import TransformListener, Buffer, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from apriltag_ros.msg import AprilTagDetectionArray
from tf.transformations import (
    quaternion_matrix, translation_matrix, inverse_matrix,
    translation_from_matrix, quaternion_from_matrix
)

# ---- names you already use ----
HAND_TOPIC = "/hand_tags/tag_detections"
GROUND_TOPIC = "/ground_tags/tag_detections"
HAND_OPTICAL = "hand_cam_color_optical_frame"
GROUND_OPTICAL = "ground_cam_color_optical_frame"
GROUND_LINK = "ground_cam_link"
BASE = "fr3_link0"
TAG_ID = 0

# store latest detections
D_hand = None   # T_{hand_optical<-tag}
D_ground = None # T_{ground_optical<-tag}

def pose_array_to_T(det_pose):
    p = det_pose.pose.pose.position
    q = det_pose.pose.pose.orientation
    return np.dot(translation_matrix([p.x, p.y, p.z]),
                  quaternion_matrix([q.x, q.y, q.z, q.w]))

def on_hand(msg):
    global D_hand
    for d in msg.detections:
        if TAG_ID in d.id:
            D_hand = pose_array_to_T(d.pose)
            break

def on_ground(msg):
    global D_ground
    for d in msg.detections:
        if TAG_ID in d.id:
            D_ground = pose_array_to_T(d.pose)
            break

def ts_from_T(parent, child, T):
    t = translation_from_matrix(T)
    q = quaternion_from_matrix(T)
    ts = TransformStamped()
    ts.header.stamp = rospy.Time.now()
    ts.header.frame_id = parent
    ts.child_frame_id = child
    ts.transform.translation.x = float(t[0])
    ts.transform.translation.y = float(t[1])
    ts.transform.translation.z = float(t[2])
    ts.transform.rotation.x = float(q[0])
    ts.transform.rotation.y = float(q[1])
    ts.transform.rotation.z = float(q[2])
    ts.transform.rotation.w = float(q[3])
    return ts

def T_from_ts(ts):
    t = ts.transform.translation
    q = ts.transform.rotation
    return np.dot(translation_matrix([t.x, t.y, t.z]),
                  quaternion_matrix([q.x, q.y, q.z, q.w]))

def rpy_deg_from_T(T):
    R = T.copy(); R[0:3,3] = [0,0,0]
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0
    return np.degrees([roll, pitch, yaw])

def main():
    rospy.init_node("tag_visualize_and_verify")

    rospy.Subscriber(HAND_TOPIC, AprilTagDetectionArray, on_hand, queue_size=1)
    rospy.Subscriber(GROUND_TOPIC, AprilTagDetectionArray, on_ground, queue_size=1)

    buf = Buffer(cache_time=rospy.Duration(30))
    listener = TransformListener(buf)
    dyn_broadcaster = tf2_ros.TransformBroadcaster()
    static_broadcaster = StaticTransformBroadcaster()

    rate = rospy.Rate(10)

    printed_ok = False
    while not rospy.is_shutdown():
        if D_hand is None or D_ground is None:
            rate.sleep()
            continue

        try:
            # Lh = T_{BASE<-HAND_OPTICAL}
            Lh = buf.lookup_transform(BASE, HAND_OPTICAL, rospy.Time(0), rospy.Duration(1.0))
            Lh_T = T_from_ts(Lh)

            # try to get Lg directly; if missing, compose via ground_cam_link
            Lg_T = None
            try:
                Lg = buf.lookup_transform(BASE, GROUND_OPTICAL, rospy.Time(0), rospy.Duration(0.2))
                Lg_T = T_from_ts(Lg)
            except:
                # T_{GROUND_OPTICAL<-GROUND_LINK}
                Gopt_Glink = buf.lookup_transform(GROUND_OPTICAL, GROUND_LINK, rospy.Time(0), rospy.Duration(1.0))
                Gopt_Glink_T = T_from_ts(Gopt_Glink)                  # T_{Gopt<-Glink}
                # we also need T_{BASE<-GROUND_LINK}
                # if you ran the earlier node once, this static exists; else we compute now via both tags
                # compute T_{BASE<-GROUND_LINK} using formula: Lh * D_hand * inv(D_ground) * (T_{Gopt<-Glink})
                T_base_groundlink = np.dot(np.dot(np.dot(Lh_T, D_hand), inverse_matrix(D_ground)), Gopt_Glink_T)
                static_broadcaster.sendTransform(ts_from_T(BASE, GROUND_LINK, T_base_groundlink))
                Lg_T = np.dot(T_base_groundlink, inverse_matrix(Gopt_Glink_T))  # BASE<-Gopt

            # Publish the two tag frames for RViz
            dyn_broadcaster.sendTransform(ts_from_T(HAND_OPTICAL, "tag_in_hand_optical", D_hand))
            dyn_broadcaster.sendTransform(ts_from_T(GROUND_OPTICAL, "tag_in_ground_optical", D_ground))

            # Compute tag in BASE via both paths
            T_base_tag_from_hand   = np.dot(Lh_T, D_hand)     # BASE<-tag using hand
            T_base_tag_from_ground = np.dot(Lg_T, D_ground)    # BASE<-tag using ground

            # Differences
            T_delta = np.dot(inverse_matrix(T_base_tag_from_hand), T_base_tag_from_ground)
            trans_err = np.linalg.norm(translation_from_matrix(T_delta))
            q_delta = quaternion_from_matrix(T_delta)
            # angular error from quaternion
            ang_err = 2.0 * np.arccos(np.clip(abs(q_delta[3]), 0.0, 1.0)) * 180.0/np.pi

            # Print all values once
            if not printed_ok:
                th = translation_from_matrix(D_hand); rh = rpy_deg_from_T(D_hand)
                tg = translation_from_matrix(D_ground); rg = rpy_deg_from_T(D_ground)
                bh = translation_from_matrix(T_base_tag_from_hand); rbh = rpy_deg_from_T(T_base_tag_from_hand)
                bg = translation_from_matrix(T_base_tag_from_ground); rbg = rpy_deg_from_T(T_base_tag_from_ground)

                rospy.loginfo(
                    "\nTag wrt %s:"
                    "\n  t = [%.3f, %.3f, %.3f]  rpy_deg = [%.2f, %.2f, %.2f]"
                    "\nTag wrt %s:"
                    "\n  t = [%.3f, %.3f, %.3f]  rpy_deg = [%.2f, %.2f, %.2f]"
                    "\nTag wrt %s via HAND:"
                    "\n  t = [%.3f, %.3f, %.3f]  rpy_deg = [%.2f, %.2f, %.2f]"
                    "\nTag wrt %s via GROUND:"
                    "\n  t = [%.3f, %.3f, %.3f]  rpy_deg = [%.2f, %.2f, %.2f]"
                    "\nConsistency error:"
                    "\n  position = %.3f m, orientation = %.2f deg"
                    % (HAND_OPTICAL, th[0], th[1], th[2], rh[0], rh[1], rh[2],
                       GROUND_OPTICAL, tg[0], tg[1], tg[2], rg[0], rg[1], rg[2],
                       BASE, bh[0], bh[1], bh[2], rbh[0], rbh[1], rbh[2],
                       BASE, bg[0], bg[1], bg[2], rbg[0], rbg[1], rbg[2],
                       trans_err, ang_err)
                )
                printed_ok = True

        except Exception as e:
            # keep looping until TFs are available
            pass

        rate.sleep()

if __name__ == "__main__":
    main()
