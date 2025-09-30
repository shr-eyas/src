#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from apriltag_ros.msg import AprilTagDetectionArray
from tf.transformations import (
    translation_matrix, quaternion_matrix,
    quaternion_from_matrix, translation_from_matrix,
    inverse_matrix
)

HAND_TOPIC = "/hand_tags/tag_detections"
GROUND_TOPIC = "/ground_tags/tag_detections"
HAND_OPTICAL = "hand_cam_color_optical_frame"
GROUND_OPTICAL = "ground_cam_color_optical_frame"
GROUND_LINK = "ground_cam_link"
BASE = "fr3_link0"
TAG_ID = 0  # the shared AprilTag id

hand_D = None  # D1 = T_{hand_optical<-tag}
ground_D = None  # D2 = T_{ground_optical<-tag}

def pose_to_T(pose_msg):
    p = pose_msg.pose.pose.position
    q = pose_msg.pose.pose.orientation
    T = np.dot(translation_matrix([p.x, p.y, p.z]),
               quaternion_matrix([q.x, q.y, q.z, q.w]))
    return T

def on_hand(msg):
    global hand_D
    for det in msg.detections:
        if TAG_ID in det.id:
            hand_D = pose_to_T(det.pose)
            break

def on_ground(msg):
    global ground_D
    for det in msg.detections:
        if TAG_ID in det.id:
            ground_D = pose_to_T(det.pose)
            break

def T_of_transform_stamped(ts):
    t = ts.transform.translation
    q = ts.transform.rotation
    T = np.dot(translation_matrix([t.x, t.y, t.z]),
               quaternion_matrix([q.x, q.y, q.z, q.w]))
    return T

def print_T(name_from, name_to, T):
    t = translation_from_matrix(T)
    q = quaternion_from_matrix(T)
    # RPY for convenience
    import math
    def rpy_from_matrix(M):
        # tf uses XYZ (roll, pitch, yaw)
        sy = math.sqrt(M[0,0]*M[0,0] + M[1,0]*M[1,0])
        singular = sy < 1e-6
        if not singular:
            roll  = math.atan2(M[2,1], M[2,2])
            pitch = math.atan2(-M[2,0], sy)
            yaw   = math.atan2(M[1,0], M[0,0])
        else:
            roll  = math.atan2(-M[1,2], M[1,1])
            pitch = math.atan2(-M[2,0], sy)
            yaw   = 0.0
        return roll, pitch, yaw
    R = T.copy()
    R[0:3,3] = [0,0,0]
    r,p,y = rpy_from_matrix(R)

    rospy.loginfo("\nTransform: {} -> {}\nTranslation: [{:.3f}, {:.3f}, {:.3f}]"
                  "\nRotation (quaternion): [{:.3f}, {:.3f}, {:.3f}, {:.3f}]"
                  "\nRPY (deg): [{:.3f}, {:.3f}, {:.3f}]".format(
        name_from, name_to, t[0], t[1], t[2],
        q[0], q[1], q[2], q[3],
        np.degrees(r), np.degrees(p), np.degrees(y)))

def main():
    rospy.init_node("fr3_to_ground_link_solver")

    # Subscribers for AprilTag detections
    rospy.Subscriber(HAND_TOPIC, AprilTagDetectionArray, on_hand, queue_size=1)
    rospy.Subscriber(GROUND_TOPIC, AprilTagDetectionArray, on_ground, queue_size=1)

    # TF2 buffer for existing transforms
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(5.0)
    got_printed = False

    rospy.loginfo("Waiting for AprilTag detections and TFs...")

    while not rospy.is_shutdown():
        try:
            # L1 = T_{BASE<-HAND_OPTICAL}
            L1 = tf_buffer.lookup_transform(
                BASE, HAND_OPTICAL, rospy.Time(0), rospy.Duration(1.0)
            )
            # L2 = T_{GROUND_OPTICAL<-GROUND_LINK}
            L2 = tf_buffer.lookup_transform(
                GROUND_OPTICAL, GROUND_LINK, rospy.Time(0), rospy.Duration(1.0)
            )
            L1_T = T_of_transform_stamped(L1)
            L2_T = T_of_transform_stamped(L2)

            if hand_D is not None and ground_D is not None:
                # D1 = T_{HAND_OPTICAL<-TAG}, D2 = T_{GROUND_OPTICAL<-TAG}
                D1_T = hand_D
                D2_T = ground_D

                # Desired: T_{BASE<-GROUND_LINK} = L1 * D1 * inv(D2) * L2
                T_base_groundlink = np.dot(np.dot(np.dot(L1_T, D1_T), inverse_matrix(D2_T)), L2_T)

                print_T(BASE, GROUND_LINK, T_base_groundlink)
                got_printed = True

                # Optional: also broadcast as a static TF once
                static_broadcaster = tf2_ros.StaticTransformBroadcaster()
                ts = TransformStamped()
                ts.header.stamp = rospy.Time.now()
                ts.header.frame_id = BASE
                ts.child_frame_id = GROUND_LINK
                t = translation_from_matrix(T_base_groundlink)
                q = quaternion_from_matrix(T_base_groundlink)
                ts.transform.translation.x = float(t[0])
                ts.transform.translation.y = float(t[1])
                ts.transform.translation.z = float(t[2])
                ts.transform.rotation.x = float(q[0])
                ts.transform.rotation.y = float(q[1])
                ts.transform.rotation.z = float(q[2])
                ts.transform.rotation.w = float(q[3])
                static_broadcaster.sendTransform(ts)

                rospy.loginfo("Published static TF {} -> {} once.".format(BASE, GROUND_LINK))
                # Exit after printing once
                break

        except Exception as e:
            # keep waiting
            pass

        rate.sleep()

    if not got_printed:
        rospy.logwarn("Failed to compute transform. Check topics, TF tree, and TAG_ID.")

if __name__ == "__main__":
    main()

