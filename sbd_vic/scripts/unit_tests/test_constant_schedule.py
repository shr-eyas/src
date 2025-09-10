#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import rospy
import actionlib

from fr3_controllers.msg import ExecuteScheduleAction, ExecuteScheduleGoal
from franka_msgs.msg import FrankaState 

def make_colmajor_diag(kx: float, ky: float, kz: float):
    """
    Return a 3x3 diagonal matrix in COLUMN-MAJOR flat list (like libfranka / Eigen).
    """
    # 3x3, column-major: [m00, m10, m20,  m01, m11, m21,  m02, m12, m22]
    return [kx, 0.0, 0.0,
            0.0, ky, 0.0,
            0.0, 0.0, kz]


def get_robot_state(topic: str, timeout: float = 5.0):
    """
    Grab ONE message of FrankaState from the given topic.
    Extract:
      - EE position (x,y,z) from column-major O_T_EE (indices 12,13,14)
      - joint position q[0:7], joint velocity dq[0:7]
    """
    msg = rospy.wait_for_message(topic, FrankaState, timeout=timeout)
    ee_pos = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]], dtype=float)
    q = np.array(msg.q, dtype=float)[:7]
    dq = np.array(msg.dq, dtype=float)[:7]
    return ee_pos, q, dq

def tile_constant_sequence(vec: np.ndarray, repeats: int):
    """
    Given a 1D vector (e.g., desired position [x,y,z]), tile it `repeats` times
    and return a flat Python list. This is what the action expects.
    """
    return np.tile(vec, (repeats,)).astype(float).tolist()


def main():
    rospy.init_node("test_constant_schedule")

    state_topic = rospy.get_param("~state_topic", "/franka_state_controller/franka_states")
    action_server = rospy.get_param("~server", "/scheduler_controller/execute_schedule")

    dt = float(rospy.get_param("~dt", 0.001))            # controller step (s)
    duration = float(rospy.get_param("~duration", 5.0))  # how long to run (s)
    steps = int(round(duration / dt))

    z_offset = float(rospy.get_param("~z_offset", 0.00))  # small move up, meters

    K_diag = [float(v) for v in rospy.get_param("~K_diag", [600.0, 600.0, 600.0])]
    D_diag = [float(v) for v in rospy.get_param("~D_diag", [30.0, 30.0, 30.0])]
    H_diag = [float(v) for v in rospy.get_param("~H_diag", [1.0, 1.0, 1.0])]

    use_nullspace = bool(rospy.get_param("~use_nullspace", True))
    alpha_kb = float(rospy.get_param("~alpha_kb", 0.5))

    rospy.loginfo("[orchestrator] Waiting for FrankaState on %s …", state_topic)
    try:
        ee_pos_0, q_0, dq_0 = get_robot_state(state_topic, timeout=10.0)
    except rospy.ROSException as e:
        rospy.logerr("Could not read FrankaState from %s: %s", state_topic, str(e))
        sys.exit(1)

    ee_pos_des = ee_pos_0.copy()
    ee_pos_des[2] += z_offset

    x_seq   = tile_constant_sequence(ee_pos_des, steps)    # len = 3*steps
    xd_seq  = [0.0] * (3 * steps)                          # hold: 0 velocity
    xdd_seq = [0.0] * (3 * steps)                          # hold: 0 acceleration

    # Build K, D, H in column-major, then repeat per tick for K_seq / D_seq
    K_block = make_colmajor_diag(*K_diag)                  # 9 numbers
    D_block = make_colmajor_diag(*D_diag)                  # 9 numbers
    H_block = make_colmajor_diag(*H_diag)                  # 9 numbers

    K_seq = K_block * steps                                # len = 9*steps
    D_seq = D_block * steps                                # len = 9*steps

    rospy.loginfo("[orchestrator] Connecting to action server: %s", action_server)
    client = actionlib.SimpleActionClient(action_server, ExecuteScheduleAction)
    client.wait_for_server()
    rospy.loginfo("[orchestrator] Connected.")

    # Fill the goal exactly how the controller expects it
    goal = ExecuteScheduleGoal()
    goal.dt = dt
    goal.T = steps
    goal.use_nullspace = use_nullspace
    goal.alpha_kb = alpha_kb
    goal.H = H_block
    goal.q_home = q_0.tolist()
    goal.x_seq = x_seq
    goal.xd_seq = xd_seq
    goal.xdd_seq = xdd_seq
    goal.K_seq = K_seq
    goal.D_seq = D_seq

    rospy.loginfo("[orchestrator] Sending goal: dt=%.6f, T=%d (%.3fs)", goal.dt, goal.T, goal.dt * goal.T)

    # Progress printer (feedback is just an index k)
    def feedback_cb(fb):
        if goal.T <= 0:
            return
        # print at ~10 checkpoints
        stride = max(1, goal.T // 10)
        if fb.k % stride == 0:
            rospy.loginfo("  progress: k=%d/%d", fb.k, goal.T)

    client.send_goal(goal, feedback_cb=feedback_cb)

    # Wait forever (Ctrl+C to stop)
    client.wait_for_result()
    result = client.get_result()

    if result is None:
        rospy.logwarn("[orchestrator] No result (goal may have been preempted).")
        sys.exit(1)

    rospy.loginfo("[orchestrator] Done. success=%s reason='%s' T_exec=%d",
                  str(result.success), result.reason, result.T_exec)
    rospy.loginfo("  t0_sec=%.6f", getattr(result, "t0_sec", 0.0))
    rospy.loginfo("  lens: x=%d, xd=%d, tau=%d, q=%d, dq=%d",
                  len(result.x_meas_seq), len(result.xd_meas_seq),
                  len(result.tau_seq), len(result.q_meas_seq), len(result.dq_meas_seq))

    # Last measured EE position 
    if len(result.x_meas_seq) >= 3:
        last_xyz = np.array(result.x_meas_seq[-3:], dtype=float)
        rospy.loginfo("  last EE pos (m): %s | desired: %s",
                      np.array2string(last_xyz, precision=4),
                      np.array2string(ee_pos_des, precision=4))


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
