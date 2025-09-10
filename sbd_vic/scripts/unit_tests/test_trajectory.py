#!/usr/bin/env python3
"""
Round-trip scheduler test:
  1) Move from current EE pose to a goal using a min-jerk trajectory
  2) On completion, move back to the start (min-jerk again)
All gains/inertia are constant; matrices are COLUMN-MAJOR to match libfranka.
"""

import sys
import numpy as np
import rospy
import actionlib

from fr3_controllers.msg import ExecuteScheduleAction, ExecuteScheduleGoal
from franka_msgs.msg import FrankaState

# ----------------------------- helpers -------------------------------------
def wait_for_state(topic="/franka_state_controller/franka_states", timeout=5.0):
    """Grab one FrankaState. Returns (x[3], q[7], dq[7])."""
    msg = rospy.wait_for_message(topic, FrankaState, timeout=timeout)
    # O_T_EE is 4x4 column-major flattened; translation is elems [12,13,14]
    x = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]], dtype=float)
    q = np.array(msg.q,  dtype=float)[:7]
    dq = np.array(msg.dq, dtype=float)[:7]
    return x, q, dq

def colmajor_diag_3x3(kx, ky, kz):
    """Column-major diagonal 3×3 = [kx, 0, 0,  0, ky, 0,  0, 0, kz]."""
    return [kx, 0.0, 0.0,  0.0, ky, 0.0,  0.0, 0.0, kz]

def minjerk_xyz(start_xyz, goal_xyz, T, dt):
    """
    Minimum-jerk in R^3. Returns (x_seq, xd_seq, xdd_seq) as flat lists:
      x_seq   length 3*T  [x,y,z for each k]
      xd_seq  length 3*T
      xdd_seq length 3*T
    """
    start = np.asarray(start_xyz, float).reshape(3)
    goal  = np.asarray(goal_xyz,  float).reshape(3)
    t  = np.linspace(0.0, (T-1)*dt, T)        # T samples over [0, tau)
    tau = max(1e-9, (T-1)*dt)

    s    = t / tau
    phi  = 10*s**3 - 15*s**4 + 6*s**5
    dphi = (30*s**2 - 60*s**3 + 30*s**4) / tau
    ddphi= (60*s - 180*s**2 + 120*s**3) / (tau**2)

    A = (goal - start)  # 3
    x   = start[None,:] + phi[:,None]*A[None,:]
    xd  = dphi[:,None]*A[None,:]
    xdd = ddphi[:,None]*A[None,:]

    # flatten as [x,y,z, x,y,z, ...]
    return x.reshape(-1).tolist(), xd.reshape(-1).tolist(), xdd.reshape(-1).tolist()

def make_constant_gain_seqs(K_diag, D_diag, H_diag, T):
    """Build column-major constant K/D/H sequences of correct lengths."""
    K_blk = colmajor_diag_3x3(*K_diag)
    D_blk = colmajor_diag_3x3(*D_diag)
    K_seq = K_blk * T                  # 9*T entries
    D_seq = D_blk * T                  # 9*T entries
    H     = colmajor_diag_3x3(*H_diag) # 9 entries
    return K_seq, D_seq, H

def send_schedule(client, dt, T, use_null, alpha_kb, H_colmaj, q_home,
                  x_seq, xd_seq, xdd_seq, K_seq, D_seq, name=""):
    """Send one schedule and wait for result. Returns the result."""
    goal = ExecuteScheduleGoal()
    goal.dt = float(dt)
    goal.T  = int(T)
    goal.use_nullspace = bool(use_null)
    goal.alpha_kb = float(alpha_kb)
    goal.H = H_colmaj
    goal.q_home = q_home.tolist()

    goal.x_seq   = x_seq
    goal.xd_seq  = xd_seq
    goal.xdd_seq = xdd_seq
    goal.K_seq   = K_seq
    goal.D_seq   = D_seq

    rospy.loginfo(f"[orchestrator] Sending goal{(' '+name) if name else ''}: dt={goal.dt:.6f}, T={goal.T} ({goal.T*goal.dt:.3f}s)")
    def fb_cb(fb):
        # print progress roughly 10 times
        step = max(1, goal.T // 10)
        if fb.k % step == 0:
            rospy.loginfo(f"  progress {name}: k={fb.k}/{goal.T}")
    client.send_goal(goal, feedback_cb=fb_cb)
    client.wait_for_result()
    res = client.get_result()
    if res is None:
        raise RuntimeError("Action returned no result (preempted?)")
    rospy.loginfo(f"[orchestrator] Done {name}. success={res.success} reason='{res.reason}' T_exec={res.T_exec}")
    return res

# ------------------------------- main ---------------------------------------
def main():
    rospy.init_node("test_minjerk_roundtrip")

    # ---- User-tweakable params (feel free to hardcode if you prefer) ----
    server_name  = rospy.get_param("~server", "/scheduler_controller/execute_schedule")
    dt           = float(rospy.get_param("~dt",        0.001))   # controller sample
    tau_forward  = float(rospy.get_param("~duration",  5.0))     # seconds (forward)
    tau_return   = float(rospy.get_param("~return_duration", 5.0))
    # Goal offset relative to current EE pose:
    dx           = float(rospy.get_param("~dx", 0.2))
    dy           = float(rospy.get_param("~dy", -0.2))
    dz           = float(rospy.get_param("~dz", -0.1))           # 5 cm up by default
    # Constant gains / inertia (diag) — tune as needed:
    K_diag       = rospy.get_param("~K_diag", [600.0, 600.0, 600.0])
    D_diag       = rospy.get_param("~D_diag", [ 30.0,  30.0,  30.0])
    H_diag       = rospy.get_param("~H_diag", [  1.0,   1.0,   1.0])
    use_null     = bool(rospy.get_param("~use_nullspace", True))
    alpha_kb     = float(rospy.get_param("~alpha_kb", 0.5))

    # ---- Prep action client ----
    rospy.loginfo(f"[orchestrator] Connecting to action server: {server_name}")
    client = actionlib.SimpleActionClient(server_name, ExecuteScheduleAction)
    client.wait_for_server()
    rospy.loginfo("[orchestrator] Connected.")

    # ---- Read current state (start pose & q_home) ----
    x_start, q_start, dq_start = wait_for_state()
    x_goal = x_start + np.array([dx, dy, dz], float)

    # ---- Build forward schedule (start → goal, min-jerk) ----
    T_fwd = max(2, int(round(tau_forward / dt)))
    x_seq_fwd, xd_seq_fwd, xdd_seq_fwd = minjerk_xyz(x_start, x_goal, T_fwd, dt)
    K_seq, D_seq, H_colmaj = make_constant_gain_seqs(K_diag, D_diag, H_diag, T_fwd)

    # ---- Send forward ----
    res_fwd = send_schedule(client, dt, T_fwd, use_null, alpha_kb, H_colmaj, q_start,
                            x_seq_fwd, xd_seq_fwd, xdd_seq_fwd, K_seq, D_seq, name="(forward)")

    # Determine the pose we actually reached (last measured sample), fallback to x_goal
    if res_fwd.x_meas_seq and len(res_fwd.x_meas_seq) >= 3:
        x_reached = np.array(res_fwd.x_meas_seq[-3:], float)
    else:
        x_reached = x_goal.copy()

    # ---- Build return schedule (reached → start, min-jerk) ----
    T_ret = max(2, int(round(tau_return / dt)))
    x_seq_ret, xd_seq_ret, xdd_seq_ret = minjerk_xyz(x_reached, x_start, T_ret, dt)
    K_seq_ret, D_seq_ret, H_colmaj_ret = make_constant_gain_seqs(K_diag, D_diag, H_diag, T_ret)

    # ---- Send return ----
    _res_ret = send_schedule(client, dt, T_ret, use_null, alpha_kb, H_colmaj_ret, q_start,
                             x_seq_ret, xd_seq_ret, xdd_seq_ret, K_seq_ret, D_seq_ret, name="(return)")

    rospy.loginfo("[orchestrator] Round-trip finished. Controller is back to bias (C+G) and idle.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
