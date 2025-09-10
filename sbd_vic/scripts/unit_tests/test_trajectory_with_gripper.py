#!/usr/bin/env python3
"""
Round-trip scheduler test + gripper routine:

  1) Move EE from current pose to a goal using a minimum-jerk trajectory
  2) At goal:   open gripper -> wait -> close gripper
  3) Move back to start (minimum-jerk)
  4) At start:  open gripper -> wait -> close gripper

All 3x3 matrices (K, D, H) are COLUMN-MAJOR to match libfranka/libfranka_msgs.
"""

import numpy as np
import rospy
import actionlib

# --- controller action (your custom scheduler) ---
from fr3_controllers.msg import ExecuteScheduleAction, ExecuteScheduleGoal
# --- state feedback ---
from franka_msgs.msg import FrankaState
# --- gripper actions ---
from franka_gripper.msg import (
    MoveAction, MoveGoal,
    GraspAction, GraspGoal, GraspEpsilon,
    HomingAction, StopAction
)

# ----------------------------- helpers -------------------------------------
def wait_for_state(topic="/franka_state_controller/franka_states", timeout=5.0):
    """Get one FrankaState. Returns (x[3], q[7], dq[7])."""
    msg = rospy.wait_for_message(topic, FrankaState, timeout=timeout)
    # O_T_EE is a 4x4 transform flattened column-major; translation = indices [12,13,14]
    x  = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]], dtype=float)
    q  = np.array(msg.q,  dtype=float)[:7]
    dq = np.array(msg.dq, dtype=float)[:7]
    return x, q, dq

def colmajor_diag_3x3(a, b, c):
    """Return a 3x3 diagonal matrix flattened in column-major order: [a,0,0, 0,b,0, 0,0,c]."""
    return [a, 0.0, 0.0,
            0.0, b, 0.0,
            0.0, 0.0, c]

def minjerk_xyz(start_xyz, goal_xyz, T, dt):
    """
    Minimum-jerk trajectory in R^3.
    Returns (x_seq, xd_seq, xdd_seq) flattened as [x,y,z, x,y,z, ...] length = 3*T each.
    """
    start = np.asarray(start_xyz, float).reshape(3)
    goal  = np.asarray(goal_xyz,  float).reshape(3)

    T = int(T)
    tau = max(1e-9, (T-1) * dt)
    t   = np.linspace(0.0, (T-1)*dt, T)
    s   = t / tau

    phi   = 10*s**3 - 15*s**4 + 6*s**5
    dphi  = (30*s**2 - 60*s**3 + 30*s**4) / tau
    ddphi = (60*s - 180*s**2 + 120*s**3) / (tau**2)

    A   = (goal - start)  # 3
    x   = start[None, :] + phi[:, None]  * A[None, :]
    xd  = dphi[:,  None] * A[None, :]
    xdd = ddphi[:, None] * A[None, :]

    return x.reshape(-1).tolist(), xd.reshape(-1).tolist(), xdd.reshape(-1).tolist()

def make_constant_gain_seqs(K_diag, D_diag, H_diag, T):
    """
    Build constant K/D/H sequences (column-major blocks).
    K_seq, D_seq each length = 9*T; H is a single 9-length array.
    """
    K_blk = colmajor_diag_3x3(*K_diag)
    D_blk = colmajor_diag_3x3(*D_diag)
    K_seq = K_blk * int(T)
    D_seq = D_blk * int(T)
    H     = colmajor_diag_3x3(*H_diag)
    return K_seq, D_seq, H

def send_schedule(client, dt, T, use_null, alpha_kb, H_colmaj, q_home,
                  x_seq, xd_seq, xdd_seq, K_seq, D_seq, name=""):
    """Send one schedule to the controller action and block for result."""
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
        step = max(1, goal.T // 10)  # print ~10 updates
        if fb.k % step == 0:
            rospy.loginfo(f"  progress {name}: k={fb.k}/{goal.T}")

    client.send_goal(goal, feedback_cb=fb_cb)
    client.wait_for_result()
    res = client.get_result()
    if res is None:
        raise RuntimeError("Schedule returned no result (preempted?)")
    rospy.loginfo(f"[orchestrator] Done {name}. success={res.success} reason='{res.reason}' T_exec={res.T_exec}")
    return res

# ----------------------------- gripper helpers ------------------------------
class FrankaHand:
    """Tiny wrapper for the gripper action clients with simple open/close calls."""
    def __init__(self):
        self.move  = actionlib.SimpleActionClient('/franka_gripper/move',  MoveAction)
        self.grasp = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.home  = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
        self.stop  = actionlib.SimpleActionClient('/franka_gripper/stop',  StopAction)

        rospy.loginfo("[gripper] Waiting for gripper action servers …")
        self.move.wait_for_server()
        self.grasp.wait_for_server()
        self.home.wait_for_server()
        self.stop.wait_for_server()
        rospy.loginfo("[gripper] Ready.")

    def homing(self, timeout_s=10.0):
        self.home.send_goal()
        ok = self.home.wait_for_result(rospy.Duration(timeout_s))
        rospy.loginfo(f"[gripper] homing -> {'OK' if ok else 'TIMEOUT'}")
        return ok

    def open(self, width=0.08, speed=0.10, timeout_s=5.0):
        """Open to a target width (m) at speed (m/s). No force control."""
        g = MoveGoal(width=float(width), speed=float(speed))
        self.move.send_goal(g)
        ok = self.move.wait_for_result(rospy.Duration(timeout_s))
        rospy.loginfo(f"[gripper] open {width:.3f} m -> {'OK' if ok else 'TIMEOUT'}")
        return ok

    def close(self, force=20.0, speed=0.05, target_width=0.0,
              eps_inner=0.005, eps_outer=0.005, timeout_s=5.0):
        """
        Close with force (N). Declares success if final aperture is within
        [target_width ± epsilon].
        """
        g = GraspGoal(
            width=float(target_width),
            speed=float(speed),
            force=float(force),
            epsilon=GraspEpsilon(inner=float(eps_inner), outer=float(eps_outer))
        )
        self.grasp.send_goal(g)
        ok = self.grasp.wait_for_result(rospy.Duration(timeout_s))
        rospy.loginfo(f"[gripper] close F={force:.1f}N -> {'OK' if ok else 'TIMEOUT'}")
        return ok

    def halt(self):
        self.stop.send_goal()
        self.stop.wait_for_result(rospy.Duration(2.0))
        rospy.loginfo("[gripper] stop.")

# --------------------------------- main -------------------------------------
def main():
    rospy.init_node("test_minjerk_roundtrip_with_gripper")

    # ---- Parameters (tweak via ROS params or leave defaults) ----------------
    # Controller action server
    server_name   = rospy.get_param("~server", "/scheduler_controller/execute_schedule")

    # Trajectory timing
    dt            = float(rospy.get_param("~dt",              0.001))
    tau_forward   = float(rospy.get_param("~duration",        5.0))
    tau_return    = float(rospy.get_param("~return_duration", 5.0))

    # Goal offset (relative to current EE pose)
    dx            = float(rospy.get_param("~dx",  0.20))
    dy            = float(rospy.get_param("~dy", -0.20))
    dz            = float(rospy.get_param("~dz", -0.10))

    # Cartesian gains/inertia (diagonal) — column-major in the goal
    K_diag        = rospy.get_param("~K_diag", [600.0, 600.0, 600.0])
    D_diag        = rospy.get_param("~D_diag", [ 30.0,  30.0,  30.0])
    H_diag        = rospy.get_param("~H_diag", [  1.0,   1.0,   1.0])
    use_nullspace = bool(rospy.get_param("~use_nullspace", True))
    alpha_kb      = float(rospy.get_param("~alpha_kb", 0.5))

    # Gripper behavior
    do_homing     = bool(rospy.get_param("~gripper_homing", False))
    open_width    = float(rospy.get_param("~open_width",   0.08))  # meters
    open_speed    = float(rospy.get_param("~open_speed",   0.10))  # m/s
    grasp_force   = float(rospy.get_param("~grasp_force", 20.00))  # N
    grasp_speed   = float(rospy.get_param("~grasp_speed",  0.05))  # m/s
    eps_inner     = float(rospy.get_param("~grasp_eps_inner", 0.005))
    eps_outer     = float(rospy.get_param("~grasp_eps_outer", 0.005))
    wait_after_open_s  = float(rospy.get_param("~wait_after_open_s",  1.0))
    wait_after_close_s = float(rospy.get_param("~wait_after_close_s", 0.5))

    # ---- Connect to controller action --------------------------------------
    rospy.loginfo(f"[orchestrator] Connecting to action server: {server_name}")
    ctrl_client = actionlib.SimpleActionClient(server_name, ExecuteScheduleAction)
    ctrl_client.wait_for_server()
    rospy.loginfo("[orchestrator] Controller ready.")

    # ---- Prepare gripper ----------------------------------------------------
    hand = FrankaHand()
    if do_homing:
        hand.homing()

    # ---- Read current state (start) ----------------------------------------
    x_start, q_start, _ = wait_for_state()
    x_goal = x_start + np.array([dx, dy, dz], float)

    # ---- Forward schedule (start -> goal) ----------------------------------
    T_fwd = max(2, int(round(tau_forward / dt)))
    x_fwd, xd_fwd, xdd_fwd = minjerk_xyz(x_start, x_goal, T_fwd, dt)
    K_fwd, D_fwd, H = make_constant_gain_seqs(K_diag, D_diag, H_diag, T_fwd)

    res_fwd = send_schedule(ctrl_client, dt, T_fwd, use_nullspace, alpha_kb, H, q_start,
                            x_fwd, xd_fwd, xdd_fwd, K_fwd, D_fwd, name="(forward)")

    # Pose we actually reached (fallback to goal if logs empty)
    x_reached = np.array(res_fwd.x_meas_seq[-3:], float) if res_fwd.x_meas_seq else x_goal.copy()

    # ---- At goal: open -> wait -> close ------------------------------------
    hand.open(width=open_width, speed=open_speed)
    rospy.sleep(wait_after_open_s)
    hand.close(force=grasp_force, speed=grasp_speed, target_width=0.0,
               eps_inner=eps_inner, eps_outer=eps_outer)
    rospy.sleep(wait_after_close_s)

    # ---- Return schedule (goal -> start) -----------------------------------
    T_ret = max(2, int(round(tau_return / dt)))
    x_ret, xd_ret, xdd_ret = minjerk_xyz(x_reached, x_start, T_ret, dt)
    K_ret, D_ret, H_ret = make_constant_gain_seqs(K_diag, D_diag, H_diag, T_ret)

    _res_ret = send_schedule(ctrl_client, dt, T_ret, use_nullspace, alpha_kb, H_ret, q_start,
                             x_ret, xd_ret, xdd_ret, K_ret, D_ret, name="(return)")

    # ---- Back at start: open -> wait -> close ------------------------------
    hand.open(width=open_width, speed=open_speed)
    rospy.sleep(wait_after_open_s)
    hand.close(force=grasp_force, speed=grasp_speed, target_width=0.0,
               eps_inner=eps_inner, eps_outer=eps_outer)
    rospy.sleep(wait_after_close_s)

    rospy.loginfo("[orchestrator] Round-trip + gripper routine complete.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
