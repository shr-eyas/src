#!/usr/bin/env python3
import sys, time
import numpy as np
import rospy
import actionlib

from fr3_controllers.msg import ExecuteScheduleAction, ExecuteScheduleGoal
from franka_msgs.msg import FrankaState

# ---- helpers ---------------------------------------------------------------
def get_state(timeout=5.0):
    msg = rospy.wait_for_message("/franka_state_controller/robot_state", FrankaState, timeout=timeout)
    # O_T_EE is 4x4 (column-major) flattened; translation is indices 12:15
    x = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]], dtype=float)
    q = np.array(msg.q, dtype=float)[:7]
    dq = np.array(msg.dq, dtype=float)[:7]
    return x, q, dq

def diag3_rowmajor(kx, ky, kz):
    # row-major 3x3: [kx,0,0, 0,ky,0, 0,0,kz]
    return [kx,0.0,0.0, 0.0,ky,0.0, 0.0,0.0,kz]

# ---- main ------------------------------------------------------------------
def main():
    rospy.init_node("test_constant_schedule")

    # params (tweak safely!)
    dt = rospy.get_param("~dt", 0.001)             # controller step
    Tsec = rospy.get_param("~duration", 2.0)       # seconds to run
    T = int(round(Tsec / dt))

    # setpoint: hold current pose or offset a bit to see motion
    dz = rospy.get_param("~z_offset", 0.02)        # 2 cm up by default
    Kdiag = rospy.get_param("~K_diag", [600.0, 600.0, 600.0])
    Ddiag = rospy.get_param("~D_diag", [30.0, 30.0, 30.0])
    Hdiag = rospy.get_param("~H_diag", [1.0, 1.0, 1.0])  # design inertia
    use_null = bool(rospy.get_param("~use_nullspace", True))
    alpha_kb = float(rospy.get_param("~alpha_kb", 0.5))

    print("[orchestrator] waiting for FrankaState…")
    x0, q0, dq0 = get_state()

    x_des = x0.copy()
    x_des[2] += float(dz)     # small test motion in z

    # build sequences
    x_seq   = np.tile(x_des, (T,)).astype(float).tolist()     # len 3*T
    xd_seq  = np.zeros(3*T, dtype=float).tolist()             # hold: 0 vel
    xdd_seq = np.zeros(3*T, dtype=float).tolist()             # hold: 0 acc

    K_block = diag3_rowmajor(*Kdiag)
    D_block = diag3_rowmajor(*Ddiag)
    K_seq   = (K_block * T)                                   # len 9*T
    D_seq   = (D_block * T)                                   # len 9*T

    H = diag3_rowmajor(*Hdiag)                                # 9 entries row-major

    # action client
    ac = actionlib.SimpleActionClient("execute_schedule", ExecuteScheduleAction)
    print("[orchestrator] waiting for action server /execute_schedule …")
    ac.wait_for_server()

    goal = ExecuteScheduleGoal()
    goal.dt = float(dt)
    goal.T  = int(T)
    goal.use_nullspace = use_null
    goal.alpha_kb = alpha_kb
    goal.H = H
    goal.q_home = q0.tolist()
    goal.x_seq   = x_seq
    goal.xd_seq  = xd_seq
    goal.xdd_seq = xdd_seq
    goal.K_seq   = K_seq
    goal.D_seq   = D_seq

    print(f"[orchestrator] sending goal: dt={goal.dt}, T={goal.T} (~{goal.T*goal.dt:.3f}s)")
    ac.send_goal(goal)

    # (optional) simple progress printer via feedback ‘k’
    def fb_cb(fb):
        if fb.k % max(1, goal.T//10) == 0:
            print(f"  progress k={fb.k}/{goal.T}")

    ac.send_goal(goal, feedback_cb=fb_cb)

    print("[orchestrator] waiting for result…")
    ac.wait_for_result()
    res = ac.get_result()

    if res is None:
        print("[orchestrator] no result (was the goal preempted?)")
        sys.exit(1)

    print(f"[orchestrator] result: success={res.success}, reason='{res.reason}', T_exec={res.T_exec}")
    print(f"[orchestrator] t0_sec={getattr(res,'t0_sec',0.0):.6f}")
    # quick sanity on lengths
    print("  len(x_meas_seq) =", len(res.x_meas_seq))
    print("  len(xd_meas_seq)=", len(res.xd_meas_seq))
    print("  len(tau_seq)    =", len(res.tau_seq))
    print("  len(q_meas_seq) =", len(res.q_meas_seq))
    print("  len(dq_meas_seq)=", len(res.dq_meas_seq))

    # tiny check: last measured position vs desired
    if res.x_meas_seq:
        x_last = np.array(res.x_meas_seq[-3:], dtype=float)
        print(f"  last EE pos (m): {x_last.round(4)}  | desired: {x_des.round(4)}")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
