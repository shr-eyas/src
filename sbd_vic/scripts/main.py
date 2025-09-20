#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd  # type: ignore
import rospy, actionlib
from fr3_controllers.msg import ExecuteScheduleAction, ExecuteScheduleGoal
from franka_msgs.msg import FrankaState

# -------------------- helpers --------------------

def wait_for_state(topic="/franka_state_controller/franka_states", timeout=5.0):
    msg = rospy.wait_for_message(topic, FrankaState, timeout=timeout)
    x = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]], float)
    q = np.array(msg.q,  float)[:7]
    dq = np.array(msg.dq, float)[:7]
    return x, q, dq

def minjerk_xyz(start_xyz, goal_xyz, T, dt):
    start = np.asarray(start_xyz, float).reshape(3)
    goal  = np.asarray(goal_xyz,  float).reshape(3)
    t  = np.linspace(0.0, (T-1)*dt, T)
    tau = max(1e-9, (T-1)*dt)
    s    = t / tau
    phi  = 10*s**3 - 15*s**4 + 6*s**5
    dphi = (30*s**2 - 60*s**3 + 30*s**4) / tau
    ddphi= (60*s - 180*s**2 + 120*s**3) / (tau**2)
    A = (goal - start)
    x   = start[None,:] + phi[:,None]*A[None,:]
    xd  = dphi[:,None]*A[None,:]
    xdd = ddphi[:,None]*A[None,:]
    return x.reshape(-1).tolist(), xd.reshape(-1).tolist(), xdd.reshape(-1).tolist()

def colmajor_diag_3x3(kx, ky, kz):
    # [m00,m10,m20, m01,m11,m21, m02,m12,m22] (column-major)
    return [kx,0.0,0.0, 0.0,ky,0.0, 0.0,0.0,kz]

def make_constant_gain_seqs(K_diag, D_diag, H_diag, T):
    K_blk = colmajor_diag_3x3(*K_diag)
    D_blk = colmajor_diag_3x3(*D_diag)
    return K_blk*T, D_blk*T, colmajor_diag_3x3(*H_diag)

def send_schedule(client, dt, T, use_null, alpha_kb, H_colmaj, q_home,
                  x_seq, xd_seq, xdd_seq, K_seq, D_seq, name=""):
    goal = ExecuteScheduleGoal()
    goal.dt = float(dt); goal.T = int(T)
    goal.use_nullspace = bool(use_null)
    goal.alpha_kb = float(alpha_kb)
    goal.H = H_colmaj
    goal.q_home = q_home.tolist()
    goal.x_seq, goal.xd_seq, goal.xdd_seq = x_seq, xd_seq, xdd_seq
    goal.K_seq, goal.D_seq = K_seq, D_seq
    rospy.loginfo(f"[csv_orchestrator] Send{name}: dt={goal.dt:.6f}, T={goal.T} ({goal.T*goal.dt:.3f}s)")
    client.send_goal(goal)
    client.wait_for_result()
    res = client.get_result()
    if res is None:
        raise RuntimeError("Action returned no result")
    rospy.loginfo(f"[csv_orchestrator] Done{name}. success={res.success} reason='{res.reason}' T_exec={res.T_exec}")
    return res

def pack_KD_colmajor(df):
    """
    CSV must provide K11..K33 and D11..D33. Expected order:
    [K11,K21,K31, K12,K22,K32, K13,K23,K33] per row, same for D.
    """
    K_cols = ["K11","K21","K31","K12","K22","K32","K13","K23","K33"]
    D_cols = ["D11","D21","D31","D12","D22","D32","D13","D23","D33"]
    K = df[K_cols].to_numpy(dtype=float)  # shape (T,9)
    D = df[D_cols].to_numpy(dtype=float)
    return K.reshape(-1).tolist(), D.reshape(-1).tolist()

def traj_from_csv(df):
    # timebase
    t = df["t"].to_numpy(dtype=float)
    if len(t) < 2:
        raise ValueError("CSV needs at least 2 rows")
    dt = float(np.median(np.diff(t)))
    T = len(df)

    # positions
    x = df[["x","y","z"]].to_numpy(dtype=float)  # (T,3)

    # velocities and accelerations via central differences
    xd  = np.zeros_like(x)
    xdd = np.zeros_like(x)
    xd[:,0]  = np.gradient(x[:,0], dt)
    xd[:,1]  = np.gradient(x[:,1], dt)
    xd[:,2]  = np.gradient(x[:,2], dt)
    xdd[:,0] = np.gradient(xd[:,0], dt)
    xdd[:,1] = np.gradient(xd[:,1], dt)
    xdd[:,2] = np.gradient(xd[:,2], dt)

    x_seq   = x.reshape(-1).tolist()
    xd_seq  = xd.reshape(-1).tolist()
    xdd_seq = xdd.reshape(-1).tolist()

    K_seq, D_seq = pack_KD_colmajor(df)
    return dt, T, x_seq, xd_seq, xdd_seq, K_seq, D_seq

def compute_rmse_xyz(x_des_seq, x_meas_seq, T_exec):
    """
    x_des_seq: flat list length 3*T_des
    x_meas_seq: flat list length 3*T_meas
    T_exec: executed steps reported by action result
    returns rmse_x, rmse_y, rmse_z and the per-step error array (N,3)
    """
    x_des = np.asarray(x_des_seq, float).reshape(-1, 3)
    x_mea = np.asarray(x_meas_seq, float).reshape(-1, 3)
    # executed N is min of T_exec, des length, meas length
    N = int(min(T_exec, x_des.shape[0], x_mea.shape[0]))
    if N <= 0:
        raise ValueError("No executed samples to evaluate RMSE.")
    e = x_mea[:N, :] - x_des[:N, :]
    rmse = np.sqrt((e**2).mean(axis=0))
    return rmse[0], rmse[1], rmse[2], e

def save_tau_csv(t0_sec, dt, tau_seq, path):
    """
    tau_seq: flat list length 7*N
    path: output CSV file
    """
    tau = np.asarray(tau_seq, float).reshape(-1, 7)
    N = tau.shape[0]
    t = t0_sec + np.arange(N, dtype=float) * float(dt)
    cols = ["t", "tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7"]
    df = pd.DataFrame(np.column_stack([t, tau]), columns=cols)
    df.to_csv(path, index=False)
    return N, path

# -------------------- main --------------------

def main():
    rospy.init_node("csv_minjerk_then_track_with_metrics")

    # params
    server_name   = rospy.get_param("~server", "/scheduler_controller/execute_schedule")
    csv_path      = rospy.get_param("~csv_path", "/home/sophia/fr3_ws/src/sbd_vic/data/final_rollout_experiment.csv")

    # phase 1: go-to-first
    dt_forward    = float(rospy.get_param("~dt_forward", 0.001))
    tau_to_first  = float(rospy.get_param("~go_to_first_duration", 3.0))
    K_diag_fwd    = rospy.get_param("~K_diag_forward", [600.0, 600.0, 600.0])
    D_diag_fwd    = rospy.get_param("~D_diag_forward", [ 30.0,  30.0,  30.0])

    # common
    H_diag        = rospy.get_param("~H_diag", [1.0, 1.0, 1.0])
    use_null      = bool(rospy.get_param("~use_nullspace", True))
    alpha_kb      = float(rospy.get_param("~alpha_kb", 0.5))

    # outputs
    torque_csv_path = rospy.get_param("~torque_csv_path", "/tmp/torques_phase2.csv")
    # optional: save per-step tracking table (disabled by default)
    tracking_csv_path = rospy.get_param("~tracking_csv_path", "")

    # connect
    rospy.loginfo(f"[csv_orchestrator] Connecting to {server_name}")
    client = actionlib.SimpleActionClient(server_name, ExecuteScheduleAction)
    client.wait_for_server()
    rospy.loginfo("[csv_orchestrator] Connected")

    # current state
    x_start, q_start, _ = wait_for_state()

    # load CSV and build phase-2 schedule
    df = pd.read_csv(csv_path)
    first_xyz = df.loc[0, ["x","y","z"]].to_numpy(dtype=float)

    # ---- Phase 1: min-jerk to first waypoint with constant gains ----
    T_fwd = max(2, int(round(tau_to_first / dt_forward)))
    x_seq_fwd, xd_seq_fwd, xdd_seq_fwd = minjerk_xyz(x_start, first_xyz, T_fwd, dt_forward)
    K_seq_fwd, D_seq_fwd, H_col = make_constant_gain_seqs(K_diag_fwd, D_diag_fwd, H_diag, T_fwd)
    res1 = send_schedule(client, dt_forward, T_fwd, use_null, alpha_kb, H_col, q_start,
                         x_seq_fwd, xd_seq_fwd, xdd_seq_fwd, K_seq_fwd, D_seq_fwd, name="(go-to-first)")

    # measured end of phase-1 becomes start for phase-2
    if res1 and res1.x_meas_seq and len(res1.x_meas_seq) >= 3:
        x_phase2_start = np.array(res1.x_meas_seq[-3:], float)
    else:
        x_phase2_start = first_xyz.copy()

    # ---- Phase 2: follow CSV trajectory and gains ----
    dt_csv, T_csv, x_seq, xd_seq, xdd_seq, K_seq, D_seq = traj_from_csv(df)

    # optional micro "catch-up" to align with first CSV sample
    if np.linalg.norm(x_phase2_start - first_xyz) > 1e-4:
        T_align = max(2, int(round(0.5 / dt_csv)))  # 0.5 s align
        xs, xds, xdds = minjerk_xyz(x_phase2_start, first_xyz, T_align, dt_csv)
        K_align, D_align, _H = make_constant_gain_seqs(K_diag_fwd, D_diag_fwd, H_diag, T_align)
        send_schedule(client, dt_csv, T_align, use_null, alpha_kb, H_col, q_start,
                      xs, xds, xdds, K_align, D_align, name="(align)")

    # main tracking
    res2 = send_schedule(client, dt_csv, T_csv, use_null, alpha_kb, H_col, q_start,
                         x_seq, xd_seq, xdd_seq, K_seq, D_seq, name="(csv-track)")

    # -------------------- metrics and logs --------------------

    # RMSE per axis on phase-2 using executed samples
    T_exec = int(res2.T_exec if res2.T_exec > 0 else T_csv)
    rmse_x, rmse_y, rmse_z, err = compute_rmse_xyz(x_seq, res2.x_meas_seq, T_exec)
    rospy.loginfo(f"[metrics] RMSE [m]  x={rmse_x:.6f}, y={rmse_y:.6f}, z={rmse_z:.6f}")

    # Save torques for phase-2
    N_tau, out_path = save_tau_csv(res2.t0_sec, dt_csv, res2.tau_seq, torque_csv_path)
    rospy.loginfo(f"[metrics] Saved {N_tau} torque rows to: {out_path}")

    # Optional: save detailed per-step tracking table
    if tracking_csv_path:
        x_des = np.asarray(x_seq, float).reshape(-1, 3)
        x_mea = np.asarray(res2.x_meas_seq, float).reshape(-1, 3)
        N = int(min(T_exec, x_des.shape[0], x_mea.shape[0]))
        t = res2.t0_sec + np.arange(N, dtype=float) * float(dt_csv)
        track_df = pd.DataFrame({
            "t": t,
            "x_des": x_des[:N, 0], "y_des": x_des[:N, 1], "z_des": x_des[:N, 2],
            "x_meas": x_mea[:N, 0], "y_meas": x_mea[:N, 1], "z_meas": x_mea[:N, 2],
            "ex": err[:N, 0], "ey": err[:N, 1], "ez": err[:N, 2],
        })
        track_df.to_csv(tracking_csv_path, index=False)
        rospy.loginfo(f"[metrics] Saved tracking table to: {tracking_csv_path}")

    # Final console report
    print("\n=== Phase-2 Tracking RMSE (m) ===")
    print(f"x: {rmse_x:.6f}")
    print(f"y: {rmse_y:.6f}")
    print(f"z: {rmse_z:.6f}")
    print(f"Torques saved to: {out_path}")

    rospy.loginfo("[csv_orchestrator] Finished.")

# -------------------- entry --------------------

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass






# #!/usr/bin/env python3
# import numpy as np
# import pandas as pd # type: ignore
# import rospy, actionlib
# from fr3_controllers.msg import ExecuteScheduleAction, ExecuteScheduleGoal
# from franka_msgs.msg import FrankaState

# # ---------- helpers ----------
# def wait_for_state(topic="/franka_state_controller/franka_states", timeout=5.0):
#     msg = rospy.wait_for_message(topic, FrankaState, timeout=timeout)
#     x = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]], float)
#     q = np.array(msg.q,  float)[:7]
#     dq = np.array(msg.dq, float)[:7]
#     return x, q, dq

# def minjerk_xyz(start_xyz, goal_xyz, T, dt):
#     start = np.asarray(start_xyz, float).reshape(3)
#     goal  = np.asarray(goal_xyz,  float).reshape(3)
#     t  = np.linspace(0.0, (T-1)*dt, T)
#     tau = max(1e-9, (T-1)*dt)
#     s    = t / tau
#     phi  = 10*s**3 - 15*s**4 + 6*s**5
#     dphi = (30*s**2 - 60*s**3 + 30*s**4) / tau
#     ddphi= (60*s - 180*s**2 + 120*s**3) / (tau**2)
#     A = (goal - start)
#     x   = start[None,:] + phi[:,None]*A[None,:]
#     xd  = dphi[:,None]*A[None,:]
#     xdd = ddphi[:,None]*A[None,:]
#     return x.reshape(-1).tolist(), xd.reshape(-1).tolist(), xdd.reshape(-1).tolist()

# def colmajor_diag_3x3(kx, ky, kz):
#     # [m00,m10,m20, m01,m11,m21, m02,m12,m22] (column-major)
#     return [kx,0.0,0.0, 0.0,ky,0.0, 0.0,0.0,kz]

# def make_constant_gain_seqs(K_diag, D_diag, H_diag, T):
#     K_blk = colmajor_diag_3x3(*K_diag)
#     D_blk = colmajor_diag_3x3(*D_diag)
#     return K_blk*T, D_blk*T, colmajor_diag_3x3(*H_diag)

# def send_schedule(client, dt, T, use_null, alpha_kb, H_colmaj, q_home,
#                   x_seq, xd_seq, xdd_seq, K_seq, D_seq, name=""):
#     goal = ExecuteScheduleGoal()
#     goal.dt = float(dt); goal.T = int(T)
#     goal.use_nullspace = bool(use_null)
#     goal.alpha_kb = float(alpha_kb)
#     goal.H = H_colmaj
#     goal.q_home = q_home.tolist()
#     goal.x_seq, goal.xd_seq, goal.xdd_seq = x_seq, xd_seq, xdd_seq
#     goal.K_seq, goal.D_seq = K_seq, D_seq
#     rospy.loginfo(f"[csv_orchestrator] Send{name}: dt={goal.dt:.6f}, T={goal.T} ({goal.T*goal.dt:.3f}s)")
#     client.send_goal(goal)
#     client.wait_for_result()
#     res = client.get_result()
#     if res is None:
#         raise RuntimeError("Action returned no result")
#     rospy.loginfo(f"[csv_orchestrator] Done{name}. success={res.success} reason='{res.reason}' T_exec={res.T_exec}")
#     return res

# def pack_KD_colmajor(df):
#     """
#     CSV must provide K11..K33 and D11..D33. Expected order:
#     [K11,K21,K31, K12,K22,K32, K13,K23,K33] per row, same for D.
#     """
#     K_cols = ["K11","K21","K31","K12","K22","K32","K13","K23","K33"]
#     D_cols = ["D11","D21","D31","D12","D22","D32","D13","D23","D33"]
#     K = df[K_cols].to_numpy(dtype=float)  # shape (T,9)
#     D = df[D_cols].to_numpy(dtype=float)
#     return K.reshape(-1).tolist(), D.reshape(-1).tolist()

# def traj_from_csv(df):
#     # timebase
#     t = df["t"].to_numpy(dtype=float)
#     if len(t) < 2:
#         raise ValueError("CSV needs at least 2 rows")
#     dt = float(np.median(np.diff(t)))
#     T = len(df)

#     # positions
#     x = df[["x","y","z"]].to_numpy(dtype=float)  # (T,3)

#     # velocities and accelerations via central differences
#     xd  = np.zeros_like(x)
#     xdd = np.zeros_like(x)
#     xd[:,0]  = np.gradient(x[:,0], dt)
#     xd[:,1]  = np.gradient(x[:,1], dt)
#     xd[:,2]  = np.gradient(x[:,2], dt)
#     xdd[:,0] = np.gradient(xd[:,0], dt)
#     xdd[:,1] = np.gradient(xd[:,1], dt)
#     xdd[:,2] = np.gradient(xd[:,2], dt)

#     x_seq   = x.reshape(-1).tolist()
#     xd_seq  = xd.reshape(-1).tolist()
#     xdd_seq = xdd.reshape(-1).tolist()

#     K_seq, D_seq = pack_KD_colmajor(df)
#     return dt, T, x_seq, xd_seq, xdd_seq, K_seq, D_seq

# # ---------- main ----------
# def main():
#     rospy.init_node("csv_minjerk_then_track")

#     # params
#     server_name   = rospy.get_param("~server", "/scheduler_controller/execute_schedule")
#     csv_path      = rospy.get_param("~csv_path", "/home/sophia/fr3_ws/src/sbd_vic/data/final_rollout_experiment.csv")
#     # phase 1: go-to-first
#     dt_forward    = float(rospy.get_param("~dt_forward", 0.001))
#     tau_to_first  = float(rospy.get_param("~go_to_first_duration", 3.0))
#     K_diag_fwd    = rospy.get_param("~K_diag_forward", [600.0, 600.0, 600.0])
#     D_diag_fwd    = rospy.get_param("~D_diag_forward", [ 30.0,  30.0,  30.0])
#     # common
#     H_diag        = rospy.get_param("~H_diag", [1.0, 1.0, 1.0])
#     use_null      = bool(rospy.get_param("~use_nullspace", True))
#     alpha_kb      = float(rospy.get_param("~alpha_kb", 0.5))

#     # connect
#     rospy.loginfo(f"[csv_orchestrator] Connecting to {server_name}")
#     client = actionlib.SimpleActionClient(server_name, ExecuteScheduleAction)
#     client.wait_for_server()
#     rospy.loginfo("[csv_orchestrator] Connected")

#     # current state
#     x_start, q_start, _ = wait_for_state()

#     # load CSV and build phase-2 schedule
#     df = pd.read_csv(csv_path)
#     # first waypoint from CSV
#     first_xyz = df.loc[0, ["x","y","z"]].to_numpy(dtype=float)

#     # ---- Phase 1: min-jerk to first waypoint with constant gains ----
#     T_fwd = max(2, int(round(tau_to_first / dt_forward)))
#     x_seq_fwd, xd_seq_fwd, xdd_seq_fwd = minjerk_xyz(x_start, first_xyz, T_fwd, dt_forward)
#     K_seq_fwd, D_seq_fwd, H_col = make_constant_gain_seqs(K_diag_fwd, D_diag_fwd, H_diag, T_fwd)
#     res1 = send_schedule(client, dt_forward, T_fwd, use_null, alpha_kb, H_col, q_start,
#                          x_seq_fwd, xd_seq_fwd, xdd_seq_fwd, K_seq_fwd, D_seq_fwd, name="(go-to-first)")

#     # if phase-1 reported last measured pose, use it as actual start for phase-2
#     if res1 and res1.x_meas_seq and len(res1.x_meas_seq) >= 3:
#         x_phase2_start = np.array(res1.x_meas_seq[-3:], float)
#     else:
#         x_phase2_start = first_xyz.copy()

#     # ---- Phase 2: follow CSV trajectory and gains ----
#     dt_csv, T_csv, x_seq, xd_seq, xdd_seq, K_seq, D_seq = traj_from_csv(df)

#     # Optional micro "catch-up" to align exactly with first CSV sample
#     # Comment out if not needed.
#     if np.linalg.norm(x_phase2_start - first_xyz) > 1e-4:
#         T_align = max(2, int(round(0.5 / dt_csv)))  # 0.5 s align
#         xs, xds, xdds = minjerk_xyz(x_phase2_start, first_xyz, T_align, dt_csv)
#         K_align, D_align, _H = make_constant_gain_seqs(K_diag_fwd, D_diag_fwd, H_diag, T_align)
#         send_schedule(client, dt_csv, T_align, use_null, alpha_kb, H_col, q_start,
#                       xs, xds, xdds, K_align, D_align, name="(align)")

#     # H is constant; use the same H for phase-2
#     res2 = send_schedule(client, dt_csv, T_csv, use_null, alpha_kb, H_col, q_start,
#                          x_seq, xd_seq, xdd_seq, K_seq, D_seq, name="(csv-track)")

#     rospy.loginfo("[csv_orchestrator] Finished.")

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass
