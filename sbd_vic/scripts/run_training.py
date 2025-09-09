#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS1 orchestrator for SbD-VIC hardware episodes.

- Builds SAFE schedules: {x_d, xd_d, xdd_d, K(t), D(t), H, dt}
- Sends one full episode per action goal to the realtime controller
- Waits for result, computes cost, updates θ with PI-BB

Requires:
  - sbd_vic_msgs: EpisodeSchedule.msg, EpisodeResult.msg, RunEpisode.action
  - A running action server at /sbd_vic_controller/run_episode
  - Your SbD-VIC core module on PYTHONPATH (imports below)
"""

from __future__ import annotations
import os
import time
import numpy as np

import rospy
import actionlib

from sbd_vic_msgs.msg import EpisodeSchedule, EpisodeResult
from sbd_vic_msgs.msg import RunEpisodeAction, RunEpisodeGoal

# ---- import your SbD-VIC core (adjust the module path/name if needed) ----
# Expect these symbols from your pasted file.
from your_sbd_vic_core import (           # TODO: change 'your_sbd_vic_core' to your module filename
    DMPGainSlackFull, finite_diff, lt_unpack,
    PIBB, PIBBConfig, CostWeights
)

# Optional: read start pose from franka_state topic
from franka_msgs.msg import FrankaState


def _sym(M):
    return 0.5 * (M + M.T)


def _pack33_seq(Aseq):
    """(T,3,3) -> flat list row-major per step"""
    Aseq = np.asarray(Aseq, dtype=np.float32)
    T = Aseq.shape[0]
    return Aseq.reshape(T, 9).reshape(-1).tolist()


def _pack3_seq(Vseq):
    """(T,3) -> flat list"""
    Vseq = np.asarray(Vseq, dtype=np.float32)
    return Vseq.reshape(-1).tolist()


class StartPoseWatcher(object):
    """Grabs the latest O_T_EE and returns x0 in meters."""
    def __init__(self, topic="/franka_state_controller/franka_states"):
        self._x = None
        self._sub = rospy.Subscriber(topic, FrankaState, self._cb, queue_size=1)

    def _cb(self, msg):
        # O_T_EE is 16 doubles, column-major (row-major reshape then transpose → col-major)
        T = np.array(msg.O_T_EE, dtype=np.float64).reshape(4, 4)
        self._x = T[0:3, 3].copy()

    def wait(self, timeout=3.0):
        t0 = rospy.Time.now().to_sec()
        while self._x is None and (rospy.Time.now().to_sec() - t0) < timeout and not rospy.is_shutdown():
            rospy.sleep(0.01)
        return None if self._x is None else self._x.copy()


class Orchestrator(object):
    def __init__(self):
        self.client = actionlib.SimpleActionClient("/sbd_vic_controller/run_episode", RunEpisodeAction)
        rospy.loginfo("Waiting for /sbd_vic_controller/run_episode action server...")
        self.client.wait_for_server()
        rospy.loginfo("Action server ready.")

        # Params
        p = rospy.get_param
        self.dt   = float(p("~dt",   0.001))
        self.tau  = float(p("~tau",  1.0))
        self.goal = np.array(p("~goal", [0.65, 0.20, 0.45]), dtype=float).reshape(3)
        self.via  = np.array(p("~via",  [0.55, 0.10, 0.48]), dtype=float).reshape(3)
        self.via_time = float(p("~via_time_frac", 0.5)) * self.tau

        self.Hd_scalar = float(p("~Hd_scalar", 1.0))
        self.alpha_kb  = float(p("~alpha_kb", 0.5))
        self.K0_init   = float(p("~K0_init", 60.0))
        self.D0_init   = float(p("~D0_init", 15.0))

        self.n_updates = int(p("~n_updates", 20))
        self.n_samples = int(p("~n_samples", 6))
        self.seed      = int(p("~seed", 0))

        self.W_via  = float(p("~W_via", 500.0))
        self.W_reg  = float(p("~W_reg", 0.0))
        self.W_gain = p("~W_gain", None)
        if self.W_gain is not None:
            self.W_gain = float(self.W_gain)

        self.save_npz = bool(p("~save_npz", True))
        self.out_npz  = p("~out_npz", "sbd_vic_hw_runs.npz")

        # Start pose
        self.start_pose_source = p("~start_pose_source", "topic")  # 'topic' or 'param'
        if self.start_pose_source == "param":
            self.start = np.array(p("~start", [0.5, 0.0, 0.5]), dtype=float).reshape(3)
        else:
            self.start = self._read_start_from_topic()

        # Optimizer and cost
        self.weights = CostWeights(W_via=self.W_via, W_reg=self.W_reg, W_gain=self.W_gain)
        self.Hd = self.Hd_scalar * np.eye(3)
        # Dummy DMP just to get sizes; will be rebuilt per-θ inside builder with the same Hd
        dmp0 = DMPGainSlackFull(self.start, self.goal, self.tau, self.dt,
                                alpha_kb=self.alpha_kb, K0=self.K0_init, D0=self.D0_init)
        dmp0.set_Hd(self.Hd)
        theta0, nf, ngd, ngk = dmp0.theta0()
        cfg = PIBBConfig(n_updates=self.n_updates, n_samples=self.n_samples, seed=self.seed)
        self.opt = PIBB(theta0, (nf, ngd, ngk), cfg)

        self.ts_last = None  # store timeline for logging

    # ------------ helpers ------------
    def _read_start_from_topic(self):
        watcher = StartPoseWatcher()
        x = watcher.wait(timeout=3.0)
        if x is None:
            rospy.logwarn("Start pose not received. Falling back to [0.5, 0, 0.5].")
            return np.array([0.5, 0.0, 0.5], dtype=float)
        return x

    def _build_safe_schedule(self, theta):
        """Return: (EpisodeSchedule msg, aux dict for cost/logs)"""
        dmp = DMPGainSlackFull(self.start, self.goal, self.tau, self.dt,
                               alpha_kb=self.alpha_kb, K0=self.K0_init, D0=self.D0_init)
        dmp.set_Hd(self.Hd)
        theta_sizes = dmp.theta0()[1:]
        dmp.set_theta(theta, theta_sizes)
        base = dmp.rollout_traj()

        ts   = base["ts"]; T = len(ts)
        y    = base["y_des"]; yd = base["yd_des"]; ydd = base["ydd_des"]
        Sd_v = base["Sd_vecs"]; Sk_v = base["Sk_vecs"]

        # Build Sd, Sk per-step
        Sd = np.zeros((T, 3, 3)); SK = np.zeros((T, 3, 3))
        for k in range(T):
            Sd[k] = lt_unpack(Sd_v[k]); SK[k] = lt_unpack(Sk_v[k])

        a = self.alpha_kb
        # SAFE D(t)
        D = np.array([_sym(a * self.Hd + Sd[k] @ Sd[k].T) for k in range(T)])
        # Ddot and B(t)
        Sdot = finite_diff(Sd, self.dt)
        Ddot = np.array([Sdot[k] @ Sd[k].T + Sd[k] @ Sdot[k].T for k in range(T)])
        B    = np.array([-a * Ddot[k] - SK[k] @ SK[k].T for k in range(T)])
        B    = np.array([_sym(B[k]) for k in range(T)])

        # Integrate K via Ż = e^{-2a t} B, Z(0)=K0 I, K=Z/e^{-2a t}
        E = np.exp(-2.0 * a * ts)
        Z = np.zeros((T, 3, 3)); Z[0] = np.eye(3) * self.K0_init
        for k in range(T - 1):
            h = ts[k + 1] - ts[k]
            Z[k + 1] = Z[k] + 0.5 * (E[k] * B[k] + E[k + 1] * B[k + 1]) * h
        K = np.array([_sym(Z[k] / max(E[k], 1e-12)) for k in range(T)])

        # Enforce PD floors and slew-limits (conservative)
        Kmin, Dmin = 5.0, 2.0
        for k in range(T):
            K[k] = _sym(K[k]); D[k] = _sym(D[k])
            for j in range(3):
                K[k][j, j] = max(K[k][j, j], Kmin)
                D[k][j, j] = max(D[k][j, j], Dmin)

        # Align start to measured pose to avoid a step
        y[0, :] = self.start

        # Pack message
        sch = EpisodeSchedule()
        sch.dt = float(self.dt)
        sch.T  = int(T)
        sch.Hd = self.Hd.astype(np.float32).T.flatten().tolist()
        sch.x_d.extend(_pack3_seq(y))
        sch.xd_d.extend(_pack3_seq(yd))
        sch.xdd_d.extend(_pack3_seq(ydd))
        sch.K.extend(_pack33_seq(K))
        sch.D.extend(_pack33_seq(D))

        aux = {
            "ts": ts,
            "Sd_vecs": Sd_v,
            "Sk_vecs": Sk_v,
            "K": K,
            "y_des": y,
        }
        return sch, aux

    def _send_episode(self, schedule_msg):
        goal = RunEpisodeGoal()
        goal.schedule = schedule_msg
        self.client.send_goal(goal)
        self.client.wait_for_result()
        res = self.client.get_result()  # EpisodeResult
        return res

    def _epi_cost(self, ts, x_meas_flat, aux):
        T = ts.size
        X = np.array(x_meas_flat, dtype=np.float64).reshape(T, 3)
        # via-point term
        idx = int(np.argmin(np.abs(ts - self.via_time)))
        c_via = float(np.linalg.norm(X[idx] - self.via))
        # reg term on slacks
        Sd = aux["Sd_vecs"]; Sk = aux["Sk_vecs"]
        c_reg = float(self.weights.W_reg * (np.mean(np.sum(Sd**2, axis=1)) + np.mean(np.sum(Sk**2, axis=1))))
        # gain penalty
        Wg = (1.0 / T) if (self.weights.W_gain is None) else float(self.weights.W_gain)
        K_diag_sum = np.sum([np.trace(aux["K"][k]) for k in range(T)])
        c_gain = float(Wg * K_diag_sum)
        return float(self.weights.W_via * c_via + c_reg + c_gain)

    # ------------ main loop ------------
    def run(self):
        mean_costs = []
        theta = self.opt.mean.copy()

        # Initial mean eval
        sch0, aux0 = self._build_safe_schedule(theta)
        res0 = self._send_episode(sch0)
        J0 = self._epi_cost(aux0["ts"], res0.x_meas, aux0)
        mean_costs.append(J0)
        rospy.loginfo("Initial mean cost: %.6f", J0)

        # PI-BB updates
        for u in range(self.opt.cfg.n_updates):
            if rospy.is_shutdown():
                break
            rospy.loginfo("=== Update %d ===", u)

            ths = self.opt.sample(self.opt.cfg.n_samples)
            costs = np.zeros(self.opt.cfg.n_samples, dtype=np.float64)

            for i, th in enumerate(ths):
                sch, aux = self._build_safe_schedule(th)
                res = self._send_episode(sch)
                Ji = self._epi_cost(aux["ts"], res.x_meas, aux)
                costs[i] = Ji
                rospy.loginfo(" sample %d cost=%.6f", i, Ji)

            self.opt.update(ths, costs)

            # Evaluate new mean
            schm, auxm = self._build_safe_schedule(self.opt.mean)
            resm = self._send_episode(schm)
            Jm = self._epi_cost(auxm["ts"], resm.x_meas, auxm)
            mean_costs.append(Jm)
            rospy.loginfo(" mean cost=%.6f", Jm)

        if self.save_npz:
            np.savez_compressed(self.out_npz,
                                mean_costs=np.array(mean_costs, dtype=np.float64),
                                theta_final=self.opt.mean.astype(np.float64))
            rospy.loginfo("Saved %s", os.path.abspath(self.out_npz))


def main():
    rospy.init_node("sbd_vic_orchestrator", anonymous=False)
    Orchestrator().run()


if __name__ == "__main__":
    main()
