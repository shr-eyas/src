"""
This file is a part of the project SbD-VIC. 

Copyright (C) 2025 Shreyas Kumar, HiRo Lab, Indian Institute of Science.
Email: shreyaskumar@iisc.ac.in

This module is licensed under the MIT License.
For more information, visit: https://opensource.org/licenses/MIT
"""

from __future__ import annotations
import os, sys
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
import mujoco
import matplotlib as mpl
if sys.platform == "darwin":
    mpl.use("Agg")
import matplotlib.pyplot as plt


import numpy as np
import rospy
import actionlib
from typing import Tuple
from fr3_controllers.msg import ExecuteScheduleAction, ExecuteScheduleGoal
from franka_msgs.msg import FrankaState


"""
Helper functions
"""
def sym(M: np.ndarray) -> np.ndarray:
    """
    @brief
        Symmetrize a square matrix.

    @param M (np.ndarray)
        Input square matrix.

    @return (np.ndarray)
        Symmetrized matrix ( (M + M.T) / 2 ).
    """
    return 0.5 * (M + M.T)

def finite_diff(Y: np.ndarray, dt: float) -> np.ndarray:
    """
    @brief
        Compute finite differences along the first axis of an array.

    @param Y (np.ndarray)
        Input array. Can be 1D, 2D, or 3D.
    @param dt (float)
        Step size.

    @return (np.ndarray)
        Array of the same shape as Y, containing finite differences.
        Central differences are used for interior points.
        Forward and backward differences are used at the boundaries.
    """
    Y = np.asarray(Y, float)
    dY = np.zeros_like(Y)

    if Y.ndim == 1:
        dY[1:-1] = (Y[2:] - Y[:-2]) / (2 * dt)
        dY[0]    = (Y[1] - Y[0]) / dt
        dY[-1]   = (Y[-1] - Y[-2]) / dt

    elif Y.ndim == 2:
        dY[1:-1, :] = (Y[2:, :] - Y[:-2, :]) / (2 * dt)
        dY[0, :]    = (Y[1, :] - Y[0, :]) / dt
        dY[-1, :]   = (Y[-1, :] - Y[-2, :]) / dt

    elif Y.ndim == 3:
        dY[1:-1, :, :] = (Y[2:, :, :] - Y[:-2, :, :]) / (2 * dt)
        dY[0, :, :]    = (Y[1, :, :] - Y[0, :, :]) / dt
        dY[-1, :, :]   = (Y[-1, :, :] - Y[-2, :, :]) / dt

    else:
        raise ValueError("finite_diff: unsupported ndim")

    return dY

def lt_pack(L: np.ndarray) -> np.ndarray:
    """
    @brief
        Pack the lower-triangular entries of a 3x3 matrix into a vector.

    @param L (np.ndarray)
        Input 3x3 matrix.

    @return (np.ndarray)
        Vector [L00, L10, L11, L20, L21, L22].
    """
    return np.array([L[0, 0],
                     L[1, 0], L[1, 1],
                     L[2, 0], L[2, 1], L[2, 2]], float)

def lt_unpack(v: np.ndarray) -> np.ndarray:
    """
    @brief
        Unpack a 6-element vector into the lower-triangular part of a 3x3 matrix.

    @param v (np.ndarray)
        Input vector of length 6, in the format [L00, L10, L11, L20, L21, L22].

    @return (np.ndarray)
        3x3 lower-triangular matrix.
    """
    v = np.asarray(v, float).reshape(-1)
    if v.shape[0] != 6:
        raise ValueError("lt_unpack: input vector must have length 6")
    L = np.zeros((3, 3), float)
    L[0, 0] = v[0]
    L[1, 0], L[1, 1] = v[1], v[2]
    L[2, 0], L[2, 1], L[2, 2] = v[3], v[4], v[5]
    return L


class MinimumJerk:
    """
    Generate minimum-jerk trajectory from start to goal in time tau with dt step.
    """
    def __init__(self, start, goal, tau, dt):
        """
        @param start (np.ndarray)
            Initial position vector.
        @param goal (np.ndarray)
            Final position vector.
        @param tau (float)
            Duration of the trajectory.
        @param dt (float)
            Time step for discretization.
        """
        self.start  =   np.asarray(start, float).reshape(3)
        self.goal   =   np.asarray(goal, float).reshape(3)
        self.tau    =   float(tau)
        self.dt     =   float(dt)
        self.ts     =   np.arange(0.0, self.tau+1e-12, self.dt)

    def generate(self):
        """
        @return:
            y   (np.ndarray): Positions over time, shape (T, 3)
            yd  (np.ndarray): Velocities over time, shape (T, 3)
            ydd (np.ndarray): Accelerations over time, shape (T, 3)
            ts  (np.ndarray): Time stamps, shape (T,)
        """
        ts = self.ts
        tau_safe = max(self.tau, np.finfo(float).eps)
        s = ts / tau_safe                       
        A = (self.goal - self.start)[None, :]       

        phi   = 10*s**3 - 15*s**4 + 6*s**5
        dphi  = (30*s**2 - 60*s**3 + 30*s**4) / tau_safe
        ddphi = (60*s - 180*s**2 + 120*s**3) / (tau_safe**2)

        y   = self.start[None, :] + phi[:, None]  * A
        yd  = dphi[:, None] * A
        ydd = ddphi[:, None]* A
        return y, yd, ydd, ts

class DynamicalSystems:
    """
    Generates the canonical phase and gating signal for DMPs.
    """
    def __init__(self, tau, decay = 0.1, D0 = 1e-7):
        """
        @param tau (float)
            Time constant for the phase variable.
        @param decay (float)
            Decay rate for the sigmoid gating function.
        @param D0 (float)
            Offset for the sigmoid gating function.
        """
        self.tau    = float(tau)
        self.decay  = float(decay) 
        self.K      = 1.0 + D0 
        self.D0     = D0
        num         = ((self.K / self.decay) - 1.0) / self.D0
        self.r      = -np.log(num) / self.tau

    def time_system(self, ts): 
        """
        @brief
            A dynamical system with linear decay.
            Dynamics:               x'   = -1 / tau
            Analytical solution:    x(t) = 1 - (t / tau)
        @param ts (np.ndarray)
            Time stamps.
        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts = np.asarray(ts, float)
        return (1.0 - ts / self.tau)
    
    def sigmoid_system(self, ts):
        """
        @brief
            A dynamical system with sigmoid decay.
            Dynamics:               x'   = r * x * (1 - x / K)
            Analytical solution:    x(t) = K / (1 + D0 * exp(-r * t))
        @param ts (np.ndarray)
            Time stamps.
        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts = np.asarray(ts, float)
        return self.K / (1.0 + self.D0*np.exp(-self.r*ts))
    
    def exponential_system(self, ts, start, goal, alpha=15.0):
        """
        @brief
            A dynamical system with exponential decay.
            Dynamics:               x'   = -(alpha / tau) * (x - goal)
            Analytical solution:    x(t) = goal + (start - goal) * exp(-(alpha / tau) * t)
        @param ts (np.ndarray)
            Time stamps.
        @param start (np.ndarray)
            Initial position vector.
        @param goal (np.ndarray)
            Final position vector.
        @param alpha (float)
            Decay rate.
        @return (np.ndarray)
            Phase variable values at the given time stamps.
        """
        ts      = np.asarray(ts, float)
        start   = np.asarray(start, float).reshape(3) 
        goal    = np.asarray(goal, float).reshape(3)
        return goal[None,:] + (start - goal)[None,:]*np.exp(-(alpha/self.tau)*ts)[:,None]

class RBF:
    def __init__(self,n,inter_height=0.7,reg=1e-6,normalize=True):
        self.n=int(n); self.h=float(inter_height); self.reg=float(reg); self.normalize=bool(normalize)
        self.centers=None; self.widths=None; self.W=None
    def _bases(self):
        if self.n>1:
            self.centers=np.linspace(0,1,self.n).reshape(-1,1)
            delta=self.centers[1]-self.centers[0]
            sigma=float(delta)/np.sqrt(-8.0*np.log(self.h))
            self.widths=np.full((self.n,1),sigma)
        else:
            self.centers=np.array([[0.5]]); self.widths=np.array([[1.0]])
    def _phi(self,x):
        X=np.asarray(x,float).reshape(-1,1); C=self.centers.T; W=self.widths.T
        Phi=np.exp(-0.5*((X-C)/W)**2)
        if self.normalize:
            s=np.sum(Phi,axis=1,keepdims=True)+1e-12; Phi=Phi/s
        return Phi
    def train(self,x,fx):
        self._bases()
        x=np.asarray(x,float).reshape(-1); FX=np.asarray(fx,float)
        if FX.ndim==1: FX=FX[:,None]
        PSI=self._phi(x)
        A=PSI.T@PSI+self.reg*np.eye(self.n); B=PSI.T@FX
        try: self.W=np.linalg.solve(A,B)
        except np.linalg.LinAlgError:
            self.W,_res,_r,_s=np.linalg.lstsq(A,B,rcond=None)
    def predict(self,x):
        if self.W is None: raise RuntimeError("RBF not trained")
        PSI=self._phi(np.asarray(x,float).reshape(-1))
        return PSI@self.W

class DMPGainSlackFull:
    """
    D(t) = α H_d + S_D S_D^T           (SAFE)
    K̇ + α Ḋ − 2α K = − S_K S_K^T     (SAFE)

    or with both signs flipped (UNSAFE).

    Slacks S_D, S_K are lower-triangular 3x3 matrices parameterized by RBFs over phase.
    """
    def __init__(self, start, goal, tau, dt,
                 n_bfs_traj=21, n_bfs_slack=7,
                 normalize_rbfs_traj=True, normalize_rbfs_slack=True,
                 alpha_kb=0.5,
                 K0=100.0, D0=20.0,
                 slack_mag=20.0, slack_rate_limit=200.0):
        self.start=np.asarray(start,float).reshape(3); self.goal=np.asarray(goal,float).reshape(3)
        self.tau=float(tau); self.dt=float(dt); self.ts=np.arange(0.0,self.tau+1e-12,self.dt); self.T=self.ts.size
        self.alpha_kb=float(alpha_kb); self.K0=float(K0); self.D0=float(D0)
        self.slack_mag=float(slack_mag); self.slack_rate=float(slack_rate_limit)
        self.canon=Canonical(self.tau)
        # trajectory forcing (RBFs)
        self.rbf_traj=[RBF(n_bfs_traj,normalize=normalize_rbfs_traj) for _ in range(3)]
        y,yd,ydd,ts=MinimumJerk(self.start,self.goal,self.tau,self.dt).generate()
        x=self.canon.phase(ts); g=self.canon.gate(ts); yg=self.canon.goal(ts,self.start,self.goal,15.0)
        d, m = 20.0, 1.0; k=(d**2)/4.0
        f_target=(self.tau**2)*ydd + (k*(y-yg) + d*self.tau*yd)/m
        f_target=f_target/(g[:,None]+1e-12)
        for i in range(3): self.rbf_traj[i].train(x,f_target[:,i])
        # slacks
        self.rbf_Sd=RBF(n_bfs_slack,normalize=normalize_rbfs_slack)
        self.rbf_Sk=RBF(n_bfs_slack,normalize=normalize_rbfs_slack)
        self.Hd=None
        self._init_slacks(np.eye(3))

    def _f_of_phase(self, x):
        xx=np.array([x]); return np.array([self.rbf_traj[i].predict(xx)[0,0] for i in range(3)])

    def _init_slacks(self,Hd):
        a=self.alpha_kb; I=np.eye(3); Hd=np.asarray(Hd,float).reshape(3,3)
        # choose S s.t. D(0)≈D0 I and K(0)=K0 I at t=0
        DeltaD0 = sym(self.D0*I - a*Hd)
        # project to PSD via eigendecomp (clip negatives)
        w,V = np.linalg.eigh(DeltaD0); w=np.clip(w,0.0,None)
        SD0 = (V*np.sqrt(w))@V.T
        SK0 = np.sqrt(max(0.0, 2.0*a*self.K0)) * I
        x=self.canon.phase(self.ts)
        Sd_demo=np.tile(lt_pack(SD0)[None,:],(self.T,1))
        Sk_demo=np.tile(lt_pack(SK0)[None,:],(self.T,1))
        self.rbf_Sd.train(x,Sd_demo); self.rbf_Sk.train(x,Sk_demo)

    def set_Hd(self,Hd):
        self.Hd=np.asarray(Hd,float).reshape(3,3)
        self._init_slacks(self.Hd)

    def theta0(self):
        nf=sum([r.W.size for r in self.rbf_traj]); ngd=self.rbf_Sd.W.size; ngk=self.rbf_Sk.W.size
        theta=np.concatenate([r.W.ravel() for r in self.rbf_traj]+[self.rbf_Sd.W.ravel(), self.rbf_Sk.W.ravel()])
        return theta, nf, ngd, ngk

    def set_theta(self,theta,sizes):
        nf,ngd,ngk=sizes; off=0
        for r in self.rbf_traj:
            n=r.W.size; r.W=theta[off:off+n].reshape(r.W.shape); off+=n
        self.rbf_Sd.W=theta[off:off+ngd].reshape(self.rbf_Sd.W.shape); off+=ngd
        self.rbf_Sk.W=theta[off:off+ngk].reshape(self.rbf_Sk.W.shape); off+=ngk

    def rollout_traj(self):
        ts=self.ts; T=self.T
        y=np.zeros((T,3)); yd=np.zeros((T,3)); ydd=np.zeros((T,3)); y[0]=self.start
        def acc(t, y_, yd_):
            phase=self.canon.phase(np.array([t]))[0]; gate=self.canon.gate(np.array([t]))[0]
            yg=self.canon.goal(np.array([t]), self.start, self.goal, 15.0)[0]
            fhat=self._f_of_phase(phase)
            d=20.0; k=(d**2)/4.0; m=1.0
            spring=k*(y_-yg); damper=d*self.tau*yd_
            return ((fhat*gate)-(spring+damper)/m)/(self.tau**2)
        for k in range(T-1):
            t0=ts[k]; h=ts[k+1]-ts[k]
            k1y=yd[k];             k1v=acc(t0,        y[k],           yd[k])
            k2y=yd[k]+0.5*h*k1v;   k2v=acc(t0+0.5*h, y[k]+0.5*h*k1y, yd[k]+0.5*h*k1v)
            k3y=yd[k]+0.5*h*k2v;   k3v=acc(t0+0.5*h, y[k]+0.5*h*k2y, yd[k]+0.5*h*k2v)
            k4y=yd[k]+h*k3v;       k4v=acc(t0+h,     y[k]+h*k3y,     yd[k]+h*k3v)
            y[k+1]=y[k]+(h/6.0)*(k1y+2*k2y+2*k3y+k4y)
            yd[k+1]=yd[k]+(h/6.0)*(k1v+2*k2v+2*k3v+k4v)
            ydd[k]=k1v
        ydd[-1]=acc(ts[-1],y[-1],yd[-1])
        x=self.canon.phase(ts)
        Sd_vecs=self.rbf_Sd.predict(x); Sk_vecs=self.rbf_Sk.predict(x)
        # rate-limit and magnitude-limit slacks to keep K(t) PD by design
        maxmag=self.slack_mag; rate=self.slack_rate
        Sd_vecs=np.clip(Sd_vecs, -maxmag, maxmag); Sk_vecs=np.clip(Sk_vecs, -maxmag, maxmag)
        # simple slew limiter
        def slew_limit(V, max_rate):
            V=V.copy()
            for t in range(1,V.shape[0]):
                dv=V[t]-V[t-1]; dv=np.clip(dv, -max_rate*self.dt, max_rate*self.dt)
                V[t]=V[t-1]+dv
            return V
        Sd_vecs=slew_limit(Sd_vecs, rate); Sk_vecs=slew_limit(Sk_vecs, rate)
        return {"ts":ts,"y_des":y,"yd_des":yd,"ydd_des":ydd,"Sd_vecs":Sd_vecs,"Sk_vecs":Sk_vecs}

# ===================== PI-BB =====================
@dataclass
class PIBBConfig:
    n_updates:int=60; n_samples:int=6; beta:float=8.0
    sigma_forcing:float=2.0; sigma_sd:float=0.3; sigma_sk:float=0.3
    decay:float=0.98; seed:int=0
class PIBB:
    def __init__(self,theta0,sizes,cfg:PIBBConfig):
        self.cfg=cfg; self.rng=np.random.default_rng(cfg.seed); self.mean=theta0.copy()
        nf,ngd,ngk=sizes
        self.sigma=np.concatenate([np.full(nf,cfg.sigma_forcing),np.full(ngd,cfg.sigma_sd),np.full(ngk,cfg.sigma_sk)])
    def sample(self,n): z=self.rng.normal(0,1,size=(n,self.mean.size)); return self.mean[None,:]+z*self.sigma[None,:]
    def update(self,ths,costs):
        cmin,cmax=float(np.min(costs)),float(np.max(costs)); scale=max(1e-8,cmax-cmin)
        w=np.exp(-self.cfg.beta*(costs-cmin)/scale); w/=np.sum(w)+1e-12
        self.mean=np.sum(ths*w[:,None],axis=0)
        diff2=np.sum(w[:,None]*(ths-self.mean[None,:])**2,axis=0)
        self.sigma=np.sqrt(self.cfg.decay*self.sigma**2 + (1.0-self.cfg.decay)*diff2 + 1e-12)

# ===================== training / eval =====================
@dataclass
class CostWeights:
    W_via: float = 100.0
    W_reg: float = 1e-4
    W_gain: float | None = None  # defaults 1/N


def run_training(model_xml, goal, via_point, tau=1.0, dt=0.002,
                 use_viewer=False,
                 MODE: str = "safe",   # "safe" or "unsafe"
                 H_d_scalar: float = 1.0,
                 alpha_kb: float = 0.6,
                 K0_init: float = 60.0,
                 D0_init: float = 15.0,
                 weights:CostWeights=CostWeights(),
                 n_updates=60, n_samples=6,
                 save_plots=True,
                 respect_limits=True,
                 use_nullspace=True,
                 haywire_error=0.07,
                 haywire_window=(0.05, 0.45),
                 unsafe_initial_like_safe: bool = True,
                 perturb_from_iter: int = 1
                 ):

    assert MODE in ("safe","unsafe")

    env=RobotEnv(model_xml,use_viewer=use_viewer,nullspace=use_nullspace,
                 kp_null=15.0,kd_null=3.0,respect_limits=respect_limits)
    env.reset()
    start=env.ee_pos(); print(f"[Init] EE start = {start.round(4)}")

    dmp=DMPGainSlackFull(start, np.asarray(goal,float).reshape(3), tau, dt,
                         n_bfs_traj=21, n_bfs_slack=7,
                         alpha_kb=alpha_kb, K0=K0_init, D0=D0_init,
                         slack_mag=20.0, slack_rate_limit=200.0)

    H_d = H_d_scalar*np.eye(3)
    dmp.set_Hd(H_d)

    # --- optimizer ---
    theta0, nf, ngd, ngk = dmp.theta0()
    opt=PIBB(theta0, (nf,ngd,ngk), PIBBConfig(n_updates=n_updates, n_samples=n_samples))

    mean_costs=[]; snapshots={}; stabs={}
    via_time=0.5*tau; via=np.asarray(via_point,float).reshape(3)

    def epi_cost(ts, pos, via, via_t, Sd, Sk, K_ts):
        T = ts.size
        idx=int(np.argmin(np.abs(ts-via_t)))
        c_via=np.linalg.norm(pos[idx]-via)
        Wg = (1.0/T) if (weights.W_gain is None) else float(weights.W_gain)
        c_reg=weights.W_reg*(np.mean(np.sum(Sd**2,axis=1))+np.mean(np.sum(Sk**2,axis=1)))
        return float(weights.W_via*c_via + c_reg + np.sum(Wg * np.sum(K_ts, axis=1)))

    def evaluate(theta, tag, iter_idx: int = 0, apply_unsafe: bool = None, inject_disturbance: bool = None):
        # Decide defaults based on MODE and training stage
        if apply_unsafe is None:
            apply_unsafe = (MODE=="unsafe")
        if inject_disturbance is None:
            inject_disturbance = (MODE=="unsafe" and iter_idx >= perturb_from_iter)
        # Local MODE selector so the rest of the code paths remain unchanged
        MODE = "unsafe" if apply_unsafe else "safe"
        dmp.set_theta(theta,(nf,ngd,ngk))
        base=dmp.rollout_traj()
        ts, Yd, Vd, Ydd = base["ts"], base["y_des"], base["yd_des"], base["ydd_des"]
        Sd_vecs, Sk_vecs = base["Sd_vecs"], base["Sk_vecs"]
        T=len(ts); a=dmp.alpha_kb

        # Build slacks per time step
        Sd=np.zeros((T,3,3)); SK=np.zeros((T,3,3))
        for k in range(T): Sd[k]=lt_unpack(Sd_vecs[k]); SK[k]=lt_unpack(Sk_vecs[k])

        # SAFE vs UNSAFE sign choices
        if MODE=="safe":
            D_design=np.array([sym(a*H_d + Sd[k]@Sd[k].T) for k in range(T)])
            Sdot=finite_diff(Sd,dt); Ddot=np.array([Sdot[k]@Sd[k].T + Sd[k]@Sdot[k].T for k in range(T)])
            B=np.array([-a*Ddot[k] - SK[k]@SK[k].T for k in range(T)])
        else:
            # Opposite sign ⇒ breaks both inequalities
            D_design=np.array([sym(a*H_d - Sd[k]@Sd[k].T) for k in range(T)])
            Sdot=finite_diff(Sd,dt); Ddot=np.array([-(Sdot[k]@Sd[k].T + Sd[k]@Sdot[k].T) for k in range(T)])
            B=np.array([+a*Ddot[k] + SK[k]@SK[k].T for k in range(T)])
        B=np.array([sym(B[k]) for k in range(T)])

        # Integrate K from Z = e^{-2αt}K,  Ż = e^{-2αt}B
        E=np.exp(-2.0*a*ts); Z=np.zeros((T,3,3)); Z[0]=np.eye(3)*dmp.K0
        for k in range(T-1):
            h=ts[k+1]-ts[k]
            Z[k+1]=Z[k]+0.5*(E[k]*B[k]+E[k+1]*B[k+1])*h
        K_design=np.array([sym(Z[k]/max(E[k],1e-12)) for k in range(T)])

        # Stability margins (design)
        lamA=np.array([np.linalg.eigvalsh(sym(a*H_d - D_design[k])).max() for k in range(T)])
        if MODE=="safe":
            C_design = np.array([- SK[k]@SK[k].T for k in range(T)])
        else:
            C_design = np.array([+ SK[k]@SK[k].T for k in range(T)])
        lamC_des=np.array([np.linalg.eigvalsh(sym(C_design[k])).max() for k in range(T)])

        # No floors in validation to preserve equality
        K_use=K_design.copy(); D_use=D_design.copy()

        # Optionally create persistent early-time error in UNSAFE mode
        if MODE=="unsafe" and inject_disturbance:
            t0,t1=haywire_window
            for k in range(T):
                if t0 <= ts[k] <= t1:
                    Yd[k] = Yd[k] + np.array([haywire_error,0.0,0.0])
                    Vd[k] = Vd[k] + np.array([0.0,0.0,0.0])
                    Ydd[k]= Ydd[k] + np.array([0.0,0.0,0.0])

        # numeric FD for applied C (sanity)
        Kdot_fd=finite_diff(K_use,dt); Ddot_fd=finite_diff(D_use,dt)
        Capp_fd=np.array([Kdot_fd[k] + a*Ddot_fd[k] - 2.0*a*K_use[k] for k in range(T)])
        lamC_app_fd=np.array([np.linalg.eigvalsh(sym(Capp_fd[k])).max() for k in range(T)])

        # rollout with OSID
        env.reset()
        # Nullspace off in UNSAFE to avoid masking instability
        nullspace_flag = (MODE=="safe") and (env.use_nullspace)
        pos=np.zeros_like(Yd); vel=np.zeros_like(Vd)
        for k in range(T):
            if k in (0,T//2,T-1):
                mineig=np.linalg.eigvalsh(K_design[k]).min()
                print(f"{tag} t={ts[k]:.3f}s | minEig(K_design)={mineig:.2f}")
            x,v=env.step_impedance_osid(Yd[k], Vd[k], Ydd[k], K_use[k], D_use[k], H_d, dt,
                                        nullspace=nullspace_flag)
            pos[k]=x; vel[k]=v

        # costs + logs
        i0=0; iv=int(np.argmin(np.abs(ts-via_time))); iT=T-1
        def diag_str(M): return np.diag(M).round(2)
        print(f"{tag} Gains:")
        print(f"  K(0) diag={diag_str(K_design[i0])}, D(0) diag={diag_str(D_design[i0])}")
        print(f"  K(via) diag={diag_str(K_design[iv])}, D(via) diag={diag_str(D_design[iv])}")
        print(f"  K(T) diag={diag_str(K_design[iT])}, D(T) diag={diag_str(D_design[iT])}")

        J=epi_cost(ts,pos,via,via_time,Sd_vecs,Sk_vecs,K_use)
        vA=int(np.sum(lamA>1e-7)); vCdes=int(np.sum(lamC_des>1e-7)); vCfd=int(np.sum(lamC_app_fd>5e-3))
        print(f"{tag} cost={J:.5f} | KB violations: A={vA}, designC={vCdes}, appliedC_fd={vCfd}")
        logs={"ts":ts,"pos":pos,"vel":vel,"K":K_use,"D":D_use,
              "K_design":K_design,"D_design":D_design}
        st={"lamA":lamA,"lamC_des":lamC_des,"lamC_app_fd":lamC_app_fd}
        return J, logs, st

    # initial
    # Initial baseline: for UNSAFE, keep signs SAFE and no disturbance if requested
    if MODE=="unsafe" and unsafe_initial_like_safe:
        J0, L0, S0 = evaluate(opt.mean, f"[mean-0-{MODE}]", iter_idx=0, apply_unsafe=False, inject_disturbance=False)
    else:
        J0, L0, S0 = evaluate(opt.mean, f"[mean-0-{MODE}]", iter_idx=0, apply_unsafe=(MODE=="unsafe"), inject_disturbance=False)
    mean_costs.append(J0); snapshots["initial"]=L0; stabs["initial"]=S0

    mid_idx = 30 if n_updates>=30 else max(1,n_updates//3)
    for u in range(n_updates):
        print(f"\n=== Update {u} ({MODE}) ===")
        ths=opt.sample(n_samples); costs=np.zeros(n_samples)
        for i,th in enumerate(ths):
            inj = (MODE=="unsafe" and (u+1) >= perturb_from_iter)
            costs[i],_,_=evaluate(th,f"[s{u}:{i}]", iter_idx=u+1, apply_unsafe=(MODE=="unsafe"), inject_disturbance=inj)
        opt.update(ths,costs)
        injm = (MODE=="unsafe" and (u+1) >= perturb_from_iter)
        Jm, Lm, Sm = evaluate(opt.mean, f"[mean{u+1}]", iter_idx=u+1, apply_unsafe=(MODE=="unsafe"), inject_disturbance=injm); mean_costs.append(Jm)
        if (u+1)==mid_idx: snapshots[f"update{mid_idx}"]=Lm; stabs[f"update{mid_idx}"]=Sm
        if (u+1)==n_updates: snapshots[f"update{n_updates}"]=Lm; stabs[f"update{n_updates}"]=Sm

    # ---- plots ----
    ts=L0["ts"]
    plt.figure(figsize=(7,4))
    plt.plot(range(len(mean_costs)),mean_costs,linewidth=2)
    plt.xlabel("update"); plt.ylabel("trajectory cost (test)"); plt.title(f"Learning curve ({MODE})"); plt.grid(True,alpha=0.3)
    if save_plots: plt.savefig(f"learning_curve_{MODE}.png",dpi=150,bbox_inches="tight"); plt.close()

    def pick(key):
        return snapshots[key] if key in snapshots else snapshots["initial"], stabs[key] if key in stabs else stabs["initial"]
    Lm, Sm = pick(f"update{n_updates}")

    # EE positions
    fig,axs=plt.subplots(1,3,figsize=(14,3.8),sharex=True); labels=["x","y","z"]
    for j in range(3):
        axs[j].plot(ts,L0["pos"][:,j],label="initial",linewidth=1.5)
        axs[j].plot(Lm["ts"],Lm["pos"][:,j],label=f"update{n_updates}",linewidth=3.0)
        axs[j].set_xlabel("time (s)"); axs[j].set_ylabel("EE pos (m)"); axs[j].set_title(labels[j]); axs[j].grid(True,alpha=0.2)
    axs[0].legend(frameon=False); fig.tight_layout()
    if save_plots: fig.savefig(f"positions_xyz_{MODE}.png",dpi=150,bbox_inches="tight"); plt.close(fig)

    # K, D diags
    fig2,axs2=plt.subplots(1,3,figsize=(14,3.8),sharex=True)
    for j in range(3):
        axs2[j].plot(ts,[L0["K"][i][j,j] for i in range(len(ts))],lw=1.5,label="initial Kjj")
        axs2[j].plot(Lm["ts"],[Lm["K"][i][j,j] for i in range(len(ts))],lw=3.0,label="final Kjj")
        axs2[j].set_xlabel("time (s)"); axs2[j].set_ylabel("K (diag)"); axs2[j].set_title(labels[j]); axs2[j].set_ylim(bottom=0); axs2[j].grid(True,alpha=0.2)
    axs2[0].legend(frameon=False); fig2.tight_layout()
    if save_plots: fig2.savefig(f"stiffness_diag_{MODE}.png",dpi=150,bbox_inches="tight"); plt.close(fig2)

    fig3,axs3=plt.subplots(1,3,figsize=(14,3.8),sharex=True)
    for j in range(3):
        axs3[j].plot(ts,[L0["D"][i][j,j] for i in range(len(ts))],lw=1.5,label="initial Djj")
        axs3[j].plot(Lm["ts"],[Lm["D"][i][j,j] for i in range(len(ts))],lw=3.0,label="final Djj")
        axs3[j].set_xlabel("time (s)"); axs3[j].set_ylabel("D (diag)"); axs3[j].set_title(labels[j]); axs3[j].set_ylim(bottom=0); axs3[j].grid(True,alpha=0.2)
    axs3[0].legend(frameon=False); fig3.tight_layout()
    if save_plots: fig3.savefig(f"damping_diag_{MODE}.png",dpi=150,bbox_inches="tight"); plt.close(fig3)

    # Stability margins
    fig4,axs4=plt.subplots(1,2,figsize=(12,3.8),sharex=True)
    axs4[0].plot(ts,Sm["lamA"],lw=3.0,label="final"); axs4[0].axhline(0.0,lw=1.0)
    axs4[0].set_title("λ_max(α H_d − D)  (≤ 0 safe)"); axs4[0].grid(True,alpha=0.2); axs4[0].legend(frameon=False)
    axs4[1].plot(ts,Sm["lamC_app_fd"],lw=1.2,ls=":",label="applied (FD)")
    axs4[1].axhline(0.0,lw=1.0); axs4[1].set_title("λ_max(K̇ + α Ḋ − 2α K)"); axs4[1].grid(True,alpha=0.2); axs4[1].legend(frameon=False)
    fig4.tight_layout()
    if save_plots: fig4.savefig(f"stability_margins_{MODE}.png",dpi=150,bbox_inches="tight"); plt.close(fig4)

    env.print_saturation_report()


# ===================== main (example) =====================
if __name__=="__main__":
    MODEL_XML = "/Users/shreyaskumar/Documents/HiRo Lab/Codes/stable_interaction/robot_description/scene.xml"
    GOAL = np.array([0.65, 0.20, 0.45])
    VIA  = np.array([0.55, 0.10, 0.48])

    # SAFE run (should satisfy certificates and behave stably)
    run_training(
        model_xml=MODEL_XML,
        goal=GOAL,
        via_point=VIA,
        tau=1.0,
        dt=0.001,
        MODE="safe",
        H_d_scalar=1.0,
        alpha_kb=0.5,         # keep 2*alpha*tau ~ O(1)
        K0_init=60.0,
        D0_init=15.0,
        weights=CostWeights(W_via=500.0, W_reg=0),
        n_updates=20,
        n_samples=6,
        save_plots=True,
        respect_limits=True,
        use_nullspace=True,
    )

    # UNSAFE run (flipped inequalities + induced early error ⇒ haywire)
    run_training(
        model_xml=MODEL_XML,
        goal=GOAL,
        via_point=VIA,
        tau=1.0,
        dt=0.001,
        MODE="unsafe",
        H_d_scalar=1.0,
        alpha_kb=0.9,
        K0_init=60.0,
        D0_init=10.0,
        weights=CostWeights(W_via=200.0, W_reg=0),
        n_updates=10,
        n_samples=4,
        save_plots=True,
        respect_limits=True,
        use_nullspace=False,  # remove nullspace damping to avoid masking
        haywire_error=0.07,
        haywire_window=(0.05,0.45),
    )



