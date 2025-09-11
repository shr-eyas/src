import numpy as np
from sbd_vic.function_approximator_rbfn import FunctionApproximatorRBFN
from sbd_vic.dynamical_systems import DynamicalSystems
from sbd_vic.trajectory_generator import MinimumJerk
from sbd_vic.utils import sym, lt_pack

class DMPWithGainScheduling:
    def __init__(
            self, 
            start, 
            end, 
            tau, 
            dt,
            n_bfs_traj=21, 
            n_bfs_slack=7,
            normalize_rbfs_traj=True, 
            normalize_rbfs_slack=True,
            alpha_kb=0.5,
            K0=100.0, 
            D0=20.0,
            slack_mag=20.0, 
            slack_rate_limit=200.0
        ):
        
        self.start      = np.asarray(start, float).reshape(3)
        self.end        = np.asarray(end, float).reshape(3)
        self.tau        = float(tau)
        self.dt         = float(dt)
        self.ts         = np.arange(0.0, self.tau+1e-12, self.dt)
        self.T          = self.ts.size
        self.alpha_kb   = float(alpha_kb)
        self.K0         = float(K0)
        self.D0         = float(D0)
        self.slack_mag  = float(slack_mag)
        self.slack_rate = float(slack_rate_limit)
        self.ds         = DynamicalSystems(self.tau)

        # trajectory forcing
        self.rbf_traj   = [FunctionApproximatorRBFN(n_bfs_traj, normalize=normalize_rbfs_traj) for _ in range(3)]
        y, yd, ydd, ts  = MinimumJerk(self.start, self.end, self.tau, self.dt).generate()
        x               = self.ds.time_system(ts)
        goal            = self.ds.polynomial_system(ts, self.start, self.end, 3)

        d, m = 20.0, 1.0
        k    = (d**2) / 4.0

        spring      = k * (y - goal)
        damper      = d * self.tau * yd
        f_target    = (self.tau**2) * ydd + (spring + damper) / m
        f_target    = f_target / (x[:,None] + 1e-12)
        
        for i in range(3): 
            self.rbf_traj[i].train(x, f_target[:,i])

        # slacks
        self.rbf_Sd = FunctionApproximatorRBFN(n_bfs_slack, normalize=normalize_rbfs_slack)
        self.rbf_Sk = FunctionApproximatorRBFN(n_bfs_slack, normalize=normalize_rbfs_slack)
        self.Hd     = None
        self._init_slacks(np.eye(3))

    def get_forcing(self, x):
        """
        @brief
            Get the forcing term f(x) from the trajectory RBFs at a particular phase x.
        
        @param x (float)
            Phase variable value.
        
        @return (np.ndarray)
            Forcing term vector f(x), shape (3,).
        """
        xx = np.array([x])
        return np.array([self.rbf_traj[i].predict(xx)[0,0] for i in range(3)])

    def _init_slacks(self, H):
        """
        @brief
            Initialize the slack RBFs to match initial desired gains.
            Find S_D, S_K such that:
                S_K^2 = 2 * alpha * K0 
                S_D^2 = D0 - alpha * H 

        @param H (np.ndarray)
            Desired inertia matrix (3x3).
        """
        a   = self.alpha_kb
        I   = np.eye(3)
        H  = np.asarray(H, float).reshape(3,3)
        # choose S such that D(0)≈D0 I and K(0)=K0 I
        DeltaD0 = sym(self.D0*I - a*H)
        w,V     = np.linalg.eigh(DeltaD0)
        w       = np.clip(w,0.0,None)
        SD0     = (V*np.sqrt(w))@V.T
        SK0     = np.sqrt(max(0.0, 2.0*a*self.K0)) * I
        x       = self.ds.time_system(self.ts)
        Sd_demo = np.tile(lt_pack(SD0)[None,:], (self.T, 1))
        Sk_demo = np.tile(lt_pack(SK0)[None,:], (self.T, 1))
        self.rbf_Sd.train(x, Sd_demo); self.rbf_Sk.train(x, Sk_demo)

    def set_Hd(self,Hd):
        self.Hd = np.asarray(Hd,float).reshape(3,3)
        self._init_slacks(self.Hd)

    def theta0(self):
        nf      = sum([r.W.size for r in self.rbf_traj])
        ngd     = self.rbf_Sd.W.size
        ngk     = self.rbf_Sk.W.size
        theta   = np.concatenate([r.W.ravel() for r in self.rbf_traj] + [self.rbf_Sd.W.ravel(), self.rbf_Sk.W.ravel()])
        return theta, nf, ngd, ngk

    def set_theta(self,theta,sizes):
        nf, ngd, ngk = sizes
        off=0
        for r in self.rbf_traj:
            n=r.W.size; r.W=theta[off:off+n].reshape(r.W.shape); off+=n
        self.rbf_Sd.W=theta[off:off+ngd].reshape(self.rbf_Sd.W.shape); off+=ngd
        self.rbf_Sk.W=theta[off:off+ngk].reshape(self.rbf_Sk.W.shape); off+=ngk

    def rollout_traj(self):
        ts   = self.ts
        T    = self.T
        y    = np.zeros((T,3))
        yd   = np.zeros((T,3))
        ydd  = np.zeros((T,3))
        y[0] = self.start

        def acc(t, y_, yd_):
            phase   = self.ds.time_system(np.array([t]))[0]
            gate    = self.ds.time_system(np.array([t]))[0]
            yg      = self.ds.polynomial_system(np.array([t]), self.start, self.end, 3)[0]
            fhat    = self._f_of_phase(phase)
            d, m    = 20.0, 1.0
            k       = (d**2)/4.0
            spring  = k * (y_ - yg)
            damper  = d * self.tau * yd_
            return ((fhat * gate) - (spring + damper) / m) / (self.tau**2)
        
        for k in range(T-1):
            t0 = ts[k]
            h  = ts[k+1]-ts[k]
            k1y = yd[k];             k1v=acc(t0,       y[k],           yd[k])
            k2y = yd[k]+0.5*h*k1v;   k2v=acc(t0+0.5*h, y[k]+0.5*h*k1y, yd[k]+0.5*h*k1v)
            k3y = yd[k]+0.5*h*k2v;   k3v=acc(t0+0.5*h, y[k]+0.5*h*k2y, yd[k]+0.5*h*k2v)
            k4y = yd[k]+h*k3v;       k4v=acc(t0+h,     y[k]+h*k3y,     yd[k]+h*k3v)
            y[k+1]  =y[k]+(h/6.0)*(k1y+2*k2y+2*k3y+k4y)
            yd[k+1] =yd[k]+(h/6.0)*(k1v+2*k2v+2*k3v+k4v)
            ydd[k]  =k1v

        ydd[-1] = acc(ts[-1],y[-1],yd[-1])
        x   = self.ds.time_system(ts)
        Sd_vecs = self.rbf_Sd.predict(x)
        Sk_vecs = self.rbf_Sk.predict(x)
        # magnitude + slew limits (keep PSD design intact)
        maxmag  = self.slack_mag
        rate    = self.slack_rate
        Sd_vecs = np.clip(Sd_vecs, -maxmag, maxmag)
        Sk_vecs = np.clip(Sk_vecs, -maxmag, maxmag)

        def slew_limit(V, max_rate):
            V = V.copy()
            for t in range(1,V.shape[0]):
                dv   = V[t]-V[t-1]
                dv   = np.clip(dv, -max_rate*self.dt, max_rate*self.dt)
                V[t] = V[t-1]+dv
            return V
        
        Sd_vecs = slew_limit(Sd_vecs, rate)
        Sk_vecs = slew_limit(Sk_vecs, rate)

        return {"ts":ts, "y_des":y, "yd_des":yd, "ydd_des":ydd, "Sd_vecs":Sd_vecs, "Sk_vecs":Sk_vecs}