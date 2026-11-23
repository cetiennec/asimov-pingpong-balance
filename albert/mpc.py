"""Model-predictive controller for the ball on Albert's paddle: a drop-in replacement for the PPO policy.

Same interface as asimov_pp.ppo.load_policy(): MPCController()(obs) -> 5 actions in [-1, 1] (joint-target deltas x 0.03 rad),
from the same 40-value observation (asimov_pp/env.py): Kalman ball estimate (u, w, vu, vw), blade gravity, accelerometer,
arm joints, last actions. It is stateful: reset() at the start of each balancing (run_policy does it at BALANCE).

Reduced model, per blade axis (u along the handle, w across), decoupled:
    ball    p' = v,  v' = KB (g + d) - C v            (rolling hollow-ish ball, KB = 3/5 x 0.87 x 9.81 as the Kalman filter)
    arm     r[k+1] = r[k] + K du[k - D]                (r: the tilt the arm settles to; du: the tilt command change sent at k)
            g[k+1] = g[k] + ALPHA (r[k+1] - g[k])     (first-order lag, identified on the sim arm, see ARM below)
    d       constant tilt error (arm model's flat, ball gain): estimated online from the ball's velocity
g is the blade's in-plane gravity (unit g, = tilt in rad for small angles), from forward kinematics of the measured joints
(+ the observation's blade gravity, low-passed, for the slow part: IMU, waist). D: command delay in control steps.

Every step: a box-constrained QP over HORIZON steps (move-blocked) per axis, minimising ball position / speed, the tilt
beyond what cancels d, and command changes, with |du| <= DU_MAX (the joints' 0.03 rad/step). The first move du is mapped to
the 5 joints by a tilt-only differential IK: least squares on the tilt Jacobian, the blade centre held in its own plane,
joints weighted by their stiction band (friction / kp of the identified arm: the high-friction shoulder pitch is used little;
antagonistic shoulder motions made the response depend on the step size), plus a slow pull to the home pose in the tilt's
null space.

Pure numpy + MuJoCo forward kinematics (common.Kinematics): ~0.3 ms per step on a Mac (QP and IK), no extra dependency.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from asimov_pp.env import DT
from asimov_pp.model import BLADE_R

ROOT = Path(__file__).resolve().parent
G = 9.81
ACTION_SCALE = 0.03


@dataclass
class MPCParams:
    # ball
    kb: float = 0.6 * 0.87 * G      # ball acceleration per unit in-plane gravity (m/s2), as the Kalman filter (roll_gain 0.87)
    damping: float = 0.5           # 1/s, as the Kalman filter
    # arm (identified on the sim with the 2026-11-22 arm and the IK below; tools: see docs/mpc.md): per axis (u, w)
    alpha: tuple = (0.27, 0.32)    # tilt: first-order lag per 20 ms step (time constant ~60 ms)
    gain: tuple = (0.8, 0.9)       # tilt: settled / commanded (friction, sag)
    alpha_c: tuple = (0.3, 0.3)    # blade-centre translation in its plane: lag per step
    gain_c: tuple = (0.9, 0.9)     # translation: settled / commanded
    delay: int = 3                 # command delay, control steps (eval: 1-4 steps; 3 and 4 tuned alike, 2 worse)
    # QP
    horizon: int = 25              # steps (0.5 s)
    blocks: tuple = (1, 1, 2, 3, 4, 6, 8)  # move blocking: lengths of the constant-input blocks (sum <= horizon)
    w_p: float = 2500.0            # ball position (m)
    w_v: float = 300.0             # ball speed (m/s)
    w_g: float = 300.0             # tilt beyond -d (unit g)
    w_s: float = 300.0             # blade-centre displacement from where balancing started (m)
    w_du: float = 1000.0           # tilt command change per step
    w_dc: float = 12000.0          # centre command change per step (m)
    w_term: float = 5.0            # x the stage cost at the horizon's end
    du_max: float = 0.06           # tilt command change per step (unit g; all joints at 0.03 rad/step give ~0.07)
    dc_max: float = 0.01           # centre command change per step (m)
    r_max: float = 0.35            # tilt command bound (clipped)
    translate: bool = True         # use the blade's translation as a second input (False: tilt only)
    # observers
    l_r: float = 0.5               # arm observer: fraction of the tilt/centre innovation (/ alpha) fed to the settled value
    l_d: float = 0.0               # tilt-error estimate d: gain on the ball-velocity innovation (0: off; 0.03-0.1 tuned worse)
    d_max: float = 0.08
    tau_gobs: float = 0.5          # s: low-pass of (observed blade gravity - kinematics), the slow part of the tilt
    # IK
    w_centre: float = 1.0          # blade centre in its plane: weight of the translation target (m vs unit-g tilt)
    w_normal: float = 1.0          # blade centre along its normal: kept still
    band_weight: float = 3.0       # x stiction band (friction / kp, rad) as the joint-motion penalty
    k_home: float = 0.01           # per step: pull of the joint targets to home in the IK targets' null space
    jac_every: int = 5             # steps between Jacobian updates
    # stiction band per joint, friction / kp (rad), of the arm identified on 2026-11-22 (albert/arm_dynamics.json); arm_dyn:
    # an identified-arm file to take them from instead (relative to albert/)
    bands: tuple = (0.01805, 0.0042, 0.00735, 0.00558, 0.00397)
    arm_dyn: str = ""

    @classmethod
    def from_spec(cls, spec: str | dict | None):
        """'k=v,k=v' or a dict or a JSON file path; tuples as 'a/b'."""
        if spec is None or spec == "":
            return cls()
        if isinstance(spec, dict):
            return cls(**spec)
        if spec.endswith(".json"):
            return cls(**json.loads(Path(spec).read_text()))
        kw = {}
        for item in spec.split(","):
            k, v = item.split("=")
            default = getattr(cls(), k)
            if isinstance(default, tuple):
                kw[k] = tuple(type(default[0])(x) for x in v.split("/"))
            elif isinstance(default, bool):
                kw[k] = v.lower() in ("1", "true", "yes")
            else:
                kw[k] = type(default)(v)
        return cls(**kw)


# ----------------------------------------------------------------------------------------------- model and condensed QP
# per-axis augmented state
I_Q, I_W, I_G, I_R, I_S, I_SR, I_VS = range(7)  # ball q = p + lam s, w = v + lam vs; tilt g, settled r; centre s, settled sr, its velocity


class AxisModel:
    """One blade axis. State x = [q, w, g, r, s, sr, vs, tilt buffer (D), centre buffer (D), d]; input u = [du, dc] (tilt and
    centre command changes, applied D steps later). The ball relative to the blade: p = q - lam s, v = w - lam vs, where
    lam = kb / g: a rolling ball on a blade that accelerates by a feels -lam a relative to it (3/5 x 0.87). In q, w the
    blade's translation drops out: q'' = kb (g + d) - c v."""

    def __init__(self, p: MPCParams, axis: int):
        self.p, self.D = p, p.delay
        self.n = 8 + 2 * p.delay
        self.i_bt, self.i_bc, self.i_d = 7, 7 + p.delay, 7 + 2 * p.delay
        self.a, self.K = p.alpha[axis], p.gain[axis]
        self.ac, self.Kc = p.alpha_c[axis], p.gain_c[axis]
        self.lam = p.kb / G
        n = self.n
        self.A = np.stack([self.step(np.eye(n)[i], np.zeros(2)) for i in range(n)], 1)
        self.B = np.stack([self.step(np.zeros(n), np.eye(2)[j]) for j in range(2)], 1)
        # outputs: ball p, v; tilt beyond -d; centre s
        C = np.zeros((4, n))
        C[0, I_Q] = 1; C[0, I_S] = -self.lam
        C[1, I_W] = 1; C[1, I_VS] = -self.lam
        C[2, I_G] = 1; C[2, self.i_d] = 1
        C[3, I_S] = 1
        self.C = C

    def step(self, x, u):
        """The model's step (linear in x, u)."""
        p, D, dt = self.p, self.D, DT
        y = np.zeros(self.n)
        ut, uc = (x[self.i_bt], x[self.i_bc]) if D > 0 else (u[0], u[1])
        if D > 0:
            y[self.i_bt:self.i_bt + D - 1] = x[self.i_bt + 1:self.i_bt + D]; y[self.i_bt + D - 1] = u[0]
            y[self.i_bc:self.i_bc + D - 1] = x[self.i_bc + 1:self.i_bc + D]; y[self.i_bc + D - 1] = u[1]
        y[I_R] = x[I_R] + self.K * ut
        y[I_G] = (1 - self.a) * x[I_G] + self.a * y[I_R]
        y[I_SR] = x[I_SR] + self.Kc * uc
        y[I_S] = (1 - self.ac) * x[I_S] + self.ac * y[I_SR]
        y[I_VS] = (y[I_S] - x[I_S]) / dt
        d = x[self.i_d]
        acc = p.kb * (0.5 * (x[I_G] + y[I_G]) + d) - p.damping * (x[I_W] - self.lam * x[I_VS])
        y[I_Q] = x[I_Q] + dt * x[I_W] + 0.5 * dt * dt * acc
        y[I_W] = x[I_W] + dt * acc
        y[self.i_d] = d
        return y


def box_qp(H, f, lo, hi, Hinv=None, iters=None):
    """min 1/2 x'Hx + f'x  s.t. lo <= x <= hi (H positive definite): unconstrained solution, then a primal active-set
    iteration (fix violated bounds, re-solve the free variables, release bounds whose multiplier has the wrong sign). Exact
    for the handful of variables used here."""
    x = -(Hinv @ f) if Hinv is not None else -np.linalg.solve(H, f)
    if np.all(x >= lo) and np.all(x <= hi):
        return x
    n = len(f)
    at_lo, at_hi = x < lo, x > hi
    for _ in range(iters or 3 * n):
        fixed = at_lo | at_hi
        xf = np.where(at_lo, lo, np.where(at_hi, hi, 0.0))
        free = ~fixed
        x = xf.copy()
        if free.any():
            x[free] = np.linalg.solve(H[np.ix_(free, free)], -(f[free] + H[np.ix_(free, fixed)] @ xf[fixed]))
        g = H @ x + f
        rel = (at_lo & (g < 0)) | (at_hi & (g > 0))  # the bound pushes the wrong way: release it
        viol_lo, viol_hi = free & (x < lo - 1e-12), free & (x > hi + 1e-12)
        if not rel.any() and not viol_lo.any() and not viol_hi.any():
            break
        at_lo = (at_lo & ~rel) | viol_lo
        at_hi = (at_hi & ~rel) | viol_hi
    return np.clip(x, lo, hi)


class _AxisQP:
    """Condensed, move-blocked QP for one axis: min 1/2 U'HU + U'F x0 over the block values of [du, dc], within their bounds."""

    def __init__(self, p: MPCParams, axis: int):
        mdl = AxisModel(p, axis)
        A, B, C = mdl.A, mdl.B, mdl.C
        N, n, ny = p.horizon, mdl.n, 4
        nu = 2 if p.translate else 1
        B = B[:, :nu]
        blocks = list(p.blocks)
        while sum(blocks) > N:
            blocks.pop()
        nb = len(blocks)
        T1 = np.zeros((N, nb))  # input per step <- block value
        k = 0
        for j, L in enumerate(blocks):
            T1[k:k + L, j] = 1.0; k += L
        T = np.kron(T1, np.eye(nu))  # (N nu, nb nu), inputs interleaved per step
        powers = [np.eye(n)]
        for i in range(N):
            powers.append(A @ powers[-1])
        Phi = np.zeros((N * ny, n)); Gam = np.zeros((N * ny, N * nu))
        for i in range(N):  # y_{i+1} = C A^{i+1} x0 + sum_j C A^{i-j} B u_j
            Phi[ny * i:ny * i + ny] = C @ powers[i + 1]
            for j in range(i + 1):
                Gam[ny * i:ny * i + ny, nu * j:nu * j + nu] = C @ powers[i - j] @ B
        w = np.array([p.w_p, p.w_v, p.w_g, p.w_s if p.translate else 0.0])
        Qd = np.tile(w, N); Qd[-ny:] *= p.w_term
        Rd = np.tile(np.array([p.w_du, p.w_dc])[:nu], N)
        Gb = Gam @ T
        self.H = Gb.T @ (Qd[:, None] * Gb) + T.T @ (Rd[:, None] * T)
        self.F = Gb.T @ (Qd[:, None] * Phi)
        self.Hinv = np.linalg.inv(self.H)
        self.umax = np.tile(np.array([p.du_max, p.dc_max])[:nu], nb)
        self.model, self.nu, self.n = mdl, nu, n

    def solve(self, x0):
        """The first step's [du, dc]."""
        U = box_qp(self.H, self.F @ x0, -self.umax, self.umax, self.Hinv)
        u = np.zeros(2); u[:self.nu] = U[:self.nu]
        return u


# ----------------------------------------------------------------------------------------------- controller
class MPCController:
    def __init__(self, params: MPCParams | str | dict | None = None, home=None, kin=None):
        from albert.common import HOME, Kinematics
        from asimov_pp.model import ARM_JOINTS
        self.p = params if isinstance(params, MPCParams) else MPCParams.from_spec(params)
        self.home = np.asarray(HOME if home is None else home, float)
        self.kin = kin if kin is not None else Kinematics()
        self.qp = [_AxisQP(self.p, 0), _AxisQP(self.p, 1)]
        dyn = json.loads((ROOT / self.p.arm_dyn).read_text()) if self.p.arm_dyn else None
        kp_nom = np.array([57.0, 86.0, 96.0, 40.0, 40.0])  # common.KP of the arm
        band = (np.array(dyn["friction"]) / (kp_nom * np.array(dyn["kp_scale"]))) if dyn else np.array(self.p.bands)
        self.W = np.diag(band) * self.p.band_weight
        m = self.kin.m
        jr = np.array([m.jnt_range[m.joint(j).id] for j in ARM_JOINTS])
        self.q_lo, self.q_hi = jr[:, 0] * 0.97, jr[:, 1] * 0.97
        self.reset()

    def reset(self, q_target=None, pull_to=None):
        """A new balancing: forget the ball, the arm observers and the blade's reference position. q_target: the arm's joint
        targets now (default: the first measured joints); pull_to: the pose the null-space pull goes back to (default home)."""
        self.k = 0
        self.q_t0 = None if q_target is None else np.asarray(q_target, float).copy()
        self.pull_to = self.home.copy() if pull_to is None else np.asarray(pull_to, float).copy()
        self.x = [None, None]           # per axis augmented state
        self.d = np.zeros(2)            # tilt error (unit g) that the ball's motion reveals
        self.g_bias = None              # low-passed (observed - kinematic) blade gravity
        self.q_t = None                 # our copy of the joint targets
        self.ref = None                 # blade centre and frame when balancing started (the translation's origin)
        self.s_prev = None
        self.last_u = np.zeros((2, 2))  # per axis [du, dc] actually commanded at the last step
        self.n_sat = 0                  # steps where the IK could not give all the MPC asked (joint rate limit)

    # --- kinematics ---
    def _frame(self, q, base_quat):
        self.kin.update(q, base_quat=base_quat)
        return self.kin.blade_frame()

    def _kin(self, q, base_quat, need_jac):
        """Blade gravity (in-plane), centre in the reference frame, and (every jac_every steps) the IK pieces."""
        c0, R0 = self._frame(q, base_quat)
        g0 = (R0.T @ np.array([0, 0, -1.0]))[:2]
        if self.ref is None:
            self.ref = (c0.copy(), R0.copy())
        s0 = self.ref[1].T @ (c0 - self.ref[0])
        if not need_jac:
            return g0, s0[:2], None
        p, eps = self.p, 1e-4
        Jg, Jc = np.zeros((2, 5)), np.zeros((3, 5))
        for j in range(5):
            dq = np.zeros(5); dq[j] = eps
            c1, R1 = self._frame(q + dq, base_quat)
            Jg[:, j] = ((R1.T @ np.array([0, 0, -1.0]))[:2] - g0) / eps
            Jc[:, j] = (self.ref[1].T @ (c1 - c0)) / eps
        self._frame(q, base_quat)
        # IK as a small box QP (in __call__): min |Jg dq - du|^2 + w_c^2 |Jc_uw dq - dc|^2 + w_n^2 |Jc_n dq|^2 + |W dq|^2,
        # |dq| <= 0.03 rad; and the projector on the targets' null space for the home pull
        Jt = np.vstack([Jg, Jc[:2]]) if p.translate else Jg
        H = Jg.T @ Jg + p.w_centre ** 2 * Jc[:2].T @ Jc[:2] + p.w_normal ** 2 * np.outer(Jc[2], Jc[2]) + self.W.T @ self.W
        Pn = np.eye(5) - np.linalg.pinv(Jt) @ Jt
        return g0, s0[:2], (H, Pn, Jg, Jc[:2])

    @staticmethod
    def _base_quat(acc):
        """Body orientation (w, x, y, z) whose up axis matches the accelerometer (yaw irrelevant for the blade's gravity)."""
        u = np.asarray(acc, float); u = u / (np.linalg.norm(u) + 1e-9)
        z = np.array([0, 0, 1.0])
        ax = np.cross(u, z); sn = np.linalg.norm(ax); cth = float(u @ z)
        if sn < 1e-9:
            return np.array([1.0, 0, 0, 0])
        ang = np.arctan2(sn, cth); ax = ax / sn
        return np.r_[np.cos(ang / 2), np.sin(ang / 2) * ax]

    # --- main ---
    def __call__(self, obs):
        p, o = self.p, np.asarray(obs, np.float64)
        ball = np.array([o[0] * BLADE_R, o[1] * BLADE_R, o[2] * 0.5, o[3] * 0.5])  # u, w, vu, vw (the Kalman filter's "now")
        g_obs, acc, q = o[4:6], o[10:13], o[13:18] * 0.3 + self.home
        if self.q_t is None:
            self.q_t, self.acc_lp = (q.copy() if self.q_t0 is None else self.q_t0.copy()), acc.copy()
        self.acc_lp += (acc - self.acc_lp) * (DT / 0.3)
        need = self.k % p.jac_every == 0 or getattr(self, "H_ik", None) is None
        g_fk, s_meas, jac = self._kin(q, self._base_quat(self.acc_lp), need)
        if jac is not None:
            self.H_ik, self.Pn, self.Jg, self.Jc = jac
        if self.g_bias is None:
            self.g_bias = g_obs - g_fk
        self.g_bias += (g_obs - g_fk - self.g_bias) * (DT / p.tau_gobs)
        g = g_fk + self.g_bias
        vs_meas = np.zeros(2) if self.s_prev is None else (s_meas - self.s_prev) / DT
        self.s_prev = s_meas

        u = np.zeros((2, 2))
        for ax in range(2):
            qp = self.qp[ax]; mdl = qp.model
            x = self.x[ax]
            if x is None:
                x = np.zeros(qp.n); x[I_G] = x[I_R] = g[ax]; x[I_S] = x[I_SR] = s_meas[ax]
            else:
                x = mdl.A @ x + mdl.B @ self.last_u[ax]
                # arm observers: the innovation of the measured tilt / centre moves the settled value it heads to
                x[I_R] += p.l_r * (g[ax] - x[I_G]) / mdl.a
                x[I_SR] += p.l_r * (s_meas[ax] - x[I_S]) / mdl.ac
                # tilt error: the ball's velocity vs the prediction (relative to the blade, with its measured motion)
                innov = np.clip(ball[2 + ax] - (x[I_W] - mdl.lam * vs_meas[ax]), -0.05, 0.05)
                self.d[ax] = np.clip(self.d[ax] + p.l_d * innov / (p.kb * DT), -p.d_max, p.d_max)
            x[I_G], x[I_S], x[I_VS] = g[ax], s_meas[ax], vs_meas[ax]
            x[I_Q] = ball[ax] + mdl.lam * s_meas[ax]
            x[I_W] = ball[2 + ax] + mdl.lam * vs_meas[ax]
            x[mdl.i_d] = self.d[ax]
            self.x[ax] = x
            u[ax] = qp.solve(x)
            lo, hi = (-p.r_max - x[I_R]) / mdl.K, (p.r_max - x[I_R]) / mdl.K  # tilt bound
            u[ax, 0] = np.clip(u[ax, 0], lo, hi)

        # [du, dc] -> joints (box QP: every joint at most 0.03 rad/step; preferred joints first, others recruited when
        # those saturate); home pull in the targets' null space. The observers get what was actually commanded.
        dq_home = np.clip(self.Pn @ (self.pull_to - self.q_t) * p.k_home, -0.002, 0.002)
        f = -(self.Jg.T @ u[:, 0] + p.w_centre ** 2 * self.Jc.T @ u[:, 1])
        dq_ik = box_qp(self.H_ik, f, -ACTION_SCALE - dq_home, ACTION_SCALE - dq_home)
        if np.any(np.abs(dq_ik) >= ACTION_SCALE - np.abs(dq_home) - 1e-9):
            self.n_sat += 1
        dq = dq_ik + dq_home
        self.last_u = np.stack([self.Jg @ dq_ik, (self.Jc @ dq_ik) if p.translate else np.zeros(2)], 1)
        self.q_t = np.clip(self.q_t + dq, self.q_lo, self.q_hi)
        self.k += 1
        return (dq / ACTION_SCALE).astype(np.float32)

    def describe(self):
        return asdict(self.p)


def load_controller(spec: str):
    """'mpc' or 'mpc:k=v,...' -> MPCController; anything else: a policy checkpoint (asimov_pp.ppo.load_policy)."""
    if spec == "mpc" or spec.startswith("mpc:"):
        return MPCController(spec[4:] if spec.startswith("mpc:") else None)
    from asimov_pp.ppo import load_policy
    return load_policy(spec)


if __name__ == "__main__":  # timing only, sends nothing to the robot: albert/onboard.sh albert.mpc
    import os
    import time

    def bench(name, ctl, n=1000):
        rng, ts = np.random.default_rng(0), []
        for _ in range(n):
            obs = rng.normal(0.0, 0.1, 40).astype(np.float32)
            t0, c0 = time.perf_counter(), time.process_time()
            ctl(obs)
            ts.append((time.perf_counter() - t0, time.process_time() - c0))
        ms = np.array(ts[50:]) * 1e3
        print(f"{name:9} step: median {np.median(ms[:, 0]):.2f} ms, p99 {np.percentile(ms[:, 0], 99):.2f} ms, "
              f"max {ms[:, 0].max():.2f} ms, CPU {ms[:, 1].mean():.2f} ms = {ms[:, 1].mean() / 20 * 100:.1f} % of one core at 50 Hz")

    mpc = MPCController()
    mpc.reset()
    bench("MPC", mpc)
    ckpt = "outputs/albert_v3_final/best.pt"
    if os.path.exists(ckpt):
        from asimov_pp.ppo import load_policy
        bench("v3_final", load_policy(ckpt))
    print("control period 20 ms; nothing was sent to the robot")
