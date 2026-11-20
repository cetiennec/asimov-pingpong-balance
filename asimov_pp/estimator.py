"""Ball estimator + predictor on the blade: a Kalman filter with delayed camera measurements. Shared by training (env.py) and
Albert (albert/common.py), so the policy is trained on exactly what it gets on the robot.

State per axis (u along the handle, w across): position (m) and velocity (m/s) of the ball on the blade.
Prediction, every control step: a ball rolling without slipping on a blade whose in-plane gravity component is g_inplane
(from the arm angles + IMU) and whose own in-plane acceleration is a_paddle accelerates by
    a = 3/5 (9.81 g_inplane - a_paddle) - damping v        (hollow sphere: I = 2/3 m r^2  ->  factor 3/5)
Correction: a camera position (u, w) captured LATENCY seconds ago. The filter goes back to that time, applies it, and replays
the stored inputs up to now: its output is the ball NOW, not where the camera last saw it. Measurements far from the
prediction (Mahalanobis gate) are rejected: a laptop or a hand at the blade's edge does not move the estimate.
"""
from __future__ import annotations

from collections import deque

import numpy as np

ROLL = 3.0 / 5.0
G = 9.81

# Tuned on Albert's head-camera recording (tools/tune_kf.py, albert_demo.csv): 2 mm camera noise, a ball that accelerates 0.87 x
# the ideal 3/5 g sin(tilt) (as record_demo.py measured), outliers down-weighted, and the arm model's flat error learnt online.
# Error of the "now" estimate vs the camera at that moment: median 19 mm -> 9 mm. Used by training (env.py) and Albert alike.
KF_PARAMS = dict(accel_noise=1.5, meas_noise=0.002, damping=0.5, gate=3.0, huber=1.5, roll_gain=0.87, reset_after=3,
                 bias_noise=0.02, bias_init=0.07)


class BallKF:
    def __init__(self, dt=0.02, accel_noise=0.8, meas_noise=0.004, damping=0.3, gate=4.0, history_s=0.6, huber=None,
                 roll_gain=1.0, g_bias=(0.0, 0.0), reset_after=4, bias_noise=None, bias_init=0.05):
        """huber: Mahalanobis distance above which a measurement's noise is inflated (robust to the camera's outliers) instead
        of trusted as is; None = plain Kalman. roll_gain scales 3/5 g (ball not an ideal hollow sphere, rubber), g_bias is a
        constant in-plane gravity error (unit g: the arm model's flat is not the true flat). reset_after: consecutive rejected
        measurements after which the filter restarts from the camera.
        bias_noise: also estimate that in-plane gravity error online (2 more states, a random walk of this std per sqrt(s),
        starting at +-bias_init): a ball that rolls where the arm model says the blade is flat reveals it. It survives a
        reset (it belongs to the arm, not to the ball). None: no bias states."""
        self.dt, self.q, self.r, self.c, self.gate = dt, accel_noise, meas_noise, damping, gate
        self.bias_noise, self.bias_init = bias_noise, bias_init
        self.n = 4 if bias_noise is None else 6
        self.b, self.Pb = np.zeros(2), np.eye(2) * bias_init ** 2
        self.huber, self.roll_gain, self.g_bias, self.reset_after = huber, roll_gain, np.asarray(g_bias, float), reset_after
        self.n_hist = 2 * int(history_s / dt) + 4  # control steps + the corrections inserted at their capture times
        self.pending = None  # after a drift reset: the first detection, waiting for a consistent second one
        self.reset()

    def reset(self):
        if getattr(self, "x", None) is not None and self.n == 6:  # keep what was learnt about the arm's flat
            self.b, self.Pb = self.x[4:6].copy(), self.P[4:6, 4:6].copy()
        self.x = None                     # [u, w, vu, vw] (+ [bias_u, bias_w])
        self.P = None
        self.t = None
        self.pending = None
        self.hist = deque(maxlen=self.n_hist)  # (t, x, P, input, dt) after each prediction step
        self.n_rejected = 0

    # --- model ---
    def _F(self):
        dt, c = self.dt, self.c
        F = np.eye(self.n)
        F[0, 2] = F[1, 3] = dt
        F[2, 2] = F[3, 3] = 1 - c * dt
        if self.n == 6:  # the bias accelerates the ball like a tilt
            k = ROLL * self.roll_gain * G
            F[0, 4] = F[1, 5] = 0.5 * dt * dt * k
            F[2, 4] = F[3, 5] = dt * k
        return F

    def _Q(self):
        dt, q = self.dt, self.q
        g = np.array([0.5 * dt * dt, dt])
        Q1 = np.outer(g, g) * q * q
        Q = np.zeros((self.n, self.n))
        for i in (0, 1):
            idx = [i, i + 2]
            Q[np.ix_(idx, idx)] = Q1
        if self.n == 6:
            Q[4, 4] = Q[5, 5] = self.bias_noise ** 2 * dt
        return Q

    def _predict(self, x, P, inp):
        a = ROLL * self.roll_gain * (G * (inp[:2] + self.g_bias) - inp[2:4])
        F = self._F()
        x = F @ x
        x[:4] += np.r_[0.5 * self.dt ** 2 * a, self.dt * a]
        P = F @ P @ F.T + self._Q()
        return x, P

    def _predict_dt(self, x, P, inp, dt):
        """One prediction over an arbitrary dt (the stored nominal dt is restored)."""
        if dt <= 0:
            return x, P
        dt_nom, self.dt = self.dt, dt
        try:
            return self._predict(x, P, inp)
        finally:
            self.dt = dt_nom

    def _init_state(self, z, v=None):
        """From a detection; with v (from two detections) the velocity is known to ~ r * sqrt(2) / dt_frames."""
        x = np.r_[z, (0.0, 0.0) if v is None else v]
        sv = 0.15 if v is None else 0.05
        P = np.diag([self.r ** 2, self.r ** 2, sv ** 2, sv ** 2])
        if self.n == 6:
            x = np.r_[x, self.b]
            P = np.block([[P, np.zeros((4, 2))], [np.zeros((2, 4)), self.Pb]])
        return x, P

    # --- API ---
    def step(self, t, g_inplane, a_paddle=(0.0, 0.0)):
        """Advance one control step to time t with the blade's in-plane gravity (unit-g) and paddle acceleration (m/s2)."""
        inp = np.r_[np.asarray(g_inplane, float)[:2], np.asarray(a_paddle, float)[:2]]
        # catch up in dt steps if more time passed (the first frame is a latency old, the loop may jitter)
        n = 1 if (self.t is None or self.x is None) else max(1, int(round((t - self.t) / self.dt)))
        dt_nom = self.dt
        if self.t is not None and self.x is not None and t > self.t:
            self.dt = (t - self.t) / n  # the real step (Albert's loop may run slower than 50 Hz)
        for i in range(n):
            ti = t - (n - 1 - i) * self.dt
            if self.x is not None:
                self.x, self.P = self._predict(self.x, self.P, inp)
            self.hist.append((ti, None if self.x is None else self.x.copy(), None if self.P is None else self.P.copy(), inp, self.dt))
        self.dt = dt_nom
        self.t = t

    def start(self, t1, z1, t2, z2):
        """(Re)start from two detections (t1 < t2): position from the second, velocity from both (the ball is rarely still)."""
        z1, z2 = np.asarray(z1, float), np.asarray(z2, float)
        v = np.clip((z2 - z1) / max(t2 - t1, 1e-3), -0.6, 0.6) if t2 > t1 else None
        self._start_at(t2, z2, v)

    def _start_at(self, t_cap, z, v=None):
        self.x, self.P = self._init_state(z, v)
        self.hist.clear(); self.hist.append((t_cap, self.x.copy(), self.P.copy(), np.zeros(4), self.dt))
        self.t = t_cap  # the next step() catches up to the present
        self.pending, self.n_rejected = None, 0

    def measure(self, t_cap, z, R=None):
        """A camera position z = (u, w) in m, captured at t_cap; R: its 2x2 covariance (m^2), default meas_noise^2 I.
        Returns True if accepted."""
        z = np.asarray(z, float)
        if self.x is None or not self.hist:
            if self.pending is None:
                if self.t is None:  # nothing yet at all (first frame of an episode): start from it
                    self._start_at(t_cap, z)
                    return True
                self.pending = (t_cap, z)  # after a drift: wait for a second, consistent detection
                return False
            t1, z1 = self.pending
            if t_cap > t1 and np.linalg.norm(z - z1) / (t_cap - t1) < 1.0 and t_cap - t1 < 0.3:
                self.start(t1, z1, t_cap, z)
                return True
            self.pending = (t_cap, z)
            return False
        H = np.zeros((2, self.n)); H[0, 0] = H[1, 1] = 1.0
        hist = list(self.hist)
        k = max(i for i in range(len(hist)) if hist[i][0] <= t_cap) if hist[0][0] <= t_cap else 0
        if hist[k][1] is None:
            return False
        x, P = hist[k][1].copy(), hist[k][2].copy()
        # bring the stored state from its step time exactly to the capture time (up to one control step later)
        lag = t_cap - hist[k][0]
        nxt = hist[k + 1] if k + 1 < len(hist) else None
        if 1e-4 < lag and nxt is not None:
            x, P = self._predict_dt(x, P, nxt[3], lag)
        Rz = np.eye(2) * self.r ** 2 if R is None else np.asarray(R, float) + np.eye(2) * (0.5e-3) ** 2
        S = H @ P @ H.T + Rz
        y = z - H @ x
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > self.gate ** 2:
            self.n_rejected += 1
            if self.n_rejected >= self.reset_after:  # the prediction drifted away: restart from the camera, but only on two
                t_keep = self.t                       # consistent detections (one could be the outlier itself)
                self.reset(); self.t = t_keep
                self.pending = (t_cap, z)
            return False
        self.n_rejected = 0
        if self.huber is not None and d2 > self.huber ** 2:  # a doubtful measurement: count it, but less
            S = H @ P @ H.T + Rz * (np.sqrt(d2) / self.huber)
        K = P @ H.T @ np.linalg.inv(S)
        x, P = x + K @ y, (np.eye(self.n) - K @ H) @ P
        # replay from the capture time up to now: first the rest of the step the frame fell in, then the stored steps
        new = hist[:k + 1] + [(t_cap, x.copy(), P.copy(), hist[k][3], 0.0)]
        rest = hist[k + 1:]
        if rest:
            t_i, _, _, inp, dt_i = rest[0]
            x, P = self._predict_dt(x, P, inp, max(t_i - t_cap, 0.0))
            new.append((t_i, x.copy(), P.copy(), inp, dt_i))
            for t_i, _, _, inp, dt_i in rest[1:]:
                x, P = self._predict_dt(x, P, inp, dt_i)
                new.append((t_i, x.copy(), P.copy(), inp, dt_i))
        self.hist = deque(new, maxlen=self.n_hist)
        self.x, self.P = x, P
        return True

    def state(self):
        """(u, w, vu, vw) now, or None before the first measurement."""
        return None if self.x is None else self.x[:4].copy()

    def bias(self):
        """Estimated in-plane gravity error (unit g), or zeros."""
        return self.x[4:6].copy() if (self.x is not None and self.n == 6) else self.b.copy()
