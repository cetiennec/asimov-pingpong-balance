"""Fit the simulated right arm to Albert's: replay the setpoints sent in a real session (albert_run.jsonl) through the
training environment's arm and compare joint angles with the ones Albert reported.

Parameters: command delay (s, applied to the setpoints' send times), per joint kp / kd scale and Coulomb friction (Nm).
    python -m tools.sysid_arm [albert_run.jsonl] [out.json, default albert/arm_dynamics.json]
"""
from __future__ import annotations

import json
import sys

import mujoco
import numpy as np

from albert.common import ARM_IDX, HOME, TAUGHT_GRAVITY, fw_to_mjcf
from asimov_pp.env import BalanceEnv, Task
from tools.load_session import load

SUB = 0.002


class Replay:
    def __init__(self, path="data/2026-11-19/albert_run.jsonl", t_from=3.0, t_to=None):
        S = load(path)
        self.ct, self.cmd = S["cmd_t"], np.array([fw_to_mjcf(c[ARM_IDX]) for c in S["cmd"]])
        self.st, self.q = S["st_t"], np.array([fw_to_mjcf(q[ARM_IDX]) for q in S["q"]])
        t0 = self.ct[0] + t_from
        t1 = self.ct[-1] - 3.5 if t_to is None else self.ct[0] + t_to   # the last 3 s: set_joints back to start
        self.t0, self.t1 = t0, t1
        self.env = BalanceEnv(Task(max_steps=10**6), domain_rand=False, home_q=HOME, stand_gravity=TAUGHT_GRAVITY, seed=0)
        self.env.reset()
        e = self.env
        self.nom = dict(kp=e.m.actuator_gainprm[e.act_ids, 0].copy(), kv=-e.m.actuator_biasprm[e.act_ids, 2].copy(),
                        fl=e.m.dof_frictionloss[e.v_adr].copy(), arm=e.m.dof_armature[e.v_adr].copy())
        e.d.qpos[e.ball_q:e.ball_q + 3] = [0, 0, -5]

    def run(self, delay, kp_s, kd_s, fric, arm_s=None):
        e, m, d = self.env, self.env.m, self.env.d
        m.actuator_gainprm[e.act_ids, 0] = self.nom["kp"] * kp_s
        m.actuator_biasprm[e.act_ids, 1] = -self.nom["kp"] * kp_s
        m.actuator_biasprm[e.act_ids, 2] = -self.nom["kv"] * kd_s
        m.dof_frictionloss[e.v_adr] = fric
        if arm_s is not None:
            m.dof_armature[e.v_adr] = self.nom["arm"] * arm_s
        mujoco.mj_resetData(m, d)
        e._set_base(0)
        i0 = np.searchsorted(self.st, self.t0)
        d.qpos[e.q_adr] = self.q[i0]; d.qvel[e.v_adr] = 0
        d.qpos[e.ball_q:e.ball_q + 3] = [0, 0, -5]
        mujoco.mj_forward(m, d)
        ts = np.arange(self.t0, self.t1, SUB)
        idx = np.clip(np.searchsorted(self.ct, ts - delay, side="right") - 1, 0, len(self.ct) - 1)
        out = np.empty((len(ts), 5))
        for k, t in enumerate(ts):
            if k % 10 == 0:
                e._set_base(0)
            d.ctrl[e.act_ids] = self.cmd[idx[k]]
            mujoco.mj_step(m, d)
            out[k] = d.qpos[e.q_adr]
        return ts, out

    def error(self, params, joints=range(5)):
        ts, out = self.run(*params)
        sel = (self.st >= ts[0]) & (self.st <= ts[-1])
        sim = np.array([np.interp(self.st[sel], ts, out[:, j]) for j in range(5)]).T
        return sim - self.q[sel]


def unpack(x):
    return x[0], np.exp(x[1:6]), np.exp(x[6:11]), np.abs(x[11:16])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/2026-11-19/albert_run.jsonl"
    from scipy.optimize import least_squares
    S = load(path)
    dur = S["cmd_t"][-1] - S["cmd_t"][0]
    split = 3.0 + 0.6 * (dur - 6.5)  # fit on the first 60 % of the motion, check on the rest
    fit = Replay(path, t_from=3.0, t_to=split)
    val = Replay(path, t_from=split)
    nominal = (0.060, np.ones(5), np.ones(5), np.zeros(5))
    def rms(r, p):
        return np.sqrt(np.mean(r.error(p) ** 2, axis=0))
    print("nominal sim (60 ms delay)  rms per joint (mrad):", np.round(1000 * rms(val, nominal)), flush=True)
    for dl in (0.0, 0.04, 0.08, 0.12):
        print(f"  delay {1000*dl:.0f} ms:", np.round(1000 * rms(fit, (dl, np.ones(5), np.ones(5), np.zeros(5)))), flush=True)
    x0 = np.r_[0.08, np.zeros(5), np.zeros(5), 0.3 * np.ones(5)]
    res = least_squares(lambda x: fit.error(unpack(x))[::4].ravel(), x0, diff_step=[1e-2] + [5e-2] * 10 + [5e-2] * 5,
                        bounds=(np.r_[0.0, -1.5 * np.ones(10), np.zeros(5)], np.r_[0.25, 1.0 * np.ones(10), 3.0 * np.ones(5)]),
                        max_nfev=250, verbose=0)
    p = unpack(res.x)
    print("fitted: delay %.0f ms" % (1000 * p[0]), "kp x", np.round(p[1], 2), "kd x", np.round(p[2], 2), "friction Nm", np.round(p[3], 2))
    print("fitted sim rms on held-out part (mrad):", np.round(1000 * rms(val, p)))
    out = sys.argv[2] if len(sys.argv) > 2 else "albert/arm_dynamics.json"
    json.dump(dict(delay=float(p[0]), kp_scale=p[1].tolist(), kd_scale=p[2].tolist(), friction=p[3].tolist(),
                   rms_val_mrad=(1000 * rms(val, p)).tolist(), rms_nominal_mrad=(1000 * rms(val, nominal)).tolist(), source=path),
              open(out, "w"), indent=1)
    print("wrote", out, "(training and the simulated Albert read albert/arm_dynamics.json)")


if __name__ == "__main__":
    main()
