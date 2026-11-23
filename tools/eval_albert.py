"""Compare paddle controllers on the sim matched to Albert (identified arm, real camera noise, outliers, dropped frames, flat
error), standing on the gantry. Success = the ball still on the blade after the episode (20 s), with or without disturbances.

    python -m tools.eval_albert outputs/albert_perturb/best.pt outputs/albert_v2A/best.pt ... [--episodes 60]
    python -m tools.eval_albert outputs/albert_v3_final/best.pt mpc --arm-dyn albert/arm_dynamics_20261120.json

A controller is a policy checkpoint, or 'mpc' / 'mpc:key=value,...' (albert/mpc.py, MPCParams; its Kalman filter is given
the blade's acceleration, env kf_paddle_acc; 'kfacc:<ckpt>' gives a policy that filter too). Besides success and time:
ball distance from the centre, and action smoothness (mean |a_t - a_t-1| over steps and joints, action units of 0.03 rad).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REALISTIC = dict(arm_dyn="albert/arm_dynamics.json", cam_outliers=0.10, cam_drops=0.08, tilt_bias=0.04)


FREE_GRAVITY = (0.203, -0.011, -0.980)  # Albert's IMU, free-standing in the centred stance (logs/B4_free.jsonl, 2026-11-20)
SCENARIOS = (("gantry", "balance"), ("free", "balance"), ("free", "perturb"))


def episode(args):
    ckpt, stage, seed, realistic, grav = args[:5]
    arm_dyn = args[5] if len(args) > 5 else None
    from albert.common import HOME
    from albert.mpc import load_controller
    from albert.sim_robot import SIM_STAND_GRAVITY as TAUGHT_GRAVITY  # fixed gantry lean: results stay comparable over time
    from asimov_pp.env import BalanceEnv
    kw = dict(REALISTIC if realistic else {})
    if ckpt.startswith("mpc") or ckpt.startswith("kfacc:"):  # the MPC's Kalman filter knows the blade's acceleration
        kw["kf_paddle_acc"] = True
        ckpt = ckpt[6:] if ckpt.startswith("kfacc:") else ckpt
    if arm_dyn and realistic:
        kw["arm_dyn"] = arm_dyn
    pol = load_controller(ckpt)
    e = BalanceEnv(stage, camera_obs=True, cam_fps=(18, 30), cam_latency=(0.07, 0.16) if realistic else (0.04, 0.15),
                   home_q=HOME, stand_gravity=TAUGHT_GRAVITY if grav == 'gantry' else FREE_GRAVITY, ball_start=(0.05, 0.25, 0.015), estimator="kf",
                   act_latency=(1, 4) if realistic else (2, 5), seed=10_000 + seed, **kw)
    o = e.reset(); n = 0; dist = []; da = []; a_prev = None
    while True:
        a = np.clip(pol(o), -1, 1)
        if a_prev is not None:
            da.append(float(np.mean(np.abs(a - a_prev))))
        a_prev = a
        o, _, term, trunc, _ = e.step(a); n += 1
        rel, _ = e.ball_in_blade(); dist.append(np.hypot(*rel[:2]))
        if term or trunc:
            return (not term), n * 0.02, float(np.mean(dist)), float(np.mean(da)) if da else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--arm-dyn", default=None, help="identified arm for the sim (default: albert/arm_dynamics.json, 2026-11-22)")
    p.add_argument("--scenarios", default="all", help="comma list of gantry-balance,free-balance,free-perturb (default all)")
    a = p.parse_args()
    scen = SCENARIOS if a.scenarios == "all" else [tuple(s.split("-")) for s in a.scenarios.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        for grav, stage in scen:
            for ck in a.ckpts:
                res = list(ex.map(episode, [(ck, stage, s, True, grav, a.arm_dyn) for s in range(a.episodes)]))
                ok, L, d, da = (np.array([r[i] for r in res]) for i in range(4))
                print(f"{grav:6s} {stage:8s} {ck:36s} success {100 * ok.mean():5.1f} %  mean time {L.mean():5.1f} s  ball {100 * d.mean():.1f} cm from centre"
                      f"  action change {da.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
