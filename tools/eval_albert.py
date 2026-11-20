"""Compare paddle policies on the sim matched to Albert (identified arm, real camera noise, outliers, dropped frames, flat
error), standing on the gantry. Success = the ball still on the blade after the episode (20 s), with or without disturbances.

    python -m tools.eval_albert outputs/albert_perturb/best.pt outputs/albert_v2A/best.pt ... [--episodes 60]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REALISTIC = dict(arm_dyn="albert/arm_dynamics.json", cam_outliers=0.10, cam_drops=0.08, tilt_bias=0.04)


FREE_GRAVITY = (0.203, -0.011, -0.980)  # Albert's IMU, free-standing in the centred stance (logs/B4_free.jsonl, 2026-11-20)


def episode(args):
    ckpt, stage, seed, realistic, grav = args
    from albert.common import HOME
    from albert.sim_robot import SIM_STAND_GRAVITY as TAUGHT_GRAVITY  # fixed gantry lean: results stay comparable over time
    from asimov_pp.env import BalanceEnv
    from asimov_pp.ppo import load_policy
    pol = load_policy(ckpt)
    e = BalanceEnv(stage, camera_obs=True, cam_fps=(18, 30), cam_latency=(0.07, 0.16) if realistic else (0.04, 0.15),
                   home_q=HOME, stand_gravity=TAUGHT_GRAVITY if grav == 'gantry' else FREE_GRAVITY, ball_start=(0.05, 0.25, 0.015), estimator="kf",
                   act_latency=(1, 4) if realistic else (2, 5), seed=10_000 + seed, **(REALISTIC if realistic else {}))
    o = e.reset(); n = 0; dist = []
    while True:
        o, _, term, trunc, _ = e.step(np.clip(pol(o), -1, 1)); n += 1
        rel, _ = e.ball_in_blade(); dist.append(np.hypot(*rel[:2]))
        if term or trunc:
            return (not term), n * 0.02, float(np.mean(dist))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        for grav, stage in (("gantry", "balance"), ("free", "balance"), ("free", "perturb")):
            for ck in a.ckpts:
                res = list(ex.map(episode, [(ck, stage, s, True, grav) for s in range(a.episodes)]))
                ok, L, d = np.array([r[0] for r in res]), np.array([r[1] for r in res]), np.array([r[2] for r in res])
                print(f"{grav:6s} {stage:8s} {ck:36s} success {100 * ok.mean():5.1f} %  mean time {L.mean():5.1f} s  ball {100 * d.mean():.1f} cm from centre",
                      flush=True)


if __name__ == "__main__":
    main()
