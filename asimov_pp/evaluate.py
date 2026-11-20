"""Success rate (ball still on the blade after 10 s) of a checkpoint in each evaluation scenario, with the deterministic policy.

    uv run python -m asimov_pp.evaluate outputs/move/best.pt --episodes 100
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np


def _run(args):
    ckpt, name, episodes, seed, camera_obs = args
    import torch

    from asimov_pp.env import EVAL_TASKS, BalanceEnv
    from asimov_pp.ppo import load_policy

    torch.set_num_threads(1)
    policy = load_policy(ckpt)
    env = BalanceEnv(EVAL_TASKS[name], seed=seed, camera_obs=camera_obs)
    succ, dists, lens = [], [], []
    for _ in range(episodes):
        o, done, ds = env.reset(), False, []
        while not done:
            o, r, te, tr, info = env.step(policy(o))
            ds.append(info["dist"]); done = te or tr
        succ.append(info["is_success"]); dists.append(np.mean(ds)); lens.append(env.t)
    return name, float(np.mean(succ)), float(np.mean(dists)) * 100, float(np.mean(lens)) * 0.02


def evaluate(ckpt, episodes=100, seed=123, camera_obs=False):
    from asimov_pp.env import EVAL_TASKS

    with mp.get_context("spawn").Pool(len(EVAL_TASKS)) as pool:
        return pool.map(_run, [(ckpt, n, episodes, seed, camera_obs) for n in EVAL_TASKS])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--json", default=None)
    p.add_argument("--camera-obs", action="store_true", help="ball seen through the head-camera model (25-30 fps, 40-120 ms)")
    a = p.parse_args()
    res = evaluate(a.ckpt, a.episodes, camera_obs=a.camera_obs)
    ball = "head-camera model 25-30 fps, 40-120 ms" if a.camera_obs else "0-40 ms"
    print(f"{a.ckpt}  ({a.episodes} episodes of 10 s per scenario, randomized dynamics, action latency 0-40 ms, ball {ball}, noisy estimate)")
    print(f"{'scenario':<15}{'success':>9}{'mean dist':>11}{'mean time':>11}")
    for n, s, d, t in res:
        print(f"{n:<15}{100 * s:8.1f}%{d:9.1f} cm{t:9.1f} s")
    if a.json:
        json.dump({n: dict(success=s, mean_dist_cm=d, mean_time_s=t) for n, s, d, t in res}, open(a.json, "w"), indent=1)
