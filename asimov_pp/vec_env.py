"""Parallel BalanceEnvs in worker processes (one pipe message per worker per step). Adapted from zecice/so101-pingpong-balance (MIT).

step() auto-resets finished envs and returns (obs, rew, terminated, truncated, final_obs, episodes).
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _worker(conn, env_kwargs, seeds):
    from asimov_pp.env import BalanceEnv

    envs = [BalanceEnv(seed=s, **env_kwargs) for s in seeds]
    ret, length = np.zeros(len(envs)), np.zeros(len(envs), int)
    for e in envs:
        e.reset()
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            conn.send(np.stack([e.reset() for e in envs]))
        elif cmd == "step":
            obs, rew, term, trunc, final, eps = [], [], [], [], [], []
            for i, (e, a) in enumerate(zip(envs, data)):
                o, r, te, tr, info = e.step(a)
                ret[i] += r; length[i] += 1
                fo = np.zeros_like(o)
                if te or tr:
                    fo = o
                    ep = dict(ret=ret[i], length=int(length[i]), success=bool(info["is_success"]), walking=bool(info["walking"]))
                    if "n_bounces" in info:  # bounce stage: bounces per second, apex error relative to the target
                        ap = np.array(info["apexes"])
                        ep.update(bounces=info["n_bounces"] / (length[i] * 0.02),
                                  apex_err=float(np.mean(np.abs(ap - info["target"]) / info["target"])) if len(ap) else 1.0)
                    eps.append(ep)
                    ret[i], length[i] = 0.0, 0
                    o = e.reset()
                obs.append(o); rew.append(r); term.append(te); trunc.append(tr); final.append(fo)
            conn.send((np.stack(obs), np.array(rew, np.float32), np.array(term), np.array(trunc), np.stack(final), eps))
        elif cmd == "close":
            conn.close()
            return


class VecBalanceEnv:
    def __init__(self, n_workers=16, envs_per_worker=16, seed=0, **env_kwargs):
        from asimov_pp.env import BOUNCE_OBS_DIM, OBS_DIM, PRIV_DIM

        ctx = mp.get_context("spawn")
        self.n_envs, self.per = n_workers * envs_per_worker, envs_per_worker
        self.conns, self.procs = [], []
        for w in range(n_workers):
            parent, child = ctx.Pipe()
            seeds = [seed + w * envs_per_worker + k for k in range(envs_per_worker)]
            p = ctx.Process(target=_worker, args=(child, env_kwargs, seeds), daemon=True)
            p.start(); child.close()
            self.conns.append(parent); self.procs.append(p)
        self.obs_dim, self.act_dim = (OBS_DIM if env_kwargs.get("bounce") is None else BOUNCE_OBS_DIM), 5
        self.full_dim = self.obs_dim + (PRIV_DIM if env_kwargs.get("privileged") else 0)  # what step() returns

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions):
        actions = np.asarray(actions, np.float32).reshape(self.n_envs, self.act_dim)
        for i, c in enumerate(self.conns):
            c.send(("step", actions[i * self.per:(i + 1) * self.per]))
        out = [c.recv() for c in self.conns]
        obs, rew, term, trunc, final = (np.concatenate([o[k] for o in out]) for k in range(5))
        return obs, rew, term, trunc, final, [ep for o in out for ep in o[5]]

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
