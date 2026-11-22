"""Train the paddle controller with PPO, one curriculum stage at a time (each stage starts from the previous one's weights):

    uv run python -m asimov_pp.train --stage balance --iters 300
    uv run python -m asimov_pp.train --stage perturb --iters 400 --init outputs/balance/best.pt
    uv run python -m asimov_pp.train --stage move    --iters 600 --init outputs/perturb/best.pt
    uv run python -m asimov_pp.train --stage bounce --bounce 0.04 0.10 --estimator kf3 --camera-obs ... (train_bounce.sh)

One iteration: rollout (steps x all envs) -> GAE -> PPO update -> log -> checkpoint. Writes outputs/<stage>/.
Loop structure adapted from zecice/so101-pingpong-balance (MIT), sim/train_controller.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from asimov_pp.ppo import ActorCritic, compute_gae, ppo_update
from asimov_pp.vec_env import VecBalanceEnv

OUT = Path(__file__).resolve().parent.parent / "outputs"


def save(obj, path: Path):
    tmp = path.with_suffix(".tmp"); torch.save(obj, tmp); tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["balance", "perturb", "move", "bounce"], default="balance")
    p.add_argument("--bounce", type=float, nargs=2, default=(0.04, 0.10), help="bounce stage: target apex range above the blade, m")
    p.add_argument("--p-rest", type=float, default=0.3, help="bounce stage: share of episodes starting with the ball at rest")
    p.add_argument("--privileged-critic", action="store_true", help="the critic also sees the sim's truth (asymmetric actor-critic)")
    p.add_argument("--name", default=None, help="run directory name (default: the stage)")
    p.add_argument("--init", default=None, help="start from this checkpoint (previous stage)")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--envs-per-worker", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--ball-noise", type=float, default=1.0)
    p.add_argument("--max-latency", type=int, default=2)
    p.add_argument("--smooth-penalty", type=float, default=0.01, help="reward -= this * sum((a - a_prev)^2)")
    p.add_argument("--air-penalty", type=float, default=0.0, help="reward -= this when the ball is > 3 mm above the blade")
    p.add_argument("--camera-obs", action="store_true", help="ball seen through the head-camera model: 25-30 fps, 40-120 ms latency")
    p.add_argument("--cam-fps", type=float, nargs=2, default=(25.0, 30.0), help="camera model: frame rate range")
    p.add_argument("--cam-latency", type=float, nargs=2, default=(0.04, 0.12), help="camera model: latency range, s")
    p.add_argument("--home", choices=["default", "albert"], default="default", help="arm home pose (albert: paddle higher, in view)")
    p.add_argument("--home-q", type=float, nargs=5, default=None, help="arm home pose, model angles (overrides --home)")
    p.add_argument("--stand-gravity", type=float, nargs=3, default=None, help="standing body leans so its IMU reads this gravity")
    p.add_argument("--ball-start", type=float, nargs="+", default=(0.02, 0.03), help="ball start: radius (m), speed per axis (m/s)[, drop height (m)]")
    p.add_argument("--estimator", choices=["fd", "kf", "kf3"], default="fd",
                   help="camera model: ball from the last frame (fd), a Kalman predictor on the blade (kf) or in 3D (kf3, bounce)")
    p.add_argument("--act-latency", type=int, nargs=2, default=None, help="extra command delay, control steps (min max)")
    p.add_argument("--arm-dyn", nargs="+", default=None, help="identified arm dynamics (albert/arm_dynamics.json, tools/sysid_arm.py); "
                   "several: one per episode")
    p.add_argument("--cam-outliers", type=float, default=0.0, help="up to this share of camera detections 1-3 cm off")
    p.add_argument("--cam-drops", type=float, default=0.0, help="up to this share of frames lost (and gaps 0.15-0.4 s)")
    p.add_argument("--tilt-bias", type=float, default=0.0, help="std of the arm model's flat error, unit g, per episode")
    p.add_argument("--stand-gravity2", type=float, nargs=3, default=None, help="second body lean: each episode between the two")
    p.add_argument("--min-std", type=float, default=0.25, help="on --init, raise the policy std to at least this (re-explore)")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed); torch.set_num_threads(2)
    run_dir = OUT / (a.name or a.stage); run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=1))
    bounce = a.stage == "bounce"
    env = VecBalanceEnv(a.workers, a.envs_per_worker, seed=a.seed * 100000, task="balance" if bounce else a.stage,
                        bounce=tuple(a.bounce) if bounce else None, p_rest=a.p_rest, privileged=a.privileged_critic, ball_noise=a.ball_noise, max_latency=a.max_latency,
                        smooth_penalty=a.smooth_penalty, air_penalty=a.air_penalty, camera_obs=a.camera_obs,
                        cam_fps=a.cam_fps, cam_latency=a.cam_latency, home=a.home,
                        home_q=a.home_q, stand_gravity=a.stand_gravity, ball_start=a.ball_start,
                        estimator=a.estimator, act_latency=a.act_latency, arm_dyn=a.arm_dyn,
                        cam_outliers=a.cam_outliers, cam_drops=a.cam_drops, tilt_bias=a.tilt_bias,
                        stand_gravity2=a.stand_gravity2)
    N, T = env.n_envs, a.steps
    ac = ActorCritic(env.obs_dim, env.act_dim, critic_dim=env.full_dim)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=True)
        sd, own = ck["model"], ac.state_dict()
        for k in ("actor.0.weight", "critic.0.weight"):
            v, w = sd[k], torch.zeros_like(own[k])
            old, new = v.shape[1], env.obs_dim  # the policy's inputs (the critic may have privileged ones after them)
            if old > new:  # a privileged critic loaded into a plain one
                v = v[:, :new]; old = new
            if old < new:  # new observation inputs (inserted before the action history): zero weights at first
                keep = 23 if old == 38 else 25  # 38: before the gait clock (23:25); 40: before the bounce inputs (25:28)
                w[:, :keep] = v[:, :keep]; w[:, new - (old - keep):new] = v[:, keep:]
                print(f"{k}: {old} -> {new} inputs (inputs {keep}:{keep + new - old} added, zero-initialised)")
            else:
                w[:, :new] = v
            if own[k].shape[1] > new and sd[k].shape[1] > env.obs_dim:  # privileged inputs carried over
                w[:, new:] = sd[k][:, new:own[k].shape[1]]
            elif own[k].shape[1] > new:
                print(f"{k}: + {own[k].shape[1] - new} privileged inputs (zero-initialised)")
            sd[k] = w
        ac.load_state_dict(sd)
        with torch.no_grad():
            ac.log_std.clamp_(min=float(np.log(a.min_std)))
        print(f"initialised from {a.init}")
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr, eps=1e-5)
    csv_f = open(run_dir / "log.csv", "w", newline=""); writer = None
    obs = env.reset()
    recent = deque(maxlen=400)
    t0, total, best = time.time(), 0, -1e9
    for it in range(1, a.iters + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * (1 - (it - 1) / a.iters)
        B = dict(obs=np.zeros((T, N, env.full_dim), np.float32), act=np.zeros((T, N, env.act_dim), np.float32),
                 logp=np.zeros((T, N), np.float32), val=np.zeros((T, N), np.float32), rew=np.zeros((T, N), np.float32),
                 done=np.zeros((T, N), np.float32), vnext=np.zeros((T, N), np.float32))
        final_idx, final_obs = [], []
        for t in range(T):
            act, logp, val = ac.act(torch.as_tensor(obs))
            act_np = act.numpy()
            nobs, rew, term, trunc, fobs, eps = env.step(act_np)
            B["obs"][t], B["act"][t], B["logp"][t], B["val"][t] = obs, act_np, logp.numpy(), val.numpy()
            B["rew"][t] = rew * a.reward_scale
            B["done"][t] = term | trunc
            for i in np.flatnonzero(trunc & ~term):
                final_idx.append((t, i)); final_obs.append(fobs[i])
            recent.extend(eps)
            obs = nobs
        total += T * N
        with torch.no_grad():
            v_all = ac.value(torch.as_tensor(np.concatenate([B["obs"][1:], obs[None]]))).numpy()
            B["vnext"][:] = np.where(B["done"] > 0, 0.0, v_all)
            if final_obs:
                for (t, i), v in zip(final_idx, ac.value(torch.as_tensor(np.stack(final_obs))).numpy()):
                    B["vnext"][t, i] = v
        adv, ret = compute_gae(B["rew"], B["val"], B["vnext"], B["done"], a.gamma, a.lam)
        flat = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]))
        s = ppo_update(ac, opt, dict(obs=flat(B["obs"]), act=flat(B["act"]), logp=flat(B["logp"]), adv=flat(adv), ret=flat(ret)))

        nan = float("nan")
        row = dict(iter=it, steps=total, fps=int(total / (time.time() - t0)), std=float(ac.log_std.exp().mean().detach()), **s,
                   ep_return=nan, ep_len=nan, success=nan, success_walk=nan, success_stand=nan,
                   **(dict(bounces=nan, apex_err=nan) if bounce else {}))
        if recent:
            row.update(ep_return=float(np.mean([e["ret"] for e in recent])), ep_len=float(np.mean([e["length"] for e in recent])),
                       success=float(np.mean([e["success"] for e in recent])))
            for key, flag in (("success_walk", True), ("success_stand", False)):
                es = [e["success"] for e in recent if e["walking"] == flag]
                row[key] = float(np.mean(es)) if es else nan
            if bounce:
                row.update(bounces=float(np.mean([e["bounces"] for e in recent])), apex_err=float(np.mean([e["apex_err"] for e in recent])))
        if writer is None:
            writer = csv.DictWriter(csv_f, fieldnames=list(row)); writer.writeheader()
        writer.writerow(row); csv_f.flush()
        if it % 5 == 0 or it == 1:
            eta = (time.time() - t0) / it * (a.iters - it) / 60
            print(f"{a.stage} it {it:4d}/{a.iters} eta {eta:4.1f} min | {total/1e6:6.2f}M steps {row['fps']} fps | ret {row['ep_return']:7.1f} "
                  f"len {row['ep_len']:5.0f} succ {row['success']:.2f} (stand {row['success_stand']:.2f} walk {row['success_walk']:.2f}) "
                  f"| kl {s['kl']:.4f} std {row['std']:.3f}" + (f" | {row['bounces']:.2f} bounces/s apex err {row['apex_err']:.2f}" if bounce else ""),
                  flush=True)
        if it % a.save_every == 0 or it == a.iters:
            ck = dict(model=ac.state_dict(), args=vars(a), iter=it, steps=total, success=row["success"])
            # bounce stage: the best return (survival alone is easy to get by catching the ball and balancing it)
            score = row["ep_return"] / 500 if bounce else row["success"]
            if len(recent) >= 200 and score >= best:
                best = score; save(ck, run_dir / "best.pt")
            save(ck, run_dir / "last.pt")
    env.close()
    print(f"done: {total/1e6:.2f}M steps in {(time.time()-t0)/60:.1f} min, best {"score" if bounce else "success"} {best:.3f}, checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
