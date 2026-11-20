"""Video of a checkpoint: the env runs the physics, its state is copied onto the full 25-DoF robot for rendering.

    uv run python -m asimov_pp.render outputs/move/best.pt --task "walk + all" --seconds 10 --out videos/walk.mp4
Several --task values are rendered one after the other into the same video.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from asimov_pp.env import EVAL_TASKS, STAGES, BalanceEnv, walk_clip
from asimov_pp.model import ARM_JOINTS, build, standing_qpos

W, H = 960, 540


def _font(size):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_episode(ac, task_name, seconds, seed, cam_mode="follow", camera_obs=False, env_kwargs=None, label=None):
    """env_kwargs: BalanceEnv arguments (e.g. the robot-matched bounce sim of tools/eval_bounce.py); label: the title."""
    task = EVAL_TASKS.get(task_name) or STAGES[task_name]
    task = type(task)(**{**task.__dict__, "max_steps": int(seconds / 0.02)})
    env = BalanceEnv(task, seed=seed, **({"camera_obs": camera_obs} if env_kwargs is None else env_kwargs))
    rm = build("render"); rd = mujoco.MjData(rm)
    rend = mujoco.Renderer(rm, H, W)
    stand = standing_qpos(rm)
    clip = walk_clip(); names = clip["names"]
    jadr = {n: rm.joint(n).qposadr[0] for n in names}
    arm_adr = [rm.joint(j).qposadr[0] for j in ARM_JOINTS]
    b_adr = rm.joint("ball_joint").qposadr[0]
    font, small = _font(26), _font(18)
    o, frames, done, falls = env.reset(), [], False, 0
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation = 1.45, -20
    yaw0 = None
    kicks = sorted(env._kicks); shoves = sorted(env._shoves)
    while not done:
        o, r, te, tr, info = env.step(ac(o))
        done = te or tr
        pos, quat, waist, cf = env.base_state()
        q = stand.copy()
        if cf is not None:
            i = int(np.floor(cf)); f = cf - i
            for k, n in enumerate(names):
                q[jadr[n]] = (1 - f) * clip["joints"][i, k] + f * clip["joints"][i + 1, k]
        q[0:3], q[3:7] = pos, quat
        q[jadr["waist_yaw_joint"]] = waist
        q[arm_adr] = env.d.qpos[env.q_adr]
        q[b_adr:b_adr + 7] = env.d.qpos[env.ball_q:env.ball_q + 7]
        if cf is not None:  # put the feet on the floor (the clip root height is for its own model)
            rd.qpos[:] = q; mujoco.mj_kinematics(rm, rd)
            low = min(rd.geom_xpos[rm.geom(g).id][2] for g in [f"{s}_foot{k}_collision" for s in ("left", "right") for k in range(1, 5)])
            dz = low - 0.005
            q[2] -= dz; q[b_adr + 2] -= dz
        rd.qpos[:] = q; mujoco.mj_forward(rm, rd)
        if env.t % 2:  # 25 fps video
            continue
        yaw = 2 * np.arctan2(quat[3], quat[0])
        yaw0 = yaw if yaw0 is None else yaw0 + 0.05 * np.angle(np.exp(1j * (yaw - yaw0)))
        cam.lookat[:] = [pos[0], pos[1], 0.82]
        cam.azimuth = np.degrees(yaw0) - 140
        rend.update_scene(rd, cam)
        img = Image.fromarray(rend.render()); dr = ImageDraw.Draw(img)
        dr.text((20, 16), label or task_name, font=font, fill=(255, 255, 255))
        t = env.t * 0.02
        if "n_bounces" in info:
            ap = info["apexes"][-1] if info["apexes"] else float("nan")
            dr.text((20, 50), f"t = {t:4.1f} s   bounces {info['n_bounces']:3d}   last apex {100 * ap:4.1f} cm (target {100 * info['target']:.0f})"
                    f"   ball {100 * info['dist']:4.1f} cm from centre", font=small, fill=(230, 230, 230))
        else:
            dr.text((20, 50), f"t = {t:4.1f} s   ball {100 * info['dist']:4.1f} cm from centre", font=small, fill=(230, 230, 230))
        ev = [("ball kick", k) for k in kicks if 0 <= env.t - k < 25] + [("paddle shove", k) for k in shoves if 0 <= env.t - k < 25] + \
             [("torso push", k) for k in env.push_steps if 0 <= env.t - k < 25]
        if ev:
            dr.text((20, 76), " + ".join(e for e, _ in ev), font=small, fill=(255, 190, 60))
        if te:
            dr.text((20, 102), "ball lost", font=font, fill=(255, 80, 80))
        frames.append(np.asarray(img))
    rend.close()
    return frames, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--task", action="append", default=None)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="videos/demo.mp4")
    p.add_argument("--camera-obs", action="store_true", help="ball seen through the head-camera model (for cam_* policies)")
    p.add_argument("--gif", action="store_true", help="also write a small GIF next to the mp4")
    a = p.parse_args()
    from asimov_pp.ppo import load_policy

    ac = load_policy(a.ckpt)
    frames = []
    for i, t in enumerate(a.task or ["stand"]):
        f, info = render_episode(ac, t, a.seconds, a.seed + i, camera_obs=a.camera_obs)
        print(f"{t}: {'balanced' if info['is_success'] else 'ball lost'} ({len(f)} frames)")
        frames += f + [f[-1]] * 10
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(a.out, frames, fps=25, quality=8, macro_block_size=8)
    if a.gif:
        imageio.mimwrite(Path(a.out).with_suffix(".gif"), [fr[::3, ::3] for fr in frames[::2]], duration=80, loop=0)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
