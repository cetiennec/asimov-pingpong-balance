"""Bounce-stage policies on the sim matched to Albert (identified arm, real camera noise, outliers, dropped frames, flat error,
3D camera model, 34 ms + 20-80 ms arm delay), on the gantry lean and free-standing. One fixed target apex per row.

    python -m tools.eval_bounce outputs/albert_bounce/best.pt [--targets 0.05 0.08 0.12] [--episodes 32] [--seconds 20]
    python -m tools.eval_bounce outputs/albert_bounce/best.pt --video videos/6_albert_bounce_sim.mp4

Per row: success (ball still bouncing or held after the episode), time kept, bounces per second, apex mean +- std (regularity:
std / mean = CV), share of apexes within +-25 % of the target, ball distance from the blade centre, and what the arm did (peak
joint speed, peak blade speed, share of frames the camera lost because the ball was out of the image).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tools.eval_albert import FREE_GRAVITY, REALISTIC

HOME_Q = (0.7577, 0.9729, -0.6564, -1.3960, -1.2616)  # train_albert.sh / train_bounce.sh (Albert's taught flat pose)
GANTRY_GRAVITY = (0.1214, 0.0081, -0.9929)              # albert.sim_robot.SIM_STAND_GRAVITY


# camera assumptions (the arm is Albert's in all: identified, 34 ms + 20-80 ms command lag):
#   albert   Albert's head camera as measured: 18-30 fps, 70-160 ms, 2.6 px centre / 1.2 px radius noise, outliers, lost frames,
#            the arm model's flat error
#   fast     30 fps, 40-60 ms, same noise
#   precise  Albert's timing, centre 1.3 px and radius 0.4 px (a lower-noise detector, or a marker of known size)
#   flat     Albert's camera, an exact arm-model flat
#   clean    all of these at once, no outliers or lost frames: what is left is the predictor's own error
#   ideal    the true ball state
CAMERAS = {"albert": dict(cam_fps=(18, 30), cam_latency=(0.07, 0.16)), "fast": dict(cam_fps=(30, 30), cam_latency=(0.04, 0.06)),
           "precise": dict(cam_fps=(18, 30), cam_latency=(0.07, 0.16), cam_noise_px=(1.3, 0.4)),
           "flat": dict(cam_fps=(18, 30), cam_latency=(0.07, 0.16), tilt_bias=0.0),
           "clean": dict(cam_fps=(30, 30), cam_latency=(0.04, 0.06), cam_noise_px=(1.3, 0.4), tilt_bias=0.0, cam_outliers=0.0,
                         cam_drops=0.0),
           "ideal": dict(camera_obs=False)}


def env_kwargs(target, grav, realistic=True, camera="albert"):
    kw = dict(camera_obs=True, home_q=HOME_Q, stand_gravity=GANTRY_GRAVITY if grav == "gantry" else FREE_GRAVITY,
              ball_start=(0.03, 0.10), estimator="kf3", act_latency=(1, 4), bounce=(target, target), p_rest=0.0,
              **(REALISTIC if realistic else {}))
    kw.update(CAMERAS[camera])
    return kw


def episode(args):
    ckpt, target, grav, seed, seconds, camera = args
    import mujoco
    from asimov_pp.env import BalanceEnv, Task
    from asimov_pp.ppo import load_policy
    pol = load_policy(ckpt)
    e = BalanceEnv(Task(max_steps=int(seconds / 0.02)), seed=20_000 + seed, **env_kwargs(target, grav, camera=camera))
    o = e.reset(); dist, qd, vb, lost, seen = [], [], [], 0, 0
    n_cam = [0, 0]
    orig = e._cam_detect

    def counted():
        det = orig(); n_cam[0] += 1; n_cam[1] += det is None
        return det
    e._cam_detect = counted
    while True:
        o, _, term, trunc, info = e.step(np.clip(pol(o), -1, 1))
        rel, _ = e.ball_in_blade(); dist.append(np.hypot(*rel[:2]))
        qd.append(np.abs(e.d.qvel[e.v_adr]))
        v = np.empty(6); mujoco.mj_objectVelocity(e.m, e.d, mujoco.mjtObj.mjOBJ_GEOM, e.blade_geom, v, 0)
        vb.append(np.linalg.norm(v[3:]))
        if term or trunc:
            t = e.t * 0.02
            return dict(ok=not term, t=t, bounces=info["n_bounces"] / t, apexes=info["apexes"], dist=float(np.mean(dist)),
                        qd=np.max(qd, 0), vb=float(np.percentile(vb, 99)), lost=n_cam[1] / max(n_cam[0], 1))


def evaluate(ex, ckpt, target, grav, episodes, seconds, camera="albert"):
    res = list(ex.map(episode, [(ckpt, target, grav, s, seconds, camera) for s in range(episodes)]))
    ap = np.concatenate([r["apexes"][1:] for r in res if len(r["apexes"]) > 1])  # the first apex is the drop's
    return dict(success=100 * np.mean([r["ok"] for r in res]), time=np.mean([r["t"] for r in res]),
                bounces=np.mean([r["bounces"] for r in res]), apex=ap.mean() if len(ap) else np.nan,
                apex_std=ap.std() if len(ap) else np.nan, within=100 * np.mean(np.abs(ap - target) < 0.25 * target) if len(ap) else 0,
                dist=100 * np.mean([r["dist"] for r in res]), qd=np.max([r["qd"] for r in res], 0),
                vb=np.mean([r["vb"] for r in res]), lost=100 * np.mean([r["lost"] for r in res]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--targets", type=float, nargs="+", default=(0.05, 0.08, 0.12))
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--cameras", nargs="+", default=["albert"], choices=list(CAMERAS), help="camera assumptions to evaluate")
    p.add_argument("--leans", nargs="+", default=["gantry", "free"], choices=["gantry", "free"])
    p.add_argument("--video", default=None, help="also render one robot-matched episode per target into this mp4")
    a = p.parse_args()
    print("| policy | camera | lean | target apex | success | time kept | bounces/s | apex (mean +- std) | CV | apex within +-25 % "
          "| ball from centre | peak joint speed (rad/s) | blade speed p99 | camera frames out of image |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    with ProcessPoolExecutor(a.workers) as ex:
        for ck in a.ckpts:
            for cam in a.cameras:
              for grav in a.leans:
                for T in a.targets:
                    r = evaluate(ex, ck, T, grav, a.episodes, a.seconds, cam)
                    print(f"| {ck.split('/')[-2]} | {cam} | {grav} | {100 * T:.0f} cm | {r['success']:.0f} % | {r['time']:.1f} s | {r['bounces']:.2f} "
                          f"| {100 * r['apex']:.1f} +- {100 * r['apex_std']:.1f} cm | {r['apex_std'] / r['apex']:.2f} | {r['within']:.0f} % "
                          f"| {r['dist']:.1f} cm | {' '.join(f'{x:.1f}' for x in r['qd'])} | {r['vb']:.2f} m/s | {r['lost']:.0f} % |",
                          flush=True)
    if a.video:
        import imageio
        from asimov_pp.ppo import load_policy
        from asimov_pp.render import render_episode
        pol = load_policy(a.ckpts[0])
        frames = []
        for i, T in enumerate(a.targets):
            f, info = render_episode(lambda o: np.clip(pol(o), -1, 1), "stand", min(a.seconds, 12), 7 + i,
                                     env_kwargs=env_kwargs(T, "free"), label=f"bounce, target {100 * T:.0f} cm (free-standing, robot-matched sim)")
            print(f"video, target {100 * T:.0f} cm: {info['n_bounces']} bounces, {'kept' if info['is_success'] else 'lost'}")
            frames += f + [f[-1]] * 10
        imageio.mimwrite(a.video, frames, fps=25, quality=8, macro_block_size=8)
        print("wrote", a.video)


if __name__ == "__main__":
    main()
