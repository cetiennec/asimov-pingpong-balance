"""Where is the paddle in the head camera? Arm at the run's home pose, legs in the stance, the neck sweeps a 4 x 3 grid of
pitch / yaw; at each the script saves the camera image and counts red blade pixels. Writes logs/aim_sweep.png (a mosaic, the
blade outlined) and prints where the blade is best seen, next to the neck angles run_policy aims with. Rope taut, no ball.

    python -m albert.aim_sweep --supported
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from albert.ball_camera import BALL_WHITE, masks
from albert.common import (CENTRED_STANCE, FLAT_HOME_FW, HOME, NECK_PITCH_IDX, NECK_YAW_IDX, WAIST_IDX, ARM_IDX, KD, KP,
                           BalanceOnExit, Kinematics, TiltGuard, fw_to_mjcf, graceful_signals, hand_back, look_at, mjcf_to_fw,
                           safe_pose, take_joints, to_balance, with_stance)

PITCHES = (-0.46, -0.30, -0.15, 0.0)
YAWS = (-0.95, -0.65, -0.35)


def main():
    graceful_signals()
    p = argparse.ArgumentParser()
    p.add_argument("--supported", action="store_true")
    p.add_argument("--stance", type=float, nargs=3, default=list(CENTRED_STANCE))
    p.add_argument("--out", default="logs/aim_sweep.png")
    a = p.parse_args()
    if not a.supported:
        sys.exit("Rope taut, then --supported.")
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from menlo.asimov import Robot
    tiles, found = [], []
    with Robot().connect("hybrid") as robot, BalanceOnExit(robot):
        if robot.get_state().faulted:
            sys.exit("Albert reports a fault: not moving.")
        take_joints(robot)
        start = safe_pose(robot.get_state().joint_pos)
        hold = with_stance(start, a.stance)
        hold[ARM_IDX] = FLAT_HOME_FW if FLAT_HOME_FW is not None else mjcf_to_fw(HOME)
        input("Rope TAUT, no ball, E-Stop in reach. Enter: legs to the stance, arm home (3 s), then the head sweeps (15 s) ")
        robot.set_joints(hold, duration=3.0, wait=False, kp=KP, kd=KD); time.sleep(3.5)
        st = robot.get_state()
        aim = look_at(Kinematics(), fw_to_mjcf(hold[ARM_IDX]), waist=hold[WAIST_IDX], base_quat=st.quat)
        print(f"run_policy would aim the neck at yaw {aim[0]:+.2f}, pitch {aim[1]:+.2f}", flush=True)
        with TiltGuard(robot, on_trip=lambda: to_balance(robot), max_abs=25.0):
            for pi in PITCHES:
                for yi in YAWS:
                    cmd = hold.copy(); cmd[NECK_YAW_IDX], cmd[NECK_PITCH_IDX] = yi, pi
                    robot.set_joints(cmd, duration=0.5, wait=False, kp=KP, kd=KD); time.sleep(0.9)
                    f = robot.camera.latest()
                    img = np.asarray(f.to_numpy())[..., :3] if f is not None else np.zeros((720, 1280, 3), np.uint8)
                    blade, _, _ = masks(img, BALL_WHITE, step=4)
                    n = int(blade.sum())
                    q = np.array(robot.get_state().joint_pos)
                    found.append((n, yi, pi, q[NECK_YAW_IDX], q[NECK_PITCH_IDX]))
                    t = img[::4, ::4].copy()
                    t[blade] = (0.5 * t[blade] + [0, 0, 127]).astype(np.uint8)  # blade region tinted blue
                    tiles.append((t, f"yaw {yi:+.2f} pitch {pi:+.2f}: {n} px"))
                    print(f"  neck yaw {yi:+.2f} pitch {pi:+.2f} (measured {q[NECK_YAW_IDX]:+.2f} {q[NECK_PITCH_IDX]:+.2f}): blade {n} px",
                          flush=True)
        hand_back(robot)
    from PIL import Image, ImageDraw
    h, w = tiles[0][0].shape[:2]
    mos = Image.new("RGB", (w * len(YAWS), h * len(PITCHES)))
    for k, (t, label) in enumerate(tiles):
        im = Image.fromarray(t); ImageDraw.Draw(im).text((4, 4), label, fill=(255, 255, 0))
        mos.paste(im, ((k % len(YAWS)) * w, (k // len(YAWS)) * h))
    mos.save(a.out)
    best = max(found)
    print(f"blade best seen at neck yaw {best[1]:+.2f}, pitch {best[2]:+.2f} ({best[0]} px); mosaic: {a.out}")


if __name__ == "__main__":
    main()
