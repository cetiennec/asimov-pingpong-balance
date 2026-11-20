"""Measure the head-camera latency on Albert: the wrist yaw wiggles +-0.1 rad at 1 Hz for 6 s while the camera watches the
paddle; the delay that best lines up the blade's motion in the images with the joint's measured motion is the latency
(capture -> this script). Writes it to albert/camera_latency.json; run_policy.py uses it. Albert must hang from its gantry.

    python -m albert.measure_latency --supported
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from albert.ball_camera import masks
from albert.common import to_balance, graceful_signals, with_stance, take_joints, hand_back, BalanceOnExit, ARM_IDX, BALL, FLAT_HOME_FW, HOME, NECK_PITCH_IDX, NECK_YAW_IDX, ROOT, TiltGuard, Kinematics, look_at, mjcf_to_fw, fw_to_mjcf, safe_pose
from albert.ball_camera import BALL_ORANGE, BALL_WHITE


def main():
    graceful_signals()
    p = argparse.ArgumentParser()
    p.add_argument("--supported", action="store_true")
    a = p.parse_args()
    if not a.supported:
        sys.exit("Hang Albert from its gantry, then run again with --supported.")
    from menlo.asimov import Robot

    ball = BALL_WHITE if BALL == "white" else BALL_ORANGE
    kin = Kinematics()
    with Robot().connect("hybrid") as robot, BalanceOnExit(robot):
        if robot.get_state().faulted:
            sys.exit("Albert reports a fault: not moving.")
        take_joints(robot)
        st = robot.get_state()
        start = safe_pose(st.joint_pos)
        hold = with_stance(start)
        input("Rope TAUT (carrying part of Albert's weight), E-Stop in reach. Enter: legs to the centred stance, arm home (3 s) ")
        hold[ARM_IDX] = FLAT_HOME_FW if FLAT_HOME_FW is not None else mjcf_to_fw(HOME)
        hold[NECK_YAW_IDX], hold[NECK_PITCH_IDX] = look_at(kin, fw_to_mjcf(hold[ARM_IDX]), base_quat=st.quat or (1, 0, 0, 0))
        print("arm to the home pose (3 s)")
        robot.set_joints(hold, duration=3.0, wait=False); time.sleep(3.5)
        # aim the head again with the body as it now hangs (the stance and the rope change the lean by up to ~15 deg)
        stq = robot.get_state()
        hold[NECK_YAW_IDX], hold[NECK_PITCH_IDX] = look_at(kin, fw_to_mjcf(hold[ARM_IDX]), base_quat=stq.quat or (1, 0, 0, 0))
        robot.set_joints(hold, duration=0.8, wait=False); time.sleep(1.0)
        # the guard's reference lean is taken now, arm at home: on the gantry the arm forward tips the hanging body ~9 deg
        # gantry test: hanging from the taut rope Albert leans up to ~17 deg, so only the relative limit (+9 deg) matters here
        with TiltGuard(robot, on_trip=lambda: to_balance(robot), max_abs=25.0):
            wi = ARM_IDX[4]
            js, im, last, n_frames = [], [], None, 0
            (ROOT / "logs").mkdir(exist_ok=True)
            print(f"  neck aimed at yaw {hold[NECK_YAW_IDX]:+.2f}, pitch {hold[NECK_PITCH_IDX]:+.2f}", flush=True)
            t0 = time.monotonic()
            print("wiggling the wrist for 6 s")
            while (t := time.monotonic() - t0) < 6.0:
                cmd = hold.copy(); cmd[wi] += 0.10 * np.sin(2 * np.pi * 1.0 * t)
                robot.trajectory(cmd)
                s = robot.get_state(); js.append((s.received_at, s.joint_pos[wi]))
                f = robot.camera.latest()
                if f is not None and f.received_at != last:
                    last = f.received_at
                    img = f.to_numpy()
                    blade, _, _ = masks(img, ball, step=4)
                    ys, xs = np.nonzero(blade)
                    n_frames += 1
                    if n_frames in (1, 60):  # what the camera saw, for when nothing is found
                        import imageio.v3 as iio
                        iio.imwrite(str(ROOT / "logs" / f"latency_frame{n_frames}.png"), np.asarray(img)[::2, ::2, :3])
                        print(f"  frame {n_frames}: {img.shape}, {getattr(f, 'encoding', '?')}, blade pixels {len(xs)}, "
                              f"neck {robot.get_state().joint_pos[NECK_YAW_IDX]:+.2f} / {robot.get_state().joint_pos[NECK_PITCH_IDX]:+.2f}", flush=True)
                    if len(xs) > 20:  # the blade's apparent width changes as it rolls: use its second moment
                        im.append((f.received_at, float(np.std(xs)), float(np.std(ys))))
                time.sleep(0.02)
            hand_back(robot)  # legs stay in the stance: Albert's own pose tips over
    js, im = np.array(js), np.array(im)
    if len(im) < 20 or len(js) < 50:
        sys.exit(f"not enough data ({len(im)} frames with the blade, {len(js)} joint samples): nothing written")
    tj = np.arange(js[0, 0] + 0.3, js[-1, 0] - 0.5, 0.005)
    q = np.interp(tj, js[:, 0], js[:, 1]); q -= q.mean()
    best = None
    for col in (1, 2):
        sig = np.interp(tj, im[:, 0], im[:, col]); sig -= sig.mean()
        for lag in np.arange(0.0, 0.4, 0.005):  # image at time t shows the joint at time t - lag
            qq = np.interp(tj - lag, tj, q)
            c = abs(np.corrcoef(qq, sig)[0, 1])
            if best is None or c > best[0]:
                best = (c, lag)
    c, lag = best
    print(f"{len(im)} frames ({len(im) / 6:.0f} fps): latency {1000 * lag:.0f} ms (correlation {c:.2f})")
    (ROOT / "camera_latency.json").write_text(json.dumps(dict(latency_s=lag, correlation=c, fps=len(im) / 6), indent=1))
    print("done: Albert goes to DAMP 2 s after the last setpoint, keep it on the gantry")


if __name__ == "__main__":
    main()
