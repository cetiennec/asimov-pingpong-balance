"""Tilt test on Albert: does the ball roll where the model says "downhill" is?

The arm holds the paddle at the home pose. For each of four tilts (tip down, tip up, left side down, right side down) you put
the ball on the centre and take your hands away; the arm tilts the blade by TILT_DEG about its centre (inverse kinematics on
the arm model), holds it while the head camera watches the ball roll, then levels again. The script compares the ball's
motion (camera) with the downhill direction (arm model + IMU): same direction = the whole chain agrees with the physics;
opposite = an axis is flipped somewhere. Also prints the measured acceleration vs 3/5 g sin(tilt).

    python -m albert.tilt_test --supported
"""
from __future__ import annotations

import argparse
import bisect
import sys
import threading
import time

import numpy as np

from albert.ball_camera import BALL_ORANGE, BALL_WHITE, CameraModel, debug_image, detect
from albert.common import (graceful_signals, take_joints, hand_back, BalanceOnExit, ARM_IDX, BALL, CAMERA_LATENCY_S, FLAT_HOME_FW, HOME, NECK_PITCH_IDX, NECK_YAW_IDX, WAIST_IDX, Kinematics,
                           TiltGuard, fw_to_mjcf, load_calibration, look_at, mjcf_to_fw, safe_pose)

TILT_DEG = 4.0
TESTS = [("tip DOWN", -1, 0), ("tip UP", 1, 0), ("LEFT side down", 0, -1), ("RIGHT side down", 0, 1)]


def tilt_q(kin, q, tip_deg, left_deg, base_quat):
    """Joint angles that tilt the blade about its centre: tip up by tip_deg, left (+w) edge up by left_deg."""
    def frame(qq):
        kin.update(qq, base_quat=base_quat)
        return kin.blade_frame()
    c0, R0 = frame(q)
    u, w, n = R0[:, 0], R0[:, 1], R0[:, 2]
    rot = -w * np.deg2rad(tip_deg) + u * np.deg2rad(left_deg)
    target_n = n + np.cross(rot, n)
    qq = q.copy()
    for _ in range(5):  # a few Gauss-Newton steps
        c1, R1 = frame(qq)
        J_n, J_c, eps = np.zeros((3, 5)), np.zeros((3, 5)), 1e-4
        for j in range(5):
            d = np.zeros(5); d[j] = eps
            c2, R2 = frame(qq + d)
            J_n[:, j], J_c[:, j] = (R2[:, 2] - R1[:, 2]) / eps, (c2 - c1) / eps
        A = np.vstack([J_n, 3 * J_c, 0.02 * np.eye(5)])
        b = np.r_[target_n - R1[:, 2], 3 * (c0 - c1), np.zeros(5)]
        qq = qq + np.linalg.lstsq(A, b, rcond=None)[0]
    return qq


def main():
    graceful_signals()
    p = argparse.ArgumentParser()
    p.add_argument("--supported", action="store_true")
    a = p.parse_args()
    if not a.supported:
        sys.exit("Hang Albert from its gantry, then run again with --supported.")
    from menlo.asimov import Robot

    calib = load_calibration()
    cam, ball = CameraModel.from_json(calib), calib.get("ball", BALL_WHITE if BALL == "white" else BALL_ORANGE)
    kin, kin_cam = Kinematics(), Kinematics()
    with Robot().connect("hybrid") as robot, BalanceOnExit(robot):
        if robot.get_state().faulted:
            sys.exit("Albert reports a fault: not moving.")
        take_joints(robot)
        st = robot.get_state()
        start = safe_pose(st.joint_pos)
        home = start.copy(); home[ARM_IDX] = FLAT_HOME_FW
        home[NECK_YAW_IDX], home[NECK_PITCH_IDX] = look_at(kin, HOME, base_quat=st.quat or (1, 0, 0, 0))
        print("arm to the home position, head aimed at the paddle (3 s)")
        robot.set_joints(home, duration=3.0, wait=False); time.sleep(3.3)
        lock, hist_t, hist_s, obs, frames = threading.Lock(), [], [], [], []
        stop = threading.Event()

        def state_logger():
            while not stop.is_set():
                s_ = robot.get_state()
                with lock:
                    hist_t.append(s_.received_at); hist_s.append(s_)
                time.sleep(0.01)

        def detector():
            last = None
            while not stop.is_set():
                f = robot.camera.latest()
                if f is None or f.received_at == last:
                    time.sleep(0.003); continue
                last = f.received_at
                t_cap = f.received_at - CAMERA_LATENCY_S
                with lock:
                    i = max(0, bisect.bisect_right(hist_t, t_cap) - 1)
                    s_ = hist_s[i] if hist_s else None
                if s_ is None:
                    continue
                q = np.array(s_.joint_pos)
                kin_cam.update(fw_to_mjcf(q[ARM_IDX]), waist=q[WAIST_IDX], neck_yaw=q[NECK_YAW_IDX], neck_pitch=q[NECK_PITCH_IDX], base_quat=s_.quat)
                img = f.to_numpy()
                det = detect(img, ball, step=4)
                frames.append((t_cap, debug_image(img, ball, step=4), det))
                if det is not None:
                    uw = cam.ball_on_blade(np.array(det[:2]), *kin_cam.head_pose(), *kin_cam.blade_frame())
                    if uw is not None:
                        with lock:
                            obs.append((t_cap, *uw))

        threading.Thread(target=state_logger, daemon=True).start()
        threading.Thread(target=detector, daemon=True).start()
        results = []
        with TiltGuard(robot, on_trip=lambda: robot.set_joints(start, duration=2.0, wait=False)):
            try:
                for name, tip, left in TESTS:
                    input(f"\n[{name}] ball on the CENTRE, hands away, then Enter (be ready to catch it) ")
                    s_ = robot.get_state()
                    q_home = fw_to_mjcf(np.array(s_.joint_pos)[ARM_IDX])
                    q_t = tilt_q(kin, fw_to_mjcf(home[ARM_IDX]), TILT_DEG * tip, TILT_DEG * left, s_.quat)
                    kin.update(q_t, base_quat=s_.quat); R = kin.blade_frame()[1]
                    downhill = (R.T @ np.array([0, 0, -1.0]))[:2]  # in-plane gravity: where the ball should go
                    tgt = home.copy(); tgt[ARM_IDX] = mjcf_to_fw(q_t)
                    t0 = time.monotonic()
                    robot.set_joints(tgt, duration=0.3, wait=False); time.sleep(1.5)
                    robot.set_joints(home, duration=0.5, wait=False)
                    with lock:
                        seg = np.array([o for o in obs if t0 - 0.2 <= o[0] <= t0 + 1.5])
                    time.sleep(0.6)
                    if len(seg) < 5:
                        print(f"  only {len(seg)} ball detections: skipped"); continue
                    tt, uu, ww = seg[:, 0] - t0, seg[:, 1], seg[:, 2]
                    moved = np.array([uu[-1] - uu[0], ww[-1] - ww[0]])
                    cos = float(moved @ downhill / (np.linalg.norm(moved) * np.linalg.norm(downhill) + 1e-9))
                    # acceleration along downhill from a parabola fit on the first 0.8 s of motion
                    m = tt < 0.9
                    proj = (np.c_[uu, ww] - [uu[0], ww[0]]) @ (downhill / np.linalg.norm(downhill))
                    acc = 2 * np.polyfit(tt[m], proj[m], 2)[0] if m.sum() >= 5 else np.nan
                    expect = 3 / 5 * 9.81 * np.linalg.norm(downhill)
                    verdict = "OK" if cos > 0.7 else ("OPPOSITE: axis flipped" if cos < -0.7 else "sideways: check")
                    print(f"  model downhill (u, w) = {np.round(downhill / np.linalg.norm(downhill), 2)}; ball moved "
                          f"{np.round(moved * 100, 1)} cm -> direction agreement {cos:+.2f}: {verdict}; "
                          f"acceleration {acc:.2f} m/s2 (expected {expect:.2f})")
                    results.append((name, cos, acc, expect))
                    # keep what the camera saw for this tilt: video + estimates
                    import csv
                    import imageio
                    from PIL import Image, ImageDraw
                    tag = name.split()[0].lower() + "_" + name.split()[1].lower()
                    with lock:
                        fr = [x for x in frames if t0 - 0.3 <= x[0] <= t0 + 1.6]
                    vid = []
                    for tc, im, det in fr:
                        pil = Image.fromarray(im); d_ = ImageDraw.Draw(pil)
                        e = [o for o in seg if abs(o[0] - tc) < 1e-6]
                        d_.text((6, 4), f"{name}  t={tc - t0:+.2f}s  " + (f"u={100*e[0][1]:+.1f} w={100*e[0][2]:+.1f} cm" if e else ""), fill=(255, 255, 0))
                        vid.append(np.asarray(pil))
                    if vid:
                        imageio.mimwrite(f"albert_tilt_{tag}.mp4", vid, fps=8, quality=7, macro_block_size=1)
                    with open(f"albert_tilt_{tag}.csv", "w", newline="") as fh:
                        csv.writer(fh).writerows([["t", "u", "w"]] + [[o[0] - t0, o[1], o[2]] for o in seg])
            except KeyboardInterrupt:
                print("\nstopped")
        stop.set()
        print("arm back to its starting pose (3 s)")
        hand_back(robot, start)
    if results:
        print("\nsummary:", ", ".join(f"{n}: {c:+.2f}" for n, c, _, _ in results))


if __name__ == "__main__":
    main()
