"""Record a human demonstration on Albert: the right arm goes limp, you balance the ball on the paddle by moving the arm, and the
script logs at 50 Hz the head-camera ball estimate, the arm angles and the IMU. Then it checks the whole perception chain
against physics: a ball rolls downhill with acceleration ~ 3/5 g x the blade's slope (hollow sphere), so the measured
acceleration of the ball estimate must match the blade tilt from the arm model + IMU, with the same sign on both axes, after
the camera latency. Writes albert_demo.csv, albert_demo_detect.mp4 and prints the check.

    python -m albert.record_demo --supported --duration 60
"""
from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path
import sys
import threading
import time

import numpy as np

from albert.ball_camera import BALL_ORANGE, BALL_WHITE, CameraModel, debug_image, detect
from albert.common import (graceful_signals, take_joints, hand_back, BalanceOnExit, ARM_IDX, BALL, CAMERA_LATENCY_S, FLAT_HOME_FW, KD, KP, NECK_PITCH_IDX, NECK_YAW_IDX, WAIST_IDX, Kinematics,
                           TiltGuard, fw_to_mjcf, load_calibration, look_at, safe_pose)

G_ROLL = 3 / 5 * 9.81  # hollow sphere rolling without slipping


def main():
    graceful_signals()
    p = argparse.ArgumentParser()
    p.add_argument("--supported", action="store_true")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--analyse", default=None, help="only analyse an existing CSV")
    a = p.parse_args()
    if a.analyse:
        analyse(a.analyse); return
    if not a.supported:
        sys.exit("Hang Albert from its gantry, then run again with --supported.")
    from menlo.asimov import Robot

    calib = load_calibration()
    cam, ball = CameraModel.from_json(calib), calib.get("ball", BALL_WHITE if BALL == "white" else BALL_ORANGE)
    kin, kin_cam = Kinematics(), Kinematics()
    kp, kd = list(KP), list(KD)
    for i in ARM_IDX:
        kp[i], kd[i] = 0.0, 1.0
    rows, dbg = [], []
    with Robot().connect("hybrid") as robot, BalanceOnExit(robot):
        if robot.get_state().faulted:
            sys.exit("Albert reports a fault: not moving.")
        take_joints(robot)
        st = robot.get_state()
        start = safe_pose(st.joint_pos)
        home = start.copy(); home[ARM_IDX] = FLAT_HOME_FW
        home[NECK_YAW_IDX], home[NECK_PITCH_IDX] = look_at(kin, fw_to_mjcf(home[ARM_IDX]), base_quat=st.quat or (1, 0, 0, 0))
        print("arm to the home position, head aimed at the paddle (3 s)")
        robot.set_joints(home, duration=3.0, wait=False); time.sleep(3.3)
        lock, hist_t, hist_s = threading.Lock(), [], []
        out = {"uw": None, "t_cap": None}
        stop = threading.Event()

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
                img = f.to_numpy(); time.sleep(0)
                det = detect(img, ball, step=4); time.sleep(0)
                uw = None if det is None else cam.ball_on_blade(np.array(det[:2]), *kin_cam.head_pose(), *kin_cam.blade_frame())
                with lock:
                    out.update(uw=uw, t_cap=t_cap)
                if len(dbg) < 22 * 300:
                    dbg.append(debug_image(img, ball, step=4))

        input("Hold the paddle (the arm goes limp), ball ready, then Enter ")
        threading.Thread(target=detector, daemon=True).start()
        with TiltGuard(robot, on_trip=lambda: robot.set_joints(start, duration=2.0, wait=False)):
            print(f"recording {a.duration:.0f} s: balance the ball by moving the arm (Ctrl+C to stop)")
            t0 = tick = time.monotonic()
            try:
                while (t := time.monotonic() - t0) < a.duration:
                    s_ = robot.get_state()
                    with lock:
                        hist_t.append(s_.received_at); hist_s.append(s_)
                        if len(hist_t) > 200:
                            del hist_t[:100], hist_s[:100]
                        uw, t_cap = out["uw"], out["t_cap"]
                    q = np.array(s_.joint_pos)
                    cmd = home.copy(); cmd[ARM_IDX] = q[ARM_IDX]  # the arm follows wherever you put it
                    robot.trajectory(cmd, kp=kp, kd=kd)
                    kin.update(fw_to_mjcf(q[ARM_IDX]), waist=q[WAIST_IDX], base_quat=s_.quat)
                    g_blade = kin.blade_frame()[1].T @ np.array([0, 0, -1.0])
                    rows.append([t, s_.received_at, *(uw if uw is not None else (np.nan, np.nan)), np.nan if t_cap is None else t_cap,
                                 *g_blade, *fw_to_mjcf(q[ARM_IDX])])
                    tick += 0.02; time.sleep(max(0.0, tick - time.monotonic()))
            except KeyboardInterrupt:
                print("\nstopped")
        stop.set()
        save(rows)  # saved before anything else can interrupt
        # keep every joint alive while waiting (the SDK re-sends a set_joints target in the background): no setpoint for 2 s
        # and Albert goes limp. The arm is held where it is, at low gain, until Enter.
        robot.set_joints(safe_pose(robot.get_state().joint_pos), duration=0.3, wait=False)
        try:
            input("Hold the paddle: Enter and the arm stiffens, then goes back down (3 s) ")
        except KeyboardInterrupt:
            pass
        hand_back(robot, start)
    if dbg:
        import imageio
        imageio.mimwrite("albert_demo_detect.mp4", dbg, fps=22, quality=7, macro_block_size=1)
    print("wrote albert_demo.csv, albert_demo_detect.mp4")
    analyse("logs/albert_demo.csv")


def save(rows):
    hdr = ["t", "state_time", "ball_u", "ball_w", "frame_capture_time", "g_u", "g_w", "g_n", "q0", "q1", "q2", "q3", "q4"]
    Path("logs").mkdir(exist_ok=True)
    with open("logs/albert_demo.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([hdr] + rows)


def analyse(path):
    """Ball acceleration (from the camera estimates, one per frame) vs 3/5 g x blade slope (arm model + IMU), over lags."""
    r = np.genfromtxt(path, delimiter=",", skip_header=1)
    st_t, u, w, tc, gu, gw = r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6]
    ok = np.isfinite(u) & np.isfinite(tc)
    # one sample per camera frame
    tc_u, idx = np.unique(tc[ok], return_index=True)
    P = np.c_[u[ok][idx], w[ok][idx]]
    if len(tc_u) < 30:
        print(f"only {len(tc_u)} frames with the ball: not enough to check"); return
    tt = np.arange(tc_u[0] + 0.2, tc_u[-1] - 0.2, 0.02)
    gaps = np.interp(tt, tc_u[1:], np.diff(tc_u)) < 0.12  # keep stretches where the ball was seen continuously
    from scipy.signal import savgol_filter
    pu, pw = np.interp(tt, tc_u, P[:, 0]), np.interp(tt, tc_u, P[:, 1])
    au, aw = savgol_filter(pu, 15, 2, deriv=2, delta=0.02), savgol_filter(pw, 15, 2, deriv=2, delta=0.02)
    print(f"{len(tc_u)} frames with the ball, {gaps.mean() * 100:.0f} % of the time seen continuously")
    for name, acc, g in (("along the handle (u)", au, gu), ("across (w)", aw, gw)):
        best = None
        for lag in np.arange(-0.10, 0.25, 0.01):  # the image of time t shows the ball at t; the blade tilt at t - lag drives it
            gg = np.interp(tt - lag, st_t, g) * G_ROLL
            m = gaps & np.isfinite(gg)
            if m.sum() < 30:
                continue
            k = float(np.dot(gg[m], acc[m]) / np.dot(gg[m], gg[m]))
            c = float(np.corrcoef(gg[m], acc[m])[0, 1])
            if best is None or abs(c) > abs(best[0]):
                best = (c, lag, k)
        c, lag, k = best
        verdict = "OK" if (c > 0.3 and 0.3 < k < 2.0) else ("SIGN FLIPPED" if (c < -0.3 and k < 0) else "weak / unclear")
        print(f"  {name:22s}: ball accel = {k:+.2f} x (3/5 g slope), correlation {c:.2f}, extra lag {1000 * lag:+.0f} ms -> {verdict}")


if __name__ == "__main__":
    main()
