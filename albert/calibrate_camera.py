"""Calibrate Albert's head camera against the paddle: pose of the camera in the head + focal length.

The arm goes to the home pose (blade level) and the head looks at the paddle. You put the (white) ball on marked spots of the blade
(centre, 4 cm towards the tip / the handle / left / right, and the four diagonals), press Enter each time; the script detects
the ball, computes where its centre is from the arm's forward kinematics, and fits the camera. Then it checks itself: every
spot is estimated back from its pixel, and the error is printed. Writes albert/camera_calibration.json and
albert/calibration_mask.png (with --sim: *_sim.json / *_sim.png) (left: the frame; right: blue = blade region, white = ball colour inside it, red cross = detection; tune the thresholds
in ball_camera.py if the ball is not the only white blob).

    python -m albert.calibrate_camera --supported          # on Albert (robot hanging from its gantry)
    python -m albert.calibrate_camera --sim                # on the simulated Albert

Mark the spots with a pen on the rubber: u runs along the handle (+u towards the tip), w across the blade (+w to the left of
the handle seen from above, robot side), as drawn in albert/calibration_spots.png (spots numbered in the order asked).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from albert.ball_camera import BALL_ORANGE, BALL_WHITE, CameraModel, debug_image, detect
from albert.common import graceful_signals, TAUGHT_GRAVITY, TiltGuard, look_at  # noqa: F401
from albert.common import (lean_deg, with_stance, to_balance, take_joints, hand_back, BalanceOnExit, safe_pose, FLAT_HOME_FW, HOME, TiltGuard, ARM_IDX, BALL, CALIB_FILE, CALIB_FILE_SIM, NECK_PITCH, NECK_PITCH_IDX, NECK_YAW, NECK_YAW_IDX, ROOT,
                           WAIST_IDX, Kinematics, fw_to_mjcf, mjcf_to_fw)

SPOTS = [(0, 0), (0.04, 0), (-0.04, 0), (0, 0.04), (0, -0.04), (0.03, 0.03), (0.03, -0.03), (-0.03, 0.03), (-0.03, -0.03)]


def look_pose(state, q_arm_mjcf, sim=False, kin=None):
    """Arm at the home pose, head looking at the paddle (aimed with the model, the camera assumed to look along the head)."""
    tgt = safe_pose(state.joint_pos)
    tgt[ARM_IDX] = FLAT_HOME_FW if (FLAT_HOME_FW is not None and not sim) else mjcf_to_fw(q_arm_mjcf)
    if kin is None:
        tgt[NECK_YAW_IDX], tgt[NECK_PITCH_IDX] = NECK_YAW, NECK_PITCH
    else:
        quat = (1, 0, 0, 0) if sim else (state.quat or (1, 0, 0, 0))
        tgt[NECK_YAW_IDX], tgt[NECK_PITCH_IDX] = look_at(kin, fw_to_mjcf(tgt[ARM_IDX]), waist=tgt[WAIST_IDX], base_quat=quat)
        print(f"  head aimed at the paddle: neck yaw {tgt[NECK_YAW_IDX]:+.2f}, pitch {tgt[NECK_PITCH_IDX]:+.2f} rad")
    return tgt


def snapshot(robot, kin, ball, sim_spot=None, hold=None):
    """One detection + the matching head pose and blade frame. In sim, the ball is put on the spot first."""
    if sim_spot is not None:
        for _ in range(20):  # let the camera latency (+ one frame) pass with the ball held on the spot
            robot.place_ball(*sim_spot); robot.trajectory(hold)
    else:
        input("  ... ball on the spot, hands away, then Enter ")
        t_ask = time.monotonic()
    for _ in range(40):
        f = robot.camera.latest()
        if f is not None and (sim_spot is not None or f.received_at > t_ask + 0.3):  # a frame taken after Enter
            det = detect(f.to_numpy(), ball)
            if det is not None:
                break
        if sim_spot is None:
            time.sleep(0.05)
    else:
        return None, f
    s = robot.get_state()
    q = np.array(s.joint_pos)
    kin.update(fw_to_mjcf(q[ARM_IDX]), waist=q[WAIST_IDX], neck_yaw=q[NECK_YAW_IDX], neck_pitch=q[NECK_PITCH_IDX], base_quat=s.quat)
    return (np.array(det[:2]), kin.head_pose(), kin.blade_frame()), f


def main():
    graceful_signals()
    try:
        _main()
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C)")


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", action="store_true")
    p.add_argument("--supported", action="store_true", help="confirms Albert hangs from its gantry (joint control turns balancing off)")
    p.add_argument("--refit", default=None, help="redo the fit from a saved calibration_spots_*.json (no robot)")
    a = p.parse_args()
    if a.refit:
        refit(a.refit); return
    if a.sim:
        from albert.sim_robot import SimAlbert
        robot = SimAlbert(seed=1)
    else:
        if not a.supported:
            sys.exit("Hang Albert from its gantry, then run again with --supported.")
        from menlo.asimov import Robot
        robot = Robot().connect("hybrid")
        if not robot.has("camera"):
            sys.exit("this connection carries no camera: use hybrid or livekit")
    ball = BALL_WHITE if BALL == "white" else BALL_ORANGE
    kin = Kinematics()
    with robot, BalanceOnExit(robot):
        s = robot.get_state()
        if getattr(s, "faulted", False):
            sys.exit("Albert reports a fault: not moving.")
        take_joints(robot)
        start = safe_pose(robot.get_state().joint_pos)
        hold = look_pose(robot.get_state(), HOME, sim=a.sim, kin=kin)
        if not a.sim:
            hold = with_stance(hold)  # Albert's own standing poses tip over under joint control
            input("Rope TAUT (carrying part of Albert's weight), E-Stop in reach. Enter: legs to the centred stance, arm home ")
        try:
            if a.sim:
                _calibrate(a, robot, kin, ball, start, hold)
            else:
                # gantry: the arm forward tips the hanging body ~10 deg; a trip goes straight to Menlo's balance controller
                with TiltGuard(robot, on_trip=lambda: to_balance(robot), max_extra=15.0, max_abs=25.0):
                    _calibrate(a, robot, kin, ball, start, hold)
        except KeyboardInterrupt:
            print("\ninterrupted: back to Menlo's balance controller")
            hand_back(robot)
            raise
    if not a.sim:
        print("done: Albert is under Menlo's balance controller")


def load_lens(sim=False):
    """Albert's lens calibration (albert/calibrate_lens.py), or None: the simulated camera is an ideal pinhole."""
    f = ROOT / "camera_intrinsics.json"
    if sim or not f.exists():
        return None
    return json.loads(f.read_text())


def robust_fit(width, height, samples, truth, min_keep=7, lens=None):
    """Fit the camera; while one spot is far off the others (> 3x the median error and > 6 mm: the ball rolled, a hand, the
    wrong spot), drop it and refit, keeping at least min_keep spots. Prints every spot's error."""
    samples, truth = list(samples), list(truth)
    while True:
        cam = CameraModel(width, height, lens=lens)
        mean_px, max_px = cam.fit(samples)
        errs = []
        for (px, _, _, _), (u, w, bc, bR, hp, hR) in zip(samples, truth):
            est = cam.ball_on_blade(px, hp, hR, bc, bR)
            errs.append(np.hypot(est[0] - u, est[1] - w) * 1000 if est is not None else np.inf)
        errs = np.array(errs)
        k = int(np.argmax(errs))
        if len(samples) > min_keep and errs[k] > max(6.0, 3 * np.median(errs)):
            u, w = truth[k][0], truth[k][1]
            print(f"  spot u={100*u:+.0f} w={100*w:+.0f} cm is {errs[k]:.0f} mm off (median {np.median(errs):.1f} mm): dropped, refit")
            del samples[k], truth[k]
            continue
        print("  per-spot error (mm): " + ", ".join(f"({100*t[0]:+.0f},{100*t[1]:+.0f}) {e:.1f}" for t, e in zip(truth, errs)))
        return cam, mean_px, max_px, list(errs), samples, truth


def refit(path):
    """Redo the fit from saved spots (no robot); writes albert/camera_calibration.json if it passes."""
    j = json.loads(Path(path).read_text())
    samples, truth = [], []
    for sp in j["spots"]:
        hp, hR, bc, bR = (np.array(sp[k]) for k in ("hp", "hR", "bc", "bR"))
        samples.append((np.array(sp["px"]), np.array(sp["P"]), hp, hR)); truth.append((sp["u"], sp["w"], bc, bR, hp, hR))
    sim = "_sim" in Path(path).name  # simulated spots never touch Albert's calibration
    lens = load_lens(sim)
    cam, mean_px, max_px, errs, samples, truth = robust_fit(j["width"], j["height"], samples, truth, lens=lens)
    ok = np.mean(errs) <= 10.0
    target = CALIB_FILE_SIM if sim else CALIB_FILE
    out = target if ok else target.with_name(target.stem + "_rejected.json")
    prev = json.loads(target.read_text()) if target.exists() else {}
    if ok and target.exists() and not sim:  # the calibration it replaces
        hist = ROOT / "calibration_history"; hist.mkdir(exist_ok=True)
        (hist / f"camera_calibration_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(target.read_text())
    out.write_text(json.dumps(dict(cam.to_json(), ball=prev.get("ball", BALL_WHITE), level_deg=prev.get("level_deg"),
                                   reproj_px=[mean_px, max_px], spot_err_mm=float(np.mean(errs)), spots=Path(path).name), indent=1))
    print(f"refit on {len(samples)} spots{' with the lens calibration' if lens else ''}: {np.mean(errs):.1f} mm mean, "
          f"{np.max(errs):.1f} mm max, reprojection {mean_px:.1f} px mean (before: {prev.get('spot_err_mm', float('nan')):.1f} mm) -> {out}")


def _calibrate(a, robot, kin, ball, start, hold):
    if True:
        print("moving the arm to the home pose and the head towards the paddle (3 s)")
        if a.sim:
            robot.set_joints(hold, duration=3.0)
        else:  # the arm sags a little under the paddle: do not wait for a tight tolerance
            robot.set_joints(hold, duration=3.0, wait=False)
            time.sleep(3.5)
            # level the blade for the body's actual lean (the stance and the rope tilt Albert by up to ~15 deg; a tilted blade
            # lets the ball roll off its spot), then aim the head again
            from albert.common import level_dq
            kin_l = Kinematics()
            for _ in range(4):
                st = robot.get_state()
                q = fw_to_mjcf(np.array(st.joint_pos)[ARM_IDX])
                dq, tilt = level_dq(kin_l, q, st.quat, st.joint_pos[WAIST_IDX])
                hold[ARM_IDX] = mjcf_to_fw(np.asarray(fw_to_mjcf(hold[ARM_IDX])) + dq)
                robot.set_joints(hold, duration=0.8, wait=False); time.sleep(1.2)
            st = robot.get_state()
            q = fw_to_mjcf(np.array(st.joint_pos)[ARM_IDX]); _, tilt = level_dq(kin_l, q, st.quat, st.joint_pos[WAIST_IDX])
            hold[NECK_YAW_IDX], hold[NECK_PITCH_IDX] = look_at(kin_l, q, waist=hold[WAIST_IDX], base_quat=st.quat)
            robot.set_joints(hold, duration=0.8, wait=False); time.sleep(1.2)
            print(f"  blade levelled: {tilt:.1f} deg left (body lean {lean_deg(st):.1f} deg)", flush=True)
            import imageio
            for _ in range(50):
                f = robot.camera.latest()
                if f is not None:
                    break
                time.sleep(0.1)
            imageio.imwrite(ROOT / "calibration_view.png", debug_image(f.to_numpy(), ball))
            print(f"  head-camera view saved to {ROOT / 'calibration_view.png'} (blue = paddle found)")
            level = None
        samples, truth, last = [], [], None
        for u, w in SPOTS:
            print(f"spot u={100*u:+.0f} cm, w={100*w:+.0f} cm")
            snap, last = snapshot(robot, kin, ball, sim_spot=(u, w) if a.sim else None, hold=hold)
            if snap is None:
                print("  ball not detected: skipped"); continue
            px, (hp, hR), (bc, bR) = snap
            P = bc + bR @ np.array([u, w, 0.005 + 0.020])
            samples.append((px, P, hp, hR)); truth.append((u, w, bc, bR, hp, hR))
        if len(samples) < 5:
            sys.exit(f"only {len(samples)} detections: check the lighting and the ball thresholds (see calibration_mask.png)")
        frame = last.to_numpy() if last is not None else None
        # every spot, so a fit can be redone without the robot: python -m albert.calibrate_camera --refit <file>
        spots_file = ROOT / ("calibration_spots_sim.json" if a.sim else f"calibration_spots_{time.strftime('%Y%m%d_%H%M%S')}.json")
        spots_file.write_text(json.dumps(dict(width=frame.shape[1], height=frame.shape[0], spots=[
            dict(u=u, w=w, px=np.asarray(px).tolist(), P=np.asarray(P).tolist(), hp=np.asarray(hp).tolist(), hR=np.asarray(hR).tolist(),
                 bc=np.asarray(bc).tolist(), bR=np.asarray(bR).tolist())
            for (px, P, _, _), (u, w, bc, bR, hp, hR) in zip(samples, truth)])))
        print(f"  spots saved to {spots_file}")
        cam, mean_px, max_px, errs, samples, truth = robust_fit(frame.shape[1], frame.shape[0], samples, truth, lens=load_lens(a.sim))
        out = CALIB_FILE_SIM if a.sim else CALIB_FILE
        if np.mean(errs) > 10.0:  # the ball was moving, held, or not on the spots
            out = out.with_name(out.stem + "_rejected.json")
            print(f"  fit rejected ({np.mean(errs):.0f} mm): was the ball resting, hands away, on the right spots?")
        out.write_text(json.dumps(dict(cam.to_json(), ball=ball, level_deg=None if a.sim else level, reproj_px=[mean_px, max_px],
                                              spot_err_mm=float(np.mean(errs))), indent=1))
        print(f"camera fitted on {len(samples)} spots: reprojection {mean_px:.1f} px mean, {max_px:.1f} px max; "
              f"ball position error {np.mean(errs):.1f} mm mean, {np.max(errs):.1f} mm max -> {out}")
        if frame is not None:
            import imageio
            imageio.imwrite(ROOT / ("calibration_mask_sim.png" if a.sim else "calibration_mask.png"), debug_image(frame, ball))
        hand_back(robot)  # the legs stay in the stance until Menlo's controller takes over


if __name__ == "__main__":
    main()
