"""Run the trained paddle policy on Albert (or on the simulated Albert with --sim).

    python -m albert.run_policy --supported                       # Albert, hanging from its gantry
    python -m albert.run_policy --sim --video videos/albert_sim.mp4 # the whole pipeline in simulation

Loop at 50 Hz: robot state (joints, IMU) + head-camera ball estimate -> the same 40-value observation as in training ->
policy -> right-arm targets (+0.03 rad x action per step) -> robot.trajectory(). Every other joint holds the pose it had when
the script started (STAND), the neck looks at the paddle. Gains: the ones the policy was trained with (common.KP / KD).

What this cannot do on the real robot (menlo-sdk 2026-11): trajectory() switches the walking policy off for every joint, so
Albert balances the ball standing on its gantry, with perturbations (push the paddle, flick the ball, shake the gantry), but
not while walking: that needs an SDK mode where the walking policy keeps the legs and the arm takes external targets.

Stops when the ball is lost (no detection for 0.5 s, or off the blade), after --duration, or on Ctrl+C. Then the arm goes back
to its starting pose and Albert is handed to Menlo's balance controller (balance()), never left to DAMP.
"""
from __future__ import annotations

import os as _os

# one maths thread: on Albert's Pi (4 cores, one taken by Asimov Edge) thread pools fight the detector and the video decoding
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import os
import bisect
import csv
import sys
import threading
import time

import numpy as np

from albert.ball_camera import CameraModel, ball_on_blade_cov, blade_roi, debug_image, detect, detect_roi
from albert.common import TAUGHT_GRAVITY, TiltGuard, lean_deg, level_dq, look_at  # noqa: F401
from albert.common import (graceful_signals, CENTRED_STANCE, STALL_S, allow_link_hiccups, state_age, unique_log, to_balance, take_joints, hand_back, BalanceOnExit, safe_pose, FLAT_HOME_FW, HOME, TiltGuard, ACTION_SCALE, ARM_IDX, ARM_SIGN, CAMERA_LATENCY_S, KD, KP, NECK_PITCH, NECK_PITCH_IDX, NECK_YAW, NECK_YAW_IDX,
                           PERIOD, WAIST_IDX, Kinematics, ObsBuilder, fw_to_mjcf, load_calibration, mjcf_to_fw, uses_paddle_acc)
from asimov_pp.model import ARM_JOINTS, BLADE_HT, BLADE_R, BALL_R
from asimov_pp.ppo import load_policy

LOST_AFTER_S = 0.5


def guard(state):
    if getattr(state, "faulted", False):
        sys.exit("Albert reports a fault: not moving.")
    hot = [j.name for j in state.joints if j.temp is not None and j.temp >= 70]
    if hot:
        sys.exit(f"hot actuators ({', '.join(hot)}): let them cool first")
    if getattr(state, "battery", None) is not None and state.battery.soc_percent < 20:
        sys.exit("battery below 20 %")


def main():
    graceful_signals()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/albert_v3_final/best.pt",
                   help="policy (default: v2A, trained on the sim matched to Albert's 2026-11-19 session; the previous one: "
                        "outputs/albert_perturb/best.pt)")
    p.add_argument("--controller", default="policy", choices=("policy", "mpc"),
                   help="policy: the PPO checkpoint (--ckpt); mpc: the model-predictive controller (albert/mpc.py), SIMULATION-TESTED ONLY")
    p.add_argument("--mpc", default="", help="--controller mpc: parameters, 'key=value,...' or a JSON file (albert/mpc.py MPCParams)")
    p.add_argument("--sim", action="store_true", help="run on the simulated Albert")
    p.add_argument("--supported", action="store_true", help="confirms Albert hangs from its gantry")
    p.add_argument("--duration", type=float, default=120.0, help="whole session, s (several catches)")
    p.add_argument("--cam-latency", type=float, default=CAMERA_LATENCY_S, help="camera capture -> reception, s (measure it)")
    p.add_argument("--edge-gains", action="store_true", help="let Asimov Edge use its own gain table instead of the training gains")
    p.add_argument("--log", default="logs/run.csv", help="run log; also <log>_frames.csv, <log>.jsonl (robot recording), <log>_detect.mp4")
    p.add_argument("--video", default=None, help="--sim only: write a video of the run (outside view + head camera)")
    p.add_argument("--seed", type=int, default=3, help="--sim only")
    p.add_argument("--place", type=float, nargs=2, default=(0.03, -0.02), help="--sim only: where the ball is put (u, w), m")
    p.add_argument("--level", action="store_true", help="while waiting, level the blade by IK (off: the policy was trained from the home pose)")
    p.add_argument("--debug-video", default=None, help="real robot: what the detector saw (frame | masks), <log>_detect.mp4 to keep it; default: only when the raw "
                   "camera is not recorded (--camera-video '')")
    p.add_argument("--camera-video", default=None, help="raw head-camera frames (halved, H.264) for the LeRobot dataset; default: "
                   "<log>_camera.mp4 (+ _camera.csv: frame times), '' to skip")
    p.add_argument("--lerobot", action=argparse.BooleanOptionalAction, default=None,
                   help="after the run, rebuild the local LeRobot dataset with the camera (default: on for real runs on the Mac; "
                        "on Albert's Pi onboard.sh does it on the Mac)")
    p.add_argument("--truth-ball", action="store_true", help="--sim only, debugging: use the true ball position, not the camera")
    p.add_argument("--kick", type=float, default=0.0, help="--sim only: flick the ball every 3 s with this speed (m/s)")
    p.add_argument("--freestand", action="store_true", help="--sim only: Albert standing on the floor (no gantry), legs held by PD")
    p.add_argument("--stand-ankle", default="motor", choices=("motor", "joint"), help="--freestand: ankle gain model")
    p.add_argument("--stand-gain", type=float, default=1.0, help="--freestand: scale of every non-arm joint gain")
    p.add_argument("--floor-friction", type=float, default=1.0, help="--freestand")
    p.add_argument("--com-shift", type=float, nargs=2, default=(0.0, 0.0), help="--freestand: torso centre of mass error, m")
    p.add_argument("--stance", type=float, nargs=3, default=None, metavar=("HIP", "KNEE", "ANKLE"),
                   help="leg pitch angles held during the run (left leg, rad; right mirrored). Default: the centred stance. "
                        "Albert's own standing poses tip over under joint control (sim_freestand.py, 2026-11-20 near-fall)")
    p.add_argument("--aim-camera", action="store_true", help="aim the camera's calibrated optical axis at the blade (default: the "
                   "head's forward axis, as in the earlier sessions)")
    p.add_argument("--roi", action=argparse.BooleanOptionalAction, default=True,
                   help="search the ball only around where the arm model puts the blade (--no-roi: the whole frame)")
    p.add_argument("--action-filter", type=float, default=0.0, help="smooth the policy's actions: a = f * a_prev + (1 - f) * a_new "
                   "(0: off). The arm jiggled on Albert: actions changed 3x more step to step than in the sim (2026-11-22 G4)")
    p.add_argument("--start-frames", type=int, default=4, help="consecutive detections on the blade before balancing starts "
                   "(they must line up, slower than 0.25 m/s: a ball resting or rolling, not one still in the hand)")
    p.add_argument("--det-hints", action=argparse.BooleanOptionalAction, default=True,
                   help="tell the detector the ball's expected size (camera model) and where the predictor expects it "
                        "(--no-det-hints: the 2026-11-20 detector, largest ball-like blob)")
    p.add_argument("--level-start", action=argparse.BooleanOptionalAction, default=True,
                   help="level the blade for the body's actual lean before balancing (--no-level-start: the taught pose as is)")
    p.add_argument("--gantry", action="store_true", help="Albert stays hanging from the taut rope (no lowering prompt; tilt limits "
                   "for the gantry: +9 deg relative, 25 deg absolute)")
    p.add_argument("--yes", action="store_true", help="skip the stencil prompt (the rope prompts stay: run in a terminal)")
    p.add_argument("--hand", action="store_true", help="--sim only: put the ball back 2 s after each loss, off-centre and rolling")
    p.add_argument("--hand-speed", type=float, default=0.10, help="--hand: up to this speed (m/s)")
    a = p.parse_args()
    if a.stance is None:  # always a known leg pose: Menlo's own varies (6 deg forward to 11 deg back on the gantry, 2026-11-20)
        a.stance = list(CENTRED_STANCE)
    if not a.sim:
        a.log = unique_log(a.log)  # never overwrite an earlier run
        allow_link_hiccups()
    if a.camera_video is None:
        a.camera_video = a.log.replace(".csv", "_camera.mp4")
    if a.debug_video is None:  # the raw recording supersedes it (the masks can be redrawn from it offline): it cost the
        # detector thread ~5 ms per frame on the Pi (14.8 instead of 22 fps reached the detector, 2026-11-22 G1)
        a.debug_video = "" if a.camera_video else a.log.replace(".csv", "_detect.mp4")
    import os
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)

    if a.sim and a.freestand:
        from albert.sim_freestand import FreeStandAlbert
        robot = FreeStandAlbert(seed=a.seed, ankle=a.stand_ankle, gain_scale=a.stand_gain, floor_friction=a.floor_friction,
                                com_shift=a.com_shift, stance=a.stance)
    elif a.sim:
        from albert.sim_robot import SimAlbert
        robot = SimAlbert(seed=a.seed)
    else:
        if not a.supported:
            sys.exit("Hang Albert from its gantry (for --stance: rope attached, slack once in the stance), then run again with "
                     "--supported. Joint control turns its balancing off.")
        from menlo.asimov import Robot
        robot = Robot().connect("hybrid")
    calib = load_calibration(sim=a.sim)
    if a.sim and a.freestand:  # the camera this sim renders, exactly (the sim calibration was fitted with the head elsewhere)
        calib = robot.exact_calibration(calib)
    cam = CameraModel.from_json(calib)
    ball = calib["ball"]  # the colour thresholds the camera was calibrated with
    kin, kin_cam = Kinematics(), Kinematics()
    if a.controller == "mpc":
        from albert.mpc import MPCController
        policy = MPCController(a.mpc or None)
        print("controller: MPC", a.mpc or "(defaults)")
    else:
        policy = load_policy(a.ckpt)
    paddle_acc = a.controller == "mpc" or uses_paddle_acc(a.ckpt)  # the predictor the controller was made with
    obs_b = ObsBuilder(kin, paddle_acc=paddle_acc)
    print("ball predictor:", "tilt + paddle acceleration (KF_PARAMS_ACC)" if paddle_acc else "tilt only (KF_PARAMS)")
    gains = {} if a.edge_gains else dict(kp=KP, kd=KD)
    q_lo = np.array([kin.m.jnt_range[kin.m.joint(j).id][0] for j in ARM_JOINTS]) * 0.97
    q_hi = np.array([kin.m.jnt_range[kin.m.joint(j).id][1] for j in ARM_JOINTS]) * 0.97
    frames, rows = [], []

    frame_rows, saved = [], []

    def save_logs():
        """The run log and the per-frame log (once)."""
        if saved:
            return
        saved.append(True)
        hdr = (["t", "balancing", "ball_u", "ball_w"] + [f"q_{j}" for j in ARM_JOINTS] + [f"a{i}" for i in range(5)]
               + ["t_state", "g_u", "g_w"] + (["true_u", "true_w"] if a.sim else []))
        with open(a.log.replace(".csv", "_frames.csv"), "w", newline="") as fh:
            csv.writer(fh).writerows([["t", "t_capture", "det_u", "det_w", "mode", "accepted", "kf_u", "kf_w", "kf_vu", "kf_vw",
                                       "bias_u", "bias_w", "g_u", "g_w"] + (["true_u", "true_w"] if a.sim else [])] + frame_rows)
        with open(a.log, "w", newline="") as fh:
            csv.writer(fh).writerows([hdr] + rows)
        print("log:", a.log, flush=True)

    try:
        _session(a, robot, calib, cam, ball, policy, kin, kin_cam, obs_b, gains, q_lo, q_hi, frames, rows, frame_rows, save_logs)
    finally:
        save_logs()
        from albert.frame_recorder import close_all
        close_all()
    import platform
    if a.lerobot if a.lerobot is not None else (not a.sim and platform.machine() not in ("aarch64", "armv7l")):
        from tools.export_lerobot import export_after_run
        export_after_run(os.path.dirname(os.path.abspath(a.log)))  # Albert is with Menlo's balance controller by now



def _session(a, robot, calib, cam, ball, policy, kin, kin_cam, obs_b, gains, q_lo, q_hi, frames, rows, frame_rows, save_logs):
    dbg = []
    with robot, (robot.record(a.log.replace(".csv", ".jsonl")) if not a.sim else robot.record(None)), BalanceOnExit(robot):
        s = robot.get_state(); guard(s)
        if not (a.stance and getattr(s.mode, "name", "") in ("MOVE", "STAND")):
            take_joints(robot)
        start = safe_pose(robot.get_state().joint_pos)
        hold = start.copy()
        # the policy's home (with the taught offsets), levelled below for the actual lean; --no-level-start: the taught flat pose
        hold[ARM_IDX] = FLAT_HOME_FW if (FLAT_HOME_FW is not None and not a.sim and not a.level_start) else mjcf_to_fw(HOME)
        if a.stance:
            from albert.sim_freestand import stance_fw
            hold[:12] = stance_fw(*a.stance, base=start)[:12]
        st0 = robot.get_state()
        real_quat = st0.quat is not None and (not a.sim or a.freestand)
        from scipy.spatial.transform import Rotation
        optical_axis = Rotation.from_rotvec(np.array(calib["params"][:3])).as_matrix()[:, 2]  # the camera's, in the head
        hold[NECK_YAW_IDX], hold[NECK_PITCH_IDX] = look_at(Kinematics(), fw_to_mjcf(hold[ARM_IDX]), waist=hold[WAIST_IDX],
                                                           base_quat=(st0.quat if real_quat else (1, 0, 0, 0)),
                                                           axis_in_head=optical_axis if a.aim_camera else (1.0, 0.0, 0.0))
        early_guard = None
        if a.stance and not a.sim:  # free-standing: take over only while the rope carries part of the weight (stand_test.py)
            input("FREE-STANDING: rope TAUT (carrying part of the weight), E-Stop in hand. Enter: legs to the stance (3 s) ")
        print("arm to the home pose, head aimed at the paddle and then fixed (3 s)" + (", legs to the stance" if a.stance else ""))
        robot.set_joints(hold, duration=3.0, wait=False, **(gains if a.stance else {}))  # held in the background until the loop
        time.sleep(3.2)
        if a.stance and not a.sim:
            # from here on a lean trips straight to Menlo's balance controller; nothing is sent to the joints after a trip.
            # On the gantry (--gantry) Albert hangs from the taut rope and leans up to ~17 deg: only the relative limit applies.
            early_guard = TiltGuard(robot, on_trip=lambda: to_balance(robot), max_extra=9.0 if a.gantry else 6.0,
                                    max_abs=25.0 if a.gantry else 16.0).__enter__()
            if not a.gantry:
                input("Lower the gantry SLOWLY until the rope is slack, then Enter ")
                print(f"rope slack: lean {lean_deg(robot.get_state()):.1f} deg (sim: 8.9)")
        if real_quat and a.level_start:  # level the blade for the body's actual lean (stance, rope): the ball must not roll at once
            kin_lv = Kinematics()
            for _ in range(4):
                stq = robot.get_state()
                q_now = fw_to_mjcf(np.array(stq.joint_pos)[ARM_IDX])
                dq, tilt = level_dq(kin_lv, q_now, stq.quat, stq.joint_pos[WAIST_IDX])
                hold[ARM_IDX] = mjcf_to_fw(np.asarray(fw_to_mjcf(hold[ARM_IDX])) + dq)
                robot.set_joints(hold, duration=0.8, wait=False, **(gains if a.stance else {})); time.sleep(1.1)
            stq = robot.get_state()
            _, tilt = level_dq(kin_lv, fw_to_mjcf(np.array(stq.joint_pos)[ARM_IDX]), stq.quat, stq.joint_pos[WAIST_IDX])
            print(f"blade levelled for a lean of {lean_deg(stq):.1f} deg: {tilt:.1f} deg left; arm moved "
                  f"{np.abs(fw_to_mjcf(hold[ARM_IDX]) - HOME).max():.2f} rad from home at most")
        if real_quat:  # aim the head again with the body as it now stands / hangs (the lean changes with the stance and the rope)
            stq = robot.get_state()
            hold[NECK_YAW_IDX], hold[NECK_PITCH_IDX] = look_at(Kinematics(), fw_to_mjcf(hold[ARM_IDX]), waist=hold[WAIST_IDX],
                                                               base_quat=stq.quat,
                                                               axis_in_head=optical_axis if a.aim_camera else (1.0, 0.0, 0.0))
            robot.set_joints(hold, duration=0.8, wait=False, **(gains if a.stance else {}))
            time.sleep(1.0)
            print(f"head re-aimed for a lean of {lean_deg(stq):.1f} deg: neck yaw {hold[NECK_YAW_IDX]:+.2f}, pitch {hold[NECK_PITCH_IDX]:+.2f}")

        hist_t, hist_s = [], []  # recent joint states, to pair each frame with the arm pose at its capture time
        det_lock = threading.Lock()

        def get_state():
            st = robot.get_state()
            with det_lock:
                hist_t.append(st.received_at); hist_s.append(st)
                if len(hist_t) > 100:
                    del hist_t[:50], hist_s[:50]
            return st

        dbg = []
        from albert.frame_recorder import FrameRecorder
        rec = FrameRecorder(a.camera_video) if a.camera_video else None
        det_out = {"uw": None, "R": None, "t_cap": None, "frame": None, "seen": False}
        det_times = []
        det_stop = threading.Event()

        def detector():
            """Background: each new camera frame -> ball (u, w) on the blade. Never dies: an error is printed, the frame skipped."""
            n_err = 0
            while not det_stop.is_set():
                try:
                    _detector()
                except Exception as exc:  # noqa: BLE001 - a dead detector thread would leave the policy blind
                    n_err += 1
                    if n_err <= 3:
                        print(f"  detector error ({exc!r}); continuing", flush=True)
                    time.sleep(0.01)

        def _detector():
            """Each new camera frame -> ball (u, w) on the blade, paired with the arm pose at its capture time."""
            last = None
            while not det_stop.is_set():
                f = robot.camera.latest()
                if f is None or f.received_at == last:
                    time.sleep(0.003); continue
                last = f.received_at
                t_cap = f.received_at - a.cam_latency
                with det_lock:
                    i = max(0, bisect.bisect_right(hist_t, t_cap) - 1)
                    st = hist_s[i] if hist_s else None
                if st is None:
                    continue
                q = np.array(st.joint_pos)
                kin_cam.update(fw_to_mjcf(q[ARM_IDX]), waist=q[WAIST_IDX], neck_yaw=q[NECK_YAW_IDX], neck_pitch=q[NECK_PITCH_IDX], base_quat=st.quat)
                img = f.to_numpy()
                time.sleep(0)  # let the 50 Hz control loop run between the heavy steps
                t_d = time.perf_counter()
                roi = blade_roi(cam, *kin_cam.head_pose(), *kin_cam.blade_frame(), BLADE_R + 0.04, step=4) if a.roi else None
                det = detect_roi(img, ball, roi, step=4, **det_hints())
                det_times.append(time.perf_counter() - t_d)
                time.sleep(0)
                uw, Rz = (None, None) if det is None else ball_on_blade_cov(cam, det[:2], *kin_cam.head_pose(), *kin_cam.blade_frame())
                with det_lock:
                    det_out.update(uw=uw, R=Rz, t_cap=t_cap, frame=f, seen=uw is not None)
                if rec:
                    rec.add(img, f.received_at, t_cap)
                if a.debug_video and len(dbg) < 22 * 300:  # keep what the detector saw (frame | masks), up to 5 min
                    dbg.append((debug_image(img, ball, step=4), uw, t_cap))

        def det_hints():
            """For the frame being detected (kin_cam at its capture pose): the ball's expected radius in pixels, and the pixel
            where the predictor puts the ball at the capture time. {} without --det-hints."""
            if not a.det_hints:
                return {}
            hp, hR = kin_cam.head_pose(); b_c, b_R = kin_cam.blade_frame()
            _, depth = cam.project(b_c, hp, hR)
            if not depth > 0.05:
                return {}
            hints = dict(r_px=float(cam.p[6] * BALL_R / depth))
            try:
                x = obs_b.kf.state()
            except (TypeError, IndexError):  # reset by the control loop at that very moment
                x = None
            if x is not None:  # the predictor's ball now, taken back to the capture time (a hint only: a race is harmless)
                u, w = x[0] - x[2] * a.cam_latency, x[1] - x[3] * a.cam_latency
                px, z = cam.project(b_c + b_R @ np.array([u, w, BLADE_HT + BALL_R]), hp, hR)
                if z > 0.05 and np.all(np.isfinite(px)):
                    hints["expect"] = (float(px[0]), float(px[1]))
            return hints

        def ball_estimate():
            """Latest result of the detector thread: (u, w) or None, the frame, its capture time."""
            if a.sim:  # simulated time: run the detector inline, on new frames only
                f = robot.camera.latest()
                if f is not None and f is not det_out.get("_last_f") and (det_out["frame"] is None or f.received_at != det_out["frame"].received_at):
                    t_cap = f.received_at - a.cam_latency
                    i = max(0, bisect.bisect_right(hist_t, t_cap) - 1)
                    st = hist_s[i] if hist_s else robot.get_state()
                    q = np.array(st.joint_pos)
                    kin_cam.update(fw_to_mjcf(q[ARM_IDX]), waist=q[WAIST_IDX], neck_yaw=q[NECK_YAW_IDX], neck_pitch=q[NECK_PITCH_IDX], base_quat=st.quat)
                    t_d = time.perf_counter()
                    img = f.to_numpy()
                    if rec:
                        rec.add(img, f.received_at, t_cap)
                    roi = blade_roi(cam, *kin_cam.head_pose(), *kin_cam.blade_frame(), BLADE_R + 0.04, step=3) if a.roi else None
                    det = detect_roi(img, ball, roi, step=3, **det_hints())
                    det_times.append(time.perf_counter() - t_d)
                    uw, Rz = (None, None) if det is None else ball_on_blade_cov(cam, det[:2], *kin_cam.head_pose(), *kin_cam.blade_frame())
                    det_out.update(uw=uw, R=Rz, t_cap=t_cap, frame=f)
            with det_lock:
                return det_out["uw"], det_out["frame"], det_out["t_cap"], det_out["R"]

        if not a.sim:
            threading.Thread(target=detector, daemon=True).start()

        # one session: WAIT (blade held level by inverse kinematics) -> ball seen on the blade -> BALANCE (policy)
        # -> ball lost -> WAIT again, without moving back. Ends after --duration or on Ctrl+C.
        if a.sim:
            for _ in range(10):
                robot.trajectory(hold, **gains)
        else:
            if a.yes:
                print("(--yes) stencil paper off the paddle, nothing white around it: starting")
            else:
                input("Stencil paper OFF the paddle, nothing white around it? Enter ")
        q_target = fw_to_mjcf(hold[ARM_IDX])  # the start pose: taught flat pose, levelled for the actual lean (= HOME in sim)
        q_home = q_target.copy()
        if early_guard is not None:
            guard_ctx, early_guard = early_guard, None  # already running
        else:
            guard_ctx = TiltGuard(robot, on_trip=lambda: robot.set_joints(start, duration=2.0, wait=False)) if not a.sim else None
        from scipy.spatial.transform import Rotation
        cam_axis = Rotation.from_rotvec(cam.p[:3]).as_matrix()[:, 2]  # the calibrated optical axis, in the head frame
        kin_look, kin_level = Kinematics(), Kinematics()
        neck = np.array([hold[NECK_YAW_IDX], hold[NECK_PITCH_IDX]])
        if guard_ctx and not guard_ctx.running:
            guard_ctx.__enter__()
        mode, seen, n_catch, t_start_bal = "wait", 0, 0, 0.0
        recent = []  # consecutive on-blade detections while waiting (t, uw): balancing starts from a straight, slow track
        t_rearm = 0.0  # after a loss, no new start before this time (the hand is putting the ball back)
        last_logged = None
        print(f"detector: {'blade region only (--roi)' if a.roi else 'whole frame'}")
        hand_wait = 0
        last_seen, last_uw, last_ft = -1.0, np.zeros(2), None
        t_prev = t_last = None
        n_steps = int(a.duration / PERIOD)
        reason = "time up"
        tick = time.monotonic()
        stall, last_cmd = 0, hold.copy()
        print("WAIT: arm at the home pose; put the ball on the paddle (Ctrl+C to end)")
        try:
            stop_file = os.path.join(os.getcwd(), "STOP")
            if os.path.exists(stop_file):
                os.remove(stop_file)
            for k in range(n_steps):
                t = k * PERIOD
                if k % 25 == 0 and os.path.exists(stop_file):  # touch STOP (e.g. over ssh): end now, hand back to balance
                    os.remove(stop_file); print("STOP file: ending the session", flush=True)
                    break
                st = get_state()
                if not a.sim and state_age(st) > STALL_S:  # Wi-Fi stall: re-send the last setpoint, compute nothing from old data
                    if not stall:
                        print(f"  link stall at t={t:.1f} s: holding the pose", flush=True)
                    stall += 1
                    robot.trajectory(last_cmd, **gains)
                    tick += PERIOD; time.sleep(max(0.0, tick - time.monotonic()))
                    continue
                if stall:
                    print(f"  link back after {stall * PERIOD:.2f} s", flush=True); stall = 0
                q = np.array(st.joint_pos)
                q_arm = fw_to_mjcf(q[ARM_IDX])
                qd_arm = np.array([st.joints[i].vel or 0.0 for i in ARM_IDX]) / ARM_SIGN
                quat = st.quat if st.quat is not None else (1, 0, 0, 0)  # IMU orientation (the body leans on the gantry)
                uw, f, t_cap, det_R = ball_estimate()
                t_now = st.received_at  # the clock of the frames' capture times (monotonic on Albert, simulated time in sim)
                if a.truth_ball:
                    uw, t_cap, det_R = robot.ball_truth(), t_now, None
                fresh = uw is not None and t_cap != last_ft
                new_frame = t_cap is not None and t_cap != last_logged
                accepted = None
                if fresh:
                    last_ft = t_cap
                obs_b.tick(t_now, q_arm, quat, q[WAIST_IDX])  # the ball predictor advances with the blade's tilt
                t_prev, t_last = t_last, t_now
                act = np.zeros(5)
                if mode == "wait":
                    if a.level and k % 5 == 0:  # level the blade (IK on the arm model + IMU), 10 Hz, at most 0.02 rad per update
                        dq, tilt = level_dq(kin_level, q_arm, quat, q[WAIST_IDX])
                        q_target = np.clip(q_target + np.clip(0.5 * dq, -0.02, 0.02), q_lo, q_hi)
                    elif not a.level:  # back to the home pose the policy starts from, slowly (0.3 rad/s at most)
                        q_target = q_target + np.clip(q_home - q_target, -0.006, 0.006)  # the (levelled) start pose
                    if fresh:
                        on = uw is not None and np.hypot(*uw) < BLADE_R - BALL_R  # well inside the blade
                        if on and (not recent or t_cap - recent[-1][0] < 0.3):
                            recent = (recent + [(t_cap, np.asarray(uw))])[-a.start_frames:]
                            last_uw = np.asarray(uw)
                        else:
                            recent = [(t_cap, np.asarray(uw))] if on else []
                    go = None
                    if len(recent) >= a.start_frames and t >= t_rearm:
                        # a ball resting or rolling moves on a straight line at a steady speed; one in the hand or bouncing
                        # does not. Velocity from the line fit, not from two frames (a hand's motion made the predictor
                        # throw the ball off the blade at once: 2026-11-22 G3, 9 starts lost within 0.1 s)
                        tt = np.array([r_[0] for r_ in recent]); pp = np.array([r_[1] for r_ in recent])
                        A = np.c_[np.ones_like(tt), tt - tt[-1]]
                        coef, *_ = np.linalg.lstsq(A, pp, rcond=None)
                        resid = np.max(np.linalg.norm(A @ coef - pp, axis=1))
                        if resid < 0.004 and np.hypot(*coef[1]) < 0.25:
                            go = coef
                    if go is not None:  # the ball is on the blade: go
                        mode, n_catch, t_start_bal, last_seen = "balance", n_catch + 1, t, t
                        act_prev = np.zeros(5)
                        if a.controller == "mpc":  # a new balancing: from the arm's current targets, pulled back to the start pose
                            policy.reset(q_target=q_target, pull_to=q_home)
                        t0_ = recent[0][0] - recent[-1][0]
                        obs_b.start(recent[0][0], go[0] + go[1] * t0_, recent[-1][0], go[0])  # the fitted line's ends
                        recent = []
                        obs_b.tick(t_now, q_arm, quat, q[WAIST_IDX])
                        print(f"BALANCE #{n_catch} at t={t:.1f} s (ball at u={100*last_uw[0]:+.1f}, w={100*last_uw[1]:+.1f} cm)")
                else:
                    if fresh:
                        accepted = obs_b.measure(t_cap, uw, det_R)  # gated: a detection far from the prediction is ignored
                    if fresh and accepted:
                        last_seen = t
                    x = obs_b.ball()
                    if x is not None:
                        last_uw = x[:2]
                    if t - last_seen > LOST_AFTER_S or x is None or np.hypot(*x[:2]) > BLADE_R - 0.5 * BALL_R:
                        print(f"  ball lost after {t - t_start_bal:.1f} s: WAIT (arm glides back to home), put the ball back")
                        mode, recent, t_rearm = "wait", [], t + 0.5
                    else:
                        obs = obs_b.build(q_arm, qd_arm, st.gravity, st.gyro)
                        act = np.clip(policy(obs), -1, 1)
                        if a.action_filter:  # what the arm gets is what the policy sees as its last action
                            act = a.action_filter * act_prev + (1 - a.action_filter) * act
                        act_prev = act
                        obs_b.push_action(act)
                        step_scale = float(np.clip((t_now - t_prev) / PERIOD, 1.0, 1.5)) if t_prev is not None else 1.0
                        q_target = np.clip(q_target + act * ACTION_SCALE * step_scale, q_lo, q_hi)  # same rad/s at a slower loop
                cmd = hold.copy(); cmd[ARM_IDX] = mjcf_to_fw(q_target)
                cmd[NECK_YAW_IDX], cmd[NECK_PITCH_IDX] = neck
                robot.trajectory(cmd, **gains); last_cmd = cmd
                rows.append([t, mode == "balance", *last_uw, *q_arm, *act, t_now, *(obs_b._R.T @ [0, 0, -1.0])[:2]]
                            + ([*robot.ball_truth()] if a.sim else []))
                if new_frame:  # every camera frame: what the detector said, whether the predictor took it, the estimate
                    last_logged = t_cap
                    xk = obs_b.ball()
                    frame_rows.append([t_now, t_cap, *(uw if uw is not None else (np.nan, np.nan)), mode, accepted,
                                       *(xk if xk is not None else [np.nan] * 4), *obs_b.kf.bias(), *obs_b._R.T[:2] @ [0, 0, -1.0]]
                                      + ([*robot.ball_truth()] if a.sim else []))
                if a.sim:
                    if k == 100:
                        robot.place_ball(*a.place)  # the "hand" puts the ball on the blade after 2 s
                    elif a.hand and mode == "wait" and k > 100 and np.hypot(*robot.ball_truth()) > BLADE_R:
                        hand_wait += 1
                        if hand_wait >= 100:  # 2 s after the ball was lost, put it back as a person does: off-centre and rolling
                            hr = robot.env.rng
                            pr, pa, va = 0.035 * np.sqrt(hr.uniform()), hr.uniform(0, 2 * np.pi), hr.uniform(0, 2 * np.pi)
                            sp = hr.uniform(0, a.hand_speed)
                            robot.place_ball(pr * np.cos(pa), pr * np.sin(pa), vel=(sp * np.cos(va), sp * np.sin(va)))
                            hand_wait = 0
                    if a.kick and k > 0 and k % 150 == 0:
                        e = robot.env; ang = robot.env.rng.uniform(0, 2 * np.pi)
                        _, R = e.blade_frame(); e.d.qvel[e.ball_v:e.ball_v + 3] += R @ [a.kick * np.cos(ang), a.kick * np.sin(ang), 0]
                    if a.video and k % 2 == 0:
                        robot.sync_render()
                        frames.append((robot.rd.qpos.copy(), None if f is None else f.to_numpy()))
                else:
                    tick += PERIOD
                    time.sleep(max(0.0, tick - time.monotonic()))
        except KeyboardInterrupt:
            reason = "Ctrl+C"
        det_stop.set()
        if guard_ctx:
            guard_ctx.__exit__()
            if guard_ctx.tripped:
                reason = f"Albert leaned {guard_ctx.tripped:.0f} deg"
        bal = sum(1 for r in rows if r[1]) * PERIOD
        if det_times:
            print(f"detector: {1000 * np.median(det_times):.1f} ms per frame (median), {1000 * np.percentile(det_times, 95):.1f} ms p95")
        print(f"session {len(rows) * PERIOD:.0f} s ({reason}): {n_catch} catches, {bal:.1f} s balancing in total")
        save_logs()  # before anything else moves the robot
        if a.stance and not a.sim:
            if guard_ctx and guard_ctx.tripped:
                print("tripped: Menlo's balance controller has the robot; nothing more is sent to the joints")
            else:
                # everything stays where it is (arm at home, legs in the stance): lowering the arm free-standing tips Albert
                # forward (sim: 12 -> 20 deg). The SDK re-sends this target in the background while we wait for Enter.
                robot.set_joints(hold, duration=1.0, wait=False, **gains)
                if not a.gantry:  # on the gantry the rope already carries Albert
                    input("Raise the gantry until the rope is taut again, then Enter: control back to Menlo's balance controller ")
                hand_back(robot)
        else:
            print("arm back to its starting pose (3 s)")
            hand_back(robot, start)

    if rec:
        print(rec.close(), flush=True)
    if not a.sim and a.debug_video and dbg:
        import imageio
        from PIL import Image, ImageDraw
        out = []
        for im, uw, tc in dbg:
            pil = Image.fromarray(im); d_ = ImageDraw.Draw(pil)
            d_.text((6, 4), f"t={tc - dbg[0][2]:6.2f}s  " + ("no ball" if uw is None else f"u={100*uw[0]:+.1f} w={100*uw[1]:+.1f} cm"), fill=(255, 255, 0))
            out.append(np.asarray(pil))
        imageio.mimwrite(a.debug_video, out, fps=22, quality=7, macro_block_size=1)
        print("detector video:", a.debug_video)
    save_logs()
    if a.sim and rows:
        r = np.array(rows, float); r = r[r[:, 1] > 0]
        err = np.hypot(r[:, 2] - r[:, -2], r[:, 3] - r[:, -1]) * 1000 if len(r) else np.array([np.nan])
        print(f"camera estimate vs truth: {np.mean(err):.1f} mm mean, {np.percentile(err, 95):.1f} mm p95")
    if a.video and frames:
        write_video(a.video, frames)


def write_video(path, frames):
    """Outside view of the simulated Albert with the head camera inset."""
    import imageio
    import mujoco

    from asimov_pp.model import build
    m = build("render"); d = mujoco.MjData(m); rend = mujoco.Renderer(m, 540, 960)
    out = []
    for qpos, head in frames:
        d.qpos[:] = qpos; mujoco.mj_forward(m, d)
        c = mujoco.MjvCamera(); c.lookat[:] = [qpos[0] + 0.15, qpos[1] - 0.1, 0.85]; c.distance = 1.3; c.azimuth = 200; c.elevation = -20
        rend.update_scene(d, c); img = rend.render().copy()
        if head is not None:
            small = head[::2, ::2]; img[10:10 + small.shape[0], -10 - small.shape[1]:-10] = small
        out.append(img)
    imageio.mimwrite(path, out, fps=25, quality=8, macro_block_size=8)
    print("video:", path)


if __name__ == "__main__":
    main()
