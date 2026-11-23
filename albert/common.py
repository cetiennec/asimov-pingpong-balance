"""Shared pieces for running the paddle policy on Albert (Asimov 1, menlo-sdk): joint mapping, gains, forward kinematics,
and the observation, built exactly as in asimov_pp/env.py but from the robot's own state.

Joint frame: menlo-sdk reports positions in the firmware's frame, in the firmware's motor order. The arm entries are mapped to
the MJCF joints through ARM_SIGN / ARM_OFFSET below. They default to "same frame" (Menlo's sim-to-real training uses this MJCF),
but NOTHING HERE HAS BEEN CHECKED ON A ROBOT: run albert/check_joints.py first and fix them if a joint moves the wrong way.
"""
from __future__ import annotations

import _thread
import json
import logging
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from asimov_pp.env import ACT_HIST, BLADE_R, DT, HOMES, FdVelTimed  # noqa: F401 (FdVelTimed: same filter as in training)
from asimov_pp.model import ARM_JOINTS, BALL_R, BLADE_HT, build

ROOT = Path(__file__).resolve().parent
logging.getLogger("livekit").setLevel(logging.CRITICAL)  # the video decoder's frame-drop warnings would bury the prompts
CALIB_FILE = ROOT / "camera_calibration.json"          # Albert
CALIB_FILE_SIM = ROOT / "camera_calibration_sim.json"  # the simulated Albert
OFFSET_FILE = ROOT / "arm_offset.json"                  # written by teach_flat.py

# asimov-firmware motor table = menlo-sdk RobotState.joint_pos order (menlo/asimov/robots.py)
FW_JOINTS = (
    "L_Hip_Pitch", "L_Hip_Roll", "L_Hip_Yaw", "L_Knee", "L_Ankle_A", "L_Ankle_B",
    "R_Hip_Pitch", "R_Hip_Roll", "R_Hip_Yaw", "R_Knee", "R_Ankle_A", "R_Ankle_B",
    "L_Shoulder_Pitch", "L_Shoulder_Roll", "L_Shoulder_Yaw", "L_Elbow", "L_Wrist_Yaw",
    "R_Shoulder_Pitch", "R_Shoulder_Roll", "R_Shoulder_Yaw", "R_Elbow", "R_Wrist_Yaw",
    "Waist_Yaw", "Neck_Yaw", "Neck_Pitch",
)
ARM_FW = ["R_Shoulder_Pitch", "R_Shoulder_Roll", "R_Shoulder_Yaw", "R_Elbow", "R_Wrist_Yaw"]
ARM_IDX = [FW_JOINTS.index(n) for n in ARM_FW]
WAIST_IDX, NECK_YAW_IDX, NECK_PITCH_IDX = FW_JOINTS.index("Waist_Yaw"), FW_JOINTS.index("Neck_Yaw"), FW_JOINTS.index("Neck_Pitch")
# q_firmware = ARM_SIGN * q_mjcf + ARM_OFFSET. Directions checked on Albert with check_joints.py on 2026-11-19: all +1.
# Zero offsets are not checked by that test: teach_flat.py measures them (you hold the paddle flat, the arm limp).
ARM_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
_TAUGHT = json.loads(OFFSET_FILE.read_text()) if OFFSET_FILE.exists() else {}
ARM_OFFSET = np.array(_TAUGHT.get("arm_offset", np.zeros(5)))
# the flat pose you taught (firmware angles of the 5 arm joints), used as the starting pose on Albert; None: the model's HOME
FLAT_HOME_FW = np.array(_TAUGHT["flat_home_fw"]) if "flat_home_fw" in _TAUGHT else None
# how Albert leans on its gantry (IMU gravity, body frame), from the taught poses
TAUGHT_GRAVITY = np.mean([s_["gravity"] for s_ in _TAUGHT["samples"]], 0) if "samples" in _TAUGHT else None

# Gains used in training (Menlo's hardware-tuned Asimov 1 table, asimov-1-amp), for every motor in firmware order.
# The arm values are the ones the policy was trained with; the others hold the standing pose. Neck: a guess (Menlo welds it).
_KP = {"Hip_Pitch": 150, "Hip_Roll": 150, "Hip_Yaw": 150, "Knee": 150, "Ankle_A": 110, "Ankle_B": 110,
       "Shoulder_Pitch": 57, "Shoulder_Roll": 86, "Shoulder_Yaw": 96, "Elbow": 40, "Wrist_Yaw": 40,
       "Waist_Yaw": 65, "Neck_Yaw": 40, "Neck_Pitch": 40}
_KD = {"Elbow": 2, "Wrist_Yaw": 2, "Neck_Yaw": 2, "Neck_Pitch": 2}
KP = tuple(float(_KP[n.split("_", 1)[1] if n[:2] in ("L_", "R_") else n]) for n in FW_JOINTS)
KD = tuple(float(_KD.get(n.split("_", 1)[1] if n[:2] in ("L_", "R_") else n, 5)) for n in FW_JOINTS)

# Head pose while balancing: look down and to the right, at the paddle (neck pitch: positive = up; limits -0.34 .. 0.40)
NECK_YAW, NECK_PITCH = -0.50, -0.46
ACTION_SCALE = 0.03
# Arm home pose on Albert, model angles. After teach_flat.py: the taught flat pose (set below, once ARM_OFFSET is known);
# before: the planned one, blade at (0.38, -0.20, 0.88) m.
HOME = HOMES["albert"]
# Camera latency: capture -> frame received by this script. Used to pair each frame with the joint state of its capture time.
# MEASURE IT ON ALBERT (see README); the policy is trained for 40-120 ms.
CAMERA_LATENCY_S = json.loads((ROOT / "camera_latency.json").read_text())["latency_s"] if (ROOT / "camera_latency.json").exists() else 0.06
# Ball colour: albert.ball_camera.BALL_WHITE (default) or BALL_ORANGE. Check calibration_mask.png after calibrating and
# adjust its thresholds if the ball is not white there (or something else is).
BALL = "white"


def _home():
    return (FLAT_HOME_FW - ARM_OFFSET) / ARM_SIGN if FLAT_HOME_FW is not None else HOMES["albert"]


def mjcf_to_fw(q_arm):
    return ARM_SIGN * np.asarray(q_arm) + ARM_OFFSET


def fw_to_mjcf(q_fw_arm):
    return (np.asarray(q_fw_arm) - ARM_OFFSET) / ARM_SIGN


class Kinematics:
    """Forward kinematics on the training model: blade frame and head pose from the robot's joint angles and IMU."""

    def __init__(self):
        self.m = build("render")
        self.d = mujoco.MjData(self.m)
        self.arm_adr = [self.m.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.adr = {n: self.m.joint(n).qposadr[0] for n in ("waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint")}
        self.blade = self.m.geom("paddle_blade").id
        self.head = self.m.body("neck_pitch_link").id

    def update(self, q_arm_mjcf, waist=0.0, neck_yaw=0.0, neck_pitch=0.0, base_quat=(1, 0, 0, 0)):
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[0:3] = [0, 0, 0.62]
        d.qpos[3:7] = base_quat
        d.qpos[self.arm_adr] = q_arm_mjcf
        d.qpos[self.adr["waist_yaw_joint"]] = waist
        d.qpos[self.adr["neck_yaw_joint"]] = neck_yaw
        d.qpos[self.adr["neck_pitch_joint"]] = neck_pitch
        mujoco.mj_kinematics(self.m, d)

    def blade_frame(self):
        """Blade centre (world) and rotation (columns: u along the handle, w, n = normal)."""
        return self.d.geom_xpos[self.blade].copy(), self.d.geom_xmat[self.blade].reshape(3, 3).copy()

    def head_pose(self):
        return self.d.xpos[self.head].copy(), self.d.xmat[self.head].reshape(3, 3).copy()

    def ball_centre_world(self, u, w):
        """World position of a ball resting on the blade at (u, w) (m, blade frame)."""
        c, R = self.blade_frame()
        return c + R @ np.array([u, w, BLADE_HT + BALL_R])


class ObsBuilder:
    """Observation of asimov_pp/env.py (40 values) from robot state + the ball estimator. The ball part comes from the same
    Kalman predictor as in training (asimov_pp/estimator.py): tick() every control step with the blade tilt from the arm
    model + IMU, measure() with each camera frame at its capture time."""

    def __init__(self, kin: Kinematics, paddle_acc: bool = False):
        """paddle_acc: the predictor also gets the blade's in-plane acceleration from the arm's motion (forward kinematics in
        the body frame, second difference over the control steps), as env.py's kf_paddle_acc. Off for the policies (trained
        without it); on for the MPC (albert/mpc.py), which moves the blade sideways, and for policies trained with
        --kf-paddle-acc (uses_paddle_acc). Then the filter runs with KF_PARAMS_ACC, as in training."""
        from asimov_pp.estimator import KF_PARAMS, KF_PARAMS_ACC, BallKF
        self.kin, self.kf = kin, BallKF(dt=DT, **(KF_PARAMS_ACC if paddle_acc else KF_PARAMS))
        self.hist = np.zeros((ACT_HIST, 5))
        self._R = np.eye(3)
        self.paddle_acc, self._pc = paddle_acc, []

    def reset(self):
        self.hist[:] = 0
        self.kf.reset()

    def push_action(self, a):
        self.hist = np.roll(self.hist, 1, axis=0); self.hist[0] = a

    def tick(self, t, q_arm, base_quat, waist):
        self.kin.update(q_arm, waist=waist, base_quat=base_quat)
        c, self._R = self.kin.blade_frame()
        a_p = np.zeros(2)
        if self.paddle_acc:  # blade centre in the body frame -> acceleration over the last two steps -> blade axes
            Rb = self.kin.d.xmat[1].reshape(3, 3)
            self._pc = (self._pc + [(t, Rb.T @ (c - self.kin.d.xpos[1]))])[-3:]
            if len(self._pc) == 3:
                (t0, p0), (t1, p1), (t2, p2) = self._pc
                if t2 > t1 > t0:
                    a_b = ((p2 - p1) / (t2 - t1) - (p1 - p0) / (t1 - t0)) / (0.5 * (t2 - t0))
                    a_p = np.clip((self._R.T @ Rb @ a_b)[:2], -20.0, 20.0)
        self.kf.step(t, (self._R.T @ np.array([0.0, 0.0, -1.0]))[:2], a_p)

    def measure(self, t_cap, uw, R=None):
        return self.kf.measure(t_cap, uw, R)

    def start(self, t1, uw1, t2, uw2):
        """Start the predictor from two detections (position from the second, velocity from both)."""
        self.kf.reset(); self.kf.start(t1, uw1, t2, uw2)

    def ball(self):
        return self.kf.state()

    def build(self, q_arm, qd_arm, gravity_body, gyro):
        g_blade = self._R.T @ np.array([0.0, 0.0, -1.0])
        x = self.kf.state()
        ball = np.zeros(4) if x is None else np.array([x[0] / BLADE_R, x[1] / BLADE_R, x[2] / 0.5, x[3] / 0.5])
        acc = -np.asarray(gravity_body, np.float64)  # standing still: the accelerometer reads -gravity
        return np.concatenate([ball, g_blade, np.asarray(gyro) * 0.25, acc, (np.asarray(q_arm) - HOME) / 0.3,
                               np.asarray(qd_arm) * 0.1, np.zeros(2), self.hist.ravel()]).astype(np.float32)


def uses_paddle_acc(ckpt):
    """Was this policy trained with the predictor that knows the blade's acceleration (train.py --kf-paddle-acc)? Read from
    the args.json next to the checkpoint; False for older policies, or a checkpoint without it."""
    try:
        return bool(json.loads((Path(ckpt).parent / "args.json").read_text()).get("kf_paddle_acc", False))
    except (OSError, ValueError):
        return False


def load_calibration(sim=False):
    f = CALIB_FILE_SIM if sim else CALIB_FILE
    if not f.exists():
        raise SystemExit(f"{f} not found: run python -m albert.calibrate_camera {'--sim' if sim else '--supported'} first")
    return json.loads(f.read_text())


PERIOD = DT


def lean_deg(state):
    """Angle of the body from vertical, from the IMU's projected gravity (deg)."""
    g = np.asarray(state.gravity, float)
    return float(np.degrees(np.arccos(np.clip(-g[2] / np.linalg.norm(g), -1, 1))))


class TiltGuard:
    """Watches the body lean at 20 Hz while a script drives the joints (the balance controller is off then). If Albert leans
    more than MAX_EXTRA deg beyond where it started, or more than MAX_ABS deg, it interrupts the script, whose Ctrl+C handling
    brings the arm back to its starting pose. It is not an emergency stop: keep the E-Stop in Asimov Manager at hand."""
    MAX_EXTRA, MAX_ABS = 9.0, 16.0

    def __init__(self, robot, on_trip=None, max_extra=None, max_abs=None):
        self.robot, self.tripped, self._stop, self.on_trip = robot, None, threading.Event(), on_trip
        self.start_lean = lean_deg(robot.get_state())
        self.running = False
        if max_extra is not None:
            self.MAX_EXTRA = max_extra
        if max_abs is not None:
            self.MAX_ABS = max_abs

    def __enter__(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                lean = lean_deg(self.robot.get_state())
            except Exception:  # noqa: BLE001 - a missing sample is not a trip
                time.sleep(0.05); continue
            if lean > self.start_lean + self.MAX_EXTRA or lean > self.MAX_ABS:
                self.tripped = lean
                print(f"\n!! Albert leans {lean:.0f} deg (started at {self.start_lean:.0f}): stopping, arm back", flush=True)
                if self.on_trip is not None:  # act at once, even if the script is waiting for Enter
                    try:
                        self.on_trip()
                    except Exception as exc:  # noqa: BLE001
                        print("  (could not send the arm back:", exc, ")", flush=True)
                _thread.interrupt_main()
                return
            time.sleep(0.05)


# The ankles are reported as their two motors (A, B); a command re-sending a reported pose must keep each ankle within the
# firmware's limits (pitch 0.35 rad, roll 0.1 rad), or menlo-sdk refuses it. Constants from menlo/asimov/robots.py.
_K_PITCH, _K_ROLL, _PITCH_LIM, _ROLL_LIM = 2.02, 0.8, 0.33, 0.095
_ANKLES = ((4, 5), (10, 11))


def safe_pose(positions):
    """A copy of a reported pose with both ankles clipped into the firmware's pitch / roll limits."""
    p = np.array(positions, float)
    for a, b in _ANKLES:
        pitch = (p[a] - p[b]) / (2 * _K_PITCH)
        roll = -(p[a] + p[b]) / (2 * _K_ROLL)
        pitch, roll = np.clip(pitch, -_PITCH_LIM, _PITCH_LIM), np.clip(roll, -_ROLL_LIM, _ROLL_LIM)
        p[a] = _K_PITCH * pitch - _K_ROLL * roll
        p[b] = -_K_PITCH * pitch - _K_ROLL * roll
    return p


HOME = _home()  # taught flat pose if teach_flat.py has run
import os as _os
if "home_mjcf" in _TAUGHT and _os.environ.get("ALBERT_HOME", "") != "taught":  # a home chosen with the model after teaching
    # (e.g. paddle closer to the body); ALBERT_HOME=taught keeps the taught pose (for policies trained around it)
    HOME = np.array(_TAUGHT["home_mjcf"])
    FLAT_HOME_FW = mjcf_to_fw(HOME)


NECK_YAW_LIM, NECK_PITCH_LIM = (-1.5, 1.5), (-0.46, 0.40)
CAM_IN_HEAD = np.array([0.10, 0.0, 0.045])  # rough camera position on the face (neck_pitch_link frame)


def look_at(kin, q_arm, target_world=None, waist=0.0, base_quat=(1, 0, 0, 0), axis_in_head=(1.0, 0.0, 0.0), q0=(NECK_YAW, NECK_PITCH)):
    """Neck yaw / pitch that point the camera (optical axis axis_in_head, from the calibration if known) at a world point,
    by default the blade centre for these arm angles. Clipped to the neck's range."""
    from scipy.optimize import least_squares
    kin.update(q_arm, waist=waist, base_quat=base_quat)
    tgt = kin.blade_frame()[0] if target_world is None else np.asarray(target_world)
    ax = np.asarray(axis_in_head, float) / np.linalg.norm(axis_in_head)

    def res(nk):
        kin.update(q_arm, waist=waist, neck_yaw=nk[0], neck_pitch=nk[1], base_quat=base_quat)
        hp, hR = kin.head_pose()
        d = tgt - (hp + hR @ CAM_IN_HEAD)
        return np.cross(hR @ ax, d / np.linalg.norm(d))
    r = least_squares(res, np.asarray(q0, float), bounds=([NECK_YAW_LIM[0], NECK_PITCH_LIM[0]], [NECK_YAW_LIM[1], NECK_PITCH_LIM[1]]))
    return float(r.x[0]), float(r.x[1])


def level_dq(kin, q_arm, base_quat=(1, 0, 0, 0), waist=0.0):
    """Inverse kinematics on the arm model: the joint change that turns the blade's normal to vertical (vertical from the IMU,
    through base_quat) while keeping the blade centre where it is. Least squares on the model Jacobian, a little damping."""
    def frame(q):
        kin.update(q, waist=waist, base_quat=base_quat)
        return kin.blade_frame()
    c0, R0 = frame(q_arm)
    n = R0[:, 2]
    J_n, J_c, eps = np.zeros((3, 5)), np.zeros((3, 5)), 1e-4
    for j in range(5):
        dq = np.zeros(5); dq[j] = eps
        c1, R1 = frame(q_arm + dq)
        J_n[:, j], J_c[:, j] = (R1[:, 2] - n) / eps, (c1 - c0) / eps
    A = np.vstack([J_n, 3.0 * J_c, 0.05 * np.eye(5)])
    b = np.r_[np.array([0.0, 0.0, 1.0]) - n, np.zeros(3), np.zeros(5)]
    return np.linalg.lstsq(A, b, rcond=None)[0], float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))


# ---- never DAMP: every script hands Albert back to Menlo's balance controller (balance()) when it ends, whatever happens ----
def take_joints(robot):
    """Before driving the joints: stand() only if Albert is limp (DAMP); in MOVE / STAND it is left as it is."""
    st = robot.get_state()
    if getattr(st.mode, "name", "") not in ("MOVE", "STAND"):
        robot.stand()
        # stand() returns once armed, while the legs may still be moving: wait until they are still (max 4 s), so a pose read
        # next is where the joints really stay
        t0 = time.monotonic()
        while time.monotonic() - t0 < 4.0:
            v = np.abs(np.array([j.vel or 0.0 for j in robot.get_state().joints]))
            if v.max() < 0.05 and time.monotonic() - t0 > 1.0:
                break
            time.sleep(0.05)


def to_balance(robot):
    """Out of joint control into Menlo's balance controller. A bare balance() right after trajectory() is accepted but Albert
    drops to DAMP 2 s later (Asimov Edge's trajectory timeout; tested 2026-11-20); stand() first, then balance(), holds."""
    try:
        robot.stand(timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        print("  (stand():", exc, ")", flush=True)
    robot.balance(timeout=5.0)


# Menlo's STAND posture for the right arm and the head (measured on Albert, logs/B1_stand.jsonl): stand() snaps them there in a
# fraction of a second, so hand_back glides them there slowly first.
STAND_ARM_FW = np.array([0.196, 0.041, 0.001, -0.381, 0.002])
STAND_NECK_FW = np.array([-0.003, -0.005])


def hand_back(robot, pose=None, seconds=4.0):
    """End of a script: glide the right arm and the head to Menlo's STAND posture over `seconds` (legs: `pose`'s if given, else
    where they are), so the stand() that follows does not snap them; then stand() -> balance(): Menlo's walking policy holds
    Albert. Never raises."""
    try:
        target = np.array(pose if pose is not None else robot.get_state().joint_pos, float)
        if pose is None:  # legs as they are now (the stance)
            target = np.array(robot.get_state().joint_pos, float)
        target = safe_pose(target)
        target[ARM_IDX] = STAND_ARM_FW
        target[NECK_YAW_IDX], target[NECK_PITCH_IDX] = STAND_NECK_FW
        robot.set_joints(target, duration=seconds, wait=False, kp=KP, kd=KD)  # same leg stiffness as the runs (free-standing)
        time.sleep(seconds + 0.3)
    except Exception as exc:  # noqa: BLE001
        print("  (could not glide back:", exc, ")", flush=True)
    try:
        to_balance(robot)
        print("control back to Menlo's balance controller (stand() -> balance())", flush=True)
        # check it holds: MOVE now and still MOVE 2.5 s later (Asimov Edge damps a trajectory 2 s after its last setpoint)
        for _ in range(2):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.5:
                time.sleep(0.1)
            mode = getattr(robot.get_state().mode, "name", "?")
            if mode != "MOVE":
                print(f"  !! Albert reports {mode}, not MOVE: balance() again", flush=True)
                to_balance(robot)
        print(f"  Albert: {getattr(robot.get_state().mode, 'name', '?')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("  !! balance() failed:", exc, "-> use Asimov Manager", flush=True)


class BalanceOnExit:
    """with BalanceOnExit(robot): ... -> balance() if the block ends with an error or Ctrl+C (a normal end calls hand_back)."""

    def __init__(self, robot):
        self.robot = robot

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            print(f"\n{exc_type.__name__}: handing Albert to its balance controller", flush=True)
            hand_back(self.robot)
        return False


# ---- link hiccups: the Mac talks to Albert over Wi-Fi --------------------------------------------------------------------------
STALL_S = 0.15  # state older than this: hold the last setpoint, do not compute anything from stale data


def allow_link_hiccups(limit_s=1.5):
    """menlo-sdk refuses to send once the latest state is 0.5 s old: on a Wi-Fi stall the setpoints stop and Asimov Edge damps
    Albert 2 s later even if the link comes back at 0.6 s. Let commands go out up to limit_s (< Edge's 2 s): the scripts then
    re-send the last pose while the state is stale (see STALL_S) and carry on if the link returns."""
    try:
        from menlo.asimov import _preflight
    except ImportError:  # a mock robot (tools/mock_real.py)
        return
    _preflight.MAX_STATE_AGE_S = limit_s


def state_age(st):
    """Seconds since this state arrived (menlo-sdk's State.age_s; 0 for the simulated robots)."""
    return float(getattr(st, "age_s", 0.0) or 0.0)


def unique_log(path):
    """path, or path with _2, _3 ... if a run already used that name (its .csv or .jsonl): never overwrite an earlier run."""
    import os
    root, ext = os.path.splitext(path)
    used = lambda r: os.path.exists(r + ext) or os.path.exists(r + ".jsonl")
    if not used(root):
        return path
    k = 2
    while used(f"{root}_{k}"):
        k += 1
    print(f"{path} exists: logging to {root}_{k}{ext}")
    return f"{root}_{k}{ext}"


# ---- legs: Albert's own standing poses tip over under stiff joint control; every script holds this one ----------------------
CENTRED_STANCE = (-0.50, 0.55, -0.21)  # left hip / knee / ankle pitch (rad), right mirrored: CoM over mid-foot (sim_freestand.py)


def with_stance(pose, stance=CENTRED_STANCE):
    """pose (firmware order) with both legs in the stance; ankle rolls kept."""
    fw = np.array(pose, float).copy()
    hip, knee, ankle = stance
    fw[0], fw[3], fw[6], fw[9] = hip, knee, -hip, -knee
    for (a, b), pitch in (((4, 5), ankle), ((10, 11), -ankle)):
        roll = -(pose[a] + pose[b]) / (2 * _K_ROLL)
        fw[a], fw[b] = _K_PITCH * pitch - _K_ROLL * roll, -_K_PITCH * pitch - _K_ROLL * roll
    return fw


def graceful_signals():
    """SIGTERM / SIGHUP (kill, the ssh session dropping) end a script like Ctrl+C: its BalanceOnExit hands Albert to the balance
    controller, instead of the process dying with Albert under joint control (Edge damps 2 s later)."""
    import signal

    def handler(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGHUP, handler)
