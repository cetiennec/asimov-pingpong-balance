"""Asimov 1 balancing a ping-pong ball on a paddle held by its right arm (MuJoCo, Gymnasium-style API).

The policy controls the five right-arm joints at 50 Hz (Asimov's control rate) through Menlo's hardware PD gains. The rest of the
body is *prescribed*: the pelvis pose and the waist yaw follow one of three base motions, chosen per episode:

    stand   standing pose, slightly tilted at random
    push    standing, hit by random pushes: the torso sways and leans (damped oscillation, 1-2.5 Hz, up to 5 cm / 6 deg)
    walk    Menlo's 49 s Asimov walking clip (0.4 m/s with turns), random segment and playback speed, optionally with pushes

and three kinds of disturbance can be switched on: torso pushes, ball kicks (velocity impulses) and shoves on the paddle (external
force). The base is made very heavy and gravity-compensated, and its velocity is set at every control step, so the arm and the
ball see the true paddle velocity (contacts drag the ball along) while the arm cannot move the body.

Observation (40, everything the real robot has: IMU, joint encoders, ball position from the head camera):
    [0:4]   ball (u, w) / blade radius in the blade plane, (vu, vw) / 0.5 m/s from finite differences of noisy, delayed positions
    [4:7]   gravity direction in the blade frame (IMU orientation + forward kinematics)
    [7:10]  torso angular velocity (gyro) x 0.25
    [10:13] torso specific force (accelerometer) / 9.81, in the torso frame
    [13:18] arm joint angles - home, / 0.3 rad           [18:23] arm joint velocities x 0.1
    [23:25] gait clock (sin, cos of the walking controller's phase; 0 when standing)
    [25:40] last 3 actions, newest first
Action: 5 joint-target deltas in [-1, 1] x 0.03 rad.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from asimov_pp.model import ARM_JOINTS, BALL_R, BLADE_HT, BLADE_R, STAND_HEIGHT, build

HOME_Q = np.array([0.6736, 0.4992, -0.5901, -1.0606, -1.6396])  # from asimov_pp/find_home.py: blade at (0.33, -0.20, 0.80) m
# Albert: blade 5 cm further forward and 8 cm higher, (0.38, -0.20, 0.88) m, so the head camera sees all of it
HOME_ALBERT = np.array([1.1617, 0.5536, -1.0749, -0.8447, -1.2251])
HOMES = {"default": HOME_Q, "albert": HOME_ALBERT}
WALK_CLIP = Path(__file__).resolve().parent.parent.parent / "supermimic_ws" / "supermimic" / "motions" / "npz" / "menlo_walk.npz"
DT = 0.02          # 50 Hz control
N_SUB = 10         # 2 ms physics
G = 9.81
FD_ALPHA = 0.5
ACT_HIST = 3
OBS_DIM = 25 + 5 * ACT_HIST


@dataclass
class Task:
    """What an episode contains. Stages of the curriculum are presets of this (see STAGES)."""
    p_walk: float = 0.0          # probability that the episode is a walking one
    push_rate: float = 0.0       # torso pushes per second (Poisson)
    push_scale: float = 1.0      # x amplitude of the pushes
    kick_rate: float = 0.0       # ball kicks per second
    kick_speed: tuple = (0.10, 0.35)
    shove_rate: float = 0.0      # paddle shoves per second
    shove_force: tuple = (4.0, 12.0)
    walk_speed: tuple = (0.8, 1.2)  # clip playback rate
    max_steps: int = 500         # 10 s


STAGES = {
    "balance": Task(),
    "perturb": Task(push_rate=0.4, kick_rate=0.3, shove_rate=0.2),
    "move": Task(p_walk=0.75, push_rate=0.2, push_scale=0.7, kick_rate=0.15, shove_rate=0.1),
}
# evaluation scenarios: one disturbance at a time, plus everything together
EVAL_TASKS = {
    "stand": Task(),
    "pushes": Task(push_rate=0.5),
    "ball kicks": Task(kick_rate=0.4),
    "paddle shoves": Task(shove_rate=0.3),
    "walk": Task(p_walk=1.0),
    "walk + all": Task(p_walk=1.0, push_rate=0.25, push_scale=0.7, kick_rate=0.2, shove_rate=0.15),
}


class FdVel:
    """Ball velocity from successive (noisy) positions: smoothed finite differences, as on the robot (same as the SO-101 project)."""

    def __init__(self, dt=DT):
        self.k = BLADE_R / dt / 0.5
        self.reset()

    def reset(self):
        self.p, self.v = None, np.zeros(2)

    def update(self, p):
        p = np.asarray(p, np.float64)
        if self.p is not None:
            self.v = (1 - FD_ALPHA) * self.v + FD_ALPHA * (p - self.p) * self.k
        self.p = p.copy()
        return np.r_[p, self.v]


class FdVelTimed:
    """FdVel for positions that arrive at a camera's rate: updated only on new frames, with their real time step, in the same
    units as FdVel. Used by the camera observation model below and, unchanged, on Albert (albert/common.py)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.p, self.t, self.v = None, None, np.zeros(2)

    def update(self, p, t):
        p = np.asarray(p, np.float64)
        if self.p is not None and t > self.t:
            self.v = (1 - FD_ALPHA) * self.v + FD_ALPHA * (p - self.p) * BLADE_R / (t - self.t) / 0.5
        self.p, self.t = p.copy(), t
        return np.r_[p, self.v]


# Head-camera model (camera_obs=True): frames at CAM_FPS, each delivered CAM_LATENCY later (randomized per episode)
CAM_FPS = (25.0, 30.0)
CAM_LATENCY = (0.04, 0.12)

_CLIP = None


def walk_clip():
    global _CLIP
    if _CLIP is None:
        d = np.load(WALK_CLIP)
        names = d["joint_names"].tolist()
        _CLIP = dict(pos=d["root_pos_w"].astype(np.float64), quat=d["root_quat_w"].astype(np.float64),
                     joints=d["joint_pos"].astype(np.float64), names=names, waist=d["joint_pos"][:, names.index("waist_yaw_joint")].astype(np.float64),
                     fps=float(d["fps"][0]))
        # gait phase, as a walking controller's clock would give it: Hilbert phase of the band-passed hip pitch motion
        from scipy.signal import butter, filtfilt, hilbert
        hp = _CLIP["joints"][:, names.index("left_hip_pitch_joint")] + _CLIP["joints"][:, names.index("right_hip_pitch_joint")]
        b, a = butter(2, [0.5 / 25, 3.0 / 25], "band")
        _CLIP["phase"] = np.unwrap(np.angle(hilbert(filtfilt(b, a, hp - hp.mean()))))
    return _CLIP


def quat_mul(a, b):
    out = np.empty(4); mujoco.mju_mulQuat(out, a, b); return out


def axis_angle_quat(axis, ang):
    out = np.empty(4); mujoco.mju_axisAngle2Quat(out, np.asarray(axis, np.float64), float(ang)); return out


def quat_rotate(q, v):
    out = np.empty(3); mujoco.mju_rotVecQuat(out, np.asarray(v, np.float64), q); return out


class BalanceEnv:
    def __init__(self, task: Task | str = "balance", domain_rand: bool = True, ball_noise: float = 1.0, max_latency: int = 2,
                 home_penalty: float = 0.05, smooth_penalty: float = 0.01, air_penalty: float = 0.0, camera_obs: bool = False,
                 cam_fps: tuple = CAM_FPS, cam_latency: tuple = CAM_LATENCY, home: str = "default", home_q=None, stand_gravity=None,
                 ball_start=(0.02, 0.03), estimator: str = "fd", act_latency=None,
                 seed: int | None = None, arm_dyn=None, kf_params=None, cam_outliers: float = 0.0, cam_drops: float = 0.0,
                 tilt_bias: float = 0.0, stand_gravity2=None,
                 model: mujoco.MjModel | None = None):
        self.task = STAGES[task] if isinstance(task, str) else task
        self.domain_rand, self.ball_noise, self.max_latency, self.home_penalty = domain_rand, ball_noise, max_latency, home_penalty
        self.smooth_penalty, self.air_penalty, self.camera_obs = smooth_penalty, air_penalty, camera_obs
        self.cam_fps, self.cam_latency = tuple(cam_fps), tuple(cam_latency)
        self.home_q = np.array(home_q, float) if home_q is not None else HOMES[home].copy()
        # ball placed within this radius (m), with up to this speed (m/s) per axis, dropped from up to this height (m, optional)
        self.ball_start = tuple(ball_start) + (0.0,) * (3 - len(ball_start))
        # ball observation from the camera model: "fd" = last frame + finite-difference velocity; "kf" = Kalman filter that
        # predicts the ball NOW from the blade tilt and corrects with each delayed frame (asimov_pp/estimator.py, as on Albert)
        self.estimator = estimator
        # extra command delay, in control steps (min, max); default 0..max_latency. Albert's arm lags ~50-65 ms more than the sim's
        self.act_latency = tuple(act_latency) if act_latency is not None else (0, max_latency)
        if estimator == "kf":
            from asimov_pp.estimator import KF_PARAMS, BallKF
            self.kf = BallKF(dt=DT, **(KF_PARAMS if kf_params is None else kf_params))
        # head camera as measured on Albert (albert_demo.csv): besides the noise below, a share of detections 1-3 cm off
        # (up to cam_outliers), dropped frames and occasional gaps of 0.15-0.4 s; and the arm model's flat off by a few degrees
        self.cam_outliers, self.cam_drops, self.tilt_bias = cam_outliers, cam_drops, tilt_bias
        # frame-to-frame jitter of the detection (m): 1.2-1.5 mm on Albert (MAD), 0.5 mm in the first runs
        self._jitter = 0.0015 if (cam_outliers or cam_drops) else 0.0005
        # a standing body that leans (e.g. Albert on its gantry): orientation whose IMU reads stand_gravity (body frame)
        self.lean = np.array([1.0, 0, 0, 0])
        # stand_gravity2: a second lean; each episode stands at a random lean between the two (e.g. Albert on its gantry ~7 deg
        # and free-standing in the centred stance ~12 deg), so one policy works in both
        self._grav = None if stand_gravity2 is None or stand_gravity is None else (
            np.asarray(stand_gravity, float), np.asarray(stand_gravity2, float))
        if stand_gravity is not None:
            from scipy.spatial.transform import Rotation
            g = np.asarray(stand_gravity, float) / np.linalg.norm(stand_gravity)
            rot, _ = Rotation.align_vectors([[0, 0, -1.0]], [g])
            self.lean = rot.as_quat()[[3, 0, 1, 2]]
        self._fdt = FdVelTimed()
        self.rng = np.random.default_rng(seed)
        self.m = model if model is not None else build("sim")
        m = self.m
        self.d = mujoco.MjData(m)
        self.q_adr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.v_adr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.act_ids = np.array([m.actuator(j.replace("_joint", "")).id for j in ARM_JOINTS])
        self.waist_q, self.waist_v = m.joint("waist_yaw_joint").qposadr[0], m.joint("waist_yaw_joint").dofadr[0]
        self.ball_q, self.ball_v = m.joint("ball_joint").qposadr[0], m.joint("ball_joint").dofadr[0]
        self.ball_body, self.ball_geom = m.body("ball").id, m.geom("ball_geom").id
        self.blade_geom, self.paddle_body = m.geom("paddle_blade").id, m.body("paddle").id
        self.torso_body = m.body("waist_yaw_link").id
        jids = [m.joint(j).id for j in ARM_JOINTS]
        lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        self.q_lo, self.q_hi = mid - 0.97 * half, mid + 0.97 * half
        # prescribed body: very heavy and weightless, so the arm cannot push it and gravity does not pull it between control steps
        arm_bodies = {m.joint(j).bodyid[0] for j in ARM_JOINTS}
        arm_sub = {b for b in range(m.nbody) if any(self._is_desc(b, a) for a in arm_bodies)}
        for b in range(1, m.nbody):
            if b not in arm_sub and b != self.ball_body:
                m.body_gravcomp[b] = 1.0
        pel = m.body("pelvis_link").id
        k = 1e4 / m.body_mass[pel]
        m.body_mass[pel] *= k; m.body_inertia[pel] *= k
        m.dof_armature[self.waist_v] = 1e3
        mujoco.mj_setConst(m, self.d)
        self._nom = dict(gain=m.actuator_gainprm[self.act_ids, 0].copy(), bias=m.actuator_biasprm[self.act_ids, :3].copy(),
                         frc=m.actuator_forcerange[self.act_ids].copy(), arm=m.dof_armature[self.v_adr].copy(),
                         ball_mass=float(m.body_mass[self.ball_body]), ball_inertia=m.body_inertia[self.ball_body].copy(),
                         ball_fric=m.geom_friction[self.ball_geom].copy(), blade_fric=m.geom_friction[self.blade_geom].copy())
        # Albert's arm as identified from a real session (tools/sysid_arm.py, albert/arm_dynamics.json): per joint kp / kd scale
        # and Coulomb friction. None: Menlo's nominal gains, no friction.
        if isinstance(arm_dyn, (str, Path)):
            import json
            arm_dyn = json.loads(Path(arm_dyn).read_text())
        self.arm_dyn = None if arm_dyn is None else dict(kp=np.array(arm_dyn["kp_scale"]), kd=np.array(arm_dyn["kd_scale"]),
                                                          fric=np.array(arm_dyn["friction"]))
        self._apply_arm(1.0, 1.0, 1.0, 1.0)
        self.obs_dim, self.act_dim = OBS_DIM, 5
        self._fd = FdVel()
        self.t = 0

    def _is_desc(self, b, a):
        while b > 0:
            if b == a:
                return True
            b = self.m.body_parentid[b]
        return False

    # ------------------------------------------------------------------ base motion
    def _base_trajectory(self, n):
        """Pelvis position, quaternion and waist yaw for control steps 0..n (inclusive), plus a label."""
        r, T = self.rng, self.task
        walking = r.uniform() < T.p_walk
        if walking:
            c = walk_clip()
            rate = r.uniform(*T.walk_speed)
            f0 = r.uniform(0, len(c["pos"]) - 2 - n * rate)
            f = f0 + np.arange(n + 1) * rate * DT * c["fps"]
            i = np.floor(f).astype(int); a = (f - i)[:, None]
            pos = (1 - a) * c["pos"][i] + a * c["pos"][i + 1]
            q0, q1 = c["quat"][i], c["quat"][i + 1]
            q1 = np.where((np.sum(q0 * q1, 1) < 0)[:, None], -q1, q1)
            quat = (1 - a) * q0 + a * q1
            quat /= np.linalg.norm(quat, axis=1, keepdims=True)
            waist = (1 - a[:, 0]) * c["waist"][i] + a[:, 0] * c["waist"][i + 1]
            pos[:, :2] -= pos[0, :2]
            self._clip_frames = f
            ph = (1 - a[:, 0]) * c["phase"][i] + a[:, 0] * c["phase"][i + 1]
            self._clock = np.stack([np.sin(ph), np.cos(ph)], 1)
        else:
            if self._grav is not None:
                from scipy.spatial.transform import Rotation
                g1, g2 = self._grav
                g = g1 + r.uniform() * (g2 - g1); g /= np.linalg.norm(g)
                rot, _ = Rotation.align_vectors([[0, 0, -1.0]], [g])
                self.lean = rot.as_quat()[[3, 0, 1, 2]]
            pos = np.tile([0.0, 0.0, STAND_HEIGHT + r.uniform(-0.01, 0.01)], (n + 1, 1))
            tilt = axis_angle_quat(np.r_[r.normal(size=2), 0.0], np.deg2rad(r.uniform(0, 1.5))) if self.domain_rand else np.array([1.0, 0, 0, 0])
            quat = np.tile(quat_mul(tilt, quat_mul(axis_angle_quat([0, 0, 1], r.uniform(-np.pi, np.pi)), self.lean)), (n + 1, 1))
            waist = np.full(n + 1, r.uniform(-0.05, 0.05) if self.domain_rand else 0.0)
            self._clip_frames = None
            self._clock = np.zeros((n + 1, 2))
        # pushes: damped oscillation of the torso, translation along the push and a lean about the perpendicular axis
        self.n_pushes, self.push_steps = 0, []
        if T.push_rate > 0:
            t = np.arange(n + 1) * DT
            tp = 1.0 + r.exponential(1 / T.push_rate)
            while tp < t[-1] - 0.5:
                d = r.uniform(-np.pi, np.pi); dirv = np.array([np.cos(d), np.sin(d), 0.0])
                amp, lean = r.uniform(0.01, 0.05) * T.push_scale, np.deg2rad(r.uniform(1, 6)) * T.push_scale
                w, z = 2 * np.pi * r.uniform(1.0, 2.5), r.uniform(0.25, 0.6)
                s = t - tp
                env = np.where(s > 0, np.exp(-z * w * np.maximum(s, 0)) * np.sin(w * np.sqrt(1 - z * z) * np.maximum(s, 0)), 0.0)
                pos += env[:, None] * amp * dirv
                ax = np.cross([0, 0, 1], dirv)
                for k in np.flatnonzero(np.abs(env) > 1e-4):
                    quat[k] = quat_mul(axis_angle_quat(ax, lean * env[k]), quat[k])
                self.n_pushes += 1; self.push_steps.append(int(tp / DT))
                tp += 0.4 + r.exponential(1 / T.push_rate)
        self.walking = walking
        return pos, quat, waist

    def _set_base(self, k):
        """Put the body on its trajectory at control step k and give it the velocity that reaches step k+1."""
        d = self.d
        d.qpos[0:3], d.qpos[3:7], d.qpos[self.waist_q] = self._bp[k], self._bq[k], self._bw[k]
        d.qvel[0:3] = (self._bp[k + 1] - self._bp[k]) / DT
        w = np.empty(3); mujoco.mju_subQuat(w, self._bq[k + 1], self._bq[k])
        d.qvel[3:6] = w / DT
        d.qvel[self.waist_v] = (self._bw[k + 1] - self._bw[k]) / DT

    def _imu(self, k):
        """Gyro and accelerometer of the torso at control step k, from the prescribed trajectory (+ noise)."""
        q = self._bq[k]
        w = np.empty(3); mujoco.mju_subQuat(w, self._bq[k + 1], self._bq[k])
        gyro = w / DT + np.array([0, 0, (self._bw[k + 1] - self._bw[k]) / DT])
        km = max(k, 1)
        acc_w = (self._bp[km + 1] - 2 * self._bp[km] + self._bp[km - 1]) / DT**2
        qi = np.array([q[0], -q[1], -q[2], -q[3]])
        acc = quat_rotate(qi, acc_w + [0, 0, G])
        if self.domain_rand:
            gyro = gyro + self.rng.normal(0, 0.02, 3); acc = acc + self.rng.normal(0, 0.2, 3)
        return gyro, acc

    # ------------------------------------------------------------------ randomization
    def _apply_arm(self, kp_f, kd_f, fric_f, arm_f):
        """Arm gains, damping, friction and armature: nominal (or identified) x the given factors."""
        m, n, a = self.m, self._nom, self.arm_dyn
        kp = n["gain"] * (a["kp"] if a else 1.0) * kp_f
        kv = -n["bias"][:, 2] * (a["kd"] if a else 1.0) * kd_f
        m.actuator_gainprm[self.act_ids, 0] = kp; m.actuator_biasprm[self.act_ids, 1] = -kp; m.actuator_biasprm[self.act_ids, 2] = -kv
        m.dof_frictionloss[self.v_adr] = (a["fric"] if a else 0.0) * fric_f
        m.dof_armature[self.v_adr] = n["arm"] * arm_f

    def _randomize(self):
        m, n, r = self.m, self._nom, self.rng
        if self.arm_dyn is not None:  # around the identified arm: narrower on the gains, wide on friction
            self._apply_arm(r.uniform(0.8, 1.2, 5), r.uniform(0.7, 1.4, 5), r.uniform(0.5, 1.5, 5), r.uniform(0.8, 1.2, 5))
        else:
            kp = n["gain"] * r.uniform(0.75, 1.25, 5); kv = -n["bias"][:, 2] * r.uniform(0.6, 1.6, 5)
            m.actuator_gainprm[self.act_ids, 0] = kp; m.actuator_biasprm[self.act_ids, 1] = -kp; m.actuator_biasprm[self.act_ids, 2] = -kv
        m.actuator_forcerange[self.act_ids] = n["frc"] * r.uniform(0.8, 1.0, (5, 1))
        if self.arm_dyn is None:
            m.dof_armature[self.v_adr] = n["arm"] * r.uniform(0.8, 1.2, 5)
        s = r.uniform(0.85, 1.2)
        # inertia ratio too: on Albert the ball accelerates 0.86-0.89 x the ideal hollow sphere's 3/5 g sin(tilt) (rubber, ball)
        ir = r.uniform(1.0, 1.4) if (self.cam_outliers or self.cam_drops) else 1.0
        m.body_mass[self.ball_body] = n["ball_mass"] * s; m.body_inertia[self.ball_body] = n["ball_inertia"] * s * ir
        m.geom_friction[self.ball_geom] = n["ball_fric"] * [r.uniform(0.6, 1.4), 1, 1]
        m.geom_friction[self.blade_geom] = n["blade_fric"] * [r.uniform(0.6, 1.4), 1, 1]
        # refresh the cached constants (body_invweight0 etc.) after changing masses: with stale values the ball-paddle contact is
        # wrong and the ball hops (this alone took walking success from 100 % to 0 %). Must run before mj_resetData.
        mujoco.mj_setConst(m, self.d)

    # ------------------------------------------------------------------ blade / ball geometry
    def blade_frame(self):
        d = self.d
        c = d.geom_xpos[self.blade_geom]
        R = d.geom_xmat[self.blade_geom].reshape(3, 3)
        return c, R  # R columns: u (along the handle), w, n (normal)

    def ball_in_blade(self):
        c, R = self.blade_frame()
        rel = R.T @ (self.d.xpos[self.ball_body] - c)
        # velocity of the ball relative to the blade material point under it
        vb = np.empty(6); mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.ball_body, vb, 0)
        vp = np.empty(6); mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_GEOM, self.blade_geom, vp, 0)
        vrel = R.T @ (vb[3:] - vp[3:] - np.cross(vp[:3], self.d.xpos[self.ball_body] - c))
        return rel, vrel

    # ------------------------------------------------------------------ API
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r, m, d, T = self.rng, self.m, self.d, self.task
        if self.domain_rand:
            self._randomize()
        mujoco.mj_resetData(m, d)
        self._bp, self._bq, self._bw = self._base_trajectory(T.max_steps + 2)
        self._set_base(0)
        q0 = self.home_q + r.uniform(-0.03, 0.03, 5)
        d.qpos[self.q_adr] = q0
        self._q_target = q0.copy()
        d.ctrl[self.act_ids] = q0
        mujoco.mj_forward(m, d)
        c, R = self.blade_frame()
        ang, rad = r.uniform(0, 2 * np.pi), self.ball_start[0] * np.sqrt(r.uniform())
        pos = c + R @ [rad * np.cos(ang), rad * np.sin(ang), BALL_R + 0.0052 + 0.002 + r.uniform(0, self.ball_start[2])]
        d.qpos[self.ball_q:self.ball_q + 3] = pos
        d.qpos[self.ball_q + 3:self.ball_q + 7] = [1, 0, 0, 0]
        vp = np.empty(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, self.blade_geom, vp, 0)
        d.qvel[self.ball_v:self.ball_v + 3] = vp[3:] + R @ np.r_[r.uniform(-self.ball_start[1], self.ball_start[1], 2), 0.0]
        mujoco.mj_forward(m, d)
        # disturbance schedule
        self._kicks = self._poisson(T.kick_rate, 1.0)
        self._shoves = self._poisson(T.shove_rate, 1.0)
        self._shove_left, self._shove_f = 0, np.zeros(3)
        self.t, self._prev_a = 0, np.zeros(5)
        self._hist = np.zeros((ACT_HIST, 5))
        self._act_delay = int(r.integers(self.act_latency[0], self.act_latency[1] + 1))
        self._ball_delay = int(r.integers(0, self.max_latency + 1))
        self._abuf = [np.zeros(5)] * self._act_delay
        k = r.uniform(0.5, 1.5) * self.ball_noise
        self._bn = dict(k=k, bias=r.normal(0, 0.004, 2) * k, ou=r.normal(0, 0.0025, 2) * k)
        self._fd.reset()
        p = self._ball_meas()
        self._bbuf = [p] * (self._ball_delay + 1)
        self._ball_obs = self._fd.update(self._bbuf[0])
        self._p_out = r.uniform(0, self.cam_outliers) if self.domain_rand else 0.0
        self._p_drop = r.uniform(0, self.cam_drops) if self.domain_rand else 0.0
        self._gap_until = -1.0
        self._g_bias = r.normal(0, self.tilt_bias, 2) if self.domain_rand else np.zeros(2)
        if self.camera_obs:
            self._cam_period, self._cam_lat = 1.0 / r.uniform(*self.cam_fps), r.uniform(*self.cam_latency)
            self._cam_next, self._cam_pending = r.uniform(0, self._cam_period), []
            self._fdt.reset()
            self._ball_obs = self._fdt.update(p, -self._cam_lat)
            if self.estimator == "kf":
                self.kf.reset()
                self.kf.measure(-self._cam_lat, p * BLADE_R)
                self._ball_obs = self._kf_obs()
        return self._obs()

    def _poisson(self, rate, t0):
        if rate <= 0:
            return set()
        out, t = set(), t0 + self.rng.exponential(1 / rate)
        while t < self.task.max_steps * DT - 0.5:
            out.add(int(t / DT)); t += 0.5 + self.rng.exponential(1 / rate)
        return out

    def _kf_obs(self):
        x = self.kf.state()
        return np.zeros(4) if x is None else np.array([x[0] / BLADE_R, x[1] / BLADE_R, x[2] / 0.5, x[3] / 0.5])

    def _ball_meas(self):
        """Ball position in the blade plane as the head-camera estimator would give it (m): bias + slow drift + jitter."""
        rel, _ = self.ball_in_blade()
        b = self._bn
        return (rel[:2] + b["bias"] + b["ou"] + self.rng.normal(0, self._jitter, 2) * b["k"]) / BLADE_R

    def step(self, action):
        m, d, r = self.m, self.d, self.rng
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        self._abuf.append(a)
        self._q_target = np.clip(self._q_target + self._abuf.pop(0) * 0.03, self.q_lo, self.q_hi)
        d.ctrl[self.act_ids] = self._q_target
        # disturbances
        if self.t in self._kicks:
            _, R = self.blade_frame()
            ang = r.uniform(0, 2 * np.pi)
            dv = R @ np.r_[np.cos(ang), np.sin(ang), r.uniform(0, 0.3)] * r.uniform(*self.task.kick_speed)
            d.qvel[self.ball_v:self.ball_v + 3] += dv
        if self.t in self._shoves:
            ang = r.uniform(0, 2 * np.pi)
            self._shove_f = np.r_[np.cos(ang), np.sin(ang), r.uniform(-0.5, 0.5)] * r.uniform(*self.task.shove_force)
            self._shove_left = int(r.integers(5, 11))
        d.xfrc_applied[self.paddle_body, :3] = self._shove_f if self._shove_left > 0 else 0.0
        self._shove_left -= 1
        self._set_base(self.t)
        for _ in range(N_SUB):
            mujoco.mj_step(m, d)
        self.t += 1

        rel, vrel = self.ball_in_blade()
        dist = float(np.hypot(rel[0], rel[1]))
        speed = float(np.hypot(vrel[0], vrel[1]))
        h = rel[2] - BALL_R - BLADE_HT
        # centred, slow ball; smooth actions; the ball stays in contact (jerky paddle motion makes it hop, then it lands off-centre)
        reward = (1.0 - min(dist / BLADE_R, 1.0)) - 0.05 * speed - self.smooth_penalty * float(np.sum((a - self._prev_a) ** 2))
        reward -= self.air_penalty * float(h > 0.003)
        if self.home_penalty:
            reward -= self.home_penalty * float(np.sum(((d.qpos[self.q_adr] - self.home_q) / 0.35) ** 2))
        # noisy, delayed ball observation
        b = self._bn
        b["ou"] = np.exp(-DT / 2.0) * b["ou"] + np.sqrt(1 - np.exp(-DT)) * r.normal(0, 0.0025, 2) * b["k"]
        if self.camera_obs:  # frames captured at the camera rate, seen after the latency, velocity from their timestamps
            now = self.t * DT
            if self.estimator == "kf":  # predict with the blade's tilt (+ IMU-like noise); paddle acceleration left to the noise
                _, R = self.blade_frame()
                g_in = (R.T @ np.array([0, 0, -1.0]))[:2] + (r.normal(0, 0.015, 2) if self.domain_rand else 0) + self._g_bias
                self.kf.step(now, g_in)
            while self._cam_next <= now:
                pm = self._ball_meas()
                if self._p_drop and r.uniform() < 0.25 * self._p_drop:  # the ball unseen for a while (hand, glare)
                    self._gap_until = self._cam_next + r.uniform(0.15, 0.4)
                lost = self._cam_next < self._gap_until or (self._p_drop and r.uniform() < self._p_drop)
                if self._p_out and r.uniform() < self._p_out:  # a misdetection: 1-3 cm off
                    ang = r.uniform(0, 2 * np.pi); pm = pm + np.r_[np.cos(ang), np.sin(ang)] * r.uniform(0.01, 0.03) / BLADE_R
                if not lost:
                    self._cam_pending.append((self._cam_next, pm))
                self._cam_next += self._cam_period
            while self._cam_pending and self._cam_pending[0][0] <= now - self._cam_lat:
                tc, pm = self._cam_pending.pop(0)
                if self.estimator == "kf":
                    self.kf.measure(tc, pm * BLADE_R)
                else:
                    self._ball_obs = self._fdt.update(pm, tc)
            if self.estimator == "kf":
                self._ball_obs = self._kf_obs()
        else:
            self._bbuf.append(self._ball_meas()); self._bbuf.pop(0)
            self._ball_obs = self._fd.update(self._bbuf[0])
        self._prev_a = a
        self._hist = np.roll(self._hist, 1, axis=0); self._hist[0] = a
        fell = dist > BLADE_R - 0.5 * BALL_R or h < -0.01 or h > 0.10
        bad = bool(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        terminated = bool(fell or bad)
        if terminated:
            reward -= 5.0
        truncated = self.t >= self.task.max_steps
        success = bool(truncated and not terminated)
        return self._obs(), float(reward), terminated, truncated, dict(dist=dist, fell=bool(fell), is_success=success, walking=self.walking)

    def _obs(self):
        d = self.d
        _, R = self.blade_frame()
        qt = self._bq[self.t]
        g_world = np.array([0, 0, -1.0])
        g_blade = R.T @ g_world
        if self.domain_rand:  # IMU orientation error ~1 deg
            g_blade = g_blade + self.rng.normal(0, 0.015, 3)
        gyro, acc = self._imu(self.t)
        return np.concatenate([
            self._ball_obs, g_blade, gyro * 0.25, acc / G,
            (d.qpos[self.q_adr] - self.home_q) / 0.3, d.qvel[self.v_adr] * 0.1, self._clock[self.t], self._hist.ravel(),
        ]).astype(np.float32)

    # for rendering on the full robot
    def base_state(self):
        return self._bp[self.t], self._bq[self.t], self._bw[self.t], (None if self._clip_frames is None else self._clip_frames[self.t])
