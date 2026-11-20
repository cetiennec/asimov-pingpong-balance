"""A simulated Albert with the slice of the menlo-sdk Robot API that the albert/ scripts use, so they can be run end to end
without the robot (--sim): state in firmware order, stand / set_joints / trajectory, and a head camera.

Physics: the training environment (standing robot, randomized dynamics), right arm driven by the trajectory targets.
Camera: rendered from the full robot model at CAM_FPS, delivered CAM_LATENCY_S late, from a mount the calibration must find
(TRUE_CAM_* below are never read by the calibration or the policy).
Time is simulated: trajectory() advances the physics by one 20 ms control step.
"""
from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np

from albert.common import ARM_IDX, ARM_SIGN, FW_JOINTS, NECK_PITCH_IDX, NECK_YAW_IDX, WAIST_IDX, fw_to_mjcf, mjcf_to_fw
from asimov_pp.env import N_SUB, BalanceEnv, Task
from asimov_pp.model import ARM_JOINTS, BALL_R, BLADE_HT, build, standing_qpos

from pathlib import Path

ARM_DYN_FILE = Path(__file__).resolve().parent / "arm_dynamics.json"
SIM_CAMERA_FILE = Path(__file__).resolve().parent / "sim_camera.json"
# body lean of the simulated Albert on its gantry (IMU gravity, body frame): Albert's of 2026-11-19, what the policies train with
SIM_STAND_GRAVITY = (0.1214, 0.0081, -0.9929)
CAM_W, CAM_H, CAM_FOVY = 640, 360, 70.0
CAM_FPS, CAM_LATENCY_S = 22.0, 0.105  # as measured on Albert (measure_latency.py)
TRUE_CAM_POS = np.array([0.099, 0.004, 0.048])  # in neck_pitch_link, on the face
TRUE_CAM_PITCH_DOWN = np.deg2rad(15.0)         # extra downward tilt of the camera in the head

_STAND_FW = {"L_Hip_Pitch": -0.15, "R_Hip_Pitch": 0.15, "L_Knee": 0.45, "R_Knee": -0.45, "L_Shoulder_Pitch": -0.25,
             "L_Shoulder_Roll": -0.10, "L_Elbow": 0.40, "L_Wrist_Yaw": 0.20, "R_Shoulder_Pitch": 0.25,
             "R_Shoulder_Roll": 0.10, "R_Elbow": -0.40, "R_Wrist_Yaw": -0.20}


class _Frame:
    def __init__(self, img, t, arrival):
        self.img, self.timestamp_ns, self.height, self.width = img, int(t * 1e9), img.shape[0], img.shape[1]
        self.received_at = arrival  # like menlo-sdk: when this script received it (sim clock)

    def to_numpy(self):
        return self.img


class SimAlbert:
    def __init__(self, seed=0, cam_latency_s=CAM_LATENCY_S, cmd_delay_steps=None, arm_dyn=True):
        from albert.common import HOME
        TAUGHT_GRAVITY = SIM_STAND_GRAVITY  # fixed: the sim does not follow the robot's latest teaching
        # Albert's arm as identified from a real session (tools/sysid_arm.py): gains, damping, friction and a 34 ms command delay.
        # Without that file: nominal gains and a 60 ms delay (Albert's arm lagged 50-65 ms more than the nominal sim's).
        dyn = ARM_DYN_FILE if (arm_dyn and ARM_DYN_FILE.exists()) else None
        if cmd_delay_steps is None:
            cmd_delay_steps = 2 if dyn else 3
        # same arm home and body lean as the real Albert (taught with teach_flat.py), so the policy sees the same situation
        self.env = BalanceEnv(Task(max_steps=10**6), seed=seed, home_q=HOME, stand_gravity=TAUGHT_GRAVITY, arm_dyn=dyn)
        self.env.reset()
        self.t = 0.0
        self.q_fw = np.array([_STAND_FW.get(n, 0.0) for n in FW_JOINTS])
        self.target_fw = self.q_fw.copy()
        self.cam_latency = cam_latency_s
        self._cmd_queue = []  # Albert's arm follows commands ~50-65 ms later than the sim's PD: delay them here as well
        self.cmd_delay_steps = cmd_delay_steps
        self.rm = build("render"); self.rd = mujoco.MjData(self.rm)
        self.rm.vis.global_.fovy = CAM_FOVY
        # if Albert's head camera has been calibrated, render the simulated one from it (position, direction, field of view)
        self.cam_pos_head, self.cam_dir_head = TRUE_CAM_POS, np.array([np.cos(TRUE_CAM_PITCH_DOWN), 0, -np.sin(TRUE_CAM_PITCH_DOWN)])
        # the simulated head camera: a fixed mount (albert/sim_camera.json, Albert's calibration of 2026-11-19), so the sim does not
        # change when the real robot is recalibrated, and runs without any robot
        CALIB_FILE = SIM_CAMERA_FILE
        if CALIB_FILE.exists():
            import json
            from scipy.spatial.transform import Rotation
            j = json.loads(CALIB_FILE.read_text()); p = np.array(j["params"])
            self.cam_pos_head, self.cam_dir_head = p[3:6], Rotation.from_rotvec(p[:3]).as_matrix()[:, 2]
            self.rm.vis.global_.fovy = float(np.degrees(2 * np.arctan(j["height"] / 2 / p[6])))
        self.rm.geom_rgba[self.rm.geom("ball_geom").id] = [0.96, 0.96, 0.94, 1.0]  # Albert's ball is white
        for b in ("right_elbow_link", "right_wrist_yaw_link"):  # Albert's forearm and wrist are dark, not white
            bid = self.rm.body(b).id
            for g in range(self.rm.ngeom):
                if self.rm.geom_bodyid[g] == bid:
                    self.rm.geom_rgba[g] = [0.18, 0.18, 0.2, 1.0]
        self.rend = mujoco.Renderer(self.rm, CAM_H, CAM_W)
        self._stand = standing_qpos(self.rm)
        self._frames, self._next_frame_t = [], 0.0
        self.camera = SimpleNamespace(latest=self._latest)
        self.info = SimpleNamespace(dof=25, joint_names=FW_JOINTS, joint_index=FW_JOINTS.index)
        self._set_arm(fw_to_mjcf(self.q_fw[ARM_IDX]))
        self.hide_ball()

    # --- menlo-sdk-like API ---
    def connect(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        self.rend.close()

    def has(self, cap):
        return cap in ("camera", "state", "drive")

    def stand(self, **k):
        return None

    def damp(self, **k):
        return None

    def balance(self, wait=True, **k):
        """Menlo's balance controller taking over: not simulated."""
        return None

    def record(self, path):
        import contextlib
        return contextlib.nullcontext()

    def get_state(self):
        e, d = self.env, self.env.d
        q_fw = self.target_fw.copy()
        q_fw[ARM_IDX] = mjcf_to_fw(d.qpos[e.q_adr])
        v_fw = np.zeros(25); v_fw[ARM_IDX] = d.qvel[e.v_adr] * ARM_SIGN
        quat = d.qpos[3:7].copy()
        R = d.xmat[1].reshape(3, 3)
        gyro = d.qvel[3:6].copy()  # free joint: body-frame angular velocity, as the IMU reports
        joints = tuple(SimpleNamespace(name=n, pos=float(q_fw[i]), vel=float(v_fw[i]), temp=35.0, current=0.0) for i, n in enumerate(FW_JOINTS))
        return SimpleNamespace(joints=joints, joint_pos=tuple(q_fw), gravity=tuple(R.T @ [0, 0, -1.0]), gyro=tuple(gyro),
                               quat=tuple(quat), faulted=False, mode=SimpleNamespace(name="STAND"), battery=None,
                               joint=lambda name: joints[FW_JOINTS.index(name)], received_at=self.t)

    def set_joints(self, positions, duration=2.0, hz=50, wait=True, **k):
        start = self.target_fw.copy(); goal = np.asarray(positions, float)
        n = max(1, int(duration * hz))
        for i in range(1, n + 1):
            s = i / n; s = s * s * s * (10 - 15 * s + 6 * s * s)  # minimum jerk
            self.trajectory(start + s * (goal - start))

    def trajectory(self, positions, kp=None, kd=None, **k):
        e = self.env
        self.target_fw = np.asarray(positions, float)
        self._cmd_queue.append(self.target_fw.copy())
        applied = self._cmd_queue.pop(0) if len(self._cmd_queue) > self.cmd_delay_steps else self._cmd_queue[0]
        e.d.ctrl[e.act_ids] = fw_to_mjcf(applied[ARM_IDX])
        e._set_base(0)
        for _ in range(N_SUB):
            mujoco.mj_step(e.m, e.d)
        self.t += 0.02
        self._camera_tick()

    # --- test helpers (not part of the SDK) ---
    def _set_arm(self, q):
        e = self.env
        e.d.qpos[e.q_adr] = q; e.d.qvel[e.v_adr] = 0; e.d.ctrl[e.act_ids] = q
        mujoco.mj_forward(e.m, e.d)

    def hide_ball(self):
        e = self.env
        e.d.qpos[e.ball_q:e.ball_q + 3] = [0, 0, -5]; e.d.qvel[e.ball_v:e.ball_v + 6] = 0
        mujoco.mj_forward(e.m, e.d)

    def place_ball(self, u, w, still=True, vel=(0.0, 0.0)):
        """Put the ball on the blade at (u, w) m. still=True: hold it there (for calibration snapshots)."""
        e = self.env
        c, R = e.blade_frame()
        e.d.qpos[e.ball_q:e.ball_q + 3] = c + R @ [u, w, BLADE_HT + BALL_R + 0.0005]
        e.d.qpos[e.ball_q + 3:e.ball_q + 7] = [1, 0, 0, 0]; e.d.qvel[e.ball_v:e.ball_v + 6] = 0
        if not still or any(vel):
            d_ = self.env.d; d_.qvel[e.ball_v:e.ball_v + 3] = R @ [vel[0], vel[1], 0.0]
        mujoco.mj_forward(e.m, e.d)

    def ball_truth(self):
        rel, _ = self.env.ball_in_blade()
        return rel[:2]

    # --- camera ---
    def _camera_tick(self):
        if self.t + 1e-9 >= self._next_frame_t:
            self._frames.append((self.t, self._render_head()))
            self._next_frame_t += 1.0 / CAM_FPS
        while len(self._frames) > 1 and self._frames[1][0] <= self.t - self.cam_latency:
            self._frames.pop(0)

    def _latest(self):
        ready = [f for f in self._frames if f[0] <= self.t - self.cam_latency + 1e-9]
        if not ready:
            return None
        t, img = ready[-1]
        return _Frame(img, t, t + self.cam_latency)

    def sync_render(self):
        e, rm, rd = self.env, self.rm, self.rd
        q = self._stand.copy()
        q[0:7] = e.d.qpos[0:7]
        for j, v in zip(ARM_JOINTS, e.d.qpos[e.q_adr]):
            q[rm.joint(j).qposadr[0]] = v
        q[rm.joint("waist_yaw_joint").qposadr[0]] = e.d.qpos[e.waist_q]
        q[rm.joint("neck_yaw_joint").qposadr[0]] = self.target_fw[NECK_YAW_IDX]
        q[rm.joint("neck_pitch_joint").qposadr[0]] = self.target_fw[NECK_PITCH_IDX]
        b = rm.joint("ball_joint").qposadr[0]
        q[b:b + 7] = e.d.qpos[e.ball_q:e.ball_q + 7]
        rd.qpos[:] = q; mujoco.mj_forward(rm, rd)

    def _render_head(self):
        self.sync_render()
        rm, rd = self.rm, self.rd
        hb = rm.body("neck_pitch_link").id
        hp, hR = rd.xpos[hb], rd.xmat[hb].reshape(3, 3)
        pos, fwd = hp + hR @ self.cam_pos_head, hR @ self.cam_dir_head
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 0.5; cam.lookat[:] = pos + fwd * cam.distance
        cam.azimuth = np.degrees(np.arctan2(fwd[1], fwd[0])); cam.elevation = np.degrees(np.arcsin(fwd[2]))
        self.rend.update_scene(rd, cam)
        return self.rend.render().copy()
