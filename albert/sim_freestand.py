"""Albert standing on its own feet (no gantry), simulated: does it stay up while the paddle policy moves the arm?

Under Menlo's trajectory command (what run_policy sends) every joint is position-controlled and the walking / balance policy is
off: the legs are only held stiff at their standing angles. This simulates exactly that: the full 25-DoF robot on the floor,
each joint a PD at Albert's gains (albert/common.py KP / KD), legs at the angles Albert reported after stand() (data/2026-11-19/albert_run.jsonl),
the right arm driven by the commands with the same 60 ms delay as SimAlbert, the head camera rendered as in SimAlbert.
Same API as SimAlbert, so the real pipeline runs unchanged:

    python -m albert.run_policy --sim --freestand --duration 60 --video videos/5_albert_freestand.mp4

Ankles: the firmware's PD acts on the two ankle motors (A, B) of the parallel linkage, pitch = (A - B) / (2 K_PITCH),
roll = -(A + B) / (2 K_ROLL). Seen at the ankle joint, a motor gain k is 2 K^2 k: ~900 Nm/rad in pitch, ~140 in roll (ankle="motor").
ankle="joint" uses the motor gain directly at the joint (110 Nm/rad): the pessimistic case if the firmware maps it otherwise.
"""
from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np

from albert.common import ARM_IDX, FW_JOINTS, KD, KP, NECK_PITCH_IDX, NECK_YAW_IDX, fw_to_mjcf, mjcf_to_fw
from albert.sim_robot import CAM_FOVY, CAM_H, CAM_LATENCY_S, CAM_W, SimAlbert
from asimov_pp.env import N_SUB
from asimov_pp.model import ARM_JOINTS, BALL_R, BLADE_HT, build

K_PITCH, K_ROLL = 2.02, 0.8  # menlo.asimov.robots ANKLE_K_PITCH / ANKLE_K_ROLL
# Albert after stand(), 10 s into the last run (data/2026-11-19/albert_run.jsonl), firmware order; body leaning 7.6 deg forward, feet flat
STAND_FW = np.array([-0.352, 0.061, 0.008, 0.545, -0.752, 0.598, 0.318, -0.076, -0.015, -0.564, 0.734, -0.628,
                     -0.27, -0.009, -0.021, 0.362, -0.011, 0.722, 0.868, -0.842, -1.391, -0.226, -0.016, -0.645, -0.459])
# firmware joint -> MJCF joint (same sign convention; the right arm goes through fw_to_mjcf, the ankles through the linkage)
_SIDE = {"L": "left", "R": "right"}
ANKLES = {"L": (4, 5), "R": (10, 11)}


def stance_fw(hip, knee, ankle, base=STAND_FW):
    """base with both legs' pitch joints set (left values; the right leg is mirrored), ankle rolls kept. Firmware order."""
    fw = np.array(base, float).copy()
    fw[0], fw[3], fw[6], fw[9] = hip, knee, -hip, -knee
    for (a, b), pitch in ((ANKLES["L"], ankle), (ANKLES["R"], -ankle)):
        roll = -(base[a] + base[b]) / (2 * K_ROLL)
        fw[a], fw[b] = K_PITCH * pitch - K_ROLL * roll, -K_PITCH * pitch - K_ROLL * roll
    return fw


LEG_IDX = list(range(12))


def _mjcf_name(fw):
    if fw[:2] in ("L_", "R_"):
        return f"{_SIDE[fw[0]]}_{fw[2:].lower()}_joint"
    return fw.lower() + "_joint"


OTHER = [i for i, n in enumerate(FW_JOINTS) if i not in ARM_IDX and "Ankle" not in n]


def fw_to_joint_targets(fw):
    """{MJCF joint: angle} for everything but the right arm."""
    out = {_mjcf_name(FW_JOINTS[i]): float(fw[i]) for i in OTHER}
    for s, (a, b) in ANKLES.items():
        out[f"{_SIDE[s]}_ankle_pitch_joint"] = (fw[a] - fw[b]) / (2 * K_PITCH)
        out[f"{_SIDE[s]}_ankle_roll_joint"] = -(fw[a] + fw[b]) / (2 * K_ROLL)
    return out


def stand_gains(ankle="motor", scale=1.0):
    g = {}
    for i in OTHER:
        g[_mjcf_name(FW_JOINTS[i])] = (KP[i] * scale, KD[i] * scale)
    for s, (a, _) in ANKLES.items():
        kp, kd = KP[a] * scale, KD[a] * scale
        if ankle == "motor":
            g[f"{_SIDE[s]}_ankle_pitch_joint"] = (2 * K_PITCH ** 2 * kp, 2 * K_PITCH ** 2 * kd)
            g[f"{_SIDE[s]}_ankle_roll_joint"] = (2 * K_ROLL ** 2 * kp, 2 * K_ROLL ** 2 * kd)
        else:
            g[f"{_SIDE[s]}_ankle_pitch_joint"] = g[f"{_SIDE[s]}_ankle_roll_joint"] = (kp, kd)
    return g


class _Env:
    """What run_policy reaches into on a simulated robot (blade frame, ball, rng)."""

    def __init__(self, m, d, rng):
        self.m, self.d, self.rng = m, d, rng
        self.blade_geom = m.geom("paddle_blade").id
        self.ball_body = m.body("ball").id
        j = m.joint("ball_joint")
        self.ball_q, self.ball_v = j.qposadr[0], j.dofadr[0]

    def blade_frame(self):
        return self.d.geom_xpos[self.blade_geom].copy(), self.d.geom_xmat[self.blade_geom].reshape(3, 3).copy()


class FreeStandAlbert(SimAlbert):
    def __init__(self, seed=0, cam_latency_s=CAM_LATENCY_S, cmd_delay_steps=None, ankle="motor", gain_scale=1.0,
                 floor_friction=1.0, com_shift=(0.0, 0.0), arm_home=None, stance=None):
        from albert.common import HOME
        self.rng = np.random.default_rng(seed)
        m = build("render", stand_pd=stand_gains(ankle, gain_scale))
        m.geom_friction[m.geom("floor").id, 0] = floor_friction
        m.jnt_range[m.joint("neck_pitch_joint").id] = [-0.60, 0.40]  # the MJCF stops at -0.34; Albert holds -0.46 (look down)
        # model error: shift the torso's centre of mass (m)
        tb = m.body("waist_yaw_link").id
        m.body_ipos[tb, :2] += com_shift
        from albert.sim_robot import ARM_DYN_FILE
        if ARM_DYN_FILE.exists():  # Albert's arm as identified (tools/sysid_arm.py): gains, damping, friction, 34 ms delay
            import json
            dyn = json.loads(ARM_DYN_FILE.read_text())
            for k, j in enumerate(ARM_JOINTS):
                a = m.actuator(j.replace("_joint", "")).id
                m.actuator_gainprm[a, 0] *= dyn["kp_scale"][k]; m.actuator_biasprm[a, 1] *= dyn["kp_scale"][k]
                m.actuator_biasprm[a, 2] *= dyn["kd_scale"][k]
                m.dof_frictionloss[m.joint(j).dofadr[0]] = dyn["friction"][k]
        if cmd_delay_steps is None:
            cmd_delay_steps = 2 if ARM_DYN_FILE.exists() else 3
        mujoco.mj_setConst(m, mujoco.MjData(m))
        d = mujoco.MjData(m)
        self.m, self.d = m, d
        self.rm, self.rd = m, d  # one model: physics, head camera and video
        self.env = _Env(m, d, self.rng)
        self.t = 0.0
        self.cam_latency, self.cmd_delay_steps, self._cmd_queue = cam_latency_s, cmd_delay_steps, []
        self.arm_q = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_v = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.act = {m.actuator(i).name: i for i in range(m.nu)}
        self.target_fw = STAND_FW.copy() if stance is None else stance_fw(*stance)
        self.target_fw[ARM_IDX] = mjcf_to_fw(HOME if arm_home is None else arm_home)
        self.pelvis = m.body("pelvis_link").id
        self.imu = m.site("imu_in_pelvis").id
        self.feet = [m.geom(f"{s}_foot{k}_collision").id for s in ("left", "right") for k in range(1, 5)]
        # camera, as SimAlbert
        m.vis.global_.fovy = CAM_FOVY
        # the head camera is Albert's calibrated one, with its full orientation (it rolls with the head)
        import json
        from scipy.spatial.transform import Rotation
        from albert.common import CALIB_FILE
        j = json.loads(CALIB_FILE.read_text()); p = np.array(j["params"])
        self.cam_pos_head, self.cam_R_head = p[3:6], Rotation.from_rotvec(p[:3]).as_matrix()  # columns: right, down, forward
        m.vis.global_.fovy = float(np.degrees(2 * np.arctan(j["height"] / 2 / p[6])))
        m.geom_rgba[m.geom("ball_geom").id] = [0.96, 0.96, 0.94, 1.0]
        for b in ("right_elbow_link", "right_wrist_yaw_link"):
            bid = m.body(b).id
            m.geom_rgba[m.geom_bodyid == bid] = [0.18, 0.18, 0.2, 1.0]
        self.rend = mujoco.Renderer(m, CAM_H, CAM_W)
        self._frames, self._next_frame_t = [], 0.0
        self.camera = SimpleNamespace(latest=self._latest)
        self.info = SimpleNamespace(dof=25, joint_names=FW_JOINTS, joint_index=FW_JOINTS.index)
        self.stats = {"max_tilt_deg": 0.0, "min_height": np.inf, "fell": False, "fell_at": None, "max_slip_mm": 0.0}
        self._place_on_floor()
        self.hide_ball()
        for _ in range(50):  # settle 1 s on the floor before anything happens
            self._apply(self.target_fw); self._physics()
        self.stats.update(max_tilt_deg=0.0, min_height=np.inf)
        self._tilt0 = self.tilt_deg()
        self._feet0 = self.d.geom_xpos[self.feet][:, :2].copy()

    # --- state ---
    def _apply(self, fw):
        d, m = self.d, self.m
        for j, v in fw_to_joint_targets(fw).items():
            d.ctrl[self.act[j.replace("_joint", "")]] = v
        for j, v in zip(ARM_JOINTS, fw_to_mjcf(fw[ARM_IDX])):
            d.ctrl[self.act[j.replace("_joint", "")]] = v

    def _place_on_floor(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        for j, v in fw_to_joint_targets(self.target_fw).items():
            d.qpos[m.joint(j).qposadr[0]] = v
        d.qpos[self.arm_q] = fw_to_mjcf(self.target_fw[ARM_IDX])
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[2] = 1.0
        mujoco.mj_kinematics(m, d)
        # pitch / roll the pelvis so that the left sole is level, then lower it onto the floor
        Rf = d.xmat[m.body("left_ankle_roll_link").id].reshape(3, 3)
        q = np.empty(4); mujoco.mju_mat2Quat(q, Rf.T.flatten())  # rotate the whole robot so that this sole is level
        d.qpos[3:7] = q
        mujoco.mj_kinematics(m, d)
        low = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in self.feet)
        d.qpos[2] -= low - 0.0005
        d.ctrl[:] = 0
        self._apply(self.target_fw)
        mujoco.mj_forward(m, d)

    def tilt_deg(self):
        R = self.d.xmat[self.pelvis].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    def _physics(self):
        for _ in range(N_SUB):
            mujoco.mj_step(self.m, self.d)
        tilt, h = self.tilt_deg(), float(self.d.xpos[self.pelvis][2])
        self.tilt_log = getattr(self, "tilt_log", []); self.tilt_log.append((self.t, tilt))
        s = self.stats
        s["max_tilt_deg"] = max(s["max_tilt_deg"], tilt)
        s["min_height"] = min(s["min_height"], h)
        if hasattr(self, "_feet0"):
            s["max_slip_mm"] = max(s["max_slip_mm"], 1000 * float(np.max(np.linalg.norm(self.d.geom_xpos[self.feet][:, :2] - self._feet0, axis=1))))
        if not s["fell"] and (tilt > 35 or h < 0.35):
            s["fell"], s["fell_at"] = True, self.t

    def get_state(self):
        m, d = self.m, self.d
        q_fw = self.target_fw.copy()
        q_fw[ARM_IDX] = mjcf_to_fw(d.qpos[self.arm_q])
        for i in OTHER:  # measured, as Albert reports them: the neck sags a little under the head's weight
            q_fw[i] = d.qpos[m.joint(_mjcf_name(FW_JOINTS[i])).qposadr[0]]
        for side, (ia, ib) in ANKLES.items():  # ankles: measured pitch / roll back to the two motors, as Albert reports them
            pitch = d.qpos[m.joint(f"{_SIDE[side]}_ankle_pitch_joint").qposadr[0]]
            roll = d.qpos[m.joint(f"{_SIDE[side]}_ankle_roll_joint").qposadr[0]]
            q_fw[ia], q_fw[ib] = K_PITCH * pitch - K_ROLL * roll, -K_PITCH * pitch - K_ROLL * roll
        v_fw = np.zeros(25); v_fw[ARM_IDX] = d.qvel[self.arm_v]
        R = d.xmat[self.pelvis].reshape(3, 3)
        joints = tuple(SimpleNamespace(name=n, pos=float(q_fw[i]), vel=float(v_fw[i]), temp=35.0, current=0.0) for i, n in enumerate(FW_JOINTS))
        return SimpleNamespace(joints=joints, joint_pos=tuple(q_fw), gravity=tuple(R.T @ [0, 0, -1.0]), gyro=tuple(d.qvel[3:6]),
                               quat=tuple(d.qpos[3:7]), faulted=False, mode=SimpleNamespace(name="MOVE"), battery=None,
                               joint=lambda name: joints[FW_JOINTS.index(name)], received_at=self.t)

    def trajectory(self, positions, kp=None, kd=None, **k):
        self.target_fw = np.asarray(positions, float)
        self._cmd_queue.append(self.target_fw.copy())
        applied = self._cmd_queue.pop(0) if len(self._cmd_queue) > self.cmd_delay_steps else self._cmd_queue[0]
        self._apply(applied)
        self._physics()
        self.t += 0.02
        self._camera_tick()

    def rope(self, fraction):
        """The gantry rope carrying this fraction of Albert's weight, pulling up at the top of the torso (harness)."""
        m, d = self.m, self.d
        tb = m.body("waist_yaw_link").id
        w = float(m.body_subtreemass[1]) * 9.81
        d.xfrc_applied[tb, :] = 0
        d.xfrc_applied[tb, 2] = fraction * w
        # applied at the harness point, 0.30 m above the torso origin: add the moment of the force about the body's CoM
        harness = d.xpos[tb] + d.xmat[tb].reshape(3, 3) @ np.array([0.0, 0.0, 0.30])
        d.xfrc_applied[tb, 3:] = np.cross(harness - d.xipos[tb], [0.0, 0.0, fraction * w])

    def balance(self, wait=True, **k):
        """Menlo's balance controller taking over: not simulated (the sim stops driving the legs here)."""
        print("  [sim] balance(): Menlo's walking policy would take the legs from here")

    # --- helpers ---
    def _set_arm(self, q):
        self.d.qpos[self.arm_q] = q; self.d.qvel[self.arm_v] = 0
        mujoco.mj_forward(self.m, self.d)

    def hide_ball(self):
        e = self.env
        self.d.qpos[e.ball_q:e.ball_q + 3] = [0, 0, -5]; self.d.qvel[e.ball_v:e.ball_v + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def place_ball(self, u, w, still=True, vel=(0.0, 0.0)):
        e = self.env
        c, R = e.blade_frame()
        self.d.qpos[e.ball_q:e.ball_q + 3] = c + R @ [u, w, BLADE_HT + BALL_R + 0.0005]
        self.d.qpos[e.ball_q + 3:e.ball_q + 7] = [1, 0, 0, 0]; self.d.qvel[e.ball_v:e.ball_v + 6] = 0
        if not still or any(vel):
            d_ = self.env.d; d_.qvel[e.ball_v:e.ball_v + 3] = R @ [vel[0], vel[1], 0.0]
        mujoco.mj_forward(self.m, self.d)

    def ball_truth(self):
        c, R = self.env.blade_frame()
        return (R.T @ (self.d.xpos[self.env.ball_body] - c))[:2]

    def sync_render(self):
        mujoco.mj_kinematics(self.m, self.d)

    def _render_head(self):
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d)
        hb = m.body("neck_pitch_link").id
        hp, hR = d.xpos[hb], d.xmat[hb].reshape(3, 3)
        pos, R = hp + hR @ self.cam_pos_head, hR @ self.cam_R_head
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.rend.update_scene(d, cam)
        for c in self.rend.scene.camera:  # set the GL camera directly: position, forward, up (= -image down)
            c.pos[:], c.forward[:], c.up[:] = pos, R[:, 2], -R[:, 1]
        return self.rend.render().copy()

    def exact_calibration(self, calib):
        """The head camera this sim renders (Albert's calibration, at the sim's image size)."""
        import json
        from albert.common import CALIB_FILE
        j = json.loads(CALIB_FILE.read_text()); p = list(j["params"]); p[6] *= CAM_H / j["height"]
        return {**calib, "width": CAM_W, "height": CAM_H, "params": p}

    def report(self):
        s = self.stats
        return (f"free-standing: {'FELL at %.1f s' % s['fell_at'] if s['fell'] else 'stayed up'}; body tilt {self._tilt0:.1f} deg "
                f"at rest, max {s['max_tilt_deg']:.1f} deg; pelvis min height {s['min_height']:.3f} m; feet slipped {s['max_slip_mm']:.0f} mm")

    def close(self):
        print(self.report())
        if getattr(self, "tilt_log", None):
            tl = np.array(self.tilt_log); late = tl[tl[:, 0] > 6.0]
            if len(late):
                i = int(np.argmax(late[:, 1]))
                print(f"  lean after the first 6 s: p50 {np.median(late[:, 1]):.1f}, p99 {np.percentile(late[:, 1], 99):.1f}, "
                      f"max {late[i, 1]:.1f} deg at t={late[i, 0]:.1f} s; first 6 s max {tl[tl[:, 0] <= 6.0][:, 1].max():.1f}")
        self.rend.close()
