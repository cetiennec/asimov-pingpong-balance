"""Asimov 1 + table-tennis paddle + ball, built from Menlo's 25-DoF MJCF with MjSpec.

Two variants of the same robot:

* ``build("sim")``    what the RL environment steps. Legs are removed; the pelvis keeps its free joint and the waist yaw joint,
                      but both are *prescribed* by the env (standing pose, push sway or the walking clip). Only the five right-arm
                      joints are simulated, with position actuators that use Menlo's hardware PD gains. Visual meshes are dropped.
* ``build("render")`` the full 25-DoF robot with meshes, used for videos: the env's state is copied onto it.

Asimov 1 without a gripper: the right forearm ends in the wrist-yaw actuator, whose output carries the stainless hand-socket cup
ASV1_300_12B. Menlo's 25-DoF MJCF puts the gripper module's wrist block (0.75 kg) on right_wrist_yaw_link; here it is replaced by
the cup (placed by registering the sim forearm on the full-body CAD: same origin, axis within 0.6 deg). The printed holder from
``hardware/paddle_holder.py`` plugs into the cup (4 radial M4) and holds the paddle handle straight ahead on the forearm axis,
so the wrist yaw rolls the blade about the handle and elbow / shoulder pitch tilt it the other way, like carrying a tray.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASIMOV_XML = ROOT.parent / "supermimic_ws" / "asimov-1" / "sim-model" / "asimov_1_25dof" / "xml" / "asimov_1_25dof.xml"
HOLDER_STL = ROOT / "hardware" / "paddle_holder.stl"
CUP_STL = ROOT / "assets" / "ASV1_300_12B_wristframe.stl"  # mm, right_wrist_yaw_link frame

ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
]
# Menlo's hardware PD gains for Asimov 1 (asimov-1-amp, assets/asimov_1.py): kp, kd, effort limit (Nm)
ARM_PD = {
    "right_shoulder_pitch_joint": (57.0, 5.0, 30.0),
    "right_shoulder_roll_joint": (86.0, 5.0, 25.0),
    "right_shoulder_yaw_joint": (96.0, 5.0, 20.0),
    "right_elbow_joint": (40.0, 2.0, 12.0),
    "right_wrist_yaw_joint": (40.0, 2.0, 12.0),
}
PRESCRIBED_JOINTS = ["waist_yaw_joint"]  # besides the pelvis free joint

# Menlo / SuperMimic standing pose (right arm excluded: it is the policy's)
STANDING = {
    "left_hip_pitch_joint": -0.15, "right_hip_pitch_joint": 0.15,
    "left_knee_joint": 0.45, "right_knee_joint": -0.45,
    "left_ankle_pitch_joint": -0.30, "right_ankle_pitch_joint": 0.30,
    "left_shoulder_pitch_joint": -0.25, "left_shoulder_roll_joint": -0.10,
    "left_elbow_joint": 0.40, "left_wrist_yaw_joint": 0.20,
}
STAND_HEIGHT = 0.62

# ---- cup, holder and paddle (m). Frame "paddle_holder": right_wrist_yaw_link frame moved to the cup rim, x = forearm axis ----
CUP_MASS = 0.056          # ASV1_300_12B, 316L (CAD volume 7.05 cm3)
TIP_X = 0.01464           # cup rim, in front of the wrist-yaw joint (full-body CAD, cup registered to 0.2 mm)
HOLDER_LEN = 0.063        # cup rim -> end of the handle cradle
HANDLE_IN = 0.055         # handle length lying in the cradle (held by 3 zip ties)
HANDLE_LEN = 0.100        # standard handle
HANDLE_W, HANDLE_T = 0.026, 0.022
BLADE_R = 0.076           # blade modelled as a disc (ITTF blades are ~150 x 158 mm)
BLADE_HT = 0.005          # half thickness: 6 mm wood + 2 x 2 mm rubber
HANDLE_X0 = HOLDER_LEN - HANDLE_IN  # handle butt against the holder's stop, 8 mm in front of the cup rim
BLADE_X = HANDLE_X0 + HANDLE_LEN + BLADE_R - 0.012  # the blade overlaps the handle end by 12 mm
PADDLE_ROLL_DEG = 135.0   # cradle rotation about the forearm axis (chosen so that home wrist yaw is mid-range, see find_home.py)

BALL_R, BALL_M = 0.020, 0.0027


def _paddle(spec: mujoco.MjSpec, light: bool, roll_deg: float):
    wrist = spec.body("right_wrist_yaw_link")
    # no gripper module: the wrist-yaw link is only the cup
    for g in list(wrist.geoms):
        spec.delete(g)
    wrist.mass, wrist.ipos, wrist.iquat = CUP_MASS, [0.006, 0, 0], [1, 0, 0, 0]
    wrist.fullinertia = [np.nan] * 6  # drop the gripper module's full inertia
    wrist.inertia = [CUP_MASS * 0.0170**2, CUP_MASS * (0.0170**2 / 2 + 0.019**2 / 12), CUP_MASS * (0.0170**2 / 2 + 0.019**2 / 12)]
    wrist.explicitinertial = True
    wrist.add_geom(name="cup_collision", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0188, 0.0097, 0], pos=[0.0052, 0, 0],
                   quat=[0.7071068, 0, 0.7071068, 0], rgba=[0.2, 0.6, 0.2, 0.3], group=3, contype=2, conaffinity=0, density=0)
    if not light:
        spec.add_mesh(name="cup_mesh", file=str(CUP_STL), scale=[1e-3] * 3)
        wrist.add_geom(name="cup_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="cup_mesh", rgba=[0.62, 0.62, 0.65, 1],
                       contype=0, conaffinity=0, group=2, density=0)
    tip = wrist.add_body(name="paddle_holder", pos=[TIP_X, 0, 0])
    # holder (hardware/paddle_holder.py): plug in the cup + flange on the rim, ~45 g printed + 4 M4 screws
    tip.add_geom(name="holder_flange_collision", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.019, 0.0025, 0], pos=[0.0025, 0, 0],
                 quat=[0.7071068, 0, 0.7071068, 0], rgba=[0.2, 0.6, 0.2, 0.3], group=3, contype=2, conaffinity=0, mass=0.035)
    if not light and HOLDER_STL.exists():
        spec.add_mesh(name="paddle_holder_mesh", file=str(HOLDER_STL), scale=[1e-3] * 3)
        tip.add_geom(name="holder_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="paddle_holder_mesh",
                     rgba=[0.12, 0.12, 0.13, 1], contype=0, conaffinity=0, group=2, density=0)  # print it dark: orange = ball colour
    h = np.deg2rad(roll_deg) / 2
    pad = tip.add_body(name="paddle", quat=[np.cos(h), np.sin(h), 0, 0])  # paddle z = blade normal, x = handle
    pad.add_geom(name="holder_cradle_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(HOLDER_LEN - HANDLE_X0) / 2, 0.022, 0.0025],
                 pos=[(HOLDER_LEN + HANDLE_X0) / 2, 0, -HANDLE_T / 2 - 0.0025], rgba=[0.2, 0.6, 0.2, 0.3], group=3, contype=2,
                 conaffinity=0, mass=0.02)
    pad.add_geom(name="handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[HANDLE_LEN / 2, HANDLE_W / 2, HANDLE_T / 2],
                 pos=[HANDLE_X0 + HANDLE_LEN / 2, 0, 0], rgba=[0.72, 0.5, 0.28, 1], contype=2, conaffinity=0, mass=0.08)
    pad.add_geom(name="paddle_blade", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BLADE_R, BLADE_HT, 0], pos=[BLADE_X, 0, 0],
                 rgba=[0.55, 0.36, 0.2, 1], friction=[1.0, 0.005, 0.0001], condim=4, solref=[0.004, 1],
                 contype=2, conaffinity=0, mass=0.09)
    if not light:  # rubbers: red top (+z), black bottom
        for name, z, rgba in (("rubber_red", 1, [0.85, 0.08, 0.08, 1]), ("rubber_black", -1, [0.05, 0.05, 0.05, 1])):
            pad.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BLADE_R - 0.001, 0.0006, 0],
                         pos=[BLADE_X, 0, z * (BLADE_HT + 0.0002)], rgba=rgba, contype=0, conaffinity=0, density=0)
    pad.add_site(name="blade_center", pos=[BLADE_X, 0, BLADE_HT], size=[0.004, 0, 0], rgba=[0, 0, 0, 0])


def _ball(spec: mujoco.MjSpec):
    b = spec.worldbody.add_body(name="ball", pos=[0.3, -0.2, 0.8])
    b.add_freejoint(name="ball_joint")
    b.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], rgba=[1, 0.5, 0.1, 1], condim=4,
               friction=[0.6, 0.005, 0.0001], solref=[0.004, 1], contype=1, conaffinity=3)
    b.mass, b.ipos = BALL_M, [0, 0, 0]
    b.inertia = [2 / 3 * BALL_M * BALL_R**2] * 3
    b.explicitinertial = True


def build(kind: str = "sim", roll_deg: float = PADDLE_ROLL_DEG, stand_pd: dict | None = None) -> mujoco.MjModel:
    """stand_pd (render only): {joint: (kp, kd)} position actuators on the other joints, and feet that touch the floor: the
    whole robot standing on its own, every joint held by PD as under Menlo's trajectory command (albert/sim_freestand.py)."""
    assert kind in ("sim", "render")
    sim = kind == "sim"
    spec = mujoco.MjSpec.from_file(str(ASIMOV_XML))
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 0

    # robot geoms collide with the ball only (contype 2 / conaffinity 0; ball: 1 / 3; floor: 1 / 0)
    for g in spec.geoms:
        if g.name == "floor":
            g.contype, g.conaffinity = 1, 0
        elif g.contype or g.conaffinity:
            g.contype, g.conaffinity = 2, 0
    if sim:
        for leg in ("left_hip_pitch_link", "right_hip_pitch_link"):
            spec.delete(spec.body(leg))
        for e in list(spec.excludes):
            spec.delete(e)
        for s in list(spec.sensors):
            spec.delete(s)
        for j in list(spec.joints):  # freeze everything that is neither prescribed nor the right arm
            if j.name not in ARM_JOINTS + PRESCRIBED_JOINTS and j.type != mujoco.mjtJoint.mjJNT_FREE:
                spec.delete(j)
        spec.delete(spec.geom("floor"))  # the ball leaving the blade ends the episode long before it reaches the floor
        for g in [g for g in spec.geoms if g.contype == 0 and g.conaffinity == 0]:
            spec.delete(g)
        for msh in list(spec.meshes):
            spec.delete(msh)
        for t in list(spec.textures):
            spec.delete(t)
        for mat in list(spec.materials):
            spec.delete(mat)
    _paddle(spec, light=sim, roll_deg=roll_deg)
    _ball(spec)
    for j in ARM_JOINTS:
        kp, kd, eff = ARM_PD[j]
        a = spec.add_actuator(name=j.replace("_joint", ""), target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=kp, kv=kd)
        a.forcelimited, a.forcerange = True, [-eff, eff]
        a.ctrllimited = False
    if stand_pd:
        spec.geom("floor").conaffinity = 2  # the robot's collision geoms (contype 2) now stand on the floor
        for j, (kp, kd) in stand_pd.items():
            a = spec.add_actuator(name=j.replace("_joint", ""), target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=kp, kv=kd)
            a.ctrllimited = False
    if not sim:
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
        spec.visual.quality.shadowsize = 4096
    return spec.compile()


def standing_qpos(m: mujoco.MjModel) -> np.ndarray:
    """qpos of the standing robot (pelvis at the origin, right arm at 0)."""
    q = m.qpos0.copy()
    q[0:3] = [0, 0, STAND_HEIGHT]
    q[3:7] = [1, 0, 0, 0]
    for k, v in STANDING.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, k)
        if jid >= 0:
            q[m.jnt_qposadr[jid]] = v
    return q
