"""Preview of the paddle holder: alone, in the ASV1_300_12B cup, and on the robot with the paddle -> hardware/paddle_holder_preview.png"""
import imageio
import mujoco
import numpy as np

from asimov_pp.env import HOME_Q
from asimov_pp.model import ARM_JOINTS, CUP_STL, HOLDER_STL, TIP_X, build, standing_qpos

xml = f"""<mujoco><asset><mesh name="h" file="{HOLDER_STL}" scale=".001 .001 .001"/><mesh name="c" file="{CUP_STL}" scale=".001 .001 .001"/></asset>
<visual><global offwidth="900" offheight="700"/><headlight diffuse=".7 .7 .7" ambient=".3 .3 .3"/></visual>
<worldbody><light pos="0 0 1" dir="0 0 -1"/>
<geom name="holder" type="mesh" mesh="h" rgba=".95 .55 .1 1"/>
<geom name="cup" type="mesh" mesh="c" pos="{-TIP_X} 0 0" rgba=".6 .62 .66 1"/></worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 700, 900); views = []
for az, el, cup_alpha in [(35, -25, 0.0), (330, -20, 1.0)]:
    m.geom_rgba[m.geom("cup").id, 3] = cup_alpha
    c = mujoco.MjvCamera(); c.lookat[:] = [0.02, 0, 0]; c.distance = 0.20; c.azimuth = az; c.elevation = el
    r.update_scene(d, c); views.append(r.render())
m2 = build("render"); d2 = mujoco.MjData(m2); q = standing_qpos(m2)
for j, v in zip(ARM_JOINTS, HOME_Q):
    q[m2.joint(j).qposadr[0]] = v
d2.qpos[:] = q; d2.qpos[m2.joint("ball_joint").qposadr[0] + 2] = -5; mujoco.mj_forward(m2, d2)
r2 = mujoco.Renderer(m2, 700, 900)
for dist, az, el, look in [(0.35, 220, -30, "paddle_holder"), (1.6, 150, -15, None)]:
    c = mujoco.MjvCamera(); c.lookat[:] = d2.xpos[m2.body(look).id] if look else [0, 0, 0.75]; c.distance = dist; c.azimuth, c.elevation = az, el
    r2.update_scene(d2, c); views.append(r2.render())
imageio.imwrite("hardware/paddle_holder_preview.png", np.concatenate([np.concatenate(views[:2], 1), np.concatenate(views[2:], 1)], 0)[::2, ::2])
