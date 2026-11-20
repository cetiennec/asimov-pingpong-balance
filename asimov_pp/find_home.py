"""Solve the home pose of the right arm: blade level, its centre 33 cm in front of the pelvis and 20 cm to the right, at 0.80 m,
with every joint as close to the middle of its range as possible. Prints the HOME_Q used by env.py.

    uv run python -m asimov_pp.find_home
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares

from asimov_pp.model import ARM_JOINTS, PADDLE_ROLL_DEG, build, standing_qpos

TARGET = np.array([0.33, -0.20, 0.80])


def solve(roll_deg=PADDLE_ROLL_DEG, target=TARGET):
    m = build("sim", roll_deg=roll_deg); d = mujoco.MjData(m)
    q0 = standing_qpos(m); adr = [m.joint(j).qposadr[0] for j in ARM_JOINTS]
    lo, hi = m.jnt_range[[m.joint(j).id for j in ARM_JOINTS]].T
    mid, span = (lo + hi) / 2, hi - lo
    blade = m.geom("paddle_blade").id

    def res(q):
        d.qpos[:] = q0; d.qpos[adr] = q; mujoco.mj_kinematics(m, d)
        n = d.geom_xmat[blade].reshape(3, 3)[:, 2]
        return np.r_[(d.geom_xpos[blade] - target) * 10, (n - [0, 0, 1]) * 10, 0.3 * (q - mid) / span]

    best = min((least_squares(res, np.random.default_rng(s).uniform(lo, hi), bounds=(lo, hi)) for s in range(30)), key=lambda r: r.cost)
    res(best.x)
    return best.x, d.geom_xpos[blade].copy(), d.geom_xmat[blade].reshape(3, 3)[:, 2].copy(), (best.x - lo) / span


if __name__ == "__main__":
    q, pos, n, frac = solve()
    print("HOME_Q =", np.round(q, 4).tolist())
    print("fraction of joint range:", np.round(frac, 2), "| blade centre", pos.round(3), "normal", n.round(4))
