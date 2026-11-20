"""The ball bounces on the blade like a ping-pong ball on a rubber: drop it from 1-30 cm on the paddle held still at Albert's
home pose (bounce stage model, model.build(bounce=True)) and measure the restitution (normal speed out / in) at the first impact.
A 40 mm ball on a table-tennis rubber: 0.85-0.9 (and ~0.88-0.9 on the table, ITTF: 30.5 cm -> 23-25 cm).

    python -m pytest tests/test_ball_bounce.py -s      or      python tests/test_ball_bounce.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asimov_pp.env import BalanceEnv  # noqa: E402
from asimov_pp.model import BALL_R, BLADE_HT  # noqa: E402

HOME_Q = [0.7577, 0.9729, -0.6564, -1.3960, -1.2616]  # train_albert.sh
HEIGHTS = (0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30)


def drop(env, h0, phase=0.0):
    """Restitution, deepest penetration (m), contact time (s) and first rebound apex (m) of a ball dropped from h0 above the
    blade (arm held still). phase: extra start height as a fraction of the distance travelled in one physics step."""
    m, d = env.m, env.d
    env.reset()
    c, R = env.blade_frame()
    n = R[:, 2]
    d.qpos[env.ball_q:env.ball_q + 3] = c + n * (BALL_R + BLADE_HT + h0 + phase * 0.005)
    d.qvel[env.ball_v:env.ball_v + 6] = 0
    qa = d.qpos[env.q_adr].copy()
    hs, v_in, v_out, t_contact = [], None, None, 0
    for _ in range(int(1.2 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        d.qpos[env.q_adr] = qa; d.qvel[env.v_adr] = 0  # paddle held still
        rel, vrel = env.ball_in_blade()
        h = rel[2] - BALL_R - BLADE_HT
        hs.append(h)
        if h < 0:
            t_contact += 1
            if v_in is None:
                v_in = -vrel[2]
        elif v_in is not None and v_out is None:
            v_out = vrel[2]
        elif v_out is not None and vrel[2] <= 0:
            return v_out / v_in, -min(hs), t_contact * m.opt.timestep, h
    raise AssertionError("no rebound")


def measure(n_phase=4):
    env = BalanceEnv("balance", domain_rand=False, home_q=HOME_Q, bounce=(0.05, 0.05), seed=0)
    out = {}
    for h0 in HEIGHTS:
        out[h0] = np.array([drop(env, h0, ph) for ph in np.linspace(0, 1, n_phase, endpoint=False)])
    return out


def test_restitution():
    out = measure()
    e_all = np.concatenate([o[:, 0] for o in out.values()])
    assert 0.83 <= e_all.mean() <= 0.91, e_all.mean()
    assert e_all.min() > 0.75 and e_all.max() < 1.0
    for h0, o in out.items():
        assert o[:, 1].max() < 0.010, (h0, o[:, 1])  # penetration under 1 cm (blade 1 cm thick)
        assert np.all(o[:, 2] <= 0.016)  # contact under 16 ms
    # the classic drop test: 30.5 cm onto the table gives 23-25 cm; on a rubber, with air drag, a bit less
    apex30 = out[0.30][:, 3].mean()
    assert 0.18 < apex30 < 0.26, apex30


if __name__ == "__main__":
    out = measure()
    print("drop (cm)  restitution      penetration (mm)  contact (ms)  first apex (cm)")
    for h0, o in out.items():
        print(f"{100 * h0:8.0f}   {o[:, 0].mean():.3f} +- {o[:, 0].std():.3f}   {1000 * o[:, 1].mean():6.1f}           "
              f"{1000 * o[:, 2].mean():5.1f}         {100 * o[:, 3].mean():5.1f}")
    e = np.concatenate([o[:, 0] for o in out.values()])
    print(f"all: restitution {e.mean():.3f} +- {e.std():.3f} (min {e.min():.3f}, max {e.max():.3f})")
