"""MPC (albert/mpc.py): the box QP is exact, the model behaves (ball rolls down the tilt, a translated blade leaves the ball
behind by lam), and the controller keeps a ball on the sim paddle (one seeded episode, gantry, matched arm).

    python -m pytest tests/test_mpc.py -s
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from albert.mpc import I_G, I_Q, I_R, I_S, I_SR, I_W, AxisModel, MPCParams, box_qp  # noqa: E402


def test_box_qp_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 4
        A = rng.normal(size=(n, n)); H = A @ A.T + 0.1 * np.eye(n); f = rng.normal(size=n) * 3
        x = box_qp(H, f, -np.ones(n) * 0.5, np.ones(n) * 0.5)
        # projected gradient to convergence as the reference
        y = np.zeros(n); L = np.linalg.eigvalsh(H).max()
        for _ in range(20000):
            y = np.clip(y - (H @ y + f) / L, -0.5, 0.5)
        assert np.allclose(x, y, atol=1e-5)


def test_model_physics():
    p = MPCParams(delay=0)
    m = AxisModel(p, 0)
    x = np.zeros(m.n); x[I_G] = x[I_R] = 0.1  # blade tilted, ball at rest
    for _ in range(50):
        x = m.step(x, np.zeros(2))
    v = x[I_W]
    assert abs(v - p.kb * 0.1 * (1 - np.exp(-p.damping)) / p.damping) < 0.02  # rolls down at kb g
    # a blade translated by 1 cm (and settled) leaves the ball lam x 1 cm behind, relative to it
    m = AxisModel(MPCParams(delay=0, damping=0.0), 0)  # without rolling resistance (it drags the ball along the blade)
    x = np.zeros(m.n)
    for k in range(80):
        x = m.step(x, np.array([0.0, 0.01 / p.gain_c[0] if k == 0 else 0.0]))
    p_rel = x[I_Q] - m.lam * x[I_S]
    assert abs(x[I_S] - 0.01) < 1e-4 and abs(x[I_SR] - 0.01) < 1e-9
    assert abs(x[I_Q]) < 1e-9 and abs(p_rel + m.lam * 0.01) < 1e-9  # q = p + lam s: unmoved by the translation


def test_mpc_keeps_the_ball():
    from tools.eval_albert import episode
    ok, t, dist, da = episode(("mpc", "balance", 1, True, "gantry"))
    assert ok and t > 9.9 and dist < 0.03
