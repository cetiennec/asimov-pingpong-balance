"""The 3D ball predictor (estimator.BallKF3D) through the bounce stage's head-camera model: a ball dropped on the paddle held
still (it bounces a few times and rolls off), seen at 18-30 fps with 70-160 ms latency, Albert-like noise, outliers and lost
frames. The predicted ball NOW (height above the blade and its normal speed, policy inputs [25:27]) against the truth.

    python tests/test_kf3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asimov_pp.env import BalanceEnv, Task  # noqa: E402
from tools.eval_bounce import env_kwargs  # noqa: E402


def errors(episodes=8):
    out = []
    for grav in ("gantry", "free"):
        kw = env_kwargs(0.12, grav)
        e = BalanceEnv(Task(max_steps=150), seed=1, **kw)
        for _ in range(episodes):
            o, done = e.reset(), False
            while not done:
                o, _, te, tr, _ = e.step(np.zeros(5)); done = te or tr
                rel, vrel = e.ball_in_blade()
                out.append([o[0] * 0.076 - rel[0], o[1] * 0.076 - rel[1], o[25] * 0.1 - (rel[2] - 0.025), o[26] * 2 - vrel[2]])
    return np.abs(np.array(out))


def test_kf3():
    E = errors()
    med = np.median(E, 0)
    assert med[0] < 0.02 and med[1] < 0.02, med  # in the blade plane, m
    assert med[2] < 0.025, med                    # height, m
    assert med[3] < 0.3, med                      # normal speed, m/s


if __name__ == "__main__":
    E = errors()
    print("median |error|: u %.1f mm, w %.1f mm, height %.1f mm, normal speed %.2f m/s" % (*(1000 * np.median(E[:, :3], 0)), np.median(E[:, 3])))
    print("90th pct:       u %.1f mm, w %.1f mm, height %.1f mm, normal speed %.2f m/s" % (*(1000 * np.percentile(E[:, :3], 90, 0)), np.percentile(E[:, 3], 90)))
    test_kf3(); print("ok")
