"""Tune the ball predictor (asimov_pp/estimator.py) on a real recording: replay the camera measurements and blade tilts of
albert_demo.csv (record_demo.py) through BallKF as run_policy does (step at 50 Hz, measure each frame at its capture time),
and score what matters to the policy: the estimate NOW against the camera position measured at that same moment (which the
filter only receives a latency later). Robust score: errors are capped at 3 cm (misdetections are not the filter's fault).

    python -m tools.tune_kf [albert_demo.csv]
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

from asimov_pp.estimator import BallKF


def load_run(path):
    """A run_policy session (albert_run.csv + albert_run_frames.csv) in load_demo's format: blade tilt at 50 Hz, detections."""
    r = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    f = np.genfromtxt(path.replace(".csv", "_frames.csv"), delimiter=",", names=True, dtype=None, encoding=None)
    st, g = r["t_state"].astype(float), np.c_[r["g_u"].astype(float), r["g_w"].astype(float)]
    z, tc = np.full((len(st), 2), np.nan), np.full(len(st), np.nan)
    for t_recv, t_c, u, w in zip(f["t"].astype(float), f["t_capture"].astype(float), f["det_u"].astype(float), f["det_w"].astype(float)):
        i = np.searchsorted(st, t_recv)  # the step at which the frame was used
        if i < len(st):
            z[i], tc[i] = (u, w), t_c
    return st, z, tc, g


def load_demo(path):
    r = np.genfromtxt(path, delimiter=",", skip_header=1)
    st, u, w, tc, gu, gw = r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6]
    return st, np.c_[u, w], tc, np.c_[gu, gw]


def replay(data, latency_shift=0.0, **kf):
    """Returns (capture time, measurement, the filter's estimate for that time computed before it arrived)."""
    st, z, tc, g = data
    f = BallKF(dt=0.02, **kf)
    est_t, est_x, last_tc = [], [], None
    for i in range(len(st)):
        f.step(st[i], g[i])
        if np.isfinite(tc[i]) and np.isfinite(z[i, 0]) and tc[i] != last_tc:
            last_tc = tc[i]
            f.measure(tc[i] + latency_shift, z[i])
        x = f.state()
        est_t.append(st[i]); est_x.append(np.full(4, np.nan) if x is None else x)
    est_t, est_x = np.array(est_t), np.array(est_x)
    # unique frames
    ok = np.isfinite(tc) & np.isfinite(z[:, 0])
    tcu, idx = np.unique(tc[ok], return_index=True)
    zu = z[ok][idx]
    # estimate at the frame's capture time, from the steps BEFORE the frame was received (received ~ at its row time)
    recv = st[ok][idx]
    pred = np.array([est_x[max(0, np.searchsorted(est_t, min(t_c, r_) , side="right") - 1)] for t_c, r_ in zip(tcu + latency_shift, recv)])
    return tcu, zu, pred


def score(data, **kf):
    tcu, zu, pred = replay(data, **kf)
    e = np.linalg.norm(pred[:, :2] - zu, axis=1)
    e = e[np.isfinite(e)]
    return float(np.mean(np.minimum(e, 0.03))), float(np.median(e)), len(e)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m tools.tune_kf <run log .csv (with its _frames.csv) or a record_demo csv>")
    path = sys.argv[1]
    import os
    data = load_run(path) if os.path.exists(path.replace(".csv", "_frames.csv")) else load_demo(path)
    from asimov_pp.estimator import KF_PARAMS
    base = dict(KF_PARAMS)
    print("current filter (KF_PARAMS): capped-mean %.1f mm, median %.1f mm over %d frames" % tuple(np.r_[np.array(score(data, **base)[:2]) * 1000, score(data, **base)[2]]))
    # what the estimate is worth without the model: last measurement held (latency uncorrected)
    st, z, tc, g = data
    best = None
    grid = dict(accel_noise=[0.8, 1.5, 3.0], meas_noise=[0.002, 0.004], damping=[0.3, 0.5, 1.0],
                gate=[3.0, 4.0], huber=[None, 1.5], roll_gain=[0.75, 0.87, 1.0], reset_after=[3], bias_noise=[None, 0.02],
                bias_init=[0.07])
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        kf = dict(zip(keys, vals))
        s = score(data, **kf)
        if best is None or s[0] < best[0][0]:
            best = (s, kf)
    print("best grid: capped-mean %.1f mm, median %.1f mm" % (best[0][0] * 1000, best[0][1] * 1000), best[1])
    for sh in (-0.04, -0.02, 0.0, 0.02, 0.04):
        s = score(data, latency_shift=sh, **best[1]); print(f"  latency {1000*sh:+.0f} ms vs the measured 105 ms: capped-mean {1000*s[0]:.1f} mm")
    for gb in ((0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)):
        s = score(data, g_bias=gb, **best[1]); print(f"  g bias {gb}: capped-mean {1000*s[0]:.1f} mm")
    json.dump(dict(best=best[1], score_mm=1000 * best[0][0], source=path), open("albert/kf_tuned.json", "w"), indent=1)
    print("(KF_PARAMS in asimov_pp/estimator.py is what runs; change it there if the best is clearly better, then retrain)")


if __name__ == "__main__":
    main()
