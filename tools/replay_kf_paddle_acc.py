"""Replay the onboard run logs (logs/onboard/G*.csv with _frames.csv) through the ball predictor as Albert runs it, with and
without the blade acceleration from ObsBuilder(paddle_acc=True), and score the "now" estimate against the camera (tune_kf.py).

    PYTHONPATH=. uv run python -m tools.replay_kf_paddle_acc logs/onboard/G*_v3*.csv logs/onboard/G5_v4*.csv
"""
import sys, numpy as np
from albert.common import Kinematics, ObsBuilder, fw_to_mjcf
from asimov_pp.estimator import BallKF, KF_PARAMS, KF_PARAMS_ACC
from tools.tune_kf import load_run

def a_from_obsbuilder(path):
    r = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    q = np.c_[[r[n].astype(float) for n in r.dtype.names if n.startswith("q_right")]].T
    ob = ObsBuilder(Kinematics(), paddle_acc=True); got = []
    ob.kf.step = lambda t, g, a=(0.0, 0.0): got.append(np.asarray(a, float))
    for t, qi in zip(r["t_state"].astype(float), q):
        ob.tick(t, fw_to_mjcf(qi), (1, 0, 0, 0), 0.0)
    return np.array(got)

def errors(data, ap, params):
    st, z, tc, g = data
    f = BallKF(dt=0.02, **params); est_t, est_x, last = [], [], None
    for i in range(len(st)):
        f.step(st[i], g[i], ap[i])
        if np.isfinite(tc[i]) and np.isfinite(z[i, 0]) and tc[i] != last:
            last = tc[i]; f.measure(tc[i], z[i])
        x = f.state(); est_t.append(st[i]); est_x.append(np.full(4, np.nan) if x is None else x)
    est_t, est_x = np.array(est_t), np.array(est_x)
    ok = np.isfinite(tc) & np.isfinite(z[:, 0]); tcu, idx = np.unique(tc[ok], return_index=True)
    zu, recv = z[ok][idx], st[ok][idx]
    pred = np.array([est_x[max(0, np.searchsorted(est_t, min(a, b), side="right") - 1)] for a, b in zip(tcu, recv)])
    e = np.linalg.norm(pred[:, :2] - zu, axis=1); return e[np.isfinite(e)]

E = {"tilt only, KF_PARAMS": [], "ObsBuilder paddle_acc, KF_PARAMS_ACC": []}
for p in sys.argv[1:]:
    d = load_run(p); ap = a_from_obsbuilder(p)
    E["tilt only, KF_PARAMS"].append(errors(d, np.zeros_like(ap), KF_PARAMS))
    E["ObsBuilder paddle_acc, KF_PARAMS_ACC"].append(errors(d, ap, KF_PARAMS_ACC))
for k, v in E.items():
    e = np.concatenate(v); print(f"{k:<40} median {1000*np.median(e):.2f} mm, capped mean {1000*np.mean(np.minimum(e,0.03)):.2f} mm, p90 {1000*np.percentile(e,90):.1f} mm")
