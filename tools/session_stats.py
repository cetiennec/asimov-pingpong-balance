"""Statistics of a run_policy log (data/2026-11-19/albert_run.csv), real or simulated, to compare sim with Albert:
catches, balancing time, how long each catch lasted, ball position spread and oscillation, arm activity."""
from __future__ import annotations

import sys

import numpy as np


def stats(path):
    r = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    bal = np.array([str(x) == "True" for x in r["balancing"]])
    t = r["t"].astype(float)
    starts = np.flatnonzero(bal & ~np.r_[False, bal[:-1]]); ends = np.flatnonzero(bal & ~np.r_[bal[1:], False])
    dur = (t[ends] - t[starts]) + 0.02
    u, w = r["ball_u"].astype(float)[bal], r["ball_w"].astype(float)[bal]
    act = np.c_[[r[f"a{i}"].astype(float) for i in range(5)]].T[bal]
    q = np.c_[[r[n].astype(float) for n in r.dtype.names if n.startswith("q_")]].T[bal]
    def freq(x):
        if len(x) < 100: return np.nan
        x = x - x.mean(); f = np.fft.rfftfreq(len(x), 0.02); P = np.abs(np.fft.rfft(x)) ** 2; m = (f > 0.2) & (f < 5)
        return float(f[m][np.argmax(P[m])])
    long_ = [i for i in range(len(starts)) if dur[i] > 3]
    seg = np.concatenate([np.arange(starts[i], ends[i] + 1) for i in long_]) if long_ else np.array([], int)
    useg, wseg = r["ball_u"].astype(float)[seg], r["ball_w"].astype(float)[seg]
    return dict(catches=len(starts), balancing_s=float(dur.sum()), longest_s=float(dur.max()) if len(dur) else 0.0,
                median_catch_s=float(np.median(dur)) if len(dur) else 0.0,
                ball_r_cm=float(100 * np.mean(np.hypot(u, w))) if len(u) else np.nan,
                ball_osc_hz=(freq(useg), freq(wseg)), arm_speed=float(np.mean(np.abs(np.diff(q, axis=0))) / 0.02) if len(q) > 1 else np.nan,
                action_rms=float(np.sqrt(np.mean(act ** 2))) if len(act) else np.nan)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        s = stats(p)
        print(f"{p}: " + ", ".join(f"{k}={np.round(v, 2) if not isinstance(v, tuple) else tuple(np.round(v, 2))}" for k, v in s.items()))
