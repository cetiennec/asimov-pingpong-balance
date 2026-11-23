"""Whole-pipeline comparison on the simulated Albert (albert/sim_robot.py): rendered head camera, real detector, Kalman
predictor, run_policy's control loop. Each run is `run_policy --sim --hand` (the ball put back 2 s after each loss, off-centre
and rolling); this sums the catches and the time balanced, and from the run logs: true ball distance from the blade centre
and action change per step (mean |a_t - a_t-1| over joints, while balancing).

    python -m tools.eval_sim_pipeline --runs 6 --duration 60 policy mpc
    SIM_ARM_DYN=albert/arm_dynamics_20261120.json python -m tools.eval_sim_pipeline policy mpc   # the older arm

A controller is 'policy' (--ckpt, default v3_final), 'policy:<ckpt>', 'mpc' or 'mpc:<key=value,...>'.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def run(ctrl, seed, duration, ckpt, outdir):
    args = [sys.executable, "-m", "albert.run_policy", "--sim", "--hand", "--duration", str(duration), "--seed", str(seed),
            "--camera-video", "", "--no-lerobot"]
    if ctrl.startswith("mpc"):
        args += ["--controller", "mpc"] + (["--mpc", ctrl[4:]] if ctrl.startswith("mpc:") else [])
    else:
        args += ["--ckpt", ctrl[7:] if ctrl.startswith("policy:") else ckpt]
    log = os.path.join(outdir, f"{re.sub(r'[^A-Za-z0-9]+', '_', ctrl)}_{seed}.csv")
    args += ["--log", log]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    out = subprocess.run(args, capture_output=True, text=True, env=env).stdout
    m = re.search(r"(\d+) catches, ([\d.]+) s balancing", out)
    catches, bal = (int(m.group(1)), float(m.group(2))) if m else (0, 0.0)
    with open(log) as fh:
        rows = list(csv.reader(fh))
    hdr, data = rows[0], np.array([[float(x) if x not in ("True", "False") else float(x == "True") for x in r] for r in rows[1:]])
    on = data[:, hdr.index("balancing")] > 0
    a = data[:, [hdr.index(f"a{i}") for i in range(5)]]
    tu, tw = data[:, hdr.index("true_u")], data[:, hdr.index("true_w")]
    dist = np.hypot(tu, tw)[on]
    da = np.abs(np.diff(a, axis=0)).mean(1)[on[1:] & on[:-1]]
    return catches, bal, dist, da


def main():
    p = argparse.ArgumentParser()
    p.add_argument("controllers", nargs="+")
    p.add_argument("--runs", type=int, default=6)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--ckpt", default="outputs/albert_v3_final/best.pt")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--outdir", default=None, help="run logs (default: a temporary folder)")
    a = p.parse_args()
    outdir = a.outdir or tempfile.mkdtemp(prefix="sim_pipeline_")
    os.makedirs(outdir, exist_ok=True)
    print("arm:", os.environ.get("SIM_ARM_DYN", "albert/arm_dynamics.json"), " logs:", outdir, flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for ctrl in a.controllers:
            res = list(ex.map(lambda s: run(ctrl, s, a.duration, a.ckpt, outdir), range(a.runs)))
            catches = sum(r[0] for r in res); bal = sum(r[1] for r in res)
            dist = np.concatenate([r[2] for r in res]); da = np.concatenate([r[3] for r in res])
            per = bal / max(catches, 1)
            print(f"{ctrl:40s} {a.runs} x {a.duration:.0f} s: {catches:3d} catches, balanced {bal:6.1f} s "
                  f"({100 * bal / (a.runs * a.duration):4.1f} % of the time), {per:5.1f} s per catch, "
                  f"ball {100 * np.mean(dist):.1f} cm from centre, action change {np.mean(da):.3f}", flush=True)


if __name__ == "__main__":
    main()
