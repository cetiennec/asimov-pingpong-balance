"""Load a real Albert session log (albert_run.jsonl, written by menlo-sdk's robot.record) as arrays.

    cmd_t, cmd (N, 25)      trajectory setpoints sent, firmware order (time: the SDK's monotonic clock)
    st_t, q, qd (M, 25)     joint states received; gravity, gyro, quat (M, 3/3/4)
"""
from __future__ import annotations

import json

import numpy as np


def load(path="data/2026-11-19/albert_run.jsonl"):
    ct, cmd, st, q, qd, grav, gyro, quat, recv = [], [], [], [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["kind"] == "sent" and d["name"] == "trajectory":
                ct.append(d["t"]); cmd.append(d["command"]["positions"])
            elif d["kind"] == "state":
                st.append(d["t"]); recv.append(d.get("received_at", d["t"]))
                q.append([j["pos"] for j in d["joints"]]); qd.append([j["vel"] for j in d["joints"]])
                grav.append(d["gravity"]); gyro.append(d["gyro"]); quat.append(d["quat"])
    A = lambda x: np.array(x, float)
    return dict(cmd_t=A(ct), cmd=A(cmd), st_t=A(st), recv=A(recv), q=A(q), qd=A(qd), gravity=A(grav), gyro=A(gyro), quat=A(quat))
