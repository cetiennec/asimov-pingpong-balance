"""Albert's paddle-balancing sessions as a LeRobot dataset (one episode per run_policy run, 50 Hz).

Per frame:
  observation.state    (31)  25 joint positions (firmware order, rad) + IMU gravity (3, body frame) + gyro (3, rad/s)
  observation.ball     (2)   the Kalman predictor's ball position on the blade (u, w, m); 0 while waiting
  action               (5)   the policy's action (right-arm joint increments / 0.03 rad, in [-1, 1])
  action.arm_target    (5)   the right-arm joint targets actually sent (firmware, rad)
  balancing            (1)   1 while the policy balances the ball, 0 while waiting for it
Joint states and commands come from the menlo-sdk recording (<log>.jsonl), the ball and the actions from the run log
(<log>.csv); the two clocks are aligned on the right-arm angles both contain. No camera frames: these runs kept only a debug
video (which shows people); new runs can save raw frames.

    uv run --with "lerobot[dataset]" python -m tools.export_lerobot --push cetiennec/asimov-pingpong-albert
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np

FW_JOINTS = ("L_Hip_Pitch", "L_Hip_Roll", "L_Hip_Yaw", "L_Knee", "L_Ankle_A", "L_Ankle_B", "R_Hip_Pitch", "R_Hip_Roll", "R_Hip_Yaw",
             "R_Knee", "R_Ankle_A", "R_Ankle_B", "L_Shoulder_Pitch", "L_Shoulder_Roll", "L_Shoulder_Yaw", "L_Elbow", "L_Wrist_Yaw",
             "R_Shoulder_Pitch", "R_Shoulder_Roll", "R_Shoulder_Yaw", "R_Elbow", "R_Wrist_Yaw", "Waist_Yaw", "Neck_Yaw", "Neck_Pitch")
ARM = [17, 18, 19, 20, 21]
TASK = "Balance the ping-pong ball on the paddle"


def load_recording(path):
    st_t, q, g, w, ct, cmd = [], [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                break
            if d["kind"] == "state":
                st_t.append(d["t"]); q.append([j["pos"] for j in d["joints"]]); g.append(d["gravity"]); w.append(d["gyro"])
            elif d["kind"] == "sent" and d["name"] == "trajectory":
                ct.append(d["t"]); cmd.append(d["command"]["positions"])
    A = np.array
    return A(st_t), A(q), A(g), A(w), A(ct), A(cmd)


def align(csv_t, csv_q_fw, st_t, st_q):
    """Offset to add to the run log's time to get the recording's clock: best match of the right-arm motion. Each joint's mean is
    removed first: the run log's angles went through the arm offsets of their day, which may have been re-taught since."""
    csv_q_fw = csv_q_fw - csv_q_fw.mean(0)
    best = None
    for off in np.arange(st_t[0] - csv_t[0], st_t[-1] - csv_t[-1] + 0.01, 0.02):
        qi = np.array([np.interp(csv_t + off, st_t, st_q[:, j]) for j in ARM]).T
        e = float(np.mean((qi - qi.mean(0) - csv_q_fw) ** 2))
        if best is None or e < best[0]:
            best = (e, off)
    # refine
    e0, off0 = best
    for off in np.arange(off0 - 0.02, off0 + 0.02, 0.002):
        qi = np.array([np.interp(csv_t + off, st_t, st_q[:, j]) for j in ARM]).T
        e = float(np.mean((qi - qi.mean(0) - csv_q_fw) ** 2))
        if e < best[0]:
            best = (e, off)
    return best[1], float(np.sqrt(best[0]))


def episodes(roots):
    import sys
    sys.path.insert(0, os.getcwd())
    from albert.common import mjcf_to_fw
    for csv in sorted(sum((glob.glob(os.path.join(r, "*.csv")) for r in roots), [])):
        if csv.endswith("_frames.csv"):
            continue
        rec = csv[:-4] + ".jsonl"
        if not os.path.exists(rec):
            continue
        r = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding=None)
        if "balancing" not in r.dtype.names or len(r) < 100:
            continue
        bal = np.array([str(x) == "True" for x in r["balancing"]], np.float32)
        if bal.sum() < 25:  # under 0.5 s of balancing: nothing to learn from
            continue
        qn = [n for n in r.dtype.names if n.startswith("q_")]
        q_mjcf = np.c_[[r[n].astype(float) for n in qn]].T
        q_fw = np.array([mjcf_to_fw(x) for x in q_mjcf])
        act = np.c_[[r[f"a{i}"].astype(float) for i in range(5)]].T
        ball = np.c_[r["ball_u"].astype(float), r["ball_w"].astype(float)]
        t = r["t"].astype(float)
        st_t, st_q, st_g, st_w, ct, cmd = load_recording(rec)
        off, rms = align(t, q_fw, st_t, st_q)
        tt = t + off
        state = np.c_[np.array([np.interp(tt, st_t, st_q[:, j]) for j in range(25)]).T,
                      np.array([np.interp(tt, st_t, st_g[:, k]) for k in range(3)]).T,
                      np.array([np.interp(tt, st_t, st_w[:, k]) for k in range(3)]).T]
        idx = np.clip(np.searchsorted(ct, tt, side="right") - 1, 0, len(ct) - 1)
        target = cmd[idx][:, ARM]
        yield os.path.basename(csv)[:-4], dict(state=state, ball=ball, action=act, target=target, balancing=bal), rms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=sorted(glob.glob("data/20*")))
    p.add_argument("--out", default="data/lerobot/asimov-pingpong-albert")
    p.add_argument("--push", default=None, help="Hugging Face dataset repo id to push to")
    a = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {
        "observation.state": {"dtype": "float32", "shape": (31,),
                              "names": list(FW_JOINTS) + ["gravity_x", "gravity_y", "gravity_z", "gyro_x", "gyro_y", "gyro_z"]},
        "observation.ball": {"dtype": "float32", "shape": (2,), "names": ["ball_u", "ball_w"]},
        "action": {"dtype": "float32", "shape": (5,), "names": [FW_JOINTS[i] for i in ARM]},
        "action.arm_target": {"dtype": "float32", "shape": (5,), "names": [FW_JOINTS[i] for i in ARM]},
        "balancing": {"dtype": "float32", "shape": (1,), "names": ["balancing"]},
    }
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    ds = LeRobotDataset.create(repo_id=a.push or "local/asimov-pingpong-albert", fps=50, features=features, root=a.out,
                               robot_type="asimov_1", use_videos=False)
    n_ep = 0
    for name, e, rms in episodes(a.data):
        n = len(e["action"])
        for k in range(n):
            ds.add_frame({"observation.state": e["state"][k].astype(np.float32), "observation.ball": e["ball"][k].astype(np.float32),
                          "action": e["action"][k].astype(np.float32), "action.arm_target": e["target"][k].astype(np.float32),
                          "balancing": e["balancing"][k:k + 1].astype(np.float32), "task": TASK})
        ds.save_episode()
        n_ep += 1
        print(f"episode {n_ep - 1}: {name}, {n} frames ({n / 50:.0f} s), {e['balancing'].sum() / 50:.1f} s balancing, "
              f"clock alignment residual {1000 * rms:.1f} mrad")
    ds.finalize()
    print(f"{n_ep} episodes -> {a.out}")
    if a.push:
        ds.push_to_hub(private=False, tags=["asimov-1", "humanoid", "ping-pong", "balancing"])
        print("pushed to", a.push)


if __name__ == "__main__":
    main()
