"""Albert's paddle-balancing sessions as a LeRobot dataset (one episode per run_policy run, 50 Hz).

Per frame:
  observation.state    (31)  25 joint positions (firmware order, rad) + IMU gravity (3, body frame) + gyro (3, rad/s)
  observation.ball     (2)   the Kalman predictor's ball position on the blade (u, w, m); 0 while waiting
  action               (5)   the policy's action (right-arm joint increments / 0.03 rad, in [-1, 1])
  action.arm_target    (5)   the right-arm joint targets actually sent (firmware, rad)
  balancing            (1)   1 while the policy balances the ball, 0 while waiting for it
  observation.images.head    the head camera (640x360), the latest frame received at each step: what the policy could see.
                             Only with --video, from runs that recorded <log>_camera.mp4 (run_policy does, since 2026-11-20)
Joint states and commands come from the menlo-sdk recording (<log>.jsonl), the ball and the actions from the run log
(<log>.csv); the two clocks are aligned on the right-arm angles both contain.

    uv run --with "lerobot[dataset]" python -m tools.export_lerobot --push cetiennec/asimov-pingpong-albert
    uv run --with "lerobot[dataset]" python -m tools.export_lerobot --video logs     # runs with camera -> ...-camera dataset

The camera frames show the room and the people in it. With --video, faces are found in every camera frame and blurred before
they go in the dataset (tools/face_blur.py; --no-blur to keep them), and a report is written next to the dataset, never in it:
<out>_privacy/faces.json (counts per episode) and <out>_privacy/<episode>_faces.jpg (each face found, before and after the blur,
and each doubtful detection left sharp), to look at before pushing. A --video dataset is pushed private unless --public.
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


class CameraFrames:
    """The run's camera recording, read forward: at(t) -> the latest frame received at or before t (None before the first)."""

    def __init__(self, mp4, blur_at=None):
        """blur_at: boxes to blur per camera frame (tools.face_blur.face_plan), applied once per frame as it is read."""
        import imageio_ffmpeg
        self.mp4, self.blur_at = mp4, blur_at
        self.t = np.genfromtxt(mp4[:-4] + ".csv", delimiter=",", names=True)["received_at"]
        self.reader = imageio_ffmpeg.read_frames(mp4)
        meta = next(self.reader)
        self.w, self.h = meta["size"]
        self.k, self.img = -1, None

    def at(self, t):
        want = int(np.searchsorted(self.t, t, side="right")) - 1
        while self.k < want:
            try:
                buf = next(self.reader)
            except StopIteration:
                break
            self.k += 1
            self.img = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            if self.blur_at is not None and self.k < len(self.blur_at) and self.blur_at[self.k]:
                from tools.face_blur import blur
                self.img = blur(self.img, self.blur_at[self.k])
        return self.img if want >= 0 else None


def episodes(roots, video=False):
    import sys
    sys.path.insert(0, os.getcwd())
    from albert.common import mjcf_to_fw
    seen = set()  # (name, size): the same run copied to several folders is exported once; two runs both named run.csv are not
    for csv in sorted(sum((glob.glob(os.path.join(r, "*.csv")) for r in roots), [])):
        key = (os.path.basename(csv), os.path.getsize(csv))
        if csv.endswith(("_frames.csv", "_camera.csv")) or key in seen:
            continue
        rec = csv[:-4] + ".jsonl"
        cam = csv[:-4] + "_camera.mp4"
        if not os.path.exists(rec) or (video and not os.path.exists(cam)):
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
        seen.add(key)
        e = dict(state=state, ball=ball, action=act, target=target, balancing=bal)
        if video:  # the camera's clock is the run's state clock (received_at): no alignment needed
            e["camera"], e["t_state"] = CameraFrames(cam), r["t_state"].astype(float)
        yield os.path.basename(csv)[:-4], e, rms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["logs", "logs/onboard"] + sorted(glob.glob("data/20*")),
                   help="folders of run logs (default: logs/, logs/onboard/, data/<day>/)")
    p.add_argument("--out", default=None, help="default: data/lerobot/asimov-pingpong-albert[-camera]")
    p.add_argument("--video", action="store_true", help="with the head camera: only runs that recorded <log>_camera.mp4")
    p.add_argument("--push", default=None, help="Hugging Face dataset repo id to push to")
    p.add_argument("--public", action="store_true", help="--video --push: make it public (the frames show people)")
    p.add_argument("--no-blur", action="store_true", help="--video: keep the faces (the dataset then stays local or private)")
    p.add_argument("--face-score", type=float, default=None, help="--video: a face track needs one detection this sure "
                   "(default: tools/face_blur.STRONG); lower it when the report shows faces left sharp")
    a = p.parse_args()
    if a.no_blur and a.public:
        raise SystemExit("--public with --no-blur: refused, the faces would be published")
    a.out = a.out or "data/lerobot/asimov-pingpong-albert" + ("-camera" if a.video else "")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {
        "observation.state": {"dtype": "float32", "shape": (31,),
                              "names": list(FW_JOINTS) + ["gravity_x", "gravity_y", "gravity_z", "gyro_x", "gyro_y", "gyro_z"]},
        "observation.ball": {"dtype": "float32", "shape": (2,), "names": ["ball_u", "ball_w"]},
        "action": {"dtype": "float32", "shape": (5,), "names": [FW_JOINTS[i] for i in ARM]},
        "action.arm_target": {"dtype": "float32", "shape": (5,), "names": [FW_JOINTS[i] for i in ARM]},
        "balancing": {"dtype": "float32", "shape": (1,), "names": ["balancing"]},
    }
    eps = list(episodes(a.data, video=a.video))
    if not eps:
        raise SystemExit("no run with a robot recording" + (" and a camera recording" if a.video else "") + f" in {a.data}")
    if a.video and not a.no_blur:
        faces_check(eps, a.out, a.face_score)
    if a.video:
        h, w = eps[0][1]["camera"].h, eps[0][1]["camera"].w
        features["observation.images.head"] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    ds = LeRobotDataset.create(repo_id=a.push or "local/asimov-pingpong-albert", fps=50, features=features, root=a.out,
                               robot_type="asimov_1", use_videos=a.video, streaming_encoding=a.video)
    n_ep = 0
    for name, e, rms in eps:
        n = len(e["action"])
        if a.video:
            cam = e["camera"]
            if (cam.h, cam.w) != (h, w):
                print(f"skipped {name}: camera {cam.w}x{cam.h}, the dataset's is {w}x{h}")
                continue
            first = None
        for k in range(n):
            fr = {"observation.state": e["state"][k].astype(np.float32), "observation.ball": e["ball"][k].astype(np.float32),
                  "action": e["action"][k].astype(np.float32), "action.arm_target": e["target"][k].astype(np.float32),
                  "balancing": e["balancing"][k:k + 1].astype(np.float32), "task": TASK}
            if a.video:
                img = cam.at(e["t_state"][k])
                if img is None:  # before the first frame arrived: the first frame (steps before the camera started)
                    first = first if first is not None else cam.at(cam.t[0])
                    img = first
                fr["observation.images.head"] = img
            ds.add_frame(fr)
        ds.save_episode()
        n_ep += 1
        print(f"episode {n_ep - 1}: {name}, {n} frames ({n / 50:.0f} s), {e['balancing'].sum() / 50:.1f} s balancing, "
              f"clock alignment residual {1000 * rms:.1f} mrad")
    ds.finalize()
    print(f"{n_ep} episodes -> {a.out}")
    if a.push:
        ds.push_to_hub(private=a.video and not a.public, tags=["asimov-1", "humanoid", "ping-pong", "balancing"])
        print("pushed to", a.push)


def faces_check(eps, out, strong=None):
    """Find the faces in each episode's camera video (before anything is written or pushed), set each episode's camera to blur
    them, and write the report next to the dataset: <out>_privacy/faces.json and a review sheet per episode."""
    from tools import face_blur as fb
    rep_dir = out.rstrip("/") + "_privacy"
    if os.path.exists(rep_dir):
        shutil.rmtree(rep_dir)
    os.makedirs(rep_dir)
    finder, rows = fb.FaceFinder(), []
    for name, e, _ in eps:
        cam = e["camera"]
        per = [finder.faces(img) for img in fb._frames(cam.mp4)]
        blur_at, faces, weak = fb.face_plan(per, cam.w, strong=strong or fb.STRONG, likely=min(fb.LIKELY, strong or 1))
        cam.blur_at = blur_at
        rows.append(fb.report_line(name, len(per), blur_at, faces, weak))
        sheet = fb.review(cam.mp4, name, blur_at, faces, weak, rep_dir)
        r = rows[-1]
        print(f"faces {name}: {r['frames_blurred']} of {r['camera_frames']} camera frames blurred ({r['face_tracks']} faces), "
              f"{r['weak_tracks_not_blurred']} doubtful detections left sharp (max score {r['max_weak_score']})"
              + (f" -> {sheet}" if sheet else ""), flush=True)
    with open(os.path.join(rep_dir, "faces.json"), "w") as fh:
        json.dump(dict(model=os.path.basename(fb.MODEL_URL), strong=strong or fb.STRONG, likely=min(fb.LIKELY, strong or 1),
                       min_width=fb.MIN_WIDTH, episodes=rows), fh, indent=1)
    print(f"face report -> {rep_dir}/ (local only: the sheets show the faces unblurred)")


def export_after_run(log_dir=None):
    """Called at the end of a run on the Mac (run_policy) and by onboard.sh: rebuild the local camera dataset with every run
    recorded so far. Needs uv (lerobot is installed on the fly, not in the robot's environment); never raises."""
    import subprocess
    if not shutil.which("uv"):
        print("LeRobot export skipped (no uv): uv run --with 'lerobot[dataset]' python -m tools.export_lerobot --video")
        return
    print("LeRobot dataset with the camera (data/lerobot/asimov-pingpong-albert-camera) ...", flush=True)
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data = ["logs", "logs/onboard"] + sorted(glob.glob(os.path.join(repo, "data", "20*")))
        if log_dir and os.path.abspath(log_dir) not in [os.path.join(repo, d) for d in data]:
            data.append(os.path.abspath(log_dir))  # a run logged somewhere else
        r = subprocess.run(["uv", "run", "--quiet", "--with", "lerobot[dataset]", "python", "-m", "tools.export_lerobot", "--video",
                            "--data", *data], cwd=repo, capture_output=True, text=True, timeout=1800)
        lines = [ln for ln in (r.stdout + r.stderr).splitlines()
                 if ln.startswith(("episode", "skipped", "no run", "face report")) or "episodes ->" in ln]
        print("\n".join(lines[-12:]) or (r.stdout + r.stderr)[-800:])
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"LeRobot export failed ({exc!r}); the run's files are saved")


if __name__ == "__main__":
    main()
