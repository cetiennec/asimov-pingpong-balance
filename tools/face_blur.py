"""Faces in Albert's head-camera frames: find them (YuNet, OpenCV's face detector) and blur them before a dataset is shared.

The camera sees the paddle, the hands that place the ball and, often, the person doing it (face from above, in profile, half out
of the image). Only faces are blurred: hands and arms stay (they hold the ball).

Per frame, YuNet gives boxes with a score. Scores alone do not separate faces from hands: real faces seen from above or half
out of the image score 0.5-0.85, and so do a hand holding the ball (G1: 0.50-0.72 for 2 s), a robot motor or a pair of
trousers. Haar cascades were worse (the neon smiley face on the wall behind the paddle, most hands). So detections are linked
over time into tracks (same place, gaps up to LINK_GAP frames) and a track is a face when one of its detections reaches
STRONG: then every frame of the track is blurred, the gaps filled, and KEEP frames more on each side (a face the detector
misses for a few frames stays hidden); a track reaching LIKELY counts too unless its boxes are tiny. Weaker tracks are not
blurred but are shown on the review sheet. Cost on the 2026-11-22 runs: a few hand tracks are blurred too (2-20 frames each;
G1_v3's hand holding the ball for 3 s), the ball's position is in observation.ball anyway.

    python -m tools.face_blur logs/onboard/G3_v3_camera.mp4 ...        # report + review sheet per video (local, unblurred)
    python -m tools.face_blur --left-half data/2026-11-20/M3_v3_gantry_detect.mp4

The model (230 kB ONNX, OpenCV model zoo, MIT) is downloaded once into .cache/models/ (gitignored) and checked by its hash.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request

import numpy as np

MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "models")
MIN_SCORE = 0.45  # detections kept for tracking
STRONG = 0.75     # a track with a detection at least this sure is a face
LIKELY = 0.70     # ... or this sure and not tiny: heads seen from above score 0.69-0.73 (G3_v3_3 64-73, G3_v3 1207-1214)
MIN_WIDTH = 0.04  # LIKELY tracks: median box width / image width (parts of the robot found as faces: 15-17 px of 640)
REVIEW = 0.6      # a track below STRONG but reaching this is shown on the review sheet (not blurred)
LINK_GAP = 12     # frames (~0.5 s at 22 fps)
KEEP = 3          # frames blurred before and after a face track


def model_path():
    path = os.path.join(CACHE, os.path.basename(MODEL_URL))
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        print(f"downloading the face detector ({MODEL_URL}) ...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, path + ".part")
        os.replace(path + ".part", path)
    with open(path, "rb") as fh:
        if hashlib.sha256(fh.read()).hexdigest() != MODEL_SHA256:
            os.remove(path)
            raise SystemExit(f"face detector model: unexpected hash, deleted {path}; run again to download it")
    return path


class FaceFinder:
    """faces(rgb) -> [(x, y, w, h, score)] in the image's pixels. The image is upscaled first (a face across the room is a few
    pixels wide at 640x360; YuNet's smallest is ~10 px)."""

    def __init__(self, score=MIN_SCORE, upscale=2.0):
        import cv2
        self.cv2, self.upscale = cv2, upscale
        self.det = cv2.FaceDetectorYN.create(model_path(), "", (320, 320), score, 0.3, 5000)

    def faces(self, rgb):
        cv2 = self.cv2
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        if self.upscale != 1:
            bgr = cv2.resize(bgr, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_LINEAR)
        self.det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, found = self.det.detect(bgr)
        if found is None:
            return []
        return [tuple(float(v) for v in f[:4] / self.upscale) + (float(f[14]),) for f in found]


def tracks(per_frame, gap=LINK_GAP):
    """Detections per frame -> tracks [{"f": [frame], "b": [(x, y, w, h, score)]}]: a detection joins the track whose last box
    (at most gap frames before) is closest, if its centre is within the larger box's size."""
    out, live = [], []
    for i, boxes in enumerate(per_frame):
        live = [t for t in live if i - t["f"][-1] <= gap]
        taken = set()
        for b in sorted(boxes, key=lambda b: -b[4]):
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            best = None
            for k, t in enumerate(live):
                if k in taken or t["f"][-1] == i:
                    continue
                x, y, w, h, _ = t["b"][-1]
                d = np.hypot(cx - x - w / 2, cy - y - h / 2)
                if d < max(w, h, b[2], b[3]) and (best is None or d < best[0]):
                    best = (d, k)
            if best is None:
                t = {"f": [i], "b": [b]}
                out.append(t)
                live.append(t)
                taken.add(len(live) - 1)
            else:
                live[best[1]]["f"].append(i)
                live[best[1]]["b"].append(b)
                taken.add(best[1])
    for t in out:
        t["max"] = max(b[4] for b in t["b"])
    return out


def face_plan(per_frame, width, strong=STRONG, likely=LIKELY, keep=KEEP):
    """-> (boxes to blur per frame, face tracks, review tracks). width: the image's, in pixels. A face track's box is
    interpolated over its gaps and held for keep frames past each end."""
    n = len(per_frame)
    ts = tracks(per_frame)

    def is_face(t):
        return t["max"] >= strong or (t["max"] >= likely and np.median([b[2] for b in t["b"]]) >= MIN_WIDTH * width)
    faces = [t for t in ts if is_face(t)]
    review = [t for t in ts if REVIEW <= t["max"] and not is_face(t)]
    blur_at = [[] for _ in range(n)]
    for t in faces:
        f, b = np.array(t["f"]), np.array([bb[:4] for bb in t["b"]])
        for i in range(max(0, f[0] - keep), min(n, f[-1] + keep + 1)):
            blur_at[i].append(tuple(np.interp(i, f, b[:, j]) for j in range(4)))

    def inside(i, d):  # the detection's centre is in a blurred area (the edge of a face already blurred)
        cx, cy = d[0] + d[2] / 2, d[1] + d[3] / 2
        return any(x - 0.5 * w <= cx <= x + 1.5 * w and y - 0.7 * h <= cy <= y + 1.4 * h for x, y, w, h in blur_at[i])
    review = [t for t in review if not all(inside(i, d) for i, d in zip(t["f"], t["b"]))]
    return blur_at, faces, review


def blur(rgb, boxes, pad=(0.5, 0.7, 0.5, 0.4)):
    """A copy of the image with each box grown by pad (left, top, right, bottom, in box sizes: the top for the hair) and
    turned into ~4 coarse blocks across, then blurred: no feature of the face is left."""
    import cv2
    out = np.array(rgb, copy=True)
    H, W = out.shape[:2]
    for x, y, w, h, *_ in boxes:
        x0, y0 = int(max(0, x - pad[0] * w)), int(max(0, y - pad[1] * h))
        x1, y1 = int(min(W, x + w * (1 + pad[2]))), int(min(H, y + h * (1 + pad[3])))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        cell = max(2, int(min(w, h) / 4))
        roi = out[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(1, (x1 - x0) // cell), max(1, (y1 - y0) // cell)), interpolation=cv2.INTER_AREA)
        roi = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        k = max(3, (cell * 2) | 1)
        out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def _frames(mp4, crop=None):
    import imageio_ffmpeg
    rd = imageio_ffmpeg.read_frames(mp4)
    w, h = next(rd)["size"]
    for buf in rd:
        img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        yield img[:, :w // 2] if crop == "left" else img


def scan(mp4, finder=None, crop=None):
    """Every frame of the video -> the plan (see face_plan) and the raw detections."""
    finder = finder or FaceFinder()
    per, width = [], None
    for img in _frames(mp4, crop):
        per.append(finder.faces(img))
        width = img.shape[1]
    return (*face_plan(per, width), per)


def review(mp4, name, blur_at, faces, weak, out_dir, crop=None, tile=(256, 144)):
    """A sheet per video for a person to check, kept local: one row per face track (the frame where the face is surest:
    as recorded with the box in red, then blurred as in the dataset) and one tile per weak track (yellow, NOT blurred: a face
    there was missed; drawn on the frame as it is in the dataset). Returns its path, or None when there is nothing to show."""
    import cv2
    if not faces and not weak:
        return None
    pick = {}
    for kind, ts in (("face", faces), ("weak", weak)):
        for t in ts:
            j = int(np.argmax([b[4] for b in t["b"]]))
            pick.setdefault(t["f"][j], []).append((kind, t, t["b"][j]))
    tw, th = tile
    tiles = []
    for i, img in enumerate(_frames(mp4, crop)):
        for kind, t, b in pick.get(i, []):
            s = tw / img.shape[1]
            shown = img if kind == "face" else blur(img, blur_at[i])
            raw = cv2.cvtColor(cv2.resize(np.ascontiguousarray(shown), (tw, th)), cv2.COLOR_RGB2BGR)
            col = (0, 0, 255) if kind == "face" else (0, 220, 255)
            x, y, w, h, sc = b
            cv2.rectangle(raw, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), col, 2)
            label = f"{name} {t['f'][0]}-{t['f'][-1]} ({len(t['f'])} det) {t['max']:.2f}"
            pair = [raw]
            if kind == "face":
                pair.append(cv2.cvtColor(cv2.resize(blur(img, blur_at[i]), (tw, th)), cv2.COLOR_RGB2BGR))
            else:
                label = f"{name} {t['f'][0]}-{t['f'][-1]} {t['max']:.2f} left sharp"
            lab = np.full((16, tw * len(pair), 3), 255, np.uint8)
            cv2.putText(lab, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
            tiles.append((kind, np.vstack([np.hstack(pair), lab])))
    rows, cur, width = [], [], 4 * tw
    for kind, t in sorted(tiles, key=lambda x: x[0]):
        if sum(c.shape[1] for c in cur) + t.shape[1] > width:
            rows.append(cur)
            cur = []
        cur.append(t)
    rows.append(cur)
    sheet = np.vstack([np.hstack(r + [np.full((th + 16, width - sum(c.shape[1] for c in r), 3), 255, np.uint8)]) for r in rows])
    path = os.path.join(out_dir, f"{name}_faces.jpg")
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def report_line(name, n_frames, blur_at, faces, weak):
    nb = sum(1 for b in blur_at if b)
    return dict(episode=name, camera_frames=n_frames, frames_blurred=nb, face_tracks=len(faces), weak_tracks_not_blurred=len(weak),
                weak_frames=sum(len(t["f"]) for t in weak), max_weak_score=round(max([t["max"] for t in weak], default=0), 2))


def main():
    import argparse
    import json
    p = argparse.ArgumentParser(description="faces in camera videos: counts and a review sheet per video (local, unblurred)")
    p.add_argument("videos", nargs="+")
    p.add_argument("--out", default="data/lerobot/face_check")
    p.add_argument("--left-half", action="store_true", help="detector debug videos: the camera is the left half")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    finder, crop, rows = FaceFinder(), "left" if a.left_half else None, []
    for mp4 in a.videos:
        name = os.path.splitext(os.path.basename(mp4))[0]
        blur_at, faces, weak, per = scan(mp4, finder, crop)
        rows.append(report_line(name, len(per), blur_at, faces, weak))
        sheet = review(mp4, name, blur_at, faces, weak, a.out, crop)
        print(json.dumps(rows[-1]) + (f" -> {sheet}" if sheet else ""), flush=True)
    with open(os.path.join(a.out, "faces.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
