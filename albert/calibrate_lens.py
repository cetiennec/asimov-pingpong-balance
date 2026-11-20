"""The head camera's lens, once (it does not change when Albert falls): focal lengths, optical centre and distortion, from
frames of a checkerboard (data/checkerboard_9x6.png: 10 x 7 squares, 9 x 6 inner corners) held at many places and angles.

    python -m albert.record_frames --seconds 90 --every 0.5 --name checkerboard     # the board: whole image, corners, tilted
    uv run --with opencv-python-headless python -m albert.calibrate_lens data/<day>/checkerboard_<time>

Writes albert/camera_intrinsics.json (the previous one goes to calibration_history/), then refits the camera's pose on the head
from the latest saved 9-spot calibration (calibration_spots_*.json) with this lens: no robot needed. The square size does not
matter for the lens; it is only printed with the board's distances.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LENS_FILE = ROOT / "camera_intrinsics.json"
PATTERN = (9, 6)


def find_corners(paths, pattern=PATTERN):
    import cv2
    found = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ok, c = cv2.findChessboardCornersSB(img, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok:
            ok, c = cv2.findChessboardCorners(img, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                c = cv2.cornerSubPix(img, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
        if ok:
            found.append((p, c.reshape(-1, 2).astype(np.float32), img.shape[::-1]))
    return found


def coverage(corners, size, grid=(6, 4)):
    """Fraction of a grid x grid over the image with at least one corner in it, and the grid as text."""
    w, h = size
    hit = np.zeros(grid[::-1], bool)
    for c in corners:
        gx = np.clip((c[:, 0] / w * grid[0]).astype(int), 0, grid[0] - 1)
        gy = np.clip((c[:, 1] / h * grid[1]).astype(int), 0, grid[1] - 1)
        hit[gy, gx] = True
    return float(hit.mean()), "\n".join("    " + " ".join("#" if v else "." for v in row) for row in hit)


def calibrate(found, square_mm=25.0, pattern=PATTERN):
    """cv2.calibrateCamera with the views far off the others dropped (blur, a wrong corner order). Returns the lens dict."""
    import cv2
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm / 1000.0
    size = found[0][2]
    views = [f for f in found if f[2] == size]
    flags = cv2.CALIB_FIX_K3 if len(views) < 15 else 0  # k3 needs many views to be determined
    while True:
        rms, K, dist, rv, tv = cv2.calibrateCamera([obj] * len(views), [v[1] for v in views], size, None, None, flags=flags)
        per = []
        for (p, c, _), r, t in zip(views, rv, tv):
            q, _ = cv2.projectPoints(obj, r, t, K, dist)
            per.append(float(np.sqrt(np.mean(np.sum((q.reshape(-1, 2) - c) ** 2, 1)))))
        per = np.array(per)
        k = int(np.argmax(per))
        if len(views) > 8 and per[k] > max(1.0, 3 * np.median(per)):
            print(f"  {os.path.basename(views[k][0])}: {per[k]:.2f} px (median {np.median(per):.2f}): dropped")
            del views[k]
            continue
        dist_m = [float(np.linalg.norm(t)) for t in tv]
        return dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                    dist=[float(x) for x in dist.ravel()[:5]], width=int(size[0]), height=int(size[1]), rms_px=float(rms),
                    n_views=len(views), board_distance_m=[min(dist_m), max(dist_m)],
                    date=time.strftime("%Y-%m-%d %H:%M")), views


def distortion_at_edges(lens):
    """How far (px) distortion moves the image's corner and edge middles: what ignoring it would cost there."""
    from albert.ball_camera import distort
    out = {}
    for name, (u, v) in {"corner": (lens["width"], lens["height"]), "right edge": (lens["width"], lens["cy"]),
                         "bottom edge": (lens["cx"], lens["height"])}.items():
        x, y = (u - lens["cx"]) / lens["fx"], (v - lens["cy"]) / lens["fy"]
        # the undistorted point that lands there
        from albert.ball_camera import undistort
        xu, yu = undistort(x, y, lens["dist"])
        xd, yd = distort(xu, yu, lens["dist"])
        out[name] = float(np.hypot((xu - xd) * lens["fx"], (yu - yd) * lens["fy"]))
    return out


def refit_pose(lens):
    """The camera's pose on the head again, with this lens, from the latest saved 9-spot calibration (no robot)."""
    from albert.calibrate_camera import refit
    spots = sorted(glob.glob(str(ROOT / "calibration_spots_2*.json")) + glob.glob(str(ROOT / "calibration_history" / "calibration_spots_2*.json")),
                   key=os.path.getmtime)
    if not spots:
        print("no saved 9-spot calibration: run python -m albert.calibrate_camera --supported (it will use this lens)")
        return
    print(f"camera pose refitted with this lens from {os.path.relpath(spots[-1])}:")
    refit(spots[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", nargs="+", help="folders (or images) of checkerboard frames")
    p.add_argument("--square-mm", type=float, default=25.0, help="one square's side (only for the printed board distances)")
    p.add_argument("--min-views", type=int, default=12)
    p.add_argument("--min-coverage", type=float, default=0.5, help="fraction of a 6 x 4 grid over the image the corners must reach")
    p.add_argument("--no-refit", action="store_true", help="do not refit the camera pose from the saved spots")
    p.add_argument("--out", default=str(LENS_FILE), help="where to write the lens (default: Albert's; other paths: no refit)")
    a = p.parse_args()
    out = Path(a.out)
    paths = sorted(sum((glob.glob(os.path.join(f, "*.png")) + glob.glob(os.path.join(f, "*.jpg")) if os.path.isdir(f) else [f]
                        for f in a.frames), []))
    found = find_corners(paths)
    print(f"{len(paths)} frames, board found in {len(found)}")
    if len(found) < a.min_views:
        raise SystemExit(f"need at least {a.min_views} frames with the whole board in view (9 x 6 inner corners): hold it "
                         "20-30 cm from the head, flat to the camera and tilted, everywhere in the image")
    lens, views = calibrate(found, a.square_mm)
    cov, grid = coverage([v[1] for v in views], (lens["width"], lens["height"]))
    lens["coverage"] = cov
    edges = distortion_at_edges(lens)
    print(f"lens: fx {lens['fx']:.1f} fy {lens['fy']:.1f} px, centre ({lens['cx']:.1f}, {lens['cy']:.1f}) "
          f"(image middle {lens['width'] / 2:.0f}, {lens['height'] / 2:.0f}), distortion {np.round(lens['dist'], 4).tolist()}")
    print(f"  {lens['n_views']} views, reprojection {lens['rms_px']:.2f} px rms; distortion moves the image corner by "
          f"{edges['corner']:.0f} px, the edges by {edges['right edge']:.0f} / {edges['bottom edge']:.0f} px")
    print(f"  corners reached {100 * cov:.0f} % of the image (# = seen):\n{grid}")
    if lens["rms_px"] > 1.5 or cov < a.min_coverage:
        bad = out.with_name(out.stem + "_rejected.json")
        bad.write_text(json.dumps(lens, indent=1))
        raise SystemExit(f"not used ({'reprojection > 1.5 px' if lens['rms_px'] > 1.5 else 'board not seen in enough of the image'})"
                         f": saved as {bad.name}. Record again, moving the board over the whole image")
    albert = out.resolve() == LENS_FILE
    if albert and LENS_FILE.exists():
        hist = ROOT / "calibration_history"; hist.mkdir(exist_ok=True)
        shutil.copy(LENS_FILE, hist / f"camera_intrinsics_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.write_text(json.dumps(lens, indent=1))
    print(f"-> {out}")
    if albert and not a.no_refit:
        refit_pose(lens)


if __name__ == "__main__":
    main()
