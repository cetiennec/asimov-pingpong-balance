"""Ball position on the blade from Albert's head camera.

detect()   the ball's colour (white by default) inside the red blade region of the image -> pixel centre and radius
CameraModel  pinhole camera fixed to neck_pitch_link: pose (6) + focal length, fitted by albert/calibrate_camera.py.
           A pixel becomes a ray, intersected with the plane of ball centres just above the blade, whose pose comes from forward
           kinematics of the arm. Result: ball (u, w) on the blade, in metres.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from asimov_pp.model import BALL_R, BLADE_HT


# Ball colours. White (default): low saturation, bright. Orange: hue range, saturated. Tune on calibration_mask.png.
BALL_WHITE = dict(color="white", max_sat=0.30, min_val=0.50)
BALL_ORANGE = dict(color="orange", hue=(10.0, 40.0), min_sat=0.72, min_val=0.45)
RED_RUBBER = dict(hue_max=15.0, hue_min=330.0, min_sat=0.45, min_val=0.15)


def _hsv(rgb, step):
    img = np.asarray(rgb)[::step, ::step, :3].astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx, mn = img.max(-1), img.min(-1)
    c = np.maximum(mx - mn, 1e-6)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    h = np.where(mx == r, (60 * (g - b) / c) % 360, np.where(mx == g, 60 * (b - r) / c + 120, 60 * (r - g) / c + 240))
    return h, sat, mx


def masks(rgb, ball=BALL_WHITE, step=2):
    """(blade region, ball-colour mask) at 1/step resolution. The blade region is the largest red-rubber blob, closed, with
    its holes filled (the ball itself makes one) and grown by about a ball radius (the ball's top can stick out of it)."""
    h, sat, val = _hsv(rgb, step)
    red = ((h <= RED_RUBBER["hue_max"]) | (h >= RED_RUBBER["hue_min"])) & (sat >= RED_RUBBER["min_sat"]) & (val >= RED_RUBBER["min_val"])
    if ball["color"] == "white":
        col = (sat <= ball["max_sat"]) & (val >= ball["min_val"])
    else:
        col = (h >= ball["hue"][0]) & (h <= ball["hue"][1]) & (sat >= ball["min_sat"]) & (val >= ball["min_val"])
    masks.red = red  # kept for detect(): is a candidate surrounded by red rubber?
    lab, n = ndimage.label(red)
    if n == 0:
        return np.zeros_like(red), col, np.zeros_like(red)
    k = int(np.argmax(ndimage.sum(red, lab, range(1, n + 1)))) + 1
    blade = ndimage.binary_closing(lab == k, iterations=3)
    blade = ndimage.binary_fill_holes(blade)
    ys, xs = np.nonzero(blade)
    if len(xs) == 0:  # no red blade in view
        return np.zeros_like(red), col, np.zeros_like(red)
    grow = max(2, int(0.15 * (xs.max() - xs.min() + 1)))  # ~ a ball radius (ball 40 mm, blade ~150 mm)
    core = ndimage.binary_dilation(blade, iterations=2)  # the blade itself (+2 px): where the ball touches it
    return ndimage.binary_dilation(blade, iterations=grow), col, core


def detect(rgb, ball=BALL_WHITE, min_area=12, step=2, min_red_ring=0.35):
    """The ball on the paddle: the largest round blob of the ball's colour inside the blade region of the image (keep anything
    else white, a screen, a sheet of paper, out of the area right around the paddle).
    Returns (x, y, radius) in full-resolution pixels, or None. Searching only on the red blade keeps Albert's white shell,
    the hand and the background out; small specular highlights on the rubber are rejected by size and shape."""
    if np.asarray(rgb).size == 0:
        return None
    blade, col, core = masks(rgb, ball, step)
    cand = ndimage.binary_opening(col & blade)
    lab, n = ndimage.label(cand)
    best = None
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        area = float((lab[sl] == k).sum())
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if area * step**2 < min_area or not (0.5 < hh / ww < 2.0) or area / (hh * ww) < 0.5:
            continue  # too small, elongated or not filled like a disc
        yy, xx = np.nonzero(lab[sl] == k)
        x, y = (xx.mean() + sl[1].start) * step + step / 2, (yy.mean() + sl[0].start) * step + step / 2
        # a ball on the blade is surrounded by red rubber; a screen or a hand behind the blade's edge is not
        cy, cx, rr = yy.mean() + sl[0].start, xx.mean() + sl[1].start, max(1.5, np.sqrt(area / np.pi))
        Y, X = np.ogrid[:masks.red.shape[0], :masks.red.shape[1]]
        ring = ((Y - cy) ** 2 + (X - cx) ** 2 >= (1.4 * rr) ** 2) & ((Y - cy) ** 2 + (X - cx) ** 2 <= (2.2 * rr) ** 2)
        if ring.sum() and masks.red[ring].mean() < min_red_ring:
            continue
        if best is None or area > best[3]:
            best = (float(x), float(y), float(np.sqrt(area / np.pi) * step), area)
    return None if best is None else best[:3]


def debug_image(rgb, ball=BALL_WHITE, step=2):
    """Frame | masks (blue: blade region, white: ball colour inside it, red cross: detection), for tuning."""
    blade, col, _ = masks(rgb, ball, step)
    img = np.asarray(rgb)[::step, ::step, :3].copy()
    m = np.zeros_like(img); m[blade] = (40, 60, 140); m[col & blade] = (255, 255, 255)
    det = detect(rgb, ball, step=step)
    if det is not None:
        x, y = int(det[0] / step), int(det[1] / step)
        m[max(0, y - 6):y + 7, x] = m[y, max(0, x - 6):x + 7] = (255, 0, 0)
    return np.concatenate([img, m], 1)


class CameraModel:
    """Pinhole camera rigidly attached to neck_pitch_link. Camera axes (OpenCV): x right, y down, z forward."""

    def __init__(self, width, height, params=None):
        self.w, self.h = width, height
        if params is None:  # initial guess: front of the head, looking along the head's +x
            # camera axes in head coordinates: x_cam = -y_head (right), y_cam = -z_head (down), z_cam = +x_head (forward)
            r0 = Rotation.from_matrix(np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float))
            params = np.r_[r0.as_rotvec(), 0.10, 0.0, 0.045, 0.8 * width]
        self.p = np.asarray(params, float)

    def _cam(self, p=None):
        p = self.p if p is None else p
        return Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6], p[6]  # R head<-cam, t in head, focal (px)

    def project(self, P_world, head_pos, head_R, p=None):
        Rc, t, f = self._cam(p)
        Pc = (Rc.T @ (head_R.T @ (np.asarray(P_world) - head_pos) - t))
        return np.array([f * Pc[0] / Pc[2] + self.w / 2, f * Pc[1] / Pc[2] + self.h / 2]), Pc[2]

    def ray(self, px, head_pos, head_R):
        Rc, t, f = self._cam()
        d_cam = np.array([(px[0] - self.w / 2) / f, (px[1] - self.h / 2) / f, 1.0])
        return head_pos + head_R @ t, head_R @ Rc @ (d_cam / np.linalg.norm(d_cam))

    def ball_on_blade(self, px, head_pos, head_R, blade_c, blade_R):
        """(u, w) in metres of the ball whose image centre is px, or None if the ray misses the plane."""
        o, d = self.ray(px, head_pos, head_R)
        n = blade_R[:, 2]
        p0 = blade_c + n * (BLADE_HT + BALL_R)
        den = d @ n
        if abs(den) < 1e-6:
            return None
        s = ((p0 - o) @ n) / den
        if s <= 0:
            return None
        rel = blade_R.T @ (o + s * d - blade_c)
        return rel[:2]

    def fit(self, samples):
        """samples: list of (pixel (2,), ball centre world (3,), head_pos, head_R). Fits pose + focal length."""
        def res(p):
            out = []
            for px, P, hp, hR in samples:
                q, z = self.project(P, hp, hR, p)
                out.append(q - px if z > 0 else np.array([1e3, 1e3]))
            return np.concatenate(out)

        # the camera sits on the face: position within 6 cm of the guess, focal length between 0.25 and 2.5 x the width
        t0 = self.p[3:6].copy()
        lo = np.r_[-np.inf * np.ones(3), t0 - 0.06, 0.25 * self.w]
        hi = np.r_[np.inf * np.ones(3), t0 + 0.06, 2.5 * self.w]
        best = None
        for df in (0.4, 0.7, 1.1):  # a few focal-length starts
            p0 = self.p.copy(); p0[6] = df * self.w
            r = least_squares(res, p0, loss="soft_l1", f_scale=3.0, bounds=(lo, hi))
            if best is None or r.cost < best.cost:
                best = r
        self.p = best.x
        err = np.linalg.norm(res(self.p).reshape(-1, 2), axis=1)
        return float(np.mean(err)), float(np.max(err))

    def to_json(self):
        return dict(width=self.w, height=self.h, params=self.p.tolist())

    @classmethod
    def from_json(cls, j):
        return cls(j["width"], j["height"], j["params"])


def blade_roi(cam, head_pos, head_R, blade_c, blade_R, radius, margin_px=24, step=4):
    """Bounding box (x0, y0, x1, y1), full-resolution pixels aligned to step, of the blade disc grown to `radius` (m) as the
    camera model sees it for these arm / head poses, or None if it is not in front of the camera."""
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts = blade_c[None, :] + radius * (np.cos(ang)[:, None] * blade_R[:, 0] + np.sin(ang)[:, None] * blade_R[:, 1])
    proj = [cam.project(p, head_pos, head_R) for p in pts]
    if any(depth < 0.05 for _, depth in proj):
        return None
    px = np.array([uv for uv, _ in proj])
    if not np.all(np.isfinite(px)):
        return None
    x0, y0 = np.floor((px.min(0) - margin_px) / step).astype(int) * step
    x1, y1 = np.ceil((px.max(0) + margin_px) / step).astype(int) * step
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(cam.w, x1), min(cam.h, y1)
    if x1 - x0 < 8 * step or y1 - y0 < 8 * step:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def detect_roi(rgb, ball, roi, **kw):
    """detect() on the region of interest only (faster; nothing outside the paddle's neighbourhood can be taken for the ball).
    roi None: the whole frame."""
    rgb = np.asarray(rgb)
    if roi is not None:  # clipped to the image; too small (e.g. another resolution than the calibration's): whole frame
        x0, y0, x1, y1 = max(0, roi[0]), max(0, roi[1]), min(rgb.shape[1], roi[2]), min(rgb.shape[0], roi[3])
        step = kw.get("step", 2)
        if x1 - x0 < 8 * step or y1 - y0 < 8 * step:
            roi = None
    if roi is None:
        return detect(rgb, ball, **kw)
    det = detect(rgb[y0:y1, x0:x1], ball, **kw)
    return None if det is None else (det[0] + x0, det[1] + y0, det[2])


def ball_on_blade_cov(cam, px, head_pos, head_R, blade_c, blade_R, sigma_px=2.0):
    """The ball on the blade (u, w) and its 2x2 covariance (m^2) for a detection noise of sigma_px pixels, from the camera
    geometry: the ray meets the blade at a slant, so the error is larger along the viewing direction. (None, None) if no hit."""
    px = np.asarray(px, float)
    uw = cam.ball_on_blade(px, head_pos, head_R, blade_c, blade_R)
    if uw is None:
        return None, None
    J = []
    for d in (np.array([2.0, 0.0]), np.array([0.0, 2.0])):
        a, b = cam.ball_on_blade(px + d, head_pos, head_R, blade_c, blade_R), cam.ball_on_blade(px - d, head_pos, head_R, blade_c, blade_R)
        if a is None or b is None:
            return np.asarray(uw), None
        J.append((np.asarray(a) - np.asarray(b)) / 4.0)
    J = np.array(J).T  # d(u, w) / d(pixel x, y)
    return np.asarray(uw), sigma_px ** 2 * J @ J.T
