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
    masks.val = val  # and its brightness, to split a ball merged with a bright background
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


def _otsu(v, bins=32):
    """Threshold splitting v into two classes with the largest between-class variance."""
    hist, edges = np.histogram(v, bins=bins, range=(float(v.min()), float(v.max()) + 1e-6))
    w = np.cumsum(hist).astype(float); mu = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2)
    tot, mt = w[-1], mu[-1]
    w1 = tot - w
    ok = (w > 0) & (w1 > 0)
    between = np.where(ok, (mt * w / tot - mu) ** 2 / np.maximum(w * w1, 1e-9), 0)
    return edges[int(np.argmax(between)) + 1]


def _ball_like(pix, step, min_area, min_red_ring, r_px):
    """One blob (bool mask on the step grid) -> (x, y, radius, area) in full-resolution pixels if it looks like the ball on the
    blade: big enough, round and filled like a disc, of the expected size (r_px, if known) and surrounded by red rubber."""
    ys, xs = np.nonzero(pix)
    area = float(len(xs))
    if area * step**2 < min_area:
        return None
    hh, ww = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if not (0.5 < hh / ww < 2.0) or area / (hh * ww) < 0.5:
        return None  # elongated or not filled like a disc
    cy, cx, rr = ys.mean(), xs.mean(), max(1.5, np.sqrt(area / np.pi))
    if r_px is not None and not (0.5 <= rr * step / r_px <= 1.7):
        return None  # not the ball's size where the blade is (a highlight, a patch of background)
    # a ball on the blade is surrounded by red rubber; a screen or a hand behind the blade's edge is not
    red = masks.red
    R = int(np.ceil(2.2 * rr)) + 1
    y0, y1, x0, x1 = max(0, int(cy) - R), min(red.shape[0], int(cy) + R + 1), max(0, int(cx) - R), min(red.shape[1], int(cx) + R + 1)
    Y, X = np.ogrid[y0:y1, x0:x1]
    d2 = (Y - cy) ** 2 + (X - cx) ** 2
    ring = (d2 >= (1.4 * rr) ** 2) & (d2 <= (2.2 * rr) ** 2)
    if ring.sum() and red[y0:y1, x0:x1][ring].mean() < min_red_ring:
        return None
    return float(cx * step + step / 2), float(cy * step + step / 2), float(rr * step), area


def detect(rgb, ball=BALL_WHITE, min_area=12, step=2, min_red_ring=0.35, r_px=None, expect=None, gate_px=None):
    """The ball on the paddle: a round blob of the ball's colour inside the blade region of the image (keep anything else white,
    a screen, a sheet of paper, out of the area right around the paddle).
    Returns (x, y, radius) in full-resolution pixels, or None. Searching only on the red blade keeps Albert's white shell,
    the hand and the background out; small specular highlights on the rubber are rejected by size and shape.
    A blob that fails because the ball merged with a bright background behind the blade's edge (a lamp, a neon sign:
    2026-11-20 M3) is split by brightness and its parts tested again.
    Optional hints from the robot: r_px, the ball's expected radius in pixels where the blade is (camera model); expect, the
    pixel where the predictor expects the ball: then the ball-like blob nearest to it (within gate_px) wins, not the largest."""
    if np.asarray(rgb).size == 0:
        return None
    blade, col, core = masks(rgb, ball, step)
    cand = ndimage.binary_opening(col & blade)
    lab, n = ndimage.label(cand)
    found = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        pix = np.zeros_like(cand); pix[sl] = lab[sl] == k
        hit = _ball_like(pix, step, min_area, min_red_ring, r_px)
        if hit is not None:
            found.append(hit)
            continue
        v = masks.val[pix]
        if len(v) * step**2 < 4 * min_area or v.max() - v.min() < 0.2:
            continue
        thr = _otsu(v)
        if v[v >= thr].mean() - v[v < thr].mean() < 0.15:
            continue  # one brightness: nothing to split
        for part in (pix & (masks.val < thr), pix & (masks.val >= thr)):
            plab, pn = ndimage.label(ndimage.binary_opening(part))
            for j, psl in enumerate(ndimage.find_objects(plab), start=1):
                q = np.zeros_like(cand); q[psl] = plab[psl] == j
                hit = _ball_like(q, step, min_area, min_red_ring, r_px)
                if hit is not None:
                    found.append(hit)
    if not found:
        return None
    if expect is not None:
        gate = gate_px if gate_px is not None else (3.0 * r_px if r_px else 60.0)
        dist = [np.hypot(f[0] - expect[0], f[1] - expect[1]) for f in found]
        if min(dist) <= gate:
            return found[int(np.argmin(dist))][:3]
    return max(found, key=lambda f: f[3])[:3]


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


def lens_for(lens, width, height):
    """The lens calibration (albert/calibrate_lens.py: fx, fy, cx, cy in pixels of its own resolution, OpenCV distortion
    k1 k2 p1 p2 k3) scaled to a width x height image of the same sensor."""
    if lens is None:
        return None
    sx, sy = width / lens["width"], height / lens["height"]
    return dict(lens, fx=lens["fx"] * sx, fy=lens["fy"] * sy, cx=lens["cx"] * sx, cy=lens["cy"] * sy, width=width, height=height)


def distort(x, y, k):
    """Normalised image coordinates -> distorted ones (OpenCV model, k = k1 k2 p1 p2 k3)."""
    k1, k2, p1, p2, k3 = (list(k) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def undistort(xd, yd, k, iters=20):
    """Inverse of distort() by fixed-point iteration (as OpenCV's undistortPoints)."""
    k1, k2, p1, p2, k3 = (list(k) + [0.0] * 5)[:5]
    x, y = xd, yd
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        x = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / radial
        y = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / radial
    return x, y


class CameraModel:
    """Camera rigidly attached to neck_pitch_link. Camera axes (OpenCV): x right, y down, z forward.
    params: pose in the head (rotation vector, translation) and a focal length. Without a lens calibration it is a pinhole
    with its centre in the middle of the image and that focal length (fitted with the pose by calibrate_camera.py). With one
    (`lens`, from calibrate_lens.py, stored in the camera calibration it was used for): the lens's focal lengths, centre and
    distortion, and only the pose is fitted."""

    def __init__(self, width, height, params=None, lens=None):
        self.w, self.h = width, height
        self.lens = lens_for(lens, width, height)
        if params is None:  # initial guess: front of the head, looking along the head's +x
            # camera axes in head coordinates: x_cam = -y_head (right), y_cam = -z_head (down), z_cam = +x_head (forward)
            r0 = Rotation.from_matrix(np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float))
            params = np.r_[r0.as_rotvec(), 0.10, 0.0, 0.045, 0.8 * width]
        self.p = np.asarray(params, float)
        if self.lens is not None:
            self.p[6] = self.lens["fx"]

    def _cam(self, p=None):
        p = self.p if p is None else p
        return Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6], p[6]  # R head<-cam, t in head, focal (px)

    def _intr(self, f):
        """fx, fy, cx, cy, distortion."""
        if self.lens is None:
            return f, f, self.w / 2, self.h / 2, None
        L = self.lens
        return L["fx"], L["fy"], L["cx"], L["cy"], L["dist"]

    def project(self, P_world, head_pos, head_R, p=None):
        Rc, t, f = self._cam(p)
        Pc = (Rc.T @ (head_R.T @ (np.asarray(P_world) - head_pos) - t))
        fx, fy, cx, cy, k = self._intr(f)
        x, y = Pc[0] / Pc[2], Pc[1] / Pc[2]
        if k is not None:
            x, y = distort(x, y, k)
        return np.array([fx * x + cx, fy * y + cy]), Pc[2]

    def ray(self, px, head_pos, head_R):
        Rc, t, f = self._cam()
        fx, fy, cx, cy, k = self._intr(f)
        x, y = (px[0] - cx) / fx, (px[1] - cy) / fy
        if k is not None:
            x, y = undistort(x, y, k)
        d_cam = np.array([x, y, 1.0])
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
        """samples: list of (pixel (2,), ball centre world (3,), head_pos, head_R). Fits the pose and, without a lens
        calibration, the focal length."""
        def res(p):
            out = []
            for px, P, hp, hR in samples:
                q, z = self.project(P, hp, hR, p)
                out.append(q - px if z > 0 else np.array([1e3, 1e3]))
            return np.concatenate(out)

        if self.lens is not None:  # the lens is known: the 6 pose numbers only (better conditioned than with the focal length)
            t0 = self.p[3:6].copy()
            f = self.p[6]
            pose = lambda q: res(np.r_[q, f])
            lo, hi = np.r_[-np.inf * np.ones(3), t0 - 0.06], np.r_[np.inf * np.ones(3), t0 + 0.06]
            r = least_squares(pose, self.p[:6], loss="soft_l1", f_scale=3.0, bounds=(lo, hi))
            self.p = np.r_[r.x, f]
            err = np.linalg.norm(res(self.p).reshape(-1, 2), axis=1)
            return float(np.mean(err)), float(np.max(err))

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
        return dict(width=self.w, height=self.h, params=self.p.tolist(), **({"lens": self.lens} if self.lens else {}))

    @classmethod
    def from_json(cls, j):
        return cls(j["width"], j["height"], j["params"], lens=j.get("lens"))


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
    if kw.get("expect") is not None:
        kw = dict(kw, expect=(kw["expect"][0] - x0, kw["expect"][1] - y0))
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
