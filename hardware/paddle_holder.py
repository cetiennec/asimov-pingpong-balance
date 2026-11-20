"""Paddle holder for Asimov 1 without a gripper: a printed plug that goes into the hand socket of the right wrist, the stainless
cup ASV1_300_12B (right arm; ASV1_400_12B is the left-arm twin), is locked by the cup's 4 radial M4 screws, and holds a
table-tennis paddle straight ahead along the forearm axis in an open cradle with zip ties (any handle shape).
Writes hardware/paddle_holder.stl in mm; the sim uses the same file.

    uv run python -m hardware.paddle_holder

The cup (ASV1_300_12B.STEP and drawing Docs/Drawing_thread_parts/ASV1_400_12B.pdf, placed with the full-body assembly):
    bolted to the wrist-yaw actuator output by 6 x M3 from inside; open bore 30.48 mm, 12.0 mm deep (bottom face to rim);
    rim 14.64 mm in front of the wrist-yaw joint; 4 x M4 x 0.7 radial threads 5.65 mm below the rim, at 0 / 90 / 180 / 270 deg
    (along +-y and +-z of right_wrist_yaw_link); two 6 mm keys run inside the bore at +-z (the +-z screws go through them).
    Only the cup turns with the wrist yaw; the forearm cover stops at x = -2 mm.

Frame of this file (= sim body "paddle_holder"): right_wrist_yaw_link frame moved to the centre of the cup rim (x = 14.64 mm),
+x = forearm axis, pointing away from the elbow.

v5 (2026-11-20): the lips follow the handle's flared shape (HANDLE_PROFILE, 0.6 mm clearance), so the handle bears along its
length instead of only at the butt. No sharp inside corners (they crack): 45 deg chamfers at the lip roots, rounds at the
handle stop and the stop / flange step, round-ended zip-tie slots with >= 3 mm of wall outside, round-bottomed tie grooves. A flat face opposite the paddle: the flange is cut flush with the
underside of the cradle plate, so the part lies flat on it. v4 is in the git history.

Print: PETG / PLA+ / PA12, lying on the flat face (cradle plate down, paddle side up): no supports under the cradle, layers along
its length (strongest way for the paddle's bending). 6 walls or 100 % infill. The plug's 4 radial holes print horizontally.
Hardware: 4 x M4 x 10 socket head screws (through the cup's threads into the plug's holes), 3 zip ties 4.8 mm wide,
          a 1-2 mm rubber strip under the handle.
"""
from __future__ import annotations

import numpy as np
import trimesh

from asimov_pp.model import HANDLE_T, HANDLE_X0, HOLDER_LEN, HOLDER_STL, PADDLE_ROLL_DEG

# --- cup interface (mm, holder frame: origin at the rim centre, x forward) ---
BORE_D = 30.46
BORE_DEPTH = 12.0          # rim -> bottom face
PLUG_D = 30.1              # 0.18 mm radial clearance
PLUG_GAP = 0.6             # plug end stays clear of the bottom face (the 6 M3 heads are flush there)
RADIAL_X = -5.65           # radial M4 threads, from the drawing
RADIAL_ANGLES = (0, 90, 180, 270)  # deg about x, from +y towards +z
RADIAL_HOLE_D, RADIAL_HOLE_R0 = 4.4, 8.0  # blind holes for the screw tips, from r = 8 mm outwards
KEY_W, KEY_FLAT = 6.0, 13.0        # two keys inside the bore at +-z (90 / 270 deg): 6 mm wide, inner flat at |z| = 13 mm
SLOT_W, SLOT_FLOOR = 6.6, 12.65    # matching slots in the plug (also stop it from turning)
FLANGE_D, FLANGE_T = 38.0, 5.0     # sits on the rim (cup outer diameter 37.6)

# --- cradle (mm, paddle frame: x along the handle, z = blade normal) ---
CR_X0 = HANDLE_X0 * 1e3    # handle butt stop
CR_X1 = HOLDER_LEN * 1e3
CR_W, CR_T = 48.0, 5.0     # base plate width (y) and thickness (>= 3 mm of wall outside the zip-tie slots)
LIP_W, LIP_H = 2.5, 6.0    # centring lips: wall thickness and height
# The handle's width along its length, from the butt (mm): a flared handle (JOOLA Team School), estimated from a photo of it
# in the v4 cradle. Measure yours with calipers and put the numbers here; the lips follow it with HANDLE_CLEAR on each side.
HANDLE_PROFILE = ((0.0, 30.0), (10.0, 28.2), (20.0, 26.4), (30.0, 25.0), (40.0, 24.0), (50.0, 23.4), (60.0, 23.2))
HANDLE_CLEAR = 0.6
LIP_CHAMFER = 1.5          # 45 deg chamfers at both lip roots (no sharp inside corner)
TIE_W, TIE_L = 2.6, 5.6    # slots for 4.8 mm zip ties (slot length along x)
TIE_X = (CR_X0 + 9, (CR_X0 + CR_X1) / 2, CR_X1 - 9)
R_STOP, R_STEP = 1.5, 2.5  # inside-corner radii (mm)


def lip_y(x):
    """Inner face of the lips (|y|, mm) at cradle position x: half the handle width there plus the clearance."""
    d = np.clip(x - CR_X0, HANDLE_PROFILE[0][0], HANDLE_PROFILE[-1][0])
    return float(np.interp(d, [p[0] for p in HANDLE_PROFILE], [p[1] for p in HANDLE_PROFILE])) / 2 + HANDLE_CLEAR


def box(x0, x1, y0, y1, z0, z1):
    b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


def cyl(r, p0, p1, sections=96):
    return trimesh.creation.cylinder(radius=r, segment=[np.asarray(p0, float), np.asarray(p1, float)], sections=sections)


def union(parts):
    return trimesh.boolean.union(parts, engine="manifold")


def fillet(corner, u, v, axis, length, r):
    """Concave round (radius r) for the inside corner at `corner` between the two faces that run along unit vectors u and v,
    the edge running along `axis` for `length`: an r x r square prism minus a cylinder centred at corner + r u + r v."""
    corner, u, v, axis = (np.asarray(x, float) for x in (corner, u, v, axis))
    pts = np.array([corner + a * u * (r + 0.01) + b * v * (r + 0.01) + c * axis * length
                    for a in (-0.01, 1) for b in (-0.01, 1) for c in (0, 1)])
    sq = box(*[f(pts[:, k]) for k in range(3) for f in (np.min, np.max)])
    centre = corner + r * u + r * v
    hole = cyl(r, centre - axis * 0.5, centre + axis * (length + 0.5), 48)
    return trimesh.boolean.difference([sq, hole], engine="manifold")


def slot(x0, x1, y, w, z0, z1):
    """Zip-tie slot along x with round ends (a stadium), from z0 to z1."""
    r = w / 2
    return union([box(x0 + r, x1 - r, y - r, y + r, z0, z1),
                  cyl(r, [x0 + r, y, z0], [x0 + r, y, z1], 32), cyl(r, [x1 - r, y, z0], [x1 - r, y, z1], 32)])


def build_holder() -> trimesh.Trimesh:
    x_plug = -(BORE_DEPTH - PLUG_GAP)
    # 1) plug in the cup bore (with a 0.8 mm lead-in chamfer), 2) flange on the rim
    plug = union([cyl(PLUG_D / 2, [x_plug + 0.8, 0, 0], [0.5, 0, 0]), cyl(PLUG_D / 2 - 0.8, [x_plug, 0, 0], [x_plug + 1.0, 0, 0])])
    flange = cyl(FLANGE_D / 2, [0, 0, 0], [FLANGE_T, 0, 0])
    roll = trimesh.transformations.rotation_matrix(np.deg2rad(PADDLE_ROLL_DEG), [1, 0, 0])
    # 3) cradle, built in the paddle frame then rolled about x
    z_base = -HANDLE_T * 1e3 / 2              # the handle's lower face (black-rubber side) rests here
    z_bot = z_base - CR_T                     # underside of the plate: the flat face
    # flat face opposite the paddle: the flange is cut flush with the plate's underside (the plug, r 15 mm, is not touched)
    flat_cut = box(-1, FLANGE_T + 1, -30, 30, z_bot - 30, z_bot); flat_cut.apply_transform(roll)
    flange = trimesh.boolean.difference([flange, flat_cut], engine="manifold")
    plate = box(0, CR_X1, -CR_W / 2, CR_W / 2, z_bot, z_base)
    # lips following the handle's taper: convex segments (lip wall + 45 deg root chamfers on both sides), hulled pairwise
    xs = np.linspace(CR_X0, CR_X1, 14)
    lips = []
    for sy in (-1, 1):
        def section(x):
            yi = lip_y(x); yo = yi + LIP_W; c = LIP_CHAMFER
            ys_zs = [(yi - c, z_base - 0.1), (yo + c, z_base - 0.1), (yo, z_base + c), (yo, z_base + LIP_H),
                     (yi, z_base + LIP_H), (yi, z_base + c)]
            return [[x, sy * y, z] for y, z in ys_zs]
        for x0, x1 in zip(xs[:-1], xs[1:]):
            lips.append(trimesh.convex.convex_hull(np.array(section(x0) + section(x1))))
    y_stop = lip_y(CR_X0) + LIP_W
    stop = box(0, CR_X0, -y_stop, y_stop, z_bot, z_base + LIP_H)
    rounds = []
    for sy in (-1, 1):  # vertical corner between a lip and the handle stop
        rounds.append(fillet([CR_X0, sy * lip_y(CR_X0), z_base], [1, 0, 0], [0, -sy, 0], [0, 0, 1], LIP_H, R_STOP))
    # handle stop to plate, across the cradle
    rounds.append(fillet([CR_X0, -lip_y(CR_X0), z_base], [1, 0, 0], [0, 0, 1], [0, 1, 0], 2 * lip_y(CR_X0), R_STOP))
    cradle = union([plate, *lips, stop, *rounds])
    # step between the top of the handle stop and the flange's front face
    step = fillet([FLANGE_T, -y_stop, z_base + LIP_H], [1, 0, 0], [0, 0, 1], [0, 1, 0], 2 * y_stop, R_STEP)
    ties = []
    for x in TIE_X:
        ty = lip_y(x) + LIP_W + TIE_W / 2 + 0.3  # just outside the lip
        for s in (-1, 1):
            ties.append(slot(x - TIE_L / 2, x + TIE_L / 2, s * ty, TIE_W, z_bot - 1, z_base + 1))
        # round-bottomed groove under the plate, 1.2 mm deep and TIE_L wide at the surface, so the tie sits flush
        rg = (1.2 ** 2 + (TIE_L / 2) ** 2) / (2 * 1.2)
        ties.append(cyl(rg, [x, -ty, z_bot + 1.2 - rg], [x, ty, z_bot + 1.2 - rg], 64))
    tie_cut = union(ties)
    cradle = union([cradle, step])
    cradle.apply_transform(roll); tie_cut.apply_transform(roll)

    body = union([plug, flange, cradle])
    cuts = [tie_cut]
    for a in np.deg2rad(RADIAL_ANGLES):  # blind holes for the tips of the 4 radial M4 screws
        d = np.array([0, np.cos(a), np.sin(a)])
        cuts.append(cyl(RADIAL_HOLE_D / 2, [RADIAL_X, 0, 0] + d * RADIAL_HOLE_R0, [RADIAL_X, 0, 0] + d * (PLUG_D / 2 + 1), 32))
    for sz in (-1, 1):  # slots for the bore keys
        z0, z1 = sorted((sz * SLOT_FLOOR, sz * (PLUG_D / 2 + 1)))
        cuts.append(box(x_plug - 1, 0.01, -SLOT_W / 2, SLOT_W / 2, z0, z1))
    return trimesh.boolean.difference([body, union(cuts)], engine="manifold")


if __name__ == "__main__":
    h = build_holder()
    HOLDER_STL.parent.mkdir(parents=True, exist_ok=True)
    h.export(HOLDER_STL)
    print(f"{HOLDER_STL}: watertight={h.is_watertight}, volume {h.volume / 1e3:.1f} cm3, "
          f"bounds (mm) {np.round(h.bounds, 1).tolist()}, ~{h.volume / 1e3 * 1.27 * 0.8:.0f} g in PETG")
