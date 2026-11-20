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

Print: PETG / PLA+ / PA12, 100 % infill for the plug (or 6 walls), standing on the flange, plug up; the cradle needs supports.
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
CR_W, CR_T = 44.0, 5.0     # base plate width (y) and thickness
LIP_Y, LIP_W, LIP_H = 15.5, 2.5, 6.0   # centring lips (handles up to 31 mm wide fit between them)
TIE_W, TIE_L = 2.6, 5.6    # slots for 4.8 mm zip ties (slot length along x)
TIE_X = (CR_X0 + 9, (CR_X0 + CR_X1) / 2, CR_X1 - 9)
TIE_Y = 19.5


def box(x0, x1, y0, y1, z0, z1):
    b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


def cyl(r, p0, p1, sections=96):
    return trimesh.creation.cylinder(radius=r, segment=[np.asarray(p0, float), np.asarray(p1, float)], sections=sections)


def union(parts):
    return trimesh.boolean.union(parts, engine="manifold")


def build_holder() -> trimesh.Trimesh:
    x_plug = -(BORE_DEPTH - PLUG_GAP)
    # 1) plug in the cup bore (with a 0.8 mm lead-in chamfer), 2) flange on the rim
    plug = union([cyl(PLUG_D / 2, [x_plug + 0.8, 0, 0], [0.5, 0, 0]), cyl(PLUG_D / 2 - 0.8, [x_plug, 0, 0], [x_plug + 1.0, 0, 0])])
    flange = cyl(FLANGE_D / 2, [0, 0, 0], [FLANGE_T, 0, 0])
    # 3) cradle, built in the paddle frame then rolled about x
    z_base = -HANDLE_T * 1e3 / 2              # the handle's lower face (black-rubber side) rests here
    plate = box(0, CR_X1, -CR_W / 2, CR_W / 2, z_base - CR_T, z_base)
    lips = [box(CR_X0, CR_X1, LIP_Y, LIP_Y + LIP_W, z_base - 0.1, z_base + LIP_H),
            box(CR_X0, CR_X1, -LIP_Y - LIP_W, -LIP_Y, z_base - 0.1, z_base + LIP_H)]
    stop = box(0, CR_X0, -LIP_Y - LIP_W, LIP_Y + LIP_W, z_base - CR_T, z_base + LIP_H)
    cradle = union([plate, *lips, stop])
    ties = []
    for x in TIE_X:
        for s in (-1, 1):
            ties.append(box(x - TIE_L / 2, x + TIE_L / 2, s * TIE_Y - TIE_W / 2, s * TIE_Y + TIE_W / 2, z_base - CR_T - 1, z_base + 1))
        ties.append(box(x - TIE_L / 2, x + TIE_L / 2, -TIE_Y, TIE_Y, z_base - CR_T - 1, z_base - CR_T + 1.2))  # flush groove
    tie_cut = union(ties)
    roll = trimesh.transformations.rotation_matrix(np.deg2rad(PADDLE_ROLL_DEG), [1, 0, 0])
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
