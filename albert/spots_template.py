"""Printable 1:1 stencil of the 9 calibration spots (A4, 300 dpi) -> albert/calibration_spots_A4.png and .pdf

Print at 100 % ("actual size", no "fit to page"), check the 10 cm bar with a ruler, cut the red disc out, lay it on the RED face
with the long line on the handle's axis and the cross on the blade's centre, prick each spot through with a pen, remove the
paper (left on, it would change the ball's grip and colours the camera sees).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from albert.calibrate_camera import SPOTS
from albert.common import ROOT

W, H = 210.0, 297.0                       # A4, mm
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
cx, cy = W / 2, 165.0                     # blade centre on the page
ax.add_patch(Circle((cx, cy), 76, color="#d42020", lw=0))                         # 152 mm red disc (cut line = its edge)
ax.plot([cx, cx], [cy - 76, cy + 76], color="w", lw=0.6)                       # handle axis (u)
ax.plot([cx, cx], [cy - 95, cy - 76], color="k", lw=0.6)
ax.plot([cx - 76, cx + 76], [cy, cy], color="w", lw=0.4, ls=":")               # w axis
for i, (u, w) in enumerate(SPOTS, 1):
    x, y = cx - w * 1000, cy + u * 1000   # +w to the left, +u up (red face towards you, handle down)
    ax.add_patch(Circle((x, y), 3.5, color="w", lw=0, zorder=3))           # white spot
    ax.plot(x, y, "k.", ms=1.5, zorder=4)                                  # centre: prick here
    ax.text(x + 4.5, y + 3.0, str(i), fontsize=11, weight="bold", color="w")
ax.annotate("", xy=(cx, cy - 92), xytext=(cx, cy - 80), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.text(cx + 3, cy - 90, "handle this way", fontsize=9)
ax.text(cx + 2, cy + 82, "tip of the blade (+u)", fontsize=9)
ax.text(cx - 84, cy + 2, "+w", fontsize=9)
ax.text(cx, H - 18, "Asimov paddle: camera-calibration spots (RED face, handle down)", ha="center", fontsize=12, weight="bold")
ax.text(cx, H - 27, "Print at 100 % / actual size. The bar below must measure exactly 10 cm.", ha="center", fontsize=9)
ax.text(cx, H - 33, "Cut along the red disc, put the long line on the handle axis and spot 1 on the blade centre,\n"
        "prick each spot through, then remove the paper.", ha="center", va="top", fontsize=8.5)
# scale bar 100 mm with 1 cm ticks
y0, x0 = 40.0, cx - 50
ax.add_patch(Rectangle((x0, y0), 100, 2.5, color="k"))
for k in range(11):
    ax.plot([x0 + 10 * k, x0 + 10 * k], [y0 - 2.5, y0], color="k", lw=0.8)
    ax.text(x0 + 10 * k, y0 - 7, str(k), ha="center", fontsize=7)
ax.text(cx, y0 - 13, "10 cm", ha="center", fontsize=9)
lines = "   ".join(f"{i}: u{100*u:+.0f} w{100*w:+.0f}" for i, (u, w) in enumerate(SPOTS, 1))
ax.text(cx, 18, "cm:  " + lines, ha="center", fontsize=6.5, family="monospace")
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"calibration_spots_A4.{ext}", dpi=300)
print("wrote", ROOT / "calibration_spots_A4.png", "and .pdf")
