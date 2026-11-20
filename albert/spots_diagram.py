"""Draw the 9 calibration spots, numbered in the order calibrate_camera.py asks for them -> albert/calibration_spots.png.
View: the red face (the face up when Albert holds the paddle) towards you, handle pointing down. Then +u (blade frame x) points
up, towards the tip, and +w (blade frame y = normal x u) points to your left."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from albert.calibrate_camera import SPOTS
from albert.common import ROOT
from asimov_pp.model import BLADE_R

R = BLADE_R * 100
fig, ax = plt.subplots(figsize=(5.5, 8))
ax.add_patch(Rectangle((-1.3, -R - 10), 2.6, 10.5, color="#b07a45"))
ax.add_patch(Circle((0, 0), R, color="#d22"))
for x in range(-6, 7, 2):  # 2 cm grid to help with the ruler
    ax.plot([x, x], [-6.5, 6.5], color="w", lw=0.4, alpha=0.5); ax.plot([-6.5, 6.5], [x, x], color="w", lw=0.4, alpha=0.5)
for i, (u, w) in enumerate(SPOTS, 1):
    x, y = -w * 100, u * 100           # +w to the left, +u up
    ax.plot(x, y, "o", ms=20, mfc="white", mec="k"); ax.text(x, y, str(i), ha="center", va="center", fontsize=11, weight="bold")
ax.annotate("", xy=(0, R + 2.5), xytext=(0, R + 0.3), arrowprops=dict(arrowstyle="->", lw=2)); ax.text(0.4, R + 1.6, "+u: towards the tip", fontsize=10)
ax.annotate("", xy=(-R - 2.5, 0), xytext=(-R - 0.3, 0), arrowprops=dict(arrowstyle="->", lw=2)); ax.text(-R - 2.6, 0.6, "+w", fontsize=10)
ax.text(0, -R - 11.5, "handle (down)", ha="center", fontsize=9)
lines = "\n".join(f"{i}: u {100*u:+.0f}, w {100*w:+.0f} cm" for i, (u, w) in enumerate(SPOTS, 1))
ax.text(R + 1.2, -R - 9, lines, fontsize=8.5, family="monospace", va="bottom")
ax.set_title("Calibration spots on the RED face (face up on Albert)\nhandle pointing down, grid 2 cm, spot 1 = blade centre", fontsize=10)
ax.set_xlim(-R - 4, R + 9); ax.set_ylim(-R - 12.5, R + 3.5); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(ROOT / "calibration_spots.png", dpi=130, bbox_inches="tight")
print("wrote", ROOT / "calibration_spots.png")
