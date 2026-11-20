"""A checkerboard to show on a screen (or print) for the head camera's lens calibration: 10 x 7 squares (9 x 6 inner corners).
After displaying it, measure one square's side on the screen in mm and pass it to the calibration.

    python -m tools.checkerboard            # -> data/checkerboard_9x6.png
"""
import numpy as np
import imageio.v3 as iio
import os

sq, nx, ny, margin = 120, 10, 7, 120
img = np.full((ny * sq + 2 * margin, nx * sq + 2 * margin), 255, np.uint8)
for i in range(ny):
    for j in range(nx):
        if (i + j) % 2 == 0:
            img[margin + i * sq: margin + (i + 1) * sq, margin + j * sq: margin + (j + 1) * sq] = 0
os.makedirs("data", exist_ok=True)
iio.imwrite("data/checkerboard_9x6.png", img)
print("data/checkerboard_9x6.png: 10 x 7 squares, 9 x 6 inner corners; show it full-screen, measure one square in mm")
