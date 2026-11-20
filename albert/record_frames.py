"""Record head-camera frames, no motion (Albert can stay in balance): for the camera's lens calibration (a checkerboard shown
at many angles) or any visual dataset. Works from the Mac over LiveKit while Albert's UDP state goes to the Pi.

    python -m albert.record_frames --seconds 90 --every 0.5 --name checkerboard

With OpenCV installed (pip install opencv-python-headless) it says live whether the 9 x 6 checkerboard is in view and which
parts of the image (a 6 x 4 grid) the board has covered so far: for the lens calibration, cover every cell, corners included,
with the board both flat to the camera and tilted. Then: python -m albert.calibrate_lens <folder>.
"""
from __future__ import annotations

import argparse
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--every", type=float, default=0.5, help="s between saved frames")
    p.add_argument("--name", default="frames")
    p.add_argument("--mode", default="livekit", choices=("livekit", "hybrid"))
    a = p.parse_args()
    import imageio.v3 as iio
    from menlo.asimov import Robot
    out = os.path.join("data", time.strftime("%Y-%m-%d"), f"{a.name}_{time.strftime('%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    with Robot().connect(a.mode) as robot:
        st = robot.get_state()
        print(f"recording {a.seconds:.0f} s, a frame every {a.every} s -> {out}  (mode {st.mode.name}; no motion is sent)", flush=True)
        try:
            import cv2  # live feedback: is the 9 x 6 checkerboard in view?
        except ImportError:
            cv2 = None
        t0 = time.monotonic(); n = 0; last = None; found = 0
        import numpy as np
        grid = np.zeros((4, 6), bool)
        while time.monotonic() - t0 < a.seconds:
            f = robot.camera.latest()
            if f is not None and f.received_at != last:
                last = f.received_at
                s = robot.get_state()
                iio.imwrite(os.path.join(out, f"{n:04d}.png"), f.to_numpy()[..., :3])
                with open(os.path.join(out, "poses.csv"), "a") as fh:
                    fh.write(f"{n},{f.received_at},{','.join(f'{v:.5f}' for v in s.joint_pos)},{','.join(f'{v:.5f}' for v in s.quat)}\n")
                n += 1
                if cv2 is not None:
                    g = cv2.cvtColor(f.to_numpy()[::2, ::2, :3].copy(), cv2.COLOR_RGB2GRAY)
                    ok, c = cv2.findChessboardCorners(g, (9, 6), cv2.CALIB_CB_FAST_CHECK)
                    found += int(ok)
                    if ok:
                        c = c.reshape(-1, 2)
                        grid[np.clip((c[:, 1] / g.shape[0] * 4).astype(int), 0, 3), np.clip((c[:, 0] / g.shape[1] * 6).astype(int), 0, 5)] = True
                    rows = " | ".join("".join("#" if v else "." for v in row) for row in grid)
                    print(f"  frame {n}: {'BOARD SEEN' if ok else 'no board  '}  ({found} so far)  covered {100 * grid.mean():3.0f} %  "
                          f"[{rows}]  (top row first)", flush=True)
                elif n % 10 == 0:
                    print(f"  {n} frames", flush=True)
                time.sleep(a.every)
            else:
                time.sleep(0.01)
    print(f"done: {n} frames in {out} (+ poses.csv: joint angles and IMU at each frame)")


if __name__ == "__main__":
    main()
