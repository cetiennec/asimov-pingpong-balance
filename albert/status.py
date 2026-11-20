"""Read-only check of Albert before any motion: connection, mode, faults, temperatures, battery, right-arm angles, IMU, and the
head camera (resolution, frame rate, one photo + what the ball detector sees in it). Sends no motion command.

    python -m albert.status                 # state over UDP (Edge's udp-state-host must be this machine) + camera
    python -m albert.status --mode livekit  # everything over LiveKit: works from the Mac while the Pi receives the UDP state
"""
from __future__ import annotations

import time

import numpy as np

from albert.ball_camera import BALL_ORANGE, BALL_WHITE, debug_image, detect
from albert.common import ARM_FW, BALL, ROOT
from albert.sim_robot import _STAND_FW


def main():
    import argparse
    from menlo.asimov import Robot
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="hybrid", choices=("hybrid", "livekit", "udp"))
    a = p.parse_args()

    ball = BALL_WHITE if BALL == "white" else BALL_ORANGE
    with Robot().connect(a.mode) as robot:
        info = robot.info
        s = robot.get_state()
        print(f"connected: {info.transport} {info.endpoint}, {info.dof} motors, protocol {info.protocol_version}")
        print(f"mode {s.mode.name}, armed {robot.armed}, faulted {s.faulted}, alerts {[a.name for a in s.alerts] if s.alerts else 'none'}")
        temps = [(j.temp, j.name) for j in s.joints if j.temp is not None]
        if temps:
            t, n = max(temps)
            print(f"hottest joint {t:.0f} C ({n})")
        if s.battery is not None:
            print(f"battery {s.battery.soc_percent:.0f} %")
        print(f"gravity (body) {np.round(s.gravity, 3) if s.gravity else None}, upright {s.upright}")
        print("right arm (firmware frame) vs the simulation's standing pose:")
        for n in ARM_FW:
            print(f"  {n:18s} {s.joint(n).pos:+.3f} rad   (sim stand {_STAND_FW.get(n, 0.0):+.3f})")
        for n in ("Waist_Yaw", "Neck_Yaw", "Neck_Pitch"):
            print(f"  {n:18s} {s.joint(n).pos:+.3f} rad")
        if not robot.has("camera"):
            print("camera: not carried by this connection")
            return
        f0 = robot.camera.photo(timeout=5)
        img = f0.to_numpy()
        print(f"camera: {f0.width}x{f0.height} {f0.encoding}")
        n, t0, seqs = 0, time.monotonic(), set()
        for f in robot.camera.frames(timeout=5):
            seqs.add(f.sequence); n += 1
            if time.monotonic() - t0 > 3.0:
                break
        dt = time.monotonic() - t0
        print(f"camera: {len(seqs)} distinct frames in {dt:.1f} s = {len(seqs) / dt:.1f} fps")
        import imageio
        imageio.imwrite(ROOT / "status_photo.png", img)
        imageio.imwrite(ROOT / "status_detect.png", debug_image(img, ball))
        det = detect(img, ball)
        print(f"ball detector on that photo: {'ball at ' + str(np.round(det, 1)) if det else 'no ball on a red blade in view'} "
              f"-> albert/status_photo.png, albert/status_detect.png")


if __name__ == "__main__":
    main()
