"""Check the arm model alone (no camera, no ball): the arm tilts the blade four ways, you read a phone level on the blade.

For each tilt the arm holds the pose; you type the phone's two readings (along the handle, across it, in degrees, the same way
for every tilt). The script prints them next to what the arm model + IMU predicted. If they disagree, the model's joint
offsets are wrong for tilting even if the home pose is flat; if they agree, look at the camera instead.

    python -m albert.tilt_phone --supported
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from albert.common import graceful_signals, take_joints, hand_back, BalanceOnExit, ARM_IDX, FLAT_HOME_FW, Kinematics, TiltGuard, fw_to_mjcf, mjcf_to_fw, safe_pose
from albert.tilt_test import TESTS, TILT_DEG, tilt_q


def predicted(kin, q, quat):
    """Blade slope along the handle (+ = tip up) and across (+ = left edge up), degrees, from the model + IMU."""
    kin.update(q, base_quat=quat)
    R = kin.blade_frame()[1]
    return float(np.degrees(np.arcsin(R[2, 0]))), float(np.degrees(np.arcsin(R[2, 1])))


def main():
    graceful_signals()
    p = argparse.ArgumentParser()
    p.add_argument("--supported", action="store_true")
    a = p.parse_args()
    if not a.supported:
        sys.exit("Hang Albert from its gantry, then run again with --supported.")
    from menlo.asimov import Robot

    kin = Kinematics()
    with Robot().connect("udp") as robot, BalanceOnExit(robot):
        if robot.get_state().faulted:
            sys.exit("Albert reports a fault: not moving.")
        take_joints(robot)
        start = safe_pose(robot.get_state().joint_pos)
        home = start.copy(); home[ARM_IDX] = FLAT_HOME_FW
        print("arm to the home position (3 s)")
        robot.set_joints(home, duration=3.0, wait=False); time.sleep(3.3)
        rows = []
        with TiltGuard(robot, on_trip=lambda: robot.set_joints(start, duration=2.0, wait=False)):
            try:
                for name, tip, left in [("HOME (no tilt)", 0, 0)] + TESTS:
                    s_ = robot.get_state()
                    q_t = tilt_q(kin, fw_to_mjcf(home[ARM_IDX]), TILT_DEG * tip, TILT_DEG * left, s_.quat)
                    tgt = home.copy(); tgt[ARM_IDX] = mjcf_to_fw(q_t)
                    robot.set_joints(tgt, duration=1.0, wait=False); time.sleep(2.0)
                    s_ = robot.get_state()
                    pa, pc = predicted(kin, fw_to_mjcf(np.array(s_.joint_pos)[ARM_IDX]), s_.quat)
                    txt = input(f"[{name}] phone level: along the handle, across (e.g. '2.5 -0.5'): ").replace(",", " ").split()
                    try:
                        ma, mc = float(txt[0]), float(txt[1])
                    except (ValueError, IndexError):
                        print("  skipped"); continue
                    rows.append((name, pa, pc, ma, mc))
                    print(f"  model: along {pa:+.1f}, across {pc:+.1f} deg | phone: along {ma:+.1f}, across {mc:+.1f} deg")
                robot.set_joints(home, duration=1.0, wait=False); time.sleep(1.2)
            except KeyboardInterrupt:
                print("\nstopped")
        print("arm back to its starting pose (3 s)")
        hand_back(robot, start)
    if len(rows) >= 3:
        P, M = np.array([r[1:3] for r in rows]), np.array([r[3:5] for r in rows])
        dP, dM = P - P[0], M - M[0]  # changes from home: the phone's own zero and sign convention cancel out
        for k, ax in enumerate(("along", "across")):
            c = np.corrcoef(dP[:, k], dM[:, k])[0, 1] if np.std(dM[:, k]) > 0 else np.nan
            g = np.dot(dP[:, k], dM[:, k]) / max(np.dot(dP[:, k], dP[:, k]), 1e-9)
            print(f"{ax:7s}: phone change = {g:+.2f} x model change (correlation {c:+.2f})")
        cross = [np.corrcoef(dP[:, 0], dM[:, 1])[0, 1], np.corrcoef(dP[:, 1], dM[:, 0])[0, 1]]
        print(f"swapped axes check: along->across {cross[0]:+.2f}, across->along {cross[1]:+.2f}")


if __name__ == "__main__":
    main()
