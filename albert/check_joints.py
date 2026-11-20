"""Check that Albert's right-arm joints mean the same as the simulation's, before running the policy. Run it first.

For each of the 5 right-arm joints: the script says, from the simulation, where the hand should go for +0.15 rad, then moves
that joint by +0.15 rad and back on Albert. Watch: if a joint goes the other way, set its ARM_SIGN to -1 in albert/common.py.
It also prints Albert's STAND pose next to the simulation's, to spot a zero offset (ARM_OFFSET).

    python -m albert.check_joints --supported      # Albert hanging from its gantry; ~25 s of slow arm motion
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from albert.common import graceful_signals, take_joints, hand_back, BalanceOnExit, safe_pose, ARM_FW, ARM_IDX, ARM_OFFSET, ARM_SIGN, Kinematics, fw_to_mjcf
from albert.sim_robot import _STAND_FW
from asimov_pp.model import ARM_JOINTS

STEP = 0.15


def _words(v, names, thr):
    out = []
    for val, (pos, neg) in sorted(zip(v, names), key=lambda t: -abs(t[0])):
        if abs(val) > thr:
            out.append(f"{abs(val):.1f} cm {pos if val > 0 else neg}")
    return out


def describe(kin, q_mjcf, j):
    """What +STEP on joint j should do, from the simulation, in the robot's frame (forward / left / up): where the blade centre
    goes, and which way the red face turns."""
    kin.update(q_mjcf); c0, R0 = kin.blade_frame()
    q1 = q_mjcf.copy(); q1[j] += STEP
    kin.update(q1); c1, R1 = kin.blade_frame()
    names = (("forward", "back"), ("left", "right"), ("up", "down"))
    move = _words((c1 - c0) * 100, names, 0.5)
    tilt = _words((R1[:, 2] - R0[:, 2]) * 100, (("forward", "backward"), ("to the robot's left", "to the robot's right"), ("up", "down")), 3.0)
    face = ("the red face turns to face more " + " and ".join(t.split(" ", 2)[2] for t in tilt[:2])) if tilt else "the red face keeps its direction"
    extra = {2: "; seen from above, the forearm swings CLOCKWISE (paddle towards the robot's right)",
             4: "; seen from behind the hand (from the elbow, looking at the paddle), the paddle rolls CLOCKWISE"}.get(j, "")
    return f"the blade centre {'moves ' + ', '.join(move) if move else 'barely moves'}; {face}{extra}"


def main():
    graceful_signals()
    p = argparse.ArgumentParser()
    p.add_argument("--supported", action="store_true")
    p.add_argument("--joints", nargs="*", default=None, help="only these, e.g. R_Shoulder_Yaw R_Wrist_Yaw")
    a = p.parse_args()
    if not a.supported:
        sys.exit("Hang Albert from its gantry, then run again with --supported.")
    from menlo.asimov import Robot

    kin = Kinematics()
    from menlo.asimov import NotReadyError, WaitTimeoutError

    with Robot().connect("udp") as robot, BalanceOnExit(robot):
        s = robot.get_state()
        if s.faulted:
            sys.exit("Albert reports a fault: not moving.")
        print("STAND: every motor holds a standing pose (no balancing)")
        take_joints(robot)
        start = safe_pose(robot.get_state().joint_pos)
        print("STAND pose, right arm (firmware frame):")
        for n, i in zip(ARM_FW, ARM_IDX):
            print(f"  {n:18s} Albert {start[i]:+.3f}   simulation standing pose {_STAND_FW.get(n, 0.0):+.3f} rad")
        q_mjcf = fw_to_mjcf(start[ARM_IDX])
        for j, (n, i) in enumerate(zip(ARM_FW, ARM_IDX)):
            if a.joints and n not in a.joints:
                continue
            print(f"\n{n} (+{STEP} rad, sign {ARM_SIGN[j]:+.0f}, offset {ARM_OFFSET[j]:+.2f}): the simulation expects: "
                  f"{describe(kin, q_mjcf, j)}")
            input("  Enter to move it ")
            tgt = start.copy(); tgt[i] += ARM_SIGN[j] * STEP
            try:
                robot.set_joints(tgt, duration=2.0)
                print(f"  reached {robot.get_state().joint_pos[i]:+.3f} rad; back")
            except (NotReadyError, WaitTimeoutError) as exc:
                print(f"  did not reach the target ({exc}); back")
            robot.set_joints(start, duration=2.0, wait=False)
            robot.wait_until(lambda st: abs(st.joint_pos[i] - start[i]) < 0.05, timeout=5)
            if input("  did the hand move as described? [y/n] ").strip().lower() == "n":
                print(f"  -> set ARM_SIGN[{j}] = {-ARM_SIGN[j]:+.0f} ({ARM_JOINTS[j]}) in albert/common.py and run this again")
        hand_back(robot)
    print("done")


if __name__ == "__main__":
    main()
