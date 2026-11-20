"""Arm system identification on Albert (on its gantry): each right-arm joint follows a gentle multisine around the paddle home
pose (0.3-4 Hz, about +-0.06 rad), then all five together, while menlo-sdk records every setpoint and joint state.
tools/sysid_arm.py then fits the simulated arm (delay, gains, damping, friction) to the recording.

    python -m albert.sysid_arm_test --supported --log logs/C1_sysid_arm.csv     # ~70 s of motion
    python -m tools.sysid_arm logs/C1_sysid_arm.jsonl
    python -m albert.sysid_arm_test --sim                                       # dry run
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from albert.common import to_balance, graceful_signals, with_stance, take_joints, hand_back, BalanceOnExit, ARM_IDX, HOME, KD, KP, PERIOD, TiltGuard, mjcf_to_fw, safe_pose

FREQS = np.array([0.3, 0.5, 0.8, 1.2, 1.7, 2.3, 3.0, 4.0])  # Hz
AMP = 0.06        # rad, peak of the sum (per joint)
PER_JOINT_S = 10.0
ALL_S = 20.0


def multisine(t, rng_phase, amp=AMP):
    # equal amplitude per frequency, random phases (low crest factor), ramped in and out over 1 s
    s = np.sin(2 * np.pi * FREQS * t + rng_phase).sum()
    return amp * s / len(FREQS) * 2.2


def main():
    graceful_signals()
    p = argparse.ArgumentParser()
    p.add_argument("--supported", action="store_true")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--amp", type=float, default=AMP)
    p.add_argument("--log", default="albert_sysid_arm.csv")
    a = p.parse_args()
    if not (a.supported or a.sim):
        sys.exit("Hang Albert from its gantry, then run again with --supported.")
    if a.sim:
        from albert.sim_robot import SimAlbert
        robot = SimAlbert(seed=0)
    else:
        from menlo.asimov import Robot
        robot = Robot().connect("udp")
    import os
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    rng = np.random.default_rng(0)
    plan = [(j, PER_JOINT_S) for j in range(5)] + [(None, ALL_S)]
    phases = rng.uniform(0, 2 * np.pi, (6, 5, len(FREQS)))
    with robot, (robot.record(a.log.replace(".csv", ".jsonl")) if not a.sim else robot.record(None)), BalanceOnExit(robot):
        if robot.get_state().faulted:
            sys.exit("Albert reports a fault: not moving.")
        take_joints(robot)
        start = safe_pose(robot.get_state().joint_pos)
        hold = with_stance(start); hold[ARM_IDX] = mjcf_to_fw(HOME)
        if not a.sim:
            input("Rope TAUT (carrying part of Albert's weight), E-Stop in reach. Enter: legs to the centred stance, arm home (3 s) ")
        print("arm to the paddle home pose (3 s); paddle mounted, NO ball")
        robot.set_joints(hold, duration=3.0)
        if not a.sim:
            input(f"Enter: {sum(d for _, d in plan):.0f} s of small arm motions (+-{a.amp:.2f} rad, up to 4 Hz) ")
        # gantry test: hanging from the taut rope Albert leans up to ~17 deg, so only the relative limit (+9 deg) matters here
        guard = TiltGuard(robot, on_trip=lambda: to_balance(robot), max_abs=25.0) if not a.sim else None
        if guard:
            guard.__enter__()
        try:
            for k_seg, (j, dur) in enumerate(plan):
                print(f"  {'all joints' if j is None else 'joint ' + str(j)}: {dur:.0f} s", flush=True)
                n = int(dur / PERIOD); tick = time.monotonic()
                for k in range(n):
                    t = k * PERIOD
                    ramp = min(1.0, t, dur - t)
                    q = HOME.copy()
                    for jj in (range(5) if j is None else [j]):
                        q[jj] += ramp * multisine(t, phases[k_seg, jj], a.amp * (0.6 if j is None else 1.0))
                    cmd = hold.copy(); cmd[ARM_IDX] = mjcf_to_fw(q)
                    robot.trajectory(cmd, kp=KP, kd=KD)
                    if not a.sim:
                        tick += PERIOD; time.sleep(max(0.0, tick - time.monotonic()))
        except KeyboardInterrupt:
            print("\nstopped")
        if guard:
            guard.__exit__()
        print("arm back to its starting pose (3 s)")
        hand_back(robot)  # legs stay in the stance: Albert's own pose tips over
    print("recording:", a.log.replace(".csv", ".jsonl"), "-> python -m tools.sysid_arm", a.log.replace(".csv", ".jsonl"))


if __name__ == "__main__":
    main()
