"""Does balance() keep Albert in MOVE (Menlo's walking policy) after our scripts? Diagnostic, no joint commands of ours.

stand() if limp (DAMP), balance(), then the mode every 0.25 s for 10 s; then this session closes and a new one reads the
mode 3 s later. Prints alerts as they change. Albert on its gantry, E-Stop in reach.

    python -m albert.balance_check --supported
"""
from __future__ import annotations

import argparse
import sys
import time


def mode_line(st):
    alerts = [getattr(a, "name", None) or str(a) for a in (st.alerts or [])]
    return f"mode {getattr(st.mode, 'name', st.mode)}, armed {getattr(st, 'armed', '?')}, alerts {alerts}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--supported", action="store_true")
    p.add_argument("--after-trajectory", choices=("direct", "stand_first"), default=None,
                   help="first hold the current pose with trajectory() for 2 s (no motion), then hand back: balance() directly, "
                        "or stand() then balance()")
    a = p.parse_args()
    if not a.supported:
        sys.exit("Albert on its gantry, then --supported.")
    from menlo.asimov import Robot
    with Robot().connect("udp") as robot:
        st = robot.get_state()
        print("start:", mode_line(st))
        if getattr(st.mode, "name", "") == "DAMP":
            print("stand() ..."); robot.stand(); print("  ", mode_line(robot.get_state()))
        if a.after_trajectory:
            import numpy as np
            from albert.common import KD, KP, safe_pose
            robot.balance(timeout=5.0); time.sleep(1.0)
            hold = safe_pose(robot.get_state().joint_pos)
            print("holding the current pose with trajectory() for 2 s ...")
            for _ in range(100):
                robot.trajectory(hold, kp=KP, kd=KD); time.sleep(0.02)
            print("  ", mode_line(robot.get_state()))
            if a.after_trajectory == "stand_first":
                print("stand() ...")
                try:
                    robot.stand(timeout=5.0)
                except Exception as exc:  # noqa: BLE001
                    print("  stand raised:", repr(exc))
                print("  ", mode_line(robot.get_state()))
        print("balance(wait=True) ...")
        try:
            robot.balance(timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            print("  balance raised:", repr(exc))
        last = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            line = mode_line(robot.get_state())
            if line != last:
                print(f"  t={time.monotonic() - t0:4.1f} s  {line}"); last = line
            time.sleep(0.25)
        print("closing this session")
    time.sleep(3.0)
    with Robot().connect("udp") as robot:
        print("3 s after closing, new session:", mode_line(robot.get_state()))


if __name__ == "__main__":
    main()
