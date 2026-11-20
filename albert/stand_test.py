"""Albert standing WITHOUT the gantry's support, legs held stiff by joint control (as run_policy does): step-by-step test.

Under trajectory() Menlo's balance controller is off. The simulation (sim_freestand.py) says Albert's own stand() pose puts the
centre of mass 3 cm behind the toes and falls as soon as the arm moves, while the centred stance below (hips and knees a bit
more bent, ankles less) stays up with ~9 deg of lean. This script checks that on the robot, one step at a time:

  1. rope TAUT (carrying part of the weight, as on the gantry today): legs to the stance, arm to the paddle home pose (3 s).
     Never take over the legs unsupported: in the simulation, Albert's own stance tips over within 2 s of any takeover;
  2. you lower the gantry until the rope is slack (the script holds the stance and prints the lean), Enter;
  3. hold --hold s, then with --arm-replay the arm replays the motion the paddle policy made on Albert (data/2026-11-19/albert_run.csv),
     no ball, scaled by --arm-scale;
  4. you raise the rope until taut again, Enter: control back to Menlo's balance controller (balance()).

Watchdog, every 20 ms: lean more than WATCH_EXTRA deg beyond the start of the hold, or more than WATCH_MAX deg, or tipping
faster than WATCH_RATE rad/s -> balance() at once (Menlo's walking policy takes the legs) and the script stops.
Keep the gantry rope ATTACHED but slack, and the E-Stop in your hand.

    python -m albert.stand_test --supported                      # step 1 only
    python -m albert.stand_test --supported --arm-replay data/2026-11-19/albert_run.csv --arm-scale 0.5
    python -m albert.stand_test --sim --arm-replay data/2026-11-19/albert_run.csv  # dry run on the simulated free-standing Albert
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

from albert.common import (graceful_signals, STALL_S, allow_link_hiccups, state_age, unique_log, take_joints, hand_back, BalanceOnExit, ARM_IDX, HOME, KD, KP, NECK_PITCH_IDX, NECK_YAW_IDX, FW_JOINTS, PERIOD, fw_to_mjcf, lean_deg,
                           mjcf_to_fw, safe_pose)

STANCE = (-0.50, 0.55, -0.21)       # left leg hip / knee / ankle pitch (rad), right mirrored: sim_freestand.py
SIM_LEAN_DEG = 8.9                  # what the simulation predicts for it (arm at home)
WATCH_EXTRA, WATCH_MAX, WATCH_RATE = 5.0, 16.0, 0.8


class Trip(Exception):
    pass


def main():
    graceful_signals()
    p = argparse.ArgumentParser()
    p.add_argument("--supported", action="store_true", help="the gantry rope is attached (slack) and the E-Stop in hand")
    p.add_argument("--sim", action="store_true", help="dry run on the simulated free-standing Albert")
    p.add_argument("--stance", type=float, nargs=3, default=STANCE, metavar=("HIP", "KNEE", "ANKLE"))
    p.add_argument("--hold", type=float, default=20.0, help="s to hold the stance with the arm at home")
    p.add_argument("--arm-replay", default=None, help="a run_policy log (data/2026-11-19/albert_run.csv): replay its arm motion while balancing")
    p.add_argument("--arm-scale", type=float, default=0.5, help="scale of the replayed motion around home (start at 0.5)")
    p.add_argument("--replay-s", type=float, default=30.0)
    p.add_argument("--log", default="albert_stand.csv")
    a = p.parse_args()
    if not (a.supported or a.sim):
        sys.exit("Keep the gantry rope attached (slack) and the E-Stop in your hand, then run again with --supported.")

    from albert.sim_freestand import stance_fw
    if a.sim:
        from albert.sim_freestand import FreeStandAlbert
        robot = FreeStandAlbert(seed=0)
    else:
        from menlo.asimov import Robot
        robot = Robot().connect("udp")
    gains = dict(kp=KP, kd=KD)
    rows = []
    import os
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    if not a.sim:
        a.log = unique_log(a.log)
        allow_link_hiccups()
    with robot, (robot.record(a.log.replace(".csv", ".jsonl")) if not a.sim else robot.record(None)), BalanceOnExit(robot):
        st = robot.get_state()
        if st.faulted:
            sys.exit("Albert reports a fault: not moving.")
        hot = [(j.name, j.temp) for j in st.joints if (j.temp or 0) > 65]
        if hot:
            print("WARNING hot joints:", ", ".join(f"{n} {t:.0f}C" for n, t in hot))
        print(f"mode {getattr(st.mode, 'name', st.mode)}, lean {lean_deg(st):.1f} deg")
        if getattr(st.mode, "name", "") not in ("MOVE", "STAND"):
            print("Albert is not standing: stand() first (STAND), then this script takes the joints.")
            take_joints(robot)
        print(f"stance: hip {a.stance[0]:+.2f}, knee {a.stance[1]:+.2f}, ankle {a.stance[2]:+.2f} rad")
        if not a.sim:
            input("Albert on its feet, rope TAUT (carrying part of its weight), E-Stop in hand. Enter: legs to the stance (3 s) ")
        # read the pose NOW, right before moving: the first setpoint must be where the joints are, or they jump
        start = safe_pose(robot.get_state().joint_pos)
        target = start.copy()
        target[:12] = stance_fw(*a.stance, base=start)[:12]
        target[ARM_IDX] = mjcf_to_fw(HOME)
        print(f"  from hip {start[0]:+.2f}, knee {start[3]:+.2f}: legs to the stance (3 s)")
        rope = {"f": 0.5}  # sim only: the share of the weight the rope carries

        state = {"lean0": None, "tripped": False}

        def watch(st, phase):
            lean = lean_deg(st)
            rate = float(np.linalg.norm(np.asarray(st.gyro)[:2]))
            rows.append([time.monotonic() if not a.sim else st.received_at, phase, lean, rate, *st.gravity,
                         *[st.joints[i].pos for i in range(12)], *[(st.joints[i].temp or 0) for i in range(12)]])
            ref = state["lean0"] if state["lean0"] is not None else lean
            if lean > ref + WATCH_EXTRA or lean > WATCH_MAX or rate > WATCH_RATE:
                raise Trip(f"lean {lean:.1f} deg (start {ref:.1f}), tipping {rate:.2f} rad/s")
            return lean

        def stream(goal_fn, seconds, phase, report=False, until_enter=None, rope_to=None):
            """Stream setpoints at 50 Hz for `seconds`, or (until_enter: a prompt) until you press Enter."""
            import threading
            tick = time.monotonic()
            n = int(seconds / PERIOD); last_print = -1
            done = threading.Event()
            if until_enter and not a.sim:
                threading.Thread(target=lambda: (input(until_enter), done.set()), daemon=True).start()
            elif until_enter:
                print(until_enter + "[sim: the rope is moved over 6 s]")
            f0 = rope["f"]
            for k in range(n):
                if until_enter and done.is_set():
                    break
                if a.sim:
                    if rope_to is not None:
                        rope["f"] = f0 + (rope_to - f0) * min(1.0, k * PERIOD / 6.0)
                        if k * PERIOD > 8.0:
                            done.set()
                    robot.rope(rope["f"])
                robot.trajectory(goal_fn(k / max(n - 1, 1)), **gains)
                st = robot.get_state()
                if not a.sim and state_age(st) > STALL_S:  # Wi-Fi stall: keep sending, judge nothing from stale data
                    tick += PERIOD; time.sleep(max(0.0, tick - time.monotonic()))
                    continue
                lean = watch(st, phase)
                if report and int(k * PERIOD) != last_print:
                    last_print = int(k * PERIOD)
                    print(f"  {phase} t={k * PERIOD:4.0f} s  lean {lean:5.1f} deg  (sim predicts {SIM_LEAN_DEG:.1f})", flush=True)
                if not a.sim:
                    tick += PERIOD; time.sleep(max(0.0, tick - time.monotonic()))

        def smooth(x):
            return x * x * x * (10 - 15 * x + 6 * x * x)

        try:
            stream(lambda s: start + smooth(s) * (target - start), 3.0, "to_stance")
            print(f"in the stance (rope still carrying): lean {lean_deg(robot.get_state()):.1f} deg")
            stream(lambda s: target, 180.0, "lower_rope", report=True, rope_to=0.0,
                   until_enter="Lower the gantry SLOWLY until the rope is slack (watch the lean), then Enter ")
            state["lean0"] = lean_deg(robot.get_state())
            print(f"rope slack: lean {state['lean0']:.1f} deg (sim: {SIM_LEAN_DEG:.1f}). Holding {a.hold:.0f} s; watchdog armed.")
            stream(lambda s: target, a.hold, "hold", report=True)
            if a.arm_replay:
                r = np.genfromtxt(a.arm_replay, delimiter=",", names=True, dtype=None, encoding=None)
                bal = np.array([str(x) == "True" for x in r["balancing"]])
                q = np.c_[[r[n].astype(float) for n in r.dtype.names if n.startswith("q_")]].T[bal]
                q = HOME + a.arm_scale * (q - q[0])  # the motion, around this home, scaled
                n = min(len(q), int(a.replay_s / PERIOD))
                if not a.sim:  # wait for Enter WHILE streaming the hold (no setpoints for 2 s = Albert goes limp)
                    stream(lambda s: target, 600.0, "hold", until_enter=f"Enter: the arm replays {n * PERIOD:.0f} s of the "
                           f"policy's motion at x{a.arm_scale} (no ball) ")
                print(f"arm replay: {n * PERIOD:.0f} s at x{a.arm_scale}, max deviation from home "
                      f"{np.abs(q[:n] - HOME).max():.2f} rad")
                def replay(s, q=q, n=n):
                    g = target.copy(); g[ARM_IDX] = mjcf_to_fw(q[min(int(s * (n - 1)), n - 1)]); return g
                stream(replay, n * PERIOD, "arm_replay", report=True)
                stream(lambda s: target, 2.0, "hold")
            lean_end = lean_deg(robot.get_state())
            print(f"done: lean {lean_end:.1f} deg, max with the rope slack {max(r_[2] for r_ in rows if r_[1] in ('hold', 'arm_replay')):.1f} deg")
        except Trip as e:
            state["tripped"] = True
            print(f"\nWATCHDOG: {e} -> balance(): Menlo's balance controller takes over")
            hand_back(robot)
        except KeyboardInterrupt:
            print("\nstopped -> balance()")
            hand_back(robot)
        else:
            state["lean0"] = None  # the rope takes the weight again: absolute limits only
            try:
                stream(lambda s: target, 180.0, "raise_rope", until_enter="Raise the gantry until the rope is taut again, then Enter ",
                       rope_to=0.5)
            except Trip as e:
                print(f"\nWATCHDOG: {e}")
            print("control back to Menlo's balance controller (balance())")
            hand_back(robot)
    hdr = ["t", "phase", "lean_deg", "tip_rate", "g_x", "g_y", "g_z"] + [f"pos_{n}" for n in FW_JOINTS[:12]] + [f"temp_{n}" for n in FW_JOINTS[:12]]
    with open(a.log, "w", newline="") as fh:
        csv.writer(fh).writerows([hdr] + rows)
    print("log:", a.log, "| tripped" if state["tripped"] else "")


if __name__ == "__main__":
    main()
