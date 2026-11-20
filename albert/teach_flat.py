"""Teach Albert where "flat" is: the right arm goes limp, you hold the paddle flat by hand, the script records the pose.

Albert holds its standing pose except the right arm (no stiffness, a little damping). Put the paddle flat (red face up, check
with a phone level) in front of the robot and press Enter; do it 3 times at slightly different places (the first one where you
want the paddle while balancing). From the recorded joint angles and the IMU (the body may lean on the gantry), the arm model
gives the 5 joint zero offsets that make every recorded pose flat. Saved to albert/arm_offset.json with the first pose, which
calibrate_camera.py and run_policy.py then use as the starting pose.

    python -m albert.teach_flat --supported
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import numpy as np
from scipy.optimize import least_squares

from albert.common import with_stance, graceful_signals, take_joints, hand_back, BalanceOnExit, safe_pose, ARM_IDX, ARM_SIGN, KD, KP, OFFSET_FILE, Kinematics, TiltGuard

LIMP_KP, LIMP_KD = 0.0, 1.0


def fit_offsets(kin, samples):
    """Offsets o (firmware = mjcf + o, signs +1) so that the model blade normal is vertical (along -gravity) in every sample."""
    def res(o):
        out = []
        for q_fw, g in samples:
            kin.update((q_fw - o) / ARM_SIGN)  # base at identity: the model's pelvis frame = the IMU's body frame
            n = kin.blade_frame()[1][:, 2]
            up = -np.asarray(g) / np.linalg.norm(g)
            out.append(n - up * (n @ up))  # the part of the normal that is not vertical
        return np.concatenate(out + [0.05 * o])
    r = least_squares(res, np.zeros(5))
    tilts = [np.degrees(np.arcsin(min(1.0, np.linalg.norm(r.fun[3 * i:3 * i + 3])))) for i in range(len(samples))]
    return r.x, tilts


def main():
    graceful_signals()
    p = argparse.ArgumentParser()
    p.add_argument("--supported", action="store_true")
    p.add_argument("--poses", type=int, default=3)
    p.add_argument("--add", action="store_true", help="add these poses to the ones already taught (refit with all), keep the home")
    a = p.parse_args()
    if not a.supported:
        sys.exit("Hang Albert from its gantry, then run again with --supported.")
    from menlo.asimov import Robot

    kin = Kinematics()
    kp, kd = list(KP), list(KD)
    for i in ARM_IDX:
        kp[i], kd[i] = LIMP_KP, LIMP_KD
    with Robot().connect("udp") as robot, BalanceOnExit(robot):
        if robot.get_state().faulted:
            sys.exit("Albert reports a fault: not moving.")
        take_joints(robot)
        start = safe_pose(robot.get_state().joint_pos)
        input("Rope TAUT (carrying part of Albert's weight), E-Stop in reach. Enter: legs to the centred stance (3 s) ")
        stance = with_stance(start)  # Albert's own standing poses tip over under joint control
        robot.set_joints(stance, duration=3.0, wait=False, kp=KP, kd=KD); time.sleep(3.3)
        start = stance
        if a.add:  # start from the current home position
            from albert.common import FLAT_HOME_FW
            home = start.copy(); home[ARM_IDX] = FLAT_HOME_FW
            print("arm to the home position (3 s)")
            robot.set_joints(home, duration=3.0, wait=False); time.sleep(3.3)
            start0, start = start, home
        input("Hold the paddle (the right arm will go limp), then Enter ")
        stop = threading.Event()

        def limp():  # hold everything but the right arm, which follows wherever you put it
            n_err = 0
            while not stop.is_set():  # never dies: no setpoint for 2 s and Albert goes limp
                try:
                    q = np.array(robot.get_state().joint_pos)
                    cmd = start.copy(); cmd[ARM_IDX] = q[ARM_IDX]
                    robot.trajectory(cmd, kp=kp, kd=kd)
                except Exception as exc:  # noqa: BLE001
                    n_err += 1
                    if n_err <= 3:
                        print(f"  (limp-arm stream: {exc!r}; continuing)", flush=True)
                time.sleep(0.02)

        th = threading.Thread(target=limp, daemon=True); th.start()
        samples = []
        try:
            for k in range(1, a.poses + 1):
                if a.add:
                    where = "where it is now (the home position)" if k == 1 else "a few cm higher" if k == 2 else "a few cm to the side"
                else:
                    where = "where you want it while balancing" if k == 1 else "about 10 cm higher" if k == 2 else "about 10 cm to the side (or further forward)"
                input(f"pose {k}/{a.poses}: paddle FLAT, red face up, {where}; hold it still, Enter ")
                qs, gs = [], []
                for _ in range(15):  # 0.3 s average
                    s = robot.get_state(); qs.append(np.array(s.joint_pos)[ARM_IDX]); gs.append(np.array(s.gravity)); time.sleep(0.02)
                samples.append((np.mean(qs, 0), np.mean(gs, 0)))
                print(f"  recorded: arm {np.round(samples[-1][0], 3).tolist()}, body gravity {np.round(samples[-1][1], 3).tolist()}")
            input("Keep holding the paddle: the arm stiffens again at pose 1. Enter ")
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            stop.set(); th.join(timeout=1)
        if samples:
            hold = start.copy(); hold[ARM_IDX] = samples[0][0]
            robot.set_joints(hold, duration=1.5, wait=False); time.sleep(1.8)
        if len(samples) >= 1:
            old = json.loads(OFFSET_FILE.read_text()) if (a.add and OFFSET_FILE.exists()) else {}
            all_s = [(np.array(x["q_fw"]), np.array(x["gravity"])) for x in old.get("samples", [])] + samples
            o, tilts = fit_offsets(kin, all_s)
            import shutil
            if OFFSET_FILE.exists():  # the previous teaching, kept
                shutil.copy(OFFSET_FILE, OFFSET_FILE.with_name(f"arm_offset_{time.strftime('%Y%m%d_%H%M%S')}.json"))
            keep = json.loads(OFFSET_FILE.read_text()) if OFFSET_FILE.exists() else {}
            if "home_mjcf" in keep:  # the home pose the policy was trained with: never dropped by a fresh teaching
                old = dict(old, home_mjcf=keep["home_mjcf"])
            out = dict(old, arm_offset=o.tolist(), samples=[dict(q_fw=q.tolist(), gravity=g.tolist()) for q, g in all_s],
                       residual_tilt_deg=tilts, date=time.strftime("%Y-%m-%d %H:%M"))
            if not a.add or "flat_home_fw" not in old:
                out["flat_home_fw"] = samples[0][0].tolist()
            OFFSET_FILE.write_text(json.dumps(out, indent=1))
            print(f"offsets {np.round(o, 3).tolist()} rad (shoulder pitch, roll, yaw, elbow, wrist yaw); "
                  f"model tilt left at each pose {np.round(tilts, 1).tolist()} deg -> {OFFSET_FILE}")
            if max(tilts) > 3:
                print("  the poses do not agree well (>3 deg): redo it, holding the paddle flatter and still")
        input("Enter: the arm goes back down (4 s), then Menlo's balance controller takes over ")
        hand_back(robot)
    print("done: Albert is under Menlo's balance controller")


if __name__ == "__main__":
    main()
