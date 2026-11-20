"""Run an albert/ script through its REAL-robot code path (prompts, watchdog, threads, logs) against a simulated robot, to
catch errors before a session on Albert. menlo.asimov.Robot is replaced by the simulated Albert; Enter is pressed for you;
the ball is put on the paddle 3 s after the loop starts.

    python -m tools.mock_real [--freestand] -- albert.run_policy --supported --duration 15 --truth-ball
"""
from __future__ import annotations

import builtins
import contextlib
import runpy
import sys
import os
import time
import types

STATE = {"held": False, "streamed_at": None}


def main():
    argv = sys.argv[1:]
    free = "--freestand" in argv[: argv.index("--")]
    mod, args = argv[argv.index("--") + 1], argv[argv.index("--") + 2:]

    class MockRobot:
        def __init__(self, *a, **k):
            pass

        def connect(self, *a, **k):
            if free:
                from albert.sim_freestand import FreeStandAlbert
                r = FreeStandAlbert(seed=0)
                r.rope(0.5)
            else:
                from albert.sim_robot import SimAlbert
                r = SimAlbert(seed=0)
            r.record = lambda path: contextlib.nullcontext()
            if not hasattr(r, "balance"):
                r.balance = lambda *a, **k: print("  [mock] balance()")
            n = {"k": 0}
            orig = r.trajectory

            def traj(pos, **k):
                orig(pos, **k)
                n["k"] += 1
                if free:
                    r.rope(0.5 if n["k"] < 150 else 0.0)
                if n["k"] == 400:
                    r.place_ball(0.02, -0.01)
            r.trajectory = traj
            # Asimov Edge damps 2 s after the last setpoint: a prompt is only safe while set_joints() holds in the background
            # (or another thread streams). Track it.
            orig_sj = r.set_joints
            def sj(*a_, **k_):
                orig_sj(*a_, **k_); STATE["held"] = True
            r.set_joints = sj
            def traj2(pos, **k):
                STATE["held"] = False; STATE["streamed_at"] = time.monotonic(); traj(pos, **k)
            r.trajectory = traj2
            # MOCK_STALL="start,duration" (simulated s): the state stops updating, as in a Wi-Fi stall
            import os
            stall = [float(x) for x in os.environ.get("MOCK_STALL", "").split(",") if x]
            orig_gs = r.get_state
            frozen = {}
            def gs():
                st_ = orig_gs()
                if stall and stall[0] <= r.t < stall[0] + stall[1]:
                    if "st" not in frozen:
                        frozen["st"], frozen["t"] = st_, r.t
                    frozen["st"].age_s = r.t - frozen["t"]
                    return frozen["st"]
                frozen.clear(); st_.age_s = 0.0
                return st_
            r.get_state = gs
            # a menlo-sdk-shaped recording (<log>.jsonl: states and sent trajectories), so the mock's runs export to LeRobot
            import json
            rec = {"fh": None}
            gs_inner, traj_inner = r.get_state, r.trajectory
            def gs_rec():
                st_ = gs_inner()
                if rec["fh"]:
                    rec["fh"].write(json.dumps({"kind": "state", "t": st_.received_at, "joints": [{"pos": j.pos} for j in st_.joints],
                                                "gravity": list(st_.gravity), "gyro": list(st_.gyro)}) + "\n")
                return st_
            def traj_rec(pos, **k):
                if rec["fh"]:
                    rec["fh"].write(json.dumps({"kind": "sent", "name": "trajectory", "t": r.t,
                                                "command": {"positions": [float(x) for x in pos]}}) + "\n")
                traj_inner(pos, **k)
            r.get_state, r.trajectory = gs_rec, traj_rec

            @contextlib.contextmanager
            def record(path):
                if path:
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    rec["fh"] = open(path, "w")
                try:
                    yield
                finally:
                    if rec["fh"]:
                        rec["fh"].close(); rec["fh"] = None
            r.record = record
            return r

    fake = types.ModuleType("menlo.asimov"); fake.Robot = MockRobot
    sys.modules["menlo"] = types.ModuleType("menlo"); sys.modules["menlo.asimov"] = fake
    import threading

    def mock_input(prompt=""):
        print(prompt + "[mock: Enter]")
        # main-thread prompt with joint control active and nothing holding: on Albert, DAMP 2 s later
        if threading.current_thread() is threading.main_thread() and STATE["streamed_at"] and not STATE["held"]:
            time.sleep(0.15)  # a streaming thread, if any, would have sent a setpoint by now
            if time.monotonic() - STATE["streamed_at"] > 0.1:
                raise RuntimeError(f"DAMP: prompt {prompt!r} waits while no setpoint is streamed and no set_joints() holds")
        return ""
    builtins.input = mock_input
    # run in a scratch folder: the scripts write their logs (albert_demo.csv, data/2026-11-19/albert_run.csv ...) to the current folder, and
    # a mock run must never overwrite a real recording. Inputs they read (run logs for --arm-replay) are copied in.
    import os
    import shutil
    import tempfile
    repo = os.getcwd()
    sys.path.insert(0, repo)
    tmp = tempfile.mkdtemp(prefix="mock_real_")
    for f in os.listdir(repo):
        if f.endswith(".csv"):
            shutil.copy(os.path.join(repo, f), tmp)
    os.symlink(os.path.join(repo, "outputs"), os.path.join(tmp, "outputs"))  # read-only use: policies
    os.chdir(tmp)
    print(f"[mock] running in {tmp}")
    sys.argv = [mod] + args
    runpy.run_module(mod, run_name="__main__")


if __name__ == "__main__":
    main()
