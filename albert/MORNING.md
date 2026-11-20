# Albert session checklist

Run every command from the repository root (`asimov-pingpong-balance/`), prefixed with `uv run --extra albert python -m`.
Logs go in `logs/` (one name per run; nothing overwrites).

## What changed since the last session

| | last session | now |
|---|---|---|
| Policy | `outputs/albert_perturb/best.pt` | `POLICY` (below): retrained on a sim matched to the last session |
| Simulated arm | nominal gains, 60 ms delay | identified on the last session (`tools/sysid_arm.py`): stiffer, with friction, 34 ms delay; reproduces each joint's lag (70–125 ms) and amplitude |
| Ball predictor | `BallKF` with hand-picked noise | tuned on Albert's camera data (`tools/tune_kf.py`): 2 mm noise, outliers down-weighted, ball rolling 0.87 × ideal, **learns the blade's flat error online**; median error 19 → 9 mm on the hand demo |
| Training camera | clean | Albert-like: 1.5 mm jitter, up to 10 % detections 1–3 cm off, dropped frames and 0.15–0.4 s gaps, ±2.3° flat error |
| Detector | whole frame | only around where the arm model puts the blade (`--roi`, default; `--no-roi` for the old way): faster, nothing off the paddle can be taken for the ball |
| Logs | `data/2026-11-19/albert_run.csv` | + `<log>_frames.csv` (every frame: detection, accepted or not, estimate, learnt flat error) and per-run names |

`POLICY = outputs/albert_v2A/best.pt`, now `run_policy`'s default (realistic sim: see the table at the end).

## Before you power on

- Paddle holder screws, zip ties, rubber; stencil paper off; white ball; nothing white near the paddle.
- `albert.status`: battery, temperatures. Hips were above 70 °C last time: let them cool below ~55 °C first.
- Gantry rope attached. E-Stop (Asimov Manager) open and in reach for every step.

## Order: A (gantry, policy), C (gantry, arm system ID), B (rope slack, step by step)

## A. On the gantry, as last session (the comparison)

1. `albert.run_policy --supported --duration 60 --log logs/A1_v2.csv`
   Same head aim and camera calibration as last time. Put the ball on, take hands away; it starts by itself.
2. Optional A/B: `albert.run_policy --supported --duration 60 --ckpt outputs/albert_perturb/best.pt --log logs/A2_old.csv`
3. After each run:
   - `python -m tools.session_stats logs/A1_v2.csv logs/A2_old.csv` (catches, balancing time, longest, ball spread, arm activity)
   - `python -m tools.tune_kf logs/A1_v2.csv` (predictor tuned on real balancing data; tell me the numbers)
   - `python -m tools.sysid_arm logs/A1_v2.jsonl` (checks the arm model still fits)

Last session for reference: 7 catches, 37 s of 60 balancing, longest 33 s, ball 1.3 cm from centre on average.

## C. Arm system ID, still on the gantry (~5 min)

1. `albert.sysid_arm_test --supported --log logs/C1_sysid_arm.csv`
   Paddle mounted, **no ball**. Each arm joint, then all five, wiggle by about ±0.06 rad at 0.3–4 Hz for 70 s;
   every setpoint and joint state is recorded (`logs/C1_sysid_arm.jsonl`).
2. `python -m tools.sysid_arm logs/C1_sysid_arm.jsonl albert/arm_dynamics_C1.json`
   Compare it with `albert/arm_dynamics.json` (fitted on yesterday's policy run): delay, kp/kd scales, friction, held-out
   error. If clearly different, copy it over `albert/arm_dynamics.json` and retrain (`job_v2.sh` on HF, ~1 h).

## B. Without the gantry's support (one step at a time; stop at the first surprise)

Under joint control Menlo's balance controller is OFF: the legs are only held stiff. In simulation, Albert's own stand pose
falls within 2 s of any takeover (centre of mass 3 cm behind the toes); the centred stance (hip −0.50, knee 0.55, ankle
−0.21 rad) stands at ~9° lean and keeps standing while the arm moves. So: **take over only with the rope carrying part of
the weight**, then lower it.

The watchdog checks the lean every 20 ms: more than 5–6° beyond the start, more than 16°, or tipping faster than 0.8 rad/s
→ `balance()`, Menlo's walking policy takes the legs, and nothing more is sent to the joints. It is not an E-Stop.

1. Stance only: `albert.stand_test --supported --hold 20 --log logs/B1_stand.csv`
   Rope taut → Enter → legs move to the stance (3 s) → lower the gantry slowly until the rope is slack → Enter → 20 s hold →
   raise the rope → Enter → `balance()`.
   **Expect a lean of ~9° (sim 8.9°)** that stays flat. Stop if it creeps up: the ankles are softer than the model assumes
   (the one thing the sim cannot settle).
   Then `python -m tools.compare_stand logs/B1_stand.csv`: lean and how much the ankles give under Albert's weight, phase
   by phase, next to the sim's (sim: 5 mrad more once the rope is slack, both ankles together). If Albert's ankles are
   less than half as stiff as modelled, stop here: the stance must be re-searched with softer ankles first.
2. Arm moving, no ball: `albert.stand_test --supported --hold 5 --arm-replay data/2026-11-19/albert_run.csv --arm-scale 0.5 --log logs/B2_arm05.csv`
   then `--arm-scale 1.0` (`logs/B3_arm10.csv`). The arm replays last session's policy motion. Sim: lean 8–10°.
3. The policy, free-standing:
   `albert.run_policy --supported --stance -0.50 0.55 -0.21 --duration 60 --log logs/B4_free.csv`
   Same rope procedure; at the end the legs stay in the stance, raise the rope, Enter → `balance()`.
   (`--aim-camera` points the camera's own axis at the blade instead of the head's forward axis; try it only if the ball
   is detected badly in this stance.)

## If something goes wrong

- Watchdog trip: Menlo's controller has the legs. Raise the rope, then E-Stop / Asimov Manager as usual.
- Ball not detected: `logs/<run>_detect.mp4` shows the frame and masks; `--no-roi` falls back to the whole-frame search.
- A script error: everything ran end to end through the real-robot code path against a simulated robot
  (`python -m tools.mock_real [--freestand] -- albert.<script> --supported ...`), but not on Albert.

## Results in the realistic sim (gantry, 128 episodes each)

Standing still (`balance`) and with flicks / pushes (`perturb`), 128 episodes of up to 20 s each, on the sim with the identified
arm, Albert-like camera noise, outliers, dropped frames and flat error:

| policy | standing: ball kept | ball from centre | disturbed: ball kept |
|---|---|---|---|
| last session's (`albert_perturb`) | 56 % | 2.8 cm | 39 % |
| **v2A** (default) | **73 %** | **2.1 cm** | **45 %** |
| v2B (smoother arm) | 70 % | 2.2 cm | 44 % |

Full camera pipeline on the simulated Albert (60 s, ball put back rolling after each loss, 2 runs each): v2A keeps the ball
0.5–0.6 cm from the centre against 0.8–1.0 cm for last session's policy, with a calmer arm. With the identified arm, last
session's policy in sim now moves the arm about as much as on Albert (arm speed 0.07–0.09 vs 0.10 rad/s real; 0.03 with
the old sim): the sim is closer to the robot, but still kinder (it keeps the ball longer than Albert did).
