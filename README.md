---
license: mit
tags:
- robotics
- humanoid
- reinforcement-learning
- sim-to-real
- mujoco
- ppo
- asimov-1
library_name: pytorch
pipeline_tag: reinforcement-learning
datasets:
- cetiennec/asimov-pingpong-albert
---

# Ping-pong ball balancing for the Asimov 1

Your [Asimov 1](https://github.com/menloresearch/asimov-1) keeps a ping-pong ball on a table-tennis paddle held in its right
hand, using only its head camera. Print a holder, calibrate, run one command.

<video src="https://huggingface.co/cetiennec/asimov-pingpong-balance/resolve/main/videos/albert_head_camera.mp4" controls></video>

*What the robot sees: Albert, an Asimov 1, balancing with `albert_v3_final` (57 s of a 60 s run). Faces blurred.*

**Try it in your browser first**: [the simulator](https://huggingface.co/spaces/cetiennec/asimov-pingpong-balance) ·
**Real runs** (head camera, joints, actions): [the dataset](https://huggingface.co/datasets/cetiennec/asimov-pingpong-albert)

## What you need

- An **Asimov 1** with menlo-sdk set up (`menlo setup`), and a **gantry** or a rope to hang it from for the first runs.
- A **table-tennis paddle** with red rubber and a **white 40 mm ping-pong ball**.
- The **paddle holder**, 3D-printed from [`hardware/paddle_holder.stl`](hardware/paddle_holder.stl): PETG, PLA+ or PA12,
  lying on its flat face, 6 walls or 100 % infill. It plugs into the right wrist cup, held by 4 x M4 x 10 screws through the
  cup's threads; the paddle's handle is held with 3 zip ties (4.8 mm) over a 1-2 mm rubber strip.
- A computer with Python 3.12 and [uv](https://docs.astral.sh/uv/) (macOS or Linux), or the robot's own computer.

## Quick start (5 minutes, no robot)

```bash
hf download cetiennec/asimov-pingpong-balance --local-dir asimov-pingpong && cd asimov-pingpong   # pip install huggingface_hub
uv sync --extra albert
tools/get_asimov_model.sh                                  # Menlo's robot model, next to this folder
uv run python -m albert.run_policy --sim --hand --duration 30
```

The last command runs the whole pipeline on a simulated Albert: rendered head camera, ball detector, ball predictor, policy,
arm. `--hand` puts the ball back when it falls.

## On your robot

1. **Mount the paddle.** Print the holder, plug it into the right wrist cup, screw it in, zip-tie the paddle, red side up.
2. **Calibrate** (about 30 minutes the first time, 5 after a fall): flat pose, camera, latency.
   Step by step: **[docs/calibration.md](docs/calibration.md)**.
3. **First run, hanging from the gantry** (rope taut, E-Stop in reach):
   ```bash
   uv run python -m albert.run_policy --supported --gantry --duration 60 --log logs/first.csv
   ```
   The arm goes to its starting pose, the head looks at the paddle and the blade is levelled. **Put the ball on the blade**:
   balancing starts by itself and waits again whenever the ball falls. At the end the robot is handed back to Menlo's balance
   controller.
4. **Standing on its own.** Leave out `--gantry`: the legs are held in a stance that keeps the robot upright under joint
   control. Start with the rope taut and lower it until slack when the script asks.
5. **Run it on the robot's computer** for steadier timing: `ALBERT_PI=user@robot.local albert/onboard.sh albert.run_policy
   --supported --gantry --duration 60 --log logs/P1.csv` copies the code over, runs it there and brings the logs back
   ([`albert/ONBOARD.md`](albert/ONBOARD.md)).

Each run writes a log (`logs/<name>.csv`, 50 Hz), the camera detections, the robot recording and the raw head camera.
`uv run python -m tools.session_stats logs/first.csv` sums it up.

**Safety, built in.** The robot is never left limp: on a normal end, Ctrl+C, an error, a dropped ssh session or a lean beyond
the watchdog's limit, the arm glides to its rest pose and the robot is handed to Menlo's balance controller. Scripts that
wait for you keep the robot's setpoints alive.

## Controllers

| | |
|---|---|
| **`albert_v3_final`** (default) | PPO policy trained in a simulation matched to Albert: its measured arm, its camera's noise and latency, leaning on the gantry or standing, with pushes, flicks and shoves |
| `albert_v4` | the same, fine-tuned on Albert's arm as measured on 2026-11-22: `--ckpt outputs/albert_v4/best.pt` |
| MPC | a model-predictive controller, no training: often smoother on a real arm, worth a try if the paddle oscillates. `--controller mpc` ([how it works](docs/mpc.md)) |

On Albert (60 s runs, the ball put back by hand after a fall):

| date | controller | balancing in 60 s | longest catch |
|---|---|---|---|
| 2026-11-20 | `albert_v2A`, standing on its own | 57 s | 46 s |
| 2026-11-22 | `albert_v3_final` | 57.2 s | 56.3 s |
| 2026-11-22 | `albert_v4` | 45.5 s | 31.3 s |
| 2026-11-23 | MPC, on the gantry | 56.9 s | 55.6 s |
| 2026-11-23 | MPC | 59.8 s | 59.8 s |

Results depend a lot on the calibration of the day: the same policy kept the ball 6 to 57 s on the same afternoon.

## Without a robot

```bash
uv run python -m albert.run_policy --sim --hand --duration 30               # the full pipeline, simulated
uv run python -m albert.run_policy --sim --freestand --truth-ball           # standing without support
uv run python -m tools.eval_albert outputs/albert_v3_final/best.pt mpc      # success rates in the robot-matched sim
```

Use a policy in your own code:

```python
from asimov_pp.ppo import load_policy
policy = load_policy("outputs/albert_v3_final/best.pt")   # numpy, ~7 us per call
action = policy(obs)   # 40 observations (ball on the blade, gravity, gyro, arm joints, last actions) -> 5 arm joint steps
```

## How it works

```
head camera (~22 fps) --> ball detector: white ball on the red blade, in the blade's region only
                             |  ray from the calibrated camera, intersected with the blade (arm model + IMU)
joints + IMU (200 Hz) ----> Kalman predictor: each frame replayed at its capture time (~105 ms old);
                             |  predicts where the ball is now from the blade's tilt and motion
                             v
               policy or MPC --> 5 right-arm joint targets, 50 Hz --> motors
```

The policies were trained with PPO in MuJoCo on a curriculum (balance, then disturbances), in a simulation matched to the
real robot: the arm's delay, gains and friction identified on Albert, the camera's jitter, outliers and dropped frames,
and the body's lean on the gantry and standing. To retrain, see `train_albert.sh` (CPU only, about 1 h on 32 cores).

## Files

| | |
|---|---|
| `albert/` | everything that runs on the robot: `run_policy.py`, the calibrations (`*.json` are Albert's), `onboard.sh`, safety |
| `asimov_pp/` | the simulation: MuJoCo model, environment, ball predictor, PPO |
| `outputs/` | the policies: `best.pt`, the training arguments, logs |
| `hardware/` | the paddle holder (`paddle_holder.stl`, and `paddle_holder.py` to change it) |
| `docs/` | [calibration](docs/calibration.md), [ball detector](docs/detector.md), [MPC](docs/mpc.md) |
| `tools/` | robot-matched evaluation, session statistics, arm identification, dataset export |

Menlo's robot model is not included: `tools/get_asimov_model.sh` fetches it from
[menloresearch/asimov-1](https://github.com/menloresearch/asimov-1).

## Limitations

- No walking while balancing: menlo-sdk's joint control turns Menlo's walking policy off.
- A fall shifts the camera: recalibrate it (step 5 of the calibration) after any fall.
- Trained on one robot. If yours differs a lot, identify its arm and retrain (calibration step 8), or use the MPC.

Inspired by [zecice/so101-pingpong-balance](https://github.com/zecice/so101-pingpong-balance) (SO-101 arm). MIT license.
