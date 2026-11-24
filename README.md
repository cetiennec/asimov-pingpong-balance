---
license: mit
tags:
- reinforcement-learning
- robotics
- humanoid
- mujoco
- ppo
- model-predictive-control
- sim-to-real
- asimov-1
library_name: pytorch
pipeline_tag: reinforcement-learning
datasets:
- cetiennec/asimov-pingpong-albert
---

# Asimov 1: balancing a ping-pong ball on a paddle

An [Asimov 1](https://github.com/menloresearch/asimov-1) humanoid (Menlo Research, 25 DoF, no gripper) balances a
ping-pong ball on a table-tennis paddle held in its right wrist, seen only by its own head camera. Everything needed to run
it on **your** Asimov 1 is here: the code that runs on the robot, two PPO policies trained in MuJoCo, a model-predictive
controller that needs no training, the paddle holder to 3D-print, and the calibrations. Developed on one robot, "Albert".

- **Dataset**: Albert's real runs, head camera included (faces blurred):
  [cetiennec/asimov-pingpong-albert](https://huggingface.co/datasets/cetiennec/asimov-pingpong-albert)
- **Try it in the browser**: [spaces/cetiennec/asimov-pingpong-balance](https://huggingface.co/spaces/cetiennec/asimov-pingpong-balance)

<video src="https://huggingface.co/cetiennec/asimov-pingpong-balance/resolve/main/videos/5_albert_freestand_sim.mp4" controls></video>

## On Albert

60 s runs (the ball put back by hand when it falls; the controller waits for it). Balancing = time with the ball on the
blade under control.

| date | controller | run | balancing in 60 s | longest |
|---|---|---|---|---|
| 2026-11-20 | `albert_v2A` | free-standing, gantry rope slack | 57 s | 46 s |
| 2026-11-22 | `albert_v3_final` | G4_v3 | 57.2 s | 56.3 s |
| 2026-11-22 | `albert_v4` | G5_v4 | 45.5 s | 31.3 s |
| 2026-11-23 | MPC | M1_mpc, gantry | 56.9 s | 55.6 s |
| 2026-11-23 | MPC | F1_mpc_2 | 59.8 s | 59.8 s |

Every run with the head camera is in the dataset; runs vary a lot with the calibration of the day (see Limitations).

## Controllers

| | what | use |
|---|---|---|
| **`outputs/albert_v3_final`** | PPO, 2 x 256 MLP, trained on a sim matched to Albert (identified arm, real camera noise and latency), across the gantry's and the free-standing lean, with pushes, ball flicks and paddle shoves | default: `--ckpt outputs/albert_v3_final/best.pt` |
| `outputs/albert_v4` | v3_final fine-tuned on the arm as identified on 2026-11-22 | `--ckpt outputs/albert_v4/best.pt` |
| `outputs/albert_v2A` | the previous policy (57 s free-standing on 2026-11-20); `train_albert.sh` starts from it | `--ckpt outputs/albert_v2A/best.pt` |
| MPC ([`albert/mpc.py`](albert/mpc.py)) | per blade axis: ball rolling on a tilted and translating blade, arm lag and command delay; a box-constrained QP every 20 ms (~0.3 ms, numpy). No training; smoother than the policies on Albert. [`docs/mpc.md`](docs/mpc.md) | `--controller mpc` |

All three take the same 40-value observation and give 5 right-arm joint-target increments (x 0.03 rad) at 50 Hz.

```python
from asimov_pp.ppo import load_policy
policy = load_policy("outputs/albert_v3_final/best.pt")   # numpy inference, ~7 us per call
action = policy(obs)                                       # obs: 40 floats -> 5 floats in [-1, 1]
```

## Run it on your Asimov 1

**You need**: an Asimov 1 with menlo-sdk set up (`menlo setup`), a gantry or a rope to hang it from (strongly advised for the
first runs), a table-tennis paddle with red rubber, a white 40 mm ping-pong ball, and the paddle holder printed from
[`hardware/paddle_holder.stl`](hardware/paddle_holder.stl) (bolted to the right wrist cup). Keep the E-Stop in reach.

```bash
uv sync --extra albert
tools/get_asimov_model.sh                  # Menlo's MJCF and meshes, next to this folder (used for the arm model)
```

1. **Calibrate your robot.** The `albert/*.json` files are Albert's: yours will differ. Joint directions, arm offsets and
   flat pose, head-camera lens and pose, camera latency, arm dynamics: the commands and what a good result looks like are in
   [`albert/README.md`](albert/README.md#calibrations-after-a-fall-or-a-hardware-change). Redo the camera ones after a fall.
2. **First run, hanging from the gantry** (rope taut):
   ```bash
   python -m albert.run_policy --supported --gantry --duration 60 --log logs/first.csv
   python -m albert.run_policy --supported --gantry --duration 60 --controller mpc --log logs/first_mpc.csv
   ```
   The arm goes to its home pose and the blade is levelled for the robot's actual lean; put the ball on the blade, it starts
   by itself and waits again when the ball falls.
3. **Free-standing** (no `--gantry`): the legs are held stiff in a centred stance (Menlo's own stand pose tips over under
   joint control); take over with the rope taut, lower it until slack when asked.
4. **On the robot's own computer** (better timing than over Wi-Fi): `ALBERT_PI=user@host albert/onboard.sh albert.run_policy
   --supported --gantry --duration 60 --log logs/P1.csv` syncs the code, runs it over ssh and copies the logs back
   ([`albert/ONBOARD.md`](albert/ONBOARD.md)).

**Safety, in every script**: the robot is never left limp. On a normal end, Ctrl+C, an error, a dropped ssh session or the
lean watchdog (every 20 ms), the arm glides to Menlo's rest pose and the robot is handed back to Menlo's balance controller
(`stand()` then `balance()`). No prompt waits without keeping the setpoints alive.

## Without a robot

```bash
python -m albert.run_policy --sim --hand --duration 30                  # the whole pipeline on a simulated Albert (camera rendered)
python -m albert.run_policy --sim --freestand --truth-ball              # standing without support, simulated
python -m tools.eval_albert outputs/albert_v3_final/best.pt mpc         # robot-matched evaluation, policy vs MPC
```

## How it works

```
head camera (1280x720, ~22 fps) --> ball detector (white ball on the red blade, searched only around the blade)
                                         |  ray from the calibrated camera, intersected with the blade plane (arm model + IMU)
joints + IMU (200 Hz) ----> arm model --> Kalman predictor: replays each frame at its capture time (~105 ms old), predicts
                                         |  the ball NOW from the blade's tilt and motion, down-weights outliers
                                         v
                    policy or MPC --> right-arm joint targets, 50 Hz --> the robot's motors
```

- **Ball detector**: colour thresholds and blob tests, no neural network ([`docs/detector.md`](docs/detector.md)).
- **Observation (40)**: ball position and velocity on the blade (predictor), gravity in the blade frame, gyro,
  accelerometer, arm angles and velocities, gait clock (zeros when standing), last 3 actions.
- **Sim matched to the robot**: the arm identified on a real session (`tools/sysid_arm.py`: gains, damping, friction,
  delay), the camera's real noise (jitter, outliers, dropped frames, flat error), the body's lean anywhere between the
  gantry's 7 deg and free-standing's 12 deg.
- **Training**: `train_albert.sh`, the recipe of `albert_v3_final` (warm-started from `outputs/albert_v2A`; needs the MJCF
  above; CPU only, ~1 h on 32 cores).

## Repository

| | |
|---|---|
| `albert/` | the real robot through menlo-sdk: `run_policy.py`, `mpc.py`, calibrations, free-standing (`stand_test.py`, `sim_freestand.py`), simulated robot, safety (`common.py`). Guides: `albert/README.md`, `albert/ONBOARD.md` |
| `asimov_pp/` | MuJoCo model builder, environment (curriculum, domain randomization, camera model), Kalman predictor, PPO |
| `tools/` | arm identification, predictor tuning, session statistics, robot-matched evaluation, LeRobot export with face blur, a mock robot that runs the real-robot code paths |
| `hardware/` | paddle holder (`paddle_holder.py` → `.stl`) |
| `outputs/` | the checkpoints: `best.pt`, `args.json` (exact training arguments), `log.csv`, `train.log` |
| `docs/` | ball detector, MPC |

Not included: Menlo's Asimov 1 MJCF and meshes (`tools/get_asimov_model.sh` fetches them).

## Limitations

- No walking with the paddle on the robot: menlo-sdk's joint control switches Menlo's walking policy off for every joint.
- The camera pose, arm offsets and neck zero shift when the robot falls; recalibrate the camera after any fall.
- Results depend on the calibrations: the same policy kept the ball 6-57 s in 60 s on Albert on the same day.
- The simulated body is prescribed (the arm's reaction on the torso is ignored), except in `albert/sim_freestand.py`.

Inspired by [zecice/so101-pingpong-balance](https://github.com/zecice/so101-pingpong-balance) (SO-101 arm).
