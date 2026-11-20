---
license: mit
tags:
- reinforcement-learning
- robotics
- humanoid
- mujoco
- ppo
- sim-to-real
library_name: pytorch
pipeline_tag: reinforcement-learning
---

# Asimov 1: ping-pong ball balancing on a paddle

A humanoid ([Asimov 1](https://github.com/menloresearch/asimov-1) by Menlo Research, 25 DoF, no gripper) balances a ping-pong
ball on a table-tennis paddle held in its right wrist, seen by its own head camera. Trained with PPO in MuJoCo, run on a
real Asimov 1 ("Albert"), on its gantry and **standing on its own feet without the gantry's support**. Inspired by
[zecice/so101-pingpong-balance](https://github.com/zecice/so101-pingpong-balance) (SO-101 arm).

| On Albert | balancing in 60 s | longest catch |
|---|---|---|
| 2026-11-19, gantry, first Kalman policy | 11.5 s in 22 s | — |
| 2026-11-19, gantry, `albert_perturb` (arm lag in training) | 37 s | 33 s |
| 2026-11-20, gantry, `albert_v2A` (sim matched to the robot) | 43.5 s | 43 s (ended by a gantry move) |
| **2026-11-20, free-standing, rope slack, `albert_v2A`** | **57 s** | **46 s** |

<video src="https://huggingface.co/cetiennec/asimov-pingpong-balance/resolve/main/videos/5_albert_freestand_sim.mp4" controls></video>

## How it works

```
head camera (1280x720, 22 fps) --> ball detector (white ball on the red blade, searched only around the blade)
                                         |  ray from the calibrated camera, intersected with the blade plane (arm model + IMU)
joints + IMU (200 Hz) ----> arm model --> Kalman predictor: replays each frame at its capture time (~105 ms old), predicts
                                         |  the ball NOW from the blade tilt, down-weights outliers, learns the blade's flat error
                                         v
                              policy (2 x 256 MLP) --> right-arm joint targets, 50 Hz --> Albert's motors
safety: lean watchdog every 20 ms; any error / Ctrl+C / ssh drop --> arm glides home --> stand() -> balance() (never DAMP)
```

- **Observation (40):** ball position / velocity on the blade (predictor), gravity in the blade frame, gyro, accelerometer,
  arm angles and velocities, gait clock (zeros when standing), last 3 actions. **Action (5):** right-arm joint increments
  (x 0.03 rad) at 50 Hz.
- **Sim matched to the robot:** the arm is identified on a real session (`tools/sysid_arm.py`: gains, damping, friction,
  34 ms delay); the camera noise is Albert's (1.5 mm jitter, up to 10 % detections 1-3 cm off, dropped frames, flat error);
  the body leans anywhere between the gantry's 7 deg and the free-standing 12 deg.
- **Standing without the gantry:** under joint control Menlo's balance controller is off, so the legs are held stiff in a
  *centred stance* (hip -0.50, knee 0.55, ankle -0.21 rad) found in a free-standing simulation (`albert/sim_freestand.py`):
  Albert's own stand pose puts the centre of mass at the toes and tips over. Take over with the gantry rope taut, then slack.

## Checkpoints (`outputs/`)

| folder | what |
|---|---|
| **`albert_v3_final`** | **default.** v2A fine-tuned across the gantry and free-standing leans (sim, free-standing: 70 % vs 65 %, ball 2.1 vs 2.7 cm off-centre) |
| `albert_v2A` | trained on the sim matched to Albert (identified arm, tuned predictor, realistic camera); ran on Albert: 57 s / 60 free-standing |
| `albert_v2B` | same, smoother-arm reward |
| `albert_perturb`, `albert_balance` | first Albert policies (Kalman predictor, 40-100 ms arm delay) |
| `albert_move` | walking stage (sim only, weak; Albert cannot walk with an external arm controller yet) |
| `cam_*`, `balance` / `perturb` / `move` | generic Asimov 1, head-camera model or ideal ball state, stages 1-3 (sim: 92-100 %) |
| `albert_*_v1..v3`, `*_balance`, `albert_v3` | intermediate runs |

Each folder: `best.pt`, `last.pt`, `args.json` (exact training arguments), `log.csv`, `train.log`.

```python
from asimov_pp.ppo import load_policy
policy = load_policy("outputs/albert_v3_final/best.pt")   # numpy inference, ~7 us per call
action = policy(obs)                                       # obs: 40 floats -> 5 floats in [-1, 1]
```

## Repository

| | |
|---|---|
| `asimov_pp/` | MuJoCo model builder, environment (curriculum, domain randomization, camera model), Kalman predictor, PPO |
| `albert/` | the real robot through menlo-sdk: `run_policy.py`, calibrations (camera, arm offsets, latency), free-standing (`stand_test.py`, `sim_freestand.py`), simulated robots, safety (`common.py`). Guides: `albert/README.md`, `albert/ONBOARD.md` |
| `tools/` | arm system identification, predictor tuning, session statistics, robot-matched evaluation, a mock robot that runs the real-robot code paths |
| `hardware/` | paddle holder (`paddle_holder.py` → `.stl`; v5 follows the flared handle, has a flat print face and no sharp inside corners; v4 kept) |
| `data/` | run logs of the sessions on Albert (CSV at 50 Hz: ball estimate, arm, actions; `*_frames.csv`: every camera detection). As a LeRobot dataset: [cetiennec/asimov-pingpong-albert](https://huggingface.co/datasets/cetiennec/asimov-pingpong-albert) (`tools/export_lerobot.py`) |
| `videos/` | simulation videos |
| `train_*.sh` | training commands |

Not included: Menlo's Asimov 1 MJCF and meshes (expected in `../supermimic_ws/asimov-1/sim-model`), the wrist-cup mesh made
from Menlo's CAD (`assets/ASV1_300_12B_wristframe.stl`), raw robot recordings and camera images.

## Run

```bash
uv sync --extra albert
uv run python -m albert.run_policy --sim --hand --duration 30          # the whole pipeline on a simulated Albert
uv run python -m albert.run_policy --sim --freestand --truth-ball      # standing without support, simulated
uv run python -m tools.eval_albert outputs/albert_v3_final/best.pt     # robot-matched sim evaluation
```

On the robot: `albert/README.md` (calibrations, gantry, free-standing), `albert/ONBOARD.md` (running on Albert's Pi).

## Limitations

- No walking with the paddle on the robot: menlo-sdk's joint control switches Menlo's walking policy off.
- The camera pose, arm offsets and neck zero shift when the robot falls; a fiducial marker on the paddle is planned to remove
  most calibrations.
- The simulated body is prescribed (the arm's reaction on the torso is ignored), except in `albert/sim_freestand.py`.
