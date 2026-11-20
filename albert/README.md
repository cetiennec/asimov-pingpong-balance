# Running the paddle policy on Albert

Albert is the default robot in your menlo-sdk setup (`~/.menlo/robots.toml`, `albert.local`, hybrid connection). These
scripts use `Robot()` with no arguments, so they connect to it the way `menlo` does; no address or credential is in the code.

**Status (2026-11-19):** run on Albert on its gantry: 7 catches and 37 s of balancing in 60 s, longest 33 s. The sim was then
matched to that session (arm identified, camera noise measured, predictor tuned) and the policy retrained. **For the next
session follow [MORNING.md](MORNING.md)** (gantry comparison, then standing without the gantry's support, step by step).

## What it can do on Albert, and what it cannot

- `robot.trajectory()` puts **every** joint under position control and **switches the walking policy off** (menlo-sdk,
  2026-11). So on Albert the policy balances the ball **standing, hanging from the gantry**: stages 1 and 2 (balance,
  perturbations: push the paddle, flick the ball, shake the gantry).
- **Walking while balancing (stage 3) is not possible with the current SDK**: it would need a mode where the walking policy
  keeps the legs and the right arm follows external targets. The stage-3 policy is ready for that day (it uses a gait clock
  input that a walking controller would provide).
- The ball comes from the **head camera** (the camera works over `hybrid` or `livekit`, not `udp`).

## Before you start

- Paddle holder printed in a **dark** colour (an orange holder looks like the ball), plugged into the right wrist cup
  (ASV1_300_12B), 4 × M4 × 10 screws, handle zip-tied in the cradle, rubber strip under it. See `../hardware/`.
- A **white** ping-pong ball, good even light, nothing white near the paddle (stencil paper off).
- Albert **hanging from its gantry**, Asimov Manager open at the E-Stop. Every script asks for `--supported`.
- Install: `uv sync --extra albert` (adds menlo-sdk), then run everything from the repository root.

## Steps

```bash
# 0. the whole pipeline on the simulated Albert (no robot needed): calibration, then 20 s of balancing, with a video
uv run python -m albert.calibrate_camera --sim
uv run --extra albert python -m albert.run_policy --sim --duration 20 --video videos/albert_sim.mp4

# 1. joint check: moves each right-arm joint 0.15 rad and back, says what the hand should do. Fix ARM_SIGN if one is wrong.
uv run --extra albert python -m albert.check_joints --supported

# 2. camera calibration: the arm goes level, you put the ball on 9 marked spots of the blade (centre, +-4 cm, diagonals).
#    Mark them with the 1:1 stencil calibration_spots_A4.pdf/.png (print at 100 %, prick through, remove the paper).
#    Check albert/calibration_mask.png: the ball must be white there and nothing else; else adjust BALL_HUE in common.py.
#    Aim for a ball position error below ~3 mm.
uv run --extra albert python -m albert.calibrate_camera --supported

# 3. balance: put the ball on the centre; it starts when it sees it, stops when the ball is lost, after --duration or Ctrl+C.
uv run --extra albert python -m albert.run_policy --supported --duration 30
```

Each run writes `data/2026-11-19/albert_run.csv` (ball estimate, arm angles, actions at 50 Hz) and `data/2026-11-19/albert_run.jsonl` (every state sample and
command, menlo-sdk recording). When a script ends (normally, on Ctrl+C, on an error or a watchdog trip), the arm returns to its starting pose and Albert is
handed to Menlo's balance controller (`balance()`): never left to DAMP.

## Things to check, in order of risk

1. **Joint signs and zeros** (`check_joints.py`). Menlo's sim-to-real training uses the same MJCF, so they should match,
   but it has not been verified here.
2. **Camera latency.** The policy is trained for a ball seen at 25–30 fps with 40–120 ms latency. Over `hybrid` the video
   goes through LiveKit; if the measured latency is higher, retrain with a larger `CAM_LATENCY` in `asimov_pp/env.py`.
   A quick measure: move the paddle sharply and compare the frame timestamps with the joint-state timestamps in the logs.
3. **Gains.** `run_policy` sends the training gains (Menlo's hardware-tuned table, `common.KP/KD`) for every joint;
   `--edge-gains` lets Asimov Edge use its own table instead.
4. **Head pose.** The neck looks down-right at the paddle (`NECK_YAW = -0.50`, `NECK_PITCH = -0.34`, its lowest). The ball
   sits low in the image; if part of the blade is out of view, lower the home pose of the paddle or adjust the neck yaw.
5. **Policy.** Use the policy trained with the camera model: `--ckpt outputs/cam_perturb/best.pt` (the default).

## Files

| | |
|---|---|
| `common.py` | firmware joint order, sim ↔ firmware mapping (`ARM_SIGN`, `ARM_OFFSET`), gains, head pose, forward kinematics, observation |
| `ball_camera.py` | ball detector (searching only around the blade, `--roi`), head-camera model, ray–blade intersection, calibration fit |
| `calibrate_camera.py` | camera calibration on the paddle (Albert or `--sim`) |
| `check_joints.py` | right-arm joint direction check on Albert |
| `run_policy.py` | the 50 Hz balancing loop (Albert or `--sim`) |
| `stand_test.py` | standing without the gantry's support, legs held by joint control: rope procedure, lean watchdog → `balance()` |
| `sim_freestand.py` | simulated Albert standing on the floor (all 25 joints at Albert's gains), same API, with a gantry rope |
| `arm_dynamics.json` | the right arm identified on a real session (`tools/sysid_arm.py`), used by training and the simulated Albert |
| `sim_robot.py` | simulated Albert: the slice of the menlo-sdk API these scripts use, on the training physics, with a rendered head camera |
