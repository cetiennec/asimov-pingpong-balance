# Running the paddle policy on Albert (a real Asimov 1)

Everything here also runs without a robot: add `--sim` (simulated Albert on its gantry) or `--sim --freestand` (standing on
the floor), which use a fixed simulated camera and body lean, independent of the real robot's calibrations.

The scripts connect with menlo-sdk's `Robot()` (the saved default robot: `menlo setup`). They run on the Mac or on Albert's
own computer (a Raspberry Pi 5, see [ONBOARD.md](ONBOARD.md)): same code, same commands.

## Safety, in every script

- Albert is never left limp: on a normal end, Ctrl+C, an error, a dropped ssh session or a watchdog trip, the arm glides to
  Menlo's rest pose (4 s) and Albert is handed to Menlo's balance controller with `stand()` then `balance()` (a bare
  `balance()` after joint control is dropped by Asimov Edge 2 s later). No prompt ever waits without a setpoint keepalive.
- Under joint control Menlo's balance controller is off: the legs are held in the **centred stance** (hip -0.50, knee 0.55,
  ankle -0.21 rad; Albert's own stand pose tips over). Take over with the gantry rope **taut**; for free-standing, lower it
  until slack when asked. A lean watchdog (every 20 ms) trips to `balance()`.
- Keep the E-Stop (Asimov Manager) in reach.

## Calibrations (after a fall or a hardware change)

| what | command | good result |
|---|---|---|
| status (read-only) | `python -m albert.status` (`--mode livekit` while the Pi has the UDP state) | not faulted, joints < 60 C |
| joint directions (new robot) | `python -m albert.check_joints --supported` | each joint moves as described |
| arm offsets + flat pose | `python -m albert.teach_flat --supported` (hold the paddle flat at 3 positions) | residual tilt < 3 deg |
| head camera lens (once: checkerboard `data/checkerboard_9x6.png` held all over the image) | `python -m albert.record_frames --seconds 90 --name checkerboard`, then `uv run --with opencv-python-headless python -m albert.calibrate_lens data/<day>/checkerboard_<time>` | < 0.5 px rms, > 50 % of the image covered; refits the pose below from the saved spots |
| head camera (9 spots, stencil `calibration_spots_A4.pdf`) | `python -m albert.calibrate_camera --supported` | ~3 mm mean |
| where the camera sees the paddle | `python -m albert.aim_sweep --supported` | blade pixels at the run's neck aim |
| camera latency | `python -m albert.measure_latency --supported` | ~105 ms over Wi-Fi |
| arm dynamics | `python -m albert.sysid_arm_test --supported`, then `python -m tools.sysid_arm <log>.jsonl` | held-out error a few mrad |

The neck's zero is Menlo's (Asimov Manager); if the camera no longer sees the paddle after a fall, recalibrate it there first.

## Balancing

```bash
python -m albert.run_policy --supported --gantry --duration 60 --log logs/gantry.csv   # rope taut, Albert hanging
python -m albert.run_policy --supported --duration 60 --log logs/free.csv              # free-standing: rope procedure
python -m albert.stand_test --supported --hold 20                                       # free-standing without the ball
```

Before the ball goes on, the blade is levelled for Albert's actual lean (arm model + IMU, `--no-level-start` to skip). Put the
ball on; it starts by itself and waits again when the ball is lost. Each run writes `<log>.csv` (50 Hz), `<log>_frames.csv`
(every camera detection), `<log>.jsonl` (menlo-sdk recording), `<log>_camera.mp4` + `_camera.csv` (the raw head camera,
640x360, every frame with its time; `--camera-video ''` to skip) and `<log>_detect.mp4` (what the detector saw, with its masks).
At the end of a run on the Mac (or after `onboard.sh` for a run on the Pi), every run with a camera recording is exported to
the local LeRobot dataset `data/lerobot/asimov-pingpong-albert-camera` (`--no-lerobot` to skip). The camera frames show the
room and the people in it: they stay local.

After a run: `python -m tools.session_stats <log>.csv`, `python -m tools.tune_kf <log>.csv` (predictor on real data).

## Files

| | |
|---|---|
| `run_policy.py` | the 50 Hz balancing loop (camera → detector → Kalman predictor → policy → arm) |
| `common.py` | joint order and mapping, gains, kinematics, observation, safety (`hand_back`, `to_balance`, `TiltGuard`, stance) |
| `ball_camera.py` | ball detector ([diagram](../docs/detector.md)), camera model (lens distortion if calibrated), ray–blade intersection, calibration fit |
| `calibrate_lens.py` | the camera's lens from checkerboard frames (focal lengths, centre, distortion) |
| `stand_test.py`, `sim_freestand.py` | standing without the gantry's support: on Albert / simulated (with a gantry rope) |
| `sim_robot.py` | simulated Albert (menlo-sdk-shaped API, rendered head camera, identified arm) |
| `calibrate_camera.py`, `teach_flat.py`, `measure_latency.py`, `sysid_arm_test.py`, `aim_sweep.py`, `check_joints.py`, `status.py`, `record_frames.py`, `record_demo.py` | calibrations and checks |
| `frame_recorder.py` | the raw camera recording (a low-priority ffmpeg process; frames dropped, never the loop slowed) |
| `onboard.sh`, [ONBOARD.md](ONBOARD.md) | running on Albert's Pi from the Mac |
| `*.json` | Albert's calibrations (camera, arm offsets, latency, arm dynamics); `sim_camera.json`: the simulated camera; history in `calibration_history/` |
