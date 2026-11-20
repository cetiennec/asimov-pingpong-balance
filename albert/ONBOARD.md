# Running on Albert's Pi

The control loop (camera → ball detector → Kalman predictor → policy → arm targets at 50 Hz) runs on Albert's own
computer (Raspberry Pi 5, 4 × Cortex-A76) instead of the Mac: no Wi-Fi in the loop, so a link drop cannot make Albert fall.
The Mac only starts runs, watches, and keeps the logs.

## One-time setup (done 2026-11-20)

- Key login from the Mac: `ssh albert@albert.local` (key installed with `ssh-copy-id`).
- On the Pi: `~/asimov-pingpong-balance/.venv` (numpy, scipy, MuJoCo 4.0.1, CPU torch, menlo-sdk, imageio-ffmpeg), the robot
  model in `~/supermimic_ws/asimov-1/sim-model`, and its own menlo-sdk profile (`menlo setup`, manager `http://127.0.0.1`).
- **Asimov Manager → Asimov Edge parameters: `udp-state-host = 127.0.0.1`** (udp-control on). Edge sends joint states to
  one address only; with the Pi there, Mac-side scripts that drive the joints stop working (they need the UDP state).

## Use

```bash
albert/onboard.sh albert.run_policy --supported --gantry --duration 60 --log logs/P5.csv   # on the gantry
albert/onboard.sh albert.run_policy --supported --duration 60 --log logs/P6.csv            # free-standing (rope procedure)
albert/onboard.sh albert.calibrate_camera --supported
albert/onboard.sh albert.measure_latency --supported
python -m albert.status --mode livekit                                                      # watch from the Mac
```

`onboard.sh` syncs the code (and the calibrations, newer file wins, both ways), runs the script on the Pi over `ssh -t`
(prompts work), then copies the Pi's `logs/` to `logs/onboard/` and, after a `run_policy`, rebuilds the local LeRobot dataset
with the camera on the Mac.

Safety is the same as on the Mac, plus: a dropped ssh session ends the script like Ctrl+C (SIGHUP is handled): the arm glides
home and Albert is handed to Menlo's balance controller (`stand()` → `balance()`). A hard crash of the Python process (kill -9,
out of memory) is the one case still left to Asimov Edge, which damps 2 s after the last setpoint.

## Performance on the Pi

| per 50 Hz step | before (2026-11-20 morning) | now |
|---|---|---|
| kinematics + predictor + policy | 5.3 ms median, 11.9 ms p99 (PyTorch thread pool) | 0.56 ms, 1.7 ms p99 (numpy policy, 1 maths thread) |
| ball detector (blade region) | 1.6 ms | 1.4 ms (whole frame: 17 ms) |
| loop rate measured on Albert | 33–37 Hz, 9–13 % of steps > 40 ms | to be measured on the next run |

## Still to do on the robot

- Measure the camera latency onboard (`albert.measure_latency`): it is 105 ms over Wi-Fi to the Mac; onboard it should be
  lower, and the predictor uses it.
- Check the loop rate of the first run with the numpy policy (`python -m tools.session_stats` on the copied log).
- Check the camera recording's cost on the Pi: the run prints `camera recording: ... N dropped` (H.264 ultrafast, one
  thread, niced). If the loop rate drops, `--camera-video ''` turns it off.
