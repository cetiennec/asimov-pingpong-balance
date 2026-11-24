# Running on the robot's own computer (optional)

Everything runs from your Mac over Wi-Fi. For steadier timing, the same control loop (camera → ball detector → ball
predictor → policy → arm targets at 50 Hz) can run on the Asimov 1's own computer (a Raspberry Pi 5), started from the Mac:
no Wi-Fi in the loop, so a dropped link cannot stall the arm. The Mac starts runs, watches and keeps the logs.

On Albert this brought the loop's computation from ~5 ms to ~0.6 ms per step (numpy policy, one maths thread).

## One-time setup

1. **Key login from the Mac** to the robot's computer: `ssh-copy-id <user>@<robot>.local`, then
   `export ALBERT_PI=<user>@<robot>.local` (the scripts below use it; without it they try `albert@albert.local`).
2. **Install** the code, Menlo's robot model, the policies and a Python environment on the robot's computer, from the Mac:
   ```bash
   albert/setup_pi.sh
   ```
   It ends with `Pi ready`.
3. **menlo-sdk profile on the robot's computer**: `ssh -t $ALBERT_PI '~/asimov-pingpong-balance/.venv/bin/menlo setup'`,
   manager `http://127.0.0.1`.
4. **Asimov Manager → Asimov Edge parameters: `udp-state-host = 127.0.0.1`.** Edge streams the joint states to one address
   only: with the robot's computer there, scripts run directly on the Mac can no longer drive the joints (set it back to your
   Mac's IP to run from the Mac again).

## Use

```bash
albert/onboard.sh albert.run_policy --supported --gantry --duration 60 --log logs/P1.csv   # on the gantry
albert/onboard.sh albert.run_policy --supported --duration 60 --log logs/P2.csv            # free-standing
albert/onboard.sh albert.calibrate_camera --supported                                      # any calibration script
python -m albert.status --mode livekit                                                     # watch from the Mac
```

`onboard.sh` syncs the code and the calibrations (both ways, the newer file wins), runs the script on the robot over `ssh -t`
(prompts work), then copies the robot's `logs/` to `logs/onboard/` on the Mac.

Safety is the same as on the Mac, plus: a dropped ssh session ends the script like Ctrl+C (the arm glides home and the robot
is handed to Menlo's balance controller). A hard crash of the Python process (kill -9, out of memory) is the one case left to
Asimov Edge, which damps the joints 2 s after the last setpoint: keep the gantry rope taut or the E-Stop in reach.

Measure the camera latency where you run (`albert/onboard.sh albert.measure_latency --supported`): it is lower on the robot
than over Wi-Fi, and the ball predictor uses it.
