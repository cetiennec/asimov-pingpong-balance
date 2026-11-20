# Bounce stage (simulation only)

Albert keeps the ping-pong ball **bouncing** on the paddle instead of balancing it. Trained in the same robot-matched MuJoCo
sim as `albert_v3_final`, warm-started from it. Nothing here has run on the real robot.

## What changed

| | balance stage | bounce stage |
|---|---|---|
| ball / blade contact | soft, critically damped (`solref` 0.004, 1): a dropped ball does not rebound | direct spring-damper (`solref` -1e5, -25, blade has priority): **restitution 0.89 ± 0.02** over drops of 1-30 cm, contact 9-10 ms, 1.5-7 mm deep (`tests/test_ball_bounce.py`); randomized 0.83-0.90 per episode |
| air | none | quadratic drag (2.7 g, 40 mm ball, terminal speed 8.4 m/s), ±20 % |
| camera | ray through the ball's image centre intersected with the blade plane (ball assumed on the blade) | ball centre in **3D**: pixel centre gives the ray, apparent radius gives the distance (f R / r). Albert's calibrated head camera (1280 x 720, f = 761 px, at the balancing neck pose), 0.44 m from the blade. Calibration error per episode (±0.6°, 3 mm), 2.6 px centre jitter (= the 1.5 mm measured on Albert), 1.2 px radius noise (assumed), motion blur (10 ms exposure), outliers, dropped frames and gaps as in the balance stage. Frames where the ball is out of the image are lost (above ~20 cm over the blade) |
| predictor | `BallKF`: ball on the blade plane, rolls with the tilt | `BallKF3D`: 3D, hybrid: free flight (gravity + drag), **impact** with the blade (normal velocity relative to the blade reversed x restitution, uncertain), or riding on the blade and rolling; the blade pose and velocity come from the arm model, so a hit is in the prediction. Delayed frames replayed at their capture time, gating, Huber, restart from two detections. Median error of the ball *now* (robot-matched camera, paddle still, ball bouncing then rolling off): height 15 mm, normal speed 0.11 m/s, in-plane 10-13 mm (`tests/test_kf3.py`) |
| observation | 40 | 43: + height above the blade / 0.1 m, normal speed relative to the blade / 2 m/s, **target apex** / 0.1 m (inputs 25:28, before the action history; zero weights when warm-starting) |
| reward | ball centred, slow | ball centred (0.4) + its *energy height* E = h + v²/2g (the apex of the current flight) near the target apex (1.2 x a tent: 0 at rest, 1 at the target, 0 at twice the target; halved off-centre) + a bonus at each apex near the target (2.0, Gaussian ±25 %) − sideways ball speed − jerk − distance from the home pose |
| termination | ball off the blade, or > 10 cm above | ball off the disc near the blade plane, > 3 cm under it, > 40 cm above, or **at rest on the blade for 0.5 s** (−1) |

## Curriculum

Straight from the balancing policy to the robot-matched bounce sim, PPO stalls: the policy first learns to catch the ball and
balance it (holding the ball still pays), then, once a ball at rest for 0.5 s ends the episode, it keeps small bounces going
for ~2 s and loses the ball sideways (`outputs/diag_bounce_direct`: 2 % of 10 s episodes after 10 M steps). With the ideal
ball state and no command delay the same task is learnt in 2 M steps (`outputs/diag_bounce_truth`: 89 %). The perception
(camera rate, ~105 ms latency, noise) and the command delay are what make it hard, so the curriculum adds them in steps
(`train_bounce.sh`):

| stage | ball seen by | command delay | targets (apex above the blade) | from |
|---|---|---|---|---|
| `albert_bounce_a` | ideal state | 0 | 3-8 cm | `albert_v3_final` |
| `albert_bounce_b` | head camera + `BallKF3D`, 25-30 fps, 30-70 ms, clean | 0-20 ms | 3-8 cm | a |
| `albert_bounce_c` | Albert's camera: 18-30 fps, 70-160 ms, outliers, lost frames, flat error | 20-80 ms | 3-10 cm | b |
| **`albert_bounce`** | as c | as c | 4-15 cm | c |

Two reward details mattered: the energy-height term is a tent (0 for a ball at rest, 1 at the target, 0 at twice the target)
rather than a narrow Gaussian, so there is a gradient from balancing to bouncing; and a ball at rest for 0.5 s ends the episode.

## What the real robot would need

- **Camera field of view.** At the balancing neck pose (yaw -0.50, pitch -0.46) the ball leaves the top of the image about
  20 cm above the blade: apexes are limited to ~15 cm, or the neck must look higher (pitch up to +0.40 rad) and re-calibrate.
- **Detector region.** `albert/ball_camera.py` searches for the ball only on the red blade (grown by a ball radius): a ball
  in flight is outside that region. The search region must extend over the column above the blade (the predicted ball
  position from `BallKF3D` is the natural centre), and the red-ring test must be skipped for a ball seen against the room.
- **Ball distance from its apparent radius.** The sim assumes 1.2 px of radius noise (35 px radius at 0.44 m: 1.5 cm of
  distance noise). Not yet measured: log the detector's radius on Albert (ball resting on the blade at known distance)
  and set `CAM_RADIUS_PX` from it. Motion blur matters: at 1.5 m/s the ball moves 25 px during a 10 ms exposure; a short
  exposure (manual, <= 5 ms, with more light) helps both the centre and the radius.
- **Camera rate and latency.** ~22 fps gives 6 frames per 10 cm bounce (0.29 s flight); the 105 ms latency is a third of it.
  The predictor bridges it (it knows the paddle's hits), but imperfectly: in sim, 30 fps and 40-60 ms (a lower-resolution
  stream, or detection on the robot's Pi) doubles the time kept for 12 cm bounces (see results, `fast`). It is worth doing,
  but the predictor's impact model matters more.
- **Arm speed.** Small bounces need little: to keep a 10 cm bounce (impact 1.4 m/s) with restitution 0.87 the blade must meet
  the ball at (1-e)/(1+e) x 1.4 = 0.1 m/s upward. The policy's command rate limit (0.03 rad per 20 ms = 1.5 rad/s per joint)
  is the binding limit, not the motors (see the peak joint and blade speeds in the results table: what Albert's arm must
  track, with its identified 34 ms lag).
- **Paddle rubber.** The sim's restitution is 0.83-0.90; drop the ball on the real paddle (30 cm) and check it rebounds to
  21-24 cm, or adjust `BOUNCE_DAMPING`.
- **Flat error.** A bounce amplifies a tilt error: 1 deg sends a ball hit at 1 m/s sideways at ~3 cm/s per bounce. The
  arm model's flat error (`--tilt-bias 0.04`, ~2 deg) is in the training; the balance predictor's online flat estimate is
  not in `BallKF3D` yet.
- **Runner.** `albert/run_policy.py` needs the 43-input observation (`ObsBuilder` with `BallKF3D`, the blade velocity from
  the joint velocities, the target apex), `detection_to_3d` from the detector's (x, y, radius), and the same safety
  (watchdog, glide home, stand() then balance(), never damp). Not done: nothing here has run on Albert.

## Results (simulation, `tools/eval_bounce.py`)

`outputs/albert_bounce/best.pt`, 32 episodes of 20 s per row, robot-matched sim (identified arm with its 34 ms lag + 20-80 ms
command delay, Albert's lean on the gantry (7 deg) or free-standing (12 deg)). The episode ends when the ball leaves the blade
or rests on it for 0.5 s. *Kept*: mean time the ball is kept bouncing. *CV*: apex std / mean (regularity). The first apex
(the drop) is excluded.

**Albert's head camera** (18-30 fps, 70-160 ms latency, Albert-like noise, outliers, lost frames, flat error):

| lean | target apex | kept (of 20 s) | full 20 s | bounces/s | apex | CV | apexes within ±25 % | ball from centre |
|---|---|---|---|---|---|---|---|---|
| gantry | 5 cm | 4.0 s | 3 % | 4.1 | 5.8 ± 2.1 cm | 0.37 | 52 % | 2.9 cm |
| gantry | 8 cm | 3.4 s | 0 % | 3.7 | 7.3 ± 2.6 cm | 0.36 | 69 % | 3.4 cm |
| gantry | 12 cm | 3.5 s | 3 % | 3.3 | 9.1 ± 3.1 cm | 0.35 | 59 % | 3.4 cm |
| free-standing | 5 cm | 3.7 s | 0 % | 4.3 | 5.7 ± 2.3 cm | 0.41 | 41 % | 3.3 cm |
| free-standing | 8 cm | 3.9 s | 6 % | 3.9 | 6.6 ± 2.5 cm | 0.38 | 63 % | 3.3 cm |
| free-standing | 12 cm | 3.3 s | 6 % | 3.1 | 9.6 ± 2.6 cm | 0.27 | 65 % | 3.9 cm |

So with today's camera the policy keeps the ball bouncing for **3-4 s, 12-15 bounces**, at roughly the asked height (apexes
within ±25 % of the target 40-70 % of the time), then loses it sideways. Gantry and free-standing are alike.

**What limits it** (gantry, same policy, other camera assumptions; `--cameras ...`):

| camera | 5 cm | 8 cm | 12 cm | |
|---|---|---|---|---|
| Albert (above) | 4.0 s | 3.4 s | 3.5 s | |
| `fast`: 30 fps, 40-60 ms, same noise | 4.0 s | 4.3 s | 7.3 s | latency matters most for high bounces |
| `precise`: Albert's timing, half the centre noise, a third of the radius noise | 2.9 s | 4.3 s | 3.9 s | detection noise alone: no |
| `flat`: Albert's camera, exact arm-model flat | 3.1 s | 3.2 s | 4.2 s | flat error alone: no |
| `clean`: all of the above, no outliers or lost frames | 4.9 s (12 %) | 6.8 s (19 %) | 6.9 s (22 %) | |
| `ideal`: the true ball state | **14.3 s (66 %)** | **16.0 s (75 %)** | **14.6 s (66 %)** | apex CV 0.18-0.23 |

The arm and the reward are not the problem: with the true ball state the same policy keeps the ball bouncing for most of
the 20 s with regular apexes. No single camera imperfection explains the gap either; even a clean camera only reaches
5-7 s. **The predictor is the bottleneck**: `BallKF3D` gives the height to ~15 mm and the normal speed to ~0.1 m/s (median),
but its p90 errors (35 mm, 0.7-1 m/s) come at the impacts: when exactly the ball meets the blade, and the velocity after it
(restitution, spin, the blade's motion), predicted across 1-3 frames of latency. The next steps, in order:

1. a better impact model in `BallKF3D` (the contact is 10 ms long; restitution and tangential change estimated online), or
   a learnt predictor (a small network trained in sim on the camera stream, replacing the hand-written hybrid);
2. give the policy the predictor's uncertainty (or the time since the last accepted frame), so it hits softer when unsure;
3. train longer with the privileged critic (stage c improved slowly to the end; 14 M steps only).

What the arm must do is modest (all rows): peak joint speeds 1.3-1.9 rad/s, blade speed 0.5-0.6 m/s at the 99th percentile,
below Albert's arm's capability (the policy's command rate limit is 1.5 rad/s per joint).

Video: [`videos/6_albert_bounce_sim.mp4`](../videos/6_albert_bounce_sim.mp4), free-standing lean: 8 cm and 12 cm with Albert's
camera, then 8 cm with the true ball state. Each segment is the longest of 6 episodes (so better than the average above).

## Files

| | |
|---|---|
| `asimov_pp/model.py` | `build(bounce=True)`: lively blade contact (`BOUNCE_SOLREF`, `BOUNCE_DAMPING`) |
| `asimov_pp/env.py` | `bounce=(lo, hi)`: target apex range; 3D camera model (`CAM_*`, `_cam_detect`), drag, reward, rest/fall termination; `privileged=True` for the critic |
| `asimov_pp/estimator.py` | `BallKF3D`, `KF3_PARAMS`, `detection_to_3d` (pixel centre + radius -> 3D + covariance, for the robot as well) |
| `asimov_pp/ppo.py`, `train.py` | asymmetric actor-critic (`--privileged-critic`), `--stage bounce --bounce lo hi --estimator kf3`, warm start 40 -> 43 inputs |
| `train_bounce.sh` | the curriculum |
| `tools/eval_bounce.py` | the tables above, `--video` |
| `tests/test_ball_bounce.py`, `tests/test_kf3.py` | restitution drop test; predictor error (run as scripts or with pytest) |
| `outputs/albert_bounce` | the policy (43 inputs), `args.json`, `log.csv`, `train.log` (stages a-d and the evaluation) |
