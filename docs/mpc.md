# MPC for the ball on the paddle (simulation only)

A model-predictive controller as an alternative to the PPO policy: same inputs (the 40-value observation: Kalman ball
estimate, blade gravity, IMU, arm joints), same output (5 joint-target deltas x 0.03 rad), 50 Hz. Code:
[`albert/mpc.py`](../albert/mpc.py). On Albert since 2026-11-23: 56.9 s of 60 on the gantry (M1_mpc), 59.8 s in one catch (F1_mpc_2).

```
python -m tools.eval_albert outputs/albert_v3_final/best.pt mpc                    # robot-matched sim, 2026-11-22 arm
python -m tools.eval_albert outputs/albert_v3_final/best.pt mpc --arm-dyn albert/arm_dynamics_20261120.json
python -m tools.eval_sim_pipeline policy mpc                                       # rendered camera + detector + run_policy loop
python -m albert.run_policy --sim --controller mpc                                 # one simulated session
```

## Model

Per blade axis (u along the handle, w across), the two axes decoupled:

| part | model | where the numbers come from |
|---|---|---|
| ball | `p'' = KB (g + d) - C v - LAM a_blade`, KB = 3/5 x 0.87 x 9.81, LAM = KB / 9.81 | the Kalman filter's model (`asimov_pp/estimator.py`) |
| tilt | `r += K du[k-D]`, `g += ALPHA (r - g)` (first-order lag, settled tilt r) | step responses on the sim arm (below) |
| translation | the same for the blade centre in its own plane, `s` | idem (assumed alike) |
| delay D | 3 control steps | eval: Albert's arm lags 1-4 steps (3 and 4 tune alike, 2 worse) |
| d | constant tilt error, estimated from the ball's velocity | in the code, but off (`l_d = 0`): it tuned worse |

`g` is the blade's in-plane gravity (unit g, ~ tilt in rad) from forward kinematics of the measured joints, with the
observation's blade gravity low-passed for the slow part (IMU, waist). `r` and the centre's settled value are estimated by
a small observer from the tilt / centre innovations (friction and sag make the settled tilt differ from the command).

The QP (per axis, condensed, move-blocked into 7 blocks over 25 steps = 0.5 s, 14 variables): minimise ball position and
speed, tilt, centre displacement and command changes; `|du| <= 0.06` (unit g per step), `|dc| <= 1 cm` per step. A
primal active-set solver in numpy (exact for this size; no osqp / cvxpy needed). The first move is mapped to the joints by
a second small box QP: tilt and centre targets by least squares on the model Jacobians, the centre held along the blade
normal, each joint penalised by its stiction band (friction / kp of the identified arm), every joint at most 0.03 rad per
step, plus a slow pull to the home pose in the targets' null space.

## What it took

* **Tilt-only control cannot catch the episode starts.** The eval starts the ball up to 5 cm off-centre at up to 0.25 m/s
  per axis, with the Kalman filter's velocity starting at 0. A tilt-only MPC held every ball it caught but lost 40-50 % in
  the first 0.5 s. The PPO policy catches them by **translating the blade** 4-10 cm under the ball (a rolling ball feels
  0.52 of the blade's acceleration at once, the tilt only after the arm's lag). Adding the blade's in-plane translation as a
  second input took gantry balance from ~45 % to ~85 %.
* **The Kalman filter must know the blade's acceleration** once the controller translates it: it predicts with
  `a_paddle = 0`, so between camera frames (115 ms late) the estimate lagged by 1-2 cm and the MPC overshot. New option
  `kf_paddle_acc` (env) / `ObsBuilder(paddle_acc=True)` (run_policy, on with `--controller mpc`): the blade centre's
  acceleration from forward kinematics in the body frame, second difference over the control steps. Off for the policies.
* **IK: no antagonistic joints.** Holding the blade centre fixed in 3D made the IK use shoulder pitch against shoulder yaw;
  with the identified friction (shoulder pitch 1.3 Nm) the step response then depended on the step's size and sign
  (gain 0.9-2.2). Holding the centre only along the normal and weighting joints by their stiction band gives a symmetric
  first-order response (u axis: 0.31 / 0.56 / 0.69 / 0.76 of the step after 2 / 4 / 7 / 15 steps).

## Results (2026-11-23, simulation only)

Tuning: a random search then a sweep around the best on gantry + free balance, 48 episodes each (2026-11-22 arm); the
weights in `MPCParams` are that best set. The variants around it all scored within the noise (+-5 points).

**Robot-matched sim** (`tools.eval_albert`, 96 episodes of 10 s per line: camera latency 70-160 ms, 18-30 fps, outliers,
dropped frames, flat error 0.04, arm command delay 1-4 steps). Action change: mean |a_t - a_t-1| over joints, action units.

| arm | scenario | v3_final success | MPC success | v3 / MPC mean time | v3 / MPC ball from centre | v3 / MPC action change |
|---|---|---|---|---|---|---|
| 2026-11-22 | gantry balance | **81.2 %** | 77.1 % | 8.6 / 8.0 s | 1.9 / 1.8 cm | 0.301 / 0.316 |
| 2026-11-22 | free balance | 75.0 % | **76.0 %** | 8.1 / 8.2 s | 2.1 / 1.7 cm | 0.370 / 0.325 |
| 2026-11-22 | free perturb | **44.8 %** | 41.7 % | 6.2 / 6.5 s | 2.7 / 2.4 cm | 0.339 / 0.326 |
| 2026-11-20 | gantry balance | **80.2 %** | 77.1 % | 8.4 / 8.1 s | 1.9 / 1.8 cm | 0.303 / 0.316 |
| 2026-11-20 | free balance | 75.0 % | **78.1 %** | 8.1 / 8.2 s | 2.1 / 1.7 cm | 0.374 / 0.324 |
| 2026-11-20 | free perturb | **46.9 %** | 41.7 % | 6.3 / 6.6 s | 2.7 / 2.4 cm | 0.343 / 0.326 |

(v3_final scores lower here than the 87.5 / 77.1 / 56.2 % quoted earlier: those were 48 episodes, these 96; one standard
error is ~4 points.) In the balance scenarios most failures of both controllers come in the first 0.5 s, the fast
off-centre starts (checked on 24 gantry episodes); with perturbations the losses come later too.

**Whole pipeline** (`tools.eval_sim_pipeline`: `run_policy --sim`, rendered head camera at 22 fps / 105 ms, real detector,
Kalman predictor, the ball put back by a "hand" 2 s after each loss; 8 sessions of 60 s per line):

| arm | controller | catches | balanced | per catch | ball from centre | action change |
|---|---|---|---|---|---|---|
| 2026-11-22 | v3_final | 33 | 414 s (86.3 %) | 12.6 s | 0.8 cm | 0.268 |
| 2026-11-22 | MPC | 8 | 462 s (96.2 %) | 57.7 s (never lost) | 0.5 cm | 0.180 |
| 2026-11-20 | v3_final | 25 | 435 s (90.7 %) | 17.4 s | 0.8 cm | 0.234 |
| 2026-11-20 | MPC | 8 | 462 s (96.2 %) | 57.7 s (never lost) | 0.5 cm | 0.044 |

The MPC caught the first ball in each session and held it to the end (the gentle starts of the hand suit it). Its action
change is ~0.04 in 13 of the 16 sessions; in 3 of the 8 on the 2026-11-22 (softer) arm it settled into a sustained
oscillation of the arm, action change ~0.4, the ball still held: the same jiggle as the policy on Albert, which this
controller does not cure on the soft arm.

Cost: 0.18 ms per step median, 0.45 ms p95 on a Mac, single thread (forward kinematics, IK QP and two 14-variable QPs);
expect ~1-2 ms on the Pi 5, within the 20 ms period. The policy's MLP is 0.02 ms.

## Worth trying on Albert?

Yes, as an experiment on the gantry: in the robot-matched sim it is on par with v3_final (within the noise; slightly
worse on the gantry and with perturbations, slightly better free-standing), and in the whole camera pipeline it held every
ball for the full minute with the ball closer to the centre and mostly smoother actions. It is not a clear win in the
harder scenarios. Before a run:

1. **Paddle acceleration in the predictor** (`ObsBuilder(paddle_acc=True)`, on with `--controller mpc`) is new and untested
   on the robot: it differentiates the forward-kinematics blade centre twice; encoder noise could make it noisy (check
   the KF estimate in a log replay first, `tools/tune_kf.py`-style).
2. **Its model of the arm** (lag 0.27-0.32 per step, gain 0.8-0.9, delay 3 steps) was identified on the sim's arm, not
   Albert's. Re-identify the tilt and translation step responses from a logged session (G4_v3) with the IK in
   `mpc.py`, or with a few small commanded tilt steps on the gantry without the ball.
3. **The jiggle on the soft arm** (3 of 8 sim sessions): raise `w_du` / `w_dc` or lower the gains if it shows up; the
   `--mpc 'key=value,...'` flag takes any parameter without code changes.
4. It moves the blade sideways by up to ~10 cm during a catch (as the policy does); check the safe-pose and tilt guards
   allow it.

Run: `python -m albert.run_policy --supported --gantry --controller mpc` (the policy flags otherwise unchanged).

Online: the Space's live simulator (`/live`, server-rendered) offers `mpc` as a controller in the three Albert scenarios
(Albert-matched sim, gantry, free-standing); its project snapshot still runs the 2026-11-20 arm. Not in the in-browser
simulator (`/`, ONNX policies): that would need the controller ported to TypeScript.
