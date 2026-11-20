#!/bin/bash
# Bounce stage: Albert keeps the ball bouncing on the paddle (sim only so far). Same robot-matched sim as train_albert.sh
# (identified arm, Albert-like camera noise, outliers and dropped frames, flat error, gantry-to-free-standing lean), plus a lively
# blade (restitution ~0.87), air drag, the head camera seeing the ball in 3D (apparent size) and the 3D hybrid predictor.
#
# Curriculum on realism, each stage warm-started from the previous (docs/bounce.md):
#   a  ideal ball state, no extra command delay                       from albert_v3_final (balancing): learns to bounce
#   b  head camera + 3D predictor, short latency, clean detections   learns to bounce from the camera
#   c  Albert's camera (18-30 fps, 70-160 ms, outliers, lost frames, flat error), 20-80 ms command delay
#   d  as c, higher targets (4-15 cm)
# Straight from albert_v3_final to c, PPO stalls at ~2 s of bouncing (outputs/diag_bounce_direct). ~1 h on 16 CPUs.
# SKIP_A=1: start from an existing outputs/albert_bounce_a.
set -e
export PYTHONPATH=.
PY=${PY:-uv run python}
W=${WORKERS:-$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu) - 2 ))}
B="--stage bounce --home-q 0.7577 0.9729 -0.6564 -1.3960 -1.2616 --stand-gravity 0.1214 0.0081 -0.9929 \
 --stand-gravity2 0.203 -0.011 -0.980 --ball-start 0.03 0.10 --arm-dyn albert/arm_dynamics.json --smooth-penalty 0.01 \
 --workers $W --envs-per-worker $(( 256 / W + 1 ))"
CAM="--camera-obs --estimator kf3"
REAL="$CAM --cam-fps 18 30 --cam-latency 0.07 0.16 --act-latency 1 4 --cam-outliers 0.10 --cam-drops 0.08 --tilt-bias 0.04"
INIT=${INIT:-outputs/albert_v3_final/best.pt}
# a: ideal ball state; then the critic also sees the sim's truth (asymmetric actor-critic: the policy keeps its 43 inputs)
[ -n "$SKIP_A" ] || $PY -m asimov_pp.train $B --act-latency 0 0 --bounce 0.03 0.08 --p-rest 0.15 --min-std 0.35 --name albert_bounce_a --iters 250 --init "$INIT"
$PY -m asimov_pp.train $B $CAM --privileged-critic --cam-fps 25 30 --cam-latency 0.03 0.07 --act-latency 0 1 --bounce 0.03 0.08 \
  --p-rest 0.15 --min-std 0.2 --name albert_bounce_b --iters 400 --init outputs/albert_bounce_a/best.pt
$PY -m asimov_pp.train $B $REAL --privileged-critic --bounce 0.03 0.10 --p-rest 0.15 --min-std 0.2 --name albert_bounce_c \
  --iters 800 --init outputs/albert_bounce_b/best.pt
$PY -m asimov_pp.train $B $REAL --privileged-critic --bounce 0.04 0.15 --p-rest 0.15 --min-std 0.15 --name albert_bounce \
  --iters 400 --init outputs/albert_bounce_c/best.pt
$PY -m tools.eval_bounce outputs/albert_bounce/best.pt
