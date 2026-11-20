#!/bin/bash
# Albert's policy, the recipe of outputs/albert_v3_final: the sim matched to the robot (identified arm, Albert-like camera
# noise and outliers, flat error), body lean between the gantry's 7 deg and free-standing 12 deg, Kalman predictor.
# Stage 1 (balance) then 2 (+ perturbations), each warm-started from the previous; ~1 h on 32 CPUs (we used HF Jobs,
# cpu-performance). From scratch: drop the first --init. Checkpoints: outputs/<name>/best.pt
set -e
export PYTHONPATH=.
W=${WORKERS:-$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu) - 2 ))}
C="--camera-obs --cam-fps 18 30 --cam-latency 0.07 0.16 \
 --home-q 0.7577 0.9729 -0.6564 -1.3960 -1.2616 --stand-gravity 0.1214 0.0081 -0.9929 --stand-gravity2 0.203 -0.011 -0.980 \
 --ball-start 0.05 0.25 0.015 --estimator kf --act-latency 1 4 --arm-dyn albert/arm_dynamics.json \
 --cam-outliers 0.10 --cam-drops 0.08 --tilt-bias 0.04 --smooth-penalty 0.01 --workers $W --envs-per-worker $(( 256 / W + 1 ))"
INIT=${INIT:-outputs/albert_v2A/best.pt}
uv run python -m asimov_pp.train $C --stage balance --name albert_v3_balance --iters 150 --init "$INIT"
uv run python -m asimov_pp.train $C --stage perturb --name albert_v3 --iters 400 --init outputs/albert_v3_balance/best.pt
uv run python -m tools.eval_albert outputs/albert_v3/best.pt
