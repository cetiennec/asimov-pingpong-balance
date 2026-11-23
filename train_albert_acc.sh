#!/bin/bash
# albert_v3_final's recipe (train_albert.sh) with the ball predictor that also knows the blade's acceleration
# (--kf-paddle-acc, KF_PARAMS_ACC: on the 10 onboard runs its error is 4.1 mm vs 5.5 mm median, tools/replay_kf_paddle_acc.py).
# Warm-started from albert_v3_final (same observation layout; only the ball estimate it sees changes).
# run_policy reads kf_paddle_acc from the checkpoint's args.json and runs the matching predictor.
set -e
export PYTHONPATH=.
W=${WORKERS:-$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu) - 2 ))}
C="--camera-obs --cam-fps 18 30 --cam-latency 0.07 0.16 \
 --home-q 0.7577 0.9729 -0.6564 -1.3960 -1.2616 --stand-gravity 0.1214 0.0081 -0.9929 --stand-gravity2 0.203 -0.011 -0.980 \
 --ball-start 0.05 0.25 0.015 --estimator kf --kf-paddle-acc --act-latency 1 4 --arm-dyn albert/arm_dynamics.json \
 --cam-outliers 0.10 --cam-drops 0.08 --tilt-bias 0.04 --smooth-penalty 0.01 --workers $W --envs-per-worker $(( 256 / W + 1 ))"
INIT=${INIT:-outputs/albert_v3_final/best.pt}
uv run python -m asimov_pp.train $C --stage balance --name albert_v4acc_balance --iters 150 --init "$INIT"
uv run python -m asimov_pp.train $C --stage perturb --name albert_v4acc --iters 400 --init outputs/albert_v4acc_balance/best.pt
uv run python -m tools.eval_albert outputs/albert_v4acc/best.pt
