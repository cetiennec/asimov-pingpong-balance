#!/bin/sh
# Stage 3 for Albert: balance while walking (Menlo's walking clip), from the Albert stage-2 policy, with all of Albert's
# measured settings (camera 18-30 fps / 40-150 ms, Kalman estimator, +40-100 ms arm delay, taught home, hand-like releases).
# Runs at low priority with 10 workers so a live Albert session on the same Mac keeps the CPU.
set -e
export PYTHONPATH=.
HOMEQ=$(uv run python -c "import albert.common as c; print(' '.join(f'{x:.4f}' for x in c.HOME))")
GRAV=$(uv run python -c "import albert.common as c; print(' '.join(f'{x:.4f}' for x in c.TAUGHT_GRAVITY))")
C="--camera-obs --cam-fps 18 30 --cam-latency 0.04 0.15 --home-q $HOMEQ --stand-gravity $GRAV --ball-start 0.05 0.25 0.015 --estimator kf --act-latency 2 5"
nice -n 15 uv run python -m asimov_pp.train $C --stage move --name albert_move --iters 800 --workers 10 --envs-per-worker 24 --init outputs/albert_perturb/best.pt > outputs/albert_move.log 2>&1
