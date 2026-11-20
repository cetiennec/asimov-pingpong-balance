#!/bin/sh
# Stages 1-2 for Albert as measured (arm follows commands 85-145 ms late: +40-100 ms command delay in training): head camera ~22 fps over the hybrid link (latency not measured yet: 40-150 ms),
# ball released by hand (up to 5 cm off-centre, 25 cm/s, dropped from up to 1.5 cm), home = the flat pose taught with albert/teach_flat.py, body leaning on the gantry as its IMU reads.
set -e
export PYTHONPATH=.
HOMEQ=$(uv run python -c "import albert.common as c; print(' '.join(f'{x:.4f}' for x in c.HOME))")
GRAV=$(uv run python -c "import albert.common as c; print(' '.join(f'{x:.4f}' for x in c.TAUGHT_GRAVITY))")
C="--camera-obs --cam-fps 18 30 --cam-latency 0.04 0.15 --home-q $HOMEQ --stand-gravity $GRAV --ball-start 0.05 0.25 0.015 --estimator kf --act-latency 2 5"
echo "home $HOMEQ | gravity $GRAV"
uv run python -m asimov_pp.train $C --stage balance --name albert_balance --iters 300 > outputs/albert_balance.log 2>&1
uv run python -m asimov_pp.train $C --stage perturb --name albert_perturb --iters 500 --init outputs/albert_balance/best.pt > outputs/albert_perturb.log 2>&1
