#!/bin/sh
# Full curriculum, each stage starting from the previous one's best checkpoint (~30 min on an M5 Max).
# The ball is seen through the head-camera model (25-30 fps, 40-120 ms), as on Albert.
set -e
export PYTHONPATH=.
N=${PREFIX:-cam}
uv run python -m asimov_pp.train --camera-obs --stage balance --name ${N}_balance --iters 300 > outputs/${N}_balance.log 2>&1
uv run python -m asimov_pp.train --camera-obs --stage perturb --name ${N}_perturb --iters 500 --init outputs/${N}_balance/best.pt > outputs/${N}_perturb.log 2>&1
uv run python -m asimov_pp.train --camera-obs --stage move --name ${N}_move --iters 800 --init outputs/${N}_perturb/best.pt > outputs/${N}_move.log 2>&1
