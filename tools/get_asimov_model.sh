#!/bin/bash
# Fetch Menlo's Asimov 1 simulation model (MJCF + meshes, https://github.com/menloresearch/asimov-1) next to this repository,
# where asimov_pp/model.py looks for it (or set ASIMOV_DIR). For the walking stage only: HybridRobotics/supermimic (SUPERMIMIC_DIR).
set -e
cd "$(dirname "$0")/../.."
mkdir -p supermimic_ws && cd supermimic_ws
if [ ! -d asimov-1 ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/menloresearch/asimov-1.git
  git -C asimov-1 sparse-checkout set sim-model
fi
[ "$1" = "--walk" ] && [ ! -d supermimic ] && git clone --depth 1 https://github.com/HybridRobotics/supermimic.git
echo "model: $(pwd)/asimov-1/sim-model"
