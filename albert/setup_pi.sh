#!/bin/bash
# Set up the robot's own computer (Raspberry Pi 5, Debian, Python 3.13) to run the albert/ scripts there. From the Mac:
#   albert/setup_pi.sh            (key login as albert@albert.local first: ssh-copy-id albert@albert.local)
# Then, on the Pi, give it its own menlo-sdk profile: ssh -t albert@albert.local '~/asimov-pingpong-balance/.venv/bin/menlo setup'
# (manager http://127.0.0.1), and set Asimov Edge's udp-state-host to 127.0.0.1 in Asimov Manager. See ONBOARD.md.
set -e
cd "$(dirname "$0")/.."
PI=${ALBERT_PI:-albert@albert.local}
ASIMOV_DIR=${ASIMOV_DIR:-../supermimic_ws/asimov-1}
ssh -o BatchMode=yes "$PI" "mkdir -p asimov-pingpong-balance/outputs asimov-pingpong-balance/logs supermimic_ws/asimov-1"
rsync -az -e "ssh -o BatchMode=yes" "$ASIMOV_DIR/sim-model" "$PI:supermimic_ws/asimov-1/"
rsync -az -e "ssh -o BatchMode=yes" --exclude "__pycache__" albert asimov_pp hardware assets pyproject.toml "$PI:asimov-pingpong-balance/"
rsync -az -e "ssh -o BatchMode=yes" $(ls -d outputs/albert_v3_final outputs/albert_v4 outputs/albert_v2A 2>/dev/null) "$PI:asimov-pingpong-balance/outputs/"
ssh -o BatchMode=yes "$PI" 'cd asimov-pingpong-balance && [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q "numpy<2.6" scipy "mujoco==4.0.1" menlo-sdk imageio imageio-ffmpeg pillow
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  OPENBLAS_NUM_THREADS=1 .venv/bin/python -c "import albert.run_policy; from asimov_pp.ppo import load_policy; load_policy(\"outputs/albert_v3_final/best.pt\"); print(\"Pi ready\")"'
