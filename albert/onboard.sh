#!/bin/bash
# Run an albert/ script ON Albert's Pi from the Mac: syncs the code, runs it interactively over ssh (prompts work; a dropped
# ssh session ends it like Ctrl+C: arm home, stand() -> balance()), then copies the run's logs back to logs/onboard/.
#   albert/onboard.sh albert.run_policy --supported --gantry --duration 60 --log logs/P5.csv
#   albert/onboard.sh albert.status --mode udp
# Needs: ssh key login as albert@albert.local, Edge's udp-state-host = 127.0.0.1 (the Pi itself). The Mac can still watch:
#   python -m albert.status --mode livekit
set -e
cd "$(dirname "$0")/.."
PI=${ALBERT_PI:-albert@albert.local}
DIR=asimov-pingpong-balance
# code and calibrations both ways, the newer file wins (-u): a calibration made on either side is kept; never the Pi's logs
rsync -azu -e "ssh -o BatchMode=yes" "$PI:$DIR/albert/*.json" albert/ 2>/dev/null || true
rsync -azu -e "ssh -o BatchMode=yes" --exclude "__pycache__" --exclude "logs" albert asimov_pp "$PI:$DIR/"
rsync -az -e "ssh -o BatchMode=yes" --exclude "*.pt.tmp" outputs/albert_v3_final outputs/albert_v2A "$PI:$DIR/outputs/" 2>/dev/null || true
MOD=$1; shift
ssh -t "$PI" "cd $DIR && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m $MOD $*" || true
mkdir -p logs/onboard
rsync -az -e "ssh -o BatchMode=yes" "$PI:$DIR/logs/" logs/onboard/ && echo "logs copied to logs/onboard/"
rsync -azu -e "ssh -o BatchMode=yes" "$PI:$DIR/albert/*.json" albert/ 2>/dev/null && echo "calibrations synced back (newer wins)"
