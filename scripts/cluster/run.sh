#!/usr/bin/env bash
set -euo pipefail
module load lmod
module use /home/thuat/.local/easybuild/modules/all/
export NUMBA_NUM_THREADS=48          
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_file.py> [args...]"
    exit 1
fi

pyfile="$1"; shift
if [ ! -f "$pyfile" ]; then
    echo "Error: file not found: $pyfile" >&2
    exit 1
fi

mkdir -p ./logs
base="$(basename "$pyfile" .py)"
log="./logs/${base}.log"

setsid nohup python3 "$pyfile" "$@" </dev/null >>"$log" 2>&1 &
tail -f "logs/${base}.log"