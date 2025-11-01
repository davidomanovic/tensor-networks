#!/usr/bin/env bash
set -euo pipefail

module load lmod
module use /home/thuat/.local/easybuild/modules/all/

export JULIA_NUM_THREADS=48
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48

if [ $# -lt 1 ]; then
  echo "Usage: $0 <script.jl> [args...]"
  exit 1
fi

jlfile="$1"; shift
if [ ! -f "$jlfile" ]; then
  echo "Error: file not found: $jlfile" >&2
  exit 1
fi

mkdir -p ./logs
base="$(basename "$jlfile" .jl)"
log="./logs/${base}.log"
pid="./logs/${base}.pid"

projdir="$(cd "$(dirname "$jlfile")" && pwd)"
if [ -f "$projdir/Project.toml" ]; then
  export JULIA_PROJECT="$projdir"
fi

setsid nohup julia --project="${JULIA_PROJECT:-@v1.11}" "$jlfile" "$@" </dev/null >>"$log" 2>&1 &
echo $! >"$pid"
echo "started pid $(cat "$pid"), logging to $log"
