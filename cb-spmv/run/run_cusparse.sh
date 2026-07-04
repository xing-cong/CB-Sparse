#!/usr/bin/env bash
#
# Compile the cuSparse baseline (benchmark/cuSparse/) and run it serially
# over every matrix in data/. Matrices are referenced by a relative path but
# converted to an absolute path at run time, because the program requires one.
#
set -euo pipefail

# Locate this script (run/) and the repo root, so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 1) Enter the build directory and compile.
cd "$ROOT_DIR/benchmark/cuSparse"
make

# 2) Serially run over every matrix in data/ (relative path -> absolute path).
EXE=./cusparse_spmv
DATA_REL=../../data

shopt -s nullglob
matrices=("$DATA_REL"/*.mtx)
shopt -u nullglob

if [ ${#matrices[@]} -eq 0 ]; then
    echo "[Error] No .mtx files found in $DATA_REL" >&2
    exit 1
fi

for mtx in "${matrices[@]}"; do
    "$EXE" "$(realpath "$mtx")" || echo "[Warn] $EXE failed on $mtx" >&2
done
