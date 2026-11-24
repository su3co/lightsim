#!/bin/bash
# One monitored run of bench_rf.py on pinned cores with stated thread counts.
#   run_rf.sh ENGINE THREADS TRIALS CORES [extra bench_rf.py arguments]
# CORES is a taskset list of physical cores (one hardware thread each), e.g. 32-39.
# Needs PY (the Python with SAX, qpdk and lightsim importable) in the environment.
set -u
E=$1; N=$2; T=$3; CORES=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
export CUDA_VISIBLE_DEVICES= RAYON_NUM_THREADS=$N OMP_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N MKL_NUM_THREADS=$N
# XLA sizes its CPU thread pool from the affinity mask (taskset below); with
# one core its Eigen threading is switched off as well
if [ "$N" = 1 ]; then export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"; else export XLA_FLAGS=""; fi
taskset -c "$CORES" "$PY" "$HERE/../../scripts/monitor.py" --jsonl "$HERE/results/monitor.jsonl" -- \
  "$PY" "$HERE/bench_rf.py" --engine "$E" --threads "$N" --trials "$T" "$@"
