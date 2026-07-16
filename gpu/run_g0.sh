#!/usr/bin/env bash
# run_g0.sh — launch an order-3 GPU script with Julia debug level 0 (`-g0`).
#
# WHY: the order-3 WENO5 + θ*-IDP kernels are enormous, and `ptxas` spends most of
# its time generating line-info/debug metadata for them, not optimizing. `-g0` sets
# Julia's debug level to 0, which drops `--generate-line-info` and the `.target debug`
# header (release-mode ptxas). MEASURED on a Tesla V100 (clean same-script A/B):
# ptxas ~742 s (default) -> ~31 s with `-g0` — a ~24x compile-time cut. Numerics are
# byte-identical (only debug info is stripped; the emitted SASS is unchanged).
#
# Usage:
#   gpu/run_g0.sh gpu/run_staged.jl <args...>
#   JULIA=/path/to/julia JULIA_PROJECT=gpu/gpuenv2 gpu/run_g0.sh <script.jl> <args...>
#
# Honors $JULIA (default: julia on PATH) and $JULIA_PROJECT (default: gpu/gpuenv2).
# A `--project=...` in the passed args still wins (Julia takes the last one).
set -euo pipefail
JULIA="${JULIA:-julia}"
PROJECT="${JULIA_PROJECT:-gpu/gpuenv2}"
exec "$JULIA" -g0 --project="$PROJECT" "$@"
