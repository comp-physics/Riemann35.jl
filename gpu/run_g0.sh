#!/usr/bin/env bash
# run_g0.sh — launch an order-3 GPU script with Julia debug level 0 (`-g0`).
#
# WHY: `-g0` sets Julia's debug level to 0, which drops `--generate-line-info` and the
# `.target debug` header (release-mode ptxas). Numerics are byte-identical.
#
# BUT DO NOT EXPECT MUCH. This header used to claim a ~24x compile-time cut (~742 s ->
# ~31 s, measured Tesla V100). That DOES NOT REPRODUCE on A100: measured 2026-07-26,
# order-3, clean same-script A/B, 863.5 s default vs 766.2 s with `-g0` — i.e. 1.13x.
# The stated mechanism fails too, not just the magnitude: the emitted PTX differs by
# only 2.4% between the two debug levels, so there is no line-info bulk to strip. It
# has not been re-checked on a V100, so the original may have been hardware-specific.
#
# The real order-3 compile cost was six-fold inlining of `_rank_face_theta` inside
# `_blend_residual!` — 36 inlined copies of the HLL closure body, 79.2% of all emitted
# PTX. Two `@noinline` annotations in residual3d_order3_gpu.jl took compile 758 s -> 66 s
# with byte-identical results. See "Compile time" in gpu/README.md.
#
# `-g0` is free and mildly positive, so this wrapper is still worth using.
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
