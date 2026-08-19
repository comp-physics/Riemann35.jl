#!/usr/bin/env bash
# Set the Wick blend parameters, invalidating Julia's precompile cache by touching the source.
# Usage: misc/set_wick5.sh <alpha> [gain]
set -euo pipefail
cd "$(dirname "$0")/.."
a=${1:?alpha required}; g=${2:-0.958}
sed -i "s/^const WICK5_ALPHA = .*/const WICK5_ALPHA = $a/;s/^const WICK5_GAIN  = .*/const WICK5_GAIN  = $g/" \
  src/numerics/wick5_config.jl
grep -E "^const WICK5" src/numerics/wick5_config.jl
