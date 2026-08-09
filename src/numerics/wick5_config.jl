# Wick/Grad degree-5 blend configuration. GENERATED/EDITED -- see `misc/set_wick5.sh`.
#
# WHY A FILE AND NOT AN ENVIRONMENT VARIABLE. The first version read these from ENV at module load.
# That silently does nothing: `const` is evaluated at PRECOMPILE time and baked into the cache, and
# ENV is not part of the precompile hash, so setting the variable afterwards leaves the stale value
# in place. An alpha ramp run that way produces four identical baselines that look like "the blend
# has no effect" -- a false negative that is indistinguishable from a real physical result. Julia
# hashes SOURCE files, so editing this one invalidates the cache and the value actually takes.
#
# alpha = 0 is the shipped closure, reproduced bit for bit (the blend branch is constant-folded
# away). alpha = 1 is the full Wick/Grad fifth moments scaled by the gain.
const WICK5_ALPHA = 0.0
const WICK5_GAIN  = 0.958
