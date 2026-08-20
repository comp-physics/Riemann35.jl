"""
    nt35(v)

Build an `NTuple{35,Float64}` from a 35-element vector.

WHY THIS EXISTS RATHER THAN `NTuple{35,Float64}(v)`. The constructor form goes through a generic,
non-inferrable path for an `AbstractVector` argument and is pathologically expensive at this length:

    NTuple{35,Float64}(v)          1184 B   1600 ns
    ntuple(i -> v[i], Val(35))      288 B     44 ns

a 4x allocation and 36x time difference, for a bitwise identical result (`===`). At four
conversions per interface flux this was the single largest allocation site in `face_flux_1d`.

`Val(35)` is load-bearing: it makes the length a compile-time constant so `ntuple` unrolls. Passing
a plain `35` reinstates the slow path.

DELIBERATELY NOT `@inbounds` AT THE CALL SITE ALONE. The bounds check is inside the closure, so it
must be written here; a caller-side `@inbounds` does not propagate into it.
"""
@inline nt35(v::AbstractVector{Float64}) = ntuple(i -> @inbounds(v[i]), Val(35))
@inline nt35(t::NTuple{35,Float64}) = t          # already a tuple: identity, no copy
