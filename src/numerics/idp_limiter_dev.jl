"""
    idp_limiter_dev.jl — device-safe θ*-IDP update limiter.

Given a realizable first-order anchor state Mlo and a candidate correction dM,
find the largest θ in [0,1] keeping Mlo+θ·dM realizable (moment cone). Bisection
on the shipped `_state_realizable`; the closed-form Hankel-pencil cubic is the
production optimization. Used by the two-pass residual: per cell, the six
one-sided half-updates each get a θ, and the interface θ is the min over the two
cells sharing it (done by the residual caller).
"""
module IdpLimiterDev

include(joinpath(@__DIR__, "riemann_flux_dev.jl"))
using .RiemannFluxDev: _state_realizable

export theta_star_update_dev

@inline function theta_star_update_dev(Mlo::NTuple{35,Float64}, dM::NTuple{35,Float64}; nb::Int = 24)
    full = ntuple(j -> Mlo[j] + dM[j], Val(35))
    _state_realizable(full) && return 1.0
    lo = 0.0; hi = 1.0
    for _ in 1:nb
        mid = 0.5 * (lo + hi)
        m = ntuple(j -> Mlo[j] + mid * dM[j], Val(35))
        _state_realizable(m) ? (lo = mid) : (hi = mid)
    end
    lo
end

end # module
