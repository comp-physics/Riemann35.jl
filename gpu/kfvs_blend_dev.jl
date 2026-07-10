"""
    kfvs_blend_dev.jl — full-cone θ* IDP blend of a high-order update toward the
    kinetic-FVS anchor (device).

Increment D of the KFVS anchor. **PURE ADDITION** — not wired into the residual,
projection35 untouched; this builds + validates the blend as a standalone device
function.

Given a realizable anchor state `U_anchor` (the `measure_update_3d_dev` result of
increment B — full-cone realizable BY CONSTRUCTION under the CFL) and a candidate
high-order raw-moment update `U_highorder` (which may exit the cone), the convex
blend is

    U(θ) = (1 − θ)·U_anchor + θ·U_highorder ,   θ ∈ [0,1],

and `θ*` is the largest θ keeping `U(θ)` **FULL-CONE realizable**. At θ=0 the blend
is the anchor (realizable); the largest feasible θ recovers as much of the
high-order update as the cone allows.

# WHY FULL-CONE (the core of this increment)
The shipped Track-2 θ* limiter (`IdpLimiterDev.theta_star_update_dev`, and its
bisection sibling) checks only the MARGINAL cone: `RiemannFluxDev._state_realizable`
uses `_marg_shape` per axis with NO cross-moment block. But the anchor's whole value
is preserving the CROSS-moment cone — the Δ2* 6×6 moment matrix that split-HLL
loses and `projection35` repairs. So the blend limiter MUST enforce FULL-cone
realizability, not just marginals, or blended states silently exit the cross-moment
cone (quantified in gpu/validation/parity_kfvs_blend.jl).

# FULL-CONE PREDICATE (`state_realizable_fullcone_dev`)
This is the SAME predicate the solver's `projection35` uses to define 35-moment
realizability beyond the marginals:
  1. `to_recon_vars_dev(M...)` → standardized moments `S…` (== `M2CS4_35`), with
     density and the three directional variances floored/exposed.
  2. require density > 0 and all three variances > 0 (the marginal 2nd-order cone).
  3. `delta2star_psd_dev(S300,…,S022, 0.0)` — is the 6×6 Δ2* moment matrix PSD?
     (Bunch–Kaufman inertia; the cross-moment cone). This is exactly
     `projection35`'s `min eig(delta2star3D) ≥ 0` test, evaluated as a boolean.

Bisection on this predicate is correct (the design's affine-Hankel-pencil closed
form, thm:idp-blend, is a later optimization — not needed for correctness). fp64,
no heap, no closures, device-compilable. `@fastmath` deliberately OFF.
"""
module KFVSBlendDev

# sibling device modules (same include order the residual uses): recon_dev first
# (realize_dev resolves `..ReconDev`), then realize_dev, then idp_limiter_dev
# (which pulls in riemann_flux_dev + roeps3_dev for the marginal-only limiter).
include(joinpath(@__DIR__, "..", "src", "numerics", "recon_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "realizability", "realize_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "idp_limiter_dev.jl"))

using .ReconDev: to_recon_vars_dev
using .RealizeDev: delta2star_psd_dev
using .IdpLimiterDev: theta_star_update_dev

export state_realizable_fullcone_dev, theta_star_blend_fullcone_dev,
       theta_star_update_dev

# ---------------------------------------------------------------------------
# FULL-CONE realizability of a raw 35-moment state. The cross-moment (Δ2*) cone
# predicate the solver's projection35 uses. Returns Bool.
# ---------------------------------------------------------------------------
@inline function state_realizable_fullcone_dev(M::NTuple{35,Float64})
    (isfinite(M[1]) && M[1] > 0.0) || return false
    V = to_recon_vars_dev(
        M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8],M[9],M[10],M[11],M[12],M[13],M[14],M[15],
        M[16],M[17],M[18],M[19],M[20],M[21],M[22],M[23],M[24],M[25],M[26],M[27],M[28],M[29],M[30],
        M[31],M[32],M[33],M[34],M[35])
    # V = (M000,u,v,w, C200,C020,C002, S300,S400,S110,…,S022). Directional variances
    # floored to 1e-12 by to_recon_vars_dev — require them strictly positive.
    (V[1] > 0.0 && V[5] > 0.0 && V[6] > 0.0 && V[7] > 0.0) || return false
    # sanity: the standardized moments must be finite (a degenerate variance can
    # produce NaN/Inf ratios upstream).
    @inbounds for k in 8:35
        isfinite(V[k]) || return false
    end
    # Δ2* 6×6 PSD test (the cross-moment cone). V[8..35] are S300…S022 in exactly
    # the delta2star_psd_dev argument order.
    return delta2star_psd_dev(
        V[8],  V[9],  V[10], V[11], V[12], V[13], V[14], V[15], V[16], V[17],
        V[18], V[19], V[20], V[21], V[22], V[23], V[24], V[25], V[26], V[27],
        V[28], V[29], V[30], V[31], V[32], V[33], V[34], V[35], 0.0)
end

# ---------------------------------------------------------------------------
# θ* = largest θ∈[0,1] keeping U(θ) = (1−θ)·Ua + θ·Uho FULL-CONE realizable.
# Bisection (nb steps) on state_realizable_fullcone_dev. θ=0 is the anchor, which
# is realizable by construction; so θ* is always well-defined and ≥ 0. Returns
# (θ*, Ustar) where Ustar = U(θ*) is full-cone realizable.
# ---------------------------------------------------------------------------
@inline function theta_star_blend_fullcone_dev(Ua::NTuple{35,Float64},
                                               Uho::NTuple{35,Float64}; nb::Int = 30)
    @inline blend(θ) = ntuple(j -> (1.0 - θ) * Ua[j] + θ * Uho[j], Val(35))
    # common path: high-order already in the full cone → keep it all
    Ufull = blend(1.0)
    if state_realizable_fullcone_dev(Ufull)
        return (1.0, Ufull)
    end
    lo = 0.0; hi = 1.0
    @inbounds for _ in 1:nb
        mid = 0.5 * (lo + hi)
        if state_realizable_fullcone_dev(blend(mid)); lo = mid; else; hi = mid; end
    end
    return (lo, blend(lo))
end

# Marginal-only θ* blend (Track-2), for the A/B comparison in validation: same
# blend, but the shipped MARGINAL-only limiter. dM = Uho - Ua so Mlo + θ·dM = U(θ).
@inline function theta_star_blend_marginal_dev(Ua::NTuple{35,Float64},
                                               Uho::NTuple{35,Float64}; nb::Int = 30)
    dM = ntuple(j -> Uho[j] - Ua[j], Val(35))
    θ = theta_star_update_dev(Ua, dM; nb = nb)
    Ustar = ntuple(j -> Ua[j] + θ * dM[j], Val(35))
    return (θ, Ustar)
end

export theta_star_blend_marginal_dev

end # module
