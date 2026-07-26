"""
    wall_ghost_dev.jl — Maxwell-accommodating wall ghost state, alloc-free and device-safe.

A wall condition is a HALF-SPACE moment problem: the outgoing half of the distribution is
prescribed by the wall, the incoming half comes from the interior. A 35-moment vector
cannot encode that split, so the ghost-cell form used here is an approximation — but a
principled one, and two of its three ingredients are exact.

## Maxwell accommodation as a convex blend

    M_ghost = alpha * M_Maxwellian(rho_w, u_w, T_w) + (1 - alpha) * M_specular(M_int)

* `alpha = 0` — **fully specular, EXACT.** Reflection maps `v_n -> -v_n`, so
  `M_{ijk} -> (-1)^i M_{ijk}` for a wall with normal along axis `i`: a pure sign flip on
  moments of odd normal order. It is a measure-preserving map, so it takes a realizable
  state to a realizable state exactly, at no cost. The tangential velocity passes through
  untouched, which is the free-slip property.
* `alpha = 1` — **fully diffuse.** The wall emits a Maxwellian at `(rho_w, u_w, T_w)`.
  `rho_w` follows from zero net normal mass flux: the incoming flux from the interior must
  equal the outgoing flux from the wall,
      integral_{v_n<0} |v_n| f_int dv  ==  rho_w * sqrt(T_w / 2pi).
  Approximating `f_int` by the Gaussian with the cell's `(rho, u, C)` makes the left side
  closed-form in `erf`/`exp`. THIS Gaussian assumption is the one real approximation in
  the design.
* Both terms are realizable, so the blend is realizable — the same convexity argument
  `bgk_relax_tup` and the theta*-IDP limiter already rest on. No projection needed and no
  new failure mode.

## Scope

Grid-aligned faces, stationary in the normal direction (tangential wall motion is
allowed, which covers Couette / Fourier / planar microchannel). Immersed or curved walls,
and normally-moving walls, are out of scope.

The half-space Gaussian assumption means the slip coefficient may not match Maxwell slip
theory. `test/validate_wall_slip.jl` MEASURES it rather than assuming it; if it is wrong,
the escalation is a prescribed wall flux (KFVS-at-wall), which would touch the residual.
"""
module WallGhostDev

using ..ReconDev: from_recon_vars_dev

export wall_ghost_tup, halfspace_influx, WALL_AXIS_X, WALL_AXIS_Y, WALL_AXIS_Z

const WALL_AXIS_X = 1
const WALL_AXIS_Y = 2
const WALL_AXIS_Z = 3

# Moments of odd order in each axis, in the canonical 35-moment (i,j,k) layout. Specular
# reflection about axis a negates exactly these. Precomputed as sign tuples so the device
# path is a multiply, not a branch.
const _SGN_X = (1.0,-1.0,1.0,-1.0,1.0, 1.0,-1.0,1.0,-1.0, 1.0,-1.0,1.0, 1.0,-1.0, 1.0,
                1.0,-1.0,1.0,-1.0, 1.0,-1.0,1.0, 1.0,-1.0, 1.0, 1.0,-1.0,1.0, 1.0,-1.0,
                1.0, 1.0,-1.0, 1.0, 1.0)
const _SGN_Y = (1.0,1.0,1.0,1.0,1.0, -1.0,-1.0,-1.0,-1.0, 1.0,1.0,1.0, -1.0,-1.0, 1.0,
                1.0,1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0, 1.0, -1.0,-1.0,-1.0, 1.0,1.0,
                -1.0, -1.0,-1.0, -1.0, 1.0)
const _SGN_Z = (1.0,1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0, 1.0,1.0, 1.0,
                -1.0,-1.0,-1.0,-1.0, 1.0,1.0,1.0, -1.0,-1.0, 1.0, -1.0,-1.0,-1.0, -1.0,-1.0,
                -1.0, 1.0,1.0, -1.0, 1.0)

@inline _sgn(axis::Int) = axis == 1 ? _SGN_X : (axis == 2 ? _SGN_Y : _SGN_Z)

"""
    specular_tup(M, axis) -> NTuple{35,Float64}

Specular reflection about `axis`. EXACT: a sign flip on moments of odd normal order,
measure-preserving, hence realizability-preserving with no approximation.
"""
@inline function specular_tup(M::NTuple{35,Float64}, axis::Int)::NTuple{35,Float64}
    s = _sgn(axis)
    ntuple(i -> s[i] * M[i], Val(35))
end

"""
    halfspace_influx(rho, un, sig) -> Float64

Mass flux of a 1-D Gaussian `N(un, sig^2)` of density `rho` crossing toward the wall,
i.e. `integral_{v<0} |v| f dv` for a wall whose OUTWARD normal is `+n`:

    rho * [ sig/sqrt(2pi) * exp(-s^2) - (un/2) * erfc(s/sqrt(2)) ] ,   s = un/sig

Both terms are needed: the first is the thermal flux, the second the drift correction.
Verified against numerical quadrature in `test/test_wall_bc.jl`.
"""
@inline function halfspace_influx(rho::Float64, un::Float64, sig::Float64)::Float64
    sig <= 0.0 && return 0.0
    s = un / sig
    # erfc(x) via erf; both are device-available
    f = sig * 0.3989422804014327 * exp(-0.5*s*s) - 0.5*un*erfc_dev(s * 0.7071067811865476)
    rho * (f > 0.0 ? f : 0.0)
end

"Abramowitz-Stegun 7.1.26-class erfc, fp64, device-safe (no libm erfc on all targets)."
@inline function erfc_dev(x::Float64)::Float64
    z = abs(x)
    t = 1.0 / (1.0 + 0.5*z)
    y = t * exp(-z*z - 1.26551223 + t*(1.00002368 + t*(0.37409196 + t*(0.09678418 +
        t*(-0.18628806 + t*(0.27886807 + t*(-1.13520398 + t*(1.48851587 +
        t*(-0.82215223 + t*0.17087277)))))))))
    x >= 0.0 ? y : 2.0 - y
end

"""
    wall_ghost_tup(M, axis, outward, Tw, uw1, uw2, alpha) -> NTuple{35,Float64}

Ghost state for a Maxwell-accommodating wall on `axis`, with `outward = +1` if the wall's
outward normal points along +axis (a hi face) and `-1` for a lo face. `uw1`/`uw2` are the
two TANGENTIAL wall velocity components in cyclic order after `axis`. `alpha` in [0,1].

`alpha = 0` returns the exact specular reflection and ignores `Tw`/`uw`.
"""
@inline function wall_ghost_tup(M::NTuple{35,Float64}, axis::Int, outward::Float64,
                                Tw::Float64, uw1::Float64, uw2::Float64,
                                alpha::Float64)::NTuple{35,Float64}
    rho = M[1]
    rho > 0.0 || return M
    Msp = specular_tup(M, axis)
    alpha <= 0.0 && return Msp                      # exact specular; no Gaussian needed

    u = M[2]/rho; v = M[6]/rho; w = M[16]/rho
    C200 = M[3]/rho  - u*u
    C020 = M[10]/rho - v*v
    C002 = M[20]/rho - w*w
    # normal-direction mean and sigma
    un   = axis == 1 ? u : (axis == 2 ? v : w)
    Cnn  = axis == 1 ? C200 : (axis == 2 ? C020 : C002)
    Cnn  = Cnn > 1e-14 ? Cnn : 1e-14
    sig  = sqrt(Cnn)
    # flux toward the wall: for an outward normal +n the incoming particles have v_n > 0,
    # so flip the sign of the drift for a hi face.
    influx = halfspace_influx(rho, outward * un, sig)
    Tw_    = Tw > 1e-14 ? Tw : 1e-14
    rho_w  = influx / sqrt(Tw_ * 0.15915494309189535)   # / sqrt(Tw/(2pi))
    rho_w  = rho_w > 0.0 ? rho_w : rho

    # wall Maxwellian: no normal velocity, tangential = wall velocity
    uwx = axis == 1 ? 0.0 : (axis == 2 ? uw2 : uw1)
    uwy = axis == 1 ? uw1 : (axis == 2 ? 0.0 : uw2)
    uwz = axis == 1 ? uw2 : (axis == 2 ? uw1 : 0.0)
    MG = from_recon_vars_dev(rho_w, uwx, uwy, uwz, Tw_, Tw_, Tw_,
        0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    # convex blend of two realizable states -> realizable, no projection required
    om = 1.0 - alpha
    ntuple(i -> alpha*MG[i] + om*Msp[i], Val(35))
end

end # module
