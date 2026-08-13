module KfvsWall
# kfvs_wall.jl — A GENUINE HALF-SPACE (KFVS) FLUX AT A DIFFUSE WALL.
#
# WHY THE GHOST CELL CANNOT DO THIS. A diffuse wall is a condition in VELOCITY SPACE: for
# molecules leaving the domain (v.n > 0 outward) f is whatever the interior supplies, and for
# molecules entering it IS the wall Maxwellian. So f is DISCONTINUOUS at v.n = 0. A ghost cell
# replaces that with one smooth state on the far side and a full-line Riemann problem, which
# has no half-space split at all: at u_n = 0 both physical flux terms vanish and the HLL
# formula leaves pure upwind dissipation acting on the density jump. That is why
# `wall_ghost_dev.jl` had to abandon the physically correct rho_w = rho*sqrt(T/Tw) -- it
# manufactured a mass source with a POSITIVE eigenvalue (interior mass 30 -> 1.9e11, growth
# rate timestep-independent) -- and settle for rho_w = rho, which is stable but discards the
# density adjustment. In Couette the gas is viscously heated so T > Tw permanently, meaning
# the discarded physics is always active.
#
# WHAT MAKES THIS AFFORDABLE HERE. The half-space integral needs a velocity-space
# representation of the interior state, and HyQMOM already carries one:
# `chyqmom_nodes_3d` inverts the 35 moments into non-negative weights and abscissas. Its own
# docstring calls it "the unblocker for a future kinetic flux". Splitting that node set by the
# sign of v.n IS the half-space integral. The method's defining feature is exactly the thing
# the boundary needs, and it has been unused precisely where it matters most.
#
#     F_psi = SUM_{nodes: outward*vn > 0} w_k vn_k psi(v_k)        leaving, from the quadrature
#           + rho_w * H_psi                                        entering, analytic Maxwellian
#
# with rho_w fixed by zero NET MASS FLUX, so conservation is exact BY CONSTRUCTION rather than
# approximately -- the DVM gets 1e-15 this way, against the ghost cell's -3.9e-4.
#
# HONEST LIMITS. The quadrature is 27 nodes for a Gaussian state, so the interior half is
# exact only to that quadrature's accuracy -- this will not reach the DVM's 1e-15 on the
# stress half-moments. `chyqmom_nodes_3d` also documents that M_310 and M_130 are not well
# represented with 3 nodes per direction, and those are stress moments, so the tangential
# momentum flux is the weak link. Expect a large improvement in conservation, not exactness.
#
# AND WHAT THIS WILL NOT FIX. The measured defect that motivated the wall work -- zeta and
# zeta_T falling far too slowly with rarefaction (1.19x against 1.75x, 1.37x against 1.98x) --
# is NOT a wall-representation failure. A degradation experiment in the DVM (collapsing the
# wall-adjacent cells onto their own Gaussian) flattened zeta only from 1.76x to 1.57x, far
# short of the closure's 1.19x, and wrecked the level while doing it. The literature says the
# same: the Knudsen layer appears in moment systems as a superposition of exponential layers
# whose number grows with moment order, errors fall only as inverse powers of that order, and
# R13-type systems both underpredict the Knudsen layer and overpredict slip above Kn ~ 0.4 --
# all of which this closure reproduces. This flux fixes CONSERVATION, which is a real defect
# in its own right (issue #36). It is not expected to fix the trend.
using LinearAlgebra
using ..Riemann35: chyqmom_nodes_3d
export kfvs_wall_flux, halfline_gauss_moments, IJK35_KFVS
# _kfvs_wall_flux_diagonal is intentionally NOT exported: it is the retired model, kept only
# so test_wall_tangential_convention.jl can demonstrate what delegating fixed.

using ..MomentIndices: IJK
using ..Riemann35: nt35
using ..KfvsWallDev
# Canonical table, imported not copied (issue #61). It was a local transcription; the
# table has been mis-transcribed once before -- positions 31-33 permuted, three moments
# mislabelled while every total and trace stayed correct, so nothing caught it.
const IJK35_KFVS = IJK

"""
    halfline_gauss_moments(pmax, u, T, side) -> Vector{Float64}

`I[p+1] = ∫ v^p g(v) dv` over the half line, for a 1D Gaussian of mean `u`, variance `T` and
unit mass. `side = +1` integrates v > 0, `side = -1` integrates v < 0.

Closed form by recursion. With `a = u/sqrt(2T)` and `g0` the density at the cut,

    I_0 = (1/2) erfc(∓a),        I_1 = u I_0 ± T g0,
    I_p = u I_{p-1} + (p-1) T I_{p-2}   for p >= 2, since the boundary term at v = 0
                                        carries a factor 0^{p-1} and vanishes.

Using the recursion rather than 2p separate special functions keeps this exact and cheap; the
p = 1 boundary term is the only place the cut enters.
"""
function halfline_gauss_moments(pmax::Int, u::Float64, T::Float64, side::Int)
    T = max(T, 1e-300)
    s = sqrt(T)
    g0 = exp(-u*u/(2T))/sqrt(2pi*T)          # density at v = 0
    a  = u/(sqrt(2.0)*s)
    I = zeros(pmax+1)
    # erfc without a special-function dependency: erfc(x) = 1 - erf(x)
    I[1] = side > 0 ? 0.5*erfc_(-a) : 0.5*erfc_(a)
    if pmax >= 1
        I[2] = u*I[1] + (side > 0 ? T*g0 : -T*g0)
    end
    for p in 2:pmax
        I[p+1] = u*I[p] + (p-1)*T*I[p-1]     # boundary term ∝ 0^{p-1} = 0
    end
    I
end

"Abramowitz & Stegun 7.1.26 complementary error function, ~1.5e-7 absolute."
@inline function erfc_(x::Float64)
    z = abs(x); t = 1.0/(1.0 + 0.5z)
    y = t*exp(-z*z - 1.26551223 + t*(1.00002368 + t*(0.37409196 + t*(0.09678418 +
        t*(-0.18628806 + t*(0.27886807 + t*(-1.13520398 + t*(1.48851587 +
        t*(-0.82215223 + t*0.17087277)))))))))
    x >= 0.0 ? y : 2.0 - y
end

"Full-line Gaussian moments `⟨v^p⟩` about mean `u`, variance `T`, unit mass."
function fullline_gauss_moments(pmax::Int, u::Float64, T::Float64)
    m = zeros(pmax+1); m[1] = 1.0
    pmax >= 1 && (m[2] = u)
    for p in 2:pmax
        m[p+1] = u*m[p] + (p-1)*T*m[p-1]
    end
    m
end

"""
    kfvs_wall_flux(M, axis, outward, Tw, uw1, uw2) -> (F::NTuple{35,Float64}, rho_w, mdot)

Half-space flux of the 35 moments through a fully diffuse wall on `axis`
(1 = x, 2 = y, 3 = z), with `outward = +1.0` for a hi face and `-1.0` for a lo face.
`uw1`, `uw2` are the tangential wall velocities in cyclic order after `axis`.

`rho_w` is chosen so the NET MASS FLUX is zero; `mdot` is returned so a caller can assert it
rather than trust it. Accommodation is full (alpha = 1); the specular blend belongs at the
call site, where the specular flux is a sign flip of the interior half.
"""
function kfvs_wall_flux(M::AbstractVector{Float64}, axis::Int, outward::Float64,
                        Tw::Float64, uw1::Float64, uw2::Float64)
    # DELEGATES to the device implementation -- see kfvs_wall_flux_full_dev's docstring.
    # This function used to carry its own copy of the half-space integral, and the copy had
    # drifted into a different physical model: it factorised the interior half with a
    # DIAGONAL covariance and dropped cxy/cxz/cyz. The two agreed to 5e-16 whenever the
    # covariance happened to be diagonal and diverged by up to 44% as soon as an off-diagonal
    # was nonzero -- and in Couette the off-diagonal IS the shear stress, i.e. they disagreed
    # most on the one quantity a wall exists to transmit.
    #
    # Nothing in production ever called this function; the GPU march calls the device one
    # directly. So every wall assertion in the suite was guarding a model that no run used,
    # while the model that every run used had no host-side test at all. Delegating costs no
    # production number and puts the tests back onto the code that actually executes.
    F, rho_w = KfvsWallDev.kfvs_wall_flux_full_dev(nt35(M), axis, outward,
                                                  Tw, uw1, uw2)
    return (F, rho_w, F[1])                # F[1] is the net mass flux: must be ~0
end

# The pre-delegation implementation is kept below under a private name, NOT because anything
# should call it, but because `test_kfvs_wall_model.jl` uses it to demonstrate the diagonal
# model's error against the correlated one. Do not use it in new code.
function _kfvs_wall_flux_diagonal(M::AbstractVector{Float64}, axis::Int, outward::Float64,
                                  Tw::Float64, uw1::Float64, uw2::Float64)
    # --- interior half ---------------------------------------------------------------------
    # NOT the raw quadrature. The HyQMOM node set has THREE abscissas per direction, so the
    # outgoing half-space sees exactly ONE of them and the half-space mass flux comes out
    # 27.6% low -- identically at every temperature, because that is a fixed geometric
    # property of a 3-point rule, not a resolvable error. A Gauss rule matches FULL-line
    # moments to 5th order and says essentially nothing about a HALF line; they are different
    # problems. `chyqmom_nodes_3d`'s docstring calls itself "the unblocker for a future
    # kinetic flux", and for the half-space split it is not.
    #
    # So the interior half is integrated from a CONTINUOUS reconstruction instead: the
    # anisotropic Gaussian carrying the interior's rho, u and diagonal second moments, whose
    # half-line moments are analytic (halfline_gauss_moments). That is exact in the
    # equilibrium limit -- rho_w -> rho*sqrt(T/Tw), the physically correct value that
    # wall_ghost_dev.jl had to abandon -- at the cost of discarding the non-Gaussian content
    # the quadrature carries. Given that the flatness defect is NOT a wall-representation
    # failure (measured: a Gaussian wall degradation in the DVM moves zeta 1.76x -> 1.57x
    # against the closure's 1.19x), buying the correct equilibrium limit with non-Gaussian
    # content is the right trade here.
    rho = M[1]
    rho > 0.0 || return (ntuple(_ -> 0.0, Val(35)), 0.0, 0.0)
    ux, uy, uz = M[2]/rho, M[6]/rho, M[16]/rho
    cxx = max(M[3]/rho - ux*ux, 1e-14)
    cyy = max(M[10]/rho - uy*uy, 1e-14)
    czz = max(M[20]/rho - uz*uz, 1e-14)
    uvec = (ux, uy, uz); cvec = (cxx, cyy, czz)
    t1_ = axis % 3 + 1; t2_ = t1_ % 3 + 1
    # outgoing half of the interior Gaussian: normal component cut at 0, tangential full-line
    # p up to 5: the FLUX carries an extra v_n beyond the moment's own normal power,
    # so psi = v^4 needs I_5. Requesting only p<=4 read past the array -- masked by the
    # @inbounds below, which returned a denormal (1.9e-309) for moment (0,4,0). Found by
    # cross-checking against the device implementation, not by the 18 assertions, which
    # never touched that moment.
    Io  = halfline_gauss_moments(5, uvec[axis], cvec[axis], outward > 0 ? +1 : -1)
    St1 = fullline_gauss_moments(4, uvec[t1_], cvec[t1_])
    St2 = fullline_gauss_moments(4, uvec[t2_], cvec[t2_])
    Fi = zeros(35)
    @inbounds for (m, (p,q,r)) in enumerate(IJK35_KFVS)
        e = (p,q,r)
        Fi[m] = rho * Io[e[axis]+2] * St1[e[t1_]+1] * St2[e[t2_]+1]
    end

    # --- wall half: an analytic Maxwellian, factorised normal x tangential ------------------
    t1 = axis % 3 + 1; t2 = t1 % 3 + 1               # cyclic tangents
    In  = halfline_gauss_moments(5, 0.0, Tw, outward > 0 ? -1 : +1)   # ENTERING the domain
    Mt1 = fullline_gauss_moments(4, uw1, Tw)
    Mt2 = fullline_gauss_moments(4, uw2, Tw)
    # unit-density wall mass flux (negative: it enters), used to fix rho_w
    # SIGNED, not abs(). On a lo face (outward = -1) the leaving half has v_n < 0 so Io[2] is
    # negative while the entering In[2] is positive; taking abs() of the outflux flips the
    # ratio and produces a NEGATIVE wall density, i.e. a large spurious mass flux. The
    # original tests only exercised outward = +1 and never saw it; the device implementation,
    # which uses the signed form, disagreed and was right.
    rho_w = In[2] != 0.0 ? -rho*Io[2]/In[2] : 0.0

    Fw = zeros(35)
    @inbounds for (m, (p,q,r)) in enumerate(IJK35_KFVS)
        e = (p, q, r)
        pn, pt1, pt2 = e[axis], e[t1], e[t2]
        Fw[m] = rho_w * In[pn+2] * Mt1[pt1+1] * Mt2[pt2+1]   # In[pn+2] carries the extra v_n
    end

    F = ntuple(m -> Fi[m] + Fw[m], Val(35))
    (F, rho_w, F[1])                                  # F[1] is the net mass flux: must be ~0
end

end # module
