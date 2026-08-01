module KfvsWallDev
# kfvs_wall_dev.jl — the half-space wall flux as an ALLOCATION-FREE DEVICE FUNCTION.
#
# The CPU version in kfvs_wall.jl builds Vectors, which cannot run in a CUDA kernel. This is
# the same mathematics in NTuple form: erfc, exp, sqrt and short recursions, nothing else.
#
# ---------------------------------------------------------------------------------------
# THE INTERIOR HALF IS A CORRELATED GAUSSIAN, and the correlation is the whole point.
# ---------------------------------------------------------------------------------------
# The first version reconstructed the interior from rho, u and the DIAGONAL second moments
# only -- 7 of the 35 -- because a diagonal Gaussian factorises into independent 1D integrals
# and the half-line moments are then trivially analytic. That is exactly wrong for a wall.
#
# A wall transmits TANGENTIAL MOMENTUM, and the flux that carries it is
# int_{vn<0} v_n v_t f dv. In a diagonal Gaussian v_n and v_t are INDEPENDENT, so that
# integral collapses to <v_n>_half * u_t and the shear stress -- which IS the off-diagonal
# C_nt -- contributes nothing at all. Measured against the DVM's exact half-space flux:
#
#     Kn      exact       diagonal-Gaussian    error        C_nt/(rho T)
#     0.038   0.030571    0.035698             +16.8%       -0.0092
#     0.142   0.014431    0.029292            +103.0%       -0.0254
#     0.730  -0.014243    0.016916            -218.8%       -0.0543
#
# At Kn = 0.73 the reconstruction gets the tangential momentum flux wrong in SIGN. That is
# the residual 5-7% slip error the ghost-cell comparison left behind, and it is an omission
# rather than a modelling limit.
#
# THE FIX needs no new model and no free parameter: use the FULL anisotropic Gaussian, ten
# moments instead of seven. A Gaussian conditioned on the normal component has
#
#     <v_t | v_n> = u_t + (C_nt/C_nn)(v_n - u_n),   Var(v_t | v_n) = C_tt - C_nt^2/C_nn
#
# so every half-space moment is still analytic -- the same half-line recursion, with the
# tangential mean now sliding linearly with v_n. Each extra power of v_t contributes one more
# term in (v_n - u_n), which the recursion already supplies.
#
# WHAT WAS RULED OUT FIRST, so it is not retried. The near-wall distribution was expected to
# be BIMODAL (two counter-propagating wall beams) and to need a two-beam reconstruction.
# Measured, it is not: the NORMAL-velocity distribution is Gaussian to skew = -0.001 and
# excess kurtosis = 0.0003 at every Kn, and the Gaussian normal half-flux is only 0.6% off.
# In low-speed Couette the two wall Maxwellians differ in their TANGENTIAL velocity only, so
# the bimodality never appears in the direction the half-space split cuts.
using ..MomentIndices: IJK
export kfvs_wall_flux_dev

# The moment-index table is the CANONICAL one, imported rather than copied. It used to be
# a local transcription; the table has been mis-transcribed once before (positions 31-33
# permuted, three moments mislabelled while every total and trace stayed correct), which
# is why nine copies existed and why they are being removed. See issue #61.
const IJK35_W = IJK

"Abramowitz & Stegun 7.1.26, ~1.5e-7 absolute."
@inline function _erfc(x::Float64)
    z = abs(x); t = 1.0/(1.0 + 0.5z)
    y = t*exp(-z*z - 1.26551223 + t*(1.00002368 + t*(0.37409196 + t*(0.09678418 +
        t*(-0.18628806 + t*(0.27886807 + t*(-1.13520398 + t*(1.48851587 +
        t*(-0.82215223 + t*0.17087277)))))))))
    x >= 0.0 ? y : 2.0 - y
end

"Half-line moments of a 1D Gaussian, p = 0..5, ABOUT ZERO. `side = +1` integrates v > 0."
@inline function _half(u::Float64, T::Float64, side::Int)
    Tp = T > 1e-300 ? T : 1e-300
    g0 = exp(-u*u/(2Tp))/sqrt(2pi*Tp)
    a  = u/(sqrt(2.0)*sqrt(Tp))
    I0 = side > 0 ? 0.5*_erfc(-a) : 0.5*_erfc(a)
    I1 = u*I0 + (side > 0 ? Tp*g0 : -Tp*g0)
    I2 = u*I1 + 1*Tp*I0
    I3 = u*I2 + 2*Tp*I1
    I4 = u*I3 + 3*Tp*I2
    I5 = u*I4 + 4*Tp*I3
    (I0, I1, I2, I3, I4, I5)
end

"Full-line central moments of a 1D Gaussian, p = 0..4, about mean `u`."
@inline function _full(u::Float64, T::Float64)
    m0 = 1.0; m1 = u
    m2 = u*m1 + 1*T*m0
    m3 = u*m2 + 2*T*m1
    m4 = u*m3 + 3*T*m2
    (m0, m1, m2, m3, m4)
end

"Central moments about ZERO mean, p = 0..4 -- for the conditional spread."
@inline function _cen(T::Float64)
    (1.0, 0.0, T, 0.0, 3T*T)
end

@inline _at6(t::NTuple{6,Float64}, i::Int) =
    i == 0 ? t[1] : i == 1 ? t[2] : i == 2 ? t[3] : i == 3 ? t[4] : i == 4 ? t[5] : t[6]
@inline _at5(t::NTuple{5,Float64}, i::Int) =
    i == 0 ? t[1] : i == 1 ? t[2] : i == 2 ? t[3] : i == 3 ? t[4] : t[5]
@inline _binom(n::Int, k::Int) =
    k == 0 ? 1.0 : k == n ? 1.0 :
    (n == 2 ? 2.0 : n == 3 ? (k == 1 ? 3.0 : 3.0) :
     n == 4 ? (k == 1 ? 4.0 : k == 2 ? 6.0 : 4.0) : 1.0)

"""
    kfvs_wall_flux_dev(M, axis, outward, Tw, uw1, uw2) -> NTuple{35,Float64}

Half-space flux through a fully diffuse wall. `outward = +1.0` on a hi face, `-1.0` on a lo
face. `uw1`, `uw2` are tangential in cyclic order after `axis`.

The interior half uses the full anisotropic Gaussian INCLUDING the off-diagonal second
moments, so the tangential momentum flux -- which the shear stress carries and a diagonal
reconstruction zeroes -- is represented. `rho_w` is set from zero net mass flux, so mass
conservation stays exact by construction.
"""
@inline function kfvs_wall_flux_dev(M::NTuple{35,Float64}, axis::Int, outward::Float64,
                                    Tw::Float64, uw1::Float64, uw2::Float64)
    rho = M[1]
    rho > 0.0 || return ntuple(_ -> 0.0, Val(35))
    ux = M[2]/rho; uy = M[6]/rho; uz = M[16]/rho
    cxx = M[3]/rho  - ux*ux; cxx = cxx > 1e-14 ? cxx : 1e-14
    cyy = M[10]/rho - uy*uy; cyy = cyy > 1e-14 ? cyy : 1e-14
    czz = M[20]/rho - uz*uz; czz = czz > 1e-14 ? czz : 1e-14
    cxy = M[7]/rho  - ux*uy          # (1,1,0)  -- the off-diagonals the first version dropped
    cxz = M[17]/rho - ux*uz          # (1,0,1)
    cyz = M[26]/rho - uy*uz          # (0,1,1)

    t1 = axis % 3 + 1; t2 = t1 % 3 + 1
    un  = axis == 1 ? ux : axis == 2 ? uy : uz
    cnn = axis == 1 ? cxx : axis == 2 ? cyy : czz
    ut1 = t1 == 1 ? ux : t1 == 2 ? uy : uz
    ct1 = t1 == 1 ? cxx : t1 == 2 ? cyy : czz
    ut2 = t2 == 1 ? ux : t2 == 2 ? uy : uz
    ct2 = t2 == 1 ? cxx : t2 == 2 ? cyy : czz
    # C_{axis,t1} and C_{axis,t2} picked out of the symmetric matrix
    cn1 = (axis == 1 && t1 == 2) || (axis == 2 && t1 == 1) ? cxy :
          (axis == 1 && t1 == 3) || (axis == 3 && t1 == 1) ? cxz : cyz
    cn2 = (axis == 1 && t2 == 2) || (axis == 2 && t2 == 1) ? cxy :
          (axis == 1 && t2 == 3) || (axis == 3 && t2 == 1) ? cxz : cyz

    # conditioning on v_n: the tangential mean slides with v_n and the spread shrinks
    b1 = cn1/cnn; b2 = cn2/cnn
    s1 = ct1 - b1*cn1; s1 = s1 > 1e-14 ? s1 : 1e-14
    s2 = ct2 - b2*cn2; s2 = s2 > 1e-14 ? s2 : 1e-14

    Io = _half(un, cnn, outward > 0 ? 1 : -1)    # interior, LEAVING
    Ii = _half(0.0, Tw, outward > 0 ? -1 : 1)    # wall, ENTERING
    Wt1 = _full(uw1, Tw); Wt2 = _full(uw2, Tw)
    R1 = _cen(s1); R2 = _cen(s2)                 # conditional spread about the sliding mean

    rho_w = Ii[2] != 0.0 ? -rho*Io[2]/Ii[2] : 0.0

    ntuple(Val(35)) do m
        e = IJK35_W[m]
        pn = e[axis]; q1 = e[t1]; q2 = e[t2]
        # E[v_n^pn v_t1^q1 v_t2^q2] over the half-space. Conditioned on v_n, each tangential
        # factor is Gaussian with mean ut + b(v_n - un), so expand in powers of (v_n - un):
        #   E[v_t^q | v_n] = sum_k C(q,k) (ut + b(v_n-un))^(q-k) * centralmoment_k
        # and each power of (v_n - un) folds back into the half-line recursion in v_n.
        acc = 0.0
        for k1 in 0:q1, k2 in 0:q2
            (isodd(k1) || isodd(k2)) && continue          # odd central moments vanish
            w1 = _binom(q1, k1)*_at5(R1, k1)
            w2 = _binom(q2, k2)*_at5(R2, k2)
            n1 = q1 - k1; n2 = q2 - k2
            # (ut1 + b1*(vn-un))^n1 * (ut2 + b2*(vn-un))^n2 expanded and integrated in vn
            for a1 in 0:n1, a2 in 0:n2
                ca = _binom(n1, a1)*_binom(n2, a2) *
                     (ut1 - b1*un)^(n1-a1) * (ut2 - b2*un)^(n2-a2) * b1^a1 * b2^a2
                ca == 0.0 && continue
                acc += w1*w2*ca*_at6(Io, pn + a1 + a2 + 1)
            end
        end
        rho*acc + rho_w*_at6(Ii, pn+1)*_at5(Wt1, q1)*_at5(Wt2, q2)
    end
end

end # module
