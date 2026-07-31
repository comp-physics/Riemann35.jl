module KfvsWallDev
# kfvs_wall_dev.jl — the half-space wall flux as an ALLOCATION-FREE DEVICE FUNCTION.
#
# The CPU version in kfvs_wall.jl builds Vectors, which cannot run in a CUDA kernel. This is
# the same mathematics in NTuple form: erfc, exp, sqrt and two short recursions, nothing else.
# `kfvs_wall.jl` is kept as the readable reference and is checked against this one, so the
# device path is not a second implementation that can silently drift.
#
# The reconstruction is the anisotropic Gaussian carrying the interior's rho, u and diagonal
# second moments -- NOT the HyQMOM quadrature. Three abscissas per direction put exactly one
# node in the outgoing half-space and get the half-space mass flux 27.6% low at every
# temperature; that is a geometric property of a 3-point rule, not a resolvable error.
export kfvs_wall_flux_dev

const IJK35_W = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
                 (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),
                 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),
                 (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))

"Abramowitz & Stegun 7.1.26, ~1.5e-7 absolute. Device-safe: no branches on globals."
@inline function _erfc(x::Float64)
    z = abs(x); t = 1.0/(1.0 + 0.5z)
    y = t*exp(-z*z - 1.26551223 + t*(1.00002368 + t*(0.37409196 + t*(0.09678418 +
        t*(-0.18628806 + t*(0.27886807 + t*(-1.13520398 + t*(1.48851587 +
        t*(-0.82215223 + t*0.17087277)))))))))
    x >= 0.0 ? y : 2.0 - y
end

"Half-line Gaussian moments p = 0..5 as a tuple. `side = +1` integrates v > 0, `-1` v < 0.
 I_0 = erfc/2; I_1 = u I_0 ± T g(0); I_p = u I_{p-1} + (p-1) T I_{p-2} thereafter, the
 boundary term vanishing because it carries 0^{p-1}."
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

"Full-line Gaussian moments p = 0..4."
@inline function _full(u::Float64, T::Float64)
    m0 = 1.0; m1 = u
    m2 = u*m1 + 1*T*m0
    m3 = u*m2 + 2*T*m1
    m4 = u*m3 + 3*T*m2
    (m0, m1, m2, m3, m4)
end

@inline _at6(t::NTuple{6,Float64}, i::Int) =
    i == 0 ? t[1] : i == 1 ? t[2] : i == 2 ? t[3] : i == 3 ? t[4] : i == 4 ? t[5] : t[6]
@inline _at5(t::NTuple{5,Float64}, i::Int) =
    i == 0 ? t[1] : i == 1 ? t[2] : i == 2 ? t[3] : i == 3 ? t[4] : t[5]

"""
    kfvs_wall_flux_dev(M, axis, outward, Tw, uw1, uw2) -> NTuple{35,Float64}

Half-space flux through a fully diffuse wall. `outward = +1.0` on a hi face, `-1.0` on a lo
face. `uw1`, `uw2` are tangential in cyclic order after `axis` (for a y-wall: uw1 -> z,
uw2 -> x, matching the march's convention).

`rho_w` is set FROM the zero-net-mass-flux condition, so mass conservation is exact by
construction rather than approximate -- which is the whole point, since no ghost density can
cancel the leak the ghost-cell form produces (#36).
"""
@inline function kfvs_wall_flux_dev(M::NTuple{35,Float64}, axis::Int, outward::Float64,
                                    Tw::Float64, uw1::Float64, uw2::Float64)
    rho = M[1]
    rho > 0.0 || return ntuple(_ -> 0.0, Val(35))
    ux = M[2]/rho; uy = M[6]/rho; uz = M[16]/rho
    cxx = M[3]/rho  - ux*ux; cxx = cxx > 1e-14 ? cxx : 1e-14
    cyy = M[10]/rho - uy*uy; cyy = cyy > 1e-14 ? cyy : 1e-14
    czz = M[20]/rho - uz*uz; czz = czz > 1e-14 ? czz : 1e-14

    t1 = axis % 3 + 1; t2 = t1 % 3 + 1
    un = axis == 1 ? ux : axis == 2 ? uy : uz
    cn = axis == 1 ? cxx : axis == 2 ? cyy : czz
    ut1 = t1 == 1 ? ux : t1 == 2 ? uy : uz
    ct1 = t1 == 1 ? cxx : t1 == 2 ? cyy : czz
    ut2 = t2 == 1 ? ux : t2 == 2 ? uy : uz
    ct2 = t2 == 1 ? cxx : t2 == 2 ? cyy : czz

    # interior half: molecules LEAVING (outward*vn > 0)
    Io  = _half(un, cn, outward > 0 ? 1 : -1)
    St1 = _full(ut1, ct1)
    St2 = _full(ut2, ct2)

    # wall half: molecules ENTERING, an isotropic Maxwellian at Tw moving with the wall
    Ii  = _half(0.0, Tw, outward > 0 ? -1 : 1)
    Wt1 = _full(uw1, Tw)
    Wt2 = _full(uw2, Tw)

    # rho_w from zero net mass flux:  rho*Io[1] + rho_w*Ii[1] = 0   (index 1 => moment p=1)
    rho_w = Ii[2] != 0.0 ? -rho*Io[2]/Ii[2] : 0.0

    ntuple(Val(35)) do m
        e = IJK35_W[m]
        pn  = e[axis]; q1 = e[t1]; q2 = e[t2]
        rho*_at6(Io, pn+1)*_at5(St1, q1)*_at5(St2, q2) +
        rho_w*_at6(Ii, pn+1)*_at5(Wt1, q1)*_at5(Wt2, q2)
    end
end

end # module
