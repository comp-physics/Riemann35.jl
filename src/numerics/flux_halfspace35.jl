"""
    flux_halfspace35.jl — the x-flux of a half-space (one-signed) 35-moment stream.

A `+` stream has x-support in [0,∞); a `-` stream in (−∞,0]. In the split axis (x)
each stream is one-signed, so donor-cell upwinding is the EXACT kinetic flux and it
preserves realizability with NO projection under the wave-speed CFL
(prop:halfline-upwind). The transverse (y,z) directions are ordinary full-line
measures, so their fluxes are the UNCHANGED production closure `flux_closure35_dev`
evaluated on the stream's own 35 moments.

Single-source scalar kernel (35 inputs → 35 x-flux outputs), matching the alloc-free
style of `flux_closure_dev.jl`. Canonical M4 ordering (M000..M022).

x-flux of the `+` stream, slot by slot:
  * carried entries (i+j+k ≤ 3): F(M_{ijk}) = M_{i+1,j,k}  — a copy of an input moment;
  * 15 closure entries (i+j+k = 4), one per fixed-(j,k) channel of order
    n_r = 5−(j+k): (0,0) → marginal h₅; else the h-functional Hankel-solve closure.

The `-` stream mirrors in x: M_{ijk} → (−1)^i M_{ijk}, apply the `+` closure, then
map the flux back with F_{ijk} → (−1)^{i+1}·(mirrored value).
"""
module FluxHalfspace35

using ..Chain: chain, clip_chain, VAC_CHAIN
using ..HalflineClosure: hseq, chan_closure, marg_closure, xspeed, gauss_nodes
using ..FluxClosureDev: flux_closure35_dev

export xflux_plus35, xflux_minus35, stream_flux_plus35, stream_flux_minus35,
       stream_xspeed, CHAIN_CLIPS, reset_chain_clips!, chain_clips

# chain-clip diagnostic (CPU; one thread per rank). ~zero away from vacuum fronts.
const CHAIN_CLIPS = Ref(0)
reset_chain_clips!() = (CHAIN_CLIPS[] = 0)
chain_clips() = CHAIN_CLIPS[]

# per-slot x-parity sign (−1)^{i}, i = x-exponent of moment slot q (M4 order).
const _XSIGN = (
    1, -1, 1, -1, 1,   1, -1, 1, -1,   1, -1, 1,   1, -1,   1,
    1, -1, 1, -1,   1, -1, 1,   1, -1,   1,
    1, -1, 1,   1, -1,   1,   1, -1,   1,   1)

const _ZERO35 = ntuple(_ -> 0.0, Val(35))

# ---------------------------------------------------------------------------
# x-flux of a `+` stream (35 raw moments in canonical M4 order → 35 x-flux outputs).
# ---------------------------------------------------------------------------
@inline function xflux_plus35(
        M000,M100,M200,M300,M400,M010,M110,M210,M310,M020,M120,M220,M030,M130,M040,
        M001,M101,M201,M301,M002,M102,M202,M003,M103,M004,M011,M111,M211,M021,M121,
        M031,M012,M112,M013,M022)

    (isfinite(M000) && M000 > VAC_CHAIN) || return _ZERO35
    # (a cheap finiteness guard on the marginal; full-vector check done by caller)
    (isfinite(M100) && isfinite(M200) && isfinite(M300) && isfinite(M400)) || return _ZERO35

    zraw = chain((M000, M100, M200, M300, M400))
    z, nc = clip_chain(zraw)
    nc > 0 && (CHAIN_CLIPS[] += nc)
    z1, z2, z3, z4 = z
    h = hseq(M000, z1, z2, z3, z4)
    # n_r-point Gauss nodes of the h-functional (computed once per cell, reused
    # across every channel of that order — stable node-form channel closure).
    nd1 = gauss_nodes(z1, z2, z3, z4, Val(1))
    nd2 = gauss_nodes(z1, z2, z3, z4, Val(2))
    nd3 = gauss_nodes(z1, z2, z3, z4, Val(3))
    nd4 = gauss_nodes(z1, z2, z3, z4, Val(4))

    # --- closure entries (i+j+k = 4) ---
    F500 = marg_closure(h)                             # channel (0,0), n_r=5
    F410 = chan_closure(nd4, (M010, M110, M210, M310)) # (1,0), n_r=4
    F320 = chan_closure(nd3, (M020, M120, M220))       # (2,0), n_r=3
    F230 = chan_closure(nd2, (M030, M130))             # (3,0), n_r=2
    F140 = chan_closure(nd1, (M040,))                  # (4,0), n_r=1
    F401 = chan_closure(nd4, (M001, M101, M201, M301)) # (0,1), n_r=4
    F302 = chan_closure(nd3, (M002, M102, M202))       # (0,2), n_r=3
    F203 = chan_closure(nd2, (M003, M103))             # (0,3), n_r=2
    F104 = chan_closure(nd1, (M004,))                  # (0,4), n_r=1
    F311 = chan_closure(nd3, (M011, M111, M211))       # (1,1), n_r=3
    F221 = chan_closure(nd2, (M021, M121))             # (2,1), n_r=2
    F131 = chan_closure(nd1, (M031,))                  # (3,1), n_r=1
    F212 = chan_closure(nd2, (M012, M112))             # (1,2), n_r=2
    F113 = chan_closure(nd1, (M013,))                  # (1,3), n_r=1
    F122 = chan_closure(nd1, (M022,))                  # (2,2), n_r=1

    # --- assemble Fx (carried entries are input moments with x-index raised) ---
    return (
        M100, M200, M300, M400, F500,          # M000 M100 M200 M300 M400
        M110, M210, M310, F410,                # M010 M110 M210 M310
        M120, M220, F320,                      # M020 M120 M220
        M130, F230,                            # M030 M130
        F140,                                  # M040
        M101, M201, M301, F401,                # M001 M101 M201 M301
        M102, M202, F302,                      # M002 M102 M202
        M103, F203,                            # M003 M103
        F104,                                  # M004
        M111, M211, F311,                      # M011 M111 M211
        M121, F221,                            # M021 M121
        F131,                                  # M031
        M112, F212,                            # M012 M112
        F113,                                  # M013
        F122)                                  # M022
end

@inline xflux_plus35(M::NTuple{35}) = xflux_plus35(M...)

# ---------------------------------------------------------------------------
# x-flux of a `-` stream: mirror in x, close as `+`, mirror the flux back.
# ---------------------------------------------------------------------------
@inline function xflux_minus35(M::NTuple{35})
    Mp = ntuple(q -> _XSIGN[q] * M[q], Val(35))     # mirror input
    Fp = xflux_plus35(Mp...)
    return ntuple(q -> -_XSIGN[q] * Fp[q], Val(35)) # mirror flux back: (−1)^{i+1}
end
@inline xflux_minus35(M000,M100,M200,M300,M400,M010,M110,M210,M310,M020,M120,M220,M030,M130,M040,
        M001,M101,M201,M301,M002,M102,M202,M003,M103,M004,M011,M111,M211,M021,M121,
        M031,M012,M112,M013,M022) =
    xflux_minus35((M000,M100,M200,M300,M400,M010,M110,M210,M310,M020,M120,M220,M030,M130,M040,
        M001,M101,M201,M301,M002,M102,M202,M003,M103,M004,M011,M111,M211,M021,M121,
        M031,M012,M112,M013,M022))

# ---------------------------------------------------------------------------
# Full (Fx,Fy,Fz) flux of a stream: half-space x-flux + UNCHANGED production
# transverse closures. Returns NTuple{105} (1..35 Fx, 36..70 Fy, 71..105 Fz),
# matching flux_closure35_dev's layout so drivers can splice uniformly.
# ---------------------------------------------------------------------------
@inline function stream_flux_plus35(M::NTuple{35})
    Fx = xflux_plus35(M...)
    Fp = flux_closure35_dev(M...)             # production Fx|Fy|Fz on this stream
    return (Fx..., ntuple(q -> Fp[35 + q], Val(35))..., ntuple(q -> Fp[70 + q], Val(35))...)
end

@inline function stream_flux_minus35(M::NTuple{35})
    Fx = xflux_minus35(M)
    Fp = flux_closure35_dev(M...)             # y,z are full-line: no mirror
    return (Fx..., ntuple(q -> Fp[35 + q], Val(35))..., ntuple(q -> Fp[70 + q], Val(35))...)
end

# ---------------------------------------------------------------------------
# CFL wave-speed bound for a stream (largest marginal support node).
# ---------------------------------------------------------------------------
@inline function stream_xspeed(M::NTuple{35})
    (isfinite(M[1]) && M[1] > VAC_CHAIN) || return 0.0
    z, _ = clip_chain(chain((M[1], M[2], M[3], M[4], M[5])))
    return xspeed(z[1], z[2], z[3], z[4])
end

end # module
