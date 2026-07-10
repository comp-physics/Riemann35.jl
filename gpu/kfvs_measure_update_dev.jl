"""
    kfvs_measure_update_dev.jl — the 3D kinetic-FVS realizable anchor (device).

Increment B of the KFVS anchor. Two device pieces, both **PURE ADDITIONS** (nothing
in the solver path calls them; projection35 / residual untouched):

1. **Storage pass** (`store4!` + host kernel `chyqmom_store_kernel!`, see
   gpu/validation/gpu_kfvs_storage.jl): invert every cell of a 3D field via
   `KFVSInversionDev.chyqmom_nodes_3d_store_dev!` and store its quadrature (weights
   + 3 abscissa components, ≤27 nodes) into a device array laid out over
   `(node, 4, nx, ny, nz)` plus a `(nx,ny,nz)` UInt8 node-count array. The `4`
   channel is `(w, Ux, Uy, Uz)`.

2. **`measure_update_3d_dev`** — the anchor. For an interior cell C with its 6
   face-neighbors (∓x, ∓y, ∓z), the first-order kinetic-FVS update is the moment
   vector of a NONNEGATIVE measure:

       M^{n+1} =
           Σ_{k∈C} nC_k·(1 − λ(|Ux|+|Uy|+|Uz|)_k) δ_{U_k^C}        (retained)
         + Σ_{k∈Lx, Ux>0}  λ·nk·Ux  δ_{U_k}                        (x inflow from left)
         + Σ_{k∈Rx, Ux<0} −λ·nk·Ux  δ_{U_k}                        (x inflow from right)
         + Σ_{k∈Ly, Uy>0}  λ·nk·Uy  δ  + Σ_{k∈Ry, Uy<0} −λ·nk·Uy  δ  (y)
         + Σ_{k∈Lz, Uz>0}  λ·nk·Uz  δ  + Σ_{k∈Rz, Uz<0} −λ·nk·Uz  δ  (z)

   The CPU reference `verify_kfvs.jl:measure_update` is **x-only**; this GENERALIZES
   it to all 6 neighbors (one upwind inflow per axis, using that axis's abscissa
   component and sign). Under the 3D CFL

       λ · max_k (|Ux|+|Uy|+|Uz|)_k ≤ 1

   the retained weight and every inflow weight are ≥ 0, so the update is the moment
   vector of a nonnegative measure ⇒ realizable in the FULL 35-moment cone by
   construction (design thm:kfvs-idp; Perthame P90 in 3D). The nonnegative-weight
   check is the CHEAP realizability certificate that replaces the Hankel test.

`measure_update_3d_dev` consumes the STORED quadratures of the 7-cell stencil
(center + 6 neighbors) and returns the updated 35-moment `NTuple{35,Float64}` plus
the minimum measure weight (the certificate). fp64, no heap, no closures, device-
compilable. `@fastmath` deliberately OFF.
"""
module KFVSMeasureUpdateDev

include(joinpath(@__DIR__, "chyqmom_nodes_3d_dev.jl"))
using .KFVSInversionDev: chyqmom_nodes_3d_store_dev!, NODEMAX

export measure_update_3d_dev, accum35_node, NODEMAX

# 35-moment exponent triples (i,j,k), standard ordering. Compile-time constant.
const KFVS_TRIPLES = (
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),
 (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))

# small integer power (exponents are 0..4)
@inline function _pw4(x::Float64, e::Int)
    if e == 0; return 1.0
    elseif e == 1; return x
    elseif e == 2; return x*x
    elseif e == 3; return x*x*x
    else; return x*x*x*x
    end
end

# Accumulate a single weighted node (w, ux, uy, uz) into the 35-moment tuple M.
# M_n += w * ux^i * uy^j * uz^k for each (i,j,k). Returns the updated NTuple{35}.
# Unrolled over the 35 fixed triples (compile-time literals).
@inline function accum35_node(M::NTuple{35,Float64}, w::Float64, ux::Float64, uy::Float64, uz::Float64)
    return ntuple(Val(35)) do n
        @inbounds begin
            (i, j, k) = KFVS_TRIPLES[n]
            M[n] + w * _pw4(ux, i) * _pw4(uy, j) * _pw4(uz, k)
        end
    end
end

# ---------------------------------------------------------------------------
# The anchor. Quadratures are passed as node getters so this is storage-layout
# agnostic (the caller closes over the stored device array + a cell's linear
# index). `getw/getx/gety/getz(cellslot, q)` return node q (1..27) of the stencil
# cell identified by `cellslot`:
#     1 = C (center), 2 = Lx, 3 = Rx, 4 = Ly, 5 = Ry, 6 = Lz, 7 = Rz.
# `count(cellslot)` returns that cell's node count.
#
# Returns (M::NTuple{35,Float64}, minw::Float64). Under the CFL the caller enforces
# (λ·max_k(|Ux|+|Uy|+|Uz|) ≤ 1) minw ≥ 0 by the theorem.
# ---------------------------------------------------------------------------
@inline function measure_update_3d_dev(getw::FW, getx::FX, gety::FY, getz::FZ,
                                       cnt::FC, λ::Float64) where {FW,FX,FY,FZ,FC}
    M = ntuple(_ -> 0.0, Val(35))
    minw = Inf

    # --- retained mass at the CENTER cell (slot 1) ---
    NC = cnt(1)
    @inbounds for q in 1:NODEMAX
        (q > NC) && break
        nk = getw(1, q)
        (nk > 0.0) || continue
        ux = getx(1, q); uy = gety(1, q); uz = getz(1, q)
        w = nk * (1.0 - λ * (abs(ux) + abs(uy) + abs(uz)))
        minw = w < minw ? w : minw
        M = accum35_node(M, w, ux, uy, uz)
    end

    # --- x inflow: left neighbor (slot 2), nodes with Ux>0 ---
    NLx = cnt(2)
    @inbounds for q in 1:NODEMAX
        (q > NLx) && break
        nk = getw(2, q); (nk > 0.0) || continue
        ux = getx(2, q)
        (ux > 0.0) || continue
        w = λ * nk * ux
        minw = w < minw ? w : minw
        M = accum35_node(M, w, ux, gety(2, q), getz(2, q))
    end
    # --- x inflow: right neighbor (slot 3), nodes with Ux<0 ---
    NRx = cnt(3)
    @inbounds for q in 1:NODEMAX
        (q > NRx) && break
        nk = getw(3, q); (nk > 0.0) || continue
        ux = getx(3, q)
        (ux < 0.0) || continue
        w = -λ * nk * ux
        minw = w < minw ? w : minw
        M = accum35_node(M, w, ux, gety(3, q), getz(3, q))
    end

    # --- y inflow: left (slot 4) Uy>0, right (slot 5) Uy<0 ---
    NLy = cnt(4)
    @inbounds for q in 1:NODEMAX
        (q > NLy) && break
        nk = getw(4, q); (nk > 0.0) || continue
        uy = gety(4, q); (uy > 0.0) || continue
        w = λ * nk * uy
        minw = w < minw ? w : minw
        M = accum35_node(M, w, getx(4, q), uy, getz(4, q))
    end
    NRy = cnt(5)
    @inbounds for q in 1:NODEMAX
        (q > NRy) && break
        nk = getw(5, q); (nk > 0.0) || continue
        uy = gety(5, q); (uy < 0.0) || continue
        w = -λ * nk * uy
        minw = w < minw ? w : minw
        M = accum35_node(M, w, getx(5, q), uy, getz(5, q))
    end

    # --- z inflow: left (slot 6) Uz>0, right (slot 7) Uz<0 ---
    NLz = cnt(6)
    @inbounds for q in 1:NODEMAX
        (q > NLz) && break
        nk = getw(6, q); (nk > 0.0) || continue
        uz = getz(6, q); (uz > 0.0) || continue
        w = λ * nk * uz
        minw = w < minw ? w : minw
        M = accum35_node(M, w, getx(6, q), gety(6, q), uz)
    end
    NRz = cnt(7)
    @inbounds for q in 1:NODEMAX
        (q > NRz) && break
        nk = getw(7, q); (nk > 0.0) || continue
        uz = getz(7, q); (uz < 0.0) || continue
        w = -λ * nk * uz
        minw = w < minw ? w : minw
        M = accum35_node(M, w, getx(7, q), gety(7, q), uz)
    end

    return (M, minw)
end

end # module
