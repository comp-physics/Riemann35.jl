"""
    standardized_to_M4(M000, umean, vmean, wmean, C200, C020, C002,
                       S300,S400,S110,S210,S310,S120,S220,S030,S130,S040,
                       S101,S201,S301,S102,S202,S003,S103,S004,
                       S011,S111,S211,S021,S121,S031,S012,S112,S013,S022)
        -> Vector{Float64}

Shared helper: given density, mean velocities, second-order central-moment
variances (C200, C020, C002), and the 28 standardized higher moments, reconstruct
the length-35 raw-moment vector.

This performs the S->C assignment block (scaling standardized->central) followed
by a call to C4toM4_3D, and extracts the 35 raw moments in canonical layout.

Used by both `from_recon_vars` (reconstruction-variable bijection) and
`realizable_3D_M4` (realizability correction), ensuring a single source of truth
for this reconstruction.
"""
function standardized_to_M4(M000::Real, umean::Real, vmean::Real, wmean::Real,
                             C200::Real, C020::Real, C002::Real,
                             S300::Real, S400::Real, S110::Real, S210::Real, S310::Real,
                             S120::Real, S220::Real, S030::Real, S130::Real, S040::Real,
                             S101::Real, S201::Real, S301::Real, S102::Real, S202::Real,
                             S003::Real, S103::Real, S004::Real,
                             S011::Real, S111::Real, S211::Real, S021::Real, S121::Real,
                             S031::Real, S012::Real, S112::Real, S013::Real, S022::Real)::Vector{Float64}
    sC200 = sqrt(C200); sC020 = sqrt(C020); sC002 = sqrt(C002)
    C110 = S110*sC200*sC020
    C101 = S101*sC200*sC002
    C011 = S011*sC020*sC002
    C300 = S300*sC200*C200
    C210 = S210*C200*sC020
    C201 = S201*C200*sC002
    C120 = S120*sC200*C020
    C111 = S111*sC200*sC020*sC002
    C102 = S102*sC200*C002
    C030 = S030*sC020*C020
    C021 = S021*C020*sC002
    C012 = S012*sC020*C002
    C003 = S003*sC002*C002
    C400 = S400*C200^2
    C310 = S310*sC200*C200*sC020
    C301 = S301*sC200*C200*sC002
    C220 = S220*C200*C020
    C211 = S211*C200*sC020*sC002
    C202 = S202*C200*C002
    C130 = S130*sC200*sC020*C020
    C121 = S121*sC200*C020*sC002
    C112 = S112*sC200*sC020*C002
    C103 = S103*sC200*sC002*C002
    C040 = S040*C020^2
    C031 = S031*sC020*C020*sC002
    C022 = S022*C020*C002
    C013 = S013*sC020*sC002*C002
    C004 = S004*C002^2

    M5 = C4toM4_3D(M000, umean, vmean, wmean, C200, C110, C101, C020, C011, C002,
                   C300, C210, C201, C120, C111, C102, C030, C021, C012, C003,
                   C400, C310, C301, C220, C211, C202, C130, C121, C112, C103,
                   C040, C031, C022, C013, C004)

    return [M5[1,1,1], M5[2,1,1], M5[3,1,1], M5[4,1,1], M5[5,1,1],
            M5[1,2,1], M5[2,2,1], M5[3,2,1], M5[4,2,1],
            M5[1,3,1], M5[2,3,1], M5[3,3,1],
            M5[1,4,1], M5[2,4,1],
            M5[1,5,1],
            M5[1,1,2], M5[2,1,2], M5[3,1,2], M5[4,1,2],
            M5[1,1,3], M5[2,1,3], M5[3,1,3],
            M5[1,1,4], M5[2,1,4],
            M5[1,1,5],
            M5[1,2,2], M5[2,2,2], M5[3,2,2],
            M5[1,3,2], M5[2,3,2],
            M5[1,4,2],
            M5[1,2,3], M5[2,2,3],
            M5[1,2,4],
            M5[1,3,3]]
end

"minmod slope limiter"
@inline function minmod(a::Real, b::Real)
    (a*b <= 0) ? 0.0 : (abs(a) < abs(b) ? Float64(a) : Float64(b))
end

"Per-component limited slope for cell V0 given neighbors Vm1, Vp1."
function muscl_slopes(Vm1::AbstractVector, V0::AbstractVector, Vp1::AbstractVector; limiter=minmod)
    n = length(V0)
    s = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        s[k] = limiter(V0[k]-Vm1[k], Vp1[k]-V0[k])
    end
    return s
end

"Left/right face recon-var states for cell V0 (V0 ∓ 0.5*slope)."
function muscl_faces(Vm1::AbstractVector, V0::AbstractVector, Vp1::AbstractVector; limiter=minmod)
    s = muscl_slopes(Vm1, V0, Vp1; limiter=limiter)
    return (V0 .- 0.5 .* s, V0 .+ 0.5 .* s)
end

"""
    to_recon_vars(M) / from_recon_vars(V)

Bijection between the 35-moment vector `M` and a bounded reconstruction-variable
vector `V = [M000, u, v, w, C200, C020, C002, <28 standardized moments>]`.
Reconstructing the bounded variables (not raw moments) limits realizability
corruption (cf. Posey/Fox/Houim arXiv:2603.13697). Reuses the same S->C->M
reconstruction as `realizable_3D_M4` via the shared helper `standardized_to_M4`.
"""

# Indices in S4 (length-35) for the 28 standardized higher moments, in canonical order:
# S300,S400,S110,S210,S310,S120,S220,S030,S130,S040,
# S101,S201,S301,S102,S202,S003,S103,S004,S011,S111,
# S211,S021,S121,S031,S012,S112,S013,S022
const _SIDX = [4,5,7,8,9,11,12,13,14,15,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

function to_recon_vars(M::AbstractVector)::Vector{Float64}
    # Delegate to the single-source, allocation-free device kernel (shared verbatim
    # with the GPU path). `to_recon_vars_dev(M...)` returns the length-35 recon-var
    # NTuple in the same canonical order; collect into the Vector callers expect.
    V = to_recon_vars_dev(M...)
    if HO_PRESSURE_RECON[]
        # opt-in pressure-tensor variant: slots 5-7 carry P_ii = rho*C2ii, so a
        # uniform-pressure contact has zero slope in every var except density.
        # Single-source transform shared with the GPU kernels (recon_dev.jl).
        V = pressurize_recon_tup(V)
    end
    return collect(V)
end

function from_recon_vars(V::AbstractVector)::Vector{Float64}
    # Delegate to the single-source device kernel; `from_recon_vars_dev(V...)` returns
    # the length-35 raw-moment NTuple in canonical layout.
    if HO_PRESSURE_RECON[]
        # P_ii -> C2ii (rho > 0 per recon_vars_ok); single-source with GPU.
        W = depressurize_recon_tup(ntuple(i -> Float64(V[i]), Val(35)))
        return collect(from_recon_vars_dev(W...))
    end
    return collect(from_recon_vars_dev(V...))
end

"""
    recon_vars_ok(V) -> Bool

True if a reconstructed recon-var vector `V` is usable for high-order face
reconstruction: all entries finite, positive density `V[1]`, and positive
directional variances `V[5],V[6],V[7]` (so `from_recon_vars`' `sqrt(C2)` is real).
MUSCL slopes can drive these negative in near-vacuum; this is the realizability
guard for the high-order path. See docs/ma100-highorder-crash-analysis.md.
"""
@inline function recon_vars_ok(V::AbstractVector)
    @inbounds return all(isfinite, V) && V[1] > 0 && V[5] > 0 && V[6] > 0 && V[7] > 0
end

"""
    scaling_limited_faces(Vm1, V0, Vp1; limiter=minmod, lam_min=0.0, nbisect=20)

Zhang--Shu / Fan--Huang--Wu realizability scaling limiter. Reconstructs cell `V0`'s two faces
from the MUSCL slope, then shrinks the slope by the largest theta in [0,1] for which BOTH faces
map (via `from_recon_vars`) to realizable 35-moment states (oracle `is_realizable`, margin
`lam_min`). theta=1 in smooth/realizable regions (design-order accuracy); theta=0 collapses to the
cell mean (locally first-order) in deep vacuum. The cell mean (theta=0) is assumed realizable
(kept so by the per-cell projection), so a feasible theta always exists.

Returns `(Vminus, Vplus, theta)`.
"""
function scaling_limited_faces(Vm1::AbstractVector, V0::AbstractVector, Vp1::AbstractVector;
                               limiter=minmod, lam_min::Real=0.0, nbisect::Int=20)
    s = muscl_slopes(Vm1, V0, Vp1; limiter=limiter)
    faces_ok(θ) = begin
        Vminus = V0 .- 0.5θ .* s
        Vplus  = V0 .+ 0.5θ .* s
        (recon_vars_ok(Vminus) && recon_vars_ok(Vplus)) || return false
        is_realizable(from_recon_vars(Vminus); lam_min=lam_min) &&
            is_realizable(from_recon_vars(Vplus);  lam_min=lam_min)
    end
    if faces_ok(1.0)                       # common path: one oracle check, unlimited
        return (V0 .- 0.5 .* s, V0 .+ 0.5 .* s, 1.0)
    end
    lo, hi = 0.0, 1.0                      # bisection for the largest feasible theta
    for _ in 1:nbisect
        mid = 0.5 * (lo + hi)
        faces_ok(mid) ? (lo = mid) : (hi = mid)
    end
    θ = lo
    return (V0 .- 0.5θ .* s, V0 .+ 0.5θ .* s, θ)
end

"""
    recon_face_pair(Vl, Vr, ML0, MR0) -> (ML, MR)

Convert the reconstructed left/right recon-var faces `(Vl, Vr)` to raw 35-moment
states, with first-order fallback to the cell-centered states `(ML0, MR0)` if
either reconstructed face is unrealizable (nonpositive/non-finite density or
directional variance) or if the reconstructed raw moments come out non-finite.
This is the single realizability gate for high-order face reconstruction: it
prevents `sqrt(negative)` / `Inf` from the reconstruction reaching the
realizability + flux machinery in near-vacuum.
"""
function recon_face_pair(Vl::AbstractVector, Vr::AbstractVector,
                         ML0::AbstractVector, MR0::AbstractVector)
    # near-vacuum gate: below the floor, cell moments are cancellation-garbage, so
    # use the first-order state (the vacuum then evolves like the robust 1st-order
    # scheme; resolved cells above the floor still get high-order reconstruction).
    vac = HO_VACUUM_FLOOR[]
    if vac > 0 && (ML0[1] < vac || MR0[1] < vac)
        return collect(ML0), collect(MR0)
    end
    if recon_vars_ok(Vl) && recon_vars_ok(Vr)
        Li = from_recon_vars(Vl); Ri = from_recon_vars(Vr)
        if Li[1] > 0 && Ri[1] > 0 && all(isfinite, Li) && all(isfinite, Ri)
            return Li, Ri
        end
    end
    return collect(ML0), collect(MR0)
end

# ---------------------------------------------------------------------------
# Per-interface face reconstruction (order-2). SINGLE SOURCE shared by the 1D
# `residual_1d` (src/numerics/highorder_flux.jl) and the 3D-line `residual_line`
# (src/numerics/highorder_3d.jl) so the three reconstruction modes are written once.
# Each takes the four stencil cells' recon vars (Vm1,V0,Vp1,Vp2) for the interface
# between the V0 cell and the Vp1 cell; the caller supplies the BC-correct stencil
# (clamped / wrapped / ghost). The DEVICE analogue of this composition is the GPU
# `_face_flux_core` (NTuple, separate by data layout + eigensolver — see misc/02).
# ---------------------------------------------------------------------------

"""
    recon_faces_default(Vm1, V0, Vp1, Vp2, M0, Mp1) -> (ML, MR)

Order-2 MUSCL faces at the V0|Vp1 interface, gated by `recon_face_pair` (vacuum floor +
recon-validity first-order fallback). `M0,Mp1` are the two cells' raw 35-moment means.
"""
function recon_faces_default(Vm1, V0, Vp1, Vp2, M0, Mp1)
    VplusL  = muscl_faces(Vm1, V0, Vp1)[2]    # right face of the V0 cell
    VminusR = muscl_faces(V0, Vp1, Vp2)[1]    # left face of the Vp1 cell
    return recon_face_pair(VplusL, VminusR, M0, Mp1)
end

"""
    recon_faces_proj(flaggedL, flaggedR, Vm1, V0, Vp1, Vp2, M0, Mp1) -> (ML, MR)

As [`recon_faces_default`](@ref) but a cell flagged for the realizability projection
(`realizability_margin < 0`) reconstructs FIRST ORDER (its face = cell-mean recon vars).
"""
function recon_faces_proj(flaggedL::Bool, flaggedR::Bool, Vm1, V0, Vp1, Vp2, M0, Mp1)
    VplusL  = flaggedL ? V0  : muscl_faces(Vm1, V0, Vp1)[2]
    VminusR = flaggedR ? Vp1 : muscl_faces(V0, Vp1, Vp2)[1]
    return recon_face_pair(VplusL, VminusR, M0, Mp1)
end

"""
    recon_faces_limited(Vm1, V0, Vp1, Vp2) -> (ML, MR)

Realizability scaling-limited faces ([`scaling_limited_faces`](@ref)). Faces are realizable
by construction, so there is no `recon_face_pair` fallback.
"""
function recon_faces_limited(Vm1, V0, Vp1, Vp2)
    _, VplusL, _  = scaling_limited_faces(Vm1, V0, Vp1)   # right face of the V0 cell
    VminusR, _, _ = scaling_limited_faces(V0, Vp1, Vp2)   # left face of the Vp1 cell
    return from_recon_vars(VplusL), from_recon_vars(VminusR)
end
