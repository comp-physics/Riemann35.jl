# ---------------------------------------------------------------------------
# Order-3 (WENO5 + θ*-IDP) modules — included here so they stay out of the
# default order=1,2 code path (byte-identical guarantee). The modules create
# sub-modules of Riemann35 and their names are brought into this file's scope.
# ---------------------------------------------------------------------------
include(joinpath(@__DIR__, "idp_limiter_dev.jl")); using .IdpLimiterDev: theta_star_update_dev, theta_star_update_closed
# Single-source device-safe order-3 reconstruction primitives, shared VERBATIM with
# the GPU order-3 kernels (hiorder3_recon_dev.jl) — residual_line3 just composes them.
include(joinpath(@__DIR__, "hiorder3_recon_dev.jl")); using .HiOrder3ReconDev:
    recon_point_dev, recon_avg_dev, weno_faces_dev, weno_scaled_face_dev, recon_vars_realizable
# Hardened device CHyQMOM inversion (KFVS increment A) — CUDA-free, host-runnable.
# Used ONLY by the opt-in kinetic-FVS anchor path (`_anchor_interior!`); the default
# order=1,2,3 paths never reference it. Its faithful condition gate removes the
# CPU `chyqmom_nodes_3d`'s wild-abscissa blowup (0.18% of real cells), which is
# essential for the anchor's measure_update to stay finite on cone-boundary cells.
include(joinpath(@__DIR__, "..", "..", "gpu", "chyqmom_nodes_3d_dev.jl"))
using .KFVSInversionDev: chyqmom_nodes_3d_dev
# SINGLE-SOURCE anchor primitives (F3 flux + full-cone θ* bisection), shared verbatim
# with the GPU order-3 residual. Included here where `chyqmom_nodes_3d_dev` is in scope.
include(joinpath(@__DIR__, "..", "..", "gpu", "kfvs_anchor_core.jl"))

# ---------------------------------------------------------------------------
# _face_flux_tup — NTuple{35} in, NTuple{35} HLL flux out.
# Mirrors face_flux_1d exactly (same realizable_3D_M4 + realize_and_speed +
# riemann_flux_dev(rs=0) chain) but operates on NTuples to stay alloc-friendly
# in the order-3 hot path.
# ---------------------------------------------------------------------------
function _face_flux_tup(mL::NTuple{35,Float64}, mR::NTuple{35,Float64},
                        axis::Int, Ma::Real, s3max::Real)
    ML  = realizable_3D_M4(collect(mL), Ma, s3max)
    MR  = realizable_3D_M4(collect(mR), Ma, s3max)
    MLr, lminL, lmaxL = realize_and_speed(ML, axis, Ma)
    MRr, lminR, lmaxR = realize_and_speed(MR, axis, Ma)
    FL  = _phys_flux(MLr, axis);  FR  = _phys_flux(MRr, axis)
    sL  = Float64(min(lminL, lminR));  sR  = Float64(max(lmaxL, lmaxR))
    riemann_flux_dev(0, axis,                         # rs=0 → HLL
                     NTuple{35,Float64}(MLr), NTuple{35,Float64}(MRr),
                     NTuple{35,Float64}(FL),  NTuple{35,Float64}(FR),
                     sL, sR)
end

# ---------------------------------------------------------------------------
# _kfvs_face_flux_tup — the KINETIC-FVS interface flux (F3 conservative anchor).
# Upwind quadrature flux: nodes of the LEFT cell with U_axis > 0 (leaving L into
# the interface) plus nodes of the RIGHT cell with U_axis < 0 (entering from R).
# For flux component m=(i,j,l): contribution = n_k · U_axis · (Ux^i Uy^j Uz^l).
# This is EXACTLY the interface flux implied by the 3D measure_update (increment B),
# expressed as an F(cL,cR) so it slots into the residual's flux-blend F_LO. Because
# it is a single-valued function of the shared (cL,cR), the face flux is identical
# for both adjacent cells ⇒ the update telescopes ⇒ mass/momentum/energy conserve
# EXACTLY (thm:idp-conservative). Uses the hardened device inversion (no wild
# abscissas / throws). axis ∈ {1=x,2=y,3=z}. Degenerate cell ⇒ HLL fallback.
@inline function _kfvs_face_flux_tup(mL::NTuple{35,Float64}, mR::NTuple{35,Float64},
                                     axis::Int, Ma::Real, s3max::Real)
    # F3 quadrature flux via the single-source device core; native HLL on degeneracy.
    f = kfvs_face_flux_dev(mL, mR, axis)
    f === nothing ? _face_flux_tup(mL, mR, axis, Ma, s3max) : f
end

# F3 flux-level θ stats (env HYQMOM_KFVS_THETA_STATS=1): mean face θ* + fraction of
# faces where θ*<1 (the flux limiter engaged). Cheap Refs, populated in PASS 2a.
const _KFVS_THETA_STATS_ENABLED = Ref{Bool}(get(ENV, "HYQMOM_KFVS_THETA_STATS", "0") != "0")
const _KFVS_THETA_SUM = Ref{Float64}(0.0)
const _KFVS_THETA_N   = Ref{Int}(0)
const _KFVS_THETA_LT1 = Ref{Int}(0)
reset_kfvs_theta_stats!() = (_KFVS_THETA_SUM[]=0.0; _KFVS_THETA_N[]=0; _KFVS_THETA_LT1[]=0; nothing)
kfvs_theta_stats() = (mean_theta = _KFVS_THETA_N[]>0 ? _KFVS_THETA_SUM[]/_KFVS_THETA_N[] : 1.0,
                      faces=_KFVS_THETA_N[], theta_lt1=_KFVS_THETA_LT1[])

# FULL-CONE + MARGINAL θ* for the face limiter (F3). Same structure as the shipped
# marginal-only `theta_star_update_dev`, but the realizability predicate is the Δ2*
# cross-moment cone (`is_realizable`) AND the marginal s3max/variance regularization
# (`_marginal_regularized`) — so the face-shared-θ flux blend keeps the update in the
# FULL cone (not just the marginals split-HLL preserves). Mlo (the KFVS first-order
# anchor) is realizable in both by construction under CFL ⇒ θ=0 feasible.
# DIAGNOSTIC env hook (F3-drift investigation; default OFF ⇒ production unchanged):
# HYQMOM_KFVS_XFLOOR=δ requires the Δ2* cross-moment margin ≥ +δ (re-interiorization
# floor, mirroring the marginal h2min floor) instead of the production ≥ 0. Set to a
# small positive value (e.g. 1e-6) to test whether parking cells slightly INTERIOR of
# the cross cone collapses the multi-step drift (VARIANT B). At 0.0 (default) this is
# byte-identical to the shipped predicate. `is_realizable(m; lam_min=δ)` already uses
# the EXACT LAPACK eigvals of the Δ2* matrix (VARIANT A = the production path).
const _KFVS_XFLOOR = Ref{Float64}(parse(Float64, get(ENV, "HYQMOM_KFVS_XFLOOR", "0.0")))
set_kfvs_xfloor!(δ) = (_KFVS_XFLOOR[] = Float64(δ); nothing)
@inline function _theta_star_fullcone(Mlo::NTuple{35,Float64}, dM::NTuple{35,Float64},
                                      Ma, s3max; nb::Int = 24)
    # CPU predicate (exact-eig is_realizable + marginal regularization); the bisection
    # loop is the single-source `theta_star_fullcone_bisect` shared with the GPU path.
    lamf = _KFVS_XFLOOR[]
    ok(m) = is_realizable(collect(m); lam_min=lamf) && _marginal_regularized(m, Ma, s3max)
    theta_star_fullcone_bisect(Mlo, dM, ok; nb=nb)
end

# ---------------------------------------------------------------------------
# residual_line3 — WENO5 high-order fluxes + first-order HLL anchors for one
# axis line.  Returns (F_HO, F_LO) as length-(n+1) Vector{NTuple{35,Float64}}.
# Requires g ≥ 4 (WENO5 ±2 stencil + deconv ±2 extension).
# Pipeline: deconv5 (smooth5-gated per component) → to_recon_vars_dev →
#           conv5 (recon-var averages) → weno5z L/R → from_recon_vars_dev →
#           realizability fallback → _face_flux_tup (HLL).
# ---------------------------------------------------------------------------
"""
    residual_line3(Mext, ds, axis, Ma; g=4, s3max=40.0) -> (F_HO, F_LO)

WENO5 reconstruction + first-order HLL anchors for a 1-D padded line.
`Mext` is `(n+2g, 35)` with `g ≥ 4` ghost cells at each end.
Returns two length-(n+1) Vectors of `NTuple{35,Float64}`: the high-order
(WENO5) flux and the first-order (cell-mean HLL) flux at each of the n+1
interior-bounding interfaces.  The driver combines these across all three
axes for the joint 6-face θ*-IDP blend.
"""
function residual_line3(Mext::AbstractMatrix, ds::Real, axis::Int, Ma::Real;
                        g::Int = 4, s3max::Real = 40.0, use_kfvs_anchor::Bool = false)
    n2g = size(Mext, 1)
    n   = n2g - 2g
    @assert g >= 4 "order=3 residual requires halo g ≥ 4 (WENO5+deconv stencils); got g=$g"
    @assert n >= 1 "line must contain ≥ 1 interior cell; got n=$n"

    # --- Step 1: raw deconvolved point values → recon-var point values -------
    # Per-component smooth5 gate: if the 5-cell stencil for component q is
    # smooth, apply deconv5 (O(h^6) avg→point); otherwise keep the cell average.
    # Cells within 2 of the array boundary cannot access a full ±2 stencil:
    # fall back to the cell average for those (they are ghost cells anyway).
    _cellrow(k) = ntuple(q -> Mext[k, q], Val(35))
    Ppt = Vector{NTuple{35,Float64}}(undef, n2g)
    for k in 1:n2g
        Ppt[k] = (k >= 3 && k <= n2g - 2) ?
            recon_point_dev(_cellrow(k-2), _cellrow(k-1), _cellrow(k),
                            _cellrow(k+1), _cellrow(k+2)) :
            to_recon_vars_dev(_cellrow(k)...)   # boundary: OOB stencil → cell average
    end

    # --- Step 2: forward-convolve recon-var point values → recon-var averages --
    # conv5 is the O(h^6) inverse of deconv5: converts point values to the
    # corresponding cell averages so weno5z reconstructs from cell averages
    # (skipping this step caps effective order at 2). Stencil indices are
    # clamped to [1..n2g] at the array boundary (replicates the outermost
    # ghost Ppt value — only affects the outermost 2 cells, which are ghosts).
    _pp(k) = Ppt[clamp(k, 1, n2g)]
    Vavg = Vector{NTuple{35,Float64}}(undef, n2g)
    for k in 1:n2g
        Vavg[k] = recon_avg_dev(_pp(k-2), _pp(k-1), _pp(k), _pp(k+1), _pp(k+2))
    end

    # --- Step 3: WENO5 face reconstruction + F_HO / F_LO per interface ------
    # Interface f (1..n+1) lies between Mext rows il = g+f-1 and ir = g+f.
    # Left reconstruction (weno5z at il, right-going stencil):
    #   uses Vavg[il-2..il+2]; for g=4 and f=1, il=4 → indices 2..6 — valid.
    # Right reconstruction (weno5z at ir, left-going = reversed stencil):
    #   uses Vavg[ir-2..ir+2]; for f=n+1, ir=g+n → indices g+n-2..g+n+2 ≤ n2g-2 — valid.
    # Clamping guards against the two outermost ghost positions (indices 1 and n2g)
    # being accessed by Vavg's conv5; it does NOT widen the face stencil itself.
    _vv(k) = Vavg[clamp(k, 1, n2g)]
    F_HO = Vector{NTuple{35,Float64}}(undef, n + 1)
    F_LO = Vector{NTuple{35,Float64}}(undef, n + 1)
    for f in 1:n+1
        il = g + f - 1                  # left cell of interface f (Mext row)
        cL = _cellrow(il)
        cR = _cellrow(il + 1)
        # WENO5 L/R faces (Vavg[il-2..il+3] straddle the interface) + continuous
        # Zhang–Shu realizability scaling toward the cell mean — all in weno_faces_dev.
        mL, mR = weno_faces_dev(_vv(il-2), _vv(il-1), _vv(il),
                                _vv(il+1), _vv(il+2), _vv(il+3), cL, cR)
        F_HO[f] = _face_flux_tup(mL, mR, axis, Ma, s3max)
        # F_LO anchor flux: default = HLL (byte-identical off); F3 = kinetic-FVS
        # upwind quadrature flux (conservative + realizable-by-construction anchor).
        F_LO[f] = use_kfvs_anchor ? _kfvs_face_flux_tup(cL, cR, axis, Ma, s3max) :
                                    _face_flux_tup(cL, cR, axis, Ma, s3max)
    end
    return F_HO, F_LO
end

# ---------------------------------------------------------------------------
# residual_ho_3d_order3! — two-pass 3D residual for order==3.
#   Pass 1: reconstruct F_HO and F_LO for all three axis sweeps.
#   Pass 2: per interior cell, compute Mlo (first-order anchor over all axes),
#           per-face θ* via theta_star_update_dev (factor 6 = N_faces), take
#           the min θ at each shared interface, blend F, accumulate R.
# Requires halo ≥ 4.  dt > 0 enables IDP limiting; dt = 0 sets all θ = 1
# (pure WENO5, useful for conservation tests).
# ---------------------------------------------------------------------------
# Runtime θ* selector (default OFF ⇒ bisection ⇒ byte-identical). `theta_closed`
# is a plain Bool threaded from the entry point — same pattern as the GPU path.
@inline _theta_star_sel(use_closed::Bool, Mlo::NTuple{35,Float64}, dM::NTuple{35,Float64}) =
    use_closed ? theta_star_update_closed(Mlo, dM) : theta_star_update_dev(Mlo, dM)

# Opt-in stage-level anchor-exit capture (diagnostic; OFF ⇒ byte-identical).
# When ENV["KFVS_CAPTURE"]=="1", the residual records the FIRST interior cell whose
# pure θ=0 KFVS anchor update `Mlo` leaves the cone (realizability_margin < tol),
# together with its 7-cell face stencil (raw moments) and the update parameters, into
# `_KFVS_CAPTURE`. This is the self-contained "first failing stencil" of Phase I.2 — a
# caller saves `_KFVS_CAPTURE[]` to JLD2 for offline U^Q/D/cone-spectrum analysis.
const _KFVS_CAPTURE = Ref{Any}(nothing)
# Companion: full haloed input state of the residual call that first sees an anchor exit
# (= the stage-2 input M1). Lets an offline pass reconstruct Mlo/U^Q/margins for EVERY
# interior cell (pass AND fail) → the gate ROC. Set once, when the first exit is captured.
const _KFVS_STATE = Ref{Any}(nothing)

function residual_ho_3d_order3!(R::Array{Float64,4}, M::Array{Float64,4},
                                nx::Int, ny::Int, nz::Int, halo::Int,
                                dx::Real, dy::Real, dz::Real, Ma::Real,
                                dt::Real; s3max::Real = 40.0,
                                theta_closed::Bool = true,
                                rank_bnd = (xlo=false, xhi=false, ylo=false, yhi=false,
                                            zlo=false, zhi=false),
                                use_kfvs_anchor::Bool = false)
    @assert halo >= 4 "order=3 residual requires halo ≥ 4; got halo=$halo"
    fill!(R, 0.0)
    g  = halo
    λx = Float64(dt) / Float64(dx)
    λy = Float64(dt) / Float64(dy)
    λz = Float64(dt) / Float64(dz)

    # --- Allocate per-axis face-flux storage (interior-relative indexing) ----
    # Face f along axis-x for y-column j, z-slice k:
    #   interface between interior cells (f-1,j,k) and (f,j,k),  f = 1..nx+1
    #   (cell 0 = last left ghost; cell nx+1 = first right ghost)
    FHO_x = Array{NTuple{35,Float64},3}(undef, nx+1, ny, nz)
    FLO_x = Array{NTuple{35,Float64},3}(undef, nx+1, ny, nz)
    FHO_y = Array{NTuple{35,Float64},3}(undef, nx, ny+1, nz)
    FLO_y = Array{NTuple{35,Float64},3}(undef, nx, ny+1, nz)
    FHO_z = Array{NTuple{35,Float64},3}(undef, nx, ny, nz+1)
    FLO_z = Array{NTuple{35,Float64},3}(undef, nx, ny, nz+1)

    # =========================================================================
    # PASS 1: reconstruct F_HO and F_LO for every interface on every axis.
    # =========================================================================

    # X-axis lines (halos already in M[:, jh, k, :])
    for k in 1:nz, j in 1:ny
        jh   = j + halo
        Mext = @view M[:, jh, k, :]           # (nx+2halo, 35) — zero-copy view
        Fho, Flo = residual_line3(Mext, dx, 1, Ma; g=g, s3max=s3max, use_kfvs_anchor=use_kfvs_anchor)
        for f in 1:nx+1
            FHO_x[f,j,k] = Fho[f];  FLO_x[f,j,k] = Flo[f]
        end
    end

    # Y-axis lines
    for k in 1:nz, i in 1:nx
        ih   = i + halo
        Mext = @view M[ih, :, k, :]
        Fho, Flo = residual_line3(Mext, dy, 2, Ma; g=g, s3max=s3max, use_kfvs_anchor=use_kfvs_anchor)
        for f in 1:ny+1
            FHO_y[i,f,k] = Fho[f];  FLO_y[i,f,k] = Flo[f]
        end
    end

    # Z-axis lines — no halo storage: pad with outflow ghosts (copy edge cells),
    # identical to the order-1/2 z-pad strategy so ghost values are consistent.
    for i in 1:nx, j in 1:ny
        ih = i + halo;  jh = j + halo
        col  = M[ih, jh, :, :]                # (nz, 35) — allocates once per line
        Mext = vcat(repeat(col[1:1,  :], g, 1),
                    col,
                    repeat(col[nz:nz,:], g, 1))   # (nz+2g, 35)
        Fho, Flo = residual_line3(Mext, dz, 3, Ma; g=g, s3max=s3max, use_kfvs_anchor=use_kfvs_anchor)
        for f in 1:nz+1
            FHO_z[i,j,f] = Fho[f];  FLO_z[i,j,f] = Flo[f]
        end
    end

    # =========================================================================
    # PASS 2a: per-cell θ* contributions for each of the six faces.
    # θ*-IDP bound (from Zhang-Shu / Vikas 2011 splitting, extended to 3D):
    #   Mlo = M - Σ_axis λ_axis (F_LO[right] - F_LO[left])   (first-order anchor)
    #   For face f of cell (i,j,k): dM_face = ∓ 6·λ_axis · G_face
    #     where G_face = F_HO_face - F_LO_face,
    #     sign "-" for a right/top/front face (flux leaves cell),
    #     sign "+" for a left/bottom/back face (flux enters cell).
    #   Factor 6 = N_faces: conservative per-face bound that keeps the cell
    #   realizable even when all six faces are simultaneously active at θ=1.
    # When dt=0, λ_axis=0 → dM_face=0 → theta_star returns 1 (no limiting).
    # =========================================================================
    Θ_xr = ones(nx, ny, nz);  Θ_xl = ones(nx, ny, nz)
    Θ_yr = ones(nx, ny, nz);  Θ_yl = ones(nx, ny, nz)
    Θ_zr = ones(nx, ny, nz);  Θ_zl = ones(nx, ny, nz)

    for k in 1:nz, j in 1:ny, i in 1:nx
        ih = i + halo;  jh = j + halo
        Mc = ntuple(q -> M[ih, jh, k, q], Val(35))

        # First-order anchor state (Euler step with F_LO on all 6 faces).
        # Short-circuit for dt=0 (λ=0): Mlo = Mc exactly, avoiding IEEE -0.0
        # sign issues from 0.0*(flux_diff) that would make _state_realizable fail.
        Mlo = if iszero(λx) && iszero(λy) && iszero(λz)
            Mc
        else
            ntuple(q -> Mc[q] - λx*(FLO_x[i+1,j,k][q]-FLO_x[i,j,k][q])
                               - λy*(FLO_y[i,j+1,k][q]-FLO_y[i,j,k][q])
                               - λz*(FLO_z[i,j,k+1][q]-FLO_z[i,j,k][q]), Val(35))
        end

        # Opt-in: capture EVERY anchor-endpoint cone exit (diagnostic; byte-id off).
        # Accumulates each failing cell's 7-cell face stencil + Mlo into _KFVS_CAPTURE
        # (lazily a Vector), capped at 400. First batch = the stage-2 failures.
        if use_kfvs_anchor && get(ENV, "KFVS_CAPTURE", "0") == "1"
            _KFVS_CAPTURE[] === nothing && (_KFVS_CAPTURE[] = Any[])
            if length(_KFVS_CAPTURE[]) < 400
                mlov = collect(Mlo)
                # only "realizable INPUT state -> anchor update exits" (the true stage-2
                # failure mode); excludes post-crash cells whose input is already out.
                if realizability_margin(collect(Mc)) >= 0.0 &&
                   all(isfinite, mlov) && realizability_margin(mlov) < -1e-8
                    _KFVS_STATE[] === nothing && (_KFVS_STATE[] = copy(M))   # stage-2 input snapshot
                    kz(dk) = clamp(k+dk, 1, nz)
                    st(di,dj,dk) = [M[ih+di, jh+dj, kz(dk), q] for q in 1:35]
                    push!(_KFVS_CAPTURE[], (cell=(i,j,k), lam=(λx,λy,λz), Ma=Float64(Ma), s3max=Float64(s3max),
                        margin=realizability_margin(mlov), Mlo=mlov,
                        C=st(0,0,0), Lx=st(-1,0,0), Rx=st(1,0,0),
                        Ly=st(0,-1,0), Ry=st(0,1,0), Lz=st(0,0,-1), Rz=st(0,0,1)))
                end
            end
        end

        # High-order corrections G = F_HO - F_LO (per face)
        Gxr = ntuple(q -> FHO_x[i+1,j,k][q] - FLO_x[i+1,j,k][q], Val(35))
        Gxl = ntuple(q -> FHO_x[i,  j,k][q] - FLO_x[i,  j,k][q], Val(35))
        Gyr = ntuple(q -> FHO_y[i,j+1,k][q] - FLO_y[i,j+1,k][q], Val(35))
        Gyl = ntuple(q -> FHO_y[i,j,  k][q] - FLO_y[i,j,  k][q], Val(35))
        Gzr = ntuple(q -> FHO_z[i,j,k+1][q] - FLO_z[i,j,k+1][q], Val(35))
        Gzl = ntuple(q -> FHO_z[i,j,k  ][q] - FLO_z[i,j,k  ][q], Val(35))

        # Per-face θ* (one-sided, factor-6 conservative bound). Default = the shipped
        # marginal-only limiter (closed-form or bisection per theta_closed); F3
        # (use_kfvs_anchor) = FULL-CONE + marginal limiter, so the face-shared-θ flux
        # blend keeps the update in the full cross-moment cone.
        kstat = use_kfvs_anchor && _KFVS_THETA_STATS_ENABLED[]
        θfun(mlo, dm) = begin
            θ = use_kfvs_anchor ? _theta_star_fullcone(mlo, dm, Ma, s3max) :
                                  _theta_star_sel(theta_closed, mlo, dm)
            if kstat
                _KFVS_THETA_SUM[] += θ; _KFVS_THETA_N[] += 1
                θ < 1.0 - 1e-9 && (_KFVS_THETA_LT1[] += 1)
            end
            θ
        end
        Θ_xr[i,j,k] = θfun(Mlo, ntuple(q -> -6λx * Gxr[q], Val(35)))
        Θ_xl[i,j,k] = θfun(Mlo, ntuple(q ->  6λx * Gxl[q], Val(35)))
        Θ_yr[i,j,k] = θfun(Mlo, ntuple(q -> -6λy * Gyr[q], Val(35)))
        Θ_yl[i,j,k] = θfun(Mlo, ntuple(q ->  6λy * Gyl[q], Val(35)))
        Θ_zr[i,j,k] = θfun(Mlo, ntuple(q -> -6λz * Gzr[q], Val(35)))
        Θ_zl[i,j,k] = θfun(Mlo, ntuple(q ->  6λz * Gzl[q], Val(35)))
    end

    # =========================================================================
    # PASS 2a': rank-boundary θ (AXIS-GENERIC: x, y, z share one path). At a RANK
    # boundary on any axis the shared interface's θ must be min(Θ_own, Θ_neighbour)
    # computed identically on both ranks — but the neighbour cell is the first HALO
    # cell, not an interior Θ_* entry. Its per-face θ is cheap and local:
    # theta_star_update_dev(Mlo_halo, ±6λ·G_shared), where
    #   Mlo_halo = the halo cell's first-order (HLL, all-6-face) anchor — the SAME
    #     function the neighbour rank evaluates for that (there interior) cell, so
    #     the two agree bit-for-bit (halo=8 exchange + corner-consistent ghosts give
    #     identical adjacent cells), and
    #   G_shared = F_HO − F_LO at the shared interface (identical across ranks; the
    #     wide halo makes the reconstruction footprint fully real data).
    # The recipe below is written ONCE and parameterized by (axis, side); only sides
    # flagged as rank boundaries (rank_bnd.*) are computed. Global-domain boundaries
    # keep the "own cell only" θ in PASS 2b (serial-identical). The CPU decomposes
    # only x/y, so the z branch is present but DORMANT (zlo/zhi always false) — the
    # code is axis-symmetric, not a per-axis copy. `crow` reads the haloed array.
    # =========================================================================
    crow(px, py, pk) = ntuple(q -> M[px, py, pk, q], Val(35))
    # First-order (HLL) anchor of the cell at haloed position (px,py,k), over all 6
    # faces; copy-rule (HLL of the cell with itself) at the z ends — mirrors the
    # interior Mlo exactly. dt=0 (λ all zero) short-circuits to the cell state.
    ff(a,b,ax) = use_kfvs_anchor ? _kfvs_face_flux_tup(a, b, ax, Ma, s3max) :
                                   _face_flux_tup(a, b, ax, Ma, s3max)
    function halo_cell_mlo(px::Int, py::Int, k::Int)
        C = crow(px, py, k)
        (iszero(λx) && iszero(λy) && iszero(λz)) && return C
        FxL = ff(crow(px-1,py,k), C, 1)
        FxR = ff(C, crow(px+1,py,k), 1)
        FyD = ff(crow(px,py-1,k), C, 2)
        FyU = ff(C, crow(px,py+1,k), 2)
        zb  = (k > 1)  ? crow(px,py,k-1) : C
        zf  = (k < nz) ? crow(px,py,k+1) : C
        FzB = ff(zb, C, 3)
        FzF = ff(C, zf, 3)
        return ntuple(q -> C[q] - λx*(FxR[q]-FxL[q]) - λy*(FyU[q]-FyD[q])
                                 - λz*(FzF[q]-FzB[q]), Val(35))
    end

    # Haloed coords of the halo cell just outside the (axis, side) boundary, for the
    # transverse index pair (a,b). (z uses raw k in 1..nz; the halo k=0/nz+1 is only
    # ever reached when zlo/zhi is set — never on the CPU.)
    halo_coords(axis, hi, a, b) =
        axis == 1 ? ((hi ? nx+halo+1 : halo), a+halo, b) :
        axis == 2 ? (a+halo, (hi ? ny+halo+1 : halo), b) :
                    (a+halo, b+halo, (hi ? nz+1 : 0))
    # Shared boundary-interface G = F_HO − F_LO for the (axis, side) at transverse (a,b).
    function shared_face_G(axis, hi, a, b)
        if axis == 1
            f = hi ? nx+1 : 1
            return ntuple(q -> FHO_x[f,a,b][q] - FLO_x[f,a,b][q], Val(35))
        elseif axis == 2
            f = hi ? ny+1 : 1
            return ntuple(q -> FHO_y[a,f,b][q] - FLO_y[a,f,b][q], Val(35))
        else
            f = hi ? nz+1 : 1
            return ntuple(q -> FHO_z[a,b,f][q] - FLO_z[a,b,f][q], Val(35))
        end
    end
    # The one axis-parameterized rank-boundary θ path. Returns a transverse-sized
    # buffer of the halo cell's θ* on its face shared with the boundary interior cell.
    # sign: -6λ on a lo side (halo cell's RIGHT/UP/FRONT face), +6λ on a hi side.
    function rank_bnd_theta(axis::Int, hi::Bool)
        t1, t2 = axis == 1 ? (ny, nz) : axis == 2 ? (nx, nz) : (nx, ny)
        λ = axis == 1 ? λx : axis == 2 ? λy : λz
        s = (hi ? 6.0 : -6.0) * λ
        Θ = Array{Float64}(undef, t1, t2)
        for b in 1:t2, a in 1:t1
            cx, cy, ck = halo_coords(axis, hi, a, b)
            Mlo = halo_cell_mlo(cx, cy, ck)
            G   = shared_face_G(axis, hi, a, b)
            dm  = ntuple(q -> s * G[q], Val(35))
            Θ[a, b] = use_kfvs_anchor ? _theta_star_fullcone(Mlo, dm, Ma, s3max) :
                                        _theta_star_sel(theta_closed, Mlo, dm)
        end
        return Θ
    end

    # Boundary halo-cell θ, computed only for flagged rank sides (empty otherwise).
    _empty = Array{Float64}(undef, 0, 0)
    Θ_xr_lo = rank_bnd.xlo ? rank_bnd_theta(1, false) : _empty
    Θ_xl_hi = rank_bnd.xhi ? rank_bnd_theta(1, true)  : _empty
    Θ_yr_lo = rank_bnd.ylo ? rank_bnd_theta(2, false) : _empty
    Θ_yl_hi = rank_bnd.yhi ? rank_bnd_theta(2, true)  : _empty
    Θ_zr_lo = rank_bnd.zlo ? rank_bnd_theta(3, false) : _empty
    Θ_zl_hi = rank_bnd.zhi ? rank_bnd_theta(3, true)  : _empty

    # =========================================================================
    # PASS 2b+2c: interface θ = min over the two adjacent cells, blend fluxes,
    # accumulate residual R = -Σ_axis (F_right - F_left) / ds_axis.
    # Conservation: both cells sharing an interface see the SAME blended flux
    # (same θ = min of the two cells' per-face bounds) → fluxes telescope exactly.
    # =========================================================================
    for k in 1:nz, j in 1:ny, i in 1:nx
        ih = i + halo;  jh = j + halo

        # Interface θ = min(this cell, neighbour cell). Interior interfaces use the
        # adjacent interior cell; RANK-boundary interfaces use the halo cell's θ
        # (single-valued/conservative across ranks); GLOBAL boundaries keep the one
        # adjacent interior cell only (serial-identical). z never has rank boundaries.
        θxr = (i < nx)      ? min(Θ_xr[i,j,k], Θ_xl[i+1,j,k]) :
              rank_bnd.xhi  ? min(Θ_xr[i,j,k], Θ_xl_hi[j,k])  : Θ_xr[i,j,k]
        θxl = (i > 1)       ? min(Θ_xl[i,j,k], Θ_xr[i-1,j,k]) :
              rank_bnd.xlo  ? min(Θ_xl[i,j,k], Θ_xr_lo[j,k])  : Θ_xl[i,j,k]
        θyr = (j < ny)      ? min(Θ_yr[i,j,k], Θ_yl[i,j+1,k]) :
              rank_bnd.yhi  ? min(Θ_yr[i,j,k], Θ_yl_hi[i,k])  : Θ_yr[i,j,k]
        θyl = (j > 1)       ? min(Θ_yl[i,j,k], Θ_yr[i,j-1,k]) :
              rank_bnd.ylo  ? min(Θ_yl[i,j,k], Θ_yr_lo[i,k])  : Θ_yl[i,j,k]
        θzr = (k < nz)      ? min(Θ_zr[i,j,k], Θ_zl[i,j,k+1]) :
              rank_bnd.zhi  ? min(Θ_zr[i,j,k], Θ_zl_hi[i,j])  : Θ_zr[i,j,k]
        θzl = (k > 1)       ? min(Θ_zl[i,j,k], Θ_zr[i,j,k-1]) :
              rank_bnd.zlo  ? min(Θ_zl[i,j,k], Θ_zr_lo[i,j])  : Θ_zl[i,j,k]

        # Blended flux F = F_LO + θ·(F_HO - F_LO) at each face
        blend(θ, FH, FL) = ntuple(q -> FL[q] + θ * (FH[q] - FL[q]), Val(35))
        Fxr = blend(θxr, FHO_x[i+1,j,k], FLO_x[i+1,j,k])
        Fxl = blend(θxl, FHO_x[i,  j,k], FLO_x[i,  j,k])
        Fyr = blend(θyr, FHO_y[i,j+1,k], FLO_y[i,j+1,k])
        Fyl = blend(θyl, FHO_y[i,j,  k], FLO_y[i,j,  k])
        Fzr = blend(θzr, FHO_z[i,j,k+1], FLO_z[i,j,k+1])
        Fzl = blend(θzl, FHO_z[i,j,k  ], FLO_z[i,j,k  ])

        # Residual R_i = -Σ_axis (F_right - F_left) / ds_axis
        for q in 1:35
            R[ih, jh, k, q] = -((Fxr[q]-Fxl[q])/dx + (Fyr[q]-Fyl[q])/dy + (Fzr[q]-Fzl[q])/dz)
        end
    end
    return R
end

"""
    residual_line(Mext, ds, axis, Ma; order=2, g=2)

1D method-of-lines residual for a line of 35-moment cells that already includes
`g` ghost cells at each end (filled externally with neighbor / BC data). Returns
the interior residual (Ni, 35) = -(Fhat[i+1/2] - Fhat[i-1/2])/ds, reconstructing
face states from the ghosts (no boundary condition applied here). This is the
shared primitive for the 3D unsplit residual (x, y via MPI halos; z via padded
ghosts). order=1 uses cell-centered states; order=2 uses MUSCL with per-interface
fallback to first order on nonpositive reconstructed density.
"""
function residual_line(Mext::AbstractMatrix, ds::Real, axis::Int, Ma::Real;
                       order::Int=2, g::Int=2, use_limiter::Bool=false,
                       use_proj_recon::Bool=false, s3max::Real = 4.0 + abs(Ma) / 2.0)
    Ntot = size(Mext, 1)
    Ni = Ntot - 2g
    # interface fluxes at i+1/2 for interior interfaces: need faces at indices
    # spanning g..Ntot-g. Compute Fhat at every interface that bounds an interior cell.
    # Interior cells are rows g+1 .. g+Ni; their bounding interfaces are g+1/2 .. g+Ni+1/2.
    Fhat = Vector{Vector{Float64}}(undef, g + Ni)
    # recon vars cached for all interior-bounding stencil rows (g >= 2 keeps iL-1..iL+2 in range).
    Vc = order >= 2 ? [to_recon_vars(@view Mext[i, :]) for i in axes(Mext, 1)] : Vector{Vector{Float64}}()
    if order == 1
        for iface in g:(g+Ni)            # interface between cell iface and iface+1
            Fhat[iface] = face_flux_1d(Mext[iface, :], Mext[iface+1, :], axis, Ma; s3max=s3max)
        end
    elseif use_proj_recon
        # Rodney's projection-triggered control: a cell flagged for the realizability
        # projection (`realizability_margin < 0`) reconstructs FIRST-ORDER (face = cell mean);
        # all others get full MUSCL. Same realizability signal as the projection itself.
        flagged = [realizability_margin(@view Mext[i, :]) < 0 for i in axes(Mext, 1)]
        for iL in g:(g+Ni)
            ML, MR = recon_faces_proj(flagged[iL], flagged[iL+1],
                                      Vc[iL-1], Vc[iL], Vc[iL+1], Vc[iL+2], Mext[iL, :], Mext[iL+1, :])
            Fhat[iL] = face_flux_1d(ML, MR, axis, Ma; s3max=s3max)
        end
    elseif use_limiter
        for iL in g:(g+Ni)
            ML, MR = recon_faces_limited(Vc[iL-1], Vc[iL], Vc[iL+1], Vc[iL+2])
            Fhat[iL] = face_flux_1d(ML, MR, axis, Ma; s3max=s3max)
        end
    else
        for iL in g:(g+Ni)
            ML, MR = recon_faces_default(Vc[iL-1], Vc[iL], Vc[iL+1], Vc[iL+2], Mext[iL, :], Mext[iL+1, :])
            Fhat[iL] = face_flux_1d(ML, MR, axis, Ma; s3max=s3max)
        end
    end
    R = zeros(Ni, 35)
    for ii in 1:Ni
        c = g + ii                   # cell row in Mext
        R[ii, :] = -(Fhat[c] .- Fhat[c-1]) ./ ds
    end
    return R
end

function residual_ho_3d!(R::Array{Float64,4}, M::Array{Float64,4},
                         nx::Int, ny::Int, nz::Int, halo::Int,
                         dx::Real, dy::Real, dz::Real, Ma::Real;
                         order::Int=2, use_limiter::Bool=false, use_proj_recon::Bool=false,
                         s3max::Real = 4.0 + abs(Ma) / 2.0,
                         dt::Real = 0.0,
                         theta_closed::Bool = true,
                         rank_bnd = (xlo=false, xhi=false, ylo=false, yhi=false,
                                     zlo=false, zhi=false),
                         use_kfvs_anchor::Bool = false)
    # order==3: two-pass WENO5 + joint 6-face θ*-IDP (separate implementation,
    # leaves the order==1,2 path below byte-identical).
    if order == 3
        return residual_ho_3d_order3!(R, M, nx, ny, nz, halo,
                                      dx, dy, dz, Ma, dt; s3max=s3max,
                                      theta_closed=theta_closed, rank_bnd=rank_bnd,
                                      use_kfvs_anchor=use_kfvs_anchor)
    end
    fill!(R, 0.0)
    g = halo
    # X: lines along i (have halos), for each interior (jh,k)
    for k in 1:nz, j in 1:ny
        jh = j + halo
        Mext = @view M[:, jh, k, :]                 # (nx+2halo, 35)
        Rl = residual_line(Mext, dx, 1, Ma; order=order, g=g, use_limiter=use_limiter, use_proj_recon=use_proj_recon, s3max=s3max)   # (nx,35)
        for i in 1:nx; R[i+halo, jh, k, :] .+= Rl[i, :]; end
    end
    # Y: lines along j, for each interior (ih,k)
    for k in 1:nz, i in 1:nx
        ih = i + halo
        Mext = @view M[ih, :, k, :]
        Rl = residual_line(Mext, dy, 2, Ma; order=order, g=g, use_limiter=use_limiter, use_proj_recon=use_proj_recon, s3max=s3max)
        for j in 1:ny; R[ih, j+halo, k, :] .+= Rl[j, :]; end
    end
    # Z: no halo in z -> pad with outflow ghosts (copy edge), for each interior (ih,jh)
    for i in 1:nx, j in 1:ny
        ih = i + halo; jh = j + halo
        col = M[ih, jh, :, :]                        # (nz,35)
        Mext = vcat(repeat(col[1:1,:], g, 1), col, repeat(col[nz:nz,:], g, 1))  # outflow pad
        Rl = residual_line(Mext, dz, 3, Ma; order=order, g=g, use_limiter=use_limiter, use_proj_recon=use_proj_recon, s3max=s3max)   # (nz,35)
        for k in 1:nz; R[ih, jh, k, :] .+= Rl[k, :]; end
    end
    return R
end

# ---------------------------------------------------------------------------
# Projection-activation diagnostic (env-gated, zero-allocation, single cheap
# branch per cell when off).
# Set HYQMOM_PROJ_COUNT=1 to count how many cells _project_interior! actually
# corrects each step.  When the env var is unset (default) the Ref holds false
# and the per-cell `if _PROJ_COUNT_ENABLED[]` costs one Ref dereference + a
# predictably-false branch — zero allocation, no extra function calls.
# ---------------------------------------------------------------------------
const _PROJ_COUNT_ENABLED = Ref{Bool}(
    get(ENV, "HYQMOM_PROJ_COUNT", "0") != "0"
)
const _PROJ_CORRECTIONS   = Ref{Int}(0)

"""
    reset_proj_counter!()

Reset the per-step projection-correction counter to zero. Always resets the counter,
regardless of `HYQMOM_PROJ_COUNT`; the counter is only *incremented* when that env var
is set (otherwise it stays 0). Exported for use by the diagnostic drivers.
"""
reset_proj_counter!() = (_PROJ_CORRECTIONS[] = 0; nothing)

"""
    proj_correction_count() -> Int

Return the current projection-correction count (cells that were unrealizable
before `_project_interior!` corrected them).  Only meaningful when
`HYQMOM_PROJ_COUNT=1`.
"""
proj_correction_count() = _PROJ_CORRECTIONS[]

function _project_interior!(M, nx,ny,nz,halo, Ma, s3max = 4.0 + abs(Ma) / 2.0)
    if _PROJ_COUNT_ENABLED[]
        for k in 1:nz, j in 1:ny, i in 1:nx
            ih=i+halo; jh=j+halo
            # count cells that are unrealizable before projection
            if realizability_margin(@view M[ih,jh,k,:]) < 0
                _PROJ_CORRECTIONS[] += 1
            end
            M[ih,jh,k,:] = realizable_3D_M4(M[ih,jh,k,:], Ma, s3max)
        end
    else
        for k in 1:nz, j in 1:ny, i in 1:nx
            ih=i+halo; jh=j+halo
            M[ih,jh,k,:] = realizable_3D_M4(M[ih,jh,k,:], Ma, s3max)
        end
    end
end

# Sparse projection (the staged-gate "simplest useful fallback"): repair ONLY the
# out-of-cone cells (realizability_margin < 0), leaving strictly-realizable cells
# byte-untouched — so F3 keeps its exactness on the ~99.4% pass set and projection35
# fires only on the thin flagged intervention set. Opt-in; default march never calls it.
const _KFVS_SPARSE_NPROJ = Ref(0)
function _project_flagged_interior!(M, nx,ny,nz,halo, Ma, s3max = 4.0 + abs(Ma)/2.0)
    n = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        ih=i+halo; jh=j+halo
        if realizability_margin(@view M[ih,jh,k,:]) < 0
            M[ih,jh,k,:] = realizable_3D_M4(M[ih,jh,k,:], Ma, s3max); n += 1
        end
    end
    _KFVS_SPARSE_NPROJ[] += n
    return nothing
end

# ---------------------------------------------------------------------------
# Kinetic-FVS anchor → full-cone θ* blend (opt-in; increment E of the KFVS anchor).
# Replaces the per-cell realizability PROJECTION with a blend of the high-order
# update toward a nonnegative-measure anchor, which is full-cone realizable BY
# CONSTRUCTION — so the projection is retired.
#
# Per interior cell (i,j,k):
#   * anchor: kinetic-FVS measure_update of the 7-cell stencil of the stage-INPUT
#     state `Ssrc` (invert each cell's quadrature with the CPU reference
#     `chyqmom_nodes_3d`; retained mass nC_k(1-λ(|Ux|+|Uy|+|Uz|)) + upwind inflow
#     per axis). Realizable by construction under λ·max_k(|Ux|+|Uy|+|Uz|) ≤ 1, and
#     the march CFL (dt = (1/3)dx/vmax) guarantees this at λ = dt/dx.
#   * blend: U(θ) = (1−θ)·U_anchor + θ·U_highorder (U_highorder = the just-updated
#     cell `M[ih,jh,k]`); θ* = largest θ∈[0,1] with U(θ) FULL-CONE realizable
#     (`is_realizable`, = the Δ2* cross-moment cone the projection uses). Bisection.
# The projection is NOT applied on this path. If any stencil cell is degenerate
# (ρ≤0 / inversion fails), the cell falls back to the shipped projection (a safe,
# rare boundary case) so the anchor path never crashes.
#
# z has no halo in the CPU layout; z-neighbors are outflow (edge-copy), matching
# the residual's z-pad. `_ANCHOR_STATS` (env HYQMOM_ANCHOR_STATS=1) records how
# many cells the projection WOULD have corrected (nonzero ⇒ projection retired),
# the mean θ*, and the count.
const _ANCHOR_STATS_ENABLED = Ref{Bool}(get(ENV, "HYQMOM_ANCHOR_STATS", "0") != "0")
const _ANCHOR_WOULD_PROJECT = Ref{Int}(0)      # cells unrealizable pre-blend (proj would fire)
const _ANCHOR_THETA_SUM     = Ref{Float64}(0.0)
const _ANCHOR_CELLS         = Ref{Int}(0)
const _ANCHOR_FALLBACK      = Ref{Int}(0)      # degenerate cells that fell back to projection
const _ANCHOR_THETA_LT1     = Ref{Int}(0)      # cells where θ* < 1 (blend engaged)
const _ANCHOR_THETA_MIN     = Ref{Float64}(1.0)
reset_anchor_stats!() = (_ANCHOR_WOULD_PROJECT[]=0; _ANCHOR_THETA_SUM[]=0.0; _ANCHOR_CELLS[]=0; _ANCHOR_FALLBACK[]=0; _ANCHOR_THETA_LT1[]=0; _ANCHOR_THETA_MIN[]=1.0; nothing)
anchor_stats() = (would_project=_ANCHOR_WOULD_PROJECT[], mean_theta = _ANCHOR_CELLS[]>0 ? _ANCHOR_THETA_SUM[]/_ANCHOR_CELLS[] : 1.0,
                  theta_lt1=_ANCHOR_THETA_LT1[], theta_min=_ANCHOR_THETA_MIN[],
                  cells=_ANCHOR_CELLS[], fallback=_ANCHOR_FALLBACK[])

# quadrature of a raw-moment cell via the HARDENED DEVICE inversion (KFVS increment
# A, FIX-1 gated — no wild-abscissa blowup, never throws). Returns the device tuple
# (n, Ux, Uy, Uz, Nn) or nothing on a degenerate cell (ρ≤0 / no nodes). All nodes
# are checked finite (a belt-and-braces guard); a non-finite node ⇒ degenerate.
@inline function _anchor_quad(m::AbstractVector)
    (length(m) >= 35 && m[1] > 0.0 && all(isfinite, m)) || return nothing
    M = ntuple(t -> Float64(m[t]), Val(35))
    (nn, nux, nuy, nuz, Nn) = chyqmom_nodes_3d_dev(M)
    Nn >= 1 || return nothing
    @inbounds for q in 1:Nn
        (isfinite(nn[q]) && isfinite(nux[q]) && isfinite(nuy[q]) && isfinite(nuz[q])) || return nothing
    end
    return (nn, nux, nuy, nuz, Nn)
end

# accumulate a weighted node into the 35-moment vector Mout (in place).
@inline function _anchor_accum!(Mout, w, ux, uy, uz)
    @inbounds for (idx,(i,j,l)) in enumerate(_ANCHOR_IJK)
        Mout[idx] += w * ux^i * uy^j * uz^l
    end
end
const _ANCHOR_IJK = _CHYQ_TRIPLES   # reuse the 35-triple table from chyqmom_nodes_3d.jl

# Collision-invariant channel indices (F4 conservative re-projection): mass (m000),
# momentum (m100,m010,m001), and the energy-trace diagonal 2nd moments (m200,m020,m002).
# Verified against _CHYQ_TRIPLES ordering.
const _INV_IDX = (1, 2, 6, 16)
const _E_IDX   = (3, 10, 20)

# F4 — CONSERVATIVE re-projection (roadmap Route B). Per-stage replacement for
# projection35's irreducible re-interiorization job (notes sec:idp-drift). Applied ONLY
# to out-of-cone cells; in-cone high-order cells are left untouched (they keep their
# accuracy and are already conservative from the face-shared flux). Each out-of-cone cell
# is set to the FIRST realizable candidate of:
#   (1) measure update M̃ (KFVS quadrature reproduction) with the collision invariants
#       restored — mass+momentum exact, diagonal 2nd scaled to the cell's energy trace;
#       keeps M̃'s high-order (3rd/4th) information when it lands in the cone (~90%);
#   (2) the GAUSSIAN (Maxwellian) closure of the cell's degree≤2 moments — realizable by
#       construction and reproducing mass, momentum, and ALL 2nd moments EXACTLY, so it
#       preserves every collision invariant (the conflict-cell fallback, ~7%).
# BOTH candidates preserve mass/momentum/energy exactly ⇒ the re-projection is
# CONSERVATIVE for the collision invariants (no accumulating energy leak, unlike a
# θ-limited partial restore). Opt-in (`anchor_reproject`, default OFF ⇒ byte-identical
# to F3). Design validated: kfvs_invariant_restore_test.jl, kfvs_f4_validate.jl.
@inline function _reproject_ok(m::NTuple{35,Float64}, Ma, s3max)
    is_realizable(collect(m)) && _marginal_regularized(m, Ma, s3max)
end
# Gaussian closure of a cell's degree≤2 moments; nothing on a degenerate cell
# (ρ≤0 or a non-positive marginal variance). Reproduces mass, momentum, and every 2nd
# moment exactly (m200=ρ(C200+u²)=cell's m200), so it is invariant-exact ⇒ conservative.
@inline function _gaussian_closure(c)
    ρ = c[1]
    ρ > 0.0 || return nothing
    u = c[2]/ρ; v = c[6]/ρ; w = c[16]/ρ
    C200 = c[3]/ρ - u*u; C020 = c[10]/ρ - v*v; C002 = c[20]/ρ - w*w
    (C200 > 0.0 && C020 > 0.0 && C002 > 0.0) || return nothing
    C110 = c[7]/ρ - u*v; C101 = c[17]/ρ - u*w; C011 = c[26]/ρ - v*w
    return InitializeM4_35(ρ, u, v, w, C200, C110, C101, C020, C011, C002)
end
function _anchor_reproject_interior!(M::Array{Float64,4}, nx, ny, nz, halo, Ma, s3max)
    @inbounds for k in 1:nz, j in halo+1:halo+ny, i in halo+1:halo+nx
        c = @view M[i, j, k, :]
        realizability_margin(c) >= 0 && continue          # in-cone: leave untouched
        newc = nothing
        # candidate 1: measure update + invariant restore (keeps M̃'s high-order info)
        q = _anchor_quad(c)
        if q !== nothing
            (nn, nux, nuy, nuz, Nn) = q
            Mt = zeros(35)
            for a in 1:Nn; _anchor_accum!(Mt, nn[a], nux[a], nuy[a], nuz[a]); end
            for t in _INV_IDX; Mt[t] = c[t]; end          # restore mass+momentum exact
            Ecur = Mt[_E_IDX[1]] + Mt[_E_IDX[2]] + Mt[_E_IDX[3]]
            Etar = c[_E_IDX[1]]  + c[_E_IDX[2]]  + c[_E_IDX[3]]
            (Ecur > 0.0) && (s = Etar / Ecur; for t in _E_IDX; Mt[t] *= s; end)
            Bt = NTuple{35,Float64}(Mt)
            _reproject_ok(Bt, Ma, s3max) && (newc = Bt)
        end
        # candidate 2 (conflict cells): Gaussian closure — invariant-exact ⇒ conservative
        if newc === nothing
            G = _gaussian_closure(c)
            if G !== nothing
                Gt = NTuple{35,Float64}(G)
                _reproject_ok(Gt, Ma, s3max) && (newc = Gt)
            end
        end
        newc === nothing && continue                       # nothing realizable: leave as-is (rare)
        for q2 in 1:35; M[i, j, k, q2] = newc[q2]; end
    end
    return nothing
end

function _anchor_interior!(M::Array{Float64,4}, Ssrc::Array{Float64,4},
                           nx,ny,nz,halo, dt, dx, Ma, s3max)
    λ = Float64(dt) / Float64(dx)
    stats = _ANCHOR_STATS_ENABLED[]
    Mout = zeros(35)
    for k in 1:nz, j in 1:ny, i in 1:nx
        ih = i + halo; jh = j + halo
        km = k > 1  ? k-1 : 1      # z outflow edge-copy (no z halo)
        kp = k < nz ? k+1 : nz
        # invert the 7-cell stencil of the stage-INPUT state
        qc  = _anchor_quad(@view Ssrc[ih,   jh,   k, :])
        qxl = _anchor_quad(@view Ssrc[ih-1, jh,   k, :]); qxr = _anchor_quad(@view Ssrc[ih+1, jh,   k, :])
        qyl = _anchor_quad(@view Ssrc[ih,   jh-1, k, :]); qyr = _anchor_quad(@view Ssrc[ih,   jh+1, k, :])
        qzl = _anchor_quad(@view Ssrc[ih,   jh,   km, :]); qzr = _anchor_quad(@view Ssrc[ih,   jh,   kp, :])
        if qc === nothing || qxl === nothing || qxr === nothing ||
           qyl === nothing || qyr === nothing || qzl === nothing || qzr === nothing
            # degenerate stencil → fall back to the shipped projection for this cell
            M[ih,jh,k,:] = realizable_3D_M4(M[ih,jh,k,:], Ma, s3max)
            stats && (_ANCHOR_FALLBACK[] += 1; _ANCHOR_CELLS[] += 1; _ANCHOR_THETA_SUM[] += 1.0)
            continue
        end
        # ---- measure_update (3D): build the nonneg-measure moments in Mout ----
        fill!(Mout, 0.0)
        (nC,uxC,uyC,uzC,NC) = qc
        for c in 1:NC
            ux=uxC[c]; uy=uyC[c]; uz=uzC[c]
            w = nC[c]*(1.0 - λ*(abs(ux)+abs(uy)+abs(uz)))
            _anchor_accum!(Mout, w, ux, uy, uz)
        end
        # x inflow: left nodes Ux>0, right nodes Ux<0
        (nn,ux_,uy_,uz_,NN)=qxl; for c in 1:NN; ux_[c]>0.0 || continue; _anchor_accum!(Mout, λ*nn[c]*ux_[c], ux_[c],uy_[c],uz_[c]); end
        (nn,ux_,uy_,uz_,NN)=qxr; for c in 1:NN; ux_[c]<0.0 || continue; _anchor_accum!(Mout,-λ*nn[c]*ux_[c], ux_[c],uy_[c],uz_[c]); end
        # y inflow
        (nn,ux_,uy_,uz_,NN)=qyl; for c in 1:NN; uy_[c]>0.0 || continue; _anchor_accum!(Mout, λ*nn[c]*uy_[c], ux_[c],uy_[c],uz_[c]); end
        (nn,ux_,uy_,uz_,NN)=qyr; for c in 1:NN; uy_[c]<0.0 || continue; _anchor_accum!(Mout,-λ*nn[c]*uy_[c], ux_[c],uy_[c],uz_[c]); end
        # z inflow
        (nn,ux_,uy_,uz_,NN)=qzl; for c in 1:NN; uz_[c]>0.0 || continue; _anchor_accum!(Mout, λ*nn[c]*uz_[c], ux_[c],uy_[c],uz_[c]); end
        (nn,ux_,uy_,uz_,NN)=qzr; for c in 1:NN; uz_[c]<0.0 || continue; _anchor_accum!(Mout,-λ*nn[c]*uz_[c], ux_[c],uy_[c],uz_[c]); end

        # ---- F2: MARGINALLY-REGULARIZE the anchor so θ=0 is realizable in BOTH the
        # Δ2* cone AND the s3max/variance marginal set. `realizable_3D_M4` applies
        # the marginal clamp (s3max cap + Hankel/variance/S2 floors); its Δ2*
        # projection is a near-no-op here since the measure-update anchor is already
        # in the cross-moment cone by construction. Ua is then the realizable-in-both
        # anchor. (Reuses the shipped projection ONLY on the anchor — a nonneg-measure
        # state — NOT on the high-order update; the blend still recovers high-order.)
        Ua = realizable_3D_M4(Mout, Ma, s3max)
        # ---- θ* blend of U_highorder toward the (regularized) anchor ----
        Uho = ntuple(q -> M[ih,jh,k,q], Val(35))
        if stats
            (realizability_margin(@view M[ih,jh,k,:]) < 0.0 ||
             !_marginal_regularized(Uho, Ma, s3max)) && (_ANCHOR_WOULD_PROJECT[] += 1)
        end
        θ = _anchor_blend_theta(Ua, Uho, Ma, s3max)
        for q in 1:35
            M[ih,jh,k,q] = (1.0-θ)*Ua[q] + θ*Uho[q]
        end
        if stats
            _ANCHOR_CELLS[] += 1; _ANCHOR_THETA_SUM[] += θ
            θ < 1.0 - 1e-9 && (_ANCHOR_THETA_LT1[] += 1)
            θ < _ANCHOR_THETA_MIN[] && (_ANCHOR_THETA_MIN[] = θ)
        end
    end
    return nothing
end

# MARGINAL-regularization predicate (F2). `realizable_3D_M4` applies, beyond the
# Δ2* cross-moment projection, a MARGINAL regularization that `is_realizable`
# (Δ2* only) does NOT test: the s3max standardized-skewness cap on |S300|,|S030|,
# |S003| (Rodney Fox's high-Ma stabilizer), the Hankel/variance floors
# H2ii ≥ h2min and S4ii ≥ S3ii²+1+h2min, the 2nd-order-cross S2 floor, and the
# S220/S202/S022 bounds. The anchor+blend must satisfy this too or high-Ma states
# blow up. This predicate returns true iff M already satisfies that regularization
# (within tolerance); it is the marginal half of the blend's θ* feasibility test.
# Uses the CPU `M2CS4_35` standardization (byte-consistent with `is_realizable`).
@inline function _marginal_regularized(M, Ma, s3max)
    (isfinite(M[1]) && M[1] > 0.0) || return false
    # M2CS4_35 needs an AbstractVector (it calls size()); collect NTuple inputs.
    Mv = M isa AbstractVector ? M : collect(M)
    C4, S4 = M2CS4_35(Mv)
    C200 = C4[3]; C020 = C4[10]; C002 = C4[20]
    (C200 > 0.0 && C020 > 0.0 && C002 > 0.0 && isfinite(C200) && isfinite(C020) && isfinite(C002)) || return false
    h2min = 1.0e-6
    S2min = 1.0e-6
    tol = 1.0e-9   # small slack so an at-the-cap state (produced BY the clamp) passes
    S300=S4[4]; S400=S4[5]; S030=S4[13]; S040=S4[15]; S003=S4[23]; S004=S4[25]
    @inbounds for q in (4,5,13,15,23,25); isfinite(S4[q]) || return false; end
    # s3max skewness cap
    (abs(S300) <= s3max + tol && abs(S030) <= s3max + tol && abs(S003) <= s3max + tol) || return false
    # Hankel/variance floors: H2ii = S4ii - S3ii^2 - 1 ≥ h2min
    (S400 - S300^2 - 1.0 >= h2min - tol) || return false
    (S040 - S030^2 - 1.0 >= h2min - tol) || return false
    (S004 - S003^2 - 1.0 >= h2min - tol) || return false
    # 2nd-order-cross S2 floor
    S110=S4[7]; S101=S4[17]; S011=S4[26]
    (isfinite(S110) && isfinite(S101) && isfinite(S011)) || return false
    S2 = 1.0 + 2.0*S110*S101*S011 - (S110^2 + S101^2 + S011^2)
    (S2 >= S2min - tol) || return false
    return true
end

# largest θ∈[0,1] keeping (1-θ)Ua + θ Uho realizable in BOTH the Δ2* cross-moment
# cone (`is_realizable`) AND the marginal s3max/variance regularization
# (`_marginal_regularized`) — F2. Ua is the anchor AFTER `realizable_3D_M4`, so it
# satisfies both ⇒ θ=0 always feasible ⇒ θ* well-defined and the blended output is
# realizable-by-construction in BOTH senses (no post-blend clamp: s3max is
# nonlinear, so a post-clamp of a blend could re-violate — instead θ* is limited to
# never leave the marginal set in the first place).
@inline function _anchor_blend_theta(Ua::Vector{Float64}, Uho::NTuple{35,Float64}, Ma, s3max; nb::Int=30)
    blend(θ) = ntuple(q -> (1.0-θ)*Ua[q] + θ*Uho[q], Val(35))
    ok(m) = is_realizable(collect(m); lam_min=0.0) && _marginal_regularized(m, Ma, s3max)
    ok(blend(1.0)) && return 1.0
    lo=0.0; hi=1.0
    for _ in 1:nb
        mid = 0.5*(lo+hi)
        ok(blend(mid)) ? (lo=mid) : (hi=mid)
    end
    return lo
end

function step_highorder_3d!(M::Array{Float64,4}, dt::Real, decomp, bc::Symbol,
                            nx,ny,nz,halo, dx,dy,dz, Ma;
                            order::Int=2, use_limiter::Bool=false, use_proj_recon::Bool=false,
                            stage_bgk_kn=nothing, s3max::Real = 4.0 + abs(Ma) / 2.0,
                            use_kfvs_anchor::Bool=false, anchor_reproject::Bool=false,
                            sparse_project::Bool=false)
    R = similar(M)
    int = (halo+1:halo+nx, halo+1:halo+ny, 1:nz, :)
    # A side is a RANK boundary iff a real neighbour rank sits there (encoded as a
    # non-negative rank; -1 = MPI_PROC_NULL sentinel = global domain boundary). Only
    # order==3 consumes rank_bnd (order 1/2 ignore the kwarg → byte-identical). z is
    # never decomposed, so its rank-boundary flags stay false (the axis-generic
    # z path in the residual is present but DORMANT on the CPU).
    rank_bnd = (xlo = decomp.neighbors.left  != -1,
                xhi = decomp.neighbors.right != -1,
                ylo = decomp.neighbors.down  != -1,
                yhi = decomp.neighbors.up    != -1,
                zlo = false, zhi = false)
    # stage helper: M_in (with halos) -> returns updated interior-only array (full M-shape, halos zero)
    function L!(Mwork)
        halo_exchange_3d!(Mwork, decomp, bc)
        residual_ho_3d!(R, Mwork, nx,ny,nz,halo, dx,dy,dz, Ma; order=order, use_limiter=use_limiter, use_proj_recon=use_proj_recon, s3max=s3max, dt=Float64(dt), rank_bnd=rank_bnd, use_kfvs_anchor=use_kfvs_anchor)
        return R
    end
    # stage_bgk (opt-in, `stage_bgk_kn` = Kn): exact-exponential BGK relaxation of
    # every interior cell after each stage's projection. Each SSP-RK3 building
    # block is then a Lie-split forward-Euler step; the convex combinations keep
    # first-order splitting consistency (same formal order as the legacy
    # once-per-step collision). At Kn=0 every stage output is Maxwellian, so
    # transient M300 can never flux M200 in a later stage — this closes the
    # collisionless-stage error channel at contacts (see test_rodney_cases.jl).
    # `bgk_relax_tup` is the single-source helper shared with the GPU path.
    function bgk!(Mwork)
        stage_bgk_kn === nothing && return nothing
        kn = Float64(stage_bgk_kn); dtf = Float64(dt)
        @inbounds for k in 1:nz, j in halo+1:halo+ny, i in halo+1:halo+nx
            Mt = ntuple(q -> Mwork[i, j, k, q], Val(35))
            out = bgk_relax_tup(Mt, dtf, kn)
            for q in 1:35
                Mwork[i, j, k, q] = out[q]
            end
        end
        return nothing
    end
    # Per-stage realizability step.
    #  * default (use_kfvs_anchor=false): the shipped `_project_interior!` projection
    #    (byte-identical to main).
    #  * F3 (use_kfvs_anchor=true): NO per-cell step. The residual's FLUX-level blend
    #    F = F_KFVS + θ·(F_HO − F_KFVS) with a FACE-SHARED θ (min over the two adjacent
    #    cells) and the FULL-CONE + marginal θ* predicate makes the update BOTH
    #    conservative (single-valued face flux ⇒ mass/mom/energy telescope exactly,
    #    thm:idp-conservative) AND full-cone realizable by construction
    #    (thm:idp-cons-real). So projection35 is RETIRED with no post-hoc per-cell
    #    projection or state blend (the E/F2 `_anchor_interior!` state blend is NOT
    #    used — it was non-conservative; see cor:proj-noncons).
    #  * F4 (use_kfvs_anchor=true, anchor_reproject=true): conservative per-stage
    #    re-projection of out-of-cone cells only (`_anchor_reproject_interior!`), the
    #    Route-B fix for the multi-step cross-cone drift (notes sec:idp-drift). Opt-in;
    #    with anchor_reproject=false (default) F3 is byte-identical.
    realize!(Mwork) =
        use_kfvs_anchor ? (sparse_project ? _project_flagged_interior!(Mwork, nx,ny,nz, halo, Ma, s3max) :
                           anchor_reproject ? _anchor_reproject_interior!(Mwork, nx,ny,nz, halo, Ma, s3max) : nothing) :
                          _project_interior!(Mwork, nx,ny,nz, halo, Ma, s3max)

    M0 = copy(M)
    # stage 1: M1 = M + dt*L(M)
    L!(M); @views M[int...] .= M0[int...] .+ dt .* R[int...]; realize!(M); bgk!(M)
    # stage 2: M2 = 3/4 M0 + 1/4 (M1 + dt L(M1))
    L!(M); @views M[int...] .= (3/4).*M0[int...] .+ (1/4).*(M[int...] .+ dt .* R[int...]); realize!(M); bgk!(M)
    # stage 3: M = 1/3 M0 + 2/3 (M2 + dt L(M2))
    L!(M); @views M[int...] .= (1/3).*M0[int...] .+ (2/3).*(M[int...] .+ dt .* R[int...]); realize!(M); bgk!(M)
    halo_exchange_3d!(M, decomp, bc)
    return nothing
end
