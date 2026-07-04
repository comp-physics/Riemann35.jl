# ---------------------------------------------------------------------------
# Order-3 (WENO5 + θ*-IDP) modules — included here so they stay out of the
# default order=1,2 code path (byte-identical guarantee). The modules create
# sub-modules of Riemann35 and their names are brought into this file's scope.
# ---------------------------------------------------------------------------
include(joinpath(@__DIR__, "weno5_dev.jl"));       using .Weno5Dev: weno5z, deconv5, conv5, smooth5
include(joinpath(@__DIR__, "idp_limiter_dev.jl")); using .IdpLimiterDev: theta_star_update_dev

# Device-safe validity of a reconstructed recon-var vector (the recon layout has
# density at slot 1 = M000 and the three variances at slots 5,6,7 = C200,C020,C002).
# from_recon_vars_dev takes sqrt of the three variances, so a WENO5 overshoot that
# drives any variance ≤ 0 (or density ≤ 0) would sqrt a negative. We must test this
# BEFORE the conversion — on the GPU there is no throw to catch, only a silent NaN.
@inline _recon_vars_realizable(v::NTuple{35,Float64}) =
    v[1] > 0.0 && v[5] > 0.0 && v[6] > 0.0 && v[7] > 0.0

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
                        g::Int = 4, s3max::Real = 40.0)
    n2g = size(Mext, 1)
    n   = n2g - 2g
    @assert g >= 4 "order=3 residual requires halo g ≥ 4 (WENO5+deconv stencils); got g=$g"
    @assert n >= 1 "line must contain ≥ 1 interior cell; got n=$n"

    # --- Step 1: raw deconvolved point values → recon-var point values -------
    # Per-component smooth5 gate: if the 5-cell stencil for component q is
    # smooth, apply deconv5 (O(h^6) avg→point); otherwise keep the cell average.
    # Cells within 2 of the array boundary cannot access a full ±2 stencil:
    # fall back to the cell average for those (they are ghost cells anyway).
    Ppt = Vector{NTuple{35,Float64}}(undef, n2g)
    for k in 1:n2g
        rawpt = ntuple(Val(35)) do q
            if k >= 3 && k <= n2g - 2
                a, b, c, d, e = Mext[k-2,q], Mext[k-1,q], Mext[k,q], Mext[k+1,q], Mext[k+2,q]
                smooth5(a, b, c, d, e) ? deconv5(a, b, c, d, e) : c
            else
                Mext[k, q]          # boundary: stencil OOB → cell average fallback
            end
        end
        Ppt[k] = to_recon_vars_dev(rawpt...)
    end

    # --- Step 2: forward-convolve recon-var point values → recon-var averages --
    # conv5 is the O(h^6) inverse of deconv5: converts point values to the
    # corresponding cell averages so weno5z reconstructs from cell averages
    # (skipping this step caps effective order at 2). Stencil indices are
    # clamped to [1..n2g] at the array boundary (replicates the outermost
    # ghost Ppt value — only affects the outermost 2 cells, which are ghosts).
    Vavg = Vector{NTuple{35,Float64}}(undef, n2g)
    for k in 1:n2g
        Vavg[k] = ntuple(Val(35)) do q
            p(j) = Ppt[clamp(k + j, 1, n2g)][q]
            conv5(p(-2), p(-1), p(0), p(1), p(2))
        end
    end

    # --- Step 3: WENO5 face reconstruction + F_HO / F_LO per interface ------
    # Interface f (1..n+1) lies between Mext rows il = g+f-1 and ir = g+f.
    # Left reconstruction (weno5z at il, right-going stencil):
    #   uses Vavg[il-2..il+2]; for g=4 and f=1, il=4 → indices 2..6 — valid.
    # Right reconstruction (weno5z at ir, left-going = reversed stencil):
    #   uses Vavg[ir-2..ir+2]; for f=n+1, ir=g+n → indices g+n-2..g+n+2 ≤ n2g-2 — valid.
    # Clamping guards against the two outermost ghost positions (indices 1 and n2g)
    # being accessed by Vavg's conv5; it does NOT widen the face stencil itself.
    F_HO = Vector{NTuple{35,Float64}}(undef, n + 1)
    F_LO = Vector{NTuple{35,Float64}}(undef, n + 1)
    for f in 1:n+1
        il = g + f - 1    # left cell in Mext (1-indexed)
        ir = il + 1

        # WENO5 left-side reconstruction (right-going, 5-cell stencil centred at il)
        vL = ntuple(Val(35)) do q
            weno5z(Vavg[clamp(il-2,1,n2g)][q], Vavg[clamp(il-1,1,n2g)][q],
                   Vavg[il][q],
                   Vavg[clamp(il+1,1,n2g)][q], Vavg[clamp(il+2,1,n2g)][q])
        end
        # WENO5 right-side reconstruction (left-going, stencil reversed around ir)
        vR = ntuple(Val(35)) do q
            weno5z(Vavg[clamp(ir+2,1,n2g)][q], Vavg[clamp(ir+1,1,n2g)][q],
                   Vavg[ir][q],
                   Vavg[clamp(ir-1,1,n2g)][q], Vavg[clamp(ir-2,1,n2g)][q])
        end

        # Realizability fallback: if a reconstructed face state exits the moment
        # cone, replace it with the cell mean (first-order at that face state).
        # STAGE 1 (pre-conversion, no throw / no NaN): a WENO5 overshoot can drive
        # a reconstructed variance ≤ 0; from_recon_vars_dev would then sqrt a
        # negative (CPU: DomainError; GPU: silent NaN). Guard the recon-var vector
        # first and fall back to the cell mean BEFORE converting.
        cL = ntuple(q -> Mext[il, q], Val(35))
        cR = ntuple(q -> Mext[ir, q], Val(35))
        mL = _recon_vars_realizable(vL) ? from_recon_vars_dev(vL...) : cL
        mR = _recon_vars_realizable(vR) ? from_recon_vars_dev(vR...) : cR
        # STAGE 2 (post-conversion): net any remaining out-of-cone raw face state.
        RiemannFluxDev._state_realizable(mL) || (mL = cL)
        RiemannFluxDev._state_realizable(mR) || (mR = cR)

        F_HO[f] = _face_flux_tup(mL, mR, axis, Ma, s3max)
        F_LO[f] = _face_flux_tup(cL, cR, axis, Ma, s3max)
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
function residual_ho_3d_order3!(R::Array{Float64,4}, M::Array{Float64,4},
                                nx::Int, ny::Int, nz::Int, halo::Int,
                                dx::Real, dy::Real, dz::Real, Ma::Real,
                                dt::Real; s3max::Real = 40.0)
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
        Fho, Flo = residual_line3(Mext, dx, 1, Ma; g=g, s3max=s3max)
        for f in 1:nx+1
            FHO_x[f,j,k] = Fho[f];  FLO_x[f,j,k] = Flo[f]
        end
    end

    # Y-axis lines
    for k in 1:nz, i in 1:nx
        ih   = i + halo
        Mext = @view M[ih, :, k, :]
        Fho, Flo = residual_line3(Mext, dy, 2, Ma; g=g, s3max=s3max)
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
        Fho, Flo = residual_line3(Mext, dz, 3, Ma; g=g, s3max=s3max)
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

        # High-order corrections G = F_HO - F_LO (per face)
        Gxr = ntuple(q -> FHO_x[i+1,j,k][q] - FLO_x[i+1,j,k][q], Val(35))
        Gxl = ntuple(q -> FHO_x[i,  j,k][q] - FLO_x[i,  j,k][q], Val(35))
        Gyr = ntuple(q -> FHO_y[i,j+1,k][q] - FLO_y[i,j+1,k][q], Val(35))
        Gyl = ntuple(q -> FHO_y[i,j,  k][q] - FLO_y[i,j,  k][q], Val(35))
        Gzr = ntuple(q -> FHO_z[i,j,k+1][q] - FLO_z[i,j,k+1][q], Val(35))
        Gzl = ntuple(q -> FHO_z[i,j,k  ][q] - FLO_z[i,j,k  ][q], Val(35))

        # Per-face θ* (one-sided, factor-6 conservative bound)
        Θ_xr[i,j,k] = theta_star_update_dev(Mlo, ntuple(q -> -6λx * Gxr[q], Val(35)))
        Θ_xl[i,j,k] = theta_star_update_dev(Mlo, ntuple(q ->  6λx * Gxl[q], Val(35)))
        Θ_yr[i,j,k] = theta_star_update_dev(Mlo, ntuple(q -> -6λy * Gyr[q], Val(35)))
        Θ_yl[i,j,k] = theta_star_update_dev(Mlo, ntuple(q ->  6λy * Gyl[q], Val(35)))
        Θ_zr[i,j,k] = theta_star_update_dev(Mlo, ntuple(q -> -6λz * Gzr[q], Val(35)))
        Θ_zl[i,j,k] = theta_star_update_dev(Mlo, ntuple(q ->  6λz * Gzl[q], Val(35)))
    end

    # =========================================================================
    # PASS 2b+2c: interface θ = min over the two adjacent cells, blend fluxes,
    # accumulate residual R = -Σ_axis (F_right - F_left) / ds_axis.
    # Conservation: both cells sharing an interface see the SAME blended flux
    # (same θ = min of the two cells' per-face bounds) → fluxes telescope exactly.
    # =========================================================================
    for k in 1:nz, j in 1:ny, i in 1:nx
        ih = i + halo;  jh = j + halo

        # Interface θ = min(this cell, neighbour cell); boundary faces use only
        # the one adjacent interior cell (ghost cells do not constrain θ here).
        θxr = (i < nx)  ? min(Θ_xr[i,j,k], Θ_xl[i+1,j,k]) : Θ_xr[i,j,k]
        θxl = (i > 1)   ? min(Θ_xl[i,j,k], Θ_xr[i-1,j,k]) : Θ_xl[i,j,k]
        θyr = (j < ny)  ? min(Θ_yr[i,j,k], Θ_yl[i,j+1,k]) : Θ_yr[i,j,k]
        θyl = (j > 1)   ? min(Θ_yl[i,j,k], Θ_yr[i,j-1,k]) : Θ_yl[i,j,k]
        θzr = (k < nz)  ? min(Θ_zr[i,j,k], Θ_zl[i,j,k+1]) : Θ_zr[i,j,k]
        θzl = (k > 1)   ? min(Θ_zl[i,j,k], Θ_zr[i,j,k-1]) : Θ_zl[i,j,k]

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
                         dt::Real = 0.0)
    # order==3: two-pass WENO5 + joint 6-face θ*-IDP (separate implementation,
    # leaves the order==1,2 path below byte-identical).
    if order == 3
        return residual_ho_3d_order3!(R, M, nx, ny, nz, halo,
                                      dx, dy, dz, Ma, dt; s3max=s3max)
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

function step_highorder_3d!(M::Array{Float64,4}, dt::Real, decomp, bc::Symbol,
                            nx,ny,nz,halo, dx,dy,dz, Ma;
                            order::Int=2, use_limiter::Bool=false, use_proj_recon::Bool=false,
                            stage_bgk_kn=nothing, s3max::Real = 4.0 + abs(Ma) / 2.0)
    R = similar(M)
    int = (halo+1:halo+nx, halo+1:halo+ny, 1:nz, :)
    # stage helper: M_in (with halos) -> returns updated interior-only array (full M-shape, halos zero)
    function L!(Mwork)
        halo_exchange_3d!(Mwork, decomp, bc)
        residual_ho_3d!(R, Mwork, nx,ny,nz,halo, dx,dy,dz, Ma; order=order, use_limiter=use_limiter, use_proj_recon=use_proj_recon, s3max=s3max, dt=Float64(dt))
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
    M0 = copy(M)
    # stage 1: M1 = M + dt*L(M)
    L!(M); @views M[int...] .= M0[int...] .+ dt .* R[int...]; _project_interior!(M,nx,ny,nz,halo,Ma,s3max); bgk!(M)
    # stage 2: M2 = 3/4 M0 + 1/4 (M1 + dt L(M1))
    L!(M); @views M[int...] .= (3/4).*M0[int...] .+ (1/4).*(M[int...] .+ dt .* R[int...]); _project_interior!(M,nx,ny,nz,halo,Ma,s3max); bgk!(M)
    # stage 3: M = 1/3 M0 + 2/3 (M2 + dt L(M2))
    L!(M); @views M[int...] .= (1/3).*M0[int...] .+ (2/3).*(M[int...] .+ dt .* R[int...]); _project_interior!(M,nx,ny,nz,halo,Ma,s3max); bgk!(M)
    halo_exchange_3d!(M, decomp, bc)
    return nothing
end
