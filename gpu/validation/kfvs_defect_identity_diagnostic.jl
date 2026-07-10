# kfvs_defect_identity_diagnostic.jl — DIAGNOSTIC (not a feature): decide whether
# the F3 anchor's Δ2* cross-cone drift is SEEDED by the quadrature reproduction
# defect D(M) = M - M̃  (M̃ = M[Q(M)], the moments regenerated from the CHyQMOM
# inversion nodes) or is an independent multi-step dynamical instability.
#
# Roadmap Opportunity 1, Phase I (research-opportunity-roadmap.md §2.1, §2.3, §2.9).
#
# The flux-form F3 update is, cell-wise and EXACTLY (the six-face fluxes depend only
# on neighbour states, not on the cell's own starting point):
#     U^F3 = M      - λ ΣΔF        (raw-moment start)
#     U^Q  = M̃      - λ ΣΔF        (measure-update start; realizable by construction)
#   ⇒ U^F3 = U^Q + D,     D = M - M̃.
# The cone is convex but NOT invariant under translation by the signed defect D, so
# U^Q ∈ ℛ does not imply U^F3 ∈ ℛ. This script measures whether D actually accounts
# for the observed cone exits.
#
# DISCRIMINATING MEASUREMENTS (not confirmatory):
#   (1) CORRELATION: across all interior cells at step n+1, is the cone violation
#       (realizability_margin < 0) concentrated in the cells with the largest ‖D^n‖?
#       Report margin split by ‖D‖ quartile. A defect-seeded drift ⇒ violations pile
#       into the high-‖D‖ cells; a pure dynamical instability need not.
#   (2) SEGMENT s*: for each DRIFTED cell, walk U(s) = U^Q + s·D, s∈[0,1], with
#       U^Q = U^F3 - D^n (D^n from the PRE-step field). Report margin at s=0 (U^Q)
#       and s=1 (U^F3=drifted), and the crossing s*. If U^Q is IN-cone (margin≥0 at
#       s=0) and the drifted state is OUT, D is provably the ejecting displacement.
#       If U^Q is ALSO out of cone, the measure update itself is off-cone ⇒ NOT a
#       pure reproduction-defect story (inversion or dynamics is implicated).
#   (3) D=0 EXACT-MEASURE IC CONTROL (non-circular): re-project the INITIAL field to
#       the quadrature image once (M_ic ← M̃_ic, so D≈0 at t=0), then march F3 with
#       NO per-step projection. If the D=0 IC delays/reduces drift onset relative to
#       the raw IC, and drift regrows as ‖D‖ regrows, the causal link is established.
#       (Per-step requad would be circular — requad lands in ℛ by construction — so
#       it is deliberately avoided; only the IC is projected.)
#   Also reported: CHyQMOM idempotency ‖M̃ - M̃̃‖ / ‖M̃‖ (is M̃ a near-fixed-point?),
#   and which moment channels dominate D on the drifted cells.
#
# CPU-only. Production path unchanged. Slow (KFVS inversion per face per stage).
#   HYQMOM_DIAG_STEPS (default 20)  — pre-step horizon n for (1)(2).
#   HYQMOM_CTRL_STEPS (default 20)  — horizon for the (3) IC control comparison.

ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
MPI.Initialized() || MPI.Init()

const RM = Riemann35

# ---- IC identical to kfvs_f3_drift_diagnostic.jl (Ma=100 crossing jets) ----------
function build_ic(Np, halo, Ma)
    C = 1.0; vj = 0.15 * Ma
    bg = InitializeM4_35(0.05, 0.0, 0.0, 0.0, C, 0.0, 0.0, C, 0.0, C)
    l  = InitializeM4_35(1.0,  vj,  vj / 2, 0.0, C, 0.0, 0.0, C, 0.0, C)
    r  = InitializeM4_35(1.0, -vj, -vj / 2, 0.0, C, 0.0, 0.0, C, 0.0, C)
    nx = ny = nz = Np; M = zeros(Float64, nx + 2halo, ny + 2halo, nz, 35)
    Cs = max(1, floor(Int, 0.2 * Np))
    lo = div(Np, 2) - Cs; hi = div(Np, 2); lo2 = div(Np, 2) + 1; hi2 = div(Np, 2) + 1 + Cs
    for k in 1:nz, i in 1:nx, j in 1:ny
        Mr = bg
        (lo <= i <= hi && lo <= j <= hi)     && (Mr = l)
        (lo2 <= i <= hi2 && lo2 <= j <= hi2) && (Mr = r)
        M[i + halo, j + halo, k, :] = Mr
    end
    M, nx, ny, nz
end

# M̃ = M[Q(M)] : regenerate the 35 moments from the CHyQMOM device inversion nodes,
# EXACTLY the nodes the F3 flux uses. Returns nothing on a degenerate (fallback) cell.
function requad(m::AbstractVector)
    q = RM._anchor_quad(collect(m))
    q === nothing && return nothing
    (nn, nux, nuy, nuz, Nn) = q
    Mt = zeros(35)
    @inbounds for c in 1:Nn
        RM._anchor_accum!(Mt, nn[c], nux[c], nuy[c], nuz[c])
    end
    return Mt
end

# scale-aware relative defect norm (nondimensionalize by |M| componentwise)
defect_relnorm(M, Mt) = norm((M .- Mt) ./ (abs.(M) .+ 1e-300)) / sqrt(35)

function march!(M, decomp, bc, nx, ny, nz, halo, dx, Ma, s3max, nsteps, dt)
    for _ in 1:nsteps
        step_highorder_3d!(M, dt, decomp, bc, nx, ny, nz, halo, dx, dx, dx, Ma;
                           order = 3, s3max = s3max, use_kfvs_anchor = true)
    end
end

function main()
    Ma = 100.0; Np = 16; halo = 4; dx = 1.0 / Np
    dt = 0.12 * dx / (Ma / 2 + 5)
    s3max = max(40.0, 4.0 + abs(Ma) / 2.0)
    nstep = parse(Int, get(ENV, "HYQMOM_DIAG_STEPS", "20"))
    cstep = parse(Int, get(ENV, "HYQMOM_CTRL_STEPS", "20"))
    comm = MPI.COMM_WORLD; bc = :copy
    set_kfvs_xfloor!(0.0)

    println("KFVS F3 defect-identity diagnostic  (Ma=$Ma, Np=$Np, dt=$(@sprintf("%.3e",dt)))")
    println("="^78)

    # ---------------------------------------------------------------- pre-step field
    M, nx, ny, nz = build_ic(Np, halo, Ma)
    decomp = setup_mpi_cartesian_3d(Np, Np, Np, halo, comm)
    halo_exchange_3d!(M, decomp, bc)
    march!(M, decomp, bc, nx, ny, nz, halo, dx, Ma, s3max, nstep, dt)   # M = M^n

    # per interior cell: D^n = M^n - M̃^n, PRE-step margin, plus idempotency ‖M̃-M̃̃‖
    idx = Tuple{Int,Int,Int}[]; Draw = Float64[]; iderr = Float64[]; Mpre = Float64[]
    Msnap = Dict{Tuple{Int,Int,Int},Vector{Float64}}()
    Dsnap = Dict{Tuple{Int,Int,Int},Vector{Float64}}()
    ndegen = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        c = Vector{Float64}(M[i+halo, j+halo, k, :])
        Mt = requad(c)
        if Mt === nothing; ndegen += 1; continue; end
        Mtt = requad(Mt)
        push!(idx, (i,j,k)); push!(Draw, defect_relnorm(c, Mt))
        push!(iderr, Mtt === nothing ? NaN : defect_relnorm(Mt, Mtt))
        push!(Mpre, realizability_margin(c))     # margin BEFORE the +1 step
        Msnap[(i,j,k)] = c; Dsnap[(i,j,k)] = c .- Mt
    end
    @printf("\n[pre-step n=%d] interior cells=%d  degenerate(fallback)=%d\n", nstep, length(idx), ndegen)
    @printf("  ‖D‖ (rel)   : min=%.2e  median=%.2e  max=%.2e\n",
            minimum(Draw), median(Draw), maximum(Draw))
    @printf("  idempotency ‖M̃-M̃̃‖/‖M̃‖ : median=%.2e  max=%.2e   (small ⇒ M̃ ≈ quadrature fixed point)\n",
            median(filter(isfinite, iderr)), maximum(filter(isfinite, iderr)))

    # ---------------------------------------------------------- one more step: M^{n+1}
    march!(M, decomp, bc, nx, ny, nz, halo, dx, Ma, s3max, 1, dt)       # M = M^{n+1}

    # (1) CORRELATION: cone violation vs ‖D^n‖ quartile ---------------------------
    rows = Tuple{Float64,Float64,Tuple{Int,Int,Int}}[]  # (‖D‖, margin_{n+1}, idx)
    for (t, id) in enumerate(idx)
        (i,j,k) = id
        mg = realizability_margin(Vector{Float64}(M[i+halo, j+halo, k, :]))
        push!(rows, (Draw[t], mg, id))
    end
    sort!(rows, by = r -> r[1])                          # ascending ‖D‖
    nq = length(rows); qs = [1, nq÷4, nq÷2, 3nq÷4, nq]
    println("\n(1) CORRELATION  cone-violation vs ‖D^n‖ quartile  (margin at step n+1):")
    println("     quartile        ‖D‖ range            #cells   #(margin<0)   worst margin")
    for b in 1:4
        a = qs[b] == 0 ? 1 : (b==1 ? 1 : qs[b]+1); z = qs[b+1]
        a = max(a, 1); seg = rows[a:z]
        nneg = count(r -> r[2] < 0, seg)
        worst = isempty(seg) ? 0.0 : minimum(r -> r[2], seg)
        @printf("     Q%d  [%.2e .. %.2e]   %5d      %5d        %.3e\n",
                b, seg[1][1], seg[end][1], length(seg), nneg, worst)
    end
    allneg = count(r -> r[2] < 0, rows)
    # rank correlation of ‖D‖ with (-margin) among all cells
    dvec = [r[1] for r in rows]; mvec = [-r[2] for r in rows]
    rho = cor(sortperm(sortperm(dvec)) .* 1.0, sortperm(sortperm(mvec)) .* 1.0)
    @printf("     total drifted (margin<0): %d / %d   Spearman ρ(‖D‖, -margin)=%.3f\n",
            allneg, nq, rho)

    # (1b) COMPETING HYPOTHESIS: cone violation vs PRE-STEP margin quartile --------
    # Tests whether proximity-to-∂ℛ (small pre-step margin), NOT ‖D‖, predicts drift.
    postmg = Dict(rows[t][3] => rows[t][2] for t in 1:nq)
    rows2 = [(Mpre[t], postmg[idx[t]], idx[t]) for t in 1:length(idx)]  # (pre-margin, post-margin, idx)
    sort!(rows2, by = r -> r[1])                          # ascending pre-step margin
    qs2 = [1, nq÷4, nq÷2, 3nq÷4, nq]
    println("\n(1b) COMPETING  cone-violation vs PRE-STEP margin quartile (proximity to ∂ℛ):")
    println("     quartile     pre-margin range        #cells   #(post margin<0)   worst post-margin")
    for b in 1:4
        a = b==1 ? 1 : qs2[b]+1; z = qs2[b+1]; seg = rows2[a:z]
        nneg = count(r -> r[2] < 0, seg)
        worst = isempty(seg) ? 0.0 : minimum(r -> r[2], seg)
        @printf("     Q%d  [%+.2e .. %+.2e]   %5d      %5d          %.3e\n",
                b, seg[1][1], seg[end][1], length(seg), nneg, worst)
    end
    pvec = [r[1] for r in rows2]; qvec = [-r[2] for r in rows2]
    rho2 = cor(sortperm(sortperm(pvec)) .* 1.0, sortperm(sortperm(qvec)) .* 1.0)
    # mean pre-step margin: drifted vs safe
    dr_pre = [Mpre[t] for t in 1:length(idx) if postmg[idx[t]] < 0]
    sf_pre = [Mpre[t] for t in 1:length(idx) if postmg[idx[t]] >= 0]
    @printf("     Spearman ρ(pre-margin, -post-margin)=%.3f   |  mean pre-margin: drifted=%.3e  safe=%.3e\n",
            rho2, isempty(dr_pre) ? NaN : mean(dr_pre), isempty(sf_pre) ? NaN : mean(sf_pre))

    # (2) SEGMENT s*: for drifted cells, U(s)=U^Q + s·D,  U^Q = M^{n+1} - D^n --------
    drifted = [r[3] for r in rows if r[2] < 0]
    println("\n(2) SEGMENT  U(s)=U^Q + s·D  on drifted cells (U^Q = M^{n+1} − D^n):")
    if isempty(drifted)
        println("     (no drifted cells at this horizon)")
    else
        nUQ_in = 0; s_list = Float64[]; margin_UF3 = Float64[]
        # dominant D-channel tally over drifted cells
        chan = zeros(Float64, 35)
        for id in drifted
            (i,j,k) = id
            UF3 = Vector{Float64}(M[i+halo, j+halo, k, :])   # s=1
            D   = Dsnap[id]
            UQ  = UF3 .- D                                    # s=0
            m0 = realizability_margin(UQ); m1 = realizability_margin(UF3)
            m0 >= 0 && (nUQ_in += 1)
            push!(margin_UF3, m1)
            # bisection crossing s* (margin ≥ 0 → < 0)
            if m0 >= 0 && m1 < 0
                lo = 0.0; hi = 1.0
                for _ in 1:40
                    mid = 0.5*(lo+hi)
                    (realizability_margin(UQ .+ mid .* D) >= 0) ? (lo = mid) : (hi = mid)
                end
                push!(s_list, lo)
            end
            chan .+= abs.(D) ./ (abs.(UF3) .+ 1e-300)
        end
        @printf("     drifted cells=%d   U^Q in-cone (margin≥0 at s=0): %d (%.0f%%)\n",
                length(drifted), nUQ_in, 100 * nUQ_in / length(drifted))
        if !isempty(s_list)
            @printf("     crossing s*  : min=%.3f  median=%.3f  max=%.3f   (fraction of D that ejects)\n",
                    minimum(s_list), median(s_list), maximum(s_list))
        end
        @printf("     drifted margins at s=1 (U^F3): median=%.3e  worst=%.3e\n",
                median(margin_UF3), minimum(margin_UF3))
        top = sortperm(chan, rev=true)[1:6]
        trip = RM._ANCHOR_IJK
        @printf("     top D channels (by rel weight over drifted cells): %s\n",
                join(["m$(trip[t])=$(@sprintf("%.2f",chan[t]/length(drifted)))" for t in top], "  "))
    end

    # (3) D=0 EXACT-MEASURE IC CONTROL --------------------------------------------
    println("\n(3) D=0 IC CONTROL  (project IC to quadrature image once, then march; no per-step projection):")
    function drift_count(M0field)
        Mc = copy(M0field)
        dcp = setup_mpi_cartesian_3d(Np, Np, Np, halo, comm)
        halo_exchange_3d!(Mc, dcp, bc)
        march!(Mc, dcp, bc, nx, ny, nz, halo, dx, Ma, s3max, cstep, dt)
        nneg = 0; worst = 0.0; d0 = Float64[]
        for k in 1:nz, j in 1:ny, i in 1:nx
            mg = realizability_margin(Vector{Float64}(Mc[i+halo, j+halo, k, :]))
            (mg < 0) && (nneg += 1; worst = min(worst, mg))
        end
        nneg, worst
    end
    # raw IC
    Mraw, _, _, _ = build_ic(Np, halo, Ma)
    nraw, wraw = drift_count(Mraw)
    # D=0 IC: project every interior cell to its quadrature image
    Mp, _, _, _ = build_ic(Np, halo, Ma)
    nproj = 0; d0proj = Float64[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        c = Vector{Float64}(Mp[i+halo, j+halo, k, :]); Mt = requad(c)
        if Mt === nothing; nproj += 1; continue; end
        push!(d0proj, defect_relnorm(Mt, requad(Mt) === nothing ? Mt : requad(Mt)))
        Mp[i+halo, j+halo, k, :] = Mt
    end
    npj, wpj = drift_count(Mp)
    @printf("     raw   IC  → after %d steps: drifted=%4d  worst=%.3e\n", cstep, nraw, wraw)
    @printf("     D=0   IC  → after %d steps: drifted=%4d  worst=%.3e   (residual IC ‖D‖ median=%.2e)\n",
            cstep, npj, wpj, isempty(d0proj) ? NaN : median(d0proj))
    println("\nDONE.")
end
main()
