# kfvs_f4_measure_reproject_scope.jl — SCOPING EXPERIMENT for F4 / roadmap Route B.
#
# The defect-identity diagnostic (kfvs_defect_identity_diagnostic.jl) established:
#   * the F3 cross-cone drift is GATED by proximity to ∂ℛ (pre-step margin, ρ=-0.96)
#     and TRIGGERED by the reproduction defect D=M-M̃ in the transverse 4th cross
#     moments (m202,m112,m022,m004); ‖D‖-magnitude does NOT predict drift (ρ=-0.47);
#   * for 100% of drifted cells the MEASURE update U^Q = M̃ - λΣΔF is IN-cone.
#
# So the measure update is already the invariant-domain-safe object. This script
# scopes the F4 fix (roadmap §2.5 Route B) with the cheapest decisive experiment:
# carry the measure update as the state — re-project M ← M̃ (quadrature reproduction)
# on the interior AFTER each step — and measure BOTH:
#   (i)  DRIFT: does removing D drive the Δ2* cross-cone violations to ~0? (expected:
#        M̃ is the moment vector of a nonneg atomic measure ⇒ realizable BY
#        CONSTRUCTION, so a re-quad'd field should not drift.)
#   (ii) CONSERVATION DEBT: re-quad changes the global moment totals by ΣD each step.
#        The linear collision invariants (mass, momentum, energy) and the transverse
#        4th-moment totals are tracked. This debt is EXACTLY what a conservative Route-B
#        redistribution must move; its size decides whether redistribution is needed or
#        the raw measure projection is already acceptable.
#
# This re-quad projection is NOT the final scheme (it is not conservative — that is the
# point being measured). It brackets the problem: (i) confirms U^Q-as-state is drift-free;
# (ii) quantifies the debt Route B must redistribute.
#
# CPU-only. Production path unchanged.  HYQMOM_F4_STEPS default "8,20,40".

ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
MPI.Initialized() || MPI.Init()
const RM = Riemann35

# transverse 4th-order cross-moment channel indices (the ejecting D channels)
const TRIP = collect(RM._ANCHOR_IJK)
chan_idx(t) = findfirst(==(t), TRIP)
const IX_TRANS = [chan_idx((2,0,2)), chan_idx((1,1,2)), chan_idx((0,2,2)), chan_idx((0,0,4))]

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

function requad(m::AbstractVector)
    q = RM._anchor_quad(collect(m)); q === nothing && return nothing
    (nn, nux, nuy, nuz, Nn) = q; Mt = zeros(35)
    @inbounds for c in 1:Nn; RM._anchor_accum!(Mt, nn[c], nux[c], nuy[c], nuz[c]); end
    return Mt
end

# interior totals: mass, |momentum|, energy, and Σ transverse-4th channels
function totals(M, nx, ny, nz, halo)
    mass = 0.0; px = 0.0; py = 0.0; pz = 0.0; en = 0.0; tr = zeros(4)
    for k in 1:nz, j in 1:ny, i in 1:nx
        c = @view M[i+halo, j+halo, k, :]
        mass += c[1]; px += c[2]; py += c[6]; pz += c[16]
        en += c[3] + c[10] + c[20]
        for (q, ix) in enumerate(IX_TRANS); tr[q] += c[ix]; end
    end
    (mass=mass, mom=sqrt(px^2+py^2+pz^2), en=en, tr=copy(tr))
end

reldrift(a, b) = abs(a - b) / max(abs(b), 1e-300)

function drift_stats(M, nx, ny, nz, halo)
    nneg = 0; worst = 0.0
    for k in 1:nz, j in 1:ny, i in 1:nx
        mg = realizability_margin(Vector{Float64}(M[i+halo, j+halo, k, :]))
        (mg < 0) && (nneg += 1; worst = min(worst, mg))
    end
    nneg, worst
end

# march nsteps of F3; if reproject, do M←M̃ on interior after each step and
# accumulate the per-step conservation debt ‖ΣD‖ (relative, in each tracked total).
function run(Np, halo, Ma, s3max, nsteps, dt, reproject::Bool)
    M, nx, ny, nz = build_ic(Np, halo, Ma)
    comm = MPI.COMM_WORLD; bc = :copy
    decomp = setup_mpi_cartesian_3d(Np, Np, Np, halo, comm)
    halo_exchange_3d!(M, decomp, bc)
    T0 = totals(M, nx, ny, nz, halo)
    debt_en = 0.0; debt_tr = 0.0; nproj_fail = 0
    for s in 1:nsteps
        step_highorder_3d!(M, dt, decomp, bc, nx, ny, nz, halo, dx_g, dx_g, dx_g, Ma;
                           order = 3, s3max = s3max, use_kfvs_anchor = true)
        if reproject
            Tpre = totals(M, nx, ny, nz, halo)
            for k in 1:nz, j in 1:ny, i in 1:nx
                c = Vector{Float64}(M[i+halo, j+halo, k, :]); Mt = requad(c)
                Mt === nothing && (nproj_fail += 1; continue)
                M[i+halo, j+halo, k, :] = Mt
            end
            halo_exchange_3d!(M, decomp, bc)
            Tpost = totals(M, nx, ny, nz, halo)
            debt_en += reldrift(Tpost.en, Tpre.en)
            debt_tr += reldrift(sum(abs, Tpost.tr), sum(abs, Tpre.tr))
        end
    end
    T1 = totals(M, nx, ny, nz, halo)
    nneg, worst = drift_stats(M, nx, ny, nz, halo)
    return (nneg=nneg, worst=worst,
            mass=reldrift(T1.mass, T0.mass), mom=reldrift(T1.mom, T0.mom),
            en=reldrift(T1.en, T0.en), tr=reldrift(sum(abs,T1.tr), sum(abs,T0.tr)),
            debt_en=debt_en, debt_tr=debt_tr, pfail=nproj_fail)
end

const dx_g = 1.0 / 16
function main()
    Ma = 100.0; Np = 16; halo = 4
    dt = 0.12 * dx_g / (Ma / 2 + 5)
    s3max = max(40.0, 4.0 + abs(Ma) / 2.0)
    set_kfvs_xfloor!(0.0)
    steps = parse.(Int, split(get(ENV, "HYQMOM_F4_STEPS", "8,20,40"), ","))
    println("KFVS F4 measure-reproject SCOPING  (Ma=$Ma, Np=$Np)   transverse chans idx=$IX_TRANS")
    println("="^96)
    @printf("%-26s %6s %11s | %9s %9s %9s %9s | %10s %10s\n",
            "variant / steps", "drift", "worst", "mass", "mom", "energy", "Σtrans", "debt-en", "debt-tr")
    for ns in steps
        b = run(Np, halo, Ma, s3max, ns, dt, false)
        @printf("%-26s %6d %11.3e | %9.2e %9.2e %9.2e %9.2e | %10s %10s\n",
                "baseline F3  $(ns)step", b.nneg, b.worst, b.mass, b.mom, b.en, b.tr, "-", "-")
        r = run(Np, halo, Ma, s3max, ns, dt, true)
        @printf("%-26s %6d %11.3e | %9.2e %9.2e %9.2e %9.2e | %10.2e %10.2e  (pfail=%d)\n",
                "F4 reproject $(ns)step", r.nneg, r.worst, r.mass, r.mom, r.en, r.tr, r.debt_en, r.debt_tr, r.pfail)
    end
    println("\nREAD: drift→0 under reproject ⇒ U^Q is the IDP-safe state (D removal kills drift).")
    println("      debt-en / debt-tr = cumulative per-step conservation debt the reproject INTRODUCES")
    println("      = exactly what a conservative Route-B redistribution must move. Compare to baseline")
    println("      energy drift: if debt ≫ baseline, redistribution is mandatory; if ≈, projection is cheap.")
    println("DONE.")
end
main()
