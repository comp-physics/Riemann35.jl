# kfvs_debt_by_order.jl — DIAGNOSTIC: decompose the re-quad reproduction defect
# D = M - M̃ (M̃ = M[Q(M)]) by MOMENT ORDER, to decide how expensive Route B is.
#
# The F4 scoping run (kfvs_f4_measure_reproject_scope.jl) showed that carrying the
# measure update (M←M̃ each step) kills the cross-cone drift but injects an ENERGY
# conservation debt ~4.2e-4 over 20 steps. Energy = m200+m020+m002 is a LINEAR
# collision invariant, so that debt is a real physical-conservation violation.
#
# QUESTION: is the energy (2nd-order) debt an inversion/regularization artifact that a
# cheap per-cell rescale could remove (→ Route B is cheap: restore the invariant part,
# tolerate the non-invariant 3rd/4th-moment debt as projection35 already did), or is it
# intrinsic (→ a full conservative redistribution / SDP is needed)?
#
# For the F3-marched field at N steps, per interior cell compute D by total degree
# p ∈ {0,1,2,3,4}, and report:
#   * per-cell RELATIVE defect ‖D_p‖/‖M_p‖ (median, max)  — which orders the inversion
#     fails to reproduce;
#   * GLOBAL conservation debt |Σ_cells D_p|_1 / |Σ_cells M_p|_1  — what re-quad injects
#     into each order's global total (this is the per-step debt, by order);
#   * ENERGY channels (m200,m020,m002) specifically: per-cell and global.
# If the global energy debt is tiny and the debt piles into the non-invariant 4th order,
# Route B is cheap. If the 2nd-order/energy debt is large and spread, it is not.
#
# CPU-only.  HYQMOM_DEBT_STEPS default "8,20".

ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
MPI.Initialized() || MPI.Init()
const RM = Riemann35

const TRIP = collect(RM._ANCHOR_IJK)
const ORD  = [t[1]+t[2]+t[3] for t in TRIP]                 # total degree per channel
const GRP  = [findall(==(p), ORD) for p in 0:4]            # channel indices by order
const IX_E = [findfirst(==(t), TRIP) for t in ((2,0,0),(0,2,0),(0,0,2))]  # energy
const IX_T = [findfirst(==(t), TRIP) for t in ((2,0,2),(1,1,2),(0,2,2),(0,0,4))]  # transverse 4th

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

function analyze(Np, halo, Ma, s3max, nsteps, dt)
    M, nx, ny, nz = build_ic(Np, halo, Ma)
    comm = MPI.COMM_WORLD; bc = :copy
    decomp = setup_mpi_cartesian_3d(Np, Np, Np, halo, comm)
    halo_exchange_3d!(M, decomp, bc)
    for _ in 1:nsteps
        step_highorder_3d!(M, dt, decomp, bc, nx, ny, nz, halo, 1.0/Np, 1.0/Np, 1.0/Np, Ma;
                           order = 3, s3max = s3max, use_kfvs_anchor = true)
    end
    # per-order relative per-cell defect + global sums
    relp = [Float64[] for _ in 0:4]
    sumD = zeros(35); sumM = zeros(35)
    ecell = Float64[]                          # per-cell relative energy defect
    ndeg = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        c = Vector{Float64}(M[i+halo, j+halo, k, :]); Mt = requad(c)
        Mt === nothing && (ndeg += 1; continue)
        D = c .- Mt
        for (pi, p) in enumerate(0:4)
            g = GRP[pi]; nM = norm(@view c[g]); nD = norm(@view D[g])
            push!(relp[pi], nD / (nM + 1e-300))
        end
        sumD .+= D; sumM .+= c
        E = sum(@view c[IX_E]); dE = sum(@view D[IX_E])
        push!(ecell, abs(dE) / (abs(E) + 1e-300))
    end
    @printf("\n[N=%d steps]  interior=%d  degenerate=%d\n", nsteps, length(ecell), ndeg)
    println("  per-cell RELATIVE defect ‖D_p‖/‖M_p‖ and GLOBAL debt |ΣD_p|/|ΣM_p| by order:")
    println("    order   #chan    per-cell median    per-cell max      global debt |ΣD_p|₁/|ΣM_p|₁")
    for (pi, p) in enumerate(0:4)
        g = GRP[pi]
        gd = sum(abs, @view sumD[g]) / (sum(abs, @view sumM[g]) + 1e-300)
        @printf("    p=%d     %3d      %.3e         %.3e         %.3e\n",
                p, length(g), median(relp[pi]), maximum(relp[pi]), gd)
    end
    gE  = abs(sum(@view sumD[IX_E])) / (abs(sum(@view sumM[IX_E])) + 1e-300)
    gT  = sum(abs, @view sumD[IX_T]) / (sum(abs, @view sumM[IX_T]) + 1e-300)
    @printf("  ENERGY (m200+m020+m002): per-cell |ΔE|/E median=%.3e max=%.3e | GLOBAL |ΣΔE|/|ΣE|=%.3e\n",
            median(ecell), maximum(ecell), gE)
    @printf("  TRANSVERSE 4th (m202,m112,m022,m004): GLOBAL debt |ΣD|/|ΣM|=%.3e\n", gT)
end

function main()
    Ma = 100.0; Np = 16; halo = 4
    dt = 0.12 * (1.0/Np) / (Ma / 2 + 5)
    s3max = max(40.0, 4.0 + abs(Ma) / 2.0)
    set_kfvs_xfloor!(0.0)
    println("KFVS re-quad DEBT-BY-ORDER  (Ma=$Ma, Np=$Np)   energy idx=$IX_E  transverse idx=$IX_T")
    println("="^92)
    for ns in parse.(Int, split(get(ENV, "HYQMOM_DEBT_STEPS", "8,20"), ","))
        analyze(Np, halo, Ma, s3max, ns, dt)
    end
    println("\nREAD: if GLOBAL energy debt |ΣΔE|/|ΣE| ≪ the transverse-4th debt and small in absolute")
    println("      terms, the invariant part is nearly conservative already → Route B is cheap (restore")
    println("      the 2nd-order part, tolerate non-invariant high-order debt). If energy debt is large")
    println("      and spread across many cells, a conservative redistribution is required.")
    println("DONE.")
end
main()
