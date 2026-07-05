"""
accuracy_vs_muscl.jl — rigorous spatial-accuracy comparison: order-3 (WENO5 + θ*-IDP)
vs order-2 (default MUSCL — exactly the "straight port"). Smooth sinusoidal density,
3-pt Gauss cell-average IC (avoids the point-value order-2 artifact). Measures the
residual self-convergence order for BOTH schemes vs a fine reference, plus the error
magnitude at a fixed grid. The claim under test: order-3 is genuinely higher-order
(≈5) than MUSCL (≈2), so it needs far fewer cells for the same error.

Run:  \$JULIA --project=. test/accuracy_vs_muscl.jl
"""

using MPI
MPI.Initialized() || MPI.Init()
using Riemann35
using Printf
using Statistics: mean

mw35_iso(rho, ux, T) = collect(InitializeM4_35(rho, ux, 0.0, 0.0, T, 0.0, 0.0, T, 0.0, T))
gauss3avg(f, xc, dx) = (xi = sqrt(3/5);
    (5/18)*f(xc - xi*dx/2) + (8/18)*f(xc) + (5/18)*f(xc + xi*dx/2))

# x-varying sinusoidal density (y,z uniform), periodic-x / copy-y halos.
function build(nx, halo; ny=4, nz=4, Ma=2.0, T0=1.0, A=0.3)
    dx = 1.0/nx; dy = 1.0/ny; dz = 1.0/nz
    M = zeros(Float64, nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:ny, i in 1:nx
        xc = (i-0.5)*dx
        rho = gauss3avg(x -> 1.0 + A*sin(2*pi*x), xc, dx)
        M[i+halo, j+halo, k, :] .= mw35_iso(rho, Ma, T0)
    end
    for g in 1:halo                                        # periodic x
        M[g,:,:,:]        .= M[nx+g,:,:,:]
        M[nx+halo+g,:,:,:] .= M[halo+g,:,:,:]
    end
    for g in 1:halo                                        # copy y
        M[:,g,:,:]        .= M[:,halo+1,:,:]
        M[:,ny+halo+g,:,:] .= M[:,ny+halo,:,:]
    end
    M, dx, dy, dz
end

# density-residual x-profile (interior, central y/z line)
function resid_profile(order, nx)
    halo = order == 3 ? 4 : 2
    ny = 4; nz = 4
    M, dx, dy, dz = build(nx, halo; ny=ny, nz=nz)
    R = zeros(Float64, size(M))
    residual_ho_3d!(R, M, nx, ny, nz, halo, dx, dy, dz, 2.0; order=order, dt=0.0, s3max=40.0)
    [R[halo+i, halo+1, 1, 1] for i in 1:nx]                # moment-1 (density) residual
end

restrict(p, r) = [mean(@view p[(i-1)*r+1 : i*r]) for i in 1:(length(p)÷r)]

const NREF = 96
const GRIDS = [16, 24, 32, 48]

function study(order)
    ref = resid_profile(order, NREF)
    errs = Float64[]
    for nx in GRIDS
        rc = resid_profile(order, nx)
        rr = restrict(ref, NREF ÷ nx)
        push!(errs, mean(abs.(rc .- rr)))
    end
    errs
end

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("="^70)
    println("SPATIAL ACCURACY: order-3 (WENO5+θ*-IDP) vs order-2 (MUSCL straight port)")
    println("smooth ρ=1+0.3sin(2πx), residual self-convergence vs $(NREF)³ ref")
    println("="^70)
    e2 = study(2); e3 = study(3)
    @printf("\n  %-6s %-13s %-8s   %-13s %-8s\n", "nx", "L1_MUSCL(o2)", "order", "L1_WENO(o3)", "order")
    println("  " * "-"^54)
    for (i, nx) in enumerate(GRIDS)
        o2 = i == 1 ? "—" : @sprintf("%.2f", log(e2[i-1]/e2[i])/log(GRIDS[i]/GRIDS[i-1]))
        o3 = i == 1 ? "—" : @sprintf("%.2f", log(e3[i-1]/e3[i])/log(GRIDS[i]/GRIDS[i-1]))
        @printf("  %-6d %-13.4e %-8s   %-13.4e %-8s\n", nx, e2[i], o2, e3[i], o3)
    end
    ord2 = log(e2[1]/e2[end])/log(GRIDS[end]/GRIDS[1])
    ord3 = log(e3[1]/e3[end])/log(GRIDS[end]/GRIDS[1])
    @printf("\n  Overall order  (nx %d→%d):  MUSCL=%.2f   WENO5+IDP=%.2f\n",
            GRIDS[1], GRIDS[end], ord2, ord3)
    @printf("  Error ratio at nx=%d (MUSCL/WENO): %.1fx  — order-3 is that much more accurate\n",
            GRIDS[end], e2[end]/e3[end])
    println()
    println(ord3 - ord2 >= 1.5 ? "RESULT: order-3 is clearly higher-order than MUSCL." :
                                  "RESULT: order gap smaller than expected — inspect.")
end
