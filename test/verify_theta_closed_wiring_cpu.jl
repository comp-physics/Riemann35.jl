"""
verify_theta_closed_wiring_cpu.jl — CPU-side validation of the theta_closed
opt-in wiring in residual_ho_3d_order3! (src/numerics/highorder_3d.jl).

Checks, on a small 3D order-3 residual with a nonzero dt (θ actually binds):
  (1) theta_closed=true (the current default, PR#19) is byte-for-byte identical to
      the historical call (no kwarg). relL2 exactly 0.0.
  (2) theta_closed=true vs false: closed-form vs bisection differ only by the θ*
      limiter's ~1e-6 (bisection 2^-24) resolution, i.e. tiny relL2, and the
      marched state stays finite/realizable.

The GPU byte-identity gate (flag off == prior kernel bit-for-bit) is run
separately (test/verify_theta_closed_gpu.jl); this is the CPU analogue and a
fast smoke that the wiring compiles and behaves.
"""

using MPI
MPI.Initialized() || MPI.Init()
using Riemann35
using Riemann35: residual_ho_3d_order3!
using Printf, Random

# Build a small haloed order-3 field (halo=4) with a smooth, realizable,
# spatially-varying state so the high-order correction is nonzero and θ binds.
function build_field(nx, ny, nz, halo, Ma)
    n2gx = nx + 2halo; n2gy = ny + 2halo
    M = zeros(n2gx, n2gy, nz, 35)
    for k in 1:nz, jh in 1:n2gy, ih in 1:n2gx
        x = (ih - halo - 0.5)/nx; y = (jh - halo - 0.5)/ny; z = (k - 0.5)/nz
        rho = 1.0 + 0.3*sin(2π*x)*cos(2π*y) + 0.2*sin(2π*z)
        u   = 0.4*cos(2π*x); v = 0.3*sin(2π*y); w = 0.2*cos(2π*z)
        T   = 0.5 + 0.2*cos(2π*x)*sin(2π*z)
        # Maxwellian-ish 35-moment (product of per-axis Gaussians about (u,v,w),T)
        # raw moments m0..m4 per axis:
        rm(uu) = (1.0, uu, T+uu^2, 3T*uu+uu^3, 3T^2+6T*uu^2+uu^4)
        mx = rm(u); my = rm(v); mz = rm(w)
        M[ih,jh,k,:] .= ntuple(35) do q
            # IJK exponents
            ijk = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
                   (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),
                   (0,3,0),(1,3,0),(0,4,0),
                   (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),
                   (0,0,3),(1,0,3),(0,0,4),
                   (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),
                   (0,1,2),(1,1,2),(0,1,3),(0,2,2))[q]
            (i,j,l) = ijk
            rho * mx[i+1]*my[j+1]*mz[l+1]
        end
    end
    return M
end

nx=ny=nz=6; halo=4; Ma=2.0; dx=1.0/nx
M = build_field(nx, ny, nz, halo, Ma)
dt = 0.15*dx   # nonzero ⇒ θ* binds

R_hist = zeros(nx+2halo, ny+2halo, nz, 35)   # historical (no kwarg)
R_off  = zeros(nx+2halo, ny+2halo, nz, 35)
R_on   = zeros(nx+2halo, ny+2halo, nz, 35)

residual_ho_3d_order3!(R_hist, M, nx, ny, nz, halo, dx, dx, dx, Ma, dt; s3max=40.0)
residual_ho_3d_order3!(R_off,  M, nx, ny, nz, halo, dx, dx, dx, Ma, dt; s3max=40.0, theta_closed=false)
residual_ho_3d_order3!(R_on,   M, nx, ny, nz, halo, dx, dx, dx, Ma, dt; s3max=40.0, theta_closed=true)

# metric helpers
relL2(a,b) = (nrm = sqrt(sum(abs2, b)); sqrt(sum(abs2, a .- b)) / (nrm == 0 ? 1.0 : nrm))
bytes_identical(a,b) = all(reinterpret(UInt64, vec(a)) .== reinterpret(UInt64, vec(b)))

@printf("== CPU wiring: theta_closed opt-in ==\n")
@printf("  default (no kwarg) vs flag ON (closed, the default):\n")
@printf("     bit-identical:  %s\n", bytes_identical(R_hist, R_on))
@printf("     relL2:          %.3e\n", relL2(R_hist, R_on))
@printf("  flag ON vs OFF (closed vs bisection):\n")
@printf("     relL2:          %.3e\n", relL2(R_on, R_off))
@printf("     max|Δ|:         %.3e\n", maximum(abs, R_on .- R_off))
@printf("     all finite ON:  %s\n", all(isfinite, R_on))

@assert bytes_identical(R_hist, R_on) "default (no kwarg) is NOT the closed-form path!"
@assert relL2(R_on, R_off) < 1e-4 "closed form disagrees too much: $(relL2(R_on,R_off))"
@assert all(isfinite, R_on) "flag ON produced non-finite residual!"

# --- marched-state stability with the flag ON: a few forward-Euler substeps of
# the θ*-limited residual must keep the interior finite, rho>0, realizable ---
using Riemann35.RiemannFluxDev: _state_realizable
Mw = copy(M)
Rw = zeros(nx+2halo, ny+2halo, nz, 35)
dt_march = 0.1*dx
nreal = ntot = 0
for s in 1:10
    residual_ho_3d_order3!(Rw, Mw, nx, ny, nz, halo, dx, dx, dx, Ma, dt_march;
                           s3max=40.0, theta_closed=true)
    @. Mw += dt_march * Rw          # interior slots get updated; halos re-used (smooth IC)
end
global nreal, ntot
for k in 1:nz, j in 1:ny, i in 1:nx
    global nreal, ntot
    m = ntuple(q -> Mw[i+halo, j+halo, k, q], 35)
    ntot += 1
    _state_realizable(m) && (nreal += 1)
end
rhomin = minimum(@view Mw[halo+1:halo+nx, halo+1:halo+ny, :, 1])
@printf("  flag ON marched-state stability (10 Euler substeps):\n")
@printf("     finite:      %s\n", all(isfinite, @view Mw[halo+1:halo+nx, halo+1:halo+ny, :, :]))
@printf("     rho_min:     %.3e  (>0: %s)\n", rhomin, rhomin > 0.0)
@printf("     realizable:  %d / %d\n", nreal, ntot)
@assert rhomin > 0.0 "flag ON march produced rho<=0!"
@assert nreal == ntot "flag ON march produced non-realizable cells: $(ntot-nreal)"
println("\nOK: default==closed byte-identical; closed vs bisection agrees to ~limiter resolution, finite, rho>0, realizable.")
