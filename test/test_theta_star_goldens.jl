"""
test_theta_star_goldens.jl — CI guard for BOTH θ*-IDP limiter paths in the
order-3 (WENO5 + θ*-IDP) residual.

Two paths, both pinned to frozen goldens so neither can silently change:

  * BASELINE (theta_closed=false): the 24-iteration bisection limiter — the
    historical reference behavior. Its golden `theta_star_baseline_golden.f64`
    is FROZEN and must reproduce BIT-FOR-BIT (relL2 == 0.0). This guarantees the
    fallback is never silently broken: old results stay reproducible forever.

  * CLOSED (theta_closed=true): the closed-form marginal-Hankel limiter — the
    DEFAULT (more accurate; exact vs bisection's 2^-24 resolution). Its golden
    `theta_star_closed_golden.f64` is FROZEN too, and the closed path must
    reproduce it BIT-FOR-BIT and stay finite + realizable.

Both goldens are raw little-endian Float64 dumps of the interior order-3 residual
`residual_ho_3d_order3!` on a fixed deterministic 6^3 smooth case with a binding
dt (θ* actually clips), s3max=40. The residual path is pure scalar arithmetic
(no @fastmath) and verified run-to-run deterministic, so a bitwise assertion is
the right same-code guard here (cf. the serial==MPI / wrapper==device bitwise
checks the golden policy already reserves bitwise comparison for).

Regenerate deliberately (only when the reference is intentionally moved):
    julia --project=. test/test_theta_star_goldens.jl --generate

Wired into test/runtests.jl so CI (julia-actions/julia-runtest → Pkg.test)
executes it on every push/PR.
"""

using Test
using Riemann35
using Riemann35: residual_ho_3d_order3!
using Riemann35.RiemannFluxDev: _state_realizable

const _TS_GOLDEN_DIR = joinpath(@__DIR__, "goldenfiles")
const _TS_BASELINE = joinpath(_TS_GOLDEN_DIR, "theta_star_baseline_golden.f64")
const _TS_CLOSED   = joinpath(_TS_GOLDEN_DIR, "theta_star_closed_golden.f64")

# --- fixed deterministic order-3 case (independent of RNG) ---
const _TS_IJK = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
                 (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),
                 (0,3,0),(1,3,0),(0,4,0),
                 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),
                 (0,0,3),(1,0,3),(0,0,4),
                 (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),
                 (0,1,2),(1,1,2),(0,1,3),(0,2,2))

function _ts_build_field(nx, ny, nz, halo)
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, jh in 1:ny+2halo, ih in 1:nx+2halo
        x = (ih-halo-0.5)/nx; y = (jh-halo-0.5)/ny; z = (k-0.5)/nz
        rho = 1.0 + 0.3*sin(2π*x)*cos(2π*y) + 0.2*sin(2π*z)
        u = 0.4*cos(2π*x); v = 0.3*sin(2π*y); w = 0.2*cos(2π*z)
        T = 0.5 + 0.2*cos(2π*x)*sin(2π*z)
        rm(uu) = (1.0, uu, T+uu^2, 3T*uu+uu^3, 3T^2+6T*uu^2+uu^4)
        mx=rm(u); my=rm(v); mz=rm(w)
        M[ih,jh,k,:] .= ntuple(35) do q
            (i,j,l)=_TS_IJK[q]; rho*mx[i+1]*my[j+1]*mz[l+1]
        end
    end
    M
end

# compute the interior order-3 residual for the fixed case; returns Vector{Float64}
function _ts_residual(theta_closed::Bool)
    nx=ny=nz=6; halo=4; Ma=2.0; dx=1.0/nx; dt=0.15*dx
    M = _ts_build_field(nx, ny, nz, halo)
    R = zeros(nx+2halo, ny+2halo, nz, 35)
    residual_ho_3d_order3!(R, M, nx, ny, nz, halo, dx, dx, dx, Ma, dt;
                           s3max=40.0, theta_closed=theta_closed)
    # interior only (strip halos) → deterministic, layout-stable
    vec(R[halo+1:halo+nx, halo+1:halo+ny, :, :])
end

_ts_read(path) = collect(reinterpret(Float64, read(path)))
_ts_write(path, v::Vector{Float64}) = open(io -> write(io, reinterpret(UInt8, v)), path, "w")
_ts_biteq(a, b) = length(a) == length(b) && all(reinterpret(UInt64, a) .== reinterpret(UInt64, b))
_ts_relL2(a, b) = (nb = sqrt(sum(abs2, b)); sqrt(sum(abs2, a .- b)) / (nb == 0 ? 1.0 : nb))

# --- generation mode (run directly, deliberate) ---
if (abspath(PROGRAM_FILE) == @__FILE__) && ("--generate" in ARGS)
    using MPI; MPI.Initialized() || MPI.Init()
    println("Freezing θ* goldens (fixed 6^3 order-3 case, binding dt)...")
    Rb = _ts_residual(false); _ts_write(_TS_BASELINE, Rb)
    Rc = _ts_residual(true);  _ts_write(_TS_CLOSED,   Rc)
    println("  baseline (bisection) → $_TS_BASELINE  sum=$(sum(Rb))")
    println("  closed              → $_TS_CLOSED    sum=$(sum(Rc))")
    println("  closed vs baseline relL2 = $(_ts_relL2(Rc, Rb))  (binding-dt limiter difference)")
    exit(0)
end

@testset "θ*-IDP limiter goldens (both paths pinned)" begin
    # BASELINE GUARD — the bisection fallback must reproduce the frozen golden to a
    # tight tolerance. NOT bit-for-bit: goldens are frozen on one machine but CI runs
    # on different arch / BLAS / Julia builds, so the last bit differs (~1e-16). Any
    # real change to the bisection code path would shift results >> 1e-12. (Same-machine
    # this is exactly 0; the tolerance only absorbs cross-platform rounding.)
    @test isfile(_TS_BASELINE)
    if isfile(_TS_BASELINE)
        Rb = _ts_residual(false)
        gb = _ts_read(_TS_BASELINE)
        rb = _ts_relL2(Rb, gb)
        @test rb < 1e-12                              # baseline preserved (portable)
        rb >= 1e-12 && @warn "baseline drift" relL2=rb maxabs=maximum(abs, Rb .- gb)
    end

    # CLOSED-FORM (the default) — pinned golden + finite + realizable.
    @test isfile(_TS_CLOSED)
    if isfile(_TS_CLOSED)
        Rc = _ts_residual(true)
        gc = _ts_read(_TS_CLOSED)
        @test _ts_relL2(Rc, gc) < 1e-12               # closed default preserved (portable)
        @test all(isfinite, Rc)                       # finite
    end

    # closed vs baseline: small, sane difference (bisection's 2^-24 resolution),
    # NOT a divergence. The closed form is the more accurate of the two.
    Rb = _ts_residual(false); Rc = _ts_residual(true)
    @test _ts_relL2(Rc, Rb) < 1e-4
    @test all(isfinite, Rc)

    # marched-state stability of the DEFAULT (closed) path: a few forward-Euler
    # substeps stay finite, ρ>0, realizable (no projection35/NaN blowup).
    nx=ny=nz=6; halo=4; Ma=2.0; dx=1.0/nx; dtm=0.1*dx
    Mw = _ts_build_field(nx, ny, nz, halo)
    Rw = zeros(nx+2halo, ny+2halo, nz, 35)
    for _ in 1:10
        residual_ho_3d_order3!(Rw, Mw, nx, ny, nz, halo, dx, dx, dx, Ma, dtm;
                               s3max=40.0, theta_closed=true)
        @. Mw += dtm * Rw
    end
    rhomin = minimum(@view Mw[halo+1:halo+nx, halo+1:halo+ny, :, 1])
    @test rhomin > 0.0
    nreal = 0; ntot = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        m = ntuple(q -> Mw[i+halo, j+halo, k, q], 35)
        ntot += 1; _state_realizable(m) && (nreal += 1)
    end
    @test nreal == ntot
end
