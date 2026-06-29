#!/usr/bin/env julia
# run_gpu_crossing_jets.jl — end-to-end GPU example: build a crossing-jet IC, advance
# it on the GPU, and stream snapshots to JLD2 in the canonical schema so the existing
# readers / visualization consume the output unchanged.
#
# Runs under the GPU project (gpuenv2). Single-GPU by default; pass an MPI comm to
# GPURun.run_gpu_3d for multi-GPU (z-slab). The IC is built self-contained here (a
# diagonal-covariance Gaussian per cell — same state InitializeM4_35 produces — so the
# example needs no package internals).
#
# Run:
#   srun --mpi=pmix -n 1 --gpus=1 julia --project=gpu/gpuenv2 examples/run_gpu_crossing_jets.jl
#   # then visualize/analyze in the MAIN package env:
#   julia --project=. -e 'using Riemann35; Riemann35.interactive_3d_timeseries_streaming("crossing_jets_gpu.jl…")'
#
# Env: set RIEMANN35_DATA / output dir as you like; defaults to the repo (CWD).
using CUDA, Printf
include(joinpath(@__DIR__, "..", "gpu", "gpu_run.jl")); using .GPURun

# ---- canonical 35-moment exponent ordering (M_n = <vx^i vy^j vz^k f>) ----
const TRIPLES = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),
 (1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),
 (1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),
 (0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]

# raw moment of a 1D Gaussian(mean=mu, var=s2), order m (0..4)
@inline _g(m, mu, s2) = m == 0 ? 1.0 : m == 1 ? mu : m == 2 ? mu^2 + s2 :
                        m == 3 ? mu^3 + 3mu*s2 : mu^4 + 6mu^2*s2 + 3s2^2
# 35-moment Gaussian cell (diagonal covariance T) -> writes into M[:,i,j,k]
@inline function set_cell!(M, i, j, k, rho, u, v, w, T)
    @inbounds for n in 1:35
        a, b, c = TRIPLES[n]
        M[n, i, j, k] = rho * _g(a, u, T) * _g(b, v, T) * _g(c, w, T)
    end
end

# ---- parameters (override via ENV) ----
Nx = parse(Int, get(ENV, "NX", "48")); Ny = Nx
Nz = parse(Int, get(ENV, "NZ", "16"))          # set NZ=1 for a 2D (single-GPU) run
Ma = parse(Float64, get(ENV, "MA", "5.0"))
nstep = parse(Int, get(ENV, "NSTEP", "20"))
snap_int = parse(Int, get(ENV, "SNAP", "5"))
T0 = 1.0
dx = 1.0 / Nx
outfile = get(ENV, "OUT", joinpath(pwd(), @sprintf("crossing_jets_gpu_Nx%d_Nz%d_Ma%.1f.jld2", Nx, Nz, Ma)))

# ---- crossing-jet IC: low-density background + two dense cold jets moving toward
#      the center from opposite diagonal corners (they cross at the middle). ----
M0 = Array{Float64}(undef, 35, Nx, Ny, Nz)
U = Ma                                            # jet speed (T0=1 => Ma ~ |u|)
rho_bg, rho_jet, T_jet = 1.0e-2, 1.0, 0.2
incube(cx, cy, x, y, hw) = (abs(x - cx) <= hw && abs(y - cy) <= hw)
for k in 1:Nz, j in 1:Ny, i in 1:Nx
    x = (i - 0.5) * dx; y = (j - 0.5) * dx
    if incube(0.30, 0.30, x, y, 0.10)
        set_cell!(M0, i, j, k, rho_jet,  U,  U, 0.0, T_jet)   # lower-left -> up-right
    elseif incube(0.70, 0.70, x, y, 0.10)
        set_cell!(M0, i, j, k, rho_jet, -U, -U, 0.0, T_jet)   # upper-right -> down-left
    else
        set_cell!(M0, i, j, k, rho_bg, 0.0, 0.0, 0.0, T0)
    end
end

CUDA.device!(0)
@printf("GPU crossing-jets: Nx=Ny=%d Nz=%d Ma=%.1f nstep=%d snap_int=%d\n", Nx, Nz, Ma, nstep, snap_int)
@printf("GPU: %s\n", CUDA.name(CUDA.device()))
t0 = time()
# params as a NamedTuple so the GLMakie viz (which reads params.Nx/.Ma/.Kn) works directly.
# Kn=Inf marks the collisionless GPU transport path.
out = run_gpu_3d(M0, dx, Ma, nstep;
                 snapshot_interval=snap_int, snapshot_filename=outfile,
                 params=(Nx=Nx, Ny=Ny, Nz=Nz, Ma=Ma, Kn=Inf, dx=dx,
                         T_jet=T_jet, rho_jet=rho_jet, source="gpu_crossing_jets"))
@printf("done in %.1f s -> %s\n", time() - t0, out)
println("""
Analyze (MAIN package env; verified):
  julia --project=. -e 'using Riemann35, JLD2
    jf=jldopen("$out","r"); M=jf["snapshots/000001/M"]
    S=Riemann35.compute_standardized_field(M); C=Riemann35.compute_central_field(M); close(jf)'

Interactive GLMakie time-slider (MAIN env, GLMakie installed):
  julia --project=. -e 'using Riemann35, JLD2
    p = jldopen(f->f["meta/params"], "$out", "r")           # NamedTuple (Nx,Ny,Nz,Ma,Kn,dx,…)
    grid = (xm=((1:p.Nx).-0.5).*p.dx, ym=((1:p.Ny).-0.5).*p.dx, zm=((1:p.Nz).-0.5).*p.dx)
    Riemann35.interactive_3d_timeseries_streaming("$out", grid, p)'
""")
