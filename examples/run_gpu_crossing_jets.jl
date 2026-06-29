#!/usr/bin/env julia
# run_gpu_crossing_jets.jl — end-to-end GPU example: build a crossing/colliding-jet IC,
# advance it on the GPU (single- or multi-GPU), and stream snapshots to JLD2 in the
# canonical schema so the existing readers / visualization consume the output unchanged.
#
# Runs under the GPU project (gpuenv2). The IC is built self-contained here (a diagonal-
# covariance Gaussian per cell — same state InitializeM4_35 produces — so the example
# needs no package internals). Multi-GPU decomposes along z (one slab per rank).
#
# Run (single GPU):
#   srun --mpi=pmix -n 1 --gpus=1 julia --project=gpu/gpuenv2 examples/run_gpu_crossing_jets.jl
# Run (both GPUs, Ma=100 colliding jets, with a vacuum floor for high-Ma robustness):
#   MA=100 NZ=64 NX=64 NSTEP=100 SNAP=20 VACF=0.01 \
#     srun --mpi=pmix -n 2 --gpus=2 julia --project=gpu/gpuenv2 examples/run_gpu_crossing_jets.jl
# Then analyze/visualize in the MAIN package env (commands printed at the end).
#
# NOTE on realizability: the GPU residual uses MUSCL + per-face realizable_3D_M4
# projection + the `ho_vacuum_floor` (VACF) + recon-validity fallback (the CPU DEFAULT
# path). It does NOT yet implement `ho_proj_first_order` / the Zhang–Shu scaling limiter
# — the knobs HIGHORDER.md identifies for Ma=100 robustness — so at very high Ma raise VACF.
using CUDA, Printf, MPI
include(joinpath(@__DIR__, "..", "gpu", "gpu_run.jl")); using .GPURun

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)
CUDA.device!(RANK % CUDA.ndevices())

# ---- canonical 35-moment exponent ordering (M_n = <vx^i vy^j vz^k f>) ----
const TRIPLES = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),
 (1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),
 (1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),
 (0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]

# raw moment of a 1D Gaussian(mean=mu, var=s2), order m (0..4)
@inline _g(m, mu, s2) = m == 0 ? 1.0 : m == 1 ? mu : m == 2 ? mu^2 + s2 :
                        m == 3 ? mu^3 + 3mu*s2 : mu^4 + 6mu^2*s2 + 3s2^2
# 35-moment diagonal-covariance Gaussian cell -> writes into M[:,i,j,k]
@inline function set_cell!(M, i, j, k, rho, u, v, w, T)
    @inbounds for n in 1:35
        a, b, c = TRIPLES[n]
        M[n, i, j, k] = rho * _g(a, u, T) * _g(b, v, T) * _g(c, w, T)
    end
end

# ---- parameters (override via ENV) ----
Nx = parse(Int, get(ENV, "NX", "48")); Ny = Nx
Nz = parse(Int, get(ENV, "NZ", "16"))
Ma = parse(Float64, get(ENV, "MA", "5.0"))
nstep = parse(Int, get(ENV, "NSTEP", "20"))
snap_int = parse(Int, get(ENV, "SNAP", "5"))
vacf = parse(Float64, get(ENV, "VACF", "0.001"))   # ho_vacuum_floor; raise for high-Ma
T0 = 1.0
dx = 1.0 / Nx
@assert Nz % NRANKS == 0 "Nz ($Nz) must be divisible by nranks ($NRANKS)"
outfile = get(ENV, "OUT", joinpath(pwd(), @sprintf("crossing_jets_gpu_Nx%d_Nz%d_Ma%.1f.jld2", Nx, Nz, Ma)))

# ---- IC: faithful port of Rodney's main_crossing_3DHyQMOM35.m ----
#  Domain [-0.5,0.5]^3 (length 1), cubic grid Np=Nx=Ny=Nz, variance T=1 (C200=C020=C002=1,
#  no correlation), 1000:1 density ratio. Two dense cubes ADJACENT at the box center, each
#  Csize=floor(0.1*Np) cells wide, moving toward each other on the main 3D diagonal with
#  per-axis speed Uc=Ma/sqrt(3) (|v|=Ma). set_cell! with T=1 == InitializeM4_35(rho,u,v,w,1,0,0,1,0,1).
@assert Nx == Ny == Nz "Rodney's 3D crossing wants a cubic grid (Nx=Ny=Nz); got $Nx,$Ny,$Nz"
Np = Nx
M0 = Array{Float64}(undef, 35, Np, Np, Np)
rho_jet, rho_bg, T_jet = 1.0, 1.0e-3, T0     # rhol=1, rhor=0.001 (1000:1); variance T=1
Uc = Ma / sqrt(3.0)                          # |v| = Ma
Csize = floor(Int, 0.1 * Np)
half = div(Np, 2)
# background everywhere (rhor, at rest)
for k in 1:Np, j in 1:Np, i in 1:Np
    set_cell!(M0, i, j, k, rho_bg, 0.0, 0.0, 0.0, T0)
end
# "bottom" cube [half-Csize : half], v = +Uc (toward center); "top" cube [half+1 : half+1+Csize], v = -Uc
for k in (half-Csize):half, j in (half-Csize):half, i in (half-Csize):half
    set_cell!(M0, i, j, k, rho_jet,  Uc,  Uc,  Uc, T_jet)
end
for k in (half+1):(half+1+Csize), j in (half+1):(half+1+Csize), i in (half+1):(half+1+Csize)
    set_cell!(M0, i, j, k, rho_jet, -Uc, -Uc, -Uc, T_jet)
end

# ---- decompose along z for multi-GPU (each rank owns a z-slab interior); single-GPU = whole field ----
nzloc = div(Nz, NRANKS); z0 = RANK * nzloc
Mloc = NRANKS > 1 ? Array(@view M0[:, :, :, z0+1:z0+nzloc]) : M0
comm_arg = NRANKS > 1 ? COMM : nothing

if RANK == 0
    @printf("GPU crossing/colliding jets: Nx=Ny=%d Nz=%d Ma=%.1f nstep=%d snap_int=%d vacf=%.3g  [%d GPU(s)]\n",
            Nx, Nz, Ma, nstep, snap_int, vacf, NRANKS)
    @printf("GPU(rank0): %s\n", CUDA.name(CUDA.device()))
end

t0 = time()
# params as a NamedTuple so the GLMakie viz (params.Nx/.Ma/.Kn) works directly; Kn=Inf = collisionless.
out = run_gpu_3d(Mloc, dx, Ma, nstep;
                 comm=comm_arg, vacuum_floor=vacf,
                 snapshot_interval=snap_int, snapshot_filename=outfile,
                 params=(Nx=Nx, Ny=Ny, Nz=Nz, Ma=Ma, Kn=1000.0, dx=dx,
                         T_jet=T_jet, rho_jet=rho_jet, source="gpu_crossing_jets"))

if RANK == 0
    @printf("done in %.1f s -> %s\n", time() - t0, out)
    println("""
    Analyze (MAIN package env):
      julia --project=. -e 'using Riemann35, JLD2
        jf=jldopen("$outfile","r"); M=jf["snapshots/000001/M"]
        S=Riemann35.compute_standardized_field(M); C=Riemann35.compute_central_field(M); close(jf)'

    Interactive GLMakie time-slider (MAIN env, GLMakie installed):
      julia --project=. -e 'using Riemann35, JLD2
        p = jldopen(f->f["meta/params"], "$outfile", "r")
        grid = (xm=((1:p.Nx).-0.5).*p.dx, ym=((1:p.Ny).-0.5).*p.dx, zm=((1:p.Nz).-0.5).*p.dx)
        Riemann35.interactive_3d_timeseries_streaming("$outfile", grid, p)'
    """)
end
MPI.Barrier(COMM)
MPI.Finalize()
