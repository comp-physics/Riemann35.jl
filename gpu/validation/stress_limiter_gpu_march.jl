# stress_limiter_gpu_march.jl — GPU half of the limiter stress test (gpuenv2).
# Reads the IC + dt sequence written by stress_limiter_cpu_march.jl and marches the SAME
# steps on the GPU (limiter=true and default), writing the final density fields. The dt
# sequence is fixed (read from file) so the ONLY CPU/GPU difference is the per-step
# residual+projection numerics — exactly what we want to measure.
#   srun --mpi=pmix -n 1 --gpus=1 $JULIA --project=gpu/gpuenv2 gpu/validation/stress_limiter_gpu_march.jl
using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_gpu.jl")); using .Timestep3DGPU
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
meta = split(strip(read(joinpath(DATA,"stress.meta"),String)),'\n')
n=parse(Int,meta[1]); nstep=parse(Int,meta[2]); Ma=parse(Float64,meta[3]); vacf=parse(Float64,meta[4]); dx=parse(Float64,meta[5])
Mint = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"stress_M0.f64")))),35,n,n,n)
dts  = collect(reinterpret(Float64,read(joinpath(DATA,"stress_dts.f64"))))

function march(Mint, dts, n, dx, Ma, vacf; limiter)
    Md = CuArray(Mint)
    march3d_gpu!(Md, dx, Ma, length(dts); dts=dts, vacuum_floor=vacf, order=2, limiter=limiter)
    H = Array(Md)
    rho = Array{Float64}(undef,n,n,n)
    for k in 1:n, j in 1:n, i in 1:n; rho[i,j,k] = H[1,i,j,k]; end
    return rho
end

@printf("GPU stress march: n=%d nstep=%d Ma=%.0f dt=%.3e vacf=%.3g  [%s]\n", n,nstep,Ma,dts[1],vacf,CUDA.name(CUDA.device()))
rho_lim = march(Mint, dts, n, dx, Ma, vacf; limiter=true)
rho_def = march(Mint, dts, n, dx, Ma, vacf; limiter=false)
@printf("  GPU limiter: rho range [%.4e, %.4e] mass=%.6e\n", minimum(rho_lim), maximum(rho_lim), sum(rho_lim))
@printf("  GPU default: rho range [%.4e, %.4e] mass=%.6e\n", minimum(rho_def), maximum(rho_def), sum(rho_def))
write(joinpath(DATA,"stress_gpu_lim.f64"), reinterpret(UInt8, vec(rho_lim)))
write(joinpath(DATA,"stress_gpu_def.f64"), reinterpret(UInt8, vec(rho_def)))
println("wrote stress_{gpu_lim,gpu_def}")
