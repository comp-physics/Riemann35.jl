# profile_kernels.jl — manual per-op timing of the order-3 march (CUDA.@elapsed,
# no CUPTI). Answers where the per-step time goes and how much the θ*-IDP/WENO5
# machinery costs vs a plain first-order HLL residual.
#   julia --project=gpu/gpuenv2 gpu/profile_kernels.jl <stage_dir>
using CUDA, Printf
include(joinpath(@__DIR__, "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
const T = Timestep3DOrder3GPU; const R = T.Residual3DOrder3GPU
include(joinpath(@__DIR__, "staging_common.jl"))

dir = ARGS[1]
m = read_stage_meta(joinpath(dir, "meta.txt"))
nx = parse(Int, m["nx"]); ny = parse(Int, m["ny"]); nz = parse(Int, m["nz"])
dx = parse(Float64, m["dx"]); Ma = parse(Float64, m["Ma"]); Kn = parse(Float64, m["Kn"])
s3 = haskey(m,"s3max") ? parse(Float64,m["s3max"]) : max(40.0,4.0+abs(Ma)/2.0)
M0 = reshape(collect(reinterpret(Float64, read(joinpath(dir,"M0.f64")))), 35, nx, ny, nz)
inlet = CuArray(M0[:,1,1,1])
G = build_haloed_cube(CuArray(M0)); g=8
nfx,nfy,nfz = size(G,2),size(G,3),size(G,4)
Rb=CUDA.zeros(Float64,35,nx,ny,nz); G0=CUDA.zeros(Float64,35,nx,ny,nz); svec=CUDA.zeros(Float64,nx*ny*nz)
thr=128; bc=cld(nfx*nfy*nfz,thr); bi=cld(nx*ny*nz,thr); dt=7.3e-4
@printf("grid %dx%dx%d  device=%s\n", nx,ny,nz, CUDA.name(CUDA.device()))

# average ms/call over reps (warm first)
function tm(f; n=100); f(); CUDA.synchronize(); e=CUDA.@elapsed for _ in 1:n; f(); end; 1000*e/n; end

t_full   = tm(()->march3d_order3_gpu!(G,dx,Ma,1;s3max=s3,stage_bgk=true,Kn=Kn,bc=:crossflow,inlet=Array(inlet)); n=30)
t_resid3 = tm(()->R.residual3d_order3_box_gpu!(Rb,G,nx,ny,nz,g,dx,dx,dx,Ma,dt;s3max=s3,threads=thr,theta_closed=true,first_order=false))
t_resid1 = tm(()->R.residual3d_order3_box_gpu!(Rb,G,nx,ny,nz,g,dx,dx,dx,Ma,dt;s3max=s3,threads=thr,theta_closed=true,first_order=true))
t_refill = tm(()->(@cuda threads=thr blocks=bc T._refill_halo_crossflow!(G,nfx,nfy,nfz,g,nx,ny,nz,inlet)))
t_proj   = tm(()->(@cuda threads=thr blocks=bi T._proj_interior!(G,nx,ny,nz,g,Ma,s3)))
t_bgk    = tm(()->(@cuda threads=thr blocks=bi T._bgk_interior!(G,nx,ny,nz,g,dt,Kn)))
t_rk     = tm(()->(@cuda threads=thr blocks=bi T._rk_combine!(G,G0,Rb,nx,ny,nz,g,0.5,0.5,dt)))
t_speed  = tm(()->(@cuda threads=thr blocks=bi T._speed_interior!(svec,G,nx,ny,nz,g)))

@printf("\n=== per-CALL ms ===\n")
@printf("residual (order-3, θ*+WENO5) : %7.3f\n", t_resid3)
@printf("residual (first_order HLL)   : %7.3f   => θ*+WENO5 overhead = %.3f ms (%.0f%% of resid)\n",
        t_resid1, t_resid3-t_resid1, 100*(t_resid3-t_resid1)/t_resid3)
@printf("refill (crossflow halo)      : %7.3f\n", t_refill)
@printf("proj (realizability)         : %7.3f\n", t_proj)
@printf("bgk                          : %7.3f\n", t_bgk)
@printf("rk_combine                   : %7.3f\n", t_rk)
@printf("speed (dt)                   : %7.3f\n", t_speed)
# per STEP: 3 stages x (refill+resid+rk+proj+bgk) + 1 speed
perstep = 3*(t_refill+t_resid3+t_rk+t_proj+t_bgk) + t_speed
@printf("\nreconstructed per-step (3 stages): %.3f ms   | measured full step: %.3f ms\n", perstep, t_full)
@printf("residual is %.0f%% of the reconstructed step; θ*+WENO5 is %.0f%%\n",
        100*3*t_resid3/perstep, 100*3*(t_resid3-t_resid1)/perstep)
