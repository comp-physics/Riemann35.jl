# kfvs_f3_gpu_timing.jl — GPU throughput: F3 anchor march vs the default (projection)
# march. Quantifies the cost of the kinetic-FVS anchor (which inverts each cell ~2x per
# face in the weno flux kernels). Reports ms/step and Mcell/s for each and the ratio, so
# we can decide whether a store-once node pass (invert each cell once, not per face) is
# warranted for "very fast".  Run under gpuenv2.  HYQMOM_TIME_NP (default 32), _STEPS (20).
using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl"))
using .Timestep3DOrder3GPU: march3d_order3_gpu!, build_haloed_cube, interior_from_cube!

function build_ic(Np)
    mk(rho,u,v,w) = begin
        C=1.0; m=zeros(35); m[1]=rho; m[2]=rho*u; m[6]=rho*v; m[16]=rho*w
        m[3]=rho*(u*u+C); m[10]=rho*(v*v+C); m[20]=rho*(w*w+C)
        m[5]=rho*(u^4+6u*u*C+3C*C); m[15]=rho*(v^4+6v*v*C+3C*C); m[25]=rho*(w^4+6w*w*C+3C*C)
        m[12]=rho*((u*u+C)*(v*v+C)); m[22]=rho*((u*u+C)*(w*w+C)); m[35]=rho*((v*v+C)*(w*w+C))
        m[4]=rho*(u^3+3u*C); m[13]=rho*(v^3+3v*C); m[23]=rho*(w^3+3w*C)
        ntuple(t->m[t],Val(35))
    end
    Mi = zeros(35, Np, Np, Np)
    bg=mk(0.05,0.0,0.0,0.0); l=mk(1.0,1.5,0.75,0.0); r=mk(1.0,-1.5,-0.75,0.0)
    Cs=max(1,floor(Int,0.2*Np)); lo=div(Np,2)-Cs;hi=div(Np,2);lo2=div(Np,2)+1;hi2=div(Np,2)+1+Cs
    for k in 1:Np,j in 1:Np,i in 1:Np
        m=bg; (lo<=i<=hi&&lo<=j<=hi)&&(m=l); (lo2<=i<=hi2&&lo2<=j<=hi2)&&(m=r)
        for q in 1:35; Mi[q,i,j,k]=m[q]; end
    end
    Mi
end

function timeit(use_anchor, Np, Ma, dx, dt, nstep)
    G = build_haloed_cube(CuArray(build_ic(Np)))
    dts = fill(dt, nstep)
    # warmup (compile + 2 steps)
    march3d_order3_gpu!(G, dx, Ma, 2; dts=fill(dt,2), s3max=max(40.0,4.0+abs(Ma)/2.0), use_kfvs_anchor=use_anchor)
    CUDA.synchronize()
    G = build_haloed_cube(CuArray(build_ic(Np)))
    t0 = time()
    march3d_order3_gpu!(G, dx, Ma, nstep; dts=dts, s3max=max(40.0,4.0+abs(Ma)/2.0), use_kfvs_anchor=use_anchor)
    CUDA.synchronize()
    return time() - t0
end

function main()
    Np = parse(Int, get(ENV,"HYQMOM_TIME_NP","32")); nstep = parse(Int, get(ENV,"HYQMOM_TIME_STEPS","20"))
    Ma=10.0; dx=1.0/Np; dt=0.1*dx/12.0
    println("CUDA: ", CUDA.name(CUDA.device()), "  Np=$Np  nstep=$nstep")
    toff = timeit(false, Np, Ma, dx, dt, nstep)
    ton  = timeit(true,  Np, Ma, dx, dt, nstep)
    cells = Np^3
    @printf("default (projection): %.1f ms/step   %.2f Mcell/s\n", 1e3*toff/nstep, cells*nstep/toff/1e6)
    @printf("F3 (kinetic anchor) : %.1f ms/step   %.2f Mcell/s\n", 1e3*ton/nstep,  cells*nstep/ton/1e6)
    @printf("F3 / default slowdown: %.2fx\n", ton/toff)
    println("(>~3x slowdown ⇒ store-once node pass warranted; ~1-2x ⇒ inversion not dominant.)")
end
main()
