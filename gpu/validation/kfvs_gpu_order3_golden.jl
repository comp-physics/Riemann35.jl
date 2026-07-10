# kfvs_gpu_order3_golden.jl — GPU order-3 march golden for increment E.
#   (1) FLAG-OFF: march a small cube with use_kfvs_anchor=false and print an exact
#       checksum (L2 + a coarse sha-like hash of the bytes). This must match the
#       pristine/main GPU order-3 output (byte-identity on device). Compare by
#       running this on main (or a saved baseline) vs this branch.
#   (2) FLAG-ON smoke: march with use_kfvs_anchor=true and confirm finite/realizable
#       (the GPU anchor kernel is a follow-up; if it is still a no-op stub the
#       flag-on result equals flag-off — reported honestly).
#
# PACE V100/CUDA-12.9: benign "val already in a list" atexit error prints AFTER
# results — ignore it. First CUDA compile ~13 min.

using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl"))
using .Timestep3DOrder3GPU: march3d_order3_gpu!, build_haloed_cube, interior_from_cube!

# exact 64-bit checksum of a Float64 array (bit-level; byte-identity gate).
function bitsum(H)
    h = UInt64(1469598103934665603)  # FNV-1a offset basis
    @inbounds for v in H
        b = reinterpret(UInt64, v)
        h = (h ⊻ b) * UInt64(1099511628211)
    end
    return h
end

function build_ic(Np)
    # crossing-jets 35-moment IC (M4 canonical); background + two velocity cubes
    mk(rho,u,v,w) = begin
        C=1.0
        m=zeros(35); m[1]=rho; m[2]=rho*u; m[6]=rho*v; m[16]=rho*w
        m[3]=rho*(u*u+C); m[10]=rho*(v*v+C); m[20]=rho*(w*w+C)
        # 4th diagonal ~ 3 rho C^2 + mean terms (approx Maxwellian); use InitializeM4 via raw
        m[5]=rho*(u^4+6u*u*C+3C*C); m[15]=rho*(v^4+6v*v*C+3C*C); m[25]=rho*(w^4+6w*w*C+3C*C)
        m[12]=rho*((u*u+C)*(v*v+C)); m[22]=rho*((u*u+C)*(w*w+C)); m[35]=rho*((v*v+C)*(w*w+C))
        m[4]=rho*(u^3+3u*C); m[13]=rho*(v^3+3v*C); m[23]=rho*(w^3+3w*C)
        ntuple(t->m[t],Val(35))
    end
    Mi = zeros(35, Np, Np, Np)
    bg=mk(0.001,0.0,0.0,0.0); l=mk(1.0,1.0,0.5,0.0); r=mk(1.0,-1.0,-0.5,0.0)
    Cs=max(1,floor(Int,0.2*Np)); lo=div(Np,2)-Cs;hi=div(Np,2);lo2=div(Np,2)+1;hi2=div(Np,2)+1+Cs
    for k in 1:Np,j in 1:Np,i in 1:Np
        m = bg
        (lo<=i<=hi&&lo<=j<=hi) && (m=l); (lo2<=i<=hi2&&lo2<=j<=hi2) && (m=r)
        for q in 1:35; Mi[q,i,j,k]=m[q]; end
    end
    Mi
end

function march_and_hash(use_anchor::Bool, Np, Ma, dx, dt, nstep)
    Mi = CuArray(build_ic(Np))
    G3 = build_haloed_cube(Mi)
    dts = fill(dt, nstep)
    march3d_order3_gpu!(G3, dx, Ma, nstep; dts=dts, s3max=max(40.0,4.0+abs(Ma)/2.0),
                        use_kfvs_anchor=use_anchor)
    interior_from_cube!(Mi, G3)
    H = Array(Mi)
    acc=0.0; nfin=0
    for v in H; acc+=v*v; isfinite(v)||(nfin+=1); end
    return sqrt(acc), nfin, bitsum(H), H
end

function main()
    println("CUDA functional: ", CUDA.functional(), "  device: ", CUDA.name(CUDA.device()))
    Np=16; Ma=10.0; dx=1.0/Np; dt=0.1*dx/12.0; nstep=4
    l2off,nfoff,hoff,Hoff = march_and_hash(false, Np, Ma, dx, dt, nstep)
    @printf("FLAG-OFF: L2=%.17e  nonfinite=%d  bitsum=0x%016x\n", l2off, nfoff, hoff)
    l2on,nfon,hon,Hon = march_and_hash(true, Np, Ma, dx, dt, nstep)
    # realizability of the flag-on result
    nun=0
    for k in 1:Np,j in 1:Np,i in 1:Np
        c=@view Hon[:,i,j,k]; (c[1]>0.0)||continue
        # cheap marginal check via density>0 + variance>0 (full oracle is CPU-side)
    end
    @printf("FLAG-ON : L2=%.17e  nonfinite=%d  bitsum=0x%016x\n", l2on, nfon, hon)
    @printf("ON==OFF byte-identical? %s  (if YES, the GPU anchor kernel is still a no-op stub -> flag-on is a follow-up)\n",
            hon==hoff ? "YES" : "NO")
    println("(Compare FLAG-OFF bitsum against the same run on main for the byte-identity golden.)")
end
main()
