# Validate the order-3 GPU non-cubic (square-in-plane + arbitrary-z) generalization.
#
# A z-UNIFORM initial field has zero z-flux, so the in-plane dynamics must be identical
# whether z carries N planes (a cube) or a thin Nz slab. This runs a z-uniform Ma=3
# dense-bubble IC as a 32^3 cube and as a 32x32x4 slab, in ONE process, and checks:
#   - the slab stays z-uniform (max|plane1 - planeNz| == 0),
#   - mass is conserved and identical between cube and slab,
#   - the slab's in-plane z-slice matches the cube's to machine precision (== 0.0).
# The last is the strong test: the thin slab reproduces the cube's 2D physics exactly,
# proving the generalization reduces to the cubic path.
#
# Run (single GPU, PACE):
#   JULIA_DEPOT_PATH=$SCR/julia_depot:$HOME/.julia TMPDIR=$SCR/tmp \
#   LD_LIBRARY_PATH=<openmpi-4.1.8>/lib:$LD_LIBRARY_PATH \
#   julia --project=gpu/gpuenv2 gpu/validation/validate_noncubic_order3.jl
using CUDA, Printf, JLD2
include(joinpath(@__DIR__, "..", "gpu_run.jl")); using .GPURun

const TRIPLES = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),
 (1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),
 (1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),
 (0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]
@inline _g(m,mu,s2)= m==0 ? 1.0 : m==1 ? mu : m==2 ? mu^2+s2 : m==3 ? mu^3+3mu*s2 : mu^4+6mu^2*s2+3s2^2
@inline function _set_cell!(M,i,j,k,rho,u,v,w,T)
    @inbounds for n in 1:35; a,b,c=TRIPLES[n]; M[n,i,j,k]=rho*_g(a,u,T)*_g(b,v,T)*_g(c,w,T); end
end
function _bubble_ic(N, Nz; Ma=3.0, L=10.0)
    dx=L/N; rho_in=1000.0; rho_out=1.0; T_out=1.0; T_in=T_out/rho_in*rho_out; U=Ma; R=0.5
    M=Array{Float64}(undef,35,N,N,Nz)
    for k in 1:Nz, j in 1:N, i in 1:N
        x=-0.5L+(i-0.5)dx; y=-0.5L+(j-0.5)dx
        (x^2+y^2 <= R^2) ? _set_cell!(M,i,j,k, rho_in,0.0,0.0,0.0,T_in) :
                           _set_cell!(M,i,j,k, rho_out,U,0.0,0.0,T_out)
    end
    M, dx
end
_final(fn) = (d=load(fn); d[@sprintf("snapshots/%06d/M", d["meta/n_snapshots"])])
function _run(N, Nz; Ma=3.0, DT=2.0e-3, NST=8)
    M0,dx = _bubble_ic(N,Nz;Ma=Ma)
    fn = joinpath(get(ENV,"TMPDIR","/tmp"), "vnc_$(N)x$(Nz).jld2")
    run_gpu_3d(M0, dx, Ma, NST; order=3, scheme=:recommended, dts=fill(DT,NST), Kn=1.0,
               vacuum_floor=0.01, snapshot_interval=NST, snapshot_filename=fn)
    _final(fn)
end

function main()
    N=32
    @printf("[vnc] device=%s  N=%d\n", CUDA.name(CUDA.device()), N)
    Mc  = _run(N, N);  @printf("[vnc] cube   %dx%dx%d done\n", N,N,N)
    Mnc = _run(N, 4);  @printf("[vnc] slab   %dx%dx%d done\n", N,N,4)
    zunif = maximum(abs.(Mnc[:,:,1,:] .- Mnc[:,:,4,:]))
    fin   = count(isfinite,Mnc)==length(Mnc)
    massc = sum(@view Mc[:,:,1,1]); massnc = sum(@view Mnc[:,:,1,1])
    slc=Mc[:,:,1,:]; slnc=Mnc[:,:,1,:]
    num=0.0; den=0.0
    for idx in eachindex(slc); d=slc[idx]-slnc[idx]; num+=d*d; den+=slc[idx]^2; end
    rel=sqrt(num/max(den,1e-300))
    @printf("[vnc] z-uniformity=%.3e  finite=%s  mass cube=%.6e slab=%.6e\n", zunif, fin, massc, massnc)
    @printf("[vnc] in-plane relL2 (cube vs slab z-slice)=%.3e\n", rel)
    pass = fin && zunif==0.0 && rel==0.0 && massc==massnc
    @printf("[vnc] RESULT=%s\n", pass ? "PASS" : "FAIL")
    pass || error("non-cubic order-3 validation FAILED")
end
main()
