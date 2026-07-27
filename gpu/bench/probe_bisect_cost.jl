# probe_bisect_cost.jl — how much does the Zhang-Shu bisection ACTUALLY cost, and how
# often does it even run?
#
# The previous probe measured (B) weno5z+realizability minus (B') weno5z alone = 3.93 ms,
# 32% of the fused flux kernel, and it is tempting to call that "the bisection". It is NOT:
# (B') writes recon variables straight out and never calls from_recon_vars_tup, while (B)
# must convert recon -> raw to return a face state. That conversion is MANDATORY. So the
# 3.93 ms is bisection + conversion, and attributing it all to the bisection would be the
# same class of error as comparing two codes at matched labels instead of matched lambda.
#
# Three variants, each adding exactly one thing:
#   (1) weno5z only                              -- no conversion, no realizability
#   (2) weno5z + from_recon_vars_tup             -- conversion, still no realizability
#   (3) weno5z + full weno_scaled_face_dev       -- what ships
# (3)-(2) is the bisection's true cost. (2)-(1) is the conversion, which is not removable.
#
# Plus the thing that decides whether any of this is worth attacking: the EARLY-OUT RATE.
# weno_scaled_face_dev returns immediately when the unlimited face is already realizable.
# On smooth data that should be ~100% of faces, in which case the 20-iteration loop is
# nearly free and the cost is all in the check. If the rate is low, the loop is really
# running and a closed-form theta (as already done for theta*-IDP via theta_star_update_closed)
# is worth deriving.
using CUDA, Printf, Statistics
const GPUDIR = get(ENV, "R35_GPUDIR", "/fastscratch/sbryngelson3/Riemann35.jl/gpu")
include(joinpath(GPUDIR, "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU
const R = Residual3DOrder3GPU
using Riemann35
using Riemann35.HiOrder3ReconDev: weno_faces_dev, recon_vars_realizable
using Riemann35.Weno5Dev: weno5z
using Riemann35.ReconDev: from_recon_vars_tup, to_recon_vars_tup
using Riemann35.RiemannFluxDev: _state_realizable
const cellG = R._cellG; const clmp = R._clamp

macro stencil()
    esc(quote
        f=(idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1; b=g+j; c=g+k; il=g+f-1
        W1=cellG(V,clmp(il-2,nfx),b,c); W2=cellG(V,clmp(il-1,nfx),b,c)
        W3=cellG(V,clmp(il,  nfx),b,c); W4=cellG(V,clmp(il+1,nfx),b,c)
        W5=cellG(V,clmp(il+2,nfx),b,c); W6=cellG(V,clmp(il+3,nfx),b,c)
        vL = ntuple(q -> weno5z(W1[q],W2[q],W3[q],W4[q],W5[q]), Val(35))
        vR = ntuple(q -> weno5z(W6[q],W5[q],W4[q],W3[q],W2[q]), Val(35))
    end)
end

# (1) weno5z only
function k1!(ML,MR,V,nx,ny,nz,g,nfx)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            @stencil
            for m in 1:35; ML[m,f,j,k]=vL[m]; MR[m,f,j,k]=vR[m]; end
        end
    end
    nothing
end
# (2) weno5z + the mandatory recon -> raw conversion
function k2!(ML,MR,V,nx,ny,nz,g,nfx)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            @stencil
            mL = from_recon_vars_tup(vL); mR = from_recon_vars_tup(vR)
            for m in 1:35; ML[m,f,j,k]=mL[m]; MR[m,f,j,k]=mR[m]; end
        end
    end
    nothing
end
# (3) what ships
function k3!(ML,MR,G,V,nx,ny,nz,g,nfx)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f=(idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1; b=g+j; c=g+k; il=g+f-1
            cL=cellG(G,il,b,c); cR=cellG(G,il+1,b,c)
            W1=cellG(V,clmp(il-2,nfx),b,c); W2=cellG(V,clmp(il-1,nfx),b,c)
            W3=cellG(V,clmp(il,  nfx),b,c); W4=cellG(V,clmp(il+1,nfx),b,c)
            W5=cellG(V,clmp(il+2,nfx),b,c); W6=cellG(V,clmp(il+3,nfx),b,c)
            mL,mR = weno_faces_dev(W1,W2,W3,W4,W5,W6,cL,cR)
            for m in 1:35; ML[m,f,j,k]=mL[m]; MR[m,f,j,k]=mR[m]; end
        end
    end
    nothing
end
# early-out census: 1.0 if BOTH faces took the fast path, else 0.0
function kcensus!(OUT,G,V,nx,ny,nz,g,nfx)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            @stencil
            okL = recon_vars_realizable(vL) && _state_realizable(from_recon_vars_tup(vL))
            okR = recon_vars_realizable(vR) && _state_realizable(from_recon_vars_tup(vR))
            OUT[idx] = (okL && okR) ? 1.0 : 0.0
        end
    end
    nothing
end

nx=ny=nz=64; g=8; nfx=nx+2g
Mc = InitializeM4_35(1.0,0.0,0.0,0.0,1.0,0,0,1.0,0,1.0)
M0 = zeros(35,nfx,nfx,nfx)
for k in 1:nfx, j in 1:nfx, i in 1:nfx
    s = 1.0 + 0.05*sin(2pi*i/nfx)
    for m in 1:35; M0[m,i,j,k] = Mc[m]*s; end
end
G=CuArray(M0)
ML=CUDA.zeros(Float64,35,nx+1,ny,nz); MR=CUDA.zeros(Float64,35,nx+1,ny,nz)
nf=(nx+1)*ny*nz; thr=128; bl=cld(nf,thr)
OUT=CUDA.zeros(Float64,nf)

# V MUST BE RECON VARIABLES, NOT RAW MOMENTS. weno5z reconstructs in recon-variable space
# and from_recon_vars_tup takes sqrt of slots 5,6,7 (the variances) on the way back. A first
# version of this probe set V = copy(M0), i.e. fed RAW MOMENTS in where recon variables were
# expected; every face then failed _state_realizable and the early-out rate read 0.00%, which
# looked like a finding ("the bisection always runs") but was just the wrong input. Build V
# the way the residual actually does: _ppt_x! (deconv5 -> recon point values) then _vavg_x!
# (conv5 -> recon cell averages).
P=CUDA.zeros(Float64,35,nfx,nfx,nfx); V=CUDA.zeros(Float64,35,nfx,nfx,nfx)
bcube=cld(nfx*nfx*nfx,thr)
@cuda threads=thr blocks=bcube R._ppt_x!(P,G,nfx,nfx,nfx)
@cuda threads=thr blocks=bcube R._vavg_x!(V,P,nfx,nfx,nfx)
CUDA.synchronize()

tm(f;n=50)=(f();CUDA.synchronize();e=CUDA.@elapsed for _ in 1:n; f(); end; 1000*e/n)
t1=tm(()->(@cuda threads=thr blocks=bl k1!(ML,MR,V,nx,ny,nz,g,nfx)))
t2=tm(()->(@cuda threads=thr blocks=bl k2!(ML,MR,V,nx,ny,nz,g,nfx)))
t3=tm(()->(@cuda threads=thr blocks=bl k3!(ML,MR,G,V,nx,ny,nz,g,nfx)))
@cuda threads=thr blocks=bl kcensus!(OUT,G,V,nx,ny,nz,g,nfx); CUDA.synchronize()
rate = mean(Array(OUT))

println("="^80); println("TRUE COST OF THE ZHANG-SHU REALIZABILITY BISECTION"); println("="^80)
@printf("(1) weno5z only                        %8.3f ms\n", t1)
@printf("(2) + from_recon_vars_tup (MANDATORY)  %8.3f ms   (+%.3f ms conversion)\n", t2, t2-t1)
@printf("(3) + full weno_scaled_face_dev        %8.3f ms   (+%.3f ms realizability)\n", t3, t3-t2)
println("-"^80)
@printf("bisection+check true cost = %.3f ms; naive (3)-(1) would have claimed %.3f ms\n", t3-t2, t3-t1)
@printf("early-out rate on this smooth field: %.2f%% of faces take the fast path\n", 100*rate)
println("="^80)
