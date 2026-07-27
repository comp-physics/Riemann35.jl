# probe_prefilter.jl — HEADROOM of the redundant realizability prefilter in _hll_states.
#
# _hll_states runs realizable_3D_M4_dev on BOTH input states before anything else. For the
# first-order anchor those inputs are cell means straight out of G, and G's interior was
# already projected to realizability by _proj_interior! earlier in the same step. So that
# prefilter is re-projecting states that are already in the cone.
#
# This does NOT propose removing it -- halo cells are not projected by _proj_interior!, the
# WENO face states genuinely need it, and the projection is not exactly idempotent, so
# removal is a semantic change requiring its own validation. This measures ONLY the headroom:
# if skipping it saves little, the question is closed; if it saves a lot, it is worth the
# validation work of applying it selectively.
using CUDA, Printf
const GPUDIR = get(ENV, "R35_GPUDIR", "/fastscratch/sbryngelson3/Riemann35.jl/gpu")
include(joinpath(GPUDIR, "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU
const R = Residual3DOrder3GPU
using Riemann35
using Riemann35.RiemannFluxDev: riemann_flux_dev
using Riemann35.WavespeedDev: realize_and_speed_Mr_dev
using Riemann35.FluxClosureDev: flux_closure35_dev
const cellG = R._cellG

@inline function _hll_norealize(mL::NTuple{35,Float64}, mR::NTuple{35,Float64},
                          ::Val{A}, Ma::Float64, s3f::Float64) where {A}
    MLf = mL
    MRf = mR

    MLr, lminL, lmaxL = realize_and_speed_Mr_dev(
        MLf[1],  MLf[2],  MLf[3],  MLf[4],  MLf[5],  MLf[6],  MLf[7],
        MLf[8],  MLf[9],  MLf[10], MLf[11], MLf[12], MLf[13], MLf[14],
        MLf[15], MLf[16], MLf[17], MLf[18], MLf[19], MLf[20], MLf[21],
        MLf[22], MLf[23], MLf[24], MLf[25], MLf[26], MLf[27], MLf[28],
        MLf[29], MLf[30], MLf[31], MLf[32], MLf[33], MLf[34], MLf[35], A, Ma)
    MRr, lminR, lmaxR = realize_and_speed_Mr_dev(
        MRf[1],  MRf[2],  MRf[3],  MRf[4],  MRf[5],  MRf[6],  MRf[7],
        MRf[8],  MRf[9],  MRf[10], MRf[11], MRf[12], MRf[13], MRf[14],
        MRf[15], MRf[16], MRf[17], MRf[18], MRf[19], MRf[20], MRf[21],
        MRf[22], MRf[23], MRf[24], MRf[25], MRf[26], MRf[27], MRf[28],
        MRf[29], MRf[30], MRf[31], MRf[32], MRf[33], MRf[34], MRf[35], A, Ma)

    FLall = flux_closure35_dev(
        MLr[1],  MLr[2],  MLr[3],  MLr[4],  MLr[5],  MLr[6],  MLr[7],
        MLr[8],  MLr[9],  MLr[10], MLr[11], MLr[12], MLr[13], MLr[14],
        MLr[15], MLr[16], MLr[17], MLr[18], MLr[19], MLr[20], MLr[21],
        MLr[22], MLr[23], MLr[24], MLr[25], MLr[26], MLr[27], MLr[28],
        MLr[29], MLr[30], MLr[31], MLr[32], MLr[33], MLr[34], MLr[35])
    FRall = flux_closure35_dev(
        MRr[1],  MRr[2],  MRr[3],  MRr[4],  MRr[5],  MRr[6],  MRr[7],
        MRr[8],  MRr[9],  MRr[10], MRr[11], MRr[12], MRr[13], MRr[14],
        MRr[15], MRr[16], MRr[17], MRr[18], MRr[19], MRr[20], MRr[21],
        MRr[22], MRr[23], MRr[24], MRr[25], MRr[26], MRr[27], MRr[28],
        MRr[29], MRr[30], MRr[31], MRr[32], MRr[33], MRr[34], MRr[35])

    off = (A - 1) * 35
    sL = min(lminL, lminR)
    sR = max(lmaxL, lmaxR)
    return riemann_flux_dev(0, A, MLr, MRr,
                            ntuple(j -> FLall[off + j], Val(35)),
                            ntuple(j -> FRall[off + j], Val(35)), sL, sR)
end


function k_base!(F,G,nx,ny,nz,g,Ma,s3f)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f=(idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1; b=g+j; c=g+k; il=g+f-1
            FL = R._hll_states(cellG(G,il,b,c), cellG(G,il+1,b,c), 1, Ma, s3f)
            for m in 1:35; F[m,f,j,k]=FL[m]; end
        end
    end
    nothing
end
function k_nopre!(F,G,nx,ny,nz,g,Ma,s3f)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f=(idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1; b=g+j; c=g+k; il=g+f-1
            FL = _hll_norealize(cellG(G,il,b,c), cellG(G,il+1,b,c), Val(1), Ma, s3f)
            for m in 1:35; F[m,f,j,k]=FL[m]; end
        end
    end
    nothing
end

nx=ny=nz=64; g=8; nfx=nx+2g
Mc = InitializeM4_35(1.0,0.0,0.0,0.0,1.0,0,0,1.0,0,1.0)
M0 = zeros(35,nfx,nfx,nfx)
for k in 1:nfx, j in 1:nfx, i in 1:nfx
    s=1.0+0.05*sin(2pi*i/nfx); for m in 1:35; M0[m,i,j,k]=Mc[m]*s; end
end
G=CuArray(M0); F1=CUDA.zeros(Float64,35,nx+1,ny,nz); F2=CUDA.zeros(Float64,35,nx+1,ny,nz)
nf=(nx+1)*ny*nz; thr=128; bl=cld(nf,thr)
tm(f;n=50)=(f();CUDA.synchronize();e=CUDA.@elapsed for _ in 1:n; f(); end; 1000*e/n)
tb=tm(()->(@cuda threads=thr blocks=bl k_base!(F1,G,nx,ny,nz,g,1.0,40.0)))
tn=tm(()->(@cuda threads=thr blocks=bl k_nopre!(F2,G,nx,ny,nz,g,1.0,40.0)))
fr(k)=CUDA.memory(k).local
kb=@cuda launch=false k_base!(F1,G,nx,ny,nz,g,1.0,40.0)
kn=@cuda launch=false k_nopre!(F2,G,nx,ny,nz,g,1.0,40.0)
println("="^80); println("HEADROOM: the realizable_3D_M4_dev prefilter inside _hll_states"); println("="^80)
@printf("with prefilter (ships)   %7.3f ms   frame=%5d B\n", tb, fr(kb))
@printf("without prefilter        %7.3f ms   frame=%5d B\n", tn, fr(kn))
@printf("\nheadroom: %.2fx  (%.3f ms of %.3f ms = %.0f%% of the HLL)\n", tb/tn, tb-tn, tb, 100*(tb-tn)/tb)
@cuda threads=thr blocks=bl k_base!(F1,G,nx,ny,nz,g,1.0,40.0)
@cuda threads=thr blocks=bl k_nopre!(F2,G,nx,ny,nz,g,1.0,40.0); CUDA.synchronize()
d=maximum(abs.(Array(F1).-Array(F2)))
@printf("max |with - without| on this ALREADY-REALIZABLE field = %.3e\n", d)
println("(nonzero here would mean the prefilter is not idempotent even on realizable input)")
