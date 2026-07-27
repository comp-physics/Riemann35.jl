# probe_weno_time.jl — DOES THE SPLIT ACTUALLY HELP? Measure it, don't infer it.
#
# The parked #15 proposal was: "split the two _hll_states per face into separate kernels".
# The frame data already argues against it (every piece is independently at 255 regs and
# 12.5% occupancy, and the split frames SUM to more than the fused one), but frame size is
# a compile-time proxy. This times the real thing on a realistic grid:
#
#   FUSED   : _weno_flux_x!  (what ships today)
#   SPLIT   : k_lo!  +  k_faces!  +  k_hll_faces!   (the proposal, end to end)
#
# The split also has to write mL,mR to global and read them back -- 2 * 35 * 8 B per face
# each way -- which the fused version keeps in registers/frame. That traffic is the price
# of the proposal and it is included here by construction.
#
# Grid is sized to a real production case, not the 16^3 used for the frame probe, so the
# numbers are throughput-representative rather than launch-overhead-dominated.
using CUDA, Printf, Statistics
const GPUDIR = get(ENV, "R35_GPUDIR", "/fastscratch/sbryngelson3/Riemann35.jl/gpu")
include(joinpath(GPUDIR, "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU
const R = Residual3DOrder3GPU
using Riemann35
using Riemann35.HiOrder3ReconDev: weno_faces_dev
using Riemann35.Weno5Dev: weno5z
const cellG = R._cellG; const hll = R._hll_states; const clmp = R._clamp

function k_lo!(FLO, G, nx, ny, nz, g, Ma, s3f)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f=(idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1; b=g+j; c=g+k; il=g+f-1
            FL = hll(cellG(G,il,b,c), cellG(G,il+1,b,c), 1, Ma, s3f)
            for m in 1:35; FLO[m,f,j,k]=FL[m]; end
        end
    end
    nothing
end
function k_faces!(ML, MR, G, V, nx, ny, nz, g, nfx)
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
function k_hll_faces!(FHO, ML, MR, nx, ny, nz, Ma, s3f)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f=(idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1
            mL = ntuple(q -> ML[q,f,j,k], Val(35)); mR = ntuple(q -> MR[q,f,j,k], Val(35))
            FH = hll(mL, mR, 1, Ma, s3f)
            for m in 1:35; FHO[m,f,j,k]=FH[m]; end
        end
    end
    nothing
end


# (B') the SAME 6-point WENO5-Z reconstruction with the realizability bisection removed.
# weno_scaled_face_dev runs up to 20 bisection iterations, each one a 35-component blend
# plus a full recon->raw conversion and a realizability test. On smooth data it early-outs,
# but a warp pays for any lane that does not. (B) minus (B') is what that machinery costs.
function k_faces_raw!(ML, MR, G, V, nx, ny, nz, g, nfx)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f=(idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1; b=g+j; c=g+k; il=g+f-1
            W1=cellG(V,clmp(il-2,nfx),b,c); W2=cellG(V,clmp(il-1,nfx),b,c)
            W3=cellG(V,clmp(il,  nfx),b,c); W4=cellG(V,clmp(il+1,nfx),b,c)
            W5=cellG(V,clmp(il+2,nfx),b,c); W6=cellG(V,clmp(il+3,nfx),b,c)
            vL = ntuple(q -> weno5z(W1[q],W2[q],W3[q],W4[q],W5[q]), Val(35))
            vR = ntuple(q -> weno5z(W6[q],W5[q],W4[q],W3[q],W2[q]), Val(35))
            for m in 1:35; ML[m,f,j,k]=vL[m]; MR[m,f,j,k]=vR[m]; end
        end
    end
    nothing
end

nx=ny=nz=64; g=8; nfx=nx+2g
# a smooth, realizable field: unit density Maxwellian with a mild wave, so the
# realizability bisection early-outs the way it does in a real smooth run
M0 = zeros(35,nfx,nfx,nfx)
Mc = InitializeM4_35(1.0,0.0,0.0,0.0,1.0,0,0,1.0,0,1.0)
for k in 1:nfx, j in 1:nfx, i in 1:nfx
    s = 1.0 + 0.05*sin(2pi*i/nfx)
    for m in 1:35; M0[m,i,j,k] = Mc[m]*s; end
end
G = CuArray(M0)
# V MUST BE RECON VARIABLES (output of _ppt -> _vavg), NOT raw moments. Setting V = copy(M0)
# feeds raw moments where recon variables are expected; every face then fails
# _state_realizable, the Zhang-Shu bisection runs all 20 iterations on every face instead of
# early-outing, and the kernel times come out ~1.7x too high with the excess misattributed to
# realizability. Build V the way the residual actually does.
Vp = CUDA.zeros(Float64,35,nfx,nfx,nfx); V = CUDA.zeros(Float64,35,nfx,nfx,nfx)
let thr0=128, bcube=cld(nfx*nfx*nfx,128)
    @cuda threads=thr0 blocks=bcube R._ppt_x!(Vp,CuArray(M0),nfx,nfx,nfx)
    @cuda threads=thr0 blocks=bcube R._vavg_x!(V,Vp,nfx,nfx,nfx)
    CUDA.synchronize()
end
FHO=CUDA.zeros(Float64,35,nx+1,ny,nz); FLO=CUDA.zeros(Float64,35,nx+1,ny,nz)
ML =CUDA.zeros(Float64,35,nx+1,ny,nz); MR =CUDA.zeros(Float64,35,nx+1,ny,nz)
nf=(nx+1)*ny*nz; thr=128; bl=cld(nf,thr)

tm(f;n=50)=(f();CUDA.synchronize();e=CUDA.@elapsed for _ in 1:n; f(); end; 1000*e/n)

t_fused = tm(()->(@cuda threads=thr blocks=bl R._weno_flux_x!(FHO,FLO,G,V,nx,ny,nz,g,nfx,1.0,40.0,false)))
t_lo    = tm(()->(@cuda threads=thr blocks=bl k_lo!(FLO,G,nx,ny,nz,g,1.0,40.0)))
t_fc    = tm(()->(@cuda threads=thr blocks=bl k_faces!(ML,MR,G,V,nx,ny,nz,g,nfx)))
t_hf    = tm(()->(@cuda threads=thr blocks=bl k_hll_faces!(FHO,ML,MR,nx,ny,nz,1.0,40.0)))
t_fcraw = tm(()->(@cuda threads=thr blocks=bl k_faces_raw!(ML,MR,G,V,nx,ny,nz,g,nfx)))

println("="^84)
@printf("SPLIT vs FUSED, %d^3 grid, %d faces, A100 %s\n", nx, nf, CUDA.name(CUDA.device()))
println("="^84)
@printf("FUSED  _weno_flux_x!                  %8.3f ms\n", t_fused)
println("-"^84)
@printf("SPLIT  (A) k_lo!        HLL on means  %8.3f ms\n", t_lo)
@printf("       (B) k_faces!     WENO -> mL,mR %8.3f ms\n", t_fc)
@printf("       (C) k_hll_faces! HLL on faces  %8.3f ms\n", t_hf)
@printf("       TOTAL                          %8.3f ms   = %.2fx the fused kernel\n",
        t_lo+t_fc+t_hf, (t_lo+t_fc+t_hf)/t_fused)
println("="^84)
# the extra memory traffic the split pays and the fused version does not
extra = 2*35*8*nf*2   # mL,mR written once and read once
@printf("extra global traffic forced by the split: %.2f GB per axis sweep\n", extra/1e9)
println()
println("WHERE THE WENO PIECE ACTUALLY GOES:")
@printf("  (B ) weno5z + realizability bisection  %8.3f ms\n", t_fc)
@printf("  (B') weno5z alone, no bisection        %8.3f ms\n", t_fcraw)
@printf("  => the Zhang-Shu realizability scaling costs %.3f ms = %.0f%% of (B), %.0f%% of the fused kernel\n",
        t_fc-t_fcraw, 100*(t_fc-t_fcraw)/t_fc, 100*(t_fc-t_fcraw)/t_fused)
println()
# Is this kernel bandwidth-bound or compute-bound? Useful traffic only.
useful = (8*35*8 + 2*35*8)*nf     # 8 states in, 2 flux states out, per face
@printf("fused kernel useful traffic %.3f GB in %.3f ms = %.0f GB/s achieved\n",
        useful/1e9, t_fused, useful/1e9/(t_fused/1000))
@printf("A100 80GB PCIe DRAM peak is ~1935 GB/s => %.1f%% of peak.\n",
        100*(useful/1e9/(t_fused/1000))/1935)
println("Far from peak bandwidth AND at 12.5%% occupancy => this kernel is COMPUTE bound,")
println("not memory bound: the fix has to be less arithmetic, not better data movement.")
