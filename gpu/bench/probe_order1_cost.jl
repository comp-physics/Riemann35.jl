# probe_order1_cost.jl — what does an ORDER-1 run pay for order-3 code it never executes?
#
# _weno_flux_x! takes `first_order::Bool` as a RUNTIME argument, so one compiled kernel
# carries both paths. The local frame is a static, per-thread allocation sized for the worst
# case, and the register count is whatever the whole body needs -- neither depends on which
# branch a thread takes. Measured: the o1 and o3 instantiations have an IDENTICAL 12048 B
# frame, 255 registers, and 12.5% occupancy.
#
# So an order-1 launch reserves the order-3 frame and runs at the order-3 occupancy to do
# one HLL on two cell means. `k_lo!` here is exactly that work as its own kernel -- what
# order-1 would cost if `first_order` were a compile-time (Val) parameter, letting ptxas
# size the frame to the branch actually taken.
#
# This is the one clearly shippable item to come out of the #15 investigation; the order-3
# path itself is compute-bound and did not yield to any restructuring tried.
using CUDA, Printf
const GPUDIR = get(ENV, "R35_GPUDIR", "/fastscratch/sbryngelson3/Riemann35.jl/gpu")
include(joinpath(GPUDIR, "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU
const R = Residual3DOrder3GPU
using Riemann35
const cellG = R._cellG; const hll = R._hll_states

# order-1 work, standing alone: FHO = FLO = HLL(cell, cell). Same output as
# _weno_flux_x!(..., first_order=true), which also writes both arrays.
function k_lo!(FHO, FLO, G, nx, ny, nz, g, Ma, s3f)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f=(idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1; b=g+j; c=g+k; il=g+f-1
            FL = hll(cellG(G,il,b,c), cellG(G,il+1,b,c), 1, Ma, s3f)
            for m in 1:35; FHO[m,f,j,k]=FL[m]; FLO[m,f,j,k]=FL[m]; end
        end
    end
    nothing
end

nx=ny=nz=64; g=8; nfx=nx+2g
Mc = InitializeM4_35(1.0,0.0,0.0,0.0,1.0,0,0,1.0,0,1.0)
M0 = zeros(35,nfx,nfx,nfx)
for k in 1:nfx, j in 1:nfx, i in 1:nfx
    s = 1.0 + 0.05*sin(2pi*i/nfx); for m in 1:35; M0[m,i,j,k]=Mc[m]*s; end
end
G=CuArray(M0)
Vp=CUDA.zeros(Float64,35,nfx,nfx,nfx); V=CUDA.zeros(Float64,35,nfx,nfx,nfx)
thr=128; bcube=cld(nfx^3,thr)
@cuda threads=thr blocks=bcube R._ppt_x!(Vp,G,nfx,nfx,nfx)
@cuda threads=thr blocks=bcube R._vavg_x!(V,Vp,nfx,nfx,nfx); CUDA.synchronize()
FHO=CUDA.zeros(Float64,35,nx+1,ny,nz); FLO=CUDA.zeros(Float64,35,nx+1,ny,nz)
FH2=CUDA.zeros(Float64,35,nx+1,ny,nz); FL2=CUDA.zeros(Float64,35,nx+1,ny,nz)
nf=(nx+1)*ny*nz; bl=cld(nf,thr)

tm(f;n=50)=(f();CUDA.synchronize();e=CUDA.@elapsed for _ in 1:n; f(); end; 1000*e/n)
t_o1_fused = tm(()->(@cuda threads=thr blocks=bl R._weno_flux_x!(FHO,FLO,G,V,nx,ny,nz,g,nfx,1.0,40.0,true)))
t_o1_alone = tm(()->(@cuda threads=thr blocks=bl k_lo!(FH2,FL2,G,nx,ny,nz,g,1.0,40.0)))

kf = @cuda launch=false R._weno_flux_x!(FHO,FLO,G,V,nx,ny,nz,g,nfx,1.0,40.0,true)
ka = @cuda launch=false k_lo!(FH2,FL2,G,nx,ny,nz,g,1.0,40.0)
fr(k)=(CUDA.memory(k).local, CUDA.registers(k))

println("="^80); println("WHAT ORDER-1 PAYS FOR THE RUNTIME `first_order` Bool"); println("="^80)
@printf("_weno_flux_x!(first_order=true)   %7.3f ms   frame=%5d B  regs=%d\n",
        t_o1_fused, fr(kf)[1], fr(kf)[2])
@printf("k_lo!  (same work, own kernel)    %7.3f ms   frame=%5d B  regs=%d\n",
        t_o1_alone, fr(ka)[1], fr(ka)[2])
@printf("\norder-1 speedup available: %.2fx   (frame %d -> %d B)\n",
        t_o1_fused/t_o1_alone, fr(kf)[1], fr(ka)[1])

# outputs must match exactly -- same arithmetic, same inputs
@cuda threads=thr blocks=bl R._weno_flux_x!(FHO,FLO,G,V,nx,ny,nz,g,nfx,1.0,40.0,true)
@cuda threads=thr blocks=bl k_lo!(FH2,FL2,G,nx,ny,nz,g,1.0,40.0); CUDA.synchronize()
d = max(maximum(abs.(Array(FHO).-Array(FH2))), maximum(abs.(Array(FLO).-Array(FL2))))
@printf("max |fused_o1 - standalone_o1| = %.3e  %s\n", d, d==0.0 ? "(BIT-IDENTICAL)" : "*** DIFFERS ***")
