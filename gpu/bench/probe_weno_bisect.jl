# probe_weno_bisect.jl — WHICH PIECE of _weno_flux_x! owns the 12 KB local frame?
#
# Baseline (A100, 2026-07-27): _weno_flux_x! is regs=255, local=12048 B/thread, 12.5%
# occupancy at EVERY block size 64/128/256 -- which is exactly why the parked block-size
# sweep came out flat. 12048 B is ~1506 doubles, ~5x the ~315 doubles live at the peak of
# the dataflow, so the frame is not merely "the stencil doesn't fit".
#
# `@noinline weno_scaled_face_dev` was tried first and made it WORSE (12048 -> 13536 B):
# NTuple{35} arguments cross a non-inlined boundary through memory, the same mechanism
# already logged for the CPU allocation issue. So the fix is not a call-boundary tweak.
#
# This bisects the kernel into its pieces, each as a standalone kernel over the same
# geometry, and reports registers + frame for each. Whatever piece carries ~12 KB alone is
# the thing to restructure; if no single piece does, the frame is a sum and the split-kernel
# approach is the right lever after all.
using CUDA, Printf
const GPUDIR = get(ENV, "R35_GPUDIR", "/fastscratch/sbryngelson3/Riemann35.jl/gpu")
include(joinpath(GPUDIR, "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU
const R = Residual3DOrder3GPU
using Riemann35.HiOrder3ReconDev: weno_faces_dev, weno_scaled_face_dev
using Riemann35.ReconDev: to_recon_vars_tup
using Riemann35.Weno5Dev: weno5z

const cellG = R._cellG
const hll   = R._hll_states
const clmp  = R._clamp

# (A) LO only: one HLL on the two raw cell means. No reconstruction at all.
function k_lo!(FLO, G, nx, ny, nz, g, Ma, s3f)
    idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    nf = nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f = (idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1
            b=g+j; c=g+k; il=g+f-1
            FL = hll(cellG(G,il,b,c), cellG(G,il+1,b,c), 1, Ma, s3f)
            for m in 1:35; FLO[m,f,j,k]=FL[m]; end
        end
    end
    nothing
end

# (B) WENO faces only: the 6-wide stencil -> mL,mR, written out. No HLL.
function k_faces!(ML, MR, G, V, nx, ny, nz, g, nfx)
    idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    nf = nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f = (idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1
            b=g+j; c=g+k; il=g+f-1
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

# (B') WENO5-Z only, WITHOUT the realizability bisection: isolates the 20-iteration
# bisection in weno_scaled_face_dev from the plain 6-point weno5z reconstruction.
function k_faces_raw!(ML, MR, G, V, nx, ny, nz, g, nfx)
    idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    nf = nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f = (idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1
            b=g+j; c=g+k; il=g+f-1
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

# (C) HLL on precomputed faces: the second half of the split-kernel proposal.
function k_hll_faces!(FHO, ML, MR, nx, ny, nz, Ma, s3f)
    idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    nf = nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f = (idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1
            mL = ntuple(q -> ML[q,f,j,k], Val(35))
            mR = ntuple(q -> MR[q,f,j,k], Val(35))
            FH = hll(mL, mR, 1, Ma, s3f)
            for m in 1:35; FHO[m,f,j,k]=FH[m]; end
        end
    end
    nothing
end

nx=ny=nz=16; g=8; nfx=nfy=nfz=nx+2g
G=CUDA.zeros(Float64,35,nfx,nfy,nfz); V=CUDA.zeros(Float64,35,nfx,nfy,nfz)
FHO=CUDA.zeros(Float64,35,nx+1,ny,nz); FLO=CUDA.zeros(Float64,35,nx+1,ny,nz)
ML=CUDA.zeros(Float64,35,nx+1,ny,nz);  MR=CUDA.zeros(Float64,35,nx+1,ny,nz)

function rep(name, kern)
    m=CUDA.memory(kern); rg=CUDA.registers(kern)
    occ = CUDA.active_blocks(kern.fun, 128); w = occ*128/32
    @printf("%-34s regs=%3d  local=%6d B   occ@128=%5.1f%%\n", name, rg, m.local, 100*w/64)
    (rg, m.local)
end

println("="^92)
println("BISECTING THE 12 KB FRAME of _weno_flux_x!  (A100)")
println("="^92)
rep("FULL _weno_flux_x! (o3)",
    @cuda launch=false R._weno_flux_x!(FHO,FLO,G,V,nx,ny,nz,g,nfx,1.0,40.0,false))
println("-"^92)
rep("(A) HLL on cell means only",     @cuda launch=false k_lo!(FLO,G,nx,ny,nz,g,1.0,40.0))
rep("(B') weno5z faces, NO bisection", @cuda launch=false k_faces_raw!(ML,MR,G,V,nx,ny,nz,g,nfx))
rep("(B) weno_faces_dev (w/ bisect)",  @cuda launch=false k_faces!(ML,MR,G,V,nx,ny,nz,g,nfx))
rep("(C) HLL on precomputed faces",    @cuda launch=false k_hll_faces!(FHO,ML,MR,nx,ny,nz,1.0,40.0))
println("="^92)
println("READ: (B) minus (B') is the realizability bisection's own frame cost.")
println("      If (B) alone is ~12 KB, splitting off the HLLs cannot help and the")
println("      bisection is the target. If every piece is small, the frame is a SUM")
println("      and splitting into separate kernels is the right lever.")
