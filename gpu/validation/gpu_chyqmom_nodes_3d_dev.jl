# gpu_chyqmom_nodes_3d_dev.jl — compile the device CHyQMOM inversion into a
# one-thread-per-cell CUDA kernel on the V100 and MEASURE registers / occupancy /
# throughput, for BOTH:
#   * FUSED  : `chyqmom_nodes_3d_dev` returning the 27-node NTuple, written to global.
#   * SPLIT  : `chyqmom_nodes_3d_store_dev!` writing each node directly to global as
#              produced (design §1.5 invert-and-store phase-1; no 27×4 NTuple
#              accumulator held live). This informs the storage-based split design.
# Also checks device output realizability (min weight) + mass.
#
# NOTE (PACE V100/CUDA-12.9): a benign "val already in a list" atexit teardown
# error prints AFTER results — ignore it, the numbers above it are valid. First
# CUDA compile is slow (several minutes).

using CUDA, JLD2, Printf, Random

include(joinpath(@__DIR__, "..", "chyqmom_nodes_3d_dev.jl"))
using .KFVSInversionDev

# ---- FUSED kernel: returns NTuple, writes all 27 slots ----
function chyq_fused!(NW, UX, UY, UZ, NN, M, ncell)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= ncell
        @inbounds begin
            m = ntuple(t -> M[i, t], Val(35))
            (nn, nux, nuy, nuz, Nn) = chyqmom_nodes_3d_dev(m)
            for q in 1:27
                NW[i, q] = nn[q]; UX[i, q] = nux[q]; UY[i, q] = nuy[q]; UZ[i, q] = nuz[q]
            end
            NN[i] = Nn
        end
    end
    return nothing
end

# ---- SPLIT kernel: incremental direct global stores (no 27-slot NTuple) ----
@inline function _store4!(NW, UX, UY, UZ, ci, q, w, ux, uy, uz)
    @inbounds begin
        NW[ci, q] = w; UX[ci, q] = ux; UY[ci, q] = uy; UZ[ci, q] = uz
    end
    return nothing
end
function chyq_split!(NW, UX, UY, UZ, NN, M, ncell)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= ncell
        @inbounds begin
            m = ntuple(t -> M[i, t], Val(35))
            Nn = chyqmom_nodes_3d_store_dev!(_store4!, NW, UX, UY, UZ, i, m)
            NN[i] = Nn
        end
    end
    return nothing
end

function measure(name, kern, args, ncell; threads=128)
    kernel = @cuda launch=false kern(args...)
    regs = CUDA.registers(kernel)
    shmem = CUDA.memory(kernel)
    maxthreads = CUDA.maxthreads(kernel)
    active = CUDA.active_blocks(kernel.fun, threads)
    wpb = cld(threads, 32)
    occ = active * wpb / 64
    println("\n========== $name (V100) ==========")
    @printf("Registers per thread   : %d  (255-wall: %s)\n", regs,
            regs > 255 ? "EXCEEDED by $(regs-255)" : "under by $(255-regs)")
    @printf("Local mem / thread     : %d B (spill)   shared=%d B\n",
            get(shmem, :local, 0), get(shmem, :shared, 0))
    @printf("Max threads/block      : %d\n", maxthreads)
    @printf("At %d thr/blk          : active blocks/SM=%d => occupancy=%.1f%% (%d/64 warps)\n",
            threads, active, 100*occ, active*wpb)
    # run + time
    blocks = cld(ncell, threads)
    kernel(args...; threads=threads, blocks=blocks); CUDA.synchronize()
    nrep = 20
    t = CUDA.@elapsed begin
        for _ in 1:nrep; kernel(args...; threads=threads, blocks=blocks); end
        CUDA.synchronize()
    end
    us = t/nrep/ncell*1e6
    @printf("Throughput             : %.4f us/cell  (%d cells, %d reps; CPU baseline ~130 us/cell)\n", us, ncell, nrep)
    return regs, occ, us
end

function main()
    println("CUDA functional: ", CUDA.functional(), "  device: ", CUDA.name(CUDA.device()))
    f = "/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma100_o1.jld2"
    data = jldopen(f,"r") do jf
        arr=nothing; for k in keys(jf); v=jf[k]; if v isa AbstractArray && ndims(v)==4 && size(v,4)==35; arr=v; break; end; end; arr
    end
    nx,ny,nz,_ = size(data)
    flat = reshape(Float64.(data), nx*ny*nz, 35)
    good = findall(>(0.0), flat[:,1])
    take = min(length(good), 1_000_000)
    Mh = Array(flat[good[1:take], :]); ncell = size(Mh,1)
    println("Batch cells: ", ncell)

    Md = CuArray(Mh)
    NW=CUDA.zeros(Float64,ncell,27); UX=CUDA.zeros(Float64,ncell,27)
    UY=CUDA.zeros(Float64,ncell,27); UZ=CUDA.zeros(Float64,ncell,27)
    NN=CUDA.zeros(Int32,ncell)

    rf,of,uf = measure("FUSED  (NTuple accumulator + global write)",
                       chyq_fused!, (NW,UX,UY,UZ,NN,Md,ncell), ncell)
    # verify fused device output realizability + mass
    NWh=Array(NW); NNh=Array(NN); minw=Inf; masserr=0.0
    for c in 1:min(ncell,200000)
        s=0.0; for q in 1:NNh[c]; w=NWh[c,q]; minw=min(minw,w); s+=w; end
        masserr=max(masserr, abs(s-Mh[c,1])/max(Mh[c,1],1e-300))
    end
    @printf("FUSED device min weight: %.3e (cert >= -1e-12 ? %s)   mass rel err: %.3e\n",
            minw, minw>=-1e-12 ? "YES" : "NO", masserr)

    # reset outputs, measure split
    NW.=0; UX.=0; UY.=0; UZ.=0; NN.=0
    rs,os,us = measure("SPLIT  (invert-and-store, incremental global stores)",
                       chyq_split!, (NW,UX,UY,UZ,NN,Md,ncell), ncell)
    # verify split == fused (byte-identical stored quadrature)
    NWh2=Array(NW); NNh2=Array(NN); ndiff=0; maxd=0.0
    for c in 1:min(ncell,200000)
        NNh2[c]==NNh[c] || (ndiff+=1)
        for q in 1:NNh2[c]; maxd=max(maxd, abs(NWh2[c,q]-NWh[c,q])); end
    end
    @printf("SPLIT vs FUSED: node-count diffs=%d  max weight diff=%.3e (should be 0 / machine)\n", ndiff, maxd)

    println("\n================= SUMMARY =================")
    @printf("FUSED : %d regs, occ %.1f%%, %.4f us/cell\n", rf, 100*of, uf)
    @printf("SPLIT : %d regs, occ %.1f%%, %.4f us/cell\n", rs, 100*os, us)
    @printf("Register delta (split - fused): %d\n", rs - rf)
    println("==========================================")
end
main()
