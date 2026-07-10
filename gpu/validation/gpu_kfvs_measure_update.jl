# gpu_kfvs_measure_update.jl — GPU storage pass + 3D measure_update anchor on the
# V100. Measures the store-pass footprint/throughput and runs the anchor over
# interior cells reading the stored quadratures, reporting the min-weight
# realizability certificate on-device + throughput.
#
# PACE V100/CUDA-12.9: benign "val already in a list" atexit error prints AFTER
# results — ignore it. First CUDA compile is slow (several minutes).

using CUDA, JLD2, Printf, Random

include(joinpath(@__DIR__, "..", "chyqmom_nodes_3d_dev.jl"))
using .KFVSInversionDev
include(joinpath(@__DIR__, "..", "kfvs_measure_update_dev.jl"))
using .KFVSMeasureUpdateDev

# storage layout: S[node, chan, cell] with chan 1..4 = (w, Ux, Uy, Uz); flat cell
# index over (nx,ny,nz). Counts NC[cell] :: Int32.
# matches the device store4! signature (NW,UX,UY,UZ, ci,q, w,ux,uy,uz); we pass S
# for all four array slots and write the 4 channels of the single (27,4,cell) array.
@inline function _store4_g!(NW, UX, UY, UZ, ci, q, w, ux, uy, uz)
    @inbounds begin
        NW[q, 1, ci] = w; NW[q, 2, ci] = ux; NW[q, 3, ci] = uy; NW[q, 4, ci] = uz
    end
    return nothing
end

# ---- STORAGE kernel: one thread per cell, invert + store its quadrature ----
function store_kernel!(S, NC, M, ncell)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= ncell
        @inbounds begin
            m = ntuple(t -> M[i, t], Val(35))
            Nn = chyqmom_nodes_3d_store_dev!(_store4_g!, S, S, S, S, i, m)  # extra ptrs unused
            NC[i] = Nn
        end
    end
    return nothing
end

# ---- MEASURE_UPDATE kernel: one thread per interior cell (i,j,k), reads the
# 7-cell stencil from S/NC, writes the updated 35-moment state + min-weight ----
function measure_kernel!(Mout, MINW, S, NC, nx, ny, nz, CFL)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nint = (nx-2)*(ny-2)*(nz-2)
    if t <= nint
        @inbounds begin
            # decode interior (i,j,k) in 2:nx-1 etc.
            ii = (t - 1) % (nx-2) + 2
            r  = (t - 1) ÷ (nx-2)
            jj = r % (ny-2) + 2
            kk = r ÷ (ny-2) + 2
            lin(i,j,k) = ((k-1)*ny + (j-1))*nx + i
            cC  = lin(ii,jj,kk)
            cLx = lin(ii-1,jj,kk); cRx = lin(ii+1,jj,kk)
            cLy = lin(ii,jj-1,kk); cRy = lin(ii,jj+1,kk)
            cLz = lin(ii,jj,kk-1); cRz = lin(ii,jj,kk+1)
            cells = (cC, cLx, cRx, cLy, cRy, cLz, cRz)
            # skip if any stencil cell has no nodes (leave output zero / minw=Inf)
            ok = true
            for s in 1:7; (NC[cells[s]] > 0) || (ok = false); end
            if ok
                # 3D CFL: λ * max over stencil of max_k(|Ux|+|Uy|+|Uz|) <= CFL
                smax = 1.0e-300
                for s in 1:7
                    c = cells[s]
                    for q in 1:Int(NC[c])
                        sp = abs(S[q,2,c]) + abs(S[q,3,c]) + abs(S[q,4,c])
                        smax = sp > smax ? sp : smax
                    end
                end
                λ = CFL / smax
                cnt(slot) = Int(NC[cells[slot]])
                gw(slot,q)=S[q,1,cells[slot]]; gx(slot,q)=S[q,2,cells[slot]]
                gy(slot,q)=S[q,3,cells[slot]]; gz(slot,q)=S[q,4,cells[slot]]
                (Mup, minw) = measure_update_3d_dev(gw, gx, gy, gz, cnt, λ)
                for n in 1:35; Mout[t, n] = Mup[n]; end
                MINW[t] = minw
            else
                MINW[t] = Inf
            end
        end
    end
    return nothing
end

function main()
    println("CUDA functional: ", CUDA.functional(), "  device: ", CUDA.name(CUDA.device()))
    f = "/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma100_o1.jld2"
    data = jldopen(f,"r") do jf
        arr=nothing; for k in keys(jf); v=jf[k]; if v isa AbstractArray && ndims(v)==4 && size(v,4)==35; arr=v; break; end; end; arr
    end
    NX,NY,NZ,_ = size(data)
    # use a 128^3 (full) field
    nx,ny,nz = NX,NY,NZ
    println("Field: ($nx,$ny,$nz)")
    sub = Float64.(data)  # (nx,ny,nz,35)
    ncell = nx*ny*nz
    # flatten to (ncell,35) column-major matching lin(i,j,k)=((k-1)*ny+(j-1))*nx+i
    Mh = Array{Float64}(undef, ncell, 35)
    for k in 1:nz, j in 1:ny, i in 1:nx
        c = ((k-1)*ny + (j-1))*nx + i
        for t in 1:35; Mh[c,t] = sub[i,j,k,t]; end
    end
    Md = CuArray(Mh)

    S  = CUDA.zeros(Float64, 27, 4, ncell)
    NC = CUDA.zeros(Int32, ncell)
    store_bytes = 27*4*8*ncell + 4*ncell
    @printf("Storage footprint: %.3f GB (S: 27x4x8xNcell) + %.1f MB counts\n",
            27*4*8*ncell/1e9, 4*ncell/1e6)

    # compile + measure store kernel
    thr = 128
    kst = @cuda launch=false store_kernel!(S, NC, Md, ncell)
    @printf("STORE kernel: %d regs, local %d B, occ maxthreads %d\n",
            CUDA.registers(kst), get(CUDA.memory(kst), :local, 0), CUDA.maxthreads(kst))
    blk = cld(ncell, thr)
    kst(S, NC, Md, ncell; threads=thr, blocks=blk); CUDA.synchronize()
    nrep=10
    tstore = CUDA.@elapsed begin
        for _ in 1:nrep; kst(S, NC, Md, ncell; threads=thr, blocks=blk); end; CUDA.synchronize()
    end
    @printf("STORE pass: %.4f us/cell (%d cells) => %.1f ms for the %d^3 field\n",
            tstore/nrep/ncell*1e6, ncell, tstore/nrep*1e3, nx)

    # measure_update kernel over interior cells
    nint = (nx-2)*(ny-2)*(nz-2)
    Mout = CUDA.zeros(Float64, nint, 35)
    MINW = CUDA.fill(Inf, nint)
    CFL = 0.4
    kmu = @cuda launch=false measure_kernel!(Mout, MINW, S, NC, nx, ny, nz, CFL)
    @printf("MEASURE kernel: %d regs, local %d B\n",
            CUDA.registers(kmu), get(CUDA.memory(kmu), :local, 0))
    blk2 = cld(nint, thr)
    kmu(Mout, MINW, S, NC, nx, ny, nz, CFL; threads=thr, blocks=blk2); CUDA.synchronize()
    tmu = CUDA.@elapsed begin
        for _ in 1:nrep; kmu(Mout, MINW, S, NC, nx, ny, nz, CFL; threads=thr, blocks=blk2); end; CUDA.synchronize()
    end
    @printf("MEASURE pass: %.4f us/cell (%d interior cells) => %.1f ms\n",
            tmu/nrep/nint*1e6, nint, tmu/nrep*1e3)

    # ---- realizability certificate on device output ----
    MINWh = Array(MINW)
    valid = findall(isfinite, MINWh)
    minw_all = isempty(valid) ? Inf : minimum(MINWh[valid])
    nneg = count(<(-1e-12), MINWh[valid])
    @printf("\n3D measure_update on-device certificate (CFL=%.2f, %d valid interior stencils):\n", CFL, length(valid))
    @printf("  min measure weight over all stencils : %.3e\n", minw_all)
    @printf("  stencils with weight < -1e-12        : %d  (cert >= -1e-12 ? %s)\n",
            nneg, nneg==0 ? "YES (100%)" : "NO")
    # mass positivity of updated states
    Mouth = Array(Mout)
    nbadrho = 0
    for c in valid; (Mouth[c,1] > 0.0) || (nbadrho += 1); end
    @printf("  updated states with rho<=0           : %d\n", nbadrho)
    println("==========================================")
end
main()
