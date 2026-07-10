# gpu_kfvs_blend.jl — GPU cost of the full-cone θ* blend limiter on the V100.
# Measures registers + us/cell of theta_star_blend_fullcone_dev over real stencils,
# and confirms 0 cross-moment-cone exits on-device.
#
# PACE V100/CUDA-12.9: benign "val already in a list" atexit error prints AFTER
# results — ignore it. First CUDA compile is slow (several minutes).

using CUDA, JLD2, Printf, Random

include(joinpath(@__DIR__, "..", "chyqmom_nodes_3d_dev.jl"))
using .KFVSInversionDev
include(joinpath(@__DIR__, "..", "kfvs_measure_update_dev.jl"))
using .KFVSMeasureUpdateDev
include(joinpath(@__DIR__, "..", "kfvs_blend_dev.jl"))
using .KFVSBlendDev

# Blend kernel: one thread per (Ua, Uho) pair. Reads two 35-moment states from
# global arrays, computes the full-cone θ* blend, writes θ* and a realizability
# flag. (Ua/Uho are precomputed on the host from real stencils — this kernel
# isolates the LIMITER cost, which is the question.)
function blend_kernel!(THETA, OKFLAG, Ua, Uho, npair)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= npair
        @inbounds begin
            ua = ntuple(j -> Ua[t, j], Val(35))
            uh = ntuple(j -> Uho[t, j], Val(35))
            (θ, Ustar) = theta_star_blend_fullcone_dev(ua, uh)
            THETA[t] = θ
            OKFLAG[t] = state_realizable_fullcone_dev(Ustar) ? Int32(1) : Int32(0)
        end
    end
    return nothing
end

# Host: build (Ua, Uho) pairs from real stencils (reuse the CPU measure_update +
# high-order construction from the parity script, inlined minimally here).
const TRIPLES = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),
 (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]

function invert_store(M)
    S = zeros(27,4)
    st!(NW,UX,UY,UZ,ci,q,w,ux,uy,uz)=(S[q,1]=w; S[q,2]=ux; S[q,3]=uy; S[q,4]=uz; nothing)
    Nn = chyqmom_nodes_3d_store_dev!(st!, nothing,nothing,nothing,nothing, 1, M)
    return S, Nn
end
function kfvs_split_axis(S, Nn, axis)
    Fp=zeros(35); Fm=zeros(35)
    for k in 1:Nn
        nk=S[k,1]; ua=S[k,1+axis]
        for m in 1:35; (i,j,l)=TRIPLES[m]; val=nk*ua*(S[k,2]^i)*(S[k,3]^j)*(S[k,4]^l); ua>=0 ? (Fp[m]+=val) : (Fm[m]+=val); end
    end
    Fp,Fm
end
iface(SL,NL,SR,NR,axis)=kfvs_split_axis(SL,NL,axis)[1] .+ kfvs_split_axis(SR,NR,axis)[2]

function build_pairs(sub, B; want=200000)
    Sarr=Array{Matrix{Float64}}(undef,B,B,B); Narr=zeros(Int,B,B,B)
    for i in 1:B,j in 1:B,k in 1:B
        M=ntuple(t->sub[i,j,k,t],Val(35))
        if M[1]>0.0; Sarr[i,j,k],Narr[i,j,k]=invert_store(M); else; Sarr[i,j,k]=zeros(27,4); Narr[i,j,k]=0; end
    end
    Ua=Float64[]; Uho=Float64[]; np=0
    UaM=Matrix{Float64}(undef, 0, 35); # will build as vectors of rows
    rowsA=Vector{NTuple{35,Float64}}(); rowsH=Vector{NTuple{35,Float64}}()
    for i in 2:B-1,j in 2:B-1,k in 2:B-1
        cells=((i,j,k),(i-1,j,k),(i+1,j,k),(i,j-1,k),(i,j+1,k),(i,j,k-1),(i,j,k+1))
        all(c->Narr[c...]>0, cells) || continue
        SC=Sarr[i,j,k]; NC=Narr[i,j,k]
        smax=1e-300; for c in cells; S=Sarr[c...]; for q in 1:Narr[c...]; smax=max(smax,abs(S[q,2])+abs(S[q,3])+abs(S[q,4])); end; end
        λ=0.4/smax
        cnt(s)=Narr[cells[s]...]; gw(s,q)=Sarr[cells[s]...][q,1]; gx(s,q)=Sarr[cells[s]...][q,2]
        gy(s,q)=Sarr[cells[s]...][q,3]; gz(s,q)=Sarr[cells[s]...][q,4]
        (ua,minw)=measure_update_3d_dev(gw,gx,gy,gz,cnt,λ); minw<-1e-12 && continue
        MC=ntuple(t->sub[i,j,k,t],Val(35)); dU=zeros(35)
        for (axis,(cl,cr)) in ((1,((i-1,j,k),(i+1,j,k))),(2,((i,j-1,k),(i,j+1,k))),(3,((i,j,k-1),(i,j,k+1))))
            dU .+= (iface(SC,NC,Sarr[cr...],Narr[cr...],axis) .- iface(Sarr[cl...],Narr[cl...],SC,NC,axis))
        end
        uho=ntuple(t->MC[t]-λ*dU[t],Val(35))
        push!(rowsA,ua); push!(rowsH,uho); np+=1
        np>=want && break
    end
    Ah=Matrix{Float64}(undef,np,35); Hh=Matrix{Float64}(undef,np,35)
    for r in 1:np, c in 1:35; Ah[r,c]=rowsA[r][c]; Hh[r,c]=rowsH[r][c]; end
    return Ah, Hh
end

function main()
    println("CUDA functional: ", CUDA.functional(), "  device: ", CUDA.name(CUDA.device()))
    f="/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma100_o1.jld2"
    data=jldopen(f,"r") do jf; a=nothing; for k in keys(jf); v=jf[k]; if v isa AbstractArray && ndims(v)==4 && size(v,4)==35; a=v; break; end; end; a; end
    B=48; sub=Float64.(data[41:41+B-1, 41:41+B-1, 41:41+B-1, :])
    println("Building (Ua,Uho) pairs from real stencils...")
    Ah, Hh = build_pairs(sub, B; want=200000)
    npair = size(Ah,1); println("pairs: ", npair)
    Ad=CuArray(Ah); Hd=CuArray(Hh)
    THETA=CUDA.zeros(Float64, npair); OK=CUDA.zeros(Int32, npair)

    thr=128
    k = @cuda launch=false blend_kernel!(THETA, OK, Ad, Hd, npair)
    @printf("BLEND kernel (full-cone θ*): %d regs, local %d B, maxthreads %d\n",
            CUDA.registers(k), get(CUDA.memory(k), :local, 0), CUDA.maxthreads(k))
    blk=cld(npair,thr)
    k(THETA,OK,Ad,Hd,npair; threads=thr, blocks=blk); CUDA.synchronize()
    nrep=20
    t=CUDA.@elapsed begin
        for _ in 1:nrep; k(THETA,OK,Ad,Hd,npair; threads=thr, blocks=blk); end; CUDA.synchronize()
    end
    @printf("BLEND throughput: %.4f us/cell (%d pairs, %d reps)\n", t/nrep/npair*1e6, npair, nrep)
    THh=Array(THETA); OKh=Array(OK)
    @printf("mean θ* = %.4f ; θ*=1 on %.2f%% ; U(θ*) full-cone realizable on %d/%d (%.4f%%) [exits: %d]\n",
            sum(THh)/npair, 100*count(>=(1-1e-9),THh)/npair, count(==(1),OKh), npair,
            100*count(==(1),OKh)/npair, count(==(0),OKh))
    println("==========================================")
end
main()
