# fix1_itereig_probe.jl — verify the DEVICE-STYLE extreme-eigenvalue estimator
# (fixed-iteration power / inverse-power through the Cholesky factor) reproduces
# the CPU svdvals(B) gate decision, so it can replace LAPACK eigvals on device.
#
# The device gate must compute κ(B)=sqrt(λmax(G)/λmin(G)) WITHOUT LAPACK. We use:
#   λmax : power iteration on G (fixed iters).
#   λmin : since G=LLᵀ (Cholesky, already needed for the SPD solve), inverse power
#          iteration solves G y = x via two triangular solves; converges to 1/λmin.
# This probe runs both against LAPACK eigvals on the harvested real-cell Grams and
# confirms the gate decision matches the CPU svdvals(B) gate on ~100%.
#
# GPU-light (CPU LinearAlgebra only). Run before the GPU compile.

using Riemann35
using JLD2, Printf, Random, LinearAlgebra
const R = Riemann35

const TRIPLES = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),
 (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]

_monomial(coords, p, e) = begin
    v=1.0; @inbounds for d in eachindex(e); ed=e[d]; ed==0 || (v*=coords[p,d]^ed); end; v
end

# fixed-iteration extreme-eig estimate on SPD G (n×n, n≤9), mirroring the device
# algorithm. Returns κ(G)=λmax/λmin (so κ(B)=sqrt of it). Inf if Cholesky fails.
function itereig_condG(G::Matrix{Float64}; itp::Int=40, iti::Int=40)
    n = size(G,1)
    # Cholesky G=LLᵀ (fails → linearly dependent → reject)
    L = zeros(n,n)
    for i in 1:n
        s = G[i,i]
        for k in 1:i-1; s -= L[i,k]^2; end
        s <= 0 && return Inf
        L[i,i] = sqrt(s)
        for j in i+1:n
            t = G[j,i]
            for k in 1:i-1; t -= L[j,k]*L[i,k]; end
            L[j,i] = t/L[i,i]
        end
    end
    # power iteration for λmax
    x = ones(n);
    lam_max = 0.0
    for _ in 1:itp
        y = G*x
        ny = norm(y); ny == 0 && break
        y ./= ny
        lam_max = dot(y, G*y)
        x = y
    end
    # inverse power iteration for λmin: solve G z = x via L Lᵀ z = x
    z = ones(n)
    lam_min = lam_max
    for _ in 1:iti
        # forward solve L w = z
        w = copy(z)
        for i in 1:n
            s = w[i]
            for k in 1:i-1; s -= L[i,k]*w[k]; end
            w[i] = s/L[i,i]
        end
        # back solve Lᵀ y = w
        yv = copy(w)
        for i in n:-1:1
            s = yv[i]
            for k in i+1:n; s -= L[k,i]*yv[k]; end
            yv[i] = s/L[i,i]
        end
        ny = norm(yv); ny == 0 && break
        yv ./= ny
        # Rayleigh quotient of G on yv (→ λmin)
        lam_min = dot(yv, G*yv)
        z = yv
    end
    lam_min <= 0 && return Inf
    return lam_max/lam_min
end

function harvest!(store, pw, coords, targets)
    Np=length(pw); nt=length(targets)
    order=sortperm([sum(targets[m][1]) for m in 1:nt])
    sw=sqrt.(max.(pw,0.0)); condmax=1e4; sel=Int[]
    B=Matrix{Float64}(undef,Np,0)
    @inbounds for m in order
        col=Float64[sw[p]*_monomial(coords,p,targets[m][1]) for p in 1:Np]
        all(iszero,col) && continue
        Btry=hcat(B,col)
        s=svdvals(Btry); condB=(isempty(s)||s[end]<=0) ? Inf : s[1]/s[end]
        cpu_accept=(size(Btry,2)<=Np)&&(condB<condmax)
        push!(store,(Btry'*Btry, condB, cpu_accept))
        if cpu_accept; B=Btry; push!(sel,m); length(sel)==Np && break; end
    end
end

function cm_all(M)
    Mraw=zeros(5,5,5)
    for n in 1:35; (i,j,k)=TRIPLES[n]; Mraw[i+1,j+1,k+1]=M[n]; end
    rho=Mraw[1,1,1]; bu=Mraw[2,1,1]/rho; bv=Mraw[1,2,1]/rho; bw=Mraw[1,1,2]/rho
    cm(i,j,k)=begin s=0.0
        for a in 0:i,b in 0:j,d in 0:k
            s+=binomial(i,a)*binomial(j,b)*binomial(k,d)*(-bu)^(i-a)*(-bv)^(j-b)*(-bw)^(k-d)*Mraw[a+1,b+1,d+1]
        end; s/rho end
    cm,bu,bv,bw,rho
end

function main()
    Random.seed!(7)
    files=["/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma100_o1.jld2",
           "/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma10_o1.jld2"]
    cells=NTuple{35,Float64}[]
    for f in files
        isfile(f)||continue
        data=jldopen(f,"r") do jf
            arr=nothing; for k in keys(jf); v=jf[k]; if v isa AbstractArray && ndims(v)==4 && size(v,4)==35; arr=v; break; end; end; arr
        end
        data===nothing && continue
        nx,ny,nz,_=size(data); nc=nx*ny*nz
        for lin in randperm(nc)[1:min(nc,10000)]
            k=(lin-1)÷(nx*ny)+1; r=(lin-1)%(nx*ny); j=r÷nx+1; i=r%nx+1
            m=ntuple(t->Float64(data[i,j,k,t]),Val(35)); m[1]>0||continue; push!(cells,m)
        end
    end
    println("Cells: ", length(cells))
    store=Tuple{Matrix{Float64},Float64,Bool}[]
    for M in cells
        local ncpu,Ucpu
        try; ncpu,Ucpu=R.chyqmom_nodes_3d(collect(M)); catch; continue; end
        cm,bu,bv,bw,rho=cm_all(M)
        kxy=Dict{Tuple{Float64,Float64},Float64}()
        for q in eachindex(ncpu); key=(Ucpu[q,1],Ucpu[q,2]); kxy[key]=get(kxy,key,0.0)+ncpu[q]/rho; end
        pw=Float64[];cx=Float64[];cy=Float64[]
        for (key,w) in kxy; push!(pw,w);push!(cx,key[1]-bu);push!(cy,key[2]-bv);end
        coords=hcat(cx,cy)
        mt=Tuple{NTuple{2,Int},Float64}[]; for i in 0:3,j in 0:(3-i); push!(mt,((i,j),cm(i,j,1)));end
        harvest!(store,pw,coords,mt)
        vt=Tuple{NTuple{2,Int},Float64}[]; for i in 0:2,j in 0:(2-i); push!(vt,((i,j),cm(i,j,2)));end
        harvest!(store,pw,coords,vt)
    end
    println("Decisions: ", length(store))
    condmax=1e4
    nmis=0; maxrel=0.0; nacc=0;nrej=0
    for (G,condB,cpu_acc) in store
        cpu_cond = condB<condmax
        cpu_cond ? (nacc+=1) : (nrej+=1)
        kG = itereig_condG(G)
        kB = sqrt(kG)
        dev_acc = kB < condmax
        (dev_acc != cpu_cond) && (nmis+=1)
        # compare kB vs LAPACK on well-conditioned (accepted) cases
        if cpu_cond
            ev=eigvals(Symmetric(G)); lo=minimum(ev);hi=maximum(ev)
            if lo>0; kref=sqrt(hi/lo); maxrel=max(maxrel,abs(kB-kref)/kref); end
        end
    end
    N=length(store)
    println("\n===== FIX 1 ITER-EIG (device-style) PROBE =====")
    @printf("Decisions: %d (%d acc, %d rej)\n", N,nacc,nrej)
    @printf("iter-eig gate mismatches vs CPU svd gate : %d / %d (%.4f%%)\n", nmis,N,100*nmis/N)
    @printf("max rel err kB(iter) vs kB(LAPACK) on accepted : %.3e\n", maxrel)
    println("===============================================")
end
main()
