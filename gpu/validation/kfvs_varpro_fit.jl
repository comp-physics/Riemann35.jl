# kfvs_varpro_fit.jl — constructive CFL-safe positive cubature by VARIABLE PROJECTION.
# Outer: atom positions X_k (nonlinear).  Inner: weights w>=0 solved exactly by NNLS for the
# row-scaled moment system (so weights never enter the nonlinear search -> removes the weight
# ill-conditioning that traps joint LM at ~1%).  Outer objective = NNLS residual; drive -> 0 by a
# robust optimizer (Optim, multi-start).  When residual->0 the positions admit a positive cubature.
# Positions initialized from the jet-marginal structure; CFL enforced; verified in BigFloat.
#
# usage: julia ... kfvs_varpro_fit.jl [N=30] [seeds=8] [outer_iters=3000]

using JLD2, Printf, LinearAlgebra, Random, Optim, NonNegLeastSquares
setprecision(BigFloat, 256)
N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
NSEED = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
OITER = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3000

const D = load("gpu/validation/kfvs_defect_counterexample.jld2")
const LAM = D["lam"][1]; const C = D["center_state"]; const CLS = D["class"]
const TRIP = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX = Dict(TRIP[n] => n for n in 1:35)
bino(n,k) = (k < 0 || k > n) ? big(0) : big(binomial(n,k))
s0 = findfirst(==(3), CLS)
M = big.(collect(C[:, s0])); ρ = M[1]
ux=M[2]/ρ; uy=M[6]/ρ; uz=M[16]/ρ
σx=sqrt(M[3]/ρ-ux^2); σy=sqrt(M[10]/ρ-uy^2); σz=sqrt(M[20]/ρ-uz^2)
sstdB = zeros(BigFloat,35)
for n in 1:35
    (i,j,k)=TRIP[n]; acc=big(0.0)
    for p in 0:i,q in 0:j,r in 0:k
        haskey(IDX,(p,q,r))||continue
        acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ)
    end
    sstdB[n]=acc/(σx^i*σy^j*σz^k)
end
sstd = Float64.(sstdB)
R = 1/LAM; mean=(Float64(ux),Float64(uy),Float64(uz)); sig=(Float64(σx),Float64(σy),Float64(σz))
Ls = sqrt(sqrt(sstd[5])); rowsc = [1.0/Ls^sum(TRIP[n]) for n in 1:35]; br = sstd .* rowsc
const Lp = 10.0
cflval(X)=LAM*(abs(mean[1]+sig[1]*X[1])+abs(mean[2]+sig[2]*X[2])+abs(mean[3]+sig[3]*X[3]))
eS=(1/sqrt(3),1/sqrt(3),1/sqrt(3)); eT1=(1/sqrt(2),-1/sqrt(2),0.0); eT2=(1/sqrt(6),1/sqrt(6),-2/sqrt(6))
Xof(S,T1,T2)=(S*eS[1]+T1*eT1[1]+T2*eT2[1], S*eS[2]+T1*eT1[2]+T2*eT2[2], S*eS[3]+T1*eT1[3]+T2*eT2[3])
σS=sqrt(1.336); σT=sqrt(0.832)

# build row-scaled moment matrix for positions Y (X=Lp*Y), CFL-projected
function positions(Yv)
    Xs = Vector{NTuple{3,Float64}}(undef, N)
    for kk in 1:N
        X=[Lp*Yv[3(kk-1)+1], Lp*Yv[3(kk-1)+2], Lp*Yv[3(kk-1)+3]]
        c=cflval(X)
        if c>1.0; s=1.0; for _ in 1:80; (cflval(s.*X)<=1.0)&&break; s*=0.92; end; X=s.*X; end
        Xs[kk]=(X[1],X[2],X[3])
    end
    Xs
end
function Phimat(Xs)
    Φ=Array{Float64}(undef,35,N)
    for kk in 1:N, n in 1:35
        (i,j,k)=TRIP[n]; Φ[n,kk]=Xs[kk][1]^i*Xs[kk][2]^j*Xs[kk][3]^k
    end
    Φ .* rowsc
end
# inner: exact NNLS (column-normalized for conditioning); returns (residual, weights)
function inner(Xs)
    Φr=Phimat(Xs); cn=[max(norm(@view Φr[:,j]),1e-300) for j in 1:N]; Φn=Φr./cn'
    u=nonneg_lsq(Φn, br; alg=:nnls)[:,1]; w=u./cn
    (norm(Φr*w .- br), w)
end
objf(Yv) = inner(positions(Yv))[1]

function init_Y(seed)
    Random.seed!(seed); Y=zeros(3N); ntail=max(4,N÷5); nbulk=N-ntail
    for kk in 1:nbulk
        X=Xof(σS*randn()*0.9, σT*randn(), σT*randn()); Y[3(kk-1)+1:3kk]=[X[1],X[2],X[3]]./Lp
    end
    for t in 1:ntail
        Sl = -47.0 + 10.0*randn() - (t<=2 ? 18.0*rand() : 0.0)      # cluster near -47 + a couple intermediate
        X=Xof(Sl, σT*randn()*0.9, σT*randn()*0.9); Y[3(nbulk+t-1)+1:3(nbulk+t)]=[X[1],X[2],X[3]]./Lp
    end
    Y
end

best=(Inf,nothing)
for seed in 1:NSEED
    Y0=init_Y(seed)
    res=optimize(objf, Y0, NelderMead(), Optim.Options(iterations=OITER, g_tol=1e-12))
    Yb=Optim.minimizer(res); fb=Optim.minimum(res)
    # restart NelderMead once from the found point (helps escape simplex collapse)
    res2=optimize(objf, Yb, NelderMead(), Optim.Options(iterations=OITER, g_tol=1e-12))
    if Optim.minimum(res2) < fb; Yb=Optim.minimizer(res2); fb=Optim.minimum(res2); end
    @printf("[seed %d] varpro NNLS residual (row-scaled) = %.3e\n", seed, fb); flush(stdout)
    if fb < best[1]; global best=(fb, Yb); end
end

Xs=positions(best[2]); _,w=inner(Xs)
function verify_bf(w,Xs)
    wB=big.(w); mx=big(0.0); wr=0
    for n in 1:35
        (i,j,k)=TRIP[n]; val=sum(wB[kk]*big(Xs[kk][1])^i*big(Xs[kk][2])^j*big(Xs[kk][3])^k for kk in 1:length(w))
        e=abs(val-sstdB[n]); e>mx && (mx=e; wr=n)
    end
    (Float64(mx), wr)
end
maxerr,worst=verify_bf(w,Xs); cflmax=maximum(cflval(collect(Xs[kk])) for kk in 1:N)
@printf("\nBEST varpro: N=%d | BigFloat max std-moment resid=%.3e (worst %s) | CFL max=%.4f | sumw=%.4f minw=%.3e nnz=%d\n",
        N, maxerr, TRIP[worst], cflmax, sum(w), minimum(w), count(>(1e-10),w))
if maxerr < 1e-6 && cflmax <= 1.0+1e-9
    println("==> CONSTRUCTIVE CFL-SAFE POSITIVE CUBATURE FOUND (variable projection, BigFloat-verified).")
    save("gpu/validation/kfvs_varpro_solution.jld2","nodes_std",[collect(Xs[kk]) for kk in 1:N],
         "weights",w,"mean",collect(mean),"sig",collect(sig),"R",R,"maxerr",maxerr)
else
    @printf("==> best residual %.2e on %s — raise N/seeds/iters.\n", maxerr, TRIP[worst])
end
