# kfvs_continuation_fit.jl — reviewer step 4: overcomplete NONLINEAR moment-fit cubature.
# Fixed N atoms; positions X_k and weights w_k=a_k^2>=0 moved by Levenberg-Marquardt to match all
# 35 standardized moments. Initialized from the jet-marginal structure (near-Gaussian bulk + a
# ONE-SIDED counter-stream at S~-47). Row-scaled residuals; positions rescaled Y=X/Lp for
# conditioning; CFL enforced by projection. Verified in BigFloat.
#
# usage: julia ... kfvs_continuation_fit.jl [N=24] [iters=400] [seedcount=6]

using JLD2, Printf, LinearAlgebra, Random
setprecision(BigFloat, 256)
N       = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 24
ITERS   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 400
NSEED   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 6

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
Ls = sqrt(sqrt(sstd[5])); rowsc = [1.0/Ls^sum(TRIP[n]) for n in 1:35]
const Lp = 10.0                                    # position rescale Y=X/Lp for conditioning
cflval(X) = LAM*(abs(mean[1]+sig[1]*X[1])+abs(mean[2]+sig[2]*X[2])+abs(mean[3]+sig[3]*X[3]))

eS=(1/sqrt(3),1/sqrt(3),1/sqrt(3)); eT1=(1/sqrt(2),-1/sqrt(2),0.0); eT2=(1/sqrt(6),1/sqrt(6),-2/sqrt(6))
Xof(S,T1,T2)=(S*eS[1]+T1*eT1[1]+T2*eT2[1], S*eS[2]+T1*eT1[2]+T2*eT2[2], S*eS[3]+T1*eT1[3]+T2*eT2[3])
σS=sqrt(1.336); σT=sqrt(0.832)

# params theta = [ Y(3N) ; a(N) ] ; X_k = Lp*Y_k ; w_k = a_k^2
unpack(θ) = (reshape(θ[1:3N],3,N), θ[3N+1:end])
function resid(θ)
    Y,a = unpack(θ); r = zeros(35)
    for n in 1:35
        (i,j,k)=TRIP[n]; s=0.0
        for kk in 1:N
            X=(Lp*Y[1,kk],Lp*Y[2,kk],Lp*Y[3,kk]); s += a[kk]^2 * X[1]^i*X[2]^j*X[3]^k
        end
        r[n] = (s - sstd[n])*rowsc[n]
    end
    r
end
function jac(θ)
    m=length(θ); J=zeros(35,m); h=1e-6; r0=resid(θ)
    for p in 1:m
        θ2=copy(θ); step=h*max(abs(θ[p]),1e-3); θ2[p]+=step
        J[:,p]=(resid(θ2).-r0)./step
    end
    J
end
function project!(θ)                               # keep atoms CFL-feasible
    Y,a = unpack(θ)
    for kk in 1:N
        X=[Lp*Y[1,kk],Lp*Y[2,kk],Lp*Y[3,kk]]
        if cflval(X) > 1.0
            s=1.0; for _ in 1:60; (cflval(s.*X)<=1.0)&&break; s*=0.9; end
            θ[3(kk-1)+1:3kk] = (s.*X)./Lp
        end
    end
    θ
end

function init_theta(seed)
    Random.seed!(seed)
    Y=zeros(3,N); a=zeros(N)
    ntail = max(3, N÷5)
    nbulk = N - ntail
    # bulk: near mean, small spread ~ marginal sigmas
    for kk in 1:nbulk
        S=σS*randn()*0.9; T1=σT*randn(); T2=σT*randn()
        X=Xof(S,T1,T2); Y[:,kk]=[X[1],X[2],X[3]]./Lp; a[kk]=sqrt((1-6e-4)/nbulk)
    end
    # tail: one-sided along -S near the marginal counter-stream, plus an intermediate for skew
    for t in 1:ntail
        Sl = -47.0 + 12.0*randn()*0.5 - (t==ntail ? 20.0 : 0.0)   # cluster near -47, one intermediate
        X=Xof(Sl, σT*randn()*0.8, σT*randn()*0.8); Y[:,nbulk+t]=[X[1],X[2],X[3]]./Lp
        a[nbulk+t]=sqrt(6e-4/ntail)
    end
    θ=vcat(vec(Y),a); project!(θ)
end

# ---- Levenberg-Marquardt, multi-seed ----
best=(Inf,nothing)
for seed in 1:NSEED
    θ=init_theta(seed); λ=1e-3; r=resid(θ); f=0.5*dot(r,r)
    for it in 1:ITERS
        J=jac(θ); g=J'*r; H=J'*J
        local θn, fn
        accepted=false
        for _ in 1:12
            δ = -(H+λ*(I+Diagonal(diag(H))))\g
            θn=project!(θ.+δ); rn=resid(θn); fn=0.5*dot(rn,rn)
            if fn < f; θ=θn; r=rn; f=fn; λ=max(λ*0.5,1e-10); accepted=true; break
            else; λ=min(λ*4,1e8); end
        end
        accepted || break
        f < 1e-16 && break
    end
    rms=sqrt(2f/35)
    @printf("[seed %d] final row-scaled RMS resid=%.3e  maxabs=%.3e\n", seed, rms, maximum(abs,r)); flush(stdout)
    if f < best[1]; global best=(f,copy(θ)); end
end

θ=best[2]; Y,a=unpack(θ); w=a.^2
Xs=[(Lp*Y[1,kk],Lp*Y[2,kk],Lp*Y[3,kk]) for kk in 1:N]
function verify_bf(w, Xs)
    wB=big.(w); maxerr=big(0.0); worst=0
    for n in 1:35
        (i,j,k)=TRIP[n]; val=sum(wB[kk]*big(Xs[kk][1])^i*big(Xs[kk][2])^j*big(Xs[kk][3])^k for kk in 1:length(w))
        e=abs(val-sstdB[n]); e>maxerr && (maxerr=e; worst=n)
    end
    (Float64(maxerr), worst)
end
maxerr, worst = verify_bf(w, Xs)
cflmax=maximum(cflval(collect(Xs[kk])) for kk in 1:N)
@printf("\nBEST: N=%d atoms | BigFloat max std-moment resid=%.3e (worst %s) | CFL max=%.4f | sumw=%.4f minw=%.3e\n",
        N, Float64(maxerr), TRIP[worst], cflmax, sum(w), minimum(w))
if Float64(maxerr) < 1e-6 && cflmax <= 1.0+1e-9 && minimum(w) >= -1e-12
    println("==> CONSTRUCTIVE CFL-SAFE POSITIVE CUBATURE FOUND (continuation moment-fit, BigFloat-verified).")
    save("gpu/validation/kfvs_continuation_solution.jld2",
         "nodes_std",[collect(Xs[kk]) for kk in 1:N],"weights",w,"mean",collect(mean),
         "sig",collect(sig),"R",R,"maxerr",Float64(maxerr))
else
    @printf("==> best residual %.2e on %s — raise N / iters / seeds or refine init.\n", Float64(maxerr), TRIP[worst])
end
