# kfvs_column_generation.jl — reviewer step 6: constructive CFL-safe positive cubature by
# LP column generation. Membership of the target moment vector in cone{ phi(X) : X in K_CFL }.
#
#   Phase-1 LP (HiGHS simplex, exact):  min sum(sp+sn)  s.t.  Phi_r W + sp - sn = br, W,sp,sn>=0
#     obj -> 0  => feasible: exact positive cubature over current nodes.
#   Pricing: duals y of the equalities define p(X)=sum_n y_n rowsc_n X^alpha_n; a new column with
#     moment vector phi_r(X) improves iff reduced cost -p(X) < 0, i.e. p(X) > 0.  So MAXIMIZE p(X)
#     over the CFL polytope K (nonconvex quartic) by multistart projected-gradient ascent, add the
#     best node(s), repeat.  If max p(X) <= tol while obj>0 -> genuine obstruction over K.
# Verify the final cubature reproduces all 35 standardized moments in BigFloat.
#
# usage: julia ... kfvs_column_generation.jl [maxit=60] [nstart=4000]

using JLD2, Printf, LinearAlgebra, JuMP, HiGHS, Random
setprecision(BigFloat, 256); Random.seed!(7)
MAXIT  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 60
NSTART = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4000

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
br = sstd .* rowsc

# moment feature vector phi_r(X) (row-scaled), and its gradient
phir(X) = [rowsc[n]*X[1]^TRIP[n][1]*X[2]^TRIP[n][2]*X[3]^TRIP[n][3] for n in 1:35]
cflval(X) = LAM*(abs(mean[1]+sig[1]*X[1])+abs(mean[2]+sig[2]*X[2])+abs(mean[3]+sig[3]*X[3]))
cflok(X) = cflval(X) <= 1.0
# p(X)=sum y_n rowsc_n X^a ; grad wrt X
function pval_grad(X, y)
    p = 0.0; g = zeros(3)
    for n in 1:35
        a = TRIP[n]; c = y[n]*rowsc[n]
        xp = X[1]^a[1]*X[2]^a[2]*X[3]^a[3]; p += c*xp
        for d in 1:3
            if a[d] > 0
                gd = a[d]*X[1]^(a[1]-(d==1)) * X[2]^(a[2]-(d==2)) * X[3]^(a[3]-(d==3))
                g[d] += c*gd
            end
        end
    end
    p, g
end
# project X to CFL polytope by scaling deviation toward mean (v = mean+sig.*X ; shrink X if outside)
function projK(X)
    c = cflval(X)
    c <= 1.0 && return X
    # scale X by factor s<1 so that lambda*sum|mean+sig*s*X| <= 1 (mean itself is inside)
    s = 1.0
    for _ in 1:60
        Xs = s .* X
        cflval(Xs) <= 1.0 && break
        s *= 0.9
    end
    s .* X
end

# ---- initial node set: jet-aligned bulk + one-sided tail (feasible start columns) ----
eS=(1/sqrt(3),1/sqrt(3),1/sqrt(3)); eT1=(1/sqrt(2),-1/sqrt(2),0.0); eT2=(1/sqrt(6),1/sqrt(6),-2/sqrt(6))
Xof(S,T1,T2)=(S*eS[1]+T1*eT1[1]+T2*eT2[1], S*eS[2]+T1*eT1[2]+T2*eT2[2], S*eS[3]+T1*eT1[3]+T2*eT2[3])
σS=sqrt(1.336); σT=sqrt(0.832)
nodes = Vector{NTuple{3,Float64}}()
for hs in -2.0:1.0:2.0, h1 in -2.0:1.0:2.0, h2 in -2.0:1.0:2.0
    X=Xof(σS*hs,σT*h1,σT*h2); cflok(X)&&push!(nodes,X)
end
for Sl in -60.0:6.0:-6.0, t1 in (-1.0,0.0,1.0), t2 in (-1.0,0.0,1.0)
    X=Xof(Sl,σT*t1,σT*t2); cflok(X)&&push!(nodes,X)
end
@printf("init nodes=%d  (bulk+tail seed)\n", length(nodes)); flush(stdout)

# ---- pricing: maximize p(X)=y.phi_r(X) over K by multistart projected-grad ascent ----
function price(y)
    best=(-Inf, (0.0,0.0,0.0))
    # structured starts along +/- jet axis and random in a generous box intersect K
    starts = Vector{NTuple{3,Float64}}()
    for Sl in range(-75,20;length=60), tt in ((0,0),(1,0),(0,1),(-1,0),(1,1))
        push!(starts, Xof(Sl, σT*tt[1], σT*tt[2]))
    end
    for _ in 1:NSTART
        push!(starts, ntuple(_->(rand()-0.5)*140, 3))     # box ~|X|<=70
    end
    for X0 in starts
        X = projK(collect(X0));
        η = 1.0
        for it in 1:80
            p,g = pval_grad(X, y); ng = norm(g)
            ng < 1e-14 && break
            Xn = projK(X .+ (η/ng).*g)
            pn,_ = pval_grad(Xn, y)
            if pn > p; X = Xn; else; η *= 0.5; end
            η < 1e-8 && break
        end
        p,_ = pval_grad(X, y)
        if p > best[1] && cflok(X); best=(p, (X[1],X[2],X[3])); end
    end
    best
end

# ---- column generation loop ----
sol_w = Float64[]; sol_nodes = NTuple{3,Float64}[]; feasible=false
for iter in 1:MAXIT
    Nn = length(nodes)
    Φr = Array{Float64}(undef, 35, Nn)
    for (jn,X) in enumerate(nodes); Φr[:,jn]=phir(X); end
    lp = Model(HiGHS.Optimizer); set_silent(lp); set_attribute(lp,"solver","simplex")
    @variable(lp, W[1:Nn] >= 0); @variable(lp, sp[1:35] >= 0); @variable(lp, sn[1:35] >= 0)
    @constraint(lp, con, Φr*W .+ sp .- sn .== br)
    @objective(lp, Min, sum(sp)+sum(sn))
    optimize!(lp)
    obj = objective_value(lp); y = dual.(con)
    if obj < 1e-9
        global feasible = true; global sol_w = value.(W); global sol_nodes = copy(nodes)
        @printf("[it %d] Nn=%d phase1-obj=%.3e  => FEASIBLE\n", iter, Nn, obj); flush(stdout); break
    end
    pbest, Xstar = price(y)
    @printf("[it %d] Nn=%d phase1-obj=%.3e  best reduced (max p)=%.3e at X=(%.2f,%.2f,%.2f)\n",
            iter, Nn, obj, pbest, Xstar...); flush(stdout)
    if pbest <= 1e-7
        @printf("  no improving column (max p=%.2e) while obj=%.2e -> OBSTRUCTION over K.\n", pbest, obj)
        flush(stdout); break
    end
    push!(nodes, Xstar)
end

if feasible
    act = findall(>(1e-12), sol_w)
    wB=big.(sol_w); maxerr=big(0.0); worst=0
    for n in 1:35
        (i,j,k)=TRIP[n]; val=sum(wB[a]*big(sol_nodes[a][1])^i*big(sol_nodes[a][2])^j*big(sol_nodes[a][3])^k for a in act)
        e=abs(val-sstdB[n]); e>maxerr && (maxerr=e; worst=n)
    end
    cflmax = maximum(cflval(sol_nodes[a]) for a in act)
    @printf("FEASIBLE: %d active atoms | BigFloat max std-moment resid=%.3e (worst %s) | CFL max=%.4f | minw=%.3e\n",
            length(act), Float64(maxerr), TRIP[worst], cflmax, minimum(sol_w[act])); flush(stdout)
    if Float64(maxerr) < 1e-6 && cflmax <= 1.0+1e-9
        println("==> CONSTRUCTIVE CFL-SAFE POSITIVE CUBATURE FOUND (column generation, BigFloat-verified).")
        save("gpu/validation/kfvs_colgen_solution.jld2",
             "nodes_std",[collect(sol_nodes[a]) for a in act],"weights",sol_w[act],
             "mean",collect(mean),"sig",collect(sig),"R",R,"maxerr",Float64(maxerr))
    else
        println("==> feasible LP but BigFloat residual/CFL imperfect — tighten pricing / precision.")
    end
else
    println("==> not feasible within iteration budget (see obstruction note above or raise maxit).")
end
