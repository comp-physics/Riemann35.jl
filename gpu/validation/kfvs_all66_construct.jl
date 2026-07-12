# kfvs_all66_construct.jl — PHASE 1 of the all-66 generalization experiment.
# For every class-C cell in the clean stage-2 artifact, construct a bounded positive CFL-safe
# cubature by the jet-aligned conditional lift, with a PER-CELL principal tail direction (chosen
# by maximizing directional kurtosis). Save each cubature + diagnostics. (Donor-repair = phase 2.)
#
# Per cell records: success, #atoms, max CFL coord, min weight, standardized residual (BigFloat),
# principal-tail direction, conditional t*, solve time.  Env: sdp_env (JuMP+Hypatia+HiGHS).
using JLD2, Printf, LinearAlgebra, JuMP, Hypatia, HiGHS, Random
setprecision(BigFloat, 512); Random.seed!(20260712)
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX=Dict(TRIP[n]=>n for n in 1:35); bino(n,k)=(k<0||k>n) ? big(0) : big(binomial(n,k))
ce=load("gpu/validation/kfvs_defect_counterexample.jld2"); Cst=ce["center_state"]; cls=ce["class"]; λ=ce["lam"][1]; R=1/λ
Ccells=findall(==(3),cls); @printf("class-C cells: %d\n",length(Ccells)); flush(stdout)

# standardized central moments of a raw 35-vector (BigFloat)
function standardize(Mraw)
    M=big.(Mraw); ρ=M[1]; ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
    σx=sqrt(M[3]/ρ-ux^2);σy=sqrt(M[10]/ρ-uy^2);σz=sqrt(M[20]/ρ-uz^2)
    s=Dict{NTuple{3,Int},BigFloat}()
    for n in 1:35; (i,j,k)=TRIP[n]; acc=big(0.0)
        for p in 0:i,q in 0:j,r in 0:k; haskey(IDX,(p,q,r))||continue
            acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ); end
        s[TRIP[n]]=acc/(σx^i*σy^j*σz^k); end
    (s,(Float64(ux),Float64(uy),Float64(uz)),(Float64(σx),Float64(σy),Float64(σz)))
end
smf(s,t)=get(s,t,big(0.0))
# directional 4th standardized moment E[(l.X)^4] via linear-form expansion
function dirkurt(s,l)
    P=Dict((0,0,0)=>big(1.0))
    for _ in 1:4; np=Dict{NTuple{3,Int},BigFloat}()
        for (e,co) in P, d in 1:3; e2=(e[1]+(d==1),e[2]+(d==2),e[3]+(d==3)); np[e2]=get(np,e2,big(0.0))+co*big(l[d]); end; P=np; end
    m4=sum(co*smf(s,e) for (e,co) in P)
    # variance of l.X = sum_{ij} l_i l_j E[X_i X_j] (standardized: E[X_i^2]=1)
    v=big(0.0); comp=((2,0,0),(0,2,0),(0,0,2)); cross=((1,1,0),(1,0,1),(0,1,1))
    for d in 1:3; v+=big(l[d])^2*smf(s,comp[d]); end
    v+=2*(big(l[1])*big(l[2])*smf(s,cross[1])+big(l[1])*big(l[3])*smf(s,cross[2])+big(l[2])*big(l[3])*smf(s,cross[3]))
    m4/v^2
end
# principal direction: maximize directional kurtosis over a sphere sample
function principal_dir(s)
    best=(-Inf,[1.0,1.0,1.0]./sqrt(3))
    dirs=[[1.,1,1]./sqrt(3),[1.,-1,1]./sqrt(3),[1.,1,-1]./sqrt(3),[1.,-1,-1]./sqrt(3)]
    for _ in 1:300; g=randn(3); push!(dirs,g./norm(g)); end
    for l in dirs; k=Float64(dirkurt(s,l)); k>best[1] && (best=(k,l)); end
    best[2]
end
orthobasis(eS)= begin
    a = abs(eS[1])<0.9 ? [1.0,0,0] : [0.0,1,0]
    t1 = a .- (a'eS).*eS; t1 ./= norm(t1); t2 = cross(eS,t1); (Tuple(eS),Tuple(t1),Tuple(t2))
end

mon2(d)=sort([(a,b) for a in 0:d for b in 0:d if a+b<=d]; by=m->(sum(m),m))
const B2=mon2(2); const B1=mon2(1); const ALL4=mon2(4); addt2(p,q)=(p[1]+q[1],p[2]+q[2])
polymul_lin(P,lc)= begin np=Dict{NTuple{3,Int},BigFloat}()
    for (e,co) in P, d in 1:3; e2=(e[1]+(d==1),e[2]+(d==2),e[3]+(d==3)); np[e2]=get(np,e2,big(0.0))+co*big(lc[d]); end; np; end
function rotmoms(s,eS,eT1,eT2)
    mr=Dict{NTuple{3,Int},BigFloat}()
    for k in 0:4,a in 0:4,b in 0:4; (k+a+b<=4) || continue
        P=Dict((0,0,0)=>big(1.0)); for _ in 1:k; P=polymul_lin(P,eS); end; for _ in 1:a; P=polymul_lin(P,eT1); end; for _ in 1:b; P=polymul_lin(P,eT2); end
        mr[(k,a,b)]=sum(co*smf(s,e) for (e,co) in P); end
    mr
end
cflS(s,mu,sig,eS)=λ*(abs(mu[1]+sig[1]*s*eS[1])+abs(mu[2]+sig[2]*s*eS[2])+abs(mu[3]+sig[3]*s*eS[3]))
function facets(s,mu,sig,eS,eT1,eT2)
    fs=NTuple{3,Float64}[]
    for sg in Iterators.product((-1,1),(-1,1),(-1,1))
        g0=R-sum(sg[j]*(mu[j]+sig[j]*s*eS[j]) for j in 1:3); g1=-sum(sg[j]*sig[j]*eT1[j] for j in 1:3); g2=-sum(sg[j]*sig[j]*eT2[j] for j in 1:3)
        push!(fs,(g0,g1,g2)); end
    fs
end
# positive quadrature weights for given nodes via LP (min sum w s.t. Vw=m, w>=0); nothing if none
function pos_weights(sv,mvec)
    n=length(sv); V=[sv[j]^k for k in 0:4, j in 1:n]
    lp=Model(HiGHS.Optimizer); set_silent(lp); set_attribute(lp,"solver","simplex")
    @variable(lp,wv[1:n]>=0); @constraint(lp,V*wv.==mvec); @objective(lp,Min,sum(wv)); optimize!(lp)
    has_values(lp) || return nothing
    w=value.(wv)
    (maximum(abs(sum(w[j]*sv[j]^k for j in 1:n)-mvec[k+1]) for k in 0:4)<1e-6 && all(>=(-1e-10),w)) ? max.(w,0.0) : nothing
end
# n-node marginal generator: sample positions (2 tails + bulk + intermediates), LP weights, positive+CFL
function nnode_members(mS,mu,sig,eS,nnode; ntry=3000, want=50)
    mem=Vector{Tuple{Vector{Float64},Vector{Float64}}}(); mvec=[mS[k+1] for k in 0:4]
    supp=sqrt(max(mS[5]/mS[3],1.0)); t=0
    while length(mem)<want && t<ntry; t+=1
        pts=Float64[]; push!(pts,-(0.85+0.6rand())*supp); push!(pts,-(0.45+0.4rand())*supp); push!(pts,-1.5+3rand())
        for _ in 1:(nnode-3); push!(pts,-0.7*supp+1.4*supp*rand()); end
        sv=sort(pts); (minimum(diff(sv))<0.8) && continue
        all(x->cflS(x,mu,sig,eS)<=1.0,sv) || continue
        w=pos_weights(sv,mvec); w===nothing && continue
        push!(mem,(sv,w)); end
    mem
end
function inner_sdp(svec,wvec,mr,mu,sig,eS,eT1,eT2)
    Nn=length(svec); m=Model(Hypatia.Optimizer); set_silent(m)
    cvar=[Dict{NTuple{2,Int},Any}() for _ in 1:Nn]
    for i in 1:Nn, ab in ALL4; cvar[i][ab]= ab==(0,0) ? 1.0 : @variable(m); end
    cf(i,ab)=convert(AffExpr,cvar[i][ab]); @variable(m,t)
    for k in 0:4, ab in ALL4; (ab==(0,0)||k+ab[1]+ab[2]>4)&&continue
        @constraint(m,sum(wvec[i]*svec[i]^k*cf(i,ab) for i in 1:Nn)==Float64(mr[(k,ab[1],ab[2])])); end
    for i in 1:Nn
        M2=[cf(i,addt2(B2[p],B2[q])) for p in eachindex(B2),q in eachindex(B2)]; @constraint(m,Symmetric(M2 .- t*Matrix(I,6,6)) in PSDCone())
        for (g0,g1,g2) in facets(svec[i],mu,sig,eS,eT1,eT2)
            L=[g0*cf(i,addt2(B1[p],B1[q]))+g1*cf(i,addt2(addt2(B1[p],B1[q]),(1,0)))+g2*cf(i,addt2(addt2(B1[p],B1[q]),(0,1))) for p in eachindex(B1),q in eachindex(B1)]
            @constraint(m,Symmetric(L) in PSDCone()); end; end
    @objective(m,Max,t); optimize!(m)
    has_values(m) ? (value(t), [Dict(ab=>(ab==(0,0) ? 1.0 : value(cvar[i][ab])) for ab in ALL4) for i in 1:Nn]) : (-Inf,nothing)
end
function extract2d(s,cd,mu,sig,eS,eT1,eT2)
    fs=facets(s,mu,sig,eS,eT1,eT2); inside(a,b)=all(g->g[1]+g[2]*a+g[3]*b>=-1e-9,fs)
    gr=range(-8,8;length=81); nd=NTuple{2,Float64}[]; for a in gr,b in gr; inside(a,b)&&push!(nd,(a,b)); end
    isempty(nd) && return nothing
    Φ=[nd[j][1]^ab[1]*nd[j][2]^ab[2] for ab in ALL4,j in eachindex(nd)]; tgt=[cd[ab] for ab in ALL4]
    rs=[1.0/max(abs(tgt[i]),1e-3) for i in eachindex(tgt)]; Φr=Φ.*rs; br=tgt.*rs
    cn=[max(norm(@view Φr[:,j]),1e-300) for j in 1:length(nd)]; Φn=Φr./cn'
    lp=Model(HiGHS.Optimizer); set_silent(lp); set_attribute(lp,"solver","simplex")
    @variable(lp,u[1:length(nd)]>=0); @constraint(lp,Φn*u.==br); @objective(lp,Min,sum(u)); optimize!(lp)
    has_values(lp) || return nothing
    w=max.(value.(u),0.0)./cn; act=findall(>(1e-10),w)
    # VERIFY the extracted 2D atoms actually match the conditional moments (else reject -> clean fail)
    rel=maximum(abs(sum(w[j]*nd[j][1]^ab[1]*nd[j][2]^ab[2] for j in act)-cd[ab])*rs[i] for (i,ab) in enumerate(ALL4))
    rel>1e-6 && return nothing
    ([nd[j] for j in act],w[act])
end

results=[]; cubatures=Dict{Int,Any}()
for (ii,ccol) in enumerate(Ccells)
    tstart=time()
    Mraw=Cst[:,ccol]; sd,mu,sig=standardize(Mraw); ρ=Mraw[1]
    eSv=principal_dir(sd); eS,eT1,eT2=orthobasis(eSv)
    mr=rotmoms(sd,eS,eT1,eT2); mS=[Float64(mr[(k,0,0)]) for k in 0:4]
    tstar=-Inf; cbest=nothing; nnode_used=5
    for nnode in (5,7)                                    # escalate 5 -> 7 S-nodes if 5 fails
        mem=nnode_members(mS,mu,sig,eS,nnode; want=(nnode==5 ? 40 : 25))
        for (sv,wv) in mem
            tv,cv=inner_sdp(sv,wv,mr,mu,sig,eS,eT1,eT2)
            if tv>tstar && cv!==nothing; tstar=tv; cbest=(sv,wv,cv); nnode_used=nnode; end
            tstar>0.15 && break
        end
        tstar>0.05 && break
    end
    if cbest===nothing || tstar<=1e-6
        push!(results,(cell=ccol,ok=false,reason="conditional infeasible (5+7 node)",tstar=tstar,dir=eSv,t=time()-tstart));
        @printf("[%2d/%d] col%d FAIL conditional t*=%.2e dir=(%.2f,%.2f,%.2f)\n",ii,length(Ccells),ccol,tstar,eSv...); flush(stdout); continue
    end
    sv,wv,cv=cbest; atomsX=NTuple{3,Float64}[]; W=Float64[]
    okext=true
    for i in eachindex(sv); r=extract2d(sv[i],cv[i],mu,sig,eS,eT1,eT2); r===nothing && (okext=false;break)
        Ts,wp=r; for q in eachindex(Ts); T1,T2=Ts[q]
            X=(sv[i]*eS[1]+T1*eT1[1]+T2*eT2[1], sv[i]*eS[2]+T1*eT1[2]+T2*eT2[2], sv[i]*eS[3]+T1*eT1[3]+T2*eT2[3])
            push!(atomsX,X); push!(W,wv[i]*wp[q]); end; end
    if !okext; push!(results,(cell=ccol,ok=false,reason="2D extraction empty slice",tstar=tstar,dir=eSv,t=time()-tstart))
        @printf("[%2d/%d] col%d FAIL extraction\n",ii,length(Ccells),ccol); flush(stdout); continue; end
    # polish weights (BigFloat min-norm, nodes fixed)
    nA=length(W); A=Array{BigFloat}(undef,35,nA)
    for j in 1:nA,n in 1:35; (i,a,b)=TRIP[n]; A[n,j]=big(atomsX[j][1])^i*big(atomsX[j][2])^a*big(atomsX[j][3])^b; end
    tgt=[sd[TRIP[n]] for n in 1:35]; wB=big.(W); r0=tgt.-A*wB; dw=A'*((A*A')\r0)
    wpol=wB.+dw; (minimum(wpol)>=0) && (wB=wpol)        # guard: only accept polish if it stays positive
    resid=Float64(maximum(abs,tgt.-A*wB)); minw=Float64(minimum(wB))
    cfl=maximum(λ*(abs(mu[1]+sig[1]*X[1])+abs(mu[2]+sig[2]*X[2])+abs(mu[3]+sig[3]*X[3])) for X in atomsX)
    ok = resid<1e-6 && minw>=0 && cfl<=1+1e-9
    cubatures[ccol]=(atomsX=[collect(X) for X in atomsX], weightsBF=wB, weights=Float64.(wB), mu=collect(mu), sig=collect(sig))
    push!(results,(cell=ccol,ok=ok,reason=ok ? "OK" : "polish resid/pos/cfl",tstar=tstar,natoms=nA,resid=resid,minw=minw,cfl=cfl,dir=eSv,t=time()-tstart))
    @printf("[%2d/%d] col%d %s natoms=%d t*=%.3f resid=%.1e minw=%.1e cfl=%.3f dir=(%.2f,%.2f,%.2f) %.1fs\n",
            ii,length(Ccells),ccol, ok ? "OK  " : "WARN", nA,tstar,resid,minw,cfl,eSv...,time()-tstart); flush(stdout)
end
nok=count(r->r.ok,results)
@printf("\n=== PHASE 1 COMPLETE: %d/%d class-C cells constructed OK ===\n",nok,length(Ccells))
save("gpu/validation/kfvs_all66_cubatures.jld2","cubatures",cubatures,
     "cells_ok",[r.cell for r in results if r.ok],"cells_fail",[r.cell for r in results if !r.ok],
     "resid",[get(r,:resid,NaN) for r in results],"natoms",[get(r,:natoms,0) for r in results],
     "cfl",[get(r,:cfl,NaN) for r in results],"tstar",[r.tstar for r in results])
@printf("worst std residual over OK cells: %.2e ; atom counts: %s\n",
        maximum([r.resid for r in results if r.ok]; init=0.0), string(sort(unique([r.natoms for r in results if r.ok]))))
