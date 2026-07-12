# kfvs_assemble.jl — final constructive step: from the feasible 5-node conditional lift, extract
# each node's 2D conditional measure, map to (s_i,T1,T2)->standardized X->physical v, assemble the
# full 3D cubature (weight w_i * w_perp), verify ALL 35 joint standardized moments in BigFloat.
#
# Re-solve the conditional SDP for the saved best member to obtain c^{(i)}_{ab}; for each S-node fit
# ~4-6 transverse atoms by 2D NNLS on the CFL-slice grid (transverse moments are O(1) -> clean).

using JLD2, Printf, LinearAlgebra, JuMP, Hypatia, HiGHS
setprecision(BigFloat, 512)
const D=load("gpu/validation/kfvs_defect_counterexample.jld2")
const LAM=D["lam"][1]; const C=D["center_state"]; const CLS=D["class"]
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX=Dict(TRIP[n]=>n for n in 1:35); bino(n,k)=(k<0||k>n) ? big(0) : big(binomial(n,k))
s0i=findfirst(==(3),CLS); M=big.(collect(C[:,s0i])); ρ=M[1]
ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
σx=sqrt(M[3]/ρ-ux^2);σy=sqrt(M[10]/ρ-uy^2);σz=sqrt(M[20]/ρ-uz^2)
sstd=Dict{NTuple{3,Int},BigFloat}()
for n in 1:35
    (i,j,k)=TRIP[n]; acc=big(0.0)
    for p in 0:i,q in 0:j,r in 0:k; haskey(IDX,(p,q,r))||continue
        acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ); end
    sstd[TRIP[n]]=acc/(σx^i*σy^j*σz^k)
end
smf(t)=get(sstd,t,big(0.0)); sstdF=[Float64(sstd[TRIP[n]]) for n in 1:35]
R=1/LAM; mu=(Float64(ux),Float64(uy),Float64(uz)); sig=(Float64(σx),Float64(σy),Float64(σz))
s3=sqrt(big(3));s2=sqrt(big(2));s6=sqrt(big(6))
eS=(1/s3,1/s3,1/s3); eT1=(1/s2,-1/s2,big(0)); eT2=(1/s6,1/s6,-2/s6)
eSf=Float64.(eS); eT1f=Float64.(eT1); eT2f=Float64.(eT2)

# rotated moments m_{kab}
function polymul_lin!(P,lc)
    newP=Dict{NTuple{3,Int},BigFloat}()
    for (e,co) in P, d in 1:3; lc[d]==0 && continue
        e2=(e[1]+(d==1),e[2]+(d==2),e[3]+(d==3)); newP[e2]=get(newP,e2,big(0.0))+co*lc[d]; end
    newP
end
function rotmom(k,a,b)
    P=Dict((0,0,0)=>big(1.0))
    for _ in 1:k; P=polymul_lin!(P,eS); end
    for _ in 1:a; P=polymul_lin!(P,eT1); end
    for _ in 1:b; P=polymul_lin!(P,eT2); end
    sum(co*smf(e) for (e,co) in P)
end
mrot=Dict{NTuple{3,Int},BigFloat}(); for k in 0:4,a in 0:4,b in 0:4; (k+a+b<=4)&&(mrot[(k,a,b)]=rotmom(k,a,b)); end

bm=load("gpu/validation/kfvs_conditional_best.jld2"); svec=bm["svec"]; wvec=bm["wvec"]; Nn=length(svec)
@printf("best member: nodes=%s weights=%s t*=%.3f\n",
        join((@sprintf("%.2f",x) for x in svec),","), join((@sprintf("%.1e",x) for x in wvec),","), bm["tstar"])

mon2(d)=sort([(a,b) for a in 0:d for b in 0:d if a+b<=d]; by=m->(sum(m),m))
B2=mon2(2);B1=mon2(1);ALL4=mon2(4); addt2(p,q)=(p[1]+q[1],p[2]+q[2])
function facets(s)
    fs=NTuple{3,Float64}[]
    for sg in Iterators.product((-1,1),(-1,1),(-1,1))
        g0=Float64(R)-sum(sg[j]*(mu[j]+sig[j]*Float64(s)*eSf[j]) for j in 1:3)
        g1=-sum(sg[j]*sig[j]*eT1f[j] for j in 1:3); g2=-sum(sg[j]*sig[j]*eT2f[j] for j in 1:3)
        push!(fs,(g0,g1,g2)); end
    fs
end
# re-solve SDP, return conditional moments c[i][(a,b)]
function solve_conditional()
    m=Model(Hypatia.Optimizer); set_silent(m)
    cvar=[Dict{NTuple{2,Int},Any}() for _ in 1:Nn]
    for i in 1:Nn, ab in ALL4; cvar[i][ab]= ab==(0,0) ? 1.0 : @variable(m,base_name="c$(i)_$(ab[1])_$(ab[2])"); end
    cf(i,ab)=convert(AffExpr,cvar[i][ab]); @variable(m,t)
    for k in 0:4, ab in ALL4
        (ab==(0,0)||k+ab[1]+ab[2]>4) && continue
        @constraint(m, sum(wvec[i]*svec[i]^k*cf(i,ab) for i in 1:Nn)==Float64(mrot[(k,ab[1],ab[2])])); end
    for i in 1:Nn
        M2=[cf(i,addt2(B2[p],B2[q])) for p in eachindex(B2),q in eachindex(B2)]
        @constraint(m,Symmetric(M2 .- t*Matrix(I,length(B2),length(B2))) in PSDCone())
        for (g0,g1,g2) in facets(svec[i])
            L=[g0*cf(i,addt2(B1[p],B1[q]))+g1*cf(i,addt2(addt2(B1[p],B1[q]),(1,0)))+
               g2*cf(i,addt2(addt2(B1[p],B1[q]),(0,1))) for p in eachindex(B1),q in eachindex(B1)]
            @constraint(m,Symmetric(L) in PSDCone()); end; end
    @objective(m,Max,t); optimize!(m)
    [Dict(ab=>(ab==(0,0) ? 1.0 : value(cvar[i][ab])) for ab in ALL4) for i in 1:Nn]
end
cval=solve_conditional()

# per-node 2D NNLS on CFL-slice grid -> transverse atoms
function extract2d(s, cd)
    fs=facets(s); inslice(T1,T2)=all(g->g[1]+g[2]*T1+g[3]*T2>=-1e-9, fs)
    gr=range(-5.0,5.0;length=61); nodes=NTuple{2,Float64}[]
    for a in gr, b in gr; inslice(a,b)&&push!(nodes,(a,b)); end
    isempty(nodes) && return nothing
    Φ=[nodes[jn][1]^ab[1]*nodes[jn][2]^ab[2] for ab in ALL4, jn in eachindex(nodes)]
    tgt=[cd[ab] for ab in ALL4]; rs=[1.0/max(abs(tgt[m2]),1e-3) for m2 in eachindex(tgt)]
    Φr=Φ.*rs; br=tgt.*rs; cn=[max(norm(@view Φr[:,j]),1e-300) for j in 1:length(nodes)]
    Φn=Φr./cn'
    lp=Model(HiGHS.Optimizer); set_silent(lp); set_attribute(lp,"solver","simplex")
    @variable(lp,u[1:length(nodes)]>=0); @constraint(lp,Φn*u.==br); @objective(lp,Min,sum(u))
    optimize!(lp)
    has_values(lp) || return nothing
    w=max.(value.(u),0.0)./cn; act=findall(>(1e-10),w)
    ([nodes[j] for j in act], w[act])
end

gAtoms=NTuple{3,Float64}[]; gW=Float64[]
for i in 1:Nn
    r=extract2d(svec[i],cval[i]); r===nothing && (@printf("node %d: no 2D extraction\n",i); continue)
    Ts,wp=r
    for q in eachindex(Ts)
        T1,T2=Ts[q]
        X=(svec[i]*eSf[1]+T1*eT1f[1]+T2*eT2f[1], svec[i]*eSf[2]+T1*eT1f[2]+T2*eT2f[2], svec[i]*eSf[3]+T1*eT1f[3]+T2*eT2f[3])
        push!(gAtoms,X); push!(gW, wvec[i]*wp[q])
    end
    @printf("node %d (S=%.1f,w=%.1e): %d transverse atoms\n", i, svec[i], wvec[i], length(Ts))
end
@printf("total assembled atoms: %d  sumW=%.6f\n", length(gAtoms), sum(gW))

# BigFloat verify all 35 joint standardized moments
function verify_joint(gW,gAtoms)
    wB=big.(gW); maxerr=big(0.0); worst=0
    for n in 1:35
        (i,j,k)=TRIP[n]; val=sum(wB[a]*big(gAtoms[a][1])^i*big(gAtoms[a][2])^j*big(gAtoms[a][3])^k for a in eachindex(gAtoms))
        e=abs(val-sstd[TRIP[n]]); e>maxerr && (maxerr=e; worst=n)
    end
    (maxerr,worst)
end
maxerr,worst=verify_joint(gW,gAtoms)
cflmax=maximum(Float64(LAM)*(abs(mu[1]+sig[1]*X[1])+abs(mu[2]+sig[2]*X[2])+abs(mu[3]+sig[3]*X[3])) for X in gAtoms)
@printf("\nVERIFY(BigFloat): max joint std-moment resid=%.3e (worst %s) | CFL λΣ|v|max=%.4f | minW=%.3e\n",
        Float64(maxerr), TRIP[worst], cflmax, minimum(gW))
if Float64(maxerr)<1e-6 && cflmax<=1.0+1e-9 && minimum(gW)>=-1e-12
    println("==> CONSTRUCTIVE CFL-SAFE POSITIVE CUBATURE ASSEMBLED (conditional lift, BigFloat-verified).")
    save("gpu/validation/kfvs_assembled_solution.jld2","atoms_std",[collect(X) for X in gAtoms],"weights",gW,
         "mu",collect(mu),"sig",collect(sig),"R",R,"maxerr",Float64(maxerr),"natoms",length(gAtoms))
else
    @printf("==> assembled but resid %.2e on %s — refine 2D extraction (more grid / atomic solve).\n", Float64(maxerr), TRIP[worst])
end
