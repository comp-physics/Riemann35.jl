# kfvs_conditional_lift.jl — reviewer step 3: outer(marginal family) / inner(conditional SDP).
# For each member of the exact 3-node S-quadrature family (fixed s_i,w_i), solve the inner convex
# conditional-moment SDP in NORMALIZED conditional moments c^{(i)}_{ab}=E[T1^a T2^b | S=s_i]:
#   maximize t   s.t.   sum_i w_i s_i^k c^{(i)}_{ab} = m_{kab}  (k+a+b<=4, a+b>=1)
#                       M_2(c^{(i)}) >= t I               (6x6, per node)
#                       M_1(g_ij c^{(i)}) >= 0            (CFL-slice facets, 3x3)
#   c^{(i)}_{00}=1.  t*(theta)>0 => structured constructive route; everywhere<0 => need 5 S-nodes.
# Also report the pure linear equality least-squares residual per transverse degree (distinguishes
# marginal-incompatibility from PSD-infeasibility). Rotated moments m_{kab} from the 35 X-moments.
#
# usage: julia ... kfvs_conditional_lift.jl [nmember=25] [Snodes=3]

using JLD2, Printf, LinearAlgebra, JuMP, Hypatia
setprecision(BigFloat, 512)
NMEM = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 25
SNODES = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3

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
smf(t)=get(sstd,t,big(0.0))
R=1/LAM; mu=(Float64(ux),Float64(uy),Float64(uz)); sig=(Float64(σx),Float64(σy),Float64(σz))
s3=sqrt(big(3)); s2=sqrt(big(2)); s6=sqrt(big(6))
eS=(1/s3,1/s3,1/s3); eT1=(1/s2,-1/s2,big(0)); eT2=(1/s6,1/s6,-2/s6)

# ---- rotated moments m_{kab}=E[S^k T1^a T2^b], k+a+b<=4 (BigFloat, exact) ----
function polymul_lin!(P, lc)                       # multiply poly dict by linear form lc=(c1,c2,c3)
    newP=Dict{NTuple{3,Int},BigFloat}()
    for (e,co) in P, d in 1:3
        lc[d]==0 && continue
        e2=(e[1]+(d==1),e[2]+(d==2),e[3]+(d==3))
        newP[e2]=get(newP,e2,big(0.0))+co*lc[d]
    end
    newP
end
function rotmom(k,a,b)
    P=Dict((0,0,0)=>big(1.0))
    for _ in 1:k; P=polymul_lin!(P,eS); end
    for _ in 1:a; P=polymul_lin!(P,eT1); end
    for _ in 1:b; P=polymul_lin!(P,eT2); end
    sum(co*smf(e) for (e,co) in P)
end
mrot=Dict{NTuple{3,Int},BigFloat}()
for k in 0:4, a in 0:4, b in 0:4; (k+a+b<=4) && (mrot[(k,a,b)]=rotmom(k,a,b)); end

# ---- 3-node S-quadrature family (Prony over free m5) ----
mS=[Float64(mrot[(k,0,0)]) for k in 0:4]
function threeatom(m5)
    A=[mS[1] mS[2] mS[3]; mS[2] mS[3] mS[4]; mS[3] mS[4] mS[5]]; rhs=[mS[4],mS[5],m5]
    c=A\rhs; comp=[0.0 0.0 c[1];1.0 0.0 c[2];0.0 1.0 c[3]]; rt=eigvals(comp)
    (any(abs.(imag.(rt)).>1e-6*(maximum(abs,real.(rt))+1e-9))) && return nothing
    nodes=sort(real.(rt)); V=[nodes[j]^k for k in 0:2,j in 1:3]; w=V\[mS[1],mS[2],mS[3]]
    (nodes,w)
end
cflv(s)=Float64(LAM)*(abs(mu[1]+sig[1]*Float64(s/s3))+abs(mu[2]+sig[2]*Float64(s/s3))+abs(mu[3]+sig[3]*Float64(s/s3)))
import Random; Random.seed!(11)
members=Vector{Tuple{Vector{Float64},Vector{Float64}}}()
if SNODES == 3
    for m5 in range(-6e5,-1.5e4;length=400)
        r=threeatom(m5); r===nothing && continue
        nodes,w=r
        (all(>(-1e-9),w) && all(s->cflv(s)<=1.0,nodes)) && push!(members,(nodes,w))
    end
    idxs=round.(Int,range(1,length(members);length=min(NMEM,length(members)))); members=members[unique(idxs)]
else  # 5-node (or more): random positions (2 tails + intermediate + bulk + flex), exact weights, positive+CFL
    mvec=[mS[k+1] for k in 0:4]
    while length(members) < NMEM
        st1=-95+30*rand(); st2=-60+35*rand(); smid=-25+20*rand(); sbulk=-1.5+3*rand(); sflex=-6+40*rand()
        sv=sort([st1,st2,smid,sbulk,sflex]); (minimum(diff(sv))<1.0) && continue
        V=[sv[j]^k for k in 0:4, j in 1:5]; w=V\mvec
        (all(>(1e-7),w) && all(s->cflv(s)<=1.0,sv)) || continue
        # verify m0..m4 exactly
        (maximum(abs(sum(w[j]*sv[j]^k for j in 1:5)-mvec[k+1]) for k in 0:4) < 1e-6) || continue
        push!(members,(sv,w))
    end
end
@printf("rotated moments computed; %d-node positive+CFL family members to scan: %d\n", SNODES, length(members)); flush(stdout)

# ---- 2D moment machinery ----
mon2(d)=sort([(a,b) for a in 0:d for b in 0:d if a+b<=d]; by=m->(sum(m),m))
B2=mon2(2); B1=mon2(1); ALL4=mon2(4)
addt2(p,q)=(p[1]+q[1],p[2]+q[2])
# CFL-slice affine facets g(T1,T2)=g0+g1 T1+g2 T2 >=0 at S=s
function facets(s)
    fs=Vector{NTuple{3,Float64}}()
    for sg in Iterators.product((-1,1),(-1,1),(-1,1))
        g0=Float64(R)-sum(sg[j]*(mu[j]+sig[j]*Float64(s)*Float64(eS[j])) for j in 1:3)
        g1=-sum(sg[j]*sig[j]*Float64(eT1[j]) for j in 1:3)
        g2=-sum(sg[j]*sig[j]*Float64(eT2[j]) for j in 1:3)
        push!(fs,(g0,g1,g2))
    end
    fs
end

# ---- inner SDP for a marginal member (svec,wvec) ----
function inner_sdp(svec,wvec)
    Nn=length(svec)
    m=Model(Hypatia.Optimizer); set_silent(m)
    # c[i][(a,b)] normalized conditional moments; (0,0)=1
    cvar=[Dict{NTuple{2,Int},Any}() for _ in 1:Nn]
    for i in 1:Nn, ab in ALL4
        cvar[i][ab] = ab==(0,0) ? 1.0 : @variable(m, base_name="c$(i)_$(ab[1])_$(ab[2])")
    end
    cf(i,ab)=convert(AffExpr,cvar[i][ab])
    @variable(m,t)
    # joint moment equalities (transverse only; pure-S satisfied by marginal)
    for k in 0:4, ab in ALL4
        (ab==(0,0)) && continue
        (k+ab[1]+ab[2] > 4) && continue
        @constraint(m, sum(wvec[i]*svec[i]^k*cf(i,ab) for i in 1:Nn) == Float64(mrot[(k,ab[1],ab[2])]))
    end
    # per-node M_2 >= t I and CFL-slice localizing M_1(g c) >= 0
    for i in 1:Nn
        M2=[cf(i,addt2(B2[p],B2[q])) for p in eachindex(B2), q in eachindex(B2)]
        @constraint(m, Symmetric(M2 .- t*Matrix(I,length(B2),length(B2))) in PSDCone())
        for (g0,g1,g2) in facets(svec[i])
            L=[g0*cf(i,addt2(B1[p],B1[q])) + g1*cf(i,addt2(addt2(B1[p],B1[q]),(1,0))) +
               g2*cf(i,addt2(addt2(B1[p],B1[q]),(0,1))) for p in eachindex(B1), q in eachindex(B1)]
            @constraint(m, Symmetric(L) in PSDCone())
        end
    end
    @objective(m,Max,t); optimize!(m)
    (termination_status(m), has_values(m) ? value(t) : NaN)
end

# ---- linear equality residual per transverse degree (diagnostic, no PSD) ----
function eqresid(svec,wvec)
    Nn=length(svec)
    res=Dict(1=>0.0,2=>0.0,3=>0.0,4=>0.0)
    # for each transverse (a,b) solve LSQ over c^{(i)}_{ab} the (5-d) equations, record residual
    for ab in ALL4
        d=ab[1]+ab[2]; d==0 && continue
        A=Float64[ wvec[i]*svec[i]^k for k in 0:(4-d), i in 1:Nn ]
        rhs=Float64[ Float64(mrot[(k,ab[1],ab[2])]) for k in 0:(4-d) ]
        c=A\rhs; r=norm(A*c-rhs)/(norm(rhs)+1e-30)
        res[d]=max(res[d],r)
    end
    res
end

@printf("\n%-4s %-10s %-9s %-9s | eq-resid by transverse degree (LSQ, no PSD)\n","mem","tail_S","t*","status")
bestt=(-Inf,nothing)
for (mi,(sv,wv)) in enumerate(members)
    er=eqresid(sv,wv)
    st,tv=inner_sdp(sv,wv)
    @printf("%-4d %-10.2f %-9.3e %-9s | d1=%.2e d2=%.2e d3=%.2e d4=%.2e  nodes=%s w=%s\n",
            mi, minimum(sv), tv, string(st)[1:min(8,end)], er[1],er[2],er[3],er[4],
            join((@sprintf("%.1f",x) for x in sv),","), join((@sprintf("%.1e",x) for x in wv),","))
    flush(stdout)
    (isfinite(tv) && tv>bestt[1]) && (global bestt=(tv,(sv,wv)))
end
@printf("\nBEST over family: t* = %.4e  %s\n", bestt[1],
        bestt[1]>1e-8 ? "=> conditional lift FEASIBLE with margin (structured constructive route)" :
        bestt[1]>-1e-7 ? "=> marginal boundary; try more members / 5 S-nodes" :
        "=> 3-node conditional INFEASIBLE everywhere => escalate to 5 S-nodes (NOT an impossibility)")
if bestt[2]!==nothing && bestt[1]>1e-8
    save("gpu/validation/kfvs_conditional_best.jld2","svec",bestt[2][1],"wvec",bestt[2][2],"tstar",bestt[1])
    println("saved best marginal member -> kfvs_conditional_best.jld2 (extract 2D conditional atoms next)")
end
