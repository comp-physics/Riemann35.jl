# kfvs_audit_witness.jl — AUTHORITATIVE audit: does the physical-coordinate 7e-6 witness
# actually reproduce the STANDARDIZED (central) moments, or was the small raw residual a
# mean-shift-cancellation artifact?  Reproduce the physical grid witness (same ±7σ grid as
# the 66/66 oracle), then transform nodes to standardized coords ṽ=(v−mean)/σ and compare
# Σw ṽ^α against the target standardized moments (computed independently in BigFloat).
# Verdict hinges on the componentwise 4th-moment errors.
using JLD2, Printf, LinearAlgebra
d=load("gpu/validation/kfvs_defect_counterexample.jld2"); λ=d["lam"][1]; C=d["center_state"]; cls=d["class"]
TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
IDX=Dict(TRIP[n]=>n for n in 1:35); cfl(v)=λ*(abs(v[1])+abs(v[2])+abs(v[3]))
bino(n,k)= k<0||k>n ? big(0) : big(binomial(n,k))
D4=("m400"=>5,"m040"=>15,"m004"=>25,"m220"=>12,"m202"=>22,"m022"=>35,"m300"=>4,"m030"=>13)

s=findfirst(==(3),cls); M=collect(C[:,s]); ρ=M[1]; ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
σx=sqrt(max(M[3]/ρ-ux^2,1e-14));σy=sqrt(max(M[10]/ρ-uy^2,1e-14));σz=sqrt(max(M[20]/ρ-uz^2,1e-14))
# --- target standardized central moments in BigFloat (per unit mass) ---
Mb=big.(M); uxb=Mb[2]/Mb[1];uyb=Mb[6]/Mb[1];uzb=Mb[16]/Mb[1]
σxb=sqrt(Mb[3]/Mb[1]-uxb^2);σyb=sqrt(Mb[10]/Mb[1]-uyb^2);σzb=sqrt(Mb[20]/Mb[1]-uzb^2)
sstd=zeros(BigFloat,35)
for n in 1:35;(i,j,k)=TRIP[n];acc=big(0.0)
    for p in 0:i,q in 0:j,r in 0:k; haskey(IDX,(p,q,r))||continue
        acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-uxb)^(i-p)*(-uyb)^(j-q)*(-uzb)^(k-r)*(Mb[IDX[(p,q,r)]]/Mb[1]); end
    sstd[n]=acc/(σxb^i*σyb^j*σzb^k)
end
@printf("class-C: mean≈58  σ≈1.1  target std kurtosis(m400)=%.1f  cross(m202)=%.1f  -> support ≥ √%.0f ≈ %.1fσ\n",
        Float64(sstd[5]),Float64(sstd[22]),Float64(sstd[5]),sqrt(Float64(sstd[5])))

# --- physical grid witness (SAME ±7σ grid as the 66/66 oracle) ---
K=7.0; ng=17; gs=range(-K,K;length=ng)
gnodes=[[a,b,c] for a in gs for b in gs for c in gs if cfl((ux+σx*a,uy+σy*b,uz+σz*c))≤1.0]  # standardized grid coords
S=[ρ*σx^TRIP[n][1]*σy^TRIP[n][2]*σz^TRIP[n][3] for n in 1:35]
Φ=Array{Float64}(undef,35,length(gnodes))
for (bi,g) in enumerate(gnodes),n in 1:35;(i,j,k)=TRIP[n];Φ[bi>0 ? n : n,bi]=(ux+σx*g[1])^i*(uy+σy*g[2])^j*(uz+σz*g[3])^k;end
Φs=Φ./S; tgt=M./S
function fista(Φ,b;it=25000)
    w=zeros(size(Φ,2));wp=copy(w);t=1.0
    vv=randn(length(w)); for _ in 1:25; vv=Φ'*(Φ*vv); vv./=norm(vv); end; η=1/norm(Φ*(vv./norm(vv)))^2
    for _ in 1:it; y=w.+((t-1)/t).*(w.-wp); wp=copy(w); w=max.(y.-η.*(Φ'*(Φ*y.-b)),0.0); t=(1+sqrt(1+4t^2))/2; end
    w
end
w=fista(Φs,tgt)
rawres=norm(Φs*w.-tgt)/norm(tgt)
@printf("physical witness (±7σ grid): active=%d  RAW rel residual=%.2e\n",count(>(1e-10*ρ),w),rawres)

# --- transform witness to standardized coords: ṽ=g ; standardized moments = (1/ρ)Σ w g^α ---
mtil=zeros(35)
for bi in eachindex(w); w[bi]>1e-14 || continue; g=gnodes[bi]
    for n in 1:35;(i,j,k)=TRIP[n]; mtil[n]+=w[bi]*g[1]^i*g[2]^j*g[3]^k; end
end
mtil./=ρ
@printf("\nAUDIT — physical witness in STANDARDIZED coordinates vs BigFloat target:\n")
@printf("   variance m200: witness=%.4f target=%.4f\n", mtil[3], Float64(sstd[3]))
for (nm,idx) in D4
    @printf("   %s: witness=%.3f  target=%.3f  |err|=%.2e\n", nm, mtil[idx], Float64(sstd[idx]), abs(mtil[idx]-Float64(sstd[idx])))
end
max4=maximum(abs(mtil[i]-Float64(sstd[i])) for i in (5,15,25,12,22,35))
@printf("VERDICT: raw residual %.1e BUT standardized 4th-moment max error = %.2e  =>  %s\n",
        rawres, max4, max4<1e-3 ? "physical witness is LEGIT (66/66 stands; std solver just had a bad grid)" :
        "physical witness FAILS in standardized coords — the 66/66 result was a MEAN-SHIFT CANCELLATION ARTIFACT")
