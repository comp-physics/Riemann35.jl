# kfvs_classA_construct.jl — class-specific constructor for class-A (and A/B) cells: a DETERMINISTIC
# full-3D positive-cubature LP on a standardized multiscale grid, NOT the class-C jet lift. Class-A has
# moderate node speeds + diffuse 4th-order defect (no 26-sigma tail), so a well-conditioned standardized
# grid contains the support and HiGHS simplex finds an exact positive representation.
# Deterministic (no RNG); degree-row-scaled + column-normalized; BigFloat-verified. Explicit failure codes.
using JLD2, Printf, LinearAlgebra, JuMP, HiGHS
setprecision(BigFloat, 512)
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX=Dict(TRIP[n]=>n for n in 1:35); bino(n,k)=(k<0||k>n) ? big(0) : big(binomial(n,k))
ce=load("gpu/validation/kfvs_defect_counterexample.jld2"); Cst=ce["center_state"]; cls=ce["class"]; λ=ce["lam"][1]; R=1/λ
which = length(ARGS)>=1 ? parse(Int,ARGS[1]) : 1     # class to build (1=A default)
targets = findall(==(which), cls)
@printf("class-%d cells: %d\n", which, length(targets)); flush(stdout)

function standardize(Mraw)
    M=big.(Mraw); ρ=M[1]; ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
    σx=sqrt(M[3]/ρ-ux^2);σy=sqrt(M[10]/ρ-uy^2);σz=sqrt(M[20]/ρ-uz^2)
    s=zeros(BigFloat,35)
    for n in 1:35; (i,j,k)=TRIP[n]; acc=big(0.0)
        for p in 0:i,q in 0:j,r in 0:k; haskey(IDX,(p,q,r))||continue
            acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ); end
        s[n]=acc/(σx^i*σy^j*σz^k); end
    (s,(Float64(ux),Float64(uy),Float64(uz)),(Float64(σx),Float64(σy),Float64(σz)))
end
cflX(X,mu,sig)=λ*(abs(mu[1]+sig[1]*X[1])+abs(mu[2]+sig[2]*X[2])+abs(mu[3]+sig[3]*X[3]))

# directional moments E[(l.X)^k] (Float64) from standardized moments
function linmom(sstdF, l, deg)
    tot=0.0
    for a in 0:deg, b in 0:(deg-a); c=deg-a-b
        mult=factorial(deg)/(factorial(a)*factorial(b)*factorial(c))
        tot += mult * l[1]^a * l[2]^b * l[3]^c * sstdF[IDX[(a,b,c)]]
    end
    tot
end
# box half-width needed to contain atoms out to the directional support R(l)=sqrt(E[(lX)^4]/E[(lX)^2])
# over a deterministic sphere sample (axes + diagonals + fibonacci sphere): K = max_l R(l)*max_i|l_i|
function box_K(sstdF)
    dirs=[[1.,0,0],[0,1.,0],[0,0,1.],[1.,1,0],[1.,0,1],[0,1.,1],[1.,-1,0],[1.,0,-1],[0,1.,-1],
          [1.,1,1],[1.,1,-1],[1.,-1,1],[1.,-1,-1]]
    ga=Float64(π)*(3-sqrt(5.0))
    for n in 0:199; z=1-2*(n+0.5)/200; r=sqrt(max(1-z^2,0)); th=ga*n; push!(dirs,[r*cos(th),r*sin(th),z]); end
    K=0.0; suppmax=0.0
    for d in dirs; l=d./norm(d); v2=linmom(sstdF,l,2); v4=linmom(sstdF,l,4)
        (v2<=0)&&continue; Rl=sqrt(max(v4,0)/v2); suppmax=max(suppmax,Rl); K=max(K, Rl*maximum(abs,l)); end
    (K, suppmax)
end
# deterministic full-3D positive cubature: grid over ±K std, CFL-restricted, exact HiGHS LP
function construct_A(sstdF, sstd, mu, sig)
    Kdir, supp = box_K(sstdF)
    suppax = maximum(sqrt(max(sstdF[i],1.0)) for i in (5,15,25))     # axis support (for diagnostics)
    supp = max(Kdir, suppax)                                          # directional box half-width
    Ls = supp                                                          # degree row-scale
    rowsc=[1.0/Ls^sum(TRIP[n]) for n in 1:35]; tgt=sstdF
    for K in (1.4*supp, 2.0*supp, 2.8*supp)                            # deterministic escalation
        ng=clamp(round(Int, 2K/0.33), 25, 47)                          # hold spacing ~0.33σ (resolution scales with support)
        gr=range(-K,K;length=ng); nodes=NTuple{3,Float64}[]
        for a in gr,b in gr,c in gr; cflX((a,b,c),mu,sig)<=1.0 && push!(nodes,(a,b,c)); end
        isempty(nodes) && continue
        Nn=length(nodes)
        Φ=Array{Float64}(undef,35,Nn); for (jn,X) in enumerate(nodes),n in 1:35; (i,j,k)=TRIP[n]; Φ[n,jn]=X[1]^i*X[2]^j*X[3]^k; end
        Φr=Φ.*rowsc; br=tgt.*rowsc; cn=[max(norm(@view Φr[:,j]),1e-300) for j in 1:Nn]; Φn=Φr./cn'
        lp=Model(HiGHS.Optimizer); set_silent(lp); set_attribute(lp,"solver","simplex")
        @variable(lp,u[1:Nn]>=0); @constraint(lp,Φn*u.==br); @objective(lp,Min,sum(u)); optimize!(lp)
        has_values(lp) || continue
        w=max.(value.(u),0.0)./cn; act=findall(>(1e-12),w)
        # BigFloat verify standardized
        wB=big.(w); mx=big(0.0)
        for n in 1:35; (i,j,k)=TRIP[n]; val=sum(wB[a]*big(nodes[a][1])^i*big(nodes[a][2])^j*big(nodes[a][3])^k for a in act); mx=max(mx,abs(val-sstd[n])); end
        if Float64(mx)<1e-6
            return (:OK, [collect(nodes[a]) for a in act], w[act], Float64(mx), maximum(cflX(nodes[a],mu,sig) for a in act), K)
        end
    end
    (:EXTRACT_FAIL, nothing,nothing,NaN,NaN,NaN)
end

results=[]; cubs=Dict{Int,Any}()
for (ii,ccol) in enumerate(targets)
    sstd,mu,sig=standardize(Cst[:,ccol]); sstdF=Float64.(sstd)
    st,atoms,w,resid,cfl,K=construct_A(sstdF,sstd,mu,sig)
    ok = st==:OK
    ok && (cubs[ccol]=(atoms=atoms,weights=w,mu=collect(mu),sig=collect(sig)))
    push!(results,(cell=ccol,ok=ok,code=st,natoms=ok ? length(w) : 0,resid=resid,cfl=cfl))
    @printf("[%3d/%d] col%d(cl%d) %-12s natoms=%d resid=%.1e cfl=%.3f K=%.1f\n",
            ii,length(targets),ccol,cls[ccol], string(st), ok ? length(w) : 0, resid, cfl, K); flush(stdout)
end
nok=count(r->r.ok,results)
@printf("\n=== CLASS-%d full-3D constructor: %d/%d OK ===\n", which, nok, length(targets))
@printf("worst resid over OK: %.2e ; atom counts range %s\n",
        maximum([r.resid for r in results if r.ok];init=0.0),
        isempty([r.natoms for r in results if r.ok]) ? "-" : string(extrema([r.natoms for r in results if r.ok])))
save("gpu/validation/kfvs_classA_cubatures.jld2","cols_ok",[r.cell for r in results if r.ok],
     "resid",[r.resid for r in results],"natoms",[r.natoms for r in results])
