# kfvs_redistribute.jl — EXPERIMENT 2 (reviewer two-shot): conservative regional redistribution over the
# A/B cells. Each A/B cell has a REALIZABLE measure base U_measure_i (margin>=0). The F3 flux update would
# require the (out-of-cone) endpoint U_flux_i; the flux/measure discrepancy is D_i = U_flux_i - U_measure_i.
# QUESTION: can we pick corrections R_i with  U_measure_i + R_i in R_4 (realizable)  and  Σ_i R_i = Σ_i D_i
# (so the component-total update equals the conservative flux total)? Realizable set is CONVEX and the
# conservation constraint is affine, so we run PROJECTION ONTO CONVEX SETS (POCS):
#   (P1) per-cell: project U_measure_i+R_i onto R_4 via realizable_3D_M4  -> R_i := proj - U_measure_i
#   (P2) conservation: subtract the mean discrepancy so Σ R_i = ΣD
# A converged fixed point (all margins>=0 AND ΣR=ΣD) is a CONSTRUCTIVE feasibility certificate. Failure to
# converge is suggestive (not a proof) of infeasibility. Env: r35env.
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, JLD2, Printf, LinearAlgebra, Statistics
ce=load(joinpath(@__DIR__,"kfvs_defect_counterexample.jld2"))
cls=ce["class"]; Uf=ce["U_flux"]; Um=ce["U_measure"]; cell=ce["cell"]; Ma=ce["Ma"]; s3max=ce["s3max"]
# component selection: default = all A/B (class 1 or 2). Optional arg "conn" = largest 6-connected A/B blob.
mode = length(ARGS)>=1 ? ARGS[1] : "all"
ab = findall(x->x==1||x==2, cls)
if mode=="conn"
    coords=Dict(ab[k]=>Tuple(round.(Int,cell[:,ab[k]])) for k in eachindex(ab))
    cset=Set(values(coords)); adj(c)=[(c[1]+d[1],c[2]+d[2],c[3]+d[3]) for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))]
    # BFS components
    seen=Set{NTuple{3,Int}}(); comps=Vector{Vector{NTuple{3,Int}}}()
    for c in cset; c in seen && continue; q=[c];push!(seen,c);comp=NTuple{3,Int}[]
        while !isempty(q); x=pop!(q); push!(comp,x); for y in adj(x); (y in cset && !(y in seen)) && (push!(seen,y);push!(q,y)); end; end
        push!(comps,comp); end
    big=comps[argmax(length.(comps))]; bset=Set(big)
    ab=[ab[k] for k in eachindex(ab) if coords[ab[k]] in bset]
    @printf("connected A/B component: %d cells (of %d comps, sizes %s)\n", length(ab), length(comps), string(sort(length.(comps);rev=true)[1:min(5,end)]))
end
N=length(ab)
D=Uf[:,ab].-Um[:,ab]; T=vec(sum(D,dims=2))           # total discrepancy to conserve
base=Um[:,ab]
# conservation set: "full" = all 35 moments ; "phys" = physical conservation laws only
#   mass M[1]; momentum M[2],M[6],M[16]; 2nd-order/energy M[3],M[7],M[17],M[10],M[26],M[20]
consmode = length(ARGS)>=2 ? ARGS[2] : "full"
CONS = consmode=="phys" ? [1,2,6,16,3,7,17,10,26,20] : collect(1:35)
@printf("conservation set: %s (%d moments)\n", consmode, length(CONS))
margin(v)=realizability_margin(collect(v))
@printf("EXPERIMENT 2 redistribution over %d A/B cells (mode=%s)\n", N, mode)
@printf("  total discrepancy T: |T[ρ]|=%.2e  |T|_2=%.3e  (energy T[3]+T[10]+T[20]=%.3e)\n", abs(T[1]), norm(T), T[3]+T[10]+T[20])
@printf("  base U_measure margins: min=%.2e all>=0=%s\n", minimum(margin(base[:,k]) for k in 1:N), all(margin(base[:,k])>=0 for k in 1:N))

R=zeros(35,N)
function run_pocs(R,base,T,N; iters=400, α=1.0)
    hist=Float64[]
    for it in 1:iters
        Rold=copy(R)
        # P1: per-cell realizability projection of base+R
        for k in 1:N
            U=base[:,k].+R[:,k]
            proj=realizable_3D_M4(U, Ma, s3max)
            R[:,k]=proj.-base[:,k]
        end
        # P2: conservation projection Σ R[cons] = T[cons] on the conserved rows only (subtract mean discrepancy)
        for r in CONS; disc=sum(R[r,:])-T[r]; R[r,:].-=disc/N; end
        α<1.0 && (R .= Rold .+ α.*(R .- Rold))   # under-relaxation to damp alternating-projection oscillation
        # metrics AFTER both projections: worst realizability of base+R and conservation residual (conserved rows)
        wm=minimum(margin(base[:,k].+R[:,k]) for k in 1:N)
        cr=norm([sum(R[r,:])-T[r] for r in CONS])/max(norm(T[CONS]),1e-30)
        push!(hist,wm)
        (it<=5 || it%50==0) && @printf("   it%-4d worst_margin=%+.3e  conservation_resid(rel)=%.3e\n", it, wm, cr)
        (wm>=-1e-10 && cr<1e-8) && return (:FEASIBLE,it,wm,cr,hist)
    end
    wm=minimum(margin(base[:,k].+R[:,k]) for k in 1:N); cr=norm([sum(R[r,:])-T[r] for r in CONS])/max(norm(T[CONS]),1e-30)
    (:NOCONV,iters,wm,cr,hist)
end
αr = length(ARGS)>=3 ? parse(Float64,ARGS[3]) : 1.0
st,it,wm,cr,hist=run_pocs(R,base,T,N; α=αr)
@printf("\n=== EXPERIMENT 2 (%s A/B, %d cells): %s after %d iters ; worst_margin=%+.3e conservation_resid=%.3e ===\n",
        mode, N, st, it, wm, cr)
if st==:FEASIBLE
    println("CONSTRUCTIVE certificate: A/B component can conservatively absorb the flux/measure discrepancy while staying realizable.")
else
    # diagnose: how many cells are the blockers, and the tail of worst-margin history (converging or stuck?)
    nbad=count(margin(base[:,k].+R[:,k])<-1e-10 for k in 1:N)
    @printf("NOT converged: %d/%d cells still out-of-cone; worst-margin last-10 trend: %s\n", nbad, N,
            join([@sprintf("%.1e",x) for x in hist[max(1,end-9):end]], " "))
end
save(joinpath(@__DIR__,"kfvs_redistribute.jld2"),"ab",ab,"R",R,"T",T,"status",string(st),"worst_margin",wm,"cons_resid",cr)
