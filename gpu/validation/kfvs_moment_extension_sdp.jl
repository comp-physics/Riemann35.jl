# kfvs_moment_extension_sdp.jl — moment-extension SDP on ONE class-C state.
#
# DECISIVE test of whether a CFL-support-constrained positive cubature can match the
# 35 standardized degree-<=4 moments of a class-C failing cell.
#
# Support set (standardized coords X_i = (v_i - mu_i)/sigma_i):
#   K = { X : sum_i |mu_i + sigma_i X_i| <= R },  R = 1/lambda   (physical CFL diamond)
#     = { X : g_s(X) >= 0  for all 8 sign vectors s in {-1,+1}^3 }
#   g_s(X) = R - sum_i s_i mu_i - sum_i s_i sigma_i X_i           (affine)
#
# Lasserre order-3 relaxation of the truncated K-moment problem:
#   unknown moments y_alpha, |alpha| <= 6 (84 total); the 35 with |alpha|<=4 are FIXED to
#   the measured standardized moments; the 49 of degree 5,6 are free.  y_0 = 1.
#   Necessary PSD conditions (Curto-Fialkow / Lasserre):
#     M_3(y) >= 0                      (20x20, moment matrix, monomials deg<=3)
#     M_2(g_s . y) >= 0  for all 8 s   (10x10, localizing matrices, monomials deg<=2)
#
# STEP 1 (decisive): maximize slack t s.t. M_3 >= t I, M_2(g_s) >= t I.
#   t* >= 0  => PSD extension exists  => NO obstruction at this (necessary) relaxation
#   t* <  0  => relaxation infeasible => GENUINE compact-support obstruction (rigorous)
# STEP 2 (if feasible): minimize trace(M_3) to drive toward a flat/low-rank extension;
#   report spectra of M_3, M_2; flatness rank M_3 == rank M_2 => finitely-atomic measure.
# STEP 3 (if flat): extract atoms via commuting multiplication matrices (constructive nodes).
#
# All moment DATA in BigFloat; SDP solved in Float64 (well-conditioned in std coords).

using JLD2, Printf, LinearAlgebra, JuMP, Hypatia

setprecision(BigFloat, 256)
const D = load("gpu/validation/kfvs_defect_counterexample.jld2")
const LAM = D["lam"][1]
const C = D["center_state"]
const CLS = D["class"]
const TRIP = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX = Dict(TRIP[n] => n for n in 1:35)
bino(n,k) = (k < 0 || k > n) ? big(0) : big(binomial(n,k))

# ---- pick the class-C representative state, compute standardized moments in BigFloat ----
s0 = findfirst(==(3), CLS)
M = big.(collect(C[:, s0])); ρ = M[1]
ux = M[2]/ρ; uy = M[6]/ρ; uz = M[16]/ρ
σx = sqrt(M[3]/ρ - ux^2); σy = sqrt(M[10]/ρ - uy^2); σz = sqrt(M[20]/ρ - uz^2)
sstd = Dict{NTuple{3,Int},BigFloat}()
for n in 1:35
    (i,j,k) = TRIP[n]; acc = big(0.0)
    for p in 0:i, q in 0:j, r in 0:k
        haskey(IDX,(p,q,r)) || continue
        acc += bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ)
    end
    sstd[TRIP[n]] = acc/(σx^i*σy^j*σz^k)
end

R      = 1/LAM
meanv  = (Float64(ux), Float64(uy), Float64(uz))
sigv   = (Float64(σx), Float64(σy), Float64(σz))
@printf("class-C state: mean=(%.3g,%.3g,%.3g) sigma=(%.3g,%.3g,%.3g)  R_CFL=%.1f\n",
        meanv..., sigv..., R)
@printf("std moments: var=%.4f  m400=%.1f  m220=%.1f  m111=%.3g\n",
        Float64(sstd[(2,0,0)]), Float64(sstd[(4,0,0)]), Float64(sstd[(2,2,0)]), Float64(sstd[(1,1,1)]))

# ---- monomial bases ----
mons(maxdeg) = [(i,j,k) for i in 0:maxdeg for j in 0:maxdeg for k in 0:maxdeg if i+j+k <= maxdeg]
B3 = mons(3); B2 = mons(2)                     # 20 and 10
addt(a,b) = (a[1]+b[1], a[2]+b[2], a[3]+b[3])

# ---- the model ----
model = Model(Hypatia.Optimizer)
set_silent(model)
# one JuMP variable per moment of degree in {5,6}; deg<=4 are fixed BigFloat->Float64 constants
yvar = Dict{NTuple{3,Int},Any}()
for a in mons(6)
    dg = sum(a)
    if dg <= 4
        yvar[a] = Float64(sstd[a])            # fixed data (deg<=4, includes y_0=1)
    else
        yvar[a] = @variable(model, base_name="y_$(a[1])_$(a[2])_$(a[3])")
    end
end
ymom(a) = convert(AffExpr, yvar[a])   # unify fixed Float64 data and free VariableRefs to AffExpr

@variable(model, t)
# moment matrix M_3
M3 = [ymom(addt(B3[a],B3[b])) for a in 1:length(B3), b in 1:length(B3)]
@constraint(model, Symmetric(M3 - t*Matrix(I,length(B3),length(B3))) in PSDCone())
# localizing matrices for the 8 CFL facets
for sgn in Iterators.product((-1,1),(-1,1),(-1,1))
    c0 = R - (sgn[1]*meanv[1] + sgn[2]*meanv[2] + sgn[3]*meanv[3])
    cc = (-sgn[1]*sigv[1], -sgn[2]*sigv[2], -sgn[3]*sigv[3])
    L = Array{AffExpr}(undef, length(B2), length(B2))
    for a in 1:length(B2), b in 1:length(B2)
        base = addt(B2[a], B2[b])
        L[a,b] = c0*ymom(base) + cc[1]*ymom(addt(base,(1,0,0))) +
                 cc[2]*ymom(addt(base,(0,1,0))) + cc[3]*ymom(addt(base,(0,0,1)))
    end
    @constraint(model, Symmetric(L - t*Matrix(I,length(B2),length(B2))) in PSDCone())
end

@objective(model, Max, t)
@printf("\n[STEP 1] max-margin feasibility SDP  (vars=%d, M3=20x20, 8x M2=10x10) ...\n",
        length(mons(6)) - 35)
optimize!(model)
st = termination_status(model)
tstar = has_values(model) ? value(t) : NaN
@printf("  status=%s   t* = %.6e\n", st, tstar)
if tstar >= -1e-9
    println("  VERDICT STEP1: PSD moment extension FEASIBLE (t* >= 0) => NO obstruction at Lasserre-3.")
    println("                 => a CFL-support positive cubature is NOT ruled out; proceed to extraction.")
else
    println("  VERDICT STEP1: relaxation INFEASIBLE (t* < 0) => GENUINE compact-support obstruction.")
    println("                 => no CFL-safe positive cubature can match these moments (rigorous, necessary-cond).")
end

# ---- STEP 2: only if feasible, push toward a flat (low-rank) extension ----
if tstar >= -1e-9
    model2 = Model(Hypatia.Optimizer); set_silent(model2)
    y2 = Dict{NTuple{3,Int},Any}()
    for a in mons(6)
        y2[a] = sum(a) <= 4 ? Float64(sstd[a]) : @variable(model2, base_name="z_$(a[1])_$(a[2])_$(a[3])")
    end
    ym(a) = convert(AffExpr, y2[a])
    M3b = [ym(addt(B3[a],B3[b])) for a in 1:length(B3), b in 1:length(B3)]
    @constraint(model2, Symmetric(M3b) in PSDCone())
    for sgn in Iterators.product((-1,1),(-1,1),(-1,1))
        c0 = R - (sgn[1]*meanv[1] + sgn[2]*meanv[2] + sgn[3]*meanv[3])
        cc = (-sgn[1]*sigv[1], -sgn[2]*sigv[2], -sgn[3]*sigv[3])
        L = Array{AffExpr}(undef, length(B2), length(B2))
        for a in 1:length(B2), b in 1:length(B2)
            base = addt(B2[a], B2[b])
            L[a,b] = c0*ym(base) + cc[1]*ym(addt(base,(1,0,0))) +
                     cc[2]*ym(addt(base,(0,1,0))) + cc[3]*ym(addt(base,(0,0,1)))
        end
        @constraint(model2, Symmetric(L) in PSDCone())
    end
    @objective(model2, Min, sum(ym(addt(B3[a],B3[a])) for a in 1:length(B3)))   # trace M_3
    @printf("\n[STEP 2] min-trace(M_3) toward flat extension ...\n")
    optimize!(model2)
    @printf("  status=%s\n", termination_status(model2))
    M3val = [value(ym(addt(B3[a],B3[b]))) for a in 1:length(B3), b in 1:length(B3)]
    ev3 = sort(eigvals(Symmetric(M3val)); rev=true)
    tol = 1e-8 * maximum(abs, ev3)
    rank3 = count(>(tol), ev3)
    # rank of M_2 (deg<=2 principal block == leading 10x10)
    M2val = [value(ym(addt(B2[a],B2[b]))) for a in 1:length(B2), b in 1:length(B2)]
    ev2 = sort(eigvals(Symmetric(M2val)); rev=true)
    rank2 = count(>(1e-8*maximum(abs,ev2)), ev2)
    @printf("  eig(M_3) top: %s\n", join((@sprintf("%.2e",e) for e in ev3[1:min(8,end)]), " "))
    @printf("  rank M_3 = %d   rank M_2 = %d   (tol rel 1e-8)\n", rank3, rank2)
    if rank3 == rank2
        @printf("  VERDICT STEP2: FLAT extension (rank M_3 == rank M_2 = %d) => finitely-atomic measure with %d atoms EXISTS.\n", rank3, rank3)
        # persist for extraction
        save("gpu/validation/kfvs_sdp_flat_extension.jld2",
             "M3", M3val, "B3", collect(B3), "rank", rank3, "mean", collect(meanv), "sig", collect(sigv), "R", R)
        println("  saved flat extension -> kfvs_sdp_flat_extension.jld2 (atom extraction next)")
    else
        @printf("  VERDICT STEP2: not flat at order 3 (rank M_3=%d > rank M_2=%d). Extension exists but higher order needed for finite atoms.\n", rank3, rank2)
    end
end
