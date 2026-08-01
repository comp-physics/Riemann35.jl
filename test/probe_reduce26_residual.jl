#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# THE ACTUAL TEST OF FOX'S 26-MOMENT REDUCTION IN WALL-BOUNDED SHEAR.
#
# test/probe_odd_cross_wall.jl established that the nine dropped odd fourth-order
# cross-moments are NOT small in planar Couette: the three that survive the flow's
# z-symmetry (S310, S130, S112) run about 3x the standardized shear stress, flat
# across Kn_H = 1.0 down to 0.1, and 10-50x the deviation |S220-1| carried by the
# kept even moment.
#
# THAT MEASUREMENT DOES NOT SETTLE THE QUESTION, and reading it as "the reduction
# throws away the physics" would be wrong. The reduction does not delete those
# moments -- it closes them algebraically in terms of the retained ones. What
# matters is whether that closure REPRODUCES them, not whether they are large.
#
# In fact the flat ratio of 3 is what the closure itself predicts. Its leading term is
#     S310 = S110*S400 + (3/2) S300 (S210 - S110 S300),
# and for a near-Maxwellian state S400 -> 3, so S310 -> 3*S110: the closure says the
# odd cross-moment IS three times the shear stress. A measured ratio near 3 is
# therefore evidence FOR the reduction, not against it. The constancy in Kn_H is the
# signature of the odd moments being slaved to the retained ones, exactly as an
# algebraic closure assumes.
#
# So this script measures the residual
#     ||R(S) - S|| / ||S||   restricted to the nine dropped slots,
# with R the reduction map (src/moments/moment_reduce26.jl), on the converged Couette
# field. Small residual => slaved, reduction is right here. Large => independent
# content, and Fox's hypothesis about micro-flows has teeth.
#
# CAVEAT ON PROVENANCE: moment_reduce26.jl is a REIMPLEMENTATION from the formulas as
# printed in the notes; the original scripts are missing from both repositories. A
# small residual is only as trustworthy as those formulas.
#
# Env: PR_KNH, PR_CPL, PR_UW, PR_ORDER, PR_TEND
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
include(joinpath(@__DIR__, "couette_driver.jl"))
using Riemann35: reduce26_S, reduce26_residual, S_INDEX, DROPPED_KEYS
using Riemann35: envparam, print_run_header
# `using MPI` is EXPLICIT here. Every test file is included into the same Main, so a
# file that omits it still works as long as some earlier include did `using MPI` --
# an order-dependent coupling that breaks the moment the file is run on its own, or
# the include order changes. See issue #62.
using MPI
MPI.Initialized() || MPI.Init()

const KNH   = [parse(Float64, s) for s in split(envparam("PR_KNH", "1.0,0.5,0.2,0.1"), ",")]
const CPL   = parse(Float64, envparam("PR_CPL", "6.0"))
const UW    = parse(Float64, envparam("PR_UW", "0.1"))
const ORDER = parse(Int,     envparam("PR_ORDER", "2"))
const TENDF = parse(Float64, envparam("PR_TEND", "0.3"))

"standardized 35-vector of a raw 35-vector"
stdz(Mv) = (M2CS4_35(collect(Float64, Mv)))[2]

print_run_header("CLOSURE RESIDUAL OF THE 26-MOMENT REDUCTION IN PLANAR COUETTE";
                 extra = ("collision" => "ES-BGK, Pr=2/3, omega=0.81", "walls" => "fully diffuse"))
println("residual = ||R(S)-S||/||S|| over the NINE dropped slots. R is reimplemented from the")
println("notes' formulas (the original moment_reduce26.jl is missing from the repo).")
println("="^112)

# ---- equilibrium sanity gate ----
# The gate must be ABSOLUTE, not relative. At equilibrium all nine dropped moments are
# zero, so ||R(S)-S||/||S|| is 0/0: with S at roundoff (~1e-17) and R predicting exactly
# 0, the relative residual comes out at 1.0 and looks like a catastrophic failure while
# the absolute agreement is perfect. A relative error is meaningless about a quantity
# whose true value is zero.
let Meq = collect(Float64, InitializeM4_35(1.0, 0.3, -0.2, 0.1, 1.0,0,0, 1.0,0, 1.0))
    S  = stdz(Meq)
    R  = reduce26_S(S)
    mx  = maximum(abs(S[S_INDEX[k]]) for k in DROPPED_KEYS)
    mxa = maximum(abs(R[S_INDEX[k]] - S[S_INDEX[k]]) for k in DROPPED_KEYS)
    @printf("GATE  Maxwellian: max|S_dropped| = %.3e, max ABSOLUTE closure error = %.3e  %s\n\n",
            mx, mxa, (mx < 1e-12 && mxa < 1e-12) ? "PASS" : "FAIL")
end

@printf("%6s %8s %11s %11s %11s %11s %11s %9s\n",
        "Kn_H", "where", "S310 meas", "S310 pred", "S130 meas", "S130 pred", "residual", "drift")
flush(stdout)
for KnH in KNH
    col, ny, nst, want, drift, kn_tau, cpl_act =
        couette_field(KnH; cpl = CPL, Uw = UW, order = ORDER, tendf = TENDF)
    nst < want && @printf("# Kn_H=%.3f TRUNCATED %d/%d steps -- UNCONVERGED\n", KnH, nst, want)

    # worst-cell residual over the whole column, plus the two sample points
    worst = 0.0; worst_j = 0
    for j in 1:ny
        r, _ = reduce26_residual(stdz(col[j]))
        r > worst && (worst = r; worst_j = j)
    end
    for (tag, j) in (("wall", 1), ("centre", max(1, ny÷2)))
        S = stdz(col[j])
        R = reduce26_S(S)
        r, _ = reduce26_residual(S)
        @printf("%6.3f %8s %11.3e %11.3e %11.3e %11.3e %11.3e %9.1e\n",
                KnH, tag,
                S[S_INDEX[(3,1,0)]], R[S_INDEX[(3,1,0)]],
                S[S_INDEX[(1,3,0)]], R[S_INDEX[(1,3,0)]],
                r, drift)
        flush(stdout)
    end
    @printf("%6.3f %8s %11s %11s %11s %11s %11.3e  (cell %d of %d)\n",
            KnH, "WORST", "", "", "", "", worst, worst_j, ny)
    flush(stdout)
end
println("="^112)
println("READING THIS. residual << 1 means the nine dropped moments are SLAVED to the retained")
println("ones and the reduction reproduces them: it is a change of variables, not a loss of")
println("physics, and Fox's micro-flow concern does not bite in shear-driven flow. residual = O(1)")
println("means they carry independent content the closure cannot recover, and the reduction is a")
println("real modelling approximation whose cost has to be measured in the Knudsen layer.")
println("Either way the magnitude result alone (S_odd ~ 3s) was never the deciding evidence.")
