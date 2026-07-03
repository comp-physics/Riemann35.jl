"""
    moment_correction_dev.jl — hyperbolicity moment correction, single source.

The scalar/register form of `correct_moments_hyperbolic_3D` (zero the cross
third-order standardized moments, tie S112/S121/S211 to the second-order
correlations, floor S220/S202/S022 at 1/3, rebuild). Consumed by BOTH the GPU
wave-speed path (`gpu/wavespeed_dev.jl`) and the CPU
`correct_moments_hyperbolic_3D` (`src/numerics/eigenvalues6_hyperbolic_3D.jl`),
which delegates here since the 2026-07-03 unification (the previous CPU
formulation via the autogen conversions differed by ~1 ulp; the CI golden
battery is tolerance-based at 1e-10, so the unification is regression-safe —
verified by a 2000-state reassociation battery).

Device-safe plain Julia; no dependencies.
"""
module MomentCorrectionDev

export correct_moments_dev

# ---------------------------------------------------------------------------
# correct_moments_dev: hyperbolicity correction (port of
# correct_moments_hyperbolic_3D, src/numerics/eigenvalues6_hyperbolic_3D.jl).
# Computes the needed central moments (M4toC4_3D), standardizes (M2CS4_35 rules,
# eps-floored variances), zeros the cross 3rd-order standardized moments, floors
# S220/S202/S022 at 1/3, rebuilds central moments (0-floored variances), and
# converts back to raw moments (C4toM4_3D). Returns the 35 corrected raw moments
# in canonical order. @noinline to cap kernel register pressure.
# ---------------------------------------------------------------------------
@noinline function correct_moments_dev(
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
        m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35)
    begin
        M000 = m1
        u = m2/m1; v = m6/m1; w = m16/m1
        t8 = 1.0/m1; t9 = t8*t8; t10 = t9*t8; t11 = t9*t9
        # central moments (M4toC4_3D expressions, only those used downstream)
        C200 = m3*t8 - m2^2*t9
        C300 = m4*t8 + m2^3*t10*2.0 - m2*m3*t9*3.0
        C400 = m5*t8 - m2^4*t11*3.0 - m2*m4*t9*4.0 + m3*m2^2*t10*6.0
        C110 = m7*t8 - m6*m2*t9
        C310 = m9*t8 - m6*m4*t9 - m2*m8*t9*3.0 - m6*m2^3*t11*3.0 + m7*m2^2*t10*3.0 + m6*m2*m3*t10*3.0
        C020 = m10*t8 - m6^2*t9
        C220 = m12*t8 - m6*m8*t9*2.0 - m2*m11*t9*2.0 + m10*m2^2*t10 + m3*m6^2*t10 - m6^2*m2^2*t11*3.0 + m6*m2*m7*t10*4.0
        C030 = m13*t8 + m6^3*t10*2.0 - m6*m10*t9*3.0
        C130 = m14*t8 - m6*m11*t9*3.0 - m13*m2*t9 - m2*m6^3*t11*3.0 + m7*m6^2*t10*3.0 + m6*m10*m2*t10*3.0
        C040 = m15*t8 - m6^4*t11*3.0 - m6*m13*t9*4.0 + m10*m6^2*t10*6.0
        C101 = m17*t8 - m16*m2*t9
        C301 = m19*t8 - m16*m4*t9 - m2*m18*t9*3.0 - m16*m2^3*t11*3.0 + m17*m2^2*t10*3.0 + m16*m2*m3*t10*3.0
        C002 = m20*t8 - m16^2*t9
        C202 = m22*t8 - m16*m18*t9*2.0 - m2*m21*t9*2.0 + m20*m2^2*t10 + m3*m16^2*t10 - m16^2*m2^2*t11*3.0 + m16*m2*m17*t10*4.0
        C003 = m23*t8 + m16^3*t10*2.0 - m16*m20*t9*3.0
        C103 = m24*t8 - m16*m21*t9*3.0 - m23*m2*t9 + m17*m16^2*t10*3.0 - m2*m16^3*t11*3.0 + m16*m20*m2*t10*3.0
        C004 = m25*t8 - m16^4*t11*3.0 - m16*m23*t9*4.0 + m20*m16^2*t10*6.0
        C011 = m26*t8 - m16*m6*t9
        C111 = m27*t8 - m16*m7*t9 - m6*m17*t9 - m26*m2*t9 + m16*m6*m2*t10*2.0
        C031 = m31*t8 - m16*m13*t9 - m6*m29*t9*3.0 - m16*m6^3*t11*3.0 + m26*m6^2*t10*3.0 + m16*m6*m10*t10*3.0
        C013 = m34*t8 - m16*m32*t9*3.0 - m23*m6*t9 + m26*m16^2*t10*3.0 - m6*m16^3*t11*3.0 + m16*m20*m6*t10*3.0
        C022 = m35*t8 - m16*m29*t9*2.0 - m6*m32*t9*2.0 + m20*m6^2*t10 + m10*m16^2*t10 - m16^2*m6^2*t11*3.0 + m16*m6*m26*t10*4.0

        # standardized moments (M2CS4_35: variances eps-floored)
        EPS = 2.220446049250313e-16
        sC200 = sqrt(max(C200, EPS)); sC020 = sqrt(max(C020, EPS)); sC002 = sqrt(max(C002, EPS))
        S110 = C110/(sC200*sC020); S101 = C101/(sC200*sC002); S011 = C011/(sC020*sC002)
        S220 = C220/(sC200^2*sC020^2); S202 = C202/(sC200^2*sC002^2); S022 = C022/(sC020^2*sC002^2)

        # force hyperbolic set
        s22min = 1.0/3.0
        S112 = S110; S121 = S101; S211 = S011
        S220 = max(S220, s22min); S202 = max(S202, s22min); S022 = max(S022, s22min)

        # rebuild central moments (variances 0-floored, cross 3rd-order zeroed)
        sC200c = sqrt(max(0.0, C200)); sC020c = sqrt(max(0.0, C020)); sC002c = sqrt(max(0.0, C002))
        C112 = S112*sC200c*sC020c*C002
        C121 = S121*sC200c*C020*sC002c
        C211 = S211*C200*sC020c*sC002c
        C220 = S220*C200*C020
        C202 = S202*C200*C002
        C022 = S022*C020*C002

        # raw moments (C4toM4_3D; C210=C120=C201=C102=C021=C012=0)
        tu2 = u*u; tu3 = tu2*u; tv2 = v*v; tv3 = tv2*v; tw2 = w*w; tw3 = tw2*w
        o1  = M000
        o2  = M000*u
        o3  = M000*tu2 + C200*M000
        o4  = M000*tu3 + C300*M000 + C200*M000*u*3.0
        o5  = M000*tu2^2 + C400*M000 + C200*M000*tu2*6.0 + C300*M000*u*4.0
        o6  = M000*v
        o7  = C110*M000 + M000*u*v
        o8  = C110*M000*u*2.0 + C200*M000*v + M000*tu2*v
        o9  = C310*M000 + C110*M000*tu2*3.0 + C300*M000*v + M000*tu3*v + C200*M000*u*v*3.0
        o10 = M000*tv2 + C020*M000
        o11 = C020*M000*u + C110*M000*v*2.0 + M000*tv2*u
        o12 = C220*M000 + C020*M000*tu2 + C200*M000*tv2 + M000*tu2*tv2 + C110*M000*u*v*4.0
        o13 = M000*tv3 + C030*M000 + C020*M000*v*3.0
        o14 = C130*M000 + C110*M000*tv2*3.0 + C030*M000*u + M000*tv3*u + C020*M000*u*v*3.0
        o15 = M000*tv2^2 + C040*M000 + C020*M000*tv2*6.0 + C030*M000*v*4.0
        o16 = M000*w
        o17 = C101*M000 + M000*u*w
        o18 = C101*M000*u*2.0 + C200*M000*w + M000*tu2*w
        o19 = C301*M000 + C101*M000*tu2*3.0 + C300*M000*w + M000*tu3*w + C200*M000*u*w*3.0
        o20 = M000*tw2 + C002*M000
        o21 = C002*M000*u + C101*M000*w*2.0 + M000*tw2*u
        o22 = C202*M000 + C002*M000*tu2 + C200*M000*tw2 + M000*tu2*tw2 + C101*M000*u*w*4.0
        o23 = M000*tw3 + C003*M000 + C002*M000*w*3.0
        o24 = C103*M000 + C101*M000*tw2*3.0 + C003*M000*u + M000*tw3*u + C002*M000*u*w*3.0
        o25 = M000*tw2^2 + C004*M000 + C002*M000*tw2*6.0 + C003*M000*w*4.0
        o26 = C011*M000 + M000*v*w
        o27 = C111*M000 + C011*M000*u + C101*M000*v + C110*M000*w + M000*u*v*w
        o28 = C211*M000 + C011*M000*tu2 + C111*M000*u*2.0 + M000*tu2*v*w + C101*M000*u*v*2.0 + C110*M000*u*w*2.0 + C200*M000*v*w
        o29 = C011*M000*v*2.0 + C020*M000*w + M000*tv2*w
        o30 = C121*M000 + C101*M000*tv2 + C111*M000*v*2.0 + M000*tv2*u*w + C011*M000*u*v*2.0 + C020*M000*u*w + C110*M000*v*w*2.0
        o31 = C031*M000 + C011*M000*tv2*3.0 + C030*M000*w + M000*tv3*w + C020*M000*v*w*3.0
        o32 = C002*M000*v + C011*M000*w*2.0 + M000*tw2*v
        o33 = C112*M000 + C110*M000*tw2 + C111*M000*w*2.0 + M000*tw2*u*v + C002*M000*u*v + C011*M000*u*w*2.0 + C101*M000*v*w*2.0
        o34 = C013*M000 + C011*M000*tw2*3.0 + C003*M000*v + M000*tw3*v + C002*M000*v*w*3.0
        o35 = C022*M000 + C002*M000*tv2 + C020*M000*tw2 + M000*tv2*tw2 + C011*M000*v*w*4.0
        return (o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15,o16,o17,o18,
                o19,o20,o21,o22,o23,o24,o25,o26,o27,o28,o29,o30,o31,o32,o33,o34,o35)
    end
end

end # module
