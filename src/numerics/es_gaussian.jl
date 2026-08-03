module ESGaussian
# es_gaussian.jl -- the CORRELATED ES-BGK equilibrium, shared by the GPU kernel and the CPU
# path. Pure scalar arithmetic, no allocation, no dynamic dispatch, so it GPU-compiles
# directly and runs unchanged on the CPU.
#
# WHY THIS EXISTS (issue #71). `_collide_es_kernel!` built its equilibrium as a PRODUCT of
# three independent 1D fits:
#
#     feq = A*exp(bx*vx + cx*vx^2) * exp(by*vy + cy*vy^2) * exp(bz*vz + cz*vz^2)
#
# A product of 1D Gaussians has a strictly DIAGONAL covariance. The ES-BGK equilibrium is
#
#     Lambda = (1-k) Theta I + k C
#
# whose off-diagonal entries are k*C_ij. Product form cannot hold them, so feq carried
# sigma_xy = 0 and the update degenerated to sigma_xy*exp(-Pr*y): shear stress relaxing at
# Pr/tau while diagonal stress relaxed at 1/tau. Measured split at Pr=2/3: 0.666667 against
# 1.000000, both fits R2 = 1.00000.
#
# WHAT k IS FOR, since it looks arbitrary. `_kappa_es` is constructed so that
# k + (1-k)exp(-Pr*y) = exp(-y), which forces the DIAGONAL deviatoric stress to relax at
# exactly 1/tau despite the exp(-Pr*y) factor. Giving the off-diagonal the same k makes it
# relax as sigma_xy*[k + (1-k)e] = sigma_xy*exp(-y) -- the same rate. So the formula already
# in the code is right; only the off-diagonal entries were being dropped.
#
# THE FIT, and why it is a fixed point rather than a 9x9 Newton. A general correlated
# exponential fit has 9 parameters (b, and symmetric C) and its Newton Jacobian needs third-
# and fourth-order moment sums over the full n^3 grid -- ~35 accumulators per iteration, and
# n^3 = 2.1e6 at nv=128. But the exponential family here is exactly Gaussian, so the map
# between parameters and moments is known in closed form:
#
#     g = exp(b.v + v'Cv)   =>   Sigma = (-2C)^-1,   mu = Sigma b
#
# Given the measured discrete moments (m, Lm) and the targets (u, Lt), the update
#
#     C <- C + (Lm^-1 - Lt^-1)/2,       b <- b - Lm^-1 m + Lt^-1 u
#
# is EXACT in one step in the continuum limit (substitute Lm^-1 = -2C to get C = -Lt^-1/2)
# and converges in a couple of iterations discretely, because the grid correction is small.
# It needs only the 9 first- and second-order moments and two 3x3 inverses -- no 4th moments,
# no 9x9 solve. That is what makes the correlated form affordable inside the kernel.
export sym3_inv, es_kappa, es_lambda, es_seed, es_refine, es_logw

"Inverse of a symmetric 3x3 given its upper triangle; returns the upper triangle and det."
@inline function sym3_inv(a11::Float64, a12::Float64, a13::Float64,
                          a22::Float64, a23::Float64, a33::Float64)
    c11 = a22*a33 - a23*a23
    c12 = a13*a23 - a12*a33
    c13 = a12*a23 - a13*a22
    det = a11*c11 + a12*c12 + a13*c13
    # A covariance that has gone singular is a failed state, not a small number: signal it
    # rather than dividing and propagating Inf into the exponent.
    (abs(det) < 1e-300) && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    id = 1.0/det
    (c11*id, c12*id, c13*id,
     (a11*a33 - a13*a13)*id, (a13*a12 - a11*a23)*id, (a11*a22 - a12*a12)*id, det)
end

"""
    es_kappa(Pr, y) -> k

The ES anisotropy weight, chosen so `k + (1-k)exp(-Pr*y) = exp(-y)`: deviatoric stress then
relaxes at exactly `1/tau` for EVERY component, diagonal and off-diagonal alike, which is
this codebase's viscous-time convention (`stage_bgk` does the same, and a Couette core fit
recovers DSMC's own `Kn_mu = lambda_mu/H` to 0.8%).
"""
@inline function es_kappa(Pr::Float64, y::Float64)
    Pr == 1.0 && return 0.0
    -exp(-Pr*y) * expm1((Pr - 1.0)*y) / expm1(-Pr*y)
end

"ES target covariance `Lambda = (1-k) Theta I + k C`, upper triangle, INCLUDING off-diagonals."
@inline function es_lambda(k::Float64, Theta::Float64,
                           cxx::Float64, cyy::Float64, czz::Float64,
                           cxy::Float64, cxz::Float64, cyz::Float64)
    om = 1.0 - k
    (om*Theta + k*cxx, k*cxy, k*cxz,
     om*Theta + k*cyy, k*cyz, om*Theta + k*czz)
end

"""
    es_seed(l11,l12,l13,l22,l23,l33, ux,uy,uz) -> (b1,b2,b3, c11,c12,c13,c22,c23,c33, ok)

Continuum quadratic parameters for a Gaussian of covariance `Lambda` and mean `u`:
`C = -Lambda^-1/2`, `b = Lambda^-1 u`. This is the fixed point's starting guess, and it is
already correct to the grid's discretisation error -- which is what keeps `es_refine` to a
couple of iterations instead of the forty a cold-started Newton needs.
"""
@inline function es_seed(l11::Float64, l12::Float64, l13::Float64,
                         l22::Float64, l23::Float64, l33::Float64,
                         ux::Float64, uy::Float64, uz::Float64)
    i11, i12, i13, i22, i23, i33, det = sym3_inv(l11, l12, l13, l22, l23, l33)
    det == 0.0 && return (0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, false)
    b1 = i11*ux + i12*uy + i13*uz
    b2 = i12*ux + i22*uy + i23*uz
    b3 = i13*ux + i23*uy + i33*uz
    (b1, b2, b3, -0.5i11, -0.5i12, -0.5i13, -0.5i22, -0.5i23, -0.5i33, true)
end

"""
    es_refine(b..., c..., measured mean/cov, target mean/cov) -> updated (b..., c..., ok)

One fixed-point correction. Exact in one step in the continuum limit; discretely it removes
the grid's moment error. See the module header for the derivation.
"""
@inline function es_refine(b1::Float64, b2::Float64, b3::Float64,
                           c11::Float64, c12::Float64, c13::Float64,
                           c22::Float64, c23::Float64, c33::Float64,
                           m1::Float64, m2::Float64, m3::Float64,
                           p11::Float64, p12::Float64, p13::Float64,
                           p22::Float64, p23::Float64, p33::Float64,
                           u1::Float64, u2::Float64, u3::Float64,
                           t11::Float64, t12::Float64, t13::Float64,
                           t22::Float64, t23::Float64, t33::Float64)
    q11, q12, q13, q22, q23, q33, dq = sym3_inv(p11, p12, p13, p22, p23, p33)   # measured^-1
    r11, r12, r13, r22, r23, r33, dr = sym3_inv(t11, t12, t13, t22, t23, t33)   # target^-1
    (dq == 0.0 || dr == 0.0) && return (b1,b2,b3, c11,c12,c13,c22,c23,c33, false)
    nb1 = b1 - (q11*m1 + q12*m2 + q13*m3) + (r11*u1 + r12*u2 + r13*u3)
    nb2 = b2 - (q12*m1 + q22*m2 + q23*m3) + (r12*u1 + r22*u2 + r23*u3)
    nb3 = b3 - (q13*m1 + q23*m2 + q33*m3) + (r13*u1 + r23*u2 + r33*u3)
    (nb1, nb2, nb3,
     c11 + 0.5*(q11 - r11), c12 + 0.5*(q12 - r12), c13 + 0.5*(q13 - r13),
     c22 + 0.5*(q22 - r22), c23 + 0.5*(q23 - r23), c33 + 0.5*(q33 - r33), true)
end

"Unnormalised log-weight of the correlated Gaussian at velocity `v` -- the ONLY place the
 quadratic form is written out, so the GPU kernel and the CPU path cannot disagree on it."
@inline function es_logw(vx::Float64, vy::Float64, vz::Float64,
                         b1::Float64, b2::Float64, b3::Float64,
                         c11::Float64, c12::Float64, c13::Float64,
                         c22::Float64, c23::Float64, c33::Float64)
    b1*vx + b2*vy + b3*vz +
    c11*vx*vx + c22*vy*vy + c33*vz*vz +
    2.0*(c12*vx*vy + c13*vx*vz + c23*vy*vz)
end

end # module
