module ESCollide
# es_collide.jl -- CPU ES-BGK collision, built on the SAME scalar core as the GPU kernel.
#
# WHY A CPU PATH AT ALL. `collide_es!` was GPU-only, so the ES operator could not be exercised
# in CI at all -- which is part of why issue #71 (shear stress relaxing at Pr/tau instead of
# 1/tau) survived: the only tests that could touch it were local, and all of them initialised
# a DIAGONAL anisotropy that never excited the defect. A CPU path on the shared core makes the
# invariant testable everywhere.
#
# THE DRY LINE. Everything that is arithmetic -- the ES anisotropy weight, the target
# covariance including off-diagonals, the continuum seed, the fixed-point refinement, and the
# quadratic form itself -- lives in `ESGaussian` and is called verbatim by both this function
# and `_collide_es_kernel!`. What differs between them is ONLY the moment reduction: a serial
# loop here, a block-wide shared-memory reduction on the device. That is the one thing that
# genuinely cannot be shared, and it is deliberately the only thing that isn't.
#
# Direction matters and matches PR #72: the DEVICE-SAFE core is the source and both callers
# wrap it. Device code cannot call host code, so a host-side source would have to be re-ported
# by hand -- which is exactly how the wall flux drifted into two different physical models.
using ..ESGaussian: es_kappa, es_lambda, es_seed, es_refine, es_logw, sym3_inv
export collide_es_cpu!

"""
    collide_es_cpu!(f, vh, dv, dt, Kn, Pr, omega; iters=8, tol=1e-14)

In-place ES-BGK relaxation of `f[a,b,c,i]` on the velocity grid `vh` (spacing `dv`).
Semantics identical to `DVMBGKGPU.collide_es!`.

The equilibrium is a CORRELATED Gaussian: its covariance is
`Lambda = (1-k) Theta I + k C` with the FULL `C`, off-diagonals included. Dropping those was
issue #71 -- it made the equilibrium's shear stress identically zero, so `sigma_xy` decayed at
the bare `exp(-Pr*y)` rate while diagonal components decayed at `exp(-y)`.
"""
function collide_es_cpu!(f::AbstractArray{Float64,4}, vh::AbstractVector{Float64},
                         dv::Float64, dt::Float64, Kn::Float64, Pr::Float64, omega::Float64;
                         iters::Int = 8, tol::Float64 = 1e-14)
    n = length(vh); dv3 = dv^3
    Nx = size(f, 4)
    gw = Array{Float64}(undef, n, n, n)          # scratch for the equilibrium weights
    @inbounds for i in 1:Nx
        # ---- moments of the current state, INCLUDING off-diagonals --------------------------
        rho = 0.0; m1 = 0.0; m2 = 0.0; m3 = 0.0
        q11 = 0.0; q12 = 0.0; q13 = 0.0; q22 = 0.0; q23 = 0.0; q33 = 0.0
        for c in 1:n, b in 1:n, a in 1:n
            w = f[a,b,c,i]; w == 0.0 && continue
            vx = vh[a]; vy = vh[b]; vz = vh[c]
            rho += w
            m1 += w*vx; m2 += w*vy; m3 += w*vz
            q11 += w*vx*vx; q12 += w*vx*vy; q13 += w*vx*vz
            q22 += w*vy*vy; q23 += w*vy*vz; q33 += w*vz*vz
        end
        rho *= dv3; rho > 0.0 || continue
        ux = m1*dv3/rho; uy = m2*dv3/rho; uz = m3*dv3/rho
        cxx = q11*dv3/rho - ux*ux; cxy = q12*dv3/rho - ux*uy; cxz = q13*dv3/rho - ux*uz
        cyy = q22*dv3/rho - uy*uy; cyz = q23*dv3/rho - uy*uz; czz = q33*dv3/rho - uz*uz

        Theta = (cxx + cyy + czz)/3
        Theta = Theta > 1e-14 ? Theta : 1e-14
        tau = (Kn/2)*Theta^(omega - 1.0)/rho
        y = dt/tau
        y > 0.0 || continue

        k = es_kappa(Pr, y)
        l11, l12, l13, l22, l23, l33 = es_lambda(k, Theta, cxx, cyy, czz, cxy, cxz, cyz)
        b1, b2, b3, d11, d12, d13, d22, d23, d33, ok =
            es_seed(l11, l12, l13, l22, l23, l33, ux, uy, uz)
        ok || continue

        # ---- fixed-point refinement against the DISCRETE moments -----------------------------
        # The seed is the continuum answer; this removes the grid's own moment error so mass,
        # momentum and energy are conserved to round-off rather than to quadrature accuracy.
        for _ in 1:iters
            s0 = 0.0; s1 = 0.0; s2 = 0.0; s3 = 0.0
            p11 = 0.0; p12 = 0.0; p13 = 0.0; p22 = 0.0; p23 = 0.0; p33 = 0.0
            for c in 1:n, b in 1:n, a in 1:n
                vx = vh[a]; vy = vh[b]; vz = vh[c]
                w = exp(es_logw(vx, vy, vz, b1, b2, b3, d11, d12, d13, d22, d23, d33))
                gw[a,b,c] = w
                s0 += w; s1 += w*vx; s2 += w*vy; s3 += w*vz
                p11 += w*vx*vx; p12 += w*vx*vy; p13 += w*vx*vz
                p22 += w*vy*vy; p23 += w*vy*vz; p33 += w*vz*vz
            end
            (s0 > 0.0 && isfinite(s0)) || break
            e1 = s1/s0; e2 = s2/s0; e3 = s3/s0
            a11 = p11/s0 - e1*e1; a12 = p12/s0 - e1*e2; a13 = p13/s0 - e1*e3
            a22 = p22/s0 - e2*e2; a23 = p23/s0 - e2*e3; a33 = p33/s0 - e3*e3
            err = abs(e1-ux) + abs(e2-uy) + abs(e3-uz) +
                  abs(a11-l11) + abs(a12-l12) + abs(a13-l13) +
                  abs(a22-l22) + abs(a23-l23) + abs(a33-l33)
            err < tol && break
            b1, b2, b3, d11, d12, d13, d22, d23, d33, ok2 =
                es_refine(b1, b2, b3, d11, d12, d13, d22, d23, d33,
                          e1, e2, e3, a11, a12, a13, a22, a23, a33,
                          ux, uy, uz, l11, l12, l13, l22, l23, l33)
            ok2 || break
        end

        # ---- apply: f <- feq + (f - feq) e -------------------------------------------------
        s0 = 0.0
        for c in 1:n, b in 1:n, a in 1:n
            w = exp(es_logw(vh[a], vh[b], vh[c], b1, b2, b3, d11, d12, d13, d22, d23, d33))
            gw[a,b,c] = w; s0 += w
        end
        (s0 > 0.0 && isfinite(s0)) || continue
        A = rho/(s0*dv3)
        e = exp(-Pr*y)
        for c in 1:n, b in 1:n, a in 1:n
            feq = A*gw[a,b,c]
            f[a,b,c,i] = feq + (f[a,b,c,i] - feq)*e
        end
    end
    f
end

end # module
