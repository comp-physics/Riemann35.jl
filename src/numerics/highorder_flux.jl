"""
    realize_and_speed(M, axis, Ma)

Hyperbolicity-correct M for the given axis and return (Mr, vpmin, vpmax) with the
combined 6x6 + 1D-closure wave speeds, matching the interior flux path.
"""
function realize_and_speed(M::AbstractVector, axis::Int, Ma::Real)
    if axis == 1
        v6min, v6max, Mr = eigenvalues6_hyperbolic_3D(M, 1, 0, Ma)
        _, v5min, v5max = closure_and_eigenvalues(Mr[MomentIndices.MARG_VEC[1]])
    elseif axis == 2
        v6min, v6max, Mr = eigenvalues6_hyperbolic_3D(M, 2, 0, Ma)
        _, v5min, v5max = closure_and_eigenvalues(Mr[MomentIndices.MARG_VEC[2]])
    else
        v6min, v6max, Mr = eigenvalues6z_hyperbolic_3D(M, 0, Ma)
        _, v5min, v5max = closure_and_eigenvalues(Mr[MomentIndices.MARG_VEC[3]])
    end
    return Mr, min(v5min, v6min), max(v5max, v6max)
end

"Physical flux (length 35) of moment vector M in the given axis direction."
function _phys_flux(M::AbstractVector, axis::Int)
    Fx, Fy, Fz = Flux_closure35_3D(M)
    return axis == 1 ? Fx : (axis == 2 ? Fy : Fz)
end

"""
Interface-flux (Riemann-solver) selector. `:hll` (default) is the original, validated
two-wave HLL flux (byte-identical). `:rusanov` is a robust, more diffusive local
Lax–Friedrichs flux. These are the only two supported solvers: HLLC/HLLEM/kinetic were
removed — for the nonlinear 35-moment HyQMOM closure they failed or were blocked (no net
benefit over HLL; see `docs/riemann-solver-scope.md`). Set from `simulation_runner` via
the `riemann_solver` param, or directly (`Riemann35.RIEMANN_SOLVER[] = :rusanov`). OPT-IN:
anything other than `:hll` must be requested explicitly.
"""
const RIEMANN_SOLVER = Ref{Symbol}(:hll)

"""
    face_flux_1d(M_L, M_R, axis, Ma)

Interface flux from left/right face moment states. Each side is projected
(realizable_3D_M4) and hyperbolicity-corrected before fluxing. The flux formula is
chosen by `RIEMANN_SOLVER[]` (default `:hll`, byte-identical to the original scheme).
"""
function face_flux_1d(M_L::AbstractVector, M_R::AbstractVector, axis::Int, Ma::Real;
                      s3max::Real = 4.0 + abs(Ma) / 2.0)
    ML = realizable_3D_M4(M_L, Ma, s3max)
    MR = realizable_3D_M4(M_R, Ma, s3max)
    MLr, lminL, lmaxL = realize_and_speed(ML, axis, Ma)
    MRr, lminR, lmaxR = realize_and_speed(MR, axis, Ma)
    FL = _phys_flux(MLr, axis)
    FR = _phys_flux(MRr, axis)
    sL = min(lminL, lminR)
    sR = max(lmaxL, lmaxR)
    rs = RIEMANN_SOLVER[]
    if rs === :hll
        if sL >= 0
            return FL
        elseif sR <= 0
            return FR
        else
            return (sR .* FL .- sL .* FR .+ (sL*sR) .* (MRr .- MLr)) ./ (sR - sL)
        end
    elseif rs === :rusanov
        # local Lax–Friedrichs (Rusanov): robust, more diffusive than HLL.
        a = max(abs(sL), abs(sR))
        return 0.5 .* (FL .+ FR) .- 0.5a .* (MRr .- MLr)
    elseif rs === :roeps3
        # parity-split Roe (RoePS3): wave-resolved |λ| on the reflection-odd
        # sector of the face-normal marginal block, constant q(u) on the even
        # sector, scalar sector coefficients on the remaining components.
        # Uniform-pressure contacts are preserved exactly (parity theorem);
        # single source with the GPU path: src/numerics/roeps3_dev.jl.
        D = roeps3_diss_dev(NTuple{35,Float64}(MLr), NTuple{35,Float64}(MRr),
                            axis, Float64(sL), Float64(sR))
        return 0.5 .* (FL .+ FR) .- 0.5 .* collect(D)
    else
        throw(ArgumentError("unknown riemann_solver=$(rs); available: :hll (default), :rusanov, :roeps3"))
    end
end

"""
    residual_1d(Mline, dx, Ma; order=2, bc=:outflow, use_limiter=false)

Method-of-lines spatial residual for a 1D row of 35-moment cells (Ncell x 35) in
the x-direction. order=1: first-order (cell-centered). order=2: MUSCL on the
bounded reconstruction variables, with local fallback to first order if a
reconstructed face has nonpositive density.

bc=:outflow (default): zero-gradient boundary conditions — boundary cells i=1 and
  i=Nc receive zero residual (no net flux through the domain walls).
bc=:periodic: wrap neighbor indices so the domain is periodic. All Nc interfaces
  i+1/2 (i=1..Nc, with i+1 wrapping) are computed and every cell gets a residual.

use_limiter=false (default): existing muscl_faces + recon_face_pair path (byte-identical
  to the pre-existing behavior). use_limiter=true: order==2 faces built with
  scaling_limited_faces instead; faces are realizable by construction so no fallback
  is needed. The order==1 path is unaffected by this flag.
"""
function residual_1d(Mline::AbstractMatrix, dx::Real, Ma::Real;
                     order::Int=2, bc::Symbol=:outflow, use_limiter::Bool=false)
    Nc = size(Mline, 1)
    axis = 1
    R = zeros(Nc, 35)

    # order-2 face reconstruction uses the SHARED per-interface helpers
    # (recon_faces_{default,limited}, src/numerics/reconstruction.jl) — same ones
    # `residual_line` uses; only the BC-correct stencil (wrapped / clamped) differs here.
    if bc == :periodic
        wrap(i) = mod(i-1, Nc) + 1
        # Face states at interface i+1/2 for i=1..Nc (i+1 wraps)
        ML = [zeros(35) for _ in 1:Nc]
        MR = [zeros(35) for _ in 1:Nc]
        if order == 1
            for i in 1:Nc
                ML[i] = Mline[i, :]; MR[i] = Mline[wrap(i+1), :]
            end
        else
            Vc = [to_recon_vars(@view Mline[i, :]) for i in 1:Nc]
            for i in 1:Nc
                vm1, v0, vp1, vp2 = Vc[wrap(i-1)], Vc[i], Vc[wrap(i+1)], Vc[wrap(i+2)]
                ML[i], MR[i] = use_limiter ?
                    recon_faces_limited(vm1, v0, vp1, vp2) :
                    recon_faces_default(vm1, v0, vp1, vp2, Mline[i, :], Mline[wrap(i+1), :])
            end
        end
        Fhat = [face_flux_1d(ML[i], MR[i], axis, Ma) for i in 1:Nc]
        for i in 1:Nc
            R[i, :] = -(Fhat[i] .- Fhat[wrap(i-1)]) ./ dx
        end
    elseif bc == :outflow  # zero-gradient BCs
        # Right-face L/R moment states at each interface i+1/2, i=1..Nc-1
        ML = [zeros(35) for _ in 1:Nc-1]   # left state at interface i+1/2 (from cell i)
        MR = [zeros(35) for _ in 1:Nc-1]   # right state at interface i+1/2 (from cell i+1)
        if order == 1
            for i in 1:Nc-1
                ML[i] = Mline[i, :]; MR[i] = Mline[i+1, :]
            end
        else
            Vc = [to_recon_vars(@view Mline[i, :]) for i in 1:Nc]
            for i in 1:Nc-1
                vm1, v0, vp1, vp2 = Vc[max(i-1,1)], Vc[i], Vc[i+1], Vc[min(i+2,Nc)]
                ML[i], MR[i] = use_limiter ?
                    recon_faces_limited(vm1, v0, vp1, vp2) :
                    recon_faces_default(vm1, v0, vp1, vp2, Mline[i, :], Mline[i+1, :])
            end
        end
        Fhat = [face_flux_1d(ML[i], MR[i], axis, Ma) for i in 1:Nc-1]
        for i in 2:Nc-1
            R[i, :] = -(Fhat[i] .- Fhat[i-1]) ./ dx
        end
        # zero-gradient BC: no net flux at the physical boundary cells (i=1, i=Nc remain zero)
    else
        throw(ArgumentError("residual_1d: unknown bc=$bc (use :outflow or :periodic)"))
    end
    return R
end
