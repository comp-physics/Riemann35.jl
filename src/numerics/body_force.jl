"""
    body_force.jl — uniform body force on a 35-moment state, applied EXACTLY.

A spatially uniform acceleration `g` enters the kinetic equation as `-g . grad_v f`. Its
solution operator over a step `dt` is a pure translation in velocity space,

    f(v, t+dt) = f(v - g*dt, t),

so the moment update is not an approximation of any kind: every CENTRAL moment is
invariant and only the mean velocity shifts,

    u -> u + g*dt,   C_{ijk} unchanged.

Two reasons this form was chosen over the obvious one:

* **Exact, and closure-free.** The naive route adds the source `g_a * i_a * M_{..i_a-1..}`
  to each moment equation and integrates it explicitly: correct to O(dt) only, and it
  couples every moment to a lower one. The translation form is exact for any `dt`.
* **Realizability-preserving by construction.** A velocity-space translation is a
  measure-preserving bijection, so it carries a realizable moment vector to a realizable
  one with no projection, limiter, or gate — the same argument that makes specular wall
  reflection exact in `wall_ghost_dev.jl`. An explicit source term carries no such
  guarantee and can eject a near-boundary state from the cone.

Scope: UNIFORM `g` only. A non-uniform force (self-consistent electric field, gravity with
a gradient) is not a rigid translation and would need the source form plus a realizability
backstop. That is the plasma case, and it is deliberately out of scope here.
"""

# Linear indices of the 35 canonical moments inside the 5x5x5 array that C4toM4_3D
# returns; identical to the extraction list in M2CS4_35.
const _BF_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 21,
                 26, 27, 28, 29, 51, 52, 53, 76, 77, 101, 31, 32, 33, 36, 37,
                 41, 56, 57, 81, 61]

"""
    body_force_shift(M, gx, gy, gz, dt) -> Vector{Float64}

35-moment vector after a uniform acceleration `(gx,gy,gz)` acting for `dt`. Exact: central
moments preserved, mean velocity shifted by `g*dt`.
"""
function body_force_shift(M::AbstractVector{Float64}, gx, gy, gz, dt)
    rho = M[1]
    rho > 0 || return collect(Float64, M)
    C = M4toC4_3D(M...)
    u = M[2]/rho  + gx*dt
    v = M[6]/rho  + gy*dt
    w = M[16]/rho + gz*dt
    # Rebuild raw moments about the SHIFTED mean from the UNCHANGED central moments.
    A = C4toM4_3D(rho, u, v, w,
        C[3,1,1], C[2,2,1], C[2,1,2], C[1,3,1], C[1,2,2], C[1,1,3],
        C[4,1,1], C[3,2,1], C[3,1,2], C[2,3,1], C[2,2,2], C[2,1,3],
        C[1,4,1], C[1,3,2], C[1,2,3], C[1,1,4],
        C[5,1,1], C[4,2,1], C[4,1,2], C[3,3,1], C[3,2,2], C[3,1,3],
        C[2,4,1], C[2,3,2], C[2,2,3], C[2,1,4],
        C[1,5,1], C[1,4,2], C[1,3,3], C[1,2,4], C[1,1,5])
    [A[i] for i in _BF_IDX]
end

"""
    apply_body_force!(M, gx, gy, gz, dt, nx, ny, nz, halo)

In-place uniform body force over the interior cells of a 4-D moment field.
"""
function apply_body_force!(M::Array{Float64,4}, gx, gy, gz, dt, nx, ny, nz, halo)
    (gx == 0 && gy == 0 && gz == 0) && return M
    @inbounds for k in 1:nz, j in (halo+1):(halo+ny), i in (halo+1):(halo+nx)
        out = body_force_shift(M[i,j,k,:], gx, gy, gz, dt)
        for q in 1:35; M[i,j,k,q] = out[q]; end
    end
    M
end
