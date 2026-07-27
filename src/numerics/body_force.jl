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
    body_force_shift_dev(M::NTuple{35,Float64}, gx, gy, gz, dt) -> NTuple{35,Float64}

Device-safe, allocation-free form of `body_force_shift`, for the GPU march and any other
alloc-free caller.

SAME OPERATOR, DIFFERENT ROUTE. `body_force_shift` goes through `M4toC4_3D` / `C4toM4_3D`,
which return 5x5x5 ARRAYS and therefore allocate — fine on the CPU, impossible on the
device. This form uses the recon-var round trip instead: `to_recon_vars_dev` returns
`(rho, u, v, w, C200, C020, C002, S300, ...)` in exactly the argument order
`from_recon_vars_dev` expects, so shifting the three mean-velocity slots and rebuilding is
the whole operation. Central and standardized moments are carried across untouched, which
is precisely the statement that a uniform force is a rigid velocity-space translation.

NOT expected to be bit-identical to `body_force_shift`: the two rebuild the raw moments by
different algebra, so they agree to roundoff, not to the last bit. The CPU path is left
alone rather than switched to this one, so existing byte-identity baselines still hold;
`test/test_body_force_dev.jl` measures the agreement instead of assuming it.
"""
@inline function body_force_shift_dev(M::NTuple{35,Float64}, gx::Float64, gy::Float64,
                                      gz::Float64, dt::Float64)::NTuple{35,Float64}
    # ZERO FORCE MUST BE EXACTLY THE IDENTITY, not the identity to roundoff. The recon-var
    # round trip is not exact, so without this guard a march that calls the body-force
    # kernel unconditionally would perturb EVERY run that has no body force, silently
    # breaking byte-identity against existing baselines. `apply_body_force!` has the same
    # short-circuit for the same reason. Caught by test_body_force_dev.jl.
    (gx == 0.0 && gy == 0.0 && gz == 0.0) && return M
    @inbounds (M[1] > 0.0) || return M
    # Arguments written out rather than splatted. `to_recon_vars_dev(M...)` is valid Julia
    # and works on the CPU, but SPLATTING DOES NOT LOWER ON THE DEVICE -- it emits
    # jl_f__apply_iterate and the kernel fails with InvalidIRError. That is why every other
    # device kernel here (e.g. _proj_interior!) spells out all 35 arguments.
    V = @inbounds to_recon_vars_dev(
        M[1],  M[2],  M[3],  M[4],  M[5],  M[6],  M[7],  M[8],  M[9],  M[10],
        M[11], M[12], M[13], M[14], M[15], M[16], M[17], M[18], M[19], M[20],
        M[21], M[22], M[23], M[24], M[25], M[26], M[27], M[28], M[29], M[30],
        M[31], M[32], M[33], M[34], M[35])
    @inbounds from_recon_vars_dev(
        V[1], V[2] + gx*dt, V[3] + gy*dt, V[4] + gz*dt,
        V[5],  V[6],  V[7],  V[8],  V[9],  V[10], V[11], V[12], V[13], V[14],
        V[15], V[16], V[17], V[18], V[19], V[20], V[21], V[22], V[23], V[24],
        V[25], V[26], V[27], V[28], V[29], V[30], V[31], V[32], V[33], V[34], V[35])
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
