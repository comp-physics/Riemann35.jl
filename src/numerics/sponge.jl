# sponge.jl — absorbing (non-reflecting) boundary layer for the moment field.
#
# A `:sponge` face (see face_bc.jl) damps outgoing waves before they reach the
# domain edge by relaxing the moments toward a fixed freestream Maxwellian
# `Mref` over a zone of `width` cells. The relaxation rate ramps from 0 at the
# zone's inner edge to `rate` at the boundary face. The update is exact-
# exponential — the same integrator as the BGK collision (collision35) — so it
# is unconditionally stable and is a bit-for-bit no-op wherever the ramp is 0:
#
#   M <- Mref + (M - Mref) * exp(-rate * ramp(cell) * dt)
#
# Direction-agnostic: any of the six faces may be a sponge. A cell's ramp is the
# MAX over all sponge faces reaching it, so corners get the stronger absorption.
# The face itself is zero-gradient (:outflow) at the halo — waves leave through
# it — and the sponge zone absorbs the reflection.

"""
    build_sponge_ramp(faces, nx, ny, nz, width; power=2) -> Array{Float64,3}

Per-interior-cell absorption ramp in `[0,1]`: 1 at a `:sponge` face, decaying to
0 over `width` cells inward (raised to `power` for a smooth profile). Zero
everywhere if no face is a sponge. `faces` is an expanded six-face spec.
"""
function build_sponge_ramp(faces::NamedTuple, nx::Int, ny::Int, nz::Int, width::Int; power::Int=2)
    ramp = zeros(Float64, nx, ny, nz)
    (width <= 0) && return ramp
    n = (nx, ny, nz)
    for (a, lokey, hikey) in ((1, :xlo, :xhi), (2, :ylo, :yhi), (3, :zlo, :zhi))
        na = n[a]
        w = min(width, na)
        if faces[lokey] === :sponge
            for d in 1:w                              # d=1 is the face-adjacent cell (strongest)
                _rampmax_plane!(ramp, a, d, ((width - d + 1) / width)^power)
            end
        end
        if faces[hikey] === :sponge
            for d in 1:w
                _rampmax_plane!(ramp, a, na - d + 1, ((width - d + 1) / width)^power)
            end
        end
    end
    ramp
end

# Max-combine ramp value `r` into the plane at index `pos` along axis `a`.
@inline function _rampmax_plane!(ramp::Array{Float64,3}, a::Int, pos::Int, r::Float64)
    pl = selectdim(ramp, a, pos)
    @inbounds @. pl = max(pl, r)
    return ramp
end

"""
    apply_sponge!(M, ramp, Mref, rate, dt, halo)

Exact-exponential relaxation of the interior of a haloed moment array `M`
(interior cell `(i,j,k)` at `M[i+halo, j+halo, k, :]`) toward the freestream
`Mref`, using the precomputed `ramp`. No-op wherever `ramp == 0`.
"""
function apply_sponge!(M::AbstractArray{T,4}, ramp::Array{Float64,3}, Mref::AbstractVector,
                       rate::Real, dt::Real, halo::Int) where T
    nx, ny, nz = size(ramp)
    nv = size(M, 4)
    length(Mref) == nv || error("sponge Mref length $(length(Mref)) != nvar $nv")
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        r = ramp[i, j, k]
        r == 0.0 && continue
        f = exp(-rate * r * dt)
        ih = i + halo; jh = j + halo
        for m in 1:nv
            M[ih, jh, k, m] = Mref[m] + (M[ih, jh, k, m] - Mref[m]) * f
        end
    end
    return M
end
