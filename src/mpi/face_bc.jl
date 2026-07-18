# face_bc.jl — direction-agnostic per-face boundary-condition spec.
#
# The solver's boundary condition is a per-face assignment: each of the six
# faces (xlo, xhi, ylo, yhi, zlo, zhi) carries an independent type drawn from
#   :outflow  — zero-gradient (copy nearest interior cell)
#   :inlet    — Dirichlet Maxwellian (CROSSFLOW_INLET[]); low-x by convention
#   :periodic — wrap-around (must be set on BOTH faces of an axis)
#   :sponge   — absorbing layer: the HALO behaves as :outflow, and a spatially
#               ramped exact-exponential relaxation to a freestream Maxwellian
#               is applied in the interior near the face (see sponge.jl). This
#               is the non-reflecting boundary.
#
# `bc` (a params field / kwarg) may be a preset Symbol or an explicit NamedTuple
# of face types; `expand_bc` canonicalizes either into the six-face NamedTuple.
# The legacy presets :copy and :crossflow expand to specs whose halo behavior is
# byte-for-byte identical to the hand-written branches they replace.

const FACE_KEYS     = (:xlo, :xhi, :ylo, :yhi, :zlo, :zhi)
const FACE_BC_TYPES = (:outflow, :inlet, :periodic, :sponge)

# Preset -> canonical six-face spec.
const _BC_PRESETS = Dict{Symbol,NamedTuple}(
    # legacy: zero-gradient everywhere (default; only x/y have CPU halos)
    :copy => (xlo=:outflow, xhi=:outflow, ylo=:outflow, yhi=:outflow, zlo=:outflow, zhi=:outflow),
    # dense-bubble crossflow: inlet-x / outflow-x / periodic-y / outflow-z
    :crossflow => (xlo=:inlet, xhi=:outflow, ylo=:periodic, yhi=:periodic, zlo=:outflow, zhi=:outflow),
    # crossflow with the periodic-y box mode replaced by absorbing (sponge) y-faces
    :crossflow_absorb_y => (xlo=:inlet, xhi=:outflow, ylo=:sponge, yhi=:sponge, zlo=:outflow, zhi=:outflow),
)

"""
    expand_bc(bc) -> NamedTuple{FACE_KEYS}

Canonicalize a boundary-condition spec into a six-face NamedTuple of face types.
`bc` is either a preset `Symbol` (`:copy`, `:crossflow`, `:crossflow_absorb_y`)
or a NamedTuple giving some/all faces explicitly (missing faces default to
`:outflow`). Validates the face types and the periodic-pairing invariant.
"""
function expand_bc(bc)::NamedTuple{FACE_KEYS}
    spec = bc isa Symbol ? get(_BC_PRESETS, bc, nothing) : bc
    spec === nothing && error("Unknown bc preset :$bc (known: $(sort(collect(keys(_BC_PRESETS)))))")
    faces = NamedTuple{FACE_KEYS}(ntuple(i -> get(spec, FACE_KEYS[i], :outflow), length(FACE_KEYS)))
    for k in FACE_KEYS
        faces[k] in FACE_BC_TYPES ||
            error("face $k has invalid BC type :$(faces[k]) (valid: $FACE_BC_TYPES)")
    end
    # periodicity is an axis property: both faces of an axis are periodic, or neither
    for (lo, hi) in ((:xlo, :xhi), (:ylo, :yhi), (:zlo, :zhi))
        (faces[lo] == :periodic) == (faces[hi] == :periodic) ||
            error("periodic BC must be on BOTH faces of an axis (got $lo=:$(faces[lo]), $hi=:$(faces[hi]))")
    end
    faces
end

# How a face type presents to the HALO refill: a sponge face is zero-gradient at
# the halo (its absorbing effect is an interior source, not a ghost value).
@inline halo_face_type(t::Symbol) = t === :sponge ? :outflow : t

# The faces (subset of FACE_KEYS) carrying an absorbing sponge layer.
sponge_faces(spec::NamedTuple) = Tuple(k for k in FACE_KEYS if spec[k] === :sponge)

# Whether any face is periodic / inlet / sponge (cheap predicates for dispatch).
has_periodic(spec::NamedTuple) = any(spec[k] === :periodic for k in FACE_KEYS)
has_inlet(spec::NamedTuple)    = any(spec[k] === :inlet    for k in FACE_KEYS)
has_sponge(spec::NamedTuple)   = any(spec[k] === :sponge   for k in FACE_KEYS)
