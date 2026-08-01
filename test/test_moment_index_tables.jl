# test_moment_index_tables.jl -- the 35-moment index table is duplicated NINE times, and
# every copy must match the canonical one in src/moments/moment_indices.jl.
#
# WHY THIS EXISTS. The table has already been transcribed wrongly once: a copy permuted
# positions 31-33, reading (0,1,2),(1,1,2),(0,3,1) where canonical is (0,3,1),(0,1,2),(1,1,2).
# The multiset was right, so no moment went missing and every total and trace stayed correct
# -- three moments were simply MISLABELLED. Nothing published was affected only because every
# wall observable uses indices <= 30; the three permuted entries are exactly the odd
# fourth-order cross-moments that moment_reduce26.jl drops, so a reduced-26 comparison run
# through that copy would have been silently wrong. It surfaced by accident, when a heat-flux
# estimator built on index 32 returned exactly -u on a pure Maxwellian where the answer had
# to be 0.
#
# The existing guard (validate_dvm_gpu.jl) checks ONE pair. Nine hand-maintained copies with
# one pairwise check is a recurrence waiting to happen.
#
# THIS READS SOURCE TEXT rather than importing values, deliberately. The failure mode is
# somebody editing one copy, which is a textual event; and several copies live in scripts and
# in GPU modules that cannot be loaded without CUDA, so a value-comparison test would silently
# skip exactly the copies most likely to drift.
#
# THE REAL FIX is to delete eight copies and import one -- `IJK` is already exported from
# MomentIndices. This test is a stopgap that closes the recurrence path; it is not a substitute
# for the deduplication, and it should be deleted along with the duplicates.
using Test

const REPO = normpath(joinpath(@__DIR__, ".."))

"Extract the 35 (i,j,k) triples following `const <name> =` in a source file."
function parse_table(relpath::String, name::String)
    path = joinpath(REPO, relpath)
    isfile(path) || return nothing
    src = read(path, String)
    m = match(Regex("const\\s+" * name * "\\s*=\\s*"), src)
    m === nothing && return nothing
    j = m.offset + length(m.match)
    open = src[j]
    close = open == '(' ? ')' : ']'
    depth = 0; stop = j
    for k in j:lastindex(src)
        c = src[k]
        c == open  && (depth += 1)
        if c == close
            depth -= 1
            depth == 0 && (stop = k; break)
        end
    end
    [(parse(Int,t[1]), parse(Int,t[2]), parse(Int,t[3]))
     for t in [m.captures for m in eachmatch(r"\(\s*(\d)\s*,\s*(\d)\s*,\s*(\d)\s*\)", src[j:stop])]]
end

const COPIES = [
    ("src/reference/dvm_bgk.jl",                  "IJK35"),
    ("src/numerics/moment_correction_minnorm.jl", "_MN_IJK35"),
    ("src/numerics/logjacobi_recon_dev.jl",       "_IJK35"),
    ("src/numerics/kfvs_wall_dev.jl",             "IJK35_W"),
    ("src/numerics/kfvs_wall.jl",                 "IJK35_KFVS"),
    ("gpu/dvm_bgk_gpu.jl",                        "IJK35"),
    ("gpu/validation/validate_esbgk_gpu.jl",      "IJK35"),
    ("gpu/validation/verify_theta_closed_gpu.jl", "IJK"),
]

@testset "35-moment index tables all match canonical" begin
    canon = parse_table("src/moments/moment_indices.jl", "IJK")
    # assert the canonical parsed BEFORE comparing anything against it: a test that silently
    # compares an empty reference to an empty copy passes while checking nothing. An earlier
    # ad-hoc version of this audit reported "ALL FOUR TABLES AGREE" having found only three.
    @test canon !== nothing
    @test length(canon) == 35
    @test allunique(canon)
    @test all(sum(t) <= 4 for t in canon)      # M4 closure: no moment above fourth order

    for (relpath, name) in COPIES
        @testset "$relpath :: $name" begin
            t = parse_table(relpath, name)
            @test t !== nothing                # a renamed/moved copy must fail loudly
            if t !== nothing
                @test length(t) == 35
                @test t == canon
                if t != canon && length(t) == 35
                    for i in 1:35
                        t[i] == canon[i] || @info "position $i" canonical=canon[i] found=t[i]
                    end
                end
            end
        end
    end
end
