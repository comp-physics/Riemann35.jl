# test_face_bc.jl — direction-agnostic per-face BC spec (expand_bc + helpers) and
# the byte-identical regression gate on the CPU halo refill.
using Test
using Riemann35: expand_bc, halo_face_type, sponge_faces, has_sponge, has_periodic,
                 has_inlet, FACE_KEYS, apply_physical_bc_3d!, CROSSFLOW_INLET,
                 build_sponge_ramp, apply_sponge!

# Verbatim copy of the pre-refactor apply_physical_bc_3d! (the byte-identical
# reference). If the new axis-generic refill ever diverges from this on :copy or
# :crossflow, the regression testset below fails.
function _legacy_bc!(A::Array{T,4}, decomp, bc::Symbol) where T
    h = decomp.halo; nx = decomp.local_size[1]; ny = decomp.local_size[2]; nz = decomp.local_size[3]
    h == 0 && return A
    if bc == :copy
        decomp.neighbors.left  == -1 && (for ih in 1:h; A[ih, :, :, :] .= view(A, h+1, :, :, :); end)
        decomp.neighbors.right == -1 && (for ih in 1:h; A[h+nx+ih, :, :, :] .= view(A, h+nx, :, :, :); end)
        decomp.neighbors.down  == -1 && (for ih in 1:h; A[:, ih, :, :] .= view(A, :, h+1, :, :); end)
        decomp.neighbors.up    == -1 && (for ih in 1:h; A[:, h+ny+ih, :, :] .= view(A, :, h+ny, :, :); end)
    elseif bc == :crossflow
        inlet = CROSSFLOW_INLET[]; nv = size(A, 4)
        if decomp.neighbors.left == -1
            @inbounds for m in 1:nv, ih in 1:h; A[ih, :, :, m] .= inlet[m]; end
        end
        decomp.neighbors.right == -1 && (for ih in 1:h; A[h+nx+ih, :, :, :] .= view(A, h+nx, :, :, :); end)
        if decomp.neighbors.down == -1 && decomp.neighbors.up == -1
            for ih in 1:h
                A[:, ih, :, :]      .= view(A, :, ny+ih, :, :)
                A[:, h+ny+ih, :, :] .= view(A, :, h+ih, :, :)
            end
        end
    end
    return A
end

_mock_decomp(h, nx, ny, nz) =
    (halo=h, local_size=(nx, ny, nz), neighbors=(left=-1, right=-1, down=-1, up=-1))

@testset "face BC spec" begin
    @testset "presets expand to the legacy specs" begin
        cp = expand_bc(:copy)
        @test all(cp[k] === :outflow for k in FACE_KEYS)

        cf = expand_bc(:crossflow)
        @test cf == (xlo=:inlet, xhi=:outflow, ylo=:periodic, yhi=:periodic, zlo=:outflow, zhi=:outflow)

        ab = expand_bc(:crossflow_absorb_y)
        @test ab == (xlo=:inlet, xhi=:outflow, ylo=:sponge, yhi=:sponge, zlo=:outflow, zhi=:outflow)
    end

    @testset "explicit NamedTuple: missing faces default to outflow" begin
        spec = expand_bc((xlo=:inlet, ylo=:sponge, yhi=:sponge))
        @test spec.xlo === :inlet
        @test spec.xhi === :outflow      # defaulted
        @test spec.ylo === :sponge && spec.yhi === :sponge
        @test spec.zlo === :outflow && spec.zhi === :outflow
    end

    @testset "validation" begin
        @test_throws ErrorException expand_bc(:not_a_preset)
        @test_throws ErrorException expand_bc((xlo=:bogus,))                 # invalid type
        @test_throws ErrorException expand_bc((ylo=:periodic, yhi=:outflow)) # unpaired periodic
        @test_throws ErrorException expand_bc((xlo=:periodic,))              # xhi defaults outflow -> unpaired
    end

    @testset "helpers" begin
        @test halo_face_type(:sponge) === :outflow      # sponge is zero-gradient at the halo
        @test halo_face_type(:inlet) === :inlet
        @test halo_face_type(:periodic) === :periodic

        ab = expand_bc(:crossflow_absorb_y)
        @test sponge_faces(ab) == (:ylo, :yhi)
        @test has_sponge(ab) && !has_periodic(ab) && has_inlet(ab)

        cf = expand_bc(:crossflow)
        @test has_periodic(cf) && !has_sponge(cf)
        @test sponge_faces(cf) == ()
    end

    @testset "CPU refill byte-identical to legacy (:copy, :crossflow)" begin
        h, nx, ny, nz, nv = 2, 6, 5, 3, 35
        dec = _mock_decomp(h, nx, ny, nz)
        CROSSFLOW_INLET[] = collect(1.0:nv) .* 0.1     # arbitrary 35-vector inlet
        for bc in (:copy, :crossflow)
            A0 = randn(h+nx+h, h+ny+h, nz, nv)
            Aref = copy(A0); Anew = copy(A0)
            _legacy_bc!(Aref, dec, bc)
            apply_physical_bc_3d!(Anew, dec, bc)
            @test Anew == Aref                         # exact bit-for-bit
        end
    end

    @testset "sponge face is :outflow at the halo (differs from periodic)" begin
        h, nx, ny, nz, nv = 2, 6, 5, 3, 35
        dec = _mock_decomp(h, nx, ny, nz)
        CROSSFLOW_INLET[] = collect(1.0:nv) .* 0.1
        A0 = randn(h+nx+h, h+ny+h, nz, nv)
        Aper = copy(A0); Aspg = copy(A0)
        apply_physical_bc_3d!(Aper, dec, :crossflow)             # y periodic
        apply_physical_bc_3d!(Aspg, dec, :crossflow_absorb_y)    # y sponge -> outflow halo
        # x faces identical (both inlet/outflow); y ghost differs (periodic wrap vs zero-grad)
        @test Aspg[1:h, :, :, :] == Aper[1:h, :, :, :]           # x-lo inlet same
        # y-lo ghost: sponge => copy interior row h+1 ; periodic => wrap top interior ny+ih
        for ih in 1:h
            @test Aspg[:, ih, :, :] == Aspg[:, h+1, :, :]        # zero-gradient
        end
        @test Aspg[:, 1, :, :] != Aper[:, 1, :, :]               # genuinely different BC
    end
    CROSSFLOW_INLET[] = nothing                                  # reset global

    @testset "sponge ramp geometry" begin
        nx, ny, nz = 8, 10, 2
        faces = expand_bc(:crossflow_absorb_y)                   # ylo, yhi sponge
        ramp = build_sponge_ramp(faces, nx, ny, nz, 3)
        @test size(ramp) == (nx, ny, nz)
        @test all(ramp[:, 1, :] .== 1.0)                         # face-adjacent row: strongest
        @test all(ramp[:, ny, :] .== 1.0)
        @test all(ramp[:, 5, :] .== 0.0)                         # interior: untouched
        @test all(ramp[:, 6, :] .== 0.0)
        @test ramp[1, 2, 1] ≈ (2/3)^2                            # ramp decays inward
        @test ramp[1, 3, 1] ≈ (1/3)^2
        # x faces are inlet/outflow (not sponge) => no x ramp
        @test all(ramp[1, 4:7, :] .== 0.0)
        # no sponge face => empty-of-effect ramp
        @test all(build_sponge_ramp(expand_bc(:crossflow), nx, ny, nz, 3) .== 0.0)
    end

    @testset "sponge exact-exp relaxation + no-op where ramp=0" begin
        nx, ny, nz, nv, halo = 4, 6, 2, 35, 2
        faces = expand_bc(:crossflow_absorb_y)
        ramp = build_sponge_ramp(faces, nx, ny, nz, 2)
        Mref = collect(1.0:nv)
        M = randn(halo+nx+halo, halo+ny+halo, nz, nv)
        M0 = copy(M)
        rate, dt = 7.0, 0.01
        apply_sponge!(M, ramp, Mref, rate, dt, halo)
        f = exp(-rate * dt)                                      # ramp=1 at the boundary row
        for m in 1:nv
            got = M[halo+1, halo+1, 1, m]                        # interior cell (1,1): y-face row, ramp=1
            @test got ≈ Mref[m] + (M0[halo+1, halo+1, 1, m] - Mref[m]) * f
        end
        # interior row (ramp=0) is byte-identical unchanged
        jmid = halo + 3
        @test M[halo+1:halo+nx, jmid, :, :] == M0[halo+1:halo+nx, jmid, :, :]
    end
end
