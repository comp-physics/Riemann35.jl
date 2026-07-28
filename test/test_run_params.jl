# test_run_params.jl — the run-provenance helper must record what was actually read.
#
# This exists because a results header drifted out of step with the code it described.
# probe_poiseuille.jl printed g, cells-per-mfp and the order but not PS_TEND or PS_CFL, both
# ENV-configurable and both answer-changing; the published 26-moment flow-rate table was
# produced at PS_TEND=1.2 rather than the default 0.4, nothing recorded it, and the number
# turned out to be a transient sampled at a tenth of its settling time.
#
# The guarantee under test is therefore not "the header is long" but "the header is exactly
# what was read": a parameter cannot be read without being registered, and cannot be
# registered without being printed. These assertions are what keep that true.
using Test, Riemann35

@testset "run_params" begin
    reset_run_params!()

    # 1. A default read is registered, returned verbatim, and NOT flagged as overridden.
    delete!(ENV, "RP_TEST_A")
    @test envparam("RP_TEST_A", "0.4") == "0.4"
    p = run_params()
    @test length(p) == 1
    @test p[1] == ("RP_TEST_A", "0.4", false)

    # 2. An overridden read is flagged. This is the PS_TEND case: same script, same printed
    #    settings, different answer -- the flag is what makes the two runs distinguishable.
    ENV["RP_TEST_B"] = "1.2"
    @test envparam("RP_TEST_B", "0.4") == "1.2"
    @test run_params()[2] == ("RP_TEST_B", "1.2", true)

    # 3. Re-reading a key updates in place rather than duplicating, so a parameter read
    #    inside a loop does not produce a header with one line per iteration.
    @test envparam("RP_TEST_B", "0.4") == "1.2"
    @test length(run_params()) == 2

    # 4. EVERY registered parameter appears in the printed header, with the override mark.
    #    This is the assertion that would have failed before the fix.
    buf = IOBuffer()
    print_run_header("HEADER TEST"; io = buf, extra = ("collision" => "ES-BGK",))
    out = String(take!(buf))
    @test occursin("HEADER TEST", out)
    @test occursin("RP_TEST_A", out)
    @test occursin("RP_TEST_B", out)
    @test occursin("1.2", out)
    @test occursin("collision", out)
    @test occursin("ES-BGK", out)
    # the overridden one carries the marker and the defaulted one does not
    bline = only(filter(l -> occursin("RP_TEST_B", l), split(out, '\n')))
    aline = only(filter(l -> occursin("RP_TEST_A", l), split(out, '\n')))
    @test occursin("*", bline)
    @test !occursin("*", aline)

    # 5. reset clears, so one process running several cases does not accumulate.
    reset_run_params!()
    @test isempty(run_params())
    delete!(ENV, "RP_TEST_B")
end
