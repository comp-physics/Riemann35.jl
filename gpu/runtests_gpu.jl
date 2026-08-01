# runtests_gpu.jl -- the GPU test entry point.
#
# WHY THIS EXISTS SEPARATELY FROM test/runtests.jl. `CUDA` is not a dependency of the main
# project; it lives only in `gpu/gpuenv2`. So a GPU test cannot be included from
# test/runtests.jl -- on CI that fails outright with
#     ArgumentError: Package CUDA not found in current path
# which is worse than not having the test. And the CI runners have no GPU regardless, so
# even with CUDA installed every GPU test would only ever skip.
#
# The honest consequence, stated plainly because it is easy to assume otherwise:
# **GPU TESTS ARE NOT COVERED BY CI.** They pass or fail only when someone runs this file
# on a machine with a device. Before merging anything that touches gpu/ or a *_dev.jl
# device path, run it.
#
#     julia --project=gpu/gpuenv2 gpu/runtests_gpu.jl
#
# Tests that need no device belong in test/runtests.jl instead, where CI will actually
# execute them.
using Test

const GPU_TESTS = [
    joinpath(@__DIR__, "..", "test", "test_dvm_weno5.jl"),
    joinpath(@__DIR__, "..", "test", "test_wall_dv_guidance.jl"),
]

@testset "Riemann35 GPU" begin
    for f in GPU_TESTS
        isfile(f) || error("GPU test not found: $f")
        @info "running" test=basename(f)
        include(f)
    end
end
