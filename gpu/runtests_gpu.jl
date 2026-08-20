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
#
# THAT COMMAND DID NOT WORK IN A GIT WORKTREE, which matters because a worktree is exactly where
# you are when preparing a branch -- i.e. precisely when the instruction above tells you to run
# this. gpu/gpuenv2/Manifest.toml is gitignored, so a fresh worktree has only Project.toml, and
# `Pkg.instantiate()` then tries to resolve Riemann35 from a registry it is not in:
#
#     ERROR: expected package `Riemann35 [ef841222]` to be registered
#
# The bootstrap below dev-links the environment to THE TREE THIS FILE IS IN and instantiates. It
# runs only when Riemann35 cannot already be loaded, so an instantiated environment is untouched.
#
# COPYING THE MAIN CHECKOUT'S MANIFEST ACROSS IS THE WRONG FIX AND IS WORTH NAMING, because it is
# the obvious one: that Manifest pins Riemann35 by ABSOLUTE path to the main checkout, so the
# worktree's tests would silently exercise the main checkout's source. They would pass, and they
# would be testing the code you did not change.
let
    try
        Base.require(Base.PkgId(Base.UUID("ef841222-879f-4f48-b9ef-62e4b8485d01"), "Riemann35"))
    catch
        @info "gpuenv2 is not instantiated for this tree; dev-linking and instantiating"
        import Pkg
        Pkg.develop(path = dirname(@__DIR__))   # the tree containing gpu/
        Pkg.instantiate()
    end
end

using Test

const GPU_TESTS = [
    joinpath(@__DIR__, "..", "test", "test_dvm_weno5.jl"),
    joinpath(@__DIR__, "..", "test", "test_wall_dv_guidance.jl"),
    joinpath(@__DIR__, "..", "test", "test_dvm_wall_weno5.jl"),
    joinpath(@__DIR__, "..", "test", "test_esbgk_offdiagonal.jl"),
]

@testset "Riemann35 GPU" begin
    for f in GPU_TESTS
        isfile(f) || error("GPU test not found: $f")
        @info "running" test=basename(f)
        include(f)
    end
end
