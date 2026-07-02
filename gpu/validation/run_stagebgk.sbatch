#!/bin/bash
# GPU half of the stage_bgk + pressure_recon validation (run the CPU half first:
#   $JULIA --project=. gpu/validation/stagebgk_cpu_march.jl
# with the same RIEMANN35_DATA). Submit: sbatch gpu/validation/run_stagebgk.sbatch
#SBATCH -J stagebgk-val
#SBATCH --account=gts-sbryngelson3
#SBATCH -N 1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o /storage/scratch1/6/sbryngelson3/stagebgk_val.%j.out
export JULIA=/usr/local/pace-apps/manual/packages/julia/1.11.3/bin/julia
export JULIA_DEPOT_PATH=/storage/scratch1/6/sbryngelson3/julia_depot:/storage/home/hcoda1/6/sbryngelson3/.julia
export TMPDIR=/storage/scratch1/6/sbryngelson3/tmp; mkdir -p $TMPDIR
export LD_LIBRARY_PATH=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/openmpi-4.1.8-iit4xaslnjxkchcc6n62b5kluzibl2v2/lib:$LD_LIBRARY_PATH
export RIEMANN35_DATA=${RIEMANN35_DATA:-/storage/scratch1/6/sbryngelson3/stagebgk_data}
cd /storage/project/r-sbryngelson3-0/sbryngelson3/Riemann35.jl
$JULIA --project=gpu/gpuenv2 gpu/validation/stagebgk_gpu_march.jl
