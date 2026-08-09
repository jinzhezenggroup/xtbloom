#!/usr/bin/env bash
set -uo pipefail

repo=/home/jzzeng/codes/gpuxtb4-pr231
out="$repo/build/benchmarks/issue-13-c9c0a43-final"
cpu_library="$repo/build/pr231-evidence-c9c0a43-cpu/libgpuxtb.so.0.1.0"
cuda_library="$repo/build/pr231-evidence-c9c0a43-cuda/libgpuxtb.so.0.1.0"
xtb_library=/tmp/pr231-1fc8698-xtb-final/libxtb.so.6.7.1
xtb_source=/home/jzzeng/codes/xtb
dxtb_source=/home/jzzeng/codes/dxtb
base_python=/home/jzzeng/miniconda3/bin/python3
dxtb_python=/tmp/dxtb-env/bin/python

cd "$repo"

export PATH="/tmp/lammps-qmmm-xtb-env/bin:/group/software/cuda-12.9.1/bin:$PATH"
export PKG_CONFIG_PATH=/tmp/lammps-qmmm-xtb-env/lib/pkgconfig
export CMAKE_PREFIX_PATH=/tmp/lammps-qmmm-xtb-env
runtime_path=/tmp/lammps-qmmm-xtb-env/lib:/group/software/deepmd-kit-3.1.1/lib:/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/group/software/cuda-12.9.1/lib64
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
  runtime_path="$runtime_path:$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="$runtime_path"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=SEQUENTIAL
unset PYTHONPATH

common=(
  --warmups 1
  --repetitions 3
  --cpu-threads 16
  --energy-atol 2e-3
  --force-atol 2e-3
  --repeatability-energy-atol 1e-10
  --repeatability-force-atol 1e-8
  --scc-charge-tolerance 1e-4
  --scc-energy-tolerance 1e-6
  --scc-max-iterations 500
)

run_case() {
  local python_exe=$1
  local output_stem=$2
  local gpuxtb_library=$3
  local engine=$4
  local panel=$5
  shift 5

  local panel_args=()
  case "$panel" in
    b128)
      panel_args=(--natoms 14,32,62,122 --batch-sizes 128 --start-policy auto-warm)
      ;;
    b512)
      panel_args=(--natoms 14,32,62,122 --batch-sizes 512 --start-policy cold)
      ;;
    *)
      printf 'unknown panel: %s\n' "$panel" >&2
      return 2
      ;;
  esac

  "$python_exe" benchmarks/natoms_cross_engine.py \
    --library "$gpuxtb_library" \
    --engines "$engine" \
    "${panel_args[@]}" \
    "${common[@]}" \
    "$@" \
    --output-json "$out/$output_stem.json" \
    --output-csv "$out/$output_stem.csv"
}

run_and_retain_unavailable() {
  local output_stem=$2
  run_case "$@"
  local status=$?
  if [[ $status -eq 0 ]]; then
    return 0
  fi
  if [[ $status -eq 2 && -s "$out/$output_stem.json" && -s "$out/$output_stem.csv" ]]; then
    printf 'retained explicit unavailable/error rows in %s (runner status 2)\n' "$output_stem"
    return 0
  fi
  return "$status"
}

run_and_retain_unavailable "$dxtb_python" dxtb-cuda-b128 "$cuda_library" dxtb-cuda b128 \
  --dxtb-source "$dxtb_source" --reference-json "$out/ref-tblite-b128.json"

run_and_retain_unavailable "$base_python" gpuxtb-cpu-b512 "$cpu_library" gpuxtb-cpu b512 \
  --reference-json "$out/ref-tblite-b512.json"
run_and_retain_unavailable "$base_python" gpuxtb-cuda-b512 "$cuda_library" gpuxtb-cuda b512 \
  --reference-json "$out/ref-tblite-b512.json"
run_and_retain_unavailable "$base_python" xtb-b512 "$cpu_library" xtb b512 \
  --xtb-library "$xtb_library" --xtb-source "$xtb_source" \
  --reference-json "$out/ref-tblite-b512.json"
run_and_retain_unavailable "$dxtb_python" dxtb-cpu-b512 "$cpu_library" dxtb-cpu b512 \
  --dxtb-source "$dxtb_source" --reference-json "$out/ref-tblite-b512.json"
run_and_retain_unavailable "$dxtb_python" dxtb-cuda-b512 "$cuda_library" dxtb-cuda b512 \
  --dxtb-source "$dxtb_source" --reference-json "$out/ref-tblite-b512.json"
