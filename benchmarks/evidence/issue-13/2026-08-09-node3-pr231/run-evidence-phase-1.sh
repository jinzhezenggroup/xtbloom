#!/usr/bin/env bash
set -euo pipefail

repo=/home/jzzeng/codes/xtbloom4-pr231
out="$repo/build/benchmarks/issue-13-c9c0a43-final"
cpu_library="$repo/build/pr231-evidence-c9c0a43-cpu/libxtbloom.so.0.1.0"
cuda_library="$repo/build/pr231-evidence-c9c0a43-cuda/libxtbloom.so.0.1.0"
xtb_library=/tmp/pr231-1fc8698-xtb-final/libxtb.so.6.7.1
tblite_library=/tmp/pr231-477d5f4-tblite-final/libtblite.so.0.7.0
xtb_source=/home/jzzeng/codes/xtb
tblite_source=/home/jzzeng/codes/tblite
dxtb_source=/home/jzzeng/codes/dxtb
base_python=/home/jzzeng/miniconda3/bin/python3
dxtb_python=/tmp/dxtb-env/bin/python

mkdir -p "$out"
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
  local xtbloom_library=$3
  local engine=$4
  local panel=$5
  shift 5

  local panel_args=()
  case "$panel" in
    cold)
      panel_args=(
        --natoms 14,32,62,122,242,362
        --batch-sizes 1
        --start-policy cold
      )
      ;;
    b128)
      panel_args=(
        --natoms 14,32,62,122
        --batch-sizes 128
        --start-policy auto-warm
      )
      ;;
    b512)
      panel_args=(
        --natoms 14,32,62,122
        --batch-sizes 512
        --start-policy cold
      )
      ;;
    *)
      printf 'unknown panel: %s\n' "$panel" >&2
      return 2
      ;;
  esac

  "$python_exe" benchmarks/natoms_cross_engine.py \
    --library "$xtbloom_library" \
    --engines "$engine" \
    "${panel_args[@]}" \
    "${common[@]}" \
    "$@" \
    --output-json "$out/$output_stem.json" \
    --output-csv "$out/$output_stem.csv"
}

run_case "$base_python" ref-tblite-cold "$cpu_library" tblite cold \
  --tblite-library "$tblite_library" --tblite-source "$tblite_source" \
  --make-reference
run_case "$base_python" ref-tblite-b128 "$cpu_library" tblite b128 \
  --tblite-library "$tblite_library" --tblite-source "$tblite_source" \
  --make-reference
run_case "$base_python" ref-tblite-b512 "$cpu_library" tblite b512 \
  --tblite-library "$tblite_library" --tblite-source "$tblite_source" \
  --make-reference

for panel in cold b128 b512; do
  reference="$out/ref-tblite-$panel.json"
  run_case "$base_python" "xtbloom-cpu-$panel" "$cpu_library" xtbloom-cpu "$panel" \
    --reference-json "$reference"
  run_case "$base_python" "xtbloom-cuda-$panel" "$cuda_library" xtbloom-cuda "$panel" \
    --reference-json "$reference"
  run_case "$base_python" "xtb-$panel" "$cpu_library" xtb "$panel" \
    --xtb-library "$xtb_library" --xtb-source "$xtb_source" \
    --reference-json "$reference"
  run_case "$dxtb_python" "dxtb-cpu-$panel" "$cpu_library" dxtb-cpu "$panel" \
    --dxtb-source "$dxtb_source" --reference-json "$reference"
  run_case "$dxtb_python" "dxtb-cuda-$panel" "$cuda_library" dxtb-cuda "$panel" \
    --dxtb-source "$dxtb_source" --reference-json "$reference"
done
