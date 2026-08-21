#!/usr/bin/env bash
# Recommended SI: repeat exactly three representative points on second hardware.
#SBATCH --job-name=paper-si-hw2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin si-second-hardware 'Recommended SI second-hardware three-point replicate'
paper_require_complete p0a-canonical-gpu
[[ "${PAPER_RUN_SECOND_HARDWARE:-0}" == 1 ]] || paper_die 'second-hardware job was not enabled'
[[ -n "${PAPER_SECOND_HARDWARE_ID:-}" && "$PAPER_SECOND_HARDWARE_ID" != "$PAPER_HARDWARE_ID" ]] || \
  paper_die 'second hardware requires a distinct nonempty hardware ID'
[[ -n "${PAPER_SECOND_GPU_REGEX:-}" ]] || paper_die 'PAPER_SECOND_GPU_REGEX is required'
PAPER_EXPECT_GPU_REGEX=$PAPER_SECOND_GPU_REGEX
PAPER_HARDWARE_ID=$PAPER_SECOND_HARDWARE_ID
export PAPER_EXPECT_GPU_REGEX PAPER_HARDWARE_ID
paper_check_gpu
PRIMARY_IDENTITY="$PAPER_RUN_ROOT/p0a-canonical-gpu/derived/hardware-identity.tsv"
paper_require_file "$PRIMARY_IDENTITY"
IFS=$'\t' read -r PRIMARY_HOST PRIMARY_UUID PRIMARY_NAME <"$PRIMARY_IDENTITY"
CURRENT_HOST=$(hostname)
CURRENT_UUID=$PAPER_RESOLVED_GPU_UUID
CURRENT_NAME=$PAPER_RESOLVED_GPU_NAME
[[ "$CURRENT_HOST" != "$PRIMARY_HOST" && "$CURRENT_UUID" != "$PRIMARY_UUID" ]] || \
  paper_die "second-hardware allocation reuses the primary host or GPU UUID: $CURRENT_HOST $CURRENT_UUID"
{
  printf 'second_hardware_id=%s\n' "$PAPER_HARDWARE_ID"
  printf 'second_gpu_regex=%s\n' "$PAPER_EXPECT_GPU_REGEX"
  printf 'primary_hardware=%s\t%s\t%s\n' "$PRIMARY_HOST" "$PRIMARY_UUID" "$PRIMARY_NAME"
  printf 'current_hardware=%s\t%s\t%s\n' "$CURRENT_HOST" "$CURRENT_UUID" "$CURRENT_NAME"
  nvidia-smi -L
} >>"$PAPER_OUTPUT_DIR/logs/environment.txt"
: "${PAPER_CMAKE:?}" "${PAPER_NINJA:?}" "${PAPER_CPU_LINALG_LIBRARY:?}"
SECOND_CUDA_COMPILER=${PAPER_SECOND_CUDA_COMPILER:-$PAPER_CUDA_COMPILER}
paper_require_file "$SECOND_CUDA_COMPILER"
paper_hash_inputs "$SECOND_CUDA_COMPILER"
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$PAPER_RESOLVED_GPU_TOKEN" | head -n1)
ARCH=${CAP/./}
[[ "$ARCH" =~ ^[0-9]+$ ]] || paper_die "invalid second-hardware CUDA architecture: $ARCH"
BUILD="$PAPER_OUTPUT_DIR/build"
paper_run "$PAPER_CMAKE" -S "$PAPER_REPO_ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PAPER_NINJA" -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$SECOND_CUDA_COMPILER" -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DXTBLOOM_CPU_LINALG_LIBRARY="$PAPER_CPU_LINALG_LIBRARY" \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PAPER_PYTHON"
paper_run "$PAPER_CMAKE" --build "$BUILD" --parallel "$SLURM_CPUS_PER_TASK"
SECOND_CUDA_LIBRARY="$BUILD/libxtbloom.so"
paper_require_file "$SECOND_CUDA_LIBRARY"
REFERENCE="$PAPER_DATA_ROOT/$PAPER_RUN_ID/performance-reference"
DXTB_ARGS=(); [[ -n "${PAPER_DXTB_SOURCE:-}" && -d "$PAPER_DXTB_SOURCE" ]] && DXTB_ARGS=(--dxtb-source "$PAPER_DXTB_SOURCE")
DXTB_GATE=(); [[ "${PAPER_REQUIRE_DXTB:-0}" == 1 ]] && DXTB_GATE=(--require-dxtb)
for cell in '1-64:1' '65-128:256' '513-1024:16'; do
  ao=${cell%%:*}; batch=${cell#*:}; name=${ao//-/_}-b$batch
  paper_run_python "$ROOT/python/performance_matrix.py" --repo "$PAPER_REPO_ROOT" \
    --manifest "$PAPER_OMOL25_MANIFEST" --dataset omol25 --subset performance --cuda-library "$SECOND_CUDA_LIBRARY" \
    "${DXTB_ARGS[@]}" --suite second-hardware --ao-bins "$ao" --batch-sizes "$batch" --cpu-threads 16 \
    --device-id "$PAPER_DEVICE_ID" --property force --warmups "$PAPER_WARMUPS" --repetitions "$PAPER_REPETITIONS" \
    --bootstrap-samples "$PAPER_BOOTSTRAP_SAMPLES" --seed "$PAPER_SEED" \
    --reference-root "$REFERENCE" --require-correctness --charge-atol "$PAPER_CHARGE_ATOL_E" \
    --accuracy "$PAPER_ACCURACY" --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
    --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" "${DXTB_GATE[@]}" \
    --output "$PAPER_OUTPUT_DIR/raw/$name.json"
done
paper_finish
