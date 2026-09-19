#!/usr/bin/env bash
# P0-A CUDA: real-GPU host/device/mixed public conformance and complete CTest.
#SBATCH --job-name=paper-p0a-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/lib/common.sh"
paper_begin p0a-canonical-gpu 'P0-A canonical independent validation, CUDA host/device/mixed'
paper_check_gpu
: "${PAPER_CMAKE:?}" "${PAPER_NINJA:?}" "${PAPER_CUDA_COMPILER:?}" "${PAPER_CPU_LINALG_LIBRARY:?}"
ARCH=${PAPER_CUDA_ARCH:-auto}
if [[ "$ARCH" == auto ]]; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$PAPER_RESOLVED_GPU_TOKEN" | head -n1)
  ARCH=${CAP/./}
fi
[[ "$ARCH" =~ ^[0-9]+$ ]] || paper_die "invalid CUDA architecture: $ARCH"
BUILD="$PAPER_OUTPUT_DIR/build"
paper_run "$PAPER_CMAKE" -S "$PAPER_REPO_ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PAPER_NINJA" -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$PAPER_CUDA_COMPILER" -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DXTBLOOM_CPU_LINALG_LIBRARY="$PAPER_CPU_LINALG_LIBRARY" \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PAPER_PYTHON"
paper_run "$PAPER_CMAKE" --build "$BUILD" --parallel "$SLURM_CPUS_PER_TASK"
paper_record_canonical_library "$BUILD/libxtbloom.so" "$PAPER_OUTPUT_DIR/derived/canonical-library.txt"
GPU_UUID=$PAPER_RESOLVED_GPU_UUID
GPU_NAME=$PAPER_RESOLVED_GPU_NAME
printf '%s\t%s\t%s\n' "$(hostname)" "$GPU_UUID" "$GPU_NAME" >"$PAPER_OUTPUT_DIR/derived/hardware-identity.tsv"
paper_run ctest --test-dir "$BUILD" -N | tee "$PAPER_OUTPUT_DIR/logs/ctest-registration.txt"
paper_run ctest --test-dir "$BUILD" --no-tests=error --output-on-failure \
  --output-junit "$PAPER_OUTPUT_DIR/derived/ctest-results.xml" | tee "$PAPER_OUTPUT_DIR/logs/ctest.txt"
paper_finish
