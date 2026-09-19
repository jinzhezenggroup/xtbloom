#!/usr/bin/env bash
# P0-A CPU: independent canonical conformance, invariance, oracle and full CTest.
#SBATCH --job-name=paper-p0a-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/lib/common.sh"
paper_begin p0a-canonical-cpu 'P0-A canonical independent validation, CPU'
: "${PAPER_CMAKE:?}" "${PAPER_NINJA:?}" "${PAPER_CPU_LINALG_LIBRARY:?}"
BUILD="$PAPER_OUTPUT_DIR/build"
paper_hash_inputs "$PAPER_CPU_LINALG_LIBRARY"
paper_run "$PAPER_CMAKE" -S "$PAPER_REPO_ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PAPER_NINJA" -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_CPU_LINALG_LIBRARY="$PAPER_CPU_LINALG_LIBRARY" \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PAPER_PYTHON"
paper_run "$PAPER_CMAKE" --build "$BUILD" --parallel "$SLURM_CPUS_PER_TASK"
paper_record_canonical_library "$BUILD/libxtbloom.so" "$PAPER_OUTPUT_DIR/derived/canonical-library.txt"
paper_run ctest --test-dir "$BUILD" -N | tee "$PAPER_OUTPUT_DIR/logs/ctest-registration.txt"
paper_run ctest --test-dir "$BUILD" --no-tests=error --output-on-failure \
  --output-junit "$PAPER_OUTPUT_DIR/derived/ctest-results.xml" | tee "$PAPER_OUTPUT_DIR/logs/ctest.txt"
cd "$PAPER_REPO_ROOT"
paper_run_python tools/parameters/generate_gfn2.py --check
paper_run_python tools/conformance/xtbloom_conformance.py check
paper_run_python -m unittest discover -s tests/conformance -p 'test_*.py' -v
paper_run_python -m unittest discover -s tests/oracle -p 'test_*.py' -v
paper_finish
