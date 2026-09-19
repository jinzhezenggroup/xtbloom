#!/usr/bin/env bash
# SI: QM/MM energy, QM force, point-charge force, finite difference/conservation.
#SBATCH --job-name=paper-si-qmmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin si-qmmm 'SI QM/MM correctness and force accounting'
paper_run_python "$ROOT/python/extract_ctest_evidence.py" \
  --junit "$PAPER_RUN_ROOT/p0a-canonical-cpu/derived/ctest-results.xml" \
  --regex 'qmmm|conformance.public_cpu|conformance.invariants_cpu|force' \
  --output "$PAPER_OUTPUT_DIR/derived/canonical-ctest-evidence.json"
paper_finish
