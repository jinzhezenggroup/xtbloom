#!/usr/bin/env bash
# SI: GFN1 CPU energy/force/charge validation. GFN1 CUDA remains unsupported.
#SBATCH --job-name=paper-si-gfn1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin si-gfn1-cpu 'SI GFN1 CPU numerical validation'
paper_run_python "$ROOT/python/extract_ctest_evidence.py" \
  --junit "$PAPER_RUN_ROOT/p0a-canonical-cpu/derived/ctest-results.xml" --regex 'gfn1' \
  --output "$PAPER_OUTPUT_DIR/derived/canonical-ctest-evidence.json"
paper_finish
