#!/usr/bin/env bash
# P0-E: exact three-fold-degeneracy public stress coordinate; narrow claim only.
#SBATCH --job-name=paper-p0e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin p0e-degeneracy 'P0-E exact-degeneracy cross-engine probe'
DXTB_ARGS=(); [[ -n "${PAPER_DXTB_SOURCE:-}" && -d "$PAPER_DXTB_SOURCE" ]] && DXTB_ARGS=(--dxtb-source "$PAPER_DXTB_SOURCE")
paper_run_python "$ROOT/python/degeneracy_probe.py" --repo "$PAPER_REPO_ROOT" \
  --library "$PAPER_CPU_LIBRARY" --xtb-library "$PAPER_XTB_LIBRARY" \
  --tblite-library "$PAPER_TBLITE_LIBRARY" "${DXTB_ARGS[@]}" \
  --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" --accuracy "$PAPER_ACCURACY" \
  --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
  --output "$PAPER_OUTPUT_DIR/derived/probe.json"
paper_run_python "$ROOT/python/extract_ctest_evidence.py" \
  --junit "$PAPER_RUN_ROOT/p0a-canonical-cpu/derived/ctest-results.xml" \
  --regex 'cpu.public_inference' \
  --output "$PAPER_OUTPUT_DIR/derived/canonical-ctest-evidence.json"
paper_finish
