#!/usr/bin/env bash
# Experiment 3-A / Table 1B / Figure 4A-B: paired convergence and SCC statistics.
#SBATCH --job-name=paper-exp3a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin exp3a-convergence 'Experiment 3-A paired convergence and P0-B equivalence analysis'
INPUT="$PAPER_DATA_ROOT/$PAPER_RUN_ID"
DXTB_GATE=(); [[ "${PAPER_REQUIRE_DXTB:-0}" == 1 ]] && DXTB_GATE=(--require-dxtb)
paper_run_python "$ROOT/python/dataset_analysis.py" --input-root "$INPUT" \
  --selection "$PAPER_SELECTION_FILE" \
  --output-json "$PAPER_OUTPUT_DIR/derived/table1-convergence.json" \
  --outliers-csv "$PAPER_OUTPUT_DIR/derived/equivalence-outliers.csv" \
  --paired-errors-jsonl "$PAPER_OUTPUT_DIR/raw/paired-errors.jsonl" \
  --energy-atol "$PAPER_ENERGY_ATOL_HARTREE" --force-atol "$PAPER_FORCE_ATOL_HARTREE_PER_BOHR" \
  --charge-atol "$PAPER_CHARGE_ATOL_E" --cpu-cuda-atol "$PAPER_CPU_CUDA_ATOL" \
  --require-complete-p0 "${DXTB_GATE[@]}"
paper_finish
