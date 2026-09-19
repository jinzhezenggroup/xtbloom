#!/usr/bin/env bash
# Final stage: compact Table/Figure inputs, complete inventory and checksums.
#SBATCH --job-name=paper-archive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin analyze-archive 'Table 1 / Figures 2-4 compact archive integrity'
paper_run_python "$ROOT/python/make_figure_inputs.py" --run-root "$PAPER_RUN_ROOT" \
  --output-dir "$PAPER_OUTPUT_DIR/derived/figure-inputs"
OPTIONAL_REQUIRED=()
[[ "${PAPER_RUN_WARM_SI:-0}" != 1 ]] || OPTIONAL_REQUIRED+=(--require-experiment si-warm-trajectory)
[[ "${PAPER_RUN_SECOND_HARDWARE:-0}" != 1 ]] || OPTIONAL_REQUIRED+=(--require-experiment si-second-hardware)
paper_run_python "$ROOT/python/archive_bundle.py" --run-root "$PAPER_RUN_ROOT" \
  --ignore-experiment analyze-archive --ignore-experiment slurm --ignore-experiment submission \
  --ignore-relative "slurm/paper-archive-$SLURM_JOB_ID.out" \
  --ignore-relative "slurm/paper-archive-$SLURM_JOB_ID.err" \
  "${OPTIONAL_REQUIRED[@]}" \
  --output "$PAPER_OUTPUT_DIR/derived/archive-index.json"
paper_finish
