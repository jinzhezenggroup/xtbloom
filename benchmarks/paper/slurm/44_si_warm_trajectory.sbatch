#!/usr/bin/env bash
# Conditional SI only: real continuous-geometry FRESH/WARM safety and timing.
#SBATCH --job-name=paper-si-warm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin si-warm-trajectory 'Conditional SI WARM real-trajectory sanity'
if [[ "${PAPER_RUN_WARM_SI:-0}" != 1 ]]; then
  printf 'SKIP: paper claims do not require WARM; no experiment was run.\n' >"$PAPER_OUTPUT_DIR/derived/status.txt"
  paper_finish; exit 0
fi
paper_check_gpu
paper_require_file "$PAPER_TRAJECTORY_MANIFEST"
paper_run_python "$ROOT/python/warm_trajectory.py" --repo "$PAPER_REPO_ROOT" \
  --trajectory "$PAPER_TRAJECTORY_MANIFEST" --library "$PAPER_CUDA_LIBRARY" \
  --backend cuda --device-id "$PAPER_DEVICE_ID" --cpu-threads 1 \
  --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
  --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
  --energy-atol "$PAPER_ENERGY_ATOL_HARTREE" \
  --force-atol "$PAPER_FORCE_ATOL_HARTREE_PER_BOHR" --bootstrap-samples "$PAPER_BOOTSTRAP_SAMPLES" \
  --output "$PAPER_OUTPUT_DIR/raw/warm-trajectory.json"
paper_finish
