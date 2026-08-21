#!/usr/bin/env bash
# P0-C CPU finite differences: 16 systems/dataset x 8 frozen coordinates.
#SBATCH --job-name=paper-p0c-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin p0c-fd-cpu 'P0-C real-system finite differences, CPU'
SELECTION="$PAPER_DATA_ROOT/$PAPER_RUN_ID/freeze-manifests/derived/paper-selection.json"
for dataset in qm9 omol25; do
  manifest=$PAPER_QM9_MAIN_MANIFEST; [[ "$dataset" == omol25 ]] && manifest=$PAPER_OMOL25_MANIFEST
  paper_run_python "$ROOT/python/finite_difference_check.py" --repo "$PAPER_REPO_ROOT" \
    --manifest "$manifest" --selection "$SELECTION" --dataset "$dataset" --library "$PAPER_CPU_LIBRARY" \
    --backend cpu --cpu-threads 1 --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
    --atol "$PAPER_FD_ATOL_HARTREE_PER_BOHR" --output "$PAPER_OUTPUT_DIR/derived/$dataset.json"
done
paper_finish
