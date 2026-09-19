#!/usr/bin/env bash
# Experiment 2 capacity: heterogeneous distinct-real batches to the frozen
# dataset limit. A sample-censored boundary is archived but is not a claim of
# saturation/OOM; such a claim requires a larger frozen capacity-only corpus.
#SBATCH --job-name=paper-capacity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin exp2-gpu-capacity 'Experiment 2 saturation, VRAM and OOM boundary'
paper_check_gpu
REFERENCE="$PAPER_DATA_ROOT/$PAPER_RUN_ID/performance-reference"
for spec in "qm9:$PAPER_QM9_PERFORMANCE_MANIFEST" "omol25:$PAPER_OMOL25_MANIFEST"; do
  dataset=${spec%%:*}; manifest=${spec#*:}
  if [[ "$dataset" == qm9 ]]; then batches=$PAPER_QM9_GPU_CAPACITY_BATCH_SIZES
  else batches=$PAPER_OMOL25_GPU_CAPACITY_BATCH_SIZES
  fi
  paper_run_python "$ROOT/python/performance_matrix.py" --repo "$PAPER_REPO_ROOT" \
    --manifest "$manifest" --dataset "$dataset" --subset performance --cuda-library "$PAPER_CUDA_LIBRARY" \
    --suite capacity --ao-bins "$PAPER_AO_BINS" --batch-sizes "$batches" --selection-mode real-order \
    --cpu-threads 1 --device-id "$PAPER_DEVICE_ID" --property force --warmups "$PAPER_WARMUPS" \
    --repetitions "$PAPER_REPETITIONS" --bootstrap-samples "$PAPER_BOOTSTRAP_SAMPLES" \
    --seed "$PAPER_SEED" --reference-root "$REFERENCE" \
    --require-correctness --charge-atol "$PAPER_CHARGE_ATOL_E" \
    --accuracy "$PAPER_ACCURACY" --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
    --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" --output "$PAPER_OUTPUT_DIR/raw/$dataset.json"
done
paper_finish
