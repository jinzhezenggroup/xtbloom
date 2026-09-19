#!/usr/bin/env bash
# Experiment 3-B / Figure 4C-D CUDA: controlled-wide and real-order workloads.
#SBATCH --job-name=paper-ragged-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin exp3b-ragged-gpu 'Experiment 3-B controlled and real ragged CUDA workloads'
paper_check_gpu
REFERENCE="$PAPER_DATA_ROOT/$PAPER_RUN_ID/performance-reference"
for spec in "qm9:$PAPER_QM9_PERFORMANCE_MANIFEST" "omol25:$PAPER_OMOL25_MANIFEST"; do
  dataset=${spec%%:*}; manifest=${spec#*:}
  for mode in controlled-wide real-order; do
    batches=128,512
    if [[ "$mode" == real-order ]]; then
      batches=2048; [[ "$dataset" != omol25 ]] || batches=4096
    fi
    paper_run_python "$ROOT/python/performance_matrix.py" --repo "$PAPER_REPO_ROOT" \
      --manifest "$manifest" --dataset "$dataset" --subset performance --cuda-library "$PAPER_CUDA_LIBRARY" \
      --suite ragged --selection-mode "$mode" --ao-bins "$PAPER_AO_BINS" --batch-sizes "$batches" \
      --cpu-threads 1 --device-id "$PAPER_DEVICE_ID" --property force --warmups "$PAPER_WARMUPS" \
      --repetitions "$PAPER_REPETITIONS" --bootstrap-samples "$PAPER_BOOTSTRAP_SAMPLES" \
      --seed "$PAPER_SEED" --reference-root "$REFERENCE" \
      --require-correctness --charge-atol "$PAPER_CHARGE_ATOL_E" \
      --accuracy "$PAPER_ACCURACY" --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
      --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" --output "$PAPER_OUTPUT_DIR/raw/$dataset-$mode.json"
  done
done
paper_finish
