#!/usr/bin/env bash
# SI: representative energy-only points; intentionally not a full main matrix.
#SBATCH --job-name=paper-si-energy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin si-energy-only 'SI representative energy-only coordinates'
paper_check_gpu
paper_run_python "$ROOT/python/performance_matrix.py" --repo "$PAPER_REPO_ROOT" \
  --manifest "$PAPER_OMOL25_MANIFEST" --dataset omol25 --subset performance \
  --cpu-library "$PAPER_CPU_LIBRARY" --cuda-library "$PAPER_CUDA_LIBRARY" \
  --suite energy-only --ao-bins 1-64,129-256,513-1024 --batch-sizes 1,64,256 --cpu-threads 32 \
  --device-id "$PAPER_DEVICE_ID" --property energy --warmups "$PAPER_WARMUPS" --repetitions "$PAPER_REPETITIONS" \
  --bootstrap-samples "$PAPER_BOOTSTRAP_SAMPLES" --seed "$PAPER_SEED" \
  --reference-root "$PAPER_DATA_ROOT/$PAPER_RUN_ID/performance-reference" --require-correctness \
  --charge-atol "$PAPER_CHARGE_ATOL_E" --accuracy "$PAPER_ACCURACY" \
  --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
  --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
  --output "$PAPER_OUTPUT_DIR/raw/energy-only.json"
paper_finish
