#!/usr/bin/env bash
# SI: CUDA mixed-descriptor correctness plus representative performance spots.
#SBATCH --job-name=paper-si-mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin si-cuda-mixed 'SI CUDA mixed descriptor correctness/performance spot check'
paper_check_gpu
paper_run_python "$ROOT/python/extract_ctest_evidence.py" \
  --junit "$PAPER_RUN_ROOT/p0a-canonical-gpu/derived/ctest-results.xml" \
  --regex 'cuda.*(mixed|public)|conformance.*cuda' \
  --output "$PAPER_OUTPUT_DIR/derived/canonical-ctest-evidence.json"
paper_run_python "$ROOT/python/performance_matrix.py" --repo "$PAPER_REPO_ROOT" \
  --manifest "$PAPER_OMOL25_MANIFEST" --dataset omol25 --subset performance --cuda-library "$PAPER_CUDA_LIBRARY" \
  --suite mixed-spot --ao-bins 65-128,257-512 --batch-sizes 16,128 --cpu-threads 1 \
  --device-id "$PAPER_DEVICE_ID" --property force --warmups "$PAPER_WARMUPS" \
  --repetitions "$PAPER_REPETITIONS" \
  --bootstrap-samples "$PAPER_BOOTSTRAP_SAMPLES" --seed "$PAPER_SEED" \
  --reference-root "$PAPER_DATA_ROOT/$PAPER_RUN_ID/performance-reference" --require-correctness \
  --charge-atol "$PAPER_CHARGE_ATOL_E" --accuracy "$PAPER_ACCURACY" \
  --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
  --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
  --output "$PAPER_OUTPUT_DIR/raw/mixed-spots.json"
paper_finish
