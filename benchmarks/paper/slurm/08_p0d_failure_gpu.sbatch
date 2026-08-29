#!/usr/bin/env bash
# P0-D CUDA: peer-local failures, NaN slices, commit/device restore semantics.
#SBATCH --job-name=paper-p0d-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin p0d-failure-gpu 'P0-D failure isolation/publication, CUDA'
paper_check_gpu
paper_run_python "$ROOT/python/extract_ctest_evidence.py" \
  --junit "$PAPER_RUN_ROOT/p0a-canonical-gpu/derived/ctest-results.xml" \
  --regex 'cuda.*(failure|public|runtime|validation)|conformance.*cuda' \
  --output "$PAPER_OUTPUT_DIR/derived/canonical-ctest-evidence.json"
for memory in host device mixed; do
  paper_run_python "$ROOT/python/failure_peer_probe.py" --repo "$PAPER_REPO_ROOT" \
    --manifest "$PAPER_OMOL25_MANIFEST" \
    --source-results "$PAPER_DATA_ROOT/$PAPER_RUN_ID/p0b-dataset-gpu/raw/omol25-main,stress-xtbloom-$memory/results.jsonl" \
    --library "$PAPER_CUDA_LIBRARY" --backend cuda --memory-mode "$memory" \
    --device-id "$PAPER_DEVICE_ID" --cpu-threads 1 --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
    --atol "$PAPER_PEER_UNCHANGED_ATOL" \
    --output "$PAPER_OUTPUT_DIR/derived/real-peer-$memory.json"
done
paper_finish
