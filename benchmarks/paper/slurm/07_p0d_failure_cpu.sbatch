#!/usr/bin/env bash
# P0-D CPU: mixed-peer, NaN publication, validation and injected failure tests.
#SBATCH --job-name=paper-p0d-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin p0d-failure-cpu 'P0-D failure isolation/publication, CPU'
paper_run_python "$ROOT/python/extract_ctest_evidence.py" \
  --junit "$PAPER_RUN_ROOT/p0a-canonical-cpu/derived/ctest-results.xml" \
  --regex 'batch_validation|runtime|cpu.public_inference|conformance.invariants_cpu|failure' \
  --output "$PAPER_OUTPUT_DIR/derived/canonical-ctest-evidence.json"
paper_run_python "$ROOT/python/failure_peer_probe.py" --repo "$PAPER_REPO_ROOT" \
  --manifest "$PAPER_OMOL25_MANIFEST" \
  --source-results "$PAPER_DATA_ROOT/$PAPER_RUN_ID/p0b-dataset-cpu/raw/omol25-main,stress/results.jsonl" \
  --library "$PAPER_CPU_LIBRARY" --backend cpu --memory-mode host --cpu-threads 1 \
  --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" --atol "$PAPER_PEER_UNCHANGED_ATOL" \
  --output "$PAPER_OUTPUT_DIR/derived/real-peer.json"
paper_finish
