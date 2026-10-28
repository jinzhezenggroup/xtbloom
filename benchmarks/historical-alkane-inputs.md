# Historical alkane input assembly

`historical_alkane_inputs.py` converts the saved `alkane32-b256` and
`alkane62-b256` arrays into input records accepted by the existing public batch
assembler. It does not regenerate geometries, run a model, collect CUDA
telemetry, or measure performance. The original 256 slots per workload remain
distinct and ordered, including repeated geometries.

## Materialize a pinned source

Supply the SHA-256 of the exact historical JSON bytes and a new output directory
whose parent already exists:

```console
python3 benchmarks/historical_alkane_inputs.py \
  --input /absolute/path/historical-inputs.json \
  --expected-sha256 <verified-64-character-sha256> \
  --output-dir /absolute/path/new-historical-inputs
```

The converter rejects a hash mismatch, duplicate JSON keys, unexpected workload
order, malformed extents, nonfinite values, or unsupported tags before creating
output. It preserves atom order, bohr coordinates with round-trip binary64
decimals, molecular charges, unpaired electrons, and explicit spin channels.
The recorded seeds and perturbation widths are provenance, not instructions to
generate replacement inputs. Existing output paths are refused; staging is
published only after all files have been written. Linux uses atomic
`renameat2(RENAME_NOREPLACE)` and Windows uses no-replace `rename`; no output
reservation is removed during cleanup. Missing Linux libc/kernel/filesystem
support and other platforms fail closed rather than falling back to a
check-then-replace operation.

The output contains `manifest.json`, 512 coordinate files, 512 explicitly empty
reference placeholders, and `SHA256SUMS`. The manifest retains source hash and
source metadata, source-slot IDs, input hashes, and per-slot geometry hashes.
Verify all generated payloads before collection:

```console
cd /absolute/path/new-historical-inputs
sha256sum -c SHA256SUMS
```

## Use the existing assembler

The following is offline input assembly only; it loads neither a native library
nor a GPU:

```python
import json
from pathlib import Path

from benchmarks import historical_alkane_inputs as historical
from benchmarks import run

manifest_path = Path("/absolute/path/new-historical-inputs/manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
cases = historical.select_workload_cases(manifest, "alkane32-b256")
storage = run.public_api.assemble_batch(manifest_path, manifest, cases)
```

A separately reviewed collector can pass that exact tuple as the existing
`run.XTBloomAdapter` constructor's `case_sequence`, with a matching batch size
and explicitly frozen options. `run.py`'s named-workload CLI is unchanged; this
converter does not add `--workloads alkane32-b256` or `alkane62-b256`.

## Qualification boundary

Empty placeholders exist only because the current assembler also loads reference
metadata. They contain no energy, force, or charge references. Every case and
the corpus are explicitly ineligible for correctness and performance evidence.
Do not feed them to correctness qualification or present input preservation as
scientific validation.

Native OFF/ON collection, paired outputs and iterations, repeated instrumentation
perturbation timing, independent references, and the full affected runtime,
sanitizer, and profiler gates remain separate work. Real-GPU collection must use
a finite Slurm allocation and preserve its assigned device visibility. See
[CUDA SCC diagnostics](cuda-diagnostics.md) for the telemetry protocol.

Run the hardware-free converter tests with:

```console
python3 -m unittest -v benchmarks.test_historical_alkane_inputs
```
