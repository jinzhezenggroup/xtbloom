# Controlled build receipts

This is an opt-in **record-only** provenance boundary for issue #513. It binds
a controlled build's immutable source archive and retained output bytes. It
does not turn a benchmark row into an accepted performance claim. The default
runner remains diagnostic and every paired performance-eligibility flag stays
false, with or without a matching receipt.

## Trust and scope

The recorder and its local execution environment are trusted. The consumer
verifies externally pinned records without executing their commands or loading
their libraries. A digest provides integrity, not a signed remote attestation
or proof against a hostile host that fabricates an entire experiment. The
reviewed recorder must produce the receipt; hand-authored metadata and the
current state of an adjacent CMake cache are not producer evidence.

The source revision is independent of the runner and producer-tool revisions.
Raw Git commit-object bytes bind the revision to a tree; reconstruction of the
complete tree binds every recorded path/mode/blob. The Git archive binds all
source payload bytes and its commit identity. Only the existing
`.git_archival.txt` export-substitution is supported: original Git metadata and
expanded archive bytes are recorded separately. Submodules, source symlinks,
unknown archive substitutions, links and traversal are rejected rather than
silently omitted. The private source snapshot is read-only and rechecked.

The fixed build uses CMake/Ninja, Release, shared libraries, an explicit CPU
LP64 provider and CPU `OFF` or CUDA `ON`; CUDA also requires an explicit nvcc
and numeric architectures. It builds the `xtbloom` target and its dependencies,
not tests, installs or wheels. Compiler launchers use available `ccache` unless
explicitly disabled. Source/compiler inputs are not retuned after failure.
The allowlisted build environment retains assigned `CUDA_VISIBLE_DEVICES` and
does not inherit hidden compiler flags or preload settings. No GPU probe,
test, profiler or benchmark is part of this operation.

The archive source-capture argv, literal exit/timeout and hashed log are retained
even if Git fails before producing a complete archive. Recognized CUDA `-real`
and `-virtual` suffixes are supported; nonpositive or duplicate numeric/suffix
requests are rejected consistently by the recorder and consumer.

The archive, commands and literal exits, logs, compiler/tool/provider bytes,
CMake cache, generated/build files and library are pinned. A fresh output
directory is mandatory; failed/partial operations remain failed receipts and
old evidence is never overwritten. Transitive build/runtime dependency closure,
actual OS mapped-image identity, independent scientific qualification, live
backend execution and performance/adoption remain separate incomplete gates.

## Record and inspect

Run from a clean reviewed recorder checkout. Choose a new Git-external output
directory; do not put large archives, binaries or raw build logs in tracked
benchmark evidence. The complete source archive preserves existing source
licenses/notices; provider binaries are referenced and hashed, not copied into
source or distribution payloads. No new dependency or linking policy is added.

```bash
python -m benchmarks.record_build \
  --source-root "$PWD" --revision "$(git rev-parse HEAD)" \
  --output /absolute/path/to/new-receipt-bundle \
  --backend cpu --cpu-linalg-library /absolute/path/to/verified-lp64-runtime.so \
  --jobs 8
```

CUDA compilation additionally uses `--backend cuda --nvcc /absolute/path/to/nvcc`
and `--cuda-architectures '<actual numeric architecture>'`. Select the actual
toolkit and architecture; do not infer CUDA compilation from `AUTO` or copy a
machine-specific architecture. Compilation alone is not real-GPU evidence.
Every later real-GPU command still requires finite-time `srun`, assigned device
visibility, and an allowed node; this tool provides no scheduler bypass.

The recorder reports the literal result and externally preservable receipt
SHA-256. Preserve that pin from the reviewed producing operation rather than
recomputing it from an arbitrary input file at consumption time.

```bash
python -m benchmarks.build_receipt \
  --receipt /absolute/path/to/new-receipt-bundle/receipt.json \
  --expected-sha256 '<independently preserved receipt SHA-256>' \
  --expected-source-revision '<reviewed full producer source revision>' \
  --library /absolute/path/to/the/exact/selected/libxtbloom.so
```

Matching equal-byte library copies are checked against the retained original
producer bundle. A library digest alone, an absent bundle, wrong source/tool
identity, failed command or postflight mutation is insufficient. The consumer
reports `LOCAL_RECEIPT_MATCHES`, **not** scientific or performance acceptance.
Receipts are canonical finite JSON, externally digest-pinned, schema-versioned
and limited to 1 MiB. Their archives/binaries/logs remain Git-external; existing
tracked-evidence size limits and narrow exclusions are unchanged.

## Optional paired-runner integration

Add all three flags together, only to the existing AO-only paired mode:

```text
--build-receipt /absolute/path/to/bundle/receipt.json
--build-receipt-sha256 <independently preserved receipt SHA-256>
--build-source-revision <full reviewed source revision>
```

Invalid pins fail before workload/native setup. Backend mismatch is rejected;
a CPU-only receipt cannot bind a requested CUDA coordinate. Successful
preflight and postflight file checks, their separately reported costs and the
still-unestablished loaded-image binding are retained in JSON metadata and
paired-row producer provenance. Checks are outside sweep timing, not hidden
inside a compute stage. A postflight failure returns nonzero **after retaining
all computed per-ID coordinates and writing JSON/CSV**. It never erases failed
peers or rewrites a library result as a different numerical outcome.

No receipt changes SCC options, cold/warmup/measured order, persistent owner
lifetimes, rollback, memory scope or finite-list mapping. Layout comparison and
independent E/F/q checks remain distinct. Missing primary q, original-corpus
science, three layout-sensitive iteration IDs, scientific blockers, corpus
rights/family identity and timing/memory gaps remain open. Do not promote a
performance flag merely because a new provenance field matches.

## Validation

```bash
python -m unittest -v benchmarks.test_build_receipt \
  benchmarks.test_record_build benchmarks.test_build_receipt_runner
```

These use synthetic archives, fake builds and fake benchmark adapters; they do
not load a fake native library or count as a real CPU/CUDA build. Focused and
full repository checks, independent exact-hash review and separately
preregistered clean real-build evidence remain required before admission.

The module and receipt schema are prototype benchmark contracts, not changes
to the stable public C ABI. Issue #513 and its draft PR remain open until their
complete acceptance matrix is satisfied. This feature is not approval for any
replacement #515 protocol, #514 holdout, production guard/default promotion,
issue closure or merge.
