# Issue #535: real-GPU correctness checkpoint, not performance qualification

PR #538 remains **Draft** and #535 remains **open**. This bundle corrects the
earlier hardware-unavailable assessment: each remote machine has its own Slurm
cluster. GPUs on n1 are usable even though the local node3 cluster is drained.
No job used maintenance node n3 or changed Slurm's `CUDA_VISIBLE_DEVICES`.

## Identity and environment

- Clean candidate: `9a5a3833637ddb375fe96a15e4351e3eaa4f52cc`.
- Clean base: `a756b076014ee2bb7f376b438be2f6cd5264d292`.
- Native sources, public headers, CMake support and the failing native test are
  unchanged between these revisions. The follow-up commit adds tests/evidence,
  not different production code; this is not a claim of a newer native build.
- Host: n1 / node1; NVIDIA GeForce RTX 5090, sm_120, driver 580.95.05,
  CUDA 12.9.86, GCC 13.3.0, CMake 4.2.1, ccache 4.5.1.
- Shared Release builds explicitly enable CUDA, use the discovered nvcc and
  architecture, and use host-isolated LP64 MKL from
  `/group/software/deepmd-kit-3.2.0/lib/libmkl_rt.so.3`. BLAS thread variables are
  all one. Library, provider, CMake cache, wheel, runner and input identities
  are in `validation-summary.json`.
- Remote root:
  `/home/jzzeng/codes/xtbloom-issue-535-20261015-9a5a383`.
  The issue's suggested `/data/ztlu/code/xtbloom` was not writable; existing
  experiments there were not modified.
- Python uses a non-editable, same-revision **CPU wheel** with an explicit
  `XTBLOOM_LIBRARY=<root>/build-cuda/libxtbloom.so` override. It is not a bundled
  CUDA-wheel qualification. The locked project environment uses Python 3.12.3,
  NumPy 2.5.1 and SciPy 1.18.0. The bootstrap tools environment used for the
  first test run has NumPy 2.5.4 and SciPy 1.18.1 instead.

## Results and limits

| Gate | Real job | Result |
| --- | --- | --- |
| Full registered native CUDA CTest, serial, 300s/test | 8601 | **211 PASS / 1 FAIL / 0 SKIP**, exit 8 |
| Native public API, clean base twice and candidate twice | 8605 | **0 PASS / 4 FAIL**, each exit 8, same assertion |
| Independent healthy32/healthy62, CPU host and CUDA host/device/mixed | 8602 | **12 property PASS / 12 FAIL** across 8 endpoints; all four runners exit 1 |
| New actual CUDA ordering/scatter/failure/repeat tests, tools environment | 8613 | **3 PASS / 0 SKIP**, exit 0 |
| Same three tests, locked project environment | 8619 | **3 PASS / 0 SKIP**, exit 0 |
| Existing full Python suite, locked project environment | 8615 | **508 PASS / 80 SKIP**, exit 0; Torch and CuPy absent |
| Optional borrowed-Torch diagnostic, not the locked Torch version | 8612 | **579 PASS / 1 FAIL / 8 SKIP**, then heap-corruption abort, exit 134; **FAIL** |
| Product full-E2E AB/BA performance matrix | None | **NOT RUN**: correctness gates remain failing |

The full native suite's sole failing test is `xtbloom.cuda.public_api`, scenario
`context-enqueue/device`, at `tests/cuda_public_api_test.cu:3339`:
`placed_info.completion_status == XTBLOOM_STATUS_SUCCESS`. The same assertion
fails in both clean revisions on two reruns each. Issue **#539** owns native
diagnosis; baseline reproduction is not a waiver of the failed gate. The other
native tests, including public conformance/invariants in host/device/mixed
modes, parity, Graph/cache and failure/publication tests, pass. This is not a
sanitizer run or a claim that all default-ordering qualification passed.

An initial all-target build failed in unchanged benchmark-only code at
`tests/cuda_aes2_test.cu:857` (incomplete `std::array`). The subsequent build
explicitly selected every non-`_benchmark` executable and every library/fixture
target, allowing all 212 registered CTest entries to execute. Do not call this
a successful all-target build. Earlier incomplete-fixture CTest logs are retained
remotely, separately from the final 211/1 result.

The external #523 manifest is hash-pinned at
`89dabddd69cbfce1e177b225fd6b95c38a0eb0f14aec7d42e8f5d879184067ab`.
The collection verifies both input and golden hashes against that manifest.
Healthy32 forces differ by about `7.914363e-4` Hartree/bohr; healthy62 energy
and forces differ by about `1.091100e-5` Hartree and `7.992314e-4` Hartree/bohr.
The original primary `5e-7` tolerances are unchanged. These remain **#523
scientific FAILs**, not evidence for changing a golden or attributing the issue
to Python ordering. This checkpoint does not diagnose the oracle discrepancy.

`python/tests/test_batch_ordering_cuda.py` uses real `backend="cuda"` calls,
without fake solvers or eligibility monkeypatches. It asserts global exact-AO
planning before atom-bounded slicing, canonical/inverse mappings, every requested
molecular and QM/MM output, SCC diagnostics, input immutability, peer-local
NaNs, repeated FRESH calls, geometry updates and changed budgets. It compares
against original-order batches and independent singleton calls, using
`rtol=0, atol=1e-12`. Its three passes do not replace the broader matrix. The
existing full Python invocation predates registration of this new file, which
was executed separately without dirtying the production checkout.

The optional Torch diagnostic borrowed shared Torch 2.12.0/CUDA 13.0, not locked
Torch 2.13.0. A Torch compile test failed because the shared package lacks
`torch/csrc/inductor/cpp_prefix.h`; after the test summary the process aborted
with `malloc(): unsorted double linked list corrupted`. That corruption is
**undiagnosed**, not dismissed as a packaging issue or ascribed to the AO queue.
An earlier less-configured Torch attempt also failed and is identified in the
raw-log inventory. Borrowed Torch was removed before the final locked run.
The 80 skips in that run include unavailable optional integrations; do not
present it as a full Torch/CuPy/device-wrapper qualification.

No native staging, physics, ABI, precision, SCC setting, conformance golden or
tolerance was modified to obtain these results. No sanitizer/profiler or timing
qualification is claimed. #539 and #523 are blocking references, **not delegated
completion** of #535. In particular the B64/B256 OMol25/heterogeneous/alkane and
singleton/homogeneous/rare-AO controls still need the preregistered full-warmup,
three-or-more AB/BA pairs, per-ID checks, context/memory accounting and complete
`BatchCalculator.compute` lifecycle measurements. An old sorted diagnostic or
direct-C paired runner is not that product E2E experiment.

## Reproduction boundary and commands

The paths below refer to the retained n1 workspace, not the caller's local
Slurm cluster. Run GPU commands via `ssh n1` and finite-time `srun`. A suitable
replay allocation for each test group is:

```bash
ssh n1
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:20:00 bash
```

Inside the allocation, preserve device visibility and use:

```bash
root=/home/jzzeng/codes/xtbloom-issue-535-20261015-9a5a383
export PATH="$root/tools/bin:$root/tooling/bin:/group/software/cuda-12.9.1/bin:$PATH"
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XTBLOOM_LIBRARY="$root/build-cuda/libxtbloom.so"
cd "$root/candidate"
ctest --test-dir "$root/build-cuda" --output-on-failure --timeout 300 \
  --parallel 1 --output-junit "$root/logs/native-cuda-replay.xml"
```

The native command intentionally exits nonzero at the recorded failing gate.
Execute the following independently so the failure does not suppress them:

```bash
.venv/bin/python -m pytest "$root/validation/test_batch_ordering_cuda.py" \
  -q -rs --junitxml="$root/logs/ordering-cuda-locked-replay.xml"
.venv/bin/python -m pytest python/tests -q -rs \
  --junitxml="$root/logs/python-cuda-locked-replay.xml"
for build_name in build-baseline build-cuda; do
  for repeat in 1 2; do
    ctest --test-dir "$root/$build_name" -R '^xtbloom\.cuda\.public_api$' \
      --output-on-failure
  done
done
manifest=/tmp/xtbloom-goal-20261009/healthy-oracle515/validation-manifest.json
for coordinate in cpu:host cuda:host cuda:device cuda:mixed; do
  backend=${coordinate%:*}
  mode=${coordinate#*:}
  "$root/tools/bin/python" tools/conformance/xtbloom_public_api.py \
    --library "$root/build-cuda/libxtbloom.so" --manifest "$manifest" \
    --backend "$backend" --memory-mode "$mode" \
    --actual-dir "$root/healthy523-replay/$backend-$mode"
done
```

The retained build scripts `build.sh` and `baseline.sh` configure each source
with these explicit arguments, substituting source/build directories:

```bash
cmake -S "$root/candidate" -B "$root/build-cuda" -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/group/software/deepmd-kit-3.2.0/lib/libmkl_rt.so.3 \
  -DPython3_EXECUTABLE="$root/tools/bin/python" \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
```

For the original target selection, environment setup and baseline reruns, use
the retained scripts through a separate `srun --cpus-per-task=16
--time=00:40:00 bash <script>` allocation. They select executable/library targets
from `ninja -t targets all`; they do not repair or hide the benchmark build
failure. The locked test dependencies were installed with
`UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --locked --no-install-project
--extra test --python /usr/bin/python3`, followed by non-editable installation
of the identified same-HEAD wheel. Bootstrap tools and borrowed Torch are not
the locked project resolution.

## Artifact handling and provenance

`validation-summary.json` records counts parsed from JUnit XML, clean Git
identities, binary/provider/source hashes, raw log/XML byte counts and SHA256s,
input/golden pins and unavailable coordinates. The full original logs, XML,
actual JSON and CMake caches remain in the remote root. They are reproducible
diagnostic artifacts, not an immutable external archive or a published timing
dataset. The original input/oracle artifacts remain at their pinned remote
paths; no new external data or third-party code is copied into this bundle.
The installed runtime/provider/vendor libraries are identified, not redistributed.

To stay within the repository's 1 MiB/file and 16 MiB total evidence limits,
large raw XML and Python/Torch logs are omitted. Files ending in
`.normalized.log` are **derived text**, with trailing horizontal whitespace
removed from the original logs by `sed -E 's/[[:blank:]]+$//'`; no result line or
numeric value is removed. All other retained raw logs are byte-identical
copies. `SHA256SUMS` covers the retained files, while the JSON preserves the
identities of the unmodified remote originals. No hook exclusion or evidence
budget was widened.

```bash
cd benchmarks/evidence/issue-535/2026-10-15-node1
sha256sum --check SHA256SUMS
```
