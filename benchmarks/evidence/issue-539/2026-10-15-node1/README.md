# Issue #539: automatic-spin topology repair and validation checkpoint

Draft PR #540 repairs the native blocker of #535 / Draft PR #538. Both issues
remain open. This is correctness/resource evidence, **not performance
qualification**, a sanitizer waiver, or a claim that every release gate passes.

## Source and binary boundary

- Clean native production revision: `e74f6215ab9142860faa8b2cb833bc0b5b5be268`,
  based on `a756b076014ee2bb7f376b438be2f6cd5264d292`.
- Native library SHA256:
  `8a450b0d2b1468fbff874479c4016ed0d2b5f7a7ad3049c6b1f124bbd9566875`.
- Public CUDA test SHA256:
  `48c0285ec894c066964361b3057e24d4f67b19bcfb8c031798c9e935e815008f`.
- RTX5090/sm120, driver580.95.05, CUDA12.9.86, GCC13.3.0,
  CMake4.2.1, ccache4.5.1. The shared Release build explicitly enables CUDA
  and uses the discovered toolkit and host-isolated LP64 MKL provider.
- Remote root `R=/home/jzzeng/codes/xtbloom-issue-539-20261015` on n1/node1.
  `validation-summary.json` records source cleanliness, complete binary/cache/
  provider identities, environment, jobs and results. `raw-artifacts.json`
  pins raw logs, XML, scripts and diagnostic source by relative path, size
  and SHA256. Those larger artifacts remain outside Git under `R`.
- Every real-GPU command uses finite-time Slurm `srun`, preserves the assigned
  `CUDA_VISIBLE_DEVICES`, and excludes maintenance node n3. Concurrent jobs
  exist; no timing here supports a latency or speed claim.

The diagnostic unfixed request completes with INVALID_ARGUMENT and the
request-owned error `the batch topology does not match the fixed CUDA plan
topology`. A successful wait is not computation success. #522 removed the
automatic-selector normalization previously implemented by #508. The repair
restores all five CUDA paths: explicit auto0 compares by effective1/2, while
missing/NULL/ABI-v1 remains restricted1. Incoming UHF is independently checked;
changed effective topology remains invalid. No ABI, physics, golden, tolerance,
allocation, transfer, launch or synchronization change is introduced.

Regression coverage alternates auto-FRESH, equivalent explicit-WARM and
auto-WARM for host/device/mixed descriptors. Two older negative fixtures now
write invalid -1 rather than valid auto0, retaining deferred rejection and
unchanged-output assertions. A missing standard `<array>` include also repairs
the all-target AES2 benchmark compilation. Independent static native review
found no blockers; validation gaps below still prevent merge readiness.

## Results

| Gate | Job | Literal result |
| --- | --- | --- |
| CPU shared Release, full registered CTest | local | 107 PASS / 0 FAIL / 0 SKIP; exit0 |
| CPU shared install, license audit, external CPU consumer | local | PASS |
| CUDA explicit-ON all-target build, including benchmarks | 8639 | PASS; 212 CTests registered |
| Initial full CTest with diagnostic 300-second cap | 8639 | 211 PASS / 1 TIMEOUT / 0 SKIP; exit8 |
| Full public API binary, 1200-second finite cap | 8652 | exit0; 534.96 seconds; two-device-only checks internally skipped |
| Final full CUDA CTest, **two allocated GPUs** | 8661 | **212 PASS / 0 FAIL / 0 SKIP; exit0** |
| CUDA shared install, license audit, external smoke/cpu/cuda consumers | 8649 | 3 consumer modes PASS; audit exit0 |
| Steady-state context/plan request profiles | 8649 | both exit0; 10 WARM calls each; resource evidence only |
| Locked Torch compile regression | 8659 | 1 PASS; process exit0 |
| Locked Torch module, including actual CUDA paths | 8659 | 51 PASS / 1 expected CPU-only-build SKIP; process exit0 |
| Combined #535 Python suite with fixed native #539 | 8666 | 583 passing cases / 8 skips, **then heap abort134: FAIL** |

The apparent later periodic hang was an insufficient **diagnostic** cap, not
a demonstrated deadlock. Original main also exceeded a short periodic-only
cap; CUDA-GDB observed an active serial native multipole kernel. The complete
uninstrumented public binary subsequently exits0 without a source, scientific
setting or acceptance-threshold change. Final two-GPU CTest uses an appropriate
finite 1200-second per-test watchdog and exercises current-device restoration
and wrong-device rejection rather than their one-GPU skips. Full public
conformance/invariants for host/device/mixed and native failure/publication,
cache/Graph/parity tests are included among the 212 passing tests.

CPU raw evidence stays locally at `/tmp/xtbloom-539-cpu-public.log` (55993 bytes,
SHA256 `145d46f47ebd52e8b91df551161715fff630f5cec2b5135e80887b6e6412bc5b`).
`build/cpu-public/results.xml` is 39801 bytes, SHA256
`3308210408dd09e1662fc803c60a318199446075add8e970dba1a3ca95af3dea`;
CPU library SHA256 is
`026e35c2c1988671880f76007535f83690d5cf7889e42b5f64a57b0dfbc7d466`.
Its CMake cache SHA256 is
`904ad807ec0adcbf510e5343f3eea4c3795a2bf56b3d88ceb6e606fcad238261`.

## Sanitizers: unresolved, not owner-dispositioned

Both `--context-request-sanitizer` and `--request-sanitizer` traverse changed
fixed-topology validation and deferred invalid-topology publication paths.
Each ran under all four tools with no suppressions or kernel exclusions:

| Compute Sanitizer | memcheck, two modes | initcheck, two modes | racecheck, two modes | synccheck, two modes |
| --- | --- | --- | --- | --- |
| 2025.2.1.0 build35969825 / toolkit12.9 | exit0, 0 errors | exit0, 0 errors | exit11, target SIGSEGV | exit99, 800/1024 errors |
| 2025.3.1.0 build36400806 / standalone13.0 tool | exit0, 0 errors | exit0, 0 errors | exit11, target SIGSEGV | exit99, 1024/800 errors |
| 2026.2.1.0 build38334959 / standalone13.3 tool | exit0, 0 errors | exit0, 0 errors | exit11, target SIGSEGV | exit99, 1024/1024 errors |

Racecheck prints zero hazards but the target crashes; it is **not a pass**.
A diagnostic shell records signal11/exit139, and GDB captures the fault on a
worker inside `libsanitizer-collection.so`. This does not prove application
code cannot contribute. Disabling only report backtraces does not resolve it.

Synccheck includes a new divergent-warp report at electric-field
`preflight_values_kernel+0x170`, and its instrumented application also fails
settlement with cudaErrorLaunchFailure. This is **outside #279's exact reducer
signature**; neither #279 nor #80 is extended or reused as a waiver. Error
summaries retain default print-limit notices; unprinted findings are not
asserted to match the printed subset. Direct SCC and normal host-Graph controls
pass, but they do not replace coverage of the changed public request path.
The first attempted host-control flag was invalid and exited2; corrected
`--crossover-sanitizer-control 41 1 1 host-graph` receipts are kept separately.

Test-only field Graph controls isolate unchanged production field kernels:
uninstrumented direct/host/device executions and racecheck controls pass.
Synccheck direct and ordinary host Graph pass, but a Graph instantiated with
`cudaGraphInstantiateFlagDeviceLaunch` reports 256 errors at the same new
field-kernel PC, whether launched by the host or by a device parent. This is
diagnostic evidence for a new scoped tool investigation, **not an approved
disposition** and not a minimal reproduction of the public racecheck crash.
The standalone sources/binaries and literal logs remain under `R`; no
production workaround is added just to silence instrumentation.

The alternative tools are test-only NVIDIA packages extracted outside source,
build, install, sdist and wheel payloads. HTTPS repository metadata and package
digests are verified before extraction; their included CUDA EULA/copyright
notices remain with the extracted tools. No project dependency or license
boundary changes. Package pins:

- `cuda-sanitizer-13-0_13.0.85-1_amd64.deb`, SHA256
  `5913520009ecc86be1c62b5793b032f81fdffdfcd4493da6212e14c3dc1f35a4`.
- `cuda-sanitizer-13-3_13.3.75-1_amd64.deb`, SHA256
  `ab5467d9473adfb528481d1b6166b9bc718ab668563a3466d0f2c5d8dd27aa4f`.

## Python teardown boundary

Python3.13.9/Torch2.13.0+cu130 uses the locked non-editable #535 production
wheel from 9a5a383, with clean 0517c32 test sources and explicit installed
#539 core/adjacent-extension selection. Python package source is unchanged
between those #535 revisions. This is a **two-source integration diagnostic**,
not a same-revision bundled CUDA-wheel pass. Actual Torch module and mapped
`libtorch_cpu.so` paths are logged; user site/PYTHONPATH are excluded and matching
Python/Inductor headers are present.

Pure Torch compilation, CPU eager/autograd and the complete Torch module pass
in separate processes. Nevertheless the full combined suite aborts after its
583pass/8skip summary with `malloc_consolidate(): unaligned fastbin chunk
detected`. The suite remains FAIL. #541 owns reduction and root-cause repair;
missing headers or a borrowed Torch version cannot explain this new result.
The 8 skips are unavailable JAX/CuPy adapter coordinates and an intentionally
CPU-only-build test, not successful GPU coverage. The initial test-only archive
omitted root fixtures and failed collection; the final complete clean clone
is recorded separately. A probe's initial PosixPath JSON-print error is also
retained as a diagnostic-fixture error, not a native failure.

## Exact replay entry points

All scripts below are hash-pinned in `raw-artifacts.json`. CUDA configure uses
`-DXTBLOOM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`, the recorded compiler,
all supported ccache launchers, shared Release, an absolute isolated LP64 runtime
and Python3.12.3 for CTest tooling. BLAS thread variables remain one.

```bash
ssh n1 'srun --partition=main --gres=gpu:5090:2 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=24G --time=00:27:00 \
  bash /home/jzzeng/codes/xtbloom-issue-539-20261015/full-final.sh'
```

`full-final.sh` runs `ctest --test-dir "$R/build-cuda" --output-on-failure
--timeout 1200 --parallel 1 --output-junit "$R/logs/native-cuda-two-gpu-final.xml"`.
For all-target rebuild, use `full-validation.sh`; its initial 300-second
diagnostic CTest cap is historical and not the final matrix. Installed consumer
and profiler commands are in `install-profile.sh`; the two 10-iteration ranges
use cudaProfilerApi capture with `--trace=cuda,osrt --cuda-memory-usage=true`.
Derived `cuda_api_sum` counts are retained in `profile-api-counts.csv`; raw
captures/SQLite remain outside Git. Neither range allocates/frees, creates or
destroys streams/events/Graphs, queries host readiness, or calls device-/stream-
wide synchronization. Each has exactly ten cudaEventSynchronize calls for the
documented synchronous wait boundary. Existing requested input/output transfers
remain visible; no zero-transfer claim is made.

`safety.sh` runs memcheck/racecheck/initcheck; `controls-synccheck.sh` records
fresh public synccheck. `sanitizer13-matrix.sh` and
`sanitizer-version-matrix.sh sanitizer133 13.3` preserve alternative-tool
coordinates. Every sanitizer command is `timeout 300` (or the recorded600)
`compute-sanitizer --tool TOOL --error-exitcode=99 BINARY --MODE`.
`python-complete-final.sh` runs the complete locked Python integration diagnostic;
its required exit status is134 here, never interpreted as a passing suite.

CPU validation explicitly configures CUDA OFF/shared Release with absolute
LP64 MKL and ccache, builds all targets, runs full CTest, then installs, audits
licenses, builds the external install consumer and executes its cpu mode.
Local full prek stalled in isolated ty/uv check and was terminated, not passed;
all remaining hooks pass with `SKIP=ty`. `uv lock --check` and diff checks pass.
Hosted final production-head CI has 33 SUCCESS / 3 publication/deployment SKIP,
but does not replace any unresolved GPU/sanitizer/Python/oracle gate.

Next: finish public racecheck/new-signature qualification, fix #541, and obtain
the explicit #523 scientific-reference decision. No parent closure, merge,
golden replacement, tolerance change or full-E2E speed qualification follows.
