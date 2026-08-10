---
name: xtbloom-validate-cuda-change
description: Validate xTBloom CUDA kernels and public runtime behavior on a real NVIDIA GPU, including CPU parity, host/device/mixed descriptors, ragged failures, streams, Graphs, cache and WARM epochs, publication, installed consumers, sanitizers, and steady-state constraints. Use for CUDA source, CUDA-visible ABI, runtime staging, cache, SCC, force, performance, or release-gate changes.
---

# Validate a xTBloom CUDA Change

Prove both numerical behavior and runtime contracts on actual hardware. Read `AGENTS.md`, `docs/architecture.md`, the active issue, and every directly blocking CUDA issue before selecting the matrix.

## Capture the Environment

Record the tested commit, clean/dirty state, GPU model, driver, CUDA compiler and toolkit versions, selected architecture, scheduler allocation, host compiler, and LP64 runtime. Discover local paths; never copy a machine-specific `nvcc` path or `sm_120` unconditionally.

Configure a fresh CUDA build explicitly:

```bash
cmake -S . -B build/cuda -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/actual/path/to/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=<actual-architecture> \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda --parallel
ctest --test-dir build/cuda -N
```

Confirm that CUDA targets and the required public tests are registered. `AUTO`, compile-only success, an unavailable backend skip, or a hosted runner using a driver stub is not real-GPU evidence.

## Construct the Affected Matrix

Select the Cartesian product needed by the changed contract:

| Axis | Cases to consider |
| --- | --- |
| Descriptor memory | Host, CUDA device, and mixed inputs/outputs |
| Batch topology | Empty, singleton, homogeneous, heterogeneous ragged, changed geometry, changed topology |
| Electronic state | Restricted/unrestricted, charge/spin edges, FRESH, strict WARM, same/changed epoch |
| Embedding | No point charges, explicit point charges, periodic shifts, charge-response matrix |
| Execution | Default/custom stream, active stream capture rejection, repeated call, concurrent/single-flight contract |
| Outcome | Success, invalid descriptor before execution, SCC/eigensolver peer failure, injected staging/launch/publication failure |
| Publication | Every requested result combination, host/device result buffers, complete NaN slices, sentinel preservation |

Do not reduce a public-path change to a kernel unit test. Include the terminal runtime and caller-output bridge.

## Run Numerical and Public Gates

1. Run the focused CUDA unit tests while iterating.
2. Run full CUDA CTest under the required scheduler on the real GPU.
3. Run public conformance for every affected memory mode with `tools/conformance/xtbloom_public_api.py`.
4. Run `tools/conformance/xtbloom_invariants.py` for affected modes.
5. Compare CPU and CUDA on identical descriptors, including ragged peer failures and requested property combinations.
6. Verify the caller's current device is restored on success and every failure phase.
7. Verify stream capture is rejected transactionally and no output commit begins.

Use the exact commands in `tools/conformance/README.md`; keep output directories separate by backend and memory mode.

## Run Runtime Safety Gates

Run every issue-required binary under all four tools on an allocated GPU:

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 <test-binary>
compute-sanitizer --tool racecheck --error-exitcode=99 <test-binary>
compute-sanitizer --tool initcheck --error-exitcode=99 <test-binary>
compute-sanitizer --tool synccheck --error-exitcode=99 <test-binary>
```

Choose binaries that traverse the changed production path and failure path; a sanitizer-clean leaf kernel alone is insufficient. Preserve exact tool versions, commands, exit codes, and error/hazard counts.

### Apply Owner-Approved External-Tool Dispositions

Never suppress sanitizer output globally or rewrite a tool failure as a clean
run. Reuse an owner-approved disposition only when every recorded environment,
signature, control, and scope condition matches durable issue, PR, or
checked-in evidence.

Compute Sanitizer 2025.2.1 cannot express this waiver as a native suppression:
its `--suppressions` support excludes `synccheck` barrier reports.
`--kernel-name-exclude` skips complete kernels and does not safely preserve
coverage for this device-launched child-Graph case. Do not use a broad filter
such as `regex=.*` merely to obtain exit code zero.

The xTBloom Blackwell device-Graph `synccheck` disposition applies only to:

- NVIDIA GeForce RTX 5090 (`sm_120`), driver 580.95.05;
- CUDA 12.9.86 and Compute Sanitizer 2025.2.1.0 build 35969825;
- device-launched Graph execution using Fire-and-Forget or Tail Launch;
- reports containing only `Barrier error detected. Divergent thread(s) in warp`
  at `reduce_spin_atomic_charges_kernel+0x2b0`;
- an unchanged matching kernel and device-launch path relative to the archived
  issue #279 evidence;
- a clean uninstrumented production run, clean direct and host-Graph controls
  recorded by the issue #279 disposition, and zero memcheck, racecheck, and
  initcheck findings.

When every condition matches, preserve the literal `synccheck` exit code and
error count, classify the acceptance row as
`PASS (OWNER-DISPOSITIONED UPSTREAM TOOL DEFECT)`, and state explicitly that
the tool did not return a clean pass. Do not repeat the root-cause investigation
or refactor correct CUDA code solely to silence this signature. Cite issue #279
and its evidence bundle in the ledger.

When the affected kernel, Graph launch control, GPU, driver, toolkit, and tool
version are unchanged, do not rerun the known-failing device-Graph coordinate
solely to reproduce the archived result. Reuse the disposition and run the
clean direct/host-Graph control plus the other applicable sanitizer gates.

Run `synccheck` again when the affected kernel, Graph launch control, GPU,
driver, CUDA toolkit, Compute Sanitizer version, or requested path changes. Any
different kernel, program counter, report class, environment, additional
finding, failed clean control, or nonzero result from another sanitizer is not
covered and remains a blocker. Never use the disposition to claim that
`synccheck` itself returned a clean pass.

For ABI, packaging, host-runtime, or publication changes, also:

- Install the shared CUDA build into an isolated prefix.
- Build `tests/install_consumer` against that prefix.
- Run consumer modes that strictly request `smoke`, `cpu`, and `cuda` as applicable.
- Run symbol and CUDA dependency checks from `tests/abi/`.
- Confirm missing drivers or host runtimes produce the specified diagnostic rather than an ELF loader failure.

## Prove Steady-State Constraints

When execution, cache, Graph, or performance code changes, instrument repeated public calls after warmup. Prove the issue's requirements for allocation count, address stability, Graph reuse, transfers, host polling, synchronization, active-set behavior, and topology/geometry epoch transitions. Use the dedicated performance-evidence workflow for timing or profiler claims.

## Report Missing Evidence Honestly

Return these gates to the caller's validation ledger. Report exact pass/fail/skip counts and each tested matrix coordinate. Mark real-GPU, sanitizer, installed-consumer, profile, or memory-mode gaps as unavailable or not run; do not let green hosted CI close a runtime acceptance criterion.
