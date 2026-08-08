# AGENTS.md

This file applies to the entire repository. A more deeply nested `AGENTS.md`,
if one is added later, overrides this file only for its subtree.

## Project mission

gpuxtb is a C++17 library for high-throughput GFN-xTB inference through one
stable C ABI. GFN2-xTB is implemented on CPU and CUDA, including restricted
and unrestricted SCC, analytic forces, ragged batches, explicit point charges,
and periodic charge response. Python, ASE, and dpdata interfaces all call the
same public C ABI.

Correctness, ABI stability, failure isolation, reproducible scientific
evidence, and legal provenance take priority over small performance or code
size improvements. Do not weaken an existing acceptance gate to make an
implementation pass.

GFN1-xTB and ROCm values are reserved in the ABI but are not implemented.
Never report them as supported.

## Start-of-task protocol and external memory

GitHub issues are the cross-agent and cross-session source of truth. Do not
rely on chat context as the only record of plans, decisions, or remaining work.

Before starting material work:

1. Read the active issue, its relevant parent/sub-issues, and every directly
   blocking issue.
2. Read Epic #1 only when the work changes or depends on its release gates,
   priority queue, or issue hierarchy, or when no narrower issue identifies
   the next work. Epic #1 is a release dashboard, not a per-task activity log.
3. Inspect the current branch, HEAD, worktree status, open PRs, and relevant CI.
4. Record the leaf issue's branch, scope, acceptance criteria, dependencies,
   and planned test matrix if they are not already current.
5. Work in priority order: `priority: urgent`, then `priority: high`, then
   unlabelled work. Apply the repository priority labels when triaging issues.

At every material checkpoint, update the active issue with:

- branch, commit, and PR number;
- exact commands and pass/fail/skip counts;
- decisions and important invariants;
- blockers and remaining acceptance criteria;
- the next concrete action.

Record routine merge and validation evidence in the active issue. Update Epic
#1 only when its dashboard materially changes, such as a release gate being
added, completed, reopened, or re-scoped, or the priority queue changing. Do
not duplicate leaf-issue checkpoints or a chronological merge log in the Epic.

An issue may close when its primary scope is complete and only small residual
work remains, provided every residual item is transferred to an open issue
connected through GitHub's parent/sub-issue relationship. Before closing:

- map each transferred acceptance item to its sub-issue in the parent ledger;
- ensure each sub-issue has independently actionable scope, acceptance
  criteria, dependencies, and required evidence; and
- state in the closing comment that the delegated items remain incomplete and
  that closing the parent does not claim their completion.

An issue is closable when every acceptance item is either satisfied or validly
delegated under this rule; otherwise leave it open. Delegation transfers
tracking responsibility but never converts missing profiling, sanitizer,
packaging, legal, conformance, or other evidence into a pass. Do not move the
issue's core deliverable or broad unfinished scope into sub-issues merely to
make the parent closable.

When any acceptance item is delegated, use `Refs #N` in the PR rather than
auto-closing the parent at merge time. After the merge, post the final ledger,
main commit, child links, and explicit incomplete-items statement, then close
the parent manually. `Closes #N` is reserved for issues whose acceptance items
all pass without delegation.

## Repository map

- `include/gpuxtb/gpuxtb.h`: the only public C ABI. It defines contexts,
  caller-owned buffers, ragged batches, result descriptors, status values, and
  `gpuxtb_compute`.
- `src/api.cpp`: public entry points, structure initialization, context
  lifetime, error reporting, and compute dispatch.
- `src/runtime/`: backend selection, public descriptor validation, CPU batch
  execution, CUDA staging/cache ownership, and CUDA pointer validation.
- `src/model/gfn2/`: CPU GFN2 physics and the primary readable reference for
  equations and semantics.
- `src/backends/common/`: plan and workspace schemas shared across backends.
- `src/backends/cuda/`: CUDA terms and the complete setup/SCC/publication/force
  execution chain.
- `python/gpuxtb/`: ctypes ABI mirror, native runtime loading, high-level
  interfaces, ASE calculator, and dpdata plugin.
- `tests/`: C/C++/CUDA unit, ABI, conformance, install-consumer, licensing,
  parameter, and oracle tests.
- `python/tests/`: Python wrapper and integration tests.
- `data/parameters/`: generated GFN2/D4 parameter artifacts and provenance.
- `data/conformance/`: hash-pinned scientific inputs and independent goldens.
- `tools/`: parameter generation, conformance, licensing, and SCC trace tools.
- `benchmarks/`: persistent public-API benchmark adapters and harness.
- `.github/workflows/`: CPU/package CI, read-only pre-commit checks, and Linux
  x86_64/aarch64 CUDA wheel builds.

Read `README.md`, `docs/developer-guide/architecture.md`,
`docs/user-guide/qmmm.md`, `docs/theory/qmmm.md`,
`tools/conformance/README.md`, and `tools/parameters/README.md` before
changing the corresponding subsystem.

## Non-negotiable scientific and ABI invariants

### Units and numerical meaning

- Public real-valued data uses IEEE binary64 and atomic units.
- Positions are in bohr, energy is in Hartree, and forces are in Hartree/bohr.
- Forces are the negative coordinate derivative of the reported energy with
  caller-supplied periodic charge-response fields held fixed. When
  `atomic_potential_shifts` or `charge_response_matrix` is supplied, gpuxtb
  does not include `db/dR` or `dA/dR`; callers own those derivatives and the
  result sets `GPUXTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES`.
- `electronic_temperature` is the energy scale `k_B T` in Hartree, not a
  Kelvin value.
- At finite electronic temperature, the reported variational energy is the
  electronic Helmholtz free energy.
- CPU and CUDA implement the same equations, parameter values, SCC state, and
  failure semantics. A backend-specific shortcut needs parity evidence.

### Public ABI

- Do not reorder, delete, resize, or reinterpret existing public fields.
- Public tag types remain fixed-width `int32_t`, independent of C enum layout.
- Every extensible structure starts with `struct_size` and `api_version`.
  Extend structures only by appending a versioned suffix.
- Preserve short-structure compatibility: never read a suffix unless
  `struct_size` proves that it is present.
- Update size macros, initialization, validation, static layout assertions,
  Python ctypes mirrors, install consumers, and ABI tests together.
- Buffers are caller-owned borrowed views. The library never takes ownership.
- Ragged topology uses flat arrays with `int64_t` offsets. Validate extents,
  overflow, aliases, memory-space tags, and pointer ownership before execution.
- Linux exports are restricted by `cmake/gpuxtb.map`. Public symbols use the
  `gpuxtb_*` namespace; do not expose implementation symbols accidentally.

A C ABI change normally requires coordinated review of at least:

- `include/gpuxtb/gpuxtb.h`
- `src/api.cpp`
- `src/runtime/validation.*`
- `src/runtime/cuda_descriptor_validation.*`
- `python/gpuxtb/library.py`
- `python/gpuxtb/interface.py`
- `tests/c_api_test.c`
- `tests/batch_validation_test.cpp`
- `tests/abi/check_symbols.py`
- `tests/install_consumer/`

### Failure and publication semantics

- Validate the complete request before executing or touching caller outputs.
- A failure before caller-output commit leaves result flags and buffers
  unchanged.
- Per-system SCC or eigensolver failure is data-level, not call-level: peers in
  the ragged batch remain independent and successful peer results survive.
- A failed system's requested floating-point slices are fully filled with
  quiet NaNs; never publish a partial result.
- After CUDA caller-output commit begins, a catastrophic failure may have
  modified output but must return `GPUXTB_STATUS_INTERNAL_ERROR` with a useful
  diagnostic.
- CUDA paths must attempt to restore the caller's current device on every exit.
- Public CUDA compute is synchronous and rejects active stream capture.

### CPU and CUDA execution

- The CUDA backend supports host, CUDA-device, and mixed descriptors. Test all
  affected memory modes, not only the easiest one.
- Preserve fixed-topology cache validation, workspace ownership, and strict
  warm-state epoch rules. CUDA `WARM` never silently falls back to `FRESH`.
- Do not add steady-state per-call allocations, iteration-level host polling,
  device-wide synchronization, or host transfers without an explicit issue
  decision and profiling evidence.
- CPU batch parallelism is the outer parallel layer. Keep BLAS one-threaded to
  avoid nested oversubscription.
- The CPU eigensolver requires one dlopen-able monolithic LP64 LAPACKE+CBLAS
  runtime. Do not accept ILP64 providers, static archives, or introduce a hard
  `DT_NEEDED` dependency merely to simplify discovery.
- MKL providers must be host-isolated (issue #30). When CMake selects MKL, it
  builds a private shim with fixed `DT_NEEDED` dependencies on
  `libmkl_intel_lp64`, `libmkl_sequential`, and `libmkl_core`, and the runtime
  factory loads the adjacent shim with `RTLD_LOCAL` in a new glibc link-map
  namespace. `RTLD_LOCAL` in the base namespace is not isolation because
  pre-existing global symbols can still interpose. gpuxtb must never call
  `MKL_Set_Interface_Layer`, never read `MKL_INTERFACE_LAYER` for the isolated
  path, and never expose provider libraries with `RTLD_GLOBAL`. Do not regress
  host coexistence: LP64 gpuxtb calls must stay correct when the host uses
  ILP64, both before and after gpuxtb backend creation.

## Generated, canonical, and licensed artifacts

Do not hand-edit generated or hash-pinned artifacts to make a check pass.

- GFN2 files `data/parameters/gfn2.toml`, `gfn2.json`, `gfn2.hpp`, and their
  manifest are generated by `tools/parameters/generate_gfn2.py`.
- D4 tables and manifests must follow the pinned-revision procedure in
  `tools/parameters/README.md`.
- Conformance goldens are independent scientific oracle evidence. Regenerate
  them only through the pinned tblite/xTB workflows in
  `tools/conformance/README.md`; never rewrite a golden to match current code.
- SCC trace schemas, fixtures, and comparator inputs are versioned contracts.
  Preserve canonical serialization and provenance hashes.
- A new dependency, copied source, generated dataset, wheel payload, or install
  artifact requires a licensing/provenance review. Update
  `THIRD_PARTY_NOTICES.md`, `LICENSES/`, manifests, and packaging checks as
  applicable.
- Keep pre-commit exclusions narrow and exact. Do not exempt a directory or
  file class because one canonical artifact needs byte preservation.

## Code changes and tests

- Search with `rg`/`rg --files` before editing and follow nearby conventions.
- Preserve unrelated user changes in dirty worktrees. Do not reset, overwrite,
  or reformat unrelated files.
- Add source files and native tests to `CMakeLists.txt`; an unregistered file is
  not part of the build.
- Add focused tests with every behavior change. For physics and forces, prefer
  term-level checks, finite differences, invariance/conservation checks,
  public-API coverage, and CPU/CUDA parity as appropriate.
- Cover empty inputs, ragged mixed batches, hostile sizes/tags, non-finite
  values, peer-local failures, repeated calls, changed geometry, changed
  topology, and cache/warm-state transitions when relevant.
- Document public APIs, complex logic, invariants, edge cases, and
  backend-specific behavior. Comments should explain why, not restate syntax.
- Do not make performance claims from one timing. Record hardware, compiler or
  toolkit, thread count, workload, descriptors, warmup, sample count, raw
  artifacts, and correctness alongside timing.

## Validation matrix

Run the smallest relevant checks while iterating, then the full applicable
matrix before handoff. Report exact commands and results; never imply that a
skipped or unavailable backend passed.

The native build requires CMake 3.24 or newer and C/C++17. Configuring tests
requires a Python 3.11-or-newer interpreter even though the installed Python
package supports Python 3.10 or newer. The repository has no CMake presets or
wrapper task runner; the explicit CMake/CTest commands below are authoritative.
Use separate build directories for CPU, shared, static, and CUDA configurations
so an old `CMakeCache.txt` cannot hide which tests were enabled.

### Fast repository checks

```bash
uvx prek@0.3.1 run --show-diff-on-failure --color=always --all-files
uv lock --check
git diff --check
```

### CPU Release build and CTest

```bash
cmake -S . -B build/cpu -G Ninja \
  -DGPUXTB_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
```

For a production CPU inference/conformance build, provide an absolute path to
a compatible LP64 runtime when auto-discovery is unavailable:

```bash
cmake -S . -B build/cpu-public -G Ninja \
  -DGPUXTB_ENABLE_CUDA=OFF \
  -DGPUXTB_MKL_RT_LIBRARY=/absolute/path/to/libmkl_rt.so \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
```

Use `ctest --test-dir <build> -R '<regex>' --output-on-failure` for targeted
native tests. Without a compatible LP64 runtime, the core build can still be
green while eigensolver, public CPU inference, and public conformance tests are
absent. Treat that smaller CTest set as partial validation, not a full pass.

### Python

The Python dependencies are managed with uv. `uv.lock` is the canonical,
committed resolution against PyPI; resolve or install only against
`https://pypi.org/simple` (uv records the index URL in the lock, so a mirror
index makes `uv lock --check` fail and `uv sync` rewrite `uv.lock`). After
changing `pyproject.toml`, run `uv lock` and commit the lock with the change.

In a clean isolated environment, build/install the local native package with
the test extras before running the suite; a missing native library is a test
failure, not a supported skip. Install non-editable (`--no-editable`) so
`library_path()` finds the bundled `libgpuxtb` inside the wheel, and run with
`--no-sync` so `uv run` does not re-sync the environment as editable:

```bash
GPUXTB_ENABLE_CUDA=OFF uv sync --no-editable --extra test
uv run --no-sync pytest python/tests -q
```

Run the relevant focused file first for wrapper changes. Python tests that need
the CPU eigensolver also need a compatible MKL/OpenBLAS runtime available to
the loader.

### Canonical data, conformance, oracle, and licensing

```bash
python3 tools/parameters/generate_gfn2.py --check
python3 tools/conformance/gpuxtb_conformance.py check
python3 tools/licensing/check_licenses.py --source-root .
python3 -m unittest discover -s tests/parameters -p 'test_*.py' -v
python3 -m unittest discover -s tests/conformance -p 'test_*.py' -v
python3 -m unittest discover -s tests/licensing -p 'test_*.py' -v
python3 -m unittest discover -s tests/oracle -p 'test_*.py' -v
```

For benchmark harness changes, run:

```bash
python3 -m unittest -v benchmarks.test_run
# Requires PyTorch but uses a fake differentiable dxtb runtime.
python3 -m unittest -v benchmarks.test_dxtb_adapter
```

### CUDA

For CUDA changes, configure explicitly with `GPUXTB_ENABLE_CUDA=ON`; `AUTO` can
silently produce a CPU-only build when `nvcc` is not found. Select the actual
CUDA compiler/toolkit and GPU architecture rather than copying a
machine-specific path or `sm_120` unconditionally. Run the affected
`gpuxtb.cuda.*` tests on a real NVIDIA GPU. Run public conformance for host,
device, and mixed memory modes when the public execution path changes. Use the
local scheduler (for example `srun`) if the machine requires one.

Release or sanitizer issues may additionally require Compute Sanitizer
`memcheck`, `racecheck`, `initcheck`, and `synccheck`. A compile-only CUDA build
does not satisfy a runtime or sanitizer acceptance criterion. Current GitHub
wheel jobs compile CUDA but run without a real NVIDIA driver using a loader
stub and CPU fallback, so green hosted CI alone is not runtime evidence for a
CUDA backend change.

### Packaging and installation

Changes to the ABI, CMake export, dependencies, licenses, or Python packaging
must also exercise the relevant shared/static install consumer, source
distribution, and wheel checks represented in `.github/workflows/ci.yml` and
`.github/workflows/wheels.yml`.

For a shared CPU package with full public inference coverage:

```bash
cmake --build build/cpu-public --parallel
ctest --test-dir build/cpu-public --output-on-failure
cmake --install build/cpu-public --prefix "$PWD/build/cpu-public-install"
python3 tools/licensing/check_licenses.py --source-root . \
  --install-prefix "$PWD/build/cpu-public-install"
cmake -S tests/install_consumer -B build/cpu-public-consumer -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/build/cpu-public-install"
cmake --build build/cpu-public-consumer --parallel
"$PWD/build/cpu-public-consumer/gpuxtb_install_consumer" cpu
```

For source-distribution changes:

```bash
uv build --sdist --out-dir build/dist-license
python tools/licensing/check_licenses.py --source-root . \
  build/dist-license/*.tar.gz
```

`ci.yml` covers Release CPU GCC/Clang, source licenses, static package export,
and shared public CPU inference/install. `wheels.yml` builds Linux x86_64 and
native aarch64 CUDA wheels and checks size/licenses. `pre-commit.yml` runs the
pinned full-file `prek` command above. All three use `pull_request` plus
main-only `push` triggers.

## Git, pull requests, and reviews

- Begin from current `main` and inspect `git status` before changing files.
- Use a dedicated branch/worktree for concurrent PR work. Do not let agents
  edit overlapping files in the same worktree.
- Keep commits focused. Existing subjects generally use `feat:`, `fix:`,
  `test:`, `ci:`, `docs:`, `perf:`, `refactor:`, or `legal:` followed by a
  concise imperative summary.
- Never force-push shared branches, discard user changes, or rewrite another
  agent's commits without explicit authorization.
- CI validation workflows are read-only. Preserve `pull_request` coverage and
  main-only `push` triggers so same-repository PRs do not run duplicate suites.
  Do not add CI-generated autofix commits.
- Use least-privilege workflow permissions, avoid persisted checkout
  credentials when they are unnecessary, and pin new third-party automation to
  reviewed immutable revisions where practical.
- When reviewing a PR, attach line-specific findings to inline comments. Use a
  general review comment only for cross-cutting findings. Provide a minimal,
  directly applicable GitHub suggestion block when a localized fix is clear.
- If authorized to maintain the PR branch, fix blockers directly, rerun the
  applicable validation, and obtain an independent final review for risky
  changes.
- The repository owner authorizes coding agents to fix reviewed PR branches
  directly and squash merge without a separate per-merge approval once the
  final head is LGTM and every required check is green.
- Squash merge only when the final head is conflict-free, review-clean, and all
  required checks are green. After merging, verify the squash commit message
  and attribution, fast-forward local `main`, update issues, and remove clean
  temporary worktrees/branches.

## Multi-agent coordination

Use subagents for concrete, independent work such as repository inspection,
separate PR review, focused test analysis, or CPU/CUDA parity review.

- Give each agent a bounded scope, explicit acceptance criteria, and a clear
  no-edit/read-only instruction when appropriate.
- Prefer separate worktrees for agents that modify code.
- Do not assign multiple agents to edit the same files concurrently.
- The primary agent owns final integration, full diff review, tests, issue
  updates, and merge decisions. Never merge only because a subagent said LGTM.
- Preserve useful findings in the active issue so they survive agent shutdown,
  chat compaction, and future sessions.

## AI attribution for GitHub and Git

Every AI-authored GitHub comment, PR body, review, and Git commit must identify
the coding agent, the actual client version, the exact configured model, and
the configured reasoning effort. Read these values immediately before each
write; do not guess or reuse stale values.

For Codex, run:

```bash
codex --version
rg -n '^(model|model_reasoning_effort)\s*=' ~/.codex/config.toml
```

If any value cannot be verified, stop before posting or committing and ask the
user how to proceed. Non-Codex agents must obtain the equivalent actual values
from their own runtime and must not claim to be Codex.

For a submitted GitHub review, put the attribution on each inline comment when
the interface permits it. When several inline comments are submitted as one
review and repeating the block is impractical, include it at minimum in the
review summary.

Codex GitHub comments and PR bodies use:

```text
Coding agent: Codex
Codex version: <output of codex --version>
Model: <configured model>
Reasoning effort: <configured reasoning effort>
```

Codex commits keep the subject compliant with repository conventions and add:

```text
Coding-Agent: Codex
Codex-Version: <output of codex --version>
Model: <configured model>
Reasoning-Effort: <configured reasoning effort>
```

Other agents replace `Codex` and the `Codex version`/`Codex-Version` labels
with their actual agent name and client version. Do not erase or replace
attribution already present for work retained from another agent.

## Definition of done

Work is complete only when:

- the implementation satisfies the active issue's acceptance criteria, or
  small residual items are explicitly delegated under the issue-closure rule;
- focused and full validation applicable to the completed parent scope passed
  with exact results recorded, while delegated evidence remains required by
  its linked sub-issues;
- ABI, scientific, CUDA, packaging, and licensing invariants remain intact;
- documentation and useful code comments describe new public or non-obvious
  behavior;
- the PR has no unresolved review findings or conflicts and required CI is
  green;
- the narrowest active issue or linked sub-issues contain enough current
  information for a new agent to continue without reconstructing the work
  from chat history; and
- Epic #1 is current when, and only when, the release dashboard changed.
