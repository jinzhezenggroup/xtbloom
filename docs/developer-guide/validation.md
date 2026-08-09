# Validation and scientific evidence

Validation follows the changed surface and issue acceptance criteria. Record
the exact command, configuration, test count, backend, provider, and result.
An absent or skipped test is not a pass.

## One-command Nox workflow

The recommended CPU validation entry point for developers and AI agents is:

```console
UV_DEFAULT_INDEX=https://pypi.org/simple \
  uv run --isolated --locked --only-group nox nox -s agent
```

The locked `nox` dependency group fixes the runner and its transitive
dependencies without publishing Nox as a project extra or runtime dependency.
uv starts that runner in a separate isolated environment; Nox creates no
additional virtual environment and only orchestrates the explicit commands
documented below. Python validation is synchronized by uv from the canonical
lock, installs the project wheel non-editably, and runs Python tools through
`uv run --no-sync`. The `agent` session runs fast checks, a shared Release CPU
build with the reviewed LP64 runtime and complete CTest registration, the
non-editable Python wheel tests, benchmark-adapter tests, and canonical data,
conformance, oracle, and licensing checks.

Focused sessions are available with the same prefix:

- `fast`: repository formatting, lock, and whitespace checks;
- `cpu`: shared CPU public inference and full registered CTest;
- `python`: non-editable CPU wheel, Python tests, and benchmark adapters;
- `canonical`: generated-data, conformance, oracle, and licensing checks;
- `package`: CPU tests, shared/static install consumers, and CPU sdist/wheel gates;
- `cuda`: explicit CUDA `ON` validation on a real visible NVIDIA GPU; and
- `full`: `agent` plus the packaging gates.

The CUDA session discovers the visible GPUs' compute capabilities unless
`XTBLOOM_CUDA_ARCHITECTURES` is set, and requires `nvcc` on `PATH` or an exact
`CUDACXX`. Missing compilers, hardware, provider-backed test registration, or
exit-code-77 tests fail the session instead of being reported as a pass. Do
not run sessions in parallel: Python synchronization and build directories are
intentionally shared within one worktree.

The underlying commands remain authoritative for diagnosis, unusual compiler
matrices, sanitizers, profilers, wheels, and release evidence.

CI uses the Nox `fast` session to exercise the wrapper itself. The compiler,
coverage, public-package, and multi-architecture wheel jobs intentionally keep
their more specialized commands; replacing those matrices with the local Nox
defaults would reduce validation coverage.

## Common repository gates

Every change runs:

```console
uvx prek@0.3.1 run --show-diff-on-failure --color=always --all-files
uv lock --check
git diff --check
```

Documentation changes additionally need a relative-link/path audit. Changes to
PyPI metadata or source-distribution documentation also need an actual sdist
build and inspection; rendering the source Markdown does not prove what PyPI
or an archive will receive.

## CPU native build

```console
cmake -S . -B build/cpu -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu --parallel
ctest --test-dir build/cpu -N
ctest --test-dir build/cpu --output-on-failure
```

Inspect registration. Without one verified LP64 LAPACKE+CBLAS runtime,
eigensolver, public CPU inference, and conformance tests may not be registered.
A smaller core pass must be reported as partial validation.

## Python

Python bindings must exercise the real bundled native package rather than an
editable source import:

```console
XTBLOOM_ENABLE_CUDA=OFF uv sync --no-editable --extra test
uv run --no-sync pytest python/tests -q
```

Run the focused file first, then the complete suite. Missing native inference
is a failure, not a supported skip.

## Scientific changes

GFN2 physics, SCC, occupations, energies, forces, charges, QM/MM, parameter,
or oracle changes require independent evidence in addition to internal
self-consistency. Select from:

- term-level reference tests;
- central finite differences;
- invariance and conservation checks;
- public-API CPU conformance;
- real-GPU CPU/CUDA parity across host, device, and mixed descriptors;
- pinned xTB/tblite goldens; and
- canonical SCC traces and replay.

Never regenerate a golden merely because implementation output changed.

## CUDA changes

Configure with `XTBLOOM_ENABLE_CUDA=ON`, select the actual compiler and
architecture, and run affected tests on a real NVIDIA GPU. `AUTO`, a compile
success, loader-stub wheel CI, or an exit-code-77 skip is not CUDA runtime
evidence.

Depending on the issue, exercise host/device/mixed descriptors, caller streams,
Graph behavior, fixed-topology cache transitions, FRESH/WARM epochs, repeated
calls, changed geometry/topology, failure publication, and Compute Sanitizer
`memcheck`, `racecheck`, `initcheck`, and `synccheck`.

## Result vocabulary

Use these states in issue and PR ledgers:

- `PASS`: the intended implementation ran in the required environment.
- `FAIL`: the test ran and failed.
- `SKIP`: registered but did not exercise the target.
- `NOT REGISTERED`: the configuration omitted the test.
- `NOT RUN`: the test exists but was not executed.
- `UNAVAILABLE`: hardware, runtime, scheduler, oracle, or package builder is
  absent.

Only `PASS` satisfies an acceptance criterion.
