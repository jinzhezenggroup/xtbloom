---
name: gpuxtb-select-validation
description: Select and execute gpuxtb validation from the changed files and acceptance criteria, including CPU, CUDA, Python, ABI, packaging, generated-data, licensing, and benchmark gates. Use for any material gpuxtb code, test, build, packaging, or data change, especially before handoff, review, or merge when a green but incomplete test configuration would be dangerous.
---

# Select gpuxtb Validation

Build a validation ledger that distinguishes a real pass from an absent, skipped, or unavailable test. Read the root `AGENTS.md` first; its commands and invariants are authoritative.

## Inventory the Change

1. Read the active issue, its blockers, and acceptance criteria.
2. Inspect committed and uncommitted paths with `git diff --name-status <base>...HEAD`, `git diff --name-status`, and `git status --short`.
3. Map each path to every affected surface below. Use the union of the required gates; do not choose only the cheapest row.
4. Add any issue-specific sanitizer, profiler, architecture, package, or oracle gate before running tests.

| Changed surface | Required validation |
| --- | --- |
| Any repository file | Fast repository checks and `git diff --check` |
| C/C++ core, CMake, native tests | Fresh CPU Release configure, build, and full registered CTest |
| Eigensolver or public CPU inference | Shared CPU build with an absolute verified LP64 LAPACKE+CBLAS runtime; public inference and conformance tests must be registered |
| Public header, validation, symbols, install API | Invoke `$gpuxtb-evolve-c-abi`; add shared/static installs and external consumers |
| CUDA source or CUDA runtime behavior | Invoke `$gpuxtb-validate-cuda-change`; use explicit CUDA `ON` and a real NVIDIA GPU |
| Python package or bindings | Non-editable `uv` install of the native wheel plus the focused and full Python suites |
| `pyproject.toml` or dependency resolution | Regenerate `uv.lock` only against PyPI, then require `uv lock --check` |
| Parameters, conformance, oracle, or copied data | Run canonical checkers and relevant unit suites; never hand-edit generated or hash-pinned outputs |
| License, dependency, install, sdist, or wheel payload | Invoke `$gpuxtb-audit-dependency`; test every changed artifact boundary |
| Benchmark harness or performance claim | Run benchmark self-tests and invoke `$gpuxtb-record-performance` for evidence |

## Run the Common Gates

Use separate build directories so an old cache cannot change which tests exist.

```bash
uvx prek@0.3.1 run --show-diff-on-failure --color=always --all-files
uv lock --check
git diff --check

cmake -S . -B build/cpu -G Ninja \
  -DGPUXTB_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
```

For a public CPU gate, configure a separate shared build with `GPUXTB_CPU_LINALG_LIBRARY` set to an absolute compatible LP64 runtime. Do not infer full inference coverage from the smaller core-only build.

For Python changes, build and install the real local package before testing:

```bash
GPUXTB_ENABLE_CUDA=OFF uv sync --no-editable --extra test
uv run --no-sync pytest python/tests -q
```

## Audit Test Registration

After every CMake configure, run `ctest --test-dir <build> -N` and inspect the configure output. Confirm by name that the tests needed for the acceptance criteria exist.

For a public shared CPU configuration, the registration list should include the relevant names from this set: `gpuxtb.c_api`, `gpuxtb.batch_validation`, `gpuxtb.runtime`, `gpuxtb.cpu.public_inference`, `gpuxtb.gfn2.eigensolver`, `gpuxtb.conformance.public_cpu`, and `gpuxtb.conformance.invariants_cpu`. Shared UNIX builds add `gpuxtb.abi_symbols` and CUDA dependency tests. The exact list is configuration-dependent; an absent name is evidence to explain, not permission to claim coverage.

Treat these states distinctly:

- `PASS`: the intended implementation ran in the required environment and passed.
- `FAIL`: the test ran and failed.
- `SKIP`: the test was registered but did not exercise the backend, including exit code 77.
- `NOT REGISTERED`: configuration omitted the test, commonly because no LP64 runtime was found or the build was static/CPU-only.
- `NOT RUN`: the test exists but was not executed.
- `UNAVAILABLE`: required hardware, runtime, scheduler, oracle, or package builder is absent.

Never convert the last four states to `PASS`. In particular:

- A CPU build without a compatible LP64 provider does not prove eigensolver, public inference, or public conformance.
- `GPUXTB_ENABLE_CUDA=AUTO` does not prove CUDA was compiled.
- CUDA compile-only or hosted wheel loader-stub coverage does not prove real-GPU execution.
- A static build does not prove shared-library symbols or dynamic dependencies.

## Report the Ledger

Record the exact command, build directory, configuration facts, pass/fail/skip counts, and missing gates. Include compiler, LP64 provider, CUDA toolkit/GPU when relevant, and the tested commit. State the smallest honest conclusion, such as `CPU core passed; public inference not registered`, instead of `all tests passed`.
