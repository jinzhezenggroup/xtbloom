# Developer guide

gpuxtb development is governed by scientific correctness, ABI stability,
failure isolation, CPU/CUDA parity, reproducible evidence, and legal
provenance. A locally green subset is not sufficient when its configuration
omits the backend or public behavior being changed.

Read these pages before modifying the corresponding surface:

- [Architecture](architecture.md): data model, compute and publication
  semantics, runtime layering, warm state, CPU scheduling, and performance
  boundaries.
- [Validation](validation.md): selecting tests, conformance, CUDA evidence,
  and honest pass/skip reporting.
- [Packaging and licensing](packaging.md): CMake installs, PyPI metadata,
  wheels, sdists, dynamic providers, dependency provenance, and notices.
- [Theory](../theory/index.md): scientific meaning that CPU and CUDA must share.

The repository-wide [`AGENTS.md`](../../AGENTS.md) is authoritative for task
startup, issue memory, test commands, GitHub handling, review, and AI
attribution. The public C header is the authoritative ABI specification.

## Repository map

| Path | Responsibility |
| --- | --- |
| `include/gpuxtb/gpuxtb.h` | Only public C ABI |
| `src/api.cpp`, `src/runtime/` | Initialization, validation, dispatch, and publication |
| `src/model/gfn2/` | Readable CPU GFN2 physics and SCC |
| `src/backends/common/` | Backend-neutral plan and workspace schemas |
| `src/backends/cuda/` | CUDA setup, SCC, publication, and force chain |
| `python/gpuxtb/` | ctypes mirror and high-level Python integrations |
| `data/parameters/` | Generated parameters and provenance |
| `data/conformance/` | Hash-pinned independent scientific inputs/goldens |
| `tools/` | Generators, conformance, oracle, and licensing tools |
| `benchmarks/` | Persistent public-API adapters and evidence |

## Change discipline

Keep changes scoped to an active GitHub leaf issue. Translate its acceptance
criteria into evidence before implementation, and update the issue at material
checkpoints. Do not close scientific, CUDA, sanitizer, profiler, packaging, or
legal work merely because code landed.

Preserve short-structure ABI compatibility, caller-owned buffers,
transactional publication, peer-local numerical failures, and strict warm
epochs. New external bytes require provenance and license review before they
enter source, build, install, sdist, or wheel artifacts.
