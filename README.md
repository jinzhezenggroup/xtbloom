# xTBloom

<img src="docs/assets/xtbloom-logo.svg" alt="xTBloom logo" width="440">

xTBloom is a native GFN2-xTB inference library for workloads made of many
small and medium molecular systems. It combines a C++17 implementation, CPU
and CUDA backends, one stable C ABI, and Python interfaces built on that same
ABI.

Try the experimental, fully client-side browser demo at
<https://jinzhezeng.group/xtbloom/>. It compiles the CPU backend to wasm32
without requiring Memory64, targeting modern iOS Safari, Safari, Chrome, and
Firefox with WebAssembly and module Worker support. It also adds a small
Web-adapter L-BFGS optimizer and a first-class SMILES input. A pinned
OpenChemLib 9.21.0 release loads from jsDelivr in the background, adds explicit
hydrogens, generates a seeded 3D conformer, and performs an MMFF94
pre-relaxation before xTBloom calculation. Opening a URL such as
`https://jinzhezeng.group/xtbloom/?smiles=CCO` additionally runs the xTBloom
geometry optimizer automatically and writes the final angstrom coordinates
back to the page. Charged SMILES must URL-encode `+` as `%2B`. These optimizer
and SMILES facilities belong to the browser adapter, not the stable C ABI or
native library API. A wasm64 build remains in CI as an ABI and numerical parity
gate.

The current pre-release implements restricted and unrestricted GFN2-xTB
energies, analytic forces, and atomic charges. It is designed for reusable
contexts and ragged batches rather than wrapping a command-line calculation
once per molecule.

## Features

- **Native ragged batches.** Molecules share one call without padding every
  system to the largest atom or orbital count.
- **Measured high-throughput advantage.** In the public energy-plus-force
  benchmark at 62 atoms, xTBloom CPU is 7.6x-8.6x faster than xTB/tblite for
  128 systems and 8.9x-10.7x faster for 512 systems, with every dependent
  timed sample checked against the same output gate.
- **CPU and CUDA parity.** Both backends implement restricted and unrestricted
  GFN2-xTB. The CUDA ABI accepts caller-owned host, device, or mixed buffers.
- **Failure isolation.** SCC or eigensolver failure in one molecule publishes
  NaNs and diagnostics for that molecule without discarding successful peers.
- **Analytic derivatives.** Energies, QM forces, atomic charges, optional
  point-charge forces, and per-system molecular dipole moments are available
  through the public API.
- **QM/MM inputs inside SCC.** Explicit point charges, caller-supplied
  periodic charge-response operators, and a uniform external electric field
  participate in every SCC iteration.
- **Reusable execution state.** Contexts retain CPU workers, CUDA workspaces,
  fixed-topology plans, and strict compatible electronic warm starts.
- **One deployment boundary.** C, C++, Python, ASE, and dpdata all call the
  same versioned, caller-buffer C ABI.

GFN1-xTB and ROCm have reserved ABI values but are **not implemented**. The
native library also does not currently provide geometry optimization,
molecular dynamics, solvation, Hessians, or a lattice/PBC descriptor. The
browser demo's adapter-local optimizer does not change that public capability
boundary.

## Choosing an xTB implementation

The projects below serve different workflows. This is a capability comparison,
not a general performance ranking.

| Project | Best fit | Methods | Batch and accelerator model |
| --- | --- | --- | --- |
| **xTBloom** | Native high-throughput inference embedded in C/C++ or Python applications | GFN2-xTB | Ragged C-ABI batches; CPU and CUDA; caller-owned host/device buffers |
| [xTB](https://github.com/grimme-lab/xtb) | Broad end-user computational chemistry workflows | GFN0/1/2-xTB, GFN-FF, and more | Mature CLI and per-system library APIs; OpenMP and optional NVIDIA build paths |
| [tblite](https://github.com/tblite/tblite) | Lightweight, extensible single-point library | GFN1-xTB, GFN2-xTB, IPEA1-xTB | Fortran/C/Python per-structure APIs; CPU/OpenMP; molecular and periodic inputs |
| [dxtb](https://github.com/grimme-lab/dxtb) | Differentiable xTB in PyTorch and ML workflows | GFN1-xTB, GFN2-xTB | Batched PyTorch tensors on CPU/CUDA; autodiff forces and response properties |

### Where xTBloom is deliberately stronger

xTBloom makes several production-inference guarantees first-class rather than
leaving them to each calling application:

- A failed SCC or eigensolve is isolated to one ragged-batch member. Successful
  peers remain valid, while every requested floating-point slice for the failed
  member is replaced with quiet NaNs and accompanied by per-system diagnostics.
- Exactly degenerate finite-temperature occupations have a documented
  binary64 publication policy. In a reproduced three-hydrogen edge case,
  xTBloom returns a finite result where xTB, tblite, or dxtb fail on at least one
  integer-charge variant; the exact versions, inputs, and outcomes are
  [documented](docs/developer-guide/architecture.md#cross-engine-degenerate-occupation-evidence).
- Explicit point-charge screening and caller-supplied periodic charge response
  enter every SCC iteration. The same stable C ABI returns both QM and
  point-charge forces. This covers embedding inputs that remain unavailable or
  incomplete in the compared xTB/tblite C interfaces.
- The high-level Python API accepts electronic temperature in kelvin, CPU work
  partitioning is tested for bit-identical results, and strict `WARM` calls
  never silently fall back to a fresh solve.
- Archived, correctness-qualified CPU evidence shows strict `WARM` reducing
  SCC work from 17-18 iterations to 2 and running 1.09x-1.54x faster than a
  persistent tblite calculation for the measured 32-122 atom alkane corpus.

The [user-guide comparison](docs/user-guide/index.md#where-xtbloom-is-stronger)
links each claim to its upstream issue, local regression test, or archived raw
benchmark evidence.

### Cross-engine scaling benchmark

xTBloom's target workload is many distinct systems in one ragged public-API
call. The figure measures GFN2-xTB energy plus analytic-force latency for
distinct conformers of one alkane family; batch 1 provides latency context,
while batches 128 and 512 expose multi-system throughput.

![Cross-engine GFN2-xTB scaling benchmark](docs/assets/natoms_cross_engine.svg)

On an AMD EPYC 7K62 with the same 16-thread budget for every CPU engine:

- **128 systems at 62 atoms:** xTBloom CPU completes the call in 182 ms,
  versus 1555 ms for xTB and 1384 ms for tblite: 8.6x and 7.6x faster.
- **512 systems at 62 atoms:** xTBloom CPU takes 1.28 s, versus 11.47 s for
  xTB and 13.70 s for tblite: 8.9x and 10.7x faster.
- **xTBloom CUDA at batch 512:** 1.15 s at 62 atoms and 4.04 s at 122 atoms,
  1.11x and 1.37x faster than xTBloom CPU on the measured RTX 5090.

The CPU speedup is the public ragged-batch design in action: xTBloom solves the
whole batch across its worker pool, while the xTB/tblite adapters must loop
over per-structure public calls. Batch-1 results are retained in the figure as
latency context and are not used to claim a universal single-system win.

Each library retains its native public convergence controls: xTBloom uses
charge `1e-4` and energy `1e-6`; xTB and tblite use accuracy factor `1.0`;
dxtb uses `x_atol=1e-4`, `x_atol_max=1e-5`, `f_atol=1e-4`, and
`force_convergence=true`. The meanings are library-specific. `2e-3` is the
uniform energy/force output gate applied to every timed dependent sample, not
a tblite convergence default and not a replacement for xTBloom's stricter
scientific conformance.

Batch 1 and 512 use cold electronic state. Batch 128 performs one untimed cold
seed and times persistent continuation for xTBloom, xTB, and tblite; dxtb
resets every timed call. xTBloom CUDA uses host descriptors, whereas dxtb CUDA
retains device tensors, so no direct cross-library CUDA speedup is claimed.

The evidence is limited to this alkane-conformer corpus, energy plus forces,
three samples per coordinate, and the stated hardware. Raw samples, exact
commands, binary hashes, failures, and limitations are archived in the
[`issue-13 evidence bundle`](benchmarks/evidence/issue-13/2026-08-09-node3-pr231/README.md).
See the [benchmark harness](benchmarks/README.md) for the protocol.

Choose xTB for its broad CLI workflows, optimizers, dynamics, solvation, and
method coverage. Choose tblite for a mature reusable single-point library with
periodic structures and customizable components. Choose dxtb when PyTorch
autodiff and differentiable response properties are central. Choose xTBloom
when the application needs a native ragged batch, a stable deployment ABI,
direct CUDA buffers, or peer-local failure handling.

Published benchmark claims are deliberately workload-specific. Reproducible
protocols and raw results live under [`benchmarks/evidence`](benchmarks/evidence/).

## Python quickstart

xTBloom is being prepared for publication on PyPI. Once a release is published,
install the CPU runtime with:

```console
python -m pip install xtbloom
```

Optional integrations and CUDA 12 host libraries are extras:

```console
python -m pip install "xtbloom[ase,dpdata]"
python -m pip install "xtbloom[cuda12]"
```

Until the first PyPI release, install a source checkout as a non-editable
package. `XTBLOOM_ENABLE_CUDA=OFF` makes the intended backend explicit:

```console
XTBLOOM_ENABLE_CUDA=OFF python -m pip install .
```

Positions are in bohr. Energies and forces are returned in Hartree and
Hartree/bohr; the high-level Python `electronic_temperature` argument is the
temperature in kelvin.

```python
import numpy as np
from xtbloom import Calculator

numbers = np.array([8, 1, 1])
positions = np.array(
    [
        [0.0000000000, 0.0000000000, -0.7357858611],
        [1.4418315287, 0.0000000000, 0.3678929305],
        [-1.4418315287, 0.0000000000, 0.3678929305],
    ]
)

with Calculator("GFN2-xTB", numbers, positions, backend="auto") as calc:
    result = calc.singlepoint()

print(result["energy"])
print(result["forces"])
print(result["charges"])
```

`BatchCalculator` submits multiple `Structure` objects in one native call.
The high-level Python API uses host NumPy arrays even when the selected backend
is CUDA; direct CUDA-device descriptors are available through the low-level C
ABI.

See the [Python user guide](docs/user-guide/python.md) for batching, spin,
point charges, ASE, and dpdata, or the concise
[PyPI package page](python/README.md).

## C and C++ quickstart

Native consumers need CMake 3.24 or newer, a C++17 compiler to build xTBloom,
and one dlopen-able monolithic LP64 LAPACKE+CBLAS runtime for CPU inference.
An explicitly selected MKL runtime also requires its matching LP64,
sequential, and core component libraries in the same provider directory.
Shared installs place xTBloom's private MKL shim beside `libxtbloom`. Static
consumers that use MKL must stage that installed shim beside the final
executable; CMake consumers can copy
`$<TARGET_FILE:xtbloom::mkl_lp64_shim>` when that optional imported target is
present. Without the sibling artifact, CPU inference fails with
`XTBLOOM_STATUS_BACKEND_UNAVAILABLE` instead of using the host's `libmkl_rt`.
The public consumer API itself is C11-compatible and is wrapped in `extern "C"`
for C++.

```console
cmake -S . -B build/release -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel
cmake --install build/release --prefix "$PWD/build/install"
```

If auto-discovery cannot find the CPU numerical runtime, configure its absolute
path with `-DXTBLOOM_CPU_LINALG_LIBRARY=/path/to/libopenblas.so` or a compatible
LP64 `libmkl_rt`.

Installed CMake consumers use the exported target:

```cmake
find_package(xtbloom CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE xtbloom::xtbloom)
```

Every extensible descriptor must be initialized before its fields are set.
The complete request and all caller-owned output buffers are then submitted in
one synchronous call:

```c
#include <xtbloom/xtbloom.h>

xtbloom_context_options_t context_options;
xtbloom_batch_t batch;
xtbloom_compute_options_t compute_options;
xtbloom_batch_result_t result;

xtbloom_context_options_init(&context_options, sizeof(context_options));
xtbloom_batch_init(&batch, sizeof(batch));
xtbloom_compute_options_init(&compute_options, sizeof(compute_options));
xtbloom_batch_result_init(&result, sizeof(result));

/* Populate batch and result with caller-owned buffers. */
compute_options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES;

xtbloom_context_t *context = NULL;
xtbloom_context_create(&context_options, &context);
xtbloom_status_t status = xtbloom_compute(context, &batch, &compute_options, &result);
xtbloom_context_destroy(context);
```

The [C API guide](docs/user-guide/c-api.md) contains a complete runnable
single-molecule example plus descriptor, units, CUDA-memory, and failure
semantics. The installed header
[`include/xtbloom/xtbloom.h`](include/xtbloom/xtbloom.h) is the normative API.

## Documentation

- [Documentation index](docs/index.md)
- [User guide](docs/user-guide/index.md)
- [Theory guide](docs/theory/index.md)
- [Developer guide](docs/developer-guide/index.md)
- [Python package documentation](python/README.md)

## Acknowledgements and provenance

xTBloom exists because the xTB and tblite communities made both the scientific
method and high-quality reference implementations available. During xTBloom's
design and implementation, coding agents studied the xTB and tblite source
code to understand equations, numerical conventions, edge cases, and public
interface behavior. xTB also serves as an executable numerical oracle and the
reference for explicit point-charge coupling. tblite supplies pinned GFN2
parameter material and strongly influenced the familiar shape of the Python
interface. We thank their authors and contributors.

This acknowledgement is not a substitute for legal provenance. Redistributed
or derived parameter data, oracle outputs, source material, revisions, hashes,
and license terms are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the linked manifests.

## AI authorship

xTBloom is an AI-first software project. The core library architecture,
scientific implementation, CUDA backend, bindings, tests, and documentation
were designed and written primarily by AI coding agents rather than as a
conventional manually authored implementation. Humans provide project goals,
scientific and release decisions, review, infrastructure, and legal ownership.
Git commits, pull requests, and issue checkpoints record the exact coding
agent, client version, model, and reasoning effort used for agent-authored work.

This development model does not relax the correctness standard: conformance
uses pinned independent xTB/tblite evidence, CPU/CUDA parity, analytic-force
finite differences, ABI tests, sanitizers, install consumers, and package
inspection.

## License

xTBloom is licensed under `GPL-3.0-or-later`, with the narrowly scoped
[CUDA and Intel MKL additional permission](CUDA_MKL_LINKING_EXCEPTION).
Upstream material remains under the separate terms in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
