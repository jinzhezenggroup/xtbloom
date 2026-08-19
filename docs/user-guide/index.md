# User guide

xTBloom is designed for applications that evaluate GFN1/GFN2-xTB over many small
and medium molecular systems. It provides reusable CPU and CUDA contexts,
native ragged batches, analytic forces and charges, and peer-local failure
handling through one stable C ABI.

## Installation

### Prerequisites

The table separates enforced or supported requirements from versions that are
only the current CI reference. A reference version documents what this revision
is tested with; it is not an invented minimum for older toolchains that the
project does not test.

| Area | Requirement and support boundary |
| --- | --- |
| Python package | `Requires-Python` is 3.10 or newer with no upper bound; classifiers currently enumerate Python 3.10-3.14. Repository test configurations require Python 3.11 or newer. |
| Native languages | A C11 compiler and a C++17 compiler. CUDA builds also require CUDA C++17. |
| GCC and Clang | No numeric minimum is currently promised beyond complete C11/C++17 support. CI follows the Ubuntu 24.04 runner defaults; the 2026-08-10 run for this revision reported GCC 13.3 and Clang 18.1. Older versions may build but are outside the promised full-validation matrix. |
| CMake and generator | CMake 3.24 or newer. The documented commands use Ninja, so Ninja must be installed for those commands; no numeric Ninja minimum is enforced. |
| `uv` and Nox | `uv` is needed for the locked Python/source-checkout workflows. Use uv 0.10.7 for lockfile maintenance and CI parity. Nox 2026.7.11 is already pinned in the lock and does not need a separate global install. Neither tool is a runtime dependency of `libxtbloom`. |
| Source version metadata | Native CMake checkouts and non-exact-tag Python checkouts need complete tag history plus a reachable strict `vMAJOR.MINOR.PATCH` tag. An exact-tag Python build may use that tag from a shallow checkout. An sdist or expanded Git archive carries frozen version metadata instead. |
| Current distribution platforms | PyPI publishes an sdist plus wheels for Linux x86_64/aarch64, macOS x86_64/arm64, Windows AMD64/ARM64, and Pyodide wasm32. Linux wheels contain CPU and CUDA backends; macOS and Windows wheels are CPU-only. Pyodide uses its separately qualified CPU/WebAssembly path. |
| CPU linear algebra | CPU inference needs one `dlopen`-able monolithic shared LP64 runtime exporting both LAPACKE and CBLAS plus provider-local thread control. Split `liblapack` + `libblas`, ILP64 providers, and static archives do not satisfy this contract. Compatible OpenBLAS or MKL installations are common source-build candidates; CMake validates the exact file. Published Linux, macOS, and Windows wheels bundle their reviewed private OpenBLAS provider. |
| CUDA source build | The CUDA backend currently supports 64-bit Linux ELF on x86_64 and aarch64. Use `nvcc` from CUDA Toolkit 12.9 for the current qualified build baseline; the wheel CI reference is NVCC 12.9.86 with GNU 14.2.1 as host compiler. `nvcc` must be on `PATH` or selected exactly with `CUDACXX`. Configuration also needs a Python 3 interpreter for CUDA shim generation, and source builds should set the actual GPU architectures. |
| CUDA version qualification | CUDA 12.9 is the only current CI-compiled cohort. CMake contains provider-SONAME metadata for CUDA major 12 and 13, but that metadata is not by itself a full CUDA 13 build or real-GPU support claim. Toolkits outside 12.9 are unqualified and should not be treated as supported without project evidence. |
| CUDA runtime | Calling the CUDA backend requires a real NVIDIA GPU, a compatible NVIDIA driver, and cudart, cuBLAS, and cuSOLVER host libraries matching the build CUDA major. They are loaded at runtime and are not bundled in xTBloom wheels. The `cuda12` extra installs the supported CUDA 12 host packages; it cannot install the system driver. Hosted wheel CI compiles CUDA but is not real-GPU evidence. |

`XTBLOOM_ENABLE_CUDA` defaults to `AUTO`: configuration enables CUDA when a
CUDA compiler is available and otherwise produces a CPU-only build. Use `ON`
only when a missing CUDA compiler must make configuration fail.

For native CMake builds, set `CMAKE_CUDA_ARCHITECTURES` to the target GPU
capabilities. For Python source builds, set `XTBLOOM_CUDA_ARCHITECTURES`; its
default is `all-major`, which compiles one representative architecture per
computed-major family supported by the build CUDA toolkit (a smaller artifact
than `all`). `backend="auto"` can fall back to CPU when CUDA
is unavailable; `backend="cuda"` requires the CUDA runtime prerequisites above.

### Python

Install the published package from PyPI. The second command adds the supported
CUDA 12 user-space libraries for CUDA-capable Linux wheels:

```console
pip install xtbloom
pip install "xtbloom[cuda12]"
```

The `cuda12` extra cannot install an NVIDIA driver or GPU. Pass
`backend="cuda"` to require GPU execution and receive an error when CUDA is
unavailable; use `backend="auto"` only when CPU fallback is intended. Optional
integrations are available as `xtbloom[ase]`, `xtbloom[dpdata]`, or combined
extras such as `xtbloom[cuda12,ase,dpdata]`. Python 3.10 or newer is supported.
See the [Python guide](python.md) for native-library discovery and the complete
API.

For development or an unsupported target, build from a complete source checkout
as a locked, non-editable uv project environment:

```console
uv sync --locked --no-editable --no-default-groups \
  --reinstall-package xtbloom
```

Source builds use the default `AUTO` backend selection and auto-discover a
compatible system LP64 LAPACKE+CBLAS runtime. When discovery fails, set
`CMAKE_ARGS="-DXTBLOOM_CPU_LINALG_LIBRARY=/absolute/path/to/provider.so"` on the
sync command. Use `uv run --no-sync` for commands in that environment.

### C and C++

Native builds require the toolchain described above. The install-only example
keeps the default `AUTO` CUDA selection and disables repository tests, so it
does not require the test-only Python 3.11 interpreter:

```console
cmake -S . -B build/release -G Ninja \
  -DXTBLOOM_BUILD_TESTS=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel
cmake --install build/release --prefix "$PWD/build/install"
```

For a CUDA native build, the default `AUTO` selection is normally sufficient.
The command below uses `ON` deliberately so a missing compiler cannot silently
produce a CPU-only build, and selects the real target architecture. CUDA
configuration also requires a Python 3 interpreter for shim generation:

```console
cmake -S . -B build/cuda -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DXTBLOOM_BUILD_TESTS=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda --parallel
cmake --install build/cuda --prefix "$PWD/build/install-cuda"
```

Replace `89` with the compute capability required by the deployment. At run
time, request `XTBLOOM_BACKEND_CUDA` or `backend="cuda"`; `AUTO` may select CPU
when CUDA is unavailable.

On supported x86-64 builds, one library contains both baseline and AVX2/FMA
Mulliken kernels. A CPU context detects the host once and freezes the selected
kernel table, so dispatch does not occur inside numerical loops. The
experimental diagnostic override `XTBLOOM_CPU_ISA=auto|baseline|avx2` can force
a path before creating a CPU context; an unavailable forced `avx2` request
fails cleanly, and CUDA contexts ignore this CPU-only variable. Source builds
can set `-DXTBLOOM_ENABLE_AVX2_DISPATCH=OFF` to produce a baseline-only library.
Reproducible mode promises exact replay only while this context-selected ISA
and the rest of the documented execution environment remain unchanged.

CPU inference requires one dlopen-able monolithic LP64 LAPACKE+CBLAS runtime.
If auto-discovery cannot find one, set
`-DXTBLOOM_CPU_LINALG_LIBRARY=/absolute/path/to/provider`. Installed CMake
consumers link `xtbloom::xtbloom`. The [C API guide](c-api.md) contains a
complete runnable example.

## What xTBloom is good at

- Submitting differently sized molecules in one native ragged-batch call.
- Reusing CPU workers, CUDA workspaces, topology plans, and compatible SCC
  state across calls.
- Keeping a failed SCC or eigensolve local to one molecule while preserving
  successful peers.
- Returning energies, analytic QM forces, atomic charges, optional
  point-charge forces, and molecular dipoles.
- Passing explicit point charges and caller-supplied charge-response operators
  through every SCC iteration.
- Sharing one public ABI across C, C++, Python, ASE, and dpdata.

The [performance summary](performance.md) presents the current
correctness-qualified batch evidence without turning one workload into a
general performance ranking.

## Units and result meaning

The native C ABI uses IEEE binary64 atomic units:

| Quantity | Unit |
| --- | --- |
| Positions | bohr |
| Energy | Hartree |
| Forces | Hartree/bohr |
| Atomic and molecular charges | elementary charge |
| Point-charge screening parameter | Hartree |
| Electronic temperature | `k_B T` in Hartree |

The high-level Python interface accepts `electronic_temperature` in kelvin and
performs the conversion. ASE and dpdata use their conventional eV and angstrom
units at their integration boundaries.

At finite electronic temperature, the reported variational energy is the
electronic Helmholtz free energy. Forces are the negative coordinate derivative
of that reported energy.

When caller-supplied `atomic_potential_shifts` or a
`charge_response_matrix` is present, xTBloom holds those fields fixed during
differentiation. Returned forces therefore exclude their coordinate
derivatives, and the caller must add them when the external model requires
them. See the [QM/MM guide](qmmm.md).

## Backends and memory

`backend="auto"` prefers CUDA when it is available and otherwise uses CPU.
Use `"cpu"` or `"cuda"` to require one backend.

The high-level `Calculator` and `BatchCalculator` interfaces use host NumPy
arrays. `ArrayBatch` and the low-level C ABI can consume CUDA-device arrays and
write into caller-owned device outputs. CUDA compute is synchronous; active
stream capture is rejected.

## Failure behavior

xTBloom separates request failures from per-system numerical failures:

- Invalid descriptors, unsupported settings, allocation failures, and other
  call-level errors return a failing status. Before caller-output commit,
  outputs and result flags remain unchanged.
- SCC nonconvergence or eigensolver failure for one molecule is a data-level
  result. Diagnostics identify the failed molecule, its requested floating
  slices are filled with quiet NaNs, and successful peers remain available.

Python `BatchCalculator.compute()` preserves this behavior by default. Call
`BatchResult.raise_for_status()` or pass `raise_on_failure=True` for
exception-oriented control flow.

## Guides

- [Browser demo](browser-demo.md)
- [Skills for AI agents](agent-skills.md)
- [Python API](python.md)
- [Vibrational analysis](vibrations.md)
- [ASE molecular dynamics](ase-md.md)
- [C and C++ API](c-api.md)
- [QM/MM usage](qmmm.md)
- [Performance evidence](performance.md)

## Scope and limitations

GFN1-xTB and GFN2-xTB are implemented on CPU and CUDA. The high-level Python
calculators, ASE, and dpdata expose both models, while Array API/DLPack and
PyTorch autograd remain GFN2-only. The single-threaded CPU/WebAssembly
browser demo exposes both GFN1 and GFN2 and defaults to GFN2. ROCm remains
reserved.

xTBloom's ABI-v4 batch descriptor reserves validated native 3D cell and
periodic-axis input, but native periodic GFN1/GFN2 execution and the Python
periodic adapters are not implemented yet. Valid `XYZ` requests therefore
return `NOT_IMPLEMENTED` before output publication. The separate periodic
charge-response API consumes fields computed by another electrostatics
program; it does not make the QM calculation periodic by itself. Native
drivers for geometry optimization and molecular dynamics, solvation,
and native/analytic Hessians are not implemented. Python
`Calculator.hessian()` and `BatchCalculator.hessian()` provide dense numerical
QM-coordinate Hessians from batched analytic-force differences. ASE-driven
molecular dynamics, the Hessian and Python [vibrational
analysis](vibrations.md), and the browser/dpdata optimizers are higher-level
adapters built on repeated xTBloom calculations.
