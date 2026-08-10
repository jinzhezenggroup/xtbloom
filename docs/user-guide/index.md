# User guide

xTBloom is designed for applications that evaluate GFN2-xTB over many small
and medium molecular systems. It provides reusable CPU and CUDA contexts,
native ragged batches, analytic forces and charges, and peer-local failure
handling through one stable C ABI.

## Installation

### Python

xTBloom is not yet published on PyPI. Install a source checkout as a
non-editable package:

```console
XTBLOOM_ENABLE_CUDA=OFF python -m pip install .
```

Set `XTBLOOM_ENABLE_CUDA=ON` for a CUDA build with an available toolkit.
Python 3.10 or newer is supported. See the
[Python guide](python.md) for package extras, native-library discovery, and the
complete API.

### C and C++

Native builds require CMake 3.24 or newer and a C++17 compiler:

```console
cmake -S . -B build/release -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel
cmake --install build/release --prefix "$PWD/build/install"
```

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
- [Python API](python.md)
- [C and C++ API](c-api.md)
- [QM/MM usage](qmmm.md)
- [Performance evidence](performance.md)

## Scope and limitations

Only GFN2-xTB is implemented. GFN1-xTB and ROCm have reserved ABI values but
return unsupported or not-implemented statuses.

xTBloom has no lattice input. Its periodic charge-response API consumes fields
computed by another electrostatics program; it does not make the QM
calculation periodic by itself. Native geometry optimization, molecular
dynamics, solvation, vibrational analysis, and Hessians are not implemented.
The browser and dpdata optimizers are higher-level adapters built on repeated
xTBloom single-point calls.
