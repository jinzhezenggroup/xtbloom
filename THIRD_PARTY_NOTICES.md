# Third-party notices and provenance

gpuxtb is distributed under `GPL-3.0-or-later`; see `LICENSE`. The additional
permission in `CUDA_MKL_LINKING_EXCEPTION` applies only to gpuxtb material
whose copyright holder authorizes it. The following upstream material is
retained under its own terms and is not relicensed by that permission.
The manifests named below pin the source revisions and content digests used
to produce the redistributed data.

## array-api-compat

Repository: <https://github.com/data-apis/array-api-compat>

License: `MIT` (`LICENSES/array-api-compat-MIT.txt`; Copyright (c) 2022 Consortium for Python Data API Standards).

The gpuxtb Python package depends on `array-api-compat` (a backend-neutral
shim, not an array backend) to probe dense eager arrays from NumPy, CuPy,
JAX, and PyTorch without importing those packages. Reviewed release 1.15.0,
commit `076218e4f5aa18578418c7d04fad9ab581a16bb8`, Python `>=3.10`, with no
`Requires-Dist` entries. The `py3-none-any` wheel is
`array_api_compat-1.15.0-py3-none-any.whl` (SHA-256
`7b1b9c53269061403fd5f45a8de349f16e7887653328bfa0c5f2d45299ff0a8e`); the
sdist is SHA-256
`53c5f922491bf15f62847afafc4e39eedfae57d218988fefb8cce39c2a9b3dea`. It is a
runtime Python dependency only and is not bundled in gpuxtb source archives,
native installs, or wheels; the canonical resolution is recorded in
`pyproject.toml` and `uv.lock`.

## DLPack

Specification: <https://github.com/dmlc/dlpack>

License: `Apache-2.0` (`LICENSES/Apache-2.0.txt`).

No DLPack header or source is bundled, linked, or vendored. gpuxtb
independently reimplements the *byte layout* of the DLPack 1.0 managed-tensor
data structures (`DLDataType`, `DLDevice`, `DLTensor`, `DLManagedTensor`, and
`DLManagedTensorVersioned`) as literal C/C++ struct mirrors in
`src/runtime/dlpack_layout.hpp` and as matching ctypes mirrors in
`python/gpuxtb/_dlpack.py`, with byte-exact static assertions. These layouts
are the public ABI of the DLPack specification and are used only to produce
and consume capsules at the Python boundary; they are not a copy of the
upstream codebase's implementation. The upstream project's Apache-2.0 license
text is retained in `LICENSES/Apache-2.0.txt` for reference.

## tblite

Repository: <https://github.com/tblite/tblite>

License: `LGPL-3.0-or-later` (`LICENSES/LGPL-3.0-or-later.txt`; the
incorporated GPLv3 terms are in the project `LICENSE`).

Redistributed or derived material:

- `data/parameters/gfn2.toml`, `gfn2.json`, and `gfn2.hpp` are deterministic
  representations of the GFN2 parameter export from revision
  `fa8a4416e8fe093d0075bc10ac875494c2a449a9`. Exact source paths and hashes
  are recorded in `data/parameters/manifest.json`.
- The Stewart STO-nG tables in `data/parameters/tblite_sto.hpp` come from
  `src/tblite/basis/slater.f90` at that revision.
- The element spin constants in `data/parameters/tblite_spin.hpp` come from
  `src/tblite/data/spin.f90` at that revision. Their digest is recorded in
  `data/parameters/spin_manifest.json`.
- The SCC observer development tool in `tools/oracle/tblite_scc_trace/`
  carries its own pinned patch metadata and bundled GPL/LGPL texts.

## dftd4 and mctc-lib

dftd4 repository: <https://github.com/dftd4/dftd4>

dftd4 license: `LGPL-3.0-or-later`.

mctc-lib repository: <https://github.com/grimme-lab/mctc-lib>

mctc-lib license: `Apache-2.0` (`LICENSES/Apache-2.0.txt`).

The packed GFN2-D4 reference data in `data/parameters/d4.hpp` is derived
from dftd4 revision `6e1f59c3f39d919a2dbef0601d2576727c8b30e8`.
The D4 electronegativity-weighted coordination data and implementation
conventions also use mctc-lib revision
`e9de066d89f250d1cfb6de3a33f0c27c0e2f855d`. The ordinary GFN2
coordination implementation retains mctc-lib's covalent-radii data and
double-exponential convention, while the H0 implementation retains its
Mantina atomic-radii table. Exact dftd4 source blobs are recorded in
`data/parameters/d4_manifest.json`; exact mctc-lib source paths, blobs, and
hashes are recorded in `data/parameters/mctc_manifest.json`. The original
focused notice and upstream license copies remain in
`data/parameters/d4.NOTICE` and `data/parameters/licenses/`.

## xTB numerical oracle

Repository: <https://github.com/grimme-lab/xtb>

License at the pinned revision: `LGPL-3.0-or-later`.

The conformance corpus uses xTB 6.7.1 revision
`edcfbbe39d411edc225e27315fbda3a204ddb023` as an independent executable
oracle. The repository redistributes normalized numerical outputs and small
test geometries, not xTB source code or binaries. Runtime hashes, command
contracts, and the origin of the QM/MM fixtures are recorded in
`data/conformance/manifest.json`.

## xTB issue #678 difficult-SCC input

Origin: <https://github.com/grimme-lab/xtb/issues/678>

The test-only `data/conformance/inputs/tmacl.xyz` fixture reproduces the
18 atom labels and Cartesian-coordinate rows posted by GitHub user
`corinwagen` in upstream xTB issue #678. The upstream issue states no license
for that user-provided input (`SPDX: NOASSERTION`). gpuxtb retains only these
factual scientific input values, not the issue prose, xTB source, or an xTB
binary. The fixture and gpuxtb-generated SCC diagnostics are source/test data;
they are included in source distributions for test reproducibility, but are
not installed by the native CMake package or bundled in wheels.

The upstream issue node, timestamps, extraction description, fixture digest,
gpuxtb evidence-generator digest, LP64 provider identities, and every generated
output digest are pinned in
`data/conformance/evidence/tmacl-temperature-continuation/manifest.json`.
Generated diagnostics are original gpuxtb outputs under the repository
license; they are not presented as upstream xTB oracle results.

## LAMMPS documentation reference

`docs/theory/qmmm.md` cites the LAMMPS QMMM-XTB adapter at revision
`9ab8ca565e0f71d967587e0bca2015f7d689f19f` to document the external
`b + A q` interface convention. No LAMMPS source code, binary, or numerical
table is redistributed by gpuxtb.

## implib.so (CUDA loader shim generator)

Repository: <https://github.com/yugr/Implib.so>

License: `MIT` (`LICENSES/MIT.txt`).

`cmake/3rdparty/implib/` vendors the `implib-gen.py` generator and its `arch/`
templates as a build-time tool (MIT, Copyright 2017-2023 Yury Gribov), copied
from the vendored revision in deepmd-kit
`6f4fc02ae058ef11848046af01a1a756f3229c29` (which includes the upstream fix
for yugr/Implib.so#34 and a small modification to tolerate a missing CUDA
library). Exact file paths, modes, Git blobs, SHA-256 digests, and the source
tree are recorded in `cmake/3rdparty/implib_manifest.json`.

The generator runs at CMake configure time. Its generated C and assembly
sources remain in the build tree, but their compiled trampoline and initializer
code is embedded in the distributed `libgpuxtb` binary. Those trampolines
dynamically resolve the CUDA host runtime, math, and driver APIs (cudart,
cuBLAS, cuSOLVER, and libcuda) rather than retaining ordinary shared-library
`DT_NEEDED` entries for those host libraries. This mechanism is an engineering
boundary, not the legal basis for combining with CUDA.

## ccache (CI build-time tool)

Repository: <https://github.com/ccache/ccache>

License: `GPL-3.0-or-later`.

The CI workflows use ccache as a compiler launcher (via
`CMAKE_*_COMPILER_LAUNCHER`) to reuse compiled C, C++, and CUDA objects across
runs. CI downloads the pinned static release binary
`ccache-4.13.6-linux-<arch>-musl-static` (SHA-256 verified in
`python/ci/install-ccache.sh`; x86_64
`156ec57c5198cc849d92834023d09910b83dc5504c6cf405d09e6ae7b208a3e5`, aarch64
`2098d561e4a8e36bd06a29aedce53ea90c7e365f9573a93d91c230efbf96a958`) and
installs it only inside the disposable build container. ccache is a build-time
tool: it is not vendored into the repository, not redistributed in source
archives, installs, or wheels, and does not alter the licensing of build
outputs (the project is itself `GPL-3.0-or-later`, so building with ccache
introduces no incompatibility).

## PyTorch CI test dependency

Repository: <https://github.com/pytorch/pytorch>

License: `BSD-3-Clause` (retained by the separately installed distribution).

The required Python CI job installs PyTorch 2.13.0 from PyPI solely to execute
the public `gpuxtb_torch` CPU/autograd tests. The canonical resolution and
artifact hashes, including PyTorch's separately installed transitive
dependencies, are recorded in `uv.lock`. PyTorch is imported lazily by the
optional integration and is not a gpuxtb runtime dependency, project extra,
source-distribution payload, native install artifact, or bundled wheel file.
The locked Linux resolution also installs NVIDIA CUDA provider packages under
their vendor terms; those test-environment packages are likewise not
redistributed in gpuxtb artifacts.

## OpenBLAS runtime dependency

Repository: <https://github.com/MacPython/openblas-libs>

The separately installed `scipy-openblas32` distribution provides gpuxtb's
default Linux LP64 LAPACKE+CBLAS runtime. Version 0.3.34.0.0 was reviewed for
this policy: its own license payload records the MacPython wrapper under
BSD-2-Clause, OpenBLAS and LAPACK under BSD-3-Clause terms, and its GCC runtime
dependencies under GPL-3.0 with the GCC Runtime Library Exception. The runtime
is not bundled in gpuxtb source archives, native installs, or wheels; its own
Python distribution retains the complete notices and license texts.

## CUDA and Intel MKL provider components

CUDA host libraries, the NVIDIA driver, and Intel MKL are not part of gpuxtb
and remain under their vendor licenses. gpuxtb artifacts must not bundle their
shared or static library files. CUDA providers may be installed separately
through the `cuda12` Python extra or supplied by the system. MKL is not a
Python dependency; native users may explicitly select a compatible
`libmkl_rt` through `GPUXTB_CPU_LINALG_LIBRARY`. That selection is used only to
validate one coherent adjacent `libmkl_intel_lp64`, `libmkl_sequential`, and
`libmkl_core` cohort. gpuxtb's private shim loads those unbundled components in
a separate link-map namespace and does not load the selected `libmkl_rt`.

Dynamic loading does not itself resolve GPL compatibility. Jinzhe Zeng grants
the narrowly scoped GPLv3 Section 7 permission in
`CUDA_MKL_LINKING_EXCEPTION` for the enumerated provider interfaces and
compiler support. Provider code remains under vendor terms, and the permission
grants no redistribution rights beyond those terms. gpuxtb's device link
passes `--cudadevrt=none`; NVIDIA libdevice code incorporated by nvcc may still
be present and is covered expressly by the permission.

## 3Dmol.js

Repository: <https://github.com/3Dmol/3Dmol.js>

License: `BSD-3-Clause` (`LICENSES/3Dmol.js-BSD-3-Clause.txt`; Copyright (c)
2014, University of Pittsburgh and contributors; incorporates code from
GLmol, Three.js, and jQuery per the upstream license text).

Build-time npm dependency (not vendored):

- `web/package.json` pins `3dmol@2.5.5` (published 2026-05-22); the exact
  resolution and integrity is recorded in `web/package-lock.json`.
- The prebuilt browser bundle `node_modules/3dmol/build/3Dmol-min.js` is
  downloaded by `npm ci` during the CMake web build and copied into the
  artifact by `web/CMakeLists.txt`. Its content hash is SHA-256
  `f7cc78921ae72e7623e89cdd111434f58c2efddd2ffda1cd212644b406fb8016`, with the
  upstream `/*! 3dmol v2.5.5 ... */` banner retained at the top of the file.

The gpuxtb WASM web demo (`web/`) uses it only for client-side molecular
visualization of the user-supplied geometry. It is not part of the native
library, CMake installs, source archives, or Python wheels.

## Distribution policy

Generated artifacts remain accompanied by their upstream SPDX identifier,
pinned provenance manifest, and applicable license text. Source archives must
also retain the complete pinned implib source tree used by the build. Source
archives, Python wheels, and native CMake installs must retain this file, the
project license, `CUDA_MKL_LINKING_EXCEPTION`, compatible third-party license
texts, and all applicable provenance manifests.
