# Third-party notices and provenance

xTBloom is distributed under `GPL-3.0-or-later`; see `LICENSE`. The additional
permission in `CUDA_MKL_LINKING_EXCEPTION` applies only to xTBloom material
whose copyright holder authorizes it. The following upstream material is
retained under its own terms and is not relicensed by that permission.
The manifests named below pin the source revisions and content digests used
to produce the redistributed data.

## Python build and Git-version tooling

The isolated PEP 517 environment installs two direct build requirements from
PyPI:

- scikit-build-core >=1.0.3
  (<https://github.com/scikit-build/scikit-build-core>, `Apache-2.0`) provides
  the build backend and dynamic-metadata bridge;
- setuptools-scm >=10.2.1
  (<https://github.com/pypa/setuptools-scm>, `MIT`) parses Git and archive
  metadata according to xTBloom's strict tag configuration.

These tools are not vendored, linked into `libxtbloom`, or redistributed in
xTBloom source archives, native installs, or wheels. Their installed Python
distributions retain their own license metadata; the corresponding Apache and
MIT texts are also available as `LICENSES/Apache-2.0.txt` and
`LICENSES/MIT.txt`.

## array-api-compat

Repository: <https://github.com/data-apis/array-api-compat>

License: `MIT` (`LICENSES/array-api-compat-MIT.txt`; Copyright (c) 2022 Consortium for Python Data API Standards).

The xTBloom Python package depends on `array-api-compat` (a backend-neutral
shim, not an array backend) to probe dense eager arrays from NumPy, CuPy,
JAX, and PyTorch without importing those packages. Reviewed release 1.15.0,
commit `076218e4f5aa18578418c7d04fad9ab581a16bb8`, Python `>=3.10`, with no
`Requires-Dist` entries. The `py3-none-any` wheel is
`array_api_compat-1.15.0-py3-none-any.whl` (SHA-256
`7b1b9c53269061403fd5f45a8de349f16e7887653328bfa0c5f2d45299ff0a8e`); the
sdist is SHA-256
`53c5f922491bf15f62847afafc4e39eedfae57d218988fefb8cce39c2a9b3dea`. It is a
runtime Python dependency only and is not bundled in xTBloom source archives,
native installs, or wheels; the canonical resolution is recorded in
`pyproject.toml` and `uv.lock`.

## DLPack

Specification: <https://github.com/dmlc/dlpack>

License: `Apache-2.0` (`LICENSES/Apache-2.0.txt`).

No DLPack header or source is bundled, linked, or vendored. xTBloom
independently reimplements the *byte layout* of the DLPack 1.0 managed-tensor
data structures (`DLDataType`, `DLDevice`, `DLTensor`, `DLManagedTensor`, and
`DLManagedTensorVersioned`) as literal C/C++ struct mirrors in
`src/runtime/dlpack_layout.hpp` and as matching ctypes mirrors in
`python/xtbloom/_dlpack.py`, with byte-exact static assertions. These layouts
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
for that user-provided input (`SPDX: NOASSERTION`). xTBloom retains only these
factual scientific input values, not the issue prose, xTB source, or an xTB
binary. The fixture and xTBloom-generated SCC diagnostics are repository-only
validation data. They remain available from the hash-pinned Git source and are
excluded from installation-focused PyPI source distributions, native CMake
installs, and wheels.

The upstream issue node, timestamps, extraction description, fixture digest,
xTBloom evidence-generator digest, LP64 provider identities, and every generated
output digest are pinned in
`data/conformance/evidence/tmacl-temperature-continuation/manifest.json`.
Generated diagnostics are original xTBloom outputs under the repository
license; they are not presented as upstream xTB oracle results.

## LAMMPS documentation reference

`docs/theory/qmmm.md` cites the LAMMPS QMMM-XTB adapter at revision
`9ab8ca565e0f71d967587e0bca2015f7d689f19f` to document the external
`b + A q` interface convention. No LAMMPS source code, binary, or numerical
table is redistributed by xTBloom.

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
code is embedded in the distributed `libxtbloom` binary. Those trampolines
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

## Cloudflare Pages deployment tools

GitHub Action for Cloudflare Pages repository:
<https://github.com/andykenward/github-actions-cloudflare-pages>

License: `MIT` ([upstream license at the pinned revision](https://github.com/andykenward/github-actions-cloudflare-pages/blob/46d86e1caa6b86365a41d335db65a6936a1beb39/LICENSE)).

The `wasm-web-pages` workflow uses this third-party Marketplace action only to
upload the already validated wasm32 site to a pre-existing Cloudflare Pages
Direct Upload project after pushes to `main`, and to record the result as a
GitHub Deployment. The workflow pins the signed v3.5.0 release commit
`46d86e1caa6b86365a41d335db65a6936a1beb39`, rather than a mutable tag. At
that revision, `action.yml` has SHA-256
`77f3fdafc9ad4e8ff66d8007d3d68bc8b0272f61f33b7e3322e6b318e610c0fb` and
the executed `dist/deploy/index.js` bundle has SHA-256
`2a0558e74fbbd8a1080140cf90bade769c9bdadf6f632afd8581c42802616b22`.
The action runs only inside the disposable GitHub-hosted deployment job; it is
not vendored into the repository or redistributed in xTBloom source archives,
native installs, wheels, or the deployed site.

The action invokes Wrangler 3.114.17 from npm. That release corresponds to
Cloudflare Workers SDK tag commit
`f21ee75d09f24e655574e9bae920585e1a31b15e`, is offered under
`MIT OR Apache-2.0` ([upstream MIT license](https://github.com/cloudflare/workers-sdk/blob/f21ee75d09f24e655574e9bae920585e1a31b15e/LICENSE-MIT)
and [upstream Apache-2.0 license](https://github.com/cloudflare/workers-sdk/blob/f21ee75d09f24e655574e9bae920585e1a31b15e/LICENSE-APACHE)), and has npm integrity
`sha512-tAvf7ly+tB+zwwrmjsCyJ2pJnnc7SZhbnNwXbH+OIdVas3zTSmjcZOjmLKcGGptssAA3RyTKhcF9BvKZzMUycA==`.
The published npm tarball has SHA-256
`e8e0028a83a3ca86a4ac5f27048c4602bdb368a01bd0486225dc7df6341fdb32`.
The exact version input makes the reviewed runtime explicit instead of
inheriting the Action release's default. The action still lets npm resolve
Wrangler's declared transitive ranges at job runtime, so future Action or
Wrangler pin changes require a renewed audit. Those packages are downloaded
into the runner only and no Wrangler bytes enter an xTBloom distribution
artifact.

## Nox validation orchestrator

Repository: <https://github.com/wntrblm/nox>

License: `Apache-2.0` (`LICENSES/Apache-2.0.txt`).

The optional developer validation workflow uses Nox 2026.7.11 solely to
orchestrate the repository's authoritative CMake, CTest, uv, conformance, and
licensing commands. The PyPI wheel
`nox-2026.7.11-py3-none-any.whl` has SHA-256
`f5e811693ee8374d269396204eb39990d2084da67ed968239f94301805c9a169`; the
sdist has SHA-256
`dec9bd2c854540a2d5c0b841eaaf1d23a7c26cd90af36d9f1f1668b34524bfd9`.
Nox and its transitive dependencies are resolved in the locked `nox`
dependency group. They are test/development tools only: they are not project
extras, runtime dependencies, native install payloads, or bundled wheel files.

## PyTorch CI test dependency

Repository: <https://github.com/pytorch/pytorch>

License: `BSD-3-Clause` (retained by the separately installed distribution).

The required Python and wheel CI jobs install PyTorch 2.13.0 from PyPI solely
to execute the public `xtbloom_torch` CPU/autograd tests on Linux, macOS arm64,
and Windows AMD64. The canonical resolution and artifact hashes, including
PyTorch's separately installed transitive dependencies, are recorded in
`uv.lock`. PyTorch 2.13 publishes no canonical-PyPI runtime wheel for macOS
x86_64 or Windows ARM64, so those xTBloom wheel jobs require the optional
extension to be absent instead of accepting an untested binary. PyTorch is
imported lazily by the optional integration and is not an xTBloom runtime
dependency, project extra, source-distribution payload, native install
artifact, or bundled wheel file. The locked Linux resolution also installs
NVIDIA CUDA provider packages under their vendor terms; those test-environment
packages are likewise not redistributed in xTBloom artifacts.

## LibTorch Stable ABI headers (vendored build input)

Repository: <https://github.com/pytorch/pytorch>

License: `BSD-3-Clause` (`LICENSES/BSD-3-Clause.txt`; Copyright (c) 2016,
Facebook, Inc.)

The optional compiled torch integration `libxtbloom_torch_ext` is written
against the LibTorch Stable ABI. The exact transitive `#include` closure of
its stable-ABI headers is vendored in `cmake/3rdparty/torch-stable/` from the
PyPI `torch 2.12.1` wheel so the extension compiles without downloading torch.
Every file is pinned by Git blob and SHA-256 in
`cmake/3rdparty/torch-stable/manifest.json` (tree `e2df0197562bc2b0f55ee910d9899ecaac465e78`), which is
regenerated only through `tools/torch_stable_vendor.py --check`. The extension
links a build-time-only platform stub that defines exactly the
`aoti_torch_*` / `torch_library_impl` / `torch_get_mutable_data_ptr` symbols it
references. Linux gives the stub the real `libtorch_cpu.so` SONAME; macOS uses
the official `@rpath/libtorch_cpu.dylib` install name; Windows builds a private
`torch_cpu.dll` solely so CMake emits the matching architecture-specific
`torch_cpu.lib`. The shipped extension binds to the Torch runtime the Python
layer imports first. The vendored headers, generated stub source, stub shared
library, and Windows import library are build-time inputs only: they are never
copied into native installs or wheels (the sdist retains only the pinned header
tree and extension source so offline wheel builds remain possible). The
extension itself loads on any compatible Torch >= 2.10 because
`TORCH_TARGET_VERSION` floors the emitted symbol set at 2.10.

## Matplotlib publication tool

Repository: <https://github.com/matplotlib/matplotlib>

License: Matplotlib License (classified by the upstream distribution as the
Python Software Foundation License and retained by the separately installed
distribution).

The benchmark figure renderer declares Matplotlib 3.10.9 through PEP 723
inline script metadata. Its complete isolated PyPI resolution and artifact
hashes are recorded in the repository-only
`benchmarks/plot_natoms_cross_engine.py.lock`. Matplotlib, its dependencies,
and that publication lock are not xTBloom project dependencies or payloads of
installation-focused source distributions, native installs, or wheels.

## Private OpenBLAS wheel provider

Repository: <https://github.com/MacPython/openblas-libs>

The `scipy-openblas32` project states that it is a build artifact and must not
be used as an end-user dependency. xTBloom therefore never publishes it in
`Requires-Dist` and never imports its Python module. Version 0.3.34.0.0 is the
exact reviewed LP64 LAPACKE+CBLAS wheel-build input for Linux x86_64/aarch64,
macOS x86_64/arm64, and Windows AMD64/ARM64. Linux exports the local thread
control required by xTBloom's worker-scoped guard. The desktop binaries expose
only global thread control, so xTBloom loads a renamed private provider by
absolute sibling path, fixes that private image to one thread once during
initialization, and never relaxes the local-thread requirement for a system
provider.

The source manifest pins MacPython release commit
`7e5538356afac3934e872b8b572799b875900657`, OpenBLAS commit
`e0166008be8e466242aa76b2ff75ce3f0fbf574a`, all six upstream wheel hashes,
their platform-specific packaged-license hashes, exact local copies of the
Linux, macOS, Windows AMD64, and Windows ARM64 license variants, and every
redistributed native binary. CMake verifies those bytes without importing the
package.
Linux links a wheel-only private shim to the provider; `auditwheel repair`
follows that shim's `DT_NEEDED` closure, collision-renames and redistributes
OpenBLAS plus the required `libgfortran`/`libquadmath` components, rewrites the
vendored dependency closure for private relative resolution, and gives the
shim a relative RPATH into that private directory. `libxtbloom` lazily loads
the shim in a new glibc link-map namespace.

macOS copies the provider plus `libgfortran`, `libquadmath`, and `libgcc_s`,
gives every image a content-hash-qualified xTBloom-private install ID, rewrites
all intra-cohort load commands to those private names, and ad-hoc signs every
derived image. The provider is loaded by a canonical absolute sibling path,
and its dispatch symbols must resolve back to that exact image. Windows copies
only the self-contained provider DLL under a content-hash-qualified xTBloom
filename, loads it by absolute sibling path, and requires all dispatch symbols
to belong to the returned module handle; the AMD64 artifact depends only on
Windows/UCRT system DLLs, while ARM64 also uses the system `VCRUNTIME140.dll`.
These desktop mechanisms avoid name/PATH discovery but are not described as
equivalent to Linux link-map isolation.
`libxtbloom` itself has no hard OpenBLAS dependency on any platform. Source
archives and ordinary native installs contain the provenance and license
records but no OpenBLAS or compiler-runtime binaries.

The retained upstream license records the MacPython wrapper under
BSD-2-Clause, OpenBLAS and LAPACK under BSD-3-Clause terms, redistributed GCC
runtimes under GPL-3.0 with the GCC Runtime Library Exception, and
`libquadmath` under LGPL-2.1-or-later. The upstream Windows ARM64 wheel's
packaged `LICENSE.txt` contains only the 1,344-byte MacPython BSD-2 text because
that build path did not append `tools/LICENSE_win32.txt`. xTBloom therefore
also retains that exact file from the pinned release commit as
`LICENSES/scipy-openblas32-tools-LICENSE_win32.txt` (SHA-256
`1ce4c83d89bc30a0a97d4bc18d72ccaa9d3cb7c90ba1408c6b3e29ebf0c5a71c`)
so the complete OpenBLAS/LAPACK and applicable GCC runtime terms accompany
both Windows wheels.

The exact packaged-license variants are retained as
`LICENSES/scipy-openblas32-0.3.34.0.0.txt` (Linux),
`LICENSES/scipy-openblas32-0.3.34.0.0-macos.txt`,
`LICENSES/scipy-openblas32-0.3.34.0.0-windows-amd64.txt`, and
`LICENSES/scipy-openblas32-0.3.34.0.0-windows-arm64.txt`. Their hashes are
recorded per target in the source manifest; mixed CRLF/LF bytes in the upstream
Windows records are preserved verbatim.

## CUDA and Intel MKL provider components

CUDA host libraries, the NVIDIA driver, and Intel MKL are not part of xTBloom
and remain under their vendor licenses. xTBloom artifacts must not bundle their
shared or static library files. CUDA providers may be installed separately
through the `cuda12` Python extra or supplied by the system. MKL is not a
Python dependency; native users may explicitly select a compatible
`libmkl_rt` through `XTBLOOM_CPU_LINALG_LIBRARY`. That selection is used only to
validate one coherent adjacent `libmkl_intel_lp64`, `libmkl_sequential`, and
`libmkl_core` cohort. xTBloom's private shim loads those unbundled components in
a separate link-map namespace and does not load the selected `libmkl_rt`.

Dynamic loading does not itself resolve GPL compatibility. Jinzhe Zeng grants
the narrowly scoped GPLv3 Section 7 permission in
`CUDA_MKL_LINKING_EXCEPTION` for the enumerated provider interfaces and
compiler support. Provider code remains under vendor terms, and the permission
grants no redistribution rights beyond those terms. xTBloom's device link
passes `--cudadevrt=none`; NVIDIA libdevice code incorporated by nvcc may still
be present and is covered expressly by the permission.

## Eigen WebAssembly linear-algebra provider

Repository: <https://gitlab.com/libeigen/eigen>

Version: Eigen 5.0.1, tag revision
`bc3b39870ecb690a623a3f49149a358b95c5781d`.

Primary license: `MPL-2.0`. The official upstream source tree also contains
BSD-3-Clause, Apache-2.0, and other embedded permissive notices. Exact upstream
records are preserved under `LICENSES/eigen/` as `COPYING.MPL2`, `COPYING.BSD`,
`COPYING.APACHE`, `COPYING.MINPACK`, and `COPYING.README`. The Pages artifact
also carries exact copies of the notice-bearing
`Half.h`, `BFloat16.h`, `AlignedBox.h`, and `InverseSize4.h` headers under
`LICENSES/eigen/notices/`, covering the embedded Fabian Giesen, TensorFlow,
Willow Garage/OSRF, and Intel grants reached by the compiled include graph.

Eigen is a WebAssembly build input, not a vendored repository dependency.
`web/CMakeLists.txt` obtains the official release archive only when the Web demo
is enabled, verifies the fixed SHA-256 before extraction, and may instead use
the same local archive supplied through `XTBLOOM_WEB_EIGEN_ARCHIVE` for an
offline build. Native builds do not fetch Eigen. The wrapper uses Eigen's
`SelfAdjointEigenSolver`, `LLT`, matrix multiplication, and triangular solves
behind the LP64 LAPACKE/CBLAS symbols required by xTBloom's unchanged runtime
loader. The upstream `unsupported/` tree contains MINPACK-derived
implementations and is not included by the wrapper; `COPYING.MINPACK` is
retained as part of Eigen's complete legal record set.

The official release archive is
<https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz>,
SHA-256
`e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec`.
`cmake/3rdparty/eigen_manifest.json` records the archive URL, size, SHA-256,
acquisition policy, and every retained legal file's exact size and digest. The
repository and source distribution retain that compact provenance/legal
payload but not the archive or header tree. Native CMake installs and Python
wheels exclude all Eigen material. The Pages artifact carries the compiled
side module inside `xtbloom_web.data`, the five exact upstream license records,
the four notice-bearing headers above, and the provenance manifest so browser
recipients can identify and obtain the corresponding source.

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

The xTBloom WASM web demo (`web/`) uses it only for client-side molecular
visualization of the user-supplied geometry. It is not part of the native
library, CMake installs, source archives, or Python wheels.

The prebuilt 3Dmol bundle incorporates these npm transitive dependencies. They
are distributed inside `vendor/3Dmol-min.js`, rather than as separate files:

- iobuffer 5.4.0, repository <https://github.com/jDataView/iobuffer>, MIT
  (`LICENSES/iobuffer-MIT.txt`; Copyright (c) 2015 Michaël Zasso), npm
  integrity
  `sha512-DRebOWuqDvxunfkNJAlc3IzWIPD5xVxwUNbHr7xKB8E6aLJxIPfNX3CoMJghcFjpv6RWQsrcJbghtEwSPoJqMA==`;
- netcdfjs 3.0.0, repository <https://github.com/cheminfo/netcdfjs>, MIT
  (`LICENSES/netcdfjs-MIT.txt`; Copyright (c) 2016 cheminfo), npm integrity
  `sha512-LOvT8KkC308qtpUkcBPiCMBtii7ZQCN6LxcVheWgyUeZ6DQWcpSRFV9dcVXLj/2eHZ/bre9tV5HTH4Sf93vrFw==`;
- UPNG.js 2.1.0, repository <https://github.com/photopea/UPNG.js>, MIT
  (`LICENSES/upng-js-MIT.txt`; Copyright (c) 2017 Photopea), npm integrity
  `sha512-d3xzZzpMP64YkjP5pr8gNyvBt7dLk/uGI67EctzDuVp4lCZyVMo0aJO6l/VDlgbInJYDY6cnClLoBp29eKWI6g==`;
- pako 2.2.0 and pako 1.0.11, repository
  <https://github.com/nodeca/pako>, MIT for the pako wrapper
  (`LICENSES/pako-MIT.txt`) and Zlib for the ported `/lib/zlib` code
  (`LICENSES/pako-Zlib.txt`). The two locked npm integrities are
  `sha512-zJq6RP/5q+TO2OpFV3FHzlPnFjmkb7Nc99a5SNjJE+uu/PkpChs+NIZSSzbBoD+6kjiISXjfYdwj1ZRQ81dz/w==`
  and
  `sha512-4hLB8Py4zZce5s4yd9XzopqwVv/yGNhV1Bl8NTmCq1763HeK2+EwVTv+leGeL13Dnh2wfbqowVPXCIO0z4taYw==`.

  Pako 2.2.0 records zlib 1.3.2 as its original implementation; the nested
  pako 1.0.11 records zlib 1.2.8.

`web/package-lock.json` (SHA-256
`475e2213ac02fbf2d4a8c4fc287b570fc476da2fda9de3f5a72a2554b5716e71`)
is the reviewed resolution. Every Pages artifact carries the project GPL,
this notice, the additional permission, the 3Dmol license, all transitive npm
license texts above, and the parameter-data licenses and provenance manifests.

## OpenChemLib JS

Repository: <https://github.com/cheminfo/openchemlib-js>

License: `BSD-3-Clause` (`LICENSES/openchemlib-BSD-3-Clause.txt`; Copyright
(c) 2015-2017, cheminfo).

The optional browser SMILES workflow fetches OpenChemLib 9.21.0 from exact
jsDelivr URLs at runtime. The reviewed JavaScript release commit is
`36aec7791ac38e7fdc23a37ba07e19514eb1e5c9`; its OpenChemLib Java submodule is
revision `27d2b2fe2195ec0b159c3aa2cae3bc1464b41daf`. The browser imports
`https://cdn.jsdelivr.net/npm/openchemlib@9.21.0/dist/openchemlib.js` (SHA-256
`5978967b12e938208e8d36222370f88fd615a2b5ec83f02e435caab26f3f4cb3`) and
registers
`https://cdn.jsdelivr.net/npm/openchemlib@9.21.0/dist/resources.json` (SHA-256
`d2741130d5a5546aeebebc43eb3dac937881b04755fefe5925e4b228a56bee14`).
Floating `latest` and jsDelivr `+esm` transformations are not used.

OpenChemLib parses SMILES, adds explicit hydrogens during seeded 3D conformer
generation, and applies an MMFF94 pre-relaxation before the coordinates enter
the xTBloom web adapter. The complete registered `resources.json` also contains
COD bond-length and torsion statistics, MMFF94/MMFF94s parameter tables, and
bundled drug-likeness and toxicity-predictor data. The predictor data is
registered as part of the upstream resource payload but is not called by
xTBloom. Exact paths, sizes, digests, source revisions, license provenance, and
the distribution boundary are recorded in `web/openchemlib_manifest.json`.

The OpenChemLib module and resource bytes are supplied by jsDelivr directly to
the user's browser; they are not vendored into the repository, linked into
`xtbloom_web.wasm`, copied into the Pages artifact, installed with the native
library, or bundled in Python wheels. The deployed site does retain the license
text and provenance manifest next to its other legal material.

## Distribution policy

Generated artifacts remain accompanied by their upstream SPDX identifier,
pinned provenance manifest, and applicable license text. Source archives must
also retain the complete pinned implib source tree used by the build. Source
archives, Python wheels, and native CMake installs must retain this file, the
project license, `CUDA_MKL_LINKING_EXCEPTION`, compatible third-party license
texts, and all applicable provenance manifests.
