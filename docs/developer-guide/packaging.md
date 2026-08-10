# Packaging, dependencies, and licensing

xTBloom has three related distribution surfaces:

1. native CMake installs for C and C++ consumers;
2. source distributions containing build inputs and provenance; and
3. Python wheels containing the Python package and one native `libxtbloom`.

Changing one surface does not prove the other two remain correct.

## Python and native product versions

Release tags use the strict form `vMAJOR.MINOR.PATCH`, with each component
written canonically (`0` or a non-zero digit followed by digits). Native CMake
packages, target filenames, generated public macros, and
`xtbloom_version_string()` use the latest reachable tag without its leading
`v` until another release tag is created.

Python distributions identify the source revision independently. An exact
clean release tag produces the same `MAJOR.MINOR.PATCH` value as the native
library. A post-tag build uses setuptools-scm's `no-guess-dev` and
`node-and-date` schemes, for example
`0.0.0.post1.dev20+ge7c20f0ff` for the twentieth commit after native tag
`v0.0.0`. Python `__version__`, sdist/wheel metadata, and distribution
filenames use that revision-aware value; it does not change the native product
version embedded in the wheel.

`XTBLOOM_API_VERSION`, the Linux symbol-version node, and the ELF `SOVERSION`
are ABI contracts, not product versions. They change only after an explicit ABI
decision; a product tag never changes them automatically.

Native Git checkouts read the nearest reachable strict tag from complete tag
history. Python package metadata comes directly from scikit-build-core's
built-in setuptools-scm provider. During Python builds, CMake validates the
full SCM version and uses its unchanged release tuple as the native tag;
exact-tag builds map directly. Native CMake configuration rejects shallow
history, while automated Python builds fetch complete history so branch
artifacts include the real commit distance. An exact-tag Python build may use
that tag from a shallow checkout, following setuptools-scm semantics.
Repositories without a reachable strict tag fail configuration.

An sdist freezes the Python version into `PKG-INFO`; an unpacked-sdist wheel
build validates that metadata and reconstructs the unchanged native tag.
`.git_archival.txt` records a full describe value for Python SCM identity;
native CMake builds recover the unchanged nearest tag from that same value.
The `v*` namespace is reserved for product versions, so a nearer malformed tag
such as `v1.2` is rejected instead of being silently skipped. There is
intentionally no fallback version.

Release automation must check out complete history and require a clean exact
tag, so published PyPI artifacts use the exact release version without a local
identifier. Branch wheel jobs build directly from their checkout and retain
the revision-aware Python version. The independent sdist job verifies the
frozen `PKG-INFO` path used by unpacked source archives. Validate Python
metadata against Python `__version__`, and independently validate the generated
public header, CMake package, and C API string against the native release tag.

## PyPI documentation

The repository root [`README.md`](../../README.md) is the GitHub-facing overview
for Python and native users. `pyproject.toml` deliberately points
`project.readme` at [`python/README.md`](../../python/README.md), so PyPI renders
a Python-only package page without native developer/test instructions.

Use absolute repository URLs in the PyPI-facing Markdown. Relative links are
resolved by PyPI rather than GitHub and otherwise lead to the wrong location.
Build metadata and inspect the generated `METADATA` payload whenever the
selected README or project URLs change.

## Native install

The install exports `xtbloom::xtbloom`, the public header, version/config files,
licenses, third-party notices, and applicable provenance manifests. Validate
both shared and static consumers when CMake export or dependency behavior
changes.

The shared library exports only versioned `xtbloom_*` C symbols on Linux. CPU
BLAS/LAPACK and CUDA host providers are opened dynamically; they must not
become accidental `DT_NEEDED` dependencies merely to simplify discovery.

## Python wheels

scikit-build-core builds `libxtbloom` through CMake and installs it under the
Python package. The ctypes binding and LibTorch Stable ABI integration are not
CPython extension modules, so one `py3-none-<platform>` wheel serves supported
Python 3 versions. Free-threaded `cp313t`/`cp314t` interpreters therefore need
runtime validation of that same wheel, not separately published wheel bytes.

Linux CUDA wheels contain compiled xTBloom device code but do not bundle CUDA
host shared libraries or the NVIDIA driver. The optional `cuda12` extra installs
supported host providers separately from PyPI. CPU-only installations remain
usable without the proprietary stack.

Linux x86_64/aarch64, macOS x86_64/arm64, and Windows AMD64/ARM64 wheels bundle
one private LP64 OpenBLAS provider. The upstream `scipy-openblas32`
distribution is deliberately build-only: an environment-gated
scikit-build-core override installs the exact reviewed version only for
cibuildwheel's audited native wheel builds, while a non-default uv dependency
group retains every platform artifact hash in `uv.lock`. It must never appear
in project dependencies, extras, editable builds, test environments, or wheel
`METADATA`.

CMake validates the installed distribution version, target, exact packaged
license, and complete native-library inventory through
`python/ci/resolve-openblas-wheel.py` without importing the package. The source
manifest pins every upstream wheel and native payload hash plus exact local
copies of each platform-specific packaged-license variant.

On Linux, a private shim retains the only `DT_NEEDED` edge. `auditwheel repair`
then vendors and collision-renames OpenBLAS and its architecture-specific GCC
runtime closure, and `libxtbloom` loads the shim in a new glibc link-map
namespace. On macOS, every OpenBLAS/libgfortran/libquadmath/libgcc image gets a
content-hash-qualified private filename and LC_ID; all intra-cohort load
commands are rewritten before every derived image is ad-hoc signed. On
Windows, the self-contained provider DLL gets a content-hash-qualified private
filename. macOS loads by canonical absolute sibling path and verifies the
dispatch image path; Windows loads by absolute sibling path and verifies every
dispatch symbol against the returned module handle. Desktop providers expose
only global thread control, so their already-private image is fixed to one
thread once during thread-safe initialization. System providers still require
worker-local thread control.

`python/ci/inspect-openblas-wheel.py` checks final metadata, exact cohort names,
machine architecture, required symbols/exports, private IDs and dependency
paths, signatures, and native dependency closure. `libxtbloom` itself must
remain free of hard OpenBLAS, libgfortran, libquadmath, and shim dependencies.

The compiled Torch extension is included in Linux wheels, macOS arm64 wheels,
and Windows AMD64 wheels. The isolated build uses a non-installed platform stub
with the real runtime identity: `libtorch_cpu.so`,
`@rpath/libtorch_cpu.dylib`, or `torch_cpu.dll` plus its generated import
library. Wheel repair excludes that unresolved external runtime edge, and
payload/linkage checks prove that no Torch library or stub is bundled. The
installed-wheel tests import the separately installed Torch 2.13 runtime first,
then run CPU forward, forces, and autograd through the compiled op.

PyTorch 2.10+ publishes no supported PyPI runtime wheel for macOS x86_64 or
Windows ARM64. xTBloom therefore does not ship an untested extension in those
two wheel cohorts; their payload checks require its absence. Pyodide likewise
has no LibTorch runtime and contains no Torch extension.

The Pyodide `cp314-pyodide_wasm32` wheel targets Pyodide 314.x and its stable
`pyemscripten_2026_0_wasm32` ABI. It carries the reviewed official Pyodide
OpenBLAS side module under a content-qualified private name plus xTBloom's
narrow LAPACKE adapter. Release builds repair that dependency with Pyodide's
supported auditwheel path, then run installed-wheel conformance, invariance,
finite-difference, failure-isolation, and NumPy/SciPy coexistence tests. The
wheel is release-eligible and enters the same PyPI artifact prefix as the
validated native wheels; the existing Web demo continues to use its separate
preloaded Eigen LAPACKE/CBLAS side-module design.

Eigen is acquired only by Web-enabled CMake configurations from the fixed
official archive (or `XTBLOOM_WEB_EIGEN_ARCHIVE` for offline builds). The
repository and sdist retain the provenance manifest and exact legal records,
but the Eigen archive/header tree is excluded from sdists, native installs, and
Python wheels.
Windows/macOS wheels must pass installed GFN2 inference, missing-provider,
concurrency, and host numerical-coexistence tests before being release-eligible.

Final GitHub Releases publish the validated sdist and release-eligible native
wheels through PyPI Trusted Publishing. The upload job runs only for a
non-prerelease published release, uses the protected `pypi` environment, and
receives job-local `id-token: write`; build jobs retain read-only permissions.

Wheel checks cover size, license payload, native symbol and dynamic-dependency
policy, bundled-library discovery, and installed inference. Hosted wheel jobs
without an NVIDIA driver do not count as real CUDA runtime validation.

## Source distributions

The PyPI sdist is an installation artifact, not a repository snapshot. It
contains the native and Python sources needed for a PEP 517 wheel build, the
generated parameter sources plus their deterministic generators, vendored
build inputs, manifests, licenses, notices, and frozen version metadata.

Repository-only validation and publication surfaces stay in Git: native and
Python tests, benchmark harnesses and evidence, conformance/oracle corpora,
maintainer documentation, the Web demo, CI and coding-agent configuration, and
the development dependency lock. The downloaded Eigen archive and header tree
likewise remain Web-build inputs rather than sdist payload; the compact Eigen
provenance and legal records stay distributed. Excluding repository-only files
does not claim that the corresponding validation passed inside the sdist;
release evidence remains in the repository and its issue/CI records.

The sdist supports ordinary PEP 517 CPU or CUDA source builds. It retains the
Pyodide OpenBLAS provenance manifest, exact 13-file recipe source closure, and
five corresponding legal texts as auditable source material. The Pyodide
resolver, downloaded ZIP and provider binary, repair scripts, and installed-
wheel test orchestration remain release-tag checkout or wheel-only; ordinary
sdist builds do not enable Pyodide provider bundling. Official cibuildwheel
jobs therefore build from the exact release tag rather than using the sdist as
their input. The one `python/ci` helper kept in the archive is the desktop/Linux
OpenBLAS manifest resolver that CMake invokes directly when a source build
explicitly requests that reviewed provider input. Repository tests and the Web
demo are likewise checkout-only. A native CMake consumer unpacking the sdist
must configure with
`-DXTBLOOM_BUILD_TESTS=OFF`; ordinary PEP 517 builds set that option already.

Build and inspect a source archive with:

```console
uv build --sdist --out-dir build/dist-license
python3 tools/licensing/check_licenses.py --source-root . \
  build/dist-license/*.tar.gz
```

The archive must retain the license, CUDA/MKL additional permission,
third-party notices, required license texts, parameter generators and source
provenance, the pinned implib generator source, vendored LibTorch Stable ABI
headers, both README roles, and every native/Python input used by ordinary PEP
517 source builds. CI unpacks the archive and builds a CPU wheel from that
extracted tree so a missing hidden checkout dependency cannot be masked by the
repository.

## External material

Every new dependency, copied source, generated dataset, parameter table,
download, CI action, native link, or distributed payload needs an inventory:

- upstream project and immutable revision/version;
- retrieval or deterministic generation procedure;
- source and output digests;
- license and notice source;
- local destination; and
- build-only, runtime, install, sdist, and wheel boundaries.

Update [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md), `LICENSES/`,
manifests, packaging rules, and automated checks as applicable. Dynamic loading
does not itself decide license compatibility. CUDA and MKL use the precise
additional permission in
[`CUDA_MKL_LINKING_EXCEPTION`](../../CUDA_MKL_LINKING_EXCEPTION); provider code
remains under vendor terms.
