"""Reproducible local validation entry points for developers and AI agents.

Nox is deliberately only an orchestrator here.  The authoritative build and
test tools remain CMake/CTest and uv, and Python validation still installs the
real non-editable wheel before importing xTBloom.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree

import nox

ROOT = Path(__file__).resolve().parent
BUILD_ROOT = ROOT / "build" / "nox"
NOX_VERSION = "2026.7.11"
PYPI_INDEX = "https://pypi.org/simple"

CPU_REQUIRED_TESTS = {
    "xtbloom.abi_symbols",
    "xtbloom.batch_validation",
    "xtbloom.c_api",
    "xtbloom.conformance.invariants_cpu",
    "xtbloom.conformance.public_cpu",
    "xtbloom.cpu.public_inference",
    "xtbloom.gfn2.eigensolver",
    "xtbloom.runtime",
}

CUDA_REQUIRED_TESTS = CPU_REQUIRED_TESTS | {
    "xtbloom.conformance.invariants_cuda_device",
    "xtbloom.conformance.invariants_cuda",
    "xtbloom.conformance.invariants_cuda_mixed",
    "xtbloom.conformance.public_cuda_device",
    "xtbloom.conformance.public_cuda",
    "xtbloom.conformance.public_cuda_mixed",
    "xtbloom.cuda.gfn2_runtime_parity",
    "xtbloom.cuda.public_api",
}

nox.needs_version = f"=={NOX_VERSION}"
nox.options.sessions = ["agent"]


def _run(
    session: nox.Session,
    *args: str,
    env: dict[str, str] | None = None,
    silent: bool = False,
) -> str | None:
    """Run one repository tool without introducing a Nox virtual environment."""
    environment = dict(env or {})
    if args[0] in {"uv", "uvx"}:
        # Site-wide uv configuration may select a mirror.  The committed lock
        # is canonical against PyPI, so every resolving uv command must use the
        # same index or --locked/--check would fail for configuration drift.
        environment.setdefault("UV_DEFAULT_INDEX", PYPI_INDEX)
    if args[0] == "uv":
        # The documented runner is itself uv-isolated. Point nested project uv
        # commands at the repository environment explicitly so the runner's
        # temporary VIRTUAL_ENV neither redirects them nor contaminates captured
        # command output with a mismatched-environment warning.
        environment.setdefault("VIRTUAL_ENV", str(ROOT / ".venv"))
    return session.run(
        *args,
        external=True,
        env=environment or None,
        silent=silent,
    )


def _sync_python_environment(session: nox.Session, *, tests: bool) -> None:
    """Install the locked CPU wheel in the project environment.

    The documented command runs Nox in a separate uv-isolated environment, so
    the project environment contains only xTBloom and the dependencies needed
    by the selected validation session.
    """
    command = [
        "uv",
        "sync",
        "--locked",
        "--no-editable",
        # CPU validation resolves the reviewed LP64 runtime from this
        # build-only group, but the project wheel itself is still built with
        # provider bundling disabled. This keeps scipy-openblas32 out of the
        # published runtime metadata while making the documented Nox entry
        # points self-contained.
        "--group",
        "wheel-build",
    ]
    if tests:
        command.extend(["--extra", "test", "--group", "torch-testing"])
    _run(session, *command, env={"XTBLOOM_ENABLE_CUDA": "OFF"})


def _resolve_cpu_linalg(session: nox.Session) -> Path:
    """Return the reviewed scipy-openblas32 LP64 runtime as an absolute path."""
    script = (
        "from pathlib import Path; import scipy_openblas32; "
        "print((Path(scipy_openblas32.get_lib_dir()) / "
        "scipy_openblas32.get_library(fullname=True)).resolve())"
    )
    output = _run(
        session,
        "uv",
        "run",
        "--no-sync",
        "python",
        "-c",
        script,
        silent=True,
    )
    runtime = Path(str(output).strip())
    if not runtime.is_absolute() or not runtime.is_file():
        session.error(f"LP64 runtime is unavailable: {runtime}")
    session.log(f"LP64 runtime: {runtime}")
    return runtime


def _configure(
    session: nox.Session,
    build_dir: Path,
    *definitions: str,
    source_dir: Path = ROOT,
) -> None:
    """Create a fresh Ninja configuration so stale cache state cannot hide gates."""
    _run(
        session,
        "cmake",
        "--fresh",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        *definitions,
    )


def _registered_tests(session: nox.Session, build_dir: Path) -> set[str]:
    """Show and return the exact CTest inventory for one configuration."""
    _run(session, "ctest", "--test-dir", str(build_dir), "-N")
    output = _run(
        session,
        "ctest",
        "--test-dir",
        str(build_dir),
        "--show-only=json-v1",
        silent=True,
    )
    payload = json.loads(str(output))
    tests = {entry["name"] for entry in payload["tests"]}
    session.log(f"Registered CTest tests: {len(tests)}")
    return tests


def _require_registered_tests(
    session: nox.Session,
    registered: set[str],
    required: set[str],
) -> None:
    """Fail instead of treating configuration-omitted acceptance tests as passes."""
    missing = sorted(required - registered)
    if missing:
        session.error("Required CTest tests are NOT REGISTERED: " + ", ".join(missing))


def _run_ctest_strict(session: nox.Session, build_dir: Path) -> None:
    """Run CTest and reject exit-code-77 skips as incomplete validation."""
    report = build_dir / "ctest-results.xml"
    report.unlink(missing_ok=True)
    _run(
        session,
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "--no-tests=error",
        "--output-junit",
        str(report),
    )
    cases = ElementTree.parse(report).findall(".//testcase")
    skipped = sum(case.find("skipped") is not None for case in cases)
    session.log(f"CTest result: PASS={len(cases) - skipped} SKIP={skipped}")
    if skipped:
        session.error("CTest registered skips; report them as SKIP, not PASS")


def _run_fast(session: nox.Session) -> None:
    """Run formatting, lock consistency, and whitespace gates."""
    _run(
        session,
        "uvx",
        "prek@0.3.1",
        "run",
        "--show-diff-on-failure",
        "--color=always",
        "--all-files",
    )
    _run(session, "uv", "lock", "--check")
    _run(session, "git", "diff", "--check")


def _run_cpu(session: nox.Session) -> None:
    """Build and run the complete shared CPU public-inference configuration."""
    runtime = _resolve_cpu_linalg(session)
    build_dir = BUILD_ROOT / "cpu-public"
    _configure(
        session,
        build_dir,
        "-DXTBLOOM_ENABLE_CUDA=OFF",
        f"-DXTBLOOM_CPU_LINALG_LIBRARY={runtime}",
        "-DBUILD_SHARED_LIBS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    )
    _run(session, "cmake", "--build", str(build_dir), "--parallel")
    registered = _registered_tests(session, build_dir)
    _require_registered_tests(session, registered, CPU_REQUIRED_TESTS)
    _run_ctest_strict(session, build_dir)


def _installed_library(session: nox.Session) -> Path:
    """Return the native library bundled in the non-editable Python install."""
    output = _run(
        session,
        "uv",
        "run",
        "--no-sync",
        "python",
        "-c",
        "from xtbloom.library import library_path; print(library_path())",
        silent=True,
    )
    library = Path(str(output).strip())
    if not library.is_file():
        session.error(f"Installed xTBloom library is unavailable: {library}")
    session.log(f"Installed xTBloom library: {library}")
    return library


def _run_python_tests(session: nox.Session) -> None:
    """Test the source Python API against the native library bundled in the wheel."""
    library = _installed_library(session)
    runtime = _resolve_cpu_linalg(session)
    if os.name == "nt":
        loader_path_name = "PATH"
    elif sys.platform == "darwin":
        loader_path_name = "DYLD_LIBRARY_PATH"
    else:
        loader_path_name = "LD_LIBRARY_PATH"
    loader_path = str(runtime.parent)
    inherited_loader_path = os.environ.get(loader_path_name)
    if inherited_loader_path:
        loader_path = loader_path + os.pathsep + inherited_loader_path
    test_environment = {
        "PYTHONPATH": str(ROOT / "python"),
        "XTBLOOM_LIBRARY": str(library),
        # The local validation wheel deliberately does not bundle its
        # build-only provider. Give its lazy loader the reviewed runtime path
        # explicitly, matching the native CTest environment.
        loader_path_name: loader_path,
    }
    _run(
        session,
        "uv",
        "run",
        "--no-sync",
        "pytest",
        "python/tests",
        "-q",
        env=test_environment,
    )
    _run(
        session,
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "unittest",
        "-v",
        "benchmarks.test_run",
        "benchmarks.test_natoms_scaling",
        "benchmarks.test_natoms_cross_engine",
        "benchmarks.test_dxtb_adapter",
        "benchmarks.test_evidence_size",
        env=test_environment,
    )


def _run_canonical(session: nox.Session) -> None:
    """Validate canonical data, provenance, licensing, conformance, and oracles."""
    _run(
        session,
        "uv",
        "run",
        "--no-sync",
        "python",
        "tools/parameters/generate_gfn2.py",
        "--check",
    )
    _run(
        session,
        "uv",
        "run",
        "--no-sync",
        "python",
        "tools/conformance/xtbloom_conformance.py",
        "check",
    )
    _run(
        session,
        "uv",
        "run",
        "--no-sync",
        "python",
        "tools/licensing/check_licenses.py",
        "--source-root",
        ".",
    )
    for suite in ("parameters", "conformance", "licensing", "oracle"):
        _run(
            session,
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            f"tests/{suite}",
            "-p",
            "test_*.py",
            "-v",
        )


def _remove_directory(session: nox.Session, path: Path) -> None:
    """Remove one workflow-owned output directory before artifact inspection."""
    _run(session, "cmake", "-E", "remove_directory", str(path))


def _run_install_consumer(
    session: nox.Session,
    *,
    build_dir: Path,
    install_dir: Path,
    consumer_dir: Path,
    mode: str,
) -> None:
    """Install one native package and exercise its external CMake consumer."""
    _remove_directory(session, install_dir)
    _run(session, "cmake", "--install", str(build_dir), "--prefix", str(install_dir))
    _run(
        session,
        "uv",
        "run",
        "--no-sync",
        "python",
        "tools/licensing/check_licenses.py",
        "--source-root",
        ".",
        "--install-prefix",
        str(install_dir),
    )
    _configure(
        session,
        consumer_dir,
        f"-DCMAKE_PREFIX_PATH={install_dir}",
        source_dir=ROOT / "tests" / "install_consumer",
    )
    _run(session, "cmake", "--build", str(consumer_dir), "--parallel")
    _run(session, str(consumer_dir / "xtbloom_install_consumer"), mode)


def _run_package(session: nox.Session) -> None:
    """Validate shared/static native installs, the sdist, and a CPU wheel."""
    _run_install_consumer(
        session,
        build_dir=BUILD_ROOT / "cpu-public",
        install_dir=BUILD_ROOT / "cpu-public-install",
        consumer_dir=BUILD_ROOT / "cpu-public-consumer",
        mode="cpu",
    )

    static_dir = BUILD_ROOT / "cpu-static"
    _configure(
        session,
        static_dir,
        "-DXTBLOOM_ENABLE_CUDA=OFF",
        "-DXTBLOOM_BUILD_TESTS=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    )
    _run(session, "cmake", "--build", str(static_dir), "--parallel")
    _run_install_consumer(
        session,
        build_dir=static_dir,
        install_dir=BUILD_ROOT / "cpu-static-install",
        consumer_dir=BUILD_ROOT / "cpu-static-consumer",
        mode="smoke",
    )

    dist_dir = BUILD_ROOT / "dist-license"
    _remove_directory(session, dist_dir)
    _run(session, "uv", "build", "--sdist", "--out-dir", str(dist_dir))
    _run(
        session,
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        str(dist_dir),
        env={"XTBLOOM_ENABLE_CUDA": "OFF"},
    )
    archives = sorted([*dist_dir.glob("*.tar.gz"), *dist_dir.glob("*.whl")])
    if len(archives) != 2:
        session.error(f"Expected one sdist and one wheel, found: {archives}")
    _run(
        session,
        "uv",
        "run",
        "--no-sync",
        "python",
        "tools/licensing/check_licenses.py",
        "--source-root",
        ".",
        *(str(archive) for archive in archives),
    )


def _cuda_compiler(session: nox.Session) -> str:
    """Resolve the explicitly selected CUDA compiler without enabling AUTO."""
    requested = os.environ.get("CUDACXX", "nvcc")
    if Path(requested).parent != Path("."):
        candidate = Path(requested).expanduser().resolve()
        if candidate.is_file():
            return str(candidate)
    resolved = shutil.which(requested)
    if resolved is None:
        session.error("CUDA compiler is UNAVAILABLE; put nvcc on PATH or set CUDACXX")
    return resolved


def _cuda_architectures(session: nox.Session) -> str:
    """Use the visible GPUs' actual compute capabilities unless explicitly set."""
    configured = os.environ.get("XTBLOOM_CUDA_ARCHITECTURES")
    if configured:
        return configured
    _run(session, "nvidia-smi", "-L")
    output = _run(
        session,
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader",
        silent=True,
    )
    capabilities = {
        line.strip().replace(".", "")
        for line in str(output).splitlines()
        if line.strip() and line.strip() != "N/A"
    }
    if not capabilities:
        session.error("A real NVIDIA GPU is UNAVAILABLE")
    return ";".join(sorted(capabilities))


def _run_cuda(session: nox.Session, compiler: str, architectures: str) -> None:
    """Build explicit CUDA support and require real-GPU public validation."""
    runtime = _resolve_cpu_linalg(session)
    session.log(f"CUDA compiler: {compiler}")
    session.log(f"CUDA architectures: {architectures}")

    build_dir = BUILD_ROOT / "cuda"
    _configure(
        session,
        build_dir,
        "-DXTBLOOM_ENABLE_CUDA=ON",
        f"-DCMAKE_CUDA_COMPILER={compiler}",
        f"-DCMAKE_CUDA_ARCHITECTURES={architectures}",
        f"-DXTBLOOM_CPU_LINALG_LIBRARY={runtime}",
        "-DBUILD_SHARED_LIBS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    )
    _run(session, "cmake", "--build", str(build_dir), "--parallel")
    registered = _registered_tests(session, build_dir)
    _require_registered_tests(session, registered, CUDA_REQUIRED_TESTS)
    _run_ctest_strict(session, build_dir)


@nox.session(venv_backend="none")
def fast(session: nox.Session) -> None:
    """Run the fast repository checks."""
    session.chdir(ROOT)
    _run_fast(session)


@nox.session(venv_backend="none")
def cpu(session: nox.Session) -> None:
    """Run the complete shared CPU native test configuration."""
    session.chdir(ROOT)
    _sync_python_environment(session, tests=False)
    _run_cpu(session)


@nox.session(venv_backend="none")
def python(session: nox.Session) -> None:
    """Run Python and benchmark-adapter tests against a non-editable wheel."""
    session.chdir(ROOT)
    _sync_python_environment(session, tests=True)
    _run_python_tests(session)


@nox.session(venv_backend="none")
def canonical(session: nox.Session) -> None:
    """Run canonical data, conformance, oracle, and licensing checks."""
    session.chdir(ROOT)
    _sync_python_environment(session, tests=False)
    _run_canonical(session)


@nox.session(venv_backend="none")
def package(session: nox.Session) -> None:
    """Run CPU tests, native install consumers, and distribution checks."""
    session.chdir(ROOT)
    _sync_python_environment(session, tests=False)
    _run_cpu(session)
    _run_package(session)


@nox.session(venv_backend="none")
def cuda(session: nox.Session) -> None:
    """Run strict CUDA validation on a visible real NVIDIA GPU."""
    session.chdir(ROOT)
    compiler = _cuda_compiler(session)
    architectures = _cuda_architectures(session)
    _sync_python_environment(session, tests=True)
    _run_cuda(session, compiler, architectures)


@nox.session(venv_backend="none")
def agent(session: nox.Session) -> None:
    """Run the standard one-command CPU validation workflow for AI agents."""
    session.chdir(ROOT)
    _run_fast(session)
    _sync_python_environment(session, tests=True)
    _run_cpu(session)
    _run_python_tests(session)
    _run_canonical(session)


@nox.session(venv_backend="none")
def full(session: nox.Session) -> None:
    """Run the standard agent workflow plus native and archive packaging gates."""
    session.chdir(ROOT)
    _run_fast(session)
    _sync_python_environment(session, tests=True)
    _run_cpu(session)
    _run_python_tests(session)
    _run_canonical(session)
    _run_package(session)
