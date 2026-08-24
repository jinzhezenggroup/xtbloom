"""Regression tests for the release legal-material validator."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import shutil
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[2]
WHEEL_DIST_INFO = "xtbloom-test.dist-info"
CHECKER_PATH = REPOSITORY / "tools" / "licensing" / "check_licenses.py"
SPEC = importlib.util.spec_from_file_location("xtbloom_check_licenses", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
WHEEL_INSPECTOR_PATH = REPOSITORY / "python" / "ci" / "inspect-cuda-wheel.py"
WHEEL_SPEC = importlib.util.spec_from_file_location(
    "xtbloom_inspect_cuda_wheel", WHEEL_INSPECTOR_PATH
)
assert WHEEL_SPEC is not None and WHEEL_SPEC.loader is not None
WHEEL_INSPECTOR = importlib.util.module_from_spec(WHEEL_SPEC)
WHEEL_SPEC.loader.exec_module(WHEEL_INSPECTOR)
VERSION_INSPECTOR_PATH = REPOSITORY / "python" / "ci" / "check-wheel-version.py"
VERSION_SPEC = importlib.util.spec_from_file_location(
    "xtbloom_check_wheel_version", VERSION_INSPECTOR_PATH
)
assert VERSION_SPEC is not None and VERSION_SPEC.loader is not None
VERSION_INSPECTOR = importlib.util.module_from_spec(VERSION_SPEC)
VERSION_SPEC.loader.exec_module(VERSION_INSPECTOR)


class PeriodicOracleProvenanceTests(unittest.TestCase):
    """Keep generated periodic evidence separate from copied upstream bytes."""

    @staticmethod
    def _copy_reviewed_tree(root: Path) -> None:
        """Copy the corpus and its required license into an isolated root."""
        source = REPOSITORY / "data/conformance/periodic"
        shutil.copytree(source, root / "data/conformance/periodic")
        (root / "LICENSES").mkdir()
        shutil.copy2(
            REPOSITORY / "LICENSES/LGPL-3.0-or-later.txt",
            root / "LICENSES/LGPL-3.0-or-later.txt",
        )
        shutil.copy2(
            REPOSITORY / "LICENSES/Apache-2.0.txt",
            root / "LICENSES/Apache-2.0.txt",
        )
        shutil.copy2(REPOSITORY / "LICENSE", root / "LICENSE")
        tool_source = REPOSITORY / "tools/oracle/periodic_gfn2_terms"
        shutil.copytree(tool_source, root / "tools/oracle/periodic_gfn2_terms")

    def test_reviewed_periodic_corpus_passes(self) -> None:
        """Accept the pinned tblite identity and authored input boundary."""
        CHECKER._check_periodic_oracle_provenance(REPOSITORY)

    def test_reviewed_periodic_term_corpus_passes(self) -> None:
        """Accept the pinned six-family term corpus and probe boundary."""
        CHECKER._check_periodic_term_provenance(REPOSITORY)

    def test_term_fixture_paths_cannot_escape_their_directories(self) -> None:
        """Reject a raw or golden path that traverses outside the term corpus."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_reviewed_tree(root)
            manifest_path = root / "data/conformance/periodic/terms/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["fixtures"][0]["raw"] = (
                "data/conformance/periodic/terms/raw/../../../../LICENSE"
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "escapes its reviewed boundary"
            ):
                CHECKER._check_periodic_term_provenance(root)

    def test_term_runtime_machine_path_is_rejected(self) -> None:
        """Keep the standalone probe attestation independent of one host."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_reviewed_tree(root)
            manifest_path = root / "data/conformance/periodic/terms/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["probe"]["runtime"]["non_system_libraries"][0]["path"] = (
                r"C:\\oracle\\libtblite.dll"
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "machine-local"):
                CHECKER._check_periodic_term_provenance(root)

    def test_term_source_license_drift_is_rejected(self) -> None:
        """Preserve each upstream project's actual SPDX identity."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_reviewed_tree(root)
            manifest_path = root / "data/conformance/periodic/terms/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"]["mctc-lib"]["license"] = "MIT"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "source identity"):
                CHECKER._check_periodic_term_provenance(root)

    def test_machine_local_runtime_path_is_rejected(self) -> None:
        """Keep generated evidence portable across oracle installations."""
        for runtime_path in (
            "/home/user/libtblite.so",
            r"C:\Users\user\libtblite.dll",
        ):
            with (
                self.subTest(runtime_path=runtime_path),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                self._copy_reviewed_tree(root)
                golden = next(
                    (root / "data/conformance/periodic/golden").glob("*.json")
                )
                document = json.loads(golden.read_text(encoding="utf-8"))
                document["provenance"]["runtime"]["libtblite"]["path"] = runtime_path
                golden.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(CHECKER.LicenseCheckError, "machine-local"):
                    CHECKER._check_periodic_oracle_provenance(root)

    def test_non_object_runtime_artifacts_is_rejected(self) -> None:
        """Reject malformed runtime provenance through the licensing contract."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_reviewed_tree(root)
            manifest_path = root / "data/conformance/periodic/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["reference_engine"]["runtime_artifacts"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unreviewed"):
                CHECKER._check_periodic_oracle_provenance(root)

    def test_build_attestation_machine_local_path_is_rejected(self) -> None:
        """Keep compiler/build provenance portable across local installations."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_reviewed_tree(root)
            attestation_path = root / "data/conformance/periodic/tblite-build.json"
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            attestation["build"]["compiler"]["path"] = "/tmp/gfortran"
            attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "machine-local"):
                CHECKER._check_periodic_oracle_provenance(root)

    def test_ewald_reconstruction_machine_local_path_is_rejected(self) -> None:
        """Keep analytic evidence independent of one generator work directory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_reviewed_tree(root)
            reconstruction_path = (
                root / "data/conformance/periodic/ewald-reconstruction.json"
            )
            reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
            reconstruction["generator_path"] = r"C:\oracle\ewald.exe"
            reconstruction_path.write_text(json.dumps(reconstruction), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "machine-local"):
                CHECKER._check_periodic_oracle_provenance(root)

    def test_build_runtime_closure_corruption_is_rejected(self) -> None:
        """Reject any byte drift in the exact non-system runtime closure."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_reviewed_tree(root)
            attestation_path = root / "data/conformance/periodic/tblite-build.json"
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            attestation["runtime"]["non_system_libraries"][0]["sha256"] = "0" * 64
            attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "hash differs"):
                CHECKER._check_periodic_oracle_provenance(root)

    def test_case_paths_cannot_traverse_outside_corpus_directories(self) -> None:
        """Reject lexical prefixes that resolve outside the reviewed corpus."""
        for field, directory in (
            ("input", "inputs"),
            ("golden", "golden"),
        ):
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as directory_name,
            ):
                root = Path(directory_name)
                self._copy_reviewed_tree(root)
                manifest_path = root / "data/conformance/periodic/manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["cases"][0][field] = (
                    f"data/conformance/periodic/{directory}/../../../../"
                    "LICENSES/LGPL-3.0-or-later.txt"
                )
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(
                    CHECKER.LicenseCheckError, "escapes its reviewed boundary"
                ):
                    CHECKER._check_periodic_oracle_provenance(root)


class SourceDistributionBoundaryTests(unittest.TestCase):
    """Keep PyPI sdists limited to wheel-build and provenance inputs."""

    @staticmethod
    def _archive_names(relative_names: set[str]) -> set[str]:
        root = "xtbloom-0.1.0"
        return {f"{root}/{relative}" for relative in relative_names}

    @staticmethod
    def _installation_files() -> set[str]:
        return {
            "CMakeLists.txt",
            "PKG-INFO",
            "cmake/xtbloom_cuda_toolkit.cmake",
            "data/parameters/d4.hpp",
            "python/xtbloom/__init__.py",
            "src/backends/cuda/gfn2_scc_loop.cu",
            "tools/implib_stubgen.py",
        }

    def test_installation_focused_payload_is_accepted(self) -> None:
        """Accept an archive that exactly matches its tracked allow-surface."""
        expected = self._installation_files()
        CHECKER._check_sdist_installation_payload(
            self._archive_names(expected), expected
        )

    def test_repository_only_tree_is_rejected(self) -> None:
        """Prevent tests and similar repository surfaces from leaking back in."""
        expected = self._installation_files()
        names = self._archive_names(expected)
        names.add("xtbloom-0.1.0/tests/runtime_test.cpp")
        with self.assertRaisesRegex(
            CHECKER.LicenseCheckError,
            "unexpected repository-only or generated payload.*tests/",
        ):
            CHECKER._check_sdist_installation_payload(names, expected)

    def test_development_lockfile_is_rejected(self) -> None:
        """Keep the repository uv resolution out of the PEP 517 source input."""
        expected = self._installation_files()
        names = self._archive_names(expected)
        names.add("xtbloom-0.1.0/uv.lock")
        with self.assertRaisesRegex(
            CHECKER.LicenseCheckError,
            "unexpected repository-only or generated payload.*uv.lock",
        ):
            CHECKER._check_sdist_installation_payload(names, expected)

    def test_checkout_wheel_orchestration_is_rejected(self) -> None:
        """Retain only the CMake-invoked resolver from repository CI helpers."""
        expected = self._installation_files()
        names = self._archive_names(expected)
        names.add("xtbloom-0.1.0/python/ci/repair-wheel.sh")
        with self.assertRaisesRegex(
            CHECKER.LicenseCheckError,
            "unexpected repository-only or generated payload.*python/ci/repair-wheel",
        ):
            CHECKER._check_sdist_installation_payload(names, expected)

    def test_missing_build_generator_is_rejected(self) -> None:
        """Retain the CUDA shim generator required by CUDA source builds."""
        expected = self._installation_files()
        names = self._archive_names(expected)
        names.remove("xtbloom-0.1.0/tools/implib_stubgen.py")
        with self.assertRaisesRegex(
            CHECKER.LicenseCheckError, "payload differs.*missing.*implib_stubgen"
        ):
            CHECKER._check_sdist_installation_payload(names, expected)

    def test_generated_binary_is_rejected(self) -> None:
        """Source archives must not carry locally compiled native artifacts."""
        expected = self._installation_files()
        names = self._archive_names(expected)
        names.add("xtbloom-0.1.0/src/libxtbloom.so")
        with self.assertRaisesRegex(
            CHECKER.LicenseCheckError,
            "unexpected repository-only or generated payload.*libxtbloom.so",
        ):
            CHECKER._check_sdist_installation_payload(names, expected)

    def test_tracked_policy_covers_complete_cuda_and_parameter_sources(self) -> None:
        """Derive the allow-surface from tracked files, not directory sentinels."""
        expected = CHECKER._tracked_sdist_installation_files(REPOSITORY)
        for required in (
            "cmake/xtbloom_cuda_toolkit.cmake",
            "cmake/3rdparty/torch-stable/aoti_symbols.txt",
            "cmake/3rdparty/eigen_manifest.json",
            "cmake/3rdparty/pyodide_openblas_manifest.json",
            "cmake/3rdparty/pyodide-openblas/recipe/libopenblas/meta.yaml",
            "data/parameters/d4.hpp",
            "data/parameters/gfn1.hpp",
            "data/parameters/gfn1_d3.hpp",
            "data/parameters/gfn2.hpp",
            "LICENSES/pyodide-MPL-2.0.txt",
            "src/backends/cuda/gfn2_scc_loop.cu",
            "tools/eigen_dependency.py",
        ):
            with self.subTest(required=required):
                self.assertIn(required, expected)
        for excluded in (
            "cmake/3rdparty/eigen/Eigen/Core",
            "cmake/3rdparty/eigen/manifest.json",
            "python/ci/repair-wheel.sh",
            "python/ci/resolve-pyodide-openblas.py",
            "python/ci/run-pyodide-wheel-test.py",
            "tests/runtime_test.cpp",
            "data/conformance/periodic/terms/manifest.json",
            "tools/check_benchmark_evidence_size.py",
            "tools/eigen_vendor.py",
            "tools/oracle/periodic_gfn2_terms/probe.f90",
            "uv.lock",
            "web/app.js",
        ):
            with self.subTest(excluded=excluded):
                self.assertNotIn(excluded, expected)

    def test_archive_bytes_and_modes_must_match_source(self) -> None:
        """Accept only byte-exact source with the tracked executable bit."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-sdist-test-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "build.sh").write_bytes(b"#!/bin/sh\n")
            archive = root / "xtbloom-0.1.0.tar.gz"
            info = tarfile.TarInfo("xtbloom-0.1.0/build.sh")
            info.mode = 0o755
            info.size = len(b"#!/bin/sh\n")
            with tarfile.open(archive, "w:gz") as tar:
                tar.addfile(info, io.BytesIO(b"#!/bin/sh\n"))
            CHECKER._check_sdist_archive_against_manifest(
                archive, source, {"build.sh": 0o755}
            )

    def test_archive_modified_bytes_are_rejected(self) -> None:
        """Reject an allowlisted path whose archived bytes were rewritten."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-sdist-test-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "source.cpp").write_bytes(b"source\n")
            archive = root / "xtbloom-0.1.0.tar.gz"
            info = tarfile.TarInfo("xtbloom-0.1.0/source.cpp")
            info.mode = 0o644
            info.size = len(b"modified\n")
            with tarfile.open(archive, "w:gz") as tar:
                tar.addfile(info, io.BytesIO(b"modified\n"))
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "tracked file bytes differ.*source.cpp"
            ):
                CHECKER._check_sdist_archive_against_manifest(
                    archive, source, {"source.cpp": 0o644}
                )

    def test_archive_mode_drift_is_rejected(self) -> None:
        """Reject an executable source generator made non-executable in tar."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-sdist-test-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "generator.py").write_bytes(b"#!/usr/bin/env python3\n")
            archive = root / "xtbloom-0.1.0.tar.gz"
            info = tarfile.TarInfo("xtbloom-0.1.0/generator.py")
            info.mode = 0o644
            payload = b"#!/usr/bin/env python3\n"
            info.size = len(payload)
            with tarfile.open(archive, "w:gz") as tar:
                tar.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "file mode differs.*generator.py"
            ):
                CHECKER._check_sdist_archive_against_manifest(
                    archive, source, {"generator.py": 0o755}
                )

    def test_archive_link_member_is_rejected(self) -> None:
        """Reject symlink entries before extracting an installation sdist."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-sdist-test-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            archive = root / "xtbloom-0.1.0.tar.gz"
            info = tarfile.TarInfo("xtbloom-0.1.0/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside"
            with tarfile.open(archive, "w:gz") as tar:
                tar.addfile(info)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "non-regular archive member.*link"
            ):
                CHECKER._check_sdist_archive_against_manifest(archive, source, {})

    def test_archive_normalized_duplicate_is_rejected(self) -> None:
        """Reject two tar names that normalize to the same payload path."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-sdist-test-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "source.cpp").write_bytes(b"source\n")
            archive = root / "xtbloom-0.1.0.tar.gz"
            payload = b"source\n"
            with tarfile.open(archive, "w:gz") as tar:
                for name in (
                    "xtbloom-0.1.0/source.cpp",
                    "xtbloom-0.1.0/./source.cpp",
                ):
                    info = tarfile.TarInfo(name)
                    info.mode = 0o644
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "duplicate archive member"
            ):
                CHECKER._check_sdist_archive_against_manifest(
                    archive, source, {"source.cpp": 0o644}
                )


class CanonicalByteCheckoutPolicyTests(unittest.TestCase):
    """Keep hash-pinned text stable across Git checkout platforms."""

    def test_hash_pinned_text_disables_checkout_conversion(self) -> None:
        """Prevent Windows autocrlf from invalidating provenance digests."""
        attributes = (REPOSITORY / ".gitattributes").read_text(encoding="utf-8")
        for expected in (
            "data/parameters/gfn1.toml whitespace=-blank-at-eof",
            "data/parameters/tblite_sto.hpp text eol=lf",
            "data/parameters/gfn1_legacy_sto.hpp text eol=lf",
            "LICENSES/Apache-2.0.txt -text",
            "LICENSES/LGPL-3.0-or-later.txt -text",
            "LICENSES/scipy-openblas32-0.3.34.0.0.txt -text",
            "LICENSES/openchemlib-BSD-3-Clause.txt -text",
            "LICENSES/pyodide-MPL-2.0.txt -text "
            "whitespace=-blank-at-eol,-blank-at-eof,-space-before-tab",
            "LICENSES/OpenBLAS-0.3.28-BSD-3-Clause.txt -text "
            "whitespace=-blank-at-eol,-blank-at-eof,-space-before-tab",
            "LICENSES/LAPACK-OpenBLAS-0.3.28-BSD-3-Clause.txt -text "
            "whitespace=-blank-at-eol,-blank-at-eof,-space-before-tab",
            "LICENSES/CLAPACK-3.2.1-BSD-3-Clause.txt -text "
            "whitespace=-blank-at-eol,-blank-at-eof,-space-before-tab",
            "LICENSES/libf2c-AT&T-Lucent-Bellcore.txt -text "
            "whitespace=-blank-at-eol,-blank-at-eof,-space-before-tab",
            "cmake/3rdparty/implib/** -text",
            "cmake/3rdparty/torch-stable/include/** -text",
            "cmake/3rdparty/pyodide-openblas/** -text "
            "whitespace=-blank-at-eol,-blank-at-eof,-space-before-tab",
            "LICENSES/eigen/** -text",
            "LICENSES/eigen/** -whitespace",
            "tools/oracle/tblite_scc_trace/tblite-e9abc395-scc-observer-v2.patch -text",
            "tools/oracle/tblite_scc_trace/scc_trace_main_v2.f90 -text",
            "tools/oracle/tblite_scc_trace/scc_trace_recorder_v2.f90 -text",
            "tools/oracle/tblite_scc_trace/xtbloom-scc-trace-v2.schema.json -text",
            "data/conformance/scc-traces/manifest-v2.json -text",
            "data/conformance/scc-traces/oh_radical.json -text",
            "data/conformance/scc-traces/specs/oh_radical.spec -text",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, attributes.splitlines())


class LicenseArchiveTests(unittest.TestCase):
    """Verify legal payload requirements for built distribution archives."""

    def _write_wheel(
        self, path: Path, names: set[str], overrides: dict[str, bytes] | None = None
    ) -> None:
        overrides = overrides or {}
        with zipfile.ZipFile(path, "w") as archive:
            for name in sorted(names):
                if name in overrides:
                    payload = overrides[name]
                elif name.endswith("/provenance/gfn1_d3_manifest.json"):
                    payload = (
                        REPOSITORY / "data/parameters/gfn1_d3_manifest.json"
                    ).read_bytes()
                elif name.endswith("/provenance/gfn1_legacy_sto_manifest.json"):
                    payload = (
                        REPOSITORY / "data/parameters/gfn1_legacy_sto_manifest.json"
                    ).read_bytes()
                elif name.endswith("/provenance/implib_manifest.json"):
                    payload = (REPOSITORY / CHECKER.IMPLIB_MANIFEST_PATH).read_bytes()
                elif name.endswith("/provenance/scipy_openblas32_manifest.json"):
                    payload = (REPOSITORY / CHECKER.OPENBLAS_MANIFEST_PATH).read_bytes()
                elif name.endswith("/provenance/pyodide_openblas_manifest.json"):
                    payload = (
                        REPOSITORY / CHECKER.PYODIDE_OPENBLAS_MANIFEST_PATH
                    ).read_bytes()
                elif name.endswith("scipy-openblas32-0.3.34.0.0.txt"):
                    payload = (REPOSITORY / CHECKER.OPENBLAS_LICENSE).read_bytes()
                elif name.endswith("scipy-openblas32-tools-LICENSE_win32.txt"):
                    payload = (
                        REPOSITORY / CHECKER.OPENBLAS_WINDOWS_LICENSE
                    ).read_bytes()
                elif any(
                    name.endswith(Path(relative).name)
                    for relative in CHECKER.OPENBLAS_EXACT_PACKAGED_LICENSES
                ):
                    relative = next(
                        relative
                        for relative in CHECKER.OPENBLAS_EXACT_PACKAGED_LICENSES
                        if name.endswith(Path(relative).name)
                    )
                    payload = (REPOSITORY / relative).read_bytes()
                elif any(
                    Path(name).name == Path(relative).name
                    for relative in CHECKER.PYODIDE_OPENBLAS_LICENSES
                ):
                    relative = next(
                        relative
                        for relative in CHECKER.PYODIDE_OPENBLAS_LICENSES
                        if Path(name).name == Path(relative).name
                    )
                    payload = (REPOSITORY / relative).read_bytes()
                elif name.endswith("/third-party/Apache-2.0.txt") or name.endswith(
                    "/LICENSES/Apache-2.0.txt"
                ):
                    payload = (REPOSITORY / "LICENSES/Apache-2.0.txt").read_bytes()
                elif name.endswith(
                    "/third-party/LGPL-3.0-or-later.txt"
                ) or name.endswith("/LICENSES/LGPL-3.0-or-later.txt"):
                    payload = (
                        REPOSITORY / "LICENSES/LGPL-3.0-or-later.txt"
                    ).read_bytes()
                elif name.endswith("/codspeed-MIT.txt"):
                    payload = (REPOSITORY / CHECKER.CODSPEED_LICENSE).read_bytes()
                else:
                    payload = b"test\n"
                archive.writestr(name, payload)

    def _valid_wheel_names(self) -> set[str]:
        names = {
            f"{WHEEL_DIST_INFO}/licenses/{suffix}"
            for suffix in CHECKER.COMMON_ARCHIVE_SUFFIXES
        } | {f"xtbloom/{suffix}" for suffix in CHECKER.WHEEL_ARCHIVE_SUFFIXES}
        manifest = CHECKER.json.loads(
            (REPOSITORY / CHECKER.OPENBLAS_MANIFEST_PATH).read_text(encoding="utf-8")
        )
        names.add("xtbloom/lib/libxtbloom_openblas_lp64_shim.so")
        for record in manifest["targets"]["linux-x86_64"]["files"]:
            source_name = Path(record["source"]).name
            names.add(
                "xtbloom.libs/"
                + CHECKER._auditwheel_name(source_name, record["sha256"])
            )
        return names

    def _valid_pyodide_wheel_names(self) -> set[str]:
        """Return the exact legal and three-module Pyodide wheel payload."""
        names = {
            f"{WHEEL_DIST_INFO}/licenses/{suffix}"
            for suffix in CHECKER.COMMON_ARCHIVE_SUFFIXES
        } | {f"xtbloom/{suffix}" for suffix in CHECKER.WHEEL_ARCHIVE_SUFFIXES}
        manifest = CHECKER.json.loads(
            (REPOSITORY / CHECKER.PYODIDE_OPENBLAS_MANIFEST_PATH).read_text(
                encoding="utf-8"
            )
        )
        names.update(
            {
                "xtbloom/lib/libxtbloom.so",
                "xtbloom/lib/" + manifest["artifact"]["adapter_install_name"],
                "xtbloom.libs/" + manifest["artifact"]["private_install_name"],
            }
        )
        return names

    def test_project_license_cannot_be_satisfied_by_third_party_filename(self) -> None:
        """Require the project license at its exact archive location."""
        names = self._valid_wheel_names()
        names.remove(f"{WHEEL_DIST_INFO}/licenses/LICENSE")
        names.add("xtbloom/share/licenses/xtbloom/third-party/d4/mctc-lib-LICENSE")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "LICENSE"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_every_provenance_manifest(self) -> None:
        """Require all provenance manifests in wheel payloads."""
        for filename in (
            "gfn1_manifest.json",
            "gfn1_d3_manifest.json",
            "gfn1_legacy_sto_manifest.json",
            "mctc_manifest.json",
        ):
            with self.subTest(filename=filename):
                names = self._valid_wheel_names()
                missing = f"xtbloom/share/licenses/xtbloom/provenance/{filename}"
                names.remove(missing)
                with tempfile.TemporaryDirectory(
                    prefix="xtbloom-license-test-"
                ) as directory:
                    wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
                    self._write_wheel(wheel, names)
                    with self.assertRaisesRegex(
                        CHECKER.LicenseCheckError, filename.split(".")[0]
                    ):
                        CHECKER.check_archive(wheel)

    def test_wheel_rejects_corrupt_gfn1_d3_provenance(self) -> None:
        """Validate archived D3 provenance bytes, not only its filename."""
        names = self._valid_wheel_names()
        manifest = "xtbloom/share/licenses/xtbloom/provenance/gfn1_d3_manifest.json"
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names, {manifest: b"{}\n"})
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "GFN1-D3"):
                CHECKER.check_archive(wheel)

    def test_wheel_rejects_changed_gfn1_d3_generated_metadata(self) -> None:
        """Enforce exact generator/output records in the distributed manifest."""
        names = self._valid_wheel_names()
        manifest_name = (
            "xtbloom/share/licenses/xtbloom/provenance/gfn1_d3_manifest.json"
        )
        manifest = json.loads(
            (REPOSITORY / "data/parameters/gfn1_d3_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        manifest["outputs"]["gfn1_d3.hpp"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(
                wheel,
                names,
                {manifest_name: (json.dumps(manifest) + "\n").encode("utf-8")},
            )
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "generated-output metadata"
            ):
                CHECKER.check_archive(wheel)

    def test_wheel_rejects_changed_gfn1_d3_lgpl_text(self) -> None:
        """Require exact upstream LGPL bytes in the wheel legal payload."""
        names = self._valid_wheel_names()
        lgpl_name = f"{WHEEL_DIST_INFO}/licenses/LICENSES/LGPL-3.0-or-later.txt"
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names, {lgpl_name: b"changed\n"})
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "LGPL license differs"
            ):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_implib_provenance_manifest(self) -> None:
        """Require the vendored implib provenance manifest in wheels."""
        names = self._valid_wheel_names()
        missing = "xtbloom/share/licenses/xtbloom/provenance/implib_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "implib_manifest"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_torch_stable_provenance_manifest(self) -> None:
        """Require the vendored LibTorch provenance manifest in wheels."""
        names = self._valid_wheel_names()
        missing = "xtbloom/share/licenses/xtbloom/provenance/torch_stable_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "torch_stable"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_linking_exception(self) -> None:
        """Require the GPLv3 Section 7 exception in wheel archives."""
        names = self._valid_wheel_names()
        missing = f"{WHEEL_DIST_INFO}/licenses/CUDA_MKL_LINKING_EXCEPTION"
        names.remove(missing)
        names.remove("xtbloom/share/licenses/xtbloom/CUDA_MKL_LINKING_EXCEPTION")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "CUDA_MKL_LINKING_EXCEPTION"
            ):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_array_api_compat_license(self) -> None:
        """Keep the new runtime dependency's distinct MIT grant in wheels."""
        names = self._valid_wheel_names()
        suffix = "LICENSES/array-api-compat-MIT.txt"
        names.remove(f"{WHEEL_DIST_INFO}/licenses/{suffix}")
        names.remove(
            "xtbloom/share/licenses/xtbloom/third-party/array-api-compat-MIT.txt"
        )
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "array-api-compat-MIT"
            ):
                CHECKER.check_archive(wheel)

    def test_wheel_rejects_changed_codspeed_license(self) -> None:
        """Require the exact reviewed plugin/Action notice in both wheel copies."""
        names = self._valid_wheel_names()
        member = f"{WHEEL_DIST_INFO}/licenses/{CHECKER.CODSPEED_LICENSE}"
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names, {member: b"changed\n"})
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "CodSpeed MIT license differs"
            ):
                CHECKER.check_archive(wheel)

    def test_wheel_must_not_bundle_vendor_library(self) -> None:
        """Reject wheels that bundle separately licensed vendor libraries."""
        names = self._valid_wheel_names()
        names.add("xtbloom/lib/libcudart.so.12")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "libcudart"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_not_bundle_relocated_eigen_headers(self) -> None:
        """Reject Eigen even when packaging moves it outside the vendor path."""
        names = self._valid_wheel_names()
        names.add("xtbloom/include/eigen3/Eigen/Core")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "Web-only Eigen"):
                CHECKER.check_archive(wheel)

    def test_wheel_requires_complete_private_openblas_cohort(self) -> None:
        """Reject a wheel that loses one auditwheel-vendored support DSO."""
        names = self._valid_wheel_names()
        missing = next(name for name in names if "libquadmath" in name)
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "cohort differs"):
                CHECKER.check_archive(wheel)

    def test_wheel_rejects_unreviewed_openblas_binary(self) -> None:
        """Require dependency re-audit before the vendored ELF set expands."""
        names = self._valid_wheel_names()
        names.add("xtbloom.libs/libgfortran-unreviewed.so.5")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "cohort differs"):
                CHECKER.check_archive(wheel)

    def test_native_wheel_rejects_renamed_wasm_shared_module(self) -> None:
        """Detect WebAssembly by magic instead of trusting provider basenames."""
        names = self._valid_wheel_names()
        renamed = "xtbloom.libs/libinnocent_math.so"
        names.add(renamed)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(
                wheel,
                names,
                {renamed: CHECKER.WASM_V1_MAGIC + b"renamed provider"},
            )
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "native wheel.*WebAssembly"
            ):
                CHECKER.check_archive(wheel)

    def test_native_wheel_rejects_wasm_with_non_shared_extension(self) -> None:
        """Detect WebAssembly payloads even when they use a plain .wasm suffix."""
        names = self._valid_wheel_names()
        renamed = "xtbloom.libs/innocent-math.wasm"
        names.add(renamed)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(
                wheel,
                names,
                {renamed: CHECKER.WASM_V1_MAGIC + b"renamed provider"},
            )
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "native wheel.*WebAssembly"
            ):
                CHECKER.check_archive(wheel)

    def test_pyodide_wheel_rejects_shadowed_provider_basename(self) -> None:
        """Require exact module paths, not only a set of matching basenames."""
        names = self._valid_pyodide_wheel_names()
        provider = next(name for name in names if name.startswith("xtbloom.libs/"))
        shadow = "shadow/" + Path(provider).name
        names.add(shadow)
        wasm = {
            name: CHECKER.WASM_V1_MAGIC + b"module"
            for name in names
            if name
            in {
                "xtbloom/lib/libxtbloom.so",
                provider,
                shadow,
            }
            or name.endswith("libxtbloom_pyodide_lapacke.so")
        }
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-pyodide_2026_0_wasm32.whl"
            self._write_wheel(wheel, names, wasm)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "cohort differs"):
                CHECKER.check_archive(wheel)

    def test_wheel_rejects_changed_target_specific_openblas_license(self) -> None:
        """Require exact upstream packaged-license bytes for every target."""
        names = self._valid_wheel_names()
        suffix = Path(CHECKER.OPENBLAS_EXACT_PACKAGED_LICENSES[0]).name
        member = "xtbloom/share/licenses/xtbloom/third-party/" + suffix
        self.assertIn(member, names)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test-manylinux_2_28_x86_64.whl"
            self._write_wheel(wheel, names, {member: b"changed license\n"})
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "exact packaged license differs"
            ):
                CHECKER.check_archive(wheel)


class OpenBlasProvenanceTests(unittest.TestCase):
    """Pin every provenance locator for the redistributed wheel inputs."""

    def setUp(self) -> None:
        """Load a fresh manifest for each negative mutation."""
        self.manifest = CHECKER.json.loads(
            (REPOSITORY / CHECKER.OPENBLAS_MANIFEST_PATH).read_text(encoding="utf-8")
        )

    def test_current_openblas_manifest_is_accepted(self) -> None:
        """Accept the exact reviewed repositories, wheels, and native cohorts."""
        CHECKER._check_openblas_manifest(self.manifest)

    def test_openblas_manifest_rejects_changed_repository(self) -> None:
        """Do not let a provenance URL drift while payload checks stay green."""
        self.manifest["source"]["repository"] = "https://example.invalid/openblas"
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "repository"):
            CHECKER._check_openblas_manifest(self.manifest)

    def test_openblas_manifest_rejects_changed_wheel_url(self) -> None:
        """Keep the immutable PyPI artifact locator tied to reviewed bytes."""
        self.manifest["targets"]["linux-x86_64"]["wheel"]["url"] = (
            "https://example.invalid/scipy-openblas32.whl"
        )
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "wheel differs"):
            CHECKER._check_openblas_manifest(self.manifest)

    def test_openblas_manifest_rejects_macos_dependency_outside_cohort(self) -> None:
        """Keep every rewritten Mach-O dependency on a declared private image."""
        provider = self.manifest["targets"]["macos-arm64"]["files"][0]
        provider["load_rewrites"][0]["to"] = "@loader_path/libhost-gfortran.dylib"
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "escapes cohort"):
            CHECKER._check_openblas_manifest(self.manifest)

    def test_openblas_manifest_rejects_generic_windows_provider_name(self) -> None:
        """Prevent Windows loader reuse through a non-private provider basename."""
        target = self.manifest["targets"]["windows-amd64"]
        target["files"][0]["install_name"] = "libscipy_openblas.dll"
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "PE private mapping"):
            CHECKER._check_openblas_manifest(self.manifest)


class PyodideOpenBlasProvenanceTests(unittest.TestCase):
    """Require an exact retained Pyodide recipe and legal payload."""

    def _copy_payload(self, root: Path) -> None:
        manifest = root / CHECKER.PYODIDE_OPENBLAS_MANIFEST_PATH
        manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / CHECKER.PYODIDE_OPENBLAS_MANIFEST_PATH, manifest)
        shutil.copytree(
            REPOSITORY / CHECKER.PYODIDE_OPENBLAS_RECIPE_PATH,
            root / CHECKER.PYODIDE_OPENBLAS_RECIPE_PATH,
        )
        for relative in CHECKER.PYODIDE_OPENBLAS_LICENSES:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY / relative, destination)

    def _write_sdist(
        self,
        archive_path: Path,
        *,
        extra_recipe: bool = False,
        recipe_symlink: bool = False,
    ) -> None:
        root = "xtbloom-0.1.0"
        manifest = json.loads(
            (REPOSITORY / CHECKER.PYODIDE_OPENBLAS_MANIFEST_PATH).read_text(
                encoding="utf-8"
            )
        )
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(
                REPOSITORY / CHECKER.PYODIDE_OPENBLAS_MANIFEST_PATH,
                arcname=f"{root}/{CHECKER.PYODIDE_OPENBLAS_MANIFEST_PATH}",
            )
            for record in (*manifest["recipe_files"], *manifest["licenses"]):
                archive.add(
                    REPOSITORY / record["local"],
                    arcname=f"{root}/{record['local']}",
                )
            if extra_recipe:
                payload = b"unexpected\n"
                info = tarfile.TarInfo(
                    f"{root}/{CHECKER.PYODIDE_OPENBLAS_RECIPE_PATH}/unexpected.patch"
                )
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            if recipe_symlink:
                info = tarfile.TarInfo(
                    f"{root}/{CHECKER.PYODIDE_OPENBLAS_RECIPE_PATH}/shadow.patch"
                )
                info.type = tarfile.SYMTYPE
                info.linkname = "unexpected.patch"
                archive.addfile(info)

    def test_current_pyodide_openblas_payload_is_accepted(self) -> None:
        """Accept the exact manifest, recipes, and license bytes in the source tree."""
        CHECKER._check_pyodide_openblas_provenance(REPOSITORY)

    def test_source_rejects_unreviewed_recipe_file(self) -> None:
        """Do not let the broad sdist include wildcard expand the audited tree."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-pyodide-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            unexpected = (
                root
                / CHECKER.PYODIDE_OPENBLAS_RECIPE_PATH
                / "libopenblas"
                / "unexpected.patch"
            )
            unexpected.write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "unexpected.*unexpected.patch"
            ):
                CHECKER._check_pyodide_openblas_provenance(root)

    def test_sdist_rejects_unreviewed_recipe_file(self) -> None:
        """Require the archived recipe file set to equal the source manifest."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-pyodide-test-") as directory:
            archive = Path(directory) / "xtbloom-0.1.0.tar.gz"
            self._write_sdist(archive, extra_recipe=True)
            names = CHECKER._archive_names(archive)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "recipe file set differs.*unexpected"
            ):
                CHECKER._check_archived_pyodide_openblas(archive, names, wheel=False)

    def test_sdist_rejects_recipe_symlink(self) -> None:
        """Do not accept archive links in the provenance-pinned recipe tree."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-pyodide-test-") as directory:
            archive = Path(directory) / "xtbloom-0.1.0.tar.gz"
            self._write_sdist(archive, recipe_symlink=True)
            names = CHECKER._archive_names(archive)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "not a regular file.*shadow.patch"
            ):
                CHECKER._check_archived_pyodide_openblas(archive, names, wheel=False)


class InstallPayloadTests(unittest.TestCase):
    """Keep wheel-only provider binaries out of native CMake installs."""

    def _write_required_install_files(self, root: Path) -> None:
        """Create a minimal install tree with authentic reviewed D3 provenance."""
        for relative in CHECKER.INSTALL_FILES:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative.endswith("provenance/gfn1_d3_manifest.json"):
                destination.write_bytes(
                    (REPOSITORY / "data/parameters/gfn1_d3_manifest.json").read_bytes()
                )
            elif relative.endswith("provenance/gfn1_legacy_sto_manifest.json"):
                destination.write_bytes(
                    (
                        REPOSITORY / "data/parameters/gfn1_legacy_sto_manifest.json"
                    ).read_bytes()
                )
            elif relative.endswith("third-party/Apache-2.0.txt"):
                destination.write_bytes(
                    (REPOSITORY / "LICENSES/Apache-2.0.txt").read_bytes()
                )
            elif relative.endswith("third-party/LGPL-3.0-or-later.txt"):
                destination.write_bytes(
                    (REPOSITORY / "LICENSES/LGPL-3.0-or-later.txt").read_bytes()
                )
            elif relative.endswith("third-party/codspeed-MIT.txt"):
                destination.write_bytes(
                    (REPOSITORY / CHECKER.CODSPEED_LICENSE).read_bytes()
                )
            else:
                destination.write_bytes(b"test\n")

    def test_install_rejects_unrenamed_desktop_openblas_cohort(self) -> None:
        """Recognize upstream macOS/Windows provider and support filenames."""
        candidates = (
            "lib/libscipy_openblas.dylib",
            "lib/libgfortran.5.dylib",
            "lib/libquadmath.0.dylib",
            "lib/libgcc_s.1.1.dylib",
            "bin/libscipy_openblas.dll",
            "bin/scipy_openblas.dll",
        )
        for candidate in candidates:
            with (
                self.subTest(candidate=candidate),
                tempfile.TemporaryDirectory(
                    prefix="xtbloom-install-license-test-"
                ) as directory,
            ):
                root = Path(directory)
                self._write_required_install_files(root)
                bundled = root / candidate
                bundled.parent.mkdir(parents=True, exist_ok=True)
                bundled.write_bytes(b"native provider")
                with self.assertRaisesRegex(
                    CHECKER.LicenseCheckError, "wheel-only OpenBLAS"
                ):
                    CHECKER.check_install(root)

    def test_native_install_rejects_renamed_wasm_shared_module(self) -> None:
        """Reject WebAssembly side modules regardless of their basename."""
        with tempfile.TemporaryDirectory(
            prefix="xtbloom-install-license-test-"
        ) as directory:
            root = Path(directory)
            self._write_required_install_files(root)
            renamed = root / "lib" / "libinnocent_math.so"
            renamed.parent.mkdir(parents=True, exist_ok=True)
            renamed.write_bytes(CHECKER.WASM_V1_MAGIC + b"renamed provider")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "WebAssembly shared module"
            ):
                CHECKER.check_install(root)

    def test_install_rejects_relocated_eigen_source(self) -> None:
        """Keep the browser-only headers and manifest out of native installs."""
        for candidate in (
            "include/eigen3/Eigen/Core",
            "share/xtbloom/provenance/eigen_manifest.json",
            "share/licenses/xtbloom/third-party/eigen/COPYING.MPL2",
        ):
            with (
                self.subTest(candidate=candidate),
                tempfile.TemporaryDirectory(
                    prefix="xtbloom-install-license-test-"
                ) as directory,
            ):
                root = Path(directory)
                self._write_required_install_files(root)
                bundled = root / candidate
                bundled.parent.mkdir(parents=True, exist_ok=True)
                bundled.write_bytes(b"Web-only Eigen payload")
                with self.assertRaisesRegex(
                    CHECKER.LicenseCheckError, "Web-only Eigen"
                ):
                    CHECKER.check_install(root)

    def test_install_rejects_changed_codspeed_license(self) -> None:
        """Require the exact reviewed CodSpeed notice in native installs."""
        with tempfile.TemporaryDirectory(
            prefix="xtbloom-install-license-test-"
        ) as directory:
            root = Path(directory)
            self._write_required_install_files(root)
            license_path = root / "share/licenses/xtbloom/third-party/codspeed-MIT.txt"
            license_path.write_bytes(b"changed\n")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "CodSpeed MIT license differs"
            ):
                CHECKER.check_install(root)

    def test_install_rejects_corrupt_gfn1_d3_provenance(self) -> None:
        """Validate the installed D3 provenance manifest contents."""
        with tempfile.TemporaryDirectory(
            prefix="xtbloom-install-license-test-"
        ) as directory:
            root = Path(directory)
            self._write_required_install_files(root)
            manifest = root / "share/licenses/xtbloom/provenance/gfn1_d3_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "GFN1-D3"):
                CHECKER.check_install(root)

    def test_install_maps_malformed_gfn1_d3_manifest_to_checker_error(self) -> None:
        """Keep malformed installed JSON and UTF-8 on the controlled error path."""
        for content in (b"{\n", b"\xff\n"):
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory(
                    prefix="xtbloom-install-license-test-"
                ) as directory,
            ):
                root = Path(directory)
                self._write_required_install_files(root)
                manifest = (
                    root / "share/licenses/xtbloom/provenance/gfn1_d3_manifest.json"
                )
                manifest.write_bytes(content)
                with self.assertRaisesRegex(
                    CHECKER.LicenseCheckError,
                    "installed GFN1-D3 manifest is malformed",
                ):
                    CHECKER.check_install(root)

    def test_install_rejects_changed_gfn1_d3_lgpl_text(self) -> None:
        """Require exact upstream LGPL bytes in native install trees."""
        with tempfile.TemporaryDirectory(
            prefix="xtbloom-install-license-test-"
        ) as directory:
            root = Path(directory)
            self._write_required_install_files(root)
            lgpl = root / "share/licenses/xtbloom/third-party/LGPL-3.0-or-later.txt"
            lgpl.write_bytes(b"changed\n")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "LGPL license differs"
            ):
                CHECKER.check_install(root)


class CodSpeedDependencySourceTests(unittest.TestCase):
    """Keep CI-only CodSpeed inputs and their retained notice fully pinned."""

    def test_codspeed_notice_record_is_complete(self) -> None:
        """Accept all reviewed revisions and downloaded artifact hashes."""
        notice = (REPOSITORY / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        CHECKER._require_notice_tokens(notice, CHECKER.CODSPEED_NOTICE_TOKENS)

    def test_codspeed_notice_rejects_every_omitted_record(self) -> None:
        """Reject removal of a plugin, Action, runner, or Valgrind locator."""
        notice = (REPOSITORY / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for token in CHECKER.CODSPEED_NOTICE_TOKENS:
            with self.subTest(token=token):
                changed = notice.replace(token, "", 1)
                with self.assertRaisesRegex(
                    CHECKER.LicenseCheckError, "THIRD_PARTY_NOTICES.md omits"
                ):
                    CHECKER._require_notice_tokens(
                        changed, CHECKER.CODSPEED_NOTICE_TOKENS
                    )

    def test_codspeed_license_is_exact(self) -> None:
        """Accept only the byte-exact shared MIT notice from upstream."""
        payload = (REPOSITORY / CHECKER.CODSPEED_LICENSE).read_bytes()
        CHECKER._require_codspeed_license_bytes(payload, "source tree")
        with self.assertRaisesRegex(
            CHECKER.LicenseCheckError, "CodSpeed MIT license differs"
        ):
            CHECKER._require_codspeed_license_bytes(payload + b"changed\n", "test")


class WebDependencySourceTests(unittest.TestCase):
    """Keep CI-only browser tooling and deployed GFN1 provenance fully pinned."""

    def test_playwright_notice_record_is_complete(self) -> None:
        """Accept the complete reviewed package, tarball, and browser record."""
        notice = (REPOSITORY / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        CHECKER._require_notice_tokens(notice, CHECKER.PLAYWRIGHT_NOTICE_TOKENS)

    def test_playwright_notice_rejects_every_omitted_record(self) -> None:
        """Reject removal of any package, tarball, browser, or codec locator."""
        notice = (REPOSITORY / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for token in CHECKER.PLAYWRIGHT_NOTICE_TOKENS:
            with self.subTest(token=token):
                changed = notice.replace(token, "", 1)
                with self.assertRaisesRegex(
                    CHECKER.LicenseCheckError, "THIRD_PARTY_NOTICES.md omits"
                ):
                    CHECKER._require_notice_tokens(
                        changed, CHECKER.PLAYWRIGHT_NOTICE_TOKENS
                    )

    def test_gfn1_web_source_map_is_complete(self) -> None:
        """Accept exact source mappings for all three deployed GFN1 manifests."""
        CHECKER._require_gfn1_web_source_map()

    def test_gfn1_web_source_map_rejects_omissions_and_changes(self) -> None:
        """Do not let a deployed GFN1 manifest lose its byte-exact source owner."""
        for site_relative in CHECKER.REQUIRED_GFN1_WEB_SOURCE_MAP:
            with (
                self.subTest(site_relative=site_relative, mutation="missing"),
                mock.patch.dict(CHECKER.WEB_SITE_SOURCE_MAP),
            ):
                CHECKER.WEB_SITE_SOURCE_MAP.pop(site_relative)
                with self.assertRaisesRegex(
                    CHECKER.LicenseCheckError, "required GFN1 provenance"
                ):
                    CHECKER._require_gfn1_web_source_map()
            with (
                self.subTest(site_relative=site_relative, mutation="changed"),
                mock.patch.dict(
                    CHECKER.WEB_SITE_SOURCE_MAP,
                    {site_relative: "data/parameters/wrong-manifest.json"},
                ),
                self.assertRaisesRegex(
                    CHECKER.LicenseCheckError, "required GFN1 provenance"
                ),
            ):
                CHECKER._require_gfn1_web_source_map()


class WebSiteLicenseTests(unittest.TestCase):
    """Require the Pages artifact to retain its complete legal boundary."""

    def _write_valid_site(self, root: Path) -> None:
        for site_relative, source_relative in CHECKER.WEB_SITE_SOURCE_MAP.items():
            destination = root / site_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY / source_relative, destination)
        for relative in CHECKER.WEB_SITE_RUNTIME_FILES:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_threedmol = REPOSITORY / "web/node_modules/3dmol/build/3Dmol-min.js"
            if relative == "vendor/3Dmol-min.js" and source_threedmol.is_file():
                shutil.copy2(source_threedmol, destination)
            else:
                destination.write_bytes(b"test\n")
        entries = []
        version_material = ""
        for asset_id, relative in CHECKER.WEB_VERSIONED_ASSETS:
            payload = (root / relative).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            entries.append(
                {
                    "id": asset_id,
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": digest,
                }
            )
            version_material += f"{asset_id}:{relative}:{len(payload)}:{digest}\n"
        (root / "engine-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "version": hashlib.sha256(version_material.encode()).hexdigest(),
                    "assets": entries,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "index.html").write_text(
            '<a href="LICENSE">license</a>\n'
            '<a href="THIRD_PARTY_NOTICES.md">notices</a>\n'
            '<a href="LICENSES/openchemlib-BSD-3-Clause.txt">OpenChemLib</a>\n'
            '<a href="LICENSES/eigen/COPYING.MPL2">Eigen</a>\n'
            '<a href="provenance/eigen_manifest.json">Eigen source</a>\n'
            '<a href="CUDA_MKL_LINKING_EXCEPTION">permission</a>\n'
            '<a href="https://xtbloom.jinzhezeng.group">demo</a>\n'
            '<a href="https://github.com/jinzhezenggroup/xtbloom">source</a>\n',
            encoding="utf-8",
        )

    def test_complete_web_site_payload_is_accepted(self) -> None:
        """Accept exact source legal bytes beside the deployed runtime."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_requires_project_license(self) -> None:
        """The GPL-covered WASM cannot be deployed without the project grant."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "LICENSE").unlink()
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "LICENSE"):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_requires_pako_zlib_notice(self) -> None:
        """Keep the non-MIT zlib grant for code inside the 3Dmol bundle."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "LICENSES/pako-Zlib.txt").unlink()
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "pako-Zlib"):
                CHECKER.check_web_site(root, REPOSITORY)

    @unittest.skipUnless(
        (REPOSITORY / "web/node_modules/3dmol/build/3Dmol-min.js").is_file(),
        "requires the pinned npm build input",
    )
    def test_web_site_rejects_changed_3dmol_fallback(self) -> None:
        """Require the deployed local fallback to match the reviewed CDN bytes."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "vendor/3Dmol-min.js").write_bytes(b"changed\n")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "3Dmol"):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_requires_torch_bsd_license(self) -> None:
        """Keep the vendored LibTorch headers' BSD grant in Pages payloads."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "LICENSES/BSD-3-Clause.txt").unlink()
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "BSD-3-Clause"):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_requires_openchemlib_license(self) -> None:
        """Retain the BSD grant for the runtime-provided SMILES dependency."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / CHECKER.OPEN_CHEMLIB_LICENSE).unlink()
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "openchemlib"):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_requires_openchemlib_provenance(self) -> None:
        """Retain exact CDN hashes and revisions beside the deployed adapter."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "provenance/openchemlib_manifest.json").unlink()
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "openchemlib"):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_requires_eigen_license_and_provenance(self) -> None:
        """Retain Eigen's MPL source offer and exact acquisition manifest."""
        for relative in (
            "LICENSES/eigen/COPYING.MPL2",
            "LICENSES/eigen/notices/AlignedBox.h",
            "provenance/eigen_manifest.json",
        ):
            with (
                self.subTest(relative=relative),
                tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory,
            ):
                root = Path(directory)
                self._write_valid_site(root)
                (root / relative).unlink()
                with self.assertRaisesRegex(CHECKER.LicenseCheckError, "Eigen|eigen"):
                    CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_requires_all_gfn1_parameter_provenance(self) -> None:
        """Ship every GFN1 parameter source manifest exposed by the method UI."""
        for relative in CHECKER.REQUIRED_GFN1_WEB_SOURCE_MAP:
            with (
                self.subTest(relative=relative),
                tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory,
            ):
                root = Path(directory)
                self._write_valid_site(root)
                (root / relative).unlink()
                with self.assertRaisesRegex(CHECKER.LicenseCheckError, "gfn1|GFN1"):
                    CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_rejects_changed_gfn1_parameter_provenance(self) -> None:
        """Reject deployed GFN1 provenance that no longer matches its source."""
        for relative in CHECKER.REQUIRED_GFN1_WEB_SOURCE_MAP:
            with (
                self.subTest(relative=relative),
                tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory,
            ):
                root = Path(directory)
                self._write_valid_site(root)
                path = root / relative
                path.write_bytes(path.read_bytes() + b"\nchanged\n")
                with self.assertRaisesRegex(
                    CHECKER.LicenseCheckError, "legal file differs from source"
                ):
                    CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_rejects_raw_lapack_side_module(self) -> None:
        """Do not deploy a second untracked copy of the preloaded side module."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "libscipy_openblas.so").write_bytes(b"unexpected raw side module")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "raw Eigen LAPACKE/CBLAS side module"
            ):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_rejects_arbitrary_stale_artifact(self) -> None:
        """Reject obsolete engine variants because Pages uploads every file."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "xtbloom_web-old.wasm").write_bytes(b"stale engine")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "unexpected or orphaned files"
            ):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_rejects_legacy_data_payload(self) -> None:
        """Do not retain the MIME-opaque preload URL beside its Wasm replacement."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "xtbloom_web.data").write_bytes(b"legacy preload package")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "unexpected or orphaned files"
            ):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_rejects_stale_engine_manifest(self) -> None:
        """Do not let cached URLs describe different JS/WASM/data bytes."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "xtbloom_web.wasm").write_bytes(b"changed engine")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "manifest does not match"
            ):
                CHECKER.check_web_site(root, REPOSITORY)


class DependencyPolicyTests(unittest.TestCase):
    """Verify mandatory and optional dependency licensing boundaries."""

    def setUp(self) -> None:
        """Load the project metadata used by each dependency-policy test."""
        metadata = CHECKER.tomllib.loads(
            (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.metadata = metadata
        self.project = metadata["project"]

    def test_current_dependency_policy_is_accepted(self) -> None:
        """Accept the repository's reviewed dependency policy."""
        CHECKER._require_dependency_policy(self.project)
        CHECKER._require_openblas_build_policy(self.metadata)

    def test_array_api_compat_must_use_reviewed_range(self) -> None:
        """Require the provenance-reviewed runtime dependency range."""
        project = copy.deepcopy(self.project)
        project["dependencies"] = [
            requirement.replace(">=1.15,<2", ">=1")
            if requirement.startswith("array-api-compat")
            else requirement
            for requirement in project["dependencies"]
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "reviewed"):
            CHECKER._require_dependency_policy(project)

    def test_mkl_cannot_be_mandatory(self) -> None:
        """Reject MKL when added to mandatory dependencies."""
        project = copy.deepcopy(self.project)
        project["dependencies"].append("mkl; sys_platform == 'linux'")
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "mkl must not"):
            CHECKER._require_dependency_policy(project)

    def test_nvidia_provider_cannot_be_mandatory(self) -> None:
        """Reject NVIDIA providers outside the CUDA optional extra."""
        project = copy.deepcopy(self.project)
        project["dependencies"].append(project["optional-dependencies"]["cuda12"][0])
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "confined to"):
            CHECKER._require_dependency_policy(project)

    def test_openblas_cannot_be_a_runtime_dependency(self) -> None:
        """Reject the upstream build artifact from published requirements."""
        project = copy.deepcopy(self.project)
        project["dependencies"].append(
            "scipy-openblas32==0.3.34.0.0; sys_platform == 'linux'"
        )
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "must not be a runtime"):
            CHECKER._require_dependency_policy(project)

    def test_openblas_build_input_must_use_reviewed_exact_version(self) -> None:
        """Require the exact provider ABI in both build-only declarations."""
        metadata = copy.deepcopy(self.metadata)
        metadata["dependency-groups"]["wheel-build"][0] = metadata["dependency-groups"][
            "wheel-build"
        ][0].replace("==0.3.34.0.0", ">=0.3.34.0.0")
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "reviewed"):
            CHECKER._require_openblas_build_policy(metadata)

    def test_openblas_build_input_must_be_wheel_only(self) -> None:
        """Reject a build override that would also install the provider for sdists."""
        metadata = copy.deepcopy(self.metadata)
        override = metadata["tool"]["scikit-build"]["overrides"][0]
        override["if"]["state"] = ".*"
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "confined"):
            CHECKER._require_openblas_build_policy(metadata)

    def test_cuda_extra_must_be_complete(self) -> None:
        """Require the complete reviewed CUDA provider set."""
        project = copy.deepcopy(self.project)
        project["optional-dependencies"]["cuda12"].pop()
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "reviewed NVIDIA"):
            CHECKER._require_dependency_policy(project)


class BuildDependencyPolicyTests(unittest.TestCase):
    """Keep direct isolated build requirements compatible and reviewed."""

    def setUp(self) -> None:
        """Load the build-system table used by each policy mutation."""
        metadata = CHECKER.tomllib.loads(
            (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.build_system = metadata["build-system"]

    def test_current_build_dependency_policy_is_accepted(self) -> None:
        """Accept the two reviewed direct requirements with lower bounds."""
        CHECKER._require_build_dependency_policy(self.build_system)

    def test_build_dependency_lower_bound_must_not_be_weakened(self) -> None:
        """Reject a scikit-build-core release lacking the provider API."""
        build_system = copy.deepcopy(self.build_system)
        build_system["requires"] = [
            requirement.replace(">=1.0.3", ">=1.0.2")
            if requirement.startswith("scikit-build-core")
            else requirement
            for requirement in build_system["requires"]
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "direct compatible"):
            CHECKER._require_build_dependency_policy(build_system)

    def test_build_dependency_must_not_be_exactly_pinned(self) -> None:
        """Do not freeze the user's isolated build environment to one release."""
        build_system = copy.deepcopy(self.build_system)
        build_system["requires"] = [
            requirement.replace(">=1.0.3", "==1.0.3")
            if requirement.startswith("scikit-build-core")
            else requirement
            for requirement in build_system["requires"]
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "direct compatible"):
            CHECKER._require_build_dependency_policy(build_system)

    def test_direct_build_dependency_set_must_remain_complete(self) -> None:
        """Reject omission of the direct Git metadata provider."""
        build_system = copy.deepcopy(self.build_system)
        build_system["requires"] = [
            requirement
            for requirement in build_system["requires"]
            if not requirement.startswith("setuptools-scm")
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "direct compatible"):
            CHECKER._require_build_dependency_policy(build_system)

    def test_transitive_build_dependency_is_not_declared_directly(self) -> None:
        """Keep backend implementation dependencies out of the user contract."""
        build_system = copy.deepcopy(self.build_system)
        build_system["requires"].append("packaging==26.3")
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "direct compatible"):
            CHECKER._require_build_dependency_policy(build_system)


class VersionMetadataPolicyTests(unittest.TestCase):
    """Keep Python SCM metadata and native release tags strictly configured."""

    def setUp(self) -> None:
        """Load the complete project metadata used by each policy mutation."""
        self.metadata = CHECKER.tomllib.loads(
            (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        )

    def test_current_version_metadata_policy_is_accepted(self) -> None:
        """Accept the reviewed dynamic provider and strict tag grammar."""
        CHECKER._require_version_metadata_policy(self.metadata)

    def test_current_git_archival_policy_is_accepted(self) -> None:
        """Retain revision metadata from which native CMake recovers the tag."""
        CHECKER._require_git_archival_policy(REPOSITORY)

    def test_git_archival_cannot_drop_commit_distance(self) -> None:
        """Reject archives that collapse every post-tag Python build to the tag."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-archive-test-") as directory:
            root = Path(directory)
            archival = (REPOSITORY / ".git_archival.txt").read_text(encoding="utf-8")
            (root / ".git_archival.txt").write_text(
                archival.replace(
                    "describe-name: $Format:%(describe:tags=true,abbrev=40,match=v*)$",
                    "describe-name: $Format:%(describe:tags=true,abbrev=0,match=v*)$",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "revision-aware Python metadata"
            ):
                CHECKER._require_git_archival_policy(root)

    def test_expanded_git_archival_policy_is_accepted(self) -> None:
        """Accept the same reviewed fields after ``git archive`` expands them."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-archive-test-") as directory:
            root = Path(directory)
            (root / ".git_archival.txt").write_text(
                "node: 0123456789abcdef0123456789abcdef01234567\n"
                "node-date: 2026-08-10T22:14:05+08:00\n"
                "describe-name: v1.2.3-7-g0123456789abcdef0123456789abcdef01234567\n",
                encoding="utf-8",
            )
            CHECKER._require_git_archival_policy(root)

    def test_expanded_git_archival_requires_matching_node(self) -> None:
        """Reject describe metadata whose object ID is unrelated to the node."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-archive-test-") as directory:
            root = Path(directory)
            (root / ".git_archival.txt").write_text(
                "node: 0123456789abcdef0123456789abcdef01234567\n"
                "node-date: 2026-08-10T22:14:05+08:00\n"
                "describe-name: v1.2.3-7-gabcdef0123456789abcdef0123456789abcdef01\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "revision-aware Python metadata"
            ):
                CHECKER._require_git_archival_policy(root)

    def test_static_project_version_is_rejected(self) -> None:
        """Reject reintroduction of a hand-maintained project version."""
        metadata = copy.deepcopy(self.metadata)
        metadata["project"]["version"] = "0.0.0"
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "exclusively dynamic"):
            CHECKER._require_version_metadata_policy(metadata)

    def test_usable_fallback_version_is_rejected(self) -> None:
        """Reject silent version synthesis when Git/archive metadata is absent."""
        metadata = copy.deepcopy(self.metadata)
        metadata["tool"]["setuptools_scm"]["fallback_version"] = "0.0.0"
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "strict Git-tag"):
            CHECKER._require_version_metadata_policy(metadata)

    def test_revision_distance_cannot_be_disabled(self) -> None:
        """Keep branch wheels distinguishable after the nearest release tag."""
        metadata = copy.deepcopy(self.metadata)
        metadata["tool"]["setuptools_scm"]["version_scheme"] = "only-version"
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "strict Git-tag"):
            CHECKER._require_version_metadata_policy(metadata)

    def test_local_git_identity_cannot_be_disabled(self) -> None:
        """Retain the source commit identity on non-release Python artifacts."""
        metadata = copy.deepcopy(self.metadata)
        metadata["tool"]["setuptools_scm"]["local_scheme"] = "no-local-version"
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "strict Git-tag"):
            CHECKER._require_version_metadata_policy(metadata)

    def test_loose_tag_regex_is_rejected(self) -> None:
        """Reject tags outside the exact vMAJOR.MINOR.PATCH grammar."""
        metadata = copy.deepcopy(self.metadata)
        metadata["tool"]["setuptools_scm"]["tag"]["regex"] = r"^(?P<version>.+)$"
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "strict Git-tag"):
            CHECKER._require_version_metadata_policy(metadata)

    def test_describe_must_not_skip_malformed_v_tags(self) -> None:
        """Reserve the full v* namespace so malformed newer tags fail."""
        metadata = copy.deepcopy(self.metadata)
        metadata["tool"]["setuptools_scm"]["scm"]["git"]["describe_command"][-1] = (
            "v[0-9]*.[0-9]*.[0-9]*"
        )
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "strict Git-tag"):
            CHECKER._require_version_metadata_policy(metadata)

    def test_custom_provider_cannot_replace_builtin_plugin(self) -> None:
        """Keep version resolution on scikit-build-core's supported plugin."""
        metadata = copy.deepcopy(self.metadata)
        metadata["tool"]["dynamic-metadata"] = [
            {
                "provider": {
                    "path": "python/ci",
                    "module": "custom_version_provider",
                }
            }
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unreviewed"):
            CHECKER._require_version_metadata_policy(metadata)


class LinkingExceptionTests(unittest.TestCase):
    """Verify the additional-permission source and exception policy."""

    def test_current_exception_policy_is_accepted(self) -> None:
        """Accept the current source notices and linking exception."""
        CHECKER._require_exception_policy(REPOSITORY)

    def test_cudadevrt_cannot_be_added_without_review(self) -> None:
        """Reject an exception document that newly covers cudadevrt."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "src", root / "src")
            shutil.copytree(REPOSITORY / "include", root / "include")
            text = (REPOSITORY / CHECKER.EXCEPTION_FILE).read_text(encoding="utf-8")
            (root / CHECKER.EXCEPTION_FILE).write_text(
                text + "\ncudadevrt\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "renewed review"):
                CHECKER._require_exception_policy(root)


class Gfn1ParameterProvenanceTests(unittest.TestCase):
    """Pin every nested GFN1 source, diagnostic, and legal record."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the reviewed manifests and retained Apache license once."""
        cls.gfn1 = json.loads(
            (REPOSITORY / "data/parameters/gfn1_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        cls.gfn1_d3 = json.loads(
            (REPOSITORY / "data/parameters/gfn1_d3_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        cls.gfn1_legacy_sto = json.loads(
            (REPOSITORY / "data/parameters/gfn1_legacy_sto_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        cls.gfn1_legacy_sto_header = (
            REPOSITORY / "data/parameters/gfn1_legacy_sto.hpp"
        ).read_bytes()
        cls.apache = (REPOSITORY / "LICENSES/Apache-2.0.txt").read_bytes()
        cls.lgpl = (REPOSITORY / "LICENSES/LGPL-3.0-or-later.txt").read_bytes()

    def test_current_gfn1_provenance_is_accepted(self) -> None:
        """Accept the exact reviewed GFN1 and GFN1-D3 provenance records."""
        CHECKER._check_gfn1_parameter_provenance(copy.deepcopy(self.gfn1))
        CHECKER._check_gfn1_d3_provenance(
            copy.deepcopy(self.gfn1_d3), self.apache, self.lgpl
        )
        CHECKER._check_gfn1_legacy_sto_provenance(
            copy.deepcopy(self.gfn1_legacy_sto),
            self.lgpl,
            self.gfn1_legacy_sto_header,
        )

    def test_gfn1_legacy_sto_source_and_consumer_mutations_are_rejected(self) -> None:
        """The GFN1-only xTB rows and retained header remain exact."""
        manifest = copy.deepcopy(self.gfn1_legacy_sto)
        manifest["source"]["git_blob"] = "0" * 40
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "legacy STO manifest"):
            CHECKER._check_gfn1_legacy_sto_provenance(
                manifest, self.lgpl, self.gfn1_legacy_sto_header
            )
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "legacy STO consumer"):
            CHECKER._check_gfn1_legacy_sto_provenance(
                copy.deepcopy(self.gfn1_legacy_sto),
                self.lgpl,
                self.gfn1_legacy_sto_header + b"changed\n",
            )

    def test_gfn1_legacy_sto_extra_fields_are_rejected(self) -> None:
        """Reject unreviewed schema extensions at every distribution boundary."""
        manifest = copy.deepcopy(self.gfn1_legacy_sto)
        manifest["unexpected"] = True
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unexpected fields"):
            CHECKER._check_gfn1_legacy_sto_provenance(
                manifest, self.lgpl, self.gfn1_legacy_sto_header
            )

    def test_gfn1_source_digest_mutation_is_rejected(self) -> None:
        """Reject a modified aggregate digest for the tblite source set."""
        manifest = copy.deepcopy(self.gfn1)
        manifest["source"]["parameter_sources_sha256"] = "0" * 64
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unreviewed provenance"):
            CHECKER._check_gfn1_parameter_provenance(manifest)

    def test_gfn1_cross_check_blob_mutation_is_rejected(self) -> None:
        """Reject a dxtb cross-check record that names another Git blob."""
        manifest = copy.deepcopy(self.gfn1)
        manifest["cross_check"]["git_blob"] = "0" * 40
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unreviewed provenance"):
            CHECKER._check_gfn1_parameter_provenance(manifest)

    def test_gfn1_mctc_source_mutation_is_rejected(self) -> None:
        """Reject changed mctc-lib bytes used by the GFN1 generated header."""
        manifest = copy.deepcopy(self.gfn1)
        manifest["mctc"]["sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unreviewed provenance"):
            CHECKER._check_gfn1_parameter_provenance(manifest)

    def test_gfn1_mctc_legal_record_mutation_is_rejected(self) -> None:
        """Reject an incomplete Apache-2.0 legal record for mixed-source data."""
        manifest = copy.deepcopy(self.gfn1)
        manifest["mctc"]["legal_files"] = []
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unreviewed provenance"):
            CHECKER._check_gfn1_parameter_provenance(manifest)

    def test_gfn1_d3_mctc_source_mutation_is_rejected(self) -> None:
        """Reject a changed mctc-lib source digest in the D3 manifest."""
        manifest = copy.deepcopy(self.gfn1_d3)
        manifest["unit_conversion"]["sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unreviewed provenance"):
            CHECKER._check_gfn1_d3_provenance(manifest, self.apache, self.lgpl)

    def test_gfn1_d3_equation_source_mutation_is_rejected(self) -> None:
        """Reject a changed simple-dftd3 equation blob or digest."""
        manifest = copy.deepcopy(self.gfn1_d3)
        manifest["source"]["equation_sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unreviewed provenance"):
            CHECKER._check_gfn1_d3_provenance(manifest, self.apache, self.lgpl)

    def test_gfn1_d3_execution_contract_mutation_is_rejected(self) -> None:
        """Reject changed tblite cutoff or switch-width provenance."""
        manifest = copy.deepcopy(self.gfn1_d3)
        manifest["execution_contract"]["smooth_cutoff_width_bohr"] = 0.0
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unreviewed provenance"):
            CHECKER._check_gfn1_d3_provenance(manifest, self.apache, self.lgpl)

    def test_gfn1_d3_mctc_license_text_mutation_is_rejected(self) -> None:
        """Reject retained Apache license bytes outside the reviewed record."""
        with self.assertRaisesRegex(
            CHECKER.LicenseCheckError, "Apache license differs"
        ):
            CHECKER._check_gfn1_d3_provenance(
                copy.deepcopy(self.gfn1_d3), self.apache + b"changed\n", self.lgpl
            )

    def test_gfn1_d3_generated_metadata_mutation_is_rejected(self) -> None:
        """Bind generator and output metadata at every distribution boundary."""
        for field in ("generator", "outputs"):
            manifest = copy.deepcopy(self.gfn1_d3)
            if field == "generator":
                manifest[field]["sha256"] = "0" * 64
            else:
                manifest[field]["gfn1_d3.hpp"]["sha256"] = "0" * 64
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(
                    CHECKER.LicenseCheckError, "generated-output metadata"
                ),
            ):
                CHECKER._check_gfn1_d3_provenance(manifest, self.apache, self.lgpl)

    def test_gfn1_d3_non_object_and_extra_fields_are_rejected(self) -> None:
        """Reject schema changes and non-object JSON through controlled errors."""
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "root is not an object"):
            CHECKER._check_gfn1_d3_provenance([], self.apache, self.lgpl)
        manifest = copy.deepcopy(self.gfn1_d3)
        manifest["unexpected"] = True
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unexpected fields"):
            CHECKER._check_gfn1_d3_provenance(manifest, self.apache, self.lgpl)

    def test_gfn1_d3_lgpl_text_mutation_is_rejected(self) -> None:
        """Require the exact reviewed simple-dftd3/tblite LGPL grant."""
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "LGPL license differs"):
            CHECKER._check_gfn1_d3_provenance(
                copy.deepcopy(self.gfn1_d3), self.apache, self.lgpl + b"changed\n"
            )


class Gfn1FixtureProvenanceTests(unittest.TestCase):
    """Pin repository-only GFN1 fixtures without adding package payload."""

    def test_current_fixture_provenance_is_accepted(self) -> None:
        """Accept the exact reviewed dxtb, mstore, and tblite records."""
        CHECKER._check_gfn1_fixture_provenance(REPOSITORY)

    def test_windows_crlf_fixture_checkout_is_accepted(self) -> None:
        """Hash canonical fixture text across Git's Windows CRLF checkout."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            for relative in (
                "tests/gfn1_d3_test.cpp",
                "tests/gfn1_halogen_test.cpp",
                "tests/gfn1_cpu_conformance.py",
            ):
                consumer = root / relative
                consumer.write_bytes(consumer.read_bytes().replace(b"\n", b"\r\n"))
            CHECKER._check_gfn1_fixture_provenance(root)

    def test_spin2_fixture_literal_mutation_is_rejected(self) -> None:
        """Bind the unrestricted P10 oracle literals and Python marker syntax."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            consumer = root / "tests/gfn1_cpu_conformance.py"
            consumer.write_text(
                consumer.read_text(encoding="utf-8").replace(
                    "-11.539671328635730", "-11.539671328635731", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "fixture bytes"):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_fixture_source_digest_mutation_is_rejected(self) -> None:
        """Reject a changed upstream digest even when the local tests remain."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][0]["files"][0]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "source files"):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_extracted_fixture_literal_mutation_is_rejected(self) -> None:
        """Bind copied numerical literals, not only their upstream source files."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            consumer = root / "tests/gfn1_d3_test.cpp"
            consumer.write_text(
                consumer.read_text(encoding="utf-8").replace(
                    "-8.2108039012179698e-5", "-8.2108039012179697e-5", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "fixture bytes"):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_extracted_fixture_digest_mutation_is_rejected(self) -> None:
        """Keep marker-delimited extraction digests immutable."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["extracted_fixtures"][0]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "unreviewed extracted"
            ):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_fixture_required_source_file_is_rejected_when_missing(self) -> None:
        """Keep the tblite ES2 equation source tied to its scientific fixture."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tblite = next(
                source
                for source in manifest["sources"]
                if source["project"] == "tblite"
            )
            tblite["files"] = [
                item
                for item in tblite["files"]
                if item["path"] != "src/tblite/coulomb/charge/effective.f90"
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "source files"):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_halogen_fixture_source_is_rejected_when_missing(self) -> None:
        """Keep the tblite halogen geometries and energies provenance-pinned."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tblite = next(
                source
                for source in manifest["sources"]
                if source["project"] == "tblite"
            )
            tblite["files"] = [
                item
                for item in tblite["files"]
                if item["path"] != "test/unit/test_halogen.f90"
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "source files"):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_duplicate_fixture_project_is_rejected(self) -> None:
        """Reject duplicate projects that hide a required provenance source."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][1] = copy.deepcopy(manifest["sources"][0])
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "incomplete sources"
            ):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_duplicate_fixture_source_file_is_rejected(self) -> None:
        """Reject duplicate paths before they collapse into a file mapping."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tblite = next(
                source
                for source in manifest["sources"]
                if source["project"] == "tblite"
            )
            tblite["files"].append(copy.deepcopy(tblite["files"][0]))
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "duplicate source file paths"
            ):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_non_string_fixture_source_path_is_rejected(self) -> None:
        """Reject malformed source paths without leaking a Python TypeError."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][0]["files"][0]["path"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "source files"):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_fixture_source_use_mutation_is_rejected(self) -> None:
        """Keep every fixture tied to its reviewed scientific extraction role."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][0]["files"][0]["use"] = "unreviewed role"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "extraction roles"):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_fixture_source_consumer_mutation_is_rejected(self) -> None:
        """Keep copied values tied to their exact reviewed local consumers."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][0]["files"][0]["consumers"] = [
                "tests/gfn1_basis_test.cpp"
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "extraction roles"):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_missing_fixture_consumer_is_rejected(self) -> None:
        """Reject a reviewed consumer path when its local test file is absent."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            (root / "tests/gfn1_h0_test.cpp").unlink()
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "missing consumer"):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_duplicate_fixture_consumers_are_rejected(self) -> None:
        """Reject duplicate consumers instead of silently normalizing them."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            consumers = manifest["sources"][0]["files"][0]["consumers"]
            consumers.append(consumers[0])
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "extraction roles"):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_non_string_fixture_project_is_rejected(self) -> None:
        """Report malformed project scalars through the controlled checker error."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][0]["project"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "incomplete sources"
            ):
                CHECKER._check_gfn1_fixture_provenance(root)

    def test_non_string_fixture_consumer_is_rejected(self) -> None:
        """Reject malformed consumer scalars without leaking a Python TypeError."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn1-fixture-") as directory:
            root = Path(directory)
            shutil.copytree(REPOSITORY / "tests", root / "tests")
            manifest_path = root / CHECKER.GFN1_FIXTURE_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][0]["files"][0]["consumers"][0] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "extraction roles"):
                CHECKER._check_gfn1_fixture_provenance(root)


class ImplibProvenanceTests(unittest.TestCase):
    """Verify vendored implib content against its pinned provenance."""

    def _copy_payload(self, root: Path) -> None:
        source = REPOSITORY / CHECKER.IMPLIB_VENDOR_PATH
        destination = root / CHECKER.IMPLIB_VENDOR_PATH
        destination.parent.mkdir(parents=True)
        shutil.copytree(source, destination)
        manifest = root / CHECKER.IMPLIB_MANIFEST_PATH
        manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / CHECKER.IMPLIB_MANIFEST_PATH, manifest)

    def test_vendored_tree_matches_pinned_deepmd_revision(self) -> None:
        """Accept the checked-in implib tree at its pinned revision."""
        CHECKER._check_implib_provenance(REPOSITORY)

    def test_git_index_mode_overrides_unreliable_filesystem_mode(self) -> None:
        """Use tracked modes when Windows cannot represent the executable bit."""
        manifest = CHECKER.json.loads(
            (REPOSITORY / CHECKER.IMPLIB_MANIFEST_PATH).read_text(encoding="utf-8")
        )
        declared = CHECKER._check_implib_manifest(manifest)
        index_modes = {
            f"{CHECKER.IMPLIB_VENDOR_PATH}/{relative}": mode
            for relative, (mode, _blob, _sha256) in declared.items()
        }
        with tempfile.TemporaryDirectory(prefix="xtbloom-implib-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            with (
                mock.patch.object(
                    CHECKER, "_git_index_modes", return_value=index_modes
                ),
                mock.patch.object(CHECKER, "_filesystem_git_mode", return_value=None),
            ):
                CHECKER._check_implib_provenance(root)

    def test_mode_mismatch_has_a_specific_diagnostic(self) -> None:
        """Distinguish mode drift from changed vendored bytes."""
        manifest = CHECKER.json.loads(
            (REPOSITORY / CHECKER.IMPLIB_MANIFEST_PATH).read_text(encoding="utf-8")
        )
        declared = CHECKER._check_implib_manifest(manifest)
        wrong_modes = {
            f"{CHECKER.IMPLIB_VENDOR_PATH}/{relative}": (
                "100644" if relative == "implib-gen.py" else mode
            )
            for relative, (mode, _blob, _sha256) in declared.items()
        }
        with tempfile.TemporaryDirectory(prefix="xtbloom-implib-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            with (
                mock.patch.object(
                    CHECKER, "_git_index_modes", return_value=wrong_modes
                ),
                self.assertRaisesRegex(
                    CHECKER.LicenseCheckError,
                    "mode differs.*implib-gen.py.*Git index: expected 100755.*"
                    "observed 100644",
                ),
            ):
                CHECKER._check_implib_provenance(root)

    def test_unexpected_vendored_file_is_rejected(self) -> None:
        """Reject undeclared files in the vendored implib tree."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-implib-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            (root / CHECKER.IMPLIB_VENDOR_PATH / "unexpected.txt").write_text(
                "unexpected\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "unexpected unexpected.txt"
            ):
                CHECKER._check_implib_provenance(root)

    def test_modified_vendored_bytes_are_rejected(self) -> None:
        """Reject modified bytes in a declared vendored implib file."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-implib-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            modified = (
                root / CHECKER.IMPLIB_VENDOR_PATH / "arch" / "x86_64" / "config.ini"
            )
            modified.write_bytes(modified.read_bytes() + b"# modified\n")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError,
                "vendored file differs.*arch/x86_64/config.ini",
            ):
                CHECKER._check_implib_provenance(root)


class TorchStableProvenanceTests(unittest.TestCase):
    """Verify vendored LibTorch Stable ABI headers against their manifest."""

    def _copy_payload(self, root: Path) -> None:
        source = REPOSITORY / CHECKER.TORCH_STABLE_VENDOR_PATH
        destination = root / CHECKER.TORCH_STABLE_VENDOR_PATH
        destination.parent.mkdir(parents=True)
        shutil.copytree(source, destination)
        manifest = root / CHECKER.TORCH_STABLE_MANIFEST_PATH
        manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / CHECKER.TORCH_STABLE_MANIFEST_PATH, manifest)

    def test_vendored_tree_matches_pinned_torch_release(self) -> None:
        """Accept the checked-in torch-stable tree at its pinned release."""
        CHECKER._check_torch_stable_provenance(REPOSITORY)

    def test_unexpected_vendored_file_is_rejected(self) -> None:
        """Reject undeclared files in the vendored torch-stable tree."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-torch-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            vendor_root = root / CHECKER.TORCH_STABLE_VENDOR_PATH
            (
                vendor_root
                / CHECKER.TORCH_STABLE_INCLUDE_SUBDIR
                / "torch"
                / "unexpected.h"
            ).parent.mkdir(parents=True, exist_ok=True)
            (
                vendor_root
                / CHECKER.TORCH_STABLE_INCLUDE_SUBDIR
                / "torch"
                / "unexpected.h"
            ).write_text("// unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unexpected"):
                CHECKER._check_torch_stable_provenance(root)

    def test_modified_vendored_bytes_are_rejected(self) -> None:
        """Reject modified bytes in a declared vendored torch-stable header."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-torch-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            modified = (
                root
                / CHECKER.TORCH_STABLE_VENDOR_PATH
                / CHECKER.TORCH_STABLE_INCLUDE_SUBDIR
                / "torch"
                / "csrc"
                / "stable"
                / "tensor.h"
            )
            modified.write_bytes(modified.read_bytes() + b"// modified\n")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError,
                "vendored file differs.*torch/csrc/stable/tensor.h",
            ):
                CHECKER._check_torch_stable_provenance(root)

    def test_sdist_must_carry_the_pinned_header_tree(self) -> None:
        """Accept an sdist whose vendored headers match the manifest bytes."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-torch-test-") as directory:
            root = Path(directory)
            archive = root / "xtbloom-0.1.0.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(
                    REPOSITORY / CHECKER.TORCH_STABLE_MANIFEST_PATH,
                    arcname=f"xtbloom-0.1.0/{CHECKER.TORCH_STABLE_MANIFEST_PATH}",
                )
                vendor_source = REPOSITORY / CHECKER.TORCH_STABLE_VENDOR_PATH
                for relative in (vendor_source / "include").rglob("*"):
                    if relative.is_file():
                        rel = relative.relative_to(REPOSITORY).as_posix()
                        tar.add(relative, arcname=f"xtbloom-0.1.0/{rel}")
            names = CHECKER._archive_names(archive)
            CHECKER._check_archived_torch_stable(archive, names)

    def test_sdist_with_modified_header_is_rejected(self) -> None:
        """Reject an sdist whose vendored header bytes were altered."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-torch-test-") as directory:
            root = Path(directory)
            archive = root / "xtbloom-0.1.0.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(
                    REPOSITORY / CHECKER.TORCH_STABLE_MANIFEST_PATH,
                    arcname=f"xtbloom-0.1.0/{CHECKER.TORCH_STABLE_MANIFEST_PATH}",
                )
                vendor_source = REPOSITORY / CHECKER.TORCH_STABLE_VENDOR_PATH
                for relative in (vendor_source / "include").rglob("*"):
                    if relative.is_file():
                        rel = relative.relative_to(REPOSITORY).as_posix()
                        payload = relative.read_bytes()
                        if rel.endswith("torch/csrc/stable/tensor.h"):
                            payload += b"// modified\n"
                        root_child = root / rel
                        root_child.parent.mkdir(parents=True, exist_ok=True)
                        root_child.write_bytes(payload)
                        tar.add(root_child, arcname=f"xtbloom-0.1.0/{rel}")
            names = CHECKER._archive_names(archive)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError,
                "differs from pinned bytes.*tensor.h",
            ):
                CHECKER._check_archived_torch_stable(archive, names)


class EigenProvenanceTests(unittest.TestCase):
    """Pin Eigen acquisition metadata and the compact retained legal payload."""

    def _copy_payload(self, root: Path) -> None:
        for relative in (CHECKER.EIGEN_MANIFEST_PATH, *CHECKER.EIGEN_RETAINED_FILES):
            source = REPOSITORY / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copy2(REPOSITORY / ".gitattributes", root / ".gitattributes")
        shutil.copy2(REPOSITORY / "pyproject.toml", root / "pyproject.toml")

    def _write_sdist(
        self,
        root: Path,
        *,
        missing: str | None = None,
        modified: str | None = None,
        unexpected: str | None = None,
    ) -> Path:
        """Create a focused Eigen sdist fixture with optional payload drift."""
        archive_path = root / "xtbloom-0.1.0.tar.gz"
        archive_root = "xtbloom-0.1.0"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(
                REPOSITORY / CHECKER.EIGEN_MANIFEST_PATH,
                arcname=f"{archive_root}/{CHECKER.EIGEN_MANIFEST_PATH}",
            )
            for relative in CHECKER.EIGEN_RETAINED_FILES:
                if relative == missing:
                    continue
                source = REPOSITORY / relative
                payload = source.read_bytes()
                if relative == modified:
                    payload += b"// modified\n"
                member = tarfile.TarInfo(f"{archive_root}/{relative}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            if unexpected is not None:
                payload = b"// unexpected\n"
                member = tarfile.TarInfo(f"{archive_root}/{unexpected}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        return archive_path

    def test_exact_compact_payload_is_accepted(self) -> None:
        """Accept the pinned archive metadata and nine exact legal records."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-eigen-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            CHECKER._check_eigen_provenance(root)

    def test_modified_retained_legal_file_is_rejected(self) -> None:
        """Reject any change to a manifest-declared Eigen legal record."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-eigen-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            license_path = root / "LICENSES/eigen/COPYING.MPL2"
            license_path.write_bytes(license_path.read_bytes() + b"modified\n")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError,
                "differs from pinned bytes.*COPYING.MPL2",
            ):
                CHECKER._check_eigen_provenance(root)

    def test_missing_retained_legal_file_is_rejected(self) -> None:
        """Require all legal records selected from the compiled include graph."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-eigen-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            (root / "LICENSES/eigen/notices/Half.h").unlink()
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "missing.*Half.h"):
                CHECKER._check_eigen_provenance(root)

    def test_unexpected_vendored_source_is_rejected(self) -> None:
        """Keep the large Eigen header tree out of the repository."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-eigen-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            unexpected = root / "cmake/3rdparty/eigen/Eigen/Core"
            unexpected.parent.mkdir(parents=True)
            unexpected.write_text("// unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "must not vendor Eigen"
            ):
                CHECKER._check_eigen_provenance(root)

    def test_unexpected_archive_is_rejected(self) -> None:
        """Keep the official archive in build caches rather than Git."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-eigen-test-") as directory:
            root = Path(directory)
            self._copy_payload(root)
            archive = root / "vendor/eigen-5.0.1.tar.gz"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"archive")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "must not vendor Eigen"
            ):
                CHECKER._check_eigen_provenance(root)

    def test_manifest_revision_drift_is_rejected(self) -> None:
        """Require a dependency re-audit before the pinned Eigen tag changes."""
        manifest = json.loads(
            (REPOSITORY / CHECKER.EIGEN_MANIFEST_PATH).read_text(encoding="utf-8")
        )
        manifest["revision"] = "0" * 40
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "pinned provenance"):
            CHECKER._check_eigen_manifest(manifest)

    def test_sdist_compact_legal_payload_is_accepted(self) -> None:
        """Accept an sdist carrying provenance and exact legal records only."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-eigen-test-") as directory:
            archive = self._write_sdist(Path(directory))
            CHECKER._check_archived_eigen(archive, CHECKER._archive_names(archive))

    def test_sdist_missing_eigen_legal_record_is_rejected(self) -> None:
        """Reject an sdist that loses a manifest-declared legal record."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-eigen-test-") as directory:
            archive = self._write_sdist(
                Path(directory), missing="LICENSES/eigen/COPYING.BSD"
            )
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "missing.*COPYING.BSD"
            ):
                CHECKER._check_archived_eigen(archive, CHECKER._archive_names(archive))

    def test_sdist_modified_eigen_legal_record_is_rejected(self) -> None:
        """Reject an sdist whose retained Eigen legal bytes differ."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-eigen-test-") as directory:
            archive = self._write_sdist(
                Path(directory), modified="LICENSES/eigen/notices/AlignedBox.h"
            )
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "differs from pinned bytes.*AlignedBox.h"
            ):
                CHECKER._check_archived_eigen(archive, CHECKER._archive_names(archive))

    def test_sdist_unexpected_eigen_source_is_rejected(self) -> None:
        """Reject an sdist that adds the downloaded Eigen header tree."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-eigen-test-") as directory:
            archive = self._write_sdist(
                Path(directory), unexpected="include/eigen3/Eigen/Core"
            )
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "must not bundle Eigen source"
            ):
                CHECKER._check_archived_eigen(archive, CHECKER._archive_names(archive))

    def test_sdist_unexpected_eigen_archive_is_rejected(self) -> None:
        """Reject an sdist that embeds the build-time download cache."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-eigen-test-") as directory:
            archive = self._write_sdist(
                Path(directory), unexpected="vendor/eigen-5.0.1.tar.gz"
            )
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "must not bundle Eigen source"
            ):
                CHECKER._check_archived_eigen(archive, CHECKER._archive_names(archive))


class CudaWheelInspectionTests(unittest.TestCase):
    """Verify extraction and dispatch of CUDA wheel ABI inspection."""

    def test_wheel_payload_is_forwarded_to_cuda_abi_checker(self) -> None:
        """Forward the extracted ELF payload and selected readelf command."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-wheel-test-") as directory:
            root = Path(directory)
            wheel = root / "xtbloom-test.whl"
            marker = root / "checker-ran"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("xtbloom/lib/libxtbloom.so", b"\x7fELFtest")
            checker = root / "checker.py"
            checker.write_text(
                "import argparse\n"
                "from pathlib import Path\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--readelf', required=True)\n"
                "parser.add_argument('--library', required=True, type=Path)\n"
                "args = parser.parse_args()\n"
                "assert args.library.read_bytes() == b'\\x7fELFtest'\n"
                f"Path({str(marker)!r}).write_text(args.readelf, encoding='utf-8')\n",
                encoding="utf-8",
            )
            WHEEL_INSPECTOR.inspect_wheel(
                wheel,
                checker=checker,
                readelf="test-readelf",
                temporary_root=root / "extracted",
            )
            self.assertEqual(marker.read_text(encoding="utf-8"), "test-readelf")

    def test_wheel_with_multiple_elf_payloads_is_rejected(self) -> None:
        """Reject wheels containing ambiguous xtbloom ELF payloads."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-wheel-test-") as directory:
            root = Path(directory)
            wheel = root / "xtbloom-test.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("xtbloom/lib/libxtbloom.so", b"\x7fELFfirst")
                archive.writestr("xtbloom/lib/libxtbloom.so.1", b"\x7fELFsecond")
            with self.assertRaisesRegex(RuntimeError, "exactly one ELF"):
                WHEEL_INSPECTOR.inspect_wheel(
                    wheel,
                    checker=CHECKER_PATH,
                    readelf="readelf",
                    temporary_root=root / "extracted",
                )


class WheelVersionInspectionTests(unittest.TestCase):
    """Keep Python SCM versions compatible with the embedded native release."""

    def _write_version_wheel(
        self,
        path: Path,
        distribution_version: str = "1.2.3",
        native_version: str | None = None,
        header_newline: str = "\n",
    ) -> None:
        """Create the minimal archive needed for metadata-only version checks."""
        resolved_native = native_version or distribution_version
        major, minor, patch = resolved_native.split(".")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "xtbloom-test.dist-info/METADATA",
                "Metadata-Version: 2.4\nName: xtbloom\n"
                f"Version: {distribution_version}\n",
            )
            archive.writestr(
                "xtbloom/include/xtbloom/version.h",
                header_newline.join(
                    (
                        f'#define XTBLOOM_VERSION_STRING "{resolved_native}"',
                        f"#define XTBLOOM_VERSION_MAJOR {major}",
                        f"#define XTBLOOM_VERSION_MINOR {minor}",
                        f"#define XTBLOOM_VERSION_PATCH {patch}",
                        "",
                    )
                ),
            )
            archive.writestr("xtbloom/lib/libxtbloom.so", b"target-native-bytes")

    def test_native_library_names_cover_linux_macos_and_windows(self) -> None:
        """Recognize the platform-specific library names installed by CMake."""
        for name in (
            "libxtbloom.so",
            "libxtbloom.so.0",
            "libxtbloom.dylib",
            "xtbloom.dll",
        ):
            with self.subTest(name=name):
                self.assertTrue(
                    VERSION_INSPECTOR._is_native_library(
                        VERSION_INSPECTOR.PurePosixPath("xtbloom/lib") / name
                    )
                )

    def test_similar_library_names_are_rejected(self) -> None:
        """Do not accept import libraries or unrelated prefixed DLL names."""
        for name in (
            "xtbloom.lib",
            "libxtbloom.a",
            "libxtbloom.so.backup",
            "libxtbloom.dylib.backup",
            "other_xtbloom.dll",
        ):
            with self.subTest(name=name):
                self.assertFalse(
                    VERSION_INSPECTOR._is_native_library(
                        VERSION_INSPECTOR.PurePosixPath("xtbloom/lib") / name
                    )
                )

    def test_metadata_only_version_check_accepts_release(self) -> None:
        """Validate cross-compiled wheels without loading their target DSO."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            wheel = root / "xtbloom-test.whl"
            self._write_version_wheel(wheel)
            VERSION_INSPECTOR.inspect_wheel(
                wheel,
                root / "extracted",
                metadata_only=True,
                expected_version="1.2.3",
            )

    def test_metadata_only_version_check_accepts_post_tag_development(self) -> None:
        """Allow Python artifacts to identify commits after a native release."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            wheel = root / "xtbloom-test.whl"
            self._write_version_wheel(
                wheel,
                distribution_version="1.2.3.post1.dev7+g0123456789",
                native_version="1.2.3",
            )
            VERSION_INSPECTOR.inspect_wheel(
                wheel,
                root / "extracted",
                metadata_only=True,
                expected_version="1.2.3.post1.dev7+g0123456789",
            )

    def test_metadata_only_version_check_accepts_dirty_development(self) -> None:
        """Accept setuptools-scm's dirty-date suffix for local source builds."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            wheel = root / "xtbloom-test.whl"
            self._write_version_wheel(
                wheel,
                distribution_version="1.2.3.post1.dev0+g0123456789.d20260810",
                native_version="1.2.3",
            )
            VERSION_INSPECTOR.inspect_wheel(
                wheel,
                root / "extracted",
                metadata_only=True,
            )

    def test_metadata_only_version_check_accepts_windows_crlf_header(self) -> None:
        """Treat CMake's Windows newlines as the same generated version ABI."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            wheel = root / "xtbloom-test.whl"
            self._write_version_wheel(wheel, header_newline="\r\n")
            VERSION_INSPECTOR.inspect_wheel(
                wheel,
                root / "extracted",
                metadata_only=True,
                expected_version="1.2.3",
            )

    def test_native_version_check_releases_loaded_library(self) -> None:
        """Unload extracted Windows DLLs before temporary-tree cleanup."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            wheel = root / "xtbloom-test.whl"
            self._write_version_wheel(wheel)
            version_function = mock.Mock(return_value=b"1.2.3")
            library = mock.Mock(xtbloom_version_string=version_function)
            with (
                mock.patch.object(
                    VERSION_INSPECTOR.ctypes, "CDLL", return_value=library
                ),
                mock.patch.object(
                    VERSION_INSPECTOR, "_release_native_library"
                ) as release,
            ):
                VERSION_INSPECTOR.inspect_wheel(wheel, root / "extracted")
            release.assert_called_once_with(library)

    def test_metadata_only_version_check_rejects_wrong_native_release(self) -> None:
        """Reject a wheel whose native library does not match the SCM base tag."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            wheel = root / "xtbloom-test.whl"
            self._write_version_wheel(
                wheel,
                distribution_version="1.2.3.post1.dev7+g0123456789",
                native_version="1.2.2",
            )
            with self.assertRaisesRegex(RuntimeError, "native release"):
                VERSION_INSPECTOR.inspect_wheel(
                    wheel,
                    root / "extracted",
                    metadata_only=True,
                )

    def test_metadata_only_version_check_rejects_development_without_node(self) -> None:
        """Require every non-release Python artifact to identify its Git node."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            wheel = root / "xtbloom-test.whl"
            self._write_version_wheel(
                wheel,
                distribution_version="1.2.3.post1.dev7",
                native_version="1.2.3",
            )
            with self.assertRaisesRegex(RuntimeError, "unsupported Python"):
                VERSION_INSPECTOR.inspect_wheel(
                    wheel,
                    root / "extracted",
                    metadata_only=True,
                )

    def test_metadata_only_version_check_rejects_wrong_release(self) -> None:
        """Prevent a release event from publishing a differently versioned wheel."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            wheel = root / "xtbloom-test.whl"
            self._write_version_wheel(wheel)
            with self.assertRaisesRegex(
                RuntimeError, "does not match expected Python distribution"
            ):
                VERSION_INSPECTOR.inspect_wheel(
                    wheel,
                    root / "extracted",
                    metadata_only=True,
                    expected_version="1.2.4",
                )


if __name__ == "__main__":
    unittest.main()
