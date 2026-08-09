"""Regression tests for the release legal-material validator."""

from __future__ import annotations

import copy
import importlib.util
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
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


class LicenseArchiveTests(unittest.TestCase):
    """Verify legal payload requirements for built distribution archives."""

    def _write_wheel(self, path: Path, names: set[str]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for name in sorted(names):
                if name.endswith("/provenance/implib_manifest.json"):
                    payload = (REPOSITORY / CHECKER.IMPLIB_MANIFEST_PATH).read_bytes()
                elif name.endswith("/provenance/scipy_openblas32_manifest.json"):
                    payload = (REPOSITORY / CHECKER.OPENBLAS_MANIFEST_PATH).read_bytes()
                elif name.endswith("scipy-openblas32-0.3.34.0.0.txt"):
                    payload = (REPOSITORY / CHECKER.OPENBLAS_LICENSE).read_bytes()
                else:
                    payload = b"test\n"
                archive.writestr(name, payload)

    def _valid_wheel_names(self) -> set[str]:
        names = {
            f"xtbloom-0.1.0.dist-info/licenses/{suffix}"
            for suffix in CHECKER.COMMON_ARCHIVE_SUFFIXES
        } | {f"xtbloom/{suffix}" for suffix in CHECKER.WHEEL_ARCHIVE_SUFFIXES}
        manifest = CHECKER.json.loads(
            (REPOSITORY / CHECKER.OPENBLAS_MANIFEST_PATH).read_text(encoding="utf-8")
        )
        names.add("xtbloom/lib/libxtbloom_openblas_lp64_shim.so")
        for record in manifest["architectures"]["x86_64"]["files"]:
            source_name = Path(record["source"]).name
            names.add(
                "xtbloom.libs/"
                + CHECKER._auditwheel_name(source_name, record["sha256"])
            )
        return names

    def test_project_license_cannot_be_satisfied_by_third_party_filename(self) -> None:
        """Require the project license at its exact archive location."""
        names = self._valid_wheel_names()
        names.remove("xtbloom-0.1.0.dist-info/licenses/LICENSE")
        names.add("xtbloom/share/licenses/xtbloom/third-party/d4/mctc-lib-LICENSE")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "LICENSE"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_every_provenance_manifest(self) -> None:
        """Require all provenance manifests in wheel payloads."""
        names = self._valid_wheel_names()
        missing = "xtbloom/share/licenses/xtbloom/provenance/mctc_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "mctc_manifest"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_implib_provenance_manifest(self) -> None:
        """Require the vendored implib provenance manifest in wheels."""
        names = self._valid_wheel_names()
        missing = "xtbloom/share/licenses/xtbloom/provenance/implib_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "implib_manifest"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_linking_exception(self) -> None:
        """Require the GPLv3 Section 7 exception in wheel archives."""
        names = self._valid_wheel_names()
        missing = "xtbloom-0.1.0.dist-info/licenses/CUDA_MKL_LINKING_EXCEPTION"
        names.remove(missing)
        names.remove("xtbloom/share/licenses/xtbloom/CUDA_MKL_LINKING_EXCEPTION")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "CUDA_MKL_LINKING_EXCEPTION"
            ):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_array_api_compat_license(self) -> None:
        """Keep the new runtime dependency's distinct MIT grant in wheels."""
        names = self._valid_wheel_names()
        suffix = "LICENSES/array-api-compat-MIT.txt"
        names.remove(f"xtbloom-0.1.0.dist-info/licenses/{suffix}")
        names.remove(
            "xtbloom/share/licenses/xtbloom/third-party/array-api-compat-MIT.txt"
        )
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "array-api-compat-MIT"
            ):
                CHECKER.check_archive(wheel)

    def test_wheel_must_not_bundle_vendor_library(self) -> None:
        """Reject wheels that bundle separately licensed vendor libraries."""
        names = self._valid_wheel_names()
        names.add("xtbloom/lib/libcudart.so.12")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "libcudart"):
                CHECKER.check_archive(wheel)

    def test_wheel_requires_complete_private_openblas_cohort(self) -> None:
        """Reject a wheel that loses one auditwheel-vendored support DSO."""
        names = self._valid_wheel_names()
        missing = next(name for name in names if "libquadmath" in name)
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "cohort differs"):
                CHECKER.check_archive(wheel)

    def test_wheel_rejects_unreviewed_openblas_binary(self) -> None:
        """Require dependency re-audit before the vendored ELF set expands."""
        names = self._valid_wheel_names()
        names.add("xtbloom.libs/libgfortran-unreviewed.so.5")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test_x86_64.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "cohort differs"):
                CHECKER.check_archive(wheel)


class OpenBlasProvenanceTests(unittest.TestCase):
    """Pin every provenance locator for the redistributed wheel inputs."""

    def setUp(self) -> None:
        """Load a fresh manifest for each negative mutation."""
        self.manifest = CHECKER.json.loads(
            (REPOSITORY / CHECKER.OPENBLAS_MANIFEST_PATH).read_text(encoding="utf-8")
        )

    def test_current_openblas_manifest_is_accepted(self) -> None:
        """Accept the exact reviewed repositories, wheels, and ELF cohort."""
        CHECKER._check_openblas_manifest(self.manifest)

    def test_openblas_manifest_rejects_changed_repository(self) -> None:
        """Do not let a provenance URL drift while payload checks stay green."""
        self.manifest["source"]["repository"] = "https://example.invalid/openblas"
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "repository"):
            CHECKER._check_openblas_manifest(self.manifest)

    def test_openblas_manifest_rejects_changed_wheel_url(self) -> None:
        """Keep the immutable PyPI artifact locator tied to reviewed bytes."""
        self.manifest["architectures"]["x86_64"]["wheel"]["url"] = (
            "https://example.invalid/scipy-openblas32.whl"
        )
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "x86_64 wheel"):
            CHECKER._check_openblas_manifest(self.manifest)


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
            destination.write_bytes(b"test\n")
        (root / "index.html").write_text(
            '<a href="LICENSE">license</a>\n'
            '<a href="THIRD_PARTY_NOTICES.md">notices</a>\n'
            '<a href="LICENSES/openchemlib-BSD-3-Clause.txt">OpenChemLib</a>\n'
            '<a href="CUDA_MKL_LINKING_EXCEPTION">permission</a>\n'
            '<a href="https://jinzhezeng.group/xtbloom/">demo</a>\n'
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

    def test_web_site_rejects_raw_lapack_side_module(self) -> None:
        """Do not deploy a second untracked copy of the preloaded side module."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "libscipy_openblas.so").write_bytes(b"unexpected raw side module")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "raw LAPACK side module"
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


if __name__ == "__main__":
    unittest.main()
