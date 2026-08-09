"""Regression tests for the release legal-material validator."""

from __future__ import annotations

import copy
import importlib.util
import shutil
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPOSITORY / "tools" / "licensing" / "check_licenses.py"
SPEC = importlib.util.spec_from_file_location("gpuxtb_check_licenses", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
WHEEL_INSPECTOR_PATH = REPOSITORY / "python" / "ci" / "inspect-cuda-wheel.py"
WHEEL_SPEC = importlib.util.spec_from_file_location(
    "gpuxtb_inspect_cuda_wheel", WHEEL_INSPECTOR_PATH
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
                else:
                    payload = b"test\n"
                archive.writestr(name, payload)

    def _valid_wheel_names(self) -> set[str]:
        return {
            f"gpuxtb-0.1.0.dist-info/licenses/{suffix}"
            for suffix in CHECKER.COMMON_ARCHIVE_SUFFIXES
        } | {f"gpuxtb/{suffix}" for suffix in CHECKER.WHEEL_ARCHIVE_SUFFIXES}

    def test_project_license_cannot_be_satisfied_by_third_party_filename(self) -> None:
        """Require the project license at its exact archive location."""
        names = self._valid_wheel_names()
        names.remove("gpuxtb-0.1.0.dist-info/licenses/LICENSE")
        names.add("gpuxtb/share/licenses/gpuxtb/third-party/d4/mctc-lib-LICENSE")
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "LICENSE"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_every_provenance_manifest(self) -> None:
        """Require all provenance manifests in wheel payloads."""
        names = self._valid_wheel_names()
        missing = "gpuxtb/share/licenses/gpuxtb/provenance/mctc_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "mctc_manifest"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_implib_provenance_manifest(self) -> None:
        """Require the vendored implib provenance manifest in wheels."""
        names = self._valid_wheel_names()
        missing = "gpuxtb/share/licenses/gpuxtb/provenance/implib_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "implib_manifest"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_torch_stable_provenance_manifest(self) -> None:
        """Require the vendored LibTorch provenance manifest in wheels."""
        names = self._valid_wheel_names()
        missing = "gpuxtb/share/licenses/gpuxtb/provenance/torch_stable_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "torch_stable"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_linking_exception(self) -> None:
        """Require the GPLv3 Section 7 exception in wheel archives."""
        names = self._valid_wheel_names()
        missing = "gpuxtb-0.1.0.dist-info/licenses/CUDA_MKL_LINKING_EXCEPTION"
        names.remove(missing)
        names.remove("gpuxtb/share/licenses/gpuxtb/CUDA_MKL_LINKING_EXCEPTION")
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "CUDA_MKL_LINKING_EXCEPTION"
            ):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_array_api_compat_license(self) -> None:
        """Keep the new runtime dependency's distinct MIT grant in wheels."""
        names = self._valid_wheel_names()
        suffix = "LICENSES/array-api-compat-MIT.txt"
        names.remove(f"gpuxtb-0.1.0.dist-info/licenses/{suffix}")
        names.remove(
            "gpuxtb/share/licenses/gpuxtb/third-party/array-api-compat-MIT.txt"
        )
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "array-api-compat-MIT"
            ):
                CHECKER.check_archive(wheel)

    def test_wheel_must_not_bundle_vendor_library(self) -> None:
        """Reject wheels that bundle separately licensed vendor libraries."""
        names = self._valid_wheel_names()
        names.add("gpuxtb/lib/libcudart.so.12")
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "libcudart"):
                CHECKER.check_archive(wheel)


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
            '<a href="CUDA_MKL_LINKING_EXCEPTION">permission</a>\n'
            '<a href="https://jinzhezeng.group/gpuxtb/">demo</a>\n'
            '<a href="https://github.com/jinzhezenggroup/gpuxtb">source</a>\n',
            encoding="utf-8",
        )

    def test_complete_web_site_payload_is_accepted(self) -> None:
        """Accept exact source legal bytes beside the deployed runtime."""
        with tempfile.TemporaryDirectory(prefix="gpuxtb-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_requires_project_license(self) -> None:
        """The GPL-covered WASM cannot be deployed without the project grant."""
        with tempfile.TemporaryDirectory(prefix="gpuxtb-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "LICENSE").unlink()
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "LICENSE"):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_requires_pako_zlib_notice(self) -> None:
        """Keep the non-MIT zlib grant for code inside the 3Dmol bundle."""
        with tempfile.TemporaryDirectory(prefix="gpuxtb-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "LICENSES/pako-Zlib.txt").unlink()
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "pako-Zlib"):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_requires_torch_bsd_license(self) -> None:
        """Keep the vendored LibTorch headers' BSD grant in Pages payloads."""
        with tempfile.TemporaryDirectory(prefix="gpuxtb-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "LICENSES/BSD-3-Clause.txt").unlink()
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "BSD-3-Clause"):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_rejects_raw_lapack_side_module(self) -> None:
        """Do not deploy a second untracked copy of the preloaded side module."""
        with tempfile.TemporaryDirectory(prefix="gpuxtb-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "libscipy_openblas.so").write_bytes(b"unexpected raw side module")
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError, "raw LAPACK side module"
            ):
                CHECKER.check_web_site(root, REPOSITORY)

    def test_web_site_rejects_arbitrary_stale_artifact(self) -> None:
        """Reject obsolete engine variants because Pages uploads every file."""
        with tempfile.TemporaryDirectory(prefix="gpuxtb-web-license-") as directory:
            root = Path(directory)
            self._write_valid_site(root)
            (root / "gpuxtb_web-old.wasm").write_bytes(b"stale engine")
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
        self.project = metadata["project"]

    def test_current_dependency_policy_is_accepted(self) -> None:
        """Accept the repository's reviewed dependency policy."""
        CHECKER._require_dependency_policy(self.project)

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

    def test_openblas_must_cover_both_linux_architectures(self) -> None:
        """Require the reviewed OpenBLAS wheel architecture selectors."""
        project = copy.deepcopy(self.project)
        project["dependencies"] = [
            requirement.replace(" or platform_machine == 'aarch64'", "")
            if requirement.startswith("scipy-openblas32")
            else requirement
            for requirement in project["dependencies"]
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "x86_64 and aarch64"):
            CHECKER._require_dependency_policy(project)

    def test_openblas_must_use_reviewed_minimum(self) -> None:
        """Require the minimum reviewed scipy-openblas32 version."""
        project = copy.deepcopy(self.project)
        project["dependencies"] = [
            requirement.replace(">=0.3.34.0.0", "")
            if requirement.startswith("scipy-openblas32")
            else requirement
            for requirement in project["dependencies"]
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "reviewed minimum"):
            CHECKER._require_dependency_policy(project)

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
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
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
        with tempfile.TemporaryDirectory(prefix="gpuxtb-implib-test-") as directory:
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
        with tempfile.TemporaryDirectory(prefix="gpuxtb-implib-test-") as directory:
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
        with tempfile.TemporaryDirectory(prefix="gpuxtb-torch-test-") as directory:
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
        with tempfile.TemporaryDirectory(prefix="gpuxtb-torch-test-") as directory:
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
        with tempfile.TemporaryDirectory(prefix="gpuxtb-torch-test-") as directory:
            root = Path(directory)
            archive = root / "gpuxtb-0.1.0.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(
                    REPOSITORY / CHECKER.TORCH_STABLE_MANIFEST_PATH,
                    arcname=f"gpuxtb-0.1.0/{CHECKER.TORCH_STABLE_MANIFEST_PATH}",
                )
                vendor_source = REPOSITORY / CHECKER.TORCH_STABLE_VENDOR_PATH
                for relative in (vendor_source / "include").rglob("*"):
                    if relative.is_file():
                        rel = relative.relative_to(REPOSITORY).as_posix()
                        tar.add(relative, arcname=f"gpuxtb-0.1.0/{rel}")
            names = CHECKER._archive_names(archive)
            CHECKER._check_archived_torch_stable(archive, names)

    def test_sdist_with_modified_header_is_rejected(self) -> None:
        """Reject an sdist whose vendored header bytes were altered."""
        with tempfile.TemporaryDirectory(prefix="gpuxtb-torch-test-") as directory:
            root = Path(directory)
            archive = root / "gpuxtb-0.1.0.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(
                    REPOSITORY / CHECKER.TORCH_STABLE_MANIFEST_PATH,
                    arcname=f"gpuxtb-0.1.0/{CHECKER.TORCH_STABLE_MANIFEST_PATH}",
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
                        tar.add(root_child, arcname=f"gpuxtb-0.1.0/{rel}")
            names = CHECKER._archive_names(archive)
            with self.assertRaisesRegex(
                CHECKER.LicenseCheckError,
                "differs from pinned bytes.*tensor.h",
            ):
                CHECKER._check_archived_torch_stable(archive, names)


class CudaWheelInspectionTests(unittest.TestCase):
    """Verify extraction and dispatch of CUDA wheel ABI inspection."""

    def test_wheel_payload_is_forwarded_to_cuda_abi_checker(self) -> None:
        """Forward the extracted ELF payload and selected readelf command."""
        with tempfile.TemporaryDirectory(prefix="gpuxtb-wheel-test-") as directory:
            root = Path(directory)
            wheel = root / "gpuxtb-test.whl"
            marker = root / "checker-ran"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("gpuxtb/lib/libgpuxtb.so", b"\x7fELFtest")
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
        """Reject wheels containing ambiguous gpuxtb ELF payloads."""
        with tempfile.TemporaryDirectory(prefix="gpuxtb-wheel-test-") as directory:
            root = Path(directory)
            wheel = root / "gpuxtb-test.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("gpuxtb/lib/libgpuxtb.so", b"\x7fELFfirst")
                archive.writestr("gpuxtb/lib/libgpuxtb.so.1", b"\x7fELFsecond")
            with self.assertRaisesRegex(RuntimeError, "exactly one ELF"):
                WHEEL_INSPECTOR.inspect_wheel(
                    wheel,
                    checker=CHECKER_PATH,
                    readelf="readelf",
                    temporary_root=root / "extracted",
                )


if __name__ == "__main__":
    unittest.main()
