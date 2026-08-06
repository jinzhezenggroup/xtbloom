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
        names = self._valid_wheel_names()
        names.remove("gpuxtb-0.1.0.dist-info/licenses/LICENSE")
        names.add("gpuxtb/share/licenses/gpuxtb/third-party/d4/mctc-lib-LICENSE")
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "LICENSE"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_every_provenance_manifest(self) -> None:
        names = self._valid_wheel_names()
        missing = "gpuxtb/share/licenses/gpuxtb/provenance/mctc_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "mctc_manifest"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_implib_provenance_manifest(self) -> None:
        names = self._valid_wheel_names()
        missing = "gpuxtb/share/licenses/gpuxtb/provenance/implib_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "implib_manifest"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_linking_exception(self) -> None:
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

    def test_wheel_must_not_bundle_vendor_library(self) -> None:
        names = self._valid_wheel_names()
        names.add("gpuxtb/lib/libcudart.so.12")
        with tempfile.TemporaryDirectory(prefix="gpuxtb-license-test-") as directory:
            wheel = Path(directory) / "gpuxtb-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "libcudart"):
                CHECKER.check_archive(wheel)


class DependencyPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        metadata = CHECKER.tomllib.loads(
            (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.project = metadata["project"]

    def test_current_dependency_policy_is_accepted(self) -> None:
        CHECKER._require_dependency_policy(self.project)

    def test_mkl_cannot_be_mandatory(self) -> None:
        project = copy.deepcopy(self.project)
        project["dependencies"].append("mkl; sys_platform == 'linux'")
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "mkl must not"):
            CHECKER._require_dependency_policy(project)

    def test_nvidia_provider_cannot_be_mandatory(self) -> None:
        project = copy.deepcopy(self.project)
        project["dependencies"].append(project["optional-dependencies"]["cuda12"][0])
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "confined to"):
            CHECKER._require_dependency_policy(project)

    def test_openblas_must_cover_both_linux_architectures(self) -> None:
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
        project = copy.deepcopy(self.project)
        project["optional-dependencies"]["cuda12"].pop()
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "reviewed NVIDIA"):
            CHECKER._require_dependency_policy(project)


class LinkingExceptionTests(unittest.TestCase):
    def test_current_exception_policy_is_accepted(self) -> None:
        CHECKER._require_exception_policy(REPOSITORY)

    def test_cudadevrt_cannot_be_added_without_review(self) -> None:
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
    def _copy_payload(self, root: Path) -> None:
        source = REPOSITORY / CHECKER.IMPLIB_VENDOR_PATH
        destination = root / CHECKER.IMPLIB_VENDOR_PATH
        destination.parent.mkdir(parents=True)
        shutil.copytree(source, destination)
        manifest = root / CHECKER.IMPLIB_MANIFEST_PATH
        manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / CHECKER.IMPLIB_MANIFEST_PATH, manifest)

    def test_vendored_tree_matches_pinned_deepmd_revision(self) -> None:
        CHECKER._check_implib_provenance(REPOSITORY)

    def test_unexpected_vendored_file_is_rejected(self) -> None:
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


class CudaWheelInspectionTests(unittest.TestCase):
    def test_wheel_payload_is_forwarded_to_cuda_abi_checker(self) -> None:
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
