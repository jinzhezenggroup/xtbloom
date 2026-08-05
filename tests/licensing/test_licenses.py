"""Regression tests for the release legal-material validator."""

from __future__ import annotations

import importlib.util
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


class LicenseArchiveTests(unittest.TestCase):
    def _write_wheel(self, path: Path, names: set[str]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for name in sorted(names):
                archive.writestr(name, "test\n")

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


if __name__ == "__main__":
    unittest.main()
