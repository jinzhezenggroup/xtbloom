"""Regression tests for the release legal-material validator."""

from __future__ import annotations

import copy
import importlib.util
import io
import os
import shutil
import subprocess
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
VERSION_PROVIDER_PATH = REPOSITORY / "python" / "ci" / "xtbloom_version_provider.py"
VERSION_PROVIDER_SPEC = importlib.util.spec_from_file_location(
    "xtbloom_version_provider", VERSION_PROVIDER_PATH
)
assert VERSION_PROVIDER_SPEC is not None and VERSION_PROVIDER_SPEC.loader is not None
VERSION_PROVIDER = importlib.util.module_from_spec(VERSION_PROVIDER_SPEC)
VERSION_PROVIDER_SPEC.loader.exec_module(VERSION_PROVIDER)


class LicenseArchiveTests(unittest.TestCase):
    """Verify legal payload requirements for built distribution archives."""

    def _write_wheel(self, path: Path, names: set[str]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for name in sorted(names):
                if name.endswith("/provenance/implib_manifest.json"):
                    payload = (REPOSITORY / CHECKER.IMPLIB_MANIFEST_PATH).read_bytes()
                elif name.endswith("MPL-2.0.txt"):
                    payload = (REPOSITORY / CHECKER.MPL_LICENSE).read_bytes()
                else:
                    payload = b"test\n"
                archive.writestr(name, payload)

    def _valid_wheel_names(self) -> set[str]:
        return {
            f"{WHEEL_DIST_INFO}/licenses/{suffix}"
            for suffix in CHECKER.COMMON_ARCHIVE_SUFFIXES
        } | {f"xtbloom/{suffix}" for suffix in CHECKER.WHEEL_ARCHIVE_SUFFIXES}

    def test_project_license_cannot_be_satisfied_by_third_party_filename(self) -> None:
        """Require the project license at its exact archive location."""
        names = self._valid_wheel_names()
        names.remove(f"{WHEEL_DIST_INFO}/licenses/LICENSE")
        names.add("xtbloom/share/licenses/xtbloom/third-party/d4/mctc-lib-LICENSE")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "LICENSE"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_every_provenance_manifest(self) -> None:
        """Require all provenance manifests in wheel payloads."""
        names = self._valid_wheel_names()
        missing = "xtbloom/share/licenses/xtbloom/provenance/mctc_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "mctc_manifest"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_implib_provenance_manifest(self) -> None:
        """Require the vendored implib provenance manifest in wheels."""
        names = self._valid_wheel_names()
        missing = "xtbloom/share/licenses/xtbloom/provenance/implib_manifest.json"
        names.remove(missing)
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "implib_manifest"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_linking_exception(self) -> None:
        """Require the GPLv3 Section 7 exception in wheel archives."""
        names = self._valid_wheel_names()
        missing = f"{WHEEL_DIST_INFO}/licenses/CUDA_MKL_LINKING_EXCEPTION"
        names.remove(missing)
        names.remove("xtbloom/share/licenses/xtbloom/CUDA_MKL_LINKING_EXCEPTION")
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test.whl"
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
            wheel = Path(directory) / "xtbloom-test.whl"
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
            wheel = Path(directory) / "xtbloom-test.whl"
            self._write_wheel(wheel, names)
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "libcudart"):
                CHECKER.check_archive(wheel)

    def test_wheel_must_retain_exact_mpl_text(self) -> None:
        """Reject whitespace or other changes to the reviewed pathspec license."""
        names = self._valid_wheel_names()
        with tempfile.TemporaryDirectory(prefix="xtbloom-license-test-") as directory:
            wheel = Path(directory) / "xtbloom-test.whl"
            self._write_wheel(wheel, names)
            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr(
                    f"{WHEEL_DIST_INFO}/licenses/{CHECKER.MPL_LICENSE}",
                    b"modified\n",
                )
            with self.assertRaisesRegex(CHECKER.LicenseCheckError, "modified MPL"):
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
            '<a href="LICENSES/openchemlib-BSD-3-Clause.txt">OpenChemLib</a>\n'
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

    def test_openblas_must_use_reviewed_exact_version(self) -> None:
        """Require the exact scipy-openblas32 ABI reviewed for runtime loading."""
        project = copy.deepcopy(self.project)
        project["dependencies"] = [
            requirement.replace("==0.3.34.0.0", ">=0.3.34.0.0")
            if requirement.startswith("scipy-openblas32")
            else requirement
            for requirement in project["dependencies"]
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "reviewed exact"):
            CHECKER._require_dependency_policy(project)

    def test_cuda_extra_must_be_complete(self) -> None:
        """Require the complete reviewed CUDA provider set."""
        project = copy.deepcopy(self.project)
        project["optional-dependencies"]["cuda12"].pop()
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "reviewed NVIDIA"):
            CHECKER._require_dependency_policy(project)


class BuildDependencyPolicyTests(unittest.TestCase):
    """Keep isolated build inputs exact, reviewed, and complete."""

    def setUp(self) -> None:
        """Load the build-system table used by each policy mutation."""
        metadata = CHECKER.tomllib.loads(
            (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.build_system = metadata["build-system"]

    def test_current_build_dependency_policy_is_accepted(self) -> None:
        """Accept the fully pinned reviewed PEP 517 build environment."""
        CHECKER._require_build_dependency_policy(self.build_system)

    def test_build_dependency_must_remain_exactly_pinned(self) -> None:
        """Reject a range that lets versioning or build behavior drift."""
        build_system = copy.deepcopy(self.build_system)
        build_system["requires"] = [
            requirement.replace("==1.0.3", ">=1.0.3")
            if requirement.startswith("scikit-build-core")
            else requirement
            for requirement in build_system["requires"]
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "reviewed exact"):
            CHECKER._require_build_dependency_policy(build_system)

    def test_build_dependency_set_must_remain_complete(self) -> None:
        """Reject omission of a transitive package whose bytes enter builds."""
        build_system = copy.deepcopy(self.build_system)
        build_system["requires"] = [
            requirement
            for requirement in build_system["requires"]
            if not requirement.startswith("vcs-versioning")
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "reviewed exact"):
            CHECKER._require_build_dependency_policy(build_system)


class VersionMetadataPolicyTests(unittest.TestCase):
    """Keep every product version dependent on strict Git-tag metadata."""

    def setUp(self) -> None:
        """Load the complete project metadata used by each policy mutation."""
        self.metadata = CHECKER.tomllib.loads(
            (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        )

    def test_current_version_metadata_policy_is_accepted(self) -> None:
        """Accept the reviewed dynamic provider and strict tag grammar."""
        CHECKER._require_version_metadata_policy(self.metadata)

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

    def test_external_provider_cannot_replace_local_preflight(self) -> None:
        """Require the wrapper that rejects exact-tag shallow clones."""
        metadata = copy.deepcopy(self.metadata)
        metadata["tool"]["dynamic-metadata"] = [
            {"provider": "scikit_build_core.metadata.setuptools_scm"}
        ]
        with self.assertRaisesRegex(CHECKER.LicenseCheckError, "unreviewed"):
            CHECKER._require_version_metadata_policy(metadata)


class VersionProviderTests(unittest.TestCase):
    """Exercise strict Git and frozen-metadata boundaries of the provider."""

    def _run_git(self, root: Path, *arguments: str) -> str:
        """Run a deterministic local Git command for integration fixtures."""
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _make_malformed_tag_repository(self, root: Path) -> None:
        """Create a valid release tag followed by a nearer malformed v tag."""
        self._run_git(root, "init", "--quiet")
        self._run_git(root, "config", "user.name", "xTBloom version test")
        self._run_git(root, "config", "user.email", "version-test@example.invalid")
        (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
        (root / ".gitattributes").write_text(
            ".git_archival.txt export-subst\n", encoding="utf-8"
        )
        (root / ".git_archival.txt").write_text(
            "describe-name: $Format:%(describe:tags=true,abbrev=0,match=v*)$\n",
            encoding="utf-8",
        )
        self._run_git(root, "add", ".")
        self._run_git(root, "commit", "--quiet", "-m", "valid release")
        self._run_git(root, "tag", "v0.0.0")
        (root / "marker.txt").write_text("newer commit\n", encoding="utf-8")
        self._run_git(root, "add", "marker.txt")
        self._run_git(root, "commit", "--quiet", "-m", "malformed release tag")
        self._run_git(root, "tag", "v1.2")

    def test_exact_tag_shallow_clone_is_rejected(self) -> None:
        """Reject shallow history before setuptools-scm's exact-tag shortcut."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory).resolve()
            with (
                mock.patch.object(
                    VERSION_PROVIDER,
                    "_run_git",
                    side_effect=[str(root), "true"],
                ),
                self.assertRaisesRegex(RuntimeError, "shallow clone rejected"),
            ):
                VERSION_PROVIDER._git_tag_version(root)

    def test_dynamic_provider_rejects_nearer_malformed_v_tag(self) -> None:
        """Fail the PEP 517 provider instead of falling back to an older tag."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory).resolve()
            self._make_malformed_tag_repository(root)
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaisesRegex(RuntimeError, "strict vMAJOR.MINOR.PATCH"):
                    VERSION_PROVIDER.dynamic_metadata({}, {})
            finally:
                os.chdir(previous_directory)

    def test_git_archive_exposes_nearer_malformed_v_tag(self) -> None:
        """Make export-subst preserve the malformed tag for strict rejection."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory).resolve()
            self._make_malformed_tag_repository(root)
            archive_bytes = subprocess.run(
                ["git", "archive", "--format=tar", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
                archival_member = archive.extractfile(".git_archival.txt")
                assert archival_member is not None
                archival_text = archival_member.read().decode("utf-8")
            self.assertEqual(archival_text, "describe-name: v1.2\n")

            extracted = root / "archive"
            extracted.mkdir()
            (extracted / ".git_archival.txt").write_text(
                archival_text, encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "strict vMAJOR.MINOR.PATCH"):
                VERSION_PROVIDER._frozen_tag_version(extracted)

    def test_sdist_version_must_be_bare_tag_tuple(self) -> None:
        """Accept a frozen tag version and reject derived local metadata."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            (root / "PKG-INFO").write_text(
                "Metadata-Version: 2.4\nVersion: 1.2.3\n", encoding="utf-8"
            )
            self.assertEqual(VERSION_PROVIDER._frozen_tag_version(root), "1.2.3")
            (root / "PKG-INFO").write_text(
                "Metadata-Version: 2.4\nVersion: 1.2.3.post1\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "non-tag version"):
                VERSION_PROVIDER._frozen_tag_version(root)

    def test_git_archive_reads_only_expanded_strict_tag(self) -> None:
        """Require export-subst metadata to contain one strict release tag."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            archival = root / ".git_archival.txt"
            archival.write_text("describe-name: v9.8.7\n", encoding="utf-8")
            self.assertEqual(VERSION_PROVIDER._frozen_tag_version(root), "9.8.7")
            archival.write_text(
                "describe-name: $Format:%(describe:tags=true)$\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "cannot resolve"):
                VERSION_PROVIDER._frozen_tag_version(root)

    def test_leading_zero_version_is_rejected(self) -> None:
        """Prevent PEP 440 normalization from diverging from native strings."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-version-test-") as directory:
            root = Path(directory)
            (root / "PKG-INFO").write_text(
                "Metadata-Version: 2.4\nVersion: 01.2.3\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "non-tag version"):
                VERSION_PROVIDER._frozen_tag_version(root)


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
