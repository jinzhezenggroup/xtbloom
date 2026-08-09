#!/usr/bin/env python3
"""Apply and validate the pinned tblite SCC observer patch safely.

The source checkout is used only as a local Git object source.  The requested
revision is checked out in a new temporary clone (or an explicitly requested
output directory), so uncommitted files in the user's tblite tree are never
patched in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

TOOL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOL_DIR.parents[2]
METADATA_PATH = TOOL_DIR / "metadata.json"
PROBE_PATH = TOOL_DIR / "observer_probe.f90"


class ObserverPatchError(RuntimeError):
    """Raised when provenance, patch application, or the probe is invalid."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and turn failures into concise oracle diagnostics."""
    completed = subprocess.run(
        [str(argument) for argument in arguments],
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=capture,
    )
    if completed.returncode != 0:
        command = shlex.join([str(argument) for argument in arguments])
        details = ""
        if capture:
            details = f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        raise ObserverPatchError(
            f"command failed with status {completed.returncode}: {command}{details}"
        )
    return completed


def load_metadata() -> dict[str, object]:
    """Load and minimally validate immutable patch metadata."""
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ObserverPatchError(f"cannot read {METADATA_PATH}: {error}") from error

    if metadata.get("schema") != "xtbloom-tblite-scc-observer-patch-v1":
        raise ObserverPatchError(
            "unsupported or missing observer patch metadata schema"
        )
    upstream = metadata.get("upstream")
    patch = metadata.get("patch")
    if not isinstance(upstream, dict) or not isinstance(patch, dict):
        raise ObserverPatchError("metadata must contain upstream and patch objects")
    revision = upstream.get("revision")
    expected_digest = patch.get("sha256")
    filename = patch.get("file")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ObserverPatchError("metadata upstream revision must be a full commit SHA")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ObserverPatchError("metadata patch SHA-256 is invalid")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ObserverPatchError("metadata patch file must be a local basename")
    return metadata


def patch_path(metadata: dict[str, object]) -> Path:
    """Resolve the patch named by validated metadata."""
    patch = metadata["patch"]
    assert isinstance(patch, dict)
    filename = patch["file"]
    assert isinstance(filename, str)
    return TOOL_DIR / filename


def validate_bundle(metadata: dict[str, object]) -> str:
    """Validate the committed patch hash, file list, and semantic anchors."""
    license_files = metadata.get("license_files")
    if not isinstance(license_files, list) or not license_files:
        raise ObserverPatchError("metadata must list the bundled license texts")
    for entry in license_files:
        if not isinstance(entry, dict):
            raise ObserverPatchError("license metadata entries must be objects")
        filename = entry.get("file")
        expected_digest = entry.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename.startswith("LICENSES/")
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
        ):
            raise ObserverPatchError("bundled license metadata is invalid")
        license_path = TOOL_DIR / filename
        if not license_path.is_file() or sha256_file(license_path) != expected_digest:
            raise ObserverPatchError(
                f"bundled license text is missing or modified: {filename}"
            )

    patch = metadata["patch"]
    assert isinstance(patch, dict)
    path = patch_path(metadata)
    if not path.is_file():
        raise ObserverPatchError(f"observer patch is missing: {path}")
    actual_digest = sha256_file(path)
    if actual_digest != patch["sha256"]:
        raise ObserverPatchError(
            f"observer patch SHA-256 mismatch: {actual_digest} != {patch['sha256']}"
        )

    text = path.read_text(encoding="utf-8")
    patched_paths = [
        line.removeprefix("+++ b/")
        for line in text.splitlines()
        if line.startswith("+++ b/")
    ]
    expected_paths = patch.get("modified_paths")
    if patched_paths != expected_paths:
        raise ObserverPatchError(
            f"patch path list differs from metadata: {patched_paths!r} != "
            f"{expected_paths!r}"
        )

    required_tokens = (
        "type, public :: scf_observer",
        "procedure :: before_solve => no_op_before_solve",
        "procedure :: after_iteration => no_op_after_iteration",
        "procedure :: finished => no_op_finished",
        "type(wavefunction_type), intent(in) :: wfn",
        "type(potential_type), intent(in) :: pot",
        "scf_observer_status_converged",
        "scf_observer_status_max_iterations",
        "scf_observer_status_failed",
        "if (present(observer)) call observer%before_solve(iscf, wfn, pot)",
        "call observer%after_iteration(observer_iteration, wfn, eelec, elast, &",
        "call observer%finished(iscf, scf_observer_status_failed)",
    )
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise ObserverPatchError(f"patch is missing semantic anchors: {missing!r}")
    if "type, public, abstract :: scf_observer" in text or "deferred" in text:
        raise ObserverPatchError("the observer base must remain concrete and no-op")
    return actual_digest


def source_state(source_root: Path) -> tuple[str, str]:
    """Capture source HEAD and porcelain state to prove it was not modified."""
    head = run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], capture=True
    ).stdout.strip()
    status = run(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        capture=True,
    ).stdout
    return head, status


def validate_output_location(source_root: Path, checkout: Path) -> None:
    """Reject retained clones which would dirty the read-only source checkout."""
    if checkout == source_root or checkout.is_relative_to(source_root):
        raise ObserverPatchError(
            "output directory must be outside the source tblite checkout"
        )


def clone_and_apply(
    source_root: Path,
    checkout: Path,
    metadata: dict[str, object],
) -> None:
    """Clone the pinned commit, apply the patch to the index, and check the diff."""
    upstream = metadata["upstream"]
    patch = metadata["patch"]
    assert isinstance(upstream, dict) and isinstance(patch, dict)
    revision = str(upstream["revision"])
    expected_paths = patch["modified_paths"]
    if not isinstance(expected_paths, list) or not all(
        isinstance(path, str) for path in expected_paths
    ):
        raise ObserverPatchError("metadata modified_paths must be a string array")

    if checkout.exists():
        raise ObserverPatchError(f"refusing to overwrite existing output: {checkout}")
    checkout.parent.mkdir(parents=True, exist_ok=True)

    run(
        [
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            "--no-checkout",
            str(source_root),
            str(checkout),
        ]
    )
    run(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", revision])
    resolved = run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture=True
    ).stdout.strip()
    if resolved != revision:
        raise ObserverPatchError(
            f"checkout resolved to {resolved}, expected {revision}"
        )

    path = patch_path(metadata)
    run(["git", "-C", str(checkout), "apply", "--check", str(path)])
    run(["git", "-C", str(checkout), "apply", "--index", str(path)])
    run(["git", "-C", str(checkout), "diff", "--cached", "--check"])
    run(["git", "-C", str(checkout), "apply", "--check", "--reverse", str(path)])
    actual_paths = run(
        ["git", "-C", str(checkout), "diff", "--cached", "--name-only"],
        capture=True,
    ).stdout.splitlines()
    if actual_paths != expected_paths:
        raise ObserverPatchError(
            f"applied path list differs from metadata: {actual_paths!r} != "
            f"{expected_paths!r}"
        )
    validate_applied_hooks(checkout)


def validate_applied_hooks(checkout: Path) -> None:
    """Check the exact hook ordering and read-only callback contract after apply."""
    iterator = (checkout / "src/tblite/scf/iterator.f90").read_text(encoding="utf-8")
    singlepoint = (checkout / "src/tblite/xtb/singlepoint.f90").read_text(
        encoding="utf-8"
    )
    observer = (checkout / "src/tblite/scf/observer.f90").read_text(encoding="utf-8")

    iterator_order = (
        iterator.index("call add_pot_to_h1"),
        iterator.index("call set_mixer"),
        iterator.index("call observer%before_solve"),
        iterator.index("call next_density"),
    )
    if list(iterator_order) != sorted(iterator_order):
        raise ObserverPatchError(
            "before_solve is not between assembly and density solve"
        )

    callback_order = (
        singlepoint.index("converged = econverged"),
        singlepoint.index("if (.not.allocated(error) .and. present(observer))"),
        singlepoint.index("call observer%after_iteration"),
        singlepoint.index(
            "if (prlevel > 0)", singlepoint.index("call observer%after_iteration")
        ),
    )
    if list(callback_order) != sorted(callback_order):
        raise ObserverPatchError(
            "after_iteration is not after convergence and before output"
        )

    if observer.count("type(wavefunction_type), intent(in) :: wfn") != 2:
        raise ObserverPatchError(
            "both callbacks must borrow wavefunction state read-only"
        )
    if observer.count("class(scf_observer), intent(inout) :: self") != 3:
        raise ObserverPatchError("observer state must be mutable only through self")
    if singlepoint.count("call observer%finished") != 4:
        raise ObserverPatchError(
            "all mixer and outer-loop terminal paths must report status"
        )


def probe_meson_project(lapack: str) -> str:
    """Return the disposable outer Meson project used to build the probe."""
    return f"""project(
  'xtbloom-tblite-observer-probe',
  'fortran',
  default_options: ['buildtype=release', 'default_library=static'],
)

tblite_project = subproject(
  'tblite',
  default_options: [
    'default_library=static',
    'openmp=false',
    'lapack={lapack}',
    'ddx=false',
    'hdf5=disabled',
    'trexio=disabled',
    'api=false',
    'python=false',
  ],
)
tblite_dep = tblite_project.get_variable('tblite_dep')

observer_probe = executable(
  'xtbloom-tblite-observer-probe',
  'observer_probe.f90',
  dependencies: tblite_dep,
)
test('xtbloom-tblite-observer-probe', observer_probe)
"""


def build_and_run_probe(
    checkout: Path,
    *,
    meson_command: Sequence[str],
    lapack: str,
    wrap_mode: str,
) -> None:
    """Build the patched tblite subproject and run the standalone H3+ probe."""
    with tempfile.TemporaryDirectory(
        prefix="xtbloom-tblite-observer-probe-"
    ) as directory:
        outer = Path(directory) / "outer"
        subprojects = outer / "subprojects"
        subprojects.mkdir(parents=True)
        os.symlink(checkout, subprojects / "tblite", target_is_directory=True)
        (outer / "meson.build").write_text(
            probe_meson_project(lapack), encoding="utf-8"
        )
        shutil.copy2(PROBE_PATH, outer / PROBE_PATH.name)

        environment = os.environ.copy()
        environment["LC_ALL"] = "C"
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["MKL_DYNAMIC"] = "FALSE"
        build = outer / "build"
        run(
            [
                *meson_command,
                "setup",
                str(build),
                f"--wrap-mode={wrap_mode}",
            ],
            cwd=outer,
            env=environment,
        )
        run(
            [
                *meson_command,
                "compile",
                "-C",
                str(build),
                "xtbloom-tblite-observer-probe",
            ],
            cwd=outer,
            env=environment,
        )
        run(
            [
                *meson_command,
                "test",
                "-C",
                str(build),
                "xtbloom-tblite-observer-probe",
                "--print-errorlogs",
            ],
            cwd=outer,
            env=environment,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse the public validation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="local tblite Git checkout used read-only as the clone source",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="leave the patched detached checkout here; the path must not exist",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="build patched tblite and run the standalone numerical probe",
    )
    parser.add_argument(
        "--meson-command",
        default=f"{shlex.quote(sys.executable)} -m mesonbuild.mesonmain",
        help="Meson command used with --probe",
    )
    parser.add_argument(
        "--lapack",
        default="auto",
        choices=("auto", "mkl", "mkl-rt", "openblas", "netlib", "custom"),
        help="tblite LAPACK backend used by the probe",
    )
    parser.add_argument(
        "--wrap-mode",
        default="forcefallback",
        choices=("default", "nofallback", "nodownload", "forcefallback", "nopromote"),
        help="Meson dependency wrap mode used by the probe",
    )
    return parser.parse_args()


def main() -> int:
    """Validate provenance, apply to a disposable clone, and optionally probe."""
    arguments = parse_arguments()
    source_root = arguments.source_root.resolve()
    if not (source_root / ".git").exists():
        raise ObserverPatchError(f"source root is not a Git checkout: {source_root}")
    if arguments.output_dir is not None:
        validate_output_location(source_root, arguments.output_dir.resolve())

    metadata = load_metadata()
    digest = validate_bundle(metadata)
    upstream = metadata["upstream"]
    assert isinstance(upstream, dict)
    revision = str(upstream["revision"])
    run(["git", "-C", str(source_root), "cat-file", "-e", f"{revision}^{{commit}}"])
    initial_source_state = source_state(source_root)

    def validate_checkout(checkout: Path) -> None:
        clone_and_apply(source_root, checkout, metadata)
        if arguments.probe:
            meson_command = shlex.split(arguments.meson_command)
            if not meson_command:
                raise ObserverPatchError("--meson-command cannot be empty")
            build_and_run_probe(
                checkout,
                meson_command=meson_command,
                lapack=arguments.lapack,
                wrap_mode=arguments.wrap_mode,
            )

    if arguments.output_dir is not None:
        checkout = arguments.output_dir.resolve()
        validate_checkout(checkout)
        retained = f"; patched checkout retained at {checkout}"
    else:
        with tempfile.TemporaryDirectory(
            prefix="xtbloom-tblite-observer-"
        ) as directory:
            validate_checkout(Path(directory) / "tblite")
        retained = ""

    if source_state(source_root) != initial_source_state:
        raise ObserverPatchError("the source tblite checkout changed during validation")
    probe = "; numerical probe passed" if arguments.probe else ""
    print(  # noqa: T201 - CLI validation report
        f"observer patch OK: revision={revision} sha256={digest}{probe}{retained}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ObserverPatchError as error:
        print(f"error: {error}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        raise SystemExit(1) from error
