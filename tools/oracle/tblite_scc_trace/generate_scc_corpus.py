#!/usr/bin/env python3
"""Generate the pinned gpuxtb-scc-trace-v1 restricted corpus (issues #45, #48).

Consumes the immutable observer-patch bundle, builds the patched pinned tblite
oracle (the recorder executable) in a disposable outer Meson project, runs the
five restricted corpus cases, and serializes each raw recorder stream into the
canonical ``gpuxtb-scc-trace-v1`` JSON document with the canonical writer.  A
manifest records every trace hash plus complete oracle provenance (tblite
revision, every dependency revision, observer patch hash, recorder harness
hash, compiler identity, BLAS/LAPACK identity, exact command line, and the
relevant environment) and refuses to write any golden whose provenance cannot
be established.

Usage:

  python3 generate_scc_corpus.py --source-root /path/to/tblite --corpus-dir out/

  python3 generate_scc_corpus.py --source-root /path/to/tblite \
      --corpus-dir out/ --check   # verify existing goldens havehes only
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Import the reusable pinned provenance/build machinery from the observer
# patch validator so the two tools cannot drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpuxtb_scc_trace as writer
import validate_observer_patch as validator

TOOL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOL_DIR.parents[2]
RECORDER_PATH = TOOL_DIR / "scc_trace_recorder.f90"
MAIN_PATH = TOOL_DIR / "scc_trace_main.f90"

FORMAT = "gpuxtb-scc-trace-v1"
REVISION = "e9abc395b122018ed688aecb1c3a65cecaf97beb"
MANIFEST_SCHEMA = "gpuxtb-scc-trace-corpus-manifest-v1"
PATCH_PATH = TOOL_DIR / "tblite-e9abc395-scc-observer.patch"
PINNED_DEPENDENCIES = {
    "dftd4": "6e1f59c3f39d919a2dbef0601d2576727c8b30e8",
    "jonquil": "4d43ffea512977602f654ab10067fcddb3e3c107",
    "mctc-lib": "e9de066d89f250d1cfb6de3a33f0c27c0e2f855d",
    "mstore": "663245d739be0123da61c917e55116b0c3db4c74",
    "multicharge": "6a5d63f9e9e29dcf13cc47cc27f33bf9015681bf",
    "s-dftd3": "6f0b06fbfa8653a23ca55c453772ce3af4420706",
    "test-drive": "d16852743043963f294a5d9a3d5218e32c20ea7f",
    "toml-f": "51a26158c6d52bbc59cb482bdd13f00d7fd032a3",
}
DETERMINISTIC_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "LC_ALL": "C",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
ML = 1.0e-30  # not used; retained for clarity


class CorpusError(RuntimeError):
    """Raised when corpus generation violates a reproducibility contract."""


# One spec per restricted corpus case (issue #48).  Positions are in bohr and
# match the committed conformance inputs exactly.
CASES: dict[str, dict[str, object]] = {
    "h3_plus": {
        "atomic_numbers": [1, 1, 1],
        "positions": [
            [-0.47073898552969, 0.81534384004086, 0.0],
            [-0.47073898552969, -0.81534384004086, 0.0],
            [0.94147797105939, 0.0, 0.0],
        ],
        "molecular_charge": 1.0,
        "unpaired_electrons": 0,
        "temperature_kelvin": 300.0,
        "mixer_memory": 2,
        "mixer_damping": 0.4,
        "maximum_iterations": 100,
    },
    "ketene": {
        "atomic_numbers": [6, 1, 1, 6, 8],
        "positions": [
            [0.0, 0.0, 1.02601244470852],
            [-1.77896091850190, 0.0, 2.03205413122797],
            [1.77896091850190, 0.0, 2.03205413122797],
            [0.0, 0.0, -1.44607538708298],
            [0.0, 0.0, -3.64404532008145],
        ],
        "molecular_charge": 0.0,
        "unpaired_electrons": 0,
        "temperature_kelvin": 300.0,
        "mixer_memory": 2,
        "mixer_damping": 0.4,
        "maximum_iterations": 100,
    },
    "nenacl": {
        "atomic_numbers": [10, 11, 17],
        "positions": [
            [0.0, 0.0, -4.68834556988340],
            [0.0, 0.0, 0.09359792798420],
            [0.0, 0.0, 4.59474764189921],
        ],
        "molecular_charge": 0.0,
        "unpaired_electrons": 0,
        "temperature_kelvin": 300.0,
        "mixer_memory": 2,
        "mixer_damping": 0.4,
        "maximum_iterations": 100,
    },
    "water_one_pc_gamma999": {
        "atomic_numbers": [8, 1, 1],
        "positions": [
            [-2.75237178376284, 2.43247309226225, -0.01392519847964],
            [-0.93157260886974, 2.79621404458590, -0.01863384029005],
            [-3.43820531288547, 3.30583608421060, 1.42134539425148],
        ],
        "molecular_charge": 0.0,
        "unpaired_electrons": 0,
        "temperature_kelvin": 300.0,
        "mixer_memory": 2,
        "mixer_damping": 0.4,
        "maximum_iterations": 100,
        "point_charges": {
            "positions": [[2.75237178376284, -2.43247309226225, -0.01392519847964]],
            "charges": [-0.69645733],
            "gammas": [999.0],
        },
    },
    "water_dimer_6pc_hardness": {
        "atomic_numbers": [8, 1, 1, 8, 1, 1],
        "positions": [
            [-2.75237178376284, 2.43247309226225, -0.01392519847964],
            [-0.93157260886974, 2.79621404458590, -0.01863384029005],
            [-3.43820531288547, 3.30583608421060, 1.42134539425148],
            [-2.43247309226225, -2.75237178376284, 0.01392519847964],
            [-2.79621404458590, -0.93157260886974, 0.01863384029005],
            [-3.30583608421060, -3.43820531288547, -1.42134539425148],
        ],
        "molecular_charge": 0.0,
        "unpaired_electrons": 0,
        "temperature_kelvin": 300.0,
        "mixer_memory": 2,
        "mixer_damping": 0.4,
        "maximum_iterations": 100,
        "point_charges": {
            "positions": [
                [2.75237178376284, -2.43247309226225, -0.01392519847964],
                [0.93157260886974, -2.79621404458590, -0.01863384029005],
                [3.43820531288547, -3.30583608421060, 1.42134539425148],
                [2.43247309226225, 2.75237178376284, 0.01392519847964],
                [2.79621404458590, 0.93157260886974, 0.01863384029005],
                [3.30583608421060, 3.43820531288547, -1.42134539425148],
            ],
            "charges": [
                -0.69645733,
                0.36031084,
                0.33614649,
                -0.69645733,
                0.36031084,
                0.33614649,
            ],
            "gammas": [0.451896, 0.405771, 0.405771, 0.451896, 0.405771, 0.405771],
        },
    },
}


def column_major_to_rows(values: list[float], n: int) -> list[list[float]]:
    """Convert a Fortran column-major flat matrix into logical [row][column].

    The recorder streams Fortran memory order (column outer, row inner), while
    the v1 format documents matrices in logical [row][column] order.
    """
    rows = [[values[column * n + row] for column in range(n)] for row in range(n)]
    return rows


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_revisions(source_root: Path) -> dict[str, str]:
    """Require local Git object sources for every reviewed dependency pin.

    The checked-out dependency HEAD is deliberately irrelevant: the build
    clones the exact commits below into its disposable outer Meson project.
    This prevents a populated developer checkout from silently changing the
    oracle while still allowing one object store to serve several pinned
    revisions.
    """
    for name, commit in PINNED_DEPENDENCIES.items():
        directory = source_root / "subprojects" / name
        if not directory.is_dir() or not (directory / ".git").exists():
            raise CorpusError(f"missing local Git source for dependency {name}")
        result = subprocess.run(
            ["git", "-C", str(directory), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise CorpusError(
                f"dependency {name} does not contain pinned commit {commit}"
            )
    return dict(PINNED_DEPENDENCIES)


def probe_meson_project(lapack: str, custom_libraries: list[str]) -> str:
    """Return the disposable outer Meson project building the trace recorder."""
    custom_option = ""
    if custom_libraries:
        custom_option = "    'custom_libraries=" + ",".join(custom_libraries) + "',\n"
    return f"""project(
  'gpuxtb-scc-trace-oracle',
  'fortran',
  default_options: ['buildtype=release', 'default_library=static'],
)

tblite_project = subproject(
  'tblite',
  default_options: [
    'default_library=static',
    'openmp=false',
    'lapack={lapack}',
{custom_option}    'ddx=false',
    'hdf5=disabled',
    'trexio=disabled',
    'api=false',
    'python=false',
  ],
)
tblite_dep = tblite_project.get_variable('tblite_dep')

scc_trace = executable(
  'gpuxtb-tblite-scc-trace',
  'scc_trace_recorder.f90',
  'scc_trace_main.f90',
  dependencies: tblite_dep,
)
"""


def clone_pinned_dependencies(
    outer: Path, source_root: Path, revisions: dict[str, str]
) -> None:
    """Populate the outer Meson project with detached local dependency clones."""
    for name, commit in revisions.items():
        source = source_root / "subprojects" / name
        target = outer / "subprojects" / name
        validator.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-checkout",
                "--no-hardlinks",
                str(source),
                str(target),
            ],
            cwd=outer,
        )
        validator.run(["git", "checkout", "--quiet", "--detach", commit], cwd=target)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target,
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode != 0 or head.stdout.strip() != commit:
            raise CorpusError(f"failed to bind dependency {name} at {commit}")


def configured_subprojects(build: Path) -> set[str]:
    """Return every recursively configured Meson subproject name."""
    path = build / "meson-info" / "intro-projectinfo.json"
    try:
        project = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusError(
            f"cannot read Meson project introspection: {error}"
        ) from error

    names: set[str] = set()

    def collect(node: object) -> None:
        if not isinstance(node, dict):
            raise CorpusError(
                "Meson project introspection has an invalid subproject entry"
            )
        name = node.get("name")
        if not isinstance(name, str) or not name:
            raise CorpusError("Meson project introspection misses a subproject name")
        names.add(name)
        children = node.get("subprojects", [])
        if not isinstance(children, list):
            raise CorpusError(
                "Meson project introspection has invalid nested subprojects"
            )
        for child in children:
            collect(child)

    children = project.get("subprojects", [])
    if not isinstance(children, list):
        raise CorpusError("Meson project introspection has invalid subprojects")
    for child in children:
        collect(child)
    return names


def compiler_provenance(build: Path) -> dict[str, object]:
    """Record the compiler command and exact executable bytes selected by Meson."""
    path = build / "meson-info" / "intro-compilers.json"
    try:
        compilers = json.loads(path.read_text(encoding="utf-8"))
        compiler = compilers["host"]["fortran"]
        command = compiler["exelist"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise CorpusError(
            f"cannot read Meson Fortran compiler provenance: {error}"
        ) from error
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise CorpusError("Meson reported an invalid Fortran compiler command")

    executables = []
    for item in command:
        candidate = shutil.which(item)
        if candidate is None and Path(item).is_file():
            candidate = str(Path(item).resolve())
        if candidate is None:
            continue
        resolved = Path(candidate).resolve()
        record = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
        }
        if record not in executables:
            executables.append(record)
    if not executables:
        raise CorpusError(
            "cannot resolve any executable in the Fortran compiler command"
        )
    identifier = compiler.get("id")
    version = compiler.get("version")
    if not isinstance(identifier, str) or not identifier:
        raise CorpusError("Meson did not report the Fortran compiler identifier")
    if not isinstance(version, str) or not version:
        raise CorpusError("Meson did not report the Fortran compiler version")
    return {
        "command": command,
        "executables": executables,
        "id": identifier,
        "version": version,
    }


def blas_lapack_provenance(
    binary: Path, requested: str, custom_libraries: list[str]
) -> dict[str, object]:
    """Record the resolved BLAS/LAPACK provider and linked library bytes."""
    result = subprocess.run(
        ["ldd", str(binary)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise CorpusError(
            f"ldd failed for the oracle executable: {result.stderr.strip()}"
        )
    if "not found" in result.stdout:
        raise CorpusError("the oracle executable has an unresolved dynamic library")

    linked: dict[str, Path] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("linux-vdso"):
            continue
        if "=>" in line:
            soname, remainder = (part.strip() for part in line.split("=>", 1))
            token = remainder.split()[0]
        else:
            token = line.split()[0]
            soname = Path(token).name
        candidate = Path(token)
        if not candidate.is_absolute() or not candidate.exists():
            continue
        linked[soname] = candidate.resolve()

    custom_tokens = [name.lower().removeprefix("lib") for name in custom_libraries]

    def is_provider(soname: str) -> bool:
        lowered = soname.lower()
        return any(token in lowered for token in ("blas", "lapack", "mkl")) or any(
            token and token in lowered for token in custom_tokens
        )

    libraries = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "soname": soname,
        }
        for soname, path in sorted(linked.items())
        if is_provider(soname)
    ]
    if not libraries:
        raise CorpusError("cannot identify the resolved BLAS/LAPACK provider library")

    sonames = [str(entry["soname"]).lower() for entry in libraries]
    if any("mkl_rt" in name for name in sonames):
        resolved_provider = "mkl-rt"
    elif any("mkl" in name for name in sonames):
        resolved_provider = "mkl"
    elif any("openblas" in name for name in sonames):
        resolved_provider = "openblas"
    elif any("lapack" in name for name in sonames) and any(
        "blas" in name for name in sonames
    ):
        resolved_provider = "netlib"
    elif requested == "custom":
        resolved_provider = "custom"
    else:
        raise CorpusError("cannot classify the resolved BLAS/LAPACK provider")
    return {
        "libraries": libraries,
        "requested": requested,
        "resolved_provider": resolved_provider,
    }


def build_oracle(
    checkout: Path,
    source_root: Path,
    *,
    dependency_pins: dict[str, str],
    meson_command: list[str],
    lapack: str,
    wrap_mode: str,
    custom_libraries: list[str],
) -> tuple[Path, dict[str, object]]:
    """Build the recorder oracle and return its executable and toolchain.

    Every fallback is a detached local clone at the manifest's exact commit.
    Meson is forbidden to download and introspection proves that those
    subprojects, rather than system packages or wrap revisions, were configured.
    """
    if wrap_mode != "nodownload":
        raise CorpusError("oracle generation requires --wrap-mode=nodownload")
    with tempfile.TemporaryDirectory(prefix="gpuxtb-scc-trace-oracle-") as directory:
        outer = Path(directory) / "outer"
        subprojects = outer / "subprojects"
        subprojects.mkdir(parents=True)
        os.symlink(checkout, subprojects / "tblite", target_is_directory=True)
        (outer / "meson.build").write_text(
            probe_meson_project(lapack, custom_libraries), encoding="utf-8"
        )
        shutil.copy2(RECORDER_PATH, outer / "scc_trace_recorder.f90")
        shutil.copy2(MAIN_PATH, outer / "scc_trace_main.f90")
        clone_pinned_dependencies(outer, source_root, dependency_pins)

        environment = os.environ.copy()
        environment.update(DETERMINISTIC_ENVIRONMENT)
        build = outer / "build"
        forced = ",".join(["tblite", *dependency_pins])
        command = [
            *meson_command,
            "setup",
            str(build),
            f"--wrap-mode={wrap_mode}",
            f"--force-fallback-for={forced}",
        ]
        validator.run(command, cwd=outer, env=environment)
        expected_subprojects = {"tblite", *dependency_pins}
        actual_subprojects = configured_subprojects(build)
        if actual_subprojects != expected_subprojects:
            raise CorpusError(
                "Meson configured subprojects "
                f"{sorted(actual_subprojects)}, expected {sorted(expected_subprojects)}"
            )
        validator.run(
            [*meson_command, "compile", "-C", str(build), "gpuxtb-tblite-scc-trace"],
            cwd=outer,
            env=environment,
        )
        binary_path = build / "gpuxtb-tblite-scc-trace"
        if not binary_path.is_file():
            raise CorpusError("recorder oracle executable was not produced")
        meson_version = subprocess.run(
            [*meson_command, "--version"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        if meson_version.returncode != 0 or not meson_version.stdout.strip():
            raise CorpusError("cannot record the Meson version")
        toolchain = {
            "blas_lapack": blas_lapack_provenance(
                binary_path, lapack, custom_libraries
            ),
            "compiler": compiler_provenance(build),
            "meson": {
                "command": meson_command,
                "version": meson_version.stdout.strip(),
            },
        }
        # Copy out so the temp dir teardown cannot invalidate the binary.
        out = checker_cache(binary_path)
        return out, toolchain


# A tiny flyweight cache: only the last binary is kept; callers must consume it
# immediately.  This is intentionally single-shot.
_binary_cache: tuple[Path, Path] | None = None


def checker_cache(path: Path) -> Path:
    """Copy the built recorder into a stable cache location and return its path."""
    global _binary_cache
    cache_dir = Path(tempfile.gettempdir()) / "gpuxtb-scc-trace-oracle-bin"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "gpuxtb-tblite-scc-trace"
    shutil.copy2(path, target)
    _binary_cache = (path, target)
    return target


def serialize_spec(spec: dict[str, object]) -> str:
    """Return the canonical fixed-layout case spec consumed by the recorder."""
    atomic_numbers = spec["atomic_numbers"]
    positions = spec["positions"]
    lines = [str(len(atomic_numbers)), *(str(number) for number in atomic_numbers)]
    for position in positions:
        lines.extend(repr(float(value)) for value in position)
    lines.append(repr(float(spec["molecular_charge"])))
    lines.append(str(int(spec["unpaired_electrons"])))
    lines.append(repr(float(spec["temperature_kelvin"])))
    lines.append(str(int(spec["mixer_memory"])))
    lines.append(repr(float(spec["mixer_damping"])))
    lines.append(str(int(spec["maximum_iterations"])))
    point_charges = spec.get("point_charges")
    if point_charges is None:
        lines.append("0")
    else:
        positions_pc = point_charges["positions"]
        lines.append(str(len(positions_pc)))
        for index in range(len(positions_pc)):
            values = [
                *positions_pc[index],
                point_charges["charges"][index],
                point_charges["gammas"][index],
            ]
            lines.extend(repr(float(value)) for value in values)
    return "\n".join(lines) + "\n"


def write_spec(spec: dict[str, object], path: Path) -> None:
    """Write a canonical fixed-layout case spec consumed by the recorder."""
    path.write_text(serialize_spec(spec), encoding="utf-8")


def run_recorder(binary: Path, spec: Path, workdir: Path) -> str:
    """Run the recorder on one case and return its raw stdout stream."""
    result = subprocess.run(
        [str(binary), str(spec)],
        cwd=workdir,
        env={**os.environ, **DETERMINISTIC_ENVIRONMENT},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CorpusError(
            f"recorder failed with status {result.returncode}:\n{result.stderr}"
        )
    return result.stdout


def canonicalize(raw: str, spec: dict[str, object], command_line: str) -> dict:
    """Convert a raw recorder stream to a validated canonical trace document.

    Raises writer.TraceError on structural mismatch; the caller turns it into a
    CorpusError.  Complete document assembly uses the canonical writer's field
    semantics only; this function parses the fixed raw layout.
    """
    lines = raw.splitlines()
    if not lines:
        raise CorpusError("raw stream is empty")
    header = lines[0].split()
    tokens = [token for line in lines[1:] for token in line.split()]
    iterator = iter(tokens)

    def take() -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise CorpusError("raw stream ended prematurely") from None

    if (
        len(header) not in (10, 12)
        or header[0] != "nat"
        or header[2] != "nsh"
        or header[4] != "nao"
        or header[6] != "niterations"
        or header[8] != "terminal"
    ):
        raise CorpusError("raw stream header is malformed")
    nat, nsh, nao = int(header[1]), int(header[3]), int(header[5])
    niterations, terminal = int(header[7]), int(header[9])
    failed_attempt_present = False
    if len(header) == 12:
        if header[10] != "failed_attempt" or header[11] not in ("0", "1"):
            raise CorpusError("raw stream failed_attempt header is malformed")
        failed_attempt_present = header[11] == "1"

    def floats(count: int) -> list[float]:
        return [float(take()) for _ in range(count)]

    def ints(count: int) -> list[int]:
        return [int(take()) for _ in range(count)]

    atomic_numbers: list[int] = []
    label = take()
    assert label == "atomic_numbers", label
    atomic_numbers = ints(nat)
    label = take()
    assert label == "positions", label
    positions = floats(3 * nat)
    label = take()
    assert label == "molecular_charge", label
    molecular_charge = float(take())
    label = take()
    assert label == "unpaired_electrons", label
    unpaired = int(take())
    label = take()
    assert label == "temperature", label
    temperature = float(take())
    label = take()
    assert label == "n_point_charges", label
    npc = int(take())
    point_charges: list[list[float]] = []
    if npc > 0:
        for _ in range(npc):
            row = floats(5)
            point_charges.append(row)
    label = take()
    assert label == "atom_to_shell_count", label
    shell_counts = ints(nat)
    if sum(shell_counts) != nsh:
        raise CorpusError("atom_to_shell_count does not sum to n_shells")

    label = take()
    assert label == "overlap", label
    overlap = column_major_to_rows(floats(nao * nao), nao)
    label = take()
    assert label == "core_hamiltonian", label
    core_hamiltonian = column_major_to_rows(floats(nao * nao), nao)

    def matrix(values: list[float]) -> list[float]:
        # logical [1][row][col]; raw is column-major rows as emitted.
        return column_major_to_rows(values, nao)

    def multipoles(atoms: int, components: int) -> list[list[list[float]]]:
        flat = floats(atoms * components)
        return [
            [flat[atom * components + component] for component in range(components)]
            for atom in range(atoms)
        ]

    iterations = []
    for iteration_offset in range(niterations):
        label = take()
        assert label == "iteration", label
        iteration_index = int(take())
        assert iteration_index == iteration_offset + 1, iteration_index
        label = take()
        assert label == "hamiltonian", label
        hamiltonian = matrix(floats(nao * nao))
        label = take()
        assert label == "eigenvalues", label
        eigenvalues = floats(nao)
        label = take()
        assert label == "occupations", label
        occupations_alpha = floats(nao)
        occupations_beta = floats(nao)
        label = take()
        assert label == "density", label
        density = matrix(floats(nao * nao))

        label = take()
        assert label == "mixed_qsh", label
        mixed_qsh = floats(nsh)
        label = take()
        assert label == "mixed_qat", label
        mixed_qat = floats(nat)
        label = take()
        assert label == "mixed_dipoles", label
        mixed_dipoles = multipoles(nat, 3)
        label = take()
        assert label == "mixed_quadrupoles", label
        mixed_quadrupoles = multipoles(nat, 6)
        label = take()
        assert label == "raw_qsh", label
        raw_qsh = floats(nsh)
        label = take()
        assert label == "raw_qat", label
        raw_qat = floats(nat)
        label = take()
        assert label == "raw_dipoles", label
        raw_dipoles = multipoles(nat, 3)
        label = take()
        assert label == "raw_quadrupoles", label
        raw_quadrupoles = multipoles(nat, 6)

        point_charge_shell_potential = None
        point_charge_energy = None
        if npc > 0:
            label = take()
            assert label == "point_charge_shell_potential", label
            point_charge_shell_potential = floats(nsh)
            label = take()
            assert label == "point_charge_energy", label
            point_charge_energy = floats(nsh)
        label = take()
        assert label == "energy", label
        energy = float(take())
        label = take()
        assert label == "energy_delta", label
        energy_delta = float(take())
        label = take()
        assert label == "residual_rms", label
        residual_rms = float(take())
        label = take()
        assert label == "convergence", label
        conv = [bool(int(take())) for _ in range(4)]

        def flatten_multipoles(multipole_blocks: list[list[float]]) -> list[float]:
            return [value for atom in multipole_blocks for value in atom]

        residual = [
            raw_value - mixed_value
            for raw_value, mixed_value in zip(
                [
                    *raw_qsh,
                    *flatten_multipoles(raw_dipoles),
                    *flatten_multipoles(raw_quadrupoles),
                ],
                [
                    *mixed_qsh,
                    *flatten_multipoles(mixed_dipoles),
                    *flatten_multipoles(mixed_quadrupoles),
                ],
                strict=True,
            )
        ]
        reconstructed_rms = math.sqrt(
            sum(value * value / len(residual) for value in residual)
        )
        if not math.isclose(
            reconstructed_rms, residual_rms, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise CorpusError(
                "raw residual RMS does not reconstruct from raw-minus-mixed"
            )

        iterations.append(
            {
                "index": len(iterations) + 1,
                "hamiltonian": [hamiltonian],
                "eigenvalues": [eigenvalues],
                "occupations": [occupations_alpha, occupations_beta],
                "density": [density],
                "mixed_qsh": [mixed_qsh],
                "raw_qsh": [raw_qsh],
                "mixed_qat": [mixed_qat],
                "raw_qat": [raw_qat],
                "mixed_dipoles": [mixed_dipoles],
                "raw_dipoles": [raw_dipoles],
                "mixed_quadrupoles": [mixed_quadrupoles],
                "raw_quadrupoles": [raw_quadrupoles],
                "residual": residual,
                "residual_rms": residual_rms,
                "point_charge_shell_potential": (
                    [point_charge_shell_potential]
                    if point_charge_shell_potential is not None
                    else None
                ),
                "point_charge_energy": (
                    [point_charge_energy] if point_charge_energy is not None else None
                ),
                "energy": energy,
                "energy_delta": energy_delta,
                "convergence": {
                    "energy": conv[0],
                    "population": conv[1],
                    "temperature": conv[2],
                    "overall": conv[3],
                },
            }
        )
        iterations[-1] = {
            key: value for key, value in iterations[-1].items() if value is not None
        }

    failed_attempt = None
    if failed_attempt_present:
        label = take()
        assert label == "failed_attempt", label
        failed_index = int(take())
        if failed_index != len(iterations) + 1:
            raise CorpusError(
                "failed attempt index does not follow completed iterations"
            )
        label = take()
        assert label == "hamiltonian", label
        failed_hamiltonian = matrix(floats(nao * nao))
        label = take()
        assert label == "mixed_qsh", label
        failed_mixed_qsh = floats(nsh)
        label = take()
        assert label == "mixed_qat", label
        failed_mixed_qat = floats(nat)
        label = take()
        assert label == "mixed_dipoles", label
        failed_mixed_dipoles = multipoles(nat, 3)
        label = take()
        assert label == "mixed_quadrupoles", label
        failed_mixed_quadrupoles = multipoles(nat, 6)
        failed_attempt = {
            "index": failed_index,
            "hamiltonian": [failed_hamiltonian],
            "mixed_qsh": [failed_mixed_qsh],
            "mixed_qat": [failed_mixed_qat],
            "mixed_dipoles": [failed_mixed_dipoles],
            "mixed_quadrupoles": [failed_mixed_quadrupoles],
        }

    converged = terminal == 1
    trace = {
        "format": FORMAT,
        "provenance": {
            "tblite_revision": REVISION,
            "oracle_patch_sha256": validator.sha256_file(PATCH_PATH),
            "oracle_command": command_line,
        },
        "input": {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "molecular_charge": molecular_charge,
            "unpaired_electrons": unpaired,
            "spin_channels": 1,
            "temperature": temperature,
        },
        "basis": {
            "n_atoms": nat,
            "n_shells": nsh,
            "nao": nao,
            "atom_to_shell_count": shell_counts,
        },
        "statics": {
            "overlap": [overlap],
            "core_hamiltonian": [core_hamiltonian],
        },
        "residual_layout": {
            "shell_charges": nsh,
            "atomic_dipoles": 3 * nat,
            "atomic_quadrupoles": 6 * nat,
        },
        "iterations": iterations,
        "terminal": {
            "status": terminal,
            "converged": converged,
            "iterations": niterations + (1 if failed_attempt is not None else 0),
        },
    }
    if failed_attempt is not None:
        trace["failed_attempt"] = failed_attempt
    if point_charges:
        trace["input"]["point_charges"] = {
            "positions": [value for row in point_charges for value in row[:3]],
            "charges": [row[3] for row in point_charges],
            "hardnesses": [row[4] for row in point_charges],
        }
    return trace


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the corpus generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "local tblite Git checkout used read-only (must be at the pinned revision)"
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="directory receiving canonical traces and the manifest",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing goldens' hashes and provenance without rebuilding",
    )
    parser.add_argument(
        "--meson-command",
        type=str,
        default=f"{shlex.quote(sys.executable)} -m mesonbuild.mesonmain",
        help="Meson command used to build the oracle",
    )
    parser.add_argument(
        "--lapack",
        type=str,
        default="auto",
        choices=("auto", "mkl", "mkl-rt", "openblas", "netlib", "custom"),
        help="tblite LAPACK backend used by the oracle",
    )
    parser.add_argument(
        "--custom-library",
        action="append",
        default=[],
        help="library name passed to tblite when --lapack=custom (repeatable)",
    )
    parser.add_argument(
        "--wrap-mode",
        type=str,
        default="nodownload",
        choices=("nodownload",),
        help="offline Meson dependency wrap mode (only nodownload is reproducible)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate or verify the pinned restricted corpus with full provenance."""
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = build_parser().parse_args(raw_arguments)

    case_dir = arguments.corpus_dir
    if arguments.check:
        return verify_manifest(case_dir)

    if arguments.lapack == "custom" and not arguments.custom_library:
        print(  # noqa: T201 - CLI diagnostics
            "ERROR: --lapack=custom requires at least one --custom-library",
            file=sys.stderr,
        )
        return 2
    if arguments.lapack != "custom" and arguments.custom_library:
        print(  # noqa: T201 - CLI diagnostics
            "ERROR: --custom-library is valid only with --lapack=custom",
            file=sys.stderr,
        )
        return 2
    if any(
        re.fullmatch(r"[A-Za-z0-9_.+-]+", name) is None
        for name in arguments.custom_library
    ):
        print("ERROR: invalid custom library name", file=sys.stderr)  # noqa: T201
        return 2

    metadata = validator.load_metadata()
    validator.validate_bundle(metadata)
    revision = str(metadata["upstream"]["revision"])
    if revision != REVISION:
        print(  # noqa: T201 - CLI diagnostics
            f"ERROR: oracle revision {revision} != expected {REVISION}", file=sys.stderr
        )
        return 2

    _, source_status = validator.source_state(arguments.source_root)
    # The source checkout may sit on any branch; what matters is that the
    # pinned revision is reachable and the tree is clean so the clone is a
    # byte-faithful basis for the oracle.
    probe = subprocess.run(
        [
            "git",
            "-C",
            str(arguments.source_root),
            "cat-file",
            "-e",
            f"{REVISION}^{{commit}}",
        ],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        print(  # noqa: T201 - CLI diagnostics
            "ERROR: pinned revision "
            f"{REVISION} is not reachable from the source checkout",
            file=sys.stderr,
        )
        return 2
    if source_status.strip():
        print(  # noqa: T201 - CLI diagnostics
            "ERROR: source checkout is dirty; provenance cannot be established",
            file=sys.stderr,
        )
        return 2

    deps = dependency_revisions(arguments.source_root)

    # The recorded command must be independent of the output directory so that
    # regenerating into a different corpus path is byte-identical.  The
    # corpus-dir argument is redacted to a fixed token.
    redacted_command: list[str] = []
    skip_next = False
    for argument in raw_arguments:
        if skip_next:
            redacted_command.append("<corpus-dir>")
            skip_next = False
            continue
        if argument == "--corpus-dir":
            redacted_command.append(argument)
            skip_next = True
            continue
        if argument.startswith("--corpus-dir="):
            redacted_command.append("--corpus-dir=<corpus-dir>")
            continue
        redacted_command.append(argument)
    command_line = shlex.join(["generate_scc_corpus.py", *redacted_command])
    case_dir = arguments.corpus_dir
    case_dir.mkdir(parents=True, exist_ok=True)

    # Build the oracle once and run every case.
    with tempfile.TemporaryDirectory(prefix="gpuxtb-scc-corpus-work-") as directory:
        work = Path(directory)
        checkout = work / "checkout"
        validator.clone_and_apply(arguments.source_root, checkout, metadata)
        binary, toolchain = build_oracle(
            checkout,
            arguments.source_root,
            dependency_pins=deps,
            meson_command=shlex.split(arguments.meson_command),
            lapack=arguments.lapack,
            wrap_mode=arguments.wrap_mode,
            custom_libraries=arguments.custom_library,
        )

        entries = {}
        specs_dir = case_dir / "specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        for case_id, spec in CASES.items():
            print(  # noqa: T201 - CLI progress
                f"[corpus] running {case_id}", file=sys.stderr
            )
            spec_path = work / f"{case_id}.spec"
            write_spec(spec, spec_path)
            raw = run_recorder(binary, spec_path, work)
            try:
                trace = canonicalize(raw, spec, command_line)
                canonical = writer.dumps(trace)
            except (writer.TraceError, CorpusError) as error:
                print(  # noqa: T201 - CLI diagnostics
                    f"ERROR: {case_id}: {error}", file=sys.stderr
                )
                return 2
            document = {**trace, "case_id": case_id}
            # Rebuild without case_id, then include in output document.
            del document["case_id"]
            output_path = case_dir / f"{case_id}.json"
            output_path.write_text(canonical, encoding="utf-8")
            committed_spec_path = specs_dir / f"{case_id}.spec"
            shutil.copy2(spec_path, committed_spec_path)
            entries[case_id] = {
                "path": f"{case_id}.json",
                "sha256": sha256_file(output_path),
                "spec_path": f"specs/{case_id}.spec",
                "spec_sha256": sha256_file(committed_spec_path),
            }

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "format": FORMAT,
        "revision": REVISION,
        "oracle_patch_sha256": validator.sha256_file(PATCH_PATH),
        "oracle_sources": {
            RECORDER_PATH.name: sha256_file(RECORDER_PATH),
            MAIN_PATH.name: sha256_file(MAIN_PATH),
        },
        "toolchain": toolchain,
        "dependencies": deps,
        "environment": DETERMINISTIC_ENVIRONMENT,
        "command": command_line,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(),
        "cases": entries,
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(  # noqa: T201 - CLI report
        f"generated {len(entries)} restricted goldens under {case_dir}"
    )
    return 0


def require_manifest(condition: bool, message: str) -> None:
    """Raise a corpus error when a manifest invariant is not satisfied."""
    if not condition:
        raise CorpusError(message)


def require_sha256(value: object, field: str) -> str:
    """Return a validated lowercase SHA-256 manifest value."""
    require_manifest(
        isinstance(value, str) and LOWER_HEX_64.fullmatch(value) is not None,
        f"{field} must be a lowercase SHA-256 digest",
    )
    return value


def validate_manifest_document(manifest: object, case_dir: Path) -> None:
    """Validate the complete offline corpus and its repository-owned provenance."""
    require_manifest(isinstance(manifest, dict), "manifest root must be an object")
    expected_fields = {
        "cases",
        "command",
        "dependencies",
        "environment",
        "format",
        "generated_at",
        "host",
        "oracle_patch_sha256",
        "oracle_sources",
        "revision",
        "schema",
        "toolchain",
    }
    require_manifest(
        set(manifest) == expected_fields,
        f"manifest fields {sorted(manifest)} do not match {sorted(expected_fields)}",
    )
    require_manifest(manifest["schema"] == MANIFEST_SCHEMA, "manifest schema mismatch")
    require_manifest(manifest["format"] == FORMAT, "manifest format mismatch")
    require_manifest(manifest["revision"] == REVISION, "manifest revision mismatch")

    patch_digest = require_sha256(
        manifest["oracle_patch_sha256"], "oracle_patch_sha256"
    )
    require_manifest(
        patch_digest == validator.sha256_file(PATCH_PATH),
        "oracle patch digest does not match the current patch bytes",
    )
    sources = manifest["oracle_sources"]
    expected_sources = {
        RECORDER_PATH.name: sha256_file(RECORDER_PATH),
        MAIN_PATH.name: sha256_file(MAIN_PATH),
    }
    require_manifest(isinstance(sources, dict), "oracle_sources must be an object")
    for name, digest in sources.items():
        require_sha256(digest, f"oracle_sources.{name}")
    require_manifest(
        sources == expected_sources,
        "oracle source digests do not match the compiled repository bytes",
    )

    dependencies = manifest["dependencies"]
    require_manifest(isinstance(dependencies, dict), "dependencies must be an object")
    for name, revision in dependencies.items():
        require_manifest(
            isinstance(name, str)
            and isinstance(revision, str)
            and LOWER_HEX_40.fullmatch(revision) is not None,
            f"dependency {name!r} is not pinned to a lowercase commit",
        )
    require_manifest(
        dependencies == PINNED_DEPENDENCIES,
        "dependency pins do not match the reviewed oracle inputs",
    )
    require_manifest(
        manifest["environment"] == DETERMINISTIC_ENVIRONMENT,
        "deterministic oracle environment mismatch",
    )
    command = manifest["command"]
    require_manifest(isinstance(command, str) and command, "command must be nonempty")
    host = manifest["host"]
    require_manifest(isinstance(host, str) and host, "host must be nonempty")
    generated_at = manifest["generated_at"]
    require_manifest(isinstance(generated_at, str), "generated_at must be a string")
    try:
        timestamp = datetime.datetime.fromisoformat(generated_at)
    except ValueError as error:
        raise CorpusError(f"generated_at is not ISO-8601: {error}") from error
    require_manifest(
        timestamp.tzinfo is not None, "generated_at must include a timezone"
    )

    toolchain = manifest["toolchain"]
    require_manifest(
        isinstance(toolchain, dict)
        and set(toolchain) == {"blas_lapack", "compiler", "meson"},
        "toolchain fields are incomplete",
    )
    compiler = toolchain["compiler"]
    require_manifest(isinstance(compiler, dict), "compiler must be an object")
    require_manifest(
        set(compiler) == {"command", "executables", "id", "version"},
        "compiler fields are incomplete",
    )
    require_manifest(
        isinstance(compiler["command"], list)
        and compiler["command"]
        and all(isinstance(item, str) and item for item in compiler["command"]),
        "compiler command is invalid",
    )
    require_manifest(
        isinstance(compiler["id"], str)
        and bool(compiler["id"])
        and isinstance(compiler["version"], str)
        and bool(compiler["version"]),
        "compiler identity is incomplete",
    )
    executables = compiler["executables"]
    require_manifest(
        isinstance(executables, list) and executables,
        "compiler executable records are missing",
    )
    for index, executable in enumerate(executables):
        require_manifest(
            isinstance(executable, dict) and set(executable) == {"path", "sha256"},
            f"compiler executable {index} fields are invalid",
        )
        require_manifest(
            isinstance(executable["path"], str)
            and Path(executable["path"]).is_absolute(),
            f"compiler executable {index} path must be absolute",
        )
        require_sha256(executable["sha256"], f"compiler.executables[{index}].sha256")

    meson = toolchain["meson"]
    require_manifest(
        isinstance(meson, dict) and set(meson) == {"command", "version"},
        "Meson provenance fields are incomplete",
    )
    require_manifest(
        isinstance(meson["command"], list)
        and meson["command"]
        and all(isinstance(item, str) and item for item in meson["command"])
        and isinstance(meson["version"], str)
        and bool(meson["version"]),
        "Meson identity is incomplete",
    )

    blas_lapack = toolchain["blas_lapack"]
    require_manifest(isinstance(blas_lapack, dict), "blas_lapack must be an object")
    require_manifest(
        set(blas_lapack) == {"libraries", "requested", "resolved_provider"},
        "BLAS/LAPACK provenance fields are incomplete",
    )
    require_manifest(
        isinstance(blas_lapack["requested"], str)
        and blas_lapack["requested"]
        in {"auto", "mkl", "mkl-rt", "openblas", "netlib", "custom"},
        "requested BLAS/LAPACK provider is invalid",
    )
    require_manifest(
        isinstance(blas_lapack["resolved_provider"], str)
        and blas_lapack["resolved_provider"]
        in {"mkl", "mkl-rt", "openblas", "netlib", "custom"},
        "resolved BLAS/LAPACK provider is invalid",
    )
    libraries = blas_lapack["libraries"]
    require_manifest(
        isinstance(libraries, list) and libraries,
        "resolved BLAS/LAPACK libraries are missing",
    )
    seen_sonames: set[str] = set()
    for index, library in enumerate(libraries):
        require_manifest(
            isinstance(library, dict) and set(library) == {"path", "sha256", "soname"},
            f"BLAS/LAPACK library {index} fields are invalid",
        )
        require_manifest(
            isinstance(library["path"], str) and Path(library["path"]).is_absolute(),
            f"BLAS/LAPACK library {index} path must be absolute",
        )
        soname = library["soname"]
        require_manifest(
            isinstance(soname, str) and soname and soname not in seen_sonames,
            f"BLAS/LAPACK library {index} SONAME is invalid or duplicated",
        )
        seen_sonames.add(soname)
        require_sha256(library["sha256"], f"blas_lapack.libraries[{index}].sha256")

    cases = manifest["cases"]
    require_manifest(isinstance(cases, dict), "cases must be an object")
    require_manifest(set(cases) == set(CASES), "manifest case set mismatch")
    for case_id, spec in CASES.items():
        entry = cases[case_id]
        require_manifest(
            isinstance(entry, dict)
            and set(entry) == {"path", "sha256", "spec_path", "spec_sha256"},
            f"manifest entry for {case_id} is malformed",
        )
        expected_path = f"{case_id}.json"
        expected_spec_path = f"specs/{case_id}.spec"
        require_manifest(entry["path"] == expected_path, f"invalid path for {case_id}")
        require_manifest(
            entry["spec_path"] == expected_spec_path,
            f"invalid spec path for {case_id}",
        )
        expected_digest = require_sha256(entry["sha256"], f"cases.{case_id}.sha256")
        expected_spec_digest = require_sha256(
            entry["spec_sha256"], f"cases.{case_id}.spec_sha256"
        )
        trace_path = case_dir / expected_path
        spec_path = case_dir / expected_spec_path
        require_manifest(trace_path.is_file(), f"missing golden {expected_path}")
        require_manifest(spec_path.is_file(), f"missing spec {expected_spec_path}")
        trace_bytes = trace_path.read_bytes()
        require_manifest(
            hashlib.sha256(trace_bytes).hexdigest() == expected_digest,
            f"{case_id} trace hash mismatch",
        )
        spec_bytes = spec_path.read_bytes()
        require_manifest(
            hashlib.sha256(spec_bytes).hexdigest() == expected_spec_digest,
            f"{case_id} spec hash mismatch",
        )
        require_manifest(
            spec_bytes == serialize_spec(spec).encode("utf-8"),
            f"{case_id} spec is not the canonical generator input",
        )
        try:
            trace = json.loads(trace_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CorpusError(f"{case_id} is not canonical JSON: {error}") from error
        try:
            writer.validate(trace)
            canonical = writer.dumps(trace).encode("utf-8")
        except writer.TraceError as error:
            raise CorpusError(f"{case_id} fails trace validation: {error}") from error
        require_manifest(canonical == trace_bytes, f"{case_id} is not canonical JSON")
        provenance = trace["provenance"]
        require_manifest(
            provenance["tblite_revision"] == REVISION
            and provenance["oracle_patch_sha256"] == patch_digest
            and provenance.get("oracle_command") == command,
            f"{case_id} trace provenance does not match the corpus manifest",
        )


def verify_manifest(case_dir: Path) -> int:
    """Check existing goldens and all repository-owned provenance offline."""
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.is_file():
        print("ERROR: no corpus directory manifest", file=sys.stderr)  # noqa: T201
        return 2
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest_document(manifest, case_dir)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, CorpusError) as error:
        print(f"ERROR: invalid corpus manifest: {error}", file=sys.stderr)  # noqa: T201
        return 2
    print(  # noqa: T201 - CLI report
        f"verified {len(CASES)} goldens and complete oracle provenance"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CorpusError, validator.ObserverPatchError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)  # noqa: T201
        sys.exit(2)
