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
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Import the reusable pinned provenance/build machinery from the observer
# patch validator so the two tools cannot drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_observer_patch as validator  # noqa: E402
import gpuxtb_scc_trace as writer  # noqa: E402

TOOL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOL_DIR.parents[2]
RECORDER_PATH = TOOL_DIR / "scc_trace_recorder.f90"
MAIN_PATH = TOOL_DIR / "scc_trace_main.f90"

FORMAT = "gpuxtb-scc-trace-v1"
REVISION = "e9abc395b122018ed688aecb1c3a65cecaf97beb"
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
            [2.43247309226225, 2.75237178376284, 0.01392519847964],
            [2.79621404458590, 0.93157260886974, 0.01863384029005],
            [3.30583608421060, 3.43820531288547, -1.42134539425148],
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
    rows = []
    for row in range(n):
        rows.append([values[column * n + row] for column in range(n)])
    return rows


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_revisions(source_root: Path) -> dict[str, str]:
    """Resolve the pinned tblite fallback dependencies to full commit SHAs.

    tblite's Meson wrap files may name tags; this generator resolves the
    actual checked-out HEAD of every fallback subproject and refuses to accept
    a golden when a dependency is not a pinned full 40-hex commit.  The source
    checkout's subprojects must already be populated at the pinned commits.
    """
    revisions: dict[str, str] = {}
    subprojects = source_root / "subprojects"
    for directory in sorted(subprojects.iterdir()):
        if not directory.is_dir() or (directory / ".git").exists() is False:
            continue
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise CorpusError(f"cannot resolve dependency HEAD: {directory}")
        commit = result.stdout.strip()
        if (
            len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise CorpusError(f"dependency is not a pinned commit: {directory}")
        revisions[directory.name] = commit
    return revisions


def probe_meson_project(lapack: str) -> str:
    """Return the disposable outer Meson project building the trace recorder."""
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
    'ddx=false',
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


def build_oracle(
    checkout: Path,
    source_root: Path,
    *,
    meson_command: list[str],
    lapack: str,
    wrap_mode: str,
) -> Path:
    """Build the recorder oracle in a disposable outer project and return the
    executable path.

    Every tblite fallback dependency is consumed from the source checkout's
    already-pinned subproject directories (symlinked read-only), so the build
    is offline, deterministic, and matches the commit SHAs recorded in the
    manifest.  Unpinned fallbacks that are not present are rejected here rather
    than silently resolved from a movable tag.
    """
    with tempfile.TemporaryDirectory(prefix="gpuxtb-scc-trace-oracle-") as directory:
        outer = Path(directory) / "outer"
        subprojects = outer / "subprojects"
        subprojects.mkdir(parents=True)
        os.symlink(checkout, subprojects / "tblite", target_is_directory=True)
        (outer / "meson.build").write_text(
            probe_meson_project(lapack), encoding="utf-8"
        )
        shutil.copy2(RECORDER_PATH, outer / "scc_trace_recorder.f90")
        shutil.copy2(MAIN_PATH, outer / "scc_trace_main.f90")

        # Bind every available dependency to the pinned local checkout so the
        # fallback resolution cannot move with a tag or need the network.
        def bind_dependencies(root: Path) -> None:
            subprojects_dir = root / "subprojects"
            if not subprojects_dir.is_dir():
                return
            for dependency in sorted(subprojects_dir.iterdir()):
                if not dependency.is_dir() or dependency.name == "ddx":
                    continue
                target = subprojects_dir / dependency.name
                if target.is_symlink() or target.exists():
                    continue
                local = source_root / "subprojects" / dependency.name
                if not local.is_dir():
                    # The wrap may carry its own subproject fallbacks; bind the
                    # registered nested dependency when available.
                    continue
                os.symlink(local, target, target_is_directory=True)
                bind_dependencies(local)

        bind_dependencies(checkout)

        environment = os.environ.copy()
        environment["LC_ALL"] = "C"
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["MKL_DYNAMIC"] = "FALSE"
        build = outer / "build"
        command = [*meson_command, "setup", str(build), f"--wrap-mode={wrap_mode}"]
        validator.run(command, cwd=outer, env=environment)
        validator.run(
            [*meson_command, "compile", "-C", str(build), "gpuxtb-tblite-scc-trace"],
            cwd=outer,
            env=environment,
        )
        binary_path = build / "gpuxtb-tblite-scc-trace"
        if not binary_path.is_file():
            raise CorpusError("recorder oracle executable was not produced")
        # Copy out so the temp dir teardown cannot invalidate the binary.
        out = checker_cache(binary_path)
        return out


# A tiny flyweight cache: only the last binary is kept; callers must consume it
# immediately.  This is intentionally single-shot.
_binary_cache: tuple[Path, Path] | None = None


def checker_cache(path: Path) -> Path:
    global _binary_cache
    cache_dir = Path(tempfile.gettempdir()) / "gpuxtb-scc-trace-oracle-bin"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "gpuxtb-tblite-scc-trace"
    shutil.copy2(path, target)
    _binary_cache = (path, target)
    return target


def write_spec(spec: dict[str, object], path: Path) -> None:
    """Write a fixed-layout case spec consumed by the Fortran recorder."""
    atomic_numbers = spec["atomic_numbers"]
    positions = spec["positions"]
    lines = [str(len(atomic_numbers))]
    for number in atomic_numbers:
        lines.append(str(number))
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
            values = [*positions_pc[index], point_charges["charges"][index],
                      point_charges["gammas"][index]]
            lines.extend(repr(float(value)) for value in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_recorder(binary: Path, spec: Path, workdir: Path) -> str:
    """Run the recorder on one case and return its raw stdout stream."""
    result = subprocess.run(
        [str(binary), str(spec)],
        cwd=workdir,
        env={**os.environ, "LC_ALL": "C", "OMP_NUM_THREADS": "1"},
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
    tokens = [token for line in raw.splitlines() for token in line.split()]
    iterator = iter(tokens)

    def take() -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise CorpusError("raw stream ended prematurely")

    header = take().split()
    while len(header) < 10:
        header += take().split()
    if header[0] != "nat" or header[2] != "nsh" or header[4] != "nao":
        raise CorpusError("raw stream header is malformed")
    nat, nsh, nao = int(header[1]), int(header[3]), int(header[5])
    niterations, terminal = int(header[7]), int(header[9])

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

    iterations = []
    for _ in range(niterations):
        label = take()
        assert label == "iteration", label
        iteration_index = int(take())
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

        def multipoles(nat: int, components: int) -> list[list[list[float]]]:
            flat = floats(nat * components)
            return [
                [flat[atom * components + component] for component in range(components)]
                for atom in range(nat)
            ]

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
        if not math.isclose(reconstructed_rms, residual_rms, rel_tol=1e-9, abs_tol=1e-9):
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

    converged = terminal == 1
    trace = {
        "format": FORMAT,
        "provenance": {
            "tblite_revision": REVISION,
            "oracle_patch_sha256": validator.sha256_file(TOOL_DIR / "tblite-e9abc395-scc-observer.patch"),
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
            "iterations": niterations,
        },
    }
    if point_charges:
        trace["input"]["point_charges"] = {
            "positions": [value for row in point_charges for value in row[:3]],
            "charges": [row[3] for row in point_charges],
            "hardnesses": [row[4] for row in point_charges],
        }
    return trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="local tblite Git checkout used read-only (must be at the pinned revision)",
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
        "--wrap-mode",
        type=str,
        default="forcefallback",
        choices=("default", "nofallback", "nodownload", "forcefallback", "nopromote"),
        help="Meson dependency wrap mode used by the oracle",
    )
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser()
    if isinstance(argv, list):
        import argparse as _argparse

        arguments = _argparse.Namespace(**dict(vars(arguments)))

    case_dir = arguments.corpus_dir
    if arguments.check:
        return verify_manifest(case_dir)

    metadata = validator.load_metadata()
    validator.validate_bundle(metadata)
    revision = str(metadata["upstream"]["revision"])
    if revision != REVISION:
        print(f"ERROR: oracle revision {revision} != expected {REVISION}", file=sys.stderr)
        return 2

    source_head, source_status = validator.source_state(arguments.source_root)
    # The source checkout may sit on any branch; what matters is that the
    # pinned revision is reachable and the tree is clean so the clone is a
    # byte-faithful basis for the oracle.
    probe = subprocess.run(
        ["git", "-C", str(arguments.source_root), "cat-file", "-e", f"{REVISION}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        print(
            f"ERROR: pinned revision {REVISION} is not reachable from the source checkout",
            file=sys.stderr,
        )
        return 2
    if source_status.strip():
        print("ERROR: source checkout is dirty; provenance cannot be established",
              file=sys.stderr)
        return 2

    deps = dependency_revisions(arguments.source_root)
    compiler = validate_compiler()

    # The recorded command must be independent of the output directory so that
    # regenerating into a different corpus path is byte-identical.  The
    # corpus-dir argument is redacted to a fixed token.
    raw_command = sys.argv[1:] if argv is None else argv
    redacted_command: list[str] = []
    skip_next = False
    for argument in raw_command:
        if skip_next:
            redacted_command.append("<corpus-dir>")
            skip_next = False
            continue
        if argument == "--corpus-dir":
            redacted_command.append(argument)
            skip_next = True
            continue
        redacted_command.append(argument)
    command_line = shlex.join(["generate_scc_corpus.py", *redacted_command])
    case_dir = arguments.corpus_dir
    case_dir.mkdir(parents=True, exist_ok=True)

    if arguments.check:
        return verify_manifest(case_dir)

    # Build the oracle once and run every case.
    with tempfile.TemporaryDirectory(prefix="gpuxtb-scc-corpus-work-") as directory:
        work = Path(directory)
        checkout = work / "checkout"
        validator.clone_and_apply(arguments.source_root, checkout, metadata)
        binary = build_oracle(
            checkout,
            arguments.source_root,
            meson_command=shlex.split(arguments.meson_command),
            lapack=arguments.lapack,
            wrap_mode=arguments.wrap_mode,
        )

        entries = {}
        for case_id, spec in CASES.items():
            print(f"[corpus] running {case_id}", file=sys.stderr)
            spec_path = work / f"{case_id}.spec"
            write_spec(spec, spec_path)
            raw = run_recorder(binary, spec_path, work)
            debug_raw = Path("/tmp") / f"{case_id}.raw"
            debug_raw.write_text(raw, encoding="utf-8")
            try:
                trace = canonicalize(raw, spec, command_line)
                canonical = writer.dumps(trace)
            except (writer.TraceError, CorpusError) as error:
                print(f"ERROR: {case_id}: {error}", file=sys.stderr)
                return 2
            document = {**trace, "case_id": case_id}
            # Rebuild without case_id, then include in output document.
            del document["case_id"]
            output_path = case_dir / f"{case_id}.json"
            output_path.write_text(canonical, encoding="utf-8")
            entries[case_id] = {
                "path": f"{case_id}.json",
                "sha256": sha256_file(output_path),
            }

    manifest = {
        "schema": "gpuxtb-scc-trace-corpus-manifest-v1",
        "format": FORMAT,
        "revision": REVISION,
        "oracle_patch_sha256": validator.sha256_file(
            TOOL_DIR / "tblite-e9abc395-scc-observer.patch"
        ),
        "recorder_sha256": sha256_file(RECORDER_PATH),
        "compiler": compiler,
        "dependencies": deps,
        "command": command_line,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(),
        "cases": entries,
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"generated {len(entries)} restricted goldens under {case_dir}")
    return 0


def verify_manifest(case_dir: Path) -> int:
    """Check existing goldens against their pinned manifest hashes."""
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.is_file():
        print("ERROR: no corpus directory manifest", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for case_id in sorted(CASES):
        entry = manifest["cases"].get(case_id)
        if entry is None:
            print(f"ERROR: manifest misses {case_id}", file=sys.stderr)
            return 2
        path = case_dir / entry["path"]
        if not path.is_file():
            print(f"ERROR: missing golden {entry['path']}", file=sys.stderr)
            return 2
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            print(
                f"ERROR: {case_id} sha256 {digest} != {entry['sha256']}",
                file=sys.stderr,
            )
            return 1
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            print(f"ERROR: {case_id} is not valid JSON: {error}", file=sys.stderr)
            return 2
        try:
            writer.validate(trace)
        except writer.TraceError as error:
            print(f"ERROR: {case_id} fails schema validation: {error}", file=sys.stderr)
            return 2
    print(f"verified {len(CASES)} goldens against the corpus manifest")
    return 0


def validate_compiler() -> dict[str, str]:
    """Record the Fortran compiler identity consumed by the oracle build."""
    candidate = os.environ.get("FC")
    identity: dict[str, str] = {"FC": candidate or ""}
    if candidate:
        result = subprocess.run(
            [candidate, "--version"], capture_output=True, text=True, check=False
        )
        identity["version"] = (result.stdout or result.stderr).strip().splitlines()[0]
    return identity


if __name__ == "__main__":
    sys.exit(main())