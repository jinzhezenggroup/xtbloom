"""Direct Python geometry optimization built on xTBloom analytic forces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from ._optimizer import curvature, initial_step_size, lbfgs_direction
from .exceptions import XTBloomRuntimeError, XTBloomValueError
from .interface import BatchCalculator

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .interface import Calculator, Structure

_ACCEPT_TOLERANCE = 1.0e-8
_ALPHA_DECAY = 0.5
_ALPHA_MIN = 1.0e-4


@dataclass(frozen=True)
class OptimizationResult:
    """Accepted final states from one molecular geometry optimization."""

    positions: tuple[np.ndarray, ...]
    energies: np.ndarray
    forces: tuple[np.ndarray, ...]
    converged: np.ndarray
    steps: np.ndarray
    evaluations: int

    @property
    def all_converged(self) -> bool:
        """Return whether every system met the requested force threshold."""
        return bool(np.all(self.converged))


def _validated_controls(
    fmax: float, max_steps: int | None, memory: int
) -> tuple[float, int | None, int]:
    threshold = float(fmax)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise XTBloomValueError("fmax must be finite and positive")
    if isinstance(max_steps, bool):
        raise XTBloomValueError("max_steps must be a positive integer or None")
    if max_steps is not None:
        try:
            steps = int(max_steps)
        except (TypeError, ValueError, OverflowError):
            raise XTBloomValueError(
                "max_steps must be a positive integer or None"
            ) from None
        if steps != max_steps or steps < 1:
            raise XTBloomValueError("max_steps must be a positive integer or None")
        max_steps = steps
    if isinstance(memory, bool):
        raise XTBloomValueError("memory must be a positive integer")
    try:
        memory_value = int(memory)
    except (TypeError, ValueError, OverflowError):
        raise XTBloomValueError("memory must be a positive integer") from None
    if memory_value != memory or memory_value < 1:
        raise XTBloomValueError("memory must be a positive integer")
    return threshold, max_steps, memory_value


def _max_force(force: np.ndarray) -> float:
    """Return the maximum per-atom force norm in native Hartree/bohr units."""
    return float(np.max(np.linalg.norm(force, axis=1)))


def _optimize_structures(
    structures: Sequence[Structure],
    evaluate: Callable[[], Sequence[tuple[float, np.ndarray]]],
    *,
    fmax: float,
    max_steps: int | None,
    memory: int,
) -> OptimizationResult:
    """Run the shared energy-accepted L-BFGS controller."""
    fmax, max_steps, memory = _validated_controls(fmax, max_steps, memory)
    if not structures:
        raise XTBloomValueError("cannot optimize an empty structure sequence")
    nsystems = len(structures)
    accepted_pos: list[np.ndarray | None] = [None] * nsystems
    accepted_energy = np.full(nsystems, np.nan, dtype=np.float64)
    accepted_force: list[np.ndarray | None] = [None] * nsystems
    previous_gradient: list[np.ndarray | None] = [None] * nsystems
    alpha = np.zeros(nsystems, dtype=np.float64)
    s_history: list[list[np.ndarray]] = [[] for _ in range(nsystems)]
    y_history: list[list[np.ndarray]] = [[] for _ in range(nsystems)]
    rho_history: list[list[float]] = [[] for _ in range(nsystems)]
    converged = np.zeros(nsystems, dtype=bool)
    accepted_steps = np.zeros(nsystems, dtype=np.int64)

    def restore_accepted() -> None:
        for index, position in enumerate(accepted_pos):
            if position is not None:
                structures[index].update(positions=position)

    def checked_evaluate() -> list[tuple[float, np.ndarray]]:
        values = list(evaluate())
        if len(values) != nsystems:
            raise XTBloomRuntimeError(
                "optimizer evaluator returned a different system count"
            )
        failed: list[int] = []
        normalized: list[tuple[float, np.ndarray]] = []
        for index, (energy, force) in enumerate(values):
            force_array = np.asarray(force, dtype=np.float64)
            if force_array.shape != structures[index].positions.shape:
                raise XTBloomRuntimeError(
                    f"optimizer evaluator returned an invalid force shape for system {index}"
                )
            if not np.isfinite(energy) or not np.isfinite(force_array).all():
                failed.append(index)
            normalized.append((float(energy), force_array.copy()))
        if failed:
            restore_accepted()
            indices = ", ".join(str(index) for index in failed)
            raise XTBloomRuntimeError(
                f"xTBloom optimization produced failed systems: {indices}"
            )
        return normalized

    initial = checked_evaluate()
    evaluations = 1
    for index, (energy, force) in enumerate(initial):
        position = structures[index].positions.copy()
        accepted_pos[index] = position
        accepted_energy[index] = energy
        accepted_force[index] = force
        converged[index] = _max_force(force) <= fmax
        if converged[index]:
            continue
        gradient = -force
        previous_gradient[index] = gradient
        alpha[index] = initial_step_size(gradient)
        direction = lbfgs_direction(gradient, [], [], [])
        structures[index].update(positions=position + alpha[index] * direction)

    moves = 0
    while not bool(np.all(converged)) and (max_steps is None or moves < max_steps):
        values = checked_evaluate()
        evaluations += 1
        moves += 1
        for index, (energy, force) in enumerate(values):
            if converged[index]:
                # Converged peers stay fixed but remain in the ragged batch so
                # the caller-owned BatchCalculator can reuse one stable topology.
                continue
            baseline_pos = cast("np.ndarray", accepted_pos[index])
            baseline_gradient = cast("np.ndarray", previous_gradient[index])
            acceptance_limit = accepted_energy[index] + _ACCEPT_TOLERANCE * max(
                1.0, abs(accepted_energy[index])
            )
            if energy > acceptance_limit:
                structures[index].update(positions=baseline_pos)
                if alpha[index] <= _ALPHA_MIN:
                    restore_accepted()
                    raise XTBloomRuntimeError(
                        "xTBloom optimization line search stalled at the minimum "
                        f"step size for system {index}"
                    )
                alpha[index] = max(alpha[index] * _ALPHA_DECAY, _ALPHA_MIN)
                direction = lbfgs_direction(
                    baseline_gradient,
                    s_history[index],
                    y_history[index],
                    rho_history[index],
                )
                structures[index].update(
                    positions=baseline_pos + alpha[index] * direction
                )
                continue

            evaluated_pos = structures[index].positions.copy()
            gradient = -force
            s_step = evaluated_pos - baseline_pos
            y_step = gradient - baseline_gradient
            pair_curvature = curvature(s_step, y_step)
            if pair_curvature > 0.0:
                s_history[index].append(s_step)
                y_history[index].append(y_step)
                rho_history[index].append(1.0 / pair_curvature)
                if len(s_history[index]) > memory:
                    s_history[index].pop(0)
                    y_history[index].pop(0)
                    rho_history[index].pop(0)

            accepted_pos[index] = evaluated_pos
            accepted_energy[index] = energy
            accepted_force[index] = force
            previous_gradient[index] = gradient
            accepted_steps[index] += 1
            converged[index] = _max_force(force) <= fmax
            if converged[index]:
                continue

            alpha[index] = 1.0
            direction = lbfgs_direction(
                gradient,
                s_history[index],
                y_history[index],
                rho_history[index],
            )
            structures[index].update(positions=evaluated_pos + direction)

    restore_accepted()
    return OptimizationResult(
        positions=tuple(cast("np.ndarray", value).copy() for value in accepted_pos),
        energies=accepted_energy.copy(),
        forces=tuple(cast("np.ndarray", value).copy() for value in accepted_force),
        converged=converged.copy(),
        steps=accepted_steps.copy(),
        evaluations=evaluations,
    )


def optimize(
    calculator: Calculator,
    *,
    fmax: float = 5.0e-4,
    max_steps: int | None = 200,
    memory: int = 5,
) -> OptimizationResult:
    """Optimize one existing :class:`~xtbloom.Calculator` in place.

    ``fmax`` is the maximum per-atom force norm in Hartree/bohr. The calculator
    keeps its configured backend, SCC policy, and warm-start behavior.
    """

    def evaluate() -> list[tuple[float, np.ndarray]]:
        result = calculator.singlepoint()
        return [(result.energy, result.forces)]

    return _optimize_structures(
        [calculator], evaluate, fmax=fmax, max_steps=max_steps, memory=memory
    )


def optimize_batch(
    structures: Sequence[Structure],
    method: str = "GFN2-xTB",
    *,
    fmax: float = 5.0e-4,
    max_steps: int | None = 200,
    memory: int = 5,
    backend: str | int = "auto",
    device_id: int | None = None,
    cpu_threads: int = 1,
    max_scc_iterations: int = 250,
    charge_tolerance: float = 1.0e-6,
    energy_tolerance: float = 1.0e-8,
    electronic_temperature: float = 300.0,
    scc_mixer: str | int = "modified_broyden",
    scc_mixer_history: int = 8,
    scc_mixer_damping: float = 0.4,
    determinism: str | int = "default",
    warm_start: bool = True,
) -> OptimizationResult:
    """Optimize a ragged sequence through one reusable :class:`BatchCalculator`."""
    structures = list(structures)
    if not structures:
        raise XTBloomValueError("cannot optimize an empty structure sequence")
    calculator = BatchCalculator(
        structures,
        method,
        backend=backend,
        device_id=device_id,
        cpu_threads=cpu_threads,
        max_scc_iterations=max_scc_iterations,
        charge_tolerance=charge_tolerance,
        energy_tolerance=energy_tolerance,
        electronic_temperature=electronic_temperature,
        scc_mixer=scc_mixer,
        scc_mixer_history=scc_mixer_history,
        scc_mixer_damping=scc_mixer_damping,
        determinism=determinism,
        warm_start=warm_start,
    )

    def evaluate() -> list[tuple[float, np.ndarray]]:
        result = calculator.compute()
        values: list[tuple[float, np.ndarray]] = []
        for index in range(len(structures)):
            single = result[index]
            values.append((single.energy, single.forces))
        return values

    try:
        return _optimize_structures(
            structures,
            evaluate,
            fmax=fmax,
            max_steps=max_steps,
            memory=memory,
        )
    finally:
        calculator.close()


__all__ = ["OptimizationResult", "optimize", "optimize_batch"]
