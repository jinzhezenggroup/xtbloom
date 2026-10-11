"""Direct Python geometry optimization built on xTBloom analytic forces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import library
from ._optimizer import _LBFGSState, _validated_controls
from .exceptions import XTBloomRuntimeError, XTBloomValueError
from .interface import BatchCalculator

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .interface import Calculator, Structure


@dataclass(frozen=True)
class OptimizationResult:
    """Accepted final states from one molecular geometry optimization."""

    positions: tuple[np.ndarray, ...]
    energies: np.ndarray
    forces: tuple[np.ndarray, ...]
    converged: np.ndarray
    failed: np.ndarray
    failure_messages: tuple[str | None, ...]
    steps: np.ndarray
    evaluations: int

    @property
    def all_converged(self) -> bool:
        """Return whether every system met the requested force threshold."""
        return bool(np.all(self.converged))

    @property
    def failed_indices(self) -> np.ndarray:
        """Return indices stopped by peer-local numerical failures."""
        return np.flatnonzero(self.failed)

    def raise_for_status(self) -> None:
        """Raise a combined exception while retaining successful peer results."""
        messages = [
            f"system {int(index)}: {self.failure_messages[int(index)]}"
            for index in self.failed_indices
        ]
        if messages:
            raise XTBloomRuntimeError(
                "xTBloom optimization produced failed systems: " + "; ".join(messages)
            )


@dataclass(frozen=True)
class _Evaluation:
    """One evaluator output plus an optional peer-local failure diagnostic."""

    energy: float
    force: np.ndarray
    error: str | None = None


def _max_force(force: np.ndarray) -> float:
    """Return the maximum per-atom force norm in native Hartree/bohr units."""
    return float(np.max(np.linalg.norm(force, axis=1)))


def _optimize_structures(
    structures: Sequence[Structure],
    evaluate: Callable[[], Sequence[_Evaluation]],
    *,
    fmax: float,
    max_steps: int | None,
    memory: int,
    peer_local_failures: bool,
) -> OptimizationResult:
    """Run the shared controller and restore accepted positions on every exit."""
    fmax, max_steps, memory = _validated_controls(fmax, max_steps, memory)
    if not structures:
        raise XTBloomValueError("cannot optimize an empty structure sequence")
    nsystems = len(structures)
    original_positions = [structure.positions.copy() for structure in structures]
    states: list[_LBFGSState | None] = [None] * nsystems
    converged = np.zeros(nsystems, dtype=bool)
    failed = np.zeros(nsystems, dtype=bool)
    failure_messages: list[str | None] = [None] * nsystems

    def restore_accepted() -> None:
        """Leave every structure at its last valid accepted geometry."""
        for index, state in enumerate(states):
            position = original_positions[index] if state is None else state.position
            structures[index].update(positions=position)

    def checked_evaluate() -> list[_Evaluation]:
        """Normalize evaluator data and reject call-level contract violations."""
        values = list(evaluate())
        if len(values) != nsystems:
            raise XTBloomRuntimeError(
                "optimizer evaluator returned a different system count"
            )
        normalized: list[_Evaluation] = []
        for index, value in enumerate(values):
            if not isinstance(value, _Evaluation):
                raise XTBloomRuntimeError(
                    f"optimizer evaluator returned an invalid entry for system {index}"
                )
            try:
                energy = float(value.energy)
                force_array = np.asarray(value.force, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raise XTBloomRuntimeError(
                    f"optimizer evaluator returned non-numeric data for system {index}"
                ) from None
            if force_array.shape != original_positions[index].shape:
                raise XTBloomRuntimeError(
                    "optimizer evaluator returned an invalid force shape "
                    f"for system {index}"
                )
            error = value.error
            if error is None and (
                not np.isfinite(energy) or not np.isfinite(force_array).all()
            ):
                error = "evaluator returned non-finite energy or forces"
            normalized.append(
                _Evaluation(energy=energy, force=force_array.copy(), error=error)
            )
        return normalized

    def stop_failed(index: int, message: str) -> None:
        """Stop one peer at its accepted state or raise for strict callers."""
        if not peer_local_failures:
            raise XTBloomRuntimeError(
                f"xTBloom optimization failed for system {index}: {message}"
            )
        failed[index] = True
        converged[index] = False
        failure_messages[index] = message
        state = states[index]
        position = original_positions[index] if state is None else state.position
        structures[index].update(positions=position)

    try:
        initial = checked_evaluate()
        evaluations = 1
        for index, value in enumerate(initial):
            if value.error is not None:
                stop_failed(index, value.error)
                continue
            energy = float(value.energy)
            force = np.asarray(value.force, dtype=np.float64)
            state = _LBFGSState.from_evaluation(
                structures[index].positions, energy, force, memory
            )
            states[index] = state
            converged[index] = _max_force(force) <= fmax
            if not converged[index]:
                structures[index].update(positions=state.initial_trial())

        moves = 0
        while not bool(np.all(converged | failed)) and (
            max_steps is None or moves < max_steps
        ):
            values = checked_evaluate()
            evaluations += 1
            moves += 1
            for index, value in enumerate(values):
                if converged[index] or failed[index]:
                    # Finished peers stay fixed but remain in a stable ragged
                    # batch so the native context and warm state can be reused.
                    continue
                if value.error is not None:
                    stop_failed(index, value.error)
                    continue

                energy = float(value.energy)
                force = np.asarray(value.force, dtype=np.float64)
                state = states[index]
                assert state is not None
                if not state.accepts(energy):
                    trial = state.backoff_trial()
                    if trial is None:
                        stop_failed(
                            index,
                            "line search stalled at the minimum step size",
                        )
                    else:
                        structures[index].update(positions=trial)
                    continue

                state.accept_trial(structures[index].positions, energy, force)
                converged[index] = _max_force(force) <= fmax
                if not converged[index]:
                    structures[index].update(positions=state.next_trial())

        positions = tuple(
            (original_positions[index] if state is None else state.position).copy()
            for index, state in enumerate(states)
        )
        energies = np.asarray(
            [np.nan if state is None else state.energy for state in states],
            dtype=np.float64,
        )
        forces = tuple(
            (
                np.full_like(original_positions[index], np.nan)
                if state is None
                else state.force.copy()
            )
            for index, state in enumerate(states)
        )
        steps = np.asarray(
            [0 if state is None else state.steps for state in states],
            dtype=np.int64,
        )
        return OptimizationResult(
            positions=positions,
            energies=energies,
            forces=forces,
            converged=converged.copy(),
            failed=failed.copy(),
            failure_messages=tuple(failure_messages),
            steps=steps,
            evaluations=evaluations,
        )
    finally:
        restore_accepted()


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

    def evaluate() -> list[_Evaluation]:
        result = calculator.singlepoint()
        return [_Evaluation(result.energy, result.forces)]

    return _optimize_structures(
        [calculator],
        evaluate,
        fmax=fmax,
        max_steps=max_steps,
        memory=memory,
        peer_local_failures=False,
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
    scc_mixer_history: int = library.DEFAULT_SCC_MIXER_HISTORY,
    scc_mixer_damping: float = library.DEFAULT_SCC_MIXER_DAMPING,
    determinism: str | int = "default",
    warm_start: bool = True,
) -> OptimizationResult:
    """Optimize distinct mutable structures through one reusable batch context.

    Repeating the same :class:`Structure` object is rejected because each batch
    member owns an independent accepted-state ledger while updates occur in place.
    """
    structures = list(structures)
    if not structures:
        raise XTBloomValueError("cannot optimize an empty structure sequence")
    if len({id(structure) for structure in structures}) != len(structures):
        raise XTBloomValueError(
            "optimize_batch structures must be distinct mutable objects"
        )
    # Validate optimizer-only arguments before acquiring native resources.
    fmax, max_steps, memory = _validated_controls(fmax, max_steps, memory)
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

    def evaluate() -> list[_Evaluation]:
        result = calculator.compute()
        values: list[_Evaluation] = []
        for index in range(len(structures)):
            single = result[index]
            error = None
            if single.scc_status != library.STATUS_SUCCESS or not single.scc_converged:
                error = (
                    f"{library.status_string(single.scc_status)}, "
                    f"scc_converged={int(single.scc_converged)}, "
                    f"iterations={single.scc_iterations}"
                )
            values.append(_Evaluation(single.energy, single.forces, error))
        return values

    try:
        return _optimize_structures(
            structures,
            evaluate,
            fmax=fmax,
            max_steps=max_steps,
            memory=memory,
            peer_local_failures=True,
        )
    finally:
        calculator.close()


__all__ = ["OptimizationResult", "optimize", "optimize_batch"]
