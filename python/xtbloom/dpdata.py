"""dpdata Driver and Minimizer plugins for xTBloom.

The driver labels a whole :class:`dpdata.System` (all frames at once) through a
single xTBloom ragged-batch ``xtbloom_compute`` call, which is the native
execution model of the C API. The minimizer optimizes the geometry of every
frame in lockstep and evaluates energies and forces for all *active* frames in
one ragged-batch call per step, so a whole batch of molecules is relaxed with
CUDA/CPU throughput instead of one frame per step as the reference ``ase``
minimizer does. Energies are returned in eV and forces in eV/Angstrom, matching
dpdata conventions.

The driver and minimizer are registered under the key ``"xtbloom"``:
``dpdata.Driver.get_driver("xtbloom")(...)`` and
``dpdata.Minimizer.get_minimizer("xtbloom")(...)``. Because the xTBloom Python
build declares the ``dpdata.plugins`` entry point pointing at this module,
importing ``dpdata`` loads it automatically and registers both plugins.

Net charge and spin multiplicity are handled per frame:

* ``charge`` may be a fixed value (constructor) or, when left as ``None``,
  taken from the per-frame ``data["charge"]`` key if present (else 0).
* ``uhf`` (``multiplicity - 1``) may be fixed (constructor) or, when ``None``,
  taken from per-frame ``data["uhf"]``/``data["multiplicity"]`` keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from dpdata.driver import Driver, Minimizer

from ._optimizer import _LBFGSState, _validated_controls
from .exceptions import XTBloomNotSupportedError, XTBloomRuntimeError, XTBloomValueError
from .interface import BatchCalculator, Structure, symbols_to_numbers

if TYPE_CHECKING:
    from collections.abc import Sequence

# dpdata reports energies in eV and forces in eV/Angstrom, while the xTBloom C
# API reports Hartree and Hartree/bohr.
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
_FORCE_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANGSTROM


class _SymbolMap:
    """Convert dpdata atom names to atomic numbers."""

    def __init__(self, atom_names: Sequence[str]) -> None:
        self._numbers: dict[str, int] = {}
        for symbol, number in zip(
            atom_names, symbols_to_numbers(atom_names), strict=True
        ):
            if symbol in self._numbers:
                raise XTBloomValueError(f"duplicate dpdata atom name {symbol!r}")
            self._numbers[symbol] = number

    def __getitem__(self, symbol: str) -> int:
        try:
            return self._numbers[symbol]
        except KeyError:
            raise XTBloomValueError(f"unknown dpdata atom name {symbol!r}") from None


def _structures_from_data(
    data: dict,
    *,
    charge: float | None,
    uhf: int | None,
    multiplicity: int | None,
    spin_channels: int | None,
) -> list[Structure]:
    """Build one :class:`~xtbloom.interface.Structure` per frame (bohr).

    Shared by the driver and the minimizer so both label and minimize honor the
    same per-frame charge/spin conventions.  Positions are converted from
    dpdata's Angstrom convention to bohr.
    """
    atom_names = list(data["atom_names"])
    raw_atom_types = np.asarray(data["atom_types"])
    if raw_atom_types.ndim != 1 or raw_atom_types.size == 0:
        raise XTBloomValueError(
            "atom_types must be a nonempty one-dimensional array of integer indices"
        )
    if raw_atom_types.dtype.kind not in "iu":
        raise XTBloomValueError("atom_types must contain exact integer indices")
    if np.any(raw_atom_types < 0) or np.any(raw_atom_types >= len(atom_names)):
        raise XTBloomValueError("atom_types contains an index outside atom_names")
    atom_types = np.asarray(raw_atom_types, dtype=np.int64)
    coords = np.asarray(data["coords"], dtype=np.float64)
    if coords.ndim != 3 or coords.shape[1:] != (atom_types.size, 3):
        raise XTBloomValueError("coords must have shape (nframes, natoms, 3)")
    nframes = coords.shape[0]

    symbol_map = _SymbolMap(atom_names)
    numbers = np.array(
        [symbol_map[atom_names[index]] for index in atom_types], dtype=np.int64
    )
    positions_bohr = coords / BOHR_TO_ANGSTROM

    structures = []
    for frame in range(nframes):
        if uhf is not None:
            uhf_value = uhf
            multiplicity_value = None
        elif multiplicity is not None:
            uhf_value = None
            multiplicity_value = multiplicity
        else:
            uhf_value = _frame_value(data, "uhf", None, frame, nframes, default=None)
            multiplicity_value = _frame_value(
                data, "multiplicity", None, frame, nframes, default=None
            )
            if uhf_value is None and multiplicity_value is None:
                uhf_value = 0
        structures.append(
            Structure(
                numbers,
                positions_bohr[frame],
                charge=cast(
                    "float",
                    _frame_value(data, "charge", charge, frame, nframes, default=0.0),
                ),
                uhf=cast("int | None", uhf_value),
                multiplicity=cast("int | None", multiplicity_value),
                spin_channels=spin_channels,
            )
        )
    return structures


@Driver.register("xtbloom")
class XTBloomDriver(Driver):
    """Label molecular frames with GFN1/GFN2-xTB using the xTBloom library.

    Parameters
    ----------
    method : str, default "GFN2-xTB"
        Underlying tight-binding method. Both GFN1-xTB and GFN2-xTB use the
        shared CPU/CUDA backend policy.
    charge : float, optional
        Fixed total charge applied to every frame. When ``None`` the per-frame
        ``data["charge"]`` key (or 0) is used.
    uhf : int, optional
        Fixed number of unpaired electrons applied to every frame. When
        ``None`` the per-frame ``data["uhf"]`` key (or 0) is used.
    multiplicity : int, optional
        Alternative to ``uhf`` (``uhf = multiplicity - 1``).
    spin_channels : int, optional
        Orbital channels (1 restricted / 2 unrestricted). Defaults to
        unrestricted for open-shell frames.
    **kwargs
        Forwarded to :class:`xtbloom.interface.BatchCalculator`: ``backend``,
        ``device_id``, ``cpu_threads``, ``max_scc_iterations``,
        ``charge_tolerance``, ``energy_tolerance``, ``electronic_temperature``,
        ``scc_mixer``, ``scc_mixer_history``, ``scc_mixer_damping``, and
        ``determinism``.
    """

    def __init__(
        self,
        method: str = "GFN2-xTB",
        charge: float | None = None,
        uhf: int | None = None,
        multiplicity: int | None = None,
        spin_channels: int | None = None,
        **kwargs: object,
    ) -> None:
        self.method = method
        self.charge = charge
        self.uhf = uhf
        self.multiplicity = multiplicity
        self.spin_channels = spin_channels
        self.kwargs: dict[str, object] = kwargs

    def label(self, data: dict) -> dict:
        """Label a dpdata system dict and return it with energies and forces.

        Parameters
        ----------
        data : dict
            dpdata system data with ``atom_names``, ``atom_types``, and
            ``coords`` (Angstrom).

        Returns
        -------
        dict
            Copy of ``data`` with ``energies`` (eV) and ``forces`` (eV/Angstrom).
        """
        _reject_periodic(data)
        structures = _structures_from_data(
            data,
            charge=self.charge,
            uhf=self.uhf,
            multiplicity=self.multiplicity,
            spin_channels=self.spin_channels,
        )
        # Derive the frame count only after the shared shape validation above;
        # malformed scalar coordinates must raise XTBloomValueError, not IndexError.
        nframes = len(structures)

        calculator = BatchCalculator(
            structures,
            self.method,
            # The driver forwards arbitrary BatchCalculator options. Values are
            # `object` at the boundary (no `Any` in the public signature); the
            # cast narrows them for the typed keyword expansion only.
            **cast("dict[str, Any]", self.kwargs),
        )
        try:
            # dpdata has no peer-status channel. Returning the non-strict
            # batch result would silently publish NaN labels for failed SCC or
            # eigensolver frames, so labeling must remain all-or-error.
            result = calculator.compute(raise_on_failure=True)
        finally:
            calculator.close()

        labeled = dict(data)
        labeled["energies"] = (
            np.asarray(result.energies, dtype=np.float64) * HARTREE_TO_EV
        )
        labeled["forces"] = (
            np.asarray(result.forces, dtype=np.float64).reshape(nframes, -1, 3)
            * _FORCE_TO_EV_ANG
        )
        return labeled


@Minimizer.register("xtbloom")
class XTBloomMinimizer(Minimizer):
    """Minimize the geometry of every frame in one ragged-batch pipeline.

    The reference dpdata ``ase`` minimizer optimizes one frame per ASE
    ``dyn.run`` call, so a batch of molecules is relaxed at batch size one.
    This minimizer instead moves all frames in lockstep and evaluates
    energies and forces for every *active* frame in a single xTBloom
    ragged-batch ``xtbloom_compute`` call per step.  Frames that reach the force
    threshold are frozen and removed from the batch, so the ragged batch
    shrinks as the optimization proceeds and only the still-active frames are
    recomputed.  The stepping mathematics is implemented on the Array API (see
    :mod:`xtbloom._optimizer`), so it can later be reused with any array backend
    that produces Array API gradients.

    Parameters
    ----------
    driver : XTBloomDriver, optional
        Configured driver that fixes the method, charge, spin, backend, and
        execution settings for every frame.  Defaults to a fresh
        ``XTBloomDriver()`` (CPU backend).
    fmax : float, default 5e-3
        Force convergence threshold in eV/Angstrom (dpdata units), matching the
        ASE convention.
    max_steps : int, optional
        Maximum number of geometry moves across the whole batch, excluding the
        initial force evaluation; ``None`` runs until every frame converges.
        Frames still active when it is reached are reported at their last
        energy-accepted geometry.
    memory : int, default 5
        Number of L-BFGS history pairs kept per frame.

    Examples
    --------
    >>> system.minimize("xtbloom", driver=XTBloomDriver(backend="cuda"), fmax=1e-2)
    """

    def __init__(
        self,
        driver: XTBloomDriver | None = None,
        *,
        fmax: float = 5e-3,
        max_steps: int | None = None,
        memory: int = 5,
    ) -> None:
        fmax, max_steps, memory = _validated_controls(fmax, max_steps, memory)
        self._driver = driver if driver is not None else XTBloomDriver()
        self._fmax = fmax
        self._max_steps = max_steps
        self._memory = memory

    def minimize(self, data: dict) -> dict:
        """Minimize the system and label it with the relaxed geometries.

        Parameters
        ----------
        data : dict
            dpdata system data with ``atom_names``, ``atom_types``, and
            ``coords`` (Angstrom).

        Returns
        -------
        dict
            Copy of ``data`` with ``coords`` replaced by the minimized
            geometries (Angstrom), and ``energies`` (eV) and ``forces``
            (eV/Angstrom) evaluated at those geometries.

        Raises
        ------
        XTBloomRuntimeError
            If any frame failed SCC or the eigensolver at its last evaluated
            geometry, or if its line search cannot find an energy-lowering step
            at the minimum step size. The caller never receives silently bogus
            or rejected labels.
        """
        _reject_periodic(data)
        driver = self._driver
        structures = _structures_from_data(
            data,
            charge=driver.charge,
            uhf=driver.uhf,
            multiplicity=driver.multiplicity,
            spin_channels=driver.spin_channels,
        )
        nframes = len(structures)
        if nframes == 0:
            raise XTBloomValueError("cannot minimize a system without frames")

        # The shared state object guarantees that rejected trials never replace
        # a frame's published position, energy, force, or L-BFGS history.
        states: dict[int, _LBFGSState] = {}
        done: set[int] = set()
        failed: set[int] = set()
        line_search_failed: set[int] = set()
        active = list(range(nframes))

        def make_calculator(frames: list[int]) -> BatchCalculator:
            return BatchCalculator(
                [structures[i] for i in frames],
                driver.method,
                **cast("dict[str, Any]", driver.kwargs),
            )

        calculator = make_calculator(active)
        try:
            evaluations = 0
            # The first pass establishes the input baselines. Each subsequent
            # pass evaluates one geometry move and consumes one max_steps slot.
            while active and (
                self._max_steps is None or evaluations <= self._max_steps
            ):
                can_move = self._max_steps is None or evaluations < self._max_steps
                result = calculator.compute()
                # Snapshot the evaluated positions before any trial moves.
                evaluated = [structures[i].positions.copy() for i in active]
                for position, frame in enumerate(active):
                    single = result[position]
                    energy = single.energy
                    force = single.forces
                    evaluated_pos = evaluated[position]

                    if not np.isfinite(energy) or not np.isfinite(force).all():
                        # Per-frame SCC/eigensolver failure is data-level, not
                        # call-level: poison only this frame and revert it to
                        # its last successfully evaluated geometry.
                        failed.add(frame)
                        done.add(frame)
                        if frame in states:
                            structures[frame].update(positions=states[frame].position)
                        continue

                    if frame not in states:
                        # The input geometry is the first accepted baseline.
                        state = _LBFGSState.from_evaluation(
                            evaluated_pos, energy, force, self._memory
                        )
                        states[frame] = state
                        max_force = float(
                            np.max(np.linalg.norm(force * _FORCE_TO_EV_ANG, axis=1))
                        )
                        if max_force <= self._fmax:
                            done.add(frame)
                            continue
                        if not can_move:
                            continue

                        structures[frame].update(positions=state.initial_trial())
                    else:
                        state = states[frame]
                        if not state.accepts(energy):
                            # Trial did not improve: retry from the accepted
                            # baseline with a shorter step, reusing the same
                            # history. Once the minimum step has itself been
                            # rejected, stop this frame instead of retrying the
                            # identical geometry forever.
                            trial = state.backoff_trial()
                            if trial is None:
                                line_search_failed.add(frame)
                                done.add(frame)
                                structures[frame].update(positions=state.position)
                                continue
                            if not can_move:
                                continue
                            structures[frame].update(positions=trial)
                            continue

                        state.accept_trial(evaluated_pos, energy, force)

                        # A rejected geometry cannot establish convergence; the
                        # force criterion is checked only after energy acceptance.
                        max_force = float(
                            np.max(np.linalg.norm(force * _FORCE_TO_EV_ANG, axis=1))
                        )
                        if max_force <= self._fmax:
                            done.add(frame)
                            continue
                        if not can_move:
                            continue

                        structures[frame].update(positions=state.next_trial())

                evaluations += 1
                if done:
                    keep = [frame for frame in active if frame not in done]
                    if keep:
                        if len(keep) < len(active):
                            calculator.close()
                            calculator = make_calculator(keep)
                        active = keep
                    else:
                        active = []
        finally:
            for frame, state in states.items():
                structures[frame].update(positions=state.position)
            calculator.close()

        if failed:
            indices = ", ".join(str(i) for i in sorted(failed))
            line_search_detail = ""
            if line_search_failed:
                stalled = ", ".join(str(i) for i in sorted(line_search_failed))
                line_search_detail = (
                    "; line search stalled at the minimum step size for "
                    f"frames {stalled}"
                )
            raise XTBloomRuntimeError(
                "xTBloom minimization produced failed systems: "
                f"frames {indices} failed SCC or the eigensolver{line_search_detail}"
            )
        if line_search_failed:
            indices = ", ".join(str(i) for i in sorted(line_search_failed))
            raise XTBloomRuntimeError(
                "xTBloom minimization line search stalled at the minimum step "
                f"size for frames {indices}"
            )

        # Frames not finished (max_steps reached) are reported at their last
        # energy-accepted geometry, never at an unevaluated or rejected trial.
        final_pos = (
            np.stack([states[frame].position for frame in range(nframes)])
            * BOHR_TO_ANGSTROM
        )
        final_energies = (
            np.asarray(
                [states[frame].energy for frame in range(nframes)], dtype=np.float64
            )
            * HARTREE_TO_EV
        )
        final_forces = (
            np.stack([states[frame].force for frame in range(nframes)])
            * _FORCE_TO_EV_ANG
        )

        labeled = dict(data)
        labeled["coords"] = final_pos
        labeled["energies"] = final_energies
        labeled["forces"] = final_forces
        return labeled


def _reject_periodic(data: dict) -> None:
    """Raise until the ABI-v4 lattice foundation has a periodic adapter."""
    if not bool(np.asarray(data.get("nopbc", True)).all()):
        raise XTBloomNotSupportedError(
            "the xTBloom Python driver does not support periodic systems yet "
            "(the ABI-v4 lattice descriptor exists, but native periodic "
            "execution and the dpdata adapter are not implemented)"
        )


def _frame_value(
    data: dict,
    key: str,
    fixed: float | None,
    frame: int,
    nframes: int,
    default: float | None,
) -> int | float | None:
    """Return a fixed scalar, a per-frame value, or the default."""
    if fixed is not None:
        return fixed
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.ndim == 0:
        return value.item()
    if value.ndim != 1 or value.shape[0] != nframes:
        raise XTBloomValueError(
            f"{key} must be a scalar or a one-dimensional array with "
            "one value per frame"
        )
    return value[frame].item()


__all__ = ["XTBloomDriver", "XTBloomMinimizer"]
