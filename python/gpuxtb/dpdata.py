"""dpdata :class:`~dpdata.driver.Driver` plugin for gpuxtb.

The driver labels a whole :class:`dpdata.System` (all frames at once) through a
single gpuxtb ragged-batch ``gpuxtb_compute`` call, which is the native
execution model of the C API. Energies are returned in eV and forces in
eV/Angstrom, matching dpdata conventions.

The driver is registered under the key ``"gpuxtb"``:
``dpdata.Driver.get_driver("gpuxtb")(...)``. Because the gpuxtb Python build
declares the ``dpdata.plugins`` entry point pointing at this module, importing
``dpdata`` loads it automatically and registers the driver.

Net charge and spin multiplicity are handled per frame:

* ``charge`` may be a fixed value (constructor) or, when left as ``None``,
  taken from the per-frame ``data["charge"]`` key if present (else 0).
* ``uhf`` (``multiplicity - 1``) may be fixed (constructor) or, when ``None``,
  taken from per-frame ``data["uhf"]``/``data["multiplicity"]`` keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from dpdata.driver import Driver

from .exceptions import GPUxtbNotSupportedError, GPUxtbValueError
from .interface import BatchCalculator, Structure, symbols_to_numbers

if TYPE_CHECKING:
    from collections.abc import Sequence

# dpdata reports energies in eV and forces in eV/Angstrom, while the gpuxtb C
# API reports Hartree and Hartree/bohr.
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


class _SymbolMap:
    """Convert dpdata atom names to atomic numbers."""

    def __init__(self, atom_names: Sequence[str]) -> None:
        self._numbers: dict[str, int] = {}
        for symbol, number in zip(
            atom_names, symbols_to_numbers(atom_names), strict=True
        ):
            if symbol in self._numbers:
                raise GPUxtbValueError(f"duplicate dpdata atom name {symbol!r}")
            self._numbers[symbol] = number

    def __getitem__(self, symbol: str) -> int:
        try:
            return self._numbers[symbol]
        except KeyError:
            raise GPUxtbValueError(f"unknown dpdata atom name {symbol!r}") from None


@Driver.register("gpuxtb")
class GPUxtbDriver(Driver):
    """Label molecular frames with GFN2-xTB using the gpuxtb library.

    Parameters
    ----------
    method : str, default "GFN2-xTB"
        Underlying tight-binding method (only GFN2-xTB is currently supported).
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
        unrestricted for open-shell frames on the CPU backend.
    **kwargs
        Forwarded to :class:`gpuxtb.interface.BatchCalculator`: ``backend``,
        ``device_id``, ``cpu_threads``, ``max_scc_iterations``,
        ``charge_tolerance``, ``energy_tolerance``, ``electronic_temperature``.
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
        self.kwargs = kwargs

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
        if not bool(np.asarray(data.get("nopbc", True)).all()):
            raise GPUxtbNotSupportedError(
                "the gpuxtb Python driver does not support periodic systems "
                "(the public C ABI has no lattice input)"
            )

        atom_names = list(data["atom_names"])
        atom_types = np.asarray(data["atom_types"], dtype=np.int64)
        coords = np.asarray(data["coords"], dtype=np.float64)
        if coords.ndim != 3 or coords.shape[1:] != (atom_types.size, 3):
            raise GPUxtbValueError("coords must have shape (nframes, natoms, 3)")
        nframes = coords.shape[0]

        symbol_map = _SymbolMap(atom_names)
        numbers = np.array(
            [symbol_map[atom_names[index]] for index in atom_types], dtype=np.int64
        )
        positions_bohr = coords / BOHR_TO_ANGSTROM

        structures = []
        for frame in range(nframes):
            if self.uhf is not None:
                uhf_value = self.uhf
                multiplicity_value = None
            elif self.multiplicity is not None:
                uhf_value = None
                multiplicity_value = self.multiplicity
            else:
                uhf_value = _frame_value(
                    data, "uhf", None, frame, nframes, default=None
                )
                multiplicity_value = _frame_value(
                    data, "multiplicity", None, frame, nframes, default=None
                )
                if uhf_value is None and multiplicity_value is None:
                    uhf_value = 0
            structures.append(
                Structure(
                    numbers,
                    positions_bohr[frame],
                    charge=_frame_value(
                        data, "charge", self.charge, frame, nframes, default=0.0
                    ),
                    uhf=uhf_value,
                    multiplicity=multiplicity_value,
                    spin_channels=self.spin_channels,
                )
            )

        calculator = BatchCalculator(structures, self.method, **self.kwargs)
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
        labeled["forces"] = np.asarray(result.forces, dtype=np.float64).reshape(
            nframes, -1, 3
        ) * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
        return labeled


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
    if key in data:
        value = np.asarray(data[key])
        if value.ndim == 0:
            return value.item()
        if value.size == nframes:
            return value[frame].item()
    return default


__all__ = ["GPUxtbDriver"]
