"""ASE calculator interface for xTBloom.

The calculator exposes the same properties and parameter names as the ASE
calculators of comparable tight-binding codes (``tblite``). After attaching it
to an :class:`ase.atoms.Atoms` object, energies are returned in eV, forces in
eV/Angstrom, and Mulliken charges in elementary-charge units.

Example
-------
>>> from ase.build import molecule
>>> from xtbloom.ase import XTBloom
>>> atoms = molecule("H2O")
>>> atoms.calc = XTBloom(method="GFN2-xTB")
>>> atoms.get_potential_energy()  # in eV
"""

try:
    import ase.calculators.calculator
    from ase.units import Bohr, Hartree
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("This submodule requires ASE installed") from e

import typing
from typing import Any

import numpy as np

from .exceptions import (
    XTBloomNotSupportedError,
    XTBloomRuntimeError,
    XTBloomValueError,
)
from .interface import (
    Calculator,
    Result,
    _normalize_periodic,
    _resolve_uhf,
    _validated_compute_setting,
)


class XTBloom(ase.calculators.calculator.Calculator):
    r"""ASE calculator for GFN1/GFN2-xTB energies and forces from xTBloom.

    Supported properties: ``energy`` (alias ``free_energy``), ``forces``, and
    ``charges``.

    ======================== ================= =========================================
     Keyword                  Default           Description
    ======================== ================= =========================================
     method                   "GFN2-xTB"        GFN1-xTB or GFN2-xTB
     charge                   None              Total charge (sum of initial charges)
     multiplicity             None              Total spin multiplicity
     electronic_temperature   300.0             Electronic temperature in kelvin
     max_scc_iterations       250               SCC iteration ceiling
     charge_tolerance         1e-6              SCC charge tolerance (e)
     energy_tolerance         1e-8              SCC energy tolerance (Hartree)
     scc_mixer               "modified_broyden" SCC mixing algorithm
     scc_mixer_history        8                 Broyden history vectors (1..64)
     scc_mixer_damping        0.4               Broyden damping in (0, 1]
     determinism             "default"          default/reproducible execution policy
     backend                  "auto"            Execution backend for either model
     device_id                None              CUDA device id
     cpu_threads              1                 CPU batch-parallelism ceiling
     cache_api                True              Reuse the underlying API calculator
     warm_start               True              Seed SCC from the previous converged
                                               state when compatible (auto)
    ======================== ================= =========================================
    """

    # Class-level defaults in the ASE calculator convention. ASE reads them
    # (and `set()` validates against them); instances receive their own copy
    # through ``Calculator.__init__``, so they must never be mutated here.
    # ASE's base class types these names as instance variables, so a ClassVar
    # override is rejected by the type checker; they are still class-level
    # defaults that ASE copies per-instance in ``Calculator.__init__``, so
    # there is no shared-mutable-default hazard.
    implemented_properties: list[str] = [  # noqa: RUF012
        "energy",
        "free_energy",
        "forces",
        "charges",
    ]

    default_parameters: dict[str, Any] = {  # noqa: RUF012
        "method": "GFN2-xTB",
        "charge": None,
        "multiplicity": None,
        "electronic_temperature": 300.0,
        "max_scc_iterations": 250,
        "charge_tolerance": 1.0e-6,
        "energy_tolerance": 1.0e-8,
        "scc_mixer": "modified_broyden",
        "scc_mixer_history": 8,
        "scc_mixer_damping": 0.4,
        "determinism": "default",
        "backend": "auto",
        "device_id": None,
        "cpu_threads": 1,
        "cache_api": True,
        "warm_start": True,
    }

    _res: Result | None = None
    _xtb: Calculator | None = None

    def _api_parameters(self) -> ase.calculators.calculator.Parameters:
        """Return the validated ASE parameter mapping used by this calculator.

        ASE's ``Parameters`` stores values in a dict but exposes attribute
        access, so after :meth:`set` this is always an attribute-accessible
        mapping. The cast only restores that runtime contract for the type
        checker; no runtime conversion happens.
        """
        return typing.cast("ase.calculators.calculator.Parameters", self.parameters)

    def __init__(self, atoms: ase.Atoms | None = None, **kwargs: object) -> None:
        ase.calculators.calculator.Calculator.__init__(self, atoms=atoms, **kwargs)

    def set(self, **kwargs: object) -> dict[str, object]:
        """Update parameters and reset cached results when they change."""

        def _validate(kwargs: dict[str, Any]) -> None:
            allowed = set(self.default_parameters)
            unknown = set(kwargs) - allowed
            if unknown:
                raise XTBloomValueError(
                    f"unknown calculator parameters: {sorted(unknown)}"
                )

        _validate(kwargs)
        for attribute in (
            "max_scc_iterations",
            "charge_tolerance",
            "energy_tolerance",
            "electronic_temperature",
            "scc_mixer",
            "scc_mixer_history",
            "scc_mixer_damping",
            "determinism",
        ):
            if attribute in kwargs:
                _validated_compute_setting(attribute, kwargs[attribute])
        if kwargs.get("multiplicity") is not None:
            _resolve_uhf(None, typing.cast("int | None", kwargs["multiplicity"]))

        changed = ase.calculators.calculator.Calculator.set(self, **kwargs)
        if not changed:
            return changed

        self.reset()

        # A structural parameter change requires rebuilding the API calculator;
        # numerical SCC settings are pushed onto an existing one in place.
        if self._xtb is not None and not any(
            key in changed
            for key in ("method", "backend", "device_id", "cpu_threads", "warm_start")
        ):
            parameters = self._api_parameters()
            if "electronic_temperature" in changed:
                self._xtb.set(
                    "electronic_temperature", parameters.electronic_temperature
                )
            if "max_scc_iterations" in changed:
                self._xtb.set("max_scc_iterations", parameters.max_scc_iterations)
            if "charge_tolerance" in changed:
                self._xtb.set("charge_tolerance", parameters.charge_tolerance)
            if "energy_tolerance" in changed:
                self._xtb.set("energy_tolerance", parameters.energy_tolerance)
            if "scc_mixer" in changed:
                self._xtb.set("scc_mixer", parameters.scc_mixer)
            if "scc_mixer_history" in changed:
                self._xtb.set("scc_mixer_history", parameters.scc_mixer_history)
            if "scc_mixer_damping" in changed:
                self._xtb.set("scc_mixer_damping", parameters.scc_mixer_damping)
            if "determinism" in changed:
                self._xtb.set("determinism", parameters.determinism)
        else:
            self._close_api_calculator()
            self._res = None
        return changed

    def reset(self) -> None:
        """Clear calculated properties and optional native API state."""
        ase.calculators.calculator.Calculator.reset(self)
        if not self._api_parameters().cache_api:
            self._close_api_calculator()
            self._res = None

    def _close_api_calculator(self) -> None:
        """Release native worker pools, handles, and caches before replacement."""
        if self._xtb is not None:
            self._xtb.close()
            self._xtb = None

    def close(self) -> None:
        """Release the cached native calculator explicitly."""
        self._close_api_calculator()
        self._res = None

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = ase.calculators.calculator.all_changes,
    ) -> None:
        """Calculate requested ASE properties for molecular or XYZ-periodic input."""
        if not properties:
            properties = ["energy"]
        ase.calculators.calculator.Calculator.calculate(
            self, atoms, properties, system_changes
        )

        # ASE's ``Calculator.calculate`` assigns ``self.atoms`` (and validates
        # it) before returning, so the cast below only restores that runtime
        # contract for the type checker; the object is the real ``ase.Atoms``.
        atoms = typing.cast("ase.Atoms", self.atoms)
        parameters = self._api_parameters()

        _validate_ase_atoms(atoms)
        # Atomic numbers are immutable in the native fixed-topology
        # calculator.  Reusing it after ASE changes the species would silently
        # evaluate the new coordinates with the old Hamiltonian parameters.
        xtb = self._xtb
        if xtb is None or not np.array_equal(xtb.numbers, atoms.numbers):
            self._close_api_calculator()
            self._xtb = _create_api_calculator(atoms, parameters)
            xtb = self._xtb
        else:
            xtb.update(
                positions=atoms.positions / Bohr,
                charge=_get_charge(atoms, parameters),
                uhf=_get_uhf(atoms, parameters),
                cell=_ase_cell(atoms),
                pbc=_ase_pbc(atoms),
            )

        assert xtb is not None
        try:
            self._res = xtb.singlepoint()
        except XTBloomRuntimeError as e:
            raise ase.calculators.calculator.CalculationFailed(str(e)) from e

        # Attribute access is behaviorally identical to ``Result["energy"]``:
        # the string getter maps straight onto these attributes.
        self.results["energy"] = self._res.energy * Hartree
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = self._res.forces * Hartree / Bohr
        self.results["charges"] = self._res.charges


def _validate_ase_atoms(atoms: ase.Atoms) -> None:
    """Validate ASE's cell/PBC pair against the released native 3D contract."""
    try:
        _normalize_periodic(_ase_cell(atoms), _ase_pbc(atoms))
    except (XTBloomNotSupportedError, XTBloomValueError) as error:
        raise ase.calculators.calculator.InputError(str(error)) from error


def _ase_pbc(atoms: ase.Atoms) -> bool | np.ndarray:
    """Return ASE periodicity while preserving partial-axis information."""
    pbc = np.asarray(atoms.pbc)
    if pbc.ndim == 0:
        return bool(pbc)
    return np.asarray(pbc, dtype=bool)


def _ase_cell(atoms: ase.Atoms) -> np.ndarray | None:
    """Return an ASE direct cell in bohr, or ``None`` for a molecule."""
    if not np.any(np.asarray(atoms.pbc)):
        return None
    return np.asarray(atoms.cell.array, dtype=np.float64) / Bohr


def _create_api_calculator(
    atoms: ase.Atoms, parameters: ase.calculators.calculator.Parameters
) -> Calculator:
    """Build the underlying xTBloom API calculator for an ASE atoms object."""
    try:
        return Calculator(
            parameters.method,
            atoms.numbers,
            atoms.positions / Bohr,
            charge=_get_charge(atoms, parameters),
            uhf=_get_uhf(atoms, parameters),
            cell=_ase_cell(atoms),
            pbc=_ase_pbc(atoms),
            backend=parameters.backend,
            device_id=parameters.device_id,
            cpu_threads=parameters.cpu_threads,
            max_scc_iterations=parameters.max_scc_iterations,
            charge_tolerance=parameters.charge_tolerance,
            energy_tolerance=parameters.energy_tolerance,
            electronic_temperature=parameters.electronic_temperature,
            scc_mixer=parameters.scc_mixer,
            scc_mixer_history=parameters.scc_mixer_history,
            scc_mixer_damping=parameters.scc_mixer_damping,
            determinism=parameters.determinism,
            warm_start=bool(parameters.warm_start),
        )
    except (XTBloomNotSupportedError, XTBloomValueError) as e:
        raise ase.calculators.calculator.InputError(str(e)) from e


def _get_charge(
    atoms: ase.Atoms, parameters: ase.calculators.calculator.Parameters
) -> float:
    """Total charge from the explicit parameter or the initial atomic charges."""
    return (
        float(atoms.get_initial_charges().sum())
        if parameters.charge is None
        else float(parameters.charge)
    )


def _get_uhf(
    atoms: ase.Atoms, parameters: ase.calculators.calculator.Parameters
) -> int:
    """Return the nonnegative unpaired-electron count from ASE spin metadata."""
    if parameters.multiplicity is not None:
        return _resolve_uhf(None, parameters.multiplicity)

    total_moment = float(np.asarray(atoms.get_initial_magnetic_moments()).sum())
    if not np.isfinite(total_moment):
        raise ase.calculators.calculator.InputError(
            "initial magnetic moments must sum to a finite integer number of "
            "unpaired electrons; set multiplicity explicitly"
        )
    rounded_moment = round(total_moment)
    if not np.isclose(total_moment, rounded_moment, rtol=0.0, atol=1.0e-8):
        raise ase.calculators.calculator.InputError(
            "initial magnetic moments must sum to an integer number of unpaired "
            "electrons; set multiplicity explicitly for a nonintegral total"
        )
    return abs(int(rounded_moment))


if "xtbloom" not in ase.calculators.calculator.external_calculators:
    ase.calculators.calculator.register_calculator_class("xtbloom", XTBloom)
