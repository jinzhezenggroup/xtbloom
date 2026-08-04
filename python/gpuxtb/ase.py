"""ASE calculator interface for gpuxtb.

The calculator exposes the same properties and parameter names as the ASE
calculators of comparable tight-binding codes (``tblite``). After attaching it
to an :class:`ase.atoms.Atoms` object, energies are returned in eV, forces in
eV/Angstrom, and Mulliken charges in elementary-charge units.

Example
-------
>>> from ase.build import molecule
>>> from gpuxtb.ase import GPUxtb
>>> atoms = molecule("H2O")
>>> atoms.calc = GPUxtb(method="GFN2-xTB")
>>> atoms.get_potential_energy()  # in eV
"""

try:
    import ase.calculators.calculator
    from ase.units import Bohr, Hartree
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("This submodule requires ASE installed") from e

from typing import Any, Dict, List, Optional

import numpy as np

from .exceptions import GPUxtbRuntimeError, GPUxtbValueError
from .interface import Calculator, _resolve_uhf, _validated_compute_setting


class GPUxtb(ase.calculators.calculator.Calculator):
    r"""ASE calculator for GFN2-xTB energies and analytic forces from gpuxtb.

    Supported properties: ``energy`` (alias ``free_energy``), ``forces``, and
    ``charges``.

    ======================== ================= =========================================
     Keyword                  Default           Description
    ======================== ================= =========================================
     method                   "GFN2-xTB"        Underlying tight-binding method
     charge                   None              Total charge (sum of initial charges)
     multiplicity             None              Total spin multiplicity
     electronic_temperature   300.0             Electronic temperature in kelvin
     max_scc_iterations       250               SCC iteration ceiling
     charge_tolerance         1e-6              SCC charge tolerance (e)
     energy_tolerance         1e-8              SCC energy tolerance (Hartree)
     backend                  "auto"            Execution backend: auto/cpu/cuda
     device_id                None              CUDA device id
     cpu_threads              1                 CPU batch-parallelism ceiling
     cache_api                True              Reuse the underlying API calculator
    ======================== ================= =========================================
    """

    implemented_properties = ["energy", "free_energy", "forces", "charges"]

    default_parameters = {
        "method": "GFN2-xTB",
        "charge": None,
        "multiplicity": None,
        "electronic_temperature": 300.0,
        "max_scc_iterations": 250,
        "charge_tolerance": 1.0e-6,
        "energy_tolerance": 1.0e-8,
        "backend": "auto",
        "device_id": None,
        "cpu_threads": 1,
        "cache_api": True,
    }

    _res = None
    _xtb = None

    def __init__(self, atoms=None, **kwargs):
        ase.calculators.calculator.Calculator.__init__(self, atoms=atoms, **kwargs)

    def set(self, **kwargs) -> dict:
        """Update parameters and reset cached results when they change."""

        def _validate(kwargs: Dict[str, Any]) -> None:
            allowed = set(self.default_parameters)
            unknown = set(kwargs) - allowed
            if unknown:
                raise GPUxtbValueError(
                    f"unknown calculator parameters: {sorted(unknown)}"
                )

        _validate(kwargs)
        for attribute in (
            "max_scc_iterations",
            "charge_tolerance",
            "energy_tolerance",
            "electronic_temperature",
        ):
            if attribute in kwargs:
                _validated_compute_setting(attribute, kwargs[attribute])
        if kwargs.get("multiplicity") is not None:
            _resolve_uhf(None, kwargs["multiplicity"])

        changed = ase.calculators.calculator.Calculator.set(self, **kwargs)
        if not changed:
            return changed

        self.reset()

        # A structural parameter change requires rebuilding the API calculator;
        # numerical SCC settings are pushed onto an existing one in place.
        if self._xtb is not None and not any(
            key in changed for key in ("method", "backend", "device_id", "cpu_threads")
        ):
            if "electronic_temperature" in changed:
                self._xtb.set(
                    "electronic_temperature", self.parameters.electronic_temperature
                )
            if "max_scc_iterations" in changed:
                self._xtb.set("max_scc_iterations", self.parameters.max_scc_iterations)
            if "charge_tolerance" in changed:
                self._xtb.set("charge_tolerance", self.parameters.charge_tolerance)
            if "energy_tolerance" in changed:
                self._xtb.set("energy_tolerance", self.parameters.energy_tolerance)
        else:
            self._close_api_calculator()
            self._res = None
        return changed

    def reset(self) -> None:
        ase.calculators.calculator.Calculator.reset(self)
        if not self.parameters.cache_api:
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
        atoms=None,
        properties: Optional[List[str]] = None,
        system_changes: List[str] = ase.calculators.calculator.all_changes,
    ) -> None:
        if not properties:
            properties = ["energy"]
        ase.calculators.calculator.Calculator.calculate(
            self, atoms, properties, system_changes
        )

        _validate_ase_atoms(self.atoms)
        # Atomic numbers are immutable in the native fixed-topology
        # calculator.  Reusing it after ASE changes the species would silently
        # evaluate the new coordinates with the old Hamiltonian parameters.
        needs_rebuild = self._xtb is None or not np.array_equal(
            self._xtb.numbers, self.atoms.numbers
        )
        if needs_rebuild:
            self._close_api_calculator()
            self._xtb = _create_api_calculator(self.atoms, self.parameters)
        else:
            self._xtb.update(
                positions=self.atoms.positions / Bohr,
                charge=_get_charge(self.atoms, self.parameters),
                uhf=_get_uhf(self.atoms, self.parameters),
            )

        try:
            self._res = self._xtb.singlepoint()
        except GPUxtbRuntimeError as e:
            raise ase.calculators.calculator.CalculationFailed(str(e)) from e

        self.results["energy"] = self._res["energy"] * Hartree
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = self._res["forces"] * Hartree / Bohr
        self.results["charges"] = self._res["charges"]


def _validate_ase_atoms(atoms) -> None:
    """Reject periodic ASE inputs that the public molecular ABI cannot model."""
    if np.any(atoms.pbc):
        raise ase.calculators.calculator.InputError(
            "gpuxtb does not support periodic ASE systems; the public C ABI "
            "has no lattice or periodic-boundary descriptor"
        )


def _create_api_calculator(atoms, parameters) -> Calculator:
    """Build the underlying gpuxtb API calculator for an ASE atoms object."""
    try:
        return Calculator(
            parameters.method,
            atoms.numbers,
            atoms.positions / Bohr,
            charge=_get_charge(atoms, parameters),
            uhf=_get_uhf(atoms, parameters),
            backend=parameters.backend,
            device_id=parameters.device_id,
            cpu_threads=parameters.cpu_threads,
            max_scc_iterations=parameters.max_scc_iterations,
            charge_tolerance=parameters.charge_tolerance,
            energy_tolerance=parameters.energy_tolerance,
            electronic_temperature=parameters.electronic_temperature,
        )
    except GPUxtbValueError as e:
        raise ase.calculators.calculator.InputError(str(e)) from e


def _get_charge(atoms, parameters) -> float:
    """Total charge from the explicit parameter or the initial atomic charges."""
    return (
        float(atoms.get_initial_charges().sum())
        if parameters.charge is None
        else float(parameters.charge)
    )


def _get_uhf(atoms, parameters) -> int:
    """Number of unpaired electrons from the multiplicity or initial magmoms."""
    if parameters.multiplicity is None:
        return int(atoms.get_initial_magnetic_moments().sum().round())
    return _resolve_uhf(None, parameters.multiplicity)


if "gpuxtb" not in ase.calculators.calculator.external_calculators:
    ase.calculators.calculator.register_calculator_class("gpuxtb", GPUxtb)
