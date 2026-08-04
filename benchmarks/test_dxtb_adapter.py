"""Protocol tests for the persistent in-process dxtb benchmark adapter."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import torch

from benchmarks.dxtb_adapter import DxtbAdapter, DxtbError, timed_invoke


class FakeCalculator:
    """Minimal differentiable dxtb calculator used without dxtb dependencies."""

    instances: ClassVar[list[FakeCalculator]] = []

    def __init__(
        self,
        numbers: torch.Tensor,
        parameters: object,
        *,
        opts: dict[str, object],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.numbers = numbers
        self.parameters = parameters
        self.opts = opts
        self.device = device
        self.dtype = dtype
        self.reset_count = 0
        self.singlepoint_count = 0
        self.singlepoint_kwargs: list[dict[str, object]] = []
        self.__class__.instances.append(self)

    def reset(self) -> None:
        self.reset_count += 1

    def singlepoint(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        spins: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        del charges, spins
        self.singlepoint_count += 1
        self.singlepoint_kwargs.append(kwargs)
        # One atom-resolved Hartree contribution per coordinate row.  This
        # leaves an exact force of -2R for adapter normalization checks.
        return SimpleNamespace(total=positions.square().sum(dim=-1))


def fake_dxtb() -> SimpleNamespace:
    """Return the public dxtb surface consumed by the adapter."""

    return SimpleNamespace(
        Calculator=FakeCalculator,
        GFN2_XTB=object(),
        __version__="test-version",
        __file__="/fake/dxtb/__init__.py",
        timer=SimpleNamespace(cuda_sync=True),
    )


def make_storage(batch_size: int) -> SimpleNamespace:
    """Build a conformer batch with two real atoms in every system."""

    atomic_numbers: list[int] = []
    positions: list[float] = []
    slices: list[SimpleNamespace] = []
    for system in range(batch_size):
        begin = len(atomic_numbers)
        atomic_numbers.extend((6, 8))
        # Different coordinates ensure the adapter does not accidentally
        # publish one cached result for every logical system.
        positions.extend((float(system + 1), 0.0, 0.0, 0.0, 2.0, -1.0))
        slices.append(
            SimpleNamespace(
                atom_begin=begin,
                atom_end=begin + 2,
                point_begin=0,
                point_end=0,
            )
        )
    return SimpleNamespace(
        atomic_numbers=atomic_numbers,
        positions=positions,
        molecular_charges=[0.0] * batch_size,
        unpaired_electrons=[0] * batch_size,
        point_charge_values=[],
        slices=slices,
    )


class DxtbAdapterTest(unittest.TestCase):
    """Exercise batch construction, cache semantics, units, and synchronization."""

    def setUp(self) -> None:
        FakeCalculator.instances.clear()

    def test_force_conformer_batches_1_8_32_128_are_persistent(self) -> None:
        original_threads = torch.get_num_threads()
        for batch_size in (1, 8, 32, 128):
            runtime_dxtb = fake_dxtb()
            adapter = DxtbAdapter(
                make_storage(batch_size),
                "force",
                "cpu",
                cpu_threads=original_threads,
                _runtime=(torch, runtime_dxtb),
            )
            try:
                self.assertEqual(adapter.batch_mode, 2)
                self.assertEqual(tuple(adapter.numbers.shape), (batch_size, 2))
                self.assertTrue(adapter.positions.requires_grad)
                self.assertFalse(runtime_dxtb.timer.cuda_sync)
                adapter.invoke()
                first = adapter.results()
                adapter.invoke()
                second = adapter.results()
                calculator = adapter.calculator
                self.assertEqual(calculator.reset_count, 2)
                self.assertEqual(calculator.singlepoint_count, 2)
                self.assertIs(calculator, FakeCalculator.instances[-1])
                self.assertEqual(first, second)
                self.assertEqual(
                    first["energies_hartree"],
                    [float((system + 1) ** 2 + 5) for system in range(batch_size)],
                )
                expected_forces: list[float] = []
                for system in range(batch_size):
                    expected_forces.extend(
                        (-2.0 * (system + 1), 0.0, 0.0, 0.0, -4.0, 2.0)
                    )
                self.assertEqual(first["forces_hartree_per_bohr"], expected_forces)
                self.assertEqual(
                    calculator.singlepoint_kwargs,
                    [
                        {"cuda_sync_in_scf": False},
                        {"cuda_sync_in_scf": False},
                    ],
                )
            finally:
                adapter.close()
            self.assertTrue(runtime_dxtb.timer.cuda_sync)
            self.assertEqual(torch.get_num_threads(), original_threads)

    def test_ragged_batch_uses_padding_but_filters_padded_forces(self) -> None:
        storage = SimpleNamespace(
            atomic_numbers=[1, 1, 8],
            positions=[1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            molecular_charges=[0.0, 0.0],
            unpaired_electrons=[0, 0],
            point_charge_values=[],
            slices=[
                SimpleNamespace(atom_begin=0, atom_end=2),
                SimpleNamespace(atom_begin=2, atom_end=3),
            ],
        )
        adapter = DxtbAdapter(
            storage,
            "force",
            "cpu",
            cpu_threads=torch.get_num_threads(),
            _runtime=(torch, fake_dxtb()),
        )
        try:
            self.assertEqual(adapter.batch_mode, 1)
            self.assertEqual(adapter.numbers.tolist(), [[1, 1], [8, 0]])
            adapter.invoke()
            self.assertEqual(
                adapter.results(),
                {
                    "energies_hartree": [2.0, 4.0],
                    "forces_hartree_per_bohr": [
                        -2.0,
                        0.0,
                        0.0,
                        2.0,
                        0.0,
                        0.0,
                        0.0,
                        -4.0,
                        0.0,
                    ],
                },
            )
        finally:
            adapter.close()

    def test_energy_does_not_create_force_output(self) -> None:
        adapter = DxtbAdapter(
            make_storage(1),
            "energy",
            "cpu",
            cpu_threads=torch.get_num_threads(),
            _runtime=(torch, fake_dxtb()),
        )
        try:
            self.assertFalse(adapter.positions.requires_grad)
            with self.assertRaisesRegex(DxtbError, "before a successful invoke"):
                adapter.results()
            adapter.invoke()
            self.assertEqual(adapter.results(), {"energies_hartree": [6.0]})
        finally:
            adapter.close()

    def test_rejects_point_charges_before_importing_runtime(self) -> None:
        storage = make_storage(1)
        storage.point_charge_values = [0.5]
        with self.assertRaisesRegex(DxtbError, "external point-charge SCC"):
            DxtbAdapter(storage, "energy", "cpu")

    def test_rejects_cuda_when_pytorch_has_no_visible_device(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            self.assertRaisesRegex(DxtbError, "torch.cuda is unavailable"),
        ):
            DxtbAdapter(
                make_storage(1),
                "energy",
                "cuda",
                _runtime=(torch, fake_dxtb()),
            )

    def test_cuda_synchronize_uses_selected_device(self) -> None:
        cuda = SimpleNamespace(synchronize=mock.Mock())
        adapter = object.__new__(DxtbAdapter)
        adapter._closed = False
        adapter.backend = "cuda"
        adapter.device = "cuda:3"
        adapter.torch = SimpleNamespace(cuda=cuda)
        adapter.synchronize()
        cuda.synchronize.assert_called_once_with("cuda:3")

    def test_timed_invoke_places_completion_after_submission(self) -> None:
        events: list[str] = []
        adapter = SimpleNamespace(
            invoke=lambda: events.append("invoke"),
            synchronize=lambda: events.append("synchronize"),
        )
        elapsed = timed_invoke(adapter)  # type: ignore[arg-type]
        self.assertEqual(events, ["invoke", "synchronize"])
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
