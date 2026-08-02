# Architecture

gpuxtb is organized around one stable C ABI and replaceable compute backends. The first physics
target is gas-phase GFN2-xTB energy and analytic forces, followed by external point-charge
embedding inside the SCC loop. GFN1-xTB and ROCm remain explicit extension points.

## Data model

The public API represents a molecular batch as flat arrays plus `int64_t` offsets. This ragged
layout avoids padding small molecules to the largest basis in a batch, while still allowing
backend-specific bucketing by atom and orbital count. Every buffer carries a memory-space tag, so
the CUDA backend can consume device pointers directly and stage host pointers when necessary.

Point-charge embedding uses a caller-provided per-site screening gamma so the softened short-range
Coulomb form is unambiguous. Optional per-atom potential shifts `b` and symmetric charge-response
matrices `A` support periodic QM/MM embeddings through `b + A*q` in every SCC iteration and the
variational energy `q^T*b + 0.5*q^T*A*q`. The caller remains responsible for coordinate derivatives
of `b` and `A`, classical MM-MM interactions, periodic electrostatics, and virtual-site force
redistribution. The exact xTB-compatible equations and initial external-charge golden are documented
in [qmmm.md](qmmm.md).

All public real-valued quantities use binary64 and atomic units. Conversion belongs at language or
simulation-package bindings, not inside numerical kernels. Keeping one unit system is particularly
important for external-charge forces and finite-difference conformance tests.

## Layering

1. The C API validates ABI versions, pointer locations, shapes, and requested outputs.
2. The runtime selects CPU, CUDA, or a future ROCm backend and owns reusable workspaces.
3. Geometry preprocessing builds coordination data, basis metadata, and neighbor/pair lists.
4. Model terms build overlap, core Hamiltonian, repulsion, D4, and GFN2 multipole electrostatics.
5. The SCC engine batches eigensolves, occupations, density updates, and convergence acceleration.
6. Analytic derivatives accumulate atomic and external-point-charge forces without host round trips.

Immutable element and pair parameters should be packed once per device. Per-call allocations are
forbidden on the steady-state inference path; the context grows and reuses workspace instead.

## Correctness strategy

Correctness gates optimization. Golden cases will be generated independently with tblite and xtb,
covering elements, charges, spin states, SCC edge cases, and external point charges. Each physical
term is tested in isolation before total energies and forces. Analytic forces must also pass central
finite differences, including forces on embedding charges.

The initial tolerance target is defined by the conformance issue rather than hard-coded here. CPU
and CUDA use the same equations and parameters, with deterministic reductions available for
debugging even if the fastest production path relaxes reduction ordering.

## Performance strategy

The primary workload is many small and medium molecules. Batches will be bucketed by basis size so
that dense operations have similar shapes, while pairwise kernels operate on compact neighbor
lists. The implementation must avoid materializing the full GFN2 dipole-dipole pair tensor and must
never form an explicit inverse in generalized eigensolves. CUDA graphs, cuSOLVER batched
factorizations/eigensolvers, fused SCC updates, active-system compaction, and asynchronous copies
are introduced after profiling identifies their break-even points.

Benchmarks report latency and throughput separately, include warm-up, exclude one-time parameter
upload, and compare against current pinned revisions of xtb, tblite, and dxtb on the same hardware.
