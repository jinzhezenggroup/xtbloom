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
Accordingly, `electronic_temperature` is the energy scale `k_B T` in Hartree; the default is
`300 K * 3.166808578545117e-6 Eh/K`, not the dimensionful value `300.0`.

## Compute semantics

At finite electronic temperature, the reported energy is the total electronic Helmholtz free
energy `F = E_internal - T*S_electronic`, including the Fermi-occupation entropy term used by
tblite. Forces are `-dF/dR`, which preserves the stationary finite-temperature SCC derivative. At
zero electronic temperature, `F` reduces to the internal energy.

The public batch call has two failure levels. Any failure detected before the final caller-output
commit begins leaves every result buffer and result flag unchanged. Once a CUDA caller-output
commit has begun, a later catastrophic failure returns `INTERNAL_ERROR` with an explicit diagnostic;
results may already have been modified because accepted device work cannot be rolled back. CUDA
attempts to restore the caller's current device on every exit. A restoration failure also returns
`INTERNAL_ERROR` and may leave that device selection changed, independently of whether output
commit began. Once the call returns success, all per-system diagnostics are valid even when
individual systems failed. `per_system_status` is then one of `SUCCESS`, `SCC_NOT_CONVERGED`, or
`EIGENSOLVER_FAILED`, and `scc_converged` is one exactly for `SUCCESS`. Requested floating-point
outputs for a failed system are quiet NaNs, committed for the complete system slice rather than as
partial energy, force, or charge results. Other systems in the ragged batch remain independent.

`scc_iterations` counts the SCC iteration whose eigensolve was last attempted. It is therefore the
converged iteration for a successful system, `max_scc_iterations` for ordinary nonconvergence, and
the failing iteration for an eigensolver error. Input-dependent electron-count and spin-parity
errors are validated for the complete batch before execution and are call-level invalid arguments,
not per-system SCC failures.

The ABI-v2 `scc_start_mode` suffix controls the electronic initial state for one call. `FRESH`
restores the immutable setup state and is also the meaning of every ABI-v1 or short options
prefix. CPU and CUDA both support strict `WARM`: it consumes the checkpoint from the latest fully
converged compatible batch call on the same context and never falls back to a fresh solve. A
compatible identity covers the complete topology and compute policy (molecular charge, spin, and
unpaired electrons; point-charge and periodic structure; SCC tolerances; iteration limit; and
electronic temperature). Geometry is not part of the identity, so a WARM call reuses the previous
converged electronic state as the initial SCC guess for the new coordinates and reconverges;
CUDA additionally keys its checkpoint to a geometry epoch and keeps modifying-Broyden history only
for a same-epoch reuse, while the CPU always restarts a fresh mixing window from the converged
state. A missing checkpoint or any identity change is a call-level invalid argument and leaves
caller outputs unchanged. On the CPU, requested-property/out-put flags are not part of the
identity. High-level Python calculators intentionally select `FRESH`; persistent warm policy is
exposed only by the low-level C/ctypes descriptor for now.

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

### CPU batch scheduling

One CPU context owns a fixed worker pool and reusable request/result staging. A public compute call
remains one serialized context transaction: it stages and validates the complete request, runs
independent `SystemExecution` objects concurrently, waits for every worker, and publishes outputs
in system-index order. This retains call-level transactionality, per-system numerical-failure
isolation, and bitwise batch-versus-serial results. Worker scheduling changes only which system
runs first, never the arithmetic order within a system.

`cpu_threads=1` executes on the calling thread without waking a background worker. A positive value
greater than one is clamped to the CPUs available in the process affinity mask; the calling thread
participates and the context retains the other workers. `cpu_threads=0` selects the affinity count
capped at 64, which avoids silently constructing an unbounded pool on large shared hosts. The
verified MKL provider remains LP64 and every factorization/eigensolve installs a thread-local MKL
limit of one, making the outer system scheduler the sole CPU parallel layer.

The reproducible CPU benchmark protocol compares identical warm public-C-API runs using
`cpu_threads=1` and an explicit affinity-constrained worker count, recording compiler, ISA,
affinity, MKL runtime, warm-up count, raw samples, and batch size. Batch-one latency is reported
separately and is not combined with throughput. A pinned regression threshold will be promoted to
CI only after the benchmark corpus includes gas, QM/MM, homogeneous, and heterogeneous workloads
on a named runner; local development measurements are evidence, not a portable CI gate.
