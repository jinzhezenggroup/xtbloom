# Architecture

gpuxtb is organized around one stable C ABI and replaceable compute backends.
The implemented physics is restricted and unrestricted GFN2-xTB energy,
charges, and analytic forces, including external point charges and periodic
caller-supplied response inside SCC. GFN1-xTB and ROCm remain explicit,
unimplemented extension points.

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
in the [QM/MM theory guide](../theory/qmmm.md).

All public real-valued quantities use binary64 and atomic units. Conversion belongs at language or
simulation-package bindings, not inside numerical kernels. Keeping one unit system is particularly
important for external-charge forces and finite-difference conformance tests.
Accordingly, `electronic_temperature` is the energy scale `k_B T` in Hartree; the default is
`300 K * 3.166808578545117e-6 Eh/K`, not the dimensionful value `300.0`.

## Compute semantics

At finite electronic temperature, the reported energy is the total electronic
Helmholtz free energy
$F = E_{\mathrm{internal}} - (k_{\mathrm B}T)S_{\mathrm{electronic}}$,
including the Fermi-occupation entropy term used by tblite. Forces are
$-\partial F/\partial R$, which preserves the stationary finite-temperature
SCC derivative. At zero electronic temperature, $F$ reduces to the internal
energy.

Finite-temperature occupations of an exactly degenerate eigenspace are published symmetrically:
every orbital in an equal-energy block shares one binary64 value, keeping the populations unitary-
and permutation-invariant. When the requested electron/hole count falls strictly between two such
symmetric states (for example three exactly degenerate orbitals with `nel = nextafter(3, 0)`, whose
single residual hole cannot be split equally across the three published occupations), the solver
relaxes to the nearest representable symmetric state instead of failing the system. The published
electron/hole error is then bounded absolutely by the block quantization scale
`2 * eps_double * block_count` (a few ULPs of an electron, where `eps_double` is `2^-52` and
`block_count` is the number of orbitals in the selected exact-degeneracy block). Relaxation is never
authorized by a singleton or by the total spectrum size. The solver collects candidates across all
energy blocks before selecting: any state inside the strict publication tolerance globally precedes
every relaxed state. Within either class, it minimizes the binary64 compensated-sum count error,
then prefers a fractional candidate, the lower occupation at an exact error tie, and finally the
lower block index. Candidate evaluation includes the directly solved block occupation, its two
nearest binary64 neighbors on each side, and the original publication value with its immediate
neighbors. CPU reconstructs this rare-path candidate baseline with the same translated binary64
root and compensated summation order used by CUDA; its long-double root remains the independent
ideal-conservation check. If binary64 root spacing is exhausted outside the ordinary root tolerance,
the final bracket still straddles the target, and exactly one multi-orbital degenerate block changes
electron or hole contribution across that bracket, both backends may retry once with that frontier
as the translated reference. A failed or non-improving retry leaves the original root unchanged.
Only that uniquely identified causal frontier can supply the block-quantization floor for root
acceptance, including when publication subsequently finds a strict singleton rescue; an unrelated
degeneracy cannot widen the root gate. If the first phase must select a relaxed block, one bounded
second phase searches strict candidates only from that vector; if none exists, the first relaxed
state remains. Every shared rare-path final strict or relaxed decision is audited with the same
double-double residual interval comparison. The reported entropy is derived from those same
published occupations. When an exactly degenerate block's valid subnormal total cannot be divided
into a nonzero binary64 per-member fraction, CPU and CUDA use the same analytic equal-level logit for
the finite chemical potential. Outside that fully degenerate analytic override, a mixed-spectrum
subnormal jump has no canonical cross-backend binary64 chemical potential: both backends require a
finite diagnostic while the shared publication, count, and entropy remain deterministic; broader
chemical-potential parity remains tracked by #54. A residual beyond the selected block's quantization
bound indicates genuine electron non-conservation and still fails deterministically. Nondegenerate
spectra retain the strict publication tolerance, and the zero-temperature Aufbau path is unaffected.
See issue #31 for the original representability question.

### Cross-engine degenerate-occupation evidence

The representability policy above is observable through the public API, rather
than only through an internal occupation helper. Issue
[#205](https://github.com/njzjz/gpuxtb/issues/205) records a cross-engine probe
measured on 2026-08-07 with GFN2-xTB at 300 K. The common geometry is three
hydrogen atoms on the x axis:

```text
atomic_numbers = [1, 1, 1]
positions_bohr = [[0, 0, 0], [1e20, 0, 0], [2e20, 0, 0]]
```

From a source checkout with a compatible CPU linear-algebra runtime, the gpuxtb
rows are reproduced through the same installed Python package and public C ABI
used by applications:

```console
GPUXTB_ENABLE_CUDA=OFF uv sync --no-editable --extra test
uv run --no-sync python - <<'PY'
import numpy as np
from gpuxtb import Calculator

numbers = np.array([1, 1, 1])
positions = np.array([[0.0, 0.0, 0.0], [1.0e20, 0.0, 0.0], [2.0e20, 0.0, 0.0]])
fractional_charge = 3.0 - 2.0 * np.nextafter(3.0, 0.0)

for charge, uhf in [(0.0, 1), (-3.0, 0), (fractional_charge, 0)]:
    with Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        backend="cpu",
        electronic_temperature=300.0,
    ) as calc:
        result = calc.singlepoint()
    print(charge, uhf, result.scc_status, f"{result.energy:.16f}")
PY
```

At this separation the three one-center orbital blocks are bitwise identical
and the inter-center interactions safely tend to zero. This creates an exact
three-fold degeneracy without relying on an approximate molecular symmetry.

| Electron case | gpuxtb, current public CPU path | xTB 6.7.1 (`edcfbbe`) | tblite | dxtb 0.4.0 (`libcint`) |
| --- | --- | --- | --- | --- |
| charge `0`, `uhf=1` | `SUCCESS`, finite energy `-1.1960140851592562 Eh` | SCC does not converge; abnormal termination with floating-point exception flags | 0.7.0 succeeds at `-1.184076585161 Eh`; 0.6.0 CLI segfaults | returns an invalid overflow-scale value near `3.46e47 Eh` |
| charge `-3`, `uhf=0` | `SUCCESS`, finite energy `-1.8322400836158348 Eh` | Hamiltonian diagonalization fails and the process terminates abnormally | 0.7.0 raises `(sygvd) failed to solve eigenvalue problem, info=2` | raises `Fermi energy failed to converge` |
| charge `3 - 2*nextafter(3,0)`, `uhf=0` | `SUCCESS`, finite energy `-1.8322400836158348 Eh`; each restricted spin requests exactly `nextafter(3,0)` electrons | not expressible through integer `--chrg`; no failure claim is made | 0.7.0 Python succeeds at `-1.832240082284 Eh`; the CLI expressibility difference is not treated as a failure | raises `Fermi energy failed to converge` |

The comparison establishes a specific numerical advantage, not a universal
ranking: tblite 0.7 handles two rows, while gpuxtb handles all three under one
documented symmetric-publication rule. The fractional-charge row is included
to exercise the exact binary64 policy and is not counted as an xTB failure
because the xTB CLI cannot represent the input. The gpuxtb regression is
`test_degenerate_occupation_representability_is_publicly_successful` in
[`tests/cpu_public_inference_test.cpp`](../../tests/cpu_public_inference_test.cpp).
Issue #205 retains the reference-engine environment and summarized outcomes.

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
compatible identity covers the complete topology and compute policy (requested-property flags;
molecular charge, spin, and unpaired electrons; point-charge and periodic structure; SCC tolerances;
iteration limit; and electronic temperature). Geometry is not part of the identity, so a WARM call
reuses the previous converged electronic state as the initial SCC guess for the new coordinates and
reconverges;
CUDA additionally keys its checkpoint to a geometry epoch and keeps modifying-Broyden history only
for a same-epoch reuse, while the CPU always restarts a fresh mixing window from the converged
state. A missing checkpoint or any identity change is a call-level invalid argument and leaves
caller outputs unchanged. CPU and CUDA use the same compute-options identity, including
requested-property/output flags. High-level Python calculators select `FRESH` by default;
`Calculator` and `BatchCalculator` also expose opt-in transparent warm start, which retries one
`FRESH` solve when the strict native gate rejects an incompatible checkpoint. The ASE calculator
enables that policy by default for dynamics-like geometry sequences. Automatic batch slicing
remains incompatible with warm start because one native context retains only its latest whole-batch
checkpoint, not one checkpoint per logical chunk.

## External interaction attachments

The ABI-v3 batch suffix adds a generic attachment slot — `total_interactions`,
`interaction_descriptors`, and `interaction_payload` — so external potentials
and self-consistent models (uniform electric field, field gradient, multipole
point charges, ALPB/GBSA/GB/GBE/ddX solvation, D3/D4 dispersion variants,
halogen-bond corrections) do not each regrow the batch layout. One
`gpuxtb_interaction_t` descriptor ties a versioned payload block to one batch
item; block contents are versioned per tag through a leading `block_version`.

Reserved attachments follow a strict validate-then-refuse policy: the common
validator proves descriptor/payload extents, memory-space tags, and every
semantic relationship available from host-resident storage, then the request
is refused with `NOT_IMPLEMENTED` before any backend execution or caller-output
commit. A caller can therefore never observe a result where a reserved
interaction was silently ignored. Unknown or `NONE` tags and structurally
malformed host-resident attachments are `INVALID_ARGUMENT`. Device-resident
descriptor content is marked with `kInteractionDescriptorsNeedStaging`, while
device payload content is marked independently with
`kInteractionPayloadNeedsStaging`. P3 must stage and validate every marked
interaction byte before enabling CUDA execution; P1's availability gate
refuses the request first because no backend can consume it yet. Host-resident
electric-field blocks are byte-loaded and checked for version 1, a zero
reserved field, and finite values before that gate. The ABI-v2 result suffix
reserves the dipole-moment outlet and
`GPUXTB_RESULT_DIPOLE_MOMENTS` publication flag alongside `quadrupole_moments`,
`wiberg_orders`, and `spin_populations`; the latter three have no released
shape contract, must remain NULL until published, and return `NOT_SUPPORTED`
when supplied.

## Fixed-topology plans and workspace sizing

`gpuxtb_compute` remains the convenience path: it validates the descriptor set, prepares the
per-system plans, runs the batch, and publishes outputs on one context transaction. For workloads
that reuse one ragged topology and compute policy across many geometries, `gpuxtb_plan_create`
binds the immutable topology (atom offsets, element numbers, molecular charges, unpaired electrons,
spin channels, and point-charge/response structure) plus the model, requested properties, SCC
tolerances, iteration limit, and electronic temperature at creation time. `FRESH` versus `WARM`
remains a per-call choice. Geometry is not part of the plan, so repeated `gpuxtb_plan_compute`
calls can change positions, point-charge positions and values, and periodic `b/A` values freely.

Plan creation is the allocation-permitted setup path: it validates the request and pre-warms the
plan-owned backend execution cache so the first and subsequent `gpuxtb_plan_compute` calls for the
same topology and policy perform zero steady-state allocations. Independent plans cannot evict one
another through the context convenience cache. `gpuxtb_plan_query_workspace` returns the reusable
host/device workspace bytes and alignment the plan keeps alive for its property set; sizes differ by backend
(the CPU backend uses no device memory) and by requested properties (forces reserve device arenas
and CPU output staging). CUDA accounting measures every prepared numerical, result, validation,
setup-owner, model-plan, and mixed-memory topology staging allocation instead of reconstructing
their layouts at the public boundary. Opaque provider and Graph bookkeeping remains outside caller
workspace totals.
A plan is bound to its creating context, must be destroyed before that
context, and a batch whose immutable topology differs from the plan fails with
`GPUXTB_STATUS_INVALID_ARGUMENT` before any caller output is modified, giving the corruption gate
for reused handles. The plan handle and workspace query are ABI-versioned in `gpuxtb.h` and are
mirrored by the Python `Plan` binding and the installed consumer.

## Layering

1. The C API validates ABI versions, pointer locations, shapes, and requested outputs.
2. The runtime selects CPU, CUDA, or a future ROCm backend and owns reusable workspaces.
3. Geometry preprocessing builds coordination data, basis metadata, and neighbor/pair lists.
4. Model terms build overlap, core Hamiltonian, repulsion, D4, and GFN2 multipole electrostatics.
5. The SCC engine batches eigensolves, occupations, density updates, and convergence acceleration.
6. Analytic derivatives accumulate atomic and external-point-charge forces without host round trips.

Immutable element and pair parameters should be packed once per device. Per-call allocations are
forbidden on the steady-state inference path; the context grows and reuses workspace instead.

### Optional host-runtime loading

libgpuxtb does not require a proprietary BLAS or CUDA host shared library merely to load. The CPU
eigensolver dlopens an LP64 BLAS/LAPACK runtime (Intel MKL or OpenBLAS) by SONAME on first use
(`src/model/gfn2/eigensolver.cpp`), so a machine without a compatible provider still loads the
library. The MKL path is host-isolated: CMake builds a private
`libgpuxtb_mkl_lp64_shim` with fixed `DT_NEEDED` dependencies on `libmkl_intel_lp64`,
`libmkl_sequential`, and `libmkl_core`. The factory loads the adjacent shim with `RTLD_LOCAL`
in a new glibc link-map namespace; `RTLD_LOCAL` in the base namespace would still allow a
globally loaded host runtime to interpose on the component dependencies. The shim deliberately
uses `DT_RPATH` rather than `DT_RUNPATH`, so `LD_LIBRARY_PATH` cannot substitute same-SONAME
components from a different MKL installation for the configure-time cohort.
gpuxtb never loads `libmkl_rt`, never calls `MKL_Set_Interface_Layer`, and never reads
`MKL_INTERFACE_LAYER`, so an embedding process's MKL interface/threading state is untouched and
LP64 gpuxtb calls stay correct even when the host uses ILP64. A shared `libgpuxtb` locates the shim
in its own directory. Because a static archive has no runtime module directory, a static MKL
consumer must stage the installed shim beside its final executable; a missing sibling produces a
deterministic backend-unavailable error rather than a base-namespace or `libmkl_rt` fallback. On
Linux the CUDA build generates
one ELF trampoline shim per wrapped host library
(cudart, cuBLAS, cuSOLVER, and libcuda) from the byte-pinned
`cmake/3rdparty/implib` source and compiles those shims into libgpuxtb itself
(`src/runtime/cuda_dlopen.c`). An early ELF constructor opens the exact build-major SONAMEs and
pre-resolves each complete symbol cohort before ordinary NVCC registration constructors run. The
resolved tables are then immutable, avoiding races on concurrent CUDA calls. A host without the
NVIDIA runtime can therefore load libgpuxtb and receive a backend-unavailable diagnostic instead
of failing at the ELF loader boundary.

This host-library indirection is not a GPL compatibility claim: dynamic loading can still combine
works under copyright law. The distribution basis is instead the GPLv3 Section 7 additional
permission in `CUDA_MKL_LINKING_EXCEPTION`. gpuxtb passes `--cudadevrt=none` at device link because
it uses no CUDA Dynamic Parallelism, but nvcc may still incorporate NVIDIA libdevice code. That
code and every separately installed CUDA or MKL provider remain under vendor terms. The source
provenance and packaging contract are recorded in `cmake/3rdparty/implib_manifest.json` and
`THIRD_PARTY_NOTICES.md`.

## Correctness strategy

Correctness gates optimization. Committed conformance cases are generated
independently with pinned tblite and xTB workflows and cover charges, spin
states, SCC behavior, and external point charges. Each physical term is tested
in isolation before total energies and forces. Analytic forces also pass
central finite differences, including forces on embedding charges.

Tolerance targets belong to the versioned conformance corpus and executable
tests rather than prose documentation. CPU and CUDA use the same equations and
parameters, with deterministic reductions available for debugging even when a
production path uses a different valid reduction order.

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
verified BLAS provider remains LP64 and every factorization/eigensolve installs a provider-local
thread limit of one, making the outer system scheduler the sole CPU parallel layer.

The reproducible CPU benchmark protocol compares identical warm public-C-API runs using
`cpu_threads=1` and an explicit affinity-constrained worker count, recording compiler, ISA,
affinity, BLAS runtime, warm-up count, raw samples, and batch size. Batch-one latency is reported
separately and is not combined with throughput. A pinned regression threshold will be promoted to
CI only after the benchmark corpus includes gas, QM/MM, homogeneous, and heterogeneous workloads
on a named runner; local development measurements are evidence, not a portable CI gate.
