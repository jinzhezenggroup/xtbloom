# Issue #84 pre-migration baseline and Step 1-2 evidence

Leaf: #220 branch `issue84-projection-migration`, derived from `main` @ e16bfa2.

## Validation baseline (before any change)

Command:
```
cmake -S . -B build/cuda-sm120 [-DXTBLOOM_ENABLE_CUDA=ON]
cmake --build build/cuda-sm120 --parallel
unset CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/cuda-12.9.1/targets/x86_64-linux/lib
ctest --test-dir build/cuda-sm120
```

Result: **93/93 passed** (RTX 5090, sm_120, CUDA 12.9.1 V12.9.86, Release).
CPU: **33/33 passed** (build/cpu).

Note: `CUDA_VISIBLE_DEVICES=` set causes 41 CUDA tests to fail with
"no CUDA-capable device"; this is an environment artifact, not a regression.

## Step 1: inventory of duplicated common fields

`Gfn2SccSetupInputs::bind_device_arena_and_upload_async`
(src/backends/cuda/gfn2_scc_setup_inputs.cu) reconstructs `Gfn2SccIterationDevicePlan`
leaves at setup from `device_topology` + its own `impl_->layout` segments. The
following authoritative common offsets/maps are re-subscribed verbatim from
`device_topology` into every leaf batch (the checkpoint's "duplicate authority"):

- `atom_offsets`: geometry_batch, scc_batch, spin_batch, potential_batch, es2_batch,
  aes2_batch, d4_batch, point_charge_batch, periodic_batch, scalar_bridge_batch,
  hamiltonian_batch, mulliken_batch, publication_plan
- `batch_shell_offsets`: scc_batch, spin_batch, es2_batch, es3_batch, point_charge_batch,
  hamiltonian_batch, mulliken_batch, publication_plan
- `batch_orbital_offsets`: hamiltonian_batch, eigensolver_batch, occupations_batch,
  density_batch, mulliken_batch, publication_plan
- `matrix_offsets`: hamiltonian_batch, eigensolver_batch, density_batch, mulliken_batch,
  electronic_energy_batch, publication_plan
- `atom_shell_offsets`: spin_batch, es2_batch, hamiltonian_batch, mulliken_batch
- `shell_orbital_offsets`: hamiltonian_batch, mulliken_batch
- `shell_to_atom`: spin_batch?, potential_batch, es2_batch, point_charge_batch,
  hamiltonian_batch, mulliken_batch, publication_plan
- `orbital_to_shell` / `orbital_to_atom`: hamiltonian_batch, mulliken_batch

Setup also re-proves equality between `basis`, `integrals`, `es2`, `aes2`,
`mulliken`, `wavefunction`, `point`, and `periodic` plan arrays and the master
topology (lines 450-620 of gfn2_scc_setup_inputs.cu), i.e. validators repeatedly
prove that equal-sized offset/map fields are the same plan.

Also retained on the plan: `candidate.topology` (full master) and
`candidate.wavefunction_layout` plus every repeated leaf above.

## Step 2: projection-only leaf (landed on this branch)

Added common-schema projection views + binders + identity tests. No numerical
or launch-ordering change: no consumer kernel or descriptor was migrated yet.

- `Gfn2AtomProjectionView` / shell-ownership / AO-matrix / packed-all-pair /
  AO-bucket / element-identity projections in
  `src/backends/common/gfn2_plan_schema.hpp` (device-neutral POD, borrowed
  pointers, fail-closed `project_*_host` builders).
- `gfn2_element_identity_fingerprint_host`: order-sensitive seal over the
  atomic-number ordering and plan token.
- Structural `validate_gfn2_*_projection_binding` (never dereferences; works in
  every address space) plus host builders with exact pointer identity.
- CUDA binders `bind_gfn2_*_projection_cuda` in `src/backends/cuda/gfn2_plan_schema.cu`
  (host-side, no kernel/transfer/sync; require the master to have passed its own
  CUDA binding; element identity proves CUDA accessibility of the uploaded
  atomic-number array).
- Host identity tests in `tests/plan_schema_test.cpp` (test_projections):
  exact pointer/count identity, cross-plan/memory-space rejection, fingerprint
  order-sensitivity and failures, empty element identity, hostile-master clearing.
- CUDA identity tests in `tests/cuda_plan_schema_test.cu` (test_device_projections).

Results: host plan-schema test and CUDA plan-schema test pass; full CPU 33/33,
full CUDA 93/93 unchanged.
