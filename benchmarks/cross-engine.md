# Cross-engine molecule-size scaling

`natoms_cross_engine.py` measures GFN2-xTB energy plus analytic-force latency
through the public interfaces of xTBloom CPU/CUDA, xTB, tblite, and dxtb. Each
multi-system batch contains distinct seeded conformers of one alkane
stoichiometry, so a row cannot benefit from duplicating one geometry.

This protocol produced the figure used in the repository README and
[performance summary](../docs/user-guide/performance.md).

## Start policies

- `cold`: every measured sample starts independently. xTBloom performs
  `FRESH` initialization inside the timed public call; xTB and tblite rebuild
  their calculator outside timing; dxtb times reset, calculation,
  synchronization, and host tensor publication.
- `auto-warm`: an untimed cold seed precedes strict `WARM` xTBloom calls and
  persistent xTB/tblite calls. dxtb has no equivalent continuation path in this
  adapter and remains reset/cold.

The published batch-1 and batch-512 panels use `cold`. Batch 128 uses
`auto-warm`. Panel titles and accompanying prose must state that distinction.

## Correctness qualification

Every interval ends with host-visible energy and forces. Each library retains
its native public convergence controls:

| Library | Controls in the published evidence |
| --- | --- |
| xTBloom | charge `1e-4`, energy `1e-6`, maximum 500 iterations |
| xTB and tblite | public accuracy factor `1.0`, maximum 500 iterations |
| dxtb | `x_atol=1e-4`, `x_atol_max=1e-5`, `f_atol=1e-4`, force convergence, maximum 500 iterations |

Every timed dependent sample in the published figure is checked against its
panel-matched clean tblite reference:

```text
max_s |Delta E_s| <= 2e-3 Eh
max_i |Delta F_i| <= 2e-3 Eh/bohr
```

This owner-authorized output-compatibility gate determines benchmark
eligibility. It is not a tblite convergence default and does not replace
xTBloom's primary scientific conformance thresholds.

## Timing interpretation

xTBloom submits a complete ragged batch through one public call. The xTB and
tblite adapters loop over their per-structure public APIs. The measured result
therefore reflects the complete engine-and-interface paths used by this
workload; it must not be reduced to a universal batching-only or
single-molecule claim.

xTBloom CUDA uses host descriptors staged by the public ABI, while dxtb CUDA
retains device tensors. Their curves show observed behavior, but the public
evidence makes no direct cross-library CUDA speedup claim.

## Artifacts and rendering

JSON retains every raw sample, final force vectors, convergence and correctness
state, source/build/runtime identities, and binary hashes. CSV is the compact
tabular view.

`plot_natoms_cross_engine.py` rejects dirty or protocol-incompatible inputs and
preserves failed or unavailable coordinates. It declares Matplotlib through
PEP 723 metadata and has an adjacent locked resolution:

```bash
uv run --script benchmarks/plot_natoms_cross_engine.py \
  --artifact /path/to/result-1.json \
  --artifact /path/to/result-2.json \
  --output /path/to/natoms_cross_engine.svg
```

The publication bundle, exact commands, raw numbers, hashes, unavailable rows,
and limitations are archived in
[issue #13 evidence](evidence/issue-13/2026-08-09-node3-pr231/README.md).

## Validation

```bash
python3 -m unittest -v benchmarks.test_natoms_cross_engine
```
