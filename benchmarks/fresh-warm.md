# CPU FRESH/WARM molecule-size scaling

`natoms_scaling.py` measures strict xTBloom SCC start policies and a persistent
public-C-API tblite reference across a deterministic alkane size sweep. This is
a separate protocol from the public cross-engine figure.

## Eligibility

A latency row is eligible only when its within-engine correctness and explicit
FRESH reference comparison both pass. Each cell retains one context, ragged
descriptor, options image, and caller-owned result buffers.

- `fresh` performs independent SCC initialization in every timed call.
- `warm` performs one untimed compatible `FRESH` seed, then uses strict
  `WARM` for every warmup and measured call.

JSON is authoritative and retains every latency, energy, requested force
vector, SCC iteration count, convergence flag, and per-system status. CSV is a
compact summary and intentionally omits repeated identity and full observable
payloads.

## Numerical contract

The archived issue #168 evidence uses the conformance SCC policy: 500 maximum
iterations, charge tolerance `1e-10`, energy tolerance `1e-12`, and the public
300 K electronic-temperature default. FRESH repeatability and WARM-versus-FRESH
force drift use the manifest's primary `5e-7` Hartree/bohr gate. Comparisons
with another engine use `5e-7` Hartree for energy and the separate live
cross-engine `5e-6` Hartree/bohr force gate.

The runner rejects a FRESH artifact with missing raw samples, wider gates,
different workload/options identity, or—for xTBloom WARM—a different library
hash or clean source revision.

These settings differ from the public cross-engine figure. Do not reuse one
protocol's thresholds or timings in the other.

## Example sequence

Run each engine and start mode as a separate clean process:

```bash
python3 benchmarks/natoms_scaling.py \
  --engine xtbloom \
  --library /absolute/path/to/libxtbloom.so \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode fresh \
  --output-json build/benchmarks/fresh.json \
  --output-csv build/benchmarks/fresh.csv

python3 benchmarks/natoms_scaling.py \
  --engine xtbloom \
  --library /absolute/path/to/libxtbloom.so \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode warm \
  --energy-reference-json build/benchmarks/fresh.json \
  --cross-engine-energy-atol 5e-7 \
  --cross-engine-force-atol 5e-6 \
  --output-json build/benchmarks/warm.json \
  --output-csv build/benchmarks/warm.csv
```

Pin process affinity and keep BLAS one-threaded in final CPU evidence. The JSON
records the exact argv, environment, build/runtime identity, and hashes.

The xTB 6.7.1 analytic-force result for the issue #168 long-chain geometry
fails the unchanged force gate and is retained only as a negative diagnostic;
no speed claim is made from it.

Exact machine paths, commands, results, and the diagnostic are preserved in
[issue #168 evidence](evidence/issue-168/2026-08-06-epyc7k62/README.md).

## Validation

```bash
python3 -m unittest -v benchmarks.test_natoms_scaling
```
