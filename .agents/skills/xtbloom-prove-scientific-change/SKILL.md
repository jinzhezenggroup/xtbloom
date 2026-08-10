---
name: xtbloom-prove-scientific-change
description: Design and execute independent scientific evidence for xTBloom GFN2 physics, SCC, occupations, energies, forces, charges, QM/MM, periodic response, parameters, CPU/CUDA parity, conformance goldens, and SCC traces. Use for any numerical behavior change or any edit under model physics, conformance, parameters, or oracle tooling.
---

# Prove a xTBloom Scientific Change

Use independent evidence to distinguish a correct implementation from two implementations that agree with the same bug. Read `AGENTS.md` and the subsystem documents before changing numerical code.

## Route to the Authoritative Contract

- Read `docs/developer-guide/architecture.md` for public energy, force, finite-temperature, failure, and WARM semantics.
- Read `docs/user-guide/qmmm.md` and `docs/theory/qmmm.md` for point charges and caller-owned periodic `b + A*q` operators.
- Read `tools/conformance/README.md` before touching the conformance corpus or public runners.
- Read `tools/parameters/README.md` before changing parameter generators or tables.
- Read `tools/oracle/tblite_scc_trace/README.md` before changing SCC observer, trace, or replay work.

Record the equation, units, sign convention, variational quantity, backend consumers, and oracle source affected by the change. Forces are negative derivatives of the reported Helmholtz free energy, not merely derivatives of a convenient internal term.

## Establish the Baseline

Run immutable-data checks before editing so pre-existing drift is visible:

```bash
python3 tools/parameters/generate_gfn2.py --check
python3 tools/conformance/xtbloom_conformance.py check
python3 -m unittest discover -s tests/parameters -p 'test_*.py' -v
python3 -m unittest discover -s tests/conformance -p 'test_*.py' -v
python3 -m unittest discover -s tests/oracle -p 'test_*.py' -v
```

Do not hand-edit generated parameters, manifests, conformance goldens, trace fixtures, or hashes to make the implementation pass.

## Build the Evidence Ladder

Require every applicable level:

1. **Term level:** compare the changed term with an independently derived value or fixture; cover signs, units, empty input, edge elements, charge/spin, and ragged indexing.
2. **Derivative level:** use central finite differences at multiple stable step sizes, analytic force comparison, translation/rotation covariance, net-force or torque checks where physically applicable, and QM plus point-charge force conservation.
3. **Composition level:** verify energy/component ordering, SCC state, requested properties, finite temperature, restricted/unrestricted behavior, and CPU/CUDA parity.
4. **Public level:** run the exported C ABI on sequential one-system calls and true homogeneous and heterogeneous ragged batches. Do not compare a ragged batch to another view of the same ragged result.
5. **Failure level:** cover non-finite inputs, difficult/nonconvergent SCC, eigensolver failure when injectable, successful peers beside failed systems, complete NaN publication, and unchanged outputs on call-level failure.

For CUDA-visible changes, cover host, device, and mixed descriptors plus the complete real-GPU runtime and sanitizer matrix. Self-consistency or CPU/CUDA agreement never replaces the independent golden.

## Preserve Oracle Independence

Regenerate a golden only when the active issue explicitly requires a reviewed oracle update. Generate into a separate build directory first, then compare:

```bash
python3 tools/conformance/xtbloom_conformance.py generate \
  --executable /path/to/pinned/tblite \
  --output-dir build/conformance/tblite
python3 tools/conformance/xtbloom_conformance.py compare \
  --actual-dir build/conformance/tblite
```

For xTB-primary cases, use the pinned xTB workflow from the README. Preserve explicit `--acc 0.0001`, single-thread execution, cleared `XTB*` variables, and executable, shared-library, parameter, input, and materialized-file hashes. Never copy current xTBloom output into a golden or widen a tolerance merely to admit it.

Distinguish primary-oracle tolerances from cross-engine diagnostics. xTBloom must satisfy primary tolerances; a looser cross-engine diagnostic is not an implementation gate.

## Keep SCC Trace Claims Precise

Treat these as separate deliverables:

- Schema/writer validation and canonical binary64 serialization.
- Comparator behavior and stable exit codes.
- Pinned corpus, manifest, revisions, and hashes.
- Injection of golden mixed `q/d/Q` into one production iteration.
- Independent replay of golden residual history through the production mixer.

Do not call comparator-only work a backend replay. Preserve callback lifecycle and failed-attempt semantics from the trace README.

## Report the Evidence

Return these gates to the caller's validation ledger. Report exact commands and counts, oracle versions and hashes, maximum errors with units, tolerances and their source, tested memory modes, and unavailable evidence. Leave the issue open when finite differences, real-GPU parity, independent oracle generation, or pinned replay remains outstanding.
