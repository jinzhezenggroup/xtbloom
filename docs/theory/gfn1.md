# GFN1-xTB implementation contract

GFN1-xTB is a distinct tight-binding model, not a second parameter table for
the GFN2 equations. xTBloom reserves its public model tag, but keeps execution
disabled until the complete CPU and CUDA implementations and their independent
evidence are available. This page records the contract those implementations
must satisfy.

## Pinned references

The initial model audit uses three clean owner-requested local checkouts:

- tblite `133f91efb94b47f05848e1f86832f40a1accc385` as the structured parameter
  and readable calculator contract;
- xTB `b31754bf3c7cccf8c242c469b03ae675e04bd608` as the canonical production
  behavior and analytic-gradient reference; and
- dxtb `b529b5ddb75c0554274955082a189f9f88437cb2` as an independently structured
  implementation cross-check.

Canonical redistributed GFN1 parameter material is pinned to tblite 0.7.0
commit `fa8a4416e8fe093d0075bc10ac875494c2a449a9`. It is an ancestor of the local
tblite checkout; the intervening GFN1/export-source changes are formatting and
workflow maintenance rather than parameter changes. Primary closed-shell
goldens use tblite 0.7.0 with explicit `--method gfn1 --acc 0.0001 --grad
--json`; pinned xTB 6.7.1 supplies unrestricted, point-charge, and
halogen-specific reference cases.

Redistributed parameter bytes are generated from the reviewed tblite source
and covered by its LGPL-3.0-or-later grant. xTB and dxtb are oracle and review
inputs unless a later provenance manifest explicitly identifies redistributed
material from them.

## Differences from GFN2

The GFN1 implementation requires its own scientific composition:

- the exponential coordination-number model rather than the GFN2
  double-exponential model;
- the GFN1 basis, including orthogonalization of repeated angular-momentum
  shells and a first-shell valence mask;
- GFN1 zeroth-order Hamiltonian shell scales and element-pair overrides;
- harmonic averaging in the isotropic second-order Coulomb kernel;
- atom-resolved third-order charge electrostatics rather than GFN2's
  shell-resolved third-order term;
- charge-independent D3(BJ) dispersion with the reviewed GFN1 parameters;
- the GFN1 halogen-bond correction; and
- no GFN2 anisotropic AES2/multipole SCC term and no self-consistent D4 term.

Shared numerical utilities such as generalized eigensolution, occupations,
density construction, mixing, and failure publication may be reused only when
their equations and state layouts are genuinely model-independent.

## Energy, SCC, and forces

At finite electronic temperature, the reported variational energy remains the
electronic Helmholtz free energy

```math
F = E_{\mathrm{GFN1}} - (k_{\mathrm B}T)S_{\mathrm{electronic}}.
```

Forces are the negative coordinate derivative of that converged free energy.
Term-level derivatives, total analytic forces, invariance, and conservation
must be checked independently of xTBloom before the public tag is enabled.
Restricted and unrestricted spin behavior must likewise be compared with the
pinned reference engines rather than assumed to match GFN2.

## Embedding is model-specific until proven

xTB contains distinct GFN1 point-charge potential and gradient paths. The
existing GFN2 screened point-charge implementation therefore cannot be reused
by naming alone. Explicit point charges and caller-supplied periodic `b + A*q`
response must either have GFN1-specific oracle and derivative evidence or be
rejected transactionally for GFN1 requests.

For a GFN1 shell hardness `g_i` and point-site hardness `g_p`, xTB's softened
interaction uses `x = 2 / (1/g_i + 1/g_p)` and
`J = (r^2 + x^-2)^(-1/2)`. This differs from the released GFN2 convention and
requires a separate implementation and finite-difference gate.

## Publication boundary

The stable C ABI already reserves `XTBLOOM_MODEL_GFN1_XTB`. While foundation
work is incomplete, it remains a known but unsupported model: validated GFN1
requests return `XTBLOOM_STATUS_NOT_SUPPORTED` before execution and leave all
caller outputs unchanged. CPU support may be advertised only after complete
energy, requested properties, analytic forces, ragged failure isolation, and
installed public-API evidence pass. CUDA support additionally requires real
GPU host/device/mixed parity and the repository sanitizer matrix.
