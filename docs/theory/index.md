# Theory guide

This guide explains the physical and numerical meaning of xTBloom's public
results. The GFN1-xTB and GFN2-xTB implementations follow their distinct
upstream conventions and pinned independent xTB/tblite evidence; xTBloom does
not define a new parameterization or a new tight-binding method.

- [GFN2-xTB, SCC, occupations, and forces](gfn2.md)
- [GFN1-xTB model and publication contract](gfn1.md)
- [Explicit point charges and periodic response](qmmm.md)

All native calculations use binary64 atomic units. At finite electronic
temperature the reported variational energy is the electronic Helmholtz free
energy, and forces are its negative coordinate derivative under the public
operator-holding convention.

Implementation details and backend invariants belong in the
[developer architecture guide](../developer-guide/architecture.md). Pinned
upstream revisions, redistributed parameter material, and license terms belong
in [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) and the data
manifests.
