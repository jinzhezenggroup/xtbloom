// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

// Private wheel-only OpenBLAS provider shim.
//
// This DSO intentionally exports no API. Its sole purpose is to retain a
// DT_NEEDED edge to the provenance-checked scipy-openblas32 provider so
// auditwheel vendors and collision-renames the complete OpenBLAS/GCC runtime
// dependency closure. libxtbloom itself remains free of a hard BLAS dependency
// and lazily dlmopens this sibling shim in a new glibc link-map namespace.

static int xtbloom_openblas_lp64_shim_unit_marker;
