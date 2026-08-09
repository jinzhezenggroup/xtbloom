// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

// Private host-isolated MKL provider shim.
//
// xtbloom never dlopens libmkl_rt directly. libmkl_rt is the MKL interface-layer
// dispatcher: initializing it in LP64 mode mutates process-global MKL state that
// an embedding application may already own, and reading MKL_INTERFACE_LAYER can
// make xtbloom depend on the host's MKL lifecycle. Instead, CMake links this
// translation unit into a private shared object with fixed DT_NEEDED dependencies
// on libmkl_intel_lp64, libmkl_sequential, and libmkl_core. Loading those three
// component libraries directly (never libmkl_rt) yields an LP64 + sequential
// provider. The runtime loads this shim in a new glibc link-map namespace because
// RTLD_LOCAL alone would still allow pre-existing global host symbols to
// interpose. The host's interface/threading state is therefore unchanged, and
// LP64 xtbloom calls remain correct even when the host uses ILP64.
//
// The shim intentionally exports nothing; the eigensolver factory only dlopens it
// and resolves the LAPACKE/CBLAS/thread-control symbols through its dependency
// scope.

namespace xtbloom {
namespace detail {
namespace gfn2 {
namespace {

int xtbloom_mkl_lp64_shim_unit_marker = 0;

}  // namespace
}  // namespace gfn2
}  // namespace detail
}  // namespace xtbloom
