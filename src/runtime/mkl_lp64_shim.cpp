// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

// Private host-isolated MKL provider shim.
//
// gpuxtb never dlopens libmkl_rt directly. libmkl_rt is the MKL interface-layer
// dispatcher: initializing it in LP64 mode mutates process-global MKL state that
// an embedding application may already own, and reading MKL_INTERFACE_LAYER can
// make gpuxtb depend on the host's MKL lifecycle. Instead, CMake links this
// translation unit into a private shared object with fixed DT_NEEDED dependencies
// on libmkl_intel_lp64, libmkl_sequential, and libmkl_core. Loading those three
// component libraries directly (never libmkl_rt) yields an LP64 + sequential
// provider whose symbols resolve inside the shim's own RTLD_LOCAL scope. The host
// namespace is not polluted and the host's interface/threading state is never
// changed, so LP64 gpuxtb calls remain correct even when the host uses ILP64.
//
// The shim intentionally exports nothing; the eigensolver factory only dlopens it
// and resolves the LAPACKE/CBLAS/thread-control symbols through its dependency
// scope.

namespace gpuxtb {
namespace detail {
namespace gfn2 {
namespace {

int gpuxtb_mkl_lp64_shim_unit_marker = 0;

}  // namespace
}  // namespace gfn2
}  // namespace detail
}  // namespace gpuxtb
