#include "runtime/nvidia_host_api.h"

#include <cstdint>
#include <type_traits>

static_assert(std::is_same_v<cublasStatus_t, std::uint32_t>);
static_assert(std::is_same_v<cusolverStatus_t, std::uint32_t>);
static_assert(!std::is_enum_v<cublasStatus_t>);
static_assert(!std::is_enum_v<cusolverStatus_t>);
static_assert(sizeof(cublasStatus_t) == 4);
static_assert(sizeof(cusolverStatus_t) == 4);

static_assert(CUBLAS_STATUS_SUCCESS == 0);
static_assert(CUBLAS_STATUS_NOT_INITIALIZED == 1);
static_assert(CUBLAS_STATUS_ALLOC_FAILED == 3);
static_assert(CUBLAS_STATUS_INVALID_VALUE == 7);
static_assert(CUBLAS_STATUS_ARCH_MISMATCH == 8);
static_assert(CUBLAS_STATUS_MAPPING_ERROR == 11);
static_assert(CUBLAS_STATUS_EXECUTION_FAILED == 13);
static_assert(CUBLAS_STATUS_INTERNAL_ERROR == 14);
static_assert(CUBLAS_STATUS_NOT_SUPPORTED == 15);
static_assert(CUBLAS_STATUS_LICENSE_ERROR == 16);

static_assert(CUSOLVER_STATUS_SUCCESS == 0);
static_assert(CUSOLVER_STATUS_NOT_INITIALIZED == 1);
static_assert(CUSOLVER_STATUS_ALLOC_FAILED == 2);
static_assert(CUSOLVER_STATUS_INVALID_VALUE == 3);
static_assert(CUSOLVER_STATUS_ARCH_MISMATCH == 4);
static_assert(CUSOLVER_STATUS_MAPPING_ERROR == 5);
static_assert(CUSOLVER_STATUS_EXECUTION_FAILED == 6);
static_assert(CUSOLVER_STATUS_INTERNAL_ERROR == 7);
static_assert(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED == 8);
static_assert(CUSOLVER_STATUS_NOT_SUPPORTED == 9);
static_assert(CUSOLVER_STATUS_ZERO_PIVOT == 10);
static_assert(CUSOLVER_STATUS_INVALID_LICENSE == 11);
static_assert(CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED == 12);
static_assert(CUSOLVER_STATUS_IRS_PARAMS_INVALID == 13);
static_assert(CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC == 14);
static_assert(CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE == 15);
static_assert(CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER == 16);
static_assert(CUSOLVER_STATUS_IRS_INTERNAL_ERROR == 20);
static_assert(CUSOLVER_STATUS_IRS_NOT_SUPPORTED == 21);
static_assert(CUSOLVER_STATUS_IRS_OUT_OF_RANGE == 22);
static_assert(CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES == 23);
static_assert(CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED == 25);
static_assert(CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED == 26);
static_assert(CUSOLVER_STATUS_IRS_MATRIX_SINGULAR == 30);
static_assert(CUSOLVER_STATUS_INVALID_WORKSPACE == 31);

int main() {
  constexpr cublasStatus_t kFutureBlasStatus = 0x10203040u;
  constexpr cusolverStatus_t kFutureSolverStatus = 0x50607080u;
  const cublasStatus_t blas_status = kFutureBlasStatus;
  const cusolverStatus_t solver_status = kFutureSolverStatus;
  return blas_status == kFutureBlasStatus && solver_status == kFutureSolverStatus ? 0 : 1;
}
