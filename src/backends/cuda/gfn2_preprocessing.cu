#include <algorithm>
#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "backends/cuda/gfn2_preprocessing.cuh"
#include "backends/cuda/periodic_wrap.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
/* Primitive candidates are unpublished and need only a valid nonzero scalar;
 * the final composer publishes the authoritative device epoch per peer. */
constexpr std::uint64_t kUnpublishedPrimitiveGeneration = 1u;

/* The periodic image evaluator keeps one complete shell pair in a 64-thread
 * CTA, matching the molecular integral primitive.  Its launch grid is a
 * bounded work queue: a CTA may consume several image/shell tasks, which
 * keeps the 1-D grid valid even for a large translation superset. */
constexpr int kNativePeriodicIntegralThreadsPerBlock = 64;
constexpr int kNativePeriodicMaximumCartesianBlock = 36;
constexpr int kNativePeriodicMultipoleComponents = 9;
constexpr double kNativePeriodicSqrtThree = 1.732050807568877293527446341505872367;
constexpr double kNativePeriodicSqrtPiCubed = 5.5683279968317061;
constexpr double kNativePeriodicMinimumImageDistanceSquared =
    2.220446049250313080847263336181640625e-16;
constexpr unsigned int kNativePeriodicMaximumGridBlocks = 65535u;
constexpr std::int64_t kNativePeriodicInt64Maximum = 9223372036854775807LL;

struct GeometryGenerationSource {
  std::uint64_t scalar = 0u;
  const std::uint64_t* device = nullptr;
};

using BindingDiagnostic = Gfn2PreprocessingBindingDiagnostic;
using BindingError = Gfn2PreprocessingBindingError;
using BindingField = Gfn2PreprocessingBindingField;

BindingDiagnostic binding_failure(BindingError error, BindingField field,
                                  std::int64_t index = -1) noexcept {
  BindingDiagnostic result{};
  result.error = error;
  result.field = field;
  result.index = index;
  return result;
}

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t& product) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  product = first * second;
  return true;
}

template <typename T>
bool canonical_pointer(const T* pointer, std::int64_t elements) noexcept {
  if (elements < 0) {
    return false;
  }
  if (elements == 0) {
    return pointer == nullptr;
  }
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) == 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

template <std::size_t Capacity>
class RangeList {
 public:
  template <typename T>
  bool add(const T* pointer, std::int64_t elements) noexcept {
    if (elements < 0 || size_ == Capacity ||
        static_cast<std::uint64_t>(elements) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
      return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
    if (bytes == 0u) {
      ranges_[size_++] = {};
      return pointer == nullptr;
    }
    if (pointer == nullptr) {
      return false;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
      return false;
    }
    ranges_[size_++] = {begin, begin + bytes};
    return true;
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] const AddressRange& operator[](std::size_t index) const noexcept {
    return ranges_[index];
  }

 private:
  std::array<AddressRange, Capacity> ranges_{};
  std::size_t size_ = 0u;
};

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t ReadCapacity, std::size_t WriteCapacity>
bool writes_are_disjoint(const RangeList<ReadCapacity>& reads,
                         const RangeList<WriteCapacity>& writes) noexcept {
  for (std::size_t write = 0u; write < writes.size(); ++write) {
    for (std::size_t read = 0u; read < reads.size(); ++read) {
      if (overlaps(writes[write], reads[read])) {
        return false;
      }
    }
    for (std::size_t peer = write + 1u; peer < writes.size(); ++peer) {
      if (overlaps(writes[write], writes[peer])) {
        return false;
      }
    }
  }
  return true;
}

__device__ bool native_periodic_sequence_is_active(const Gfn2IntegralDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool native_periodic_system_is_valid(const std::uint32_t* system_errors,
                                                std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors + system), 0u) ==
         static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess);
}

/* Keep periodic failures in the same primitive domain consumed by the
 * composer.  A first peer-local error is also retained in the sticky scalar,
 * while the publication gate still uses the per-system slot for isolation. */
__device__ void native_periodic_record_error(std::uint32_t* system_errors, std::int64_t system,
                                             std::uint32_t* device_error,
                                             Gfn2IntegralDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system,
                static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess)) {
    atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess), code);
  }
}

__device__ int native_periodic_cartesian_count(std::uint8_t angular_momentum) {
  const int l = static_cast<int>(angular_momentum);
  return (l + 1) * (l + 2) / 2;
}

__device__ int native_periodic_spherical_count(std::uint8_t angular_momentum) {
  return 2 * static_cast<int>(angular_momentum) + 1;
}

__device__ void native_periodic_cartesian_exponent(std::uint8_t angular_momentum, int function,
                                                   int* x, int* y, int* z) {
  if (angular_momentum == 0u) {
    *x = 0;
    *y = 0;
    *z = 0;
  } else if (angular_momentum == 1u) {
    *x = function == 0 ? 1 : 0;
    *y = function == 1 ? 1 : 0;
    *z = function == 2 ? 1 : 0;
  } else {
    constexpr int exponents[6][3] = {{2, 0, 0}, {1, 1, 0}, {1, 0, 1},
                                     {0, 2, 0}, {0, 1, 1}, {0, 0, 2}};
    *x = exponents[function][0];
    *y = exponents[function][1];
    *z = exponents[function][2];
  }
}

__device__ void native_periodic_multipole_power(int component, int* x, int* y, int* z) {
  constexpr int powers[kNativePeriodicMultipoleComponents][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                                                                 {2, 0, 0}, {1, 1, 0}, {0, 2, 0},
                                                                 {1, 0, 1}, {0, 1, 1}, {0, 0, 2}};
  *x = powers[component][0];
  *y = powers[component][1];
  *z = powers[component][2];
}

/* Real spherical rows and CCA Cartesian columns used by the molecular CUDA
 * evaluator.  Keeping the transform local makes the periodic path share the
 * exact basis convention without exposing private molecular helpers. */
__device__ double native_periodic_spherical_coefficient(std::uint8_t angular_momentum,
                                                        int spherical, int cartesian) {
  if (angular_momentum == 0u) {
    return spherical == 0 && cartesian == 0 ? 1.0 : 0.0;
  }
  if (angular_momentum == 1u) {
    const int selected = spherical == 0 ? 1 : (spherical == 1 ? 2 : 0);
    return cartesian == selected ? 1.0 : 0.0;
  }
  if (spherical == 0) return cartesian == 1 ? kNativePeriodicSqrtThree : 0.0;
  if (spherical == 1) return cartesian == 4 ? kNativePeriodicSqrtThree : 0.0;
  if (spherical == 2) {
    return cartesian == 0 || cartesian == 3 ? -0.5 : (cartesian == 5 ? 1.0 : 0.0);
  }
  if (spherical == 3) return cartesian == 2 ? kNativePeriodicSqrtThree : 0.0;
  return cartesian == 0 ? 0.5 * kNativePeriodicSqrtThree
                        : (cartesian == 3 ? -0.5 * kNativePeriodicSqrtThree : 0.0);
}

__device__ void native_periodic_make_axis_overlap(double product_minus_i, double product_minus_j,
                                                  double inverse_twice_sum, int maximum_a,
                                                  int maximum_b, double overlap[6][3]) {
#pragma unroll
  for (int a = 0; a < 6; ++a) {
#pragma unroll
    for (int b = 0; b < 3; ++b) overlap[a][b] = 0.0;
  }
  overlap[0][0] = 1.0;
  for (int a = 1; a <= maximum_a; ++a) {
    overlap[a][0] = product_minus_i * overlap[a - 1][0];
    if (a > 1) {
      overlap[a][0] += static_cast<double>(a - 1) * inverse_twice_sum * overlap[a - 2][0];
    }
  }
  for (int b = 1; b <= maximum_b; ++b) {
    overlap[0][b] = product_minus_j * overlap[0][b - 1];
    if (b > 1) {
      overlap[0][b] += static_cast<double>(b - 1) * inverse_twice_sum * overlap[0][b - 2];
    }
    for (int a = 1; a <= maximum_a; ++a) {
      overlap[a][b] = product_minus_i * overlap[a - 1][b] +
                      static_cast<double>(b) * inverse_twice_sum * overlap[a - 1][b - 1];
      if (a > 1) {
        overlap[a][b] += static_cast<double>(a - 1) * inverse_twice_sum * overlap[a - 2][b];
      }
    }
  }
}

__device__ bool native_periodic_image_vector(const double* wrapped_positions, std::int64_t bra_atom,
                                             std::int64_t ket_atom,
                                             const Gfn2CudaPeriodicTranslation& translation,
                                             double cutoff, double cutoff_squared, double vector[3],
                                             double& distance_squared) {
  const double* const bra = wrapped_positions + bra_atom * 3;
  const double* const ket = wrapped_positions + ket_atom * 3;
  for (int axis = 0; axis < 3; ++axis) {
    const double difference = periodic_wrap_detail::rounded_subtract(ket[axis], bra[axis]);
    vector[axis] = periodic_wrap_detail::rounded_subtract(difference, translation.cartesian[axis]);
    if (!isfinite(vector[axis]) || fabs(vector[axis]) > cutoff) return false;
  }
  const double x_squared = periodic_wrap_detail::rounded_multiply(vector[0], vector[0]);
  const double y_squared = periodic_wrap_detail::rounded_multiply(vector[1], vector[1]);
  const double z_squared = periodic_wrap_detail::rounded_multiply(vector[2], vector[2]);
  distance_squared = periodic_wrap_detail::rounded_add(
      periodic_wrap_detail::rounded_add(x_squared, y_squared), z_squared);
  return isfinite(distance_squared) && distance_squared <= cutoff_squared &&
         distance_squared >= kNativePeriodicMinimumImageDistanceSquared;
}

__device__ bool native_periodic_h0_factor(const Gfn2IntegralDeviceBatch& batch,
                                          const Gfn2H0DevicePlan& h0,
                                          const double* coordination_numbers, std::int64_t system,
                                          std::int64_t bra_atom, std::int64_t ket_atom,
                                          std::int64_t bra_shell, std::int64_t ket_shell,
                                          bool onsite, double distance_squared, double& factor) {
  const double bra_cn = coordination_numbers[bra_atom];
  const double ket_cn = coordination_numbers[ket_atom];
  const double bra_level =
      h0.shell_levels[bra_shell] - h0.shell_coordination_scale[bra_shell] * bra_cn;
  const double ket_level =
      h0.shell_levels[ket_shell] - h0.shell_coordination_scale[ket_shell] * ket_cn;
  if (!isfinite(bra_cn) || !isfinite(ket_cn) || !isfinite(bra_level) || !isfinite(ket_level)) {
    return false;
  }
  const double average = 0.5 * (bra_level + ket_level);
  if (!isfinite(average)) return false;
  if (onsite) {
    factor = average;
    return true;
  }
  const double distance = sqrt(distance_squared);
  const double radius_sum = h0.atomic_radii[bra_atom] + h0.atomic_radii[ket_atom];
  if (!(distance > 0.0) || !isfinite(distance) || !(radius_sum > 0.0) || !isfinite(radius_sum)) {
    return false;
  }
  const double reduced = sqrt(distance / radius_sum);
  const double bra_polynomial = 1.0 + h0.shell_polynomial[bra_shell] * reduced;
  const double ket_polynomial = 1.0 + h0.shell_polynomial[ket_shell] * reduced;
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t shell_count = shell_end - shell_begin;
  const std::int64_t local_bra = bra_shell - shell_begin;
  const std::int64_t local_ket = ket_shell - shell_begin;
  const std::int64_t pair = batch.shell_pair_offsets[system] + local_bra * shell_count + local_ket;
  const double pair_scale = h0.shell_pair_scale[pair];
  if (!isfinite(reduced) || !isfinite(bra_polynomial) || !isfinite(ket_polynomial) ||
      !(pair_scale > 0.0) || !isfinite(pair_scale)) {
    return false;
  }
  factor = average * pair_scale * bra_polynomial * ket_polynomial;
  return isfinite(factor);
}

__device__ void native_periodic_shifted_multipoles(const double vector[3], double overlap,
                                                   const double dipole[3],
                                                   const double quadrupole[6],
                                                   double shifted_dipole[3],
                                                   double shifted_quadrupole[6]) {
  for (int component = 0; component < 3; ++component) {
    shifted_dipole[component] = dipole[component] + vector[component] * overlap;
  }
  const double shift[6] = {
      2.0 * vector[0] * dipole[0] + vector[0] * vector[0] * overlap,
      vector[0] * dipole[1] + vector[1] * dipole[0] + vector[0] * vector[1] * overlap,
      2.0 * vector[1] * dipole[1] + vector[1] * vector[1] * overlap,
      vector[0] * dipole[2] + vector[2] * dipole[0] + vector[0] * vector[2] * overlap,
      vector[1] * dipole[2] + vector[2] * dipole[1] + vector[1] * vector[2] * overlap,
      2.0 * vector[2] * dipole[2] + vector[2] * vector[2] * overlap};
  const double trace = 0.5 * (shift[0] + shift[2] + shift[5]);
  for (int component = 0; component < 6; ++component) {
    const bool diagonal = component == 0 || component == 2 || component == 5;
    shifted_quadrupole[component] =
        quadrupole[component] + 1.5 * shift[component] - (diagonal ? trace : 0.0);
  }
}

/* One CTA owns one system at a time and evaluates its complete image/shell
 * queue in canonical order.  The shell-pair AO work remains parallel inside
 * the CTA, but the image reduction itself is deliberately serialized per
 * system: every output matrix element then observes the same image order as
 * the CPU evaluator instead of an unspecified cross-CTA atomicAdd order.  A
 * bounded grid lets a block process more than 65,535 systems by taking a
 * system-stride through the batch; systems still never share output slices. */
__global__ void native_periodic_integral_kernel(
    Gfn2IntegralDeviceBatch batch, Gfn2H0DevicePlan h0,
    Gfn2NativePeriodicIntegralDeviceBatch periodic, const double* wrapped_positions,
    const double* coordination_numbers, Gfn2IntegralDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, std::int64_t total_tasks,
    std::int64_t image_task_count, std::int64_t image_system_stride,
    std::int64_t maximum_shell_pair_count) {
  __shared__ int task_active;
  __shared__ int task_is_image;
  __shared__ std::int64_t task_system;
  __shared__ std::int64_t task_bra_atom;
  __shared__ std::int64_t task_ket_atom;
  __shared__ std::int64_t task_bra_shell;
  __shared__ std::int64_t task_ket_shell;
  __shared__ int bra_cartesian_count;
  __shared__ int ket_cartesian_count;
  __shared__ int bra_spherical_count;
  __shared__ int ket_spherical_count;
  __shared__ int cartesian_block_size;
  __shared__ int spherical_block_size;
  __shared__ double task_vector[3];
  __shared__ double task_distance_squared;
  __shared__ double task_h0_factor;
  __shared__ double cartesian_overlap[kNativePeriodicMaximumCartesianBlock];
  __shared__ double cartesian_multipole[kNativePeriodicMultipoleComponents *
                                        kNativePeriodicMaximumCartesianBlock];

  const std::int64_t task_count_per_system = image_system_stride + maximum_shell_pair_count;
  for (std::int64_t owned_system = static_cast<std::int64_t>(blockIdx.x);
       owned_system < batch.batch_size; owned_system += static_cast<std::int64_t>(gridDim.x)) {
    for (std::int64_t local_task = 0; local_task < task_count_per_system; ++local_task) {
      const std::int64_t task = local_task < image_system_stride
                                    ? owned_system * image_system_stride + local_task
                                    : image_task_count + owned_system * maximum_shell_pair_count +
                                          (local_task - image_system_stride);
      if (threadIdx.x == 0) {
        task_active = 0;
        task_is_image = 0;
        task_system = -1;
        task_bra_atom = -1;
        task_ket_atom = -1;
        task_bra_shell = -1;
        task_ket_shell = -1;
        bra_cartesian_count = 0;
        ket_cartesian_count = 0;
        bra_spherical_count = 0;
        ket_spherical_count = 0;
        cartesian_block_size = 0;
        spherical_block_size = 0;
        task_vector[0] = 0.0;
        task_vector[1] = 0.0;
        task_vector[2] = 0.0;
        task_distance_squared = 0.0;
        task_h0_factor = 0.0;

        if (native_periodic_sequence_is_active(workspace)) {
          bool metadata_valid = true;
          bool image_inside = true;
          std::int64_t local_pair = 0;
          std::int64_t image_slot = -1;
          if (task < image_task_count) {
            task_is_image = 1;
            if (image_system_stride <= 0 || maximum_shell_pair_count <= 0) {
              metadata_valid = false;
            } else {
              task_system = task / image_system_stride;
              const std::int64_t remainder = task - task_system * image_system_stride;
              image_slot = remainder / maximum_shell_pair_count;
              local_pair = remainder - image_slot * maximum_shell_pair_count;
            }
          } else {
            const std::int64_t onsite_task = task - image_task_count;
            if (maximum_shell_pair_count <= 0) {
              metadata_valid = false;
            } else {
              task_system = onsite_task / maximum_shell_pair_count;
              local_pair = onsite_task - task_system * maximum_shell_pair_count;
            }
          }

          if (metadata_valid && (task_system < 0 || task_system >= batch.batch_size ||
                                 task_system >= periodic.translation_offset_elements - 1)) {
            metadata_valid = false;
          }
          if (metadata_valid) {
            const std::int64_t atom_begin = batch.atom_offsets[task_system];
            const std::int64_t atom_end = batch.atom_offsets[task_system + 1];
            const std::int64_t shell_begin = batch.batch_shell_offsets[task_system];
            const std::int64_t shell_end = batch.batch_shell_offsets[task_system + 1];
            const std::int64_t orbital_begin = batch.batch_orbital_offsets[task_system];
            const std::int64_t orbital_end = batch.batch_orbital_offsets[task_system + 1];
            const std::int64_t matrix_begin = batch.matrix_offsets[task_system];
            const std::int64_t matrix_end = batch.matrix_offsets[task_system + 1];
            const std::int64_t shell_pair_begin = batch.shell_pair_offsets[task_system];
            const std::int64_t shell_pair_end = batch.shell_pair_offsets[task_system + 1];
            const std::int64_t shell_count = shell_end - shell_begin;
            const std::int64_t orbitals = orbital_end - orbital_begin;
            std::int64_t expected_shell_pairs = 0;
            std::int64_t expected_matrix = 0;
            const bool ranges_valid =
                atom_begin >= 0 && atom_begin <= atom_end && atom_end <= batch.total_atoms &&
                shell_begin >= 0 && shell_begin <= shell_end && shell_end <= batch.total_shells &&
                orbital_begin >= 0 && orbital_begin <= orbital_end &&
                orbital_end <= batch.total_orbitals && matrix_begin >= 0 &&
                matrix_begin <= matrix_end && matrix_end <= batch.total_matrix_elements &&
                shell_pair_begin >= 0 && shell_pair_begin <= shell_pair_end &&
                shell_pair_end <= batch.total_shell_pair_elements && shell_count > 0 &&
                shell_count <= batch.maximum_system_shells && orbitals > 0 &&
                shell_count <= kNativePeriodicInt64Maximum / shell_count &&
                orbitals <= kNativePeriodicInt64Maximum / orbitals;
            if (!ranges_valid) {
              metadata_valid = false;
            } else {
              expected_shell_pairs = shell_count * shell_count;
              expected_matrix = orbitals * orbitals;
              metadata_valid = shell_pair_end - shell_pair_begin == expected_shell_pairs &&
                               matrix_end - matrix_begin == expected_matrix;
            }

            /* The queue is rectangular in the system dimension: systems with
             * fewer shells than the plan maximum contribute padding slots.
             * Padding is an ordinary no-op, not damaged immutable metadata;
             * only malformed ranges above are peer-local errors. */
            if (metadata_valid && local_pair >= expected_shell_pairs) {
              image_inside = false;
            }

            if (metadata_valid && task_is_image != 0) {
              const std::int64_t translation_begin = periodic.translation_offsets[task_system];
              const std::int64_t translation_end = periodic.translation_offsets[task_system + 1];
              const std::int64_t translation_count = translation_end - translation_begin;
              if (translation_begin < 0 || translation_begin > translation_end ||
                  translation_end > periodic.translation_elements || translation_count <= 0 ||
                  translation_count > periodic.max_translations_per_system ||
                  (task_system == 0 && translation_begin != 0) ||
                  (task_system + 1 == batch.batch_size &&
                   translation_end != periodic.translation_elements)) {
                metadata_valid = false;
              } else if (image_slot < 0 || image_slot >= translation_count) {
                image_inside = false;
              } else {
                const auto& translation = periodic.translations[translation_begin + image_slot];
                if (!isfinite(translation.cartesian[0]) || !isfinite(translation.cartesian[1]) ||
                    !isfinite(translation.cartesian[2])) {
                  metadata_valid = false;
                } else {
                  task_distance_squared = 0.0;
                  const std::int64_t shell_pair = local_pair;
                  task_bra_shell = shell_begin + shell_pair / shell_count;
                  task_ket_shell = shell_begin + shell_pair % shell_count;
                }
              }
            }

            if (metadata_valid && image_inside &&
                native_periodic_system_is_valid(system_errors, task_system)) {
              const std::int64_t shell_pair = local_pair;
              if (shell_pair < 0 || shell_pair >= expected_shell_pairs) {
                metadata_valid = false;
              } else {
                task_bra_shell = shell_begin + shell_pair / shell_count;
                task_ket_shell = shell_begin + shell_pair % shell_count;
                task_bra_atom = batch.shell_to_atom[task_bra_shell];
                task_ket_atom = batch.shell_to_atom[task_ket_shell];
                if (task_bra_atom < atom_begin || task_bra_atom >= atom_end ||
                    task_ket_atom < atom_begin || task_ket_atom >= atom_end ||
                    (task_is_image == 0 && task_bra_atom != task_ket_atom) ||
                    (task_is_image != 0 && task_bra_atom > task_ket_atom)) {
                  /* Shells belonging to the opposite atom ordering are not
                   * errors: the canonical atom traversal handles them in the
                   * complementary shell block. */
                  image_inside = false;
                }
              }
            }

            if (metadata_valid && image_inside &&
                native_periodic_system_is_valid(system_errors, task_system)) {
              const std::uint8_t bra_l = batch.angular_momenta[task_bra_shell];
              const std::uint8_t ket_l = batch.angular_momenta[task_ket_shell];
              bra_cartesian_count = native_periodic_cartesian_count(bra_l);
              ket_cartesian_count = native_periodic_cartesian_count(ket_l);
              bra_spherical_count = native_periodic_spherical_count(bra_l);
              ket_spherical_count = native_periodic_spherical_count(ket_l);
              cartesian_block_size = bra_cartesian_count * ket_cartesian_count;
              spherical_block_size = bra_spherical_count * ket_spherical_count;
              const std::int64_t bra_orbital_begin = batch.shell_orbital_offsets[task_bra_shell];
              const std::int64_t bra_orbital_end = batch.shell_orbital_offsets[task_bra_shell + 1];
              const std::int64_t ket_orbital_begin = batch.shell_orbital_offsets[task_ket_shell];
              const std::int64_t ket_orbital_end = batch.shell_orbital_offsets[task_ket_shell + 1];
              const std::int64_t bra_primitive_begin =
                  batch.shell_primitive_offsets[task_bra_shell];
              const std::int64_t bra_primitive_end =
                  batch.shell_primitive_offsets[task_bra_shell + 1];
              const std::int64_t ket_primitive_begin =
                  batch.shell_primitive_offsets[task_ket_shell];
              const std::int64_t ket_primitive_end =
                  batch.shell_primitive_offsets[task_ket_shell + 1];
              metadata_valid =
                  bra_l <= 2u && ket_l <= 2u && cartesian_block_size > 0 &&
                  cartesian_block_size <= kNativePeriodicMaximumCartesianBlock &&
                  spherical_block_size > 0 && spherical_block_size <= 25 &&
                  bra_orbital_begin >= orbital_begin && bra_orbital_end <= orbital_end &&
                  ket_orbital_begin >= orbital_begin && ket_orbital_end <= orbital_end &&
                  bra_orbital_end - bra_orbital_begin == bra_spherical_count &&
                  ket_orbital_end - ket_orbital_begin == ket_spherical_count &&
                  bra_primitive_begin >= 0 && bra_primitive_begin < bra_primitive_end &&
                  bra_primitive_end <= batch.total_primitives && ket_primitive_begin >= 0 &&
                  ket_primitive_begin < ket_primitive_end &&
                  ket_primitive_end <= batch.total_primitives;
              if (metadata_valid) {
                for (std::int64_t primitive = bra_primitive_begin; primitive < bra_primitive_end;
                     ++primitive) {
                  if (!(batch.primitive_exponents[primitive] > 0.0) ||
                      !isfinite(batch.primitive_exponents[primitive]) ||
                      !isfinite(batch.primitive_coefficients[primitive])) {
                    metadata_valid = false;
                    break;
                  }
                }
              }
              if (metadata_valid) {
                for (std::int64_t primitive = ket_primitive_begin; primitive < ket_primitive_end;
                     ++primitive) {
                  if (!(batch.primitive_exponents[primitive] > 0.0) ||
                      !isfinite(batch.primitive_exponents[primitive]) ||
                      !isfinite(batch.primitive_coefficients[primitive])) {
                    metadata_valid = false;
                    break;
                  }
                }
              }
              if (metadata_valid) {
                const double* const bra_position = wrapped_positions + task_bra_atom * 3;
                const double* const ket_position = wrapped_positions + task_ket_atom * 3;
                if (!isfinite(bra_position[0]) || !isfinite(bra_position[1]) ||
                    !isfinite(bra_position[2]) || !isfinite(ket_position[0]) ||
                    !isfinite(ket_position[1]) || !isfinite(ket_position[2])) {
                  native_periodic_record_error(system_errors, task_system, device_error,
                                               Gfn2IntegralDeviceError::kNonfinitePosition);
                  metadata_valid = false;
                } else if (task_is_image != 0) {
                  const auto& translation =
                      periodic.translations[periodic.translation_offsets[task_system] + image_slot];
                  if (!native_periodic_image_vector(
                          wrapped_positions, task_bra_atom, task_ket_atom, translation,
                          periodic.realspace_cutoff,
                          periodic.realspace_cutoff * periodic.realspace_cutoff, task_vector,
                          task_distance_squared)) {
                    image_inside = false;
                  }
                }
              }
              if (metadata_valid && image_inside) {
                if (task_is_image == 0) {
                  task_vector[0] = 0.0;
                  task_vector[1] = 0.0;
                  task_vector[2] = 0.0;
                  task_distance_squared = 0.0;
                }
                if (!native_periodic_h0_factor(batch, h0, coordination_numbers, task_system,
                                               task_bra_atom, task_ket_atom, task_bra_shell,
                                               task_ket_shell, task_is_image == 0,
                                               task_distance_squared, task_h0_factor)) {
                  native_periodic_record_error(system_errors, task_system, device_error,
                                               Gfn2IntegralDeviceError::kInvalidH0Parameter);
                  metadata_valid = false;
                }
              }
              if (metadata_valid && image_inside) task_active = 1;
            }
          }
          if (!metadata_valid && task_system >= 0 && task_system < batch.batch_size) {
            native_periodic_record_error(system_errors, task_system, device_error,
                                         Gfn2IntegralDeviceError::kInvalidOffsets);
          }
        }
      }
      __syncthreads();

      if (task_active != 0) {
        if (static_cast<int>(threadIdx.x) < cartesian_block_size) {
          const int cartesian_index = static_cast<int>(threadIdx.x);
          const int bra_cartesian = cartesian_index / ket_cartesian_count;
          const int ket_cartesian = cartesian_index % ket_cartesian_count;
          int bra_power[3];
          int ket_power[3];
          native_periodic_cartesian_exponent(batch.angular_momenta[task_bra_shell], bra_cartesian,
                                             &bra_power[0], &bra_power[1], &bra_power[2]);
          native_periodic_cartesian_exponent(batch.angular_momenta[task_ket_shell], ket_cartesian,
                                             &ket_power[0], &ket_power[1], &ket_power[2]);
          double overlap_value = 0.0;
          double multipoles[kNativePeriodicMultipoleComponents] = {};
          const std::int64_t bra_primitive_begin = batch.shell_primitive_offsets[task_bra_shell];
          const std::int64_t bra_primitive_end = batch.shell_primitive_offsets[task_bra_shell + 1];
          const std::int64_t ket_primitive_begin = batch.shell_primitive_offsets[task_ket_shell];
          const std::int64_t ket_primitive_end = batch.shell_primitive_offsets[task_ket_shell + 1];
          for (std::int64_t ket_primitive = ket_primitive_begin; ket_primitive < ket_primitive_end;
               ++ket_primitive) {
            const double ket_alpha = batch.primitive_exponents[ket_primitive];
            for (std::int64_t bra_primitive = bra_primitive_begin;
                 bra_primitive < bra_primitive_end; ++bra_primitive) {
              const double bra_alpha = batch.primitive_exponents[bra_primitive];
              const double alpha_sum = ket_alpha + bra_alpha;
              const double inverse_sum = 1.0 / alpha_sum;
              const double product_exponent =
                  ket_alpha * bra_alpha * task_distance_squared * inverse_sum;
              if (!(alpha_sum > 0.0) || !isfinite(alpha_sum) || !isfinite(inverse_sum) ||
                  !isfinite(product_exponent)) {
                native_periodic_record_error(system_errors, task_system, device_error,
                                             Gfn2IntegralDeviceError::kInvalidPrimitiveData);
                continue;
              }
              if (product_exponent > batch.integral_cutoff) continue;
              const double sqrt_inverse_sum = sqrt(inverse_sum);
              const double primitive_prefactor = exp(-product_exponent) *
                                                 kNativePeriodicSqrtPiCubed * sqrt_inverse_sum *
                                                 sqrt_inverse_sum * sqrt_inverse_sum *
                                                 batch.primitive_coefficients[ket_primitive] *
                                                 batch.primitive_coefficients[bra_primitive];
              if (!isfinite(primitive_prefactor)) {
                native_periodic_record_error(system_errors, task_system, device_error,
                                             Gfn2IntegralDeviceError::kNonfiniteIntegralArithmetic);
                continue;
              }
              const double inverse_twice_sum = 0.5 * inverse_sum;
              double axis[3][6][3];
              for (int coordinate = 0; coordinate < 3; ++coordinate) {
                const double product_minus_i = -task_vector[coordinate] * bra_alpha * inverse_sum;
                const double product_minus_j = +task_vector[coordinate] * ket_alpha * inverse_sum;
                native_periodic_make_axis_overlap(
                    product_minus_i, product_minus_j, inverse_twice_sum,
                    static_cast<int>(batch.angular_momenta[task_ket_shell]) + 2,
                    static_cast<int>(batch.angular_momenta[task_bra_shell]), axis[coordinate]);
              }
              overlap_value += primitive_prefactor * axis[0][ket_power[0]][bra_power[0]] *
                               axis[1][ket_power[1]][bra_power[1]] *
                               axis[2][ket_power[2]][bra_power[2]];
              if (batch.model == XtbModelFlavor::kGfn2) {
                for (int component = 0; component < kNativePeriodicMultipoleComponents;
                     ++component) {
                  int moment_power[3];
                  native_periodic_multipole_power(component, &moment_power[0], &moment_power[1],
                                                  &moment_power[2]);
                  multipoles[component] += primitive_prefactor *
                                           axis[0][ket_power[0] + moment_power[0]][bra_power[0]] *
                                           axis[1][ket_power[1] + moment_power[1]][bra_power[1]] *
                                           axis[2][ket_power[2] + moment_power[2]][bra_power[2]];
                }
              }
            }
          }
          bool finite = isfinite(overlap_value);
          if (batch.model == XtbModelFlavor::kGfn2) {
            for (int component = 0; component < kNativePeriodicMultipoleComponents; ++component) {
              finite = finite && isfinite(multipoles[component]);
            }
          }
          cartesian_overlap[cartesian_index] = overlap_value;
          if (batch.model == XtbModelFlavor::kGfn2) {
            for (int component = 0; component < kNativePeriodicMultipoleComponents; ++component) {
              cartesian_multipole[component * kNativePeriodicMaximumCartesianBlock +
                                  cartesian_index] = multipoles[component];
            }
          }
          if (!finite) {
            native_periodic_record_error(system_errors, task_system, device_error,
                                         Gfn2IntegralDeviceError::kNonfiniteIntegralArithmetic);
          }
        }
      }
      __syncthreads();

      if (task_active != 0 && native_periodic_sequence_is_active(workspace) &&
          native_periodic_system_is_valid(system_errors, task_system)) {
        const int spherical_index = static_cast<int>(threadIdx.x);
        if (spherical_index < spherical_block_size) {
          const int bra_ao = spherical_index / ket_spherical_count;
          const int ket_ao = spherical_index % ket_spherical_count;
          double overlap_value = 0.0;
          double raw_multipoles[kNativePeriodicMultipoleComponents] = {};
          for (int bra_cartesian = 0; bra_cartesian < bra_cartesian_count; ++bra_cartesian) {
            const double bra_coefficient = native_periodic_spherical_coefficient(
                batch.angular_momenta[task_bra_shell], bra_ao, bra_cartesian);
            if (bra_coefficient == 0.0) continue;
            for (int ket_cartesian = 0; ket_cartesian < ket_cartesian_count; ++ket_cartesian) {
              const double ket_coefficient = native_periodic_spherical_coefficient(
                  batch.angular_momenta[task_ket_shell], ket_ao, ket_cartesian);
              if (ket_coefficient == 0.0) continue;
              const int index = bra_cartesian * ket_cartesian_count + ket_cartesian;
              overlap_value += bra_coefficient * cartesian_overlap[index] * ket_coefficient;
              if (batch.model == XtbModelFlavor::kGfn2) {
                for (int component = 0; component < kNativePeriodicMultipoleComponents;
                     ++component) {
                  raw_multipoles[component] +=
                      bra_coefficient *
                      cartesian_multipole[component * kNativePeriodicMaximumCartesianBlock +
                                          index] *
                      ket_coefficient;
                }
              }
            }
          }
          double dipole[3] = {raw_multipoles[0], raw_multipoles[1], raw_multipoles[2]};
          const double trace = 0.5 * (raw_multipoles[3] + raw_multipoles[5] + raw_multipoles[8]);
          double quadrupole[6] = {1.5 * raw_multipoles[3] - trace, 1.5 * raw_multipoles[4],
                                  1.5 * raw_multipoles[5] - trace, 1.5 * raw_multipoles[6],
                                  1.5 * raw_multipoles[7],         1.5 * raw_multipoles[8] - trace};
          bool finite = isfinite(overlap_value);
          if (batch.model == XtbModelFlavor::kGfn2) {
            for (double value : dipole) finite = finite && isfinite(value);
            for (double value : quadrupole) finite = finite && isfinite(value);
          }
          const double h0_value = overlap_value * task_h0_factor;
          finite = finite && isfinite(h0_value);
          if (!finite) {
            native_periodic_record_error(system_errors, task_system, device_error,
                                         Gfn2IntegralDeviceError::kNonfiniteIntegralArithmetic);
          } else {
            const std::int64_t orbital_begin = batch.batch_orbital_offsets[task_system];
            const std::int64_t orbital_count =
                batch.batch_orbital_offsets[task_system + 1] - orbital_begin;
            const std::int64_t matrix_begin = batch.matrix_offsets[task_system];
            const std::int64_t bra_orbital =
                batch.shell_orbital_offsets[task_bra_shell] - orbital_begin + bra_ao;
            const std::int64_t ket_orbital =
                batch.shell_orbital_offsets[task_ket_shell] - orbital_begin + ket_ao;
            const std::int64_t forward = matrix_begin + bra_orbital * orbital_count + ket_orbital;
            const std::int64_t reverse = matrix_begin + ket_orbital * orbital_count + bra_orbital;
            /* Each system is owned by exactly one CTA and each AO thread owns a
             * distinct forward/reverse output element for the current shell
             * pair.  Plain += is therefore race-free here and preserves the
             * canonical task order across images. */
            workspace.overlap_scratch[forward] += overlap_value;
            workspace.h0_scratch[forward] += h0_value;
            if (batch.model == XtbModelFlavor::kGfn2) {
              for (int component = 0; component < 3; ++component) {
                workspace.dipole_scratch[component * batch.total_matrix_elements + forward] +=
                    dipole[component];
              }
              for (int component = 0; component < 6; ++component) {
                workspace.quadrupole_scratch[component * batch.total_matrix_elements + forward] +=
                    quadrupole[component];
              }
            }
            const bool publish_reverse = task_is_image != 0 && task_bra_atom != task_ket_atom;
            if (publish_reverse) {
              workspace.overlap_scratch[reverse] += overlap_value;
              workspace.h0_scratch[reverse] += h0_value;
              if (batch.model == XtbModelFlavor::kGfn2) {
                double shifted_dipole[3];
                double shifted_quadrupole[6];
                native_periodic_shifted_multipoles(task_vector, overlap_value, dipole, quadrupole,
                                                   shifted_dipole, shifted_quadrupole);
                for (int component = 0; component < 3; ++component) {
                  if (!isfinite(shifted_dipole[component])) {
                    native_periodic_record_error(
                        system_errors, task_system, device_error,
                        Gfn2IntegralDeviceError::kNonfiniteIntegralArithmetic);
                  } else {
                    workspace.dipole_scratch[component * batch.total_matrix_elements + reverse] +=
                        shifted_dipole[component];
                  }
                }
                for (int component = 0; component < 6; ++component) {
                  if (!isfinite(shifted_quadrupole[component])) {
                    native_periodic_record_error(
                        system_errors, task_system, device_error,
                        Gfn2IntegralDeviceError::kNonfiniteIntegralArithmetic);
                  } else {
                    workspace
                        .quadrupole_scratch[component * batch.total_matrix_elements + reverse] +=
                        shifted_quadrupole[component];
                  }
                }
              }
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ void native_periodic_integral_clear_kernel(Gfn2IntegralDeviceWorkspace workspace,
                                                      const std::uint32_t* device_error,
                                                      std::int64_t matrix_elements,
                                                      std::int64_t dipole_elements,
                                                      std::int64_t quadrupole_elements) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t element = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       element < matrix_elements || element < dipole_elements || element < quadrupole_elements;
       element += stride) {
    if (element < matrix_elements) {
      workspace.overlap_scratch[element] = 0.0;
      workspace.h0_scratch[element] = 0.0;
    }
    if (element < dipole_elements) workspace.dipole_scratch[element] = 0.0;
    if (element < quadrupole_elements) workspace.quadrupole_scratch[element] = 0.0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

__global__ void native_periodic_integral_publish_kernel(Gfn2IntegralDeviceBatch batch,
                                                        Gfn2IntegralDeviceWorkspace workspace,
                                                        double* overlap, double* dipole,
                                                        double* quadrupole, double* hamiltonian,
                                                        const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!native_periodic_sequence_is_active(workspace) ||
      !native_periodic_system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t begin = batch.matrix_offsets[system];
  const std::int64_t end = batch.matrix_offsets[system + 1];
  for (std::int64_t element = begin + threadIdx.x; element < end; element += blockDim.x) {
    overlap[element] = workspace.overlap_scratch[element];
    hamiltonian[element] = workspace.h0_scratch[element];
    if (batch.model == XtbModelFlavor::kGfn2) {
      for (std::int64_t component = 0; component < kGfn2IntegralDipoleComponents; ++component) {
        dipole[component * batch.total_matrix_elements + element] =
            workspace.dipole_scratch[component * batch.total_matrix_elements + element];
      }
      for (std::int64_t component = 0; component < kGfn2IntegralQuadrupoleComponents; ++component) {
        quadrupole[component * batch.total_matrix_elements + element] =
            workspace.quadrupole_scratch[component * batch.total_matrix_elements + element];
      }
    }
  }
}

bool all_tokens_match(const Gfn2PreprocessingDeviceBinding& binding) noexcept {
  const std::uint64_t token = binding.plan_token;
  const bool aes2_enabled = binding.plan.geometry.model == XtbModelFlavor::kGfn2;
  const bool native_periodic_enabled = binding.plan.native_short_range.topology.plan_token != 0u;
  const bool native_integrals_enabled = binding.plan.native_integrals.plan_token != 0u;
  const bool epoch_token_matches =
      (binding.geometry_epoch.value == nullptr && binding.geometry_epoch.value_elements == 0 &&
       binding.geometry_epoch.plan_token == 0u) ||
      binding.geometry_epoch.plan_token == token;
  /* The sparse pair-list leaf is optional.  A disabled leaf contributes no
   * pointers at all and must not participate in the plan-token proof; an
   * enabled leaf requires every sparse view wired to the same plan token. */
  const bool pairlist_enabled = binding.plan.pairlist.batch_size > 0;
  bool pairlist_matches = true;
  if (pairlist_enabled) {
    pairlist_matches = binding.plan.pairlist.plan_token == token &&
                       binding.plan.pairlist.atom_offsets == binding.plan.geometry.atom_offsets &&
                       binding.workspace.pairlist_candidate.plan_token == token &&
                       binding.workspace.pairlist.plan_token == token &&
                       binding.workspace.pairlist_candidate.pair_counts != nullptr &&
                       binding.workspace.pairlist_candidate.neighbor_counts != nullptr &&
                       binding.output.pairlist.plan_token == token &&
                       binding.output.pairlist.state == Gfn2PairListState::kCommitted &&
                       binding.output.pairlist.role == Gfn2PairListRole::kCoordination &&
                       binding.diagnostics.sparse_system_errors != nullptr &&
                       binding.diagnostics.sparse_device_error != nullptr;
  } else {
    const auto& batch = binding.plan.pairlist;
    const auto& candidate = binding.workspace.pairlist_candidate;
    const auto& workspace = binding.workspace.pairlist;
    const auto& output = binding.output.pairlist;
    pairlist_matches =
        batch.batch_size == 0 && batch.total_atoms == 0 && batch.atom_offset_elements == 0 &&
        batch.cutoff == 0.0 && batch.max_cells_per_system == 0 &&
        batch.max_neighbors_per_atom == 0 && batch.max_pairs_per_system == 0 &&
        batch.mode == Gfn2PairListMode::kSparse && batch.plan_token == 0u &&
        batch.atom_offsets == nullptr && batch.flags == 0u && batch.system_modes == nullptr &&
        batch.system_mode_elements == 0 && candidate.pairs == nullptr &&
        candidate.pair_elements == 0 && candidate.pair_offsets == nullptr &&
        candidate.pair_offset_elements == 0 && candidate.pair_counts == nullptr &&
        candidate.pair_count_elements == 0 && candidate.neighbor_offsets == nullptr &&
        candidate.neighbor_offset_elements == 0 && candidate.neighbor_counts == nullptr &&
        candidate.neighbor_count_elements == 0 && candidate.neighbors == nullptr &&
        candidate.neighbor_elements == 0 && candidate.pair_generations == nullptr &&
        candidate.generation_elements == 0 && candidate.plan_token == 0u &&
        workspace.system_meta == nullptr && workspace.system_meta_elements == 0 &&
        workspace.atom_cells == nullptr && workspace.atom_cell_elements == 0 &&
        workspace.cell_counts == nullptr && workspace.cell_count_elements == 0 &&
        workspace.cell_offsets == nullptr && workspace.cell_offset_elements == 0 &&
        workspace.cell_fill == nullptr && workspace.cell_fill_elements == 0 &&
        workspace.cell_atoms == nullptr && workspace.cell_atom_elements == 0 &&
        workspace.neighbor_cursor == nullptr && workspace.neighbor_cursor_elements == 0 &&
        workspace.neighbor_scratch == nullptr && workspace.neighbor_scratch_elements == 0 &&
        workspace.pair_cursor == nullptr && workspace.pair_cursor_elements == 0 &&
        workspace.sequence_active == nullptr && workspace.sequence_elements == 0 &&
        workspace.plan_token == 0u && binding.workspace.sparse_coordination == nullptr &&
        binding.workspace.sparse_coordination_elements == 0 &&
        output.memory_space == Gfn2PlanMemorySpace::kHost &&
        output.state == Gfn2PairListState::kCommitted &&
        output.role == Gfn2PairListRole::kCoordination &&
        output.pair_map_kind == Gfn2PairMapKind::kExplicit && output.plan_token == 0u &&
        output.cutoff_bohr == 0.0 && output.list_builder_cutoff_bohr == 0.0 &&
        output.batch_size == 0 && output.total_atoms == 0 && output.max_pairs_per_system == 0 &&
        output.max_neighbors_per_atom == 0 && output.pair_offset_count == 0 &&
        output.neighbor_offset_count == 0 && output.pair_count == 0 && output.neighbor_count == 0 &&
        output.pair_offsets == nullptr && output.pairs == nullptr &&
        output.pair_count_elements == 0 && output.neighbor_count_elements == 0 &&
        output.pair_counts == nullptr && output.neighbor_counts == nullptr &&
        output.neighbor_offsets == nullptr && output.neighbors == nullptr &&
        output.committed_generation_count == 0 && output.eligible_mask_count == 0 &&
        output.active_mask_count == 0 && output.committed_generations == nullptr &&
        output.eligible_mask == nullptr && output.active_mask == nullptr &&
        binding.diagnostics.sparse_system_errors == nullptr &&
        binding.diagnostics.sparse_system_elements == 0 &&
        binding.diagnostics.sparse_device_error == nullptr;
  }
  return token != 0u && binding.plan.plan_token == token &&
         binding.plan.geometry.plan_token == token && binding.plan.integrals.plan_token == token &&
         binding.plan.h0.plan_token == token && binding.plan.es2.plan_token == token &&
         (!aes2_enabled || binding.plan.aes2.plan_token == token) &&
         (!native_periodic_enabled ||
          (native_integrals_enabled &&
           binding.plan.native_short_range.topology.plan_token == token &&
           binding.plan.native_short_range.atomic_number_elements ==
               binding.plan.geometry.total_atoms &&
           binding.plan.native_short_range.covalent_radius_elements ==
               binding.plan.geometry.total_atoms &&
           binding.plan.native_short_range.position_elements ==
               binding.plan.geometry.coordinate_elements &&
           binding.workspace.native_short_range.plan_token == token)) &&
         (!native_integrals_enabled ||
          (native_periodic_enabled && binding.plan.native_integrals.plan_token == token)) &&
         binding.input.plan_token == token && binding.activity.plan_token == token &&
         binding.output.plan_token == token && binding.output.geometry.plan_token == token &&
         binding.output.es2.plan_token == token &&
         (!aes2_enabled || binding.output.aes2.plan_token == token) &&
         binding.diagnostics.plan_token == token && binding.workspace.plan_token == token &&
         binding.workspace.geometry_candidate.plan_token == token &&
         binding.workspace.geometry.plan_token == token &&
         binding.workspace.integrals.plan_token == token &&
         binding.workspace.es2_candidate.plan_token == token &&
         (!aes2_enabled || binding.workspace.aes2_candidate.plan_token == token) &&
         pairlist_matches && epoch_token_matches;
}

/* Hash the byte-stable POD projection. Dynamic requested-generation metadata
 * is normalized so callers may advance it without rebuilding any descriptor. */
std::uint64_t binding_seal(const Gfn2PreprocessingDeviceBinding& binding) noexcept {
  Gfn2PreprocessingDeviceBinding normalized{};
  std::memcpy(&normalized, &binding, sizeof(normalized));
  normalized.binding_seal = 0u;
  normalized.output.es2.geometry_generation = 0u;
  normalized.output.aes2.geometry_generation = 0u;
  normalized.workspace.es2_candidate.geometry_generation = 0u;
  normalized.workspace.aes2_candidate.geometry_generation = 0u;
  if (normalized.plan.geometry.model == XtbModelFlavor::kGfn1) {
    /* GFN1 owns only scalar overlap/H0/ES2 preprocessing. Make every
     * multipole-only leaf irrelevant to the binding identity so a disabled
     * AES2 or S/D/Q pointer cannot become an accidental cache dependency. */
    normalized.plan.aes2 = {};
    normalized.output.dipole_integrals = nullptr;
    normalized.output.dipole_elements = 0;
    normalized.output.quadrupole_integrals = nullptr;
    normalized.output.quadrupole_elements = 0;
    normalized.output.aes2 = {};
    normalized.workspace.dipole_candidate = nullptr;
    normalized.workspace.dipole_elements = 0;
    normalized.workspace.quadrupole_candidate = nullptr;
    normalized.workspace.quadrupole_elements = 0;
    normalized.workspace.integrals.dipole_scratch = nullptr;
    normalized.workspace.integrals.dipole_elements = 0;
    normalized.workspace.integrals.quadrupole_scratch = nullptr;
    normalized.workspace.integrals.quadrupole_elements = 0;
    normalized.workspace.aes2_candidate = {};
    normalized.workspace.aes2 = {};
    normalized.diagnostics.aes2_system_errors = nullptr;
    normalized.diagnostics.aes2_system_elements = 0;
    normalized.diagnostics.aes2_device_error = nullptr;
  }

  constexpr std::uint64_t kOffsetBasis = 1469598103934665603ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  std::uint64_t hash = kOffsetBasis;
  const auto* bytes = reinterpret_cast<const unsigned char*>(&normalized);
  for (std::size_t index = 0u; index < sizeof(normalized); ++index) {
    hash ^= bytes[index];
    hash *= kPrime;
  }
  return hash == 0u ? 1u : hash;
}

BindingDiagnostic validate_structure(const Gfn2PreprocessingDeviceBinding& binding,
                                     bool require_seal) noexcept {
  if (binding.plan.abi_version != kGfn2PreprocessingAbiVersion || binding.plan.reserved != 0u) {
    return binding_failure(BindingError::kInvalidAbi, BindingField::kPlan);
  }
  if (binding.plan_token == 0u) {
    return binding_failure(BindingError::kInvalidPlanToken, BindingField::kBinding);
  }
  /* Once sealed, any descriptor mutation is stale before its new value is
   * interpreted.  seal_gfn2_preprocessing_binding_cuda still runs the full
   * structural/canonical validation with require_seal=false. */
  if (require_seal && binding.binding_seal != binding_seal(binding)) {
    return binding_failure(
        binding.binding_seal == 0u ? BindingError::kUnsealedBinding : BindingError::kStaleSeal,
        BindingField::kSeal);
  }
  if (!all_tokens_match(binding)) {
    return binding_failure(BindingError::kCrossPlan, BindingField::kPlan);
  }
  const bool epoch_disabled = binding.geometry_epoch.value == nullptr &&
                              binding.geometry_epoch.value_elements == 0 &&
                              binding.geometry_epoch.plan_token == 0u;
  const bool epoch_enabled = binding.geometry_epoch.value_elements == 1 &&
                             binding.geometry_epoch.plan_token == binding.plan_token &&
                             canonical_pointer(binding.geometry_epoch.value, 1);
  if (!epoch_disabled && !epoch_enabled) {
    return binding_failure(BindingError::kInvalidEpoch, BindingField::kEpoch);
  }
  const bool admission_disabled = binding.admission.error == nullptr &&
                                  binding.admission.error_elements == 0 &&
                                  binding.admission.plan_token == 0u;
  const bool admission_enabled = binding.admission.error_elements == 1 &&
                                 binding.admission.plan_token == binding.plan_token &&
                                 canonical_pointer(binding.admission.error, 1);
  if (!admission_disabled && !admission_enabled) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kActivity);
  }

  const Gfn2GeometryDeviceBatch& geometry = binding.plan.geometry;
  const Gfn2IntegralDeviceBatch& integrals = binding.plan.integrals;
  const Gfn2H0DevicePlan& h0 = binding.plan.h0;
  const Gfn2ES2DeviceBatch& es2 = binding.plan.es2;
  const Gfn2AES2DeviceBatch& aes2 = binding.plan.aes2;
  const Gfn2NativePeriodicIntegralDeviceBatch& native_integrals = binding.plan.native_integrals;
  const bool multipoles_enabled = geometry.model == XtbModelFlavor::kGfn2;
  const bool aes2_enabled = multipoles_enabled;
  const std::int64_t batch = geometry.batch_size;
  const std::int64_t atoms = geometry.total_atoms;
  const std::int64_t pairs = geometry.total_pairs;
  const std::int64_t shells = integrals.total_shells;
  const std::int64_t matrices = integrals.total_matrix_elements;
  const std::int64_t shell_matrices = es2.total_matrix_elements;
  std::int64_t coordinates = 0;
  std::int64_t geometry_pair_elements = 0;
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  std::int64_t aes2_pair_elements = 0;
  std::int64_t shell_grid_per_system = 0;
  std::int64_t shell_grid_blocks = 0;
  const bool legacy_shell_grid_valid =
      integrals.use_compact_tasks != 0u ||
      (checked_multiply(integrals.maximum_system_shells, integrals.maximum_system_shells,
                        shell_grid_per_system) &&
       checked_multiply(shell_grid_per_system, batch, shell_grid_blocks) &&
       shell_grid_blocks <= static_cast<std::int64_t>(std::numeric_limits<int>::max()));
  if (batch <= 0 || atoms <= 0 || pairs < 0 || shells <= 0 || matrices <= 0 ||
      shell_matrices <= 0 || integrals.total_orbitals <= 0 || integrals.total_primitives <= 0 ||
      integrals.total_shell_pair_elements <= 0 || integrals.maximum_system_shells <= 0 ||
      integrals.linear_tiles_per_system <= 0 ||
      integrals.linear_tiles_per_system > kGfn2IntegralLinearBlockBudget ||
      !(integrals.integral_cutoff > 0.0) || !std::isfinite(integrals.integral_cutoff) ||
      atoms == std::numeric_limits<std::int64_t>::max() ||
      shells == std::numeric_limits<std::int64_t>::max() ||
      batch > static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max()) ||
      batch > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      integrals.use_compact_tasks > 1u || integrals.reserved != 0u ||
      integrals.forward_generic_task_count < 0 || integrals.forward_ss_task_count < 0 ||
      integrals.h0_generic_task_count < 0 || integrals.h0_ss_task_count < 0 ||
      integrals.force_generic_task_count < 0 || integrals.force_ss_task_count < 0 ||
      integrals.forward_generic_task_count >
          static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      integrals.forward_ss_task_count >
          static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      integrals.h0_generic_task_count >
          static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      integrals.h0_ss_task_count > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      integrals.force_generic_task_count >
          static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      integrals.force_ss_task_count > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      !legacy_shell_grid_valid || !checked_multiply(atoms, 3, coordinates) ||
      !checked_multiply(pairs, kGfn2GeometryPairDataElements, geometry_pair_elements) ||
      !valid_xtb_model_flavor(geometry.model) || integrals.model != geometry.model ||
      es2.model != geometry.model ||
      (multipoles_enabled &&
       !checked_multiply(matrices, kGfn2IntegralDipoleComponents, dipole_elements)) ||
      (multipoles_enabled &&
       !checked_multiply(matrices, kGfn2IntegralQuadrupoleComponents, quadrupole_elements)) ||
      (aes2_enabled && !checked_multiply(pairs, kGfn2AES2PairDataElements, aes2_pair_elements))) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kPlan);
  }

  const bool compatible_extents =
      geometry.atom_offset_elements == batch + 1 && geometry.pair_offset_elements == batch + 1 &&
      geometry.covalent_radius_elements == atoms && geometry.coordinate_elements == coordinates &&
      integrals.batch_size == batch && integrals.total_atoms == atoms &&
      integrals.atom_offset_count == batch + 1 && integrals.batch_shell_offset_count == batch + 1 &&
      integrals.batch_orbital_offset_count == batch + 1 &&
      integrals.matrix_offset_count == batch + 1 &&
      integrals.shell_pair_offset_count == batch + 1 &&
      integrals.atom_shell_offset_count == atoms + 1 &&
      integrals.shell_orbital_offset_count == shells + 1 &&
      integrals.shell_primitive_offset_count == shells + 1 &&
      integrals.shell_to_atom_count == shells && integrals.angular_momentum_count == shells &&
      integrals.primitive_exponent_count == integrals.total_primitives &&
      integrals.primitive_coefficient_count == integrals.total_primitives &&
      h0.atomic_radius_count == atoms && h0.shell_level_count == shells &&
      h0.shell_coordination_scale_count == shells && h0.shell_polynomial_count == shells &&
      h0.shell_pair_scale_count == integrals.total_shell_pair_elements && es2.batch_size == batch &&
      es2.total_atoms == atoms && es2.total_shells == shells &&
      es2.atom_offset_count == batch + 1 && es2.batch_shell_offset_count == batch + 1 &&
      es2.atom_shell_offset_count == atoms + 1 && es2.matrix_offset_count == batch + 1 &&
      es2.shell_to_atom_count == shells && es2.shell_hardness_count == shells &&
      (!aes2_enabled ||
       (aes2.batch_size == batch && aes2.total_atoms == atoms && aes2.total_pairs == pairs &&
        aes2.atom_offset_count == batch + 1 && aes2.pair_offset_count == batch + 1 &&
        aes2.dipole_kernel_count == atoms && aes2.quadrupole_kernel_count == atoms &&
        aes2.multipole_radius_count == atoms && aes2.multipole_valence_cn_count == atoms));
  if (!compatible_extents) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kPlan);
  }

  const bool task_domains_absent =
      integrals.forward_generic_task_count == 0 && integrals.forward_ss_task_count == 0 &&
      integrals.h0_generic_task_count == 0 && integrals.h0_ss_task_count == 0 &&
      integrals.force_generic_task_count == 0 && integrals.force_ss_task_count == 0 &&
      integrals.forward_generic_tasks == nullptr && integrals.forward_ss_tasks == nullptr &&
      integrals.h0_generic_tasks == nullptr && integrals.h0_ss_tasks == nullptr &&
      integrals.force_generic_tasks == nullptr && integrals.force_ss_tasks == nullptr;
  const std::int64_t forward_tasks =
      integrals.forward_generic_task_count + integrals.forward_ss_task_count;
  const std::int64_t h0_tasks = integrals.h0_generic_task_count + integrals.h0_ss_task_count;
  const bool task_domains_present =
      forward_tasks > 0 && h0_tasks > 0 &&
      canonical_pointer(integrals.forward_generic_tasks, integrals.forward_generic_task_count) &&
      canonical_pointer(integrals.forward_ss_tasks, integrals.forward_ss_task_count) &&
      canonical_pointer(integrals.h0_generic_tasks, integrals.h0_generic_task_count) &&
      canonical_pointer(integrals.h0_ss_tasks, integrals.h0_ss_task_count) &&
      canonical_pointer(integrals.force_generic_tasks, integrals.force_generic_task_count) &&
      canonical_pointer(integrals.force_ss_tasks, integrals.force_ss_task_count);
  if ((!task_domains_absent && !task_domains_present) ||
      (integrals.use_compact_tasks != 0u && !task_domains_present)) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kPlan);
  }

  const bool canonical_topology = geometry.atom_offsets == integrals.atom_offsets &&
                                  geometry.atom_offsets == es2.atom_offsets &&
                                  (!aes2_enabled || geometry.atom_offsets == aes2.atom_offsets) &&
                                  (!aes2_enabled || geometry.pair_offsets == aes2.pair_offsets) &&
                                  integrals.batch_shell_offsets == es2.batch_shell_offsets &&
                                  integrals.atom_shell_offsets == es2.atom_shell_offsets &&
                                  integrals.shell_to_atom == es2.shell_to_atom;
  if (!canonical_topology) {
    return binding_failure(BindingError::kCrossPlan, BindingField::kPlan);
  }

  const bool plan_pointers =
      canonical_pointer(geometry.atom_offsets, batch + 1) &&
      canonical_pointer(geometry.pair_offsets, batch + 1) &&
      canonical_pointer(geometry.covalent_radii, atoms) &&
      canonical_pointer(integrals.batch_shell_offsets, batch + 1) &&
      canonical_pointer(integrals.batch_orbital_offsets, batch + 1) &&
      canonical_pointer(integrals.matrix_offsets, batch + 1) &&
      canonical_pointer(integrals.shell_pair_offsets, batch + 1) &&
      canonical_pointer(integrals.atom_shell_offsets, atoms + 1) &&
      canonical_pointer(integrals.shell_orbital_offsets, shells + 1) &&
      canonical_pointer(integrals.shell_primitive_offsets, shells + 1) &&
      canonical_pointer(integrals.shell_to_atom, shells) &&
      canonical_pointer(integrals.angular_momenta, shells) &&
      canonical_pointer(integrals.primitive_exponents, integrals.total_primitives) &&
      canonical_pointer(integrals.primitive_coefficients, integrals.total_primitives) &&
      canonical_pointer(integrals.forward_generic_tasks, integrals.forward_generic_task_count) &&
      canonical_pointer(integrals.forward_ss_tasks, integrals.forward_ss_task_count) &&
      canonical_pointer(integrals.h0_generic_tasks, integrals.h0_generic_task_count) &&
      canonical_pointer(integrals.h0_ss_tasks, integrals.h0_ss_task_count) &&
      canonical_pointer(integrals.force_generic_tasks, integrals.force_generic_task_count) &&
      canonical_pointer(integrals.force_ss_tasks, integrals.force_ss_task_count) &&
      canonical_pointer(h0.atomic_radii, atoms) && canonical_pointer(h0.shell_levels, shells) &&
      canonical_pointer(h0.shell_coordination_scale, shells) &&
      canonical_pointer(h0.shell_polynomial, shells) &&
      canonical_pointer(h0.shell_pair_scale, integrals.total_shell_pair_elements) &&
      canonical_pointer(es2.matrix_offsets, batch + 1) &&
      canonical_pointer(es2.shell_hardness, shells) &&
      (!aes2_enabled || (canonical_pointer(aes2.dipole_kernel, atoms) &&
                         canonical_pointer(aes2.quadrupole_kernel, atoms) &&
                         canonical_pointer(aes2.multipole_radius, atoms) &&
                         canonical_pointer(aes2.multipole_valence_cn, atoms)));
  if (!plan_pointers) {
    return binding_failure(BindingError::kInvalidPointer, BindingField::kPlan);
  }

  const bool native_periodic_enabled = binding.plan.native_short_range.topology.plan_token != 0u;
  if (native_periodic_enabled) {
    const auto& native = binding.plan.native_short_range;
    const auto& topology = native.topology;
    const auto& workspace_native = binding.workspace.native_short_range;
    if (native.atomic_number_elements != atoms || native.covalent_radius_elements != atoms ||
        native.position_elements != coordinates || topology.plan_token != binding.plan_token ||
        topology.batch_size != batch || topology.total_atoms != atoms ||
        topology.atom_offset_count != batch + 1 || topology.translation_offset_count != batch + 1 ||
        topology.cell_elements != batch * 9 || topology.periodic_axes_elements != batch ||
        topology.atom_offsets == nullptr || topology.cell_matrices == nullptr ||
        topology.periodic_axes == nullptr || topology.translation_offsets == nullptr ||
        topology.translations == nullptr || !canonical_pointer(native.atomic_numbers, atoms) ||
        !canonical_pointer(native.positions, coordinates) ||
        !canonical_pointer(native.covalent_radii, atoms) ||
        workspace_native.plan_token != binding.plan_token ||
        workspace_native.wrapped_position_elements != coordinates ||
        workspace_native.coordination_elements != atoms ||
        workspace_native.repulsion_energy_elements != batch ||
        workspace_native.repulsion_gradient_elements != coordinates ||
        workspace_native.repulsion_strain_elements != batch * 9 ||
        !canonical_pointer(workspace_native.wrapped_positions, coordinates) ||
        !canonical_pointer(workspace_native.coordination, atoms) ||
        !canonical_pointer(workspace_native.repulsion_energies, batch) ||
        !canonical_pointer(workspace_native.repulsion_gradients, coordinates) ||
        !canonical_pointer(workspace_native.repulsion_strain, batch * 9)) {
      return binding_failure(BindingError::kInvalidExtent, BindingField::kPlan);
    }
  } else {
    const auto& native = binding.plan.native_short_range;
    const auto& workspace_native = binding.workspace.native_short_range;
    if (native.topology.plan_token != 0u || native.atomic_numbers != nullptr ||
        native.atomic_number_elements != 0 || native.positions != nullptr ||
        native.position_elements != 0 || native.covalent_radii != nullptr ||
        native.covalent_radius_elements != 0 || workspace_native.plan_token != 0u ||
        workspace_native.wrapped_positions != nullptr ||
        workspace_native.wrapped_position_elements != 0 ||
        workspace_native.coordination != nullptr || workspace_native.coordination_elements != 0 ||
        workspace_native.repulsion_energies != nullptr ||
        workspace_native.repulsion_energy_elements != 0 ||
        workspace_native.repulsion_gradients != nullptr ||
        workspace_native.repulsion_gradient_elements != 0 ||
        workspace_native.repulsion_strain != nullptr ||
        workspace_native.repulsion_strain_elements != 0) {
      return binding_failure(BindingError::kCrossPlan, BindingField::kPlan);
    }
  }

  const bool native_integrals_enabled = native_integrals.plan_token != 0u;
  if (native_integrals_enabled) {
    const bool cutoff_squared_finite =
        std::isfinite(native_integrals.realspace_cutoff * native_integrals.realspace_cutoff);
    const bool native_integral_extents =
        native_periodic_enabled && native_integrals.plan_token == binding.plan_token &&
        native_integrals.translation_offset_elements == batch + 1 &&
        native_integrals.translation_elements > 0 &&
        native_integrals.max_translations_per_system > 0 &&
        native_integrals.max_translations_per_system <= native_integrals.translation_elements &&
        native_integrals.realspace_cutoff > 0.0 && cutoff_squared_finite &&
        canonical_pointer(native_integrals.translation_offsets,
                          native_integrals.translation_offset_elements) &&
        canonical_pointer(native_integrals.translations, native_integrals.translation_elements);
    if (!native_integral_extents) {
      return binding_failure(BindingError::kInvalidExtent, BindingField::kPlan);
    }
  } else if (native_integrals.translation_offset_elements != 0 ||
             native_integrals.translation_elements != 0 ||
             native_integrals.max_translations_per_system != 0 ||
             native_integrals.realspace_cutoff != 0.0 ||
             native_integrals.translation_offsets != nullptr ||
             native_integrals.translations != nullptr) {
    return binding_failure(BindingError::kCrossPlan, BindingField::kPlan);
  }

  if (binding.input.position_elements != coordinates ||
      !canonical_pointer(binding.input.positions, coordinates)) {
    return binding_failure(BindingError::kInvalidPointer, BindingField::kPositions);
  }
  if (binding.activity.requested_elements != batch ||
      binding.activity.published_elements != batch ||
      !canonical_pointer(binding.activity.requested_mask, batch) ||
      !canonical_pointer(binding.activity.published_mask, batch)) {
    return binding_failure(BindingError::kInvalidActivity, BindingField::kActivity);
  }

  const Gfn2PreprocessingDeviceOutput& output = binding.output;
  const bool output_valid =
      output.geometry.pair_data_elements == geometry_pair_elements &&
      output.geometry.coordination_elements == atoms &&
      output.geometry.generation_elements == batch &&
      canonical_pointer(output.geometry.pair_data, geometry_pair_elements) &&
      canonical_pointer(output.geometry.coordination_numbers, atoms) &&
      canonical_pointer(output.geometry.geometry_generations, batch) &&
      output.overlap_elements == matrices && canonical_pointer(output.overlap, matrices) &&
      output.dipole_elements == dipole_elements &&
      canonical_pointer(output.dipole_integrals, dipole_elements) &&
      output.quadrupole_elements == quadrupole_elements &&
      canonical_pointer(output.quadrupole_integrals, quadrupole_elements) &&
      output.h0_elements == matrices && canonical_pointer(output.h0, matrices) &&
      output.es2.matrix_elements == shell_matrices &&
      canonical_pointer(output.es2.coulomb_matrix, shell_matrices) &&
      output.aes2.pair_data_elements == aes2_pair_elements &&
      canonical_pointer(output.aes2.pair_data, aes2_pair_elements) &&
      output.generation_elements == batch && canonical_pointer(output.operator_generations, batch);
  if (!output_valid) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kOutput);
  }

  const Gfn2PreprocessingDeviceDiagnostics& diagnostics = binding.diagnostics;
  const bool diagnostics_valid =
      diagnostics.geometry_system_elements == batch &&
      canonical_pointer(diagnostics.geometry_system_errors, batch) &&
      canonical_pointer(diagnostics.geometry_device_error, 1) &&
      diagnostics.integral_system_elements == batch &&
      canonical_pointer(diagnostics.integral_system_errors, batch) &&
      canonical_pointer(diagnostics.integral_device_error, 1) &&
      canonical_pointer(diagnostics.es2_device_error, 1) &&
      (!aes2_enabled || (diagnostics.aes2_system_elements == batch &&
                         canonical_pointer(diagnostics.aes2_system_errors, batch) &&
                         canonical_pointer(diagnostics.aes2_device_error, 1))) &&
      diagnostics.system_stage_elements == batch &&
      canonical_pointer(diagnostics.system_stages, batch) &&
      canonical_pointer(diagnostics.plan_error, 1);
  if (!diagnostics_valid) {
    return binding_failure(BindingError::kInvalidDiagnostics, BindingField::kDiagnostics);
  }

  const Gfn2PreprocessingDeviceWorkspace& workspace = binding.workspace;
  const bool workspace_valid =
      workspace.position_elements == coordinates &&
      canonical_pointer(workspace.positions_scratch, coordinates) &&
      workspace.geometry_candidate.pair_data_elements == geometry_pair_elements &&
      workspace.geometry_candidate.coordination_elements == atoms &&
      workspace.geometry_candidate.generation_elements == batch &&
      canonical_pointer(workspace.geometry_candidate.pair_data, geometry_pair_elements) &&
      canonical_pointer(workspace.geometry_candidate.coordination_numbers, atoms) &&
      canonical_pointer(workspace.geometry_candidate.geometry_generations, batch) &&
      workspace.geometry.pair_elements == geometry_pair_elements &&
      workspace.geometry.coordination_elements == atoms &&
      workspace.geometry.gradient_elements == 0 && workspace.geometry.gradient_scratch == nullptr &&
      workspace.geometry.sequence_elements == 1 &&
      canonical_pointer(workspace.geometry.pair_scratch, geometry_pair_elements) &&
      canonical_pointer(workspace.geometry.coordination_scratch, atoms) &&
      canonical_pointer(workspace.geometry.sequence_active, 1) &&
      workspace.overlap_elements == matrices &&
      canonical_pointer(workspace.overlap_candidate, matrices) &&
      workspace.dipole_elements == dipole_elements &&
      canonical_pointer(workspace.dipole_candidate, dipole_elements) &&
      workspace.quadrupole_elements == quadrupole_elements &&
      canonical_pointer(workspace.quadrupole_candidate, quadrupole_elements) &&
      workspace.h0_elements == matrices && canonical_pointer(workspace.h0_candidate, matrices) &&
      workspace.integrals.overlap_elements == matrices &&
      canonical_pointer(workspace.integrals.overlap_scratch, matrices) &&
      workspace.integrals.dipole_elements == dipole_elements &&
      canonical_pointer(workspace.integrals.dipole_scratch, dipole_elements) &&
      workspace.integrals.quadrupole_elements == quadrupole_elements &&
      canonical_pointer(workspace.integrals.quadrupole_scratch, quadrupole_elements) &&
      workspace.integrals.h0_elements == matrices &&
      canonical_pointer(workspace.integrals.h0_scratch, matrices) &&
      workspace.integrals.sequence_elements == 1 &&
      canonical_pointer(workspace.integrals.sequence_active, 1) &&
      workspace.es2_candidate.matrix_elements == shell_matrices &&
      canonical_pointer(workspace.es2_candidate.coulomb_matrix, shell_matrices) &&
      workspace.es2.matrix_elements == shell_matrices &&
      canonical_pointer(workspace.es2.matrix_scratch, shell_matrices) &&
      workspace.es2.shell_elements == 0 && workspace.es2.shell_scratch == nullptr &&
      workspace.es2.batch_elements == 0 && workspace.es2.batch_scratch == nullptr &&
      workspace.es2.gradient_elements == 0 && workspace.es2.gradient_scratch == nullptr &&
      workspace.aes2_candidate.pair_data_elements == aes2_pair_elements &&
      canonical_pointer(workspace.aes2_candidate.pair_data, aes2_pair_elements) &&
      workspace.aes2.pair_elements == aes2_pair_elements &&
      canonical_pointer(workspace.aes2.pair_scratch, aes2_pair_elements) &&
      workspace.aes2.potential_elements == 0 && workspace.aes2.potential_scratch == nullptr &&
      workspace.aes2.batch_elements == 0 && workspace.aes2.batch_scratch == nullptr &&
      workspace.aes2.gradient_elements == 0 && workspace.aes2.gradient_scratch == nullptr &&
      workspace.aes2.coordination_elements == 0 && workspace.aes2.coordination_scratch == nullptr &&
      workspace.aes2.scc_peer_error_elements == 0 &&
      workspace.aes2.scc_peer_error_scratch == nullptr &&
      (!native_periodic_enabled ||
       (workspace.native_short_range.plan_token == binding.plan_token &&
        workspace.native_short_range.wrapped_position_elements == coordinates &&
        workspace.native_short_range.coordination_elements == atoms &&
        workspace.native_short_range.repulsion_energy_elements == batch &&
        workspace.native_short_range.repulsion_gradient_elements == coordinates &&
        workspace.native_short_range.repulsion_strain_elements == batch * 9 &&
        canonical_pointer(workspace.native_short_range.wrapped_positions, coordinates) &&
        canonical_pointer(workspace.native_short_range.coordination, atoms) &&
        canonical_pointer(workspace.native_short_range.repulsion_energies, batch) &&
        canonical_pointer(workspace.native_short_range.repulsion_gradients, coordinates) &&
        canonical_pointer(workspace.native_short_range.repulsion_strain, batch * 9)));
  if (!workspace_valid) {
    return binding_failure(BindingError::kInvalidWorkspace, BindingField::kWorkspace);
  }

  /* Optional sparse pair-list leaf.  A disabled leaf carries no pointers; an
   * enabled leaf must describe exactly the canonical geometry partition with
   * per-system capacities and fully wired candidate/workspace/output domains.
   * The host scheduler chooses the batch mode with gfn2_pairlist_use_sparse_for
   * and provisions the fixed-topology capacities up front. */
  const Gfn2PairListDeviceBatch& pairlist = binding.plan.pairlist;
  const bool pairlist_enabled = pairlist.batch_size > 0;
  if (pairlist_enabled) {
    std::int64_t pairlist_neighbors_capacity = 0;
    std::int64_t pairlist_cells_capacity = 0;
    std::int64_t pairlist_pairs_capacity = 0;
    const bool pairlist_capacity_products =
        checked_multiply(atoms, pairlist.max_neighbors_per_atom, pairlist_neighbors_capacity) &&
        checked_multiply(batch, pairlist.max_cells_per_system, pairlist_cells_capacity) &&
        checked_multiply(batch, pairlist.max_pairs_per_system, pairlist_pairs_capacity) &&
        pairlist.max_cells_per_system < std::numeric_limits<std::int64_t>::max() &&
        checked_multiply(batch, pairlist.max_cells_per_system + 1, pairlist_cells_capacity);
    std::int64_t pairlist_scratch = 0;
    const bool pairlist_extents =
        pairlist_capacity_products && pairlist.plan_token == binding.plan_token &&
        pairlist.batch_size == batch && pairlist.total_atoms == atoms &&
        pairlist.atom_offset_elements == batch + 1 &&
        pairlist.atom_offsets == geometry.atom_offsets && std::isfinite(pairlist.cutoff) &&
        pairlist.cutoff >= kDefaultPairlistCutoffBohr && pairlist.max_cells_per_system > 0 &&
        pairlist.max_neighbors_per_atom > 0 && pairlist.max_pairs_per_system > 0 &&
        (pairlist.mode == Gfn2PairListMode::kSparse || pairlist.mode == Gfn2PairListMode::kDense) &&
        (pairlist.flags & ~kGfn2PairListAllowDenseFallback) == 0u &&
        pairlist.system_modes != nullptr && pairlist.system_mode_elements == batch &&
        canonical_pointer(pairlist.system_modes, batch) &&
        checked_multiply(atoms, pairlist.max_neighbors_per_atom, pairlist_scratch) &&
        workspace.sparse_coordination_elements == atoms &&
        canonical_pointer(workspace.sparse_coordination, atoms) &&
        workspace.pairlist_candidate.pair_elements >= pairlist_pairs_capacity &&
        workspace.pairlist_candidate.pair_offset_elements == batch + 1 &&
        workspace.pairlist_candidate.pair_count_elements >= batch &&
        workspace.pairlist_candidate.neighbor_offset_elements == atoms + 1 &&
        workspace.pairlist_candidate.neighbor_count_elements >= atoms &&
        workspace.pairlist_candidate.neighbor_elements >= pairlist_neighbors_capacity &&
        workspace.pairlist_candidate.generation_elements >= batch &&
        workspace.pairlist_candidate.plan_token == binding.plan_token &&
        canonical_pointer(workspace.pairlist_candidate.pairs,
                          workspace.pairlist_candidate.pair_elements) &&
        canonical_pointer(workspace.pairlist_candidate.pair_offsets,
                          workspace.pairlist_candidate.pair_offset_elements) &&
        canonical_pointer(workspace.pairlist_candidate.pair_counts,
                          workspace.pairlist_candidate.pair_count_elements) &&
        canonical_pointer(workspace.pairlist_candidate.neighbor_offsets,
                          workspace.pairlist_candidate.neighbor_offset_elements) &&
        canonical_pointer(workspace.pairlist_candidate.neighbor_counts,
                          workspace.pairlist_candidate.neighbor_count_elements) &&
        canonical_pointer(workspace.pairlist_candidate.neighbors,
                          workspace.pairlist_candidate.neighbor_elements) &&
        canonical_pointer(workspace.pairlist_candidate.pair_generations,
                          workspace.pairlist_candidate.generation_elements) &&
        workspace.pairlist.system_meta_elements >= batch &&
        workspace.pairlist.atom_cell_elements >= atoms &&
        workspace.pairlist.cell_count_elements >= pairlist_cells_capacity &&
        workspace.pairlist.cell_offset_elements >= pairlist_cells_capacity &&
        workspace.pairlist.cell_fill_elements >= pairlist_cells_capacity &&
        workspace.pairlist.cell_atom_elements >= atoms &&
        workspace.pairlist.neighbor_cursor_elements >= atoms &&
        workspace.pairlist.neighbor_scratch_elements >= pairlist_neighbors_capacity &&
        workspace.pairlist.pair_cursor_elements >= batch &&
        workspace.pairlist.sequence_elements >= 1 &&
        workspace.pairlist.plan_token == binding.plan_token &&
        canonical_pointer(workspace.pairlist.system_meta, batch) &&
        canonical_pointer(workspace.pairlist.atom_cells, atoms) &&
        canonical_pointer(workspace.pairlist.cell_counts, pairlist_cells_capacity) &&
        canonical_pointer(workspace.pairlist.cell_offsets, pairlist_cells_capacity) &&
        canonical_pointer(workspace.pairlist.cell_fill, pairlist_cells_capacity) &&
        canonical_pointer(workspace.pairlist.cell_atoms, workspace.pairlist.cell_atom_elements) &&
        canonical_pointer(workspace.pairlist.neighbor_cursor, atoms) &&
        canonical_pointer(workspace.pairlist.neighbor_scratch, pairlist_neighbors_capacity) &&
        canonical_pointer(workspace.pairlist.pair_cursor, batch) &&
        canonical_pointer(workspace.pairlist.sequence_active, 1) &&
        diagnostics.sparse_system_elements == batch &&
        canonical_pointer(diagnostics.sparse_system_errors, batch) &&
        canonical_pointer(diagnostics.sparse_device_error, 1);
    if (!pairlist_extents) {
      return binding_failure(BindingError::kInvalidWorkspace, BindingField::kPairlist);
    }
    /* Committed output view (ABI step 4): the final per-system gate publishes
     * the candidate list into fixed-stride per-system slices with explicit
     * counts, per-system committed generations, and an eligibility mask, so a
     * failed peer is ineligible and never shifts a later peer's slice. */
    const Gfn2PairListConsumerView& committed = binding.output.pairlist;
    std::int64_t committed_pairs_capacity = 0;
    std::int64_t committed_neighbors_capacity = 0;
    const bool committed_extents =
        committed.memory_space == Gfn2PlanMemorySpace::kCudaDevice &&
        committed.state == Gfn2PairListState::kCommitted &&
        committed.role == Gfn2PairListRole::kCoordination &&
        committed.pair_map_kind == Gfn2PairMapKind::kExplicit &&
        committed.plan_token == binding.plan_token && committed.batch_size == batch &&
        committed.total_atoms == atoms &&
        committed.max_pairs_per_system == pairlist.max_pairs_per_system &&
        committed.max_neighbors_per_atom == pairlist.max_neighbors_per_atom &&
        committed.cutoff_bohr == kDefaultPairlistCutoffBohr &&
        committed.list_builder_cutoff_bohr == pairlist.cutoff &&
        committed.pair_offset_count == batch + 1 && committed.neighbor_offset_count == atoms + 1 &&
        committed.pair_count_elements == batch && committed.neighbor_count_elements == atoms &&
        committed.pair_count == pairlist_pairs_capacity &&
        committed.neighbor_count == pairlist_neighbors_capacity &&
        committed.committed_generation_count == batch && committed.eligible_mask_count == batch &&
        committed.active_mask_count == 0 &&
        checked_multiply(batch, pairlist.max_pairs_per_system, committed_pairs_capacity) &&
        checked_multiply(atoms, pairlist.max_neighbors_per_atom, committed_neighbors_capacity) &&
        committed.pair_count == committed_pairs_capacity &&
        committed.neighbor_count == committed_neighbors_capacity &&
        canonical_pointer(committed.pair_offsets, committed.pair_offset_count) &&
        canonical_pointer(committed.pairs, committed.pair_count) &&
        canonical_pointer(committed.pair_counts, committed.pair_count_elements) &&
        canonical_pointer(committed.neighbor_offsets, committed.neighbor_offset_count) &&
        canonical_pointer(committed.neighbor_counts, committed.neighbor_count_elements) &&
        canonical_pointer(committed.neighbors, committed.neighbor_count) &&
        canonical_pointer(committed.committed_generations, committed.committed_generation_count) &&
        canonical_pointer(committed.eligible_mask, committed.eligible_mask_count) &&
        committed.active_mask == nullptr;
    if (!committed_extents) {
      return binding_failure(BindingError::kInvalidExtent, BindingField::kOutput);
    }
  }

  RangeList<64> reads;
  RangeList<80> writes;
  const bool ranges_valid =
      reads.add(geometry.atom_offsets, batch + 1) && reads.add(geometry.pair_offsets, batch + 1) &&
      reads.add(geometry.covalent_radii, atoms) &&
      reads.add(integrals.batch_shell_offsets, batch + 1) &&
      reads.add(integrals.batch_orbital_offsets, batch + 1) &&
      reads.add(integrals.matrix_offsets, batch + 1) &&
      reads.add(integrals.shell_pair_offsets, batch + 1) &&
      reads.add(integrals.atom_shell_offsets, atoms + 1) &&
      reads.add(integrals.shell_orbital_offsets, shells + 1) &&
      reads.add(integrals.shell_primitive_offsets, shells + 1) &&
      reads.add(integrals.shell_to_atom, shells) && reads.add(integrals.angular_momenta, shells) &&
      reads.add(integrals.primitive_exponents, integrals.total_primitives) &&
      reads.add(integrals.primitive_coefficients, integrals.total_primitives) &&
      reads.add(integrals.forward_generic_tasks, integrals.forward_generic_task_count) &&
      reads.add(integrals.forward_ss_tasks, integrals.forward_ss_task_count) &&
      reads.add(integrals.h0_generic_tasks, integrals.h0_generic_task_count) &&
      reads.add(integrals.h0_ss_tasks, integrals.h0_ss_task_count) &&
      reads.add(integrals.force_generic_tasks, integrals.force_generic_task_count) &&
      reads.add(integrals.force_ss_tasks, integrals.force_ss_task_count) &&
      reads.add(h0.atomic_radii, atoms) && reads.add(h0.shell_levels, shells) &&
      reads.add(h0.shell_coordination_scale, shells) && reads.add(h0.shell_polynomial, shells) &&
      reads.add(h0.shell_pair_scale, integrals.total_shell_pair_elements) &&
      reads.add(es2.matrix_offsets, batch + 1) && reads.add(es2.shell_hardness, shells) &&
      (!aes2_enabled ||
       (reads.add(aes2.dipole_kernel, atoms) && reads.add(aes2.quadrupole_kernel, atoms) &&
        reads.add(aes2.multipole_radius, atoms) && reads.add(aes2.multipole_valence_cn, atoms))) &&
      (!native_periodic_enabled ||
       (reads.add(binding.plan.native_short_range.topology.cell_matrices, batch * 9) &&
        reads.add(binding.plan.native_short_range.topology.periodic_axes, batch) &&
        reads.add(binding.plan.native_short_range.topology.translation_offsets, batch + 1) &&
        reads.add(binding.plan.native_short_range.topology.translations,
                  binding.plan.native_short_range.topology.total_translations) &&
        reads.add(binding.plan.native_short_range.atomic_numbers, atoms) &&
        reads.add(binding.plan.native_short_range.covalent_radii, atoms))) &&
      (!native_integrals_enabled ||
       (reads.add(native_integrals.translation_offsets,
                  native_integrals.translation_offset_elements) &&
        reads.add(native_integrals.translations, native_integrals.translation_elements))) &&
      reads.add(binding.input.positions, coordinates) &&
      reads.add(binding.admission.error, binding.admission.error_elements) &&
      reads.add(binding.activity.requested_mask, batch) &&
      (!pairlist_enabled ||
       reads.add(binding.plan.pairlist.system_modes, binding.plan.pairlist.system_mode_elements)) &&
      writes.add(binding.activity.published_mask, batch) &&
      writes.add(output.geometry.pair_data, geometry_pair_elements) &&
      writes.add(output.geometry.coordination_numbers, atoms) &&
      writes.add(output.geometry.geometry_generations, batch) &&
      writes.add(output.overlap, matrices) &&
      writes.add(output.dipole_integrals, dipole_elements) &&
      writes.add(output.quadrupole_integrals, quadrupole_elements) &&
      writes.add(output.h0, matrices) && writes.add(output.es2.coulomb_matrix, shell_matrices) &&
      writes.add(output.aes2.pair_data, aes2_pair_elements) &&
      writes.add(output.operator_generations, batch) &&
      writes.add(workspace.positions_scratch, coordinates) &&
      writes.add(workspace.geometry_candidate.pair_data, geometry_pair_elements) &&
      writes.add(workspace.geometry_candidate.coordination_numbers, atoms) &&
      writes.add(workspace.geometry_candidate.geometry_generations, batch) &&
      writes.add(workspace.geometry.pair_scratch, geometry_pair_elements) &&
      writes.add(workspace.geometry.coordination_scratch, atoms) &&
      writes.add(workspace.geometry.sequence_active, 1) &&
      writes.add(workspace.overlap_candidate, matrices) &&
      writes.add(workspace.dipole_candidate, dipole_elements) &&
      writes.add(workspace.quadrupole_candidate, quadrupole_elements) &&
      writes.add(workspace.h0_candidate, matrices) &&
      writes.add(workspace.integrals.overlap_scratch, matrices) &&
      writes.add(workspace.integrals.dipole_scratch, dipole_elements) &&
      writes.add(workspace.integrals.quadrupole_scratch, quadrupole_elements) &&
      writes.add(workspace.integrals.h0_scratch, matrices) &&
      writes.add(workspace.integrals.sequence_active, 1) &&
      writes.add(workspace.es2_candidate.coulomb_matrix, shell_matrices) &&
      writes.add(workspace.es2.matrix_scratch, shell_matrices) &&
      writes.add(workspace.aes2_candidate.pair_data, aes2_pair_elements) &&
      writes.add(workspace.aes2.pair_scratch, aes2_pair_elements) &&
      (!native_periodic_enabled ||
       (writes.add(workspace.native_short_range.wrapped_positions, coordinates) &&
        writes.add(workspace.native_short_range.coordination, atoms) &&
        writes.add(workspace.native_short_range.repulsion_energies, batch) &&
        writes.add(workspace.native_short_range.repulsion_gradients, coordinates) &&
        writes.add(workspace.native_short_range.repulsion_strain, batch * 9))) &&
      writes.add(diagnostics.geometry_system_errors, batch) &&
      writes.add(diagnostics.geometry_device_error, 1) &&
      writes.add(diagnostics.integral_system_errors, batch) &&
      writes.add(diagnostics.integral_device_error, 1) &&
      writes.add(diagnostics.es2_device_error, 1) &&
      (!aes2_enabled || (writes.add(diagnostics.aes2_system_errors, batch) &&
                         writes.add(diagnostics.aes2_device_error, 1))) &&
      writes.add(diagnostics.system_stages, batch) && writes.add(diagnostics.plan_error, 1);
  const bool sparse_ranges_valid =
      !pairlist_enabled ||
      (writes.add(workspace.sparse_coordination, workspace.sparse_coordination_elements) &&
       writes.add(workspace.pairlist_candidate.pairs, workspace.pairlist_candidate.pair_elements) &&
       writes.add(workspace.pairlist_candidate.pair_offsets,
                  workspace.pairlist_candidate.pair_offset_elements) &&
       writes.add(workspace.pairlist_candidate.pair_counts,
                  workspace.pairlist_candidate.pair_count_elements) &&
       writes.add(workspace.pairlist_candidate.neighbor_offsets,
                  workspace.pairlist_candidate.neighbor_offset_elements) &&
       writes.add(workspace.pairlist_candidate.neighbor_counts,
                  workspace.pairlist_candidate.neighbor_count_elements) &&
       writes.add(workspace.pairlist_candidate.neighbors,
                  workspace.pairlist_candidate.neighbor_elements) &&
       writes.add(workspace.pairlist_candidate.pair_generations,
                  workspace.pairlist_candidate.generation_elements) &&
       writes.add(workspace.pairlist.system_meta, workspace.pairlist.system_meta_elements) &&
       writes.add(workspace.pairlist.atom_cells, workspace.pairlist.atom_cell_elements) &&
       writes.add(workspace.pairlist.cell_counts, workspace.pairlist.cell_count_elements) &&
       writes.add(workspace.pairlist.cell_offsets, workspace.pairlist.cell_offset_elements) &&
       writes.add(workspace.pairlist.cell_fill, workspace.pairlist.cell_fill_elements) &&
       writes.add(workspace.pairlist.cell_atoms, workspace.pairlist.cell_atom_elements) &&
       writes.add(workspace.pairlist.neighbor_cursor,
                  workspace.pairlist.neighbor_cursor_elements) &&
       writes.add(workspace.pairlist.neighbor_scratch,
                  workspace.pairlist.neighbor_scratch_elements) &&
       writes.add(workspace.pairlist.pair_cursor, workspace.pairlist.pair_cursor_elements) &&
       writes.add(workspace.pairlist.sequence_active, workspace.pairlist.sequence_elements) &&
       writes.add(output.pairlist.pairs, output.pairlist.pair_count) &&
       writes.add(output.pairlist.pair_offsets, output.pairlist.pair_offset_count) &&
       writes.add(output.pairlist.pair_counts, output.pairlist.pair_count_elements) &&
       writes.add(output.pairlist.neighbor_offsets, output.pairlist.neighbor_offset_count) &&
       writes.add(output.pairlist.neighbor_counts, output.pairlist.neighbor_count_elements) &&
       writes.add(output.pairlist.neighbors, output.pairlist.neighbor_count) &&
       writes.add(output.pairlist.committed_generations,
                  output.pairlist.committed_generation_count) &&
       writes.add(output.pairlist.eligible_mask, output.pairlist.eligible_mask_count) &&
       writes.add(diagnostics.sparse_system_errors, diagnostics.sparse_system_elements) &&
       writes.add(diagnostics.sparse_device_error, 1));
  const bool epoch_range_valid =
      epoch_disabled ||
      writes.add(binding.geometry_epoch.value, binding.geometry_epoch.value_elements);
  if (!ranges_valid || !sparse_ranges_valid || !epoch_range_valid) {
    return binding_failure(BindingError::kInvalidExtent, BindingField::kWorkspace);
  }
  if (!writes_are_disjoint(reads, writes)) {
    return binding_failure(BindingError::kInvalidAlias, BindingField::kWorkspace);
  }

  return {};
}

__device__ std::uint32_t read_u32(const std::uint32_t* value) {
  return atomicAdd(const_cast<std::uint32_t*>(value), 0u);
}

__device__ void record_plan_error(std::uint32_t* plan_error, Gfn2PreprocessingDeviceError error) {
  atomicCAS(plan_error, static_cast<std::uint32_t>(Gfn2PreprocessingDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ std::uint64_t load_geometry_generation(GeometryGenerationSource source) {
  return source.device == nullptr
             ? source.scalar
             : atomicAdd(
                   reinterpret_cast<unsigned long long*>(const_cast<std::uint64_t*>(source.device)),
                   0ULL);
}

/* The CAS loop prevents wraparound even if a caller violates the documented
 * single-flight contract. Concurrent use still has no inference-level
 * ordering guarantee, but it cannot publish a duplicate or zero epoch. */
__global__ void advance_geometry_epoch_kernel(Gfn2GeometryEpochDevice epoch,
                                              Gfn2DeviceAdmission admission,
                                              std::uint32_t* plan_error) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || !gfn2_request_admitted(admission) ||
      read_u32(plan_error) != 0u) {
    return;
  }
  auto* const value = reinterpret_cast<unsigned long long*>(epoch.value);
  unsigned long long observed = atomicAdd(value, 0ULL);
  while (observed != ~0ULL) {
    const unsigned long long previous = atomicCAS(value, observed, observed + 1ULL);
    if (previous == observed) {
      return;
    }
    observed = previous;
  }
  record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kGeometryEpochOverflow);
}

__device__ void store_safe_positions(std::int64_t atom_begin, std::int64_t atom,
                                     double* positions) {
  const double local = static_cast<double>(atom - atom_begin);
  positions[atom * 3] = 2.0 * local;
  positions[atom * 3 + 1] = 0.125 * local;
  positions[atom * 3 + 2] = -0.0625 * local;
}

__global__ void prepare_positions_kernel(Gfn2GeometryDeviceBatch batch, const double* input,
                                         const std::uint8_t* requested, double* scratch,
                                         std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  const std::uint8_t active = requested[system];
  if (begin < 0 || begin > end || end > batch.total_atoms || (system == 0 && begin != 0) ||
      (system + 1 == batch.batch_size && end != batch.total_atoms)) {
    if (threadIdx.x == 0) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kInvalidComposerOffsets);
    }
    return;
  }
  if (active > 1u) {
    if (threadIdx.x == 0) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kInvalidActivity);
    }
  }
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    if (active == 1u) {
      scratch[atom * 3] = input[atom * 3];
      scratch[atom * 3 + 1] = input[atom * 3 + 1];
      scratch[atom * 3 + 2] = input[atom * 3 + 2];
    } else {
      store_safe_positions(begin, atom, scratch);
    }
  }
}

__global__ void gate_composer_plan_kernel(
    std::int64_t batch_size, const std::uint32_t* plan_error, std::uint32_t* geometry_system_errors,
    std::uint32_t* geometry_device_error, std::uint32_t* integral_system_errors,
    std::uint32_t* integral_device_error, std::uint32_t* es2_device_error,
    std::uint32_t* aes2_system_errors, std::uint32_t* aes2_device_error, bool aes2_enabled) {
  if (read_u32(plan_error) == 0u) {
    return;
  }
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size) {
    geometry_system_errors[system] =
        static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kInvalidOffsets);
    integral_system_errors[system] =
        static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidOffsets);
    if (aes2_enabled) {
      aes2_system_errors[system] = static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidOffsets);
    }
  }
  if (system == 0) {
    *geometry_device_error = static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kInvalidOffsets);
    *integral_device_error = static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidOffsets);
    *es2_device_error = static_cast<std::uint32_t>(Gfn2ES2DeviceError::kInvalidOffsets);
    if (aes2_enabled) {
      *aes2_device_error = static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidOffsets);
    }
  }
}

__global__ void gate_h0_kernel(std::int64_t batch_size, const std::uint8_t* requested,
                               const std::uint32_t* geometry_errors,
                               std::uint32_t* integral_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size && requested[system] == 1u && geometry_errors[system] != 0u) {
    atomicCAS(integral_errors + system, 0u,
              static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidCoordination));
  }
}

/* evaluate_gfn2_h0_cuda snapshots the same sticky primitive-domain scalar as
 * evaluate_gfn2_integrals_cuda. A peer numerical error from S/D/Q must not be
 * reinterpreted as a plan-wide H0 gate, so clear only when the integral
 * topology snapshot proved healthy. Per-system codes remain authoritative. */
__global__ void prepare_h0_sequence_kernel(const std::uint32_t* integral_sequence,
                                           std::uint32_t* integral_device_error) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && read_u32(integral_sequence) == 1u) {
    *integral_device_error = 0u;
  }
}

__global__ void prepare_late_stages_kernel(Gfn2GeometryDeviceBatch batch,
                                           const std::uint8_t* requested,
                                           const std::uint32_t* geometry_errors,
                                           const std::uint32_t* integral_errors, double* positions,
                                           std::uint32_t* aes2_errors, std::uint32_t* plan_error,
                                           bool aes2_enabled) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  if (begin < 0 || begin > end || end > batch.total_atoms) {
    if (threadIdx.x == 0) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kInvalidComposerOffsets);
    }
    return;
  }
  const bool failed =
      requested[system] != 1u || geometry_errors[system] != 0u || integral_errors[system] != 0u;
  if (aes2_enabled && threadIdx.x == 0 && requested[system] == 1u && failed) {
    atomicCAS(aes2_errors + system, 0u,
              static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidCoordination));
  }
  if (failed) {
    for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
      store_safe_positions(begin, atom, positions);
    }
  }
}

__device__ bool geometry_plan_code(std::uint32_t code) {
  return code == static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kInvalidOffsets) ||
         code == static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kInvalidCovalentRadius);
}

__device__ bool integral_plan_code(std::uint32_t code) {
  return code == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidOffsets) ||
         code == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidShellMetadata) ||
         code == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidPrimitiveData) ||
         code == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kInvalidH0Parameter);
}

__device__ bool aes2_plan_code(std::uint32_t code) {
  return code == static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidOffsets) ||
         code == static_cast<std::uint32_t>(Gfn2AES2DeviceError::kInvalidElementParameter);
}

/*
 * Bitwise dense/sparse CN consistency gate.  One block per system compares the
 * dense geometry-cache coordination numbers (the authoritative production
 * output) against the bucketed sparse pair-list coordination numbers for every
 * healthy, requested peer.  Any disagreement records kSparseCoordinationMismatch
 * in the peer's geometry error slot (and the sparse domain markers), so the
 * existing publication gate rejects that peer closed.  Failed or inactive
 * peers are intentionally skipped, preserving peer isolation.
 */
__global__ void gate_sparse_coordination_kernel(Gfn2GeometryDeviceBatch geometry,
                                                Gfn2PairListDeviceBatch pairlist,
                                                const double* dense_coordination,
                                                const double* sparse_coordination,
                                                const std::uint32_t* sparse_sequence_active,
                                                Gfn2PreprocessingDeviceActivity activity,
                                                Gfn2PreprocessingDeviceDiagnostics diagnostics) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const bool requested = activity.requested_mask[system] == 1u;
  const bool geometry_healthy =
      atomicAdd(const_cast<std::uint32_t*>(diagnostics.geometry_system_errors + system), 0u) ==
      static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess);
  const bool sparse_healthy =
      atomicAdd(const_cast<std::uint32_t*>(diagnostics.sparse_system_errors + system), 0u) ==
      static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess);
  if (!requested || !geometry_healthy) {
    return;
  }
  const bool sparse_sequence_ok =
      atomicAdd(const_cast<std::uint32_t*>(sparse_sequence_active), 0u) == 1u;
  if (!sparse_sequence_ok || !sparse_healthy) {
    /* A peer-local sparse error is folded into the existing geometry
     * publication gate; a topology-wide sequence failure closes every
     * requested healthy peer. */
    atomicCAS(diagnostics.geometry_system_errors + system,
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSparseCoordinationMismatch));
    return;
  }
  if (pairlist.batch_size != geometry.batch_size || pairlist.total_atoms != geometry.total_atoms ||
      pairlist.atom_offsets != geometry.atom_offsets) {
    atomicCAS(diagnostics.geometry_system_errors + system,
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSparseCoordinationMismatch));
    return;
  }
  const std::int64_t atom_begin = geometry.atom_offsets[system];
  const std::int64_t atom_end = geometry.atom_offsets[system + 1];
  __shared__ unsigned int shared_mismatch;
  if (threadIdx.x == 0) {
    shared_mismatch = 0u;
  }
  __syncthreads();
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::uint64_t dense_bits =
        *reinterpret_cast<const std::uint64_t*>(dense_coordination + atom);
    const std::uint64_t sparse_bits =
        *reinterpret_cast<const std::uint64_t*>(sparse_coordination + atom);
    if (dense_bits != sparse_bits) {
      atomicExch(&shared_mismatch, 1u);
    }
  }
  __syncthreads();
  if (shared_mismatch != 0u && threadIdx.x == 0) {
    atomicCAS(diagnostics.geometry_system_errors + system,
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSparseCoordinationMismatch));
    atomicCAS(diagnostics.sparse_system_errors + system,
              static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2PairListDeviceError::kInvalidCache));
    atomicCAS(diagnostics.sparse_device_error,
              static_cast<std::uint32_t>(Gfn2PairListDeviceError::kSuccess),
              static_cast<std::uint32_t>(Gfn2PairListDeviceError::kInvalidCache));
  }
}

/*
 * Make the sparse pair list the authoritative CN producer for the current
 * epoch.  The gate above proved the sparse coordination numbers equal the
 * dense geometry cache bitwise for every healthy requested peer; this kernel
 * then overwrites the dense CN in the geometry candidate with the sparse
 * value, so H0, AES2 geometry, and the publication path all consume the
 * sparse-produced CN.  The dense path remains as the differential reference
 * and its seven-value pair cache still feeds the coordination VJP until the
 * sparse VJP is wired into the force path.  Failed/inactive peers are skipped.
 */
__global__ void promote_sparse_coordination_kernel(Gfn2GeometryDeviceBatch geometry,
                                                   Gfn2PairListDeviceBatch pairlist,
                                                   const double* sparse_coordination,
                                                   double* dense_coordination,
                                                   Gfn2PreprocessingDeviceActivity activity,
                                                   Gfn2PreprocessingDeviceDiagnostics diagnostics) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const bool requested = activity.requested_mask[system] == 1u;
  if (!requested) {
    return;
  }
  const bool geometry_healthy =
      atomicAdd(const_cast<std::uint32_t*>(diagnostics.geometry_system_errors + system), 0u) ==
      static_cast<std::uint32_t>(Gfn2GeometryDeviceError::kSuccess);
  if (!geometry_healthy) {
    return;
  }
  const std::int64_t atom_begin = geometry.atom_offsets[system];
  const std::int64_t atom_end = geometry.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    dense_coordination[atom] = sparse_coordination[atom];
  }
  static_cast<void>(pairlist);
}

/*
 * Commit the sparse pair list into the stable public output view through the
 * final per-system gate.  Only requested, healthy peers publish: their
 * committed generation advances and their eligibility byte flips.  A failed or
 * inactive peer keeps its last good payload/counts/generation and receives an
 * ineligible byte for the attempted refresh.  Consumers read
 * only eligible peers and trust explicit per-peer counts + committed
 * generations (the consumer-view contract), so one peer's failure can never
 * publish a partial or shifted slice for another.
 *
 * Committed offsets address fixed-capacity slots.  Explicit counts delimit the
 * live prefix copied from the compact candidate, so a failed peer cannot move
 * another peer's bytes.  Fixed slot offsets may be rewritten after plan-wide
 * validation; the per-peer kernel changes persistent metadata and payload only
 * for a peer that passed the final publication gate.
 */
__global__ void initialize_committed_pairlist_metadata_kernel(Gfn2PairListDeviceBatch pairlist,
                                                              Gfn2PairListConsumerView committed,
                                                              Gfn2DeviceAdmission admission,
                                                              const std::uint32_t* plan_error) {
  if (!gfn2_request_admitted(admission) ||
      atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) != 0u) {
    return;
  }
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto* const pair_offsets = const_cast<std::int64_t*>(committed.pair_offsets);
  auto* const neighbor_offsets = const_cast<std::int64_t*>(committed.neighbor_offsets);
  if (index <= pairlist.batch_size) {
    pair_offsets[index] = index * pairlist.max_pairs_per_system;
  }
  if (index <= pairlist.total_atoms) {
    neighbor_offsets[index] = index * pairlist.max_neighbors_per_atom;
  }
}

/* Kernel descriptors must be copied into CUDA argument storage.  A reference
 * parameter instead exposes the launcher's host address to device code, which
 * is invalid on GPUs that cannot directly access host stack memory. */
__global__ void commit_pairlist_kernel(Gfn2PairListDeviceBatch pairlist,
                                       Gfn2PairListDeviceCache candidate,
                                       Gfn2PairListConsumerView committed,
                                       GeometryGenerationSource generation_source,
                                       Gfn2PreprocessingDeviceActivity activity,
                                       Gfn2DeviceAdmission admission,
                                       const std::uint32_t* plan_error) {
  /* A plan-wide failure is transactional for every committed array.  In
   * particular, do not clear generations or eligibility after the metadata
   * initializer has already declined to run. */
  if (!gfn2_request_admitted(admission) ||
      atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) != 0u) {
    return;
  }
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::uint64_t generation = load_geometry_generation(generation_source);
  /* published_mask is the single final transaction gate.  Geometry can be
   * healthy while integral/H0/AES2 or a plan-wide failure has already closed
   * publication, so re-deriving eligibility from the geometry error alone
   * would expose a partial operator transaction to sparse consumers. */
  const bool commit = activity.published_mask[system] == 1u && generation != 0u;
  /* The committed consumer view is a const projection; the output arrays are
   * the mutable publication target owned by the caller, so publish through
   * mutable local aliases. */
  auto* const committed_pairs = const_cast<xtbloom::detail::Gfn2AtomPair*>(committed.pairs);
  auto* const committed_pair_counts = const_cast<std::int64_t*>(committed.pair_counts);
  auto* const committed_neighbor_counts = const_cast<std::int64_t*>(committed.neighbor_counts);
  auto* const committed_neighbors = const_cast<std::int64_t*>(committed.neighbors);
  auto* const committed_generations = const_cast<std::uint64_t*>(committed.committed_generations);
  auto* const committed_eligible_mask = const_cast<std::uint8_t*>(committed.eligible_mask);
  if (threadIdx.x == 0) {
    if (committed_eligible_mask != nullptr) {
      committed_eligible_mask[system] = commit ? 1u : 0u;
    }
    if (commit && committed_generations != nullptr) {
      committed_generations[system] = generation;
    }
  }
  if (!commit) {
    return;
  }
  const std::int64_t atom_begin = pairlist.atom_offsets[system];
  const std::int64_t atom_end = pairlist.atom_offsets[system + 1];
  const std::int64_t candidate_pair_begin = candidate.pair_offsets[system];
  const std::int64_t pair_count = candidate.pair_counts[system];
  const std::int64_t committed_pair_begin = system * pairlist.max_pairs_per_system;
  for (std::int64_t index = threadIdx.x; index < pair_count; index += blockDim.x) {
    committed_pairs[committed_pair_begin + index] = candidate.pairs[candidate_pair_begin + index];
  }
  if (threadIdx.x == 0) {
    committed_pair_counts[system] = pair_count;
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t count = candidate.neighbor_counts[atom];
    const std::int64_t candidate_begin = candidate.neighbor_offsets[atom];
    const std::int64_t committed_begin = atom * pairlist.max_neighbors_per_atom;
    for (std::int64_t index = 0; index < count; ++index) {
      committed_neighbors[committed_begin + index] = candidate.neighbors[candidate_begin + index];
    }
    committed_neighbor_counts[atom] = count;
  }
}

__global__ void classify_plan_kernel(
    std::int64_t batch_size, const std::uint32_t* geometry_sequence,
    const std::uint32_t* sparse_sequence, const std::uint32_t* integral_sequence,
    const std::uint32_t* geometry_errors, const std::uint32_t* integral_errors,
    const std::uint32_t* es2_error, const std::uint32_t* aes2_errors, std::uint32_t* plan_error,
    bool aes2_enabled) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || read_u32(plan_error) != 0u) {
    return;
  }
  if (read_u32(geometry_sequence) == 0u) {
    record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kGeometryPlanFailure);
    return;
  }
  if (sparse_sequence != nullptr && read_u32(sparse_sequence) == 0u) {
    /* Pair-list preflight validates one shared topology and dispatch vector.
     * A closed sequence is plan-wide and must stop committed metadata
     * initialization, rather than being reclassified as peer-local CN drift. */
    record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kSparsePairlistFailure);
    return;
  }
  if (read_u32(integral_sequence) == 0u) {
    record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kIntegralPlanFailure);
    return;
  }
  if (read_u32(es2_error) != 0u) {
    record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kEs2PlanFailure);
    return;
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    if (geometry_plan_code(geometry_errors[system])) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kGeometryPlanFailure);
      return;
    }
    if (integral_plan_code(integral_errors[system])) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kIntegralPlanFailure);
      return;
    }
    if (aes2_enabled && aes2_plan_code(aes2_errors[system])) {
      record_plan_error(plan_error, Gfn2PreprocessingDeviceError::kAes2PlanFailure);
      return;
    }
  }
}

__global__ void publish_preprocessing_kernel(Gfn2PreprocessingDevicePlan plan,
                                             Gfn2PreprocessingDeviceActivity activity,
                                             Gfn2PreprocessingDeviceOutput output,
                                             Gfn2PreprocessingDeviceWorkspace workspace,
                                             Gfn2PreprocessingDeviceDiagnostics diagnostics,
                                             Gfn2DeviceAdmission admission,
                                             GeometryGenerationSource generation_source) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!gfn2_request_admitted(admission)) return;
  const bool requested = activity.requested_mask[system] == 1u;
  const std::uint32_t geometry_error = diagnostics.geometry_system_errors[system];
  const std::uint32_t integral_error = diagnostics.integral_system_errors[system];
  const bool multipoles_enabled = plan.geometry.model == XtbModelFlavor::kGfn2;
  const bool aes2_enabled = multipoles_enabled;
  const std::uint32_t aes2_error = aes2_enabled ? diagnostics.aes2_system_errors[system] : 0u;
  std::uint32_t stage = static_cast<std::uint32_t>(Gfn2PreprocessingSystemStage::kSuccess);
  if (requested) {
    if (geometry_error != 0u) {
      stage = static_cast<std::uint32_t>(Gfn2PreprocessingSystemStage::kGeometry);
    } else if (integral_error != 0u) {
      stage = static_cast<std::uint32_t>(Gfn2PreprocessingSystemStage::kIntegralsOrH0);
    } else if (aes2_error != 0u) {
      stage = static_cast<std::uint32_t>(Gfn2PreprocessingSystemStage::kAes2);
    }
  }
  if (threadIdx.x == 0) {
    diagnostics.system_stages[system] = stage;
  }

  const bool publish = requested && stage == 0u && read_u32(diagnostics.plan_error) == 0u;
  if (!publish) {
    if (threadIdx.x == 0) {
      activity.published_mask[system] = 0u;
    }
    return;
  }

  const std::uint64_t generation = load_geometry_generation(generation_source);
  if (generation == 0u) {
    if (threadIdx.x == 0) {
      activity.published_mask[system] = 0u;
      record_plan_error(diagnostics.plan_error,
                        Gfn2PreprocessingDeviceError::kGeometryEpochOverflow);
    }
    return;
  }

  const std::int64_t atom_begin = plan.geometry.atom_offsets[system];
  const std::int64_t atom_end = plan.geometry.atom_offsets[system + 1];
  const std::int64_t pair_begin = plan.geometry.pair_offsets[system];
  const std::int64_t pair_end = plan.geometry.pair_offsets[system + 1];
  const std::int64_t matrix_begin = plan.integrals.matrix_offsets[system];
  const std::int64_t matrix_end = plan.integrals.matrix_offsets[system + 1];
  const std::int64_t es2_begin = plan.es2.matrix_offsets[system];
  const std::int64_t es2_end = plan.es2.matrix_offsets[system + 1];

  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    output.geometry.coordination_numbers[atom] =
        workspace.geometry_candidate.coordination_numbers[atom];
  }
  for (std::int64_t pair = pair_begin + threadIdx.x; pair < pair_end; pair += blockDim.x) {
    for (std::int64_t component = 0; component < kGfn2GeometryPairDataElements; ++component) {
      output.geometry.pair_data[pair * kGfn2GeometryPairDataElements + component] =
          workspace.geometry_candidate.pair_data[pair * kGfn2GeometryPairDataElements + component];
    }
    if (aes2_enabled) {
      for (std::int64_t component = 0; component < kGfn2AES2PairDataElements; ++component) {
        output.aes2.pair_data[pair * kGfn2AES2PairDataElements + component] =
            workspace.aes2_candidate.pair_data[pair * kGfn2AES2PairDataElements + component];
      }
    }
  }
  for (std::int64_t element = matrix_begin + threadIdx.x; element < matrix_end;
       element += blockDim.x) {
    output.overlap[element] = workspace.overlap_candidate[element];
    output.h0[element] = workspace.h0_candidate[element];
    if (multipoles_enabled) {
      for (std::int64_t component = 0; component < kGfn2IntegralDipoleComponents; ++component) {
        output.dipole_integrals[component * plan.integrals.total_matrix_elements + element] =
            workspace.dipole_candidate[component * plan.integrals.total_matrix_elements + element];
      }
      for (std::int64_t component = 0; component < kGfn2IntegralQuadrupoleComponents; ++component) {
        output.quadrupole_integrals[component * plan.integrals.total_matrix_elements + element] =
            workspace
                .quadrupole_candidate[component * plan.integrals.total_matrix_elements + element];
      }
    }
  }
  for (std::int64_t element = es2_begin + threadIdx.x; element < es2_end; element += blockDim.x) {
    output.es2.coulomb_matrix[element] = workspace.es2_candidate.coulomb_matrix[element];
  }
  /* The sparse list is committed by commit_pairlist_kernel after this final
   * operator gate has published the authoritative per-peer decision. */
  if (threadIdx.x == 0) {
    output.geometry.geometry_generations[system] = generation;
    output.operator_generations[system] = generation;
    activity.published_mask[system] = 1u;
  }
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

Gfn2PreprocessingLaunchDiagnostic launch_failure(const BindingDiagnostic& binding,
                                                 cudaError_t status) noexcept {
  Gfn2PreprocessingLaunchDiagnostic result{};
  result.binding = binding;
  result.cuda_status = status;
  return result;
}

}  // namespace

cudaError_t evaluate_gfn2_native_periodic_integrals_h0_cuda(
    const Gfn2IntegralDeviceBatch& batch, const Gfn2H0DevicePlan& h0,
    const Gfn2NativePeriodicIntegralDeviceBatch& periodic, const double* wrapped_positions,
    const double* coordination_numbers, double* overlap, double* dipole, double* quadrupole,
    double* hamiltonian, const Gfn2IntegralDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const bool multipoles_enabled = batch.model == XtbModelFlavor::kGfn2;
  std::int64_t coordinates = 0;
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  std::int64_t maximum_shell_pair_count = 0;
  std::int64_t image_system_stride = 0;
  std::int64_t image_task_count = 0;
  std::int64_t onsite_task_count = 0;
  std::int64_t total_tasks = 0;
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_shells <= 0 ||
      batch.total_orbitals <= 0 || batch.total_primitives <= 0 ||
      batch.total_matrix_elements <= 0 || batch.total_shell_pair_elements <= 0 ||
      batch.maximum_system_shells <= 0 || batch.linear_tiles_per_system <= 0 ||
      !valid_xtb_model_flavor(batch.model) || batch.plan_token == 0u ||
      periodic.plan_token != batch.plan_token ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max()) ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.batch_orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.shell_pair_offset_count != batch.batch_size + 1 ||
      batch.atom_shell_offset_count != batch.total_atoms + 1 ||
      batch.shell_orbital_offset_count != batch.total_shells + 1 ||
      batch.shell_primitive_offset_count != batch.total_shells + 1 ||
      batch.shell_to_atom_count != batch.total_shells ||
      batch.angular_momentum_count != batch.total_shells ||
      batch.primitive_exponent_count != batch.total_primitives ||
      batch.primitive_coefficient_count != batch.total_primitives ||
      !(batch.integral_cutoff > 0.0) || !std::isfinite(batch.integral_cutoff) ||
      batch.use_compact_tasks > 1u || batch.reserved != 0u || h0.plan_token != batch.plan_token ||
      h0.atomic_radius_count != batch.total_atoms || h0.shell_level_count != batch.total_shells ||
      h0.shell_coordination_scale_count != batch.total_shells ||
      h0.shell_polynomial_count != batch.total_shells ||
      h0.shell_pair_scale_count != batch.total_shell_pair_elements ||
      periodic.translation_offset_elements != batch.batch_size + 1 ||
      periodic.translation_elements <= 0 || periodic.max_translations_per_system <= 0 ||
      periodic.max_translations_per_system > periodic.translation_elements ||
      !(periodic.realspace_cutoff > 0.0) || !std::isfinite(periodic.realspace_cutoff) ||
      !std::isfinite(periodic.realspace_cutoff * periodic.realspace_cutoff) ||
      periodic.translation_offsets == nullptr || periodic.translations == nullptr ||
      workspace.plan_token != batch.plan_token || workspace.sequence_elements != 1 ||
      workspace.overlap_elements != batch.total_matrix_elements ||
      workspace.h0_elements != batch.total_matrix_elements || overlap == nullptr ||
      hamiltonian == nullptr || coordination_numbers == nullptr || system_errors == nullptr ||
      device_error == nullptr || !checked_multiply(batch.total_atoms, 3, coordinates) ||
      !checked_multiply(batch.total_matrix_elements, kGfn2IntegralDipoleComponents,
                        dipole_elements) ||
      !checked_multiply(batch.total_matrix_elements, kGfn2IntegralQuadrupoleComponents,
                        quadrupole_elements) ||
      !checked_multiply(batch.maximum_system_shells, batch.maximum_system_shells,
                        maximum_shell_pair_count) ||
      !checked_multiply(periodic.max_translations_per_system, maximum_shell_pair_count,
                        image_system_stride) ||
      !checked_multiply(batch.batch_size, image_system_stride, image_task_count) ||
      !checked_multiply(batch.batch_size, maximum_shell_pair_count, onsite_task_count) ||
      image_task_count > std::numeric_limits<std::int64_t>::max() - onsite_task_count) {
    return cudaErrorInvalidValue;
  }
  total_tasks = image_task_count + onsite_task_count;

  const bool pointers_valid =
      canonical_pointer(batch.atom_offsets, batch.atom_offset_count) &&
      canonical_pointer(batch.batch_shell_offsets, batch.batch_shell_offset_count) &&
      canonical_pointer(batch.batch_orbital_offsets, batch.batch_orbital_offset_count) &&
      canonical_pointer(batch.matrix_offsets, batch.matrix_offset_count) &&
      canonical_pointer(batch.shell_pair_offsets, batch.shell_pair_offset_count) &&
      canonical_pointer(batch.atom_shell_offsets, batch.atom_shell_offset_count) &&
      canonical_pointer(batch.shell_orbital_offsets, batch.shell_orbital_offset_count) &&
      canonical_pointer(batch.shell_primitive_offsets, batch.shell_primitive_offset_count) &&
      canonical_pointer(batch.shell_to_atom, batch.shell_to_atom_count) &&
      canonical_pointer(batch.angular_momenta, batch.angular_momentum_count) &&
      canonical_pointer(batch.primitive_exponents, batch.primitive_exponent_count) &&
      canonical_pointer(batch.primitive_coefficients, batch.primitive_coefficient_count) &&
      canonical_pointer(h0.atomic_radii, h0.atomic_radius_count) &&
      canonical_pointer(h0.shell_levels, h0.shell_level_count) &&
      canonical_pointer(h0.shell_coordination_scale, h0.shell_coordination_scale_count) &&
      canonical_pointer(h0.shell_polynomial, h0.shell_polynomial_count) &&
      canonical_pointer(h0.shell_pair_scale, h0.shell_pair_scale_count) &&
      canonical_pointer(periodic.translation_offsets, periodic.translation_offset_elements) &&
      canonical_pointer(periodic.translations, periodic.translation_elements) &&
      canonical_pointer(wrapped_positions, coordinates) &&
      canonical_pointer(coordination_numbers, batch.total_atoms) &&
      canonical_pointer(overlap, batch.total_matrix_elements) &&
      canonical_pointer(hamiltonian, batch.total_matrix_elements) &&
      canonical_pointer(dipole, multipoles_enabled ? dipole_elements : 0) &&
      canonical_pointer(quadrupole, multipoles_enabled ? quadrupole_elements : 0) &&
      canonical_pointer(workspace.overlap_scratch, workspace.overlap_elements) &&
      canonical_pointer(workspace.dipole_scratch,
                        multipoles_enabled ? workspace.dipole_elements : 0) &&
      canonical_pointer(workspace.quadrupole_scratch,
                        multipoles_enabled ? workspace.quadrupole_elements : 0) &&
      canonical_pointer(workspace.h0_scratch, workspace.h0_elements) &&
      canonical_pointer(workspace.sequence_active, workspace.sequence_elements) &&
      workspace.dipole_elements == (multipoles_enabled ? dipole_elements : 0) &&
      workspace.quadrupole_elements == (multipoles_enabled ? quadrupole_elements : 0);
  if (!pointers_valid) return cudaErrorInvalidValue;

  RangeList<64> reads;
  RangeList<32> writes;
  const bool ranges_valid =
      reads.add(batch.atom_offsets, batch.atom_offset_count) &&
      reads.add(batch.batch_shell_offsets, batch.batch_shell_offset_count) &&
      reads.add(batch.batch_orbital_offsets, batch.batch_orbital_offset_count) &&
      reads.add(batch.matrix_offsets, batch.matrix_offset_count) &&
      reads.add(batch.shell_pair_offsets, batch.shell_pair_offset_count) &&
      reads.add(batch.atom_shell_offsets, batch.atom_shell_offset_count) &&
      reads.add(batch.shell_orbital_offsets, batch.shell_orbital_offset_count) &&
      reads.add(batch.shell_primitive_offsets, batch.shell_primitive_offset_count) &&
      reads.add(batch.shell_to_atom, batch.shell_to_atom_count) &&
      reads.add(batch.angular_momenta, batch.angular_momentum_count) &&
      reads.add(batch.primitive_exponents, batch.primitive_exponent_count) &&
      reads.add(batch.primitive_coefficients, batch.primitive_coefficient_count) &&
      reads.add(h0.atomic_radii, h0.atomic_radius_count) &&
      reads.add(h0.shell_levels, h0.shell_level_count) &&
      reads.add(h0.shell_coordination_scale, h0.shell_coordination_scale_count) &&
      reads.add(h0.shell_polynomial, h0.shell_polynomial_count) &&
      reads.add(h0.shell_pair_scale, h0.shell_pair_scale_count) &&
      reads.add(periodic.translation_offsets, periodic.translation_offset_elements) &&
      reads.add(periodic.translations, periodic.translation_elements) &&
      reads.add(wrapped_positions, coordinates) &&
      reads.add(coordination_numbers, batch.total_atoms) &&
      writes.add(overlap, batch.total_matrix_elements) &&
      writes.add(dipole, multipoles_enabled ? dipole_elements : 0) &&
      writes.add(quadrupole, multipoles_enabled ? quadrupole_elements : 0) &&
      writes.add(hamiltonian, batch.total_matrix_elements) &&
      writes.add(workspace.overlap_scratch, workspace.overlap_elements) &&
      writes.add(workspace.dipole_scratch, multipoles_enabled ? workspace.dipole_elements : 0) &&
      writes.add(workspace.quadrupole_scratch,
                 multipoles_enabled ? workspace.quadrupole_elements : 0) &&
      writes.add(workspace.h0_scratch, workspace.h0_elements) &&
      writes.add(workspace.sequence_active, workspace.sequence_elements) &&
      writes.add(system_errors, batch.batch_size) && writes.add(device_error, 1);
  if (!ranges_valid || !writes_are_disjoint(reads, writes)) return cudaErrorInvalidValue;

  const std::int64_t clear_elements =
      std::max(batch.total_matrix_elements, std::max(dipole_elements, quadrupole_elements));
  /* Use quotient/remainder ceiling division so a hostile, but otherwise
   * representable, element count cannot overflow the +threads-1 expression
   * before the launch-grid cap is applied. */
  const std::int64_t clear_block_count =
      clear_elements / kNativePeriodicIntegralThreadsPerBlock +
      (clear_elements % kNativePeriodicIntegralThreadsPerBlock == 0 ? 0 : 1);
  const unsigned int clear_blocks = static_cast<unsigned int>(
      std::min<std::int64_t>(kNativePeriodicMaximumGridBlocks, clear_block_count));
  if (clear_blocks == 0u) return cudaErrorInvalidValue;
  native_periodic_integral_clear_kernel<<<clear_blocks, kNativePeriodicIntegralThreadsPerBlock, 0,
                                          stream>>>(
      workspace, device_error, batch.total_matrix_elements, dipole_elements, quadrupole_elements);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) return status;

  /* The kernel now serializes each system's image queue inside its owner CTA.
   * Keep the grid bounded and let a CTA take a system-stride for unusually
   * large batches; this retains deterministic per-system reductions without
   * imposing a one-CTA bottleneck on independent peers. */
  const unsigned int task_blocks = static_cast<unsigned int>(
      std::min<std::int64_t>(kNativePeriodicMaximumGridBlocks, batch.batch_size));
  if (task_blocks == 0u) return cudaErrorInvalidValue;
  native_periodic_integral_kernel<<<task_blocks, kNativePeriodicIntegralThreadsPerBlock, 0,
                                    stream>>>(
      batch, h0, periodic, wrapped_positions, coordination_numbers, workspace, system_errors,
      device_error, total_tasks, image_task_count, image_system_stride, maximum_shell_pair_count);
  status = check_launch();
  if (status != cudaSuccess) return status;

  native_periodic_integral_publish_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                            kNativePeriodicIntegralThreadsPerBlock, 0, stream>>>(
      batch, workspace, overlap, dipole, quadrupole, hamiltonian, system_errors);
  return check_launch();
}

Gfn2PreprocessingBindingDiagnostic validate_gfn2_preprocessing_binding_cuda(
    const Gfn2PreprocessingDeviceBinding& binding) noexcept {
  return validate_structure(binding, true);
}

Gfn2PreprocessingBindingDiagnostic seal_gfn2_preprocessing_binding_cuda(
    Gfn2PreprocessingDeviceBinding& binding) noexcept {
  binding.binding_seal = 0u;
  const BindingDiagnostic diagnostic = validate_structure(binding, false);
  if (!diagnostic.success()) {
    return diagnostic;
  }
  binding.binding_seal = binding_seal(binding);
  return {};
}

namespace {

Gfn2PreprocessingLaunchDiagnostic compose_preprocessing_impl(
    Gfn2PreprocessingDeviceBinding& binding, GeometryGenerationSource generation_source,
    bool advance_epoch, cudaStream_t stream) noexcept {
  const BindingDiagnostic descriptor = validate_structure(binding, true);
  if (!descriptor.success()) {
    return launch_failure(descriptor, cudaErrorInvalidValue);
  }
  const bool epoch_enabled = binding.geometry_epoch.value != nullptr &&
                             binding.geometry_epoch.value_elements == 1 &&
                             binding.geometry_epoch.plan_token == binding.plan_token;
  if ((!advance_epoch && (generation_source.scalar == 0u || epoch_enabled)) ||
      (advance_epoch &&
       (!epoch_enabled || generation_source.device != binding.geometry_epoch.value))) {
    return launch_failure(
        binding_failure(
            advance_epoch ? BindingError::kInvalidEpoch : BindingError::kInvalidGeneration,
            advance_epoch ? BindingField::kEpoch : BindingField::kGeneration),
        cudaErrorInvalidValue);
  }
  const std::uint64_t primitive_generation =
      advance_epoch ? kUnpublishedPrimitiveGeneration : generation_source.scalar;

  const std::int64_t batch = binding.plan.geometry.batch_size;
  const bool multipoles_enabled = binding.plan.geometry.model == XtbModelFlavor::kGfn2;
  const bool aes2_enabled = multipoles_enabled;
  cudaError_t status =
      reset_gfn2_geometry_device_errors_cuda(batch, binding.diagnostics.geometry_system_errors,
                                             binding.diagnostics.geometry_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  status =
      reset_gfn2_integral_device_errors_cuda(batch, binding.diagnostics.integral_system_errors,
                                             binding.diagnostics.integral_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  status = reset_gfn2_es2_device_error_cuda(binding.diagnostics.es2_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  if (aes2_enabled) {
    status = reset_gfn2_aes2_device_errors_cuda(batch, binding.diagnostics.aes2_system_errors,
                                                binding.diagnostics.aes2_device_error, stream);
    if (status != cudaSuccess) return launch_failure({}, status);
  }
  status = cudaMemsetAsync(binding.diagnostics.system_stages, 0,
                           static_cast<std::size_t>(batch) * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  status = cudaMemsetAsync(binding.diagnostics.plan_error, 0, sizeof(std::uint32_t), stream);
  if (status != cudaSuccess) return launch_failure({}, status);
  if (advance_epoch) {
    advance_geometry_epoch_kernel<<<1, 1, 0, stream>>>(binding.geometry_epoch, binding.admission,
                                                       binding.diagnostics.plan_error);
    status = check_launch();
    if (status != cudaSuccess) return launch_failure({}, status);
  }

  prepare_positions_kernel<<<static_cast<unsigned int>(batch), kThreadsPerBlock, 0, stream>>>(
      binding.plan.geometry, binding.input.positions, binding.activity.requested_mask,
      binding.workspace.positions_scratch, binding.diagnostics.plan_error);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);
  const unsigned int gate_blocks = static_cast<unsigned int>(
      (static_cast<std::uint64_t>(batch) + kThreadsPerBlock - 1u) / kThreadsPerBlock);
  gate_composer_plan_kernel<<<gate_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, binding.diagnostics.plan_error, binding.diagnostics.geometry_system_errors,
      binding.diagnostics.geometry_device_error, binding.diagnostics.integral_system_errors,
      binding.diagnostics.integral_device_error, binding.diagnostics.es2_device_error,
      binding.diagnostics.aes2_system_errors, binding.diagnostics.aes2_device_error, aes2_enabled);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);

  status = update_gfn2_geometry_cache_cuda(
      binding.plan.geometry, binding.workspace.positions_scratch, primitive_generation,
      binding.workspace.geometry_candidate, binding.workspace.geometry,
      binding.diagnostics.geometry_system_errors, binding.diagnostics.geometry_device_error,
      stream);
  if (status != cudaSuccess) return launch_failure({}, status);

  /* Optional sparse pair-list consistency gate.  The host scheduler enables
   * plan.pairlist when the batch crosses the measured dense-fallback crossover;
   * the bucketed builder then reproduces coordination numbers that must match
   * the dense geometry cache bitwise.  A mismatch fails that peer closed, so a
   * sparse/dense regression can never silently publish different physics. */
  if (binding.plan.pairlist.batch_size > 0) {
    status =
        reset_gfn2_pairlist_device_errors_cuda(batch, binding.diagnostics.sparse_system_errors,
                                               binding.diagnostics.sparse_device_error, stream);
    if (status != cudaSuccess) return launch_failure({}, status);
    status = update_gfn2_pairlist_cache_cuda(
        binding.plan.pairlist, binding.workspace.positions_scratch, primitive_generation,
        binding.workspace.pairlist_candidate, binding.workspace.pairlist,
        binding.diagnostics.sparse_system_errors, binding.diagnostics.sparse_device_error, stream);
    if (status != cudaSuccess) return launch_failure({}, status);
    status = evaluate_gfn2_pairlist_coordination_cuda(
        binding.plan.pairlist, binding.workspace.positions_scratch,
        binding.plan.geometry.covalent_radii, primitive_generation,
        binding.workspace.pairlist_candidate, binding.workspace.sparse_coordination,
        binding.workspace.pairlist, binding.diagnostics.sparse_system_errors,
        binding.diagnostics.sparse_device_error, stream);
    if (status != cudaSuccess) return launch_failure({}, status);
    const unsigned int sparse_blocks = static_cast<unsigned int>(batch);
    gate_sparse_coordination_kernel<<<sparse_blocks, kThreadsPerBlock, 0, stream>>>(
        binding.plan.geometry, binding.plan.pairlist,
        binding.workspace.geometry_candidate.coordination_numbers,
        binding.workspace.sparse_coordination, binding.workspace.pairlist.sequence_active,
        binding.activity, binding.diagnostics);
    status = check_launch();
    if (status != cudaSuccess) return launch_failure({}, status);
    /* After the bitwise gate passed, the sparse list is the authoritative CN
     * producer: publish its values into the geometry candidate so H0, AES2,
     * and the public cache consume sparse-produced CN.  The dense path remains
     * the differential reference behind the gate. */
    promote_sparse_coordination_kernel<<<sparse_blocks, kThreadsPerBlock, 0, stream>>>(
        binding.plan.geometry, binding.plan.pairlist, binding.workspace.sparse_coordination,
        binding.workspace.geometry_candidate.coordination_numbers, binding.activity,
        binding.diagnostics);
    status = check_launch();
    if (status != cudaSuccess) return launch_failure({}, status);
  }

  /* Native XYZ requests use the complete 25-bohr image topology for
   * coordination.  Run this leaf after the molecular/sparse geometry pass
   * but before integrals/H0, so the periodic CN is the value consumed by the
   * Hamiltonian and AES2 assembly.  Native diagnostics share the geometry
   * stage: a peer-local short-range failure therefore closes only that peer,
   * while the existing publication gate remains transactional. */
  if (binding.plan.native_short_range.topology.plan_token != 0u) {
    Gfn2NativePeriodicShortRangeDeviceBatch native = binding.plan.native_short_range;
    native.positions = binding.input.positions;
    native.position_elements = binding.input.position_elements;
    status = evaluate_gfn2_native_periodic_short_range_cuda(
        native, binding.workspace.native_short_range,
        binding.workspace.geometry_candidate.coordination_numbers,
        binding.workspace.native_short_range.repulsion_energies,
        binding.workspace.native_short_range.repulsion_gradients,
        binding.workspace.native_short_range.repulsion_strain,
        binding.diagnostics.geometry_system_errors, binding.diagnostics.geometry_device_error,
        stream);
    if (status != cudaSuccess) return launch_failure({}, status);
  }
  /* A native short-range peer failure invalidates its coordination candidate.
   * Fold that peer-local geometry status into the integral domain before either
   * one-electron implementation consumes the candidate. */
  gate_h0_kernel<<<gate_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, binding.activity.requested_mask, binding.diagnostics.geometry_system_errors,
      binding.diagnostics.integral_system_errors);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);
  const bool native_periodic_enabled = binding.plan.native_short_range.topology.plan_token != 0u;
  if (native_periodic_enabled) {
    /* Native XYZ owns the complete one-electron image traversal.  It consumes
     * the wrapped positions produced by the periodic short-range stage and
     * writes the same unpublished S/D/Q/H0 candidate slices used by the
     * molecular publication gate.  Do not run the molecular evaluator after
     * this branch: its central-cell-only H0 would overwrite the periodic
     * result. */
    status = evaluate_gfn2_native_periodic_integrals_h0_cuda(
        binding.plan.integrals, binding.plan.h0, binding.plan.native_integrals,
        binding.workspace.native_short_range.wrapped_positions,
        binding.workspace.geometry_candidate.coordination_numbers,
        binding.workspace.overlap_candidate, binding.workspace.dipole_candidate,
        binding.workspace.quadrupole_candidate, binding.workspace.h0_candidate,
        binding.workspace.integrals, binding.diagnostics.integral_system_errors,
        binding.diagnostics.integral_device_error, stream);
  } else {
    status = evaluate_gfn2_integrals_cuda(
        binding.plan.integrals, binding.workspace.positions_scratch,
        binding.workspace.overlap_candidate, binding.workspace.dipole_candidate,
        binding.workspace.quadrupole_candidate, binding.workspace.integrals,
        binding.diagnostics.integral_system_errors, binding.diagnostics.integral_device_error,
        stream);
    if (status == cudaSuccess) {
      prepare_h0_sequence_kernel<<<1, 1, 0, stream>>>(binding.workspace.integrals.sequence_active,
                                                      binding.diagnostics.integral_device_error);
      status = check_launch();
    }
    if (status == cudaSuccess) {
      status = evaluate_gfn2_h0_cuda(
          binding.plan.integrals, binding.plan.h0, binding.workspace.positions_scratch,
          binding.workspace.geometry_candidate.coordination_numbers,
          binding.workspace.overlap_candidate, binding.workspace.h0_candidate,
          binding.workspace.integrals, binding.diagnostics.integral_system_errors,
          binding.diagnostics.integral_device_error, stream);
    }
  }
  if (status != cudaSuccess) return launch_failure({}, status);

  prepare_late_stages_kernel<<<static_cast<unsigned int>(batch), kThreadsPerBlock, 0, stream>>>(
      binding.plan.geometry, binding.activity.requested_mask,
      binding.diagnostics.geometry_system_errors, binding.diagnostics.integral_system_errors,
      binding.workspace.positions_scratch, binding.diagnostics.aes2_system_errors,
      binding.diagnostics.plan_error, aes2_enabled);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);

  Gfn2ES2DeviceCache es2_candidate = binding.workspace.es2_candidate;
  es2_candidate.geometry_generation = primitive_generation;
  status = update_gfn2_es2_geometry_cache_cuda(
      binding.plan.es2, binding.workspace.positions_scratch, es2_candidate, binding.workspace.es2,
      binding.diagnostics.es2_device_error, stream);
  if (status != cudaSuccess) return launch_failure({}, status);

  if (aes2_enabled) {
    Gfn2AES2DeviceCache aes2_candidate = binding.workspace.aes2_candidate;
    aes2_candidate.geometry_generation = primitive_generation;
    status = update_gfn2_aes2_geometry_cache_cuda(
        binding.plan.aes2, binding.workspace.positions_scratch,
        binding.workspace.geometry_candidate.coordination_numbers, aes2_candidate,
        binding.workspace.aes2, binding.diagnostics.aes2_system_errors,
        binding.diagnostics.aes2_device_error, stream);
    if (status != cudaSuccess) return launch_failure({}, status);
  }

  classify_plan_kernel<<<1, 1, 0, stream>>>(
      batch, binding.workspace.geometry.sequence_active,
      binding.plan.pairlist.batch_size > 0 ? binding.workspace.pairlist.sequence_active : nullptr,
      binding.workspace.integrals.sequence_active, binding.diagnostics.geometry_system_errors,
      binding.diagnostics.integral_system_errors, binding.diagnostics.es2_device_error,
      binding.diagnostics.aes2_system_errors, binding.diagnostics.plan_error, aes2_enabled);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);
  publish_preprocessing_kernel<<<static_cast<unsigned int>(batch), kThreadsPerBlock, 0, stream>>>(
      binding.plan, binding.activity, binding.output, binding.workspace, binding.diagnostics,
      binding.admission, generation_source);
  status = check_launch();
  if (status != cudaSuccess) return launch_failure({}, status);

  /* Committed pair-list transaction: initialize fixed-capacity slot metadata
   * only after plan validation, then publish each healthy peer into its own
   * stable slot with the current generation and eligibility. */
  if (binding.plan.pairlist.batch_size > 0) {
    const std::int64_t metadata_elements =
        std::max<std::int64_t>(batch + 1, binding.plan.pairlist.total_atoms + 1);
    initialize_committed_pairlist_metadata_kernel<<<
        static_cast<unsigned int>((metadata_elements + kThreadsPerBlock - 1) / kThreadsPerBlock),
        kThreadsPerBlock, 0, stream>>>(binding.plan.pairlist, binding.output.pairlist,
                                       binding.admission, binding.diagnostics.plan_error);
    status = check_launch();
    if (status != cudaSuccess) return launch_failure({}, status);
    commit_pairlist_kernel<<<static_cast<unsigned int>(batch), kThreadsPerBlock, 0, stream>>>(
        binding.plan.pairlist, binding.workspace.pairlist_candidate, binding.output.pairlist,
        generation_source, binding.activity, binding.admission, binding.diagnostics.plan_error);
    status = check_launch();
    if (status != cudaSuccess) return launch_failure({}, status);
  }

  /* Only the legacy path can update host-side descriptor scalars. In either
   * mode, device-resident per-system generations plus published_mask are the
   * authoritative record that a peer's complete public cache was committed. */
  if (!advance_epoch) {
    binding.output.es2.geometry_generation = generation_source.scalar;
    binding.workspace.es2_candidate.geometry_generation = generation_source.scalar;
    if (aes2_enabled) {
      binding.output.aes2.geometry_generation = generation_source.scalar;
      binding.workspace.aes2_candidate.geometry_generation = generation_source.scalar;
    }
  }
  return {};
}

}  // namespace

Gfn2PreprocessingLaunchDiagnostic compose_gfn2_preprocessing_cuda(
    Gfn2PreprocessingDeviceBinding& binding, std::uint64_t geometry_generation,
    cudaStream_t stream) noexcept {
  return compose_preprocessing_impl(binding, {geometry_generation, nullptr}, false, stream);
}

Gfn2PreprocessingLaunchDiagnostic compose_gfn2_preprocessing_epoch_cuda(
    Gfn2PreprocessingDeviceBinding& binding, cudaStream_t stream) noexcept {
  return compose_preprocessing_impl(binding, {0u, binding.geometry_epoch.value}, true, stream);
}

Gfn2PreprocessingLaunchDiagnostic gate_gfn2_sparse_coordination_cuda(
    Gfn2PreprocessingDeviceBinding& binding, cudaStream_t stream) noexcept {
  const BindingDiagnostic descriptor = validate_structure(binding, true);
  if (descriptor.error != BindingError::kSuccess) {
    return launch_failure(descriptor, cudaErrorInvalidValue);
  }
  if (binding.plan.pairlist.batch_size <= 0) {
    return launch_failure(binding_failure(BindingError::kInvalidWorkspace, BindingField::kPairlist),
                          cudaErrorInvalidValue);
  }
  const std::int64_t batch = binding.plan.geometry.batch_size;
  const unsigned int blocks = static_cast<unsigned int>(batch);
  gate_sparse_coordination_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      binding.plan.geometry, binding.plan.pairlist,
      binding.workspace.geometry_candidate.coordination_numbers,
      binding.workspace.sparse_coordination, binding.workspace.pairlist.sequence_active,
      binding.activity, binding.diagnostics);
  return {binding_failure(BindingError::kSuccess, BindingField::kNone), check_launch()};
}

}  // namespace xtbloom::detail::cuda
