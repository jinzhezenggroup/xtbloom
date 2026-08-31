// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "backends/cuda/periodic_topology.cuh"
#include "model/gfn2/lattice.hpp"

namespace xtbloom::detail::cuda {
namespace {

using DeviceError = Gfn2CudaPeriodicTopologyError;
using DeviceField = Gfn2CudaPeriodicTopologyField;
using Diagnostic = Gfn2CudaPeriodicTopologyDiagnostic;

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::int64_t kInt64Maximum = std::numeric_limits<std::int64_t>::max();

Diagnostic failure(xtbloom_status_t status, DeviceError error, DeviceField field,
                   std::int64_t index = -1, cudaError_t cuda_status = cudaSuccess) noexcept {
  Diagnostic result{};
  result.status = status;
  result.error = error;
  result.field = field;
  result.index = index;
  result.cuda_status = static_cast<std::int32_t>(cuda_status);
  return result;
}

Diagnostic cuda_failure(DeviceField field, cudaError_t status) noexcept {
  const xtbloom_status_t public_status = status == cudaErrorMemoryAllocation
                                             ? XTBLOOM_STATUS_ALLOCATION_FAILED
                                             : XTBLOOM_STATUS_INTERNAL_ERROR;
  return failure(public_status,
                 status == cudaErrorMemoryAllocation ? DeviceError::kAllocationFailed
                                                     : DeviceError::kCudaFailure,
                 field, -1, status);
}

bool checked_add(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first > std::numeric_limits<std::size_t>::max() - second) return false;
  result = first + second;
  return true;
}

bool checked_multiply(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) return false;
  result = first * second;
  return true;
}

bool to_size(std::int64_t value, std::size_t& result) noexcept {
  if (value < 0 || static_cast<std::uint64_t>(value) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  result = static_cast<std::size_t>(value);
  return true;
}

double canonical_zero(double value) noexcept { return value == 0.0 ? 0.0 : value; }

template <typename T>
void hash_scalar(std::uint64_t& hash, const T& value) noexcept {
  unsigned char bytes[sizeof(T)]{};
  std::memcpy(bytes, &value, sizeof(T));
  for (unsigned char byte : bytes) {
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= kFnvPrime;
  }
}

void hash_double(std::uint64_t& hash, double value) noexcept {
  hash_scalar(hash, canonical_zero(value));
}

std::uint64_t topology_hash(const Gfn2CudaPeriodicTopology& topology) noexcept {
  std::uint64_t hash = kFnvOffset;
  hash_scalar(hash, topology.host_atom_offsets().size());
  for (const std::int64_t value : topology.host_atom_offsets()) hash_scalar(hash, value);
  hash_scalar(hash, topology.host_periodic_axes().size());
  for (const std::int32_t value : topology.host_periodic_axes()) hash_scalar(hash, value);
  hash_scalar(hash, topology.host_cell_matrices().size());
  for (const double value : topology.host_cell_matrices()) hash_double(hash, value);
  hash_scalar(hash, topology.host_translation_offsets().size());
  for (const std::int64_t value : topology.host_translation_offsets()) hash_scalar(hash, value);
  hash_scalar(hash, topology.host_translations().size());
  for (const Gfn2CudaPeriodicTranslation& translation : topology.host_translations()) {
    for (const std::int64_t value : translation.index) hash_scalar(hash, value);
    for (const double value : translation.cartesian) hash_double(hash, value);
  }
  hash_double(hash, topology.image_cutoff());
  return hash == 0u ? 1u : hash;
}

__host__ __device__ bool all_zero(const double* values, std::size_t count) noexcept {
  for (std::size_t index = 0; index < count; ++index) {
    if (values[index] != 0.0) return false;
  }
  return true;
}

Gfn2CudaPeriodicTranslation origin_translation() noexcept { return {}; }

template <typename T>
cudaError_t upload_array(T*& destination, const std::vector<T>& source,
                         cudaStream_t stream) noexcept {
  const std::size_t bytes = source.size() * sizeof(T);
  if (bytes == 0u) {
    destination = nullptr;
    return cudaSuccess;
  }
  cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&destination), bytes);
  if (status != cudaSuccess) return status;
  status = cudaMemcpyAsync(destination, source.data(), bytes, cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) {
    (void)cudaFree(destination);
    destination = nullptr;
  }
  return status;
}

__device__ void device_failure(Gfn2CudaPeriodicTopologyDeviceDiagnostic* diagnostic,
                               DeviceError error, DeviceField field, std::int64_t index) {
  diagnostic->error = static_cast<std::uint32_t>(error);
  diagnostic->field = static_cast<std::uint32_t>(field);
  diagnostic->index = index;
}

__device__ bool device_is_origin(const Gfn2CudaPeriodicTranslation& translation) {
  return translation.index[0] == 0 && translation.index[1] == 0 && translation.index[2] == 0 &&
         translation.cartesian[0] == 0.0 && translation.cartesian[1] == 0.0 &&
         translation.cartesian[2] == 0.0;
}

__global__ void validate_periodic_topology_kernel(
    Gfn2CudaPeriodicTopologyView view, Gfn2CudaPeriodicTopologyDeviceDiagnostic* diagnostic) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *diagnostic = {};
  diagnostic->index = -1;

  if (view.batch_size <= 0 || view.total_atoms < 0 || view.total_translations <= 0 ||
      view.batch_size == kInt64Maximum || view.atom_offset_count != view.batch_size + 1 ||
      view.translation_offset_count != view.batch_size + 1 || view.batch_size > kInt64Maximum / 9 ||
      view.cell_elements != view.batch_size * 9 || view.periodic_axes_elements != view.batch_size ||
      view.plan_token == 0u || view.cell_generation == 0u) {
    device_failure(diagnostic, DeviceError::kInvalidDimensions, DeviceField::kDimensions, -1);
    return;
  }
  if (view.atom_offsets == nullptr || view.cell_matrices == nullptr ||
      view.periodic_axes == nullptr || view.translation_offsets == nullptr ||
      view.translations == nullptr) {
    device_failure(diagnostic, DeviceError::kInvalidDimensions, DeviceField::kDimensions, -1);
    return;
  }

  if (view.atom_offsets[0] != 0 || view.atom_offsets[view.batch_size] != view.total_atoms) {
    device_failure(diagnostic, DeviceError::kInvalidOffsets, DeviceField::kAtomOffsets, 0);
    return;
  }
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    if (view.atom_offsets[system] < 0 ||
        view.atom_offsets[system] > view.atom_offsets[system + 1] ||
        view.atom_offsets[system + 1] > view.total_atoms) {
      device_failure(diagnostic, DeviceError::kInvalidOffsets, DeviceField::kAtomOffsets, system);
      return;
    }
  }
  if (view.translation_offsets[0] != 0 ||
      view.translation_offsets[view.batch_size] != view.total_translations) {
    device_failure(diagnostic, DeviceError::kInvalidOffsets, DeviceField::kTranslationOffsets, 0);
    return;
  }
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    if (view.translation_offsets[system] < 0 ||
        view.translation_offsets[system] > view.translation_offsets[system + 1] ||
        view.translation_offsets[system + 1] > view.total_translations) {
      device_failure(diagnostic, DeviceError::kInvalidOffsets, DeviceField::kTranslationOffsets,
                     system);
      return;
    }
    const std::int32_t mask = view.periodic_axes[system];
    const double* const cell = view.cell_matrices + system * 9;
    const std::int64_t translation_begin = view.translation_offsets[system];
    const std::int64_t translation_end = view.translation_offsets[system + 1];
    if (mask == XTBLOOM_PERIODIC_AXES_NONE) {
      if (!all_zero(cell, 9u)) {
        device_failure(diagnostic, DeviceError::kInvalidPeriodicity, DeviceField::kCellMatrices,
                       system);
        return;
      }
      if (translation_end - translation_begin != 1 ||
          !device_is_origin(view.translations[translation_begin])) {
        device_failure(diagnostic, DeviceError::kInvalidTranslations, DeviceField::kTranslations,
                       system);
        return;
      }
    } else if ((mask & ~XTBLOOM_PERIODIC_AXES_XYZ) != 0) {
      device_failure(diagnostic, DeviceError::kInvalidPeriodicity, DeviceField::kPeriodicAxes,
                     system);
      return;
    } else if (mask != XTBLOOM_PERIODIC_AXES_XYZ) {
      device_failure(diagnostic, DeviceError::kUnsupportedPeriodicity, DeviceField::kPeriodicAxes,
                     system);
      return;
    } else if (!gfn2::valid_lattice_cell_3d_binary64(cell)) {
      device_failure(diagnostic, DeviceError::kInvalidCell, DeviceField::kCellMatrices, system);
      return;
    } else if (translation_end <= translation_begin ||
               !device_is_origin(view.translations[translation_begin])) {
      device_failure(diagnostic, DeviceError::kInvalidTranslations, DeviceField::kTranslations,
                     system);
      return;
    }
    for (std::int64_t translation = translation_begin; translation < translation_end;
         ++translation) {
      const Gfn2CudaPeriodicTranslation& value = view.translations[translation];
      for (int component = 0; component < 3; ++component) {
        if (!gfn2::lattice_binary64_detail::finite(value.cartesian[component])) {
          device_failure(diagnostic, DeviceError::kNonfiniteTranslation, DeviceField::kTranslations,
                         translation);
          return;
        }
      }
    }
  }
}

}  // namespace

bool same_gfn2_cuda_periodic_graph_identity(const Gfn2CudaPeriodicGraphIdentity& first,
                                            const Gfn2CudaPeriodicGraphIdentity& second) noexcept {
  return first.plan_token == second.plan_token &&
         first.topology_fingerprint == second.topology_fingerprint &&
         first.cell_generation == second.cell_generation &&
         first.geometry_generation == second.geometry_generation &&
         first.requested_result_flags == second.requested_result_flags &&
         first.unsupported_interaction_mask == second.unsupported_interaction_mask &&
         first.strain_requested == second.strain_requested;
}

Gfn2CudaPeriodicTopology::~Gfn2CudaPeriodicTopology() { release_device(); }

Gfn2CudaPeriodicTopology::Gfn2CudaPeriodicTopology(Gfn2CudaPeriodicTopology&& other) noexcept {
  move_from(std::move(other));
}

Gfn2CudaPeriodicTopology& Gfn2CudaPeriodicTopology::operator=(
    Gfn2CudaPeriodicTopology&& other) noexcept {
  if (this != &other) {
    release_device();
    move_from(std::move(other));
  }
  return *this;
}

void Gfn2CudaPeriodicTopology::release_device() noexcept {
  if (device_atom_offsets_ != nullptr) (void)cudaFree(device_atom_offsets_);
  if (device_cell_matrices_ != nullptr) (void)cudaFree(device_cell_matrices_);
  if (device_periodic_axes_ != nullptr) (void)cudaFree(device_periodic_axes_);
  if (device_translation_offsets_ != nullptr) (void)cudaFree(device_translation_offsets_);
  if (device_translations_ != nullptr) (void)cudaFree(device_translations_);
  device_atom_offsets_ = nullptr;
  device_cell_matrices_ = nullptr;
  device_periodic_axes_ = nullptr;
  device_translation_offsets_ = nullptr;
  device_translations_ = nullptr;
}

void Gfn2CudaPeriodicTopology::move_from(Gfn2CudaPeriodicTopology&& other) noexcept {
  device_id_ = std::exchange(other.device_id_, -1);
  batch_size_ = std::exchange(other.batch_size_, 0);
  total_atoms_ = std::exchange(other.total_atoms_, 0);
  image_cutoff_ = std::exchange(other.image_cutoff_, 0.0);
  plan_token_ = std::exchange(other.plan_token_, 0u);
  cell_generation_ = std::exchange(other.cell_generation_, 0u);
  topology_fingerprint_ = std::exchange(other.topology_fingerprint_, 0u);
  atom_offsets_ = std::move(other.atom_offsets_);
  cell_matrices_ = std::move(other.cell_matrices_);
  periodic_axes_ = std::move(other.periodic_axes_);
  translation_offsets_ = std::move(other.translation_offsets_);
  translations_ = std::move(other.translations_);
  device_atom_offsets_ = std::exchange(other.device_atom_offsets_, nullptr);
  device_cell_matrices_ = std::exchange(other.device_cell_matrices_, nullptr);
  device_periodic_axes_ = std::exchange(other.device_periodic_axes_, nullptr);
  device_translation_offsets_ = std::exchange(other.device_translation_offsets_, nullptr);
  device_translations_ = std::exchange(other.device_translations_, nullptr);
}

Gfn2CudaPeriodicTopologyDiagnostic Gfn2CudaPeriodicTopology::create(
    const Gfn2CudaPeriodicTopologyInput& input, cudaStream_t stream,
    Gfn2CudaPeriodicTopology& output) noexcept {
  if (input.batch_size <= 0 || input.batch_size == kInt64Maximum || input.total_atoms < 0 ||
      input.atom_offsets == nullptr || input.cell_matrices == nullptr ||
      input.periodic_axes == nullptr) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidDimensions,
                   DeviceField::kDimensions);
  }
  if (input.plan_token == 0u || input.cell_generation == 0u) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidPlanIdentity,
                   DeviceField::kPlanIdentity);
  }
  if (!(input.image_cutoff >= 0.0) || !std::isfinite(input.image_cutoff)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidDimensions,
                   DeviceField::kDimensions);
  }

  std::size_t batch_count = 0u;
  std::size_t atom_count = 0u;
  std::size_t offset_count = 0u;
  std::size_t cell_count = 0u;
  if (!to_size(input.batch_size, batch_count) || !to_size(input.total_atoms, atom_count) ||
      !checked_add(batch_count, 1u, offset_count) ||
      !checked_multiply(batch_count, 9u, cell_count) || input.batch_size > kInt64Maximum - 1 ||
      batch_count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      atom_count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      cell_count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidDimensions,
                   DeviceField::kDimensions);
  }
  if (input.atom_offsets[0] != 0 || input.atom_offsets[input.batch_size] != input.total_atoms) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidOffsets,
                   DeviceField::kAtomOffsets, 0);
  }

  try {
    Gfn2CudaPeriodicTopology candidate;
    candidate.batch_size_ = input.batch_size;
    candidate.total_atoms_ = input.total_atoms;
    candidate.image_cutoff_ = input.image_cutoff;
    candidate.plan_token_ = input.plan_token;
    candidate.cell_generation_ = input.cell_generation;
    candidate.atom_offsets_.assign(input.atom_offsets, input.atom_offsets + offset_count);
    candidate.cell_matrices_.resize(cell_count);
    candidate.periodic_axes_.assign(input.periodic_axes, input.periodic_axes + batch_count);
    candidate.translation_offsets_.assign(offset_count, 0);

    for (std::int64_t system = 0; system < input.batch_size; ++system) {
      const std::int64_t begin = candidate.atom_offsets_[static_cast<std::size_t>(system)];
      const std::int64_t end = candidate.atom_offsets_[static_cast<std::size_t>(system + 1)];
      if (begin < 0 || begin > end || end > input.total_atoms) {
        return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidOffsets,
                       DeviceField::kAtomOffsets, system);
      }
      std::array<double, 9> direct{};
      for (std::size_t element = 0; element < direct.size(); ++element) {
        direct[element] = canonical_zero(
            input.cell_matrices[static_cast<std::size_t>(system) * direct.size() + element]);
        candidate.cell_matrices_[static_cast<std::size_t>(system) * direct.size() + element] =
            direct[element];
      }

      const std::int32_t mask = candidate.periodic_axes_[static_cast<std::size_t>(system)];
      std::vector<gfn2::LatticeTranslation> local;
      if (mask == XTBLOOM_PERIODIC_AXES_NONE) {
        if (!all_zero(direct.data(), direct.size())) {
          return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidPeriodicity,
                         DeviceField::kCellMatrices, system);
        }
        candidate.translations_.push_back(origin_translation());
      } else {
        if ((mask & ~XTBLOOM_PERIODIC_AXES_XYZ) != 0) {
          return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidPeriodicity,
                         DeviceField::kPeriodicAxes, system);
        }
        if (mask != XTBLOOM_PERIODIC_AXES_XYZ) {
          return failure(XTBLOOM_STATUS_NOT_SUPPORTED, DeviceError::kUnsupportedPeriodicity,
                         DeviceField::kPeriodicAxes, system);
        }
        gfn2::Lattice3D lattice;
        std::string lattice_error;
        xtbloom_status_t status = gfn2::make_lattice_3d(direct.data(), lattice, lattice_error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return failure(status, DeviceError::kInvalidCell, DeviceField::kCellMatrices, system);
        }
        status = gfn2::make_lattice_translations(
            lattice, input.image_cutoff, gfn2::LatticeOriginPolicy::kInclude, local, lattice_error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return failure(status, DeviceError::kInvalidTranslations, DeviceField::kTranslations,
                         system);
        }
        if (local.size() >
                std::numeric_limits<std::size_t>::max() - candidate.translations_.size() ||
            candidate.translations_.size() + local.size() >
                static_cast<std::size_t>(kInt64Maximum)) {
          return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidTranslations,
                         DeviceField::kTranslations, system);
        }
        candidate.translations_.reserve(candidate.translations_.size() + local.size());
        for (const gfn2::LatticeTranslation& translation : local) {
          Gfn2CudaPeriodicTranslation value{};
          for (int component = 0; component < 3; ++component) {
            value.index[component] = translation.index[component];
            value.cartesian[component] = canonical_zero(translation.cartesian[component]);
          }
          candidate.translations_.push_back(value);
        }
      }
      if (candidate.translations_.size() > static_cast<std::size_t>(kInt64Maximum)) {
        return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidTranslations,
                       DeviceField::kTranslations, system);
      }
      candidate.translation_offsets_[static_cast<std::size_t>(system + 1)] =
          static_cast<std::int64_t>(candidate.translations_.size());
    }
    if (candidate.translations_.empty() ||
        candidate.translations_.size() > static_cast<std::size_t>(kInt64Maximum)) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidTranslations,
                     DeviceField::kTranslations);
    }
    candidate.topology_fingerprint_ = topology_hash(candidate);

    cudaError_t cuda_status = cudaGetDevice(&candidate.device_id_);
    if (cuda_status != cudaSuccess) return cuda_failure(DeviceField::kPlanIdentity, cuda_status);
    cuda_status = upload_array(candidate.device_atom_offsets_, candidate.atom_offsets_, stream);
    if (cuda_status != cudaSuccess) return cuda_failure(DeviceField::kAtomOffsets, cuda_status);
    cuda_status = upload_array(candidate.device_cell_matrices_, candidate.cell_matrices_, stream);
    if (cuda_status != cudaSuccess) return cuda_failure(DeviceField::kCellMatrices, cuda_status);
    cuda_status = upload_array(candidate.device_periodic_axes_, candidate.periodic_axes_, stream);
    if (cuda_status != cudaSuccess) return cuda_failure(DeviceField::kPeriodicAxes, cuda_status);
    cuda_status =
        upload_array(candidate.device_translation_offsets_, candidate.translation_offsets_, stream);
    if (cuda_status != cudaSuccess) {
      return cuda_failure(DeviceField::kTranslationOffsets, cuda_status);
    }
    cuda_status = upload_array(candidate.device_translations_, candidate.translations_, stream);
    if (cuda_status != cudaSuccess) return cuda_failure(DeviceField::kTranslations, cuda_status);

    output = std::move(candidate);
    return {};
  } catch (const std::bad_alloc&) {
    return failure(XTBLOOM_STATUS_ALLOCATION_FAILED, DeviceError::kAllocationFailed,
                   DeviceField::kDimensions);
  } catch (const std::length_error&) {
    return failure(XTBLOOM_STATUS_ALLOCATION_FAILED, DeviceError::kAllocationFailed,
                   DeviceField::kDimensions);
  }
}

Gfn2CudaPeriodicTopologyDiagnostic Gfn2CudaPeriodicTopology::validate(
    const Gfn2CudaPeriodicTopologyView& view,
    Gfn2CudaPeriodicTopologyDeviceDiagnostic* device_diagnostic, cudaStream_t stream) noexcept {
  if (device_diagnostic == nullptr || view.batch_size <= 0 || view.total_atoms < 0 ||
      view.total_translations <= 0 || view.batch_size == kInt64Maximum ||
      view.atom_offset_count != view.batch_size + 1 ||
      view.translation_offset_count != view.batch_size + 1 || view.batch_size > kInt64Maximum / 9 ||
      view.cell_elements != view.batch_size * 9 || view.periodic_axes_elements != view.batch_size ||
      view.plan_token == 0u || view.cell_generation == 0u || view.atom_offsets == nullptr ||
      view.cell_matrices == nullptr || view.periodic_axes == nullptr ||
      view.translation_offsets == nullptr || view.translations == nullptr ||
      reinterpret_cast<std::uintptr_t>(device_diagnostic) %
              alignof(Gfn2CudaPeriodicTopologyDeviceDiagnostic) !=
          0u) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, DeviceError::kInvalidDimensions,
                   DeviceField::kDimensions);
  }
  validate_periodic_topology_kernel<<<1, 1, 0, stream>>>(view, device_diagnostic);
  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) return cuda_failure(DeviceField::kPlanIdentity, launch_status);
  return {};
}

bool Gfn2CudaPeriodicTopology::valid() const noexcept {
  return device_id_ >= 0 && batch_size_ > 0 && total_atoms_ >= 0 && plan_token_ != 0u &&
         cell_generation_ != 0u && !atom_offsets_.empty() && !cell_matrices_.empty() &&
         !periodic_axes_.empty() && !translation_offsets_.empty() && !translations_.empty() &&
         device_atom_offsets_ != nullptr && device_cell_matrices_ != nullptr &&
         device_periodic_axes_ != nullptr && device_translation_offsets_ != nullptr &&
         device_translations_ != nullptr;
}

Gfn2CudaPeriodicTopologyView Gfn2CudaPeriodicTopology::device_view() const noexcept {
  Gfn2CudaPeriodicTopologyView view{};
  view.batch_size = batch_size_;
  view.total_atoms = total_atoms_;
  view.total_translations = static_cast<std::int64_t>(translations_.size());
  view.atom_offset_count = static_cast<std::int64_t>(atom_offsets_.size());
  view.cell_elements = static_cast<std::int64_t>(cell_matrices_.size());
  view.periodic_axes_elements = static_cast<std::int64_t>(periodic_axes_.size());
  view.translation_offset_count = static_cast<std::int64_t>(translation_offsets_.size());
  view.plan_token = plan_token_;
  view.cell_generation = cell_generation_;
  view.atom_offsets = device_atom_offsets_;
  view.cell_matrices = device_cell_matrices_;
  view.periodic_axes = device_periodic_axes_;
  view.translation_offsets = device_translation_offsets_;
  view.translations = device_translations_;
  return view;
}

Gfn2CudaPeriodicGraphIdentity Gfn2CudaPeriodicTopology::graph_identity(
    std::uint64_t geometry_generation, std::uint32_t requested_result_flags, bool strain_requested,
    std::uint32_t unsupported_interaction_mask) const noexcept {
  Gfn2CudaPeriodicGraphIdentity identity{};
  identity.plan_token = plan_token_;
  identity.topology_fingerprint = topology_fingerprint_;
  identity.cell_generation = cell_generation_;
  identity.geometry_generation = geometry_generation;
  identity.requested_result_flags = requested_result_flags;
  identity.unsupported_interaction_mask = unsupported_interaction_mask;
  identity.strain_requested = strain_requested ? 1u : 0u;
  return identity;
}

}  // namespace xtbloom::detail::cuda
