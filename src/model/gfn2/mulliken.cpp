#include "model/gfn2/mulliken.hpp"
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include "data/parameters/gfn2.hpp"

namespace gpuxtb::detail::gfn2 {

struct MullikenPlanData final {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t matrix_elements = 0;
  std::int64_t density_elements = 0;
  std::int64_t shell_population_elements = 0;
  std::int64_t atom_population_elements = 0;
  std::int64_t dipole_population_elements = 0;
  std::int64_t quadrupole_population_elements = 0;
  std::int64_t population_scratch_elements = 0;
  std::int64_t hamiltonian_scratch_elements = 0;

  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> shell_orbital_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int64_t> orbital_to_atom;
  std::vector<std::int32_t> spin_channels;
  std::vector<double> reference_shell_occupations;

  std::vector<std::int64_t> density_offsets;
  std::vector<std::int64_t> shell_population_offsets;
  std::vector<std::int64_t> atom_population_offsets;
  std::vector<std::int64_t> dipole_population_offsets;
  std::vector<std::int64_t> quadrupole_population_offsets;
};

namespace {

static_assert(std::is_trivially_copyable_v<MullikenIntegralView>);
static_assert(std::is_standard_layout_v<MullikenIntegralView>);
static_assert(std::is_trivially_copyable_v<MullikenDensityView>);
static_assert(std::is_standard_layout_v<MullikenDensityView>);
static_assert(std::is_trivially_copyable_v<MullikenPopulationView>);
static_assert(std::is_standard_layout_v<MullikenPopulationView>);
static_assert(std::is_trivially_copyable_v<MullikenPotentialView>);
static_assert(std::is_standard_layout_v<MullikenPotentialView>);
static_assert(std::is_trivially_copyable_v<MullikenHamiltonianView>);
static_assert(std::is_standard_layout_v<MullikenHamiltonianView>);
static_assert(std::is_trivially_copyable_v<MullikenWorkspace>);
static_assert(std::is_standard_layout_v<MullikenWorkspace>);

bool count_fits_storage(std::int64_t count, std::size_t element_size, bool add_sentinel = false) {
  if (count < 0) {
    return false;
  }
  const auto value = static_cast<std::uint64_t>(count);
  const auto extra = add_sentinel ? std::uint64_t{1} : std::uint64_t{0};
  if (value > std::numeric_limits<std::uint64_t>::max() - extra) {
    return false;
  }
  const auto length = value + extra;
  return length <=
             static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / element_size &&
         length <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool checked_add(std::int64_t increment, std::int64_t& total) {
  if (increment < 0 || total > std::numeric_limits<std::int64_t>::max() - increment) {
    return false;
  }
  total += increment;
  return true;
}

bool checked_product(std::initializer_list<std::int64_t> factors, std::int64_t& result) {
  result = 1;
  for (const std::int64_t factor : factors) {
    if (factor < 0 || (factor != 0 && result > std::numeric_limits<std::int64_t>::max() / factor)) {
      return false;
    }
    result *= factor;
  }
  return true;
}

bool byte_count(std::int64_t elements, std::size_t& bytes) {
  if (!count_fits_storage(elements, sizeof(double))) {
    return false;
  }
  bytes = static_cast<std::size_t>(elements) * sizeof(double);
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) {
  if (first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  const auto first_end = first_begin + first_bytes;
  const auto second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

bool is_aligned_double(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(double) == 0u;
}

struct MemoryRange {
  const void* data = nullptr;
  std::size_t size_bytes = 0;
};

template <std::size_t N>
bool pairwise_disjoint(const std::array<MemoryRange, N>& ranges) {
  for (std::size_t first = 0; first < N; ++first) {
    for (std::size_t second = first + 1u; second < N; ++second) {
      if (ranges_overlap(ranges[first].data, ranges[first].size_bytes, ranges[second].data,
                         ranges[second].size_bytes)) {
        return false;
      }
    }
  }
  return true;
}

std::array<MemoryRange, 17> plan_storage_ranges(const MullikenPlan& plan) {
  const MullikenPlanData* data = plan.identity();
  return {
      {{&plan, sizeof(plan)},
       {data, sizeof(MullikenPlanData)},
       {data->atom_offsets.data(), data->atom_offsets.capacity() * sizeof(std::int64_t)},
       {data->batch_shell_offsets.data(),
        data->batch_shell_offsets.capacity() * sizeof(std::int64_t)},
       {data->batch_orbital_offsets.data(),
        data->batch_orbital_offsets.capacity() * sizeof(std::int64_t)},
       {data->matrix_offsets.data(), data->matrix_offsets.capacity() * sizeof(std::int64_t)},
       {data->shell_orbital_offsets.data(),
        data->shell_orbital_offsets.capacity() * sizeof(std::int64_t)},
       {data->shell_to_atom.data(), data->shell_to_atom.capacity() * sizeof(std::int64_t)},
       {data->orbital_to_shell.data(), data->orbital_to_shell.capacity() * sizeof(std::int64_t)},
       {data->orbital_to_atom.data(), data->orbital_to_atom.capacity() * sizeof(std::int64_t)},
       {data->spin_channels.data(), data->spin_channels.capacity() * sizeof(std::int32_t)},
       {data->reference_shell_occupations.data(),
        data->reference_shell_occupations.capacity() * sizeof(double)},
       {data->density_offsets.data(), data->density_offsets.capacity() * sizeof(std::int64_t)},
       {data->shell_population_offsets.data(),
        data->shell_population_offsets.capacity() * sizeof(std::int64_t)},
       {data->atom_population_offsets.data(),
        data->atom_population_offsets.capacity() * sizeof(std::int64_t)},
       {data->dipole_population_offsets.data(),
        data->dipole_population_offsets.capacity() * sizeof(std::int64_t)},
       {data->quadrupole_population_offsets.data(),
        data->quadrupole_population_offsets.capacity() * sizeof(std::int64_t)}}};
}

bool overlaps_plan_storage(const MullikenPlan& plan, const MemoryRange& candidate) {
  for (const MemoryRange& range : plan_storage_ranges(plan)) {
    if (ranges_overlap(candidate.data, candidate.size_bytes, range.data, range.size_bytes)) {
      return true;
    }
  }
  return false;
}

template <std::size_t N>
bool overlaps_control_storage(const MullikenPlan& plan, const MemoryRange& candidate,
                              const std::array<MemoryRange, N>& descriptors) {
  if (overlaps_plan_storage(plan, candidate)) {
    return true;
  }
  for (const MemoryRange& descriptor : descriptors) {
    if (ranges_overlap(candidate.data, candidate.size_bytes, descriptor.data,
                       descriptor.size_bytes)) {
      return true;
    }
  }
  return false;
}

gpuxtb_status_t validate_plan(const MullikenPlan& plan, std::string& error) {
  if (!plan.sealed()) {
    error = "Mulliken plan is default-constructed, moved-from, or otherwise unsealed";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_pointer_count(const void* pointer, std::int64_t actual,
                                       std::int64_t expected, const char* message,
                                       std::string& error) {
  if (pointer == nullptr || actual != expected || !is_aligned_double(pointer)) {
    error = message;
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_integral_view(const MullikenPlan& plan,
                                       const MullikenIntegralView& integrals, std::string& error) {
  if (integrals.plan_identity != plan.identity()) {
    error = "Mulliken integral view belongs to a different plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (validate_pointer_count(integrals.overlap, integrals.matrix_elements, plan.matrix_elements(),
                             "Mulliken overlap view is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS) {
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (integrals.dipole == nullptr || integrals.quadrupole == nullptr ||
      !is_aligned_double(integrals.dipole) || !is_aligned_double(integrals.quadrupole)) {
    error = "Mulliken multipole integral views must not be NULL or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_view_identity(const MullikenPlan& plan, const MullikenPlanData* identity,
                                       const char* message, std::string& error) {
  if (identity != plan.identity()) {
    error = message;
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_workspace(const MullikenWorkspace& workspace, std::int64_t required,
                                   std::string& error) {
  if (workspace.scratch == nullptr || workspace.elements < required ||
      !is_aligned_double(workspace.scratch)) {
    error = "Mulliken workspace is NULL, misaligned, or too small";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

bool add_product(double left, double right, double& accumulator) {
  /* fma avoids a spurious overflow when the product and accumulator cancel. */
  const double updated = std::fma(left, right, accumulator);
  if (!std::isfinite(updated)) {
    return false;
  }
  accumulator = updated;
  return true;
}

}  // namespace

namespace {

const std::vector<std::int64_t> kEmptyInt64Vector;
const std::vector<std::int32_t> kEmptyInt32Vector;
const std::vector<double> kEmptyDoubleVector;

/* One-shot chunked contraction state shared with the SCC driver's optional
 * batch==1 parallel executor. Chunks own disjoint output elements, so every
 * computed value is bit-identical to the serial path regardless of how the
 * executor assigns chunks. failure packs a serial-traversal position in the
 * high bits and the matching failure code in the low bits; peer chunks record
 * the memory-order-independent minimum so the caller reproduces the first
 * serially-encountered data-level failure text exactly as the serial path
 * would. */
struct MullikenPopulationTask {
  const MullikenPlanData* data = nullptr;
  const MullikenIntegralView* integrals = nullptr;
  const double* density = nullptr;
  double* qsh_scratch = nullptr;
  double* dipole_scratch = nullptr;
  double* quadrupole_scratch = nullptr;
  std::int64_t orbital_begin = 0;
  std::int64_t shell_begin = 0;
  std::int64_t atom_begin = 0;
  std::int64_t atoms = 0;
  std::int64_t shells = 0;
  std::int64_t orbitals = 0;
  std::int64_t matrix_base = 0;
  std::int64_t density_base = 0;
  std::int64_t qsh_base = 0;
  std::int64_t dipole_base = 0;
  std::int64_t quadrupole_base = 0;
  std::int64_t chunk_count = 1;
  std::int32_t nspin = 1;
  std::atomic<std::uint64_t> failure{0};
};

/* Installs `candidate` (serial position, code) as the failure only when it is
 * the earliest in serial traversal seen so far; the CAS loop converges to the
 * global minimum across chunks and threads. */
void mulliken_population_record_failure(MullikenPopulationTask& task,
                                        std::uint64_t candidate) noexcept {
  std::uint64_t current = task.failure.load(std::memory_order_relaxed);
  while ((current == 0u || candidate < current) &&
         !task.failure.compare_exchange_weak(current, candidate, std::memory_order_relaxed,
                                             std::memory_order_relaxed)) {
  }
}

void mulliken_population_fail(MullikenPopulationTask& task, int code,
                              std::uint64_t position) noexcept {
  mulliken_population_record_failure(task, (position << 16u) | static_cast<std::uint32_t>(code));
}

void mulliken_population_chunk(void* opaque, std::size_t chunk) noexcept {
  MullikenPopulationTask& task = *static_cast<MullikenPopulationTask*>(opaque);
  if (task.failure.load(std::memory_order_relaxed) != 0u) {
    return;
  }
  const std::int64_t per_chunk = (task.atoms + task.chunk_count - 1) / task.chunk_count;
  const std::int64_t local_atom_begin =
      std::min<std::int64_t>(static_cast<std::int64_t>(chunk) * per_chunk, task.atoms);
  const std::int64_t local_atom_end =
      std::min<std::int64_t>(local_atom_begin + per_chunk, task.atoms);
  /* Atoms are contiguous orbital ranges (constant orbital_to_atom), so each
   * chunk's ket interval is the first orbital of its first atom through the
   * last orbital of its last atom. Shell- and atom-keyed population
   * accumulators are therefore disjoint across chunks. */
  const std::int64_t* orbitals_begin = task.data->orbital_to_atom.data() + task.orbital_begin;
  const std::int64_t* orbitals_end = orbitals_begin + task.orbitals;
  const std::int64_t local_ket_begin =
      std::lower_bound(orbitals_begin, orbitals_end, task.atom_begin + local_atom_begin) -
      orbitals_begin;
  const std::int64_t local_ket_end =
      std::lower_bound(orbitals_begin, orbitals_end, task.atom_begin + local_atom_end) -
      orbitals_begin;
  const std::int64_t matrix_elements = task.data->matrix_elements;
  for (std::int32_t spin = 0; spin < task.nspin; ++spin) {
    const std::int64_t spin_matrix_base = task.density_base + spin * task.orbitals * task.orbitals;
    for (std::int64_t local_ket = local_ket_begin; local_ket < local_ket_end; ++local_ket) {
      const std::int64_t ket = task.orbital_begin + local_ket;
      const std::int64_t local_shell =
          task.data->orbital_to_shell[static_cast<std::size_t>(ket)] - task.shell_begin;
      const std::int64_t local_atom =
          task.data->orbital_to_atom[static_cast<std::size_t>(ket)] - task.atom_begin;
      double& shell_charge = task.qsh_scratch[static_cast<std::size_t>(
          task.qsh_base + spin * task.shells + local_shell)];
      for (std::int64_t local_bra = 0; local_bra < task.orbitals; ++local_bra) {
        const std::int64_t matrix_index = task.matrix_base + local_bra * task.orbitals + local_ket;
        const std::int64_t density_index = spin_matrix_base + local_bra * task.orbitals + local_ket;
        /* Serial traversal index: spin outer, ket middle, bra inner; the per
         * element site sub-order follows the serial check sequence below. */
        const std::uint64_t element_position =
            static_cast<std::uint64_t>(
                ((spin * task.orbitals + local_ket) * task.orbitals + local_bra)) *
            11u;
        const double density_value = task.density[static_cast<std::size_t>(density_index)];
        const double overlap_value =
            task.integrals->overlap[static_cast<std::size_t>(matrix_index)];
        if (!std::isfinite(density_value) || !std::isfinite(overlap_value)) {
          mulliken_population_fail(task, 1, element_position);
          return;
        }
        if (!add_product(-density_value, overlap_value, shell_charge)) {
          mulliken_population_fail(task, 4, element_position + 1u);
          return;
        }
        for (std::int64_t component = 0; component < 3; ++component) {
          double& value = task.dipole_scratch[static_cast<std::size_t>(
              task.dipole_base + (spin * task.atoms + local_atom) * 3 + component)];
          const double integral =
              task.integrals
                  ->dipole[static_cast<std::size_t>(component * matrix_elements + matrix_index)];
          if (!std::isfinite(integral)) {
            mulliken_population_fail(task, 2,
                                     element_position + 2u + static_cast<std::uint64_t>(component));
            return;
          }
          if (!add_product(-density_value, integral, value)) {
            mulliken_population_fail(task, 5,
                                     element_position + 2u + static_cast<std::uint64_t>(component));
            return;
          }
        }
        for (std::int64_t component = 0; component < 6; ++component) {
          double& value = task.quadrupole_scratch[static_cast<std::size_t>(
              task.quadrupole_base + (spin * task.atoms + local_atom) * 6 + component)];
          const double integral = task.integrals->quadrupole[static_cast<std::size_t>(
              component * matrix_elements + matrix_index)];
          if (!std::isfinite(integral)) {
            mulliken_population_fail(task, 3,
                                     element_position + 5u + static_cast<std::uint64_t>(component));
            return;
          }
          if (!add_product(-density_value, integral, value)) {
            mulliken_population_fail(task, 6,
                                     element_position + 5u + static_cast<std::uint64_t>(component));
            return;
          }
        }
      }
    }
  }
}

const char* mulliken_population_failure_message(int code) noexcept {
  switch (code) {
    case 1:
      return "Mulliken target density or overlap contains NaN or infinity";
    case 2:
      return "Mulliken target dipole integral contains NaN or infinity";
    case 3:
      return "Mulliken target quadrupole integral contains NaN or infinity";
    case 4:
      return "Mulliken target qsh contraction exceeded floating-point range";
    case 5:
      return "Mulliken target dipole contraction exceeded floating-point range";
    case 6:
      return "Mulliken target quadrupole contraction exceeded floating-point range";
    default:
      return "Mulliken target population contraction failed";
  }
}

/* One-shot chunked Hamiltonian assembly state for the SCC driver's optional
 * batch==1 parallel executor. Rows are split per chunk, and every matrix
 * element pair is written only by the row with the smaller index, so chunks
 * own disjoint output elements and results are bit-identical to serial.
 * failure packs a serial-traversal position and the matching failure code;
 * peer chunks record the memory-order-independent minimum so the caller
 * reproduces the serially-first data-level failure exactly. */
struct MullikenHamiltonianTask {
  const MullikenPlanData* data = nullptr;
  const MullikenIntegralView* integrals = nullptr;
  const double* vat_scratch = nullptr;
  const double* vsh_scratch = nullptr;
  const double* dipole_scratch = nullptr;
  const double* quadrupole_scratch = nullptr;
  double* hamiltonian_scratch = nullptr;
  std::int64_t orbital_begin = 0;
  std::int64_t shell_begin = 0;
  std::int64_t atom_begin = 0;
  std::int64_t orbitals = 0;
  std::int64_t atoms = 0;
  std::int64_t shells = 0;
  std::int64_t matrix_base = 0;
  std::int64_t hamiltonian_base = 0;
  std::int64_t vat_base = 0;
  std::int64_t vsh_base = 0;
  std::int64_t dipole_base = 0;
  std::int64_t quadrupole_base = 0;
  std::int64_t chunk_count = 1;
  std::int32_t nspin = 1;
  std::atomic<std::uint64_t> failure{0};
};

void mulliken_hamiltonian_record_failure(MullikenHamiltonianTask& task,
                                         std::uint64_t candidate) noexcept {
  std::uint64_t current = task.failure.load(std::memory_order_relaxed);
  while ((current == 0u || candidate < current) &&
         !task.failure.compare_exchange_weak(current, candidate, std::memory_order_relaxed,
                                             std::memory_order_relaxed)) {
  }
}

void mulliken_hamiltonian_fail(MullikenHamiltonianTask& task, int code,
                               std::uint64_t position) noexcept {
  mulliken_hamiltonian_record_failure(task, (position << 16u) | static_cast<std::uint32_t>(code));
}

void mulliken_hamiltonian_chunk(void* opaque, std::size_t chunk) noexcept {
  MullikenHamiltonianTask& task = *static_cast<MullikenHamiltonianTask*>(opaque);
  if (task.failure.load(std::memory_order_relaxed) != 0u) {
    return;
  }
  const std::int64_t per_chunk = (task.orbitals + task.chunk_count - 1) / task.chunk_count;
  const std::int64_t local_row_begin =
      std::min<std::int64_t>(static_cast<std::int64_t>(chunk) * per_chunk, task.orbitals);
  const std::int64_t local_row_end =
      std::min<std::int64_t>(local_row_begin + per_chunk, task.orbitals);
  const std::int64_t matrix_elements = task.data->matrix_elements;
  for (std::int32_t spin = 0; spin < task.nspin; ++spin) {
    const std::int64_t spin_matrix_base =
        task.hamiltonian_base + spin * task.orbitals * task.orbitals;
    for (std::int64_t local_row = local_row_begin; local_row < local_row_end; ++local_row) {
      const std::int64_t row = task.orbital_begin + local_row;
      const std::int64_t row_shell = task.data->orbital_to_shell[static_cast<std::size_t>(row)];
      const std::int64_t row_atom = task.data->orbital_to_atom[static_cast<std::size_t>(row)];
      const std::int64_t local_row_shell = row_shell - task.shell_begin;
      const std::int64_t local_row_atom = row_atom - task.atom_begin;
      const double row_vat = task.vat_scratch[static_cast<std::size_t>(
          task.vat_base + spin * task.atoms + local_row_atom)];
      const double row_vsh = task.vsh_scratch[static_cast<std::size_t>(
          task.vsh_base + spin * task.shells + local_row_shell)];

      for (std::int64_t local_column = local_row; local_column < task.orbitals; ++local_column) {
        const std::int64_t column = task.orbital_begin + local_column;
        const std::int64_t column_shell =
            task.data->orbital_to_shell[static_cast<std::size_t>(column)];
        const std::int64_t column_atom =
            task.data->orbital_to_atom[static_cast<std::size_t>(column)];
        const std::int64_t local_column_shell = column_shell - task.shell_begin;
        const std::int64_t local_column_atom = column_atom - task.atom_begin;
        const double column_vat = task.vat_scratch[static_cast<std::size_t>(
            task.vat_base + spin * task.atoms + local_column_atom)];
        const double column_vsh = task.vsh_scratch[static_cast<std::size_t>(
            task.vsh_base + spin * task.shells + local_column_shell)];

        const std::int64_t forward_matrix =
            task.matrix_base + local_row * task.orbitals + local_column;
        const std::int64_t reverse_matrix =
            task.matrix_base + local_column * task.orbitals + local_row;
        const std::int64_t forward_hamiltonian =
            spin_matrix_base + local_row * task.orbitals + local_column;
        const std::int64_t reverse_hamiltonian =
            spin_matrix_base + local_column * task.orbitals + local_row;
        double shift = 0.0;
        const double overlap = task.integrals->overlap[static_cast<std::size_t>(forward_matrix)];
        const double reverse_overlap =
            task.integrals->overlap[static_cast<std::size_t>(reverse_matrix)];
        const double half_overlap = -0.5 * overlap;
        /* Serial traversal index: spin outer, row middle, column inner; the
         * per element site sub-order follows the serial check sequence below. */
        const std::uint64_t element_position =
            static_cast<std::uint64_t>(
                ((spin * task.orbitals + local_row) * task.orbitals + local_column)) *
            12u;
        if (!std::isfinite(overlap) || !std::isfinite(reverse_overlap)) {
          mulliken_hamiltonian_fail(task, 1, element_position);
          return;
        }
        if (!std::isfinite(half_overlap) || !add_product(half_overlap, row_vat, shift) ||
            !add_product(half_overlap, row_vsh, shift) ||
            !add_product(half_overlap, column_vat, shift) ||
            !add_product(half_overlap, column_vsh, shift)) {
          mulliken_hamiltonian_fail(task, 4, element_position + 1u);
          return;
        }

        for (std::int64_t component = 0; component < 3; ++component) {
          const double row_potential = task.dipole_scratch[static_cast<std::size_t>(
              task.dipole_base + (spin * task.atoms + local_row_atom) * 3 + component)];
          const double column_potential = task.dipole_scratch[static_cast<std::size_t>(
              task.dipole_base + (spin * task.atoms + local_column_atom) * 3 + component)];
          const double forward_integral =
              -0.5 *
              task.integrals
                  ->dipole[static_cast<std::size_t>(component * matrix_elements + forward_matrix)];
          const double reverse_integral =
              -0.5 *
              task.integrals
                  ->dipole[static_cast<std::size_t>(component * matrix_elements + reverse_matrix)];
          if (!std::isfinite(forward_integral) || !std::isfinite(reverse_integral)) {
            mulliken_hamiltonian_fail(
                task, 2, element_position + 2u + static_cast<std::uint64_t>(component));
            return;
          }
          if (!std::isfinite(row_potential) || !std::isfinite(column_potential) ||
              !add_product(forward_integral, column_potential, shift) ||
              !add_product(reverse_integral, row_potential, shift)) {
            mulliken_hamiltonian_fail(
                task, 5, element_position + 2u + static_cast<std::uint64_t>(component));
            return;
          }
        }

        for (std::int64_t component = 0; component < 6; ++component) {
          const double row_potential = task.quadrupole_scratch[static_cast<std::size_t>(
              task.quadrupole_base + (spin * task.atoms + local_row_atom) * 6 + component)];
          const double column_potential = task.quadrupole_scratch[static_cast<std::size_t>(
              task.quadrupole_base + (spin * task.atoms + local_column_atom) * 6 + component)];
          const double forward_integral =
              -0.5 * task.integrals->quadrupole[static_cast<std::size_t>(
                         component * matrix_elements + forward_matrix)];
          const double reverse_integral =
              -0.5 * task.integrals->quadrupole[static_cast<std::size_t>(
                         component * matrix_elements + reverse_matrix)];
          if (!std::isfinite(forward_integral) || !std::isfinite(reverse_integral)) {
            mulliken_hamiltonian_fail(
                task, 3, element_position + 5u + static_cast<std::uint64_t>(component));
            return;
          }
          if (!std::isfinite(row_potential) || !std::isfinite(column_potential) ||
              !add_product(forward_integral, column_potential, shift) ||
              !add_product(reverse_integral, row_potential, shift)) {
            mulliken_hamiltonian_fail(
                task, 6, element_position + 5u + static_cast<std::uint64_t>(component));
            return;
          }
        }
        const double forward_value =
            task.hamiltonian_scratch[static_cast<std::size_t>(forward_hamiltonian)] + shift;
        if (!std::isfinite(forward_value)) {
          mulliken_hamiltonian_fail(task, 7, element_position + 11u);
          return;
        }
        task.hamiltonian_scratch[static_cast<std::size_t>(forward_hamiltonian)] = forward_value;
        if (forward_hamiltonian != reverse_hamiltonian) {
          const double reverse_value =
              task.hamiltonian_scratch[static_cast<std::size_t>(reverse_hamiltonian)] + shift;
          if (!std::isfinite(reverse_value)) {
            mulliken_hamiltonian_fail(task, 7, element_position + 11u);
            return;
          }
          task.hamiltonian_scratch[static_cast<std::size_t>(reverse_hamiltonian)] = reverse_value;
        }
      }
    }
  }
}

/* The system-level assembly reports every data-level failure as an internal
 * error and leaves the staged Hamiltonian unchanged (see the batch wrapper for
 * the INVALID_ARGUMENT integral validation). */
gpuxtb_status_t mulliken_hamiltonian_failure_status(int code) noexcept {
  return code != 0 ? GPUXTB_STATUS_INTERNAL_ERROR : GPUXTB_STATUS_SUCCESS;
}

const char* mulliken_hamiltonian_failure_message(int code) noexcept {
  switch (code) {
    case 1:
      return "Mulliken target overlap input contains NaN or infinity";
    case 2:
      return "Mulliken target dipole integral input contains NaN or infinity";
    case 3:
      return "Mulliken target quadrupole integral input contains NaN or infinity";
    case 4:
      return "Mulliken target scalar Hamiltonian assembly exceeded floating-point range";
    case 5:
      return "Mulliken target dipole Hamiltonian assembly exceeded floating-point range";
    case 6:
      return "Mulliken target quadrupole Hamiltonian assembly exceeded floating-point range";
    case 7:
      return "Mulliken target Hamiltonian accumulation exceeded floating-point range";
    default:
      return "Mulliken target Hamiltonian assembly failed";
  }
}

}  // namespace

MullikenPlan::MullikenPlan(std::shared_ptr<const MullikenPlanData> data) noexcept
    : data_(std::move(data)) {}

bool MullikenPlan::sealed() const noexcept { return data_ != nullptr; }

std::int64_t MullikenPlan::batch_size() const noexcept {
  return data_ == nullptr ? 0 : data_->batch_size;
}

std::int64_t MullikenPlan::total_atoms() const noexcept {
  return data_ == nullptr ? 0 : data_->total_atoms;
}

std::int64_t MullikenPlan::total_shells() const noexcept {
  return data_ == nullptr ? 0 : data_->total_shells;
}

std::int64_t MullikenPlan::total_orbitals() const noexcept {
  return data_ == nullptr ? 0 : data_->total_orbitals;
}

std::int64_t MullikenPlan::matrix_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->matrix_elements;
}

std::int64_t MullikenPlan::density_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->density_elements;
}

std::int64_t MullikenPlan::shell_population_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->shell_population_elements;
}

std::int64_t MullikenPlan::atom_population_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->atom_population_elements;
}

std::int64_t MullikenPlan::dipole_population_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->dipole_population_elements;
}

std::int64_t MullikenPlan::quadrupole_population_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->quadrupole_population_elements;
}

std::int64_t MullikenPlan::population_scratch_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->population_scratch_elements;
}

std::int64_t MullikenPlan::hamiltonian_scratch_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->hamiltonian_scratch_elements;
}

std::size_t MullikenPlan::resident_bytes() const noexcept {
  if (data_ == nullptr) {
    return 0u;
  }
  return sizeof(*data_) + data_->atom_offsets.capacity() * sizeof(std::int64_t) +
         data_->batch_shell_offsets.capacity() * sizeof(std::int64_t) +
         data_->batch_orbital_offsets.capacity() * sizeof(std::int64_t) +
         data_->matrix_offsets.capacity() * sizeof(std::int64_t) +
         data_->shell_orbital_offsets.capacity() * sizeof(std::int64_t) +
         data_->shell_to_atom.capacity() * sizeof(std::int64_t) +
         data_->orbital_to_shell.capacity() * sizeof(std::int64_t) +
         data_->orbital_to_atom.capacity() * sizeof(std::int64_t) +
         data_->spin_channels.capacity() * sizeof(std::int32_t) +
         data_->reference_shell_occupations.capacity() * sizeof(double) +
         data_->density_offsets.capacity() * sizeof(std::int64_t) +
         data_->shell_population_offsets.capacity() * sizeof(std::int64_t) +
         data_->atom_population_offsets.capacity() * sizeof(std::int64_t) +
         data_->dipole_population_offsets.capacity() * sizeof(std::int64_t) +
         data_->quadrupole_population_offsets.capacity() * sizeof(std::int64_t);
}

const std::vector<std::int64_t>& MullikenPlan::atom_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->atom_offsets;
}

const std::vector<std::int64_t>& MullikenPlan::batch_shell_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->batch_shell_offsets;
}

const std::vector<std::int64_t>& MullikenPlan::batch_orbital_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->batch_orbital_offsets;
}

const std::vector<std::int64_t>& MullikenPlan::matrix_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->matrix_offsets;
}

const std::vector<std::int64_t>& MullikenPlan::shell_orbital_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->shell_orbital_offsets;
}

const std::vector<std::int64_t>& MullikenPlan::shell_to_atom() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->shell_to_atom;
}

const std::vector<std::int64_t>& MullikenPlan::orbital_to_shell() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->orbital_to_shell;
}

const std::vector<std::int64_t>& MullikenPlan::orbital_to_atom() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->orbital_to_atom;
}

const std::vector<std::int32_t>& MullikenPlan::spin_channels() const noexcept {
  return data_ == nullptr ? kEmptyInt32Vector : data_->spin_channels;
}

const std::vector<double>& MullikenPlan::reference_shell_occupations() const noexcept {
  return data_ == nullptr ? kEmptyDoubleVector : data_->reference_shell_occupations;
}

bool MullikenPlan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  return size_bytes != 0u && (data_ == nullptr || overlaps_plan_storage(*this, {data, size_bytes}));
}

const MullikenPlanData* MullikenPlan::identity() const noexcept { return data_.get(); }

gpuxtb_status_t make_mulliken_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                                   const WavefunctionLayout& wavefunction, MullikenPlan& plan,
                                   std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      basis.total_orbitals <= 0 ||
      !count_fits_storage(basis.batch_size, sizeof(std::int64_t), true) ||
      !count_fits_storage(basis.total_atoms, sizeof(std::int64_t), true) ||
      !count_fits_storage(basis.total_shells, sizeof(std::int64_t), true) ||
      !count_fits_storage(basis.total_orbitals, sizeof(std::int64_t))) {
    error = "Mulliken plan requires positive, representable basis dimensions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch_count = static_cast<std::size_t>(basis.batch_size);
  const std::size_t atom_count = static_cast<std::size_t>(basis.total_atoms);
  const std::size_t shell_count = static_cast<std::size_t>(basis.total_shells);
  const std::size_t orbital_count = static_cast<std::size_t>(basis.total_orbitals);
  if (basis.atom_offsets.size() != batch_count + 1u ||
      basis.batch_shell_offsets.size() != batch_count + 1u ||
      basis.batch_orbital_offsets.size() != batch_count + 1u ||
      basis.atom_shell_offsets.size() != atom_count + 1u ||
      basis.atom_orbital_offsets.size() != atom_count + 1u ||
      basis.shell_orbital_offsets.size() != shell_count + 1u ||
      basis.shell_to_atom.size() != shell_count ||
      basis.principal_quantum_numbers.size() != shell_count ||
      basis.angular_momenta.size() != shell_count || basis.slater_exponents.size() != shell_count ||
      basis.atom_offsets.front() != 0 || basis.atom_offsets.back() != basis.total_atoms ||
      basis.batch_shell_offsets.front() != 0 ||
      basis.batch_shell_offsets.back() != basis.total_shells ||
      basis.batch_orbital_offsets.front() != 0 ||
      basis.batch_orbital_offsets.back() != basis.total_orbitals ||
      basis.atom_shell_offsets.front() != 0 ||
      basis.atom_shell_offsets.back() != basis.total_shells ||
      basis.atom_orbital_offsets.front() != 0 ||
      basis.atom_orbital_offsets.back() != basis.total_orbitals ||
      basis.shell_orbital_offsets.front() != 0 ||
      basis.shell_orbital_offsets.back() != basis.total_orbitals) {
    error = "Mulliken plan received an incomplete or inconsistent basis";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  std::int64_t dipole_integral_elements = 0;
  std::int64_t quadrupole_integral_elements = 0;
  if (!checked_product({integrals.total_matrix_elements, 3}, dipole_integral_elements) ||
      !checked_product({integrals.total_matrix_elements, 6}, quadrupole_integral_elements) ||
      integrals.batch_size != basis.batch_size || integrals.total_matrix_elements <= 0 ||
      integrals.matrix_offsets.size() != batch_count + 1u ||
      integrals.matrix_offsets.front() != 0 ||
      integrals.matrix_offsets.back() != integrals.total_matrix_elements ||
      !count_fits_storage(integrals.total_matrix_elements, sizeof(double)) ||
      !count_fits_storage(dipole_integral_elements, sizeof(double)) ||
      !count_fits_storage(quadrupole_integral_elements, sizeof(double))) {
    error = "Mulliken integral plan is incompatible with the basis";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  if (wavefunction.batch_size != basis.batch_size ||
      wavefunction.total_atoms != basis.total_atoms ||
      wavefunction.total_shells != basis.total_shells ||
      wavefunction.total_orbitals != basis.total_orbitals ||
      wavefunction.atom_offsets != basis.atom_offsets ||
      wavefunction.batch_shell_offsets != basis.batch_shell_offsets ||
      wavefunction.batch_orbital_offsets != basis.batch_orbital_offsets ||
      wavefunction.atomic_numbers.size() != atom_count ||
      wavefunction.spin_channels.size() != batch_count ||
      wavefunction.reference_shell_occupations.size() != shell_count ||
      wavefunction.reference_atom_occupations.size() != atom_count) {
    error = "Mulliken wavefunction layout is incompatible with the basis";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::array<const WavefunctionFieldLayout*, 5> fields{
      &wavefunction.density, &wavefunction.qsh, &wavefunction.qat, &wavefunction.dipole,
      &wavefunction.quadrupole};
  for (const WavefunctionFieldLayout* field : fields) {
    if (field->element_count <= 0 || field->system_offsets.size() != batch_count + 1u ||
        field->system_offsets.front() != 0 ||
        field->system_offsets.back() != field->element_count ||
        !count_fits_storage(field->element_count, sizeof(double))) {
      error = "Mulliken wavefunction fields have invalid ragged extents";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  std::int64_t expected_matrix_total = 0;
  std::int64_t expected_density_total = 0;
  std::int64_t expected_shell_total = 0;
  std::int64_t expected_atom_total = 0;
  std::int64_t expected_dipole_total = 0;
  std::int64_t expected_quadrupole_total = 0;
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const std::int64_t atom_begin = basis.atom_offsets[batch];
    const std::int64_t atom_end = basis.atom_offsets[batch + 1u];
    const std::int64_t shell_begin = basis.batch_shell_offsets[batch];
    const std::int64_t shell_end = basis.batch_shell_offsets[batch + 1u];
    const std::int64_t orbital_begin = basis.batch_orbital_offsets[batch];
    const std::int64_t orbital_end = basis.batch_orbital_offsets[batch + 1u];
    const std::int64_t atoms = atom_end - atom_begin;
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t orbitals = orbital_end - orbital_begin;
    const std::int32_t nspin = wavefunction.spin_channels[batch];
    if (atom_begin < 0 || atom_begin >= atom_end || atom_end > basis.total_atoms ||
        shell_begin < 0 || shell_begin >= shell_end || shell_end > basis.total_shells ||
        orbital_begin < 0 || orbital_begin >= orbital_end || orbital_end > basis.total_orbitals ||
        (nspin != 1 && nspin != 2) ||
        shell_begin != basis.atom_shell_offsets[static_cast<std::size_t>(atom_begin)] ||
        shell_end != basis.atom_shell_offsets[static_cast<std::size_t>(atom_end)] ||
        orbital_begin != basis.atom_orbital_offsets[static_cast<std::size_t>(atom_begin)] ||
        orbital_end != basis.atom_orbital_offsets[static_cast<std::size_t>(atom_end)]) {
      error = "Mulliken basis offsets are not valid nonempty ragged partitions";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    std::int64_t matrix = 0;
    std::int64_t density = 0;
    std::int64_t shell_population = 0;
    std::int64_t atom_population = 0;
    std::int64_t dipole_population = 0;
    std::int64_t quadrupole_population = 0;
    if (!checked_product({orbitals, orbitals}, matrix) ||
        !checked_product({matrix, nspin}, density) ||
        !checked_product({shells, nspin}, shell_population) ||
        !checked_product({atoms, nspin}, atom_population) ||
        !checked_product({atom_population, 3}, dipole_population) ||
        !checked_product({atom_population, kWavefunctionQuadrupoleComponents},
                         quadrupole_population) ||
        integrals.matrix_offsets[batch] != expected_matrix_total ||
        wavefunction.density.system_offsets[batch] != expected_density_total ||
        wavefunction.qsh.system_offsets[batch] != expected_shell_total ||
        wavefunction.qat.system_offsets[batch] != expected_atom_total ||
        wavefunction.dipole.system_offsets[batch] != expected_dipole_total ||
        wavefunction.quadrupole.system_offsets[batch] != expected_quadrupole_total ||
        !checked_add(matrix, expected_matrix_total) ||
        !checked_add(density, expected_density_total) ||
        !checked_add(shell_population, expected_shell_total) ||
        !checked_add(atom_population, expected_atom_total) ||
        !checked_add(dipole_population, expected_dipole_total) ||
        !checked_add(quadrupole_population, expected_quadrupole_total)) {
      error = "Mulliken ragged field dimensions overflow or disagree";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  if (expected_matrix_total != integrals.total_matrix_elements ||
      expected_density_total != wavefunction.density.element_count ||
      expected_shell_total != wavefunction.qsh.element_count ||
      expected_atom_total != wavefunction.qat.element_count ||
      expected_dipole_total != wavefunction.dipole.element_count ||
      expected_quadrupole_total != wavefunction.quadrupole.element_count) {
    error = "Mulliken ragged field offsets do not span their expected storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    const std::int64_t shell_begin = basis.atom_shell_offsets[atom];
    const std::int64_t shell_end = basis.atom_shell_offsets[atom + 1u];
    const std::int64_t orbital_begin = basis.atom_orbital_offsets[atom];
    const std::int64_t orbital_end = basis.atom_orbital_offsets[atom + 1u];
    const std::int32_t atomic_number = wavefunction.atomic_numbers[atom];
    const auto* element = parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
    if (shell_begin < 0 || shell_begin >= shell_end || shell_end > basis.total_shells ||
        orbital_begin < 0 || orbital_begin >= orbital_end || orbital_end > basis.total_orbitals ||
        orbital_begin != basis.shell_orbital_offsets[static_cast<std::size_t>(shell_begin)] ||
        orbital_end != basis.shell_orbital_offsets[static_cast<std::size_t>(shell_end)] ||
        element == nullptr || element->atomic_number != atomic_number ||
        shell_end - shell_begin != element->shell_count ||
        element->shell_offset > parameters::gfn2::kShells.size() ||
        element->shell_count > parameters::gfn2::kShells.size() - element->shell_offset) {
      error = "Mulliken atom identity or shell/orbital topology is inconsistent";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    double atom_reference = 0.0;
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::size_t shell_index = static_cast<std::size_t>(shell);
      const std::size_t local_shell = static_cast<std::size_t>(shell - shell_begin);
      const auto& expected = parameters::gfn2::kShells[element->shell_offset + local_shell];
      const std::int64_t shell_orbital_begin = basis.shell_orbital_offsets[shell_index];
      const std::int64_t shell_orbital_end = basis.shell_orbital_offsets[shell_index + 1u];
      const std::int64_t expected_orbitals =
          2 * static_cast<std::int64_t>(expected.angular_momentum) + 1;
      atom_reference += expected.reference_occupation;
      if (shell_orbital_begin < 0 || shell_orbital_begin >= shell_orbital_end ||
          shell_orbital_end > basis.total_orbitals ||
          shell_orbital_end - shell_orbital_begin != expected_orbitals ||
          basis.shell_to_atom[shell_index] != static_cast<std::int64_t>(atom) ||
          basis.principal_quantum_numbers[shell_index] != expected.principal_quantum_number ||
          basis.angular_momenta[shell_index] != expected.angular_momentum ||
          basis.slater_exponents[shell_index] != expected.slater ||
          wavefunction.reference_shell_occupations[shell_index] != expected.reference_occupation) {
        error = "Mulliken basis shell metadata or reference occupation is inconsistent";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
    if (!std::isfinite(atom_reference) ||
        wavefunction.reference_atom_occupations[atom] != atom_reference) {
      error = "Mulliken atomic reference occupation is inconsistent";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  std::int64_t population_scratch = expected_shell_total;
  if (!checked_add(expected_atom_total, population_scratch) ||
      !checked_add(expected_dipole_total, population_scratch) ||
      !checked_add(expected_quadrupole_total, population_scratch) ||
      !count_fits_storage(population_scratch, sizeof(double))) {
    error = "Mulliken population scratch extent is not representable";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  std::int64_t hamiltonian_scratch = expected_density_total;
  if (!checked_add(population_scratch, hamiltonian_scratch) ||
      !count_fits_storage(hamiltonian_scratch, sizeof(double))) {
    error = "Mulliken Hamiltonian scratch extent is not representable";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  try {
    MullikenPlanData created;
    created.batch_size = basis.batch_size;
    created.total_atoms = basis.total_atoms;
    created.total_shells = basis.total_shells;
    created.total_orbitals = basis.total_orbitals;
    created.matrix_elements = expected_matrix_total;
    created.density_elements = expected_density_total;
    created.shell_population_elements = expected_shell_total;
    created.atom_population_elements = expected_atom_total;
    created.dipole_population_elements = expected_dipole_total;
    created.quadrupole_population_elements = expected_quadrupole_total;
    created.population_scratch_elements = population_scratch;
    created.hamiltonian_scratch_elements = hamiltonian_scratch;
    created.atom_offsets = basis.atom_offsets;
    created.batch_shell_offsets = basis.batch_shell_offsets;
    created.batch_orbital_offsets = basis.batch_orbital_offsets;
    created.matrix_offsets = integrals.matrix_offsets;
    created.shell_orbital_offsets = basis.shell_orbital_offsets;
    created.shell_to_atom = basis.shell_to_atom;
    created.orbital_to_shell.resize(orbital_count);
    created.orbital_to_atom.resize(orbital_count);
    created.spin_channels = wavefunction.spin_channels;
    created.reference_shell_occupations = wavefunction.reference_shell_occupations;
    created.density_offsets = wavefunction.density.system_offsets;
    created.shell_population_offsets = wavefunction.qsh.system_offsets;
    created.atom_population_offsets = wavefunction.qat.system_offsets;
    created.dipole_population_offsets = wavefunction.dipole.system_offsets;
    created.quadrupole_population_offsets = wavefunction.quadrupole.system_offsets;

    for (std::size_t shell = 0; shell < shell_count; ++shell) {
      for (std::int64_t orbital = basis.shell_orbital_offsets[shell];
           orbital < basis.shell_orbital_offsets[shell + 1u]; ++orbital) {
        const std::size_t orbital_index = static_cast<std::size_t>(orbital);
        created.orbital_to_shell[orbital_index] = static_cast<std::int64_t>(shell);
        created.orbital_to_atom[orbital_index] = basis.shell_to_atom[shell];
      }
    }

    plan = MullikenPlan(std::make_shared<const MullikenPlanData>(std::move(created)));
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate immutable Mulliken plan metadata";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

gpuxtb_status_t evaluate_mulliken_population_system_cpu(
    const MullikenPlan& plan, const MullikenIntegralView& integrals,
    const MullikenDensityView& density, const MullikenPopulationView& population,
    std::int64_t system, const MullikenWorkspace& workspace, std::string& error,
    const SccParallelExecutor* parallel) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_integral_view(plan, integrals, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_view_identity(plan, density.plan_identity,
                                  "Mulliken density view belongs to a different plan", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_view_identity(plan, population.plan_identity,
                                  "Mulliken population view belongs to a different plan", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (validate_pointer_count(density.density, density.elements, plan.density_elements(),
                             "Mulliken density view is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(population.qsh, population.qsh_elements,
                             plan.shell_population_elements(),
                             "Mulliken qsh output is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(population.qat, population.qat_elements,
                             plan.atom_population_elements(),
                             "Mulliken qat output is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(population.dipole, population.dipole_elements,
                             plan.dipole_population_elements(),
                             "Mulliken dipole output is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(population.quadrupole, population.quadrupole_elements,
                             plan.quadrupole_population_elements(),
                             "Mulliken quadrupole output is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS) {
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (system < 0 || system >= plan.batch_size()) {
    error = "Mulliken population system index is out of range";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_workspace(workspace, plan.population_scratch_elements(), error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::size_t matrix_bytes = 0;
  std::size_t dipole_integral_bytes = 0;
  std::size_t quadrupole_integral_bytes = 0;
  std::size_t density_bytes = 0;
  std::size_t qsh_bytes = 0;
  std::size_t qat_bytes = 0;
  std::size_t dipole_population_bytes = 0;
  std::size_t quadrupole_population_bytes = 0;
  std::size_t scratch_bytes = 0;
  if (!byte_count(plan.matrix_elements(), matrix_bytes) ||
      !byte_count(3 * plan.matrix_elements(), dipole_integral_bytes) ||
      !byte_count(6 * plan.matrix_elements(), quadrupole_integral_bytes) ||
      !byte_count(plan.density_elements(), density_bytes) ||
      !byte_count(plan.shell_population_elements(), qsh_bytes) ||
      !byte_count(plan.atom_population_elements(), qat_bytes) ||
      !byte_count(plan.dipole_population_elements(), dipole_population_bytes) ||
      !byte_count(plan.quadrupole_population_elements(), quadrupole_population_bytes) ||
      !byte_count(plan.population_scratch_elements(), scratch_bytes)) {
    error = "Mulliken population byte extents are not representable";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 8> active_ranges{
      {{integrals.overlap, matrix_bytes},
       {integrals.dipole, dipole_integral_bytes},
       {integrals.quadrupole, quadrupole_integral_bytes},
       {density.density, density_bytes},
       {population.qsh, qsh_bytes},
       {population.qat, qat_bytes},
       {population.dipole, dipole_population_bytes},
       {population.quadrupole, quadrupole_population_bytes}}};
  if (!pairwise_disjoint(active_ranges)) {
    error = "Mulliken population inputs and outputs must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const MemoryRange scratch_range{workspace.scratch, scratch_bytes};
  const std::array<MemoryRange, 4> descriptor_ranges{{{&integrals, sizeof(integrals)},
                                                      {&density, sizeof(density)},
                                                      {&population, sizeof(population)},
                                                      {&workspace, sizeof(workspace)}}};
  for (const MemoryRange& range : active_ranges) {
    if (ranges_overlap(scratch_range.data, scratch_range.size_bytes, range.data,
                       range.size_bytes)) {
      error = "Mulliken population workspace must not overlap inputs or outputs";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    if (overlaps_control_storage(plan, range, descriptor_ranges)) {
      error = "Mulliken population buffers must not overlap plan or descriptor storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  if (overlaps_control_storage(plan, scratch_range, descriptor_ranges)) {
    error = "Mulliken population workspace must not overlap plan or descriptor storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const MullikenPlanData& data = *plan.identity();
  const std::int64_t atom_begin = data.atom_offsets[system];
  const std::int64_t atom_end = data.atom_offsets[system + 1u];
  const std::int64_t shell_begin = data.batch_shell_offsets[system];
  const std::int64_t shell_end = data.batch_shell_offsets[system + 1u];
  const std::int64_t orbital_begin = data.batch_orbital_offsets[system];
  const std::int64_t orbital_end = data.batch_orbital_offsets[system + 1u];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t orbitals = orbital_end - orbital_begin;
  const std::int32_t nspin = data.spin_channels[system];
  const std::int64_t matrix_base = data.matrix_offsets[system];
  const std::int64_t density_base = data.density_offsets[system];
  const std::int64_t qsh_base = data.shell_population_offsets[system];
  const std::int64_t qat_base = data.atom_population_offsets[system];
  const std::int64_t dipole_base = data.dipole_population_offsets[system];
  const std::int64_t quadrupole_base = data.quadrupole_population_offsets[system];
  if (atom_begin < 0 || atom_begin > atom_end || atom_end > data.total_atoms || shell_begin < 0 ||
      shell_begin > shell_end || shell_end > data.total_shells || orbital_begin < 0 ||
      orbital_begin > orbital_end || orbital_end > data.total_orbitals || qsh_base < 0 ||
      dipole_base < 0 || quadrupole_base < 0) {
    error = "Mulliken target system partition is structurally invalid";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  double* qsh_scratch = workspace.scratch;
  double* qat_scratch = qsh_scratch + data.shell_population_elements;
  double* dipole_scratch = qat_scratch + data.atom_population_elements;
  double* quadrupole_scratch = dipole_scratch + data.dipole_population_elements;
  const std::size_t target_shells =
      static_cast<std::size_t>(nspin) * static_cast<std::size_t>(shells);
  const std::size_t target_atoms =
      static_cast<std::size_t>(nspin) * static_cast<std::size_t>(atoms);
  std::fill_n(qsh_scratch + qsh_base, target_shells, 0.0);
  std::fill_n(qat_scratch + qat_base, target_atoms, 0.0);
  std::fill_n(dipole_scratch + dipole_base, target_atoms * 3u, 0.0);
  std::fill_n(quadrupole_scratch + quadrupole_base, target_atoms * 6u, 0.0);

  const bool use_parallel = parallel != nullptr && scc_parallel_enabled(*parallel) && atoms >= 2;
  MullikenPopulationTask population_task;
  population_task.data = &data;
  population_task.integrals = &integrals;
  population_task.density = density.density;
  population_task.qsh_scratch = qsh_scratch;
  population_task.dipole_scratch = dipole_scratch;
  population_task.quadrupole_scratch = quadrupole_scratch;
  population_task.orbital_begin = orbital_begin;
  population_task.shell_begin = shell_begin;
  population_task.atom_begin = atom_begin;
  population_task.atoms = atoms;
  population_task.shells = shells;
  population_task.orbitals = orbitals;
  population_task.matrix_base = matrix_base;
  population_task.density_base = density_base;
  population_task.qsh_base = qsh_base;
  population_task.dipole_base = dipole_base;
  population_task.quadrupole_base = quadrupole_base;
  population_task.nspin = nspin;
  if (use_parallel) {
    /* A bounded number of contiguous atom-block chunks (~2 per worker) keeps
     * dispatch overhead low and shared cache lines hot; each chunk still owns
     * whole atoms so its shell/atom accumulators are disjoint and every value
     * is bit-identical to the serial path. */
    population_task.chunk_count =
        std::min<std::int64_t>(atoms, 2 * static_cast<std::int64_t>(parallel->worker_count));
    parallel->dispatch_chunks(parallel->pool_context,
                              static_cast<std::size_t>(population_task.chunk_count),
                              &mulliken_population_chunk, &population_task);
  } else {
    population_task.chunk_count = 1;
    mulliken_population_chunk(&population_task, 0u);
  }
  const std::uint64_t population_failure = population_task.failure.load(std::memory_order_relaxed);
  if (population_failure != 0u) {
    error = mulliken_population_failure_message(static_cast<int>(population_failure & 0xFFFFu));
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }

  if (nspin == 2) {
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      const std::size_t charge_index = static_cast<std::size_t>(qsh_base + local_shell);
      const std::size_t magnetization_index =
          static_cast<std::size_t>(qsh_base + shells + local_shell);
      const double alpha = qsh_scratch[charge_index];
      const double beta = qsh_scratch[magnetization_index];
      const double charge = alpha + beta;
      const double magnetization = alpha - beta;
      if (!std::isfinite(charge) || !std::isfinite(magnetization)) {
        error = "Mulliken target spin conversion exceeded floating-point range";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      qsh_scratch[charge_index] = charge;
      qsh_scratch[magnetization_index] = magnetization;
    }
  }

  for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
    const std::size_t charge_index = static_cast<std::size_t>(qsh_base + local_shell);
    const double reference =
        data.reference_shell_occupations[static_cast<std::size_t>(shell_begin + local_shell)];
    if (nspin == 1) {
      const double charge = qsh_scratch[charge_index] + reference;
      if (!std::isfinite(charge)) {
        error = "Mulliken target reference-charge addition exceeded floating-point range";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      qsh_scratch[charge_index] = charge;
    } else {
      const double charge = qsh_scratch[charge_index] + reference;
      if (!std::isfinite(charge)) {
        error = "Mulliken target reference-charge addition exceeded floating-point range";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      qsh_scratch[charge_index] = charge;
    }
  }

  if (nspin == 2) {
    for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
      for (std::int64_t component = 0; component < 3; ++component) {
        const std::size_t charge_index =
            static_cast<std::size_t>(dipole_base + local_atom * 3 + component);
        const std::size_t magnetization_index =
            static_cast<std::size_t>(dipole_base + (atoms + local_atom) * 3 + component);
        const double alpha = dipole_scratch[charge_index];
        const double beta = dipole_scratch[magnetization_index];
        const double charge = alpha + beta;
        const double magnetization = alpha - beta;
        if (!std::isfinite(charge) || !std::isfinite(magnetization)) {
          error = "Mulliken target dipole spin conversion exceeded floating-point range";
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        dipole_scratch[charge_index] = charge;
        dipole_scratch[magnetization_index] = magnetization;
      }
      for (std::int64_t component = 0; component < 6; ++component) {
        const std::size_t charge_index =
            static_cast<std::size_t>(quadrupole_base + local_atom * 6 + component);
        const std::size_t magnetization_index =
            static_cast<std::size_t>(quadrupole_base + (atoms + local_atom) * 6 + component);
        const double alpha = quadrupole_scratch[charge_index];
        const double beta = quadrupole_scratch[magnetization_index];
        const double charge = alpha + beta;
        const double magnetization = alpha - beta;
        if (!std::isfinite(charge) || !std::isfinite(magnetization)) {
          error = "Mulliken target quadrupole spin conversion exceeded floating-point range";
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        quadrupole_scratch[charge_index] = charge;
        quadrupole_scratch[magnetization_index] = magnetization;
      }
    }
  }

  for (std::int32_t channel = 0; channel < nspin; ++channel) {
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      const std::int64_t shell = shell_begin + local_shell;
      const std::int64_t local_atom =
          data.shell_to_atom[static_cast<std::size_t>(shell)] - atom_begin;
      const std::size_t atom_index =
          static_cast<std::size_t>(qat_base + channel * atoms + local_atom);
      const double updated =
          qat_scratch[atom_index] +
          qsh_scratch[static_cast<std::size_t>(qsh_base + channel * shells + local_shell)];
      if (!std::isfinite(updated)) {
        error = "Mulliken target shell-to-atom charge reduction exceeded floating-point range";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      qat_scratch[atom_index] = updated;
    }
  }

  std::copy_n(qsh_scratch + qsh_base, static_cast<std::size_t>(nspin) * shells,
              population.qsh + qsh_base);
  std::copy_n(qat_scratch + qat_base, static_cast<std::size_t>(nspin) * atoms,
              population.qat + qat_base);
  std::copy_n(dipole_scratch + dipole_base, static_cast<std::size_t>(nspin) * atoms * 3,
              population.dipole + dipole_base);
  std::copy_n(quadrupole_scratch + quadrupole_base, static_cast<std::size_t>(nspin) * atoms * 6,
              population.quadrupole + quadrupole_base);
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t evaluate_mulliken_population_cpu(const MullikenPlan& plan,
                                                 const MullikenIntegralView& integrals,
                                                 const MullikenDensityView& density,
                                                 const MullikenPopulationView& population,
                                                 const MullikenWorkspace& workspace,
                                                 std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_integral_view(plan, integrals, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_view_identity(plan, density.plan_identity,
                                  "Mulliken density view belongs to a different plan", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_view_identity(plan, population.plan_identity,
                                  "Mulliken population view belongs to a different plan", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (validate_pointer_count(density.density, density.elements, plan.density_elements(),
                             "Mulliken density view is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(population.qsh, population.qsh_elements,
                             plan.shell_population_elements(),
                             "Mulliken qsh output is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(population.qat, population.qat_elements,
                             plan.atom_population_elements(),
                             "Mulliken qat output is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(population.dipole, population.dipole_elements,
                             plan.dipole_population_elements(),
                             "Mulliken dipole output is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(population.quadrupole, population.quadrupole_elements,
                             plan.quadrupole_population_elements(),
                             "Mulliken quadrupole output is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS) {
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_workspace(workspace, plan.population_scratch_elements(), error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::size_t matrix_bytes = 0;
  std::size_t dipole_integral_bytes = 0;
  std::size_t quadrupole_integral_bytes = 0;
  std::size_t density_bytes = 0;
  std::size_t qsh_bytes = 0;
  std::size_t qat_bytes = 0;
  std::size_t dipole_population_bytes = 0;
  std::size_t quadrupole_population_bytes = 0;
  std::size_t scratch_bytes = 0;
  if (!byte_count(plan.matrix_elements(), matrix_bytes) ||
      !byte_count(3 * plan.matrix_elements(), dipole_integral_bytes) ||
      !byte_count(6 * plan.matrix_elements(), quadrupole_integral_bytes) ||
      !byte_count(plan.density_elements(), density_bytes) ||
      !byte_count(plan.shell_population_elements(), qsh_bytes) ||
      !byte_count(plan.atom_population_elements(), qat_bytes) ||
      !byte_count(plan.dipole_population_elements(), dipole_population_bytes) ||
      !byte_count(plan.quadrupole_population_elements(), quadrupole_population_bytes) ||
      !byte_count(plan.population_scratch_elements(), scratch_bytes)) {
    error = "Mulliken population byte extents are not representable";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 8> active_ranges{
      {{integrals.overlap, matrix_bytes},
       {integrals.dipole, dipole_integral_bytes},
       {integrals.quadrupole, quadrupole_integral_bytes},
       {density.density, density_bytes},
       {population.qsh, qsh_bytes},
       {population.qat, qat_bytes},
       {population.dipole, dipole_population_bytes},
       {population.quadrupole, quadrupole_population_bytes}}};
  if (!pairwise_disjoint(active_ranges)) {
    error = "Mulliken population inputs and outputs must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const MemoryRange scratch_range{workspace.scratch, scratch_bytes};
  const std::array<MemoryRange, 4> descriptor_ranges{{{&integrals, sizeof(integrals)},
                                                      {&density, sizeof(density)},
                                                      {&population, sizeof(population)},
                                                      {&workspace, sizeof(workspace)}}};
  for (const MemoryRange& range : active_ranges) {
    if (ranges_overlap(scratch_range.data, scratch_range.size_bytes, range.data,
                       range.size_bytes)) {
      error = "Mulliken population workspace must not overlap inputs or outputs";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    if (overlaps_control_storage(plan, range, descriptor_ranges)) {
      error = "Mulliken population buffers must not overlap plan or descriptor storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  if (overlaps_control_storage(plan, scratch_range, descriptor_ranges)) {
    error = "Mulliken population workspace must not overlap plan or descriptor storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const MullikenPlanData& data = *plan.identity();
  double* qsh_scratch = workspace.scratch;
  double* qat_scratch = qsh_scratch + data.shell_population_elements;
  double* dipole_scratch = qat_scratch + data.atom_population_elements;
  double* quadrupole_scratch = dipole_scratch + data.dipole_population_elements;
  std::fill_n(workspace.scratch, static_cast<std::size_t>(data.population_scratch_elements), 0.0);

  for (std::size_t system = 0; system < static_cast<std::size_t>(data.batch_size); ++system) {
    const std::int64_t atom_begin = data.atom_offsets[system];
    const std::int64_t atom_end = data.atom_offsets[system + 1u];
    const std::int64_t shell_begin = data.batch_shell_offsets[system];
    const std::int64_t shell_end = data.batch_shell_offsets[system + 1u];
    const std::int64_t orbital_begin = data.batch_orbital_offsets[system];
    const std::int64_t orbital_end = data.batch_orbital_offsets[system + 1u];
    const std::int64_t atoms = atom_end - atom_begin;
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t orbitals = orbital_end - orbital_begin;
    const std::int32_t nspin = data.spin_channels[system];
    const std::int64_t matrix_base = data.matrix_offsets[system];
    const std::int64_t density_base = data.density_offsets[system];
    const std::int64_t qsh_base = data.shell_population_offsets[system];
    const std::int64_t qat_base = data.atom_population_offsets[system];
    const std::int64_t dipole_base = data.dipole_population_offsets[system];
    const std::int64_t quadrupole_base = data.quadrupole_population_offsets[system];

    for (std::int32_t spin = 0; spin < nspin; ++spin) {
      const std::int64_t spin_matrix_base = density_base + spin * orbitals * orbitals;
      /*
       * One ket-major density pass accumulates shell charge and every atomic
       * multipole. This preserves tblite's ket ownership while avoiding a
       * separate P:S pass and validates each numerical input before use.
       */
      for (std::int64_t local_ket = 0; local_ket < orbitals; ++local_ket) {
        const std::int64_t ket = orbital_begin + local_ket;
        const std::int64_t local_shell =
            data.orbital_to_shell[static_cast<std::size_t>(ket)] - shell_begin;
        const std::int64_t local_atom =
            data.orbital_to_atom[static_cast<std::size_t>(ket)] - atom_begin;
        double& shell_charge =
            qsh_scratch[static_cast<std::size_t>(qsh_base + spin * shells + local_shell)];
        for (std::int64_t local_bra = 0; local_bra < orbitals; ++local_bra) {
          const std::int64_t matrix_index = matrix_base + local_bra * orbitals + local_ket;
          const std::int64_t density_index = spin_matrix_base + local_bra * orbitals + local_ket;
          const double density_value = density.density[static_cast<std::size_t>(density_index)];
          const double overlap_value = integrals.overlap[static_cast<std::size_t>(matrix_index)];
          if (!std::isfinite(density_value) || !std::isfinite(overlap_value)) {
            error = "Mulliken population inputs contain NaN or infinity";
            return GPUXTB_STATUS_INVALID_ARGUMENT;
          }
          if (!add_product(-density_value, overlap_value, shell_charge)) {
            error = "Mulliken qsh contraction exceeded floating-point range";
            return GPUXTB_STATUS_INTERNAL_ERROR;
          }
          for (std::int64_t component = 0; component < 3; ++component) {
            double& value = dipole_scratch[static_cast<std::size_t>(
                dipole_base + (spin * atoms + local_atom) * 3 + component)];
            const double integral = integrals.dipole[static_cast<std::size_t>(
                component * data.matrix_elements + matrix_index)];
            if (!std::isfinite(integral)) {
              error = "Mulliken population inputs contain NaN or infinity";
              return GPUXTB_STATUS_INVALID_ARGUMENT;
            }
            if (!add_product(-density_value, integral, value)) {
              error = "Mulliken dipole contraction exceeded floating-point range";
              return GPUXTB_STATUS_INTERNAL_ERROR;
            }
          }
          for (std::int64_t component = 0; component < 6; ++component) {
            double& value = quadrupole_scratch[static_cast<std::size_t>(
                quadrupole_base + (spin * atoms + local_atom) * 6 + component)];
            const double integral = integrals.quadrupole[static_cast<std::size_t>(
                component * data.matrix_elements + matrix_index)];
            if (!std::isfinite(integral)) {
              error = "Mulliken population inputs contain NaN or infinity";
              return GPUXTB_STATUS_INVALID_ARGUMENT;
            }
            if (!add_product(-density_value, integral, value)) {
              error = "Mulliken quadrupole contraction exceeded floating-point range";
              return GPUXTB_STATUS_INTERNAL_ERROR;
            }
          }
        }
      }
    }

    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      const std::size_t charge_index = static_cast<std::size_t>(qsh_base + local_shell);
      const double reference =
          data.reference_shell_occupations[static_cast<std::size_t>(shell_begin + local_shell)];
      if (nspin == 1) {
        const double charge = qsh_scratch[charge_index] + reference;
        if (!std::isfinite(charge)) {
          error = "Mulliken reference-charge addition exceeded floating-point range";
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        qsh_scratch[charge_index] = charge;
      } else {
        const std::size_t magnetization_index =
            static_cast<std::size_t>(qsh_base + shells + local_shell);
        const double alpha = qsh_scratch[charge_index];
        const double beta = qsh_scratch[magnetization_index];
        const double charge = alpha + beta + reference;
        const double magnetization = alpha - beta;
        if (!std::isfinite(charge) || !std::isfinite(magnetization)) {
          error = "Mulliken spin conversion exceeded floating-point range";
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        qsh_scratch[charge_index] = charge;
        qsh_scratch[magnetization_index] = magnetization;
      }
    }

    if (nspin == 2) {
      for (std::int64_t local_atom = 0; local_atom < atoms; ++local_atom) {
        for (std::int64_t component = 0; component < 3; ++component) {
          const std::size_t charge_index =
              static_cast<std::size_t>(dipole_base + local_atom * 3 + component);
          const std::size_t magnetization_index =
              static_cast<std::size_t>(dipole_base + (atoms + local_atom) * 3 + component);
          const double alpha = dipole_scratch[charge_index];
          const double beta = dipole_scratch[magnetization_index];
          const double charge = alpha + beta;
          const double magnetization = alpha - beta;
          if (!std::isfinite(charge) || !std::isfinite(magnetization)) {
            error = "Mulliken dipole spin conversion exceeded floating-point range";
            return GPUXTB_STATUS_INTERNAL_ERROR;
          }
          dipole_scratch[charge_index] = charge;
          dipole_scratch[magnetization_index] = magnetization;
        }
        for (std::int64_t component = 0; component < 6; ++component) {
          const std::size_t charge_index =
              static_cast<std::size_t>(quadrupole_base + local_atom * 6 + component);
          const std::size_t magnetization_index =
              static_cast<std::size_t>(quadrupole_base + (atoms + local_atom) * 6 + component);
          const double alpha = quadrupole_scratch[charge_index];
          const double beta = quadrupole_scratch[magnetization_index];
          const double charge = alpha + beta;
          const double magnetization = alpha - beta;
          if (!std::isfinite(charge) || !std::isfinite(magnetization)) {
            error = "Mulliken quadrupole spin conversion exceeded floating-point range";
            return GPUXTB_STATUS_INTERNAL_ERROR;
          }
          quadrupole_scratch[charge_index] = charge;
          quadrupole_scratch[magnetization_index] = magnetization;
        }
      }
    }

    for (std::int32_t channel = 0; channel < nspin; ++channel) {
      for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
        const std::int64_t shell = shell_begin + local_shell;
        const std::int64_t local_atom =
            data.shell_to_atom[static_cast<std::size_t>(shell)] - atom_begin;
        const std::size_t atom_index =
            static_cast<std::size_t>(qat_base + channel * atoms + local_atom);
        const double updated =
            qat_scratch[atom_index] +
            qsh_scratch[static_cast<std::size_t>(qsh_base + channel * shells + local_shell)];
        if (!std::isfinite(updated)) {
          error = "Mulliken shell-to-atom charge reduction exceeded floating-point range";
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        qat_scratch[atom_index] = updated;
      }
    }
  }

  std::copy_n(qsh_scratch, static_cast<std::size_t>(data.shell_population_elements),
              population.qsh);
  std::copy_n(qat_scratch, static_cast<std::size_t>(data.atom_population_elements), population.qat);
  std::copy_n(dipole_scratch, static_cast<std::size_t>(data.dipole_population_elements),
              population.dipole);
  std::copy_n(quadrupole_scratch, static_cast<std::size_t>(data.quadrupole_population_elements),
              population.quadrupole);
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_mulliken_hamiltonian_cpu(const MullikenPlan& plan,
                                             const MullikenIntegralView& integrals,
                                             const MullikenPotentialView& potential,
                                             const MullikenHamiltonianView& hamiltonian,
                                             const MullikenWorkspace& workspace,
                                             std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_integral_view(plan, integrals, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_view_identity(plan, potential.plan_identity,
                                  "Mulliken potential view belongs to a different plan", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_view_identity(plan, hamiltonian.plan_identity,
                                  "Mulliken Hamiltonian view belongs to a different plan", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (validate_pointer_count(potential.vat, potential.vat_elements, plan.atom_population_elements(),
                             "Mulliken vat view is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(potential.vsh, potential.vsh_elements,
                             plan.shell_population_elements(),
                             "Mulliken vsh view is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(potential.dipole, potential.dipole_elements,
                             plan.dipole_population_elements(),
                             "Mulliken dipole potential is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(
          potential.quadrupole, potential.quadrupole_elements,
          plan.quadrupole_population_elements(),
          "Mulliken quadrupole potential is NULL, misaligned, or has wrong extent",
          error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(hamiltonian.matrix, hamiltonian.elements, plan.density_elements(),
                             "Mulliken Hamiltonian is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS) {
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_workspace(workspace, plan.hamiltonian_scratch_elements(), error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::size_t matrix_bytes = 0;
  std::size_t dipole_integral_bytes = 0;
  std::size_t quadrupole_integral_bytes = 0;
  std::size_t atom_bytes = 0;
  std::size_t shell_bytes = 0;
  std::size_t dipole_bytes = 0;
  std::size_t quadrupole_bytes = 0;
  std::size_t hamiltonian_bytes = 0;
  std::size_t scratch_bytes = 0;
  if (!byte_count(plan.matrix_elements(), matrix_bytes) ||
      !byte_count(3 * plan.matrix_elements(), dipole_integral_bytes) ||
      !byte_count(6 * plan.matrix_elements(), quadrupole_integral_bytes) ||
      !byte_count(plan.atom_population_elements(), atom_bytes) ||
      !byte_count(plan.shell_population_elements(), shell_bytes) ||
      !byte_count(plan.dipole_population_elements(), dipole_bytes) ||
      !byte_count(plan.quadrupole_population_elements(), quadrupole_bytes) ||
      !byte_count(plan.density_elements(), hamiltonian_bytes) ||
      !byte_count(plan.hamiltonian_scratch_elements(), scratch_bytes)) {
    error = "Mulliken Hamiltonian byte extents are not representable";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 8> active_ranges{{{integrals.overlap, matrix_bytes},
                                                  {integrals.dipole, dipole_integral_bytes},
                                                  {integrals.quadrupole, quadrupole_integral_bytes},
                                                  {potential.vat, atom_bytes},
                                                  {potential.vsh, shell_bytes},
                                                  {potential.dipole, dipole_bytes},
                                                  {potential.quadrupole, quadrupole_bytes},
                                                  {hamiltonian.matrix, hamiltonian_bytes}}};
  if (!pairwise_disjoint(active_ranges)) {
    error = "Mulliken Hamiltonian inputs and output must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const MemoryRange scratch_range{workspace.scratch, scratch_bytes};
  const std::array<MemoryRange, 4> descriptor_ranges{{{&integrals, sizeof(integrals)},
                                                      {&potential, sizeof(potential)},
                                                      {&hamiltonian, sizeof(hamiltonian)},
                                                      {&workspace, sizeof(workspace)}}};
  for (const MemoryRange& range : active_ranges) {
    if (ranges_overlap(scratch_range.data, scratch_range.size_bytes, range.data,
                       range.size_bytes)) {
      error = "Mulliken Hamiltonian workspace must not overlap inputs or output";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    if (overlaps_control_storage(plan, range, descriptor_ranges)) {
      error = "Mulliken Hamiltonian buffers must not overlap plan or descriptor storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  if (overlaps_control_storage(plan, scratch_range, descriptor_ranges)) {
    error = "Mulliken Hamiltonian workspace must not overlap plan or descriptor storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const MullikenPlanData& data = *plan.identity();
  double* hamiltonian_scratch = workspace.scratch;
  double* vat_scratch = hamiltonian_scratch + data.density_elements;
  double* vsh_scratch = vat_scratch + data.atom_population_elements;
  double* dipole_scratch = vsh_scratch + data.shell_population_elements;
  double* quadrupole_scratch = dipole_scratch + data.dipole_population_elements;
  for (std::int64_t element = 0; element < data.density_elements; ++element) {
    const double value = hamiltonian.matrix[static_cast<std::size_t>(element)];
    if (!std::isfinite(value)) {
      error = "Mulliken Hamiltonian input contains NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    hamiltonian_scratch[static_cast<std::size_t>(element)] = value;
  }

  for (std::size_t system = 0; system < static_cast<std::size_t>(data.batch_size); ++system) {
    const std::int64_t atom_begin = data.atom_offsets[system];
    const std::int64_t atom_end = data.atom_offsets[system + 1u];
    const std::int64_t shell_begin = data.batch_shell_offsets[system];
    const std::int64_t shell_end = data.batch_shell_offsets[system + 1u];
    const std::int64_t orbital_begin = data.batch_orbital_offsets[system];
    const std::int64_t orbital_end = data.batch_orbital_offsets[system + 1u];
    const std::int64_t atoms = atom_end - atom_begin;
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t orbitals = orbital_end - orbital_begin;
    const std::int32_t nspin = data.spin_channels[system];
    const std::int64_t matrix_base = data.matrix_offsets[system];
    const std::int64_t hamiltonian_base = data.density_offsets[system];
    const std::int64_t vsh_base = data.shell_population_offsets[system];
    const std::int64_t vat_base = data.atom_population_offsets[system];
    const std::int64_t dipole_base = data.dipole_population_offsets[system];
    const std::int64_t quadrupole_base = data.quadrupole_population_offsets[system];

    const auto convert_potential = [&](const double* source, double* destination, std::int64_t base,
                                       std::int64_t channel_elements) {
      if (nspin == 1) {
        for (std::int64_t element = 0; element < channel_elements; ++element) {
          const std::size_t index = static_cast<std::size_t>(base + element);
          const double value = source[index];
          if (!std::isfinite(value)) {
            return GPUXTB_STATUS_INVALID_ARGUMENT;
          }
          destination[index] = value;
        }
        return GPUXTB_STATUS_SUCCESS;
      }
      for (std::int64_t element = 0; element < channel_elements; ++element) {
        const std::size_t charge_index = static_cast<std::size_t>(base + element);
        const std::size_t magnetization_index =
            static_cast<std::size_t>(base + channel_elements + element);
        const double charge = source[charge_index];
        const double magnetization = source[magnetization_index];
        if (!std::isfinite(charge) || !std::isfinite(magnetization)) {
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        /* Half-before-add matches tblite's magnet_to_updown conversion. */
        const double alpha = 0.5 * charge + 0.5 * magnetization;
        const double beta = 0.5 * charge - 0.5 * magnetization;
        if (!std::isfinite(alpha) || !std::isfinite(beta)) {
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        destination[charge_index] = alpha;
        destination[magnetization_index] = beta;
      }
      return GPUXTB_STATUS_SUCCESS;
    };
    const gpuxtb_status_t vat_status =
        convert_potential(potential.vat, vat_scratch, vat_base, atoms);
    const gpuxtb_status_t vsh_status =
        convert_potential(potential.vsh, vsh_scratch, vsh_base, shells);
    const gpuxtb_status_t dipole_status =
        convert_potential(potential.dipole, dipole_scratch, dipole_base, atoms * 3);
    const gpuxtb_status_t quadrupole_status =
        convert_potential(potential.quadrupole, quadrupole_scratch, quadrupole_base, atoms * 6);
    if (vat_status != GPUXTB_STATUS_SUCCESS || vsh_status != GPUXTB_STATUS_SUCCESS ||
        dipole_status != GPUXTB_STATUS_SUCCESS || quadrupole_status != GPUXTB_STATUS_SUCCESS) {
      if (vat_status == GPUXTB_STATUS_INVALID_ARGUMENT ||
          vsh_status == GPUXTB_STATUS_INVALID_ARGUMENT ||
          dipole_status == GPUXTB_STATUS_INVALID_ARGUMENT ||
          quadrupole_status == GPUXTB_STATUS_INVALID_ARGUMENT) {
        error = "Mulliken potentials contain NaN or infinity";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      error = "Mulliken potential spin conversion exceeded floating-point range";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }

    for (std::int32_t spin = 0; spin < nspin; ++spin) {
      const std::int64_t spin_matrix_base = hamiltonian_base + spin * orbitals * orbitals;
      for (std::int64_t local_row = 0; local_row < orbitals; ++local_row) {
        const std::int64_t row = orbital_begin + local_row;
        const std::int64_t row_shell = data.orbital_to_shell[static_cast<std::size_t>(row)];
        const std::int64_t row_atom = data.orbital_to_atom[static_cast<std::size_t>(row)];
        const std::int64_t local_row_shell = row_shell - shell_begin;
        const std::int64_t local_row_atom = row_atom - atom_begin;
        const double row_vat =
            vat_scratch[static_cast<std::size_t>(vat_base + spin * atoms + local_row_atom)];
        const double row_vsh =
            vsh_scratch[static_cast<std::size_t>(vsh_base + spin * shells + local_row_shell)];

        /*
         * The effective Hamiltonian is symmetric. Evaluate one triangle using
         * both ket-origin integral directions, then add the same shift to both
         * stored directions exactly as tblite does.
         */
        for (std::int64_t local_column = local_row; local_column < orbitals; ++local_column) {
          const std::int64_t column = orbital_begin + local_column;
          const std::int64_t column_shell = data.orbital_to_shell[static_cast<std::size_t>(column)];
          const std::int64_t column_atom = data.orbital_to_atom[static_cast<std::size_t>(column)];
          const std::int64_t local_column_shell = column_shell - shell_begin;
          const std::int64_t local_column_atom = column_atom - atom_begin;
          const double column_vat =
              vat_scratch[static_cast<std::size_t>(vat_base + spin * atoms + local_column_atom)];
          const double column_vsh =
              vsh_scratch[static_cast<std::size_t>(vsh_base + spin * shells + local_column_shell)];

          const std::int64_t forward_matrix = matrix_base + local_row * orbitals + local_column;
          const std::int64_t reverse_matrix = matrix_base + local_column * orbitals + local_row;
          const std::int64_t forward_hamiltonian =
              spin_matrix_base + local_row * orbitals + local_column;
          const std::int64_t reverse_hamiltonian =
              spin_matrix_base + local_column * orbitals + local_row;
          double shift = 0.0;
          const double overlap = integrals.overlap[static_cast<std::size_t>(forward_matrix)];
          const double reverse_overlap =
              integrals.overlap[static_cast<std::size_t>(reverse_matrix)];
          const double half_overlap = -0.5 * overlap;
          if (!std::isfinite(overlap) || !std::isfinite(reverse_overlap)) {
            error = "Mulliken overlap input contains NaN or infinity";
            return GPUXTB_STATUS_INVALID_ARGUMENT;
          }
          if (!std::isfinite(half_overlap) || !add_product(half_overlap, row_vat, shift) ||
              !add_product(half_overlap, row_vsh, shift) ||
              !add_product(half_overlap, column_vat, shift) ||
              !add_product(half_overlap, column_vsh, shift)) {
            error = "Mulliken scalar Hamiltonian assembly exceeded floating-point range";
            return GPUXTB_STATUS_INTERNAL_ERROR;
          }

          for (std::int64_t component = 0; component < 3; ++component) {
            const double row_potential = dipole_scratch[static_cast<std::size_t>(
                dipole_base + (spin * atoms + local_row_atom) * 3 + component)];
            const double column_potential = dipole_scratch[static_cast<std::size_t>(
                dipole_base + (spin * atoms + local_column_atom) * 3 + component)];
            const double forward_integral =
                -0.5 * integrals.dipole[static_cast<std::size_t>(component * data.matrix_elements +
                                                                 forward_matrix)];
            const double reverse_integral =
                -0.5 * integrals.dipole[static_cast<std::size_t>(component * data.matrix_elements +
                                                                 reverse_matrix)];
            if (!std::isfinite(forward_integral) || !std::isfinite(reverse_integral)) {
              error = "Mulliken dipole integral input contains NaN or infinity";
              return GPUXTB_STATUS_INVALID_ARGUMENT;
            }
            if (!std::isfinite(row_potential) || !std::isfinite(column_potential) ||
                !add_product(forward_integral, column_potential, shift) ||
                !add_product(reverse_integral, row_potential, shift)) {
              error = "Mulliken dipole Hamiltonian assembly exceeded floating-point range";
              return GPUXTB_STATUS_INTERNAL_ERROR;
            }
          }

          for (std::int64_t component = 0; component < 6; ++component) {
            const double row_potential = quadrupole_scratch[static_cast<std::size_t>(
                quadrupole_base + (spin * atoms + local_row_atom) * 6 + component)];
            const double column_potential = quadrupole_scratch[static_cast<std::size_t>(
                quadrupole_base + (spin * atoms + local_column_atom) * 6 + component)];
            const double forward_integral =
                -0.5 * integrals.quadrupole[static_cast<std::size_t>(
                           component * data.matrix_elements + forward_matrix)];
            const double reverse_integral =
                -0.5 * integrals.quadrupole[static_cast<std::size_t>(
                           component * data.matrix_elements + reverse_matrix)];
            if (!std::isfinite(forward_integral) || !std::isfinite(reverse_integral)) {
              error = "Mulliken quadrupole integral input contains NaN or infinity";
              return GPUXTB_STATUS_INVALID_ARGUMENT;
            }
            if (!std::isfinite(row_potential) || !std::isfinite(column_potential) ||
                !add_product(forward_integral, column_potential, shift) ||
                !add_product(reverse_integral, row_potential, shift)) {
              error = "Mulliken quadrupole Hamiltonian assembly exceeded floating-point range";
              return GPUXTB_STATUS_INTERNAL_ERROR;
            }
          }
          const double forward_value =
              hamiltonian_scratch[static_cast<std::size_t>(forward_hamiltonian)] + shift;
          if (!std::isfinite(forward_value)) {
            error = "Mulliken Hamiltonian accumulation exceeded floating-point range";
            return GPUXTB_STATUS_INTERNAL_ERROR;
          }
          hamiltonian_scratch[static_cast<std::size_t>(forward_hamiltonian)] = forward_value;
          if (forward_hamiltonian != reverse_hamiltonian) {
            const double reverse_value =
                hamiltonian_scratch[static_cast<std::size_t>(reverse_hamiltonian)] + shift;
            if (!std::isfinite(reverse_value)) {
              error = "Mulliken Hamiltonian accumulation exceeded floating-point range";
              return GPUXTB_STATUS_INTERNAL_ERROR;
            }
            hamiltonian_scratch[static_cast<std::size_t>(reverse_hamiltonian)] = reverse_value;
          }
        }
      }
    }
  }

  std::copy_n(hamiltonian_scratch, static_cast<std::size_t>(data.density_elements),
              hamiltonian.matrix);
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_mulliken_hamiltonian_system_cpu(
    const MullikenPlan& plan, const MullikenIntegralView& integrals,
    const MullikenPotentialView& potential, const MullikenHamiltonianView& hamiltonian,
    std::int64_t system, const MullikenWorkspace& workspace, std::string& error,
    const SccParallelExecutor* parallel) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_integral_view(plan, integrals, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_view_identity(plan, potential.plan_identity,
                                  "Mulliken potential view belongs to a different plan", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_view_identity(plan, hamiltonian.plan_identity,
                                  "Mulliken Hamiltonian view belongs to a different plan", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (validate_pointer_count(potential.vat, potential.vat_elements, plan.atom_population_elements(),
                             "Mulliken vat view is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(potential.vsh, potential.vsh_elements,
                             plan.shell_population_elements(),
                             "Mulliken vsh view is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(potential.dipole, potential.dipole_elements,
                             plan.dipole_population_elements(),
                             "Mulliken dipole potential is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(
          potential.quadrupole, potential.quadrupole_elements,
          plan.quadrupole_population_elements(),
          "Mulliken quadrupole potential is NULL, misaligned, or has wrong extent",
          error) != GPUXTB_STATUS_SUCCESS ||
      validate_pointer_count(hamiltonian.matrix, hamiltonian.elements, plan.density_elements(),
                             "Mulliken Hamiltonian is NULL, misaligned, or has wrong extent",
                             error) != GPUXTB_STATUS_SUCCESS) {
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (system < 0 || system >= plan.batch_size()) {
    error = "Mulliken Hamiltonian system index is out of range";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_workspace(workspace, plan.hamiltonian_scratch_elements(), error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::size_t matrix_bytes = 0;
  std::size_t dipole_integral_bytes = 0;
  std::size_t quadrupole_integral_bytes = 0;
  std::size_t atom_bytes = 0;
  std::size_t shell_bytes = 0;
  std::size_t dipole_bytes = 0;
  std::size_t quadrupole_bytes = 0;
  std::size_t hamiltonian_bytes = 0;
  std::size_t scratch_bytes = 0;
  if (!byte_count(plan.matrix_elements(), matrix_bytes) ||
      !byte_count(3 * plan.matrix_elements(), dipole_integral_bytes) ||
      !byte_count(6 * plan.matrix_elements(), quadrupole_integral_bytes) ||
      !byte_count(plan.atom_population_elements(), atom_bytes) ||
      !byte_count(plan.shell_population_elements(), shell_bytes) ||
      !byte_count(plan.dipole_population_elements(), dipole_bytes) ||
      !byte_count(plan.quadrupole_population_elements(), quadrupole_bytes) ||
      !byte_count(plan.density_elements(), hamiltonian_bytes) ||
      !byte_count(plan.hamiltonian_scratch_elements(), scratch_bytes)) {
    error = "Mulliken Hamiltonian byte extents are not representable";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 8> active_ranges{{{integrals.overlap, matrix_bytes},
                                                  {integrals.dipole, dipole_integral_bytes},
                                                  {integrals.quadrupole, quadrupole_integral_bytes},
                                                  {potential.vat, atom_bytes},
                                                  {potential.vsh, shell_bytes},
                                                  {potential.dipole, dipole_bytes},
                                                  {potential.quadrupole, quadrupole_bytes},
                                                  {hamiltonian.matrix, hamiltonian_bytes}}};
  if (!pairwise_disjoint(active_ranges)) {
    error = "Mulliken Hamiltonian inputs and output must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const MemoryRange scratch_range{workspace.scratch, scratch_bytes};
  const std::array<MemoryRange, 4> descriptor_ranges{{{&integrals, sizeof(integrals)},
                                                      {&potential, sizeof(potential)},
                                                      {&hamiltonian, sizeof(hamiltonian)},
                                                      {&workspace, sizeof(workspace)}}};
  for (const MemoryRange& range : active_ranges) {
    if (ranges_overlap(scratch_range.data, scratch_range.size_bytes, range.data,
                       range.size_bytes)) {
      error = "Mulliken Hamiltonian workspace must not overlap inputs or output";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    if (overlaps_control_storage(plan, range, descriptor_ranges)) {
      error = "Mulliken Hamiltonian buffers must not overlap plan or descriptor storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  if (overlaps_control_storage(plan, scratch_range, descriptor_ranges)) {
    error = "Mulliken Hamiltonian workspace must not overlap plan or descriptor storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const MullikenPlanData& data = *plan.identity();
  const std::int64_t atom_begin = data.atom_offsets[system];
  const std::int64_t atom_end = data.atom_offsets[system + 1u];
  const std::int64_t shell_begin = data.batch_shell_offsets[system];
  const std::int64_t shell_end = data.batch_shell_offsets[system + 1u];
  const std::int64_t orbital_begin = data.batch_orbital_offsets[system];
  const std::int64_t orbital_end = data.batch_orbital_offsets[system + 1u];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t orbitals = orbital_end - orbital_begin;
  const std::int32_t nspin = data.spin_channels[system];
  const std::int64_t matrix_base = data.matrix_offsets[system];
  const std::int64_t hamiltonian_base = data.density_offsets[system];
  const std::int64_t vsh_base = data.shell_population_offsets[system];
  const std::int64_t vat_base = data.atom_population_offsets[system];
  const std::int64_t dipole_base = data.dipole_population_offsets[system];
  const std::int64_t quadrupole_base = data.quadrupole_population_offsets[system];
  if (atom_begin < 0 || atom_begin > atom_end || atom_end > data.total_atoms || shell_begin < 0 ||
      shell_begin > shell_end || shell_end > data.total_shells || orbital_begin < 0 ||
      orbital_begin > orbital_end || orbital_end > data.total_orbitals || vat_base < 0 ||
      vsh_base < 0 || dipole_base < 0 || quadrupole_base < 0) {
    error = "Mulliken target system partition is structurally invalid";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  double* hamiltonian_scratch = workspace.scratch;
  double* vat_scratch = hamiltonian_scratch + data.density_elements;
  double* vsh_scratch = vat_scratch + data.atom_population_elements;
  double* dipole_scratch = vsh_scratch + data.shell_population_elements;
  double* quadrupole_scratch = dipole_scratch + data.dipole_population_elements;
  const std::size_t target_hamiltonian_elements = static_cast<std::size_t>(nspin) *
                                                  static_cast<std::size_t>(orbitals) *
                                                  static_cast<std::size_t>(orbitals);
  for (std::size_t element = 0; element < target_hamiltonian_elements; ++element) {
    const double value = hamiltonian.matrix[static_cast<std::size_t>(hamiltonian_base) + element];
    if (!std::isfinite(value)) {
      error = "Mulliken target Hamiltonian input contains NaN or infinity";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    hamiltonian_scratch[static_cast<std::size_t>(hamiltonian_base) + element] = value;
  }

  const auto convert_potential = [&](const double* source, double* destination, std::int64_t base,
                                     std::int64_t channel_elements) {
    if (nspin == 1) {
      for (std::int64_t element = 0; element < channel_elements; ++element) {
        const std::size_t index = static_cast<std::size_t>(base + element);
        const double value = source[index];
        if (!std::isfinite(value)) {
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        destination[index] = value;
      }
      return GPUXTB_STATUS_SUCCESS;
    }
    for (std::int64_t element = 0; element < channel_elements; ++element) {
      const std::size_t charge_index = static_cast<std::size_t>(base + element);
      const std::size_t magnetization_index =
          static_cast<std::size_t>(base + channel_elements + element);
      const double charge = source[charge_index];
      const double magnetization = source[magnetization_index];
      if (!std::isfinite(charge) || !std::isfinite(magnetization)) {
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      /* Half-before-add matches tblite's magnet_to_updown conversion. */
      const double alpha = 0.5 * charge + 0.5 * magnetization;
      const double beta = 0.5 * charge - 0.5 * magnetization;
      if (!std::isfinite(alpha) || !std::isfinite(beta)) {
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      destination[charge_index] = alpha;
      destination[magnetization_index] = beta;
    }
    return GPUXTB_STATUS_SUCCESS;
  };
  const gpuxtb_status_t vat_status = convert_potential(potential.vat, vat_scratch, vat_base, atoms);
  const gpuxtb_status_t vsh_status =
      convert_potential(potential.vsh, vsh_scratch, vsh_base, shells);
  const gpuxtb_status_t dipole_status =
      convert_potential(potential.dipole, dipole_scratch, dipole_base, atoms * 3);
  const gpuxtb_status_t quadrupole_status =
      convert_potential(potential.quadrupole, quadrupole_scratch, quadrupole_base, atoms * 6);
  if (vat_status != GPUXTB_STATUS_SUCCESS || vsh_status != GPUXTB_STATUS_SUCCESS ||
      dipole_status != GPUXTB_STATUS_SUCCESS || quadrupole_status != GPUXTB_STATUS_SUCCESS) {
    error = "Mulliken target potential data or spin conversion is not finite";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }

  const bool use_parallel = parallel != nullptr && scc_parallel_enabled(*parallel) && orbitals >= 2;
  MullikenHamiltonianTask assembly_task;
  assembly_task.data = &data;
  assembly_task.integrals = &integrals;
  assembly_task.vat_scratch = vat_scratch;
  assembly_task.vsh_scratch = vsh_scratch;
  assembly_task.dipole_scratch = dipole_scratch;
  assembly_task.quadrupole_scratch = quadrupole_scratch;
  assembly_task.hamiltonian_scratch = hamiltonian_scratch;
  assembly_task.orbital_begin = orbital_begin;
  assembly_task.shell_begin = shell_begin;
  assembly_task.atom_begin = atom_begin;
  assembly_task.orbitals = orbitals;
  assembly_task.atoms = atoms;
  assembly_task.shells = shells;
  assembly_task.matrix_base = matrix_base;
  assembly_task.hamiltonian_base = hamiltonian_base;
  assembly_task.vat_base = vat_base;
  assembly_task.vsh_base = vsh_base;
  assembly_task.dipole_base = dipole_base;
  assembly_task.quadrupole_base = quadrupole_base;
  assembly_task.nspin = nspin;
  if (use_parallel) {
    /* Each chunk owns a contiguous row range; the (row, column) matrix pair
     * is written only by the row with the smaller index, so chunks write
     * disjoint elements and every value is bit-identical to the serial path.
     * Chunk the rows into ~2 per worker for the same locality/dispatch balance
     * as the population contraction. */
    assembly_task.chunk_count =
        std::min<std::int64_t>(orbitals, 2 * static_cast<std::int64_t>(parallel->worker_count));
    parallel->dispatch_chunks(parallel->pool_context,
                              static_cast<std::size_t>(assembly_task.chunk_count),
                              &mulliken_hamiltonian_chunk, &assembly_task);
  } else {
    assembly_task.chunk_count = 1;
    mulliken_hamiltonian_chunk(&assembly_task, 0u);
  }
  const std::uint64_t assembly_failure = assembly_task.failure.load(std::memory_order_relaxed);
  if (assembly_failure != 0u) {
    const int assembly_code = static_cast<int>(assembly_failure & 0xFFFFu);
    error = mulliken_hamiltonian_failure_message(assembly_code);
    return mulliken_hamiltonian_failure_status(assembly_code);
  }

  std::copy_n(hamiltonian_scratch + hamiltonian_base, target_hamiltonian_elements,
              hamiltonian.matrix + hamiltonian_base);
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail::gfn2
