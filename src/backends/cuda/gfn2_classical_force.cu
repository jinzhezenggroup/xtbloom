#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_classical_force.cuh"
#include "backends/cuda/gfn2_parameters.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr double kRepulsionCutoffSquared = 25.0 * 25.0;
constexpr double kMinimumDistanceSquared = 1.0e-24;
constexpr std::int64_t kInt64Maximum = 9223372036854775807LL;
constexpr std::uint32_t kPrimitiveInactiveMarker = 0xffffffffu;

[[nodiscard]] constexpr bool component_enabled(std::uint32_t mask,
                                               Gfn2ClassicalForceComponent component) noexcept {
  return (mask & static_cast<std::uint32_t>(component)) != 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
  bool empty = true;
};

template <typename T>
bool make_range(const T* pointer, std::int64_t elements, AddressRange* range) noexcept {
  if (range == nullptr || elements < 0) {
    return false;
  }
  if (elements == 0) {
    *range = {};
    return true;
  }
  if (pointer == nullptr || reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) != 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *range = {begin, begin + bytes, false};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return !first.empty && !second.empty && first.begin < second.end && second.begin < first.end;
}

template <std::size_t N>
bool ranges_are_disjoint(const std::array<AddressRange, N>& ranges, std::size_t count) noexcept {
  if (count > N) {
    return false;
  }
  for (std::size_t first = 0; first < count; ++first) {
    for (std::size_t second = first + 1; second < count; ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

template <std::size_t N, typename T>
bool append_range(std::array<AddressRange, N>& ranges, std::size_t* count, const T* pointer,
                  std::int64_t elements) noexcept {
  if (count == nullptr || *count >= N || !make_range(pointer, elements, &ranges[*count])) {
    return false;
  }
  ++*count;
  return true;
}

bool checked_product(std::int64_t first, std::int64_t second, std::int64_t* result) noexcept {
  if (result == nullptr || first < 0 || second < 0 ||
      (first != 0 && second > kInt64Maximum / first)) {
    return false;
  }
  *result = first * second;
  return true;
}

template <typename T>
bool aligned_pointer(const T* pointer) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) == 0u;
}

bool same_pairlist_storage(const Gfn2PairListConsumerView& first,
                           const Gfn2PairListConsumerView& second) noexcept {
  return first.memory_space == second.memory_space && first.state == second.state &&
         first.pair_map_kind == second.pair_map_kind && first.plan_token == second.plan_token &&
         first.list_builder_cutoff_bohr == second.list_builder_cutoff_bohr &&
         first.batch_size == second.batch_size && first.total_atoms == second.total_atoms &&
         first.max_pairs_per_system == second.max_pairs_per_system &&
         first.max_neighbors_per_atom == second.max_neighbors_per_atom &&
         first.pair_offset_count == second.pair_offset_count &&
         first.neighbor_offset_count == second.neighbor_offset_count &&
         first.pair_count == second.pair_count && first.neighbor_count == second.neighbor_count &&
         first.pair_offsets == second.pair_offsets && first.pairs == second.pairs &&
         first.pair_count_elements == second.pair_count_elements &&
         first.neighbor_count_elements == second.neighbor_count_elements &&
         first.pair_counts == second.pair_counts &&
         first.neighbor_counts == second.neighbor_counts &&
         first.neighbor_offsets == second.neighbor_offsets && first.neighbors == second.neighbors &&
         first.committed_generation_count == second.committed_generation_count &&
         first.eligible_mask_count == second.eligible_mask_count &&
         first.active_mask_count == second.active_mask_count &&
         first.committed_generations == second.committed_generations &&
         first.eligible_mask == second.eligible_mask && first.active_mask == second.active_mask;
}

bool valid_pairlist_role(const Gfn2PairListConsumerView& view, Gfn2PairListRole role, double cutoff,
                         std::int64_t batch, std::int64_t atoms, std::uint64_t token) noexcept {
  if (batch <= 0 || atoms <= 0 || view.max_pairs_per_system <= 0 ||
      view.max_neighbors_per_atom <= 0 ||
      view.max_pairs_per_system > std::numeric_limits<std::int64_t>::max() / batch ||
      view.max_neighbors_per_atom > std::numeric_limits<std::int64_t>::max() / atoms) {
    return false;
  }
  return view.memory_space == Gfn2PlanMemorySpace::kCudaDevice &&
         view.state == Gfn2PairListState::kCommitted && view.role == role &&
         view.pair_map_kind == Gfn2PairMapKind::kExplicit && view.plan_token == token &&
         view.cutoff_bohr == cutoff && view.list_builder_cutoff_bohr == kGfn2D4TwoBodyCutoffBohr &&
         view.batch_size == batch && view.total_atoms == atoms &&
         view.pair_count == batch * view.max_pairs_per_system &&
         view.neighbor_count == atoms * view.max_neighbors_per_atom &&
         view.pair_offset_count == batch + 1 && view.neighbor_offset_count == atoms + 1 &&
         view.pair_count_elements == batch && view.neighbor_count_elements == atoms &&
         view.committed_generation_count == batch && view.eligible_mask_count == batch &&
         (view.active_mask_count == 0 || view.active_mask_count == batch) &&
         aligned_pointer(view.pair_offsets) && aligned_pointer(view.pairs) &&
         aligned_pointer(view.pair_counts) && aligned_pointer(view.neighbor_counts) &&
         aligned_pointer(view.neighbor_offsets) && aligned_pointer(view.neighbors) &&
         aligned_pointer(view.committed_generations) && aligned_pointer(view.eligible_mask) &&
         (view.active_mask_count == 0 ? view.active_mask == nullptr
                                      : aligned_pointer(view.active_mask));
}

bool valid_d4_pairlist_cache(const Gfn2ClassicalForceDevicePlan& plan,
                             const Gfn2ClassicalForceDeviceInput& input) noexcept {
  const auto& cache = plan.d4_pairlist_cache;
  const auto& coordination = cache.coordination_pairs;
  const auto& two_body = cache.two_body_pairs;
  const auto& atm = cache.atm_pairs;
  return cache.plan_token == plan.plan_token && cache.positions == input.positions &&
         cache.position_elements == plan.total_atoms * 3 &&
         cache.coordination_elements == plan.total_atoms &&
         cache.coordination_generation_elements == plan.batch_size &&
         cache.coordination_eligible_elements == plan.batch_size &&
         aligned_pointer(cache.coordination_numbers) &&
         aligned_pointer(cache.coordination_generations) &&
         aligned_pointer(cache.coordination_eligible_mask) &&
         valid_pairlist_role(coordination, Gfn2PairListRole::kD4Coordination,
                             kGfn2D4CoordinationCutoffBohr, plan.batch_size, plan.total_atoms,
                             plan.plan_token) &&
         valid_pairlist_role(two_body, Gfn2PairListRole::kD4TwoBody, kGfn2D4TwoBodyCutoffBohr,
                             plan.batch_size, plan.total_atoms, plan.plan_token) &&
         valid_pairlist_role(atm, Gfn2PairListRole::kD4Atm, kGfn2D4AtmCutoffBohr, plan.batch_size,
                             plan.total_atoms, plan.plan_token) &&
         same_pairlist_storage(coordination, two_body) && same_pairlist_storage(coordination, atm);
}

bool validate_common_descriptors(const Gfn2ClassicalForceDevicePlan& plan,
                                 const Gfn2ForceDeviceActivity& activity,
                                 const Gfn2ClassicalForceDeviceInput& input,
                                 const Gfn2ClassicalForceDeviceOutput& output,
                                 const Gfn2ClassicalForceDeviceWorkspace& workspace,
                                 const Gfn2GeometryEpochDevice* geometry_epoch,
                                 std::uint32_t* system_errors,
                                 std::uint32_t* device_error) noexcept {
  const bool extent_representable =
      plan.batch_size > 0 && plan.total_atoms > 0 && plan.total_shells > 0 &&
      plan.batch_size <= static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max()) &&
      plan.total_atoms <= kInt64Maximum / 6 && plan.total_atoms <= kInt64Maximum / 3;
  if (!extent_representable || !valid_xtb_model_flavor(plan.model) || plan.plan_token == 0u ||
      plan.geometry_generation == 0u ||
      (plan.enabled_components & ~kGfn2ClassicalForceAllComponents) != 0u ||
      !aligned_pointer(plan.atom_offsets) || !aligned_pointer(plan.atomic_numbers) ||
      activity.requested_mask == nullptr || activity.system_statuses == nullptr ||
      activity.batch_elements != plan.batch_size || activity.plan_token != plan.plan_token ||
      input.plan_token != plan.plan_token || output.plan_token != plan.plan_token ||
      workspace.plan_token != plan.plan_token || input.position_elements != plan.total_atoms * 3 ||
      output.force_elements != plan.total_atoms * 3 ||
      workspace.gradient_elements < plan.total_atoms * 3 ||
      workspace.force_elements < plan.total_atoms * 3 ||
      workspace.coordination_elements < plan.total_atoms ||
      workspace.selected_elements < plan.batch_size ||
      workspace.primitive_system_error_elements < plan.batch_size ||
      workspace.primitive_device_error_elements != 1 || workspace.sequence_elements != 1 ||
      !aligned_pointer(input.positions) || !aligned_pointer(output.forces) ||
      !aligned_pointer(workspace.gradient_scratch) || !aligned_pointer(workspace.force_scratch) ||
      !aligned_pointer(workspace.coordination_adjoints) ||
      !aligned_pointer(workspace.selected_mask) ||
      !aligned_pointer(workspace.primitive_system_errors) ||
      !aligned_pointer(workspace.primitive_device_error) ||
      !aligned_pointer(workspace.sequence_active) || !aligned_pointer(system_errors) ||
      !aligned_pointer(device_error)) {
    return false;
  }
  if (geometry_epoch != nullptr &&
      (geometry_epoch->plan_token != plan.plan_token || geometry_epoch->value_elements != 1 ||
       !aligned_pointer(geometry_epoch->value))) {
    return false;
  }
  if (plan.model == XtbModelFlavor::kGfn1) {
    const auto& correction = plan.gfn1_correction;
    const std::uint32_t forbidden =
        static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kAES2) |
        static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4TwoBody) |
        static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM);
    if ((plan.enabled_components & forbidden) != 0u ||
        plan.repulsion_sqrt_alpha_elements != plan.total_atoms ||
        plan.repulsion_effective_charge_elements != plan.total_atoms ||
        !aligned_pointer(plan.repulsion_sqrt_alpha) ||
        !aligned_pointer(plan.repulsion_effective_charge) ||
        plan.geometry_batch.model != plan.model ||
        plan.geometry_batch.batch_size != plan.batch_size ||
        plan.geometry_batch.total_atoms != plan.total_atoms ||
        plan.geometry_batch.plan_token != plan.plan_token ||
        plan.geometry_batch.atom_offsets != plan.atom_offsets || correction.model != plan.model ||
        correction.batch_size != plan.batch_size || correction.total_atoms != plan.total_atoms ||
        correction.total_pairs != plan.geometry_batch.total_pairs ||
        correction.plan_token != plan.plan_token || correction.atom_offsets != plan.atom_offsets ||
        correction.pair_offsets != plan.geometry_batch.pair_offsets ||
        correction.covalent_radii != plan.geometry_batch.covalent_radii) {
      return false;
    }
  } else if (plan.repulsion_sqrt_alpha != nullptr || plan.repulsion_effective_charge != nullptr ||
             plan.repulsion_sqrt_alpha_elements != 0 ||
             plan.repulsion_effective_charge_elements != 0 ||
             plan.gfn1_correction.plan_token != 0u || workspace.gfn1_correction.plan_token != 0u) {
    return false;
  }

  const bool needs_es2 =
      component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kES2);
  const bool needs_aes2 =
      component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kAES2);
  const bool needs_d4 =
      component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4TwoBody) ||
      component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4ATM);
  const bool needs_gfn1_correction = plan.model == XtbModelFlavor::kGfn1;
  const bool native_periodic = plan.native_short_range_batch.topology.plan_token != 0u ||
                               plan.native_ewald_batch.topology.plan_token != 0u ||
                               plan.native_multipole_batch.topology.plan_token != 0u;
  const bool native_d4 = native_periodic && needs_d4;

  if (native_periodic) {
    /* Native XYZ terms are a GFN2-only post-SCC family.  All three batches
     * borrow the same committed geometry and stationary multipole views; a
     * mismatched projection would otherwise silently mix molecular and image
     * derivatives in one force slice. */
    const auto& short_range = plan.native_short_range_batch;
    const auto& ewald = plan.native_ewald_batch;
    const auto& multipole = plan.native_multipole_batch;
    const auto& native_d4_batch = plan.native_d4_batch;
    if (plan.model != XtbModelFlavor::kGfn2 || short_range.topology.plan_token != plan.plan_token ||
        ewald.topology.plan_token != plan.plan_token ||
        multipole.topology.plan_token != plan.plan_token ||
        short_range.topology.batch_size != plan.batch_size ||
        short_range.topology.total_atoms != plan.total_atoms ||
        ewald.topology.batch_size != plan.batch_size ||
        ewald.topology.total_atoms != plan.total_atoms ||
        multipole.topology.batch_size != plan.batch_size ||
        multipole.topology.total_atoms != plan.total_atoms ||
        short_range.atomic_numbers != plan.atomic_numbers ||
        short_range.atomic_number_elements != plan.total_atoms ||
        short_range.positions != input.positions ||
        short_range.position_elements != plan.total_atoms * 3 ||
        short_range.covalent_radii != plan.geometry_batch.covalent_radii ||
        short_range.covalent_radius_elements != plan.total_atoms ||
        ewald.positions != input.positions || ewald.position_elements != plan.total_atoms * 3 ||
        ewald.shell_charges != input.shell_charges ||
        ewald.shell_charge_elements != plan.total_shells ||
        multipole.positions != input.positions ||
        multipole.position_elements != plan.total_atoms * 3 ||
        multipole.coordination_numbers != input.coordination_numbers ||
        multipole.coordination_number_elements != plan.total_atoms ||
        multipole.atomic_charges != input.atomic_charges ||
        multipole.atomic_charge_elements != plan.total_atoms ||
        multipole.atomic_dipoles != input.atomic_dipoles ||
        multipole.atomic_dipole_elements != plan.total_atoms * 3 ||
        multipole.atomic_quadrupoles != input.atomic_quadrupoles ||
        multipole.atomic_quadrupole_elements != plan.total_atoms * 6 ||
        (ewald.active_mask != nullptr && ewald.active_mask_elements != plan.batch_size) ||
        (multipole.active_mask != nullptr && multipole.active_mask_elements != plan.batch_size) ||
        workspace.native_short_range_workspace.plan_token != plan.plan_token ||
        workspace.native_ewald_workspace.plan_token != plan.plan_token ||
        workspace.native_multipole_workspace.plan_token != plan.plan_token ||
        workspace.native_short_range_workspace.repulsion_gradient_elements !=
            plan.total_atoms * 3 ||
        workspace.native_short_range_workspace.repulsion_strain_elements != plan.batch_size * 9 ||
        workspace.native_ewald_workspace.gradient_elements != plan.total_atoms * 3 ||
        workspace.native_ewald_workspace.strain_elements != plan.batch_size * 9 ||
        workspace.native_multipole_workspace.gradient_elements != plan.total_atoms * 3 ||
        workspace.native_multipole_workspace.strain_elements != plan.batch_size * 9 ||
        !aligned_pointer(workspace.native_short_range_workspace.repulsion_gradients) ||
        !aligned_pointer(workspace.native_ewald_workspace.gradients) ||
        !aligned_pointer(workspace.native_multipole_workspace.gradients) ||
        !aligned_pointer(workspace.native_multipole_workspace.coordination_adjoint)) {
      return false;
    }
    if (native_d4 &&
        (native_d4_batch.topology.plan_token != plan.plan_token ||
         native_d4_batch.topology.batch_size != plan.batch_size ||
         native_d4_batch.topology.total_atoms != plan.total_atoms ||
         native_d4_batch.plan_token != plan.plan_token ||
         native_d4_batch.atomic_numbers != plan.atomic_numbers ||
         native_d4_batch.atomic_number_elements != plan.total_atoms ||
         native_d4_batch.positions != input.positions ||
         native_d4_batch.position_elements != plan.total_atoms * 3 ||
         native_d4_batch.coordination_numbers != input.coordination_numbers ||
         native_d4_batch.coordination_number_elements != plan.total_atoms ||
         native_d4_batch.atomic_charges != input.atomic_charges ||
         native_d4_batch.atomic_charge_elements != plan.total_atoms ||
         native_d4_batch.image_cutoff < 50.0 ||
         workspace.native_d4_workspace.plan_token != plan.plan_token ||
         workspace.native_d4_workspace.wrapped_position_elements != plan.total_atoms * 3 ||
         workspace.native_d4_workspace.weight_elements !=
             plan.total_atoms * kGfn2D4MaximumReferences ||
         workspace.native_d4_workspace.coordination_elements != plan.total_atoms ||
         workspace.native_d4_workspace.atom_energy_elements != plan.total_atoms ||
         workspace.native_d4_workspace.atom_potential_elements != plan.total_atoms ||
         workspace.native_d4_workspace.gradient_elements != plan.total_atoms * 3 ||
         workspace.native_d4_workspace.strain_elements != plan.batch_size * 9 ||
         workspace.native_d4_workspace.coordination_adjoint_elements != plan.total_atoms ||
         !aligned_pointer(workspace.native_d4_workspace.wrapped_positions) ||
         !aligned_pointer(workspace.native_d4_workspace.weights) ||
         !aligned_pointer(workspace.native_d4_workspace.weight_cn_derivatives) ||
         !aligned_pointer(workspace.native_d4_workspace.weight_charge_derivatives) ||
         !aligned_pointer(workspace.native_d4_workspace.coordination) ||
         !aligned_pointer(workspace.native_d4_workspace.atom_energy) ||
         !aligned_pointer(workspace.native_d4_workspace.atom_potential) ||
         !aligned_pointer(workspace.native_d4_workspace.gradient) ||
         !aligned_pointer(workspace.native_d4_workspace.strain) ||
         !aligned_pointer(workspace.native_d4_workspace.coordination_adjoint))) {
      return false;
    }
  } else if (plan.native_short_range_batch.topology.plan_token != 0u ||
             plan.native_ewald_batch.topology.plan_token != 0u ||
             plan.native_multipole_batch.topology.plan_token != 0u ||
             workspace.native_short_range_workspace.plan_token != 0u ||
             workspace.native_ewald_workspace.plan_token != 0u ||
             workspace.native_multipole_workspace.plan_token != 0u) {
    return false;
  }

  if (needs_gfn1_correction &&
      (input.coordination_elements != plan.total_atoms ||
       input.coordination_numbers != plan.geometry_cache.coordination_numbers ||
       !aligned_pointer(input.coordination_numbers) ||
       !validate_gfn1_classical_correction_binding(
           plan.gfn1_correction, input.positions, input.coordination_numbers, nullptr, nullptr,
           workspace.gradient_scratch, workspace.gfn1_correction, workspace.primitive_system_errors,
           workspace.primitive_device_error))) {
    return false;
  }

  if (needs_es2 &&
      (input.shell_elements != plan.total_shells || !aligned_pointer(input.shell_charges) ||
       plan.es2_batch.batch_size != plan.batch_size ||
       plan.es2_batch.total_atoms != plan.total_atoms ||
       plan.es2_batch.total_shells != plan.total_shells || plan.es2_batch.model != plan.model ||
       plan.es2_batch.plan_token != plan.plan_token ||
       plan.es2_batch.atom_offsets != plan.atom_offsets ||
       plan.es2_cache.plan_token != plan.plan_token ||
       plan.es2_cache.geometry_generation != plan.geometry_generation)) {
    return false;
  }
  if (needs_aes2 &&
      (input.coordination_elements != plan.total_atoms || input.atom_elements != plan.total_atoms ||
       input.dipole_elements != plan.total_atoms * 3 ||
       input.quadrupole_elements != plan.total_atoms * 6 ||
       !aligned_pointer(input.coordination_numbers) || !aligned_pointer(input.atomic_charges) ||
       !aligned_pointer(input.atomic_dipoles) || !aligned_pointer(input.atomic_quadrupoles) ||
       plan.aes2_batch.batch_size != plan.batch_size ||
       plan.aes2_batch.total_atoms != plan.total_atoms ||
       plan.aes2_batch.plan_token != plan.plan_token ||
       plan.aes2_batch.atom_offsets != plan.atom_offsets ||
       plan.aes2_cache.plan_token != plan.plan_token ||
       plan.aes2_cache.geometry_generation != plan.geometry_generation ||
       plan.geometry_batch.batch_size != plan.batch_size ||
       plan.geometry_batch.total_atoms != plan.total_atoms ||
       plan.geometry_batch.plan_token != plan.plan_token ||
       plan.geometry_batch.atom_offsets != plan.atom_offsets ||
       plan.geometry_cache.plan_token != plan.plan_token ||
       workspace.aes2_workspace.gradient_elements < plan.total_atoms * 3 ||
       workspace.aes2_workspace.coordination_elements < plan.total_atoms ||
       !aligned_pointer(workspace.aes2_workspace.gradient_scratch) ||
       !aligned_pointer(workspace.aes2_workspace.coordination_scratch) ||
       workspace.geometry_workspace.gradient_elements < plan.total_atoms * 3 ||
       workspace.geometry_workspace.sequence_elements != 1 ||
       !aligned_pointer(workspace.geometry_workspace.gradient_scratch) ||
       !aligned_pointer(workspace.geometry_workspace.sequence_active))) {
    return false;
  }
  if (needs_d4 && (plan.d4_batch.batch_size != plan.batch_size ||
                   plan.d4_batch.total_atoms != plan.total_atoms ||
                   plan.d4_batch.plan_token != plan.plan_token ||
                   plan.d4_batch.atom_offsets != plan.atom_offsets ||
                   plan.d4_batch.atomic_numbers != plan.atomic_numbers ||
                   !valid_d4_pairlist_cache(plan, input) ||
                   workspace.d4_workspace.system_errors != workspace.primitive_system_errors ||
                   workspace.d4_workspace.system_error_elements < plan.batch_size)) {
    return false;
  }
  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4TwoBody) &&
      (input.atom_elements != plan.total_atoms || !aligned_pointer(input.atomic_charges))) {
    return false;
  }

  std::int64_t geometry_pair_elements = 0;
  std::int64_t aes2_pair_elements = 0;
  std::int64_t d4_weight_elements = 0;
  if ((needs_aes2 && (!checked_product(plan.geometry_batch.total_pairs,
                                       kGfn2GeometryPairDataElements, &geometry_pair_elements) ||
                      !checked_product(plan.aes2_batch.total_pairs, kGfn2AES2PairDataElements,
                                       &aes2_pair_elements))) ||
      (needs_d4 &&
       !checked_product(plan.total_atoms, kGfn2D4MaximumReferences, &d4_weight_elements))) {
    return false;
  }

  /*
   * Keep this fixed-capacity inventory synchronized with the nested primitive
   * calls below. Validation is a hot-path host operation and must not allocate.
   * d4_workspace.system_errors is the documented exact projection of
   * primitive_system_errors, so that range is represented only once.
   */
  std::array<AddressRange, 80> writes{};
  std::size_t write_count = 0u;
  if (!append_range(writes, &write_count, output.forces, plan.total_atoms * 3) ||
      !append_range(writes, &write_count, workspace.gradient_scratch, plan.total_atoms * 3) ||
      !append_range(writes, &write_count, workspace.force_scratch, plan.total_atoms * 3) ||
      !append_range(writes, &write_count, workspace.coordination_adjoints, plan.total_atoms) ||
      !append_range(writes, &write_count, workspace.selected_mask, plan.batch_size) ||
      !append_range(writes, &write_count, workspace.primitive_system_errors, plan.batch_size) ||
      !append_range(writes, &write_count, workspace.primitive_device_error, 1) ||
      !append_range(writes, &write_count, workspace.sequence_active, 1) ||
      !append_range(writes, &write_count, system_errors, plan.batch_size) ||
      !append_range(writes, &write_count, device_error, 1)) {
    return false;
  }
  if (needs_aes2 &&
      (!append_range(writes, &write_count, workspace.aes2_workspace.pair_scratch,
                     workspace.aes2_workspace.pair_elements) ||
       !append_range(writes, &write_count, workspace.aes2_workspace.potential_scratch,
                     workspace.aes2_workspace.potential_elements) ||
       !append_range(writes, &write_count, workspace.aes2_workspace.batch_scratch,
                     workspace.aes2_workspace.batch_elements) ||
       !append_range(writes, &write_count, workspace.aes2_workspace.gradient_scratch,
                     plan.total_atoms * 3) ||
       !append_range(writes, &write_count, workspace.aes2_workspace.coordination_scratch,
                     plan.total_atoms) ||
       !append_range(writes, &write_count, workspace.aes2_workspace.scc_peer_error_scratch,
                     workspace.aes2_workspace.scc_peer_error_elements) ||
       !append_range(writes, &write_count, workspace.geometry_workspace.pair_scratch,
                     workspace.geometry_workspace.pair_elements) ||
       !append_range(writes, &write_count, workspace.geometry_workspace.coordination_scratch,
                     workspace.geometry_workspace.coordination_elements) ||
       !append_range(writes, &write_count, workspace.geometry_workspace.gradient_scratch,
                     plan.total_atoms * 3) ||
       !append_range(writes, &write_count, workspace.geometry_workspace.sequence_active, 1))) {
    return false;
  }
  if (needs_d4 &&
      (!append_range(writes, &write_count, workspace.d4_workspace.weights, d4_weight_elements) ||
       !append_range(writes, &write_count, workspace.d4_workspace.weight_cn_derivatives,
                     d4_weight_elements) ||
       !append_range(writes, &write_count, workspace.d4_workspace.weight_charge_derivatives,
                     d4_weight_elements) ||
       !append_range(writes, &write_count, workspace.d4_workspace.atom_scratch, plan.total_atoms) ||
       !append_range(writes, &write_count, workspace.d4_workspace.coordination_adjoints,
                     plan.total_atoms) ||
       !append_range(writes, &write_count, workspace.d4_workspace.batch_scratch, plan.batch_size) ||
       !append_range(writes, &write_count, workspace.d4_workspace.gradient_scratch,
                     plan.total_atoms * 3))) {
    return false;
  }
  if (needs_gfn1_correction &&
      (!append_range(writes, &write_count, workspace.gfn1_correction.weights,
                     workspace.gfn1_correction.weight_elements) ||
       !append_range(writes, &write_count, workspace.gfn1_correction.weight_cn_derivatives,
                     workspace.gfn1_correction.weight_cn_derivative_elements) ||
       !append_range(writes, &write_count, workspace.gfn1_correction.coordination_adjoints,
                     workspace.gfn1_correction.coordination_adjoint_elements) ||
       !append_range(writes, &write_count, workspace.gfn1_correction.axis_neighbors,
                     workspace.gfn1_correction.axis_neighbor_elements) ||
       !append_range(writes, &write_count, workspace.gfn1_correction.batch_scratch,
                     workspace.gfn1_correction.batch_scratch_elements) ||
       !append_range(writes, &write_count, workspace.gfn1_correction.gradient_scratch,
                     workspace.gfn1_correction.gradient_scratch_elements))) {
    return false;
  }
  if (native_periodic) {
    const auto& sr = workspace.native_short_range_workspace;
    const auto& ew = workspace.native_ewald_workspace;
    const auto& mp = workspace.native_multipole_workspace;
    if (!append_range(writes, &write_count, sr.wrapped_positions, sr.wrapped_position_elements) ||
        !append_range(writes, &write_count, sr.coordination, sr.coordination_elements) ||
        !append_range(writes, &write_count, sr.repulsion_energies, sr.repulsion_energy_elements) ||
        !append_range(writes, &write_count, sr.repulsion_gradients,
                      sr.repulsion_gradient_elements) ||
        !append_range(writes, &write_count, sr.repulsion_strain, sr.repulsion_strain_elements) ||
        !append_range(writes, &write_count, ew.wrapped_positions, ew.wrapped_position_elements) ||
        !append_range(writes, &write_count, ew.matrix, ew.matrix_elements) ||
        !append_range(writes, &write_count, ew.shell_potentials, ew.shell_potential_elements) ||
        !append_range(writes, &write_count, ew.energies, ew.energy_elements) ||
        !append_range(writes, &write_count, ew.gradients, ew.gradient_elements) ||
        !append_range(writes, &write_count, ew.strain, ew.strain_elements) ||
        !append_range(writes, &write_count, mp.wrapped_positions, mp.wrapped_position_elements) ||
        !append_range(writes, &write_count, mp.charge_dipole_matrix,
                      mp.charge_dipole_matrix_elements) ||
        !append_range(writes, &write_count, mp.dipole_dipole_matrix,
                      mp.dipole_dipole_matrix_elements) ||
        !append_range(writes, &write_count, mp.charge_quadrupole_matrix,
                      mp.charge_quadrupole_matrix_elements) ||
        !append_range(writes, &write_count, mp.charge_potentials, mp.charge_potential_elements) ||
        !append_range(writes, &write_count, mp.dipole_potentials, mp.dipole_potential_elements) ||
        !append_range(writes, &write_count, mp.quadrupole_potentials,
                      mp.quadrupole_potential_elements) ||
        !append_range(writes, &write_count, mp.energies, mp.energy_elements) ||
        !append_range(writes, &write_count, mp.gradients, mp.gradient_elements) ||
        !append_range(writes, &write_count, mp.strain, mp.strain_elements) ||
        !append_range(writes, &write_count, mp.coordination_adjoint,
                      mp.coordination_adjoint_elements)) {
      {
        return false;
      }
      if (native_d4) {
        const auto& d4 = workspace.native_d4_workspace;
        if (!append_range(writes, &write_count, d4.wrapped_positions,
                          d4.wrapped_position_elements) ||
            !append_range(writes, &write_count, d4.weights, d4.weight_elements) ||
            !append_range(writes, &write_count, d4.weight_cn_derivatives, d4.weight_elements) ||
            !append_range(writes, &write_count, d4.weight_charge_derivatives, d4.weight_elements) ||
            !append_range(writes, &write_count, d4.coordination, d4.coordination_elements) ||
            !append_range(writes, &write_count, d4.atom_energy, d4.atom_energy_elements) ||
            !append_range(writes, &write_count, d4.atom_potential, d4.atom_potential_elements) ||
            !append_range(writes, &write_count, d4.gradient, d4.gradient_elements) ||
            !append_range(writes, &write_count, d4.strain, d4.strain_elements) ||
            !append_range(writes, &write_count, d4.coordination_adjoint,
                          d4.coordination_adjoint_elements)) {
          return false;
        }
      }
    }
  }
  if (!ranges_are_disjoint(writes, write_count)) {
    {
      return false;
    }
  }

  std::array<AddressRange, 80> reads{};
  std::size_t read_count = 0u;
  if (!append_range(reads, &read_count, plan.atom_offsets, plan.batch_size + 1) ||
      !append_range(reads, &read_count, plan.atomic_numbers, plan.total_atoms) ||
      (plan.model == XtbModelFlavor::kGfn1 &&
       (!append_range(reads, &read_count, plan.repulsion_sqrt_alpha, plan.total_atoms) ||
        !append_range(reads, &read_count, plan.repulsion_effective_charge, plan.total_atoms))) ||
      !append_range(reads, &read_count, activity.requested_mask, plan.batch_size) ||
      !append_range(reads, &read_count, activity.system_statuses, plan.batch_size) ||
      !append_range(reads, &read_count, input.positions, plan.total_atoms * 3)) {
    {
      return false;
    }
  }
  if (needs_es2 &&
      (!append_range(reads, &read_count, plan.es2_batch.batch_shell_offsets, plan.batch_size + 1) ||
       !append_range(reads, &read_count, plan.es2_batch.atom_shell_offsets, plan.total_atoms + 1) ||
       !append_range(reads, &read_count, plan.es2_batch.matrix_offsets, plan.batch_size + 1) ||
       !append_range(reads, &read_count, plan.es2_batch.shell_to_atom, plan.total_shells) ||
       !append_range(reads, &read_count, plan.es2_batch.shell_hardness, plan.total_shells) ||
       !append_range(reads, &read_count, plan.es2_cache.coulomb_matrix,
                     plan.es2_batch.total_matrix_elements) ||
       !append_range(reads, &read_count, input.shell_charges, plan.total_shells))) {
    {
      return false;
    }
  }
  if (needs_aes2 &&
      (!append_range(reads, &read_count, plan.aes2_batch.pair_offsets, plan.batch_size + 1) ||
       !append_range(reads, &read_count, plan.aes2_batch.dipole_kernel, plan.total_atoms) ||
       !append_range(reads, &read_count, plan.aes2_batch.quadrupole_kernel, plan.total_atoms) ||
       !append_range(reads, &read_count, plan.aes2_batch.multipole_radius, plan.total_atoms) ||
       !append_range(reads, &read_count, plan.aes2_batch.multipole_valence_cn, plan.total_atoms) ||
       !append_range(reads, &read_count, plan.aes2_cache.pair_data, aes2_pair_elements) ||
       !append_range(reads, &read_count, input.coordination_numbers, plan.total_atoms) ||
       !append_range(reads, &read_count, input.atomic_charges, plan.total_atoms) ||
       !append_range(reads, &read_count, input.atomic_dipoles, plan.total_atoms * 3) ||
       !append_range(reads, &read_count, input.atomic_quadrupoles, plan.total_atoms * 6) ||
       !append_range(reads, &read_count, plan.geometry_batch.pair_offsets, plan.batch_size + 1) ||
       !append_range(reads, &read_count, plan.geometry_batch.covalent_radii, plan.total_atoms) ||
       !append_range(reads, &read_count, plan.geometry_cache.pair_data, geometry_pair_elements) ||
       !append_range(reads, &read_count, plan.geometry_cache.coordination_numbers,
                     plan.total_atoms) ||
       !append_range(reads, &read_count, plan.geometry_cache.geometry_generations,
                     plan.batch_size))) {
    {
      return false;
    }
  }
  if (needs_gfn1_correction) {
    const auto& correction = plan.gfn1_correction;
    if (!append_range(reads, &read_count, input.coordination_numbers, plan.total_atoms) ||
        !append_range(reads, &read_count, correction.pair_offsets,
                      correction.pair_offset_elements) ||
        !append_range(reads, &read_count, correction.covalent_radii,
                      correction.covalent_radius_elements) ||
        !append_range(reads, &read_count, correction.reference_counts,
                      correction.reference_count_elements) ||
        !append_range(reads, &read_count, correction.reference_cn,
                      correction.reference_cn_elements) ||
        !append_range(reads, &read_count, correction.reference_c6,
                      correction.reference_c6_elements) ||
        !append_range(reads, &read_count, correction.pair_rrij, correction.pair_rrij_elements) ||
        !append_range(reads, &read_count, correction.pair_damping_radii,
                      correction.pair_damping_radius_elements) ||
        !append_range(reads, &read_count, correction.halogen_scaled_radii,
                      correction.halogen_scaled_radius_elements) ||
        !append_range(reads, &read_count, correction.halogen_bond_strength,
                      correction.halogen_bond_strength_elements) ||
        !append_range(reads, &read_count, correction.halogen_donor,
                      correction.halogen_donor_elements) ||
        !append_range(reads, &read_count, correction.halogen_acceptor,
                      correction.halogen_acceptor_elements)) {
      return false;
    }
  }
  if (needs_d4) {
    const auto& cache = plan.d4_pairlist_cache;
    const auto& pairs = cache.coordination_pairs;
    if (!append_range(reads, &read_count, plan.d4_parameters.elements,
                      plan.d4_parameters.element_count) ||
        !append_range(reads, &read_count, plan.d4_parameters.references,
                      plan.d4_parameters.reference_count) ||
        !append_range(reads, &read_count, plan.d4_parameters.reference_c6,
                      plan.d4_parameters.reference_c6_elements) ||
        !append_range(reads, &read_count, cache.coordination_numbers, plan.total_atoms) ||
        !append_range(reads, &read_count, cache.coordination_generations, plan.batch_size) ||
        !append_range(reads, &read_count, cache.coordination_eligible_mask, plan.batch_size) ||
        !append_range(reads, &read_count, pairs.pair_offsets, pairs.pair_offset_count) ||
        !append_range(reads, &read_count, pairs.pairs, pairs.pair_count) ||
        !append_range(reads, &read_count, pairs.pair_counts, pairs.pair_count_elements) ||
        !append_range(reads, &read_count, pairs.neighbor_offsets, pairs.neighbor_offset_count) ||
        !append_range(reads, &read_count, pairs.neighbor_counts, pairs.neighbor_count_elements) ||
        !append_range(reads, &read_count, pairs.neighbors, pairs.neighbor_count) ||
        !append_range(reads, &read_count, pairs.committed_generations,
                      pairs.committed_generation_count) ||
        !append_range(reads, &read_count, pairs.eligible_mask, pairs.eligible_mask_count) ||
        !append_range(reads, &read_count, pairs.active_mask, pairs.active_mask_count)) {
      return false;
    }
  }
  /* q is already represented by AES2 when both charge-dependent components run. */
  if (!needs_aes2 &&
      component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4TwoBody) &&
      !append_range(reads, &read_count, input.atomic_charges, plan.total_atoms)) {
    {
      return false;
    }
  }
  if (native_periodic) {
    const auto& sr = plan.native_short_range_batch;
    const auto& ew = plan.native_ewald_batch;
    const auto& mp = plan.native_multipole_batch;
    if (!append_range(reads, &read_count, sr.topology.cell_matrices, sr.topology.cell_elements) ||
        !append_range(reads, &read_count, sr.topology.periodic_axes,
                      sr.topology.periodic_axes_elements) ||
        !append_range(reads, &read_count, sr.topology.translation_offsets,
                      sr.topology.translation_offset_count) ||
        !append_range(reads, &read_count, sr.topology.translations,
                      sr.topology.total_translations) ||
        !append_range(reads, &read_count, ew.batch_shell_offsets, ew.batch_shell_offset_elements) ||
        !append_range(reads, &read_count, ew.atom_shell_offsets, ew.atom_shell_offset_elements) ||
        !append_range(reads, &read_count, ew.matrix_offsets, ew.matrix_offset_elements) ||
        !append_range(reads, &read_count, ew.shell_hardness, ew.shell_hardness_elements) ||
        !append_range(reads, &read_count, ew.alphas, ew.alpha_elements) ||
        !append_range(reads, &read_count, ew.direct_translation_offsets,
                      ew.direct_translation_offset_elements) ||
        !append_range(reads, &read_count, ew.direct_translations, ew.direct_translation_elements) ||
        !append_range(reads, &read_count, ew.reciprocal_translation_offsets,
                      ew.reciprocal_translation_offset_elements) ||
        !append_range(reads, &read_count, ew.reciprocal_translations,
                      ew.reciprocal_translation_elements) ||
        !append_range(reads, &read_count, mp.matrix_offsets, mp.matrix_offset_elements) ||
        !append_range(reads, &read_count, mp.volumes, mp.volume_elements) ||
        !append_range(reads, &read_count, mp.alphas, mp.alpha_elements) ||
        !append_range(reads, &read_count, mp.direct_translation_offsets,
                      mp.direct_translation_offset_elements) ||
        !append_range(reads, &read_count, mp.direct_translations, mp.direct_translation_elements) ||
        !append_range(reads, &read_count, mp.reciprocal_translation_offsets,
                      mp.reciprocal_translation_offset_elements) ||
        !append_range(reads, &read_count, mp.reciprocal_translations,
                      mp.reciprocal_translation_elements) ||
        !append_range(reads, &read_count, mp.dipole_kernel, mp.dipole_kernel_elements) ||
        !append_range(reads, &read_count, mp.quadrupole_kernel, mp.quadrupole_kernel_elements) ||
        !append_range(reads, &read_count, mp.multipole_radius, mp.multipole_radius_elements) ||
        !append_range(reads, &read_count, mp.multipole_valence_cn,
                      mp.multipole_valence_cn_elements)) {
      return false;
    }
  }
  if (geometry_epoch != nullptr && !append_range(reads, &read_count, geometry_epoch->value, 1)) {
    return false;
  }

  for (std::size_t read_index = 0; read_index < read_count; ++read_index) {
    for (std::size_t write_index = 0; write_index < write_count; ++write_index) {
      const AddressRange& read = reads[read_index];
      const AddressRange& write = writes[write_index];
      if (ranges_overlap(read, write)) {
        return false;
      }
    }
  }
  return true;
}

__device__ bool sequence_is_open(const std::uint32_t* sequence_active) {
  return atomicAdd(const_cast<std::uint32_t*>(sequence_active), 0u) == 1u;
}

__device__ bool system_is_selected(const Gfn2ClassicalForceDeviceWorkspace& workspace,
                                   const std::uint32_t* system_errors, std::int64_t system) {
  return sequence_is_open(workspace.sequence_active) && workspace.selected_mask[system] == 1u &&
         atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) == 0u;
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error,
                                    Gfn2ClassicalForceDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system, 0u, code) == 0u) {
    atomicCAS(device_error, 0u, code);
  }
}

__device__ void record_plan_error(std::uint32_t* device_error, std::uint32_t* sequence_active,
                                  Gfn2ClassicalForceDeviceError error) {
  atomicCAS(device_error, 0u, static_cast<std::uint32_t>(error));
  atomicExch(sequence_active, 0u);
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) == 0u ? 1u : 0u;
  }
}

__global__ void common_topology_preflight_kernel(Gfn2ClassicalForceDevicePlan plan,
                                                 std::uint32_t* device_error,
                                                 std::uint32_t* sequence_active) {
  if (blockIdx.x != 0 || !sequence_is_open(sequence_active)) {
    return;
  }
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = plan.atom_offsets[0] == 0 && plan.atom_offsets[plan.batch_size] == plan.total_atoms;
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < plan.batch_size; system += blockDim.x) {
    const std::int64_t begin = plan.atom_offsets[system];
    const std::int64_t end = plan.atom_offsets[system + 1];
    if (begin < 0 || begin > end || end > plan.total_atoms) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && valid == 0) {
    record_plan_error(device_error, sequence_active,
                      Gfn2ClassicalForceDeviceError::kInvalidTopology);
  }
}

__global__ void gate_and_seed_kernel(Gfn2ClassicalForceDevicePlan plan,
                                     Gfn2ForceDeviceActivity activity,
                                     Gfn2ClassicalForceDeviceOutput output,
                                     Gfn2ClassicalForceDeviceWorkspace workspace,
                                     std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int selected;
  if (threadIdx.x == 0) {
    selected = 0;
    workspace.selected_mask[system] = 0u;
    if (sequence_is_open(workspace.sequence_active) &&
        atomicAdd(system_errors + system, 0u) == 0u) {
      const std::uint8_t requested = activity.requested_mask[system];
      if (requested > 1u) {
        record_system_error(system_errors, system, device_error,
                            Gfn2ClassicalForceDeviceError::kInvalidActivity);
      } else if (requested == 1u && activity.system_statuses[system] == XTBLOOM_STATUS_SUCCESS) {
        selected = 1;
        workspace.selected_mask[system] = 1u;
      }
    }
  }
  __syncthreads();
  if (selected == 0) {
    return;
  }

  const std::int64_t atom_begin = plan.atom_offsets[system];
  const std::int64_t atom_end = plan.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    bool finite_seed = true;
    for (int axis = 0; axis < 3; ++axis) {
      const double seed = output.forces[coordinate + axis];
      if (!isfinite(seed)) {
        finite_seed = false;
      } else {
        workspace.force_scratch[coordinate + axis] = seed;
        workspace.gradient_scratch[coordinate + axis] = 0.0;
      }
    }
    workspace.coordination_adjoints[atom] = 0.0;
    if (!finite_seed) {
      record_system_error(system_errors, system, device_error,
                          Gfn2ClassicalForceDeviceError::kNonfiniteForceSeed);
    }
  }
}

__device__ bool add_finite_atomic(double* target, double contribution) {
  if (!isfinite(contribution)) {
    return false;
  }
  const double previous = atomic_add_fp64(target, contribution);
  return isfinite(previous) && isfinite(previous + contribution);
}

__global__ void repulsion_gradient_kernel(Gfn2ClassicalForceDevicePlan plan,
                                          const double* positions,
                                          Gfn2ClassicalForceDeviceWorkspace workspace,
                                          std::uint32_t* system_errors,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = system_is_selected(workspace, system_errors, system) ? 1 : 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    const std::int32_t atomic_number = plan.atomic_numbers[atom];
    const std::int64_t coordinate = atom * 3;
    if (atomic_number < 1 || !isfinite(positions[coordinate]) ||
        !isfinite(positions[coordinate + 1]) || !isfinite(positions[coordinate + 2])) {
      atomicExch(&valid, 0);
    } else if (plan.model == XtbModelFlavor::kGfn1) {
      if (plan.repulsion_sqrt_alpha == nullptr || plan.repulsion_effective_charge == nullptr ||
          !(plan.repulsion_sqrt_alpha[atom] > 0.0) ||
          !(plan.repulsion_effective_charge[atom] > 0.0) ||
          !isfinite(plan.repulsion_sqrt_alpha[atom]) ||
          !isfinite(plan.repulsion_effective_charge[atom])) {
        atomicExch(&valid, 0);
      }
    } else if (atomic_number > static_cast<std::int32_t>(parameters::gfn2::kElementCount)) {
      atomicExch(&valid, 0);
    } else {
      const parameters::gfn2::ElementParameters element = g_gfn2_elements[atomic_number - 1];
      if (element.atomic_number != atomic_number || !(element.arep > 0.0) ||
          !(element.zeff > 0.0) || !isfinite(element.arep) || !isfinite(element.zeff)) {
        atomicExch(&valid, 0);
      }
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2ClassicalForceDeviceError::kRepulsionFailure);
    }
    return;
  }

  for (std::int64_t upper = begin + 1 + threadIdx.x; upper < end; upper += blockDim.x) {
    const std::int32_t upper_number = plan.atomic_numbers[upper];
    const parameters::gfn2::ElementParameters upper_element =
        plan.model == XtbModelFlavor::kGfn2 ? g_gfn2_elements[upper_number - 1]
                                            : parameters::gfn2::ElementParameters{};
    const double upper_sqrt_alpha = plan.model == XtbModelFlavor::kGfn1
                                        ? plan.repulsion_sqrt_alpha[upper]
                                        : sqrt(upper_element.arep);
    const std::int64_t upper_coordinate = upper * 3;
    for (std::int64_t lower = begin; lower < upper; ++lower) {
      const std::int64_t lower_coordinate = lower * 3;
      const double dx = positions[upper_coordinate] - positions[lower_coordinate];
      const double dy = positions[upper_coordinate + 1] - positions[lower_coordinate + 1];
      const double dz = positions[upper_coordinate + 2] - positions[lower_coordinate + 2];
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (!isfinite(distance_squared)) {
        atomicExch(&valid, 0);
        continue;
      }
      if (plan.model == XtbModelFlavor::kGfn1) {
        if (distance_squared < 1.0e-12) continue;
      } else if (!(distance_squared > kMinimumDistanceSquared)) {
        atomicExch(&valid, 0);
        continue;
      }
      if (distance_squared > kRepulsionCutoffSquared) {
        continue;
      }
      const std::int32_t lower_number = plan.atomic_numbers[lower];
      const parameters::gfn2::ElementParameters lower_element =
          plan.model == XtbModelFlavor::kGfn2 ? g_gfn2_elements[lower_number - 1]
                                              : parameters::gfn2::ElementParameters{};
      const double distance = sqrt(distance_squared);
      const bool light_pair =
          plan.model == XtbModelFlavor::kGfn2 && upper_number <= 2 && lower_number <= 2;
      const double exponent =
          plan.model == XtbModelFlavor::kGfn1
              ? 1.5
              : (light_pair ? g_gfn2_global.repulsion_klight : g_gfn2_global.repulsion_kexp);
      const double distance_power =
          plan.model == XtbModelFlavor::kGfn1 || !light_pair ? distance * sqrt(distance) : distance;
      const double lower_sqrt_alpha = plan.model == XtbModelFlavor::kGfn1
                                          ? plan.repulsion_sqrt_alpha[lower]
                                          : sqrt(lower_element.arep);
      const double pair_alpha = upper_sqrt_alpha * lower_sqrt_alpha;
      const double pair_charge =
          plan.model == XtbModelFlavor::kGfn1
              ? plan.repulsion_effective_charge[upper] * plan.repulsion_effective_charge[lower]
              : upper_element.zeff * lower_element.zeff;
      const double pair_energy = pair_charge * exp(-pair_alpha * distance_power) / distance;
      const double force_scale =
          (pair_alpha * exponent * distance_power + 1.0) * pair_energy / distance_squared;
      const double force[3] = {force_scale * dx, force_scale * dy, force_scale * dz};
      bool finite_pair = isfinite(pair_energy) && isfinite(force_scale);
      for (int axis = 0; axis < 3; ++axis) {
        /* Existing repulsion primitive publishes force; this composer stores dE/dR. */
        finite_pair =
            add_finite_atomic(workspace.gradient_scratch + upper_coordinate + axis, -force[axis]) &&
            add_finite_atomic(workspace.gradient_scratch + lower_coordinate + axis, force[axis]) &&
            finite_pair;
      }
      if (!finite_pair) {
        atomicExch(&valid, 0);
      }
    }
  }
  __syncthreads();
  if (valid == 0 && threadIdx.x == 0) {
    record_system_error(system_errors, system, device_error,
                        Gfn2ClassicalForceDeviceError::kRepulsionFailure);
  }
}

__global__ void add_native_periodic_gradients_kernel(Gfn2ClassicalForceDevicePlan plan,
                                                     Gfn2ClassicalForceDeviceWorkspace workspace,
                                                     std::uint32_t* system_errors,
                                                     std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!system_is_selected(workspace, system_errors, system)) return;
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    bool finite = true;
    for (int axis = 0; axis < 3; ++axis) {
      const double contribution =
          workspace.native_short_range_workspace.repulsion_gradients[coordinate + axis] +
          workspace.native_ewald_workspace.gradients[coordinate + axis] +
          workspace.native_multipole_workspace.gradients[coordinate + axis];
      const double updated = workspace.gradient_scratch[coordinate + axis] + contribution;
      if (!isfinite(contribution) || !isfinite(updated)) {
        finite = false;
      } else {
        workspace.gradient_scratch[coordinate + axis] = updated;
      }
    }
    /* Native D4 contracts its own D4-CN adjoint internally because its CN
     * model differs from the periodic GFN2 coordination model.  Only the
     * multipole radius adjoint belongs in the standard periodic CN VJP. */
    const double coordination = workspace.native_multipole_workspace.coordination_adjoint[atom];
    const double updated_coordination = workspace.coordination_adjoints[atom] + coordination;
    if (!isfinite(coordination) || !isfinite(updated_coordination))
      finite = false;
    else
      workspace.coordination_adjoints[atom] = updated_coordination;
    if (!finite) {
      record_system_error(system_errors, system, device_error,
                          Gfn2ClassicalForceDeviceError::kNativeMultipoleFailure);
    }
  }
}

/* Contract the stationary AES2 radius adjoint through the image-aware GFN2
 * coordination model.  The molecular geometry VJP cannot be reused here: its
 * pair cache excludes lattice images and therefore misses self-image and
 * wrapped-coordinate contributions.  One thread owns each ragged peer to
 * keep the lower-triangular translation order deterministic and to stage the
 * complete peer result before touching the common gradient accumulator. */
__global__ void add_native_periodic_coordination_vjp_kernel(
    Gfn2ClassicalForceDevicePlan plan, Gfn2ClassicalForceDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  constexpr double kCutoff = 25.0;
  constexpr double kCutoffSquared = kCutoff * kCutoff;
  constexpr double kMinimumDistanceSquared = 1.0e-12;
  constexpr double kFirstSteepness = 10.0;
  constexpr double kSecondSteepness = 20.0;
  constexpr double kSecondRadiusShift = 2.0;
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || system >= plan.batch_size ||
      !system_is_selected(workspace, system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = plan.atom_offsets[system];
  const std::int64_t atom_end = plan.atom_offsets[system + 1];
  const std::int64_t translation_begin =
      plan.native_short_range_batch.topology.translation_offsets[system];
  const std::int64_t translation_end =
      plan.native_short_range_batch.topology.translation_offsets[system + 1];
  double* const staged = workspace.geometry_workspace.gradient_scratch;
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    staged[atom * 3] = 0.0;
    staged[atom * 3 + 1] = 0.0;
    staged[atom * 3 + 2] = 0.0;
  }
  /* The short-range wrapped view belongs to the preprocessing transaction and
   * is not refreshed by this stationary force pass.  Multipole evaluation just
   * above performs the same canonical wrapping with the force geometry, so use
   * its freshly written view for the CN adjoint.  Reusing the stale short-range
   * scratch would turn an uninitialized/zero image into a coincident pair. */
  const double* const wrapped = workspace.native_multipole_workspace.wrapped_positions;
  const double* const radii = plan.geometry_batch.covalent_radii;
  const double* const adjoints = workspace.coordination_adjoints;
  for (std::int64_t first = atom_begin; first < atom_end; ++first) {
    const double* const center = wrapped + first * 3;
    for (std::int64_t second = atom_begin; second <= first; ++second) {
      const double* const image = wrapped + second * 3;
      const double radius = radii[first] + radii[second];
      if (!(radius > 0.0) || !isfinite(radius)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2ClassicalForceDeviceError::kNativeMultipoleFailure);
        return;
      }
      const double shifted_radius = radius + kSecondRadiusShift;
      const double adjoint = adjoints[first] + (first == second ? 0.0 : adjoints[second]);
      if (!isfinite(adjoint)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2ClassicalForceDeviceError::kNativeMultipoleFailure);
        return;
      }
      for (std::int64_t translation = translation_begin; translation < translation_end;
           ++translation) {
        const auto& value = plan.native_short_range_batch.topology.translations[translation];
        if (first == second && value.index[0] == 0 && value.index[1] == 0 && value.index[2] == 0 &&
            value.cartesian[0] == 0.0 && value.cartesian[1] == 0.0 && value.cartesian[2] == 0.0) {
          continue;
        }
        double vector[3]{};
        bool within_cutoff = true;
        for (int axis = 0; axis < 3; ++axis) {
          vector[axis] = image[axis] + value.cartesian[axis] - center[axis];
          if (!isfinite(vector[axis]) || fabs(vector[axis]) > kCutoff) {
            /* The native topology is a deliberately conservative image
             * superset (it also serves D4).  Images outside the CN cutoff
             * are simply absent from this VJP; zeroing the vector here would
             * turn an out-of-range image into a false coincident pair. */
            within_cutoff = false;
            break;
          }
        }
        if (!within_cutoff) continue;
        const double distance_squared =
            fma(vector[0], vector[0], fma(vector[1], vector[1], vector[2] * vector[2]));
        if (!isfinite(distance_squared) || distance_squared > kCutoffSquared) continue;
        if (distance_squared < kMinimumDistanceSquared) {
          record_system_error(system_errors, system, device_error,
                              Gfn2ClassicalForceDeviceError::kNativeMultipoleFailure);
          return;
        }
        const double distance = sqrt(distance_squared);
        const auto logistic = [](double argument) {
          const double exponential = argument >= 0.0 ? exp(-argument) : exp(argument);
          return argument >= 0.0 ? 1.0 / (1.0 + exponential) : exponential / (1.0 + exponential);
        };
        const double first_argument = kFirstSteepness * (radius / distance - 1.0);
        const double second_argument = kSecondSteepness * (shifted_radius / distance - 1.0);
        const double first_value = logistic(first_argument);
        const double second_value = logistic(second_argument);
        const double derivative = first_value * (1.0 - first_value) *
                                      (-kFirstSteepness * radius / distance_squared) *
                                      second_value +
                                  first_value * second_value * (1.0 - second_value) *
                                      (-kSecondSteepness * shifted_radius / distance_squared);
        const double scale = -adjoint * derivative / distance;
        if (!isfinite(derivative) || !isfinite(scale)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2ClassicalForceDeviceError::kNativeMultipoleFailure);
          return;
        }
        if (first != second) {
          for (int axis = 0; axis < 3; ++axis) {
            const double contribution = scale * vector[axis];
            staged[first * 3 + axis] += contribution;
            staged[second * 3 + axis] -= contribution;
          }
        }
      }
    }
  }
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    for (int axis = 0; axis < 3; ++axis) {
      if (!isfinite(staged[atom * 3 + axis])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2ClassicalForceDeviceError::kNativeMultipoleFailure);
        return;
      }
    }
  }
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    for (int axis = 0; axis < 3; ++axis) {
      const std::int64_t coordinate = atom * 3 + axis;
      const double updated = workspace.gradient_scratch[coordinate] + staged[coordinate];
      if (!isfinite(updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2ClassicalForceDeviceError::kNativeMultipoleFailure);
        return;
      }
      workspace.gradient_scratch[coordinate] = updated;
    }
  }
}

__global__ void es2_topology_preflight_kernel(Gfn2ClassicalForceDevicePlan plan,
                                              std::uint32_t* device_error,
                                              std::uint32_t* sequence_active) {
  if (blockIdx.x != 0 || !sequence_is_open(sequence_active)) {
    return;
  }
  const Gfn2ES2DeviceBatch batch = plan.es2_batch;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = batch.atom_offsets[0] == 0 &&
            batch.atom_offsets[batch.batch_size] == batch.total_atoms &&
            batch.batch_shell_offsets[0] == 0 &&
            batch.batch_shell_offsets[batch.batch_size] == batch.total_shells &&
            batch.matrix_offsets[0] == 0 &&
            batch.matrix_offsets[batch.batch_size] == batch.total_matrix_elements;
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t shell_begin = batch.batch_shell_offsets[system];
    const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
    const std::int64_t matrix_begin = batch.matrix_offsets[system];
    const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
    const std::int64_t shells = shell_end - shell_begin;
    const bool square_ok = shells >= 0 && (shells == 0 || shells <= kInt64Maximum / shells);
    bool system_valid = atom_begin >= 0 && atom_begin <= atom_end &&
                        atom_end <= batch.total_atoms && shell_begin >= 0 &&
                        shell_begin <= shell_end && shell_end <= batch.total_shells &&
                        matrix_begin >= 0 && matrix_begin <= matrix_end &&
                        matrix_end <= batch.total_matrix_elements && square_ok &&
                        matrix_end - matrix_begin == shells * shells;
    if (system_valid) {
      system_valid = batch.atom_shell_offsets[atom_begin] == shell_begin &&
                     batch.atom_shell_offsets[atom_end] == shell_end;
    }
    if (!system_valid) {
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t atom = threadIdx.x; atom < batch.total_atoms; atom += blockDim.x) {
    const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
    const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
    if (shell_begin < 0 || shell_begin >= shell_end || shell_end > batch.total_shells) {
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t shell = threadIdx.x; shell < batch.total_shells; shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    if (atom < 0 || atom >= batch.total_atoms || shell < batch.atom_shell_offsets[atom] ||
        shell >= batch.atom_shell_offsets[atom + 1] || !(batch.shell_hardness[shell] > 0.0) ||
        !isfinite(batch.shell_hardness[shell])) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && valid == 0) {
    record_plan_error(device_error, sequence_active,
                      Gfn2ClassicalForceDeviceError::kInvalidTopology);
  }
}

__global__ void es2_gradient_kernel(Gfn2ClassicalForceDevicePlan plan,
                                    Gfn2ClassicalForceDeviceInput input,
                                    Gfn2ClassicalForceDeviceWorkspace workspace,
                                    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = system_is_selected(workspace, system_errors, system) ? 1 : 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  const Gfn2ES2DeviceBatch batch = plan.es2_batch;
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t shells = shell_end - shell_begin;
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(input.positions[coordinate]) || !isfinite(input.positions[coordinate + 1]) ||
        !isfinite(input.positions[coordinate + 2])) {
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    if (!isfinite(input.shell_charges[shell])) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2ClassicalForceDeviceError::kES2Failure);
    }
    return;
  }

  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    double contribution[3] = {0.0, 0.0, 0.0};
    bool finite_result = true;
    for (std::int64_t peer = atom_begin; finite_result && peer < atom_end; ++peer) {
      if (peer == atom) {
        continue;
      }
      const std::int64_t upper = atom > peer ? atom : peer;
      const std::int64_t lower = atom > peer ? peer : atom;
      const std::int64_t upper_shell_begin = batch.atom_shell_offsets[upper];
      const std::int64_t upper_shell_end = batch.atom_shell_offsets[upper + 1];
      const std::int64_t lower_shell_begin = batch.atom_shell_offsets[lower];
      const std::int64_t lower_shell_end = batch.atom_shell_offsets[lower + 1];
      double weighted = 0.0;
      for (std::int64_t upper_shell = upper_shell_begin; upper_shell < upper_shell_end;
           ++upper_shell) {
        for (std::int64_t lower_shell = lower_shell_begin; lower_shell < lower_shell_end;
             ++lower_shell) {
          const std::int64_t matrix =
              matrix_begin + (upper_shell - shell_begin) * shells + lower_shell - shell_begin;
          const double kernel = plan.es2_cache.coulomb_matrix[matrix];
          double term = input.shell_charges[upper_shell] * kernel;
          term *= input.shell_charges[lower_shell];
          term *= kernel;
          term *= kernel;
          const double updated = weighted + term;
          if (!(kernel > 0.0) || !isfinite(kernel) || !isfinite(term) || !isfinite(updated)) {
            finite_result = false;
            break;
          }
          weighted = updated;
        }
      }
      const double sign = atom == upper ? -1.0 : 1.0;
      for (int axis = 0; finite_result && axis < 3; ++axis) {
        const double displacement =
            input.positions[upper * 3 + axis] - input.positions[lower * 3 + axis];
        const double term = sign * weighted * displacement;
        const double updated = contribution[axis] + term;
        if (!isfinite(displacement) || !isfinite(term) || !isfinite(updated)) {
          finite_result = false;
        } else {
          contribution[axis] = updated;
        }
      }
    }
    const std::int64_t coordinate = atom * 3;
    for (int axis = 0; finite_result && axis < 3; ++axis) {
      const double updated = workspace.gradient_scratch[coordinate + axis] + contribution[axis];
      if (!isfinite(updated)) {
        finite_result = false;
      } else {
        workspace.gradient_scratch[coordinate + axis] = updated;
      }
    }
    if (!finite_result) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0 && threadIdx.x == 0) {
    record_system_error(system_errors, system, device_error,
                        Gfn2ClassicalForceDeviceError::kES2Failure);
  }
}

__global__ void prepare_primitive_stage_kernel(std::int64_t batch_size,
                                               Gfn2ClassicalForceDeviceWorkspace workspace,
                                               const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size) {
    const bool active = sequence_is_open(workspace.sequence_active) &&
                        workspace.selected_mask[system] == 1u &&
                        atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) == 0u;
    workspace.primitive_system_errors[system] = active ? 0u : kPrimitiveInactiveMarker;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0 && !sequence_is_open(workspace.sequence_active)) {
    *workspace.primitive_device_error = kPrimitiveInactiveMarker;
  }
}

__global__ void merge_primitive_stage_kernel(std::int64_t batch_size,
                                             Gfn2ClassicalForceDeviceWorkspace workspace,
                                             Gfn2ClassicalForceDeviceError mapped_error,
                                             bool device_code_is_plan_only,
                                             std::uint32_t* system_errors,
                                             std::uint32_t* device_error) {
  __shared__ int device_code_localized;
  __shared__ std::uint32_t primitive_device_code;
  if (threadIdx.x == 0) {
    device_code_localized = 0;
    primitive_device_code = atomicAdd(workspace.primitive_device_error, 0u);
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < batch_size; system += blockDim.x) {
    const std::uint32_t raw = workspace.primitive_system_errors[system];
    if (raw != 0u && raw != kPrimitiveInactiveMarker) {
    }
    if (raw != 0u && raw != kPrimitiveInactiveMarker && workspace.selected_mask[system] == 1u) {
      record_system_error(system_errors, system, device_error, mapped_error);
      if (raw == primitive_device_code) {
        atomicExch(&device_code_localized, 1);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && primitive_device_code != 0u &&
      primitive_device_code != kPrimitiveInactiveMarker &&
      (device_code_is_plan_only || device_code_localized == 0)) {
    record_plan_error(device_error, workspace.sequence_active,
                      Gfn2ClassicalForceDeviceError::kInvalidTopology);
  }
}

__global__ void finalize_force_kernel(Gfn2ClassicalForceDevicePlan plan,
                                      Gfn2ClassicalForceDeviceWorkspace workspace,
                                      std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!system_is_selected(workspace, system_errors, system)) {
    return;
  }
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    bool finite_result = true;
    for (int axis = 0; axis < 3; ++axis) {
      const double updated = workspace.force_scratch[coordinate + axis] -
                             workspace.gradient_scratch[coordinate + axis];
      if (!isfinite(updated)) {
        finite_result = false;
      } else {
        workspace.force_scratch[coordinate + axis] = updated;
      }
    }
    if (!finite_result) {
      record_system_error(system_errors, system, device_error,
                          Gfn2ClassicalForceDeviceError::kNonfiniteForceArithmetic);
    }
  }
}

__global__ void publish_force_kernel(Gfn2ClassicalForceDevicePlan plan,
                                     Gfn2ClassicalForceDeviceOutput output,
                                     Gfn2ClassicalForceDeviceWorkspace workspace,
                                     const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!system_is_selected(workspace, system_errors, system)) {
    return;
  }
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    output.forces[coordinate] = workspace.force_scratch[coordinate];
    output.forces[coordinate + 1] = workspace.force_scratch[coordinate + 1];
    output.forces[coordinate + 2] = workspace.force_scratch[coordinate + 2];
  }
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

cudaError_t prepare_primitive_stage(const Gfn2ClassicalForceDevicePlan& plan,
                                    const Gfn2ClassicalForceDeviceWorkspace& workspace,
                                    const std::uint32_t* system_errors,
                                    cudaStream_t stream) noexcept {
  cudaError_t status = cudaMemsetAsync(workspace.primitive_device_error, 0,
                                       sizeof(*workspace.primitive_device_error), stream);
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((plan.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  prepare_primitive_stage_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(plan.batch_size,
                                                                          workspace, system_errors);
  return check_launch();
}

cudaError_t merge_primitive_stage(const Gfn2ClassicalForceDevicePlan& plan,
                                  const Gfn2ClassicalForceDeviceWorkspace& workspace,
                                  Gfn2ClassicalForceDeviceError mapped_error,
                                  bool device_code_is_plan_only, std::uint32_t* system_errors,
                                  std::uint32_t* device_error, cudaStream_t stream) noexcept {
  merge_primitive_stage_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      plan.batch_size, workspace, mapped_error, device_code_is_plan_only, system_errors,
      device_error);
  return check_launch();
}

}  // namespace

cudaError_t reset_gfn2_classical_force_device_errors_cuda(std::int64_t batch_size,
                                                          std::uint32_t* system_errors,
                                                          std::uint32_t* device_error,
                                                          cudaStream_t stream) noexcept {
  if (batch_size <= 0 || !aligned_pointer(system_errors) || !aligned_pointer(device_error) ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                     sizeof(*system_errors))) {
    return cudaErrorInvalidValue;
  }
  AddressRange systems;
  AddressRange device;
  if (!make_range(system_errors, batch_size, &systems) || !make_range(device_error, 1, &device) ||
      ranges_overlap(systems, device)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream)
                               : status;
}

static cudaError_t add_classical_forces_impl(
    const Gfn2ClassicalForceDevicePlan& plan, const Gfn2ForceDeviceActivity& activity,
    const Gfn2ClassicalForceDeviceInput& input, const Gfn2ClassicalForceDeviceOutput& output,
    const Gfn2ClassicalForceDeviceWorkspace& workspace,
    const Gfn2GeometryEpochDevice* geometry_epoch, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_common_descriptors(plan, activity, input, output, workspace, geometry_epoch,
                                   system_errors, device_error)) {
    return cudaErrorInvalidValue;
  }

  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  common_topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(plan, device_error,
                                                                       workspace.sequence_active);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  gate_and_seed_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0, stream>>>(
      plan, activity, output, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  const bool native_periodic = plan.native_short_range_batch.topology.plan_token != 0u;
  const bool native_d4 =
      native_periodic &&
      (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4TwoBody) ||
       component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4ATM));
  if (native_periodic) {
    /* Re-evaluate the converged q/d/Q state with derivatives enabled.  SCC
     * iteration deliberately requests only potentials/energies; doing this
     * after the terminal SCC status is known keeps the force reverse pass
     * stationary and lets failed peers drop out before any caller output is
     * touched. */
    status = prepare_primitive_stage(plan, workspace, system_errors, stream);
    if (status != cudaSuccess) return status;
    auto ewald = plan.native_ewald_batch;
    ewald.active_mask = workspace.selected_mask;
    ewald.active_mask_elements = plan.batch_size;
    status = evaluate_gfn2_native_periodic_ewald_cuda(
        ewald, workspace.native_ewald_workspace, nullptr, nullptr, nullptr, nullptr, nullptr,
        workspace.primitive_system_errors, workspace.primitive_device_error, stream);
    if (status != cudaSuccess) return status;
    status =
        merge_primitive_stage(plan, workspace, Gfn2ClassicalForceDeviceError::kNativeEwaldFailure,
                              false, system_errors, device_error, stream);
    if (status != cudaSuccess) return status;

    status = prepare_primitive_stage(plan, workspace, system_errors, stream);
    if (status != cudaSuccess) return status;
    auto multipole = plan.native_multipole_batch;
    multipole.active_mask = workspace.selected_mask;
    multipole.active_mask_elements = plan.batch_size;
    status = evaluate_gfn2_native_periodic_multipole_cuda(
        multipole, workspace.native_multipole_workspace, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, workspace.primitive_system_errors,
        workspace.primitive_device_error, stream);
    if (status != cudaSuccess) return status;
    status = merge_primitive_stage(plan, workspace,
                                   Gfn2ClassicalForceDeviceError::kNativeMultipoleFailure, false,
                                   system_errors, device_error, stream);
    if (status != cudaSuccess) return status;

    if (native_d4) {
      /* Native D4 owns both the image-aware two-body/ATM derivatives and the
       * corresponding CN adjoint.  Its gradient is accumulated directly into
       * the common dE/dR scratch; only the CN adjoint is folded by the helper
       * kernel below before the periodic CN reverse pass. */
      status = prepare_primitive_stage(plan, workspace, system_errors, stream);
      if (status != cudaSuccess) return status;
      auto d4 = plan.native_d4_batch;
      d4.active_mask = workspace.selected_mask;
      d4.active_mask_elements = plan.batch_size;
      d4.coordination_numbers = workspace.native_d4_workspace.coordination;
      d4.coordination_number_elements = plan.total_atoms;
      status = evaluate_gfn2_native_periodic_d4_coordination_cuda(
          d4, workspace.native_d4_workspace, workspace.native_d4_workspace.coordination,
          workspace.primitive_system_errors, workspace.primitive_device_error, stream);
      if (status != cudaSuccess) return status;
      status =
          merge_primitive_stage(plan, workspace, Gfn2ClassicalForceDeviceError::kNativeD4Failure,
                                false, system_errors, device_error, stream);
      if (status != cudaSuccess) return status;

      status = prepare_primitive_stage(plan, workspace, system_errors, stream);
      if (status != cudaSuccess) return status;
      status = add_gfn2_native_periodic_d4_gradients_cuda(
          d4, workspace.native_d4_workspace, workspace.gradient_scratch,
          workspace.native_d4_workspace.strain, workspace.primitive_system_errors,
          workspace.primitive_device_error, stream);
      if (status != cudaSuccess) return status;
      status =
          merge_primitive_stage(plan, workspace, Gfn2ClassicalForceDeviceError::kNativeD4Failure,
                                false, system_errors, device_error, stream);
      if (status != cudaSuccess) return status;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kRepulsion) &&
      !native_periodic) {
    repulsion_gradient_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0,
                                stream>>>(plan, input.positions, workspace, system_errors,
                                          device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kES2) &&
      !native_periodic) {
    es2_topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(plan, device_error,
                                                                      workspace.sequence_active);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
    es2_gradient_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0,
                          stream>>>(plan, input, workspace, system_errors, device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kAES2) &&
      !native_periodic) {
    {
      status = prepare_primitive_stage(plan, workspace, system_errors, stream);
      if (status != cudaSuccess) {
        return status;
      }
      status = add_gfn2_aes2_vjp_cuda(
          plan.aes2_batch, plan.aes2_cache, input.positions, input.coordination_numbers,
          plan.geometry_generation, input.atomic_charges, input.atomic_dipoles,
          input.atomic_quadrupoles, workspace.gradient_scratch, workspace.coordination_adjoints,
          workspace.aes2_workspace, workspace.primitive_system_errors,
          workspace.primitive_device_error, stream);
      if (status != cudaSuccess) {
        return status;
      }
      status = merge_primitive_stage(plan, workspace, Gfn2ClassicalForceDeviceError::kAES2Failure,
                                     false, system_errors, device_error, stream);
      if (status != cudaSuccess) {
        return status;
      }
    }
    status = prepare_primitive_stage(plan, workspace, system_errors, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = geometry_epoch == nullptr
                 ? add_gfn2_coordination_vjp_cuda(
                       plan.geometry_batch, plan.geometry_cache, plan.geometry_generation,
                       workspace.coordination_adjoints, workspace.gradient_scratch,
                       workspace.geometry_workspace, workspace.primitive_system_errors,
                       workspace.primitive_device_error, stream)
                 : add_gfn2_coordination_vjp_cuda(
                       plan.geometry_batch, plan.geometry_cache, *geometry_epoch,
                       workspace.coordination_adjoints, workspace.gradient_scratch,
                       workspace.geometry_workspace, workspace.primitive_system_errors,
                       workspace.primitive_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = merge_primitive_stage(plan, workspace,
                                   Gfn2ClassicalForceDeviceError::kAES2CoordinationFailure, false,
                                   system_errors, device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (native_periodic) {
    add_native_periodic_gradients_kernel<<<static_cast<unsigned int>(plan.batch_size),
                                           kThreadsPerBlock, 0, stream>>>(
        plan, workspace, system_errors, device_error);
    status = check_launch();
    if (status != cudaSuccess) return status;
    /* Native multipole radii depend on CN.  Contract that adjoint through the
     * periodic image topology; the molecular pair cache omits self-images. */
    add_native_periodic_coordination_vjp_kernel<<<static_cast<unsigned int>(plan.batch_size),
                                                  kThreadsPerBlock, 0, stream>>>(
        plan, workspace, system_errors, device_error);
    status = check_launch();
    if (status != cudaSuccess) return status;
  }

  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4TwoBody) &&
      !native_d4) {
    status = prepare_primitive_stage(plan, workspace, system_errors, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = geometry_epoch == nullptr
                 ? add_gfn2_d4_two_body_gradient_pairlist_cuda(
                       plan.d4_batch, plan.d4_parameters, plan.geometry_generation,
                       plan.d4_pairlist_cache, input.atomic_charges, workspace.gradient_scratch,
                       workspace.d4_workspace, workspace.primitive_device_error, stream)
                 : add_gfn2_d4_two_body_gradient_pairlist_cuda(
                       plan.d4_batch, plan.d4_parameters, *geometry_epoch, plan.d4_pairlist_cache,
                       input.atomic_charges, workspace.gradient_scratch, workspace.d4_workspace,
                       workspace.primitive_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status =
        merge_primitive_stage(plan, workspace, Gfn2ClassicalForceDeviceError::kD4TwoBodyFailure,
                              true, system_errors, device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4ATM) &&
      !native_d4) {
    status = prepare_primitive_stage(plan, workspace, system_errors, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = geometry_epoch == nullptr
                 ? add_gfn2_d4_atm_gradient_pairlist_cuda(
                       plan.d4_batch, plan.d4_parameters, plan.geometry_generation,
                       plan.d4_pairlist_cache, workspace.gradient_scratch, workspace.d4_workspace,
                       workspace.primitive_device_error, stream)
                 : add_gfn2_d4_atm_gradient_pairlist_cuda(
                       plan.d4_batch, plan.d4_parameters, *geometry_epoch, plan.d4_pairlist_cache,
                       workspace.gradient_scratch, workspace.d4_workspace,
                       workspace.primitive_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = merge_primitive_stage(plan, workspace, Gfn2ClassicalForceDeviceError::kD4ATMFailure,
                                   true, system_errors, device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (plan.model == XtbModelFlavor::kGfn1) {
    status = prepare_primitive_stage(plan, workspace, system_errors, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = add_gfn1_classical_corrections_cuda(
        plan.gfn1_correction, input.positions, input.coordination_numbers, nullptr, nullptr,
        workspace.gradient_scratch, workspace.gfn1_correction, workspace.primitive_system_errors,
        workspace.primitive_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = merge_primitive_stage(plan, workspace,
                                   Gfn2ClassicalForceDeviceError::kGfn1CorrectionFailure, false,
                                   system_errors, device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
  }

  finalize_force_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0,
                          stream>>>(plan, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_force_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0, stream>>>(
      plan, output, workspace, system_errors);
  return check_launch();
}

cudaError_t add_gfn2_classical_forces_cuda(
    const Gfn2ClassicalForceDevicePlan& plan, const Gfn2ForceDeviceActivity& activity,
    const Gfn2ClassicalForceDeviceInput& input, const Gfn2ClassicalForceDeviceOutput& output,
    const Gfn2ClassicalForceDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return add_classical_forces_impl(plan, activity, input, output, workspace, nullptr, system_errors,
                                   device_error, stream);
}

cudaError_t add_gfn2_classical_forces_cuda(
    const Gfn2ClassicalForceDevicePlan& plan, const Gfn2ForceDeviceActivity& activity,
    const Gfn2ClassicalForceDeviceInput& input, const Gfn2ClassicalForceDeviceOutput& output,
    const Gfn2ClassicalForceDeviceWorkspace& workspace,
    const Gfn2GeometryEpochDevice& geometry_epoch, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  return add_classical_forces_impl(plan, activity, input, output, workspace, &geometry_epoch,
                                   system_errors, device_error, stream);
}

}  // namespace xtbloom::detail::cuda
