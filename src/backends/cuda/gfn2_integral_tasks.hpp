#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_INTEGRAL_TASKS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_INTEGRAL_TASKS_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_integrals.cuh"
#include "model/common/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::cuda {

/* Ordered shell/primitive signature ledger retained with the immutable plan.
 * Angular momenta are deliberately ordered rather than normalized so the
 * evidence describes the exact bra/ket work seen by each consumer. */
struct Gfn2IntegralPrimitiveSignatureAccounting {
  std::uint8_t bra_angular_momentum = 0u;
  std::uint8_t ket_angular_momentum = 0u;
  std::int64_t bra_primitives = 0;
  std::int64_t ket_primitives = 0;
  std::int64_t forward_tasks = 0;
  std::int64_t h0_tasks = 0;
  std::int64_t force_tasks = 0;
  std::uint64_t forward_primitive_work = 0u;
  std::uint64_t h0_primitive_work = 0u;
  std::uint64_t force_primitive_work = 0u;
};

/* Host-retained accounting for the setup-time canonical task construction.
 * Shell classes use the stable order ss, sp, sd, pp, pd, dd. */
struct Gfn2IntegralTaskAccounting {
  std::int64_t capacity_slots = 0;
  std::int64_t dense_live_pairs = 0;
  std::int64_t capacity_tail_slots = 0;
  std::int64_t forward_symmetry_exits = 0;
  std::int64_t force_ineligibility_exits = 0;
  std::array<std::int64_t, 6> forward_shell_classes{};
  std::array<std::int64_t, 6> h0_shell_classes{};
  std::array<std::int64_t, 6> force_shell_classes{};
  std::uint64_t forward_primitive_work = 0u;
  std::uint64_t h0_primitive_work = 0u;
  std::uint64_t force_primitive_work = 0u;
  std::vector<Gfn2IntegralPrimitiveSignatureAccounting> primitive_signatures;
};

/* Very small domains do not amortize the extra packed/generic kernel launches.
 * The threshold is topology-only so default selection remains stable across
 * SCC activity, direct execution, and Graph replay. Explicit compact/legacy
 * environment overrides bypass this automatic choice for ABBA diagnostics. */
inline constexpr std::int64_t kGfn2IntegralCompactMinimumDensePairs = 256;

inline bool prefer_gfn2_compact_integral_tasks(
    const Gfn2IntegralTaskAccounting& accounting) noexcept {
  return accounting.dense_live_pairs >= kGfn2IntegralCompactMinimumDensePairs;
}

/* Immutable setup-owned task domains. Each vector preserves the corresponding
 * legacy system-major, row-major shell-pair order after removing only the
 * consumer's no-op entries. Executing the packed ss force bucket separately
 * changes only the cross-bucket order of atomic gradient additions; compact
 * force parity and finite-difference tests qualify that rounding difference. */
struct Gfn2IntegralHostTaskDomains {
  std::vector<Gfn2IntegralShellPairTask> forward_generic;
  std::vector<Gfn2IntegralShellPairTask> forward_ss;
  std::vector<Gfn2IntegralShellPairTask> h0_generic;
  std::vector<Gfn2IntegralShellPairTask> h0_ss;
  std::vector<Gfn2IntegralShellPairTask> force_generic;
  std::vector<Gfn2IntegralShellPairTask> force_ss;
  Gfn2IntegralTaskAccounting accounting{};
};

namespace integral_tasks_detail {

inline bool checked_add(std::uint64_t first, std::uint64_t second, std::uint64_t& result) noexcept {
  if (second > std::numeric_limits<std::uint64_t>::max() - first) return false;
  result = first + second;
  return true;
}

inline bool checked_multiply(std::int64_t first, std::int64_t second,
                             std::int64_t& result) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  result = first * second;
  return true;
}

inline std::size_t shell_class(std::uint8_t first, std::uint8_t second) noexcept {
  const std::uint8_t lower = std::min(first, second);
  const std::uint8_t upper = std::max(first, second);
  if (lower == 0u) return static_cast<std::size_t>(upper);
  return lower == 1u ? static_cast<std::size_t>(upper + 2u) : 5u;
}

inline bool add_primitive_work(std::int64_t first, std::int64_t second,
                               std::uint64_t& total) noexcept {
  std::int64_t product = 0;
  if (!checked_multiply(first, second, product)) return false;
  std::uint64_t updated = 0u;
  if (!checked_add(total, static_cast<std::uint64_t>(product), updated)) return false;
  total = updated;
  return true;
}

inline Gfn2IntegralPrimitiveSignatureAccounting& primitive_signature(
    std::vector<Gfn2IntegralPrimitiveSignatureAccounting>& signatures, std::uint8_t bra_l,
    std::uint8_t ket_l, std::int64_t bra_primitives, std::int64_t ket_primitives) {
  const auto existing =
      std::find_if(signatures.begin(), signatures.end(), [&](const auto& signature) {
        return signature.bra_angular_momentum == bra_l && signature.ket_angular_momentum == ket_l &&
               signature.bra_primitives == bra_primitives &&
               signature.ket_primitives == ket_primitives;
      });
  if (existing != signatures.end()) return *existing;
  signatures.push_back({bra_l, ket_l, bra_primitives, ket_primitives, 0, 0, 0, 0u, 0u, 0u});
  return signatures.back();
}

}  // namespace integral_tasks_detail

/* Build the exact accepted domains from immutable basis topology. Dense H0
 * scale storage is intentionally retained; the compact H0 tasks reference its
 * original row-major local-pair index so numerical meaning is unchanged. */
inline xtbloom_status_t make_gfn2_integral_task_domains(
    const common::BasisPlan& basis, const std::vector<std::int64_t>& shell_pair_offsets,
    Gfn2IntegralHostTaskDomains& domains, std::string& error) {
  using integral_tasks_detail::add_primitive_work;
  using integral_tasks_detail::checked_multiply;
  using integral_tasks_detail::primitive_signature;
  using integral_tasks_detail::shell_class;

  const auto expected_batch_entries =
      basis.batch_size >= 0 ? static_cast<std::uint64_t>(basis.batch_size) + 1u : 0u;
  if (basis.batch_size <= 0 || basis.total_shells <= 0 ||
      static_cast<std::uint64_t>(basis.batch_shell_offsets.size()) != expected_batch_entries ||
      static_cast<std::uint64_t>(shell_pair_offsets.size()) != expected_batch_entries ||
      basis.batch_shell_offsets.front() != 0 || shell_pair_offsets.front() != 0 ||
      basis.batch_shell_offsets.back() != basis.total_shells ||
      basis.shell_to_atom.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.angular_momenta.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.shell_primitive_offsets.size() != static_cast<std::size_t>(basis.total_shells) + 1u) {
    error = "CUDA integral task setup received inconsistent basis topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  Gfn2IntegralHostTaskDomains created;
  std::int64_t maximum_shells = 0;
  for (std::int64_t system = 0; system < basis.batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    const std::int64_t shell_begin = basis.batch_shell_offsets[index];
    const std::int64_t shell_end = basis.batch_shell_offsets[index + 1u];
    if (shell_begin < 0 || shell_begin > shell_end || shell_end > basis.total_shells) {
      error = "CUDA integral task setup found invalid batch shell offsets";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::int64_t shells = shell_end - shell_begin;
    std::int64_t pairs = 0;
    const std::int64_t pair_begin = shell_pair_offsets[index];
    const std::int64_t pair_end = shell_pair_offsets[index + 1u];
    if (!checked_multiply(shells, shells, pairs) || pair_begin < 0 || pair_end < pair_begin ||
        pair_end - pair_begin != pairs) {
      error = "CUDA integral task setup found invalid dense shell-pair offsets";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    maximum_shells = std::max(maximum_shells, shells);
  }

  std::int64_t capacity_per_system = 0;
  if (!checked_multiply(maximum_shells, maximum_shells, capacity_per_system) ||
      !checked_multiply(capacity_per_system, basis.batch_size, created.accounting.capacity_slots) ||
      shell_pair_offsets.back() < 0 ||
      shell_pair_offsets.back() > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      basis.total_shells > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
    error = "CUDA integral task setup exceeds the supported launch domain";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  created.accounting.dense_live_pairs = shell_pair_offsets.back();
  created.accounting.capacity_tail_slots =
      created.accounting.capacity_slots - created.accounting.dense_live_pairs;

  try {
    const std::size_t dense_pairs = static_cast<std::size_t>(shell_pair_offsets.back());
    created.h0_generic.reserve(dense_pairs);
    created.h0_ss.reserve(dense_pairs);
    created.forward_generic.reserve(dense_pairs / 2u + 1u);
    created.forward_ss.reserve(dense_pairs / 2u + 1u);
    created.force_generic.reserve(dense_pairs / 2u + 1u);
    created.force_ss.reserve(dense_pairs / 2u + 1u);

    for (std::int64_t system = 0; system < basis.batch_size; ++system) {
      const std::size_t system_index = static_cast<std::size_t>(system);
      const std::int64_t shell_begin = basis.batch_shell_offsets[system_index];
      const std::int64_t shell_end = basis.batch_shell_offsets[system_index + 1u];
      const std::int64_t shells = shell_end - shell_begin;
      std::int64_t pairs = 0;
      if (!checked_multiply(shells, shells, pairs)) {
        /* Keep this construction pass independently overflow-safe instead of
         * relying on the earlier topology-validation pass remaining paired
         * with this loop during future maintenance. */
        error = "CUDA integral task setup shell-pair count overflowed";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      for (std::int64_t local_pair = 0; local_pair < pairs; ++local_pair) {
        const std::int64_t bra_shell = shell_begin + local_pair / shells;
        const std::int64_t ket_shell = shell_begin + local_pair % shells;
        const std::int64_t bra_atom = basis.shell_to_atom[static_cast<std::size_t>(bra_shell)];
        const std::int64_t ket_atom = basis.shell_to_atom[static_cast<std::size_t>(ket_shell)];
        if (bra_atom < 0 || ket_atom < 0 || bra_atom >= basis.total_atoms ||
            ket_atom >= basis.total_atoms) {
          error = "CUDA integral task setup found an invalid shell-to-atom map";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }

        const auto task = Gfn2IntegralShellPairTask{
            static_cast<std::uint32_t>(system), static_cast<std::uint32_t>(local_pair),
            static_cast<std::uint32_t>(bra_shell), static_cast<std::uint32_t>(ket_shell)};
        const std::uint8_t bra_l = basis.angular_momenta[static_cast<std::size_t>(bra_shell)];
        const std::uint8_t ket_l = basis.angular_momenta[static_cast<std::size_t>(ket_shell)];
        if (bra_l > 2u || ket_l > 2u) {
          error = "CUDA integral task setup found unsupported angular momentum";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        const std::int64_t bra_primitives =
            basis.shell_primitive_offsets[static_cast<std::size_t>(bra_shell + 1)] -
            basis.shell_primitive_offsets[static_cast<std::size_t>(bra_shell)];
        const std::int64_t ket_primitives =
            basis.shell_primitive_offsets[static_cast<std::size_t>(ket_shell + 1)] -
            basis.shell_primitive_offsets[static_cast<std::size_t>(ket_shell)];
        if (bra_primitives <= 0 || ket_primitives <= 0) {
          error = "CUDA integral task setup found an empty primitive range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        const std::size_t klass = shell_class(bra_l, ket_l);
        auto& signature = primitive_signature(created.accounting.primitive_signatures, bra_l, ket_l,
                                              bra_primitives, ket_primitives);

        (bra_l == 0u && ket_l == 0u ? created.h0_ss : created.h0_generic).push_back(task);
        ++created.accounting.h0_shell_classes[klass];
        ++signature.h0_tasks;
        if (!add_primitive_work(bra_primitives, ket_primitives,
                                created.accounting.h0_primitive_work) ||
            !add_primitive_work(bra_primitives, ket_primitives, signature.h0_primitive_work)) {
          error = "CUDA integral H0 primitive-work accounting overflowed";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }

        if (bra_atom < ket_atom || (bra_atom == ket_atom && bra_shell <= ket_shell)) {
          (bra_l == 0u && ket_l == 0u ? created.forward_ss : created.forward_generic)
              .push_back(task);
          ++created.accounting.forward_shell_classes[klass];
          ++signature.forward_tasks;
          if (!add_primitive_work(bra_primitives, ket_primitives,
                                  created.accounting.forward_primitive_work) ||
              !add_primitive_work(bra_primitives, ket_primitives,
                                  signature.forward_primitive_work)) {
            error = "CUDA forward integral primitive-work accounting overflowed";
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
        }
        if (bra_atom < ket_atom) {
          (bra_l == 0u && ket_l == 0u ? created.force_ss : created.force_generic).push_back(task);
          ++created.accounting.force_shell_classes[klass];
          ++signature.force_tasks;
          if (!add_primitive_work(bra_primitives, ket_primitives,
                                  created.accounting.force_primitive_work) ||
              !add_primitive_work(bra_primitives, ket_primitives, signature.force_primitive_work)) {
            error = "CUDA integral-force primitive-work accounting overflowed";
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
        }
      }
    }
  } catch (const std::bad_alloc&) {
    error = "CUDA integral task setup allocation failed";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }

  const std::size_t forward_tasks = created.forward_generic.size() + created.forward_ss.size();
  const std::size_t h0_tasks = created.h0_generic.size() + created.h0_ss.size();
  const std::size_t force_tasks = created.force_generic.size() + created.force_ss.size();
  if (forward_tasks == 0u || h0_tasks == 0u ||
      forward_tasks > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      h0_tasks > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      force_tasks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    error = "CUDA integral task setup produced an unsupported task count";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  created.accounting.forward_symmetry_exits =
      created.accounting.dense_live_pairs - static_cast<std::int64_t>(forward_tasks);
  created.accounting.force_ineligibility_exits =
      created.accounting.dense_live_pairs - static_cast<std::int64_t>(force_tasks);
  domains = std::move(created);
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_INTEGRAL_TASKS_HPP
