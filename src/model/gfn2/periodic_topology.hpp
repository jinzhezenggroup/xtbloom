// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_MODEL_GFN2_PERIODIC_TOPOLOGY_HPP
#define XTBLOOM_MODEL_GFN2_PERIODIC_TOPOLOGY_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "model/gfn2/lattice.hpp"

namespace xtbloom::detail::gfn2 {

inline constexpr double kPeriodicShortRangeCutoffBohr = 25.0;
inline constexpr double kPeriodicD4CoordinationCutoffBohr = 30.0;
inline constexpr double kPeriodicD4TwoBodyCutoffBohr = 50.0;
inline constexpr std::size_t kPeriodicShortRangeWorkspaceAlignment = 64u;

enum class PeriodicTranslationCutoff : std::int32_t {
  kShortRange25 = 0,
  kD4Coordination30 = 1,
  kD4TwoBody50 = 2,
};

struct LatticeTranslationView {
  const LatticeTranslation* data = nullptr;
  std::int64_t size = 0;
};

struct PeriodicShortRangePlanData;

/*
 * Immutable batch topology for the complete real-space image sums used by
 * periodic CN, repulsion, and D4. Unlike Wigner--Seitz topology, these lists
 * retain every translation in the cutoff-complete rectangular repeat box;
 * the term evaluators perform the final distance cutoff for each atom pair.
 * Cell changes rebuild this plan, while geometry-only changes reuse it.
 */
class PeriodicShortRangePlan {
 public:
  PeriodicShortRangePlan() noexcept = default;
  PeriodicShortRangePlan(const PeriodicShortRangePlan&) noexcept = default;
  PeriodicShortRangePlan(PeriodicShortRangePlan&&) noexcept = default;
  PeriodicShortRangePlan& operator=(const PeriodicShortRangePlan&) noexcept = default;
  PeriodicShortRangePlan& operator=(PeriodicShortRangePlan&&) noexcept = default;
  ~PeriodicShortRangePlan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const Lattice3D& lattice(std::int64_t system) const noexcept;
  [[nodiscard]] LatticeTranslationView translations(
      std::int64_t system, PeriodicTranslationCutoff cutoff) const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  /* Reject caller-owned numerical storage that overlaps immutable plan data. */
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] const PeriodicShortRangePlanData* identity() const noexcept;

 private:
  explicit PeriodicShortRangePlan(std::shared_ptr<const PeriodicShortRangePlanData> data) noexcept;
  std::shared_ptr<const PeriodicShortRangePlanData> data_;

  friend xtbloom_status_t make_periodic_short_range_plan(std::int64_t, std::int64_t,
                                                         const std::int64_t*, const double*,
                                                         PeriodicShortRangePlan&, std::string&);
};

/* Caller-owned scratch shared by the three short-range periodic term families. */
struct PeriodicShortRangeWorkspace {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  double* wrapped_positions = nullptr;
  std::int64_t wrapped_position_elements = 0;
  double* atom_scratch = nullptr;
  double* secondary_atom_scratch = nullptr;
  std::int64_t atom_elements = 0;
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
  double* strain_scratch = nullptr;
  std::int64_t strain_elements = 0;
  double* batch_scratch = nullptr;
  std::int64_t batch_elements = 0;
  const PeriodicShortRangePlanData* plan_identity = nullptr;
};

/*
 * Prepared wrapped coordinates for one geometry generation. The cache points
 * into the bound workspace and is published only after all coordinates have
 * been validated and wrapped successfully.
 */
struct PeriodicShortRangeGeometry {
  const double* wrapped_positions = nullptr;
  std::int64_t wrapped_position_elements = 0;
  std::uint64_t geometry_generation = 0u;
  const PeriodicShortRangePlanData* plan_identity = nullptr;
};

xtbloom_status_t make_periodic_short_range_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                                const std::int64_t* atom_offsets,
                                                const double* cell_matrices,
                                                PeriodicShortRangePlan& plan, std::string& error);

xtbloom_status_t bind_periodic_short_range_workspace(const PeriodicShortRangePlan& plan,
                                                     void* workspace, std::size_t workspace_size,
                                                     PeriodicShortRangeWorkspace& view,
                                                     std::string& error);

/*
 * Verify that a retained workspace descriptor still has the canonical layout
 * produced by bind_periodic_short_range_workspace and does not overlap the
 * immutable topology plan. Term evaluators call this before touching scratch.
 */
xtbloom_status_t validate_periodic_short_range_workspace(
    const PeriodicShortRangePlan& plan, const PeriodicShortRangeWorkspace& workspace,
    std::string& error);

xtbloom_status_t update_periodic_short_range_geometry_cpu(
    const PeriodicShortRangePlan& plan, const double* positions, std::uint64_t geometry_generation,
    const PeriodicShortRangeWorkspace& workspace, PeriodicShortRangeGeometry& geometry,
    std::string& error);

/*
 * Wigner--Seitz topology keeps only the closest periodic image or equally
 * close images of an atom pair. Every retained image receives weight 1/n,
 * where n is the number of closest images whose Cartesian distances differ
 * by strictly less than xTB's 0.01-bohr tolerance. This is a
 * topology/canonicalization weight, not a generic real-space pair-energy
 * factor.
 */
constexpr double kWignerSeitzDistanceTolerance = 0.01;
constexpr double kPeriodicTopologyMinimumDistanceSquared = 1.0e-12;

enum class WignerSeitzPairMode : std::int32_t {
  /* Store one lower-triangular atom pair (image_atom <= center_atom). */
  kUnique = 0,
  /* Store every ordered atom pair. */
  kDirected = 1,
};

/*
 * One canonical closest image. translation is applied to image_atom after
 * both input atoms have been wrapped into the central cell. displacement is
 * then r(image_atom + translation) - r(center_atom), in bohr.
 */
struct WignerSeitzImage {
  std::int64_t center_atom = 0;
  std::int64_t image_atom = 0;
  std::array<std::int64_t, 3> translation{};
  std::array<double, 3> displacement{};
  double distance_squared = 0.0;
  double weight = 0.0;
};

namespace periodic_topology_detail {

inline bool representable_geometry_size(std::int64_t atom_count) noexcept {
  if (atom_count < 0) return false;
  const auto count = static_cast<std::uint64_t>(atom_count);
  return count <= std::numeric_limits<std::size_t>::max() / 3u &&
         count <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 3u;
}

inline bool is_origin(const LatticeTranslation& translation) noexcept {
  return translation.index[0] == 0 && translation.index[1] == 0 && translation.index[2] == 0;
}

enum class ImageGeometryStatus {
  kInsideCutoff,
  kOutsideCutoff,
};

inline ImageGeometryStatus image_geometry(const std::array<double, 3>& center,
                                          const std::array<double, 3>& image,
                                          const LatticeTranslation& translation, double cutoff,
                                          double cutoff_squared,
                                          std::array<double, 3>& displacement,
                                          double& distance_squared) noexcept {
  using lattice_binary64_detail::absolute;
  using lattice_binary64_detail::finite;
  using lattice_binary64_detail::rounded_add;
  using lattice_binary64_detail::rounded_multiply;
  using lattice_binary64_detail::rounded_subtract;

  for (std::size_t component = 0; component < 3u; ++component) {
    const double translated = rounded_add(image[component], translation.cartesian[component]);
    displacement[component] = rounded_subtract(translated, center[component]);
    /*
     * The lattice generator deliberately returns a rectangular superset. A
     * far image can therefore overflow while squaring even though another
     * image of the same pair is safely inside the cutoff. An infinite
     * displacement produced from finite inputs is also provably outside every
     * accepted finite cutoff. All inputs have already been validated as
     * finite, so these operations can produce infinity through overflow but
     * cannot produce NaN.
     */
    if (!finite(displacement[component]) || absolute(displacement[component]) > cutoff) {
      return ImageGeometryStatus::kOutsideCutoff;
    }
  }
  distance_squared = rounded_add(rounded_add(rounded_multiply(displacement[0], displacement[0]),
                                             rounded_multiply(displacement[1], displacement[1])),
                                 rounded_multiply(displacement[2], displacement[2]));
  if (!finite(distance_squared) || distance_squared > cutoff_squared) {
    return ImageGeometryStatus::kOutsideCutoff;
  }
  return ImageGeometryStatus::kInsideCutoff;
}

inline bool within_wigner_seitz_distance_tolerance(double distance_squared,
                                                   double minimum_distance) noexcept {
  using lattice_binary64_detail::absolute;
  using lattice_binary64_detail::rounded_square_root;
  using lattice_binary64_detail::rounded_subtract;

  const double distance = rounded_square_root(distance_squared);
  return absolute(rounded_subtract(distance, minimum_distance)) < kWignerSeitzDistanceTolerance;
}

}  // namespace periodic_topology_detail

/*
 * Build a deterministic Wigner--Seitz image topology from atom-major Cartesian
 * coordinates in bohr. Inputs are wrapped internally, so integer-lattice
 * shifts of individual atoms produce the same canonical topology. For each
 * atom pair, only the closest image(s) inside cutoff are retained. Self pairs
 * exclude the zero translation but retain periodic self images.
 *
 * kUnique emits lower-triangular pairs in (center_atom, image_atom) order;
 * kDirected emits the complete ordered pair map. Within each pair, the origin
 * is first when retained and all other translations follow the lattice
 * generator's lexicographic integer-triplet order.
 *
 * The output is transactional: it remains unchanged on error.
 */
inline xtbloom_status_t make_wigner_seitz_topology(const Lattice3D& lattice,
                                                   std::int64_t atom_count, const double* positions,
                                                   double cutoff, WignerSeitzPairMode pair_mode,
                                                   std::vector<WignerSeitzImage>& topology,
                                                   std::string& error) {
  using namespace periodic_topology_detail;
  using lattice_binary64_detail::finite;
  using lattice_binary64_detail::rounded_multiply;
  using lattice_binary64_detail::rounded_square_root;

  if (!representable_geometry_size(atom_count)) {
    error = "periodic topology atom count is outside the supported host range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_count > 0 && positions == nullptr) {
    error = "periodic topology positions must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!(cutoff >= 0.0) || !finite(cutoff)) {
    error = "periodic topology cutoff must be finite and nonnegative";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (pair_mode != WignerSeitzPairMode::kUnique && pair_mode != WignerSeitzPairMode::kDirected) {
    error = "periodic topology pair mode is invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const double cutoff_squared = rounded_multiply(cutoff, cutoff);
  if (!finite(cutoff_squared)) {
    error = "periodic topology cutoff squared is outside the binary64 range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    const auto count = static_cast<std::size_t>(atom_count);
    std::vector<std::array<double, 3>> wrapped;
    if (count > wrapped.max_size()) {
      error = "periodic topology atom count exceeds the wrapped-coordinate vector limit";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    wrapped.resize(count);
    std::string local_error;
    for (std::size_t atom = 0; atom < count; ++atom) {
      const xtbloom_status_t status =
          wrap_cartesian(lattice, positions + atom * 3u, wrapped[atom].data(), local_error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        error = std::move(local_error);
        return status;
      }
    }

    /*
     * Validate and canonicalize every coordinate before enumerating the
     * cutoff-dependent translation superset. Malformed geometry must not
     * trigger potentially large allocation or enumeration work.
     */
    std::vector<LatticeTranslation> translations;
    const xtbloom_status_t translation_status = make_lattice_translations(
        lattice, cutoff, LatticeOriginPolicy::kInclude, translations, local_error);
    if (translation_status != XTBLOOM_STATUS_SUCCESS) {
      error = std::move(local_error);
      return translation_status;
    }

    std::vector<WignerSeitzImage> created;
    for (std::size_t center = 0; center < count; ++center) {
      const std::size_t image_end =
          pair_mode == WignerSeitzPairMode::kDirected ? count : center + 1u;
      for (std::size_t image = 0; image < image_end; ++image) {
        double minimum = std::numeric_limits<double>::infinity();
        for (const auto& translation : translations) {
          if (center == image && is_origin(translation)) continue;
          std::array<double, 3> displacement{};
          double distance_squared = 0.0;
          const ImageGeometryStatus geometry =
              image_geometry(wrapped[center], wrapped[image], translation, cutoff, cutoff_squared,
                             displacement, distance_squared);
          if (geometry == ImageGeometryStatus::kInsideCutoff && distance_squared < minimum) {
            minimum = distance_squared;
          }
        }
        if (!finite(minimum)) continue;
        if (minimum < kPeriodicTopologyMinimumDistanceSquared) {
          error = "periodic topology is undefined for coincident or near-coincident images";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }

        const double minimum_distance = rounded_square_root(minimum);
        std::size_t multiplicity = 0u;
        for (const auto& translation : translations) {
          if (center == image && is_origin(translation)) continue;
          std::array<double, 3> displacement{};
          double distance_squared = 0.0;
          const ImageGeometryStatus geometry =
              image_geometry(wrapped[center], wrapped[image], translation, cutoff, cutoff_squared,
                             displacement, distance_squared);
          if (geometry == ImageGeometryStatus::kInsideCutoff &&
              within_wigner_seitz_distance_tolerance(distance_squared, minimum_distance)) {
            ++multiplicity;
          }
        }
        if (multiplicity == 0u) {
          error = "periodic topology lost its closest Wigner-Seitz image";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        if (multiplicity > created.max_size() - created.size()) {
          error = "periodic topology image count exceeds the vector implementation limit";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        const double weight = 1.0 / static_cast<double>(multiplicity);
        for (const auto& translation : translations) {
          if (center == image && is_origin(translation)) continue;
          std::array<double, 3> displacement{};
          double distance_squared = 0.0;
          const ImageGeometryStatus geometry =
              image_geometry(wrapped[center], wrapped[image], translation, cutoff, cutoff_squared,
                             displacement, distance_squared);
          if (geometry == ImageGeometryStatus::kOutsideCutoff ||
              !within_wigner_seitz_distance_tolerance(distance_squared, minimum_distance)) {
            continue;
          }
          WignerSeitzImage entry;
          entry.center_atom = static_cast<std::int64_t>(center);
          entry.image_atom = static_cast<std::int64_t>(image);
          entry.translation = translation.index;
          entry.displacement = displacement;
          entry.distance_squared = distance_squared;
          entry.weight = weight;
          created.push_back(entry);
        }
      }
    }

    topology = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate periodic Wigner-Seitz topology";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_PERIODIC_TOPOLOGY_HPP
