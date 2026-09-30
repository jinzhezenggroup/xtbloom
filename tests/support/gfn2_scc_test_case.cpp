#include "tests/support/gfn2_scc_test_case.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace xtbloom::test::gfn2 {
namespace {

using namespace xtbloom::detail::gfn2;

constexpr std::size_t kHostAlignment = 64u;

/* Own one zero-initialized allocation satisfying every current host binding. */
class AlignedBuffer {
 public:
  AlignedBuffer() noexcept = default;
  ~AlignedBuffer() { std::free(data_); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  [[nodiscard]] bool allocate(std::size_t requested) noexcept {
    if (data_ != nullptr || requested > std::numeric_limits<std::size_t>::max() - 63u) {
      return false;
    }
    size_ = (std::max<std::size_t>(requested, 1u) + 63u) & ~std::size_t{63u};
    data_ = std::aligned_alloc(kHostAlignment, size_);
    if (data_ == nullptr) {
      size_ = 0u;
      return false;
    }
    std::memset(data_, 0, size_);
    return true;
  }

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0u;
};

LapackInt tiny_dpotrf(LapackInt, char, LapackInt n, double* matrix, LapackInt) {
  for (LapackInt column = 0; column < n; ++column) {
    for (LapackInt row = column; row < n; ++row) {
      double value = matrix[column * n + row];
      for (LapackInt inner = 0; inner < column; ++inner) {
        value -= matrix[inner * n + row] * matrix[inner * n + column];
      }
      if (row == column) {
        if (!(value > 0.0) || !std::isfinite(value)) {
          return column + 1;
        }
        matrix[column * n + column] = std::sqrt(value);
      } else {
        matrix[column * n + row] = value / matrix[column * n + column];
      }
    }
  }
  return 0;
}

LapackInt tiny_dpocon(LapackInt, char, LapackInt n, const double*, LapackInt, double,
                      double* reciprocal_condition, double*, LapackInt*) {
  if (n <= 0) {
    return -3;
  }
  *reciprocal_condition = 1.0;
  return 0;
}

/* Jacobi is sufficient for the fixture's at-most-six-orbital molecules. */
LapackInt tiny_dsyevd(LapackInt, char, char, LapackInt n, double* matrix, LapackInt,
                      double* eigenvalues, double*, LapackInt, LapackInt*, LapackInt) {
  if (n <= 0 || n > 16) {
    return -4;
  }
  std::array<double, 16u * 16u> values{};
  std::array<double, 16u * 16u> vectors{};
  for (LapackInt row = 0; row < n; ++row) {
    vectors[static_cast<std::size_t>(row * n + row)] = 1.0;
    for (LapackInt column = 0; column < n; ++column) {
      values[static_cast<std::size_t>(row * n + column)] = matrix[column * n + row];
    }
  }

  bool converged = false;
  for (int sweep = 0; sweep < 128 * n * n; ++sweep) {
    LapackInt p = 0;
    LapackInt q = 0;
    double largest = 0.0;
    for (LapackInt row = 0; row < n; ++row) {
      for (LapackInt column = row + 1; column < n; ++column) {
        const double magnitude = std::abs(values[static_cast<std::size_t>(row * n + column)]);
        if (magnitude > largest) {
          largest = magnitude;
          p = row;
          q = column;
        }
      }
    }
    if (largest <= 1.0e-14) {
      converged = true;
      break;
    }

    const double app = values[static_cast<std::size_t>(p * n + p)];
    const double aqq = values[static_cast<std::size_t>(q * n + q)];
    const double apq = values[static_cast<std::size_t>(p * n + q)];
    const double tau = (aqq - app) / (2.0 * apq);
    const double tangent = std::copysign(1.0, tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
    const double sine = tangent * cosine;
    for (LapackInt index = 0; index < n; ++index) {
      if (index != p && index != q) {
        const double aip = values[static_cast<std::size_t>(index * n + p)];
        const double aiq = values[static_cast<std::size_t>(index * n + q)];
        const double updated_p = cosine * aip - sine * aiq;
        const double updated_q = sine * aip + cosine * aiq;
        values[static_cast<std::size_t>(index * n + p)] = updated_p;
        values[static_cast<std::size_t>(p * n + index)] = updated_p;
        values[static_cast<std::size_t>(index * n + q)] = updated_q;
        values[static_cast<std::size_t>(q * n + index)] = updated_q;
      }
      const double vip = vectors[static_cast<std::size_t>(index * n + p)];
      const double viq = vectors[static_cast<std::size_t>(index * n + q)];
      vectors[static_cast<std::size_t>(index * n + p)] = cosine * vip - sine * viq;
      vectors[static_cast<std::size_t>(index * n + q)] = sine * vip + cosine * viq;
    }
    values[static_cast<std::size_t>(p * n + p)] =
        cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
    values[static_cast<std::size_t>(q * n + q)] =
        sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
    values[static_cast<std::size_t>(p * n + q)] = 0.0;
    values[static_cast<std::size_t>(q * n + p)] = 0.0;
  }
  if (!converged) {
    return 1;
  }

  std::array<LapackInt, 16> order{};
  for (LapackInt index = 0; index < n; ++index) {
    order[static_cast<std::size_t>(index)] = index;
  }
  std::sort(order.begin(), order.begin() + n, [&](LapackInt first, LapackInt second) {
    return values[static_cast<std::size_t>(first * n + first)] <
           values[static_cast<std::size_t>(second * n + second)];
  });
  for (LapackInt column = 0; column < n; ++column) {
    const LapackInt source = order[static_cast<std::size_t>(column)];
    eigenvalues[column] = values[static_cast<std::size_t>(source * n + source)];
    for (LapackInt row = 0; row < n; ++row) {
      matrix[column * n + row] = vectors[static_cast<std::size_t>(row * n + source)];
    }
  }
  return 0;
}

void tiny_dtrsm(int, int side, int, int transpose, int, LapackInt rows, LapackInt columns,
                double alpha, const double* triangular, LapackInt, double* rhs, LapackInt) {
  constexpr int kLeft = 141;
  constexpr int kNoTrans = 111;
  if (side == kLeft) {
    for (LapackInt column = 0; column < columns; ++column) {
      if (transpose == kNoTrans) {
        for (LapackInt row = 0; row < rows; ++row) {
          double value = alpha * rhs[column * rows + row];
          for (LapackInt inner = 0; inner < row; ++inner) {
            value -= triangular[inner * rows + row] * rhs[column * rows + inner];
          }
          rhs[column * rows + row] = value / triangular[row * rows + row];
        }
      } else {
        for (LapackInt row = rows; row-- > 0;) {
          double value = alpha * rhs[column * rows + row];
          for (LapackInt inner = row + 1; inner < rows; ++inner) {
            value -= triangular[row * rows + inner] * rhs[column * rows + inner];
          }
          rhs[column * rows + row] = value / triangular[row * rows + row];
        }
      }
    }
  } else {
    /* The eigensolver only requests right-side L^T solves in this path. */
    for (LapackInt row = 0; row < rows; ++row) {
      for (LapackInt column = 0; column < columns; ++column) {
        double value = alpha * rhs[column * rows + row];
        for (LapackInt inner = 0; inner < column; ++inner) {
          value -= rhs[inner * rows + row] * triangular[inner * columns + column];
        }
        rhs[column * rows + row] = value / triangular[column * columns + column];
      }
    }
  }
}

void tiny_dgemm(int, int, int, LapackInt rows, LapackInt columns, LapackInt inner, double alpha,
                const double* left, LapackInt leading_left, const double* right,
                LapackInt leading_right, double beta, double* result, LapackInt leading_result) {
  for (LapackInt column = 0; column < columns; ++column) {
    for (LapackInt row = 0; row < rows; ++row) {
      double value = 0.0;
      for (LapackInt k = 0; k < inner; ++k) {
        value += left[k * leading_left + row] * right[k * leading_right + column];
      }
      result[column * leading_result + row] =
          alpha * value + beta * result[column * leading_result + row];
    }
  }
}

xtbloom_status_t allocate(AlignedBuffer& buffer, std::size_t bytes, const char* purpose,
                          std::string& error) {
  if (buffer.allocate(bytes)) {
    return XTBLOOM_STATUS_SUCCESS;
  }
  error = std::string("failed to allocate host SCC fixture ") + purpose;
  return XTBLOOM_STATUS_ALLOCATION_FAILED;
}

std::vector<std::byte> copy_bytes(const AlignedBuffer& buffer) {
  std::vector<std::byte> copy(buffer.size());
  std::memcpy(copy.data(), buffer.data(), buffer.size());
  return copy;
}

}  // namespace

struct HostSccCase::Impl {
  HostSccCaseOptions options;
  std::int64_t batch_size = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<double> coordination_numbers;

  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0_plan;
  WavefunctionLayout wavefunction_layout;
  ES2Plan es2_plan;
  ES3Plan es3_plan;
  AES2Plan aes2_plan;
  MullikenPlan mulliken_plan;
  EigensolverPlan eigensolver_plan;
  SccMixerPlan mixer_plan;
  D4Plan d4_plan;
  PeriodicEmbeddingPlan periodic_plan;
  ExternalPointChargePlan point_charge_plan;
  SccDriverPlan driver_plan;

  std::vector<std::int64_t> point_charge_offsets;
  std::vector<double> point_charge_positions;
  std::vector<double> point_charge_charges;
  std::vector<double> point_charge_hardnesses;
  std::vector<double> explicit_point_charge_shell_potential;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response_matrices;

  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> h0;
  AlignedBuffer integral_scratch;

  AlignedBuffer es2_storage;
  AlignedBuffer es2_scratch_storage;
  ES2GeometryCache es2_cache;
  ES2Workspace es2_scratch;
  AlignedBuffer aes2_storage;
  AlignedBuffer aes2_scratch_storage;
  AES2GeometryCache aes2_cache;
  AES2Workspace aes2_scratch;
  std::vector<double> d4_pair_data;
  std::vector<double> d4_coordination;
  D4GeometryCache d4_cache;

  AlignedBuffer wavefunction_storage;
  WavefunctionView wavefunction;
  AlignedBuffer overlap_cache_storage;
  EigensolverOverlapCache overlap_cache;
  AlignedBuffer eigensolver_scratch_storage;
  EigensolverWorkspace eigensolver_scratch;
  AlignedBuffer mixer_state_storage;
  SccMixerState mixer_state;
  AlignedBuffer driver_state_storage;
  SccDriverState driver_state;
  AlignedBuffer driver_workspace_storage;
  SccDriverWorkspace driver_workspace;
  SccDriverGeometryView geometry;
  CpuLinearAlgebraBackend cpu_backend;

  xtbloom_status_t build(std::string& error);
  bool append_system(SmallSystemKind kind, std::int64_t system);
};

bool HostSccCase::Impl::append_system(SmallSystemKind kind, std::int64_t system) {
  const double shift = 8.0 * static_cast<double>(system);
  const auto atom = [&](std::int32_t atomic_number, double x, double y, double z) {
    atomic_numbers.push_back(atomic_number);
    positions.insert(positions.end(), {x + shift, y, z});
  };
  const auto append_alkane = [&](std::int32_t carbon_count) {
    constexpr double kAngstromPerBohr = 0.529177210903;
    constexpr double bond = 1.54 / kAngstromPerBohr;
    constexpr double h_bond = 1.09 / kAngstromPerBohr;
    constexpr double h_y = h_bond * std::sin(109.5 * 3.14159265358979323846 / 180.0 / 2.0);
    constexpr double h_z = h_bond * std::cos(109.5 * 3.14159265358979323846 / 180.0 / 2.0);
    atom(6, 0.0, 0.0, 0.0);
    for (std::int32_t carbon = 0; carbon < carbon_count; ++carbon) {
      const double z = static_cast<double>(carbon) * bond;
      if (carbon > 0) {
        atom(6, 0.0, 0.0, z);
      }
      if (carbon == 0 || carbon == carbon_count - 1) {
        atom(1, 0.0, h_y, z + (carbon == 0 ? -h_z : h_z));
        atom(1, h_y, 0.0, z);
        atom(1, -h_y, 0.0, z);
      } else {
        atom(1, 0.0, h_y, z);
        atom(1, 0.0, -h_y, z);
      }
    }
  };
  const auto append_benchmark_alkane = [&](std::int32_t carbon_count) {
    /* Match benchmarks/natoms_scaling.py exactly so the focused CUDA SCC test
     * and the retained public performance sweep use one physical coordinate. */
    using Point = std::array<double, 3>;
    constexpr double kAngstromToBohr = 1.8897261254579021;
    constexpr double cc = 1.54;
    constexpr double ch = 1.09;
    const auto unit = [](Point value) {
      const double norm =
          std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
      for (double& component : value) {
        component /= norm;
      }
      return value;
    };
    const auto cross = [](const Point& left, const Point& right) {
      return Point{left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2],
                   left[0] * right[1] - left[1] * right[0]};
    };
    const auto orthogonal = [&](const Point& axis) {
      return unit(std::abs(axis[0]) < 0.9 ? Point{-axis[1], axis[0], 0.0}
                                          : Point{0.0, -axis[2], axis[1]});
    };

    const double half_external = 0.5 * std::acos(1.0 / 3.0);
    const Point even_step{cc * std::cos(half_external), cc * std::sin(half_external), 0.0};
    const Point odd_step{cc * std::cos(half_external), -cc * std::sin(half_external), 0.0};
    std::vector<Point> carbons{{0.0, 0.0, 0.0}};
    for (std::int32_t index = 1; index < carbon_count; ++index) {
      const Point& step = index % 2 == 1 ? even_step : odd_step;
      const Point& previous = carbons.back();
      carbons.push_back({previous[0] + step[0], previous[1] + step[1], previous[2] + step[2]});
    }

    std::vector<Point> hydrogens;
    hydrogens.reserve(static_cast<std::size_t>(2 * carbon_count + 2));
    for (std::int32_t index = 0; index < carbon_count; ++index) {
      const Point& position = carbons[static_cast<std::size_t>(index)];
      std::vector<Point> neighbors;
      if (index > 0) {
        neighbors.push_back(carbons[static_cast<std::size_t>(index - 1)]);
      }
      if (index + 1 < carbon_count) {
        neighbors.push_back(carbons[static_cast<std::size_t>(index + 1)]);
      }
      std::vector<Point> directions;
      if (neighbors.size() == 1u) {
        Point axis = unit({position[0] - neighbors[0][0], position[1] - neighbors[0][1],
                           position[2] - neighbors[0][2]});
        const Point perpendicular = orthogonal(axis);
        const Point binormal = unit(cross(axis, perpendicular));
        constexpr double axial = 1.0 / 3.0;
        const double radial = std::sqrt(8.0 / 9.0);
        for (const double azimuth :
             {0.0, 2.0 * 3.14159265358979323846 / 3.0, 4.0 * 3.14159265358979323846 / 3.0}) {
          directions.push_back({axial * axis[0] + radial * (std::cos(azimuth) * perpendicular[0] +
                                                            std::sin(azimuth) * binormal[0]),
                                axial * axis[1] + radial * (std::cos(azimuth) * perpendicular[1] +
                                                            std::sin(azimuth) * binormal[1]),
                                axial * axis[2] + radial * (std::cos(azimuth) * perpendicular[2] +
                                                            std::sin(azimuth) * binormal[2])});
        }
      } else {
        const Point first = unit({neighbors[0][0] - position[0], neighbors[0][1] - position[1],
                                  neighbors[0][2] - position[2]});
        const Point second = unit({neighbors[1][0] - position[0], neighbors[1][1] - position[1],
                                   neighbors[1][2] - position[2]});
        const Point bisector =
            unit({first[0] + second[0], first[1] + second[1], first[2] + second[2]});
        const Point normal = unit(cross(first, second));
        const double axial = -1.0 / std::sqrt(3.0);
        const double radial = std::sqrt(2.0 / 3.0);
        directions.push_back({axial * bisector[0] + radial * normal[0],
                              axial * bisector[1] + radial * normal[1],
                              axial * bisector[2] + radial * normal[2]});
        directions.push_back({axial * bisector[0] - radial * normal[0],
                              axial * bisector[1] - radial * normal[1],
                              axial * bisector[2] - radial * normal[2]});
      }
      for (const Point& direction : directions) {
        hydrogens.push_back({position[0] + ch * direction[0], position[1] + ch * direction[1],
                             position[2] + ch * direction[2]});
      }
    }
    for (const Point& point : hydrogens) {
      atom(1, point[0] * kAngstromToBohr, point[1] * kAngstromToBohr, point[2] * kAngstromToBohr);
    }
    for (const Point& point : carbons) {
      atom(6, point[0] * kAngstromToBohr, point[1] * kAngstromToBohr, point[2] * kAngstromToBohr);
    }
  };

  switch (kind) {
    case SmallSystemKind::kYttriumSpinComplex:
      /* Coordinates in bohr from gs2-problem-data-threeway-20260930.tar.gz,
       * SHA256 8716d1dde0c4f9106dcdb295387c044c826769090ed8d546a4d5def43d2554eb.
       * Keep the original atom order and binary64 geometry for replay. */
      atom(39, -8.1105344515425895, 62.696275116204006, 29.957733767386102);
      atom(17, -15.339228207028563, 65.728397374949807, 27.968589151343771);
      atom(8, -16.498783054360182, 63.562072037462549, 26.82623081174625);
      atom(8, -15.020393615281703, 67.499316418120344, 25.946997937741617);
      atom(8, -16.76788005450689, 66.804086176870527, 29.969468966620028);
      atom(8, -12.886968409624194, 64.924451189750258, 28.751823938217417);
      atom(6, -8.6686461651895641, 61.602747299666824, 21.651404792071041);
      atom(6, -10.159432207645589, 62.414309081148602, 23.990337713781603);
      atom(8, -8.1675852832450406, 63.335928512128589, 25.700691034657893);
      atom(1, -7.1276500996022341, 60.266503059682691, 21.938794341104128);
      atom(1, -8.2959354816296234, 63.150073947771645, 20.343865491919978);
      atom(1, -10.049979270507263, 60.544122724651466, 20.54961887236923);
      atom(1, -10.994294312244007, 60.804697059976107, 24.967552887348081);
      atom(1, -11.751772132039001, 63.634240678162008, 23.522422628063016);
      atom(1, -6.8732362714438668, 64.078364111971567, 24.668938365134714);
      atom(6, -12.869091600485234, 71.725235361575969, 33.16461789813728);
      atom(6, -10.862958346582516, 71.340014691071019, 35.240765504957388);
      atom(8, -9.3332250486979547, 69.246387117598118, 34.755691706027193);
      atom(1, -13.931911367497259, 73.453805642293659, 33.518412423189716);
      atom(1, -14.20327603899352, 70.156309143866679, 33.130451649804051);
      atom(1, -12.062556490494956, 71.7076798058782, 31.269354875966357);
      atom(1, -11.785428154318586, 70.864824159772624, 37.020093829382517);
      atom(1, -9.7024586361885845, 72.998060392817663, 35.624039757553987);
      atom(1, -8.2649439731857601, 69.224919828822365, 36.221779028034355);
      atom(6, -21.250934031740343, 73.911327234326805, 35.775104463956566);
      atom(6, -19.982946699377699, 71.561868538102075, 34.843601765344786);
      atom(8, -19.615035920174307, 71.758173287928201, 32.197701731950012);
      atom(1, -22.017217975276093, 73.564354620584254, 37.655325266175467);
      atom(1, -22.951574160336062, 74.168613446194598, 34.641779015234754);
      atom(1, -20.212661807087208, 75.690013051869556, 35.742714558180481);
      atom(1, -21.060770891819256, 69.845279121014514, 35.210227530783428);
      atom(1, -18.077838204097478, 71.25747145394736, 35.56519368603314);
      atom(1, -18.974702222844865, 70.262209395890693, 31.395758656442577);
      atom(6, -13.999431281930137, 64.767282667965134, 36.406084016969118);
      atom(6, -11.740377083507507, 63.187112579814311, 35.604462194902858);
      atom(8, -10.559506125490111, 64.306888692222557, 33.414118438371624);
      atom(1, -13.47941644695562, 66.410626300462198, 37.533815876362034);
      atom(1, -15.331234665521396, 63.701949565207357, 37.561141316124122);
      atom(1, -15.030957184318362, 65.47804645795938, 34.770979590375418);
      atom(1, -12.509117671005271, 61.326450442985283, 35.168804734131633);
      atom(1, -10.349236299603001, 63.071461340987213, 37.119096581051657);
      atom(1, -10.678162028855363, 66.084780824792972, 33.754892750425391);
      atom(6, -16.639133013636137, 76.562650781699261, 25.563723685145018);
      atom(6, -14.76696244470814, 74.795322218165495, 26.900213589525347);
      atom(8, -12.348736614808283, 75.871634629707344, 27.232200675099602);
      atom(1, -18.572398428173287, 75.851905888966257, 25.55436954082812);
      atom(1, -16.059913059177095, 76.867954934393794, 23.610767324389268);
      atom(1, -16.913899192156727, 78.315522940379623, 26.610046143089058);
      atom(1, -14.813487501896429, 72.952688068565394, 25.980767343588681);
      atom(1, -15.496415626074935, 74.24044193619163, 28.74488864334004);
      atom(1, -12.373076287293463, 77.118305851384207, 28.549887804539907);
      atom(6, -23.371641380579124, 71.022521804872639, 23.030696993174143);
      atom(6, -21.015322978522004, 70.50607855227365, 24.56020352392375);
      atom(8, -20.720582394864124, 67.812122783529659, 24.935465337751936);
      atom(1, -24.990078422753619, 70.294750479756757, 24.076565916848274);
      atom(1, -23.647144552288314, 69.931450632297455, 21.305471527697044);
      atom(1, -23.606118598122691, 73.04010679909058, 22.688448694743169);
      atom(1, -21.207035693865286, 71.165498483461818, 26.502161678634174);
      atom(1, -19.267949922864293, 71.209699177516811, 23.72690989200877);
      atom(1, -18.937153364748553, 67.53416296785845, 25.117634936165857);
      break;
    case SmallSystemKind::kH2:
      atom(1, -0.7, 0.0, 0.0);
      atom(1, 0.7, 0.0, 0.0);
      break;
    case SmallSystemKind::kHe:
      atom(2, 0.0, 0.0, 0.0);
      break;
    case SmallSystemKind::kLiH:
      atom(3, -1.5, 0.0, 0.0);
      atom(1, 1.5, 0.0, 0.0);
      break;
    case SmallSystemKind::kCH2:
      atom(6, 0.0, 0.0, 0.0);
      atom(1, 1.6, 0.0, 1.0);
      atom(1, -1.6, 0.0, 1.0);
      break;
    case SmallSystemKind::kH2Stretched:
      atom(1, -4.0, 0.0, 0.0);
      atom(1, 4.0, 0.0, 0.0);
      break;
    case SmallSystemKind::kC20H42: {
      /* A 20-carbon n-alkane C20H42 (62 atoms).  Carbons form a straight
       * chain along z at 1.54 angstrom spacing; each interior carbon
       * carries two hydrogens in y, and each terminal carbon carries three
       * hydrogens, so every carbon has four neighbors.  The molecule is
       * neutral, closed-shell, and wide-gap, so restricted 0 K SCC converges
       * reliably, and 62 atoms crosses the 40-atom sparse pair-list crossover
       * used by the production bucketed CN consistency gate. */
      append_alkane(20);
      break;
    }
    case SmallSystemKind::kC90H182: {
      /* Keep the same deterministic all-trans construction as C20H42 while
       * crossing CUDA 12.9's 512-orbital vector-capture boundary. */
      append_benchmark_alkane(90);
      break;
    }
    default:
      return false;
  }
  atom_offsets.push_back(static_cast<std::int64_t>(atomic_numbers.size()));
  const std::size_t electronic_index = static_cast<std::size_t>(system);
  molecular_charges.push_back(
      options.molecular_charges.empty() ? 0.0 : options.molecular_charges[electronic_index]);
  unpaired_electrons.push_back(
      options.unpaired_electrons.empty() ? 0 : options.unpaired_electrons[electronic_index]);
  spin_channels.push_back(options.spin_channels.empty() ? 1
                                                        : options.spin_channels[electronic_index]);
  return true;
}

xtbloom_status_t HostSccCase::Impl::build(std::string& error) {
  error.clear();
  if (options.systems.empty()) {
    error = "host SCC fixture requires at least one system";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.systems.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error = "host SCC fixture batch size exceeds int64 range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.geometry_generation == 0u) {
    error = "host SCC fixture geometry generation must be nonzero";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const auto valid_optional_batch = [&](std::size_t elements) {
    return elements == 0u || elements == options.systems.size();
  };
  if (!valid_optional_batch(options.molecular_charges.size()) ||
      !valid_optional_batch(options.unpaired_electrons.size()) ||
      !valid_optional_batch(options.spin_channels.size())) {
    error = "host SCC fixture electronic vectors must be empty or match systems.size()";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  batch_size = static_cast<std::int64_t>(options.systems.size());
  atom_offsets.reserve(options.systems.size() + 1u);
  atom_offsets.push_back(0);
  for (std::size_t system = 0; system < options.systems.size(); ++system) {
    if (!append_system(options.systems[system], static_cast<std::int64_t>(system))) {
      error = "host SCC fixture contains an unknown small-system kind";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  const std::int64_t total_atoms = atom_offsets.back();
  coordination_numbers.assign(static_cast<std::size_t>(total_atoms), 0.0);

  xtbloom_status_t status = make_basis_plan(batch_size, total_atoms, atom_offsets.data(),
                                            atomic_numbers.data(), basis, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = make_integral_plan(basis, integrals, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = make_h0_plan(basis, integrals, atomic_numbers.data(), h0_plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = make_wavefunction_layout(basis, atomic_numbers.data(), molecular_charges.data(),
                                    unpaired_electrons.data(), spin_channels.data(),
                                    wavefunction_layout, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = make_es2_plan(basis, atomic_numbers.data(), es2_plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = make_es3_plan(basis, atomic_numbers.data(), es3_plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = make_aes2_plan(basis, atomic_numbers.data(), aes2_plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = make_mulliken_plan(basis, integrals, wavefunction_layout, mulliken_plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = make_eigensolver_plan(wavefunction_layout, eigensolver_plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = make_scc_mixer_plan(wavefunction_layout, options.mixer_history, options.mixer_damping,
                               options.residual_tolerance, options.residual_tolerance, mixer_plan,
                               error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  if (options.enable_d4) {
    status = make_d4_plan(batch_size, total_atoms, atom_offsets.data(), atomic_numbers.data(),
                          d4_plan, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }
  if (options.enable_periodic_embedding) {
    status = make_periodic_embedding_plan(batch_size, total_atoms, atom_offsets.data(),
                                          periodic_plan, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }
  if (options.enable_explicit_point_charges) {
    point_charge_offsets.resize(static_cast<std::size_t>(batch_size) + 1u);
    for (std::int64_t system = 0; system <= batch_size; ++system) {
      point_charge_offsets[static_cast<std::size_t>(system)] = system;
    }
    point_charge_positions.reserve(3u * static_cast<std::size_t>(batch_size));
    point_charge_charges.reserve(static_cast<std::size_t>(batch_size));
    point_charge_hardnesses.assign(static_cast<std::size_t>(batch_size), 1.5);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const std::int64_t first_atom = atom_offsets[static_cast<std::size_t>(system)];
      const std::size_t xyz = 3u * static_cast<std::size_t>(first_atom);
      point_charge_positions.insert(
          point_charge_positions.end(),
          {positions[xyz] + 2.2, positions[xyz + 1u] + 0.4, positions[xyz + 2u] - 0.3});
      point_charge_charges.push_back(system % 2 == 0 ? 0.05 : -0.04);
    }
    status = make_external_point_charge_plan(basis, atomic_numbers.data(), batch_size,
                                             point_charge_offsets.data(), point_charge_plan, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }

  status = make_scc_driver_plan(
      wavefunction_layout, mulliken_plan, es2_plan, es3_plan, aes2_plan, eigensolver_plan,
      mixer_plan, options.enable_d4 ? &d4_plan : nullptr,
      options.enable_periodic_embedding ? &periodic_plan : nullptr, options.maximum_iterations,
      options.electronic_temperature, options.energy_tolerance, driver_plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  if (options.enable_periodic_embedding) {
    periodic_shifts.resize(static_cast<std::size_t>(total_atoms));
    for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
      const double magnitude = 0.004 * static_cast<double>(1 + atom % 3);
      periodic_shifts[static_cast<std::size_t>(atom)] = atom % 2 == 0 ? magnitude : -magnitude;
    }
    periodic_response_matrices.assign(
        static_cast<std::size_t>(periodic_plan.total_matrix_elements()), 0.0);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const std::int64_t atoms = atom_offsets[static_cast<std::size_t>(system + 1)] -
                                 atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t matrix_base =
          periodic_plan.matrix_offsets()[static_cast<std::size_t>(system)];
      for (std::int64_t row = 0; row < atoms; ++row) {
        for (std::int64_t column = 0; column < atoms; ++column) {
          const std::int64_t separation = row >= column ? row - column : column - row;
          const double value = row == column ? 0.02 + 0.001 * static_cast<double>(row)
                                             : 0.002 / static_cast<double>(1 + separation);
          const std::int64_t index = matrix_base + row * atoms + column;
          periodic_response_matrices[static_cast<std::size_t>(index)] = value;
        }
      }
    }
  }

  const std::size_t matrix_elements = static_cast<std::size_t>(integrals.total_matrix_elements);
  overlap.resize(matrix_elements);
  dipole_integrals.resize(3u * matrix_elements);
  quadrupole_integrals.resize(6u * matrix_elements);
  h0.resize(matrix_elements);
  status = allocate(integral_scratch, integrals.workspace_size_bytes, "integral workspace", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(),
                                integral_scratch.data(), integral_scratch.size(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = evaluate_multipole_cpu(basis, integrals, positions.data(), dipole_integrals.data(),
                                  quadrupole_integrals.data(), integral_scratch.data(),
                                  integral_scratch.size(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = evaluate_h0_cpu(basis, integrals, h0_plan, positions.data(), coordination_numbers.data(),
                           overlap.data(), h0.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  const std::size_t es2_elements = static_cast<std::size_t>(es2_plan.total_matrix_elements());
  status = allocate(es2_storage, es2_elements * sizeof(double), "ES2 cache", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(es2_scratch_storage, es2_elements * sizeof(double), "ES2 scratch", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  es2_scratch.matrix_scratch = static_cast<double*>(es2_scratch_storage.data());
  es2_scratch.matrix_elements = es2_plan.total_matrix_elements();
  status = update_es2_geometry_cache_cpu(es2_plan, positions.data(), options.geometry_generation,
                                         static_cast<double*>(es2_storage.data()), es2_elements,
                                         es2_scratch, es2_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  const std::size_t aes2_elements = static_cast<std::size_t>(aes2_plan.pair_data_elements());
  status = allocate(aes2_storage, aes2_elements * sizeof(double), "AES2 cache", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(aes2_scratch_storage, aes2_elements * sizeof(double), "AES2 scratch", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  aes2_scratch.pair_scratch = static_cast<double*>(aes2_scratch_storage.data());
  aes2_scratch.pair_elements = aes2_plan.pair_data_elements();
  status = update_aes2_geometry_cache_cpu(
      aes2_plan, positions.data(), coordination_numbers.data(), options.geometry_generation,
      static_cast<double*>(aes2_storage.data()), aes2_elements, aes2_scratch, aes2_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  if (options.enable_explicit_point_charges) {
    explicit_point_charge_shell_potential.resize(
        static_cast<std::size_t>(wavefunction_layout.total_shells));
    status = evaluate_external_point_charge_potential_cpu(
        point_charge_plan, positions.data(), point_charge_positions.data(),
        point_charge_charges.data(), point_charge_hardnesses.data(),
        explicit_point_charge_shell_potential.data(), error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }

  status =
      options.use_production_linalg
          ? make_mkl_rt_lp64_backend(cpu_backend, error)
          : make_internal_test_lp64_backend(&tiny_dpotrf, &tiny_dpocon, &tiny_dsyevd, &tiny_dtrsm,
                                            &tiny_dgemm, nullptr, cpu_backend, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(wavefunction_storage, wavefunction_layout.workspace_size_bytes, "wavefunction",
                    error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(overlap_cache_storage, eigensolver_plan.overlap_cache_size_bytes(),
                    "overlap cache", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(eigensolver_scratch_storage, eigensolver_plan.workspace_size_bytes(),
                    "eigensolver workspace", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(mixer_state_storage, mixer_plan.state_size_bytes(), "mixer state", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(driver_state_storage, driver_plan.state_size_bytes(), "driver state", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(driver_workspace_storage, driver_plan.workspace_size_bytes(),
                    "driver workspace", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  status = bind_wavefunction_view(wavefunction_layout, wavefunction_storage.data(),
                                  wavefunction_storage.size(), wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = initialize_sad_multipole_state(wavefunction_layout, wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = bind_eigensolver_overlap_cache(eigensolver_plan, overlap_cache_storage.data(),
                                          overlap_cache_storage.size(), overlap_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status =
      bind_eigensolver_workspace(eigensolver_plan, eigensolver_scratch_storage.data(),
                                 eigensolver_scratch_storage.size(), eigensolver_scratch, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = factor_overlap_cpu(eigensolver_plan, overlap.data(), options.geometry_generation,
                              cpu_backend, eigensolver_scratch, overlap_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = bind_scc_mixer_state(mixer_plan, mixer_state_storage.data(), mixer_state_storage.size(),
                                mixer_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = bind_scc_driver_state(driver_plan, driver_state_storage.data(),
                                 driver_state_storage.size(), driver_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = bind_scc_driver_workspace(driver_plan, driver_workspace_storage.data(),
                                     driver_workspace_storage.size(), driver_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status =
      initialize_scc_driver_state_cpu(driver_plan, wavefunction, mixer_state, driver_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  geometry.h0 = h0.data();
  geometry.h0_elements = integrals.total_matrix_elements;
  geometry.integrals = {overlap.data(), dipole_integrals.data(), quadrupole_integrals.data(),
                        integrals.total_matrix_elements, mulliken_plan.identity()};
  geometry.es2_cache = es2_cache;
  geometry.aes2_cache = aes2_cache;
  geometry.geometry_generation = options.geometry_generation;

  if (options.enable_explicit_point_charges) {
    geometry.explicit_point_charge_shell_potential = explicit_point_charge_shell_potential.data();
    geometry.explicit_point_charge_shell_elements = wavefunction_layout.total_shells;
  }
  if (options.enable_d4) {
    const std::size_t pair_elements =
        static_cast<std::size_t>(d4_plan.total_pairs()) * kD4PairDataElements;
    d4_pair_data.assign(std::max<std::size_t>(pair_elements, 1u), 0.0);
    d4_coordination.assign(static_cast<std::size_t>(total_atoms), 0.0);
    status = update_d4_geometry_cache_cpu(d4_plan, positions.data(), options.geometry_generation,
                                          d4_pair_data.data(), d4_pair_data.size(),
                                          d4_coordination.data(), d4_coordination.size(),
                                          driver_workspace.d4_workspace, d4_cache, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    geometry.d4_cache = d4_cache;
  }
  if (options.enable_periodic_embedding) {
    geometry.periodic_shifts = periodic_shifts.data();
    geometry.periodic_shift_elements = total_atoms;
    geometry.periodic_response_matrices = periodic_response_matrices.data();
    geometry.periodic_response_elements = periodic_plan.total_matrix_elements();
    geometry.periodic_embedding_generation = options.geometry_generation;
    geometry.periodic_plan_identity = periodic_plan.identity();
  }
  return XTBLOOM_STATUS_SUCCESS;
}

HostSccCase::HostSccCase() noexcept = default;
HostSccCase::~HostSccCase() = default;
HostSccCase::HostSccCase(HostSccCase&&) noexcept = default;
HostSccCase& HostSccCase::operator=(HostSccCase&&) noexcept = default;

xtbloom_status_t HostSccCase::create(const HostSccCaseOptions& options, HostSccCase& output,
                                     std::string& error) {
  try {
    auto candidate = std::make_unique<Impl>();
    candidate->options = options;
    const xtbloom_status_t status = candidate->build(error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    output.impl_ = std::move(candidate);
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate host SCC fixture metadata";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

bool HostSccCase::valid() const noexcept { return impl_ != nullptr; }

xtbloom_status_t HostSccCase::run_one_iteration(std::string& error) {
  if (!valid()) {
    error = "host SCC fixture is not initialized";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return iterate_scc_driver_batch_cpu(impl_->driver_plan, impl_->geometry, impl_->cpu_backend,
                                      impl_->overlap_cache, impl_->wavefunction, impl_->mixer_state,
                                      impl_->driver_state, impl_->driver_workspace, error);
}

HostSccCheckpoint HostSccCase::checkpoint() const {
  HostSccCheckpoint result;
  if (!valid()) {
    return result;
  }
  result.wavefunction = copy_bytes(impl_->wavefunction_storage);
  result.mixer_state = copy_bytes(impl_->mixer_state_storage);
  result.driver_state = copy_bytes(impl_->driver_state_storage);
  result.driver_workspace = copy_bytes(impl_->driver_workspace_storage);
  return result;
}

xtbloom_status_t HostSccCase::restore(const HostSccCheckpoint& checkpoint, std::string& error) {
  if (!valid()) {
    error = "host SCC fixture is not initialized";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (checkpoint.wavefunction.size() != impl_->wavefunction_storage.size() ||
      checkpoint.mixer_state.size() != impl_->mixer_state_storage.size() ||
      checkpoint.driver_state.size() != impl_->driver_state_storage.size() ||
      checkpoint.driver_workspace.size() != impl_->driver_workspace_storage.size()) {
    error = "host SCC checkpoint extents do not match this fixture";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  /* Extents are validated as one transaction before any destination changes. */
  std::memcpy(impl_->wavefunction_storage.data(), checkpoint.wavefunction.data(),
              checkpoint.wavefunction.size());
  std::memcpy(impl_->mixer_state_storage.data(), checkpoint.mixer_state.data(),
              checkpoint.mixer_state.size());
  std::memcpy(impl_->driver_state_storage.data(), checkpoint.driver_state.data(),
              checkpoint.driver_state.size());
  std::memcpy(impl_->driver_workspace_storage.data(), checkpoint.driver_workspace.data(),
              checkpoint.driver_workspace.size());
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

const HostSccCaseOptions& HostSccCase::options() const noexcept { return impl_->options; }
std::int64_t HostSccCase::batch_size() const noexcept { return impl_->batch_size; }
std::int64_t HostSccCase::total_atoms() const noexcept { return impl_->atom_offsets.back(); }
const std::vector<std::int64_t>& HostSccCase::atom_offsets() const noexcept {
  return impl_->atom_offsets;
}
const std::vector<std::int32_t>& HostSccCase::atomic_numbers() const noexcept {
  return impl_->atomic_numbers;
}
const std::vector<double>& HostSccCase::positions() const noexcept { return impl_->positions; }
const std::vector<double>& HostSccCase::molecular_charges() const noexcept {
  return impl_->molecular_charges;
}
const std::vector<std::int32_t>& HostSccCase::unpaired_electrons() const noexcept {
  return impl_->unpaired_electrons;
}
const std::vector<std::int32_t>& HostSccCase::spin_channels() const noexcept {
  return impl_->spin_channels;
}
const std::vector<double>& HostSccCase::coordination_numbers() const noexcept {
  return impl_->coordination_numbers;
}
const std::vector<std::int64_t>& HostSccCase::point_charge_offsets() const noexcept {
  return impl_->point_charge_offsets;
}
const std::vector<double>& HostSccCase::point_charge_positions() const noexcept {
  return impl_->point_charge_positions;
}
const std::vector<double>& HostSccCase::point_charge_charges() const noexcept {
  return impl_->point_charge_charges;
}
const std::vector<double>& HostSccCase::point_charge_hardnesses() const noexcept {
  return impl_->point_charge_hardnesses;
}
const std::vector<double>& HostSccCase::explicit_point_charge_shell_potential() const noexcept {
  return impl_->explicit_point_charge_shell_potential;
}
std::vector<double>& HostSccCase::explicit_point_charge_shell_potential() noexcept {
  return impl_->explicit_point_charge_shell_potential;
}
const std::vector<double>& HostSccCase::periodic_shifts() const noexcept {
  return impl_->periodic_shifts;
}
std::vector<double>& HostSccCase::periodic_shifts() noexcept { return impl_->periodic_shifts; }
const std::vector<double>& HostSccCase::periodic_response_matrices() const noexcept {
  return impl_->periodic_response_matrices;
}
std::vector<double>& HostSccCase::periodic_response_matrices() noexcept {
  return impl_->periodic_response_matrices;
}
const std::vector<double>& HostSccCase::overlap() const noexcept { return impl_->overlap; }
std::vector<double>& HostSccCase::overlap() noexcept { return impl_->overlap; }
const std::vector<double>& HostSccCase::dipole_integrals() const noexcept {
  return impl_->dipole_integrals;
}
std::vector<double>& HostSccCase::dipole_integrals() noexcept { return impl_->dipole_integrals; }
const std::vector<double>& HostSccCase::quadrupole_integrals() const noexcept {
  return impl_->quadrupole_integrals;
}
std::vector<double>& HostSccCase::quadrupole_integrals() noexcept {
  return impl_->quadrupole_integrals;
}
const std::vector<double>& HostSccCase::h0() const noexcept { return impl_->h0; }
std::vector<double>& HostSccCase::h0() noexcept { return impl_->h0; }

const BasisPlan& HostSccCase::basis_plan() const noexcept { return impl_->basis; }
const IntegralPlan& HostSccCase::integral_plan() const noexcept { return impl_->integrals; }
const H0Plan& HostSccCase::h0_plan() const noexcept { return impl_->h0_plan; }
const WavefunctionLayout& HostSccCase::wavefunction_layout() const noexcept {
  return impl_->wavefunction_layout;
}
const ES2Plan& HostSccCase::es2_plan() const noexcept { return impl_->es2_plan; }
const ES3Plan& HostSccCase::es3_plan() const noexcept { return impl_->es3_plan; }
const AES2Plan& HostSccCase::aes2_plan() const noexcept { return impl_->aes2_plan; }
const MullikenPlan& HostSccCase::mulliken_plan() const noexcept { return impl_->mulliken_plan; }
const EigensolverPlan& HostSccCase::eigensolver_plan() const noexcept {
  return impl_->eigensolver_plan;
}
const SccMixerPlan& HostSccCase::mixer_plan() const noexcept { return impl_->mixer_plan; }
const D4Plan* HostSccCase::d4_plan() const noexcept {
  return impl_->options.enable_d4 ? &impl_->d4_plan : nullptr;
}
const PeriodicEmbeddingPlan* HostSccCase::periodic_plan() const noexcept {
  return impl_->options.enable_periodic_embedding ? &impl_->periodic_plan : nullptr;
}
const ExternalPointChargePlan* HostSccCase::point_charge_plan() const noexcept {
  return impl_->options.enable_explicit_point_charges ? &impl_->point_charge_plan : nullptr;
}
const SccDriverPlan& HostSccCase::driver_plan() const noexcept { return impl_->driver_plan; }
const ES2GeometryCache& HostSccCase::es2_cache() const noexcept { return impl_->es2_cache; }
ES2GeometryCache& HostSccCase::es2_cache() noexcept { return impl_->es2_cache; }
const AES2GeometryCache& HostSccCase::aes2_cache() const noexcept { return impl_->aes2_cache; }
AES2GeometryCache& HostSccCase::aes2_cache() noexcept { return impl_->aes2_cache; }
const D4GeometryCache* HostSccCase::d4_cache() const noexcept {
  return impl_->options.enable_d4 ? &impl_->d4_cache : nullptr;
}
D4GeometryCache* HostSccCase::d4_cache() noexcept {
  return impl_->options.enable_d4 ? &impl_->d4_cache : nullptr;
}
const EigensolverOverlapCache& HostSccCase::overlap_cache() const noexcept {
  return impl_->overlap_cache;
}
const WavefunctionView& HostSccCase::wavefunction() const noexcept { return impl_->wavefunction; }
WavefunctionView& HostSccCase::wavefunction() noexcept { return impl_->wavefunction; }
const SccMixerState& HostSccCase::mixer_state() const noexcept { return impl_->mixer_state; }
SccMixerState& HostSccCase::mixer_state() noexcept { return impl_->mixer_state; }
const SccDriverState& HostSccCase::driver_state() const noexcept { return impl_->driver_state; }
SccDriverState& HostSccCase::driver_state() noexcept { return impl_->driver_state; }
const SccDriverWorkspace& HostSccCase::driver_workspace() const noexcept {
  return impl_->driver_workspace;
}
SccDriverWorkspace& HostSccCase::driver_workspace() noexcept { return impl_->driver_workspace; }
const SccDriverGeometryView& HostSccCase::geometry() const noexcept { return impl_->geometry; }
SccDriverGeometryView& HostSccCase::geometry() noexcept { return impl_->geometry; }
const CpuLinearAlgebraBackend& HostSccCase::cpu_backend() const noexcept {
  return impl_->cpu_backend;
}

}  // namespace xtbloom::test::gfn2
