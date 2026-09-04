#include <cuda_runtime_api.h>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/external_energy_device_evaluator.cuh"

namespace xtbloom::detail::cuda {
namespace {

/* The evaluator deliberately bounds local arrays.  This keeps one-thread-per
 * molecule execution graph-safe while rejecting pathological checkpoints at
 * setup instead of causing silent local-memory explosions in a kernel. */
constexpr int kMaxHidden = 256;
constexpr int kMaxElectronicWidth = 32;
constexpr double kCutoff = 6.0;
constexpr std::int64_t kInt64Max = 9223372036854775807LL;

struct MlpLayout {
  std::int64_t w0 = 0;
  std::int64_t b0 = 0;
  std::int64_t w1 = 0;
  std::int64_t b1 = 0;
  std::int64_t w2 = 0;
  std::int64_t b2 = 0;
};

struct ParameterLayout {
  MlpLayout geometry{};
  MlpLayout electronic{};
  MlpLayout condition{};
  MlpLayout fitting{};
  std::int64_t embedding = 0;
  std::int64_t constants = 0;
  std::int64_t total = 0;
};

__host__ __device__ inline bool checked_add(std::int64_t a, std::int64_t b, std::int64_t& out) {
  if (a < 0 || b < 0 || a > kInt64Max - b) return false;
  out = a + b;
  return true;
}

__host__ __device__ inline bool checked_mul(std::int64_t a, std::int64_t b, std::int64_t& out) {
  if (a < 0 || b < 0 || (a != 0 && b > kInt64Max / a)) {
    return false;
  }
  out = a * b;
  return true;
}

__host__ __device__ bool append_mlp(std::int64_t in, std::int64_t hidden, std::int64_t out,
                                    std::int64_t& cursor, MlpLayout& layout) noexcept {
  std::int64_t count = 0;
  layout.w0 = cursor;
  if (!checked_mul(hidden, in, count) || !checked_add(cursor, count, cursor)) return false;
  layout.b0 = cursor;
  if (!checked_add(cursor, hidden, cursor)) return false;
  layout.w1 = cursor;
  if (!checked_mul(hidden, hidden, count) || !checked_add(cursor, count, cursor)) return false;
  layout.b1 = cursor;
  if (!checked_add(cursor, hidden, cursor)) return false;
  layout.w2 = cursor;
  if (!checked_mul(out, hidden, count) || !checked_add(cursor, count, cursor)) return false;
  layout.b2 = cursor;
  if (!checked_add(cursor, out, cursor)) return false;
  return true;
}

__host__ __device__ bool make_layout(const ExternalEnergyDeviceModel& model,
                                     ParameterLayout& layout) noexcept {
  if (model.geometry_dim < 1 || model.electronic_dim < 1 || model.hidden_dim < 1 ||
      model.max_atomic_number < 1 || model.projection_width < 1 || model.radial_count < 1 ||
      model.geometry_dim != model.radial_count + 3 ||
      model.electronic_dim != 2 * model.projection_width || model.hidden_dim > kMaxHidden ||
      model.projection_width > kMaxElectronicWidth || model.radial_count > 256 ||
      model.geometry_dim > 259 || !isfinite(model.output_scale)) {
    return false;
  }
  std::int64_t cursor = 0;
  if (!append_mlp(model.geometry_dim, model.hidden_dim, model.hidden_dim, cursor,
                  layout.geometry) ||
      !append_mlp(model.electronic_dim, model.hidden_dim, model.hidden_dim, cursor,
                  layout.electronic) ||
      !append_mlp(3, model.hidden_dim, 2 * model.hidden_dim, cursor, layout.condition) ||
      !append_mlp(model.hidden_dim, model.hidden_dim, 1, cursor, layout.fitting)) {
    return false;
  }
  if (!checked_mul(model.max_atomic_number + 1, model.hidden_dim, layout.embedding) ||
      !checked_add(cursor, layout.embedding, cursor) ||
      !checked_add(cursor, model.max_atomic_number + 1, cursor)) {
    return false;
  }
  layout.constants = cursor - (model.max_atomic_number + 1);
  layout.total = cursor;
  return model.parameters != nullptr && model.parameter_elements == layout.total;
}

__device__ inline double gelu(double x) {
  return 0.5 * x * (1.0 + erf(x * 0.70710678118654752440));
}

__device__ inline double gelu_prime(double x) {
  constexpr double inv_sqrt_2pi = 0.39894228040143267794;
  return 0.5 * (1.0 + erf(x * 0.70710678118654752440)) + x * inv_sqrt_2pi * exp(-0.5 * x * x);
}

__device__ inline void grad_add(double* gradient, std::int64_t index, double value) {
  if (gradient != nullptr) (void)atomic_add_fp64(gradient + index, value);
}

__device__ void mlp_forward(const double* p, const MlpLayout& layout, int in, int hidden, int out,
                            const double* x, double* z0, double* a0, double* z1, double* a1,
                            double* y) {
  for (int i = 0; i < hidden; ++i) {
    double value = p[layout.b0 + i];
    for (int j = 0; j < in; ++j) value += p[layout.w0 + i * in + j] * x[j];
    z0[i] = value;
    a0[i] = gelu(value);
  }
  for (int i = 0; i < hidden; ++i) {
    double value = p[layout.b1 + i];
    for (int j = 0; j < hidden; ++j) value += p[layout.w1 + i * hidden + j] * a0[j];
    z1[i] = value;
    a1[i] = gelu(value);
  }
  for (int i = 0; i < out; ++i) {
    double value = p[layout.b2 + i];
    for (int j = 0; j < hidden; ++j) value += p[layout.w2 + i * hidden + j] * a1[j];
    y[i] = value;
  }
}

__device__ void mlp_input_vjp(const double* p, const MlpLayout& layout, int in, int hidden, int out,
                              const double* z0, const double* a0, const double* z1,
                              const double* a1, const double* upstream, double* result,
                              double* scratch0, double* scratch1) {
  for (int j = 0; j < hidden; ++j) {
    double value = 0.0;
    for (int i = 0; i < out; ++i) value += p[layout.w2 + i * hidden + j] * upstream[i];
    scratch1[j] = value * gelu_prime(z1[j]);
  }
  for (int j = 0; j < hidden; ++j) {
    double value = 0.0;
    for (int i = 0; i < hidden; ++i) value += p[layout.w1 + i * hidden + j] * scratch1[i];
    scratch0[j] = value * gelu_prime(z0[j]);
  }
  for (int j = 0; j < in; ++j) {
    double value = 0.0;
    for (int i = 0; i < hidden; ++i) value += p[layout.w0 + i * in + j] * scratch0[i];
    result[j] = value;
  }
  (void)a0;
  (void)a1;
}

/* Reverse one MLP and accumulate its parameter gradient into the context
 * checkpoint.  Each SCC batch member owns one CUDA block, so atom updates
 * within a member are serialized; atomics only arbitrate between members.
 * This keeps the gradient deterministic enough for optimization while
 * avoiding a batch-sized parameter-gradient workspace. */
__device__ void mlp_backward_accumulate(const double* p, const MlpLayout& layout, int in,
                                        int hidden, int out, const double* x, const double* z0,
                                        const double* a0, const double* z1, const double* a1,
                                        const double* output_grad, double* input_grad,
                                        double* scratch0, double* scratch1,
                                        double* parameter_gradient) {
  for (int i = 0; i < out; ++i) {
    grad_add(parameter_gradient, layout.b2 + i, output_grad[i]);
    for (int j = 0; j < hidden; ++j) {
      grad_add(parameter_gradient, layout.w2 + i * hidden + j, output_grad[i] * a1[j]);
    }
  }
  for (int j = 0; j < hidden; ++j) {
    double value = 0.0;
    for (int i = 0; i < out; ++i) {
      value += p[layout.w2 + i * hidden + j] * output_grad[i];
    }
    scratch1[j] = value * gelu_prime(z1[j]);
    grad_add(parameter_gradient, layout.b1 + j, scratch1[j]);
    for (int k = 0; k < hidden; ++k) {
      grad_add(parameter_gradient, layout.w1 + j * hidden + k, scratch1[j] * a0[k]);
    }
  }
  for (int k = 0; k < hidden; ++k) {
    double value = 0.0;
    for (int j = 0; j < hidden; ++j) {
      value += p[layout.w1 + j * hidden + k] * scratch1[j];
    }
    scratch0[k] = value * gelu_prime(z0[k]);
    grad_add(parameter_gradient, layout.b0 + k, scratch0[k]);
    for (int j = 0; j < in; ++j) {
      grad_add(parameter_gradient, layout.w0 + k * in + j, scratch0[k] * x[j]);
    }
  }
  for (int j = 0; j < in; ++j) {
    double value = 0.0;
    for (int k = 0; k < hidden; ++k) {
      value += p[layout.w0 + k * in + j] * scratch0[k];
    }
    input_grad[j] = value;
  }
}

__device__ void geometry_features(const ExternalEnergyDeviceInput& input, std::int64_t system,
                                  std::int64_t atom, int radial_count, double* features) {
  const std::int64_t atom_begin = input.atom_offsets[system];
  const std::int64_t atom_end = input.atom_offsets[system + 1];
  const double* center = input.positions + 3 * (atom_begin + atom);
  for (int k = 0; k < radial_count + 3; ++k) features[k] = 0.0;
  for (std::int64_t neighbor = 0; neighbor < atom_end - atom_begin; ++neighbor) {
    if (neighbor == atom) continue;
    const double* point = input.positions + 3 * (atom_begin + neighbor);
    const double dx = center[0] - point[0];
    const double dy = center[1] - point[1];
    const double dz = center[2] - point[2];
    const double distance = sqrt(dx * dx + dy * dy + dz * dz);
    if (!(distance < kCutoff)) continue;
    const double scaled = distance / kCutoff;
    const double envelope = 0.5 * (cos(3.14159265358979323846 * scaled) + 1.0);
    const double charge = static_cast<double>(input.atomic_numbers[atom_begin + neighbor]) / 118.0;
    const double spacing = kCutoff / static_cast<double>(radial_count);
    const double width = 1.0 / (spacing * spacing);
    for (int radial = 0; radial < radial_count; ++radial) {
      const double fraction =
          radial_count == 1 ? 0.0
                            : static_cast<double>(radial) / static_cast<double>(radial_count - 1);
      const double center_radius = kCutoff * (0.25 + 0.70 * fraction);
      const double basis = exp(-width * (distance - center_radius) * (distance - center_radius));
      features[radial] += envelope * basis * charge;
    }
    features[radial_count] += envelope * charge;
    features[radial_count + 1] += envelope / fmax(distance, 1.0e-6);
    features[radial_count + 2] += envelope;
  }
}

__device__ inline std::int64_t spin_channel_index(const Gfn2WavefunctionLayoutView& layout,
                                                  std::int64_t system, std::int64_t channel) {
  return layout.spin_channel_offsets[system] + channel;
}

__device__ inline std::int64_t density_matrix_begin(const Gfn2WavefunctionLayoutView& layout,
                                                    std::int64_t system, std::int64_t channel) {
  return layout.spin_matrix_offsets[spin_channel_index(layout, system, channel)];
}

__device__ double projected_diagonal(const ExternalEnergyDeviceInput& input, std::int64_t system,
                                     std::int64_t atom, std::int64_t slot, int width,
                                     std::int64_t channel, const double* density) {
  const std::int64_t atom_begin = input.atom_offsets[system];
  const std::int64_t orbital_begin = input.batch_orbital_offsets[system];
  const std::int64_t orbital_end = input.batch_orbital_offsets[system + 1];
  std::int64_t selected = 0;
  std::int64_t orbital = -1;
  for (std::int64_t ao = orbital_begin; ao < orbital_end; ++ao) {
    if (input.orbital_to_atom[ao] != atom_begin + atom) continue;
    if (selected == slot) {
      orbital = ao - orbital_begin;
      break;
    }
    ++selected;
  }
  if (orbital < 0 || slot >= width) return 0.0;
  const std::int64_t nao = orbital_end - orbital_begin;
  const std::int64_t matrix_begin =
      density_matrix_begin(input.wavefunction_layout, system, channel);
  const std::int64_t matrix_base = input.matrix_offsets[system];
  double value = 0.0;
  for (std::int64_t row = 0; row < nao; ++row) {
    for (std::int64_t col = 0; col < nao; ++col) {
      const double left = input.overlap[matrix_base + row * nao + orbital];
      const double right = input.overlap[matrix_base + col * nao + orbital];
      value += left * density[matrix_begin + row * nao + col] * right;
    }
  }
  return value;
}

__device__ double evaluate_atom(const ExternalEnergyDeviceModel& model,
                                const ParameterLayout& layout,
                                const ExternalEnergyDeviceInput& input, std::int64_t system,
                                std::int64_t atom, const double* density, double* d_charge,
                                double* d_spin, double* geometry_input_gradient,
                                double* parameter_gradient) {
  const int hidden = static_cast<int>(model.hidden_dim);
  const int radial = static_cast<int>(model.radial_count);
  const int width = static_cast<int>(model.projection_width);
  const int z = static_cast<int>(input.atomic_numbers[input.atom_offsets[system] + atom]);
  double geo[259], elec[64], condition[3], geo_z0[kMaxHidden], geo_a0[kMaxHidden],
      geo_z1[kMaxHidden], geo_a1[kMaxHidden], geo_y[kMaxHidden], elec_z0[kMaxHidden],
      elec_a0[kMaxHidden], elec_z1[kMaxHidden], elec_a1[kMaxHidden], elec_y[kMaxHidden],
      cond_z0[kMaxHidden], cond_a0[kMaxHidden], cond_z1[kMaxHidden], cond_a1[kMaxHidden],
      cond_y[2 * kMaxHidden], fit_z0[kMaxHidden], fit_a0[kMaxHidden], fit_z1[kMaxHidden],
      fit_a1[kMaxHidden], fit_y[1], h[kMaxHidden], fit_upstream[1], fit_grad[kMaxHidden],
      fit_s0[kMaxHidden], fit_s1[kMaxHidden], geo_upstream[kMaxHidden], geo_s0[kMaxHidden],
      geo_s1[kMaxHidden], cond_upstream[2 * kMaxHidden], cond_s0[kMaxHidden], cond_s1[kMaxHidden],
      elec_upstream[kMaxHidden], elec_s0[kMaxHidden], elec_s1[kMaxHidden], elec_grad[64];
  /* The public topology validator bounds atomic numbers for xTB itself, but
   * the private model may intentionally contain a smaller embedding table.
   * Fail closed before indexing the uploaded arena if such a model/input pair
   * is supplied through the low-level C API. */
  if (z < 0 || z > model.max_atomic_number) {
    for (int i = 0; i < width; ++i) {
      d_charge[i] = 0.0;
      d_spin[i] = 0.0;
    }
    return 0.0;
  }
  geometry_features(input, system, atom, radial, geo);
  const std::int64_t channels = input.spin_channels[system];
  for (int i = 0; i < width; ++i) {
    const double alpha = projected_diagonal(input, system, atom, i, width, 0, density);
    const double beta =
        channels == 2 ? projected_diagonal(input, system, atom, i, width, 1, density) : 0.0;
    elec[i] = alpha + beta;
    elec[width + i] = alpha - beta;
  }
  condition[0] = input.molecular_charges[system];
  condition[1] = static_cast<double>(input.unpaired_electrons[system]);
  condition[2] = static_cast<double>(input.spin_channels[system]);
  const double* p = model.parameters;
  mlp_forward(p, layout.geometry, static_cast<int>(model.geometry_dim), hidden, hidden, geo, geo_z0,
              geo_a0, geo_z1, geo_a1, geo_y);
  mlp_forward(p, layout.electronic, static_cast<int>(model.electronic_dim), hidden, hidden, elec,
              elec_z0, elec_a0, elec_z1, elec_a1, elec_y);
  mlp_forward(p, layout.condition, 3, hidden, 2 * hidden, condition, cond_z0, cond_a0, cond_z1,
              cond_a1, cond_y);
  for (int i = 0; i < hidden; ++i) {
    h[i] = (geo_y[i] + elec_y[i] + p[layout.embedding + z * hidden + i]) * (1.0 + cond_y[i]) +
           cond_y[hidden + i];
  }
  mlp_forward(p, layout.fitting, hidden, hidden, 1, h, fit_z0, fit_a0, fit_z1, fit_a1, fit_y);
  fit_upstream[0] = model.output_scale;
  if (parameter_gradient != nullptr) {
    mlp_backward_accumulate(p, layout.fitting, hidden, hidden, 1, h, fit_z0, fit_a0, fit_z1, fit_a1,
                            fit_upstream, fit_grad, fit_s0, fit_s1, parameter_gradient);
  } else {
    mlp_input_vjp(p, layout.fitting, hidden, hidden, 1, fit_z0, fit_a0, fit_z1, fit_a1,
                  fit_upstream, fit_grad, fit_s0, fit_s1);
  }
  for (int i = 0; i < hidden; ++i) elec_upstream[i] = fit_grad[i] * (1.0 + cond_y[i]);
  for (int i = 0; i < hidden; ++i) {
    const double common = geo_y[i] + elec_y[i] + p[layout.embedding + z * hidden + i];
    geo_upstream[i] = fit_grad[i] * (1.0 + cond_y[i]);
    cond_upstream[i] = fit_grad[i] * common;
    cond_upstream[hidden + i] = fit_grad[i];
  }
  if (parameter_gradient != nullptr) {
    for (int i = 0; i < hidden; ++i) {
      grad_add(parameter_gradient, layout.embedding + z * hidden + i, geo_upstream[i]);
    }
    grad_add(parameter_gradient, layout.constants + z, 1.0);
    mlp_backward_accumulate(p, layout.condition, 3, hidden, 2 * hidden, condition, cond_z0, cond_a0,
                            cond_z1, cond_a1, cond_upstream, cond_z0, cond_s0, cond_s1,
                            parameter_gradient);
    mlp_backward_accumulate(p, layout.geometry, static_cast<int>(model.geometry_dim), hidden,
                            hidden, geo, geo_z0, geo_a0, geo_z1, geo_a1, geo_upstream, fit_s0,
                            geo_s0, geo_s1, parameter_gradient);
    mlp_backward_accumulate(p, layout.electronic, static_cast<int>(model.electronic_dim), hidden,
                            hidden, elec, elec_z0, elec_a0, elec_z1, elec_a1, elec_upstream,
                            elec_grad, elec_s0, elec_s1, parameter_gradient);
  } else {
    mlp_input_vjp(p, layout.electronic, static_cast<int>(model.electronic_dim), hidden, hidden,
                  elec_z0, elec_a0, elec_z1, elec_a1, elec_upstream, elec_grad, elec_s0, elec_s1);
  }
  if (geometry_input_gradient != nullptr) {
    /* Geometry and condition branches are independent after the feature
     * fusion.  Reuse the exact upstream produced above and retain this VJP
     * for the device-side Cartesian geometry reverse pass. */
    mlp_input_vjp(p, layout.geometry, static_cast<int>(model.geometry_dim), hidden, hidden, geo_z0,
                  geo_a0, geo_z1, geo_a1, geo_upstream, geometry_input_gradient, geo_s0, geo_s1);
  }
  for (int i = 0; i < width; ++i) {
    d_charge[i] = elec_grad[i];
    d_spin[i] = elec_grad[width + i];
  }
  return fit_y[0] * model.output_scale + p[layout.constants + z];
}

/* Add one atom's projected-density derivative to both spin Hamiltonians.  The
 * evaluator runs one lane per molecule, so recomputing the AO factor for each
 * atom is deterministic and avoids a fixed-size atom scratch table (which
 * would otherwise overflow for molecules larger than 32 atoms). */
__device__ void add_projected_potential(const ExternalEnergyDeviceModel& model,
                                        const ExternalEnergyDeviceInput& input, std::int64_t system,
                                        std::int64_t atom, const double* d_charge,
                                        const double* d_spin, double* hamiltonian) {
  const int width = static_cast<int>(model.projection_width);
  const std::int64_t orbital_begin = input.batch_orbital_offsets[system];
  const std::int64_t orbital_end = input.batch_orbital_offsets[system + 1];
  const std::int64_t matrix_size = orbital_end - orbital_begin;
  const std::int64_t overlap_base = input.matrix_offsets[system];
  const std::int64_t atom_begin = input.atom_offsets[system];
  const std::int64_t channels = input.spin_channels[system];
  const std::int64_t target_atom = atom_begin + atom;
  for (std::int64_t row = 0; row < matrix_size; ++row) {
    for (std::int64_t col = 0; col < matrix_size; ++col) {
      double charge_value = 0.0;
      double spin_value = 0.0;
      for (int slot = 0; slot < width; ++slot) {
        std::int64_t selected = 0;
        std::int64_t ao = -1;
        for (std::int64_t candidate = orbital_begin; candidate < orbital_end; ++candidate) {
          if (input.orbital_to_atom[candidate] != target_atom) continue;
          if (selected == slot) {
            ao = candidate;
            break;
          }
          ++selected;
        }
        if (ao >= 0) {
          const double factor =
              input.overlap[overlap_base + row * matrix_size + (ao - orbital_begin)] *
              input.overlap[overlap_base + col * matrix_size + (ao - orbital_begin)];
          charge_value += d_charge[slot] * factor;
          spin_value += d_spin[slot] * factor;
        }
      }
      const std::int64_t matrix_base = density_matrix_begin(input.wavefunction_layout, system, 0);
      hamiltonian[matrix_base + row * matrix_size + col] += charge_value + spin_value;
      if (channels == 2) {
        const std::int64_t beta_base = density_matrix_begin(input.wavefunction_layout, system, 1);
        hamiltonian[beta_base + row * matrix_size + col] += charge_value - spin_value;
      }
    }
  }
}

__global__ void evaluate_kernel(ExternalEnergyDeviceModel model, ExternalEnergyDeviceInput input,
                                ExternalEnergyDeviceOutput output, const std::uint8_t* active_mask,
                                bool potential_phase) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || system >= input.batch_size ||
      (active_mask != nullptr && active_mask[system] != 1u)) {
    return;
  }
  ParameterLayout layout{};
  if (!make_layout(model, layout)) {
    return;
  }
  const double* density = potential_phase ? input.current_density : input.staged_density;
  if (density == nullptr) return;
  const std::int64_t atom_begin = input.atom_offsets[system];
  const std::int64_t atom_end = input.atom_offsets[system + 1];
  double charge[32], spin[32];
  double total = 0.0;
  /* Per-system execution avoids atom-level races and makes the SCC update
   * deterministic for both direct launches and CUDA Graph replay. */
  for (std::int64_t atom = 0; atom < atom_end - atom_begin; ++atom) {
    total += evaluate_atom(model, layout, input, system, atom, density, charge, spin, nullptr,
                           potential_phase ? nullptr : output.parameter_gradient);
    if (potential_phase) {
      add_projected_potential(model, input, system, atom, charge, spin, output.hamiltonian);
    }
  }
  if (!potential_phase && output.energy != nullptr) {
    output.energy[system] = total;
  }
  if (!potential_phase && output.energy_accumulator != nullptr) {
    output.energy_accumulator[system] += total;
  }
}

/* Derivative of the fixed radial feature map used by the native evaluator.
 * Returning d(feature)/d(distance) keeps the Cartesian reverse pass compact
 * and guarantees the force path differentiates the same cutoff/envelope as
 * geometry_features(). */
__device__ void geometry_feature_distance_derivative(const ExternalEnergyDeviceInput& input,
                                                     std::int64_t system, std::int64_t atom,
                                                     std::int64_t neighbor, int radial_count,
                                                     const double* feature_gradient,
                                                     double& distance_gradient) {
  const std::int64_t atom_begin = input.atom_offsets[system];
  const double* center = input.positions + 3 * (atom_begin + atom);
  const double* point = input.positions + 3 * (atom_begin + neighbor);
  const double dx = center[0] - point[0];
  const double dy = center[1] - point[1];
  const double dz = center[2] - point[2];
  const double distance = sqrt(dx * dx + dy * dy + dz * dz);
  distance_gradient = 0.0;
  if (!(distance > 1.0e-12) || !(distance < kCutoff) || !isfinite(distance)) return;
  const double scaled = distance / kCutoff;
  const double pi = 3.14159265358979323846;
  const double envelope = 0.5 * (cos(pi * scaled) + 1.0);
  const double envelope_derivative = -0.5 * pi / kCutoff * sin(pi * scaled);
  const double charge = static_cast<double>(input.atomic_numbers[atom_begin + neighbor]) / 118.0;
  const double spacing = kCutoff / static_cast<double>(radial_count);
  const double width = 1.0 / (spacing * spacing);
  for (int radial = 0; radial < radial_count; ++radial) {
    const double fraction =
        radial_count == 1 ? 0.0
                          : static_cast<double>(radial) / static_cast<double>(radial_count - 1);
    const double center_radius = kCutoff * (0.25 + 0.70 * fraction);
    const double delta = distance - center_radius;
    const double basis = exp(-width * delta * delta);
    const double basis_derivative = -2.0 * width * delta * basis;
    distance_gradient += feature_gradient[radial] * charge *
                         (envelope_derivative * basis + envelope * basis_derivative);
  }
  distance_gradient += feature_gradient[radial_count] * charge * envelope_derivative;
  distance_gradient += feature_gradient[radial_count + 1] *
                       (envelope_derivative / distance - envelope / (distance * distance));
  distance_gradient += feature_gradient[radial_count + 2] * envelope_derivative;
}

__device__ void add_projected_overlap_adjoint(const ExternalEnergyDeviceInput& input,
                                              std::int64_t system, std::int64_t atom, int slot,
                                              const double* density, std::int64_t channel,
                                              double coefficient, double* overlap_adjoint) {
  if (coefficient == 0.0) return;
  const std::int64_t orbital_begin = input.batch_orbital_offsets[system];
  const std::int64_t orbital_end = input.batch_orbital_offsets[system + 1];
  const std::int64_t nao = orbital_end - orbital_begin;
  const std::int64_t matrix_base = input.matrix_offsets[system];
  const std::int64_t matrix_begin =
      density_matrix_begin(input.wavefunction_layout, system, channel);
  const std::int64_t target_atom = input.atom_offsets[system] + atom;
  std::int64_t selected = 0;
  std::int64_t ao = -1;
  for (std::int64_t candidate = orbital_begin; candidate < orbital_end; ++candidate) {
    if (input.orbital_to_atom[candidate] != target_atom) continue;
    if (selected == slot) {
      ao = candidate - orbital_begin;
      break;
    }
    ++selected;
  }
  if (ao < 0) return;
  for (std::int64_t row = 0; row < nao; ++row) {
    double left = 0.0;
    double right = 0.0;
    for (std::int64_t col = 0; col < nao; ++col) {
      left += density[matrix_begin + row * nao + col] * input.overlap[matrix_base + col * nao + ao];
      right +=
          density[matrix_begin + col * nao + row] * input.overlap[matrix_base + col * nao + ao];
    }
    const std::int64_t row_index = matrix_base + row * nao + ao;
    const std::int64_t column_index = matrix_base + ao * nao + row;
    overlap_adjoint[row_index] += coefficient * left;
    overlap_adjoint[column_index] += coefficient * right;
  }
}

__global__ void evaluate_force_kernel(ExternalEnergyDeviceModel model,
                                      ExternalEnergyDeviceInput input,
                                      ExternalEnergyDeviceOutput output,
                                      const std::uint8_t* active_mask) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || system >= input.batch_size ||
      (active_mask != nullptr && active_mask[system] != 1u)) {
    return;
  }
  ParameterLayout layout{};
  if (!make_layout(model, layout) || input.current_density == nullptr) return;
  const int radial = static_cast<int>(model.radial_count);
  const int width = static_cast<int>(model.projection_width);
  const std::int64_t atom_begin = input.atom_offsets[system];
  const std::int64_t atom_end = input.atom_offsets[system + 1];
  const std::int64_t matrix_begin = input.matrix_offsets[system];
  const std::int64_t matrix_end = input.matrix_offsets[system + 1];
  for (std::int64_t matrix = matrix_begin; matrix < matrix_end; ++matrix) {
    output.overlap_adjoint[matrix] = 0.0;
  }
  for (std::int64_t coordinate = atom_begin * 3; coordinate < atom_end * 3; ++coordinate) {
    output.geometry_gradient[coordinate] = 0.0;
  }
  double charge[32], spin[32], geo_grad[259];
  for (std::int64_t atom = 0; atom < atom_end - atom_begin; ++atom) {
    for (int slot = 0; slot < width; ++slot) {
      charge[slot] = 0.0;
      spin[slot] = 0.0;
    }
    for (int k = 0; k < radial + 3; ++k) geo_grad[k] = 0.0;
    (void)evaluate_atom(model, layout, input, system, atom, input.current_density, charge, spin,
                        geo_grad, nullptr);
    /* dE/dS from the projected alpha/beta density channels. */
    const std::int64_t channels = input.spin_channels[system];
    for (int slot = 0; slot < width; ++slot) {
      add_projected_overlap_adjoint(input, system, atom, slot, input.current_density, 0,
                                    charge[slot] + spin[slot], output.overlap_adjoint);
      if (channels == 2) {
        add_projected_overlap_adjoint(input, system, atom, slot, input.current_density, 1,
                                      charge[slot] - spin[slot], output.overlap_adjoint);
      }
    }
    /* Explicit geometry branch dE/dR at fixed density. */
    for (std::int64_t neighbor = 0; neighbor < atom_end - atom_begin; ++neighbor) {
      if (neighbor == atom) continue;
      const std::int64_t absolute_atom = atom_begin + atom;
      const std::int64_t absolute_neighbor = atom_begin + neighbor;
      const double dx = input.positions[3 * absolute_atom] - input.positions[3 * absolute_neighbor];
      const double dy =
          input.positions[3 * absolute_atom + 1] - input.positions[3 * absolute_neighbor + 1];
      const double dz =
          input.positions[3 * absolute_atom + 2] - input.positions[3 * absolute_neighbor + 2];
      const double distance = sqrt(dx * dx + dy * dy + dz * dz);
      if (!(distance > 1.0e-12) || !(distance < kCutoff)) continue;
      double distance_gradient = 0.0;
      geometry_feature_distance_derivative(input, system, atom, neighbor, radial, geo_grad,
                                           distance_gradient);
      const double scale = distance_gradient / distance;
      output.geometry_gradient[3 * absolute_atom] += scale * dx;
      output.geometry_gradient[3 * absolute_atom + 1] += scale * dy;
      output.geometry_gradient[3 * absolute_atom + 2] += scale * dz;
      output.geometry_gradient[3 * absolute_neighbor] -= scale * dx;
      output.geometry_gradient[3 * absolute_neighbor + 1] -= scale * dy;
      output.geometry_gradient[3 * absolute_neighbor + 2] -= scale * dz;
    }
  }
}

}  // namespace

bool validate_external_energy_device_model(const ExternalEnergyDeviceModel& model) noexcept {
  ParameterLayout layout{};
  return make_layout(model, layout);
}

cudaError_t evaluate_external_energy_device_potential_cuda(const ExternalEnergyDeviceModel& model,
                                                           const ExternalEnergyDeviceInput& input,
                                                           const ExternalEnergyDeviceOutput& output,
                                                           const std::uint8_t* active_mask,
                                                           cudaStream_t stream) noexcept {
  evaluate_kernel<<<static_cast<unsigned int>(input.batch_size), 1, 0, stream>>>(
      model, input, output, active_mask, true);
  return cudaPeekAtLastError();
}

cudaError_t evaluate_external_energy_device_energy_cuda(const ExternalEnergyDeviceModel& model,
                                                        const ExternalEnergyDeviceInput& input,
                                                        const ExternalEnergyDeviceOutput& output,
                                                        const std::uint8_t* active_mask,
                                                        cudaStream_t stream) noexcept {
  if (output.parameter_gradient != nullptr && output.parameter_gradient_elements > 0) {
    cudaError_t reset = cudaMemsetAsync(
        output.parameter_gradient, 0,
        static_cast<std::size_t>(output.parameter_gradient_elements) * sizeof(double), stream);
    if (reset != cudaSuccess) return reset;
  }
  evaluate_kernel<<<static_cast<unsigned int>(input.batch_size), 1, 0, stream>>>(
      model, input, output, active_mask, false);
  return cudaPeekAtLastError();
}

cudaError_t evaluate_external_energy_device_force_cuda(const ExternalEnergyDeviceModel& model,
                                                       const ExternalEnergyDeviceInput& input,
                                                       const ExternalEnergyDeviceOutput& output,
                                                       const std::uint8_t* active_mask,
                                                       cudaStream_t stream) noexcept {
  if (output.overlap_adjoint == nullptr || output.geometry_gradient == nullptr ||
      output.overlap_adjoint_elements < input.total_matrix_elements ||
      output.geometry_gradient_elements < input.total_atoms * 3) {
    return cudaErrorInvalidValue;
  }
  evaluate_force_kernel<<<static_cast<unsigned int>(input.batch_size), 1, 0, stream>>>(
      model, input, output, active_mask);
  return cudaPeekAtLastError();
}

}  // namespace xtbloom::detail::cuda
