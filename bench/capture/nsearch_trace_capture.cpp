// Runs a learnsph scene and records consecutive neighbor searches at given
// simulated times. Built by scripts/capture_traces.sh inside a copy of learnsph.
//
// Usage:
//   nsearch_trace_capture <scene> <radius> <pbf-iterations> <out-prefix>
//                         <searches-per-window> <time> [time ...]
//
// Each window is written to <out-prefix>_t<time>.trace.
#include "../learnSPH/SPHSolver.h"
#include "../learnSPH/io.h"
#include "../learnSPH/kernel.h"
#include "../learnSPH/sampling.h"

#include <Eigen/Core>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using learnSPH::solver::SPHSolver;
using Vec = Eigen::Vector3d;

std::string mesh(const std::string &name) {
  return std::string(LEARNSPH_MESH_DIR) + "/" + name;
}

std::array<Vec, 2> mesh_bounds(const std::string &name) {
  const auto box = learnSPH::read_tri_meshes_from_obj(mesh(name))[0];
  std::array<Vec, 2> bounds{box.vertices[0], box.vertices[0]};
  for (const auto &v : box.vertices) {
    bounds[0] = bounds[0].cwiseMin(v);
    bounds[1] = bounds[1].cwiseMax(v);
  }
  return bounds;
}

void add_box(SPHSolver &solver, const Vec &bottom, const Vec &top, double radius) {
  std::vector<Vec> fluid;
  learnSPH::sampling::fluid_box(fluid, bottom, top, 2 * radius);
  const Vec size = top - bottom;
  solver.add_point_set(fluid, size.x() * size.y() * size.z());
}

void common(SPHSolver &solver, double max_step, double fluid_viscosity = 0.0025,
            double boundary_viscosity = 0.0) {
  solver.set_boundary_viscosity(boundary_viscosity);
  solver.set_fluid_viscosity(fluid_viscosity);
  solver.set_max_timestep(max_step);
  solver.set_min_timestep(0.0001);
  solver.set_pressure_stiffness(1000.0);
}

void setup(SPHSolver &solver, const std::string &scene, double radius) {
  const double pi = learnSPH::kernel::PI;
  if (scene == "dam_break") {
    const Vec offset{0.0, -0.4, -0.5};
    add_box(solver, Vec{-0.075, 0, 0} + offset, Vec{0.075, 0.25, 0.5} + offset, radius);
    solver.set_bounding_box({-0.5, -1, -1}, {0.5, 1, 2});
    solver.add_boundary_object(mesh("boundary.obj"));
    common(solver, 0.001);
  } else if (scene == "wall") {
    const auto box = mesh_bounds("fluid_box.obj");
    add_box(solver, box[0], box[1], radius);
    solver.set_bounding_box({-1.5, -1.5, -0.5}, {1.5, 1.5, 2.5});
    solver.add_boundary_object(mesh("wall.obj"));
    solver.add_boundary_object(mesh("box.obj"));
    common(solver, 0.001);
  } else if (scene == "bowl") {
    const auto box = mesh_bounds("water_box_bowl.obj");
    add_box(solver, box[0], box[1], radius);
    solver.set_bounding_box({-1.5, -1.5, -0.5}, {1.5, 1.5, 2.5});
    solver.add_boundary_object(mesh("bowl.obj"));
    common(solver, 0.001);
  } else if (scene == "sink") {
    const Vec offset{0.0, -0.5, 0.1};
    add_box(solver, Vec{-0.5, 0, 0} + offset, Vec{0.5, 1.0, 0.2} + offset, radius);
    solver.set_bounding_box({-2, -2, -1}, {2, 2, 12});
    solver.add_boundary_object(mesh("sink.obj"));
    common(solver, 0.001);
  } else if (scene == "droplet") {
    const auto box = mesh_bounds("fluid_box.obj");
    add_box(solver, box[0], box[1], radius);
    solver.set_bounding_box({-1.5, -1.5, -0.5}, {1.5, 1.5, 1.5});
    solver.set_adhesion_coefficient(0.0);
    solver.set_gravity({0, 0, 0});
    solver.set_surface_tension_coefficient(1.5);
    solver.add_boundary_object(mesh("wall.obj"));
    common(solver, 0.001);
  } else if (scene == "funnel") {
    solver.set_surface_tension_coefficient(1.0);
    solver.set_bounding_box({-3, -3, -5}, {3, 3, 3});
    solver.add_boundary_object(mesh("funnel4.obj"));
    common(solver, 0.0002);
    solver.add_emitter({-1.4, -2, 1.5}, {0, 0, -4}, {pi / 2.5, 0, pi / 2.5}, 1.1,
                       {0.4, 0.4, 0.4});
    solver.add_emitter({2, 1.4, 1.5}, {0, 0, -4}, {-pi / 2.5, 0, pi / 2.5}, 1.1,
                       {0.4, 0.4, 0.4});
  } else if (scene == "ramp_lv" || scene == "ramp_hv") {
    const bool high = scene == "ramp_hv";
    solver.add_emitter({0, 0.75, 0.5}, {0, 0, -1}, {-0.25 * 3.14, 0, 0}, 1.0,
                       {0.5, 0.5, 0.5});
    solver.set_bounding_box({-2, -2, -2}, {2, 2, 2});
    solver.add_boundary_object(mesh("ramp_split.obj"));
    solver.set_surface_tension_coefficient(high ? 2.0 : 0.5);
    common(solver, 0.0001, high ? 1.0 : 0.0025);
  } else {
    std::cerr << "unknown scene " << scene << std::endl;
    std::exit(2);
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 7) {
    std::cerr << "usage: nsearch_trace_capture <scene> <radius> <pbf> <prefix> "
                 "<searches> <time> [time ...]\n";
    return 2;
  }
  const std::string scene = argv[1];
  const double radius = std::atof(argv[2]);
  const std::string prefix = argv[4];
  const int searches = std::atoi(argv[5]);

  SPHSolver solver(radius, 1000.0);
  solver.set_pbf_iterations(static_cast<uint64_t>(std::atoi(argv[3])));
  setup(solver, scene, radius);
  solver.finalize_setup();

  constexpr double kFrame = 1.0 / 30.0;
  for (int arg = 6; arg < argc; ++arg) {
    const double time = std::atof(argv[arg]);
    while (solver.get_current_time() < time) {
      solver.advance_time(kFrame);
    }
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "_t%.2f.trace", time);
    trace_recorder().arm(prefix + suffix, searches);
    while (trace_recorder().armed()) {
      solver.advance_time(kFrame);
    }
    std::cout << prefix << suffix << ": fluid="
              << solver.get_fluid_particle_positions().size()
              << " boundary=" << solver.get_boundary_particle_positions().size()
              << " time=" << solver.get_current_time() << std::endl;
  }
  return 0;
}
