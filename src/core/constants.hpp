// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

namespace Constants {
    // UNITS: this project is strictly SI. Mesh coordinates MUST be in METERS.
    //
    // The constants below are per-meter (H/m, F/m) and are multiplied directly
    // against mesh-derived lengths during assembly, so a mesh authored in
    // millimeters does not merely rescale the answer -- it silently corrupts
    // every absolute quantity (energy, capacitance, inductance, loss) while
    // still solving cleanly. There is no length-scale factor in the schema and
    // nothing validates the mesh extent, so this convention is documented and
    // relied upon rather than enforced. Convert at mesh generation time.

    // Physical constants
    constexpr double MU_0 = 4.0 * 3.141592653589793 * 1e-7;  // Permeability of free space [H/m]
    constexpr double EPSILON_0 = 8.854187817e-12;             // Permittivity of free space [F/m]
    constexpr double TWO_PI = 2.0 * 3.141592653589793;        // 2*pi

    // Numerical tolerances
    // NOTE: axis proximity is no longer governed by a fixed absolute tolerance.
    // See axisym::kRelativeGeometryTolerance in axis_geometry.hpp, which scales
    // with the mesh bounding box. That makes axis CLASSIFICATION scale-free; it
    // does not relax the meters requirement above, which the physics imposes.

    // Default solver parameters
    constexpr double DEFAULT_SOLVER_TOLERANCE = 1e-12;
    constexpr int DEFAULT_SOLVER_MAX_ITER = 1000;
    constexpr int DEFAULT_SOLVER_PRINT_LEVEL = 1;
}
