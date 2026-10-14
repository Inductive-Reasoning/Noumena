// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @brief Physics formulation selected by "simulation.physics"
 */
enum class PhysicsType {
    Electrostatics,
    Magnetostatics,
    Magnetoquasistatics
};

/**
 * @brief Boundary condition types
 */
enum class BoundaryType {
    Dirichlet,
    Neumann,
    Robin
};

/**
 * @brief Coordinate assumption / weak form ("simulation.geometry_type")
 *
 * Axisymmetric and Planar are 2D reductions solved on a 2D mesh; Cartesian3D
 * is the full three-dimensional model solved on a 3D mesh. Everything that
 * depends on this choice (integration measure, integrator selection, output
 * units, required mesh dimension) is centralized in GeometryModel
 * (solvers/geometry_model.hpp) rather than branched on at each call site.
 */
enum class GeometryType {
    Axisymmetric,
    Planar,
    Cartesian3D
};

/// PhysicsType -> canonical JSON string (used by the factory and diagnostics).
inline const char* ToString(PhysicsType p) {
    switch (p) {
        case PhysicsType::Electrostatics:      return "electrostatics";
        case PhysicsType::Magnetostatics:      return "magnetostatics";
        case PhysicsType::Magnetoquasistatics: return "magnetoquasistatics";
    }
    return "unknown";
}

/// GeometryType -> canonical JSON string ("simulation.geometry_type"). Also the
/// value written to the HDF5 geometry_type attributes.
inline const char* ToString(GeometryType g) {
    switch (g) {
        case GeometryType::Axisymmetric: return "axisymmetric";
        case GeometryType::Planar:       return "planar";
        case GeometryType::Cartesian3D:  return "3d";
    }
    return "unknown";
}
