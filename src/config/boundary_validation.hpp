// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once
#include "mfem.hpp"
#include <vector>
#include <map>
#include <sstream>
#include <tuple>
#include <utility>
#include <unordered_map>
#include <string>
#include <algorithm>
#include "../core/marked_boundary_condition.hpp"

/// Validates that boundary constraints (bcs + terminals) are well-posed.
///
/// Three checks are performed:
///   1. Dirichlet value conflicts - a DOF pinned to two different fixed values
///      is ill-posed. Weak Neumann conditions do not participate in this check.
///   2. Duplicate closure assignments on the same boundary attributes.
///   3. Terminal ownership/overlap - terminal values are scenario-dependent, so a
///      DOF may be owned by at most ONE terminal and must not also be a closure
///      DOF. Any shared DOF is guaranteed to conflict in some scenario.
class BoundaryConditionValidator {
private:
    mfem::Mesh& mesh;
    mfem::FiniteElementSpace& fespace;

    /// Collect the orientation-normalized DOFs touched by a boundary marker,
    /// remembering one touching boundary element per DOF for diagnostics.
    void CollectMarkerDofs(const mfem::Array<int>& marker,
                           std::map<int, int>& dof_to_be) const;

    /// Human-readable geometric description of where a DOF lives: the mesh
    /// vertex, edge or face that owns it, as the space numbers them (any
    /// order, H1 or Nedelec), else near a boundary element that touches it.
    std::string DescribeDof(int dof, int fallback_be) const;

    enum class Owner { None, Vertex, Edge, Face };
    mutable std::vector<std::pair<Owner, int>> dof_owner;  // by DOF, built on demand

    void BuildDofOwners() const;

public:
    BoundaryConditionValidator(mfem::Mesh& m, mfem::FiniteElementSpace& fes)
        : mesh(m), fespace(fes) {}

    /// @param bcs           Prescribed boundary conditions with boundary markers.
    /// @param terminals     terminal name -> boundary marker for voltage terminals.
    /// @param allow_overlap If true, warn but don't throw on detected problems.
    /// @throws std::runtime_error if problems are detected and !allow_overlap.
    void ValidateBoundaryConditions(
        const std::vector<MarkedBoundaryCondition>& bcs,
        const std::unordered_map<std::string, mfem::Array<int>>& terminals,
        bool allow_overlap = false);
};
