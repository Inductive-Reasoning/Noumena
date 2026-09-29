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
                           std::map<int, int>& dof_to_be) const {
        for (int i = 0; i < fespace.GetNBE(); ++i) {
            int attr = mesh.GetBdrAttribute(i);
            if (attr < 1 || attr > marker.Size() || !marker[attr - 1]) continue;
            mfem::Array<int> vdofs;
            fespace.GetBdrElementVDofs(i, vdofs);
            for (int j = 0; j < vdofs.Size(); ++j) {
                int dof = vdofs[j];
                if (dof < 0) dof = -1 - dof;  // handle orientation
                dof_to_be.emplace(dof, i);    // first touching bdr element wins
            }
        }
    }

    /// Human-readable geometric description of where a DOF lives. In an H1 space
    /// the first NV DOFs are vertex DOFs, the next NEdges*(order-1) are edge
    /// interiors, then face/interior DOFs (located via a touching bdr element).
    std::string DescribeDof(int dof, int fallback_be) const {
        std::ostringstream oss;
        const int NV = mesh.GetNV();
        const int NE = mesh.GetNEdges();
        const int sdim = mesh.SpaceDimension();
        // "(x, y)" in 2D, "(x, y, z)" in 3D.
        auto point = [&](const double* p) {
            oss << "(" << p[0];
            for (int c = 1; c < sdim; ++c) oss << ", " << p[c];
            oss << ")";
        };
        if (dof < NV) {
            oss << "vertex " << dof << " at ";
            point(mesh.GetVertex(dof));
        } else if (dof < NV + NE) {
            int edge_idx = dof - NV;
            mfem::Array<int> ev;
            mesh.GetEdgeVertices(edge_idx, ev);
            const double* v0 = mesh.GetVertex(ev[0]);
            const double* v1 = mesh.GetVertex(ev[1]);
            double mid[3] = { 0.0, 0.0, 0.0 };
            for (int c = 0; c < sdim; ++c) mid[c] = 0.5 * (v0[c] + v1[c]);
            oss << "edge-interior on mesh edge " << edge_idx << " midpoint=";
            point(mid);
        } else if (fallback_be >= 0) {
            mfem::Array<int> verts;
            mesh.GetBdrElementVertices(fallback_be, verts);
            double centroid[3] = { 0.0, 0.0, 0.0 };
            for (int vi = 0; vi < verts.Size(); ++vi) {
                const double* v = mesh.GetVertex(verts[vi]);
                for (int c = 0; c < sdim; ++c) centroid[c] += v[c] / verts.Size();
            }
            oss << "face/interior near bdr-element centroid ";
            point(centroid);
        } else {
            oss << "index " << dof;
        }
        return oss.str();
    }

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
        bool allow_overlap = false)
    {
        std::vector<std::string> problems;
        const double tolerance = 1e-10;

        // (1) Dirichlet value conflicts: weak conditions do not pin DOFs.
        std::map<int, std::vector<std::tuple<double, int, int>>> dof_values;
        for (const auto& bc : bcs) {
            if (bc.Condition.Type != BoundaryConditionType::Dirichlet) continue;
            const auto& marker = bc.Marker;
            const double value = bc.Condition.Value;
            for (int i = 0; i < fespace.GetNBE(); ++i) {
                int attr = mesh.GetBdrAttribute(i);
                if (attr < 1 || attr > marker.Size() || !marker[attr - 1]) continue;
                mfem::Array<int> vdofs;
                fespace.GetBdrElementVDofs(i, vdofs);
                for (int j = 0; j < vdofs.Size(); ++j) {
                    int dof = vdofs[j];
                    if (dof < 0) dof = -1 - dof;  // handle orientation
                    dof_values[dof].push_back(std::make_tuple(value, attr, i));
                }
            }
        }

        for (const auto& [dof, values] : dof_values) {
            if (values.size() < 2) continue;
            double first_val = std::get<0>(values[0]);
            bool conflict = false;
            for (size_t i = 1; i < values.size(); ++i) {
                if (std::abs(std::get<0>(values[i]) - first_val) > tolerance) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) continue;

            std::ostringstream oss;
            oss << "DOF " << dof << " (" << DescribeDof(dof, std::get<2>(values[0]))
                << ") is pinned by bcs to DIFFERENT values:\n";
            std::map<std::pair<int, double>, int> grouped;
            for (const auto& t : values) grouped[{std::get<1>(t), std::get<0>(t)}]++;
            for (const auto& [key, count] : grouped) {
                oss << "    boundary attribute " << key.first
                    << ": value = " << key.second
                    << "  (" << count << " bdr element" << (count == 1 ? "" : "s") << ")\n";
            }
            problems.push_back(oss.str());
        }

        // Two configured conditions may meet at vertices, but they must not claim
        // the same boundary attribute (and therefore the same boundary elements).
        for (size_t i = 0; i < bcs.size(); ++i) {
            for (size_t j = i + 1; j < bcs.size(); ++j) {
                const auto& lhs = bcs[i];
                const auto& rhs = bcs[j];
                const int marker_size = std::min(lhs.Marker.Size(), rhs.Marker.Size());
                for (int attr = 0; attr < marker_size; ++attr) {
                    if (!lhs.Marker[attr] || !rhs.Marker[attr]) continue;
                    std::ostringstream oss;
                    oss << "Boundary attribute " << (attr + 1)
                        << " is assigned to both '" << lhs.Condition.EntityGroupName
                        << "' and '" << rhs.Condition.EntityGroupName
                        << "'. Each boundary entity must have one boundary condition.";
                    problems.push_back(oss.str());
                    break;
                }
            }
        }

        std::vector<std::string> term_names;
        term_names.reserve(terminals.size());
        for (const auto& kv : terminals) term_names.push_back(kv.first);
        std::sort(term_names.begin(), term_names.end());  // deterministic output

        // A boundary attribute has one physical owner. In particular, a weak
        // Neumann load on a voltage-terminal attribute would be silently removed
        // when the terminal's essential DOFs are eliminated.
        for (const auto& bc : bcs) {
            for (const auto& name : term_names) {
                const auto& terminal = terminals.at(name);
                const int marker_size = std::min(bc.Marker.Size(), terminal.Size());
                for (int attr = 0; attr < marker_size; ++attr) {
                    if (!bc.Marker[attr] || !terminal[attr]) continue;
                    std::ostringstream oss;
                    oss << "Boundary attribute " << (attr + 1)
                        << " is assigned to boundary condition '"
                        << bc.Condition.EntityGroupName
                        << "' and voltage terminal '" << name
                        << "'. Each boundary entity must have one physical role.";
                    problems.push_back(oss.str());
                    break;
                }
            }
        }

        // (2) Terminal ownership/overlap: only Dirichlet bcs are essential.
        std::map<int, int> bc_dofs;  // dof -> touching bdr element
        for (const auto& bc : bcs) {
            if (bc.Condition.Type == BoundaryConditionType::Dirichlet) {
                CollectMarkerDofs(bc.Marker, bc_dofs);
            }
        }

        std::map<int, std::vector<std::string>> dof_terminals;  // dof -> terminal names
        std::map<int, int> terminal_dof_be;                     // dof -> touching bdr element
        for (const auto& name : term_names) {
            std::map<int, int> td;
            CollectMarkerDofs(terminals.at(name), td);
            for (const auto& [dof, be] : td) {
                dof_terminals[dof].push_back(name);
                terminal_dof_be.emplace(dof, be);
            }
        }

        for (const auto& [dof, names] : dof_terminals) {
            const bool multi_terminal = names.size() > 1;
            const bool hits_bc   = bc_dofs.count(dof) > 0;
            if (!multi_terminal && !hits_bc) continue;

            int be = terminal_dof_be.count(dof) ? terminal_dof_be.at(dof) : -1;
            std::ostringstream oss;
            oss << "DOF " << dof << " (" << DescribeDof(dof, be)
                << ") is constrained by overlapping essential sites:\n";
            if (multi_terminal) {
                oss << "    terminals: ";
                for (size_t i = 0; i < names.size(); ++i) {
                    if (i) oss << ", ";
                    oss << names[i];
                }
                oss << "\n";
            } else {
                oss << "    terminal: " << names[0] << "\n";
            }
            if (hits_bc) {
                oss << "    boundary condition (also pins this DOF)\n";
            }
            problems.push_back(oss.str());
        }

        if (problems.empty()) return;

        std::ostringstream msg;
        msg << "\n========================================\n";
        msg << (allow_overlap ? "WARNING" : "ERROR")
            << ": Conflicting Boundary Constraints Detected!\n";
        msg << "========================================\n\n";
        msg << "Boundary entities or DOFs have incompatible assignments.\n";
        msg << "This typically indicates:\n";
        msg << "  1. Mesh: boundaries/terminals with different roles share nodes\n";
        msg << "  2. Config: a node is driven to two values at once (now or per-scenario)\n\n";
        msg << "Problems found (showing first 5):\n\n";

        int count = 0;
        for (const auto& p : problems) {
            msg << p << "\n";
            if (++count >= 5) break;
        }
        if (problems.size() > 5) {
            msg << "... and " << (problems.size() - 5) << " more\n";
        }

        if (allow_overlap) {
            msg << "\nPROCEEDING WITH OVERLAPS (values will be averaged at shared DOFs).\n";
            msg << "This may produce INCORRECT PHYSICS!\n";
            msg << "\nRecommended fixes:\n";
            msg << "  - Regenerate the mesh so distinct boundaries/terminals don't share nodes\n";
            msg << "  - Check terminal attribute_ids and boundary specification\n";
            msg << "========================================\n";
            mfem::out << msg.str() << std::endl;
        } else {
            msg << "\nPlease fix your:\n";
            msg << "  - Mesh design (separate boundaries/terminals shouldn't share nodes)\n";
            msg << "  - Terminal/boundary specification (check your config.json)\n";
            msg << "========================================\n";
            throw std::runtime_error(msg.str());
        }
    }
};
