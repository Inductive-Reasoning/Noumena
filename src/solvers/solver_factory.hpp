// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <unordered_map>
#include <functional>
#include <string>
#include <vector>
#include <stdexcept>
#include "mfem.hpp"
#include "physics_solver.hpp"
#include "electrostatic_solver.hpp"
#include "magnetostatic_solver.hpp"
#include "magnetostatic_solver_3d.hpp"
#include "magnetoquasistatic_solver_3d.hpp"
#include "magnetoquasistatic_solver.hpp"

/**
 * @brief Factory for creating physics solvers based on simulation type
 */
class SolverFactory {
public:
    using SolverCreator = std::function<std::unique_ptr<PhysicsSolver>(mfem::Mesh&, const ProblemConfig&)>;

private:
    std::unordered_map<PhysicsType, SolverCreator> registry;

    SolverFactory() {
        // Register all available solvers
        Register(PhysicsType::Electrostatics,
            [](mfem::Mesh& mesh, const ProblemConfig& config) -> std::unique_ptr<PhysicsSolver> {
                return std::make_unique<ElectrostaticSolver>(mesh, config);
            });
        
        // Magnetics is two formulations, not one with a geometry switch: the 2D
        // models solve a scalar potential (A_z / A_phi), '3d' the full vector
        // potential in H(curl). See MagneticSolverBase.
        Register(PhysicsType::Magnetostatics,
            [](mfem::Mesh& mesh, const ProblemConfig& config) -> std::unique_ptr<PhysicsSolver> {
                if (config.GeometryType == GeometryType::Cartesian3D) {
                    return std::make_unique<MagnetostaticSolver3D>(mesh, config);
                }
                return std::make_unique<MagnetostaticSolver>(mesh, config);
            });

        Register(PhysicsType::Magnetoquasistatics,
            [](mfem::Mesh& mesh, const ProblemConfig& config) -> std::unique_ptr<PhysicsSolver> {
                if (config.GeometryType == GeometryType::Cartesian3D) {
                    return std::make_unique<MagnetoquasistaticSolver3D>(mesh, config);
                }
                return std::make_unique<MagnetoquasistaticSolver>(mesh, config);
            });
    }

public:
    /**
     * @brief Get the singleton instance of the factory
     */
    static SolverFactory& Instance() {
        static SolverFactory instance;
        return instance;
    }

    /**
     * @brief Register a new solver type
     * @param physics The physics formulation this creator handles
     * @param creator Factory function that creates the solver
     */
    void Register(PhysicsType physics, SolverCreator creator) {
        registry[physics] = creator;
    }

    /**
     * @brief Create a solver based on the physics formulation
     * @param mesh The mesh object
     * @param config The decoded problem configuration
     * @return Unique pointer to the created solver
     * @throws std::runtime_error if the physics type is not registered
     */
    [[nodiscard]] std::unique_ptr<PhysicsSolver> Create(
        mfem::Mesh& mesh,
        const ProblemConfig& config) const {

        auto it = registry.find(config.PhysicsType);
        if (it == registry.end()) {
            throw std::runtime_error(std::string("Unregistered physics: ") + ToString(config.PhysicsType));
        }
        return it->second(mesh, config);
    }

    /**
     * @brief Check if a solver type is registered
     * @param physics The physics formulation
     * @return true if the type is registered
     */
    [[nodiscard]] bool IsRegistered(PhysicsType physics) const {
        return registry.find(physics) != registry.end();
    }

    // Delete copy constructor and assignment operator
    SolverFactory(const SolverFactory&) = delete;
    SolverFactory& operator=(const SolverFactory&) = delete;
};
