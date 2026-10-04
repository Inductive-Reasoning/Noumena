#pragma once

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include "mfem.hpp"

// Check the actual residual, not the Krylov recurrence or preconditioned norm.
// Essential equations have different units/scales and must not mask free rows.
inline void RequireAcceptedSolve(const mfem::Operator& a, const mfem::Vector& b,
    const mfem::Vector& x, const mfem::Array<int>& essential, double tolerance,
    const std::string& context) {
    mfem::Array<int> constrained(b.Size());
    constrained = 0;
    for (int i : essential) constrained[i] = 1;
    mfem::Vector ax(b.Size());
    a.Mult(x, ax);
    const auto* sparse = dynamic_cast<const mfem::SparseMatrix*>(&a);
    double residual[2] = {}, drive[2] = {}, rounding[2] = {};
    for (int i = 0; i < b.Size(); ++i) {
        if (!std::isfinite(x(i)) || !std::isfinite(b(i)) || !std::isfinite(ax(i))) {
            throw std::runtime_error(context + ": non-finite solution or residual");
        }
        const int group = constrained[i] ? 1 : 0;
        double scale = std::abs(ax(i));
        if (sparse) {
            scale = 0.0;
            for (int j = sparse->GetI()[i]; j < sparse->GetI()[i + 1]; ++j)
                scale += std::abs(sparse->GetData()[j] * x(sparse->GetJ()[j]));
        }
        residual[group] = std::hypot(residual[group], ax(i) - b(i));
        drive[group] = std::hypot(drive[group], b(i));
        rounding[group] = std::hypot(rounding[group], scale + std::abs(b(i)));
    }
    for (int group = 0; group < 2; ++group) {
        const double allowance = tolerance * drive[group]
            + 64 * std::numeric_limits<double>::epsilon() * rounding[group];
        if (!std::isfinite(allowance) || residual[group] > allowance) {
            std::ostringstream message;
            message << context << ": rejected " << (group ? "essential" : "free")
                << " equations; residual=" << residual[group]
                << ", allowed=" << allowance << ", rhs=" << drive[group];
            throw std::runtime_error(message.str());
        }
    }
}
