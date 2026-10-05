// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "../linalg/complex_block_layout.hpp"

// Block-diagonal preconditioner of the packed MQS system (see
// MqsMassivePortOperator): P on each field block, real and imaginary, with
// P ~ (K + omega M_sigma)^-1, and the exact inverse of each massive port's
// corner [0, G/omega; -G/omega, 0]. For the real form [K, -omega M; omega M, K]
// of K + j omega M this choice keeps the GMRES iteration count bounded
// independently of omega and the mesh. P is built by @p make_inverse from
// K + omega M_sigma with the essential rows and columns eliminated: AMG for
// the scalar 2D field, AMS for the 3D Nedelec field.
class MqsBlockPreconditioner : public mfem::Solver {
public:
	using InverseFactory = std::function<std::unique_ptr<mfem::Solver>(mfem::SparseMatrix&)>;

	MqsBlockPreconditioner(const ComplexPortLayout& layout, const mfem::SparseMatrix& K,
						   const mfem::SparseMatrix& M_sigma, double omega,
						   const mfem::Array<int>& ess_tdofs,
						   std::vector<mfem::real_t> conductances,
						   const InverseFactory& make_inverse)
		: mfem::Solver(layout.FullSize()), layout(layout), omega(omega),
		  conductances(std::move(conductances)) {
		MFEM_VERIFY(static_cast<int>(this->conductances.size()) == layout.NPorts(),
			"MqsBlockPreconditioner: one conductance per massive port is required.");
		field.reset(mfem::Add(1.0, K, omega, M_sigma));
		for (int i = 0; i < ess_tdofs.Size(); ++i) {
			field->EliminateRowCol(ess_tdofs[i], mfem::Operator::DIAG_ONE);
		}
		inverse = make_inverse(*field);
	}

	void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
		const int n = layout.NDofs(), h = layout.HalfSize();
		mfem::Vector r(n), z(n);
		for (const int offset : { 0, h }) {
			for (int i = 0; i < n; ++i) { r(i) = x(offset + i); }
			z = 0.0;
			inverse->Mult(r, z);
			for (int i = 0; i < n; ++i) { y(offset + i) = z(i); }
		}
		for (int p = 0; p < layout.NPorts(); ++p) {
			const double g = conductances[p] / omega;
			y(layout.RePortIndex(p)) = -x(layout.ImPortIndex(p)) / g;
			y(layout.ImPortIndex(p)) = x(layout.RePortIndex(p)) / g;
		}
	}

	void SetOperator(const mfem::Operator&) override {}

private:
	ComplexPortLayout layout;
	double omega;
	std::vector<mfem::real_t> conductances;
	std::unique_ptr<mfem::SparseMatrix> field;  // may be referenced by the inverse
	std::unique_ptr<mfem::Solver> inverse;
};
