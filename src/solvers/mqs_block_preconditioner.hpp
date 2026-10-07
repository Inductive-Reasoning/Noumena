// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "../linalg/complex_block_layout.hpp"

// Preconditioner of the packed MQS system (see MqsMassivePortOperator). Its
// field part is PRESB (preconditioned square block; Axelsson, Neytcheva et
// al.) for the real form [A, -B; B, A] of A + jB, here A = K and
// B = omega M_sigma:
//
//     [ A   -B     ]   [ I   0 ] [ H  -B ] [ I  0 ]
//     [ B   A + 2B ] = [ -I  I ] [ 0   H ] [ I  I ],    H = A + B,
//
// With exact solves with H the preconditioned eigenvalues lie in [1/2, 1]
// for every omega. One application costs two solves with H, the same as the
// block-diagonal diag(H, H), but the Krylov solver needs half to a third of
// the iterations once omega M_sigma is comparable to K (TEAM 7 at order 2:
// 66 against 153). Essential rows pass through unchanged. H^-1 is
// approximated by @p make_inverse, built from H with the essential rows and
// columns eliminated: AMG for the scalar 2D field, AMS for the 3D Nedelec
// field. The ports get the exact inverse of their corner
// [0, G/omega; -G/omega, 0].
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
		coupling.reset(new mfem::SparseMatrix(M_sigma));
		*coupling *= omega;
		for (int i = 0; i < ess_tdofs.Size(); ++i) {
			coupling->EliminateRowCol(ess_tdofs[i], mfem::Operator::DIAG_ZERO);
		}
	}

	void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
		const int n = layout.NDofs(), h = layout.HalfSize();
		// Solve P [y_re; y_im] = [f; g] through the factorization above:
		// w2 = H^-1 (f + g), w1 = H^-1 (f + B w2), y_re = w1, y_im = w2 - w1.
		mfem::Vector r(n), w1(n), w2(n), Bw(n);
		for (int i = 0; i < n; ++i) { r(i) = x(i) + x(h + i); }
		w2 = 0.0;
		inverse->Mult(r, w2);
		coupling->Mult(w2, Bw);
		for (int i = 0; i < n; ++i) { r(i) = x(i) + Bw(i); }
		w1 = 0.0;
		inverse->Mult(r, w1);
		for (int i = 0; i < n; ++i) {
			y(i) = w1(i);
			y(h + i) = w2(i) - w1(i);
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
	std::unique_ptr<mfem::SparseMatrix> field;     // H, may be referenced by the inverse
	std::unique_ptr<mfem::Solver> inverse;         // ~ H^-1
	std::unique_ptr<mfem::SparseMatrix> coupling;  // B = omega M_sigma
};
