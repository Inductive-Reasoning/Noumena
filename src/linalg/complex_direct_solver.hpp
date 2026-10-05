// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <complex>
#include <memory>
#include <numeric>
#include <vector>

#include "mfem.hpp"
#include "sparse_direct_solver.hpp"

#ifdef MFEM_ELECTROMAG_STRUMPACK
#include <StrumpackSparseSolver.hpp>
#endif

// Sparse direct solver for the time-harmonic (MQS) systems. They are stored in
// the packed real form of a complex system C = A + jB (see ComplexPortLayout):
//
//     [ A  -B ] [ Re x ]   [ Re b ]
//     [ B   A ] [ Im x ] = [ Im b ]
//
// and this solver takes that packed matrix and packed vectors.
//
// With STRUMPACK (USE_STRUMPACK) it factors C itself: a multifrontal LU with
// nested-dissection (METIS) ordering, threaded with OpenMP. Measured on the
// MQS systems of the test suite (31k to 66k packed unknowns, 2D and 3D) that
// is 10 to 60 times faster, with 5 to 10 times fewer factor entries, than an
// LU of the packed form. Factoring the packed form with STRUMPACK instead is
// both slower and unreliable: its pivoting is restricted to the fronts, and
// on the larger 3D system the solution's residual was 0.7.
//
// Without STRUMPACK it falls back to Eigen's SparseLU of the packed form.
class ComplexDirectSolver : public mfem::Solver {
public:
#ifdef MFEM_ELECTROMAG_STRUMPACK
	static constexpr bool kUsesStrumpack = true;
	// Complex unknowns of a 3D system above which the factorization is large:
	// 33k took a second and 0.4 GB, and the factors of a 3D mesh grow like
	// n^(4/3), to about 4 GB here.
	static constexpr int kLarge3DUnknowns = 200000;
#else
	static constexpr bool kUsesStrumpack = false;
	static constexpr int kLarge3DUnknowns = 25000;  // a minute or more
#endif

	/// Factor the packed matrix @p packed, finalized (CSR), of even size.
	explicit ComplexDirectSolver(mfem::SparseMatrix& packed)
		: mfem::Solver(packed.Height(), packed.Width()), half(packed.Height() / 2) {
		MFEM_VERIFY(packed.Height() == packed.Width() && packed.Height() % 2 == 0,
			"ComplexDirectSolver requires a square packed matrix of even size.");
#ifdef MFEM_ELECTROMAG_STRUMPACK
		Factor(packed);
#else
		fallback = std::make_unique<SparseLUSolver>(packed);
#endif
	}

	/// Solve for the packed right-hand side @p b against the stored factors.
	void Mult(const mfem::Vector& b, mfem::Vector& x) const override {
		MFEM_ASSERT(b.Size() == 2 * half && x.Size() == 2 * half,
			"ComplexDirectSolver: vector size does not match the factored matrix.");
#ifdef MFEM_ELECTROMAG_STRUMPACK
		std::vector<std::complex<double>> rhs(half), solution(half);
		for (int i = 0; i < half; ++i) { rhs[i] = { b(i), b(half + i) }; }
		const strumpack::ReturnCode status = solver->solve(rhs.data(), solution.data());
		MFEM_VERIFY(status == strumpack::ReturnCode::SUCCESS,
			"STRUMPACK solve failed (return code " << static_cast<int>(status) << ").");
		for (int i = 0; i < half; ++i) {
			x(i) = solution[i].real();
			x(half + i) = solution[i].imag();
		}
#else
		fallback->Mult(b, x);
#endif
	}

	void SetOperator(const mfem::Operator&) override {
		MFEM_ABORT("ComplexDirectSolver is bound to the matrix it factored; "
			"construct a new solver instead.");
	}

private:
	int half;  // complex unknowns

#ifdef MFEM_ELECTROMAG_STRUMPACK
	// C = A + jB from the packed matrix's left block column: A is its top
	// half, B its bottom half. Rows are given with ascending column indices.
	void Factor(const mfem::SparseMatrix& packed) {
		const int* I = packed.GetI();
		const int* J = packed.GetJ();
		const double* V = packed.GetData();
		std::vector<int> row_ptr(half + 1, 0), col;
		std::vector<std::complex<double>> val;
		std::vector<int> slot(half, -1);  // position of column j in the current row
		for (int i = 0; i < half; ++i) {
			const int start = static_cast<int>(col.size());
			auto add = [&](int row, std::complex<double> unit) {
				for (int k = I[row]; k < I[row + 1]; ++k) {
					const int j = J[k];
					if (j >= half) continue;
					if (slot[j] < start) {
						slot[j] = static_cast<int>(col.size());
						col.push_back(j);
						val.push_back(0.0);
					}
					val[slot[j]] += unit * V[k];
				}
			};
			add(i, 1.0);
			add(half + i, { 0.0, 1.0 });
			const int end = static_cast<int>(col.size());
			std::vector<int> order(end - start);
			std::iota(order.begin(), order.end(), start);
			std::sort(order.begin(), order.end(), [&](int p, int q) { return col[p] < col[q]; });
			std::vector<int> sorted_col(order.size());
			std::vector<std::complex<double>> sorted_val(order.size());
			for (size_t k = 0; k < order.size(); ++k) {
				sorted_col[k] = col[order[k]];
				sorted_val[k] = val[order[k]];
			}
			std::copy(sorted_col.begin(), sorted_col.end(), col.begin() + start);
			std::copy(sorted_val.begin(), sorted_val.end(), val.begin() + start);
			row_ptr[i + 1] = end;
		}

		solver = std::make_unique<strumpack::SparseSolver<std::complex<double>, int>>(
			/*verbose=*/false);
		solver->set_csr_matrix(half, row_ptr.data(), col.data(), val.data());
		strumpack::ReturnCode status = solver->reorder();
		if (status == strumpack::ReturnCode::SUCCESS) { status = solver->factor(); }
		MFEM_VERIFY(status == strumpack::ReturnCode::SUCCESS,
			"STRUMPACK factorization failed (return code " << static_cast<int>(status)
			<< "): the system matrix is singular. A common cause is a region left "
			"without material properties, or a massive port whose conductance is zero.");
	}

	std::unique_ptr<strumpack::SparseSolver<std::complex<double>, int>> solver;
#else
	std::unique_ptr<SparseLUSolver> fallback;
#endif
};
