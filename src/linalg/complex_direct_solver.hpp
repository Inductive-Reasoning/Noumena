// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <complex>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "mfem.hpp"
#include "complex_block_layout.hpp"
#include "sparse_direct_solver.hpp"

#ifdef MFEM_ELECTROMAG_STRUMPACK
#include <Eigen/LU>
#include <StrumpackSparseSolver.hpp>
#endif

// Sparse direct solver for the time-harmonic (MQS) systems. They are stored in
// the packed real form of a complex system (see ComplexPortLayout), a field
// block F = K + j omega M_sigma bordered by the massive ports:
//
//     [ F    Bc ] [ A ]   [ b_A ]
//     [ Br   D  ] [ V ] = [ b_V ]
//
// with Bc the port columns (each dense over its conductor's DOFs), Br the
// port rows and D the p x p port corner. This solver takes that packed matrix
// and packed vectors.
//
// With STRUMPACK (USE_STRUMPACK) it factors the complex field block F alone,
// a multifrontal LU with nested-dissection (METIS) ordering threaded with
// OpenMP, and takes the ports through the Schur complement
//
//     S = D - Br F^-1 Bc,
//
// a dense p x p matrix formed once per factorization with one F solve per
// port. The factorization is then purely sparse however many ports there are,
// and a solve is two F solves and one with S. Measured on the MQS systems of
// the test suite (31k to 66k packed unknowns, 2D and 3D), factoring the
// complex system is 10 to 60 times faster, with 5 to 10 times fewer factor
// entries, than an LU of the packed form. (Factoring the packed form with
// STRUMPACK is both slower and unreliable: its pivoting is restricted to the
// fronts, and on the larger 3D system the solution's residual was 0.7.)
//
// The sparsity pattern of F does not change with frequency, so refactoring at
// a new frequency reuses the ordering and symbolic factorization (a quarter
// to a third of the cost) and redoes only the numerical factorization.
// STRUMPACK's matching (MC64) is off for that reason: it depends on the
// values, and these matrices, whose diagonal carries the curl-curl and
// sigma-mass terms, do not need it.
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

	explicit ComplexDirectSolver(const ComplexPortLayout& layout)
		: mfem::Solver(layout.FullSize()), layout(layout) {}

	/// Factor the packed matrix @p packed (finalized, of this layout's size),
	/// replacing any earlier factorization.
	void Factor(mfem::SparseMatrix& packed) {
		MFEM_VERIFY(packed.Height() == layout.FullSize() && packed.Width() == layout.FullSize(),
			"ComplexDirectSolver: the matrix does not match the packed layout.");
#ifdef MFEM_ELECTROMAG_STRUMPACK
		Split(packed);
		FactorField();
		FormSchurComplement();
#else
		fallback = std::make_unique<SparseLUSolver>(packed);
#endif
	}

	/// Solve for the packed right-hand side @p b against the stored factors.
	void Mult(const mfem::Vector& b, mfem::Vector& x) const override {
		MFEM_ASSERT(b.Size() == layout.FullSize() && x.Size() == layout.FullSize(),
			"ComplexDirectSolver: vector size does not match the factored matrix.");
#ifdef MFEM_ELECTROMAG_STRUMPACK
		const int n = layout.NDofs(), p = layout.NPorts(), h = layout.HalfSize();
		std::vector<Complex> b_field(n), y(n);
		for (int i = 0; i < n; ++i) { b_field[i] = { b(i), b(h + i) }; }
		SolveField(b_field.data(), y.data(), 1);
		Eigen::VectorXcd V = Eigen::VectorXcd::Zero(p);
		if (p > 0) {
			// S V = b_V - Br F^-1 b_A, then A = F^-1 (b_A - Bc V).
			Eigen::VectorXcd rhs(p);
			for (int r = 0; r < p; ++r) {
				rhs(r) = { b(n + r), b(h + n + r) };
				for (const Entry& e : border_row[r]) { rhs(r) -= e.value * y[e.index]; }
			}
			V = schur.solve(rhs);
			for (int k = 0; k < p; ++k) {
				for (const Entry& e : border_column[k]) { b_field[e.index] -= e.value * V(k); }
			}
			SolveField(b_field.data(), y.data(), 1);
		}
		for (int i = 0; i < n; ++i) {
			x(i) = y[i].real();
			x(h + i) = y[i].imag();
		}
		for (int k = 0; k < p; ++k) {
			x(n + k) = V(k).real();
			x(h + n + k) = V(k).imag();
		}
#else
		fallback->Mult(b, x);
#endif
	}

	void SetOperator(const mfem::Operator&) override {
		MFEM_ABORT("ComplexDirectSolver factors with Factor(); SetOperator is not used.");
	}

private:
	ComplexPortLayout layout;

#ifdef MFEM_ELECTROMAG_STRUMPACK
	using Complex = std::complex<double>;
	struct Entry { int index; Complex value; };

	// F in CSR with ascending column indices; its pattern is kept to detect
	// whether the next factorization can reuse the ordering.
	std::vector<int> row_ptr, col;
	std::vector<Complex> val;
	std::vector<std::vector<Entry>> border_column, border_row;  // Bc and Br, by port
	Eigen::MatrixXcd corner;                                     // D
	Eigen::PartialPivLU<Eigen::MatrixXcd> schur;
	std::unique_ptr<strumpack::SparseSolver<Complex, int>> field;

	// The complex matrix is the packed matrix's left block column, its top
	// half the real part and its bottom half the imaginary part. Split it
	// into F, Bc, Br and D.
	void Split(const mfem::SparseMatrix& packed) {
		const int n = layout.NDofs(), p = layout.NPorts(), h = layout.HalfSize();
		const int* I = packed.GetI();
		const int* J = packed.GetJ();
		const double* data = packed.GetData();
		std::vector<int> new_ptr(n + 1, 0), new_col;
		std::vector<Complex> new_val;
		border_column.assign(p, {});
		border_row.assign(p, {});
		corner = Eigen::MatrixXcd::Zero(p, p);

		std::vector<int> slot(h, -1);  // position of column j in the row being built
		std::vector<int> row_cols, order;
		std::vector<Complex> row_vals;
		const std::pair<int, Complex> parts[] = { { 0, 1.0 }, { h, { 0.0, 1.0 } } };
		for (int i = 0; i < h; ++i) {
			row_cols.clear();
			row_vals.clear();
			for (const auto& [offset, unit] : parts) {
				for (int k = I[offset + i]; k < I[offset + i + 1]; ++k) {
					const int j = J[k];
					if (j >= h) continue;
					if (slot[j] < 0) {
						slot[j] = static_cast<int>(row_cols.size());
						row_cols.push_back(j);
						row_vals.push_back(0.0);
					}
					row_vals[slot[j]] += unit * data[k];
				}
			}
			for (int j : row_cols) { slot[j] = -1; }

			order.resize(row_cols.size());
			std::iota(order.begin(), order.end(), 0);
			std::sort(order.begin(), order.end(),
					  [&](int a, int b) { return row_cols[a] < row_cols[b]; });
			for (int k : order) {
				const int j = row_cols[k];
				const Complex v = row_vals[k];
				if (i < n && j < n) { new_col.push_back(j); new_val.push_back(v); }
				else if (i < n) { border_column[j - n].push_back({ i, v }); }
				else if (j < n) { border_row[i - n].push_back({ j, v }); }
				else { corner(i - n, j - n) = v; }
			}
			if (i < n) { new_ptr[i + 1] = static_cast<int>(new_col.size()); }
		}

		const bool same_pattern = field && new_ptr == row_ptr && new_col == col;
		row_ptr = std::move(new_ptr);
		col = std::move(new_col);
		val = std::move(new_val);
		if (!same_pattern) { field.reset(); }
	}

	void FactorField() {
		const int n = layout.NDofs();
		strumpack::ReturnCode status = strumpack::ReturnCode::SUCCESS;
		if (field) {
			field->update_matrix_values(n, row_ptr.data(), col.data(), val.data());
		}
		else {
			field = std::make_unique<strumpack::SparseSolver<Complex, int>>(/*verbose=*/false);
			field->options().set_matching(strumpack::MatchingJob::NONE);
			field->set_csr_matrix(n, row_ptr.data(), col.data(), val.data());
			status = field->reorder();
		}
		if (status == strumpack::ReturnCode::SUCCESS) { status = field->factor(); }
		MFEM_VERIFY(status == strumpack::ReturnCode::SUCCESS,
			"STRUMPACK factorization failed (return code " << static_cast<int>(status)
			<< "): the system matrix is singular. A common cause is a region left "
			"without material properties.");
	}

	// S = D - Br F^-1 Bc, a block of ports at a time so that the F^-1 Bc
	// columns held in memory stay bounded.
	void FormSchurComplement() {
		const int n = layout.NDofs(), p = layout.NPorts();
		if (p == 0) { return; }
		constexpr int kBlock = 32;
		Eigen::MatrixXcd S = corner;
		std::vector<Complex> rhs, w;
		for (int k0 = 0; k0 < p; k0 += kBlock) {
			const int m = std::min(kBlock, p - k0);
			rhs.assign(static_cast<size_t>(n) * m, 0.0);
			w.assign(static_cast<size_t>(n) * m, 0.0);
			for (int c = 0; c < m; ++c) {
				for (const Entry& e : border_column[k0 + c]) {
					rhs[static_cast<size_t>(c) * n + e.index] = e.value;
				}
			}
			SolveField(rhs.data(), w.data(), m);
			for (int r = 0; r < p; ++r) {
				for (const Entry& e : border_row[r]) {
					for (int c = 0; c < m; ++c) {
						S(r, k0 + c) -= e.value * w[static_cast<size_t>(c) * n + e.index];
					}
				}
			}
		}
		schur.compute(S);
		MFEM_VERIFY(std::abs(schur.determinant()) > 0.0,
			"The massive-port Schur complement is singular; a massive port whose "
			"conductance is zero?");
	}

	// x = F^-1 b for @p count right-hand sides stored one after another.
	void SolveField(const Complex* b, Complex* x, int count) const {
		const int n = layout.NDofs();
		const strumpack::ReturnCode status = field->solve(count, b, n, x, n);
		MFEM_VERIFY(status == strumpack::ReturnCode::SUCCESS,
			"STRUMPACK solve failed (return code " << static_cast<int>(status) << ").");
	}
#else
	std::unique_ptr<SparseLUSolver> fallback;
#endif
};
