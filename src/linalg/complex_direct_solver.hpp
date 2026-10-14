// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <complex>
#include <numeric>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "mfem.hpp"
#include "complex_block_layout.hpp"
#include "sparse_lu.hpp"

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
// It factors the complex field block alone (SparseLU: STRUMPACK's
// multifrontal LU, or Eigen's without it) and takes the ports through the
// Schur complement
//
//     S = D - Br F^-1 Bc,
//
// a dense p x p matrix formed once per factorization with one F solve per
// port. The sparse factorization never sees the ports' dense rows and
// columns, however many there are, and a solve is two F solves and one with
// S. Measured on the MQS systems of the test suite (31k to 66k packed
// unknowns, 2D and 3D), STRUMPACK on the complex system is 10 to 60 times
// faster, with 5 to 10 times fewer factor entries, than an LU of the packed
// form. (STRUMPACK on the packed form is both slower and unreliable: its
// pivoting is restricted to the fronts, and on the larger 3D system the
// residual was 0.7.)
//
// A 3D field block is singular in the nonconducting regions (curl-curl
// annihilates gradients). With a gauge constraint G (see
// DivergenceFreeProjector::GaugeConstraint) the field block factored is
//
//     [ F    G ]
//     [ G^T  0 ],
//
// which imposes the discrete Coulomb gauge exactly through a Lagrange
// multiplier that comes out zero. Every load is balanced (projected), so the
// field equations themselves are unchanged.
//
// The pattern of the field block does not change with frequency, so
// refactoring at a new frequency redoes only the numerical factorization.
class ComplexDirectSolver : public mfem::Solver {
public:
#ifdef NOUMENA_STRUMPACK
	static constexpr bool kUsesStrumpack = true;
	// Complex unknowns of a 3D system above which the factorization is large:
	// 33k took a second and 0.4 GB, and the factors of a 3D mesh grow like
	// n^(4/3), to about 4 GB here.
	static constexpr int kLarge3DUnknowns = 200000;
#else
	static constexpr bool kUsesStrumpack = false;
	static constexpr int kLarge3DUnknowns = 25000;  // a minute or more
#endif

	/// @param gauge  Gauge constraint split at the essential field DOFs, or
	///               null for a field block that needs none (2D). Held by
	///               reference: it must outlive this solver.
	explicit ComplexDirectSolver(const ComplexPortLayout& layout,
								 const GaugeConstraintRows* gauge = nullptr)
		: mfem::Solver(layout.FullSize()), layout(layout), gauge(gauge) {
		MFEM_VERIFY(!gauge || gauge->Interior().Height() == layout.NDofs(),
			"ComplexDirectSolver: the gauge constraint does not match the field DOFs.");
	}

	/// Factor the packed matrix @p packed (finalized, of this layout's size),
	/// replacing any earlier factorization.
	void Factor(const mfem::SparseMatrix& packed) {
		MFEM_VERIFY(packed.Height() == layout.FullSize() && packed.Width() == layout.FullSize(),
			"ComplexDirectSolver: the matrix does not match the packed layout.");
		CsrMatrix<Complex> F = Split(packed);
		diagonal = Diagonal(F);
		field.Factor(gauge ? BorderWithConstraint(F, gauge->Interior()) : F);
		FormSchurComplement();
	}

	/// Solve for the packed right-hand side @p b against the stored factors.
	void Mult(const mfem::Vector& b, mfem::Vector& x) const override {
		MFEM_ASSERT(b.Size() == layout.FullSize() && x.Size() == layout.FullSize(),
			"ComplexDirectSolver: vector size does not match the factored matrix.");
		const int n = layout.NDofs(), p = layout.NPorts(), h = layout.HalfSize();
		std::vector<Complex> b_field(field.Size(), 0.0), y(field.Size());
		for (int i = 0; i < n; ++i) { b_field[i] = { b(i), b(h + i) }; }
		if (gauge) { gauge->RightHandSide(b_field.data(), diagonal.data(), b_field.data() + n); }
		field.Solve(b_field.data(), y.data(), 1);
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
			field.Solve(b_field.data(), y.data(), 1);
		}
		for (int i = 0; i < n; ++i) {
			x(i) = y[i].real();
			x(h + i) = y[i].imag();
		}
		for (int k = 0; k < p; ++k) {
			x(n + k) = V(k).real();
			x(h + n + k) = V(k).imag();
		}
	}

	void SetOperator(const mfem::Operator&) override {
		MFEM_ABORT("ComplexDirectSolver factors with Factor(); SetOperator is not used.");
	}

private:
	using Complex = std::complex<double>;
	struct Entry { int index; Complex value; };

	ComplexPortLayout layout;
	const GaugeConstraintRows* gauge;
	SparseLU<Complex> field;
	std::vector<Complex> diagonal;  // of F, for the essential values
	std::vector<std::vector<Entry>> border_column, border_row;  // Bc and Br, by port
	Eigen::MatrixXcd corner;                                     // D
	Eigen::PartialPivLU<Eigen::MatrixXcd> schur;

	// The complex matrix is the packed matrix's left block column, its top
	// half the real part and its bottom half the imaginary part. Split it
	// into F (returned), Bc, Br and D.
	CsrMatrix<Complex> Split(const mfem::SparseMatrix& packed) {
		const int n = layout.NDofs(), p = layout.NPorts(), h = layout.HalfSize();
		const int* I = packed.GetI();
		const int* J = packed.GetJ();
		const double* data = packed.GetData();
		CsrMatrix<Complex> F;
		F.n = n;
		F.row_ptr.assign(n + 1, 0);
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
				if (i < n && j < n) { F.col.push_back(j); F.val.push_back(v); }
				else if (i < n) { border_column[j - n].push_back({ i, v }); }
				else if (j < n) { border_row[i - n].push_back({ j, v }); }
				else { corner(i - n, j - n) = v; }
			}
			if (i < n) { F.row_ptr[i + 1] = static_cast<int>(F.col.size()); }
		}
		return F;
	}

	// S = D - Br F^-1 Bc, a block of ports at a time so that the F^-1 Bc
	// columns held in memory stay bounded.
	void FormSchurComplement() {
		const int p = layout.NPorts(), size = field.Size();
		if (p == 0) { return; }
		constexpr int kBlock = 32;
		Eigen::MatrixXcd S = corner;
		std::vector<Complex> rhs, w;
		for (int k0 = 0; k0 < p; k0 += kBlock) {
			const int m = std::min(kBlock, p - k0);
			rhs.assign(static_cast<size_t>(size) * m, 0.0);
			w.assign(static_cast<size_t>(size) * m, 0.0);
			for (int c = 0; c < m; ++c) {
				for (const Entry& e : border_column[k0 + c]) {
					rhs[static_cast<size_t>(c) * size + e.index] = e.value;
				}
			}
			field.Solve(rhs.data(), w.data(), m);
			for (int r = 0; r < p; ++r) {
				for (const Entry& e : border_row[r]) {
					for (int c = 0; c < m; ++c) {
						S(r, k0 + c) -= e.value * w[static_cast<size_t>(c) * size + e.index];
					}
				}
			}
		}
		schur.compute(S);
		MFEM_VERIFY(schur.rcond() > 0.0,
			"The massive-port Schur complement is singular; a massive port whose "
			"conductance is zero?");
	}
};
