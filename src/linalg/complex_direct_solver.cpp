// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "complex_direct_solver.hpp"

void ComplexDirectSolver::Factor(const mfem::SparseMatrix& packed) {
	MFEM_VERIFY(packed.Height() == layout.FullSize() && packed.Width() == layout.FullSize(),
		"ComplexDirectSolver: the matrix does not match the packed layout.");
	CsrMatrix<Complex> F = Split(packed);
	diagonal = Diagonal(F);
	field.Factor(gauge ? BorderWithConstraint(F, gauge->Interior()) : F);
	FormSchurComplement();
}

void ComplexDirectSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const  {
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

CsrMatrix<ComplexDirectSolver::Complex> ComplexDirectSolver::Split(const mfem::SparseMatrix& packed) {
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

void ComplexDirectSolver::FormSchurComplement() {
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
