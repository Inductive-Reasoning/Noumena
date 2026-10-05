// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <complex>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "mfem.hpp"
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#ifdef MFEM_ELECTROMAG_STRUMPACK
#include <StrumpackSparseSolver.hpp>
#endif

// A square sparse matrix in CSR form (ascending column indices in each row),
// real or complex.
template <typename T>
struct CsrMatrix {
	int n = 0;
	std::vector<int> row_ptr{ 0 }, col;
	std::vector<T> val;
};

// A finalized real mfem::SparseMatrix as a CsrMatrix (columns sorted).
inline CsrMatrix<double> ToCsr(const mfem::SparseMatrix& A) {
	MFEM_VERIFY(A.Finalized() && A.Height() == A.Width(), "ToCsr: needs a finalized square matrix.");
	CsrMatrix<double> C;
	C.n = A.Height();
	C.row_ptr.assign(C.n + 1, 0);
	std::vector<std::pair<int, double>> row;
	for (int i = 0; i < C.n; ++i) {
		row.clear();
		for (int k = A.GetI()[i]; k < A.GetI()[i + 1]; ++k) {
			row.push_back({ A.GetJ()[k], A.GetData()[k] });
		}
		std::sort(row.begin(), row.end());
		for (const auto& [j, v] : row) {
			C.col.push_back(j);
			C.val.push_back(v);
		}
		C.row_ptr[i + 1] = static_cast<int>(C.col.size());
	}
	return C;
}

// [F B; B^T 0] for a CSR matrix F (n x n) and a real sparse matrix B
// (n x m, finalized): F bordered by the constraint columns B and rows B^T.
template <typename T>
CsrMatrix<T> BorderWithConstraint(const CsrMatrix<T>& F, const mfem::SparseMatrix& B) {
	const int n = F.n, m = B.Width();
	MFEM_VERIFY(B.Height() == n, "BorderWithConstraint: B has the wrong number of rows.");
	std::unique_ptr<mfem::SparseMatrix> Bt(mfem::Transpose(B));
	CsrMatrix<T> S;
	S.n = n + m;
	S.row_ptr.assign(S.n + 1, 0);
	for (int i = 0; i < n; ++i) {
		for (int k = F.row_ptr[i]; k < F.row_ptr[i + 1]; ++k) {
			S.col.push_back(F.col[k]);
			S.val.push_back(F.val[k]);
		}
		for (int k = B.GetI()[i]; k < B.GetI()[i + 1]; ++k) {
			S.col.push_back(n + B.GetJ()[k]);
			S.val.push_back(T(B.GetData()[k]));
		}
		S.row_ptr[i + 1] = static_cast<int>(S.col.size());
	}
	Bt->SortColumnIndices();
	for (int r = 0; r < m; ++r) {
		for (int k = Bt->GetI()[r]; k < Bt->GetI()[r + 1]; ++k) {
			S.col.push_back(Bt->GetJ()[k]);
			S.val.push_back(T(Bt->GetData()[k]));
		}
		S.row_ptr[n + r + 1] = static_cast<int>(S.col.size());
	}
	return S;
}

// A gauge constraint C (field DOFs x multipliers) split at the essential
// field DOFs of a system whose essential rows and columns are eliminated
// (diagonal d_i kept or set to one, right-hand side d_i A_i). Its other rows
// border the system; its essential rows move the essential values to the
// constraint's right-hand side, C_free^T A_free = -C_ess^T A_ess.
class GaugeConstraintRows {
public:
	GaugeConstraintRows(const mfem::SparseMatrix& C, const mfem::Array<int>& ess_tdofs)
		: interior(C.Height(), C.Width()) {
		std::vector<bool> essential(C.Height(), false);
		for (int i = 0; i < ess_tdofs.Size(); ++i) { essential[ess_tdofs[i]] = true; }
		for (int i = 0; i < C.Height(); ++i) {
			for (int k = C.GetI()[i]; k < C.GetI()[i + 1]; ++k) {
				const double v = C.GetData()[k];
				if (v == 0.0) continue;
				if (essential[i]) { boundary.push_back({ i, C.GetJ()[k], v }); }
				else { interior.Set(i, C.GetJ()[k], v); }
			}
		}
		interior.Finalize();
	}

	/// The constraint's non-essential rows.
	const mfem::SparseMatrix& Interior() const { return interior; }
	int Multipliers() const { return interior.Width(); }

	/// The constraint's right-hand side -C_ess^T A_ess into @p rhs (one entry
	/// per multiplier), with A_i = b_i / d_i from the eliminated system's
	/// right-hand side @p b and diagonal @p diagonal.
	template <typename T>
	void RightHandSide(const T* b, const T* diagonal, T* rhs) const {
		for (int k = 0; k < Multipliers(); ++k) { rhs[k] = T(0.0); }
		for (const Entry& e : boundary) {
			rhs[e.multiplier] -= e.value * (b[e.row] / diagonal[e.row]);
		}
	}

private:
	struct Entry { int row, multiplier; double value; };
	mfem::SparseMatrix interior;
	std::vector<Entry> boundary;
};

// The diagonal of a CsrMatrix.
template <typename T>
std::vector<T> Diagonal(const CsrMatrix<T>& A) {
	std::vector<T> d(A.n, T(0.0));
	for (int i = 0; i < A.n; ++i) {
		for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
			if (A.col[k] == i) { d[i] = A.val[k]; }
		}
	}
	return d;
}

// LU factorization of a CsrMatrix<T>, T double or std::complex<double>.
//
// With STRUMPACK it is STRUMPACK's multifrontal LU (nested-dissection METIS
// ordering, OpenMP), with its default MC64 matching and scaling, which also
// lets it pivot past a zero diagonal block (a constrained, saddle-point
// matrix). Refactoring a matrix with the same sparsity pattern keeps the
// ordering, matching and symbolic factorization and redoes only the
// numerical factorization.
//
// Without STRUMPACK it is Eigen's SparseLU with COLAMD ordering, in the
// matrix's own (real or complex) arithmetic.
template <typename T>
class SparseLU {
	static_assert(std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
				  "SparseLU supports double and std::complex<double>.");
	static constexpr bool kComplex = std::is_same_v<T, std::complex<double>>;

public:
	/// Factor @p A, reusing the previous factorization's analysis when its
	/// sparsity pattern is unchanged.
	void Factor(const CsrMatrix<T>& A) {
#ifdef MFEM_ELECTROMAG_STRUMPACK
		const bool same_pattern = solver && A.row_ptr == row_ptr && A.col == col;
		n = A.n;
		row_ptr = A.row_ptr;
		col = A.col;
		strumpack::ReturnCode status = strumpack::ReturnCode::SUCCESS;
		if (same_pattern) {
			solver->update_matrix_values(n, row_ptr.data(), col.data(), A.val.data());
		}
		else {
			solver = std::make_unique<strumpack::SparseSolver<T, int>>(/*verbose=*/false);
			solver->set_csr_matrix(n, row_ptr.data(), col.data(), A.val.data());
			status = solver->reorder();
		}
		if (status == strumpack::ReturnCode::SUCCESS) { status = solver->factor(); }
		MFEM_VERIFY(status == strumpack::ReturnCode::SUCCESS,
			"STRUMPACK factorization failed (return code " << static_cast<int>(status)
			<< "): the system matrix is singular. A common cause is a region left "
			"without material properties.");
#else
		n = A.n;
		std::vector<Eigen::Triplet<T>> entries;
		entries.reserve(A.val.size());
		for (int i = 0; i < n; ++i) {
			for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
				entries.emplace_back(i, A.col[k], A.val[k]);
			}
		}
		matrix.resize(n, n);
		matrix.setFromTriplets(entries.begin(), entries.end());
		matrix.makeCompressed();
		fallback.analyzePattern(matrix);
		fallback.factorize(matrix);
		MFEM_VERIFY(fallback.info() == Eigen::Success,
			"Sparse LU factorization failed: the system matrix is singular. A common "
			"cause is a region left without material properties.");
#endif
	}

	/// x = A^-1 b for @p count right-hand sides of length n stored one after
	/// another.
	void Solve(const T* b, T* x, int count) const {
#ifdef MFEM_ELECTROMAG_STRUMPACK
		const strumpack::ReturnCode status = solver->solve(count, b, n, x, n);
		MFEM_VERIFY(status == strumpack::ReturnCode::SUCCESS,
			"STRUMPACK solve failed (return code " << static_cast<int>(status) << ").");
#else
		for (int c = 0; c < count; ++c) {
			const Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> rhs(b + static_cast<size_t>(c) * n, n);
			Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> sol(x + static_cast<size_t>(c) * n, n);
			sol = fallback.solve(rhs);
		}
#endif
	}

	int Size() const { return n; }

private:
	int n = 0;
#ifdef MFEM_ELECTROMAG_STRUMPACK
	std::vector<int> row_ptr, col;  // pattern of the last factorization
	std::unique_ptr<strumpack::SparseSolver<T, int>> solver;
#else
	Eigen::SparseMatrix<T> matrix;
	Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> fallback;
#endif
};
