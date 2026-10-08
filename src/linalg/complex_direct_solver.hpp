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
	void Factor(const mfem::SparseMatrix& packed);

	/// Solve for the packed right-hand side @p b against the stored factors.
	void Mult(const mfem::Vector& b, mfem::Vector& x) const override;

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
	CsrMatrix<Complex> Split(const mfem::SparseMatrix& packed);

	// S = D - Br F^-1 Bc, a block of ports at a time so that the F^-1 Bc
	// columns held in memory stay bounded.
	void FormSchurComplement();
};
