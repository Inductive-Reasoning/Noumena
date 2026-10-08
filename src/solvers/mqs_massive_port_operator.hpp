// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>
#include "mfem.hpp"
#include "../linalg/bordered_block_operator.hpp"
#include "../linalg/complex_block_layout.hpp"
#include "../linalg/owning_complex_block_operator.hpp"

// The assembled time-harmonic MQS operator with massive-conductor port coupling.
// M_sigma is assembled once and scaled by the active omega at operator-application
// time:
//
//   [ K + j M_omega_sigma    -C              ] [ A      ]   [ F_source    ]
//   [ -C^T                    -j G_dc/omega    ] [ V_port ] = [ I/(j omega) ]
//
// The lower equation is the physical current balance
//
//     G_dc*V_port - j*omega*C^T*A = I,
//
// divided by j*omega to make the field/port coupling symmetric. Consequently
// the lower RHS is the scaled current phasor I/(j*omega), not the physical
// current phasor I itself. For a real in-phase current this value is -j*I/omega.
//
// K (curl-curl, 1/mu) and M_sigma (sigma mass) are the real/imag
// parts of the field SesquilinearForm; they are REFERENCED here, not owned (the
// caller keeps that form alive). The port load vectors (the columns of C) are
// owned here.
//
// This class owns only the MQS operator composition and its algebraic layout. It
// does not own scenario vectors, finite-element fields, or linear-solver policy.
// The wiring is delegated to reusable algebraic building blocks:
//   - mfem::DenseMatrix          : C, the rank-N_ports border from port loads.
//   - BorderedBlockOperator      : the real [K, -C; -C^T, 0] and imaginary
//                                  [omega M_sigma, 0; 0, -G_dc/omega] blocks.
//   - OwningComplexBlockOperator : owns both blocks and the ComplexOperator.
// With zero massive ports the borders/corners are absent and the operator
// degenerates to K + j M_omega_sigma.
class MqsMassivePortOperator
{
	class ScaledReferenceOperator : public mfem::Operator
	{
	public:
		ScaledReferenceOperator(mfem::Operator& referenced, mfem::real_t scale)
			: mfem::Operator(referenced.Height(), referenced.Width()),
			  referenced(referenced), scale(scale) {}

		void SetScale(mfem::real_t value) { scale = value; }

		void Mult(const mfem::Vector& x, mfem::Vector& y) const override
		{
			referenced.Mult(x, y);
			y *= scale;
		}

		void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
		{
			referenced.MultTranspose(x, y);
			y *= scale;
		}

	private:
		mfem::Operator& referenced;
		mfem::real_t scale;
	};

	class ConductanceCornerOperator : public mfem::Operator
	{
	public:
		ConductanceCornerOperator(const std::vector<mfem::real_t>& conductances,
								  mfem::real_t omega)
			: mfem::Operator(static_cast<int>(conductances.size())),
			  conductances(conductances), omega(omega) {}

		void SetOmega(mfem::real_t value) { omega = value; }

		void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

		void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
		{
			Mult(x, y);
		}

	private:
		std::vector<mfem::real_t> conductances;
		mfem::real_t omega;
	};

public:
	MqsMassivePortOperator(
		int n_dofs,
		mfem::SparseMatrix& K,
		mfem::SparseMatrix& M_sigma,
		std::vector<std::unique_ptr<mfem::Vector>> port_loads,
		const std::vector<mfem::real_t>& conductances,
		mfem::real_t omega);

	void SetOmega(mfem::real_t value);

	// Assemble the packed real matrix that ComplexOperator applies matrix-free,
	// in the HERMITIAN layout its Mult() implements:
	//
	//     [ R  -I ] [ x_r ]   [ y_r ]
	//     [ I   R ] [ x_i ] = [ y_i ]
	//
	// where R = [K, -C; -C^T, 0] and I = [omega*M_sigma, 0; 0, -G_dc/omega] are
	// this operator's real and imaginary halves. Building it explicitly is the
	// price of admission for a direct solve: the block operators are apply-only,
	// so a factorization needs the coefficients gathered into one CSR matrix.
	//
	// The result is frequency-dependent through both omega-scaled blocks, so it
	// must be rebuilt (and refactored) whenever SetOmega() changes the frequency.
	std::unique_ptr<mfem::SparseMatrix> AssemblePackedMatrix() const;

	mfem::ComplexOperator& Operator() { return complex_operator->GetOperator(); }
	const mfem::ComplexOperator& Operator() const { return complex_operator->GetOperator(); }

	const ComplexPortLayout& Layout() const { return layout; }

	ComplexPortVectorView View(mfem::Vector& vector) const
	{
		return ComplexPortVectorView(vector, layout.NDofs(), layout.NPorts());
	}

	ConstComplexPortVectorView View(const mfem::Vector& vector) const
	{
		return ConstComplexPortVectorView(vector, layout.NDofs(), layout.NPorts());
	}

	mfem::Array<int> MakeEssentialTDofs(const mfem::Array<int>& field_tdofs) const
	{
		return ComplexEssentialTDofs(field_tdofs, layout.HalfSize());
	}

private:
	// complex_operator transitively references port_columns, so it is declared last
	// and destroyed first. layout is independent and remains the size authority.
	ComplexPortLayout layout;
	// Owned border storage C = [c_0 | ... | c_{N-1}]; the DenseMatrix inside
	// complex_operator is a non-owning view over this data.
	mfem::DenseMatrix port_columns;
	// Referenced (not owned) field matrices, retained so the packed matrix can be
	// reassembled at each new frequency.
	mfem::SparseMatrix& stiffness;
	mfem::SparseMatrix& sigma_mass;
	std::vector<mfem::real_t> conductances;
	mfem::real_t active_omega;
	ConductanceCornerOperator* conductance_corner = nullptr;
	std::unique_ptr<ScaledReferenceOperator> owned_scaled_mass;
	std::unique_ptr<OwningComplexBlockOperator> complex_operator;
};
