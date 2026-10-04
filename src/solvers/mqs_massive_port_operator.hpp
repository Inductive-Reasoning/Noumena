// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>
#include "mfem.hpp"
#include "../linalg/bordered_block_operator.hpp"
#include "../linalg/complex_block_layout.hpp"
#include "../linalg/owning_complex_block_operator.hpp"
#include "../linalg/solve_acceptance.hpp"

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

		void Mult(const mfem::Vector& x, mfem::Vector& y) const override
		{
			for (int p = 0; p < Height(); ++p) {
				y(p) = -conductances[p] * x(p) / omega;
			}
		}

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
		mfem::real_t omega)
		: layout(n_dofs, static_cast<int>(port_loads.size())),
		  port_columns(n_dofs, static_cast<int>(port_loads.size())),
		  stiffness(K), sigma_mass(M_sigma), conductances(conductances),
		  active_omega(omega)
	{
		MFEM_VERIFY(omega > 0.0, "MQS angular frequency must be positive.");
		MFEM_VERIFY(
			static_cast<int>(conductances.size()) == layout.NPorts(),
			"Port conductance data must have one entry per port.");

		std::unique_ptr<mfem::Operator> coupling;
		if (layout.NPorts() > 0)
		{
			for (int p = 0; p < layout.NPorts(); ++p)
			{
				MFEM_VERIFY(port_loads[p] && port_loads[p]->Size() == n_dofs,
							"Port load vector size must match the field DOF count.");
				port_columns.SetCol(p, *port_loads[p]);
			}
			// A referencing view: the border action is DenseMatrix's own
			// Mult/MultTranspose over the owned column storage.
			coupling = std::make_unique<mfem::DenseMatrix>(
				port_columns.Data(), n_dofs, layout.NPorts());
		}

		auto real_block = std::make_unique<BorderedBlockOperator>(
			layout.NDofs(), layout.NPorts(), K, std::move(coupling), -1.0,
			/*corner=*/nullptr);

		std::unique_ptr<mfem::Operator> port_corner;
		if (layout.NPorts() > 0)
		{
			auto frequency_corner =
				std::make_unique<ConductanceCornerOperator>(conductances, omega);
			conductance_corner = frequency_corner.get();
			port_corner = std::move(frequency_corner);
		}

		auto frequency_mass = std::make_unique<ScaledReferenceOperator>(M_sigma, omega);
		auto imaginary_block = std::make_unique<BorderedBlockOperator>(
			layout.NDofs(), layout.NPorts(), *frequency_mass,
			/*border=*/nullptr, 1.0, std::move(port_corner));
		owned_scaled_mass = std::move(frequency_mass);

		complex_operator = std::make_unique<OwningComplexBlockOperator>(
			std::move(real_block), std::move(imaginary_block),
			mfem::ComplexOperator::HERMITIAN);
	}

	void SetOmega(mfem::real_t value)
	{
		MFEM_VERIFY(value > 0.0, "MQS angular frequency must be positive.");
		active_omega = value;
		owned_scaled_mass->SetScale(value);
		if (conductance_corner) conductance_corner->SetOmega(value);
	}

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
	std::unique_ptr<mfem::SparseMatrix> AssemblePackedMatrix() const
	{
		const int n = layout.NDofs();
		const int n_ports = layout.NPorts();

		mfem::Array<int> offsets(3);
		offsets[0] = 0;
		offsets[1] = n;
		offsets[2] = n + n_ports;

		// The imaginary half carries the omega scaling that
		// ScaledReferenceOperator applies at Mult time.
		mfem::SparseMatrix scaled_mass(sigma_mass);
		scaled_mass *= active_omega;

		mfem::SparseMatrix minus_coupling(n, n_ports);
		mfem::SparseMatrix minus_coupling_t(n_ports, n);
		mfem::SparseMatrix corner(n_ports, n_ports);
		for (int p = 0; p < n_ports; ++p)
		{
			for (int d = 0; d < n; ++d)
			{
				const mfem::real_t value = port_columns(d, p);
				if (value == 0.0) { continue; }
				minus_coupling.Add(d, p, -value);
				minus_coupling_t.Add(p, d, -value);
			}
			corner.Add(p, p, -conductances[p] / active_omega);
		}
		minus_coupling.Finalize();
		minus_coupling_t.Finalize();
		corner.Finalize();

		// R = [K, -C; -C^T, 0] and I = [omega*M_sigma, 0; 0, -G_dc/omega].
		mfem::BlockMatrix real_blocks(offsets);
		mfem::BlockMatrix imag_blocks(offsets);
		real_blocks.SetBlock(0, 0, &stiffness);		imag_blocks.SetBlock(0, 0, &scaled_mass);
		if (n_ports > 0)
		{
			real_blocks.SetBlock(0, 1, &minus_coupling);
			real_blocks.SetBlock(1, 0, &minus_coupling_t);
			imag_blocks.SetBlock(1, 1, &corner);
		}

		std::unique_ptr<mfem::SparseMatrix> real_half(real_blocks.CreateMonolithic());
		std::unique_ptr<mfem::SparseMatrix> imag_half(imag_blocks.CreateMonolithic());

		mfem::ComplexSparseMatrix complex_matrix(
			real_half.get(), imag_half.get(), /*ownReal=*/false, /*ownImag=*/false,
			mfem::ComplexOperator::HERMITIAN);
		return std::unique_ptr<mfem::SparseMatrix>(complex_matrix.GetSystemMatrix());
	}

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

	void RequireAccepted(const mfem::Vector& rhs, const mfem::Vector& physical_rhs,
		const mfem::Vector& solution,
		const mfem::Array<int>& essential, double tolerance,
		const std::string& context) const {
		auto matrix = AssemblePackedMatrix();
		for (int d : essential) matrix->EliminateRowCol(d, mfem::Operator::DIAG_ONE);
		RequireAcceptedSolve(*matrix, rhs, solution, essential, tolerance, context);
		const auto x = View(solution);
		const auto b = View(physical_rhs);
		std::vector<double> residuals(layout.NPorts()), scales(layout.NPorts());
		double drive = 0.0;
		for (int p = 0; p < layout.NPorts(); ++p) {
			double re = conductances[p] * x.RePort(p) + active_omega * b.ImPort(p);
			double im = conductances[p] * x.ImPort(p) - active_omega * b.RePort(p);
			double scale = conductances[p] * std::hypot(x.RePort(p), x.ImPort(p))
				+ active_omega * std::hypot(b.RePort(p), b.ImPort(p));
			drive = std::max(drive, active_omega * std::hypot(b.RePort(p), b.ImPort(p)));
			for (int d = 0; d < layout.NDofs(); ++d) {
				const double c = active_omega * port_columns(d, p);
				re += c * x.ImMesh(d);
				im -= c * x.ReMesh(d);
				scale += std::abs(c) * std::hypot(x.ReMesh(d), x.ImMesh(d));
			}
			drive = std::max(drive, conductances[p] * std::hypot(x.RePort(p), x.ImPort(p)));
			residuals[p] = std::hypot(re, im);
			scales[p] = scale;
		}
		for (int p = 0; p < layout.NPorts(); ++p) {
			const double allowed = tolerance * drive
				+ 64 * std::numeric_limits<double>::epsilon() * scales[p];
			if (!std::isfinite(residuals[p]) || residuals[p] > allowed) {
				std::ostringstream msg;
				msg << context << ": rejected physical current balance at port " << p
					<< "; residual=" << residuals[p] << " A, allowed=" << allowed << " A";
				throw std::runtime_error(msg.str());
			}
		}
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
