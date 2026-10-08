// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "mqs_massive_port_operator.hpp"

void MqsMassivePortOperator::ConductanceCornerOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const 
{
	for (int p = 0; p < Height(); ++p) {
		y(p) = -conductances[p] * x(p) / omega;
	}
}

MqsMassivePortOperator::MqsMassivePortOperator(
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

void MqsMassivePortOperator::SetOmega(mfem::real_t value)
{
	MFEM_VERIFY(value > 0.0, "MQS angular frequency must be positive.");
	active_omega = value;
	owned_scaled_mass->SetScale(value);
	if (conductance_corner) conductance_corner->SetOmega(value);
}

std::unique_ptr<mfem::SparseMatrix> MqsMassivePortOperator::AssemblePackedMatrix() const
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
