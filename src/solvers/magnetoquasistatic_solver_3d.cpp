// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "magnetoquasistatic_solver_3d.hpp"

std::complex<double> MagnetoquasistaticSolver3D::GetPortVoltage(const std::string& name) const {
	for (size_t k = 0; k < conductors.size(); ++k) {
		if (conductors[k].Name != name) continue;
		MFEM_VERIFY(port_of[k] >= 0, "Terminal '" + name + "' is not massive.");
		return port_voltage[port_of[k]];
	}
	MFEM_ABORT("Unknown terminal '" + name + "'.");
	return {};
}

std::vector<RegionLoss> MagnetoquasistaticSolver3D::ComputeRegionLosses() const {
	if (!A) { return {}; }
	MqsVectorLossDensityCoefficient density(*sigma_coeff, MakeElectricField());
	return IntegrateRegionLosses(density);
}

void MagnetoquasistaticSolver3D::Setup()  {
	MFEM_VERIFY(!config.Scenarios.empty(),
		"Magnetoquasistatic simulations require at least one frequency scenario.");
	WarnOnSlowComplexDirectSolve();
	for (const Region& region : config.Regions) {
		MFEM_VERIFY(region.CurrentConstraint == RegionCurrentConstraint::None,
			"Region '" + region.EntityGroupName + "' has current_constraint "
			"'open', which geometry_type '3d' does not support: a 3D conductor "
			"with no terminal carries only induced current, so model its "
			"actual extent instead.");
	}
	ActivateFrequency(config.Scenarios.front().second.Frequency);
	InitializeVectorPotential();
	WarnOnConductorsTouchingContacts();
	BuildOperators();
	ValidateVectorPotentialBoundaries();
}

void MagnetoquasistaticSolver3D::BuildOperators()  {
#ifdef MFEM_USE_MPI
	preconditioner.reset();  // refers to the space and matrices about to be replaced
#endif
	BuildSpaceAndConductors();
	const int n = fespace->GetTrueVSize();

	const bool direct = config.LinearSolver == LinearSolverType::Direct;
	{
		auto operation = Reporter().Start("field matrix assembly");
		stiffness = std::make_unique<mfem::BilinearForm>(fespace.get());
		stiffness->AddDomainIntegrator(new mfem::CurlCurlIntegrator(*nu_coeff));
		if (!direct) {
			regularization = std::make_unique<mfem::ConstantCoefficient>(EddyCurrentRegularization());
			stiffness->AddDomainIntegrator(new mfem::VectorFEMassIntegrator(*regularization));
		}
		stiffness->Assemble();
		stiffness->Finalize();

		sigma_mass = std::make_unique<mfem::BilinearForm>(fespace.get());
		sigma_mass->AddDomainIntegrator(new mfem::VectorFEMassIntegrator(*sigma_coeff));
		sigma_mass->Assemble();
		sigma_mass->Finalize();
	}

	// Stranded conductors are sources (projected, as in magnetostatics);
	// massive ones are ports, driven by the field sigma w of a unit
	// voltage. Port columns need no projection: they live on conducting
	// elements, where the sigma mass term already pins every gradient.
	std::vector<std::unique_ptr<mfem::Vector>> port_loads;
	std::vector<mfem::real_t> conductances;
	stranded_loads.assign(conductors.size(), mfem::Vector());
	port_of.assign(conductors.size(), -1);
	for (size_t k = 0; k < conductors.size(); ++k) {
		const TerminalConductor& c = conductors[k];
		if (c.Type == ConductorType::Stranded) {
			stranded_loads[k] = ProjectedUnitCurrentLoad(c);
			continue;
		}
		ProjectedUnitCurrentLoad(c);  // only to check that its DC current balances
		port_of[k] = static_cast<int>(port_loads.size());
		port_loads.push_back(std::make_unique<mfem::Vector>(AssembleConductorLoad(c, 1.0)));
		conductances.push_back(c.PathIntegral);
	}
	port_operator = std::make_unique<MqsMassivePortOperator>(
		n, stiffness->SpMat(), sigma_mass->SpMat(), std::move(port_loads),
		conductances, omega);
	ess_packed_tdofs = port_operator->MakeEssentialTDofs(ess_tdof_list);

	A = std::make_unique<mfem::ComplexGridFunction>(fespace.get());
	*A = 0.0;
	port_voltage.assign(conductances.size(), 0.0);

	// Factors belong to the old mesh and frequency.
	direct_solver.reset();
	packed_matrix.reset();
	prepared_omega = 0.0;

	// The direct path gauges the field block by a Lagrange multiplier,
	// the Coulomb gauge in the nonconducting regions (see
	// ComplexDirectSolver); the iterative one is regularized instead.
	gauge.reset();
	if (direct) {
		mfem::Array<int> conducting(mesh.attributes.Max());
		for (int a = 1; a <= conducting.Size(); ++a) {
			conducting[a - 1] = (*sigma_coeff)(a) > 0.0 ? 1 : 0;
		}
		gauge = std::make_unique<GaugeConstraintRows>(
			*projector->GaugeConstraint(conducting), ess_tdof_list);
		WarnOnLargeDirectSolve(port_operator->Layout().HalfSize(),
							   ComplexDirectSolver::kLarge3DUnknowns);
	}
}

void MagnetoquasistaticSolver3D::RunOnCurrentMesh()  {
	if (config.AnalysisType == AnalysisType::Field) {
		for (const auto& [name, scenario] : config.Scenarios) {
			auto operation = Reporter().Start("scenario '" + name + "'");
			ActivateFrequency(scenario.Frequency);
			Solve(scenario);
			const std::vector<RegionLoss> losses = ComputeRegionLosses();
			ReportRegionLosses(losses);
			SaveScenario(name, scenario, {}, losses);
		}
		return;
	}

	std::map<double, std::string> frequency_points;
	for (const auto& [name, scenario] : config.Scenarios) {
		frequency_points.emplace(scenario.Frequency, name);
	}
	coupling_results.clear();
	const int n = static_cast<int>(conductors.size());
	for (const auto& [point_frequency, point_name] : frequency_points) {
		ActivateFrequency(point_frequency);
		ImpedancePoint point;
		point.Frequency = point_frequency;
		point.Resistance.SetSize(n);
		point.Inductance.SetSize(n);
		for (int column = 0; column < n; ++column) {
			Scenario drive;
			drive.Frequency = point_frequency;
			drive.Excitations.push_back({ conductors[column].Name, 1.0 });
			auto operation = Reporter().Start(
				"scenario '" + point_name + "', terminal '" + conductors[column].Name + "'");
			Solve(drive);
			for (int row = 0; row < n; ++row) {
				// Z = V / I with I = 1 A: a port's solved voltage, or
				// j omega lambda for a stranded terminal.
				const std::complex<double> Z = port_of[row] >= 0
					? port_voltage[port_of[row]]
					: std::complex<double>(0.0, omega) * FluxLinkage(row);
				point.Resistance(row, column) = Z.real();
				point.Inductance(row, column) = Z.imag() / omega;
			}
			SaveScenario(point_name, drive, conductors[column].Name);
		}
		coupling_results.push_back(std::move(point));
	}
}

FieldExportSet MagnetoquasistaticSolver3D::CollectExportFields() const  {
	FieldExportSet fields;
	fields.AddPrimary("A_Real", A->real());
	fields.AddPrimary("A_Imag", A->imag());
	auto& b_re = fields.AddVector("B_Real",
		std::make_unique<mfem::CurlGridFunctionCoefficient>(&A->real()));
	auto& b_im = fields.AddVector("B_Imag",
		std::make_unique<mfem::CurlGridFunctionCoefficient>(&A->imag()));
	fields.AddScalar("B_Magnitude",
		std::make_unique<ComplexVectorMagnitudeCoefficient>(b_re, b_im));
	const auto e = MakeElectricField();
	using J = MqsVectorCurrentDensityCoefficient;
	fields.AddVector("J_Real", std::make_unique<J>(*sigma_coeff, e, J::Part::Real));
	fields.AddVector("J_Imag", std::make_unique<J>(*sigma_coeff, e, J::Part::Imag));
	fields.AddScalar("P_Loss", std::make_unique<MqsVectorLossDensityCoefficient>(*sigma_coeff, e));
	return fields;
}

void MagnetoquasistaticSolver3D::SaveAnalysisResults()  {
	if (config.AnalysisType == AnalysisType::CouplingMatrix) {
		WriteImpedanceSeries(coupling_results);
	}
}

void MagnetoquasistaticSolver3D::ActivateFrequency(double f) {
	MFEM_VERIFY(std::isfinite(f) && f > 0.0,
		"MQS scenario frequency must be finite and positive.");
	frequency = f;
	omega = Constants::TWO_PI * f;
	if (port_operator) { port_operator->SetOmega(omega); }
}

void MagnetoquasistaticSolver3D::WarnOnConductorsTouchingContacts() const {
	std::set<int> electrodes;
	for (const auto& [name, term] : config.Terminals) {
		const CurrentDirection& d = *term.Direction;
		if (d.Type != CurrentDirection::Kind::Electrodes) continue;
		for (const std::string& group : { d.Input, d.Output }) {
			const auto& ids = config.EntityGroups.at(group).AttributeIds;
			electrodes.insert(ids.begin(), ids.end());
		}
	}
	std::set<int> touching;
	for (int be = 0; be < mesh.GetNBE(); ++be) {
		const int a = mesh.GetBdrAttribute(be);
		if (a < 1 || a > ess_bdr.Size() || !ess_bdr[a - 1] || electrodes.count(a)) continue;
		int e1, e2;
		mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(be), &e1, &e2);
		for (int e : { e1, e2 }) {
			if (e >= 0 && (*sigma_coeff)(mesh.GetAttribute(e)) > 0.0) {
				touching.insert(mesh.GetAttribute(e));
			}
		}
	}
	for (const auto& [name, attrs] : ConductingGroups()) {
		if (std::none_of(attrs.begin(), attrs.end(),
				[&](int a) { return touching.count(a) != 0; })) continue;
		Reporter().Warning("Conductor '" + name + "' touches an n x A = 0 "
			"('dirichlet') boundary, which acts as a perfect electrical contact: "
			"eddy current can flow between it and the boundary. Intended on a "
			"symmetry plane; otherwise keep the conductor off the boundary.");
	}
}

double MagnetoquasistaticSolver3D::EddyCurrentRegularization() const {
	const double static_weight = RegularizationWeight();
	double omega_min = std::numeric_limits<double>::max();
	for (const auto& [name, scenario] : config.Scenarios) {
		omega_min = std::min(omega_min, Constants::TWO_PI * scenario.Frequency);
	}
	double sigma_min = std::numeric_limits<double>::max();
	for (int attr : mesh.attributes) {
		const double sigma = (*sigma_coeff)(attr);
		if (sigma > 0.0) sigma_min = std::min(sigma_min, sigma);
	}
	if (sigma_min == std::numeric_limits<double>::max()) { return static_weight; }

	const double weight = std::min(static_weight, kRegularization * omega_min * sigma_min);
	const double floor = kRegularization * static_weight;
	if (weight >= floor) { return weight; }
	std::ostringstream msg;
	msg << std::setprecision(3) << "The weakest conductor (sigma = " << sigma_min
		<< " S/m at " << omega_min / Constants::TWO_PI << " Hz) conducts too little "
		"for the regularization, which is held at its round-off floor: beta / "
		"(omega sigma) = " << floor / (omega_min * sigma_min) << ", and eddy-current "
		"losses there may be off by several tens of times that. Model it as "
		"nonconducting if its eddy currents do not matter.";
	Reporter().Warning(msg.str());
	return floor;
}

void MagnetoquasistaticSolver3D::Solve(const Scenario& scenario) {
	const ComplexPortLayout& layout = port_operator->Layout();
	mfem::Vector rhs(layout.FullSize()), x(layout.FullSize());
	rhs = 0.0;
	x = 0.0;
	auto b = port_operator->View(rhs);
	for (size_t k = 0; k < conductors.size(); ++k) {
		const std::complex<double> current = ExcitationFor(scenario, conductors[k].Name);
		if (current == 0.0) continue;
		if (port_of[k] < 0) {
			for (int i = 0; i < layout.NDofs(); ++i) {
				b.ReMesh(i) += current.real() * stranded_loads[k](i);
				b.ImMesh(i) += current.imag() * stranded_loads[k](i);
			}
		}
		else {
			// The port row carries I / (j omega) = -j I / omega (see
			// MqsMassivePortOperator); I is a peak phasor.
			const std::complex<double> port_rhs = current / std::complex<double>(0.0, omega);
			b.RePort(port_of[k]) = port_rhs.real();
			b.ImPort(port_of[k]) = port_rhs.imag();
		}
	}
	for (int i = 0; i < ess_packed_tdofs.Size(); ++i) { rhs(ess_packed_tdofs[i]) = 0.0; }

	{
		auto operation = Reporter().Start("linear system solve");
		PrepareSolver();
		if (direct_solver) {
			direct_solver->Mult(rhs, x);
		}
		else {
			SolveIteratively(rhs, x);
		}
	}

	auto solved = port_operator->View(x);
	for (int i = 0; i < layout.NDofs(); ++i) {
		A->real()(i) = solved.ReMesh(i);
		A->imag()(i) = solved.ImMesh(i);
	}
	for (int p = 0; p < layout.NPorts(); ++p) {
		port_voltage[p] = { solved.RePort(p), solved.ImPort(p) };
	}
}

void MagnetoquasistaticSolver3D::PrepareSolver() {
	if (prepared_omega == omega) { return; }
	if (config.LinearSolver == LinearSolverType::Direct) {
		std::ostringstream label;
		label << "sparse direct factorization at " << frequency << " Hz";
		auto operation = Reporter().Start(label.str());
		packed_matrix = port_operator->AssemblePackedMatrix();
		for (int i = 0; i < ess_packed_tdofs.Size(); ++i) {
			packed_matrix->EliminateRowCol(ess_packed_tdofs[i], mfem::Operator::DIAG_ONE);
		}
		// One solver per mesh: refactoring it at a new frequency reuses its
		// ordering.
		if (!direct_solver) {
			direct_solver = std::make_unique<ComplexDirectSolver>(port_operator->Layout(),
																  gauge.get());
		}
		direct_solver->Factor(*packed_matrix);
	}
	else {
#ifdef MFEM_USE_MPI
		auto operation = Reporter().Start("AMS preconditioner setup");
		std::vector<mfem::real_t> conductances;
		for (const TerminalConductor& c : conductors) {
			if (c.Type == ConductorType::Massive) conductances.push_back(c.PathIntegral);
		}
		preconditioner.reset();
		preconditioner = std::make_unique<MqsBlockPreconditioner>(
			port_operator->Layout(), stiffness->SpMat(), sigma_mass->SpMat(), omega,
			ess_tdof_list, std::move(conductances), [&](mfem::SparseMatrix& field) {
				return std::make_unique<SerialAmsPreconditioner>(field, *fespace,
																 /*singular=*/false);
			});
#endif
	}
	prepared_omega = omega;
}

void MagnetoquasistaticSolver3D::SolveIteratively(const mfem::Vector& rhs, mfem::Vector& x) {
#ifdef MFEM_USE_MPI
	mfem::ConstrainedOperator system(&port_operator->Operator(), ess_packed_tdofs);
	SolveNonsymmetricIteratively(system, *preconditioner, rhs, x, ess_packed_tdofs);
#else
	(void)rhs;
	(void)x;
	MFEM_ABORT("The iterative 3D MQS solver needs the MPI/HYPRE build.");
#endif
}

std::shared_ptr<const MqsVectorElectricField> MagnetoquasistaticSolver3D::MakeElectricField() const {
	std::vector<MqsVectorElectricField::Drive> drives;
	std::vector<int> drive_of_attribute(mesh.attributes.Max(), -1);
	for (size_t k = 0; k < conductors.size(); ++k) {
		if (port_of[k] < 0) continue;
		const int index = static_cast<int>(drives.size());
		drives.push_back({ conductors[k].Path.get(), port_voltage[port_of[k]] });
		for (int a = 0; a < conductors[k].Marker.Size(); ++a) {
			if (conductors[k].Marker[a]) drive_of_attribute[a] = index;
		}
	}
	return std::make_shared<const MqsVectorElectricField>(
		A->real(), A->imag(), omega, std::move(drives), std::move(drive_of_attribute));
}
