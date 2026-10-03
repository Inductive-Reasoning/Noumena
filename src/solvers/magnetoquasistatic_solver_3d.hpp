// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <map>
#include <set>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "vector_potential_solver_3d.hpp"
#include "mqs_massive_port_operator.hpp"
#include "../coefficients/complex_vector_magnitude_coefficient.hpp"
#include "../coefficients/mqs_vector_electric_field.hpp"
#include "../core/constants.hpp"
#include "../linalg/serial_ams.hpp"
#include "../linalg/sparse_direct_solver.hpp"

/**
 * @brief 3D time-harmonic magnetoquasistatics (eddy currents) in the vector
 *        potential A (H(curl)).
 *
 * Solves curl(nu curl A) + j omega sigma A = J_s + sigma V w for the phasor A
 * at each scenario frequency; see VectorPotentialSolver3D for the
 * discretization, boundary conditions, conductors and gauge.
 *
 * @par Conductors
 *  - Stranded terminals impose their current as a source, I / A_cs along the
 *    path, exactly as in 3D magnetostatics; no eddy currents are modelled in
 *    their strands.
 *  - Massive terminals are ports. The conductor's field is
 *    E = V w - j omega A, with w its DC conduction path (found with its own
 *    conductivity) and V a voltage unknown that makes its net current
 *    integral(sigma E . w) equal the imposed I:
 *        G V - j omega c^T A = I,   c_i = integral sigma w . N_i,
 *        G = integral sigma |w|^2  (the DC conductance).
 *    The block system is MqsMassivePortOperator, the same one the 2D solver
 *    uses. At low frequency the current returns to the DC distribution and
 *    R -> 1/G.
 *  - Any other conducting region (a shield, a tank wall) carries the eddy
 *    currents -j omega sigma A that the sigma mass term induces. A is the
 *    modified potential of the A-formulation: in a conductor its gradient part
 *    carries the electric scalar potential, so the eddy current is
 *    divergence-free with no normal component at the surface, weakly and with
 *    no extra unknown. A closed loop of it carries whatever net current the
 *    induction drives; 2D's "open" regions, which force that to zero, have no
 *    3D counterpart and are rejected.
 *
 * @par Coupling matrix
 * Per frequency, column j drives terminal j with 1 A. A massive row reads
 * its solved port voltage, R = Re V and L = Im V / omega; a stranded row its
 * flux linkage lambda = b'_k . A (b'_k its projected unit load), with
 * V = j omega lambda. Written as one R and one L matrix per frequency.
 *
 * @par Regularization
 * Both linear solvers solve a regularized system: in the nonconducting
 * regions curl-curl alone is singular. Tested with a gradient grad(psi),
 * the regularized field equation reads
 *     integral (beta + j omega sigma) A . grad(psi) = 0,
 * so beta enters charge conservation in, and at the surface of, every
 * conductor: the eddy current is off by a relative beta / (omega sigma). The
 * static weight beta = kRegularization nu_min / L^2 is harmless for good
 * conductors, but not for weak ones. At 50 Hz it overstated the loss in a
 * sigma = 1 S/m block with ends by 68%. Confining beta to the nonconducting
 * regions does not help: the surface term remains. So beta is scaled to the
 * weakest conductor instead,
 *     beta = kRegularization min(nu_min / L^2, omega_min sigma_min),
 * over every scenario frequency and every conducting attribute, which keeps
 * beta / (omega sigma) <= kRegularization everywhere. It is floored at
 * kRegularization times the static weight to keep the null-space pivots
 * above round-off, with a warning if the floor binds.
 *
 * @par Linear solvers
 *  - "direct": the packed real form of the complex system, factored once per
 *    frequency by sparse LU and reused for every terminal column.
 *  - "iterative" (MPI/HYPRE build only): GMRES preconditioned block-
 *    diagonally, with hypre's AMS on K + omega M_sigma for both the real and
 *    the imaginary field block (the frequency-robust choice for
 *    [K, -omega M; omega M, K]) and the exact inverse of the port corner. The
 *    rank-N_ports border is left to GMRES.
 */
class MagnetoquasistaticSolver3D : public VectorPotentialSolver3D {
public:
	MagnetoquasistaticSolver3D(mfem::Mesh& m, const ProblemConfig& c)
		: VectorPotentialSolver3D(m, c) {}

	const mfem::GridFunction& GetSolutionReal() const { return A->real(); }
	const mfem::GridFunction& GetSolutionImag() const { return A->imag(); }

	/// Solved complex voltage of massive terminal @p name for the current
	/// solution.
	std::complex<double> GetPortVoltage(const std::string& name) const {
		for (size_t k = 0; k < conductors.size(); ++k) {
			if (conductors[k].Name != name) continue;
			MFEM_VERIFY(port_of[k] >= 0, "Terminal '" + name + "' is not massive.");
			return port_voltage[port_of[k]];
		}
		MFEM_ABORT("Unknown terminal '" + name + "'.");
		return {};
	}

	/// Time-averaged dissipation of every region that can dissipate.
	std::vector<RegionLoss> ComputeRegionLosses() const {
		if (!A) { return {}; }
		MqsVectorLossDensityCoefficient density(*sigma_coeff, MakeElectricField());
		return IntegrateRegionLosses(density);
	}

	void Setup() override {
		MFEM_VERIFY(!config.Scenarios.empty(),
			"Magnetoquasistatic simulations require at least one frequency scenario.");
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

	void BuildOperators() override {
		BuildSpaceAndConductors();
		const int n = fespace->GetTrueVSize();

		{
			auto operation = Reporter().Start("field matrix assembly");
			regularization = std::make_unique<mfem::ConstantCoefficient>(EddyCurrentRegularization());
			stiffness = std::make_unique<mfem::BilinearForm>(fespace.get());
			stiffness->AddDomainIntegrator(new mfem::CurlCurlIntegrator(*nu_coeff));
			stiffness->AddDomainIntegrator(new mfem::VectorFEMassIntegrator(*regularization));
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

		// Factors and preconditioners belong to the old mesh and frequency.
		direct_solver.reset();
		packed_matrix.reset();
#ifdef MFEM_USE_MPI
		preconditioner.reset();
#endif
		prepared_omega = 0.0;
		if (config.LinearSolver == LinearSolverType::Direct) {
			WarnOnLargeDirectSolve(2 * port_operator->Layout().HalfSize());
		}
	}

	void RunOnCurrentMesh() override {
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

	// Real and imaginary parts of A and of B = curl A, |B| of the phasor and
	// the loss density.
	FieldExportSet CollectExportFields() const override {
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

	double ComputePeakFieldMagnitude() const override {
		return A ? PeakCurlMagnitude({ &A->real(), &A->imag() }) : 0.0;
	}

protected:
	void SaveAnalysisResults() override {
		if (config.AnalysisType == AnalysisType::CouplingMatrix) {
			WriteImpedanceSeries(coupling_results);
		}
	}

private:
	double frequency = 0.0;
	double omega = 0.0;

	std::unique_ptr<mfem::ConstantCoefficient> regularization;
	std::unique_ptr<mfem::BilinearForm> stiffness;   // curl-curl + beta mass
	std::unique_ptr<mfem::BilinearForm> sigma_mass;  // referenced by port_operator
	std::unique_ptr<MqsMassivePortOperator> port_operator;
	mfem::Array<int> ess_packed_tdofs;  // essential DOFs of the packed system

	std::vector<mfem::Vector> stranded_loads;  // by conductor; empty for massive
	std::vector<int> port_of;                  // conductor -> port index, or -1

	std::unique_ptr<mfem::ComplexGridFunction> A;
	std::vector<std::complex<double>> port_voltage;
	std::vector<ImpedancePoint> coupling_results;

	// Solver state for the active frequency (prepared_omega).
	double prepared_omega = 0.0;
	std::unique_ptr<mfem::SparseMatrix> packed_matrix;
	std::unique_ptr<SparseLUSolver> direct_solver;

#ifdef MFEM_USE_MPI
	// Block-diagonal preconditioner of the packed system: AMS on
	// K + omega M_sigma for each field block, the exact inverse of the
	// port corner [0, G/omega; -G/omega, 0] for each port.
	class BlockPreconditioner : public mfem::Solver {
	public:
		BlockPreconditioner(const MqsMassivePortOperator& op, mfem::SparseMatrix& K,
							mfem::SparseMatrix& M_sigma, double omega,
							const mfem::Array<int>& ess, std::vector<mfem::real_t> conductances,
							mfem::FiniteElementSpace& nd)
			: mfem::Solver(op.Layout().FullSize()), op(op), omega(omega),
			  conductances(std::move(conductances)) {
			field.reset(mfem::Add(1.0, K, omega, M_sigma));
			for (int i = 0; i < ess.Size(); ++i) {
				field->EliminateRowCol(ess[i], mfem::Operator::DIAG_ONE);
			}
			ams = std::make_unique<SerialAmsPreconditioner>(*field, nd, /*singular=*/false);
		}

		void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
			const int n = op.Layout().NDofs();
			auto in = op.View(x);
			auto out = op.View(y);
			mfem::Vector r(n), z(n);
			for (int part = 0; part < 2; ++part) {
				for (int i = 0; i < n; ++i) { r(i) = part ? in.ImMesh(i) : in.ReMesh(i); }
				z = 0.0;
				ams->Mult(r, z);
				for (int i = 0; i < n; ++i) { (part ? out.ImMesh(i) : out.ReMesh(i)) = z(i); }
			}
			for (int p = 0; p < op.Layout().NPorts(); ++p) {
				const double g = conductances[p] / omega;
				out.RePort(p) = -in.ImPort(p) / g;
				out.ImPort(p) = in.RePort(p) / g;
			}
		}

		void SetOperator(const mfem::Operator&) override {}

	private:
		const MqsMassivePortOperator& op;
		double omega;
		std::vector<mfem::real_t> conductances;
		std::unique_ptr<mfem::SparseMatrix> field;  // referenced by ams
		std::unique_ptr<SerialAmsPreconditioner> ams;
	};
	std::unique_ptr<BlockPreconditioner> preconditioner;
#endif

	void ActivateFrequency(double f) {
		MFEM_VERIFY(std::isfinite(f) && f > 0.0,
			"MQS scenario frequency must be finite and positive.");
		frequency = f;
		omega = Constants::TWO_PI * f;
		if (port_operator) { port_operator->SetOmega(omega); }
	}

	// n x A = 0 forces the tangential E = -j omega A to zero on the wall, so
	// the wall is a perfect electrical contact: a conductor touching it can
	// pass eddy current into it and back out elsewhere. Right on a symmetry
	// plane that current crosses normally; on an outer box it shorts the
	// conductor's surface to the box. The electrodes of a terminal are meant
	// to touch such a wall and are not reported.
	void WarnOnConductorsTouchingContacts() const {
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

	// beta scaled to the weakest conductor; see the class comment.
	double EddyCurrentRegularization() const {
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

	std::complex<double> FluxLinkage(size_t k) const {
		return { stranded_loads[k] * A->real(), stranded_loads[k] * A->imag() };
	}

	// Solve for @p scenario's excitations at the active frequency.
	void Solve(const Scenario& scenario) {
		const ComplexPortLayout& layout = port_operator->Layout();
		mfem::Vector rhs(layout.FullSize()), x(layout.FullSize());
		rhs = 0.0;
		x = 0.0;
		auto b = port_operator->View(rhs);
		for (size_t k = 0; k < conductors.size(); ++k) {
			const double current = ExcitationFor(scenario, conductors[k].Name);
			if (current == 0.0) continue;
			if (port_of[k] < 0) {
				for (int i = 0; i < layout.NDofs(); ++i) {
					b.ReMesh(i) += current * stranded_loads[k](i);
				}
			}
			else {
				// The port row carries I / (j omega) = -j I / omega (see
				// MqsMassivePortOperator); I is a peak phasor.
				b.ImPort(port_of[k]) = -current / omega;
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

	// Factor, or build the preconditioner, for the active frequency; both are
	// reused across every terminal column at one frequency.
	void PrepareSolver() {
		if (prepared_omega == omega) { return; }
		if (config.LinearSolver == LinearSolverType::Direct) {
			std::ostringstream label;
			label << "sparse direct factorization at " << frequency << " Hz";
			auto operation = Reporter().Start(label.str());
			direct_solver.reset();
			packed_matrix = port_operator->AssemblePackedMatrix();
			for (int i = 0; i < ess_packed_tdofs.Size(); ++i) {
				packed_matrix->EliminateRowCol(ess_packed_tdofs[i], mfem::Operator::DIAG_ONE);
			}
			direct_solver = std::make_unique<SparseLUSolver>(*packed_matrix);
		}
		else {
#ifdef MFEM_USE_MPI
			auto operation = Reporter().Start("AMS preconditioner setup");
			std::vector<mfem::real_t> conductances;
			for (const TerminalConductor& c : conductors) {
				if (c.Type == ConductorType::Massive) conductances.push_back(c.PathIntegral);
			}
			preconditioner.reset();
			preconditioner = std::make_unique<BlockPreconditioner>(
				*port_operator, stiffness->SpMat(), sigma_mass->SpMat(), omega,
				ess_tdof_list, std::move(conductances), *fespace);
#endif
		}
		prepared_omega = omega;
	}

	void SolveIteratively(const mfem::Vector& rhs, mfem::Vector& x) {
#ifdef MFEM_USE_MPI
		mfem::ConstrainedOperator system(&port_operator->Operator(), ess_packed_tdofs);
		mfem::GMRESSolver gmres;
		gmres.SetOperator(system);
		gmres.SetPreconditioner(*preconditioner);
		gmres.SetKDim(200);
		gmres.SetRelTol(config.SolverTolerance);
		gmres.SetAbsTol(0.0);
		gmres.SetMaxIter(config.SolverMaxIter);
		gmres.SetPrintLevel(Reporter().SolverPrintLevel(config.SolverPrintLevel));
		gmres.Mult(rhs, x);

		std::ostringstream msg;
		msg << std::scientific << std::setprecision(3);
		if (gmres.GetConverged()) {
			msg << "GMRES converged in " << gmres.GetNumIterations()
				<< " iterations (relative residual " << gmres.GetFinalRelNorm() << ").";
			Reporter().Diagnostic(msg.str());
		}
		else {
			msg << "GMRES did not converge: relative residual " << gmres.GetFinalRelNorm()
				<< " after " << gmres.GetNumIterations() << " iterations, above "
				   "solver_tolerance " << config.SolverTolerance << ". Raise "
				   "solver_max_iter, loosen solver_tolerance, or use the direct "
				   "solver; results may be inaccurate.";
			Reporter().Warning(msg.str());
		}
#else
		(void)rhs;
		(void)x;
		MFEM_ABORT("The iterative 3D MQS solver needs the MPI/HYPRE build.");
#endif
	}

	std::shared_ptr<const MqsVectorElectricField> MakeElectricField() const {
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
};
