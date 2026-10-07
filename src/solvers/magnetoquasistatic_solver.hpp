// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <cctype>
#include <complex>
#include <iomanip>
#include <limits>
#include <map>
#include <memory> // Required for smart pointers
#include <set>
#include <sstream>
#include "mfem.hpp"
#include "magnetic_solver.hpp"
#include "../axisym/axisymmetric_curl_curl_integrator.hpp"
#include "../axisym/axisymmetric_mass_integrator.hpp"
#include "../axisym/axisymmetric_lf_integrator.hpp"
#include "../coefficients/magnetic_field_coefficient.hpp"
#include "../coefficients/complex_vector_magnitude_coefficient.hpp"
#include "../coefficients/mqs_loss_density_coefficient.hpp"
#include "../core/constants.hpp"
#include "../config/boundary_validation.hpp"
#include "../core/problem_config.hpp"
#include "mqs_block_preconditioner.hpp"
#include "mqs_massive_port_operator.hpp"
#include "../linalg/amg_preconditioner.hpp"
#include "../linalg/complex_block_layout.hpp"
#include "../io/gmsh_results_writer.hpp"
#include "amr_support.hpp"
#include "../linalg/complex_direct_solver.hpp"

class MagnetoquasistaticSolver : public MagneticSolver {
    enum class ImprintMode { Field, CouplingPerturbation };

    double frequency = 0.0;
    mfem::real_t omega = 0.0;
    
    // Complex system objects. S_AA owns the real/imag field matrices referenced
    // by port_operator, so it is declared first and outlives that operator.
	std::unique_ptr<mfem::SesquilinearForm> S_AA;
	std::unique_ptr<mfem::ComplexGridFunction> A; // Complex field solution
	std::unique_ptr<mfem::Vector> Re_port_values; // Real part of port voltages
	std::unique_ptr<mfem::Vector> Im_port_values; // Imag part of port voltages
	std::unique_ptr<MqsMassivePortOperator> port_operator;
	std::unique_ptr<mfem::Vector> x_combined;
	std::unique_ptr<mfem::Vector> b_combined;

	// Direct-solve state for the packed real system. Both are rebuilt whenever
	// the mesh or the active frequency changes; factored_omega records which
	// frequency the current factors belong to.
	std::unique_ptr<mfem::SparseMatrix> packed_matrix;
	std::unique_ptr<ComplexDirectSolver> direct_solver;
	// Preconditioner of the iterative path, for one frequency.
	std::unique_ptr<MqsBlockPreconditioner> preconditioner;
	mfem::real_t preconditioned_omega = 0.0;
	mfem::real_t factored_omega = 0.0;

    // Coefficients
    mfem::Vector neumann_rhs;
    std::vector<mfem::real_t> port_conductances;
    
    // Two DISTINCT index spaces, both essential-DOF lists. Keeping them
    // separately named is deliberate: mixing them silently eliminates the wrong
    // rows, because the packed system is larger than the FE space.
    mfem::Array<int> ess_mesh_tdofs;    // indices into the FE space, [0, N_DOFs)
    mfem::Array<int> ess_packed_tdofs;  // indices into the packed block system,
                                        // [0, N_DOFs + N_Ports); each scalar
                                        // mesh DOF constrains its Re and Im copy.
                                        // Used INSTEAD of PhysicsSolver::ess_tdof_list,
                                        // which is an FE-space list and does not
                                        // apply to this formulation.

    struct MassivePortDefinition {
        std::string Name;
        std::vector<int> AttributeIds;
    };

    // The ordered list of ports that own a voltage unknown.
    //
    // The ordering is a contract, not an implementation detail: the solved
    // vectors Re_port_values / Im_port_values are indexed by position here, so
    // anything mapping a solved voltage back to a region must agree with the
    // order the operator was built from. It lives in one function for that
    // reason; a second copy of this loop would be free to drift.
    //
    // Explicit massive terminals come first so their solved voltage indices
    // retain terminal order. Passive open-current regions follow and receive an
    // identically zero current RHS in every scenario.
    //
    // Conductive regions with no current constraint (a flux shield, a steel
    // brace) are deliberately absent. They still carry eddy currents through the
    // sigma mass term, but nothing constrains their net current, so they have no
    // voltage unknown and their drive field is zero.
    std::vector<MassivePortDefinition> CollectMassivePorts() const
    {
        std::vector<MassivePortDefinition> massive_ports;
        massive_ports.reserve(config.Terminals.size() + config.Regions.size());
        for (const auto& [term_name, term] : config.Terminals) {
            if (term.Conductor != ConductorType::Massive) continue;
            const EntityGroup& group = config.EntityGroups.at(term.EntityGroupName);
            massive_ports.push_back({ term_name, group.AttributeIds });
        }
        for (const Region& region : config.Regions) {
            if (region.CurrentConstraint != RegionCurrentConstraint::Open) continue;
            const EntityGroup& group = config.EntityGroups.at(region.EntityGroupName);
            massive_ports.push_back({ region.EntityGroupName, group.AttributeIds });
        }
        return massive_ports;
    }

    std::vector<ImpedancePoint> coupling_results;

    // The complex block system is solved as a single real vector laid out
    // [Re_Mesh, Re_Port, Im_Mesh, Im_Port]; ComplexPortVectorView and
    // ConstComplexPortVectorView (complex_block_layout.hpp) name the four slots
    // so callers never compute packed indices by hand.

    // Lift the essential boundary values currently projected into *A onto the
    // matching slots of the packed solution vector. FormLinearSystem constrains
    // essential DOFs to whatever it finds there, so without this the projected
    // non-zero Dirichlet values would be forced back to zero.
    void LiftEssentialInto(mfem::Vector& x_packed) const {
        auto x = port_operator->View(x_packed);
        for (int k = 0; k < ess_mesh_tdofs.Size(); ++k) {
            const int d = ess_mesh_tdofs[k];
            x.ReMesh(d) = A->real()(d);
            x.ImMesh(d) = A->imag()(d);
        }
    }

	void RecoverSolvedUnknowns(mfem::Vector& x_packed) {
		auto x = port_operator->View(x_packed);
        // Copy field DOFs back into the complex grid function
        for (int i = 0; i < port_operator->Layout().NDofs(); ++i) {
            A->real()(i) = x.ReMesh(i);
            A->imag()(i) = x.ImMesh(i);
        }
        // Copy solved port voltages back into real/imaginary port vectors
		Re_port_values = std::make_unique<mfem::Vector>(port_operator->Layout().NPorts());
		Im_port_values = std::make_unique<mfem::Vector>(port_operator->Layout().NPorts());
		for (int p = 0; p < port_operator->Layout().NPorts(); ++p) {
			// Recover the solved port values from the packed vector
            (*Re_port_values)(p) = x.RePort(p);
            (*Im_port_values)(p) = x.ImPort(p);
		}
	}

	// Per-attribute drive amplitudes from the solved port voltages, indexed by
	// (attribute - 1) and sized to the mesh.
	//
	// The tables are zero-initialised and written only where a port unknown
	// exists. That default is load-bearing rather than defensive: conductive
	// regions with no current constraint (a flux shield, a steel brace) carry
	// eddy currents through the sigma mass term but own no voltage unknown, so
	// zero is their physically correct drive and the general loss expression
	// collapses to 0.5*sigma*omega^2*|A|^2 for them with no special case.
	//
	// Ordering comes from CollectMassivePorts(), the same function the operator
	// was built from, so voltage index p always refers to the same region here
	// as it does there.
	void BuildDriveTables(std::vector<double>& drive_re,
						  std::vector<double>& drive_im) const {
		const int n_attr = mesh.attributes.Max();
		drive_re.assign(n_attr, 0.0);
		drive_im.assign(n_attr, 0.0);

		if (!Re_port_values || !Im_port_values) { return; }

		const std::vector<MassivePortDefinition> massive_ports = CollectMassivePorts();
		MFEM_VERIFY(static_cast<int>(massive_ports.size()) == Re_port_values->Size(),
			"Massive port count (" + std::to_string(massive_ports.size()) +
			") does not match the solved voltage count (" +
			std::to_string(Re_port_values->Size()) +
			"). The port ordering used for loss recovery has diverged from the "
			"one the operator was assembled with.");

		for (size_t p = 0; p < massive_ports.size(); ++p) {
			for (int attr : massive_ports[p].AttributeIds) {
				const int index = attr - 1;
				if (index < 0 || index >= n_attr) { continue; }
				drive_re[index] = (*Re_port_values)(static_cast<int>(p));
				drive_im[index] = (*Im_port_values)(static_cast<int>(p));
			}
		}
	}

public:
	// Solved complex vector potential. Exposed const so verification code can
	// recompute derived quantities independently of the solver's own paths.
	const mfem::GridFunction& GetSolutionReal() const { return A->real(); }
	const mfem::GridFunction& GetSolutionImag() const { return A->imag(); }

	// Solved complex voltage of a named massive port, as (real, imaginary).
	//
	// Looked up through CollectMassivePorts() so the index always refers to the
	// same region the operator was assembled from.
	std::pair<double, double> GetPortVoltage(const std::string& port_name) const {
		MFEM_VERIFY(Re_port_values && Im_port_values,
			"Port voltages are unavailable; solve before querying them.");
		const std::vector<MassivePortDefinition> ports = CollectMassivePorts();
		for (size_t p = 0; p < ports.size(); ++p) {
			if (ports[p].Name != port_name) { continue; }
			return { (*Re_port_values)(static_cast<int>(p)),
					 (*Im_port_values)(static_cast<int>(p)) };
		}
		MFEM_ABORT("Unknown massive port '" + port_name + "'.");
		return { 0.0, 0.0 };
	}

	// Time-averaged dissipation of every region that can dissipate. Public
	// because it is a result of the analysis in its own right.
	std::vector<RegionLoss> ComputeRegionLosses() const {
		if (!A || !sigma_coeff) { return {}; }
		std::vector<double> drive_re, drive_im;
		BuildDriveTables(drive_re, drive_im);
		MqsLossDensityCoefficient density(
			*sigma_coeff, A->real(), A->imag(), omega,
			drive_re, drive_im,
			geometry == GeometryType::Axisymmetric);
        return IntegrateRegionLosses(density);
	}

public:
	// Constructor deals only with initialization, no manual nullptr assignment needed
	MagnetoquasistaticSolver(mfem::Mesh &m, const ProblemConfig &c) : MagneticSolver(m, c) {}

	void Setup() override {
        int order = config.Order;
        const int dim = mesh.Dimension();

        MFEM_VERIFY(!config.Scenarios.empty(),
            "Magnetoquasistatic simulations require at least one frequency scenario.");
        frequency = config.Scenarios.front().second.Frequency;
        omega = Constants::TWO_PI * frequency;

        InitializeMagneticGeometry();
        WarnOnSlowComplexDirectSolve();
        for (const auto& [term_name, term] : config.Terminals) {
            MFEM_VERIFY(term.DriveQuantity == Quantity::Current,
                "Magnetoquasistatic terminal '" + term_name +
                "' must use a current excitation; massive terminal voltage "
                "is a solved output.");
        }

        // Reject negative radii, record whether the domain reaches r = 0, and
        // report under-resolved near-axis curl-curl quadrature.
        ValidateMagneticAxisymmetricGeometry();

        // FE spaces
        fec = std::make_unique<mfem::H1_FECollection>(order, mesh.Dimension());
        
        // Materials
        // Real part: reluctivity nu = 1/mu.
        BuildReluctivity();

        // Assemble conductivity without frequency scaling so the mass matrix can
        // be reused at every sweep point.
        BuildConductivity();

        // MQS terminals are ports driven through the port block, not essential
        // boundaries, so they are deliberately NOT registered into the set here.
        boundary_conditions = BuildBoundaryConditions();
        BuildEssentialBoundaryMarker();
        RequireReferencePotential();

        // Build the FE space and everything bound to it for the starting mesh.
        BuildOperators();
        ValidateMagneticAxisBoundaryValues();

        // Validate that BCs don't create physical conflicts
        BoundaryConditionValidator validator(mesh, *fespace);
        validator.ValidateBoundaryConditions(
            boundary_conditions.Entries(), /*terminals=*/{}, false);  // Strict mode - reject conflicts
    }

	void BuildOperators() override {
		// Build the FE space and everything bound to it for the starting mesh.
		fespace = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get());
		Reporter().Status("Mesh has " + std::to_string(mesh.GetNE()) +
			" elements; field space has " + std::to_string(fespace->GetTrueVSize()) +
			" true DOFs.");
		neumann_rhs = AssembleNaturalBoundaryLoad();

		{
			auto operation = Reporter().Start("complex bilinear form assembly");
			// Setup Complex Billinear Form
			S_AA = std::make_unique<mfem::SesquilinearForm>(fespace.get(), mfem::ComplexOperator::HERMITIAN);
			S_AA->AddDomainIntegrator(MakeStiffnessIntegrator(), nullptr);
			S_AA->AddDomainIntegrator(nullptr, MakeMassIntegrator());

			S_AA->Assemble();
			S_AA->Finalize();
		}

        mfem::BilinearForm& K = S_AA->real();
        mfem::BilinearForm& M_sigma = S_AA->imag();

        const int n_dofs = fespace->GetTrueVSize();

        const std::vector<MassivePortDefinition> massive_ports = CollectMassivePorts();

        std::vector<std::unique_ptr<mfem::Vector>> port_loads;
        port_loads.reserve(massive_ports.size());
        port_conductances.clear();
        port_conductances.reserve(massive_ports.size());
        {
        auto operation = Reporter().Start(
            "massive port assembly (" + std::to_string(massive_ports.size()) + " ports)");
        size_t port_index = 0;
        for (const MassivePortDefinition& port : massive_ports) {
            // Winding models declare one port per turn, so report periodically
            // rather than once per port.
            if (++port_index % 25 == 0) {
                Reporter().Status("  assembled " + std::to_string(port_index) + " of " +
                    std::to_string(massive_ports.size()) + " massive ports");
            }
            ValidateMassiveConductor(port.Name, port.AttributeIds);
            port_loads.push_back(std::make_unique<mfem::Vector>(
                MassiveConductorLoad(port.Name, port.AttributeIds)));
            port_conductances.push_back(MassiveConductance(port.Name, port.AttributeIds));
        }
        }

        // Hand the field matrices and port data to the block-system owner, which
        // assembles the complex saddle-point operator and owns all the wiring.
        port_operator = std::make_unique<MqsMassivePortOperator>(
            n_dofs, K.SpMat(), M_sigma.SpMat(), std::move(port_loads),
            port_conductances, omega);

        fespace->GetEssentialTrueDofs(ess_bdr, ess_mesh_tdofs);   // indices in [0, N_DOFs)
        AddAxisTrueDofs(ess_mesh_tdofs);

        // Each scalar essential DOF constrains both its real and imaginary copy
        // in the packed [Re|Im] layout (half-size = N_DOFs + N_Ports).
        ess_packed_tdofs = port_operator->MakeEssentialTDofs(ess_mesh_tdofs);

		// Grid Function (for solution recovery later)
		A = std::make_unique<mfem::ComplexGridFunction>(fespace.get());

		// Any factorization from a previous mesh refers to the old DOF numbering.
		direct_solver.reset();
		packed_matrix.reset();
		factored_omega = 0.0;
		preconditioner.reset();
	}

	mfem::BilinearFormIntegrator* MakeMassIntegrator() {
		return Geometry().NewMassIntegrator(*sigma_coeff);
	}

    void ActivateFrequency(const Scenario& sc) {
        MFEM_VERIFY(std::isfinite(sc.Frequency) && sc.Frequency > 0.0,
            "MQS scenario frequency must be finite and positive.");
        frequency = sc.Frequency;
        omega = Constants::TWO_PI * frequency;
        port_operator->SetOmega(omega);
    }

    void ImprintScenario(const Scenario& sc, ImprintMode mode) {
        ActivateFrequency(sc);
        *A = 0.0;

        // Re-apply non-zero essential BC values on this mesh's A.
        // ess_tdof values are lifted into the RHS by FormLinearSystem at solve time,
        // so they must be set AFTER the *A = 0.0 reset, every scenario.
        if (mode == ImprintMode::Field) {
            ForEachNonzeroDirichlet([&](mfem::Array<int>& marker, double value) {
                mfem::ConstantCoefficient c_re(value);
                mfem::ConstantCoefficient c_im(0.0);
                A->ProjectBdrCoefficient(c_re, c_im, marker);
            });
        }
        // Axis stays at A=0 (already zero from the reset; no projection needed).

        // The monolithic real/imag solver vectors (size 2*(N_DOFs+N_Ports)) are
        // laid out [Re_Mesh, Re_Port, Im_Mesh, Im_Port]; assemble the RHS
        // directly into that layout through a typed view rather than raw indices.
        b_combined = std::make_unique<mfem::Vector>(port_operator->Layout().FullSize());
        x_combined = std::make_unique<mfem::Vector>(port_operator->Layout().FullSize());
        *b_combined = 0.0;
        auto b = port_operator->View(*b_combined);

        // Stranded source, its real and imaginary parts into the Re_Mesh and
        // Im_Mesh blocks.
        mfem::Vector j_re, j_im;
        BuildCurrentDensity(sc, j_re, j_im);
        mfem::PWConstCoefficient j_re_coeff(j_re), j_im_coeff(j_im);
        mfem::LinearForm b_re(fespace.get()), b_im(fespace.get());
        b_re.AddDomainIntegrator(Geometry().NewDomainLFIntegrator(j_re_coeff));
        b_im.AddDomainIntegrator(Geometry().NewDomainLFIntegrator(j_im_coeff));
        b_re.Assemble();
        b_im.Assemble();
        for (int d = 0; d < port_operator->Layout().NDofs(); ++d) {
            b.ReMesh(d) += b_re[d];
            b.ImMesh(d) += b_im[d];
            if (mode == ImprintMode::Field) {
                b.ReMesh(d) += neumann_rhs[d];
            }
        }

        // Drive the massive ports.
        //
        // The prescribed excitation IS the current phasor I, and the RHS entry
        // that produces it is I/(j*omega) = -j*I/omega in this block ordering.
        // It is used unscaled, so the PEAK-phasor convention enters the solve
        // here and every downstream quantity -- solved port voltages, the
        // coupling matrix, and the 1/2 in the loss density -- inherits it.
        int p = 0;
        for (const auto& [term_name, term] : config.Terminals) {
            if (term.Conductor != ConductorType::Massive) continue;   // keep p aligned
            const std::complex<double> port_rhs =
                ExcitationFor(sc, term_name) / std::complex<double>(0.0, omega);
            b.RePort(p) = port_rhs.real();
            b.ImPort(p) = port_rhs.imag();
            ++p;
        }

        *x_combined = 0.0; // Initial guess
        LiftEssentialInto(*x_combined);
    }

    // Solve + save on the CURRENT mesh/operators. Field analysis performs one
    // solve per prescribed scenario. Coupling analysis uses each unique frequency
    // and synthesizes one unit-current solve per terminal.
    void RunOnCurrentMesh() override {
        PrepareAnalysis();

        if (config.AnalysisType == AnalysisType::Field) {
            for (const auto& [name, scenario] : config.Scenarios) {
                auto operation = Reporter().Start("scenario '" + name + "'");
                ImprintScenario(scenario, ImprintMode::Field);
                SolveSystem();
                AccumulateScenarioError();
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
        for (const auto& [point_frequency, point_name] : frequency_points) {
            BeginCouplingPoint(point_frequency);
            for (const auto& [term_name, term] : config.Terminals) {
                Scenario column;
                column.Frequency = point_frequency;
                column.Excitations.push_back({ term_name, 1.0 });
                auto operation = Reporter().Start(
                    "scenario '" + point_name + "', terminal '" + term_name + "'");
                ImprintScenario(column, ImprintMode::CouplingPerturbation);
                SolveSystem();
                AccumulateScenarioError();
                GatherCouplingColumn(column);
                SaveScenario(point_name, column, term_name);
            }
        }
    }

    void SolveSystem() {
        auto operation = Reporter().Start("linear system solve");
        // Solve
        mfem::OperatorHandle A_op;
        mfem::Vector B_vec, X_vec;

        mfem::Operator* A_op_ptr;

		mfem::ComplexOperator& complex_system = port_operator->Operator();
		complex_system.FormLinearSystem(ess_packed_tdofs, *x_combined, *b_combined, A_op_ptr, X_vec, B_vec);
		bool own_A = (A_op_ptr != &complex_system);
		A_op.Reset(A_op_ptr, own_A);

		if (config.LinearSolver == LinearSolverType::Direct) {
			// The factorization is valid for one frequency, so it is cached and
			// reused across every terminal column at that frequency; only a
			// change of frequency (which rescales both omega-dependent blocks)
			// forces a rebuild.
			EnsureFactorizationForActiveFrequency();
			direct_solver->Mult(B_vec, X_vec);
		}
		else {
			// FGMRES preconditioned by PRESB with AMG (see
			// MqsBlockPreconditioner).
			EnsurePreconditionerForActiveFrequency();
			SolveNonsymmetricIteratively(*A_op.Ptr(), *preconditioner, B_vec, X_vec);
		}

		// X_vec is laid out [Re_Mesh, Re_Port, Im_Mesh, Im_Port]; copy the mesh
		// (field) DOFs back into the complex grid function, dropping the ports.
		RecoverSolvedUnknowns(X_vec);
	}

    // Assemble and factor the packed real system for the currently active
    // frequency, reusing the existing factors if the frequency has not moved.
    //
    // Unlike the static solvers, the MQS matrix is NOT constant over a run: both
    // the sigma-mass block (omega*M_sigma) and the port corner (G_dc/omega)
    // depend on frequency. It is, however, constant across the terminal columns
    // of a single frequency point, which is where the reuse pays off: one
    // factorization serves every terminal at that frequency.
    void EnsurePreconditionerForActiveFrequency() {
        if (preconditioner && preconditioned_omega == omega) { return; }
        auto operation = Reporter().Start("AMG preconditioner setup");
        preconditioner = std::make_unique<MqsBlockPreconditioner>(
            port_operator->Layout(), S_AA->real().SpMat(), S_AA->imag().SpMat(), omega,
            ess_mesh_tdofs, port_conductances, [](mfem::SparseMatrix& field) {
                return std::make_unique<AmgPreconditioner>(field);
            });
        preconditioned_omega = omega;
    }

    void EnsureFactorizationForActiveFrequency() {
        if (direct_solver && factored_omega == omega) { return; }

        auto operation = Reporter().Start(
            "sparse direct factorization at " + std::to_string(frequency) + " Hz");
        packed_matrix = port_operator->AssemblePackedMatrix();

        // Apply the same essential-DOF elimination that ComplexOperator's
        // constrained operator applies matrix-free. FormLinearSystem has already
        // folded the essential values into B_vec and placed them in X_vec, so a
        // unit diagonal on the eliminated rows reproduces them exactly.
        for (int i = 0; i < ess_packed_tdofs.Size(); ++i) {
            packed_matrix->EliminateRowCol(ess_packed_tdofs[i], mfem::Operator::DIAG_ONE);
        }

        // One solver per mesh: refactoring it at a new frequency reuses its
        // ordering.
        if (!direct_solver) {
            direct_solver = std::make_unique<ComplexDirectSolver>(port_operator->Layout());
        }
        direct_solver->Factor(*packed_matrix);
        factored_omega = omega;
    }

    // Estimate per-element error on the CURRENT mesh. The scenario-wide fold (a
    // running RMS over every scenario / frequency point) lives in the base class
    // AccumulateScenarioError(), so one shared mesh is refined for all scenarios
    // (spec: identical $Nodes/$Elements across every <scenario>.results.msh).
    //
    // Uses the serial recovery-based ZienkiewiczZhuEstimator (the L2 variant is
    // MPI-only). A dedicated integrator instance (separate from a's) and an H1
    // vector flux space are constructed here per call; SetFluxAveraging(1) keeps
    // the recovered flux from smoothing across material-attribute interfaces so
    // per-region reluctivity discontinuities are respected. The real and
    // imaginary indicators are combined as a complex magnitude.
    //
    // That magnitude is then normalized by the total field energy of the phasor,
    // Et = 0.5*(Re^T K Re) + 0.5*(Im^T K Im), making it a dimensionless RELATIVE
    // error. This matters more here than in the static solvers: an MQS run
    // sweeps frequency, and the field energy of a fixed excitation varies by
    // orders of magnitude across a decade sweep as eddy currents screen the
    // conductor. Folding raw indicators would let the highest-energy frequency
    // point set the mesh for the whole sweep.
    //
    // @param errors  Output: per-element error indicator (sized to NE).
    void EstimateCurrentSolutionError(mfem::Vector& errors) override {
        const int sdim = mesh.SpaceDimension();
        std::unique_ptr<mfem::BilinearFormIntegrator> flux_integ(MakeStiffnessIntegrator());
        mfem::FiniteElementSpace flux_fes(&mesh, fec.get(), sdim);
        mfem::ZienkiewiczZhuEstimator estimator_re(*flux_integ, A->real(), flux_fes);
        estimator_re.SetWithCoeff(false);     // flux = nu * grad(A)
        estimator_re.SetFluxAveraging(1);    // do not average across attribute interfaces

        mfem::ZienkiewiczZhuEstimator estimator_im(*flux_integ, A->imag(), flux_fes);
        estimator_im.SetWithCoeff(false);     // flux = nu * grad(A)
        estimator_im.SetFluxAveraging(1);    // do not average across attribute interfaces

        const mfem::Vector& errs_re = estimator_re.GetLocalErrors();
        const mfem::Vector& errs_im = estimator_im.GetLocalErrors();
        errors.SetSize(errs_re.Size());
        for (int k = 0; k < errors.Size(); ++k) {
            errors(k) = std::hypot(errs_re(k), errs_im(k));
        }

        // Energy of the phasor is the sum of the real and imaginary parts'
        // energies (the cross term vanishes in the time average). A zero/near-
        // zero solution carries no energy and no meaningful relative error;
        // leave the indicator unscaled rather than dividing by ~0.
        const double energy =
            amr::FieldEnergy(*fespace, MakeStiffnessIntegrator(), A->real()) +
            amr::FieldEnergy(*fespace, MakeStiffnessIntegrator(), A->imag());
        if (energy > 0.0) { errors /= std::sqrt(energy); }
    }

    // Peak flux density |B| over the current solution *A, sampled at element
    // nodes. AMR convergence diagnostic: the peak near a conductor corner or
    // high-permeability edge should settle as refinement resolves it.
    //
    // Planar: B = (dA/dy, -dA/dx), so |B| == |grad(A)| exactly.
    // Axisymmetric: B_r=-dA/dz, B_z=A/r+dA/dr, so the A/r term matters; reuse
    // MagneticFieldCoefficient (same B reconstruction as the exporters, incl.
    // the r->0 limit). Reflects whichever solution currently lives in *A.
    double ComputePeakFieldMagnitude() const override {
        if (!A) { return 0.0; }

        std::optional<MagneticFieldCoefficient> B_axi_re, B_axi_im;
        if (axis_geometry) {
            B_axi_re.emplace(&A->real(), axis_geometry->tolerance);
            B_axi_im.emplace(&A->imag(), axis_geometry->tolerance);
        }

        double peak = 0.0;
        mfem::Vector B_re;
        mfem::Vector B_im;
        for (int e = 0; e < fespace->GetNE(); ++e) {
            const mfem::FiniteElement* fe = fespace->GetFE(e);
            mfem::ElementTransformation* T = fespace->GetElementTransformation(e);
            const mfem::IntegrationRule& nodes = fe->GetNodes();
            for (int i = 0; i < nodes.GetNPoints(); ++i) {
                const mfem::IntegrationPoint& ip = nodes.IntPoint(i);
                T->SetIntPoint(&ip);
                if (axis_geometry) {
                    B_axi_re->Eval(B_re, *T, ip);
                    B_axi_im->Eval(B_im, *T, ip);
                }  // true |B| incl. A/r term
                else {
                    A->real().GetGradient(*T, B_re);
                    A->imag().GetGradient(*T, B_im);
                }   // |B| == |grad(A)| (planar)
                const double mag_re = B_re.Norml2();
                const double mag_im = B_im.Norml2();
                const double mag = std::hypot(mag_re, mag_im);
                if (mag > peak) { peak = mag; }
            }
        }
        return peak;
    }


	// Post-solve field recovery for the complex solution: the real/imaginary
	// vector-potential parts (primaries), the real/imaginary flux densities
	// B = curl(A) (derived vectors, geometry-dependent), and the complex flux
	// magnitude |B| = sqrt(|Re B|^2 + |Im B|^2). Serialization is handled by the
	// base class.
	FieldExportSet CollectExportFields() const override {
		FieldExportSet fields;
		fields.AddPrimary("A_Real", A->real());
		fields.AddPrimary("A_Imag", A->imag());

		mfem::VectorCoefficient* b_re;
		mfem::VectorCoefficient* b_im;
		if (geometry == GeometryType::Axisymmetric) {
			// Axisymmetric B = Curl(A_phi) = (-dA/dz, 1/r*d(rA)/dr)
			b_re = &fields.AddVector("B_Real",
				std::make_unique<MagneticFieldCoefficient>(
					&A->real(), axis_geometry->tolerance));
			b_im = &fields.AddVector("B_Imag",
				std::make_unique<MagneticFieldCoefficient>(
					&A->imag(), axis_geometry->tolerance));
		}
		else {
			// Planar B = Curl(A_z) = (dA/dy, -dA/dx)
			b_re = &fields.AddVector("B_Real",
                std::make_unique<PlanarMagneticFieldCoefficient>(&A->real()));
			b_im = &fields.AddVector("B_Imag",
                std::make_unique<PlanarMagneticFieldCoefficient>(&A->imag()));
		}

		fields.AddScalar("B_Magnitude",
			std::make_unique<ComplexVectorMagnitudeCoefficient>(*b_re, *b_im));

		// Time-averaged Joule loss density [W/m^3], defined over every
		// conductive region rather than only the ported ones: the sigma mass
		// term induces eddy currents wherever sigma > 0, so a shield or brace
		// dissipates even though it owns no voltage unknown.
		std::vector<double> drive_re, drive_im;
		BuildDriveTables(drive_re, drive_im);
		fields.AddScalar("P_Loss",
			std::make_unique<MqsLossDensityCoefficient>(
				*sigma_coeff, A->real(), A->imag(), omega,
				std::move(drive_re), std::move(drive_im),
				geometry == GeometryType::Axisymmetric));
		return fields;
	}

	void GatherCouplingColumn(const Scenario& sc) {
        MFEM_VERIFY(sc.Excitations.size() == 1,
            "MQS coupling scenarios must drive exactly one terminal.");
        const auto driven = config.Terminals.find(sc.Excitations.front().TerminalName);
        MFEM_VERIFY(driven != config.Terminals.end(),
            "MQS coupling scenario references an unknown terminal.");
        const int column = static_cast<int>(std::distance(config.Terminals.begin(), driven));

        ImpedancePoint& point = coupling_results.back();
        int massive_port = 0;
        int row = 0;
        for (const auto& [term_name, term] : config.Terminals) {
            if (term.Conductor == ConductorType::Massive) {
                point.Resistance(row, column) = (*Re_port_values)(massive_port);
                point.Inductance(row, column) =
                    (*Im_port_values)(massive_port) / omega;
                ++massive_port;
            }
            else {
                const auto [flux_re, flux_im] =
                    ComputeStrandedFluxLinkage(term_name);
                point.Resistance(row, column) = -omega * flux_im;
                point.Inductance(row, column) = flux_re;
            }
            ++row;
        }
	}

	void PrepareAnalysis() {
		if (config.AnalysisType == AnalysisType::CouplingMatrix) {
            for (const auto& [term_name, term] : config.Terminals) {
                MFEM_VERIFY(term.DriveQuantity == Quantity::Current,
                    "MQS CouplingMatrix terminal '" + term_name +
                    "' must be a Current terminal.");
            }

            coupling_results.clear();
		}
	}

    void BeginCouplingPoint(double point_frequency) {
        const int n = static_cast<int>(config.Terminals.size());
        ImpedancePoint point;
        point.Frequency = point_frequency;
        point.Resistance.SetSize(n);
        point.Inductance.SetSize(n);
        point.Resistance = 0.0;
        point.Inductance = 0.0;
        coupling_results.push_back(std::move(point));
    }

    void SaveAnalysisResults() override
	{
		if (config.AnalysisType == AnalysisType::CouplingMatrix) {
			WriteImpedanceSeries(coupling_results);
		}
	}

    std::pair<double, double> ComputeStrandedFluxLinkage(
        const std::string& terminal_name) const {
        mfem::Vector unit_density =
            BuildTerminalCurrentDensity(terminal_name, 1.0);
        mfem::PWConstCoefficient unit_density_coeff(unit_density);
        mfem::LinearForm winding_functional(fespace.get());
        winding_functional.AddDomainIntegrator(
            Geometry().NewDomainLFIntegrator(unit_density_coeff));
        winding_functional.Assemble();

        // The integrator carries the full geometric measure, so these are webers.
        return {
            winding_functional * A->real(),
            winding_functional * A->imag()
        };
    }

    // Stranded-conductor source current density for a scenario. Massive
    // conductors are driven through the port block instead, so they are
    // excluded here.
    void BuildCurrentDensity(const Scenario& sc, mfem::Vector& j_re,
                             mfem::Vector& j_im) const {
        MagneticSolver::BuildCurrentDensity(sc, [](const Terminal& term) {
            return term.Conductor == ConductorType::Stranded;
        }, j_re, j_im);
    }
};