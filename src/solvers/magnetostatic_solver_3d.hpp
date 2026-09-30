// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "magnetic_solver.hpp"
#include "divergence_free_projector.hpp"
#include "../coefficients/coil_path.hpp"
#include "../config/boundary_validation.hpp"
#include "../core/constants.hpp"
#include "../linalg/serial_ams.hpp"
#include "../linalg/sparse_direct_solver.hpp"
#include "../parallel/mpi_runtime.hpp"

/**
 * @brief 3D magnetostatics in the full vector potential A (H(curl)).
 *
 * Solves curl(nu curl A) = J on a 3D mesh with A in a Nedelec (edge element)
 * space, so the tangential continuity of A across element faces -- and
 * therefore the normal continuity of B = curl A -- holds exactly, and material
 * interfaces need no special treatment.
 *
 * @par Sources
 * Current enters through coil terminals: each current terminal gives a volume
 * group and a "direction" (CoilDirection): azimuthal about an axis, open
 * between electrodes, or closed through a cut. Every coil carries a uniform
 * current density I / A_cs along its path (see coil_path.hpp); an azimuthal
 * coil's is exactly the 2D axisymmetric model's I/area source revolved. Each coil's unit-current load is assembled once per
 * mesh and made discretely divergence-free (DivergenceFreeProjector); a
 * scenario's load is the excitation-weighted sum. A programmatic source
 * (SetSourceCurrentDensity) is added to it and projected the same way.
 *
 * @par Coupling matrix
 * Column j drives terminal j with 1 A; entry (k, j) is the flux linkage
 * lambda_k = integral(A . J_k) with J_k terminal k's unit-current density,
 * evaluated as b'_k . A with the projected load b'_k. That is the same winding
 * functional the 2D solvers use, it is gauge-invariant because b'_k is
 * orthogonal to the gradients, and it makes L = B'^T K^-1 B' symmetric by
 * construction. Units are henries.
 *
 * Not yet available, and rejected in Setup(): adaptive refinement.
 *
 * @par Boundary conditions
 * A 3D vector potential has no meaningful scalar boundary value, so the
 * configured conditions are restricted to their homogeneous forms:
 *  - Dirichlet (value 0): n x A = 0, "flux tangent" -- B has no normal
 *    component (magnetic insulation, or a symmetry plane B cannot cross);
 *  - Neumann (value 0) or no entry: the natural condition n x H = 0,
 *    "flux normal" -- B crosses the boundary normally.
 * Nonzero tangential data n x A = n x g can be imposed programmatically with
 * SetTangentialBoundaryValue(); it is used by the manufactured-solution tests.
 *
 * @par Linear solvers, gauge and regularization
 * The curl-curl operator annihilates gradients, so it is singular even with
 * n x A fixed on the whole boundary: every gradient of an interior nodal
 * function is in its null space. The two linear solvers deal with that
 * differently:
 *  - "iterative" (MPI/HYPRE build only): CG preconditioned by hypre's AMS
 *    (SerialAmsPreconditioner) on the singular system itself, which is
 *    consistent because every load is projected. The iteration count stays
 *    roughly constant under refinement. CG leaves A's gradient part
 *    arbitrary, so it is removed afterwards
 *    (DivergenceFreeProjector::RemoveGradient), putting A in the discrete
 *    Coulomb gauge.
 *  - "direct": the factorization cannot handle a singular matrix, so a small
 *    mass term beta (A, w) is added,
 *     beta = kRegularization * nu_min / L^2,
 * with L the mesh bounding-box diagonal and nu_min the smallest reluctivity.
 * With a divergence-free source this selects the Coulomb-gauged solution and
 * perturbs B by a relative O(kRegularization) in every material, while keeping
 * the null-space pivots (relative size kRegularization * (nu_min/nu_max) *
 * (h/L)^2) above round-off. Its fill-in limits it to small 3D problems.
 *
 * @warning The source must be divergence-free (and, discretely, orthogonal to
 * the gradients of the essential-boundary nodal space) or the regularized
 * solve returns a gauge-polluted A. Coil sources in M1 will be projected; a
 * programmatic source is the caller's responsibility.
 */
class MagnetostaticSolver3D : public MagneticSolverBase {
public:
	/// Relative size of the regularizing mass term; see the class comment.
	static constexpr double kRegularization = 1e-6;

	MagnetostaticSolver3D(mfem::Mesh& m, const ProblemConfig& c)
		: MagneticSolverBase(m, c) {}

	/// Source current density J [A/m^2], applied to every scenario. Not owned;
	/// must outlive the solve. nullptr (the default) means no source.
	void SetSourceCurrentDensity(mfem::VectorCoefficient* J) { source = J; }

	/// Tangential boundary data g, imposing n x A = n x g on every Dirichlet
	/// boundary. Not owned. nullptr (the default) means n x A = 0.
	void SetTangentialBoundaryValue(mfem::VectorCoefficient* g) { boundary_value = g; }

	/// The solved vector potential (Nedelec grid function).
	const mfem::GridFunction& GetSolution() const { return *A; }

	/// Magnetic energy W = 1/2 integral(nu |curl A|^2) [J] of the current
	/// solution, excluding the regularization term.
	double MagneticEnergy() const {
		mfem::BilinearForm k(fespace.get());
		k.AddDomainIntegrator(new mfem::CurlCurlIntegrator(*nu_coeff));
		k.Assemble();
		k.Finalize();
		return 0.5 * k.InnerProduct(*A, *A);
	}

	void Setup() override {
		InitializeGeometry();
		MFEM_VERIFY(geometry == GeometryType::Cartesian3D,
			"MagnetostaticSolver3D requires geometry_type '3d'.");
		for (const auto& [name, term] : config.Terminals) {
			MFEM_VERIFY(term.DriveQuantity == Quantity::Current,
				"Magnetostatic terminal '" + name + "' must use a current excitation.");
			MFEM_VERIFY(term.Direction.has_value(),
				"3D coil terminal '" + name + "' needs a 'direction'.");
		}
		MFEM_VERIFY(!config.Amr.Enabled,
			"3D magnetostatics does not yet support adaptive refinement.");
		MFEM_VERIFY(config.LinearSolver == LinearSolverType::Direct || parallel::Enabled(),
			"The iterative solver for 3D magnetostatics uses hypre's AMS "
			"preconditioner and needs the MPI/HYPRE build (-DUSE_MPI=ON). Set "
			"simulation.linear_solver to 'direct' in this serial build.");

		BuildReluctivity();
		fec = std::make_unique<mfem::ND_FECollection>(config.Order, mesh.Dimension());

		boundary_conditions = BuildBoundaryConditions();
		for (const auto& bc : boundary_conditions) {
			MFEM_VERIFY(bc.Condition.Value == 0.0,
				"Boundary group '" + bc.Condition.EntityGroupName + "' has value " +
				std::to_string(bc.Condition.Value) + ". 3D magnetostatics supports "
				"only homogeneous conditions: 'dirichlet' 0 (n x A = 0, flux "
				"tangent) and 'neumann' 0 (n x H = 0, flux normal).");
		}
		BuildEssentialBoundaryMarker();

		BuildOperators();

		BoundaryConditionValidator validator(mesh, *fespace);
		validator.ValidateBoundaryConditions(
			boundary_conditions.Entries(), /*terminals=*/{}, /*allow_overlap=*/false);
	}

	void BuildOperators() override {
		fespace = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get());
		A = std::make_unique<mfem::GridFunction>(fespace.get());
		*A = 0.0;

		const bool direct = config.LinearSolver == LinearSolverType::Direct;
		a = std::make_unique<mfem::BilinearForm>(fespace.get());
		a->AddDomainIntegrator(new mfem::CurlCurlIntegrator(*nu_coeff));
		if (direct) {
			regularization = std::make_unique<mfem::ConstantCoefficient>(RegularizationWeight());
			a->AddDomainIntegrator(new mfem::VectorFEMassIntegrator(*regularization));
		}
		a->Assemble();

		fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
		a->FormSystemMatrix(ess_tdof_list, A_op);

		auto* matrix = dynamic_cast<mfem::SparseMatrix*>(A_op.Ptr());
		MFEM_VERIFY(matrix, "Expected a SparseMatrix operator from FormSystemMatrix.");
		direct_solver.reset();
#ifdef MFEM_USE_MPI
		ams.reset();
#endif
		if (direct) {
			WarnOnLargeDirectSolve(fespace->GetTrueVSize());
			auto operation = Reporter().Start("sparse direct factorization");
			direct_solver = std::make_unique<SparseDirectSolver>(*matrix);
		}
		else {
#ifdef MFEM_USE_MPI
			auto operation = Reporter().Start("AMS preconditioner setup");
			ams = std::make_unique<SerialAmsPreconditioner>(*matrix, *fespace, /*singular=*/true);
#endif
		}

		auto operation = Reporter().Start("coil source assembly");
		projector = std::make_unique<DivergenceFreeProjector>(*fespace, ess_bdr);
		coil_loads.clear();
		for (const auto& [name, term] : config.Terminals) {
			coil_loads.push_back(AssembleUnitCoilLoad(name, term));
		}
	}

	void RunOnCurrentMesh() override {
		const bool coupling = config.AnalysisType == AnalysisType::CouplingMatrix;
		const int n = static_cast<int>(config.Terminals.size());
		if (coupling) {
			L = std::make_unique<mfem::DenseMatrix>(n, n);
			*L = 0.0;
		}
		int column = 0;
		for (const auto& [name, scenario] : BuildSolveScenarios()) {
			auto operation = Reporter().Start("scenario '" + name + "'");
			ImprintScenario(scenario);
			SolveSystem();
			if (coupling) {
				for (int row = 0; row < n; ++row) { (*L)(row, column) = coil_loads[row] * *A; }
				++column;
			}
			SaveScenario(name, scenario,
				coupling ? scenario.Excitations.front().TerminalName : "");
		}
	}

	/// Flux linkage lambda_k [Wb] of every terminal for the current solution,
	/// in config.Terminals (name) order.
	std::vector<double> FluxLinkages() const {
		std::vector<double> lambda;
		for (const auto& load : coil_loads) { lambda.push_back(load * *A); }
		return lambda;
	}

	/// The projected unit-current load of each terminal, in config.Terminals
	/// order (exposed for verification).
	const std::vector<mfem::Vector>& CoilLoads() const { return coil_loads; }
	const DivergenceFreeProjector& Projector() const { return *projector; }

	// Post-solve fields: the potential A (a vector Nedelec field) and the flux
	// density B = curl A, evaluated exactly from the element basis.
	FieldExportSet CollectExportFields() const override {
		FieldExportSet fields;
		fields.AddPrimary("A", *A);
		fields.AddVector("B", std::make_unique<mfem::CurlGridFunctionCoefficient>(A.get()));
		return fields;
	}

	// Peak |B| over quadrature points of the current solution.
	double ComputePeakFieldMagnitude() const override {
		if (!A) { return 0.0; }
		mfem::CurlGridFunctionCoefficient curl(A.get());
		mfem::Vector B;
		double peak = 0.0;
		for (int e = 0; e < mesh.GetNE(); ++e) {
			mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
			const mfem::IntegrationRule& ir =
				mfem::IntRules.Get(mesh.GetElementBaseGeometry(e), 2 * config.Order);
			for (int q = 0; q < ir.GetNPoints(); ++q) {
				const mfem::IntegrationPoint& ip = ir.IntPoint(q);
				T->SetIntPoint(&ip);
				curl.Eval(B, *T, ip);
				peak = std::max(peak, B.Norml2());
			}
		}
		return peak;
	}

protected:
	void SaveAnalysisResults() override {
		if (config.AnalysisType != AnalysisType::CouplingMatrix) return;
		if (!L) {
			Reporter().Warning("WriteCouplingMatrix: coupling matrix not computed.");
			return;
		}
		SaveCouplingMatrix(*L, "Inductance Matrix " + CouplingUnitLabel("H"),
			"Inductance", "H");
	}

	void EstimateCurrentSolutionError(mfem::Vector&) override {
		MFEM_ABORT("3D magnetostatics does not yet support adaptive refinement.");
	}

private:
	std::unique_ptr<mfem::GridFunction> A;
	std::unique_ptr<mfem::ConstantCoefficient> regularization;
	std::unique_ptr<mfem::BilinearForm> a;
	std::unique_ptr<mfem::LinearForm> b;
	mfem::OperatorHandle A_op;
	std::unique_ptr<SparseDirectSolver> direct_solver;
#ifdef MFEM_USE_MPI
	std::unique_ptr<SerialAmsPreconditioner> ams;  // iterative path
#endif

	mfem::VectorCoefficient* source = nullptr;          // not owned
	mfem::VectorCoefficient* boundary_value = nullptr;  // not owned

	std::unique_ptr<DivergenceFreeProjector> projector;
	std::vector<mfem::Vector> coil_loads;  // projected unit loads, terminal order
	std::unique_ptr<mfem::DenseMatrix> L;  // inductance matrix (coupling runs)

	// Load vector of terminal @p name driven by 1 A, made divergence-free.
	mfem::Vector AssembleUnitCoilLoad(const std::string& name, const Terminal& term) {
		const EntityGroup& group = config.EntityGroups.at(term.EntityGroupName);
		mfem::Array<int> marker =
			DomainMarkerFromAttrs(group.AttributeIds, "coil terminal '" + name + "'");

		const std::unique_ptr<CoilPath> path = MakeCoilPath(name, *term.Direction, marker);
		const double area = CoilCrossSection(mesh, marker, *path, config.Order);
		MFEM_VERIFY(area > 0.0, "Coil terminal '" + name + "' has zero cross-section.");

		CoilCurrentCoefficient density(*path, 1.0 / area);
		mfem::LinearForm load(fespace.get());
		load.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(density), marker);
		load.Assemble();

		mfem::Vector b(load);
		projector->Project(b);
		return b;
	}

	std::unique_ptr<CoilPath> MakeCoilPath(const std::string& name, const CoilDirection& d,
										   const mfem::Array<int>& coil) {
		const std::string context = "coil terminal '" + name + "'";
		if (d.Type == CoilDirection::Kind::Azimuthal) {
			auto path = std::make_unique<AzimuthalPath>(d);
			// The direction is undefined on the axis; a vertex there means the
			// coil reaches it (quadrature points alone could miss that).
			mfem::Vector lo, hi;
			mesh.GetBoundingBox(lo, hi);
			hi -= lo;
			mfem::Array<int> vertices;
			for (int e = 0; e < mesh.GetNE(); ++e) {
				if (!coil[mesh.GetAttribute(e) - 1]) continue;
				mesh.GetElementVertices(e, vertices);
				for (int v : vertices) {
					mfem::Vector x(mesh.GetVertex(v), 3);
					MFEM_VERIFY(path->RadiusOf(x) > 1e-9 * hi.Norml2(),
						"Coil terminal '" + name + "' reaches its own axis, where "
						"the azimuthal current direction is undefined.");
				}
			}
			return path;
		}

		const int n_bdr = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
		mfem::Array<int> none(n_bdr);
		none = 0;
		if (d.Type == CoilDirection::Kind::Electrodes) {
			mfem::Array<int> input = MarkerFromGroup(d.Input);
			mfem::Array<int> output = MarkerFromGroup(d.Output);
			// Current may only enter or leave the model through an n x A = 0
			// wall: anywhere else the load is not balanced, and the projection
			// would silently redistribute the missing return current.
			for (int a = 0; a < n_bdr; ++a) {
				MFEM_VERIFY(!(input[a] || output[a]) || ess_bdr[a],
					"The electrodes of " + context + " must lie on a 'dirichlet' "
					"(n x A = 0) boundary; for a closed coil use a 'cut'.");
			}
			return std::make_unique<ConductionPath>(mesh, config.Order, coil, d, input, output, none);
		}
		return std::make_unique<ConductionPath>(mesh, config.Order, coil, d, none, none,
												MarkerFromGroup(d.Cut));
	}

	double RegularizationWeight() const {
		mfem::Vector lo, hi;
		mesh.GetBoundingBox(lo, hi);
		hi -= lo;
		const double length = hi.Norml2();
		MFEM_VERIFY(length > 0.0, "3D mesh has an empty bounding box.");

		double nu_min = std::numeric_limits<double>::max();
		for (int attr : mesh.attributes) {
			nu_min = std::min(nu_min, (*nu_coeff)(attr));
		}
		return kRegularization * nu_min / (length * length);
	}

	void ImprintScenario(const Scenario& scenario) {
		*A = 0.0;
		if (boundary_value) {
			A->ProjectBdrCoefficientTangent(*boundary_value, ess_bdr);
		}
		b = std::make_unique<mfem::LinearForm>(fespace.get());
		if (source) {
			b->AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(*source));
		}
		b->Assemble();
		if (source) { projector->Project(*b); }

		int k = 0;
		for (const auto& [name, term] : config.Terminals) {
			const double current = ExcitationFor(scenario, name);
			if (current != 0.0) { b->Add(current, coil_loads[k]); }
			++k;
		}
	}

	void SolveSystem() {
		auto operation = Reporter().Start("linear system solve");
		mfem::Vector X, B;
		a->FormLinearSystem(ess_tdof_list, *A, *b, A_op, X, B);
		if (direct_solver) {
			direct_solver->Mult(B, X);
			a->RecoverFEMSolution(X, *b, *A);
			return;
		}
#ifdef MFEM_USE_MPI
		SolveSpdIteratively(*A_op, *ams, B, X);
		a->RecoverFEMSolution(X, *b, *A);
		projector->RemoveGradient(*A);
#endif
	}
};
