// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include "mfem.hpp"
#include "magnetic_solver.hpp"
#include "../config/boundary_validation.hpp"
#include "../core/constants.hpp"
#include "../linalg/sparse_direct_solver.hpp"

/**
 * @brief 3D magnetostatics in the full vector potential A (H(curl)).
 *
 * Solves curl(nu curl A) = J on a 3D mesh with A in a Nedelec (edge element)
 * space, so the tangential continuity of A across element faces -- and
 * therefore the normal continuity of B = curl A -- holds exactly, and material
 * interfaces need no special treatment.
 *
 * Status (milestone M0 of the 3D magnetics plan): the formulation, assembly,
 * boundary conditions, direct solve and field export are in place. Sources
 * are supplied programmatically through SetSourceCurrentDensity(); coil
 * terminals (M1), the coupling-matrix analysis (M1) and an iterative solver
 * (M3) are not yet available and are rejected in Setup(). The configuration
 * validator still rejects '3d' magnetostatics for the same reason.
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
 * @par Gauge and regularization
 * The curl-curl operator annihilates gradients, so it is singular even with
 * n x A fixed on the whole boundary: every gradient of an interior nodal
 * function is in its null space. The direct factorization cannot handle a
 * singular matrix, so a small mass term beta (A, w) is added,
 *     beta = kRegularization * nu_min / L^2,
 * with L the mesh bounding-box diagonal and nu_min the smallest reluctivity.
 * With a divergence-free source this selects the Coulomb-gauged solution and
 * perturbs B by a relative O(kRegularization) in every material, while keeping
 * the null-space pivots (relative size kRegularization * (nu_min/nu_max) *
 * (h/L)^2) above round-off. An auxiliary-space preconditioned iterative solver
 * handles the singular system directly and will not need this term.
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
		MFEM_VERIFY(config.Terminals.empty(),
			"3D magnetostatics does not yet support terminals: coil sources "
			"are not implemented.");
		MFEM_VERIFY(config.AnalysisType == AnalysisType::Field,
			"3D magnetostatics does not yet support the coupling-matrix "
			"analysis, which needs coil terminals.");
		MFEM_VERIFY(!config.Amr.Enabled,
			"3D magnetostatics does not yet support adaptive refinement.");
		MFEM_VERIFY(config.LinearSolver == LinearSolverType::Direct,
			"3D magnetostatics has no iterative solver yet: the curl-curl "
			"operator needs an auxiliary-space (AMS) preconditioner. Set "
			"simulation.linear_solver to 'direct'.");

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

		regularization = std::make_unique<mfem::ConstantCoefficient>(RegularizationWeight());
		a = std::make_unique<mfem::BilinearForm>(fespace.get());
		a->AddDomainIntegrator(new mfem::CurlCurlIntegrator(*nu_coeff));
		a->AddDomainIntegrator(new mfem::VectorFEMassIntegrator(*regularization));
		a->Assemble();

		fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
		a->FormSystemMatrix(ess_tdof_list, A_op);

		auto* matrix = dynamic_cast<mfem::SparseMatrix*>(A_op.Ptr());
		MFEM_VERIFY(matrix, "Expected a SparseMatrix operator from FormSystemMatrix.");
		auto operation = Reporter().Start("sparse direct factorization");
		direct_solver = std::make_unique<SparseDirectSolver>(*matrix);
	}

	void RunOnCurrentMesh() override {
		for (const auto& [name, scenario] : BuildSolveScenarios()) {
			auto operation = Reporter().Start("scenario '" + name + "'");
			ImprintScenario();
			SolveSystem();
			SaveScenario(name, scenario);
		}
	}

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
	void SaveAnalysisResults() override {}

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

	mfem::VectorCoefficient* source = nullptr;          // not owned
	mfem::VectorCoefficient* boundary_value = nullptr;  // not owned

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

	void ImprintScenario() {
		*A = 0.0;
		if (boundary_value) {
			A->ProjectBdrCoefficientTangent(*boundary_value, ess_bdr);
		}
		b = std::make_unique<mfem::LinearForm>(fespace.get());
		if (source) {
			b->AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(*source));
		}
		b->Assemble();
	}

	void SolveSystem() {
		auto operation = Reporter().Start("linear system solve");
		mfem::Vector X, B;
		a->FormLinearSystem(ess_tdof_list, *A, *b, A_op, X, B);
		direct_solver->Mult(B, X);
		a->RecoverFEMSolution(X, *b, *A);
	}
};
