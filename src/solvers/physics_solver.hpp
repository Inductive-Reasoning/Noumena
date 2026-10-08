// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once
#include <cmath>
#include <complex>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "mfem.hpp"
#include "../core/problem_config.hpp"
#include "../io/field_export.hpp"
#include "../io/matrix_writer.hpp"
#include "../io/coupling_matrix_writer.hpp"
#include "../io/result_writer.hpp"
#include "../io/status_reporter.hpp"
#include "amr_support.hpp"
#include "../axisym/axis_geometry.hpp"
#include "geometry_model.hpp"
#include "../core/marked_boundary_condition.hpp"

/**
 * @brief Base class for physics solvers using MFEM
 *
 * @warning The mesh reference must outlive this solver instance.
 */
class PhysicsSolver {
    // ---- State --------------------------------------------------------------
protected:
	mfem::Mesh& mesh;
	ProblemConfig config;

    // Held as the BASE collection type deliberately. Every solver currently
    // builds an H1 space, but nothing in this class requires H1: the FE space
    // constructor and GetOrder() are both base-class API. Naming the concrete
    // type here would commit every present and future solver to a nodal scalar
    // discretization, which is a formulation choice that belongs to the derived
    // solver, not to the shared plumbing.
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    std::unique_ptr<mfem::FiniteElementSpace> fespace;
    // The stiffness of the AMR energy norm (EstimateRelativeZZError), per mesh.
    std::unique_ptr<mfem::BilinearForm> energy_form;
    const mfem::FiniteElementSpace* energy_fes = nullptr;
    long energy_sequence = -1;
    GeometryType geometry = GeometryType::Planar;
    mfem::Array<int> ess_bdr;
    mfem::Array<int> ess_tdof_list;

    // The prescribed boundary conditions of this solve
    // Terminals are NOT here; each solver realizes those itself. See
    // docs/boundary_and_terminal_model.md.
    BoundaryConditionSet boundary_conditions;
private:
    std::unique_ptr<ResultWriter> result_writer;
    std::vector<amr::AmrIterationInfo> amr_history;
	// Non-null only while an AMR pass is in flight: the running root-mean-square
	// of the per-scenario error indicators accumulated by the current scenario
	// loop. Owned by RunAdaptive(); derived solvers reach it only through
	// AccumulateScenarioError().
	mfem::Vector* amr_errors = nullptr;

	// Number of scenario indicators folded into *amr_errors so far. Reset per
	// AMR iteration alongside amr_errors; drives the running-average update.
	int amr_error_samples = 0;

    // ---- Public API ---------------------------------------------------------
public:
    PhysicsSolver(mfem::Mesh& m, const ProblemConfig& c) : mesh(m), config(c) {}

    // Virtual destructor is essential for unique_ptr polymorphism
    virtual ~PhysicsSolver() = default;

    virtual void Setup() = 0;
    void Run();
    void SaveAnalysis();

    // Post-solve field recovery: each solver declares WHAT to export for the
    // just-solved scenario (primary solution fields + derived coefficients).
    // The set is format-agnostic; SolverFieldWriter decides HOW to serialize it.
    virtual FieldExportSet CollectExportFields() const = 0;

    // AMR per-iteration diagnostics from the most recent Run(). Empty when AMR
    // is disabled. Read-only convergence history for logging and regression tests.
    const std::vector<amr::AmrIterationInfo>& GetAmrHistory() const { return amr_history; }

    // ---- Virtual methods: each solver supplies its own physics -------------
protected:
    virtual void SaveAnalysisResults() = 0;
    virtual void BuildOperators() = 0;
    virtual void RunOnCurrentMesh() = 0;

    // Per-element error indicator for whichever solution currently lives in the
    // solver's grid function. The scenario-wide fold lives in the base class.
    virtual void EstimateCurrentSolutionError(mfem::Vector& errors) = 0;
    virtual double ComputePeakFieldMagnitude() const = 0;

    // Builds ess_bdr: every boundary attribute whose DOFs this formulation pins.
    //
    // The default is the prescribed Dirichlet conditions alone. Formulations with
    // additional essential sources override this, call the base, and merge their
    // own markers in -- voltage terminals for electrostatics, axis regularity for
    // the axisymmetric magnetic solvers. Collecting the union behind one named
    // hook is what makes "what pins DOFs here?" answerable per solver, and what
    // guarantees the merges happen before BuildOperators() consumes ess_bdr.
    //
    // Called from Setup(); the result is refinement-invariant (attribute-keyed)
    // and is reused across every AMR pass.
    virtual void BuildEssentialBoundaryMarker() {
        ess_bdr = boundary_conditions.DirichletMarker(
            mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0);
    }

    // Every connected piece of the mesh needs something that fixes the scalar
    // potential on it: a boundary in ess_bdr (a Dirichlet condition, a voltage
    // terminal, the axis of an axisymmetric magnetic run) or a Robin boundary
    // with a positive coefficient. Without one the operator is singular on that
    // piece: its solution is fixed only up to a constant, and a load with a net
    // flux into it has no solution at all -- yet a direct factorization would
    // return a field anyway. Call after BuildEssentialBoundaryMarker().
    void RequireReferencePotential() const;

    // ---- Shared helpers for derived solvers ---------------------------------

    StatusReporter& Reporter() const {
        return StatusReporter::Global();
    }

    // The coordinate model of this run: measure, scalar integrators, required
    // mesh dimension and output units. Built on demand from `geometry` so the
    // enum stays the single piece of state.
    [[nodiscard]] GeometryModel Geometry() const { return GeometryModel(geometry); }

    // Adopt the configured coordinate model and reject a mesh of the wrong
    // dimension. Every solver's Setup() starts here, before anything reads
    // `geometry` or assembles on the mesh.
    void InitializeGeometry() {
        geometry = config.GeometryType;
        Geometry().VerifyMeshDimension(mesh);
    }

    // Solve an SPD system with preconditioned CG, used by the static solvers'
    // iterative path.
    //
    // solver_tolerance is the RELATIVE residual, measured as CG does in the
    // preconditioner's norm (the MQS FGMRES path measures the true residual).
    // The mfem::PCG convenience function previously used here square-roots its
    // tolerance argument, so a configured 1e-12 used to mean 1e-6 for these
    // solvers only.
    //
    void SolveSpdIteratively(const mfem::Operator& A, mfem::Solver& preconditioner,
                             const mfem::Vector& B, mfem::Vector& X) const;

    // Solve the time-harmonic (MQS) systems, which are not symmetric in their
    // real form, with flexible GMRES from the initial guess in X (zero but for
    // the essential values). The target is ||b - A x|| <= solver_tolerance
    // ||b_free||, b_free being b without the @p essential rows: those hold the
    // essential values themselves, are satisfied by the initial guess, and are
    // not a load.
    //
    // Being right-preconditioned, FGMRES monitors the unpreconditioned
    // residual; left-preconditioned GMRES monitors the preconditioned one,
    // which does not bound the port quantities (with massive ports at 10-100
    // kHz a coupling matrix solved to 1e-10 that way was off by up to 3e-4
    // against the direct solve). Within a cycle FGMRES tracks the residual
    // through its Arnoldi recurrence, which can fall below the true residual,
    // so the true residual is computed when it stops. If that is above the
    // target, the solve restarts from there, as long as each restart at least
    // halves it. A residual that no longer falls is the round-off floor of the
    // system, set by its conditioning, not by the tolerance: at the default
    // 1e-12 the 2D impedance test's true residual stalls at 4e-12 at 50 Hz and
    // at 6e-10 at 5 kHz, where the port rows are badly scaled, while the
    // recurrence reports 1e-12; its impedances still match the direct solve
    // to 1e-8. A stalled residual is accepted and reported, unless it is above
    // kMqsStallLimit, far above any round-off floor seen.
    //
    // FGMRES keeps two vectors per iteration (the basis and the preconditioned
    // basis), against GMRES's one, so it restarts after kMqsRestart iterations,
    // holding its memory to that of 200-iteration GMRES. Restarting at 100
    // rather than 200 also converged faster on the massive-port problems at
    // high frequency (the 2D impedance test at 5 kHz, 127 against 227
    // iterations; the two_loops example at 100 kHz, 160 against 262).
    static constexpr int kMqsRestart = 100;
    static constexpr double kMqsStallLimit = 1e-6;

    void SolveNonsymmetricIteratively(const mfem::Operator& A, mfem::Solver& preconditioner,
                                      const mfem::Vector& B, mfem::Vector& X,
                                      const mfem::Array<int>& essential) const;

    // Recovery-based (Zienkiewicz-Zhu) error indicator of a scalar solution
    // @p parts (one field, or a phasor's real and imaginary parts, combined
    // in quadrature), divided by sqrt of its field energy Et = 1/2 u^T K u to
    // make it a dimensionless RELATIVE error: the raw indicator has units of
    // sqrt(energy) and scales with the excitation, so folding raw indicators
    // across scenarios would let the most strongly driven one set the mesh.
    // @p make_stiffness is the solve's stiffness integrator (the flux is
    // grad u, the coefficient applied in the energy); K is assembled once per
    // mesh. A zero solution carries no energy and no meaningful relative
    // error, so its indicator is left unscaled.
    void EstimateRelativeZZError(const std::vector<mfem::GridFunction*>& parts,
                                 const std::function<mfem::BilinearFormIntegrator*()>& make_stiffness,
                                 mfem::Vector& errors);

    // Report a finished Krylov solve. Non-convergence is an error: the last
    // iterate of a solve that missed solver_tolerance is not a result, and
    // passing it on as one (with a warning that is easily missed) would put
    // an unconverged field into every output and coupling matrix.
    void RequireConverged(const mfem::IterativeSolver& solver, const std::string& name) const {
        ReportKrylov(name, solver.GetConverged(), solver.GetNumIterations(),
                     solver.GetFinalRelNorm());
    }

    void ReportKrylov(const std::string& name, bool converged, int iterations,
                      double relative_residual) const;

    // Eigen's simplicial LDL^T is fine for 2D meshes but its fill-in grows much
    // faster in 3D: measured on a P2 Laplacian, 14 s / 0.24 GB at 36k unknowns
    // and 346 s / 1.5 GB at 118k. Warn before a 3D factorization that size so
    // the run does not just appear to hang. The time-harmonic solvers pass
    // their own limit (see ComplexDirectSolver).
    void WarnOnLargeDirectSolve(int true_dofs, int large = 50000) const;

    // Marker (1/0 over domain attributes) for a set of element attribute ids.
    // Unlike the boundary variant, a domain attribute that the mesh does not
    // carry is a configuration error: silently dropping it yields an empty
    // integration region and downstream results of exactly zero, which are much
    // harder to diagnose than a failure here.
    mfem::Array<int> DomainMarkerFromAttrs(const std::vector<int>& attrs,
                                           const std::string& context) const;

    // Marker (1/0 over bdr attributes) for a named entity group.
    //
    // Resolved from the group's own AttributeIds, which the configuration
    // supplies and validation has already checked. The mesh's
    // bdr_attribute_sets are deliberately NOT consulted: only three of MFEM's
    // readers populate them (MFEM v1.3+, Gmsh $PhysicalNames, and Cubit), so
    // depending on them would make an otherwise portable configuration resolve
    // correctly for some mesh formats and silently not at all for others --
    // Netgen and VTK carry no names whatsoever. The config is the single
    // source of truth for name -> attribute binding.
    //
    // Ids the mesh does not carry are dropped rather than rejected: a boundary
    // group naming an attribute absent from this mesh contributes no DOFs,
    // which is the same outcome as omitting it. (Domain groups take the
    // opposite line -- see DomainMarkerFromAttrs.)
    mfem::Array<int> MarkerFromGroup(const std::string& group_name) const;

    // Region/material that claims a given domain attribute, or nullptr if none
    // does. The single source of truth for attribute -> material resolution.
    // Configuration validation rejects an attribute claimed by more than one
    // region, so the match here is unique and order-independent.
    const Material* MaterialForAttr(int attr) const;

    // Per-domain-attribute material property for the integrators. Attributes
    // that no region claims fall back to `default_value`; @p pick reads the
    // wanted quantity off whichever material a region assigns.
    std::unique_ptr<mfem::PWConstCoefficient> MaterialCoefficient(
        double default_value,
        const std::function<double(const Material&)>& pick) const {
        mfem::Vector values = MaterialVector(default_value, pick);
        return std::make_unique<mfem::PWConstCoefficient>(values);
    }

    // Per-domain-attribute vector laid out as PWConstCoefficient expects
    // (attribute a -> element a-1): `value` on the listed attributes, zero
    // elsewhere. The companion to MaterialCoefficient() for quantities that
    // come from the configuration rather than from a material.
    mfem::Vector AttributeVector(const std::vector<int>& attrs,
                                 double value) const;

    // The phasor Value * exp(j Phase) a scenario applies to a terminal, or 0
    // when the scenario does not mention it. Excitations are prescribed as a
    // list rather than a map, so this is the single place that resolves one
    // against a terminal name. Static solvers take its real part: their phase
    // is zero (validated).
    //
    // The amplitude is returned unscaled. For time-harmonic solvers it is a
    // PEAK amplitude by convention; this function performs no rms/peak
    // conversion. See Excitation in problem_config.hpp.
    static std::complex<double> ExcitationFor(const Scenario& sc,
                                              const std::string& terminal_name);

    // The fixed nonzero Dirichlet values. Solvers re-imprint these on every
    // scenario (the values are lifted into the RHS by FormLinearSystem, so they
    // must be re-applied after each solution reset), and this is the one place
    // that decides which conditions qualify. Terminal drives are not included:
    // they are scenario-dependent and are projected from the excitation list.
    template <typename Fn>
    void ForEachNonzeroDirichlet(Fn&& apply) {
        for (const auto& bc : boundary_conditions) {
            if (bc.IsNonzeroDirichlet()) {
                mfem::Array<int> marker(bc.Marker);
                apply(marker, bc.Condition.Value);
            }
        }
    }

    // Validate the mesh as (r,z) input, once, at setup. Delegates to
    // axisym::ValidateMesh, which ABORTS on a materially negative radius -- an
    // invalid coordinate system for any formulation, and one that would
    // otherwise produce a negative axisymmetric measure and a silently
    // indefinite operator.
    //
    // The scan is RETURNED rather than cached here: nothing in a general
    // physics solve consumes the radial extent or the axis tolerance. Both are
    // read only by the A_phi formulations, which hold onto the result
    // themselves. A planar run has no axis, hence no result.
    std::optional<axisym::AxisGeometry> ValidateAxisymmetricGeometry();

    // Whether this formulation assembles Robin conditions. A Robin term is part
    // of the operator, so a solver that ignored one would silently apply the
    // homogeneous Neumann condition instead; solvers without it reject Robin.
    virtual bool SupportsRobin() const { return false; }

    BoundaryConditionSet BuildBoundaryConditions() const;

    // The fixed natural boundary load: integral(g v) over every Neumann
    // boundary (g = prescribed outward flux) and every Robin boundary (g = the
    // Robin data value). This is boundary DATA, so coupling analyses omit it;
    // the Robin operator term (RobinCoeff * u, v) is assembled by the solver.
    mfem::Vector AssembleNaturalBoundaryLoad();

    // The ordered list of (name, scenario) solves for the active analysis, so
    // every solver flows both analysis types through ONE imprint/solve loop:
    //   - Field:          the prescribed scenarios, as-is.
    //   - CouplingMatrix: one synthesized scenario per terminal that drives that
    //                     terminal by a unit excitation (1 V or 1 A, per physics)
    //                     while every other terminal defaults to zero. This is the
    //                     column-by-column excitation used to assemble the coupling
    //                     (capacitance / inductance) matrix.
    // Terminal order follows config.Terminals (name-sorted), which the matrix
    // assembly and WriteCouplingMatrix() also use, keeping columns aligned.
    std::vector<std::pair<std::string, Scenario>> BuildSolveScenarios() const;

    // Fold the just-solved scenario's local error indicator into the running
    // combination, so one shared mesh is refined for all scenarios.
    // Solvers call this from RunOnCurrentMesh() right after each solve; outside an
    // AMR pass it is a no-op, which is what lets a single scenario loop serve both
    // the production solve and the error estimate (no duplicate solves).
    //
    // The fold is a running ROOT-MEAN-SQUARE over the scenarios seen so far:
    //     e_k = sqrt( (1/N) * sum_s eta_k(s)^2 ).
    // Updated incrementally from the previous average and the sample count, so
    // no per-scenario history is retained:
    //     e_k <- sqrt( (e_k^2 * n + eta_k^2) / (n + 1) ).
    //
    // RMS rather than the element-wise maximum: the max lets a single outlier
    // scenario dictate the refinement pattern, so elements are spent resolving a
    // feature only one excitation cares about. Averaging instead accounts for
    // error that is moderately important to MANY solves. This matters most for
    // coupling-matrix runs, where one scenario is synthesized per terminal and
    // every column is equally part of the answer.
    //
    // Each scenario's indicator is normalized by that scenario's field energy
    // (see EstimateCurrentSolutionError implementations) so the samples entering
    // this average are dimensionless relative errors and therefore comparable.
    void AccumulateScenarioError();

    // Integral of 1 over the given domain attributes, i.e. the measure of that
    // region in the MESH's own dimension: length in 1D, area in 2D, volume in 3D.
    //
    // Note what this deliberately is NOT. It uses a plain DomainLFIntegrator, so
    // it carries no axisymmetric r-weight: on an axisymmetric mesh it returns the
    // r-z cross-section area, not the revolved volume 2*pi*Int(r dA). That is the
    // right quantity for the current caller (a conductor cross-section feeding
    // J = I/A) but it is the wrong quantity for anything that wants a true
    // physical volume, which must integrate the axisymmetric weight instead.
    double CalculateRegionMeasure(const std::vector<int>& attribute_ids) const;

    // Shared per-scenario serialization: recover the field set ONCE, then fan it
    // out to whichever formats are enabled. The writer owns the format details;
    // solvers only declare WHAT to export via CollectExportFields().
    void SaveScenario(const std::string& scenario_name, const Scenario& scenario,
        const std::string& driven_terminal = {}, const std::vector<RegionLoss>& losses = {});

    // Unit label for an extracted coupling quantity: absolute for the
    // axisymmetric and 3D models, per unit length for planar. See
    // GeometryModel::IsPerUnitLength.
    [[nodiscard]] std::string CouplingUnitLabel(const std::string& si_unit) const {
        return "[" + CouplingUnits(si_unit) + "]";
    }

    [[nodiscard]] std::string CouplingUnits(const std::string& si_unit) const {
        return Geometry().CouplingUnits(si_unit);
    }

    std::optional<matrix_io::CouplingMatrixWriter> CreateCouplingWriter() const {
        return result_writer ? result_writer->CouplingWriter(TerminalNames()) : std::nullopt;
    }

    // Static analyses write one matrix; MQS uses the same writer for a sweep.
    void SaveCouplingMatrix(const mfem::DenseMatrix& M,
        const std::string& title, const std::string& quantity,
        const std::string& si_unit) const;

    void PrintCouplingMatrix(const mfem::DenseMatrix& M,
        const std::string& title) const;

    // ---- Base-class internals -----------------------------------------------
private:

    // Per-domain-attribute values behind MaterialCoefficient(), indexed as
    // PWConstCoefficient expects (attribute a -> element a-1). Attributes that
    // no region claims keep `default_value`. Size = mesh.attributes.Max().
    mfem::Vector MaterialVector(double default_value,
        const std::function<double(const Material&)>& pick) const;

    // Terminal names in config (name-sorted) order. This is the same order
    // BuildSolveScenarios() drives the coupling columns in, so it labels the
    // rows/columns of any coupling matrix consistently.
    std::vector<std::string> TerminalNames() const;

    const char* PeakFieldLabel() const {
        return config.PhysicsType == PhysicsType::Electrostatics ? "|E|" : "|B|";
    }

    void RunAdaptive();
};
