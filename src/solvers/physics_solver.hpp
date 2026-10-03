// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include "mfem.hpp"
#include "../core/problem_config.hpp"
#include "../io/field_export.hpp"
#include "../io/matrix_writer.hpp"
#include "../io/coupling_matrix_writer.hpp"
#include "../io/result_writer.hpp"
#include "../io/status_reporter.hpp"
#include "amr_support.hpp"
#include "../axisym/axisymmetric_mesh_validation.hpp"
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
    void Run() {
        result_writer = std::make_unique<ResultWriter>(mesh, config);
        if (config.Amr.Enabled) {
            RunAdaptive();
        }
        else {
            amr_history.clear();
            result_writer->BeginMesh();
            RunOnCurrentMesh();
        }
    }
    void SaveAnalysis() {
        if (!result_writer) return;
        SaveAnalysisResults();
        result_writer.reset();
    }

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
        ess_bdr = boundary_conditions.DirichletMarker(mesh.bdr_attributes.Max());
    }

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
    // solver_tolerance is the RELATIVE residual ||b - A x|| / ||b||, the same
    // meaning it has in the MQS GMRES path. (The mfem::PCG convenience function
    // previously used here squares-roots its tolerance argument, so a
    // configured 1e-12 used to mean 1e-6 for these solvers only.)
    //
    // Non-convergence is reported rather than silent: mfem::PCG returned the
    // last iterate without comment, which reads exactly like a converged run.
    void SolveSpdIteratively(const mfem::Operator& A, mfem::Solver& preconditioner,
                             const mfem::Vector& B, mfem::Vector& X) const {
        mfem::CGSolver cg;
        cg.SetOperator(A);
        cg.SetPreconditioner(preconditioner);
        cg.SetRelTol(config.SolverTolerance);
        cg.SetAbsTol(0.0);
        cg.SetMaxIter(config.SolverMaxIter);
        cg.SetPrintLevel(Reporter().SolverPrintLevel(config.SolverPrintLevel));
        cg.Mult(B, X);

        std::ostringstream msg;
        msg << std::scientific << std::setprecision(3);
        if (cg.GetConverged()) {
            msg << "CG converged in " << cg.GetNumIterations()
                << " iterations (relative residual " << cg.GetFinalRelNorm() << ").";
            Reporter().Diagnostic(msg.str());
        }
        else {
            msg << "CG did not converge: relative residual " << cg.GetFinalRelNorm()
                << " after " << cg.GetNumIterations() << " iterations, above "
                   "solver_tolerance " << config.SolverTolerance << ". Raise "
                   "solver_max_iter, loosen solver_tolerance, or use the direct "
                   "solver; results may be inaccurate.";
            Reporter().Warning(msg.str());
        }
    }

    // Eigen's simplicial LDL^T is fine for 2D meshes but its fill-in grows much
    // faster in 3D: measured on a P2 Laplacian, 14 s / 0.24 GB at 36k unknowns
    // and 346 s / 1.5 GB at 118k. Warn before a 3D factorization that size so
    // the run does not just appear to hang.
    void WarnOnLargeDirectSolve(int true_dofs) const {
        constexpr int kLarge3DDirectDofs = 50000;
        if (config.LinearSolver != LinearSolverType::Direct) return;
        if (geometry != GeometryType::Cartesian3D || true_dofs <= kLarge3DDirectDofs) return;
        Reporter().Warning("Direct factorization of a 3D system with " +
            std::to_string(true_dofs) + " unknowns may take many minutes and "
            "gigabytes of memory. Set simulation.linear_solver to 'iterative' "
            "(algebraic multigrid), the default for geometry_type '3d'.");
    }

    // Marker (1/0 over domain attributes) for a set of element attribute ids.
    // Unlike the boundary variant, a domain attribute that the mesh does not
    // carry is a configuration error: silently dropping it yields an empty
    // integration region and downstream results of exactly zero, which are much
    // harder to diagnose than a failure here.
    mfem::Array<int> DomainMarkerFromAttrs(const std::vector<int>& attrs,
                                           const std::string& context) const {
        MFEM_VERIFY(!attrs.empty(),
            "No domain attribute ids are associated with " + context + ".");
        const int max_attr = mesh.attributes.Max();
        mfem::Array<int> ids;
        ids.Reserve(static_cast<int>(attrs.size()));
        for (int a : attrs) {
            const bool is_bdr_attr = mesh.bdr_attributes.Find(a) >= 0;
            MFEM_VERIFY(a > 0 && a <= max_attr && mesh.attributes.Find(a) >= 0,
                "Domain attribute " + std::to_string(a) + " referenced by " +
                context + " does not exist in the mesh (mesh domain attributes "
                "run up to " + std::to_string(max_attr) + ")." +
                (is_bdr_attr ? " It is a boundary attribute of this mesh; the "
                               "entity group most likely names a curve physical "
                               "group instead of the surface one." : ""));
            ids.Append(a);
        }
        return mfem::AttributeSets::AttrToMarker(max_attr, ids);
    }

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
    mfem::Array<int> MarkerFromGroup(const std::string& group_name) const {
        const auto entry = config.EntityGroups.find(group_name);
        MFEM_VERIFY(entry != config.EntityGroups.end(),
            "Unknown entity group '" + group_name + "'.");
        MFEM_VERIFY(entry->second.IsBoundary(mesh.Dimension()),
            "Entity group '" + group_name + "' declares dim " +
            std::to_string(entry->second.Dim) + ", but is used here as a "
            "boundary, which for this " + std::to_string(mesh.Dimension()) +
            "D mesh requires dim " + std::to_string(mesh.Dimension() - 1) + ".");

        const int max_bdr_attr = mesh.bdr_attributes.Size() > 0
            ? mesh.bdr_attributes.Max() : 0;
        mfem::Array<int> marker(max_bdr_attr);
        marker = 0;
        for (int a : entry->second.AttributeIds) {
            if (a > 0 && a <= max_bdr_attr && mesh.bdr_attributes.Find(a) >= 0) {
                marker[a - 1] = 1;
            }
        }
        return marker;
    }

    // Region/material that claims a given domain attribute, or nullptr if none
    // does. The single source of truth for attribute -> material resolution.
    // Configuration validation rejects an attribute claimed by more than one
    // region, so the match here is unique and order-independent.
    const Material* MaterialForAttr(int attr) const {
        for (const auto& region : config.Regions) {
            const EntityGroup& group = config.EntityGroups.at(region.EntityGroupName);
            if (std::find(group.AttributeIds.begin(), group.AttributeIds.end(), attr)
                != group.AttributeIds.end())
                return &config.Materials.at(region.MaterialName);
        }
        return nullptr;
    }

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
                                 double value) const {
        mfem::Vector v(mesh.attributes.Max());
        v = 0.0;
        for (int attr : attrs) {
            if (attr > 0 && attr <= v.Size()) { v[attr - 1] = value; }
        }
        return v;
    }

    // The drive a scenario applies to a terminal, or 0.0 when the scenario does
    // not mention it. Excitations are prescribed as a list rather than a map, so
    // this is the single place that resolves one against a terminal name.
    //
    // The value is returned unscaled. For time-harmonic solvers it is a PEAK
    // (amplitude) phasor by convention; this function is convention-agnostic
    // and performs no rms/peak conversion. See Excitation in problem_config.hpp.
    static double ExcitationFor(const Scenario& sc,
                                const std::string& terminal_name) {
        double value = 0.0;
        for (const auto& exc : sc.Excitations) {
            if (exc.TerminalName == terminal_name) { value = exc.Value; }
        }
        return value;
    }

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
    // themselves. Returns a default-constructed value for planar runs.
    axisym::AxisGeometry ValidateAxisymmetricGeometry()
    {
        if (geometry != GeometryType::Axisymmetric) { return {}; }

        const axisym::AxisGeometry info = axisym::ValidateMesh(mesh);

        std::ostringstream msg;
        msg << std::setprecision(6)
            << "Axisymmetric mesh radial extent: r in ["
            << info.min_r << ", " << info.max_r << "].";
        Reporter().Diagnostic(msg.str());

        return info;
    }

    // Whether this formulation assembles Robin conditions. A Robin term is part
    // of the operator, so a solver that ignored one would silently apply the
    // homogeneous Neumann condition instead; solvers without it reject Robin.
    virtual bool SupportsRobin() const { return false; }

    BoundaryConditionSet BuildBoundaryConditions() const {
        BoundaryConditionSet bcs;
        for (const auto& bc : config.BoundaryConditions) {
            MFEM_VERIFY(bc.Type != BoundaryConditionType::Robin || SupportsRobin(),
                "Robin boundary conditions are not implemented for " +
                std::string(ToString(config.PhysicsType)) +
                ". Use Dirichlet or Neumann for boundary group '" +
                bc.EntityGroupName + "'.");
            bcs.Add(MarkerFromGroup(bc.EntityGroupName), bc);
        }
        return bcs;
    }

    // The fixed natural boundary load: integral(g v) over every Neumann
    // boundary (g = prescribed outward flux) and every Robin boundary (g = the
    // Robin data value). This is boundary DATA, so coupling analyses omit it;
    // the Robin operator term (RobinCoeff * u, v) is assembled by the solver.
    mfem::Vector AssembleNaturalBoundaryLoad() {
        mfem::LinearForm load(fespace.get());
        std::vector<std::unique_ptr<mfem::ConstantCoefficient>> coefficients;
        // MFEM binds the marker by non-const reference and keeps the pointer, so
        // these copies must stay alive until Assemble() has run.
        std::vector<std::unique_ptr<mfem::Array<int>>> markers;

        for (const auto& bc : boundary_conditions) {
            if (!(bc.IsNeumann() || bc.IsRobin()) || bc.Condition.Value == 0.0) continue;
            coefficients.push_back(
                std::make_unique<mfem::ConstantCoefficient>(bc.Condition.Value));
            markers.push_back(std::make_unique<mfem::Array<int>>(bc.Marker));
            load.AddBoundaryIntegrator(
                Geometry().NewBoundaryLFIntegrator(*coefficients.back()),
                *markers.back());
        }

        load.Assemble();
        return mfem::Vector(load);
    }

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
    std::vector<std::pair<std::string, Scenario>> BuildSolveScenarios() const {
        std::vector<std::pair<std::string, Scenario>> out;
        if (config.AnalysisType == AnalysisType::CouplingMatrix) {
            out.reserve(config.Terminals.size());
            for (const auto& [term_name, term] : config.Terminals) {
                Scenario sc;
                sc.Excitations.push_back({ term_name, 1.0 });
                out.push_back({ "CouplingMatrix_" + term_name, std::move(sc) });
            }
        }
        else {
            out.reserve(config.Scenarios.size());
            for (const auto& [sc_name, sc] : config.Scenarios)
                out.push_back({ sc_name, sc });
        }
        return out;
    }

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
    void AccumulateScenarioError() {
        if (!amr_errors) return;

        const int ne = mesh.GetNE();
        mfem::Vector current;
        EstimateCurrentSolutionError(current);
        MFEM_VERIFY(current.Size() == ne,
            "AMR estimator returned the wrong number of element errors.");

        const double n = static_cast<double>(amr_error_samples);
        for (int element = 0; element < ne; ++element) {
            const double prev = (*amr_errors)(element);
            const double eta = current(element);
            (*amr_errors)(element) =
                std::sqrt((prev * prev * n + eta * eta) / (n + 1.0));
        }
        ++amr_error_samples;
    }

    // Integral of 1 over the given domain attributes, i.e. the measure of that
    // region in the MESH's own dimension: length in 1D, area in 2D, volume in 3D.
    //
    // Note what this deliberately is NOT. It uses a plain DomainLFIntegrator, so
    // it carries no axisymmetric r-weight: on an axisymmetric mesh it returns the
    // r-z cross-section area, not the revolved volume 2*pi*Int(r dA). That is the
    // right quantity for the current caller (a conductor cross-section feeding
    // J = I/A) but it is the wrong quantity for anything that wants a true
    // physical volume, which must integrate the axisymmetric weight instead.
    double CalculateRegionMeasure(const std::vector<int>& attribute_ids) const {
        // AddDomainIntegrator binds the marker by non-const reference, so it
        // cannot be const here.
        mfem::Array<int> marker =
            DomainMarkerFromAttrs(attribute_ids, "a region measure calculation");
        mfem::LinearForm measure_form(fespace.get());
        mfem::ConstantCoefficient one(1.0);

        measure_form.AddDomainIntegrator(new mfem::DomainLFIntegrator(one), marker);
        measure_form.Assemble();

        return measure_form.Sum();
    }

    // Shared per-scenario serialization: recover the field set ONCE, then fan it
    // out to whichever formats are enabled. The writer owns the format details;
    // solvers only declare WHAT to export via CollectExportFields().
    void SaveScenario(const std::string& scenario_name, const Scenario& scenario,
        const std::string& driven_terminal = {}, const std::vector<RegionLoss>& losses = {}) {
        if (!result_writer || !result_writer->WantsFields()) return;
        result_writer->WriteScenario(scenario_name, scenario, CollectExportFields(), driven_terminal,
            losses);
    }

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
        const std::string& si_unit) const {
        auto writer = CreateCouplingWriter();
        if (writer) writer->WriteMatrix(quantity, M, CouplingUnits(si_unit));
        PrintCouplingMatrix(M, title);
    }

    void PrintCouplingMatrix(const mfem::DenseMatrix& M,
        const std::string& title) const {
        matrix_io::MatrixWriter writer(title, TerminalNames());
        if (Reporter().IsMachineReadable()) {
            std::ostringstream table;
            writer.PrintConsole(M, table);
            Reporter().Status(table.str());
        }
        else {
            writer.PrintConsole(M);
        }
    }

    // ---- Base-class internals -----------------------------------------------
private:

    // Per-domain-attribute values behind MaterialCoefficient(), indexed as
    // PWConstCoefficient expects (attribute a -> element a-1). Attributes that
    // no region claims keep `default_value`. Size = mesh.attributes.Max().
    mfem::Vector MaterialVector(double default_value,
        const std::function<double(const Material&)>& pick) const {
        const int n = mesh.attributes.Max();
        mfem::Vector v(n);
        v = default_value;
        for (int a = 1; a <= n; ++a)
            if (const Material* mat = MaterialForAttr(a))
                v[a - 1] = pick(*mat);
        return v;
    }

    // Terminal names in config (name-sorted) order. This is the same order
    // BuildSolveScenarios() drives the coupling columns in, so it labels the
    // rows/columns of any coupling matrix consistently.
    std::vector<std::string> TerminalNames() const {
        std::vector<std::string> names;
        names.reserve(config.Terminals.size());
        for (const auto& [name, term] : config.Terminals) names.push_back(name);
        return names;
    }

    const char* PeakFieldLabel() const {
        return config.PhysicsType == PhysicsType::Electrostatics ? "|E|" : "|B|";
    }

    void RunAdaptive() {
        const AmrSettings& settings = config.Amr;
        amr_history.clear();

        const int max_iterations = std::max(1, settings.MaxIterations);
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            const long true_dofs = fespace->GetTrueVSize();

            // ONE scenario loop per AMR iteration: RunOnCurrentMesh() solves and
            // post-processes every scenario, and folds each solution's error
            // indicator into `errors` as it goes. The last iteration's results are
            // therefore already the converged-mesh results, so no extra solve pass
            // is needed after the loop.
            mfem::Vector errors(mesh.GetNE());
            errors = 0.0;
            amr_errors = &errors;
            amr_error_samples = 0;
            result_writer->BeginMesh();
            RunOnCurrentMesh();
            amr_errors = nullptr;

            double sum_squared = 0.0;
            for (int element = 0; element < errors.Size(); ++element) {
                sum_squared += errors(element) * errors(element);
            }
            const double global_error = std::sqrt(sum_squared);
            const double peak_field = ComputePeakFieldMagnitude();
            amr_history.push_back({ true_dofs, global_error, peak_field });

            std::ostringstream diagnostic;
            diagnostic << "AMR iteration " << iteration
                << ": elements=" << mesh.GetNE()
                << ", true_dofs=" << true_dofs
                << ", global_error=" << std::scientific << std::setprecision(6)
                << global_error
                << ", peak" << PeakFieldLabel() << "=" << peak_field;
            Reporter().Diagnostic(diagnostic.str());

            if (settings.ErrorTolerance > 0.0 &&
                global_error < settings.ErrorTolerance) {
                Reporter().Diagnostic("AMR: global error below tolerance. Stop.");
                break;
            }
            if (settings.MaxDofs > 0 && true_dofs > settings.MaxDofs) {
                Reporter().Diagnostic("AMR: reached the maximum number of DOFs. Stop.");
                break;
            }
            if (iteration + 1 >= max_iterations) {
                Reporter().Diagnostic("AMR: reached the maximum number of iterations. Stop.");
                break;
            }

            mfem::Array<int> marked;
            amr::MarkElementsDorfler(errors, settings.ErrorFraction, marked);
            if (marked.Size() == 0) {
                Reporter().Diagnostic("AMR: no elements marked for refinement. Stop.");
                break;
            }

            amr::RefineConforming(mesh, marked);
            BuildOperators();
        }
    }
};
