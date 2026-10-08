// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "physics_solver.hpp"

void PhysicsSolver::Run() {
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

void PhysicsSolver::SaveAnalysis() {
    if (!result_writer) return;
    SaveAnalysisResults();
    result_writer.reset();
}

void PhysicsSolver::RequireReferencePotential() const {
    std::vector<int> root(mesh.GetNE());
    for (int e = 0; e < mesh.GetNE(); ++e) { root[e] = e; }
    auto find = [&](int e) {
        while (root[e] != e) { e = root[e] = root[root[e]]; }
        return e;
    };
    for (int f = 0; f < mesh.GetNumFaces(); ++f) {
        int e1, e2;
        mesh.GetFaceElements(f, &e1, &e2);
        if (e1 >= 0 && e2 >= 0) { root[find(e1)] = find(e2); }
    }

    auto fixes_potential = [&](int attribute) {
        if (attribute <= ess_bdr.Size() && ess_bdr[attribute - 1]) { return true; }
        for (const auto& bc : boundary_conditions) {
            if (bc.IsRobin() && bc.Condition.RobinCoeff > 0.0
                && attribute <= bc.Marker.Size() && bc.Marker[attribute - 1]) {
                return true;
            }
        }
        return false;
    };
    std::vector<bool> fixed(mesh.GetNE(), false);
    for (int be = 0; be < mesh.GetNBE(); ++be) {
        if (!fixes_potential(mesh.GetBdrAttribute(be))) { continue; }
        int e1, e2;
        mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(be), &e1, &e2);
        fixed[find(e1)] = true;
    }
    for (int e = 0; e < mesh.GetNE(); ++e) {
        MFEM_VERIFY(fixed[find(e)],
            "Nothing fixes the potential on the part of the mesh containing element "
            << e << " (domain attribute " << mesh.GetAttribute(e) << "): no 'dirichlet' "
            "boundary, terminal, symmetry axis or Robin boundary with a positive "
            "robin_coefficient touches it, so its solution is determined only up to a "
            "constant. Add a 'dirichlet' boundary (or terminal) on it.");
    }
}

void PhysicsSolver::SolveSpdIteratively(const mfem::Operator& A, mfem::Solver& preconditioner,
                         const mfem::Vector& B, mfem::Vector& X) const {
    mfem::CGSolver cg;
    cg.SetOperator(A);
    cg.SetPreconditioner(preconditioner);
    cg.SetRelTol(config.SolverTolerance);
    cg.SetAbsTol(0.0);
    cg.SetMaxIter(config.SolverMaxIter);
    cg.SetPrintLevel(Reporter().SolverPrintLevel(config.SolverPrintLevel));
    cg.Mult(B, X);
    RequireConverged(cg, "CG");
}

void PhysicsSolver::SolveNonsymmetricIteratively(const mfem::Operator& A, mfem::Solver& preconditioner,
                                  const mfem::Vector& B, mfem::Vector& X,
                                  const mfem::Array<int>& essential) const {
    mfem::Vector b_free(B);
    for (const int i : essential) { b_free(i) = 0.0; }
    const double b_norm = b_free.Norml2();
    if (b_norm == 0.0) {
        X = B;  // the essential values, and zero elsewhere
        return;
    }
    const double target = config.SolverTolerance * b_norm;
    mfem::FGMRESSolver fgmres;
    fgmres.SetOperator(A);
    fgmres.SetPreconditioner(preconditioner);
    fgmres.SetKDim(kMqsRestart);
    fgmres.SetRelTol(0.0);
    fgmres.SetAbsTol(target);
    fgmres.SetPrintLevel(Reporter().SolverPrintLevel(config.SolverPrintLevel));
    mfem::Vector r(B.Size());
    int iterations = 0;
    double residual = std::numeric_limits<double>::infinity();
    bool stalled = false;
    while (iterations < config.SolverMaxIter) {
        fgmres.SetMaxIter(config.SolverMaxIter - iterations);
        fgmres.Mult(B, X);  // continues from X
        iterations += fgmres.GetNumIterations();
        A.Mult(X, r);
        subtract(B, r, r);
        const double previous = residual;
        residual = r.Norml2();
        if (residual <= target || !fgmres.GetConverged()) { break; }
        if (residual > 0.5 * previous) {
            stalled = true;
            break;
        }
    }
    const double relative = residual / b_norm;
    const bool converged = residual <= target ||
        (stalled && relative <= kMqsStallLimit);
    ReportKrylov("FGMRES", converged, iterations, relative);
    if (stalled && residual > target) {
        std::ostringstream msg;
        msg << std::scientific << std::setprecision(3)
            << "FGMRES: the true relative residual stalls at " << relative
            << " after " << iterations << " iterations, above solver_tolerance "
            << config.SolverTolerance << ": the round-off floor of this system.";
        Reporter().Diagnostic(msg.str());
    }
}

void PhysicsSolver::EstimateRelativeZZError(const std::vector<mfem::GridFunction*>& parts,
                             const std::function<mfem::BilinearFormIntegrator*()>& make_stiffness,
                             mfem::Vector& errors) {
    const std::unique_ptr<mfem::BilinearFormIntegrator> flux_integ(make_stiffness());
    mfem::FiniteElementSpace flux_fes(&mesh, fec.get(), mesh.SpaceDimension());
    if (!energy_form || energy_fes != fespace.get() ||
        energy_sequence != mesh.GetSequence()) {
        energy_form = std::make_unique<mfem::BilinearForm>(fespace.get());
        energy_form->AddDomainIntegrator(make_stiffness());
        energy_form->Assemble();
        energy_form->Finalize();
        energy_fes = fespace.get();
        energy_sequence = mesh.GetSequence();
    }
    errors.SetSize(mesh.GetNE());
    errors = 0.0;
    double energy = 0.0;
    for (mfem::GridFunction* u : parts) {
        mfem::ZienkiewiczZhuEstimator estimator(*flux_integ, *u, flux_fes);
        estimator.SetWithCoeff(false);  // the flux is grad u; the energy applies the coefficient
        estimator.SetFluxAveraging(1);  // do not average across attribute interfaces
        const mfem::Vector& local = estimator.GetLocalErrors();
        for (int k = 0; k < errors.Size(); ++k) { errors(k) = std::hypot(errors(k), local(k)); }
        // Clamp the round-off of a nearly null solution's quadratic form.
        energy += std::max(0.0, 0.5 * energy_form->InnerProduct(*u, *u));
    }
    if (energy > 0.0) { errors /= std::sqrt(energy); }
}

void PhysicsSolver::ReportKrylov(const std::string& name, bool converged, int iterations,
                  double relative_residual) const {
    std::ostringstream msg;
    msg << std::scientific << std::setprecision(3);
    if (!converged) {
        msg << name << " did not converge: relative residual " << relative_residual
            << " after " << iterations << " iterations, above "
               "solver_tolerance " << config.SolverTolerance << ". Raise "
               "solver_max_iter, loosen solver_tolerance, or use the direct solver.";
        throw std::runtime_error(msg.str());
    }
    msg << name << " converged in " << iterations
        << " iterations (relative residual " << relative_residual << ").";
    Reporter().Diagnostic(msg.str());
}

void PhysicsSolver::WarnOnLargeDirectSolve(int true_dofs, int large) const {
    if (config.LinearSolver != LinearSolverType::Direct) return;
    if (geometry != GeometryType::Cartesian3D || true_dofs <= large) return;
    Reporter().Warning("Direct factorization of a 3D system with " +
        std::to_string(true_dofs) + " unknowns may take many minutes and "
        "gigabytes of memory. Set simulation.linear_solver to 'iterative' "
        "(algebraic multigrid), the default for geometry_type '3d'.");
}

mfem::Array<int> PhysicsSolver::DomainMarkerFromAttrs(const std::vector<int>& attrs,
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

mfem::Array<int> PhysicsSolver::MarkerFromGroup(const std::string& group_name) const {
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

const Material* PhysicsSolver::MaterialForAttr(int attr) const {
    for (const auto& region : config.Regions) {
        const EntityGroup& group = config.EntityGroups.at(region.EntityGroupName);
        if (std::find(group.AttributeIds.begin(), group.AttributeIds.end(), attr)
            != group.AttributeIds.end())
            return &config.Materials.at(region.MaterialName);
    }
    return nullptr;
}

mfem::Vector PhysicsSolver::AttributeVector(const std::vector<int>& attrs,
                             double value) const {
    mfem::Vector v(mesh.attributes.Max());
    v = 0.0;
    for (int attr : attrs) {
        if (attr > 0 && attr <= v.Size()) { v[attr - 1] = value; }
    }
    return v;
}

 std::complex<double> PhysicsSolver::ExcitationFor(const Scenario& sc,
                                          const std::string& terminal_name) {
    std::complex<double> value = 0.0;
    for (const auto& exc : sc.Excitations) {
        if (exc.TerminalName == terminal_name) {
            // Not std::polar, which needs a nonnegative amplitude.
            const double radians = exc.Phase * Constants::TWO_PI / 360.0;
            value = exc.Value * std::complex<double>(std::cos(radians), std::sin(radians));
        }
    }
    return value;
}

std::optional<axisym::AxisGeometry> PhysicsSolver::ValidateAxisymmetricGeometry()
{
    if (geometry != GeometryType::Axisymmetric) { return std::nullopt; }

    const axisym::AxisGeometry info = axisym::ValidateMesh(mesh);

    std::ostringstream msg;
    msg << std::setprecision(6)
        << "Axisymmetric mesh radial extent: r in ["
        << info.min_r << ", " << info.max_r << "].";
    Reporter().Diagnostic(msg.str());

    return info;
}

BoundaryConditionSet PhysicsSolver::BuildBoundaryConditions() const {
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

mfem::Vector PhysicsSolver::AssembleNaturalBoundaryLoad() {
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

std::vector<std::pair<std::string, Scenario>> PhysicsSolver::BuildSolveScenarios() const {
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

void PhysicsSolver::AccumulateScenarioError() {
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

double PhysicsSolver::CalculateRegionMeasure(const std::vector<int>& attribute_ids) const {
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

void PhysicsSolver::SaveScenario(const std::string& scenario_name, const Scenario& scenario,
    const std::string& driven_terminal, const std::vector<RegionLoss>& losses) {
    if (!result_writer || !result_writer->WantsFields()) return;
    result_writer->WriteScenario(scenario_name, scenario, CollectExportFields(), driven_terminal,
        losses);
}

void PhysicsSolver::SaveCouplingMatrix(const mfem::DenseMatrix& M,
    const std::string& title, const std::string& quantity,
    const std::string& si_unit) const {
    auto writer = CreateCouplingWriter();
    if (writer) writer->WriteMatrix(quantity, M, CouplingUnits(si_unit));
    PrintCouplingMatrix(M, title);
}

void PhysicsSolver::PrintCouplingMatrix(const mfem::DenseMatrix& M,
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

mfem::Vector PhysicsSolver::MaterialVector(double default_value,
    const std::function<double(const Material&)>& pick) const {
    const int n = mesh.attributes.Max();
    mfem::Vector v(n);
    v = default_value;
    for (int a = 1; a <= n; ++a)
        if (const Material* mat = MaterialForAttr(a))
            v[a - 1] = pick(*mat);
    return v;
}

std::vector<std::string> PhysicsSolver::TerminalNames() const {
    std::vector<std::string> names;
    names.reserve(config.Terminals.size());
    for (const auto& [name, term] : config.Terminals) names.push_back(name);
    return names;
}

void PhysicsSolver::RunAdaptive() {
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
