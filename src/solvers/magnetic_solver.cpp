// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "magnetic_solver.hpp"

void MagneticSolverBase::WarnOnSlowComplexDirectSolve() const {
	if (ComplexDirectSolver::kUsesStrumpack || config.LinearSolver != LinearSolverType::Direct) {
		return;
	}
	Reporter().Warning("This build has no STRUMPACK (CMake option USE_STRUMPACK), so the "
		"direct solve factors the complex system with Eigen's SparseLU, which is much "
		"slower and needs much more memory in 3D.");
}

void MagneticSolverBase::BuildConductivity() {
	sigma_coeff = MaterialCoefficient(0.0, Conductivity);
	for (const auto& [name, term] : config.Terminals) {
		if (term.Conductor != ConductorType::Stranded) continue;
		bool conducts = false;
		for (int attr : config.EntityGroups.at(term.EntityGroupName).AttributeIds) {
			if (attr < 1 || attr > sigma_coeff->GetNConst()) continue;
			conducts |= (*sigma_coeff)(attr) > 0.0;
			(*sigma_coeff)(attr) = 0.0;
		}
		if (conducts) {
			Reporter().Diagnostic("Stranded conductor '" + name + "': its material "
				"conductivity is the wire's and does not enter the field solve, "
				"which imposes the winding current without eddy currents.");
		}
	}
}

void MagneticSolverBase::ValidateMassiveConductivity(const std::string& name,
								 const std::vector<int>& attributes) const {
	for (int attr : attributes) {
		const Material* material = MaterialForAttr(attr);
		MFEM_VERIFY(material != nullptr,
			"Massive conductor '" + name + "' contains domain attribute " +
			std::to_string(attr) + " without an assigned material.");
		MFEM_VERIFY(material->Conductivity > 0.0,
			"Massive conductor '" + name + "' contains domain attribute " +
			std::to_string(attr) + " with non-positive conductivity " +
			std::to_string(material->Conductivity) +
			". Assign a material with a positive 'sigma' or make it a "
			"stranded conductor.");
	}
}

std::vector<MagneticSolverBase::RegionLoss> MagneticSolverBase::IntegrateRegionLosses(mfem::Coefficient& density) const {
	std::vector<RegionLoss> losses;
	for (const auto& [name, attrs] : ConductingGroups()) {
		losses.push_back({ name, IntegrateOverAttributes(density, attrs) });
	}
	return losses;
}

std::map<std::string, std::set<int>> MagneticSolverBase::ConductingGroups() const {
	std::map<int, std::string> owner;
	for (const Region& region : config.Regions) {
		const EntityGroup& group = config.EntityGroups.at(region.EntityGroupName);
		for (int attr : group.AttributeIds) { owner[attr] = region.EntityGroupName; }
	}
	for (const auto& [name, term] : config.Terminals) {
		if (term.Conductor != ConductorType::Massive) continue;
		const EntityGroup& group = config.EntityGroups.at(term.EntityGroupName);
		for (int attr : group.AttributeIds) { owner[attr] = name; }
	}

	std::map<std::string, std::set<int>> groups;
	for (int attr = 1; attr <= mesh.attributes.Max(); ++attr) {
		if (attr > sigma_coeff->GetNConst() || (*sigma_coeff)(attr) <= 0.0) continue;
		const auto named = owner.find(attr);
		groups[named != owner.end() ? named->second
								   : "attribute " + std::to_string(attr)].insert(attr);
	}
	return groups;
}

void MagneticSolverBase::ReportRegionLosses(const std::vector<RegionLoss>& losses) const {
	if (losses.empty()) { return; }
	std::ostringstream out;
	out << "Time-averaged Joule loss " << CouplingUnitLabel("W")
		<< " (peak-phasor convention):\n";
	out << std::scientific << std::setprecision(6);
	double total = 0.0;
	for (const RegionLoss& loss : losses) {
		out << "  " << loss.Name << ": " << loss.Power << "\n";
		total += loss.Power;
	}
	out << "  total: " << total;
	Reporter().Status(out.str());
}

void MagneticSolverBase::WriteImpedanceSeries(const std::vector<ImpedancePoint>& points) const {
	if (points.empty()) {
		Reporter().Warning("WriteCouplingMatrix: MQS coupling matrices not computed.");
		return;
	}
	std::vector<double> frequencies;
	std::vector<const mfem::DenseMatrix*> resistance, inductance;
	for (const ImpedancePoint& point : points) {
		frequencies.push_back(point.Frequency);
		resistance.push_back(&point.Resistance);
		inductance.push_back(&point.Inductance);
	}
	if (auto writer = CreateCouplingWriter()) {
		writer->WriteFrequencies(frequencies);
		writer->WriteMatrixSeries("Inductance", inductance, CouplingUnits("H"));
		writer->WriteMatrixSeries("Resistance", resistance, CouplingUnits("Ohm"));
	}
	for (const ImpedancePoint& point : points) {
		std::ostringstream at;
		at << " at " << std::setprecision(std::numeric_limits<double>::max_digits10)
		   << point.Frequency << " Hz ";
		PrintCouplingMatrix(point.Inductance,
			"Inductance Matrix" + at.str() + CouplingUnitLabel("H"));
		PrintCouplingMatrix(point.Resistance,
			"Resistance Matrix" + at.str() + CouplingUnitLabel("Ohm"));
	}
}

double MagneticSolverBase::IntegrateOverAttributes(mfem::Coefficient& density,
							   const std::set<int>& attrs) const {
	double total = 0.0;
	mfem::Vector pos;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		if (!attrs.count(mesh.GetAttribute(e))) { continue; }
		mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
		const mfem::FiniteElement& fe = *fespace->GetFE(e);
		const int order = 2 * fe.GetOrder() + T.OrderW() + 2;
		const mfem::IntegrationRule& ir = geometry == GeometryType::Axisymmetric
			? axisym::RadialRule(fe.GetGeomType(), order, T, axis_geometry->tolerance)
			: mfem::IntRules.Get(fe.GetGeomType(), order);
		for (int q = 0; q < ir.GetNPoints(); ++q) {
			const mfem::IntegrationPoint& ip = ir.IntPoint(q);
			T.SetIntPoint(&ip);
			T.Transform(ip, pos);
			total += density.Eval(T, ip) * ip.weight * T.Weight() * Geometry().Measure(pos);
		}
	}
	return total;
}

void MagneticSolver::InitializeMagneticGeometry() {
	MFEM_VERIFY(config.GeometryType != GeometryType::Cartesian3D,
		"This " + std::string(ToString(config.PhysicsType)) + " solver is "
		"the 2D scalar-potential formulation; geometry_type '3d' needs the "
		"vector (H(curl)) formulation of the 3D solver classes.");
	InitializeGeometry();
}

void MagneticSolver::ValidateMagneticAxisymmetricGeometry() {
	axis_geometry = ValidateAxisymmetricGeometry();
	if (!axis_geometry) { return; }

	Reporter().Diagnostic(
		axis_geometry->TouchesAxis()
			? "Axisymmetric domain touches the symmetry axis: "
			  "axis regularity A_phi = 0 will be enforced."
			: "Axisymmetric domain is annular: no axis condition required.");

	WarnOnUnderResolvedRadialQuadrature();
}

void MagneticSolver::WarnOnUnderResolvedRadialQuadrature() {
	int worst_element = -1;
	double worst_ratio = std::numeric_limits<double>::max();

	for (int e = 0; e < mesh.GetNE(); ++e) {
		const double ratio = axisym::RadialResolution(
			*mesh.GetElementTransformation(e), axis_geometry->tolerance);
		if (ratio < worst_ratio) {
			worst_ratio = ratio;
			worst_element = e;
		}
	}

	if (worst_element < 0 || worst_ratio >= axisym::kResolvedRadiusRatio) {
		return;
	}

	std::ostringstream msg;
	msg << std::setprecision(3)
		<< "Element " << worst_element << " has radial ratio " << worst_ratio
		<< ", below the ratio " << axisym::kResolvedRadiusRatio
		<< " at which the 1/r quadrature reaches its accuracy target. The "
		   "capped rule integrates such elements approximately; refine "
		   "radially near the axis, or avoid slivers touching it, if "
		   "near-axis accuracy matters.";
	Reporter().Warning(msg.str());
}

void MagneticSolver::BuildEssentialBoundaryMarker()  {
	PhysicsSolver::BuildEssentialBoundaryMarker();

	if (!axis_geometry) { return; }

	axis_boundary = axisym::FindAxisBoundaryMarker(mesh, *axis_geometry);

	MFEM_VERIFY(ess_bdr.Size() == axis_boundary.Size(),
		"Axis boundary marker does not match the mesh boundary attributes.");
	MergeMarker(ess_bdr, axis_boundary);
}

mfem::Array<int> MagneticSolver::AxisTrueDofs() const {
	mfem::Array<int> tdofs;
	if (!axis_geometry || !axis_geometry->TouchesAxis()) { return tdofs; }
	fespace->GetEssentialTrueDofs(axis_boundary, tdofs);
	tdofs.Append(axisym::AxisVertexDofs(*fespace, *axis_geometry));
	tdofs.Sort();
	tdofs.Unique();
	return tdofs;
}

void MagneticSolver::AddAxisTrueDofs(mfem::Array<int>& tdofs) const {
	tdofs.Append(AxisTrueDofs());
	tdofs.Sort();
	tdofs.Unique();
}

void MagneticSolver::ValidateMagneticAxisBoundaryValues() const {
	if (!axis_geometry || !axis_geometry->TouchesAxis()) return;

	MFEM_VERIFY(fespace,
		"Magnetic axis boundary validation requires a finite element space.");

	const mfem::Array<int> axis_tdofs = AxisTrueDofs();
	mfem::Array<int> is_axis_tdof(fespace->GetTrueVSize());
	is_axis_tdof = 0;
	for (int i = 0; i < axis_tdofs.Size(); ++i) {
		is_axis_tdof[axis_tdofs[i]] = 1;
	}

	for (const auto& bc : boundary_conditions) {
		if (!bc.IsNonzeroDirichlet()) continue;

		mfem::Array<int> marker(bc.Marker);
		mfem::Array<int> boundary_tdofs;
		fespace->GetEssentialTrueDofs(marker, boundary_tdofs);
		for (int i = 0; i < boundary_tdofs.Size(); ++i) {
			const int tdof = boundary_tdofs[i];
			MFEM_VERIFY(!is_axis_tdof[tdof],
				"Boundary group '" + bc.Condition.EntityGroupName +
				"' assigns a nonzero Dirichlet value at true DOF " +
				std::to_string(tdof) + " on the magnetic symmetry axis. "
				"Axis regularity requires A_phi = 0 at r = 0.");
		}
	}
}

mfem::BilinearFormIntegrator* MagneticSolver::MakeStiffnessIntegrator() const {
	if (geometry == GeometryType::Axisymmetric) {
		return new AxisymmetricCurlCurlIntegrator(
			*nu_coeff, axis_geometry->tolerance);
	}
	else {
		return new mfem::DiffusionIntegrator(*nu_coeff);
	}
}

mfem::Vector MagneticSolver::BuildTerminalCurrentDensity(
	const std::string& terminal_name, double current) const {
	const Terminal& term = config.Terminals.at(terminal_name);
	const EntityGroup& group = config.EntityGroups.at(term.EntityGroupName);
	return AttributeVector(group.AttributeIds,
						   term.Turns * current / TerminalArea(terminal_name));
}

const mfem::Vector& MagneticSolver::WindingFunctional(const std::string& terminal_name) const {
	ForgetMeshCachesOnRefinement();
	const auto cached = winding_functionals.find(terminal_name);
	if (cached != winding_functionals.end()) { return cached->second; }
	mfem::Vector unit_density = BuildTerminalCurrentDensity(terminal_name, 1.0);
	mfem::PWConstCoefficient unit_density_coeff(unit_density);
	mfem::LinearForm functional(fespace.get());
	functional.AddDomainIntegrator(Geometry().NewDomainLFIntegrator(unit_density_coeff));
	functional.Assemble();
	return winding_functionals.emplace(terminal_name, mfem::Vector(functional)).first->second;
}

double MagneticSolver::TerminalArea(const std::string& terminal_name) const {
	ForgetMeshCachesOnRefinement();
	const auto cached = terminal_areas.find(terminal_name);
	if (cached != terminal_areas.end()) { return cached->second; }
	const Terminal& term = config.Terminals.at(terminal_name);
	const double area = CalculateRegionMeasure(
		config.EntityGroups.at(term.EntityGroupName).AttributeIds);
	MFEM_VERIFY(area > 0.0,
		"Current terminal '" + terminal_name + "' has zero cross-section.");
	return terminal_areas.emplace(terminal_name, area).first->second;
}

void MagneticSolver::ForgetMeshCachesOnRefinement() const {
	if (cached_sequence == mesh.GetSequence()) { return; }
	terminal_areas.clear();
	winding_functionals.clear();
	cached_sequence = mesh.GetSequence();
}

mfem::Vector MagneticSolver::MassiveConductorLoad(const std::string& name,
								  const std::vector<int>& attributes) const {
	mfem::Array<int> marker =
		DomainMarkerFromAttrs(attributes, "massive conductor '" + name + "'");
	mfem::LinearForm load(fespace.get());
	load.AddDomainIntegrator(new mfem::DomainLFIntegrator(*sigma_coeff), marker);
	load.Assemble();
	return mfem::Vector(load);
}

double MagneticSolver::MassiveConductance(const std::string& name,
						  const std::vector<int>& attributes) const {
	const std::set<int> attrs(attributes.begin(), attributes.end());
	AxisymmetricConductanceCoeff axisymmetric(*sigma_coeff);
	mfem::Coefficient& integrand = geometry == GeometryType::Axisymmetric
		? static_cast<mfem::Coefficient&>(axisymmetric) : *sigma_coeff;
	double G = 0.0;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		if (!attrs.count(mesh.GetAttribute(e))) { continue; }
		mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
		const mfem::Geometry::Type shape = mesh.GetElementBaseGeometry(e);
		const int order = 2 * config.Order + T->OrderW() + 2;
		const mfem::IntegrationRule& ir = geometry == GeometryType::Axisymmetric
			? axisym::RadialRule(shape, order, *T, axis_geometry->tolerance)
			: mfem::IntRules.Get(shape, order);
		for (int i = 0; i < ir.GetNPoints(); ++i) {
			const mfem::IntegrationPoint& ip = ir.IntPoint(i);
			T->SetIntPoint(&ip);
			G += ip.weight * T->Weight() * integrand.Eval(*T, ip);
		}
	}
	MFEM_VERIFY(G > 0.0, "Massive conductor '" + name + "' has zero conductance.");
	return G;
}

void MagneticSolver::ValidateMassiveConductor(const std::string& name,
							  const std::vector<int>& attributes) const {
	ValidateMassiveConductivity(name, attributes);
	if (geometry != GeometryType::Axisymmetric) { return; }
	const std::set<int> attrs(attributes.begin(), attributes.end());
	mfem::Vector pos(mesh.SpaceDimension());
	for (int e = 0; e < mesh.GetNE(); ++e) {
		if (!attrs.count(mesh.GetAttribute(e))) { continue; }
		double min_radius = 0.0, radial_width = 0.0;
		axisym::RadialExtent(*mesh.GetElementTransformation(e), min_radius, radial_width);
		MFEM_VERIFY(min_radius > axis_geometry->tolerance,
			"Massive conductor '" + name + "' touches the symmetry axis. Its DC "
			"conductance integral sigma/(2*pi*r) is divergent; model it as a "
			"stranded conductor or move it off the axis.");
	}
}

void MagneticSolver::BuildCurrentDensity(
	const Scenario& sc,
	const std::function<bool(const Terminal&)>& include,
	mfem::Vector& j_re, mfem::Vector& j_im) const {
	j_re.SetSize(mesh.attributes.Max());
	j_im.SetSize(mesh.attributes.Max());
	j_re = 0.0;
	j_im = 0.0;

	for (const auto& [term_name, term] : config.Terminals) {
		if (term.DriveQuantity != Quantity::Current) continue;
		if (!include(term)) continue;

		const std::complex<double> I = ExcitationFor(sc, term_name);
		if (I.real() != 0.0) { j_re += BuildTerminalCurrentDensity(term_name, I.real()); }
		if (I.imag() != 0.0) { j_im += BuildTerminalCurrentDensity(term_name, I.imag()); }
	}
}
