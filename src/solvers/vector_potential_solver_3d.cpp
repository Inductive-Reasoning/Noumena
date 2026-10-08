// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "vector_potential_solver_3d.hpp"

void VectorPotentialSolver3D::InitializeVectorPotential() {
	const std::string physics = ToString(config.PhysicsType);
	InitializeGeometry();
	MFEM_VERIFY(geometry == GeometryType::Cartesian3D,
		"The 3D " + physics + " solver requires geometry_type '3d'.");
	for (const auto& [name, term] : config.Terminals) {
		MFEM_VERIFY(term.DriveQuantity == Quantity::Current,
			"Magnetic terminal '" + name + "' must use a current excitation.");
		MFEM_VERIFY(term.Direction.has_value(),
			"3D magnetic terminal '" + name + "' needs a 'direction'.");
	}
	MFEM_VERIFY(!config.Amr.Enabled,
		"3D " + physics + " does not yet support adaptive refinement.");
	MFEM_VERIFY(config.LinearSolver == LinearSolverType::Direct || parallel::Enabled(),
		"The iterative solvers for 3D magnetics use hypre's AMS "
		"preconditioner and need the MPI/HYPRE build (-DUSE_MPI=ON). Set "
		"simulation.linear_solver to 'direct' in this serial build.");

	BuildReluctivity();
	BuildConductivity();
	fec = std::make_unique<mfem::ND_FECollection>(config.Order, mesh.Dimension());

	boundary_conditions = BuildBoundaryConditions();
	for (const auto& bc : boundary_conditions) {
		MFEM_VERIFY(bc.Condition.Value == 0.0,
			"Boundary group '" + bc.Condition.EntityGroupName + "' has value " +
			std::to_string(bc.Condition.Value) + ". 3D magnetics supports only "
			"homogeneous conditions: 'dirichlet' 0 (n x A = 0, flux tangent) "
			"and 'neumann' 0 (n x H = 0, flux normal).");
	}
	BuildEssentialBoundaryMarker();
}

void VectorPotentialSolver3D::ValidateVectorPotentialBoundaries() {
	BoundaryConditionValidator validator(mesh, *fespace);
	validator.ValidateBoundaryConditions(
		boundary_conditions.Entries(), /*terminals=*/{}, /*allow_overlap=*/false);
}

void VectorPotentialSolver3D::BuildSpaceAndConductors() {
	fespace = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get());
	fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
	Reporter().Status("Mesh has " + std::to_string(mesh.GetNE()) +
		" elements; field space has " + std::to_string(fespace->GetTrueVSize()) +
		" true DOFs.");

	auto operation = Reporter().Start("conductor paths");
	projector = std::make_unique<DivergenceFreeProjector>(*fespace, ess_bdr);
	conductors.clear();
	for (const auto& [name, term] : config.Terminals) {
		conductors.push_back(BuildConductor(name, term));
	}
}

mfem::Vector VectorPotentialSolver3D::AssembleConductorLoad(const TerminalConductor& c, double scale) {
	ConductorCurrentCoefficient density(*c.Path, ConductivityOf(c), scale);
	mfem::Array<int> marker(c.Marker);  // the form binds it non-const
	mfem::LinearForm load(fespace.get());
	load.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(density), marker);
	load.Assemble();
	return mfem::Vector(load);
}

mfem::Vector VectorPotentialSolver3D::ProjectedUnitCurrentLoad(const TerminalConductor& c) {
	const double scale = c.Turns / c.PathIntegral;
	mfem::Vector load = AssembleConductorLoad(c, scale);
	const double removed = projector->ProjectWithin(load, c.Marker);
	ConductorCurrentCoefficient J(*c.Path, ConductivityOf(c), scale);
	const double fraction = std::sqrt(std::max(removed, 0.0) /
		ConductorCurrentNormSquared(mesh, c.Marker, J, config.Order));
	std::ostringstream msg;
	msg << std::setprecision(3) << "Terminal '" << c.Name << "': the divergence-free "
		"projection removed " << 100.0 * fraction << "% of its current density";
	if (fraction <= kMaxProjectedFraction) {
		Reporter().Diagnostic(msg.str() + ".");
		return load;
	}
	msg << ", so the current as given does not stay balanced in its conductor and "
		"has been redistributed within it. ";
	if (c.Direction == CurrentDirection::Kind::Azimuthal) {
		msg << "An 'azimuthal' direction fits only a conductor that is a body of "
			"revolution about 'origin' and 'axis': check both, refine a coarsely "
			"faceted round conductor (or use curved elements), or describe the path "
			"with a 'cut' or 'electrodes'.";
	} else if (c.Type == ConductorType::Stranded) {
		msg << "A stranded current is uniform along its path, which balances only "
			"where the conductor's cross-section is constant along it and it has no "
			"dead-end branches; use a massive conductor, whose current follows its "
			"conduction path, or check the geometry.";
	} else {
		msg << "Check the conductor's mesh resolution.";
	}
	Reporter().Warning(msg.str());
	return load;
}

double VectorPotentialSolver3D::RegularizationWeight() const {
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

double VectorPotentialSolver3D::PeakCurlMagnitude(std::initializer_list<const mfem::GridFunction*> parts) const {
	std::vector<mfem::CurlGridFunctionCoefficient> curls;
	curls.reserve(parts.size());
	for (const mfem::GridFunction* a : parts) { curls.emplace_back(a); }
	mfem::Vector B;
	double peak = 0.0;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
		const mfem::IntegrationRule& ir =
			mfem::IntRules.Get(mesh.GetElementBaseGeometry(e), 2 * config.Order);
		for (int q = 0; q < ir.GetNPoints(); ++q) {
			const mfem::IntegrationPoint& ip = ir.IntPoint(q);
			T->SetIntPoint(&ip);
			double squared = 0.0;
			for (auto& curl : curls) {
				curl.Eval(B, *T, ip);
				squared += B * B;
			}
			peak = std::max(peak, std::sqrt(squared));
		}
	}
	return peak;
}

VectorPotentialSolver3D::TerminalConductor VectorPotentialSolver3D::BuildConductor(const std::string& name, const Terminal& term) {
	TerminalConductor c;
	c.Name = name;
	c.Type = term.Conductor;
	c.Turns = term.Turns;
	c.Direction = term.Direction->Type;
	const EntityGroup& group = config.EntityGroups.at(term.EntityGroupName);
	c.Marker = DomainMarkerFromAttrs(group.AttributeIds, "terminal '" + name + "'");
	if (c.Type == ConductorType::Massive) {
		ValidateMassiveConductivity(name, group.AttributeIds);
	}
	c.Path = MakeConductorPath(name, *term.Direction, c.Marker, ConductivityOf(c));
	c.PathIntegral = ConductorPathIntegral(mesh, c.Marker, *c.Path, ConductivityOf(c),
										   config.Order);
	MFEM_VERIFY(c.PathIntegral > 0.0, "Terminal '" + name + "' has zero " +
		(c.Type == ConductorType::Massive ? "conductance." : "cross-section."));
	return c;
}

double VectorPotentialSolver3D::AzimuthalExtent(const std::string& name, const AzimuthalPath& frame,
					   const mfem::Array<int>& conductor) const {
	auto inside = [&](int e) {
		return e >= 0 && conductor[mesh.GetAttribute(e) - 1] != 0;
	};
	double in_sin = 0.0, in_cos = 0.0, out_sin = 0.0, out_cos = 0.0;
	bool entries = false, exits = false;
	mfem::Vector x, n(3), t, c;
	for (int be = 0; be < mesh.GetNBE(); ++be) {
		const int a = mesh.GetBdrAttribute(be);
		if (a < 1 || a > ess_bdr.Size() || !ess_bdr[a - 1]) continue;
		const int f = mesh.GetBdrElementFaceIndex(be);
		int e1, e2;
		mesh.GetFaceElements(f, &e1, &e2);
		const int e = inside(e1) ? e1 : (inside(e2) ? e2 : -1);
		if (e < 0) continue;

		mfem::ElementTransformation* T = mesh.GetFaceTransformation(f);
		const mfem::IntegrationPoint& center = mfem::Geometries.GetCenter(T->GetGeometryType());
		T->SetIntPoint(&center);
		T->Transform(center, x);
		mfem::CalcOrtho(T->Jacobian(), n);
		mesh.GetElementTransformation(e)->Transform(
			mfem::Geometries.GetCenter(mesh.GetElementBaseGeometry(e)), c);
		mfem::Vector outward(x);
		outward -= c;
		if (outward * n < 0.0) n.Neg();  // out of the conductor
		frame.Tangent(x, t);
		const double area = n.Norml2(), crossing = (n * t) / area;
		if (std::abs(crossing) < 0.5) continue;
		const double angle = frame.Angle(x);
		if (crossing > 0.0) {
			exits = true;
			out_sin += area * std::sin(angle);
			out_cos += area * std::cos(angle);
		} else {
			entries = true;
			in_sin += area * std::sin(angle);
			in_cos += area * std::cos(angle);
		}
	}
	if (!entries && !exits) return Constants::TWO_PI;
	MFEM_VERIFY(entries && exits, "Terminal '" + name + "' is an azimuthal sector "
		"with only one end on an n x A = 0 ('dirichlet') boundary; both ends must "
		"lie on one for its current to enter and leave.");
	double extent = std::atan2(out_sin, out_cos) - std::atan2(in_sin, in_cos);
	if (extent <= 0.0) extent += Constants::TWO_PI;
	std::ostringstream msg;
	msg << std::setprecision(4) << "Terminal '" << name << "' is an azimuthal sector of "
		<< extent * 360.0 / Constants::TWO_PI << " degrees.";
	Reporter().Diagnostic(msg.str());
	return extent;
}

std::vector<int> VectorPotentialSolver3D::WallPieces() const {
	std::vector<int> root(mesh.GetNV());
	for (int v = 0; v < mesh.GetNV(); ++v) { root[v] = v; }
	auto find = [&](int v) {
		while (root[v] != v) { v = root[v] = root[root[v]]; }
		return v;
	};
	auto on_wall = [&](int be) {
		const int a = mesh.GetBdrAttribute(be);
		return a >= 1 && a <= ess_bdr.Size() && ess_bdr[a - 1];
	};
	mfem::Array<int> vertices;
	for (int be = 0; be < mesh.GetNBE(); ++be) {
		if (!on_wall(be)) continue;
		mesh.GetBdrElementVertices(be, vertices);
		for (int v : vertices) { root[find(v)] = find(vertices[0]); }
	}
	std::vector<int> piece(mesh.GetNBE(), -1);
	for (int be = 0; be < mesh.GetNBE(); ++be) {
		if (!on_wall(be)) continue;
		mesh.GetBdrElementVertices(be, vertices);
		piece[be] = find(vertices[0]);
	}
	return piece;
}

std::unique_ptr<ConductorPath> VectorPotentialSolver3D::MakeConductorPath(
	const std::string& name, const CurrentDirection& d,
	const mfem::Array<int>& conductor, mfem::Coefficient* conductivity) {
	if (d.Type == CurrentDirection::Kind::Azimuthal) {
		auto path = std::make_unique<AzimuthalPath>(
			d, AzimuthalExtent(name, AzimuthalPath(d), conductor));
		// The direction is undefined on the axis; a vertex there means the
		// conductor reaches it (quadrature points alone could miss that).
		mfem::Vector lo, hi;
		mesh.GetBoundingBox(lo, hi);
		hi -= lo;
		mfem::Array<int> vertices;
		for (int e = 0; e < mesh.GetNE(); ++e) {
			if (!conductor[mesh.GetAttribute(e) - 1]) continue;
			mesh.GetElementVertices(e, vertices);
			for (int v : vertices) {
				mfem::Vector x(mesh.GetVertex(v), 3);
				MFEM_VERIFY(path->RadiusOf(x) > 1e-9 * hi.Norml2(),
					"Terminal '" + name + "' reaches its own axis, where "
					"the azimuthal current direction is undefined.");
			}
		}
		return path;
	}

	const int n_bdr = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
	mfem::Array<int> none(n_bdr);
	none = 0;
	if (d.Type == CurrentDirection::Kind::Electrodes) {
		mfem::Array<int> input = MarkerFromGroup(d.Input);
		mfem::Array<int> output = MarkerFromGroup(d.Output);
		// Current may only enter or leave the model through an n x A = 0
		// wall: anywhere else the load is not balanced, and the projection
		// would silently redistribute the missing return current.
		for (int a = 0; a < n_bdr; ++a) {
			MFEM_VERIFY(!(input[a] || output[a]) || ess_bdr[a],
				"The electrodes of terminal '" + name + "' must lie on a "
				"'dirichlet' (n x A = 0) boundary; for a closed loop use a 'cut'.");
		}
		// The electrodes must share one connected piece of that wall. If
		// they do not, a closed loop on the rest of the boundary (n x H = 0)
		// runs between the pieces around the conductor: tangential H is
		// zero along it, so by Ampere's law no net current can pass through
		// it, yet all of the terminal's current does. The problem then has
		// no solution, and the solve would return a wrong field without
		// failing. (Discretely: a potential that is 1 on one piece and 0 on
		// the others has a gradient the n x A = 0 space contains, and
		// testing the field equation with it demands zero net current into
		// the piece.)
		const std::vector<int> pieces = WallPieces();
		std::set<int> touched;
		for (int be = 0; be < mesh.GetNBE(); ++be) {
			const int a = mesh.GetBdrAttribute(be) - 1;
			if (a < 0 || a >= n_bdr || !(input[a] || output[a])) continue;
			int e1, e2;
			mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(be), &e1, &e2);
			const bool borders = (e1 >= 0 && conductor[mesh.GetAttribute(e1) - 1])
				|| (e2 >= 0 && conductor[mesh.GetAttribute(e2) - 1]);
			if (borders) touched.insert(pieces[be]);
		}
		MFEM_VERIFY(touched.size() <= 1,
			"The electrodes of terminal '" + name + "' lie on separate pieces of the "
			"'dirichlet' (n x A = 0) boundary that do not touch. A closed loop can "
			"then be drawn on the rest of the boundary, which is n x H = 0, between "
			"the pieces and around the conductor. Tangential H is zero along it, so "
			"by Ampere's law no net current can pass through the loop, yet all of "
			"the terminal's current does: the problem has no solution. Join the "
			"electrodes by a connected 'dirichlet' region, e.g. make the walls "
			"between them 'dirichlet' too.");
		return std::make_unique<ConductionPath>(mesh, config.Order, conductor,
												conductivity, d, input, output, none);
	}
	return std::make_unique<ConductionPath>(mesh, config.Order, conductor, conductivity,
											d, none, none, MarkerFromGroup(d.Cut));
}
