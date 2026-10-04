// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "magnetic_solver.hpp"
#include "divergence_free_projector.hpp"
#include "../coefficients/conductor_path.hpp"
#include "../config/boundary_validation.hpp"
#include "../parallel/mpi_runtime.hpp"

/**
 * @brief What the 3D magnetic solvers share: the vector potential A in a
 *        Nedelec (H(curl)) space, its boundary conditions and gauge, and the
 *        current-carrying conductors that drive it.
 *
 * The tangential continuity of a Nedelec field across element faces -- and
 * therefore the normal continuity of B = curl A -- holds exactly, so material
 * interfaces need no special treatment.
 *
 * @par Conductors
 * Every current terminal is a conductor: a volume group plus a "direction"
 * (CurrentDirection) that says where its current flows -- azimuthally about an
 * axis, between electrodes, or around a closed loop through a cut. Its
 * ConductorPath (conductor_path.hpp) turns that into the direction field w;
 * how the current spreads over the conductor then follows from its type:
 * uniformly along the path (stranded) or as the conduction current sigma E of
 * a massive conductor, whose path is found with its own conductivity.
 *
 * @par Boundary conditions
 * A 3D vector potential has no meaningful scalar boundary value, so the
 * configured conditions are restricted to their homogeneous forms:
 *  - Dirichlet (value 0): n x A = 0, "flux tangent" -- B has no normal
 *    component (magnetic insulation, or a symmetry plane B cannot cross);
 *  - Neumann (value 0) or no entry: the natural condition n x H = 0,
 *    "flux normal" -- B crosses the boundary normally.
 * Current can enter or leave the model only through n x A = 0 walls, so
 * electrodes must lie on one.
 *
 * @par Gauge and regularization
 * The curl-curl operator annihilates gradients, so it is singular even with
 * n x A fixed on the whole boundary: every gradient of a nodal function that
 * vanishes on the n x A = 0 walls is in its null space. Sources are made
 * orthogonal to those gradients (DivergenceFreeProjector), so the singular
 * system is consistent. Where a solver needs a nonsingular matrix it adds a
 * small mass term beta (A, w),
 *     beta = kRegularization * nu_min / L^2,
 * with L the mesh bounding-box diagonal and nu_min the smallest reluctivity.
 * With an orthogonal source this selects the Coulomb-gauged solution and
 * perturbs B by a relative O(kRegularization) in every material, while keeping
 * the null-space pivots (relative size kRegularization * (nu_min/nu_max) *
 * (h/L)^2) above round-off. In an eddy-current solve beta also enters charge
 * conservation in the conductors, so there it is scaled down to the weakest
 * conductor (MagnetoquasistaticSolver3D).
 *
 * Not yet available, and rejected in Setup(): adaptive refinement.
 */
class VectorPotentialSolver3D : public MagneticSolverBase {
public:
	/// Relative size of the regularizing mass term; see the class comment.
	static constexpr double kRegularization = 1e-6;


	/// Largest fraction of a conductor's current density (L2 norm) the
	/// divergence-free projection may remove before it is reported; see
	/// ProjectedUnitCurrentLoad. The removed part is orthogonal to the kept
	/// one, so its effect on energies and inductances is about the fraction
	/// squared.
	static constexpr double kMaxProjectedFraction = 0.02;

	const DivergenceFreeProjector& Projector() const { return *projector; }

protected:
	/// A current terminal's conductor on the current mesh.
	struct TerminalConductor {
		std::string Name;
		ConductorType Type = ConductorType::Stranded;
		CurrentDirection::Kind Direction = CurrentDirection::Kind::Azimuthal;
		mfem::Array<int> Marker;              // domain attributes
		std::unique_ptr<ConductorPath> Path;
		/// Stranded: the cross-section A_cs = integral |w|. Massive: the DC
		/// conductance G = integral sigma |w|^2.
		double PathIntegral = 0.0;
	};

	/// Terminal conductors in config.Terminals (name) order. Rebuilt per mesh
	/// by BuildConductors().
	std::vector<TerminalConductor> conductors;
	std::unique_ptr<DivergenceFreeProjector> projector;

	VectorPotentialSolver3D(mfem::Mesh& m, const ProblemConfig& c) : MagneticSolverBase(m, c) {}

	// The part of Setup() before the FE space: checks, material tables, the
	// Nedelec collection and the essential-boundary marker.
	void InitializeVectorPotential() {
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

	// The part of Setup() after BuildOperators().
	void ValidateVectorPotentialBoundaries() {
		BoundaryConditionValidator validator(mesh, *fespace);
		validator.ValidateBoundaryConditions(
			boundary_conditions.Entries(), /*terminals=*/{}, /*allow_overlap=*/false);
	}

	// The Nedelec space, its essential DOFs, the gradient projector and the
	// terminal conductors for the current mesh.
	void BuildSpaceAndConductors() {
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

	/// sigma for a massive conductor, nullptr for a stranded one (the
	/// convention of ConductorCurrentCoefficient and ConductorPathIntegral).
	mfem::Coefficient* ConductivityOf(const TerminalConductor& c) const {
		return c.Type == ConductorType::Massive ? sigma_coeff.get() : nullptr;
	}

	/// Load vector of the current density ConductorCurrentCoefficient(scale)
	/// of @p c: stranded, scale * w / |w|; massive, scale * sigma * w.
	mfem::Vector AssembleConductorLoad(const TerminalConductor& c, double scale) {
		ConductorCurrentCoefficient density(*c.Path, ConductivityOf(c), scale);
		mfem::Array<int> marker(c.Marker);  // the form binds it non-const
		mfem::LinearForm load(fespace.get());
		load.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(density), marker);
		load.Assemble();
		return mfem::Vector(load);
	}

	/// The divergence-free load of 1 A through @p c: J = w / (|w| A_cs)
	/// stranded, the DC distribution sigma w / G massive.
	///
	/// The load is projected within the conductor
	/// (DivergenceFreeProjector::ProjectWithin), so the result is the nearest
	/// divergence-free current that stays inside it. A well-posed current is
	/// divergence-free in the continuum and has no normal component on its
	/// conductor's surface (except at electrodes), so the projection removes
	/// only discretization-level imbalance from it. A sizable removal means
	/// the current as given does not balance in its conductor and the
	/// projection has redistributed it, which is reported. That happens with an azimuthal direction about
	/// the wrong axis, or on a conductor that is not a body of revolution
	/// about it (including, to a few percent, a coarsely faceted round one),
	/// and with a stranded current -- uniform along its path -- in a
	/// conductor whose cross-section varies along it or that has a dead-end
	/// branch. It is reported as the fraction |grad psi| / |J| of the
	/// current's L2 norm that was removed.
	///
	/// Projecting within the conductor keeps the current: the removed
	/// gradient carries none along the path, since integral grad(psi) . w = 0
	/// for the harmonic path w (no lateral flux, psi = 0 at electrodes).
	mfem::Vector ProjectedUnitCurrentLoad(const TerminalConductor& c) {
		const double scale = 1.0 / c.PathIntegral;
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

	/// Peak of sqrt(sum_i |curl a_i|^2) over quadrature points: |B| of a real
	/// field ({A}) or of a phasor ({Re A, Im A}).
	double PeakCurlMagnitude(std::initializer_list<const mfem::GridFunction*> parts) const {
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

	void EstimateCurrentSolutionError(mfem::Vector&) override {
		MFEM_ABORT("3D magnetics does not yet support adaptive refinement.");
	}

private:
	TerminalConductor BuildConductor(const std::string& name, const Terminal& term) {
		TerminalConductor c;
		c.Name = name;
		c.Type = term.Conductor;
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

	// The angular extent of an azimuthal conductor, over which its
	// conduction potential falls by 1: 2 pi for a full ring, less for a
	// sector whose ends lie on n x A = 0 walls -- a symmetry model, cut by
	// meridian planes the current crosses normally. The ends are the
	// conductor's faces on such walls that phi-hat crosses (|n . phi-hat| >
	// 1/2; a ring merely lying against a wall is crossed by none), current
	// entering where phi-hat points in and leaving where it points out. The
	// extent is the arc from the one to the other along phi-hat.
	double AzimuthalExtent(const std::string& name, const AzimuthalPath& frame,
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

	// The connected pieces of the n x A = 0 boundary: a piece number for each
	// boundary element on it (-1 elsewhere). Elements sharing a vertex are in
	// one piece, since a continuous potential cannot differ between them.
	std::vector<int> WallPieces() const {
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

	std::unique_ptr<ConductorPath> MakeConductorPath(
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
};
