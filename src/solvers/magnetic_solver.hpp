// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "physics_solver.hpp"
#include "../axisym/axisymmetric_curl_curl_integrator.hpp"
#include "../axisym/magnetic_axis_boundary.hpp"
#include "../io/region_loss.hpp"

/**
 * @brief What every magnetic vector-potential solver shares, in 2D or 3D.
 *
 * The material tables, which do not change with the discretization, and the
 * bookkeeping of eddy-current losses, which depends only on which regions
 * conduct. The 2D scalar-potential solvers derive from MagneticSolver below;
 * the 3D vector-potential solvers derive from VectorPotentialSolver3D, because
 * almost nothing else in MagneticSolver -- axis regularity, the scalar
 * curl-curl operator, I/area source densities -- has a 3D meaning.
 */
class MagneticSolverBase : public PhysicsSolver {
public:
	/// One conductive region's time-averaged dissipation [W], and the label
	/// under which it reports.
	using RegionLoss = ::RegionLoss;

protected:
	// nu = 1/mu (reluctivity) and the field-solve conductivity sigma, keyed
	// by mesh DOMAIN attribute. Unclaimed attributes fall back to vacuum.
	// Built in each derived Setup(); refinement-invariant, like every material
	// table. sigma is zero on stranded conductors; see BuildConductivity().
	std::unique_ptr<mfem::PWConstCoefficient> nu_coeff;
	std::unique_ptr<mfem::PWConstCoefficient> sigma_coeff;

	MagneticSolverBase(mfem::Mesh& m, const ProblemConfig& c) : PhysicsSolver(m, c) {}

	static double Reluctivity(const Material& m) {
		return 1.0 / (Constants::MU_0 * m.RelPermeability);
	}
	static double Conductivity(const Material& m) { return m.Conductivity; }

	void BuildReluctivity() {
		nu_coeff = MaterialCoefficient(1.0 / Constants::MU_0, Reluctivity);
	}
	// The conductivity of the eddy-current term j omega sigma A, which is zero
	// on every stranded conductor whatever its material.
	//
	// A stranded conductor is a winding: insulated strands in series, so the
	// winding's connection fixes the current in every strand and none crosses
	// between them. Its current is the imposed source alone. A sigma term
	// there would add a free induced current -j omega sigma A on top, as if
	// the winding were also a solid block -- in a ring coil, a shorted turn
	// sharing its volume -- which changes the coil's actual current, screens
	// its field and dissipates power no terminal accounts for. The material's
	// sigma is the wire's conductivity; it matters for the winding's own
	// resistance and in-strand losses, which are not modelled, not for the
	// field.
	void BuildConductivity() {
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

	// A massive conductor's current is sigma E, so every attribute of it must
	// conduct: sigma = 0 there would carry no current and make its conductance
	// meaningless.
	void ValidateMassiveConductivity(const std::string& name,
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

	// Integrate a loss density over every region that can dissipate, one
	// entry per reporting owner.
	//
	// Membership is decided by the field-solve sigma > 0, not by whether a
	// region owns a port. The sigma mass term induces eddy currents in any
	// conductive material, so a flux shield or a steel brace dissipates real
	// power while appearing in no coupling matrix. Reporting only ported
	// regions would produce a "total" that silently omits it. Stranded
	// conductors have sigma = 0 there (BuildConductivity), so they report
	// nothing: the field solve dissipates nothing in them.
	//
	// Each conductive attribute has exactly one owner. Exclusive ownership is
	// essential, not cosmetic: a terminal and a region routinely share an
	// entity group (a massive conductor is usually also declared as a
	// material region), so grouping by both names independently would
	// integrate that attribute twice. Terminals win because they are the more
	// specific description of the same metal; conductive attributes no
	// terminal or region claims report individually.
	std::vector<RegionLoss> IntegrateRegionLosses(mfem::Coefficient& density) const {
		std::vector<RegionLoss> losses;
		for (const auto& [name, attrs] : ConductingGroups()) {
			losses.push_back({ name, IntegrateOverAttributes(density, attrs) });
		}
		return losses;
	}

	// The conducting attributes (field-solve sigma > 0), grouped by the name
	// they report under: a massive terminal, else a region, else
	// "attribute N". See IntegrateRegionLosses for why each attribute has
	// exactly one owner.
	std::map<std::string, std::set<int>> ConductingGroups() const {
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

	// Print per-region and total dissipation.
	//
	// Reported only for field scenarios. Coupling runs drive synthetic unit
	// currents one terminal at a time, so the loss of any single such column
	// is not the loss of a physically realised operating point.
	void ReportRegionLosses(const std::vector<RegionLoss>& losses) const {
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

	/// Resistance and inductance matrices at one frequency of an MQS
	/// coupling run.
	struct ImpedancePoint {
		double Frequency = 0.0;
		mfem::DenseMatrix Resistance, Inductance;
	};

	// Write and print an MQS coupling sweep: one R and one L per frequency.
	void WriteImpedanceSeries(const std::vector<ImpedancePoint>& points) const {
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

private:
	// Element-wise integral of @p density over the given attributes, with the
	// geometric measure (2 pi r in axisymmetry). The rule is sized for a
	// density quadratic in the solution; the axisymmetric drive field's 1/r
	// factors are why it is not borrowed from a source integrator.
	double IntegrateOverAttributes(mfem::Coefficient& density,
								   const std::set<int>& attrs) const {
		double total = 0.0;
		mfem::Vector pos;
		for (int e = 0; e < mesh.GetNE(); ++e) {
			if (!attrs.count(mesh.GetAttribute(e))) { continue; }
			mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
			const mfem::FiniteElement& fe = *fespace->GetFE(e);
			const int order = 2 * fe.GetOrder() + T.OrderW() + 2;
			const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe.GetGeomType(), order);
			for (int q = 0; q < ir.GetNPoints(); ++q) {
				const mfem::IntegrationPoint& ip = ir.IntPoint(q);
				T.SetIntPoint(&ip);
				T.Transform(ip, pos);
				total += density.Eval(T, ip) * ip.weight * T.Weight() * Geometry().Measure(pos);
			}
		}
		return total;
	}
};

/**
 * @brief Base class for the 2D solvers formulated in a scalar vector potential.
 *
 * Holds what the magnetostatic and magnetoquasistatic solvers share by virtue
 * of solving for the same unknown -- A_phi (axisymmetric) or A_z (planar) --
 * rather than by coincidence: the curl-curl stiffness term built from the
 * reluctivity, terminal current density, and the axis regularity condition.
 * None of this applies to an electrostatic run, which is why it does not belong
 * in PhysicsSolver.
 *
 * The solution field itself stays in the derived classes: magnetostatics holds
 * a real GridFunction, the time-harmonic solver a ComplexGridFunction.
 */
class MagneticSolver : public MagneticSolverBase {
protected:

	// Radial extent and scale-relative axis tolerance of the (r,z) mesh. Owned
	// here rather than by PhysicsSolver because every consumer is magnetic: the
	// tolerance feeds the curl-curl 1/r axis limit and the B-field recovery,
	// and TouchesAxis drives the A_phi = 0 regularity condition. Planar runs
	// leave it at its default.
	axisym::AxisGeometry axisymmetric_mesh;

	// Boundary attributes lying entirely on r = 0. Discovered here rather than
	// during geometric classification because only an A_phi formulation needs a
	// dedicated axis attribute; an electrostatic run on the same mesh does not.
	mfem::Array<int> axis_boundary;

	MagneticSolver(mfem::Mesh& m, const ProblemConfig& c) : MagneticSolverBase(m, c) {}

	// Adopt the configured coordinate model, restricted to the 2D reductions.
	//
	// Everything below is a scalar-potential formulation: the unknown is the
	// single out-of-plane (A_z) or azimuthal (A_phi) component. A 3D model has
	// a full vector potential, which needs an H(curl) (Nedelec) discretization,
	// a divergence-free source and a gauge -- a different formulation rather
	// than another geometry branch here, so SolverFactory routes '3d' runs to
	// the separate 3D solver classes. Running this class on a 3D mesh would
	// assemble a scalar Laplacian and report it as a magnetic field, so it is
	// rejected outright.
	void InitializeMagneticGeometry() {
		MFEM_VERIFY(config.GeometryType != GeometryType::Cartesian3D,
			"This " + std::string(ToString(config.PhysicsType)) + " solver is "
			"the 2D scalar-potential formulation; geometry_type '3d' needs the "
			"vector (H(curl)) formulation of the 3D solver classes.");
		InitializeGeometry();
	}

	// Validate the axisymmetric mesh as (r,z) input, keep the resulting radial
	// extent, then add what only an A_phi formulation cares about: whether the
	// domain reaches r = 0, and whether any near-axis element leaves the 1/r
	// quadrature under-resolved.
	//
	// Both are regularity concerns. Axis regularity exists because A_phi is the
	// component of a vector field that must vanish on the axis to stay
	// single-valued; a scalar potential carries no such constraint, so an
	// electrostatic run has no use for either report.
	void ValidateMagneticAxisymmetricGeometry() {
		axisymmetric_mesh = ValidateAxisymmetricGeometry();
		if (geometry != GeometryType::Axisymmetric) { return; }

		Reporter().Diagnostic(
			axisymmetric_mesh.TouchesAxis()
				? "Axisymmetric domain touches the symmetry axis: "
				  "axis regularity A_phi = 0 will be enforced."
				: "Axisymmetric domain is annular: no axis condition required.");

		WarnOnUnderResolvedRadialQuadrature();
	}

	// The curl-curl 1/r term is integrated by a geometry-aware rule whose cost
	// is set by s = r_min/h per element (see
	// AxisymmetricCurlCurlIntegrator::RadialExtraOrder). 1/r is rational, so no
	// polynomial rule integrates it exactly and the rule must be capped; an
	// element that is both very thin radially and very close to the axis can
	// therefore fall outside the accuracy target. Such an element is rare and
	// always a meshing choice, but the resulting error is silent, so report it
	// once. The electrostatic r-weighted diffusion integrand is polynomial and
	// is integrated exactly, so no equivalent concern exists there.
	void WarnOnUnderResolvedRadialQuadrature() {
		int worst_element = -1;
		double worst_ratio = std::numeric_limits<double>::max();

		for (int e = 0; e < mesh.GetNE(); ++e) {
			double min_radius = 0.0;
			double radial_width = 0.0;
			AxisymmetricCurlCurlIntegrator::RadialExtent(
				*mesh.GetElementTransformation(e), min_radius, radial_width);

			// Elements meeting the axis are excluded by design: there the
			// divergent directions are removed by the A_phi = 0 constraint.
			if (!(radial_width > 0.0)) { continue; }
			if (axisymmetric_mesh.IsOnAxisGeometry(min_radius)) { continue; }

			const double ratio = min_radius / radial_width;
			if (ratio < worst_ratio) {
				worst_ratio = ratio;
				worst_element = e;
			}
		}

		if (worst_element < 0) { return; }
		if (worst_ratio >= AxisymmetricCurlCurlIntegrator::kResolvedRadiusRatio) {
			return;
		}

		std::ostringstream msg;
		msg << std::setprecision(3)
			<< "Element " << worst_element << " has r_min/width = " << worst_ratio
			<< ", below the ratio " << AxisymmetricCurlCurlIntegrator::kResolvedRadiusRatio
			<< " at which the curl-curl 1/r quadrature reaches its accuracy "
			   "target. The capped rule integrates such elements approximately; "
			   "widen the innermost radial band or move it away from the axis if "
			   "near-axis accuracy matters.";
		Reporter().Warning(msg.str());
	}

	// Axis regularity, imposition half: A_phi = 0 on r = 0. The dedicated axis
	// boundary attribute joins the prescribed Dirichlet conditions in ess_bdr, so
	// the ordering (merge before BuildOperators() reads ess_bdr) is structural
	// rather than a convention the caller has to remember.
	void BuildEssentialBoundaryMarker() override {
		PhysicsSolver::BuildEssentialBoundaryMarker();

		if (geometry != GeometryType::Axisymmetric) { return; }

		axis_boundary = axisym::FindAxisBoundaryMarker(mesh, axisymmetric_mesh);

		MFEM_VERIFY(ess_bdr.Size() == axis_boundary.Size(),
			"Axis boundary marker does not match the mesh boundary attributes.");
		MergeMarker(ess_bdr, axis_boundary);
	}

	// Axis regularity, verification half: a nonzero Dirichlet value on the axis
	// contradicts the A_phi = 0 constraint imposed above. The constraint would
	// silently win, so the configuration is rejected instead. Requires the FE
	// space, so call after BuildOperators().
	void ValidateMagneticAxisBoundaryValues() const {
		if (geometry != GeometryType::Axisymmetric ||
			!axisymmetric_mesh.TouchesAxis()) return;

		MFEM_VERIFY(fespace,
			"Magnetic axis boundary validation requires a finite element space.");

		mfem::Array<int> axis_tdofs;
		fespace->GetEssentialTrueDofs(axis_boundary, axis_tdofs);
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

	// Stiffness term: axisymmetric curl-curl (nu * curl A * curl A, carrying the
	// 1/r factor) or planar diffusion (nu * grad A * grad A). A fresh instance is
	// returned each call so the solve's bilinear form and the AMR error estimator
	// can own separate copies.
	mfem::BilinearFormIntegrator* MakeStiffnessIntegrator() const {
		if (geometry == GeometryType::Axisymmetric) {
			return new AxisymmetricCurlCurlIntegrator(
				*nu_coeff, axisymmetric_mesh.tolerance);
		}
		else {
			return new mfem::DiffusionIntegrator(*nu_coeff);
		}
	}

	// Uniform current density I/area over the terminal's domain attributes,
	// laid out per mesh attribute for a PWConstCoefficient.
	//
	// This is a 2D-reduction relation. The terminal region is a conductor
	// CROSS-SECTION here, so its measure is an area and I/area is a current
	// density. In a full 3D model the same attributes would bound a volume, whose
	// measure is not a cross-section, and the current would have to be given a
	// direction as well as a magnitude; this scalar form does not generalize.
	mfem::Vector BuildTerminalCurrentDensity(
		const std::string& terminal_name, double current) const {
		const Terminal& term = config.Terminals.at(terminal_name);
		const EntityGroup& group = config.EntityGroups.at(term.EntityGroupName);
		const double area = CalculateRegionMeasure(group.AttributeIds);
		MFEM_VERIFY(area > 0.0,
			"Current terminal '" + terminal_name + "' has zero cross-section.");

		return AttributeVector(group.AttributeIds, current / area);
	}

	// Scenario source current density, summed over the terminals @p include
	// accepts. Current enters the model only through Terminals, so this is a
	// pure function of sc.Excitations: a terminal the scenario does not drive
	// contributes nothing. In CouplingMatrix mode the scenario carries a single
	// unit excitation, so this IS the drive for that column rather than
	// background data, and must not be suppressed the way boundary data is.
	mfem::Vector BuildCurrentDensity(
		const Scenario& sc,
		const std::function<bool(const Terminal&)>& include) const {
		mfem::Vector j_src(mesh.attributes.Max());
		j_src = 0.0;

		for (const auto& [term_name, term] : config.Terminals) {
			if (term.DriveQuantity != Quantity::Current) continue;
			if (!include(term)) continue;

			const double I = ExcitationFor(sc, term_name);
			if (I == 0.0) continue;
			j_src += BuildTerminalCurrentDensity(term_name, I);
		}
		return j_src;
	}
};
