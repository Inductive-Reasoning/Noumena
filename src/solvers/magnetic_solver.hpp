// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "physics_solver.hpp"
#include "../axisym/axisymmetric_curl_curl_integrator.hpp"
#include "../axisym/magnetic_axis_boundary.hpp"
#include "../axisym/radial_quadrature.hpp"
#include "../coefficients/axisymmetric_conductance_coefficient.hpp"
#include "../io/region_loss.hpp"
#include "../linalg/complex_direct_solver.hpp"

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

	// Radial extent and axis tolerance of an axisymmetric mesh; empty otherwise,
	// since a planar or 3D domain has no axis.
	std::optional<axisym::AxisGeometry> axis_geometry;

	MagneticSolverBase(mfem::Mesh& m, const ProblemConfig& c) : PhysicsSolver(m, c) {}

	static double Reluctivity(const Material& m) {
		return 1.0 / (Constants::MU_0 * m.RelPermeability);
	}
	static double Conductivity(const Material& m) { return m.Conductivity; }

	// The time-harmonic direct solve is fast only with STRUMPACK; say so when
	// this build falls back to Eigen. See ComplexDirectSolver.
	void WarnOnSlowComplexDirectSolve() const;

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
	void BuildConductivity();

	// A massive conductor's current is sigma E, so every attribute of it must
	// conduct: sigma = 0 there would carry no current and make its conductance
	// meaningless.
	void ValidateMassiveConductivity(const std::string& name,
									 const std::vector<int>& attributes) const;

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
	std::vector<RegionLoss> IntegrateRegionLosses(mfem::Coefficient& density) const;

	// The conducting attributes (field-solve sigma > 0), grouped by the name
	// they report under: a massive terminal, else a region, else
	// "attribute N". See IntegrateRegionLosses for why each attribute has
	// exactly one owner.
	std::map<std::string, std::set<int>> ConductingGroups() const;

	// Print per-region and total dissipation.
	//
	// Reported only for field scenarios. Coupling runs drive synthetic unit
	// currents one terminal at a time, so the loss of any single such column
	// is not the loss of a physically realised operating point.
	void ReportRegionLosses(const std::vector<RegionLoss>& losses) const;

	/// Resistance and inductance matrices at one frequency of an MQS
	/// coupling run.
	struct ImpedancePoint {
		double Frequency = 0.0;
		mfem::DenseMatrix Resistance, Inductance;
	};

	// Write and print an MQS coupling sweep: one R and one L per frequency.
	void WriteImpedanceSeries(const std::vector<ImpedancePoint>& points) const;

private:
	// Element-wise integral of @p density over the given attributes, with the
	// geometric measure (2 pi r in axisymmetry). The rule is sized for a
	// density quadratic in the solution and, in axisymmetry, for the 1/r of a
	// massive conductor's drive field V / (2 pi r) (radial_quadrature.hpp).
	double IntegrateOverAttributes(mfem::Coefficient& density,
								   const std::set<int>& attrs) const;
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
	void InitializeMagneticGeometry();

	// Validate the axisymmetric mesh as (r,z) input, keep the resulting radial
	// extent, then add what only an A_phi formulation cares about: whether the
	// domain reaches r = 0, and whether any near-axis element leaves the 1/r
	// quadrature under-resolved.
	//
	// Both are regularity concerns. Axis regularity exists because A_phi is the
	// component of a vector field that must vanish on the axis to stay
	// single-valued; a scalar potential carries no such constraint, so an
	// electrostatic run has no use for either report.
	void ValidateMagneticAxisymmetricGeometry();

	// The 1/r integrands (curl-curl, a massive conductor's conductance and
	// drive-field loss) are integrated by a geometry-aware rule whose order is
	// set per element by the ratio q_min/(q_max - q_min) of r with any axis
	// contact factored out (radial_quadrature.hpp). 1/r is rational, so the
	// added order is capped, and an element below kResolvedRadiusRatio falls
	// outside the accuracy target: off the axis, one radially wide compared with
	// its distance from it; on the axis, a sliver whose far side nearly touches
	// it. Such an element is a meshing choice, but the resulting error is
	// silent, so report it once. The electrostatic r-weighted diffusion
	// integrand is polynomial and is integrated exactly, so no equivalent
	// concern exists there.
	void WarnOnUnderResolvedRadialQuadrature();

	// Axis regularity, imposition half: A_phi = 0 on r = 0. The dedicated axis
	// boundary attribute joins the prescribed Dirichlet conditions in ess_bdr, so
	// the ordering (merge before BuildOperators() reads ess_bdr) is structural
	// rather than a convention the caller has to remember.
	void BuildEssentialBoundaryMarker() override;

	// Boundary attributes lying entirely on r = 0; only an A_phi formulation
	// needs the axis tagged on its own.
	mfem::Array<int> axis_boundary;

	// The true DOFs where A_phi = 0 on the axis: those of the axis attribute
	// and of every axis vertex (a domain meeting the axis at a point has no
	// axis attribute). The solvers add them to their essential DOFs.
	mfem::Array<int> AxisTrueDofs() const;

	void AddAxisTrueDofs(mfem::Array<int>& tdofs) const;

	// Axis regularity, verification half: a nonzero Dirichlet value on the axis
	// contradicts the A_phi = 0 constraint imposed above. The constraint would
	// silently win, so the configuration is rejected instead. Requires the FE
	// space, so call after BuildOperators().
	void ValidateMagneticAxisBoundaryValues() const;

	// Stiffness term: axisymmetric curl-curl (nu * curl A * curl A, carrying the
	// 1/r factor) or planar diffusion (nu * grad A * grad A). A fresh instance is
	// returned each call so the solve's bilinear form and the AMR error estimator
	// can own separate copies.
	mfem::BilinearFormIntegrator* MakeStiffnessIntegrator() const;

	// Uniform current density N I/area over the terminal's domain attributes,
	// for a winding of N turns each carrying I, laid out per mesh attribute
	// for a PWConstCoefficient.
	//
	// This is a 2D-reduction relation. The terminal region is a conductor
	// CROSS-SECTION here, so its measure is an area and I/area is a current
	// density. In a full 3D model the same attributes would bound a volume, whose
	// measure is not a cross-section, and the current would have to be given a
	// direction as well as a magnitude; this scalar form does not generalize.
	mfem::Vector BuildTerminalCurrentDensity(
		const std::string& terminal_name, double current) const;

	// A stranded terminal's winding functional, lambda_k(A) = integral of
	// (N_k/area_k) A dV: the flux linkage of the field A with terminal k, the
	// load of a unit current in it. It depends on the mesh only, so it is
	// assembled once per mesh rather than per coupling-matrix entry.
	const mfem::Vector& WindingFunctional(const std::string& terminal_name) const;

private:
	// The cross-section of a stranded terminal, measured once per mesh.
	double TerminalArea(const std::string& terminal_name) const;

	void ForgetMeshCachesOnRefinement() const;

	mutable std::map<std::string, double> terminal_areas;
	mutable std::map<std::string, mfem::Vector> winding_functionals;
	mutable long cached_sequence = -1;

protected:

	// A massive conductor carries its DC conduction distribution
	// J = sigma w V, with the path w = 1 per unit length in the plane and
	// w = 1/(2 pi r) around the axis, and V the voltage driving the current.
	// Its conductance is G = integral sigma |w|^2 dV, so a current I has
	// V = I / G; this is the distribution an MQS massive conductor tends to as
	// the frequency goes to zero, and the one 3D gives a massive conductor.
	//
	// MassiveConductorLoad is the load of J for V = 1: integral sigma w v dV.
	// Weighted by the measure (2 pi r around the axis), sigma w becomes plain
	// sigma, so it is the same unweighted domain form in both geometries.
	// Assembly is restricted to the conductor's elements, so its cost is
	// proportional to the conductor rather than to the mesh.
	mfem::Vector MassiveConductorLoad(const std::string& name,
									  const std::vector<int>& attributes) const;

	// DC conductance G of a massive conductor: the integral of sigma over its
	// elements in the plane, of sigma/(2 pi r) around the axis, where the rule
	// also resolves the 1/r factor by the element's distance from the axis
	// (radial_quadrature.hpp). A rule of fixed order cannot: a ring's
	// conductance sigma h ln(b/a) / (2 pi) grows without bound as a -> 0.
	double MassiveConductance(const std::string& name,
							  const std::vector<int>& attributes) const;

	// A massive conductor needs a positive conductivity throughout and, around
	// the axis, must not reach it: its conductance integral sigma/(2 pi r)
	// diverges there.
	void ValidateMassiveConductor(const std::string& name,
								  const std::vector<int>& attributes) const;

	// Real and imaginary parts of the scenario source current density, summed
	// over the terminals @p include accepts. Current enters the model only
	// through Terminals, so this is a pure function of sc.Excitations: a
	// terminal the scenario does not drive contributes nothing. In
	// CouplingMatrix mode the scenario carries a single unit excitation, so
	// this IS the drive for that column rather than background data, and must
	// not be suppressed the way boundary data is.
	void BuildCurrentDensity(
		const Scenario& sc,
		const std::function<bool(const Terminal&)>& include,
		mfem::Vector& j_re, mfem::Vector& j_im) const;
};
