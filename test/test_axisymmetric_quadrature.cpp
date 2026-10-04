// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "axisym/axisymmetric_curl_curl_integrator.hpp"
#include "axisym/axisymmetric_diffusion_integrator.hpp"
#include "axisym/axisymmetric_lf_integrator.hpp"
#include "axisym/axisymmetric_mass_integrator.hpp"
#include "axisym/axisymmetric_boundary_lf_integrator.hpp"
#include "axisym/axisymmetric_mesh_validation.hpp"
#include "axisym/magnetic_axis_boundary.hpp"
#include "core/constants.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

namespace {

std::unique_ptr<mfem::Mesh> MakeAxisymmetricMesh(bool curved)
{
   auto mesh = std::make_unique<mfem::Mesh>(
	  mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
								  true, 1.0, 1.0));

   for (int v = 0; v < mesh->GetNV(); ++v)
   {
	  mesh->GetVertex(v)[0] += 1.0;
   }

   if (curved)
   {
	  mesh->SetCurvature(2);
	  mfem::VectorFunctionCoefficient deformation(
		 2, [](const mfem::Vector &x, mfem::Vector &mapped)
		 {
			mapped(0) = x(0);
			mapped(1) = x(1) + 0.1 * (x(0) - 1.0) * (2.0 - x(0));
		 });
	  mesh->Transform(deformation);
   }

   return mesh;
}

double RelativeDifference(const mfem::DenseMatrix &actual,
						  const mfem::DenseMatrix &reference)
{
   double difference_squared = 0.0;
   double reference_squared = 0.0;
   for (int row = 0; row < actual.Height(); ++row)
   {
	  for (int column = 0; column < actual.Width(); ++column)
	  {
		 const double difference = actual(row, column) - reference(row, column);
		 difference_squared += difference * difference;
		 reference_squared += reference(row, column) * reference(row, column);
	  }
   }

   return std::sqrt(difference_squared / reference_squared);
}

double RelativeDifference(const mfem::Vector &actual,
						  const mfem::Vector &reference)
{
   mfem::Vector difference(actual);
   difference -= reference;
   return difference.Norml2() / reference.Norml2();
}

void CheckAutomaticQuadrature(bool curved)
{
   auto mesh = MakeAxisymmetricMesh(curved);
   mfem::H1_FECollection collection(2, mesh->Dimension());
   mfem::FiniteElementSpace space(mesh.get(), &collection);

   const mfem::FiniteElement &element = *space.GetFE(0);
   mfem::ElementTransformation &transformation =
	  *mesh->GetElementTransformation(0);
   const int reference_order = 2 * element.GetOrder()
	  + transformation.Order() + transformation.OrderW() + 6;
   const mfem::IntegrationRule &reference_rule =
	  mfem::IntRules.Get(element.GetGeomType(), reference_order);
   mfem::ConstantCoefficient coefficient(1.0);

   mfem::DenseMatrix automatic_matrix;
   mfem::DenseMatrix reference_matrix;

   AxisymmetricMassIntegrator automatic_mass(coefficient);
	  AxisymmetricMassIntegrator reference_mass(coefficient, &reference_rule);
   automatic_mass.AssembleElementMatrix(element, transformation, automatic_matrix);
   reference_mass.AssembleElementMatrix(element, transformation, reference_matrix);
   REQUIRE(RelativeDifference(automatic_matrix, reference_matrix) < 1.0e-11);

   AxisymmetricDiffusionIntegrator automatic_diffusion(coefficient);
   AxisymmetricDiffusionIntegrator reference_diffusion(coefficient);
   reference_diffusion.SetIntRule(&reference_rule);
   automatic_diffusion.AssembleElementMatrix(
		element, transformation, automatic_matrix);
   reference_diffusion.AssembleElementMatrix(
		element, transformation, reference_matrix);
   REQUIRE(RelativeDifference(automatic_matrix, reference_matrix) < 1.0e-9);

   AxisymmetricCurlCurlIntegrator automatic_curl(coefficient, 0.0);
	 AxisymmetricCurlCurlIntegrator reference_curl(coefficient, 0.0);
   // The curl-curl rule now adapts to element geometry and can exceed the
   // shared reference order, so it needs a strictly finer reference.
   const mfem::IntegrationRule &curl_reference_rule =
	  mfem::IntRules.Get(element.GetGeomType(), 2 * reference_order + 40);
   reference_curl.SetIntRule(&curl_reference_rule);
   automatic_curl.AssembleElementMatrix(element, transformation, automatic_matrix);
   reference_curl.AssembleElementMatrix(element, transformation, reference_matrix);
	  // The 1/r reaction term is non-polynomial, but the automatic rule adds a
   // geometry-dependent order for it, so on this well-separated element
   // (r_min/width = 1) it is resolved to round-off rather than merely close.
   REQUIRE(RelativeDifference(automatic_matrix, reference_matrix) < 1.0e-11);

   mfem::Vector automatic_vector;
   mfem::Vector reference_vector;
   AxisymmetricLFIntegrator automatic_load(coefficient);
   AxisymmetricLFIntegrator reference_load(coefficient);
   reference_load.SetIntRule(&reference_rule);
   automatic_load.AssembleRHSElementVect(
	  element, transformation, automatic_vector);
   reference_load.AssembleRHSElementVect(
	  element, transformation, reference_vector);
   REQUIRE(RelativeDifference(automatic_vector, reference_vector) < 1.0e-11);
}

} // namespace

TEST_CASE("Axisymmetric quadrature accounts for element transformations",
		  "[axisymmetric][quadrature]")
{
   SECTION("affine geometry")
   {
	  CheckAutomaticQuadrature(false);
   }

   SECTION("curved quadratic geometry")
   {
	  CheckAutomaticQuadrature(true);
   }
}

namespace {

// One element spanning r in [s*h, s*h + h], z in [0, 1].
std::unique_ptr<mfem::Mesh> MakeRadialBand(double r_min, double width)
{
   auto mesh = std::make_unique<mfem::Mesh>(
	  mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
								  true, width, 1.0));
   for (int v = 0; v < mesh->GetNV(); ++v)
   {
	  mesh->GetVertex(v)[0] += r_min;
   }
   return mesh;
}

double ConstantPotential(const mfem::Vector &) { return 1.0; }

// u^T K u for A_phi = 1, the magnetic energy the stiffness matrix represents.
double CurlCurlEnergy(mfem::Mesh &mesh)
{
   mfem::H1_FECollection collection(2, mesh.Dimension());
   mfem::FiniteElementSpace space(&mesh, &collection);
   mfem::ConstantCoefficient one(1.0);

   mfem::BilinearForm a(&space);
   a.AddDomainIntegrator(new AxisymmetricCurlCurlIntegrator(one, 0.0));
   a.Assemble();
   a.Finalize();

   mfem::GridFunction u(&space);
   mfem::FunctionCoefficient potential(ConstantPotential);
   u.ProjectCoefficient(potential);

   mfem::Vector Ku(u.Size());
   a.SpMat().Mult(u, Ku);
   return Ku * u;
}

} // namespace

TEST_CASE("Radial conductance and the Joule quadratic share accurate positive quadrature",
          "[axisymmetric][quadrature][power]") {
   for (auto type : {mfem::Element::TRIANGLE, mfem::Element::QUADRILATERAL})
   for (int order : {1, 2})
   for (double inner : {0.001, 0.1, 3.0})
   for (bool curved : {false, true}) {
      auto mesh = mfem::Mesh::MakeCartesian2D(2, 1, type, true, 1, 1);
      for (int e = 0; e < mesh.GetNE(); ++e) {
         mfem::Vector center;
         mesh.GetElementCenter(e, center);
         mesh.GetElement(e)->SetAttribute(center(0) < 0.5 ? 1 : 2);
      }
      mesh.SetAttributes();
      mesh.SetCurvature(2);
      mfem::VectorFunctionCoefficient map(2,
         [=](const mfem::Vector& p, mfem::Vector& mapped) {
            mapped(0) = inner + p(0);
            mapped(1) = p(1) + (curved ? 0.1 * p(0) * (1 - p(0)) : 0);
         });
      mesh.Transform(map);
      mfem::H1_FECollection collection(order, 2);
      mfem::FiniteElementSpace space(&mesh, &collection);
      mfem::Vector material(2);
      material(0) = 1; material(1) = 3;
      mfem::PWConstCoefficient sigma(material);
      mfem::BilinearForm mass(&space);
      mass.AddDomainIntegrator(new AxisymmetricMassIntegrator(sigma));
      mass.Assemble(); mass.Finalize();
      mfem::LinearForm coupling(&space);
      coupling.AddDomainIntegrator(new axisym::PortLoadIntegrator(sigma));
      coupling.Assemble();
      mfem::GridFunction ar(&space), ai(&space);
      for (int d = 0; d < ar.Size(); ++d) {
         ar(d) = std::sin(0.3 * d);
         ai(d) = std::cos(0.2 * d);
      }
      constexpr double vr = 0.7, vi = -0.2, omega = 3.0;
      double conductance = 0, power = 0;
      for (int e = 0; e < mesh.GetNE(); ++e) {
         auto& t = *mesh.GetElementTransformation(e);
         const auto& rule = axisym::RadialRule(*space.GetFE(e), t);
         for (int q = 0; q < rule.GetNPoints(); ++q) {
            const auto& ip = rule.IntPoint(q);
            t.SetIntPoint(&ip);
            mfem::Vector position;
            t.Transform(ip, position);
            const double radius = position(0);
            const double weight = ip.weight * t.Weight() * sigma.Eval(t, ip);
            conductance += weight / (Constants::TWO_PI * radius);
            const double er = vr / (Constants::TWO_PI * radius) + omega * ai.GetValue(e, ip);
            const double ei = vi / (Constants::TWO_PI * radius) - omega * ar.GetValue(e, ip);
            power += weight * Constants::TWO_PI * radius * (er * er + ei * ei);
         }
      }
      const double exact = (std::log((inner + 0.5) / inner)
         + 3 * std::log((inner + 1) / (inner + 0.5))) / Constants::TWO_PI;
      mfem::Vector mr(ar.Size()), mi(ai.Size());
      mass.SpMat().Mult(ar, mr); mass.SpMat().Mult(ai, mi);
      const double quadratic = conductance * (vr * vr + vi * vi)
         + omega * omega * (ar * mr + ai * mi)
         + 2 * omega * (vr * (coupling * ai) - vi * (coupling * ar));
      INFO("inner=" << inner << ", order=" << order << ", curved=" << curved);
      REQUIRE(conductance == Catch::Approx(exact).epsilon(1e-10));
      REQUIRE(power >= 0);
      REQUIRE(power == Catch::Approx(quadratic).epsilon(1e-11));
   }
}

// The 1/r term is non-polynomial and its quadrature difficulty is governed by
// the element's geometry, s = r_min/width, not by the basis degree: mapped to
// the reference interval, 1/r has a pole at -(1 + 2s), which approaches the
// integration interval as s -> 0. A basis-degree-only rule is therefore blind
// to the hard case, and a thin band close to the axis was integrated with tens
// of percent of error. A_phi = 1 does not vanish near the axis, so it exercises
// the term directly, and its energy has the closed form 2*pi*ln(b/a).
TEST_CASE("Near-axis annular curl-curl quadrature meets its accuracy target",
		  "[axisymmetric][quadrature]")
{
   const double width = 1.0;

   // Include the former capped-rule threshold as a regression point.
   const double ratios[] = {
	  3.0, 1.0, 0.1, 0.03,
	  AxisymmetricCurlCurlIntegrator::kResolvedRadiusRatio};

   for (double s : ratios)
   {
	  const double a = s * width;
	  const double b = a + width;
	  auto mesh = MakeRadialBand(a, width);

	  const double exact = Constants::TWO_PI * std::log(b / a);
	  const double relative_error =
		 std::abs(CurlCurlEnergy(*mesh) - exact) / exact;

	  INFO("r_min/width = " << s);
	  REQUIRE(relative_error < 1.0e-10);
   }
}

// Enrichment must resolve the near-axis band rather than silently saturate.
TEST_CASE("Curl-curl quadrature resolves the formerly capped radial band",
		  "[axisymmetric][quadrature]")
{
   const double width = 1.0;
   const double s = 1.0e-3;
   const double a = s * width;
   auto mesh = MakeRadialBand(a, width);

   const double exact = Constants::TWO_PI * std::log((a + width) / a);
   const double relative_error =
	  std::abs(CurlCurlEnergy(*mesh) - exact) / exact;

   REQUIRE(s < AxisymmetricCurlCurlIntegrator::kResolvedRadiusRatio);
   REQUIRE(relative_error < 1.0e-10);
}

TEST_CASE("Radial quadrature checks curved radii and reports unresolved geometry",
          "[axisymmetric][quadrature]") {
   auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL);
   mesh.SetCurvature(2);
   constexpr double inner = 0.01;
   mfem::VectorFunctionCoefficient deformation(2,
      [](const mfem::Vector& x, mfem::Vector& mapped) {
         mapped(0) = inner + x(0) + 0.2 * x(1) * (1 - x(1));
         mapped(1) = x(1);
      });
   mesh.Transform(deformation);
   mfem::H1_FECollection collection(2, 2);
   mfem::FiniteElementSpace space(&mesh, &collection);
   auto& t = *mesh.GetElementTransformation(0);
   const auto& rule = axisym::RadialRule(*space.GetFE(0), t);
   double integral = 0.0, reference = 0.0;
   for (int q = 0; q < rule.GetNPoints(); ++q) {
      const auto& ip = rule.IntPoint(q);
      t.SetIntPoint(&ip);
      mfem::Vector position;
      t.Transform(ip, position);
      integral += ip.weight * t.Weight() / position(0);
   }
   const auto& line = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 255);
   for (int q = 0; q < line.GetNPoints(); ++q) {
      const auto& ip = line.IntPoint(q);
      const double a = inner + 0.2 * ip.x * (1 - ip.x);
      reference += ip.weight * std::log((a + 1) / a);
   }
   REQUIRE(integral == Catch::Approx(reference).epsilon(1e-10));
   auto unresolved = MakeRadialBand(1e-8, 1.0);
   REQUIRE(axisym::ValidateMesh(*unresolved).relation == axisym::AxisRelation::Annular);
   REQUIRE_THROWS(CurlCurlEnergy(*unresolved));
}

TEST_CASE("Radial quadrature shares the constrained magnetic axis tolerance",
          "[axisymmetric][quadrature][axis]") {
   for (auto type : {mfem::Element::TRIANGLE, mfem::Element::QUADRILATERAL})
   for (int order : {1, 2})
   for (double height : {1.0, 1000.0})
   for (double offset : {-1e-12 * height, 0.0, 1e-12 * height}) {
      auto mesh = mfem::Mesh::MakeCartesian2D(2, 4, type, true, 1.0, height);
      for (int v = 0; v < mesh.GetNV(); ++v) mesh.GetVertex(v)[0] += offset;
      const auto geometry = axisym::ValidateMesh(mesh);
      REQUIRE(geometry.TouchesAxis());
      mfem::H1_FECollection collection(order, 2);
      mfem::FiniteElementSpace space(&mesh, &collection);
      mfem::Array<int> essential;
      space.GetEssentialTrueDofs(axisym::FindAxisBoundaryMarker(mesh, geometry), essential);
      REQUIRE(essential.Size() > 0);
      mfem::GridFunction potential(&space);
      mfem::FunctionCoefficient profile([=](const mfem::Vector& x) { return x(0) - offset; });
      potential.ProjectCoefficient(profile);
      for (int i = 0; i < essential.Size(); ++i) potential(essential[i]) = 0.0;
      mfem::ConstantCoefficient one(1.0);
      mfem::BilinearForm stiffness(&space), mass(&space);
      stiffness.AddDomainIntegrator(new AxisymmetricCurlCurlIntegrator(one, geometry.tolerance));
      mass.AddDomainIntegrator(new AxisymmetricMassIntegrator(one, nullptr, geometry.tolerance));
      REQUIRE_NOTHROW(stiffness.Assemble());
      stiffness.Finalize();
      REQUIRE_NOTHROW(mass.Assemble());
      mass.Finalize();
      mfem::Vector product(potential.Size());
      stiffness.SpMat().Mult(potential, product);
      REQUIRE(potential * product == Catch::Approx(2 * Constants::TWO_PI * height).epsilon(1e-8));
      mass.SpMat().Mult(potential, product);
      REQUIRE(potential * product == Catch::Approx(Constants::TWO_PI * height / 4).epsilon(1e-8));
      REQUIRE(mesh.GetVertex(0)[0] == offset);
   }
}

TEST_CASE("Axis tolerance does not admit nonpositive integration radii",
          "[axisymmetric][quadrature][axis]") {
   auto mesh = MakeRadialBand(-0.1, 1.0);
   mfem::H1_FECollection collection(1, 2);
   mfem::FiniteElementSpace space(mesh.get(), &collection);
   auto& t = *mesh->GetElementTransformation(0);
   REQUIRE_THROWS(axisym::RadialRule(*space.GetFE(0), t, 1e-10));
   // Even when vertices fall within a supplied tolerance, physical interior
   // coordinates must remain strictly positive.
   REQUIRE_THROWS(axisym::RadialRule(*space.GetFE(0), t, 0.2));
}

TEST_CASE("Axisymmetric boundary load includes radial measure",
		  "[axisymmetric][quadrature][boundary]")
{
   auto mesh = MakeAxisymmetricMesh(false); // unit square shifted to 1 <= r <= 2
   mfem::H1_FECollection collection(1, mesh->Dimension());
   mfem::FiniteElementSpace space(mesh.get(), &collection);
   mfem::ConstantCoefficient unit_load(1.0);
   mfem::LinearForm load(&space);
   load.AddBoundaryIntegrator(new AxisymmetricBoundaryLFIntegrator(unit_load));
   load.Assemble();

   // Partition of unity gives the full measure integral_boundary 2*pi*r ds:
   // bottom + top + left + right = 3/2 + 3/2 + 1 + 2 = 6, times 2*pi.
   REQUIRE(load.Sum()
           == Catch::Approx(Constants::TWO_PI * 6.0).epsilon(1.0e-12));
}

TEST_CASE("Curl-curl quadrature stays on positive-weight simplex rules",
          "[axisymmetric][quadrature][curlcurl]")
{
   // This near-axis triangle needs enrichment beyond MFEM's positive-weight
   // simplex tables, so it exercises the positive Duffy construction.
   mfem::Mesh mesh(2, 3, 1, 0, 2);
   mesh.AddVertex(1.0e-3, 0.0);
   mesh.AddVertex(1.0, 0.0);
   mesh.AddVertex(1.0e-3, 1.0);
   mesh.AddTriangle(0, 1, 2, 1);
   mesh.FinalizeTriMesh(1, 0, true);

   mfem::H1_FECollection collection(2, mesh.Dimension());
   mfem::FiniteElementSpace space(&mesh, &collection);
   const mfem::FiniteElement &element = *space.GetFE(0);
   mfem::ElementTransformation &transformation =
      *mesh.GetElementTransformation(0);

   // Positive Duffy rules enrich beyond the simplex tables.
   const mfem::IntegrationRule &rule =
      AxisymmetricCurlCurlIntegrator::GetRule(element, element, transformation);
   REQUIRE(rule.GetOrder()
           > AxisymmetricCurlCurlIntegrator::kMaxPositiveWeightSimplexOrder);

   double min_weight = rule.IntPoint(0).weight;
   for (int i = 1; i < rule.GetNPoints(); ++i)
   {
      min_weight = std::min(min_weight, (double)rule.IntPoint(i).weight);
   }
   REQUIRE(min_weight > 0.0);

   // Positive quadrature preserves the assembled operator's positive definiteness.
   // Tested by attempting a Cholesky factorization, which succeeds exactly
   // when the matrix is positive definite. MFEM here is built without LAPACK,
   // so this is also the check that does not need an eigensolver.
   mfem::ConstantCoefficient reluctivity(1.0);
   AxisymmetricCurlCurlIntegrator integrator(reluctivity, 0.0);
   mfem::DenseMatrix matrix;
   integrator.AssembleElementMatrix(element, transformation, matrix);

   const int n = matrix.Height();
   mfem::DenseMatrix factor(n);
   factor = 0.0;
   bool positive_definite = true;
   for (int j = 0; j < n && positive_definite; ++j)
   {
      for (int i = j; i < n; ++i)
      {
         // Symmetrized entry: assembly is symmetric only up to round-off.
         double sum = 0.5 * (matrix(i, j) + matrix(j, i));
         for (int k = 0; k < j; ++k) { sum -= factor(i, k) * factor(j, k); }
         if (i == j)
         {
            if (!(sum > 0.0)) { positive_definite = false; break; }
            factor(j, j) = std::sqrt(sum);
         }
         else
         {
            factor(i, j) = sum / factor(j, j);
         }
      }
   }
   REQUIRE(positive_definite);
}
