// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "mfem.hpp"
#include "axisymmetric_measure.hpp"
#include "axisymmetric_field_relations.hpp"
#include "radial_quadrature.hpp"

/**
 * @brief Axisymmetric curl-curl bilinear form integrator for magnetostatics
 *        with A = A_phi(r,z) e_phi.
 *
 * Assembles, with the full axisymmetric measure (see axisymmetric_measure.hpp):
 *
 *   integral nu * [dA_phi/dz * dv/dz + (dA_phi/dr + A_phi/r) * (dv/dr + v/r)] * 2*pi*r dr dz
 *
 * This follows from
 *
 *   curl(A_phi e_phi) = -dA_phi/dz e_r + (dA_phi/dr + A_phi/r) e_z.
 *
 * IMPORTANT:
 *   Enforce the essential BC A_phi = 0 on the symmetry axis r = 0 (regularity).
 *   This is required whenever the domain closure reaches r = 0; annular domains
 *   (r_min > 0) need no axis condition at all.
 *
 * Notes:
 *   - Assembly keeps its scratch storage local, but the axis-aware quadrature
 *     (axisym::RadialRule) mutates the element transformation and fills MFEM's
 *     global rule caches, so elements must not be assembled concurrently.
 *   - Assembly does NOT clamp r. Standard interior quadrature keeps r > 0 even for
 *     elements touching the axis, so a zero radius there signals a bad custom rule
 *     or a bad mesh and is reported rather than papered over. The 1/r term must be
 *     kept exact per basis function: individual shape functions need not vanish on
 *     the axis, so no per-basis limit exists. Regularity is a property of the
 *     constrained solution and is delivered by essential BC elimination.
 *   - Flux/energy recovery, which does see the constrained solution, uses the exact
 *     limit B_z -> 2 dA/dr on the axis (see ComputeElementFlux). Both that path and
 *     MagneticFieldCoefficient call axisym::AxialFluxDensity with the mesh's
 *     scale-relative axis tolerance, so recovery, postprocessing, and geometry
 *     classification cannot disagree about what counts as the axis.
 */
class AxisymmetricCurlCurlIntegrator : public mfem::BilinearFormIntegrator
{
public:
   // @param axis_tolerance  Scale-relative radius below which flux recovery takes
   //                        the axis limit; use axisym::AxisGeometry::tolerance so the
   //                        integrator shares the mesh's axis policy.
   explicit AxisymmetricCurlCurlIntegrator(
      mfem::Coefficient &reluctivity,
      mfem::real_t axis_tolerance,
      const mfem::IntegrationRule *ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), nu_(&reluctivity),
        axis_tolerance_(axis_tolerance)
   {
      MFEM_ASSERT(nu_ != nullptr, "Reluctivity coefficient cannot be null");
   }

   static const mfem::IntegrationRule &GetRule(
      const mfem::FiniteElement &trial_fe,
      const mfem::FiniteElement &test_fe,
      const mfem::ElementTransformation &Trans,
      mfem::real_t axis_tolerance);

   void AssembleElementMatrix(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &Trans,
                              /*out*/ mfem::DenseMatrix &elmat) override;

   void ComputeElementFlux(const mfem::FiniteElement& el,
       mfem::ElementTransformation& Trans,
       mfem::Vector& u,
       const mfem::FiniteElement& flux_elem,
       mfem::Vector& flux,
       bool with_coef = true,  // the base class's default
       const mfem::IntegrationRule* ir = nullptr) override;

   // Returns mfem::real_t to match the base class declaration exactly; see the
   // note on AxisymmetricDiffusionIntegrator::ComputeFluxEnergy for why a
   // double here would silently stop overriding in a single-precision build.
   //
   // The energy is integral nu |flux|^2 r: it takes the flux of B, as from
   // ComputeElementFlux with with_coef = false, the convention of MFEM's
   // DiffusionIntegrator (Q |flux|^2), so an estimator using it calls
   // SetWithCoeff(false). Anisotropic estimation is not supported: d_energy
   // comes back zero.
   mfem::real_t ComputeFluxEnergy(const mfem::FiniteElement& flux_elem,
       mfem::ElementTransformation& Trans,
       mfem::Vector& flux,
       mfem::Vector* d_energy = nullptr) override;

protected:
   const mfem::IntegrationRule *GetDefaultIntegrationRule(
      const mfem::FiniteElement &trial_fe,
      const mfem::FiniteElement &test_fe,
      const mfem::ElementTransformation &Trans) const override
   {
      return &GetRule(trial_fe, test_fe, Trans, axis_tolerance_);
   }

private:
   mfem::Coefficient *nu_;
   mfem::real_t axis_tolerance_;
};

