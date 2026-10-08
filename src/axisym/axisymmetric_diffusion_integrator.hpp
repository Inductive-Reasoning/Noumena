// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include "mfem.hpp"
#include "axisymmetric_measure.hpp"

// -----------------------------------------------------------------------------
// Stiffness Integrator: -div( eps * grad(u) )
// -----------------------------------------------------------------------------
// Solves: Integral( eps * grad(u) . grad(v) * 2*pi*r * dr * dz )
//
// Carries the full axisymmetric measure (see axisymmetric_measure.hpp), so the
// assembled operator is physical: the gathered terminal charge is in coulombs
// with no post-hoc scaling.
//
// Implements the Zienkiewicz-Zhu flux/energy hooks (ComputeElementFlux /
// ComputeFluxEnergy) so MFEM's recovery-based error estimators
// (ZienkiewiczZhuEstimator) drive AMR with an axisymmetrically-consistent error.
// The pointwise flux is the physical density eps*grad(u); the energy integral
// carries the same 2*pi*r measure as AssembleElementMatrix so refinement
// concentrates by physical (r-z) field energy rather than a planar approximation.
class AxisymmetricDiffusionIntegrator : public mfem::BilinearFormIntegrator
{
private:
   mfem::Coefficient *Q;

public:
   explicit AxisymmetricDiffusionIntegrator(
      mfem::Coefficient &q, const mfem::IntegrationRule *ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), Q(&q)
   {
      MFEM_ASSERT(Q != nullptr, "Coefficient cannot be null");
   }

   static const mfem::IntegrationRule &GetRule(
      const mfem::FiniteElement &trial_fe,
      const mfem::FiniteElement &test_fe,
      const mfem::ElementTransformation &Trans);

   void AssembleElementMatrix(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &Trans,
                               mfem::DenseMatrix &elmat) override;

   // Pointwise discrete flux flux = eps*grad(u), evaluated at the flux element's
   // nodes (or @p ir when supplied). Mirrors mfem::DiffusionIntegrator's
   // scalar-coefficient path, including the component-major layout
   // flux(fnd*j + i) the ZZ estimator expects. The flux is a physical density,
   // so it carries NO geometric measure here (the 2*pi*r weight lives in the
   // energy integral below, matching AssembleElementMatrix).
   void ComputeElementFlux(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &Trans,
                           mfem::Vector &u,
                           const mfem::FiniteElement &fluxelem,
                           mfem::Vector &flux, bool with_coef = true,
                           const mfem::IntegrationRule *ir = nullptr) override;

   // Energy norm of a flux expansion: Integral( eps * |flux|^2 * 2*pi*r ).
   // Mirrors mfem::DiffusionIntegrator (scalar Q) but applies the axisymmetric
   // radial measure so the ZZ error indicator sqrt(energy) reflects the physical
   // r-z field. Anisotropic splitting (d_energy) is not supported.
   //
   // The return type MUST be mfem::real_t, not double. The base class declares
   // `virtual real_t ComputeFluxEnergy(...)`, and real_t is float in an
   // MFEM_USE_SINGLE build. A double here would stop overriding, and the ZZ
   // estimator would silently fall back to the base implementation, driving AMR
   // with a planar (non-axisymmetric) error measure and no diagnostic.
   mfem::real_t ComputeFluxEnergy(const mfem::FiniteElement &fluxelem,
                            mfem::ElementTransformation &Trans,
                            mfem::Vector &flux,
                            mfem::Vector *d_energy = nullptr) override;

protected:
   const mfem::IntegrationRule *GetDefaultIntegrationRule(
      const mfem::FiniteElement &trial_fe,
      const mfem::FiniteElement &test_fe,
      const mfem::ElementTransformation &Trans) const override
   {
      return &GetRule(trial_fe, test_fe, Trans);
   }
};