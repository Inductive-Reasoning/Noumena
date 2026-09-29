// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include "mfem.hpp"

// -----------------------------------------------------------------------------
// Energy Density Coefficient
// -----------------------------------------------------------------------------
// Computes local energy density: w = 0.5 * eps * |E|^2
// Useful for visualization. Note: This does NOT include 2*pi*r. 
// It represents the physical density at that point in space.
class EnergyDensityCoefficient : public mfem::Coefficient
{
private:
   mfem::GridFunction *Phi;
   mfem::Coefficient *Eps;

public:
   EnergyDensityCoefficient(mfem::GridFunction &phi, mfem::Coefficient &eps) 
      : Phi(&phi), Eps(&eps) { }

   double Eval(mfem::ElementTransformation &T, 
               const mfem::IntegrationPoint &ip) override
   {
      // GetGradient() evaluates at T's CURRENT integration point, which it reads
      // back via T.GetIntPoint(). Binding ip here (as every built-in MFEM
      // coefficient does) is what makes this Eval() honour its own argument
      // instead of whatever point happened to be set last.
      T.SetIntPoint(&ip);

      mfem::Vector grad;
      Phi->GetGradient(T, grad); // Sized by GetGradient to the space dimension

      double mag_sq = grad * grad; // |E|^2
      double eps_val = Eps->Eval(T, ip);

      return 0.5 * eps_val * mag_sq;
   }
};
