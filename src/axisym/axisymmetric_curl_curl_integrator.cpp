// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "axisymmetric_curl_curl_integrator.hpp"

 const mfem::IntegrationRule &AxisymmetricCurlCurlIntegrator::GetRule(
   const mfem::FiniteElement &trial_fe,
   const mfem::FiniteElement &test_fe,
   const mfem::ElementTransformation &Trans,
   mfem::real_t axis_tolerance)
{
   // Polynomial part: integrated exactly by the usual order estimate.
   const int gradient_order = Trans.OrderGrad(&trial_fe)
      + Trans.OrderGrad(&test_fe) + Trans.Order();
   const int radial_reaction_order = trial_fe.GetOrder()
      + test_fe.GetOrder() + Trans.Order() + Trans.OrderW();
   const int polynomial_order = std::max(gradient_order, radial_reaction_order);

   // The N_j N_k / r part: see radial_quadrature.hpp.
   return axisym::RadialRule(trial_fe.GetGeomType(), polynomial_order, Trans,
                axis_tolerance);
}

void AxisymmetricCurlCurlIntegrator::AssembleElementMatrix(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &Trans,
                           /*out*/ mfem::DenseMatrix &elmat) 
{
   const int ndof  = el.GetDof();
   const int dim = el.GetDim();

   MFEM_VERIFY(dim == 2 && Trans.GetSpaceDim() == 2,
      "AxisymmetricCurlCurlIntegrator expects a 2D (r,z) finite element.");

   elmat.SetSize(ndof);
   elmat = 0.0;

   // Scratch storage, local to this call
   mfem::Vector      shape(ndof);
   mfem::DenseMatrix dshape_ref(ndof, dim);
   mfem::DenseMatrix dshape_phys(ndof, dim);
   mfem::Vector      pos(dim);

   const mfem::IntegrationRule *ir = GetIntegrationRule(el, Trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);

      // Physical coordinates: pos(0)=r, pos(1)=z
      Trans.Transform(ip, pos);
      const mfem::real_t r = pos(0);

      // Interior quadrature keeps r > 0 even for elements touching the axis.
      // A non-positive radius here means an invalid mesh or a custom rule
      // with boundary points; clamping would silently deform the geometry.
      MFEM_VERIFY(r > 0.0,
         "AxisymmetricCurlCurlIntegrator evaluated at a non-positive radius (r = "
         << r << "). The 1/r term is singular there: use an interior "
         "integration rule and a mesh with non-negative radii.");

      const mfem::real_t nu = nu_->Eval(Trans, ip);

      // Axisymmetric weight: ip.weight * detJ * 2*pi*r * nu
      const mfem::real_t w = ip.weight * Trans.Weight()
         * Axisymmetric::Measure(r) * nu;

      el.CalcShape(ip, /*out*/ shape);
      el.CalcDShape(ip, /*out*/ dshape_ref);

      // Map row-oriented reference derivatives to physical derivatives.
      // dshape_phys = dshape_ref * InvJ
      Mult(dshape_ref, Trans.InverseJacobian(), /*out*/ dshape_phys);

      for (int j = 0; j < ndof; j++)
      {
         const mfem::real_t Nj     = shape(j);
         const mfem::real_t dNj_dr = dshape_phys(j, 0);
         const mfem::real_t dNj_dz = dshape_phys(j, 1);

         for (int k = j; k < ndof; k++)
         {
            const mfem::real_t Nk     = shape(k);
            const mfem::real_t dNk_dr = dshape_phys(k, 0);
            const mfem::real_t dNk_dz = dshape_phys(k, 1);

            const mfem::real_t Br_j = -dNj_dz;
            const mfem::real_t Bz_j = dNj_dr + Nj / r;

            const mfem::real_t Br_k = -dNk_dz;
            const mfem::real_t Bz_k = dNk_dr + Nk / r;

            const mfem::real_t val = Br_j * Br_k + Bz_j * Bz_k;

            elmat(j, k) += w * val;
         }
      }
   }

   // Symmetrize
   for (int j = 0; j < ndof; j++)
   {
      for (int k = 0; k < j; k++)
      {
         elmat(j, k) = elmat(k, j);
      }
   }
}

void AxisymmetricCurlCurlIntegrator::ComputeElementFlux(const mfem::FiniteElement& el,
    mfem::ElementTransformation& Trans,
    mfem::Vector& u,
    const mfem::FiniteElement& flux_elem,
    mfem::Vector& flux,
    bool with_coef,  // the base class's default
    const mfem::IntegrationRule* ir) 
{
    const int ndof = el.GetDof();
    const int dim = el.GetDim();
    const int space_dim = Trans.GetSpaceDim();

    MFEM_VERIFY(dim == 2 && space_dim == 2,
        "AxisymmetricCurlCurlIntegrator expects a 2D (r,z) mesh.");
    MFEM_ASSERT(u.Size() == ndof,
        "Element solution has an unexpected size.");

    if (!ir)
    {
        ir = &flux_elem.GetNodes();
    }

    const int flux_nd = ir->GetNPoints();
    flux.SetSize(flux_nd * space_dim);

    mfem::Vector shape(ndof);
    mfem::DenseMatrix dshape_ref(ndof, dim);
    // Inverse Jacobian: (dim x space_dim), NOT shaped like dshape.
    mfem::DenseMatrix inv_jacobian(dim, space_dim);
    mfem::Vector grad_ref(dim);
    mfem::Vector grad_phys(space_dim);
    mfem::Vector pos(space_dim);

    for (int i = 0; i < flux_nd; ++i)
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);

        el.CalcShape(ip, shape);
        el.CalcDShape(ip, dshape_ref);

        dshape_ref.MultTranspose(u, grad_ref);
        mfem::CalcInverse(Trans.Jacobian(), inv_jacobian);
        inv_jacobian.MultTranspose(grad_ref, grad_phys);

        Trans.Transform(ip, pos);
        const mfem::real_t r = pos(0);
        const mfem::real_t A_phi = shape * u;

        // curl(A_phi e_phi) in the (r,z) component ordering, with the axis
        // limit applied under the shared scale-relative tolerance.
        mfem::real_t B_r = -grad_phys(1);
        mfem::real_t B_z = axisym::AxialFluxDensity(A_phi, grad_phys(0), r, axis_tolerance_);

        if (with_coef)
        {
            const mfem::real_t nu = nu_->Eval(Trans, ip);
            B_r *= nu;
            B_z *= nu;
        }

        // MFEM vector GridFunction element data is component-major.
        flux(i) = B_r;
        flux(flux_nd + i) = B_z;
    }
}

mfem::real_t AxisymmetricCurlCurlIntegrator::ComputeFluxEnergy(const mfem::FiniteElement& flux_elem,
    mfem::ElementTransformation& Trans,
    mfem::Vector& flux,
    mfem::Vector* d_energy) 
{
    const int nd = flux_elem.GetDof();
    const int space_dim = Trans.GetSpaceDim();

    MFEM_VERIFY(space_dim == 2,
        "AxisymmetricCurlCurlIntegrator expects a 2D (r,z) mesh.");
    MFEM_ASSERT(flux.Size() == nd * space_dim,
        "Flux vector has an unexpected size.");

    mfem::Vector shape(nd);
    mfem::Vector point_flux(space_dim);
    mfem::Vector pos(space_dim);

    const int order = 2 * flux_elem.GetOrder()
        + Trans.Order() + Trans.OrderW();

    const mfem::IntegrationRule* ir = GetIntRule();
    if (!ir)
    {
        ir = &mfem::IntRules.Get(flux_elem.GetGeomType(), order);
    }

    if (d_energy)
    {
        d_energy->SetSize(space_dim);
        *d_energy = 0.0;
    }

    mfem::real_t energy = 0.0;

    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const mfem::IntegrationPoint& ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);

        flux_elem.CalcPhysShape(Trans, shape);

        point_flux = 0.0;
        for (int component = 0; component < space_dim; ++component)
        {
            for (int j = 0; j < nd; ++j)
            {
                point_flux(component) +=
                    flux(component * nd + j) * shape(j);
            }
        }

        Trans.Transform(ip, pos);
        // No clamp needed: this term carries the measure r, never 1/r, so
        // the contribution simply vanishes on the axis.
        const mfem::real_t r = pos(0);
        const mfem::real_t nu = nu_->Eval(Trans, ip);
        const mfem::real_t weight = ip.weight * Trans.Weight()
            * Axisymmetric::Measure(r);

        energy += weight * nu * (point_flux * point_flux);
    }

    return energy;
}
