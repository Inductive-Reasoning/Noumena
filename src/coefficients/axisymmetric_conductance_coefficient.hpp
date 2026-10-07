#pragma once

#include "mfem.hpp"
#include "../core/constants.hpp"

// Custom coefficient for axisymmetric G_dc = sigma / (2 * pi * r)
//
// This integrand is genuinely singular on the symmetry axis: a toroidal massive
// conductor whose cross-section reaches r = 0 has no finite DC conductance, and
// there is no limit to substitute. Rather than silently returning zero (which
// would quietly under-report the conductance of the whole port), an axis-touching
// evaluation is reported as the modelling error it is. Callers are expected to
// reject such ports up front using the mesh radial extent.
class AxisymmetricConductanceCoeff : public mfem::Coefficient
{
private:
    mfem::Coefficient& conductivity;
public:
    explicit AxisymmetricConductanceCoeff(mfem::Coefficient& sigma)
        : conductivity(sigma) { }

    virtual mfem::real_t Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
    {
        T.SetIntPoint(&ip);
        mfem::Vector transip;
        T.Transform(ip, transip);
        double r = transip(0); // Assuming 'r' is mapped to the first coordinate (x)

        MFEM_VERIFY(r > 0.0,
            "Axisymmetric port conductance sigma/(2*pi*r) is singular at r = "
            << r << ". A massive azimuthal conductor cannot touch the symmetry "
            "axis; remodel the port or use a different conductor type.");

        return conductivity.Eval(T, ip) / (Constants::TWO_PI * r);
    }
};