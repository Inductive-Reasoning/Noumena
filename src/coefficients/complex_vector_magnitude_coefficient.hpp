// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Magnitude of a complex-valued vector field given its real and imaginary
// vector parts: |F| = sqrt(|Re F|^2 + |Im F|^2). Physics-agnostic (works for B,
// E, ...); used so writers can treat the magnitude as an ordinary scalar
// Coefficient instead of hand-rolling a strided GridFunction loop.

#pragma once

#include <cmath>
#include "mfem.hpp"

class ComplexVectorMagnitudeCoefficient : public mfem::Coefficient {
	mfem::VectorCoefficient& re_;
	mfem::VectorCoefficient& im_;

public:
	ComplexVectorMagnitudeCoefficient(mfem::VectorCoefficient& re,
									  mfem::VectorCoefficient& im)
		: re_(re), im_(im) {}

	mfem::real_t Eval(mfem::ElementTransformation& T,
				const mfem::IntegrationPoint& ip) override {
		mfem::Vector re_val, im_val;
		re_.Eval(re_val, T, ip);
		im_.Eval(im_val, T, ip);
		double s = 0.0;
		for (int i = 0; i < re_val.Size(); ++i) s += re_val(i) * re_val(i);
		for (int i = 0; i < im_val.Size(); ++i) s += im_val(i) * im_val(i);
		return std::sqrt(s);
	}
};
