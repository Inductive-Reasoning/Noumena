// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Shared adaptive mesh refinement (AMR) helpers. These are physics-agnostic so
// the electrostatic solver (and, later, the magnetostatic / MQS solvers) can
// reuse the same marking and conforming-refinement policy.

#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "mfem.hpp"

namespace amr {

	// One AMR iteration's diagnostics, recorded during RunAdaptive(). Exposed for
	// tests (convergence / conformity assertions) and console logging.
	struct AmrIterationInfo {
		long   true_dofs;     // global true DOFs on that iteration's mesh
		double global_error;  // sqrt(sum_k eta_k^2), combined over scenarios
		double peak_field_magnitude;     // max |B| sampled over the mesh
	};

	// Bulk (Dorfler) marking. Selects the smallest set of elements whose summed
	// SQUARED error reaches @p fraction of the total squared error.
	//
	//   total = sum_k errors[k]^2
	//   mark elements (largest error first) until accumulated squared error
	//   reaches fraction * total.
	//
	// @p errors   Per-element error indicators (NOT squared); errors[k] >= 0.
	// @p fraction Marking fraction in (0, 1]; values outside are clamped.
	// @p marked   Output: indices of elements to refine (cleared first).
	inline void MarkElementsDorfler(const mfem::Vector& errors, double fraction,
									mfem::Array<int>& marked) {
		marked.SetSize(0);
		const int ne = errors.Size();
		if (ne == 0) { return; }

		// Clamp the fraction into (0, 1]; fall back to the spec default on garbage.
		if (!(fraction > 0.0)) { fraction = 0.7; }
		if (fraction > 1.0)    { fraction = 1.0; }

		// Total "energy" = sum of squared errors.
		double total = 0.0;
		for (int i = 0; i < ne; ++i) { total += errors(i) * errors(i); }
		if (!(total > 0.0)) { return; } // No estimated error -> nothing to refine.

		// Sort element indices by descending error.
		std::vector<int> idx(static_cast<std::size_t>(ne));
		for (int i = 0; i < ne; ++i) { idx[static_cast<std::size_t>(i)] = i; }
		std::sort(idx.begin(), idx.end(),
				  [&errors](int a, int b) { return errors(a) > errors(b); });

		const double threshold = fraction * total;
		double accum = 0.0;
		for (int i = 0; i < ne; ++i) {
			const int e = idx[static_cast<std::size_t>(i)];
			accum += errors(e) * errors(e);
			marked.Append(e);
			if (accum >= threshold) { break; }
		}
	}

	// Conforming refinement of @p marked elements. For simplex (triangular) meshes
	// MFEM performs bisection (red/green) refinement that resolves without hanging
	// nodes. Passing nonconforming = 0 forces the conforming path.
	//
	// The downstream C# results pipeline (TriangleLocator / FemFieldSampler /
	// ResultsView) assumes a conforming triangle tiling, so a non-conforming result
	// is treated as an error rather than silently exported.
	inline void RefineConforming(mfem::Mesh& mesh, const mfem::Array<int>& marked) {
		if (marked.Size() == 0) { return; }

		// MFEM cannot refine quadrilaterals or hexahedra conformingly: it
		// aborts inside GeneralRefinement, so reject them first.
		for (int e = 0; e < mesh.GetNE(); ++e) {
			const mfem::Geometry::Type geometry = mesh.GetElementBaseGeometry(e);
			if (geometry != mfem::Geometry::TRIANGLE && geometry != mfem::Geometry::TETRAHEDRON) {
				throw std::runtime_error(
					"AMR: conforming refinement is only supported for simplex "
					"(triangular or tetrahedral) meshes, and this mesh has other "
					"elements. Re-mesh with simplices or disable AMR.");
			}
		}

		mesh.GeneralRefinement(marked, /*nonconforming=*/0);

		// Hard requirement: the exported mesh must be free of hanging nodes. If MFEM
		// could not satisfy the request conformingly it falls back to a
		// non-conforming representation; reject that here.
		if (mesh.Nonconforming()) {
			throw std::runtime_error(
				"AMR: conforming refinement was requested but the mesh became "
				"non-conforming (hanging nodes). Conforming AMR is only supported "
				"for simplex (triangular or tetrahedral) meshes. Re-mesh with "
				"simplices or disable AMR.");
		}
	}

} // namespace amr
