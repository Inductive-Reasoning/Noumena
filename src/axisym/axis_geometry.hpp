// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "mfem.hpp"

/**
 * @brief Radial extent of a 2D (r,z) mesh, and the scale at which a radius
 *        counts as on the symmetry axis r = 0.
 *
 * Inspected once at solver setup rather than clamping r at each quadrature
 * point, which would silently deform the geometry. The tolerance is relative to
 * the mesh bounding box, so the axis test is scale-free; the physics is not, and
 * needs coordinates in metres (see core/constants.hpp).
 */
namespace axisym {

// Relative tolerance used to decide whether a coordinate "is" on the axis.
// Loose enough to absorb mesh-generator round-off, tight enough that a real
// gap between the domain and the axis is never mistaken for contact.
inline constexpr mfem::real_t kRelativeGeometryTolerance =
   static_cast<mfem::real_t>(1.0e-10);

/// True when @p r is on the symmetry axis to within @p tolerance.
inline bool IsOnAxisGeometry(mfem::real_t r, mfem::real_t tolerance)
{
   return std::abs(r) <= tolerance;
}

/// Radial extent of an (r,z) mesh. Stays valid across conforming AMR, which
/// cannot change the bounding box.
struct AxisGeometry
{
   mfem::real_t min_r = 0.0;
   mfem::real_t max_r = 0.0;
   /// Mesh-scale-relative tolerance for axis proximity tests.
   mfem::real_t tolerance = 0.0;

   /// True when the closure of the domain reaches r = 0.
   [[nodiscard]] bool TouchesAxis() const
   {
	  return min_r <= tolerance;
   }

   /// True when @p r represents axis geometry at this mesh's scale.
   [[nodiscard]] bool IsOnAxisGeometry(mfem::real_t r) const
   {
	  return axisym::IsOnAxisGeometry(r, tolerance);
   }
};

/**
 * @brief Scan the radial coordinate of a 2D (r,z) mesh, aborting on a
 *        materially negative radius.
 *
 * Samples each element's geometry nodes, not just its vertices: a curved edge
 * can bulge across r = 0 while both endpoints stay non-negative.
 */
inline AxisGeometry ValidateMesh(mfem::Mesh &mesh)
{
   MFEM_VERIFY(mesh.Dimension() == 2,
			   "Axisymmetric geometry requires a 2D (r,z) mesh.");
   MFEM_VERIFY(mesh.GetNE() > 0, "Axisymmetric mesh has no elements.");

   AxisGeometry info;

   mfem::real_t min_r = std::numeric_limits<mfem::real_t>::max();
   mfem::real_t max_r = std::numeric_limits<mfem::real_t>::lowest();
   mfem::real_t min_z = std::numeric_limits<mfem::real_t>::max();
   mfem::real_t max_z = std::numeric_limits<mfem::real_t>::lowest();

   mfem::Vector pos(mesh.SpaceDimension());
   const mfem::FiniteElementSpace *nodal_fes = mesh.GetNodalFESpace();

   for (int e = 0; e < mesh.GetNE(); ++e)
   {
	  mfem::ElementTransformation *trans = mesh.GetElementTransformation(e);

	  // Curved meshes are sampled at their geometry nodes; linear meshes fall
	  // back to the reference element's vertices.
	  const mfem::IntegrationRule &nodes =
		 nodal_fes
			? nodal_fes->GetFE(e)->GetNodes()
			: *mfem::Geometries.GetVertices(mesh.GetElementBaseGeometry(e));

	  for (int i = 0; i < nodes.GetNPoints(); ++i)
	  {
		 const mfem::IntegrationPoint &ip = nodes.IntPoint(i);
		 trans->SetIntPoint(&ip);
		 trans->Transform(ip, pos);

		 min_r = std::min(min_r, pos(0));
		 max_r = std::max(max_r, pos(0));
		 min_z = std::min(min_z, pos(1));
		 max_z = std::max(max_z, pos(1));
	  }
   }

   info.min_r = min_r;
   info.max_r = max_r;

   // The final term is a floor that keeps the tolerance strictly positive for a
   // degenerate (zero-extent) mesh. It must come from the type itself: a literal
   // like 1e-300 is below the smallest normal float and would flush to zero in a
   // single-precision build, silently turning the tolerance into an exact-zero
   // comparison.
   const mfem::real_t scale = std::max({ std::abs(max_r), std::abs(min_r),
										max_z - min_z,
										std::numeric_limits<mfem::real_t>::min() });
   info.tolerance = kRelativeGeometryTolerance * scale;

   MFEM_VERIFY(min_r >= -info.tolerance,
			   "Axisymmetric mesh extends to a negative radius (min r = " << min_r
			   << "). The r coordinate is the mesh x coordinate and must be "
			   "non-negative; check the mesh orientation or units.");

   return info;
}

} // namespace axisym
