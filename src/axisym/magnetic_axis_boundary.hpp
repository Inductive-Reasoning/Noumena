// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "mfem.hpp"
#include "axis_geometry.hpp"

/**
 * @brief Discovery of the boundary attribute that carries the symmetry axis.
 *
 * This is a MAGNETIC concern, not a geometric one. Solving for A_phi requires
 * enforcing the essential condition A_phi = 0 on r = 0, which in turn requires
 * the axis to be tagged as its own boundary attribute so the constraint is not
 * smeared onto unrelated boundaries. An electrostatic run on the same
 * axis-touching mesh needs none of this, which is why the requirement is
 * imposed here rather than during geometric classification.
 */
namespace axisym {

/**
 * @brief Find the boundary attributes lying entirely on the symmetry axis.
 *
 * Returns a marker array sized to the mesh's boundary attribute maximum, with 1
 * for each attribute whose every boundary element sits at r = 0. Returns an
 * all-zero marker when the domain does not reach the axis.
 *
 * Aborts when an attribute is only PARTLY on the axis, or when the domain
 * reaches the axis with no attribute dedicated to it: in both cases applying
 * A_phi = 0 would constrain boundaries the user did not intend.
 */
inline mfem::Array<int> FindAxisBoundaryMarker(mfem::Mesh &mesh,
											   const AxisGeometry &geometry)
{
   const int num_bdr_attributes = mesh.bdr_attributes.Size()
	  ? mesh.bdr_attributes.Max() : 0;

   mfem::Array<int> axis_boundary(num_bdr_attributes);
   axis_boundary = 0;

   if (!geometry.TouchesAxis())
   {
	  return axis_boundary;
   }

   mfem::Array<int> on_axis_count(num_bdr_attributes);
   mfem::Array<int> total_count(num_bdr_attributes);
   on_axis_count = 0;
   total_count = 0;

   mfem::Vector pos(mesh.SpaceDimension());
   const mfem::FiniteElementSpace *boundary_nodal_fes = mesh.GetNodalFESpace();

   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
	  const int attr = mesh.GetBdrAttribute(be);
	  // MFEM accepts non-positive attributes with only a warning; no marker
	  // can select such an element.
	  if (attr <= 0) { continue; }
	  total_count[attr - 1]++;

	  mfem::ElementTransformation *trans = mesh.GetBdrElementTransformation(be);
	  const mfem::IntegrationRule &nodes =
		 boundary_nodal_fes
			? boundary_nodal_fes->GetBE(be)->GetNodes()
			: *mfem::Geometries.GetVertices(mesh.GetBdrElementGeometry(be));

	  bool on_axis = true;
	  for (int i = 0; i < nodes.GetNPoints() && on_axis; ++i)
	  {
		 const mfem::IntegrationPoint &ip = nodes.IntPoint(i);
		 trans->SetIntPoint(&ip);
		 trans->Transform(ip, pos);
		 on_axis = geometry.IsOnAxisGeometry(pos(0));
	  }

	  if (on_axis) { on_axis_count[attr - 1]++; }
   }

   bool found_axis = false;
   for (int i = 0; i < num_bdr_attributes; ++i)
   {
	  if (total_count[i] > 0 && on_axis_count[i] == total_count[i])
	  {
		 axis_boundary[i] = 1;
		 found_axis = true;
	  }
	  else if (on_axis_count[i] > 0)
	  {
		 MFEM_ABORT("Boundary attribute " + std::to_string(i + 1) + " has "
					+ std::to_string(on_axis_count[i]) + " of "
					+ std::to_string(total_count[i])
					+ " elements on the symmetry axis. Give the axis its own "
					  "boundary attribute so A_phi = 0 is not applied to "
					  "unrelated boundaries.");
	  }
   }

   MFEM_VERIFY(found_axis,
			   "The axisymmetric domain reaches r = 0 but no boundary attribute "
			   "lies entirely on the symmetry axis. Tag the axis as its own "
			   "boundary attribute so A_phi = 0 can be enforced there.");

   return axis_boundary;
}

} // namespace axisym
