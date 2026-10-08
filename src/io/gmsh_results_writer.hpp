// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Writes results in Gmsh MSH ASCII format, version 2.2 (default) or 4.1. The
// single output file contains one mesh block followed by zero or more
// $NodeData / $ElementNodeData views. View names are exposed via Gmsh's
// StringTags[0] and form the contract with downstream consumers (e.g.
// TfmrLib's FEMSolution loader).
//
// Elements are emitted as native Gmsh Lagrange elements of the solution order
// (types 2/9/21/23... for triangles, 3/10/36/37... for quads, 4/11/29/30... for
// tetrahedra, 5/12/92/93... for hexahedra), so Gmsh interpolates the field with
// the matching high-order shape functions. No refined/tessellated export copy
// of the mesh is made.
//
// An $InterpolationScheme block carries the shape functions themselves (a
// monomial exponent matrix plus Lagrange coefficients) and every view names it
// via StringTags[1]. A consumer can therefore evaluate any field at an
// arbitrary point without hardcoding Gmsh node ordering or Lagrange formulas,
// and without changes when the solution order changes.
//
// FIELD REPRESENTATION POLICY: fields are exported exactly as the FE solution
// represents them. Continuous primaries (V, A) go out as $NodeData; derived
// quantities that are genuinely discontinuous across elements (E = -grad V,
// B = curl A, and anything built from them) go out as $ElementNodeData with
// per-element values. Inter-element jumps are real results of the
// discretization and are passed through unsmoothed; how to treat them (average,
// recover, or respect) is the consumer's decision, since it depends on the
// post-processing being done.
//
// FORMAT SEAM: only the MESH sections differ between MSH 2.2 and 4.1, and all
// of that is confined to WriteMeshFormat, WriteMeshBlock22, and
// WriteMeshBlock41 behind the WriteMeshBlock dispatcher. Gmsh's
// post-processing sections are not versioned along with the mesh format, so
// $InterpolationScheme, $NodeData, and $ElementNodeData are emitted
// byte-identically for both versions. The node layout (HoLayout/GetHoLayout),
// the MFEM->Gmsh permutation, and all field sampling are likewise
// format-agnostic.
//
// The one semantic difference worth knowing: 2.2 stores each element's
// attribute on the element line, whereas 4.1 stores it once per model entity
// in $Entities and groups elements under those entities. Both paths therefore
// round-trip the same MFEM attributes.
//
// Format references:
//   2.2: https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format-version-2-_0028Legacy_0029
//   4.1: https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format

#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <ios>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "mfem.hpp"

namespace gmsh_results {

/// MSH file format version of the emitted mesh sections.
///
/// V2_2 is the default and the format the existing downstream C# consumer
/// (TfmrLib's FEMSolution loader) understands. V4_1 is opt-in.
enum class MshVersion { V2_2, V4_1 };

/// Parse a format string ("2.2" / "4.1") as used in the simulation config.
/// Throws std::runtime_error on an unrecognized value.
MshVersion ParseMshVersion(const std::string& text);

/// Description of one Gmsh view to emit alongside the mesh block.
struct View {
    enum class Kind { NodeData, ElementNodeData };

    std::string name;        ///< View name, written as StringTags[0].
    Kind        kind;
    int         num_components; ///< 1 (scalar) or 3 (vector, padded in 2D).

    /// Called once per sample point with the point in reference coordinates;
    /// out has num_components entries. Both view kinds sample by evaluation,
    /// so continuous (NodeData) and discontinuous (ElementNodeData) fields
    /// share one callback shape.
    std::function<void(int elem_id,
                       const mfem::IntegrationPoint& ip,
                       mfem::ElementTransformation& T,
                       double* out)> elem_node_eval;
};

namespace detail {

// Reference-space node layout of a Gmsh Lagrange element of a given order.
//
// Node ORDER is Gmsh's own, defined recursively: all corner nodes, then the
// nodes interior to each edge (edges traversed in Gmsh's element-local order
// and direction), then the nodes interior to each face (in 3D, each face laid
// out as a lower-order triangle/quad in that face's vertex order), then the
// nodes interior to the cell, which are themselves laid out as a lower order
// element of the same shape. Coordinates are in MFEM's reference domain
// (unit simplex / unit square / unit cube) so they feed straight into
// ElementTransformation::Transform. In 2D the third coordinate is zero.
//
// The 3D layouts reproduce gmsh.model.mesh.getElementProperties() node for
// node (after mapping Gmsh's [-1,1] hexahedron onto MFEM's unit cube); see
// tools/check_gmsh_layouts.py for the cross-check against Gmsh itself.
//
// Working in reference COORDINATES rather than a permutation of MFEM DOF
// indices keeps this independent of MFEM's internal nodal layout and of which
// basis (Gauss-Lobatto vs equispaced) the solution happens to use: every
// exported value is obtained by evaluating at a point.
using RefPt = std::array<double, 3>;

struct HoLayout {
    int gmsh_type = 0;
    std::vector<RefPt> ref;
};

inline RefPt Lerp(const RefPt& a, const RefPt& b, double t) {
    return { a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
             a[2] + (b[2] - a[2]) * t };
}

// The lattice point one step (1/p of each incident edge) in from corner @p a
// toward its @p neighbors. This is the corner of the inset element that holds
// the next recursion level's nodes.
RefPt Inset(const RefPt& a, std::initializer_list<const RefPt*> neighbors,
                   double inv);

RefPt Centroid(std::initializer_list<const RefPt*> pts);

// Nodes strictly inside the edge a -> b, in that direction.
inline void AppendEdgeNodes(int p, const RefPt& a, const RefPt& b,
                            std::vector<RefPt>& out) {
    const double inv = 1.0 / static_cast<double>(p);
    for (int i = 1; i < p; ++i) { out.push_back(Lerp(a, b, i * inv)); }
}

// Gmsh element type codes, indexed by order. Index 0 is unused.
inline int TriangleGmshType(int order) {
    static const int kTypes[] = { 0, 2, 9, 21, 23, 25, 42, 43, 44, 45, 46 };
    return (order >= 1 && order <= 10) ? kTypes[order] : 0;
}

inline int QuadGmshType(int order) {
    static const int kTypes[] = { 0, 3, 10, 36, 37, 38, 47, 48, 49, 50, 51 };
    return (order >= 1 && order <= 10) ? kTypes[order] : 0;
}

inline int TetrahedronGmshType(int order) {
    static const int kTypes[] = { 0, 4, 11, 29, 30, 31, 71, 72, 73, 74, 75 };
    return (order >= 1 && order <= 10) ? kTypes[order] : 0;
}

// Gmsh defines complete Lagrange hexahedra up to order 9 only.
inline int HexahedronGmshType(int order) {
    static const int kTypes[] = { 0, 5, 12, 92, 93, 94, 95, 96, 97, 98 };
    return (order >= 1 && order <= 9) ? kTypes[order] : 0;
}

// Emit an order-p triangular lattice over the triangle (a, b, c) in Gmsh order.
// p == 0 degenerates to the single centroid node, which is how the recursion
// terminates for orders that leave exactly one interior node.
void AppendTriangleNodes(int p, const RefPt& a, const RefPt& b,
                                const RefPt& c, std::vector<RefPt>& out);

// Emit an order-p quadrilateral lattice over (a, b, c, d) in Gmsh order.
void AppendQuadNodes(int p, const RefPt& a, const RefPt& b,
                            const RefPt& c, const RefPt& d,
                            std::vector<RefPt>& out);

// Emit an order-p tetrahedral lattice over (v0, v1, v2, v3) in Gmsh order:
// corners; edges {0,1} {1,2} {2,0} {3,0} {3,2} {3,1} (each walked from its
// first vertex); faces {0,2,1} {0,1,3} {0,3,2} {3,1,2}, each an inset triangle
// of order p-3; then the interior as an inset tetrahedron of order p-4.
void AppendTetNodes(int p, const RefPt& v0, const RefPt& v1,
                           const RefPt& v2, const RefPt& v3,
                           std::vector<RefPt>& out);

// Emit an order-p hexahedral lattice over corners v[0..7] (MFEM/Gmsh corner
// numbering) in Gmsh order: corners; the 12 edges; the 6 faces, each an inset
// quad of order p-2; then the interior as an inset hexahedron of order p-2.
void AppendHexNodes(int p, const std::array<RefPt, 8>& v,
                           std::vector<RefPt>& out);

const char* GeometryName(mfem::Geometry::Type geom);

// Layout for one (geometry, order) pair. Built once and cached: the recursion
// is cheap but this is called per element.
const HoLayout& GetHoLayout(mfem::Geometry::Type geom, int order);

// Hot-path numeric formatting helpers.
//
// std::ostream::operator<<(double) is ~5-10x slower than std::to_chars on
// MSVC because it queries the stream locale and routes through num_put. For
// MSH export we write millions of doubles, so format into a small stack
// buffer and write raw bytes through filebuf instead.
//
// Doubles are written in their shortest form that reads back to the same
// value (to_chars without a precision): exact for coordinates as well as
// field values, so high-order nodes stay distinct on large models (a fixed 10
// digits collapses a 1e-7 m lattice spacing at 1e3 m), and usually shorter
// than a fixed 17 digits. Gmsh has no representation for NaN or infinity, so
// a non-finite value is an error rather than a file readers reject.
void AppendInt(std::string& s, long long v);

void AppendDouble(std::string& s, double v);

// Per-element map from Gmsh node slot -> MFEM local DOF index.
//
// Derived by matching reference coordinates rather than by hardcoding MFEM's
// nodal ordering, so it stays correct if that ordering ever changes. All
// elements of the same geometry and order share one permutation.
std::vector<int> BuildDofPermutation(const mfem::FiniteElement& fe,
                                            const HoLayout& layout);

// Monomial exponents and Lagrange coefficients defining an element's shape
// functions, in the form Gmsh's $InterpolationScheme block expects.
//
// Gmsh's model is: given exponent matrix E (n_terms x n_vars) and coefficient
// matrix C (n_nodes x n_terms), shape function i is
//
//     phi_i(u, v, w) = sum_j C[i][j] * u^E[j][0] * v^E[j][1] * w^E[j][2]
//
// with n_vars = 2 for 2D elements (the w exponent is then zero and is not
// written) and 3 for 3D elements. (u, v, w) are MFEM's reference coordinates:
// the unit simplex, which coincides with Gmsh's, and the unit square / cube,
// where Gmsh's own parametric domain is [-1, 1]^d.
//
// and a field is reconstructed as sum_i phi_i(u, v) * value_i, where value_i
// are the nodal values listed for that element (in the same node order as the
// mesh block's connectivity).
//
// Shipping this in the file is what makes the consumer independent of Gmsh's
// node ordering AND of the element order: a reader evaluates the monomials and
// does a matrix-vector product without knowing anything about Lagrange bases.
// This matters most for DISCONTINUOUS ($ElementNodeData) fields such as E and
// B, where nodal values alone cannot be interpolated -- the consumer must
// evaluate the element-local basis to get a value anywhere but a node.
struct InterpScheme {
    std::vector<std::array<int, 3>>  exponents;  // per monomial term (u, v, w)
    std::vector<std::vector<double>> coeffs;     // [node][term]
};

// Monomial exponents spanning the polynomial space of an order-p element.
// Simplices use the total-degree space (u^a v^b [w^c], a + b [+ c] <= p);
// quads and hexahedra use the tensor-product space (each exponent <= p). These
// match the node lattices built by the Append*Nodes functions, so the
// Vandermonde system below is square and non-singular.
std::vector<std::array<int, 3>> MonomialExponents(
    mfem::Geometry::Type geom, int order);

// Build the Lagrange coefficient matrix.
//
// Simplices: requiring phi_i(node_k) = delta_ik gives V * C^T = I, where
// V[k][j] = monomial_j(node_k). So C^T = V^-1, i.e. C = (V^-1)^T. Solved with
// Gauss-Jordan and partial pivoting; the Lagrange property then holds to about
// 1e-6 at order 10 (1e-12 at order 5). Tensor-product elements take the exact
// route above instead.
// Monomial coefficients of the 1D Lagrange polynomials on the equispaced nodes
// t_b = b/p of [0, 1]: result[a][k] is the u^k coefficient of L_a(u), with
// L_a(t_b) = delta_ab. Built by expanding prod_{b != a} (u - t_b) / (t_a - t_b)
// directly, which involves no linear solve.
std::vector<std::vector<double>> LagrangeMonomials1D(int p);

// Tensor-product elements (quads, hexahedra): each shape function is a product
// of 1D Lagrange polynomials, so its monomial coefficients are exact products
// of 1D coefficients. This avoids inverting the (p+1)^d Vandermonde matrix,
// whose conditioning grows like the d-th power of the 1D one: by Gauss-Jordan
// the Lagrange property fails at order 6 for quads and order 5 for hexahedra.
//
// What remains is inherent to a monomial basis on [0, 1]: evaluating it
// cancels terms whose size grows with order, so the Lagrange property holds to
// ~1e-10 at order 5 and ~1e-7 at order 7 for quads, and to ~1e-10 at order 4
// and ~1e-7 at order 5 for hexahedra, degrading quickly above that. Gmsh's own
// [-1, 1] reference domain would be far better conditioned, but the consumer
// contract fixes MFEM's [0, 1] domain.
InterpScheme BuildTensorInterpScheme(mfem::Geometry::Type geom, int order,
                                            const HoLayout& layout);

InterpScheme BuildInterpScheme(mfem::Geometry::Type geom, int order);

// Node numbering and geometry for the exported high-order mesh.
//
// Node ids are the DOF indices of an equispaced H1 space, which are shared
// across element boundaries, so the emitted mesh is continuous. Each node also
// records one (element, reference point) pair so field values can be sampled
// there later without recomputing the layout.
struct ExportNodes {
    int order = 1;
    std::vector<std::array<double, 3>> coord;  // indexed by node id
    std::vector<int>   node_elem;              // representative element
    std::vector<RefPt> node_ref;               // reference point in that element
    std::vector<int>   elem_type;              // Gmsh type code, per element
    std::vector<std::vector<int>> elem_nodes;  // Gmsh-ordered node ids, per element
};

ExportNodes BuildExportNodes(mfem::Mesh& mesh,
                                    mfem::FiniteElementSpace& fes,
                                    int order);

void WriteMeshFormat(std::ostream& out, MshVersion version);

void WriteMeshBlock22(std::ostream& out, mfem::Mesh& mesh,
                             const ExportNodes& nodes);

// MSH 4.1 mesh sections.
//
// The structural difference from 2.2 is where an element's attribute lives.
// In 2.2 every element line carries its own tags ("2 <attr> <attr>"). In 4.1
// elements carry no tags at all; they are grouped into blocks keyed by the
// model entity they are classified on, and a separate $Entities section maps
// each entity to its physical tags. To preserve exactly the attribute
// semantics the 2.2 path exposes, we synthesize one entity of the mesh's
// dimension (a surface in 2D, a volume in 3D) per distinct element attribute,
// with entityTag == physicalTag == attribute.
//
// Node tags remain the global 1..nv numbering used by the 2.2 path, so the
// $NodeData / $ElementNodeData sections that follow are byte-identical between
// the two formats.
void WriteMeshBlock41(std::ostream& out, mfem::Mesh& mesh,
                             const ExportNodes& nodes);

void WriteMeshBlock(std::ostream& out, mfem::Mesh& mesh,
                           const ExportNodes& nodes, MshVersion version);

// Emit one $InterpolationScheme block describing the shape functions for every
// element type present in the mesh.
//
// Gmsh associates the scheme with views by name; WriteViewHeader writes the
// same name as the view's StringTags[1], which is how a reader (and Gmsh
// itself) links a view's nodal values to the basis that interpolates them.
//
// Each element topology gets two matrices per Gmsh's format: the coefficient
// matrix then the exponent matrix. A scheme is keyed by the element family
// (Gmsh's TYPE_TRI = 3, TYPE_QUA = 4, TYPE_TET = 5, TYPE_HEX = 8), not by the
// element type, which would name a different family (type 2, the 3-node
// triangle, is family TYPE_LIN) or none at all for higher orders; Gmsh itself
// writes a triangle scheme under 3.
int GmshFamily(mfem::Geometry::Type geom);

void WriteInterpolationScheme(std::ostream& out,
                                     const std::string& scheme_name,
                                     mfem::Mesh& mesh,
                                     int order);

void WriteViewHeader(std::ostream& out,
                            const char*   tag,
                            const std::string& name,
                            const std::string& scheme_name,
                            int num_components,
                            int num_entities);

void WriteNodeData(std::ostream& out, mfem::Mesh& mesh,
                          const ExportNodes& nodes, const View& v,
                          const std::string& scheme_name);

void WriteElementNodeData(std::ostream& out,
                                 mfem::Mesh& mesh,
                                 const ExportNodes& nodes,
                                 const View& v,
                                 const std::string& scheme_name);

} // namespace detail

/// Convenience: NodeData view of a continuous scalar GridFunction. The field is
/// sampled by evaluation, so it may be of any order or basis; it need not match
/// the equispaced space used to number the export nodes.
View MakeScalarNodeView(const std::string& name,
                               mfem::GridFunction& gf);

/// Convenience: ElementNodeData view of a vector-valued GridFunction (e.g. a
/// Nedelec vector potential). Per-element because such fields are continuous
/// only in their tangential component; values are the element's own, padded
/// to 3 components.
View MakeVectorGridFunctionView(const std::string& name,
                                       mfem::GridFunction& gf);

/// Convenience: ElementNodeData view of a vector Coefficient. Output is always
/// padded to 3 components.
///
/// Derived fields are sampled straight from the coefficient at each export
/// node, so there is no intermediate projection onto an L2 space and no
/// projection error; the emitted values are the coefficient's own.
View MakeVectorCoefficientView(const std::string& name,
                                      mfem::VectorCoefficient& vec_coeff);

/// Convenience: ElementNodeData scalar view sampling a scalar Coefficient.
View MakeScalarCoefficientView(const std::string& name,
                                      mfem::Coefficient& coeff);

/// The mesh-side part of a results file, which every scenario on one mesh
/// shares (see WriteGmshResults).
struct MeshExport {
    const mfem::Mesh* mesh = nullptr;
    long sequence = -1;
    int order = 0;
    MshVersion version = MshVersion::V2_2;
    detail::ExportNodes nodes;
    std::string block;  // $MeshFormat through $InterpolationScheme
};

/// Writes the mesh and all views to @p path in Gmsh MSH ASCII, using native
/// Gmsh Lagrange elements of order @p order.
///
/// @param path     Destination file (will be overwritten).
/// @param mesh     Mesh to embed. Exported at its own resolution; no refined
///                 copy is made.
/// @param order    Lagrange order of the emitted elements and of the node layout
///                 every view is sampled on. Typically the solution order.
/// @param views    Views to emit, in order.
/// @param version  MSH format of the mesh sections. Defaults to 2.2, which is
///                 what the downstream C# consumer reads; 4.1 is opt-in.
/// @param cache    Optional: the mesh-side part of the file (export nodes,
///                 mesh block, interpolation scheme), which every scenario on
///                 one mesh shares; rebuilt only when the mesh, its refinement
///                 sequence, the order or the version changes.
///
/// Throws std::runtime_error if the file cannot be opened or written.
void WriteGmshResults(const std::string& path,
                             mfem::Mesh& mesh,
                             int order,
                             const std::vector<View>& views,
                             MshVersion version = MshVersion::V2_2,
                             MeshExport* cache = nullptr);

} // namespace gmsh_results
