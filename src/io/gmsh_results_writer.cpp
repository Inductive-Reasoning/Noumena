// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "gmsh_results_writer.hpp"

namespace gmsh_results {

 MshVersion ParseMshVersion(const std::string& text) {
    if (text == "2.2") { return MshVersion::V2_2; }
    if (text == "4.1") { return MshVersion::V4_1; }
    throw std::runtime_error(
        "gmsh_results: unsupported MSH format '" + text
        + "'; expected \"2.2\" or \"4.1\"");
}

namespace detail {

 RefPt Inset(const RefPt& a, std::initializer_list<const RefPt*> neighbors,
                   double inv) {
    RefPt r = a;
    for (const RefPt* n : neighbors) {
        for (int c = 0; c < 3; ++c) { r[c] += ((*n)[c] - a[c]) * inv; }
    }
    return r;
}

 RefPt Centroid(std::initializer_list<const RefPt*> pts) {
    RefPt r = { 0.0, 0.0, 0.0 };
    for (const RefPt* q : pts) {
        for (int c = 0; c < 3; ++c) { r[c] += (*q)[c]; }
    }
    for (int c = 0; c < 3; ++c) { r[c] /= static_cast<double>(pts.size()); }
    return r;
}

 void AppendTriangleNodes(int p, const RefPt& a, const RefPt& b,
                                const RefPt& c, std::vector<RefPt>& out) {
    if (p == 0) {
        out.push_back(Centroid({ &a, &b, &c }));
        return;
    }
    out.push_back(a);
    out.push_back(b);
    out.push_back(c);

    AppendEdgeNodes(p, a, b, out);
    AppendEdgeNodes(p, b, c, out);
    AppendEdgeNodes(p, c, a, out);

    // Interior nodes form a triangle of order p-3, inset by one lattice step.
    if (p < 3) return;
    const double inv = 1.0 / static_cast<double>(p);
    AppendTriangleNodes(p - 3, Inset(a, { &b, &c }, inv),
                        Inset(b, { &a, &c }, inv), Inset(c, { &a, &b }, inv), out);
}

 void AppendQuadNodes(int p, const RefPt& a, const RefPt& b,
                            const RefPt& c, const RefPt& d,
                            std::vector<RefPt>& out) {
    if (p == 0) {
        out.push_back(Centroid({ &a, &b, &c, &d }));
        return;
    }
    out.push_back(a);
    out.push_back(b);
    out.push_back(c);
    out.push_back(d);

    AppendEdgeNodes(p, a, b, out);
    AppendEdgeNodes(p, b, c, out);
    AppendEdgeNodes(p, c, d, out);
    AppendEdgeNodes(p, d, a, out);

    // Interior nodes form a quad of order p-2, inset by one lattice step.
    if (p < 2) return;
    const double inv = 1.0 / static_cast<double>(p);
    AppendQuadNodes(p - 2, Inset(a, { &b, &d }, inv), Inset(b, { &a, &c }, inv),
                    Inset(c, { &b, &d }, inv), Inset(d, { &a, &c }, inv), out);
}

 void AppendTetNodes(int p, const RefPt& v0, const RefPt& v1,
                           const RefPt& v2, const RefPt& v3,
                           std::vector<RefPt>& out) {
    if (p == 0) {
        out.push_back(Centroid({ &v0, &v1, &v2, &v3 }));
        return;
    }
    const RefPt* v[4] = { &v0, &v1, &v2, &v3 };
    for (const RefPt* c : v) { out.push_back(*c); }

    static const int kEdges[6][2] = {
        { 0, 1 }, { 1, 2 }, { 2, 0 }, { 3, 0 }, { 3, 2 }, { 3, 1 } };
    for (const auto& e : kEdges) { AppendEdgeNodes(p, *v[e[0]], *v[e[1]], out); }

    if (p < 3) return;
    const double inv = 1.0 / static_cast<double>(p);
    static const int kFaces[4][3] = {
        { 0, 2, 1 }, { 0, 1, 3 }, { 0, 3, 2 }, { 3, 1, 2 } };
    for (const auto& f : kFaces) {
        const RefPt& a = *v[f[0]];
        const RefPt& b = *v[f[1]];
        const RefPt& c = *v[f[2]];
        AppendTriangleNodes(p - 3, Inset(a, { &b, &c }, inv),
                            Inset(b, { &a, &c }, inv), Inset(c, { &a, &b }, inv), out);
    }

    if (p < 4) return;
    AppendTetNodes(p - 4, Inset(v0, { &v1, &v2, &v3 }, inv),
                   Inset(v1, { &v0, &v2, &v3 }, inv),
                   Inset(v2, { &v0, &v1, &v3 }, inv),
                   Inset(v3, { &v0, &v1, &v2 }, inv), out);
}

 void AppendHexNodes(int p, const std::array<RefPt, 8>& v,
                           std::vector<RefPt>& out) {
    if (p == 0) {
        RefPt c = { 0.0, 0.0, 0.0 };
        for (const RefPt& q : v) {
            for (int k = 0; k < 3; ++k) { c[k] += q[k] * 0.125; }
        }
        out.push_back(c);
        return;
    }
    for (const RefPt& c : v) { out.push_back(c); }

    static const int kEdges[12][2] = {
        { 0, 1 }, { 0, 3 }, { 0, 4 }, { 1, 2 }, { 1, 5 }, { 2, 3 },
        { 2, 6 }, { 3, 7 }, { 4, 5 }, { 4, 7 }, { 5, 6 }, { 6, 7 } };
    for (const auto& e : kEdges) { AppendEdgeNodes(p, v[e[0]], v[e[1]], out); }

    if (p < 2) return;
    const double inv = 1.0 / static_cast<double>(p);
    static const int kFaces[6][4] = {
        { 0, 3, 2, 1 }, { 0, 1, 5, 4 }, { 0, 4, 7, 3 },
        { 1, 2, 6, 5 }, { 2, 3, 7, 6 }, { 4, 5, 6, 7 } };
    for (const auto& f : kFaces) {
        const RefPt& a = v[f[0]];
        const RefPt& b = v[f[1]];
        const RefPt& c = v[f[2]];
        const RefPt& d = v[f[3]];
        AppendQuadNodes(p - 2, Inset(a, { &b, &d }, inv), Inset(b, { &a, &c }, inv),
                        Inset(c, { &b, &d }, inv), Inset(d, { &a, &c }, inv), out);
    }

    // Each corner's three edge neighbors, for the inset interior hexahedron.
    static const int kNeighbors[8][3] = {
        { 1, 3, 4 }, { 0, 2, 5 }, { 1, 3, 6 }, { 0, 2, 7 },
        { 0, 5, 7 }, { 1, 4, 6 }, { 2, 5, 7 }, { 3, 4, 6 } };
    std::array<RefPt, 8> inner;
    for (int i = 0; i < 8; ++i) {
        const auto& n = kNeighbors[i];
        inner[i] = Inset(v[i], { &v[n[0]], &v[n[1]], &v[n[2]] }, inv);
    }
    AppendHexNodes(p - 2, inner, out);
}

 const char* GeometryName(mfem::Geometry::Type geom) {
    switch (geom) {
        case mfem::Geometry::SEGMENT:     return "segment";
        case mfem::Geometry::TRIANGLE:    return "triangle";
        case mfem::Geometry::SQUARE:      return "quadrilateral";
        case mfem::Geometry::TETRAHEDRON: return "tetrahedron";
        case mfem::Geometry::CUBE:        return "hexahedron";
        case mfem::Geometry::PRISM:       return "prism";
        default:                          return "unknown";
    }
}

 const HoLayout& GetHoLayout(mfem::Geometry::Type geom, int order) {
    static std::map<std::pair<int, int>, HoLayout> cache;
    const auto key = std::make_pair(static_cast<int>(geom), order);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    if (order < 1 || order > 10) {
        throw std::runtime_error(
            "gmsh_results: unsupported export order " + std::to_string(order)
            + " for a " + GeometryName(geom)
            + "; Gmsh Lagrange elements are supported for orders 1-10");
    }

    HoLayout layout;
    switch (geom) {
        case mfem::Geometry::TRIANGLE:
            layout.gmsh_type = TriangleGmshType(order);
            AppendTriangleNodes(order, { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 },
                                { 0.0, 1.0, 0.0 }, layout.ref);
            break;
        case mfem::Geometry::SQUARE:
            layout.gmsh_type = QuadGmshType(order);
            AppendQuadNodes(order, { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 },
                            { 1.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0 }, layout.ref);
            break;
        case mfem::Geometry::TETRAHEDRON:
            layout.gmsh_type = TetrahedronGmshType(order);
            AppendTetNodes(order, { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 },
                           { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 }, layout.ref);
            break;
        case mfem::Geometry::CUBE:
            layout.gmsh_type = HexahedronGmshType(order);
            if (layout.gmsh_type == 0) {
                throw std::runtime_error(
                    "gmsh_results: unsupported export order "
                    + std::to_string(order) + " for a hexahedron; Gmsh defines "
                      "Lagrange hexahedra for orders 1-9");
            }
            AppendHexNodes(order, { { { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 },
                                      { 1.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0 },
                                      { 0.0, 0.0, 1.0 }, { 1.0, 0.0, 1.0 },
                                      { 1.0, 1.0, 1.0 }, { 0.0, 1.0, 1.0 } } },
                           layout.ref);
            break;
        default:
            throw std::runtime_error(
                std::string("gmsh_results: unsupported element geometry '")
                + GeometryName(geom)
                + "' for MSH export; triangles, quadrilaterals, tetrahedra and "
                  "hexahedra are handled");
    }
    return cache.emplace(key, std::move(layout)).first->second;
}

 void AppendInt(std::string& s, long long v) {
    char buf[24];
    auto r = std::to_chars(buf, buf + sizeof(buf), v);
    s.append(buf, r.ptr);
}

 void AppendDouble(std::string& s, double v) {
    if (!std::isfinite(v)) {
        throw std::runtime_error("gmsh_results: a non-finite value (NaN or infinity) "
                                 "cannot be written to a Gmsh file");
    }
    char buf[32];
    auto r = std::to_chars(buf, buf + sizeof(buf), v);
    s.append(buf, r.ptr);
}

 std::vector<int> BuildDofPermutation(const mfem::FiniteElement& fe,
                                            const HoLayout& layout) {
    const mfem::IntegrationRule& ir = fe.GetNodes();
    const int n = ir.GetNPoints();
    if (n != static_cast<int>(layout.ref.size())) {
        throw std::runtime_error(
            "gmsh_results: node count mismatch between MFEM element ("
            + std::to_string(n) + ") and Gmsh layout ("
            + std::to_string(layout.ref.size()) + ")");
    }

    std::vector<int> perm(n, -1);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            const mfem::IntegrationPoint& ip = ir.IntPoint(j);
            if (std::fabs(ip.x - layout.ref[k][0]) < 1e-10 &&
                std::fabs(ip.y - layout.ref[k][1]) < 1e-10 &&
                std::fabs(ip.z - layout.ref[k][2]) < 1e-10) {
                perm[k] = j;
                break;
            }
        }
        if (perm[k] < 0) {
            throw std::runtime_error(
                "gmsh_results: no MFEM node matches Gmsh node slot "
                + std::to_string(k)
                + "; the export space must use equispaced (ClosedUniform) nodes");
        }
    }
    return perm;
}

 std::vector<std::array<int, 3>> MonomialExponents(
    mfem::Geometry::Type geom, int order) {
    std::vector<std::array<int, 3>> e;
    switch (geom) {
        case mfem::Geometry::TRIANGLE:
            for (int d = 0; d <= order; ++d) {
                for (int i = 0; i <= d; ++i) { e.push_back({ d - i, i, 0 }); }
            }
            break;
        case mfem::Geometry::TETRAHEDRON:
            for (int d = 0; d <= order; ++d) {
                for (int c = 0; c <= d; ++c) {
                    for (int b = 0; b <= d - c; ++b) {
                        e.push_back({ d - b - c, b, c });
                    }
                }
            }
            break;
        case mfem::Geometry::CUBE:
            for (int a = 0; a <= order; ++a) {
                for (int b = 0; b <= order; ++b) {
                    for (int c = 0; c <= order; ++c) { e.push_back({ a, b, c }); }
                }
            }
            break;
        default:  // SQUARE
            for (int a = 0; a <= order; ++a) {
                for (int b = 0; b <= order; ++b) { e.push_back({ a, b, 0 }); }
            }
            break;
    }
    return e;
}

 std::vector<std::vector<double>> LagrangeMonomials1D(int p) {
    std::vector<std::vector<double>> coeffs(p + 1, std::vector<double>(p + 1, 0.0));
    for (int a = 0; a <= p; ++a) {
        std::vector<double> poly{ 1.0 };  // ascending powers of u
        double denom = 1.0;
        for (int b = 0; b <= p; ++b) {
            if (b == a) { continue; }
            const double tb = static_cast<double>(b) / p;
            std::vector<double> next(poly.size() + 1, 0.0);
            for (size_t k = 0; k < poly.size(); ++k) {
                next[k + 1] += poly[k];
                next[k] -= tb * poly[k];
            }
            poly.swap(next);
            denom *= (static_cast<double>(a) - b) / p;
        }
        for (int k = 0; k <= p; ++k) { coeffs[a][k] = poly[k] / denom; }
    }
    return coeffs;
}

 InterpScheme BuildTensorInterpScheme(mfem::Geometry::Type geom, int order,
                                            const HoLayout& layout) {
    InterpScheme scheme;
    scheme.exponents = MonomialExponents(geom, order);
    const auto c1 = LagrangeMonomials1D(order);
    const int dim = mfem::Geometry::Dimension[geom];

    const int n = static_cast<int>(layout.ref.size());
    scheme.coeffs.assign(n, std::vector<double>(scheme.exponents.size(), 0.0));
    for (int i = 0; i < n; ++i) {
        // Lattice index of node i along each axis.
        int idx[3] = { 0, 0, 0 };
        for (int d = 0; d < dim; ++d) {
            idx[d] = static_cast<int>(std::lround(layout.ref[i][d] * order));
        }
        for (size_t j = 0; j < scheme.exponents.size(); ++j) {
            double c = 1.0;
            for (int d = 0; d < dim; ++d) { c *= c1[idx[d]][scheme.exponents[j][d]]; }
            scheme.coeffs[i][j] = c;
        }
    }
    return scheme;
}

 InterpScheme BuildInterpScheme(mfem::Geometry::Type geom, int order) {
    const HoLayout& layout = GetHoLayout(geom, order);
    if (geom == mfem::Geometry::SQUARE || geom == mfem::Geometry::CUBE) {
        return BuildTensorInterpScheme(geom, order, layout);
    }

    InterpScheme scheme;
    scheme.exponents = MonomialExponents(geom, order);

    const int n = static_cast<int>(layout.ref.size());
    const int m = static_cast<int>(scheme.exponents.size());
    if (n != m) {
        throw std::runtime_error(
            "gmsh_results: interpolation basis size (" + std::to_string(m)
            + ") does not match node count (" + std::to_string(n)
            + ") for a " + GeometryName(geom) + " of order "
            + std::to_string(order));
    }

    // Augmented [V | I]; Gauss-Jordan leaves V^-1 in the right half.
    std::vector<std::vector<double>> a(n, std::vector<double>(2 * n, 0.0));
    for (int k = 0; k < n; ++k) {
        const double u = layout.ref[k][0];
        const double v = layout.ref[k][1];
        const double w = layout.ref[k][2];
        for (int j = 0; j < n; ++j) {
            a[k][j] = std::pow(u, scheme.exponents[j][0]) *
                      std::pow(v, scheme.exponents[j][1]) *
                      std::pow(w, scheme.exponents[j][2]);
        }
        a[k][n + k] = 1.0;
    }

    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r) {
            if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) { piv = r; }
        }
        if (std::fabs(a[piv][col]) < 1e-12) {
            throw std::runtime_error(
                "gmsh_results: singular Vandermonde matrix building the "
                "interpolation scheme for a " + std::string(GeometryName(geom))
                + " of order " + std::to_string(order));
        }
        std::swap(a[col], a[piv]);

        const double inv_p = 1.0 / a[col][col];
        for (int j = 0; j < 2 * n; ++j) { a[col][j] *= inv_p; }
        for (int r = 0; r < n; ++r) {
            if (r == col) { continue; }
            const double f = a[r][col];
            if (f == 0.0) { continue; }
            for (int j = 0; j < 2 * n; ++j) { a[r][j] -= f * a[col][j]; }
        }
    }

    // C = (V^-1)^T: row i holds the monomial coefficients of phi_i.
    scheme.coeffs.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) { scheme.coeffs[i][j] = a[j][n + i]; }
    }
    return scheme;
}

 ExportNodes BuildExportNodes(mfem::Mesh& mesh,
                                    mfem::FiniteElementSpace& fes,
                                    int order) {
    ExportNodes nodes;
    nodes.order = order;

    const int nd = fes.GetNDofs();
    const int ne = mesh.GetNE();
    nodes.coord.assign(nd, { 0.0, 0.0, 0.0 });
    nodes.node_elem.assign(nd, -1);
    nodes.node_ref.assign(nd, RefPt{ 0.0, 0.0, 0.0 });
    nodes.elem_type.assign(ne, 0);
    nodes.elem_nodes.assign(ne, {});

    std::map<int, std::vector<int>> perm_cache;  // keyed by geometry
    mfem::Array<int> dofs;
    mfem::Vector phys;

    for (int e = 0; e < ne; ++e) {
        const auto geom = mesh.GetElementBaseGeometry(e);
        const HoLayout& layout = GetHoLayout(geom, order);
        nodes.elem_type[e] = layout.gmsh_type;

        auto pit = perm_cache.find(static_cast<int>(geom));
        if (pit == perm_cache.end()) {
            pit = perm_cache.emplace(
                static_cast<int>(geom),
                BuildDofPermutation(*fes.GetFE(e), layout)).first;
        }
        const std::vector<int>& perm = pit->second;

        fes.GetElementDofs(e, dofs);
        mfem::ElementTransformation* T = mesh.GetElementTransformation(e);

        const int n = static_cast<int>(layout.ref.size());
        nodes.elem_nodes[e].resize(n);
        for (int k = 0; k < n; ++k) {
            const int dof = dofs[perm[k]];
            nodes.elem_nodes[e][k] = dof;

            mfem::IntegrationPoint ip;
            ip.Set3(layout.ref[k][0], layout.ref[k][1], layout.ref[k][2]);

            // Transform() gives the true geometric position for curved
            // elements (mesh.GetNodes() populated) and reduces to the expected
            // edge midpoints/lattice for straight-sided ones, so no separate
            // curved-vs-straight handling is needed.
            T->SetIntPoint(&ip);
            T->Transform(ip, phys);

            nodes.coord[dof] = { phys(0),
                                 phys.Size() > 1 ? phys(1) : 0.0,
                                 phys.Size() > 2 ? phys(2) : 0.0 };
            nodes.node_elem[dof] = e;
            nodes.node_ref[dof] = layout.ref[k];
        }
    }
    return nodes;
}

 void WriteMeshFormat(std::ostream& out, MshVersion version) {
    // Fields are: version, file-type (0 = ASCII), data-size (sizeof(double)).
    out << (version == MshVersion::V4_1
                ? "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
                : "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n");
}

 void WriteMeshBlock22(std::ostream& out, mfem::Mesh& mesh,
                             const ExportNodes& nodes) {
    const int nv = static_cast<int>(nodes.coord.size());
    const int ne = mesh.GetNE();

    // Stage each section into a single std::string and flush with one
    // out.write(). Avoids per-token ostream overhead and lets the OS see
    // bulk I/O.
    std::string blk;
    blk.reserve(static_cast<size_t>(nv) * 48 + 64);

    blk.assign("$Nodes\n");
    AppendInt(blk, nv);
    blk.push_back('\n');
    for (int i = 0; i < nv; ++i) {
        AppendInt(blk, i + 1);
        blk.push_back(' '); AppendDouble(blk, nodes.coord[i][0]);
        blk.push_back(' '); AppendDouble(blk, nodes.coord[i][1]);
        blk.push_back(' '); AppendDouble(blk, nodes.coord[i][2]);
        blk.push_back('\n');
    }
    blk.append("$EndNodes\n");
    out.write(blk.data(), static_cast<std::streamsize>(blk.size()));

    blk.clear();
    blk.reserve(static_cast<size_t>(ne) * 32 + 64);
    blk.append("$Elements\n");
    AppendInt(blk, ne);
    blk.push_back('\n');
    for (int e = 0; e < ne; ++e) {
        const int attr = mesh.GetAttribute(e);
        AppendInt(blk, e + 1);
        blk.push_back(' '); AppendInt(blk, nodes.elem_type[e]);
        blk.append(" 2 "); AppendInt(blk, attr);
        blk.push_back(' '); AppendInt(blk, attr);
        for (int id : nodes.elem_nodes[e]) {
            blk.push_back(' ');
            AppendInt(blk, id + 1);
        }
        blk.push_back('\n');

        if (blk.size() > (1u << 20)) {
            out.write(blk.data(), static_cast<std::streamsize>(blk.size()));
            blk.clear();
        }
    }
    blk.append("$EndElements\n");
    out.write(blk.data(), static_cast<std::streamsize>(blk.size()));
}

 void WriteMeshBlock41(std::ostream& out, mfem::Mesh& mesh,
                             const ExportNodes& nodes) {
    const int nv = static_cast<int>(nodes.coord.size());
    const int ne = mesh.GetNE();
    const int dim = mesh.Dimension();
    MFEM_VERIFY(dim == 2 || dim == 3,
        "gmsh_results: MSH 4.1 export supports 2D and 3D meshes only.");

    // Group elements by (attribute, gmsh element type). A single attribute may
    // legitimately contain both triangles and quads, and MSH 4.1 requires one
    // block per element type, so the pair is the block key. std::map keeps the
    // emission order deterministic.
    std::map<std::pair<int, int>, std::vector<int>> blocks;
    for (int e = 0; e < ne; ++e) {
        blocks[{ mesh.GetAttribute(e), nodes.elem_type[e] }].push_back(e);
    }

    // Distinct attributes become the surface (2D) or volume (3D) entities.
    std::vector<int> attrs;
    for (const auto& kv : blocks) {
        if (std::find(attrs.begin(), attrs.end(), kv.first.first) == attrs.end()) {
            attrs.push_back(kv.first.first);
        }
    }
    std::sort(attrs.begin(), attrs.end());

    std::string s;

    // $Entities: numPoints numCurves numSurfaces numVolumes, then one line per
    // entity: tag, bounding box, physical tags, bounding entities of one
    // dimension lower (none, since we do not synthesize a lower topology).
    s.append(dim == 3 ? "$Entities\n0 0 0 " : "$Entities\n0 0 ");
    AppendInt(s, static_cast<long long>(attrs.size()));
    s.append(dim == 3 ? "\n" : " 0\n");
    for (int attr : attrs) {
        // Bounding box over the export nodes of every element with this
        // attribute. Gmsh tolerates a loose box; it is used for display only.
        double lo[3] = { 1e300, 1e300, 1e300 };
        double hi[3] = { -1e300, -1e300, -1e300 };
        for (const auto& kv : blocks) {
            if (kv.first.first != attr) { continue; }
            for (int e : kv.second) {
                for (int id : nodes.elem_nodes[e]) {
                    for (int c = 0; c < 3; ++c) {
                        lo[c] = std::min(lo[c], nodes.coord[id][c]);
                        hi[c] = std::max(hi[c], nodes.coord[id][c]);
                    }
                }
            }
        }
        AppendInt(s, attr);
        for (int c = 0; c < 3; ++c) { s.push_back(' '); AppendDouble(s, lo[c]); }
        for (int c = 0; c < 3; ++c) { s.push_back(' '); AppendDouble(s, hi[c]); }
        s.append(" 1 ");        // one physical tag...
        AppendInt(s, attr);     // ...which is the attribute itself
        s.append(" 0\n");       // zero bounding curves / surfaces
    }
    s.append("$EndEntities\n");
    out.write(s.data(), static_cast<std::streamsize>(s.size()));

    // $Nodes: numEntityBlocks numNodes minNodeTag maxNodeTag, then per block
    // entityDim entityTag parametric numNodesInBlock, all tags, then all
    // coordinates. Export nodes are shared across attributes, and MSH 4.1
    // requires each node tag to appear exactly once, so all nodes go in a
    // single block classified on the first entity.
    s.clear();
    s.reserve(static_cast<size_t>(nv) * 48 + 128);
    if (attrs.empty() || nv == 0) {
        s.append("$Nodes\n0 0 0 0\n$EndNodes\n");
    } else {
        s.append("$Nodes\n1 ");
        AppendInt(s, nv); s.append(" 1 "); AppendInt(s, nv); s.push_back('\n');
        AppendInt(s, dim);           // entityDim (2 = surface, 3 = volume)
        s.push_back(' ');
        AppendInt(s, attrs.front()); // entityTag
        s.append(" 0 ");            // parametric = 0
        AppendInt(s, nv); s.push_back('\n');
        for (int i = 0; i < nv; ++i) {
            AppendInt(s, i + 1);
            s.push_back('\n');
        }
        for (int i = 0; i < nv; ++i) {
            AppendDouble(s, nodes.coord[i][0]);
            s.push_back(' '); AppendDouble(s, nodes.coord[i][1]);
            s.push_back(' '); AppendDouble(s, nodes.coord[i][2]);
            s.push_back('\n');
        }
        s.append("$EndNodes\n");
    }
    out.write(s.data(), static_cast<std::streamsize>(s.size()));

    // $Elements: numEntityBlocks numElements minElementTag maxElementTag, then
    // per block entityDim entityTag elementType numElementsInBlock followed by
    // "elementTag nodeTag..." lines. Element tags keep the global 1..ne
    // numbering so they still line up with $ElementNodeData below.
    s.clear();
    s.reserve(static_cast<size_t>(ne) * 32 + 128);
    s.append("$Elements\n");
    AppendInt(s, static_cast<long long>(blocks.size()));
    s.push_back(' '); AppendInt(s, ne);
    s.append(ne > 0 ? " 1 " : " 0 ");
    AppendInt(s, ne);
    s.push_back('\n');
    for (const auto& kv : blocks) {
        AppendInt(s, dim);                       // entityDim
        s.push_back(' ');
        AppendInt(s, kv.first.first);            // entityTag = attribute
        s.push_back(' '); AppendInt(s, kv.first.second);  // elementType
        s.push_back(' ');
        AppendInt(s, static_cast<long long>(kv.second.size()));
        s.push_back('\n');
        for (int e : kv.second) {
            AppendInt(s, e + 1);
            for (int id : nodes.elem_nodes[e]) {
                s.push_back(' ');
                AppendInt(s, id + 1);
            }
            s.push_back('\n');

            if (s.size() > (1u << 20)) {
                out.write(s.data(), static_cast<std::streamsize>(s.size()));
                s.clear();
            }
        }
    }
    s.append("$EndElements\n");
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

 void WriteMeshBlock(std::ostream& out, mfem::Mesh& mesh,
                           const ExportNodes& nodes, MshVersion version) {
    WriteMeshFormat(out, version);
    if (version == MshVersion::V4_1) {
        WriteMeshBlock41(out, mesh, nodes);
    } else {
        WriteMeshBlock22(out, mesh, nodes);
    }
}

 int GmshFamily(mfem::Geometry::Type geom) {
    switch (geom) {
        case mfem::Geometry::TRIANGLE:    return 3;
        case mfem::Geometry::SQUARE:      return 4;
        case mfem::Geometry::TETRAHEDRON: return 5;
        case mfem::Geometry::CUBE:        return 8;
        default:
            throw std::runtime_error(std::string("gmsh_results: no Gmsh element family for a ")
                                     + GeometryName(geom));
    }
}

 void WriteInterpolationScheme(std::ostream& out,
                                     const std::string& scheme_name,
                                     mfem::Mesh& mesh,
                                     int order) {
    // Collect the distinct geometries actually used, preserving a stable order.
    std::vector<mfem::Geometry::Type> geoms;
    for (int e = 0; e < mesh.GetNE(); ++e) {
        const auto g = mesh.GetElementBaseGeometry(e);
        if (std::find(geoms.begin(), geoms.end(), g) == geoms.end()) {
            geoms.push_back(g);
        }
    }
    if (geoms.empty()) { return; }

    std::string s;
    s.append("$InterpolationScheme\n");
    s.append("\"" + scheme_name + "\"\n");
    AppendInt(s, static_cast<long long>(geoms.size()));
    s.push_back('\n');

    for (const auto g : geoms) {
        const HoLayout&    layout = GetHoLayout(g, order);
        const InterpScheme scheme = BuildInterpScheme(g, order);
        const int n = static_cast<int>(scheme.coeffs.size());
        const int m = static_cast<int>(scheme.exponents.size());

        AppendInt(s, GmshFamily(g));
        s.append("\n2\n");  // two matrices follow: coefficients, then exponents

        AppendInt(s, n); s.push_back(' '); AppendInt(s, m); s.push_back('\n');
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (j) { s.push_back(' '); }
                AppendDouble(s, scheme.coeffs[i][j]);
            }
            s.push_back('\n');
        }

        // One exponent column per reference coordinate of this element.
        const int n_vars = mfem::Geometry::Dimension[g];
        AppendInt(s, m); s.push_back(' '); AppendInt(s, n_vars); s.push_back('\n');
        for (int j = 0; j < m; ++j) {
            for (int c = 0; c < n_vars; ++c) {
                if (c) { s.push_back(' '); }
                AppendInt(s, scheme.exponents[j][c]);
            }
            s.push_back('\n');
        }
    }

    s.append("$EndInterpolationScheme\n");
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

 void WriteViewHeader(std::ostream& out,
                            const char*   tag,
                            const std::string& name,
                            const std::string& scheme_name,
                            int num_components,
                            int num_entities) {
    std::string s;
    s.reserve(96 + name.size() + scheme_name.size());
    s.push_back('$'); s.append(tag); s.push_back('\n');
    // StringTags: [0] view name, [1] interpolation scheme name. The second tag
    // is what binds this view to the $InterpolationScheme block above.
    s.append("2\n\""); s.append(name); s.append("\"\n");
    s.append("\""); s.append(scheme_name); s.append("\"\n");
    s.append("1\n0.0\n");
    s.append("3\n0\n");
    AppendInt(s, num_components); s.push_back('\n');
    AppendInt(s, num_entities);   s.push_back('\n');
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

 void WriteNodeData(std::ostream& out, mfem::Mesh& mesh,
                          const ExportNodes& nodes, const View& v,
                          const std::string& scheme_name) {
    const int nv = static_cast<int>(nodes.coord.size());
    WriteViewHeader(out, "NodeData", v.name, scheme_name, v.num_components, nv);

    std::vector<double> buf(v.num_components);
    std::string blk;
    blk.reserve(static_cast<size_t>(nv) * (16 + v.num_components * 18));
    for (int i = 0; i < nv; ++i) {
        // Sample at the node's recorded reference point rather than reading a
        // DOF directly: the field may live in a different space or basis than
        // the equispaced one used for node numbering.
        mfem::IntegrationPoint ip;
        ip.Set3(nodes.node_ref[i][0], nodes.node_ref[i][1], nodes.node_ref[i][2]);
        mfem::ElementTransformation* T =
            mesh.GetElementTransformation(nodes.node_elem[i]);
        T->SetIntPoint(&ip);
        v.elem_node_eval(nodes.node_elem[i], ip, *T, buf.data());

        AppendInt(blk, i + 1);
        for (int c = 0; c < v.num_components; ++c) {
            blk.push_back(' ');
            AppendDouble(blk, buf[c]);
        }
        blk.push_back('\n');

        if (blk.size() > (1u << 20)) {
            out.write(blk.data(), static_cast<std::streamsize>(blk.size()));
            blk.clear();
        }
    }
    blk.append("$EndNodeData\n");
    out.write(blk.data(), static_cast<std::streamsize>(blk.size()));
}

 void WriteElementNodeData(std::ostream& out,
                                 mfem::Mesh& mesh,
                                 const ExportNodes& nodes,
                                 const View& v,
                                 const std::string& scheme_name) {
    const int ne = mesh.GetNE();
    WriteViewHeader(out, "ElementNodeData", v.name, scheme_name,
                    v.num_components, ne);

    std::vector<double> buf(v.num_components);
    std::string blk;
    // Rough upper bound; grows on demand if needed.
    blk.reserve(static_cast<size_t>(ne) * (16 + 4 * v.num_components * 18));
    for (int e = 0; e < ne; ++e) {
        // Values must be listed in the same node order as the element's
        // connectivity in the mesh block, so drive the loop from the Gmsh
        // layout rather than from an MFEM space's own nodal ordering.
        const HoLayout& layout =
            GetHoLayout(mesh.GetElementBaseGeometry(e), nodes.order);
        mfem::ElementTransformation* T = mesh.GetElementTransformation(e);

        const int n_local = static_cast<int>(layout.ref.size());
        AppendInt(blk, e + 1);
        blk.push_back(' ');
        AppendInt(blk, n_local);
        for (int k = 0; k < n_local; ++k) {
            mfem::IntegrationPoint ip;
            ip.Set3(layout.ref[k][0], layout.ref[k][1], layout.ref[k][2]);
            T->SetIntPoint(&ip);
            v.elem_node_eval(e, ip, *T, buf.data());
            for (int c = 0; c < v.num_components; ++c) {
                blk.push_back(' ');
                AppendDouble(blk, buf[c]);
            }
        }
        blk.push_back('\n');

        // Flush periodically so the staging buffer doesn't grow without
        // bound for very large meshes.
        if (blk.size() > (1u << 20)) {
            out.write(blk.data(), static_cast<std::streamsize>(blk.size()));
            blk.clear();
        }
    }
    blk.append("$EndElementNodeData\n");
    out.write(blk.data(), static_cast<std::streamsize>(blk.size()));
}

} // namespace detail

 View MakeScalarNodeView(const std::string& name,
                               mfem::GridFunction& gf) {
    View v;
    v.name = name;
    v.kind = View::Kind::NodeData;
    v.num_components = 1;
    v.elem_node_eval = [&gf](int /*elem_id*/,
                             const mfem::IntegrationPoint& ip,
                             mfem::ElementTransformation& T,
                             double* out) {
        out[0] = gf.GetValue(T, ip);
    };
    return v;
}

 View MakeVectorGridFunctionView(const std::string& name,
                                       mfem::GridFunction& gf) {
    View v;
    v.name = name;
    v.kind = View::Kind::ElementNodeData;
    v.num_components = 3;
    v.elem_node_eval = [&gf](int /*elem_id*/,
                             const mfem::IntegrationPoint& ip,
                             mfem::ElementTransformation& T,
                             double* out) {
        mfem::Vector val;
        gf.GetVectorValue(T, ip, val);
        out[0] = val.Size() > 0 ? val(0) : 0.0;
        out[1] = val.Size() > 1 ? val(1) : 0.0;
        out[2] = val.Size() > 2 ? val(2) : 0.0;
    };
    return v;
}

 View MakeVectorCoefficientView(const std::string& name,
                                      mfem::VectorCoefficient& vec_coeff) {
    View v;
    v.name = name;
    v.kind = View::Kind::ElementNodeData;
    v.num_components = 3;
    v.elem_node_eval = [&vec_coeff](int /*elem_id*/,
                                    const mfem::IntegrationPoint& ip,
                                    mfem::ElementTransformation& T,
                                    double* out) {
        mfem::Vector val(vec_coeff.GetVDim());
        vec_coeff.Eval(val, T, ip);
        out[0] = val.Size() > 0 ? val(0) : 0.0;
        out[1] = val.Size() > 1 ? val(1) : 0.0;
        out[2] = val.Size() > 2 ? val(2) : 0.0;
    };
    return v;
}

 View MakeScalarCoefficientView(const std::string& name,
                                      mfem::Coefficient& coeff) {
    View v;
    v.name = name;
    v.kind = View::Kind::ElementNodeData;
    v.num_components = 1;
    v.elem_node_eval = [&coeff](int /*elem_id*/,
                                const mfem::IntegrationPoint& ip,
                                mfem::ElementTransformation& T,
                                double* out) {
        out[0] = coeff.Eval(T, ip);
    };
    return v;
}

 void WriteGmshResults(const std::string& path,
                             mfem::Mesh& mesh,
                             int order,
                             const std::vector<View>& views,
                             MshVersion version,
                             MeshExport* cache) {
    // std::ofstream / fopen will not create missing parent directories on
    // Windows or POSIX; opening "./foo/bar.msh" silently fails with ENOENT
    // when ./foo does not yet exist. Create the parent chain up front so a
    // results path like "<mesh-dir>/<case>.results.msh" works on a fresh
    // working tree.
    namespace fs = std::filesystem;
    fs::path out_path(path);
    if (out_path.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(out_path.parent_path(), ec);
        if (ec) {
            throw std::runtime_error(
                "WriteGmshResults: cannot create directory '"
                + out_path.parent_path().string() + "': " + ec.message());
        }
    }

    // Open in binary mode so '\n' is not translated to "\r\n" on Windows;
    // halves write traffic on text-heavy MSH output. Give the filebuf a
    // 1 MiB buffer (default is 8 KiB) so the staged std::string blocks
    // emitted by WriteMeshBlock / WriteNodeData / WriteElementNodeData
    // hit the OS in a handful of large writes instead of hundreds.
    std::ofstream out;
    static thread_local std::array<char, 1u << 20> io_buf;
    out.rdbuf()->pubsetbuf(io_buf.data(),
                           static_cast<std::streamsize>(io_buf.size()));
    out.open(out_path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::error_code cwd_ec;
        const auto cwd = fs::current_path(cwd_ec);
        throw std::runtime_error(
            "WriteGmshResults: cannot open '" + path + "': "
            + std::strerror(errno)
            + " (cwd=" + (cwd_ec ? std::string("?") : cwd.string()) + ")");
    }
    // Numeric formatting goes through std::to_chars in detail::AppendDouble
    // (locale-independent, ~5-10x faster than operator<<). No stream-side
    // setprecision / scientific needed.

    // One scheme shared by every view: all views are sampled on the same node
    // layout at the same order, so they interpolate with the same basis.
    const std::string scheme_name = "MFEM_Lagrange_P" + std::to_string(order);
    MeshExport local;
    MeshExport& part = cache ? *cache : local;
    if (part.mesh != &mesh || part.sequence != mesh.GetSequence() ||
        part.order != order || part.version != version) {
        // Equispaced nodes so the DOF positions coincide with Gmsh's Lagrange
        // node lattice; the resulting DOF indices double as shared node ids.
        mfem::H1_FECollection fec(order, mesh.Dimension(),
                                  mfem::BasisType::ClosedUniform);
        mfem::FiniteElementSpace fes(&mesh, &fec);
        part.nodes = detail::BuildExportNodes(mesh, fes, order);
        std::ostringstream block;
        detail::WriteMeshBlock(block, mesh, part.nodes, version);
        detail::WriteInterpolationScheme(block, scheme_name, mesh, order);
        part.block = block.str();
        part.mesh = &mesh;
        part.sequence = mesh.GetSequence();
        part.order = order;
        part.version = version;
    }
    const detail::ExportNodes& nodes = part.nodes;
    out.write(part.block.data(), static_cast<std::streamsize>(part.block.size()));

    for (const auto& v : views) {
        switch (v.kind) {
            case View::Kind::NodeData:
                detail::WriteNodeData(out, mesh, nodes, v, scheme_name);
                break;
            case View::Kind::ElementNodeData:
                detail::WriteElementNodeData(out, mesh, nodes, v, scheme_name);
                break;
        }
    }
    out.close();
    if (out.fail()) {
        throw std::runtime_error("WriteGmshResults: writing '" + path + "' failed: "
                                 + std::strerror(errno));
    }
}

} // namespace gmsh_results
