// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Prints every Gmsh Lagrange node layout the results writer builds, for
// tools/check_gmsh_layouts.py to compare against Gmsh itself.
#include "io/gmsh_results_writer.hpp"
#include <cstdio>
using namespace gmsh_results::detail;
int main() {
  struct G { mfem::Geometry::Type g; const char* n; int maxo; };
  for (G g : { G{mfem::Geometry::TRIANGLE,"Triangle",10}, G{mfem::Geometry::SQUARE,"Quadrangle",10},
               G{mfem::Geometry::TETRAHEDRON,"Tetrahedron",10}, G{mfem::Geometry::CUBE,"Hexahedron",9} })
    for (int o = 1; o <= g.maxo; ++o) {
      const auto& L = GetHoLayout(g.g, o);
      std::printf("%s %d %d %zu\n", g.n, o, L.gmsh_type, L.ref.size());
      for (auto& r : L.ref) std::printf("%.12f %.12f %.12f\n", r[0], r[1], r[2]);
    }
}
