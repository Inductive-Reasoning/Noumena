#!/bin/bash
# Script to generate all example meshes using Gmsh

set -e  # Exit on error

echo "Generating example meshes..."

# Check if gmsh is available
if ! command -v gmsh &> /dev/null; then
    echo "Error: gmsh not found. Please install gmsh:"
    echo "  Ubuntu/Debian: sudo apt-get install gmsh"
    echo "  macOS: brew install gmsh"
    echo "  Or download from: https://gmsh.info/"
    exit 1
fi

# Function to generate mesh
generate_mesh() {
    local geo_file=$1
    local mesh_file=$2
    local example_name=$3

    echo ""
    echo "==================================="
    echo "Generating: $example_name"
    echo "==================================="

    if [ ! -f "$geo_file" ]; then
        echo "Error: $geo_file not found"
        return 1
    fi

    # Generate mesh with Gmsh
    # -2: generate 2D mesh
    # -format msh2: output in MSH2 ASCII format (compatible with MFEM)
    # -o: output file
    # A .msh target (what the example's config reads) is Gmsh's output
    # itself; a .mesh target is converted to MFEM format, or renamed if the
    # converter is not available (MFEM reads either format).
    local gmsh_file="${mesh_file%.*}.msh"
    echo "Running: gmsh -2 -format msh2 $geo_file -o $gmsh_file"
    gmsh -2 -format msh2 "$geo_file" -o "$gmsh_file"

    if [[ "$mesh_file" == *.mesh ]]; then
        if command -v convert-mesh &> /dev/null; then
            echo "Converting to MFEM format..."
            convert-mesh "$gmsh_file" "$mesh_file"
            rm -f "$gmsh_file"
        else
            echo "Note: convert-mesh not found. Keeping Gmsh .msh format."
            echo "MFEM can read .msh files directly, or you can:"
            echo "  1. Build MFEM and use: mfem/miniapps/tools/convert-mesh"
            echo "  2. Rename .msh to .mesh (MFEM auto-detects format)"
            mv "$gmsh_file" "$mesh_file"
        fi
    fi
    echo "Generated: $mesh_file"

    # Print mesh statistics
    if [ -f "$mesh_file" ]; then
        echo ""
        echo "Mesh statistics:"
        if [[ "$mesh_file" == *.msh ]]; then
            num_elements=$(grep -A1 '\$Elements' "$mesh_file" | tail -1)
            echo "  Elements: $num_elements"
        else
            echo "  File size: $(du -h "$mesh_file" | cut -f1)"
        fi
    fi

    echo "Done: $example_name"
}

# Generate each mesh
generate_mesh \
    "eddy_current/eddy_current.geo" \
    "eddy_current/eddy_current.mesh" \
    "Eddy Current (Conducting Cylinder)"

generate_mesh \
    "simple_capacitor/capacitor.geo" \
    "simple_capacitor/capacitor.mesh" \
    "Parallel Plate Capacitor"

generate_mesh \
    "solenoid/solenoid.geo" \
    "solenoid/solenoid.mesh" \
    "Axisymmetric Solenoid"

generate_mesh \
    "two_loops/case.geo" \
    "two_loops/case.msh" \
    "Two Loops (Coupled Inductors)"

echo ""
echo "==================================="
echo "All meshes generated successfully!"
echo "==================================="
echo ""
echo "You can now run the examples:"
echo "  cd ../build"
echo "  ./noumena ../examples/eddy_current/config.json"
echo "  ./noumena ../examples/simple_capacitor/config.json"
echo "  ./noumena ../examples/solenoid/config.json"
echo "  ./noumena ../examples/two_loops/config.json"
