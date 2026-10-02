# MFEM-ElectroMag

A finite element solver for electromagnetic problems using MFEM (Modular Finite Element Methods). This solver supports electrostatic, magnetostatic, and magnetoquasistatic problems in axisymmetric and planar 2D geometries and in full 3D.

## Features

- **Electrostatics**: Solves for electric potential and field distributions
- **Magnetostatics**: Solves for magnetic vector potential and field distributions
- **Magnetoquasistatics**: Time-harmonic eddy current problems
- **Axisymmetric, Planar and 3D**: Both 2D coordinate systems, plus full 3D for every physics type
- **JSON Configuration**: Easy problem setup via JSON files
- **ParaView Output**: Direct visualization of results
- **Comprehensive Testing**: Unit and integration tests with Catch2
- **Configuration Validation**: Automatic checking of input files
- **Solver Factory Pattern**: Extensible architecture for new physics types
- **OpenMP Support**: Parallel assembly for improved performance
- **API Documentation**: Doxygen-generated documentation

## Dependencies

### Required

- **CMake** (>= 3.18): Build system (minimum required by HDF5)
- **C and C++ Compilers**: C for HDF5 and C++17 support for the solver
  - GCC (≥ 7.0)
  - Clang (≥ 5.0)
  - MSVC (≥ 2017)
- **MFEM** (v4.7): Automatically downloaded and built by CMake
- **HDF5** (1.14.6): C library, automatically downloaded and built statically by CMake
- **HighFive** (3.1.1): Header-only C++ wrapper for HDF5, automatically downloaded by CMake
- **Eigen** (3.4.0): Header-only sparse direct solvers (MPL2 subset only), automatically downloaded by CMake
- **AMGCL** (1.5.0): Header-only algebraic multigrid (MIT) for the iterative solver, automatically downloaded by CMake

No separate HDF5 installation is required. The HDF5 build excludes optional tools,
language bindings, MPI, and zlib/SZip compression dependencies. HDF5 and HighFive
provide [coupling-matrix output](docs/coupling_hdf5.md) for all three solvers.
Coupling matrices use HDF5 instead of CSV; field output formats are unchanged.

### Optional

- **MPI** (OpenMPI, MPICH, or MS-MPI on Windows): only for the MPI build
  (`-DUSE_MPI=ON`), which adds HYPRE's BoomerAMG and AMS solvers
- **HYPRE** (v3.0.0): built automatically in the MPI build, or supplied via `HYPRE_DIR`
- **METIS** (5.x): optional in the MPI build, supplied via `METIS_DIR`; only
  needed for multi-rank partitioning, which is not implemented yet
- **OpenMP**: threads the linear algebra (usually included with the compiler); see [Threads](#threads)
- **Doxygen**: For generating API documentation
- **Catch2**: For running tests (automatically downloaded)

## Installation Instructions

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y cmake g++ git

# Clone the repository
git clone https://github.com/xfmrexpert/MFEM-ElectroMag.git
cd MFEM-ElectroMag

# Build
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### macOS

```bash
# Install dependencies using Homebrew
brew install cmake git

# Clone the repository
git clone https://github.com/xfmrexpert/MFEM-ElectroMag.git
cd MFEM-ElectroMag

# Build
mkdir build
cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Windows

Using Visual Studio:

```powershell
# Install CMake and Git
# Download from: https://cmake.org/download/ and https://git-scm.com/

# Clone the repository
git clone https://github.com/xfmrexpert/MFEM-ElectroMag.git
cd MFEM-ElectroMag

# Create build directory
mkdir build
cd build

# Generate Visual Studio project
cmake ..

# Build using CMake
cmake --build . --config Release

# Or open the generated .sln file in Visual Studio and build
```

Using MinGW/MSYS2:

```bash
# Install dependencies
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc git

# Clone and build
git clone https://github.com/xfmrexpert/MFEM-ElectroMag.git
cd MFEM-ElectroMag
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j
```

## MPI/HYPRE Build

The default build is serial and needs no MPI. The MPI build links MFEM against
MPI and HYPRE, which the 3D magnetic solvers need for their iterative
(AMS-preconditioned) solves:

```bash
sudo apt-get install -y openmpi-bin libopenmpi-dev   # or mpich
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON
cmake --build build-mpi -j
```

The first configure fetches and builds HYPRE into `build-mpi/tpl/hypre`
(a few minutes, once). To use an existing HYPRE instead, pass
`-DHYPRE_DIR=<prefix>`; with vcpkg on Windows, install the `msmpi` and `hypre`
ports and point `HYPRE_DIR` at the vcpkg installation. `-DMETIS_DIR=<prefix>`
enables METIS.

An MPI build currently runs on **one rank**: run the executable directly or
with `mpirun -np 1`. Starting more ranks is rejected, since the solvers and
result writers are not distributed yet. `--version` reports whether a binary
is a `serial` or `MPI/HYPRE` build.

### Threads

With `USE_OPENMP` (on by default) the linear algebra is threaded over
`OMP_NUM_THREADS` threads (default: every core): the Krylov solvers' matrix
and vector operations (MFEM's OpenMP backend), AMGCL's multigrid and, in the
MPI build, HYPRE's AMS and BoomerAMG. Assembly and the rest stay serial. On
four cores the TEAM 7 eddy-current solve (0.9M complex unknowns) runs 2.5x
faster than on one (168 s against 428 s).

Under `mpirun`, Open MPI binds a rank to one core, which leaves every thread
sharing it. Run the executable directly, or pass `--bind-to none`
(`mpirun --bind-to none -np 1 ./mfem-electromag config.json`).

## Usage

After building, the executable `mfem-electromag` will be in the `build` directory:

```bash
# Run with a configuration file
./mfem-electromag path/to/config.json

# Example: Run test cases
./mfem-electromag ../test/electrostatic_test.json
./mfem-electromag ../test/magnetostatic_test.json
./mfem-electromag ../test/mqs_test.json

# Run with OpenMP parallelism (if enabled)
OMP_NUM_THREADS=4 ./mfem-electromag config.json
```

### Command-Line Options

| Option | Description |
| --- | --- |
| `<config.json>` | Path to the configuration file (default: `config.json`) |
| `--output-directory <directory>` | Override `output.directory`. A relative path resolves against the current working directory, not the config file directory. |
| `--verbosity <0\|1\|2>` | `0` = status/timing only, `1` = solver output, `2` = diagnostics |
| `--machine-readable` | Emit flushed JSON Lines progress on stdout. Implies `--verbosity 1` unless `--verbosity` is given explicitly. |
| `--version` | Print version/build information and exit |
| `-h`, `--help` | Show help and exit |

All other settings are configured in the JSON input file rather than on the
command line. Gmsh results are written as native high-order Lagrange elements
matching `simulation.order`, so no export refinement knob is needed.

### Running Tests

```bash
# In build directory
ctest --output-on-failure

# Or run test executable directly
./mfem_tests
```

### Generating Documentation

```bash
# In build directory (requires Doxygen)
make docs

# Documentation will be in docs/doxygen/html/index.html
```

## Configuration

Problems are configured using JSON files. All input is **SI**, and mesh
coordinates must be in **metres** -- material properties are per-metre and the
axisymmetric measure uses the radial coordinate as a physical length, so a mesh
in millimetres solves cleanly and returns silently wrong absolute quantities.
There is no unit or scale key in the schema; see
[Units](docs/config_reference.md#units).

See the `test/` directory for examples of:
- `electrostatic_test.json`: Electrostatic problem setup
- `magnetostatic_test.json`: Magnetostatic problem setup
- `mqs_test.json`: Magnetoquasistatic problem setup

See `examples/` directory for complete example problems with documentation:
- `simple_capacitor/`: Parallel plate capacitor with analytical validation
- `solenoid/`: Magnetostatic coil with field calculations
- `eddy_current/`: Time-harmonic eddy current analysis

### MQS Frequency Scenarios

Magnetoquasistatic frequency is defined on every scenario, not in the
`simulation` block. A scalar performs one solve:

```json
{"name": "60Hz", "frequency": 60.0, "excitations": []}
```

A sweep uses an inclusive linear or logarithmic range:

```json
{
  "name": "Sweep",
  "frequency": {"scale": "log", "start": 10.0, "stop": 1000.0, "points": 5},
  "excitations": []
}
```

`points` includes both endpoints; with `points: 1`, only `start` is solved.
Field analyses write one result per expanded point. For MQS
`coupling_matrix` analyses, the solver writes one matrix pair per unique frequency
to the archive selected by `output.hdf5`. The `Inductance` and `Resistance` groups
share a sorted numeric frequency axis. See [the schema and C# indexing
example](docs/coupling_hdf5.md).

Enable result formats in a top-level block:

```json
"output": {
  "directory": "results",
  "export_fields_for_coupling_matrix": false,
  "paraview": {},
  "gmsh": {"version": "2.2"},
  "hdf5": {"file": "results.h5"}
}
```

Omitted formats are disabled. The HDF5 archive holds the final mesh, scenario
fields, and coupling matrices. Coupling fields are disabled by default to keep
matrix runs compact; set `export_fields_for_coupling_matrix` to true to export every unit-drive
field in all enabled formats. Relative format paths resolve beneath the root.

### Documentation

| Document | Purpose |
|----------|---------|
| [`docs/input_file_format.md`](docs/input_file_format.md) | Guide to the configuration model, its vocabulary, and a worked example |
| [`docs/config_reference.md`](docs/config_reference.md) | Normative reference: every key, type, default, and validation rule |
| [`docs/faq.md`](docs/faq.md) | Conventions and sharp edges, including the peak-vs-RMS excitation rule |
| [`docs/boundary_and_terminal_model.md`](docs/boundary_and_terminal_model.md) | Boundary, boundary-condition, and terminal modeling rules |
| [`docs/math_formulation.md`](docs/math_formulation.md) | Mathematical formulation of each physics type |
| [`docs/open_boundary.md`](docs/open_boundary.md) | Far-field truncation error, the measured convergence study, and options for true open boundaries |
| [`docs/mesh_formats.md`](docs/mesh_formats.md) | Which mesh formats carry named groups, why the config owns name to attribute binding, and Netgen feasibility |

> **Note:** excitation values in time-harmonic runs are **peak (amplitude)
> phasors**, not RMS. This is not enforced by the solver; see the
> [FAQ](docs/faq.md#are-excitations-peak-or-rms).

> **Note:** mesh coordinates must be in **metres**. This is likewise not
> enforced; see the [FAQ](docs/faq.md#what-units-does-the-mesh-use).

## Output

Results are saved in ParaView format (`.pvd`, `.vtu` files) and can be visualized using:
- [ParaView](https://www.paraview.org/): Open-source visualization application
- [VisIt](https://visit-dav.github.io/visit-website/): Scientific data visualization tool

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 T. C. Raymond

## Performance

For large-scale simulations:
- Use `Release` build type: `cmake .. -DCMAKE_BUILD_TYPE=Release`
- Enable OpenMP: `cmake .. -DUSE_OPENMP=ON` (default)
- Use configurable solver parameters in JSON:
  ```json
  {
    "simulation": {
      "solver_tolerance": 1e-10,
      "solver_max_iter": 2000,
      "solver_print_level": 1
    }
  }
  ```
- For > 100k DOFs, consider using HYPRE preconditioners (see `docs/solver_performance.md`)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**Development workflow:**
1. Run tests: `ctest` in build directory
2. Check code compiles without warnings: `-Wall -Wextra`
3. Ensure examples still run correctly
4. Update documentation for API changes

## Acknowledgments

This project uses the [MFEM](https://mfem.org) library for finite element computations and [nlohmann/json](https://github.com/nlohmann/json) for JSON parsing.
