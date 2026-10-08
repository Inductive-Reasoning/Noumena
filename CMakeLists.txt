cmake_minimum_required(VERSION 3.18) # HDF5 requires 3.18+
project(Noumena VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Build options
option(BUILD_TESTING "Build tests" ON)
option(ENABLE_NATIVE_ARCH "Enable -march=native for Release builds" OFF)
# OpenMP threads the linear algebra: AMGCL, MFEM's sparse matrix-vector
# products and vector operations (the Krylov solvers' own work), and, in the
# MPI build, HYPRE. OMP_NUM_THREADS sets the thread count.
option(USE_OPENMP "Thread the linear algebra with OpenMP" ON)
# MPI build: MFEM with MPI and HYPRE (BoomerAMG, AMS, ...). Off by default;
# the serial build needs no MPI installation. HYPRE is built automatically
# unless HYPRE_DIR points at an existing install; see cmake/ParallelTPLs.cmake.
option(USE_MPI "Build with MPI and HYPRE" OFF)
# STRUMPACK factors the time-harmonic (MQS) systems for linear_solver
# "direct", far faster than the Eigen fallback. It needs a Fortran compiler,
# BLAS/LAPACK and METIS, and is left out with a warning if any is missing;
# see cmake/Strumpack.cmake.
option(USE_STRUMPACK "Factor the MQS systems with STRUMPACK" ON)

# Compiler flags for different build types
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2 /DNDEBUG)
        if(ENABLE_NATIVE_ARCH)
            add_compile_options(/arch:AVX2)
        endif()
    else()
        if(ENABLE_NATIVE_ARCH)
            add_compile_options(-O3 -DNDEBUG -march=native)
        else()
            add_compile_options(-O3 -DNDEBUG)
        endif()
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        add_compile_options(/Od /Zi)
        # MSVC supports AddressSanitizer (/fsanitize=address) but not UBSan.
        option(ENABLE_SANITIZERS "Enable address sanitizer" OFF)
        if(ENABLE_SANITIZERS)
            add_compile_options(/fsanitize=address)
        endif()
    else()
        add_compile_options(-O0 -g)
        # Enable sanitizers for Debug builds (optional, can be disabled)
        option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" ON)
        if(ENABLE_SANITIZERS)
            add_compile_options(-fsanitize=address -fsanitize=undefined)
            add_link_options(-fsanitize=address -fsanitize=undefined)
        endif()
    endif()
endif()

include(FetchContent)

# Make MFEM throw mfem::ErrorException on failed MFEM_ASSERT/MFEM_VERIFY
# instead of calling std::abort(). This lets us catch malformed-mesh errors
# raised from inside the mfem::Mesh constructor (e.g. CheckBdrElementOrientation
# indexing faces_info with -1 for orphan boundary edges) and report them
# uniformly. Must be set before FetchContent_MakeAvailable(mfem).
set(MFEM_USE_EXCEPTIONS ON CACHE BOOL "Throw mfem::ErrorException on MFEM errors" FORCE)

# Enable MFEM's internal debug checks (MFEM_ASSERT and the extra MFEM_VERIFY
# blocks guarded by #ifdef MFEM_DEBUG, e.g. the interior-face orientation
# check in Mesh::Finalize). Independent of CMAKE_BUILD_TYPE. Override with
# -DMFEM_DEBUG=OFF on the cmake command line if you need to match a release
# build's looser behavior. Must be set before FetchContent_MakeAvailable(mfem).
# set(MFEM_DEBUG ON CACHE BOOL "Enable MFEM internal debug checks" FORCE)

# MFEM's OpenMP backend, used once an "omp" Device is configured (see
# parallel::Initialize). The deprecated MFEM_USE_LEGACY_OPENMP is not used:
# it requires MFEM_THREAD_SAFE and gave wrong results here.
# MFEM's own configure requires OpenMP once enabled, so a compiler without it
# (Apple clang without libomp) is detected here and the build continues
# without threads instead of failing inside MFEM.
if(USE_OPENMP)
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        message(WARNING "USE_OPENMP: the compiler has no OpenMP support; building without threads.")
        set(USE_OPENMP OFF)
    endif()
endif()
set(MFEM_USE_OPENMP ${USE_OPENMP} CACHE BOOL "MFEM OpenMP backend" FORCE)

# Automatically download and build MFEM
FetchContent_Declare(
  mfem
  GIT_REPOSITORY https://github.com/mfem/mfem.git
  GIT_TAG        v4.10  # Pin to a stable version for reproducibility
  GIT_PROGRESS   TRUE  # Stream clone progress to the CMake output pane so
                       # configure does not appear to hang after
                       # "Detecting CXX compile features".
  EXCLUDE_FROM_ALL  # Don't build MFEM's miniapps/examples as part of "Build All"
)

if(USE_MPI)
    include(${CMAKE_SOURCE_DIR}/cmake/ParallelTPLs.cmake)
else()
    # ParallelTPLs.cmake forces these into the cache; reset them so that
    # reconfiguring an MPI build directory with USE_MPI=OFF builds MFEM serial.
    set(MFEM_USE_MPI OFF CACHE BOOL "Build MFEM with MPI" FORCE)
    set(MFEM_USE_METIS OFF CACHE BOOL "Enable METIS (multi-rank partitioning)" FORCE)
endif()

# This creates the 'mfem' target available for linking.
# EXCLUDE_FROM_ALL above keeps MFEM out of the default build set so Visual
# Studio's Output window isn't dominated by MFEM compile messages; the 'mfem'
# library is still built on demand when noumena links it.
FetchContent_MakeAvailable(mfem)
if(USE_MPI)
    noumena_wrap_build_tree_tpl_includes(mfem)
endif()

# Eigen supplies the sparse Cholesky (SimplicialLDLT) behind the "direct"
# linear solver option. Eigen is header-only, so SOURCE_SUBDIR points at a
# directory that does not exist: that skips Eigen's own (heavy) CMake project
# entirely and just downloads the headers, which we add as an include dir.
# Only the MPL2-licensed subset is used; EIGEN_MPL2_ONLY makes any accidental
# use of an LGPL-licensed Eigen module a compile error, keeping this project
# safely MIT-licensed.
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0  # Pin to a stable version for reproducibility
  GIT_PROGRESS   TRUE   # See MFEM block above for rationale.
  SOURCE_SUBDIR  do-not-configure-eigen
)
FetchContent_MakeAvailable(eigen)

add_library(eigen_headers INTERFACE)
target_include_directories(eigen_headers SYSTEM INTERFACE ${eigen_SOURCE_DIR})
target_compile_definitions(eigen_headers INTERFACE EIGEN_MPL2_ONLY)

# AMGCL supplies the algebraic multigrid preconditioner behind the "iterative"
# linear solver of the static solvers. Header-only and MIT-licensed, and
# threaded with OpenMP (when enabled below) rather than MPI, so it adds no
# build requirement. As with Eigen, its own CMake project is skipped and only
# the headers are used. AMGCL_NO_BOOST drops its optional Boost.PropertyTree
# runtime-configuration interface; parameters are set in code instead.
FetchContent_Declare(
  amgcl
  GIT_REPOSITORY https://github.com/ddemidov/amgcl.git
  GIT_TAG        1.5.0  # Pin to a stable version for reproducibility
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE   # See MFEM block above for rationale.
  SOURCE_SUBDIR  do-not-configure-amgcl
)
FetchContent_MakeAvailable(amgcl)

add_library(amgcl_headers INTERFACE)
target_include_directories(amgcl_headers SYSTEM INTERFACE ${amgcl_SOURCE_DIR})
target_compile_definitions(amgcl_headers INTERFACE AMGCL_NO_BOOST)

# nlohmann/json supplies the configuration parser. Fetched rather than vendored
# so its version is pinned in one visible place, consistent with MFEM and Eigen.
# JSON_DIAGNOSTICS/JSON_DIAGNOSTIC_POSITIONS include the JSON pointer path and
# byte positions in exception messages so config errors (e.g. type_error.302)
# point at the offending field/line. They are set here as INTERFACE definitions
# on the target: the ABI tag depends on them, so carrying them with the target
# keeps every linked TU consistent automatically.
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.12.0  # Matches the previously vendored src/json.hpp
  GIT_PROGRESS   TRUE     # See MFEM block above for rationale.
)
FetchContent_MakeAvailable(nlohmann_json)
target_compile_definitions(nlohmann_json
    INTERFACE JSON_DIAGNOSTICS=1 JSON_DIAGNOSTIC_POSITIONS=1)

# HighFive wraps the HDF5 C API; no HDF5 C++ bindings are needed. Scope the
# generic HDF5 options so disabling its tests and shared libraries does not
# change BUILD_TESTING or BUILD_SHARED_LIBS for the rest of this project.
function(configure_hdf5)
    set(BUILD_TESTING OFF)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_STATIC_LIBS ON)
    set(HDF5_BUILD_TOOLS OFF)
    set(HDF5_BUILD_EXAMPLES OFF)
    set(HDF5_BUILD_HL_LIB OFF)
    set(HDF5_BUILD_CPP_LIB OFF)
    set(HDF5_BUILD_FORTRAN OFF)
    set(HDF5_BUILD_JAVA OFF)
    set(HDF5_ENABLE_PARALLEL OFF)
    set(HDF5_ENABLE_Z_LIB_SUPPORT OFF)
    set(HDF5_ENABLE_SZIP_SUPPORT OFF)
    set(HDF5_NO_PACKAGES ON)
    FetchContent_Declare(
        hdf5
        GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
        GIT_TAG        hdf5_1.14.6
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(hdf5)
endfunction()
configure_hdf5()

# Use the fetched HDF5 target rather than searching for a system installation.
set(HIGHFIVE_FIND_HDF5 OFF CACHE BOOL "Use the fetched HDF5 library" FORCE)
set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "Build HighFive tests" FORCE)
set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "Build HighFive examples" FORCE)
set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "Build HighFive documentation" FORCE)
FetchContent_Declare(
    highfive
    GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
    GIT_TAG        v3.1.1
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(highfive)

if(USE_STRUMPACK)
    include(${CMAKE_SOURCE_DIR}/cmake/Strumpack.cmake)
endif()

# The solvers, configuration and writers, compiled once and linked by both
# the executable and the tests, which then compile only their own code.
add_library(noumena_core STATIC
    src/axisym/axisymmetric_curl_curl_integrator.cpp
    src/axisym/axisymmetric_diffusion_integrator.cpp
    src/axisym/radial_quadrature.cpp
    src/coefficients/conductor_path.cpp
    src/config/boundary_validation.cpp
    src/config/config_validator.cpp
    src/config/input_parser.cpp
    src/io/gmsh_results_writer.cpp
    src/io/hdf5_results_writer.cpp
    src/io/probe_sampler.cpp
    src/io/solver_field_writer.cpp
    src/io/status_reporter.cpp
    src/linalg/complex_direct_solver.cpp
    src/solvers/divergence_free_projector.cpp
    src/solvers/electrostatic_solver.cpp
    src/solvers/magnetic_solver.cpp
    src/solvers/magnetoquasistatic_solver.cpp
    src/solvers/magnetoquasistatic_solver_3d.cpp
    src/solvers/magnetostatic_solver.cpp
    src/solvers/magnetostatic_solver_3d.cpp
    src/solvers/mqs_massive_port_operator.cpp
    src/solvers/physics_solver.cpp
    src/solvers/vector_potential_solver_3d.cpp
)
target_link_libraries(noumena_core PUBLIC mfem eigen_headers amgcl_headers nlohmann_json::nlohmann_json HighFive::HighFive hdf5-static)
if(USE_MPI)
    target_link_libraries(noumena_core PUBLIC MPI::MPI_CXX)
endif()
if(USE_OPENMP)
    target_link_libraries(noumena_core PUBLIC OpenMP::OpenMP_CXX)
endif()
# STRUMPACK, when the build has it. NOUMENA_STRUMPACK is public: it changes
# class layouts in the headers, which every user must see alike.
if(NOUMENA_HAVE_STRUMPACK)
    target_link_libraries(noumena_core PUBLIC STRUMPACK::strumpack)
    target_compile_definitions(noumena_core PUBLIC NOUMENA_STRUMPACK)
endif()
# Project headers are included by bare name ("physics_solver.hpp"), so that
# neither src/ nor test/ depends on the relative path between them.
target_include_directories(noumena_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_executable(noumena src/main.cpp)
target_link_libraries(noumena PRIVATE noumena_core)

# Heavily templated MFEM and nlohmann/json code can exceed MSVC's default COFF
# section limit (65279) in one translation unit, which fails with C1128.
# /bigobj raises it; it has no runtime cost.
if(MSVC)
    target_compile_options(noumena_core PRIVATE /bigobj)
    target_compile_options(noumena PRIVATE /bigobj)
endif()

# Build identity banner: regenerate build_info.hpp on every build so a stale
# executable is obvious at runtime. GenerateBuildInfo.cmake uses configure_file,
# which only rewrites the header when the commit/dirty state actually changes,
# so this does not trigger a rebuild on every invocation.
add_custom_target(build_info ALL
    COMMAND ${CMAKE_COMMAND}
        -DPROJECT_VERSION=${PROJECT_VERSION}
        -DGIT_DIR=${CMAKE_SOURCE_DIR}
        -DSRC=${CMAKE_SOURCE_DIR}/src/io/build_info.hpp.in
        -DDST=${CMAKE_BINARY_DIR}/generated/build_info.hpp
        -P ${CMAKE_SOURCE_DIR}/cmake/GenerateBuildInfo.cmake
    BYPRODUCTS ${CMAKE_BINARY_DIR}/generated/build_info.hpp
    COMMENT "Generating build info header"
    VERBATIM
)
add_dependencies(noumena build_info)
target_include_directories(noumena PRIVATE ${CMAKE_BINARY_DIR}/generated)

# OpenMP (linked through noumena_core; USE_OPENMP is switched off above if
# the compiler lacks it)
if(USE_OPENMP)
    message(STATUS "OpenMP enabled. Use OMP_NUM_THREADS to control parallelism.")
endif()

# Documentation with Doxygen
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    message(STATUS "Doxygen found. Use 'make docs' to generate documentation.")
else()
    message(STATUS "Doxygen not found. Documentation generation will be skipped.")
endif()

# Testing
if(BUILD_TESTING)
    enable_testing()

    # Fetch Catch2
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2  # Pin to stable version
        GIT_PROGRESS   TRUE    # See MFEM block above for rationale.
    )
    FetchContent_MakeAvailable(Catch2)

    # Create test executable
    add_executable(mfem_tests
        test/test_main.cpp
        test/test_axisymmetric_axis.cpp
        test/test_axisymmetric_quadrature.cpp
        test/test_closed_forms.cpp
        test/test_coaxial_rings.cpp
        test/test_config_validator.cpp
        test/test_gmsh_interpolation.cpp
        test/test_gmsh_msh_format.cpp
        test/test_hdf5.cpp
        test/test_input_parser.cpp
        test/test_electrostatic_3d.cpp
        test/test_magnetostatic_3d.cpp
        test/test_magnetoquasistatic_3d.cpp
        test/test_mqs_port_operator.cpp
        test/test_probes.cpp
        test/test_solvers.cpp
        test/test_status_reporter.cpp
        test/test_team7.cpp
        test/test_team15.cpp
        test/test_team21a.cpp
        test/test_hypre.cpp
    )

    # The test runner supplies its own main (test_main.cpp) so an MPI build
    # can initialize MPI and HYPRE around the Catch2 session.
    target_link_libraries(mfem_tests PRIVATE noumena_core Catch2::Catch2)
    # Committed test inputs (meshes and the Gmsh scripts that generate them).
    target_compile_definitions(mfem_tests PRIVATE
        NOUMENA_TEST_DATA="${CMAKE_SOURCE_DIR}/test/data"
        NOUMENA_EXAMPLES="${CMAKE_SOURCE_DIR}/examples")
    # Gmsh, for the on-demand benchmarks whose meshes are generated rather
    # than committed (they skip without it).
    find_program(GMSH_EXECUTABLE gmsh)
    if(GMSH_EXECUTABLE)
        target_compile_definitions(mfem_tests PRIVATE NOUMENA_GMSH="${GMSH_EXECUTABLE}")
    endif()

    # Same COFF section limit as noumena.
    if(MSVC)
        target_compile_options(mfem_tests PRIVATE /bigobj)
    endif()

    # Register tests with CTest
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
    catch_discover_tests(mfem_tests)
endif()

