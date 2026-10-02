# Copyright (c) 2026 T. C. Raymond
# SPDX-License-Identifier: MIT
#
# Third-party libraries for the MPI build (USE_MPI=ON): HYPRE, and optionally
# METIS.
#
# MFEM is itself fetched and configured in the parent CMakeLists, and its
# configure step calls find_package(HYPRE). HYPRE therefore has to be
# INSTALLED before MFEM is configured, not merely declared as a target, so it
# is fetched, built and installed here at configure time, into the build tree.
# A stamp file keyed on the pinned version makes re-configuring free.
#
# To use an existing installation instead (a system package, vcpkg on
# Windows, a site build), set HYPRE_DIR to its prefix (containing include/ and
# lib/); nothing is then built here.
#
# METIS is only needed to PARTITION a mesh across several MPI ranks. The
# solvers currently run on one rank, so METIS is not built by default; set
# METIS_DIR to an installed METIS 5 to enable it. (Building METIS 5 from
# source portably is awkward: its CMake needs a generated include directory,
# does not install under MSVC, and needs GKlib linked alongside it.)

set(ELECTROMAG_HYPRE_VERSION "v3.0.0" CACHE STRING "HYPRE release built when HYPRE_DIR is not set")

find_package(MPI REQUIRED COMPONENTS C CXX)

# HYPRE is threaded with OpenMP along with the rest of the build (USE_OPENMP).
if(USE_OPENMP)
    set(_hypre_openmp ON)
else()
    set(_hypre_openmp OFF)
endif()

# The bundled build is re-checked on every configure (HYPRE_DIR is cached
# pointing at it after the first), so changing its version or USE_OPENMP
# rebuilds it; a HYPRE_DIR set to anything else is used as given.
set(_hypre_bundled "${CMAKE_BINARY_DIR}/tpl/hypre")
if(NOT HYPRE_DIR OR HYPRE_DIR STREQUAL _hypre_bundled)
    set(_hypre_prefix "${_hypre_bundled}")
    set(_hypre_stamp "${_hypre_prefix}/.built-${ELECTROMAG_HYPRE_VERSION}-openmp-${_hypre_openmp}")
    if(NOT EXISTS "${_hypre_stamp}")
        message(STATUS "Building HYPRE ${ELECTROMAG_HYPRE_VERSION} (one-time, at configure)...")
        FetchContent_Declare(
            hypre
            GIT_REPOSITORY https://github.com/hypre-space/hypre.git
            GIT_TAG        ${ELECTROMAG_HYPRE_VERSION}
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
            SOURCE_SUBDIR  do-not-configure-hypre  # built below, not added
        )
        FetchContent_MakeAvailable(hypre)

        set(_hypre_build "${CMAKE_BINARY_DIR}/tpl/hypre-build")
        # A multi-config generator (Visual Studio) needs the configuration
        # at build time; single-config ones take it at configure time.
        set(_hypre_config Release)
        execute_process(
            COMMAND ${CMAKE_COMMAND}
                -S "${hypre_SOURCE_DIR}/src" -B "${_hypre_build}"
                -G "${CMAKE_GENERATOR}"
                -DCMAKE_BUILD_TYPE=${_hypre_config}
                -DCMAKE_INSTALL_PREFIX=${_hypre_prefix}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                -DBUILD_SHARED_LIBS=OFF
                -DHYPRE_ENABLE_MPI=ON
                -DHYPRE_ENABLE_OPENMP=${_hypre_openmp}
                -DHYPRE_ENABLE_FORTRAN=OFF
                -DHYPRE_BUILD_TESTS=OFF
                -DHYPRE_BUILD_EXAMPLES=OFF
            RESULT_VARIABLE _hypre_result)
        if(NOT _hypre_result EQUAL 0)
            message(FATAL_ERROR "Configuring HYPRE failed (see output above).")
        endif()
        include(ProcessorCount)
        ProcessorCount(_ncpu)
        if(_ncpu EQUAL 0)
            set(_ncpu 1)
        endif()
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build "${_hypre_build}" --config ${_hypre_config}
                    --target install --parallel ${_ncpu}
            RESULT_VARIABLE _hypre_result)
        if(NOT _hypre_result EQUAL 0)
            message(FATAL_ERROR "Building HYPRE failed (see output above).")
        endif()
        file(WRITE "${_hypre_stamp}" "${ELECTROMAG_HYPRE_VERSION}\n")
    endif()
    set(HYPRE_DIR "${_hypre_prefix}" CACHE PATH "HYPRE installation prefix" FORCE)
endif()
message(STATUS "HYPRE: ${HYPRE_DIR}")

# MFEM caches a default METIS_DIR of its own (a sibling "metis-4.0" that
# normally does not exist), so only a directory that exists counts as the
# user's choice.
if(METIS_DIR AND EXISTS "${METIS_DIR}")
    set(MFEM_USE_METIS ON CACHE BOOL "Enable METIS (multi-rank partitioning)" FORCE)
    set(MFEM_USE_METIS_5 ON CACHE BOOL "METIS 5 API" FORCE)
    message(STATUS "METIS: ${METIS_DIR}")
else()
    set(MFEM_USE_METIS OFF CACHE BOOL "Enable METIS (multi-rank partitioning)" FORCE)
    message(STATUS "METIS: not used (single-rank runs only; set METIS_DIR to enable)")
endif()

set(MFEM_USE_MPI ON CACHE BOOL "Build MFEM with MPI" FORCE)

# MFEM exports its 'mfem' target for installation unconditionally, and CMake
# refuses to export an include directory inside the build tree -- which is
# where the HYPRE built above lives. This project never installs MFEM, so
# after MFEM has been added the build-tree TPL include paths are wrapped in
# $<BUILD_INTERFACE:...>: they still apply to everything built here, and the
# install export no longer sees them.
function(electromag_wrap_build_tree_tpl_includes target)
    get_target_property(_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _dirs)
        return()
    endif()
    set(_wrapped "")
    foreach(_dir IN LISTS _dirs)
        string(FIND "${_dir}" "${CMAKE_BINARY_DIR}/tpl/" _pos)
        if(_pos EQUAL 0)
            list(APPEND _wrapped "$<BUILD_INTERFACE:${_dir}>")
        else()
            list(APPEND _wrapped "${_dir}")
        endif()
    endforeach()
    set_target_properties(${target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_wrapped}")
endfunction()
