# Copyright (c) 2026 T. C. Raymond
# SPDX-License-Identifier: MIT
#
# STRUMPACK (USE_STRUMPACK), the sparse direct solver of the time-harmonic
# (MQS) systems: it factors the complex system itself, 10-60 times faster and
# 5-10 times smaller than Eigen's LU of the packed real form, which the
# "direct" MQS solve falls back to without it. See
# src/linalg/complex_direct_solver.hpp.
#
# STRUMPACK needs a Fortran compiler (gfortran, LLVM Flang or Intel ifx),
# BLAS/LAPACK and METIS 5 (set METIS_DIR for one outside the default search
# paths). If any is missing it is left out, with a warning.
#
# Its CMake project cannot be added as a subproject (it resolves paths against
# CMAKE_SOURCE_DIR), so, like HYPRE in the MPI build, it is fetched, built and
# installed into the build tree at configure time, once per version, and
# imported as STRUMPACK::strumpack. To use an existing installation instead,
# set STRUMPACK_ROOT to its prefix.
#
# Sets ELECTROMAG_HAVE_STRUMPACK.

set(ELECTROMAG_STRUMPACK_VERSION "v8.0.0" CACHE STRING "STRUMPACK release built when STRUMPACK_ROOT is not set")
set(ELECTROMAG_HAVE_STRUMPACK OFF)

include(CheckLanguage)
check_language(Fortran)
if(NOT CMAKE_Fortran_COMPILER)
    message(WARNING "USE_STRUMPACK: no Fortran compiler found (gfortran, LLVM Flang or "
        "Intel ifx), so STRUMPACK is not used and the direct MQS solve falls back to "
        "Eigen's much slower SparseLU.")
    return()
endif()
enable_language(Fortran)

find_package(LAPACK QUIET)  # finds BLAS too
if(METIS_DIR AND EXISTS "${METIS_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${METIS_DIR}")
endif()
find_path(ELECTROMAG_METIS_INCLUDE_DIR metis.h)
find_library(ELECTROMAG_METIS_LIBRARY metis)
if(NOT LAPACK_FOUND OR NOT ELECTROMAG_METIS_INCLUDE_DIR OR NOT ELECTROMAG_METIS_LIBRARY)
    message(WARNING "USE_STRUMPACK: STRUMPACK needs BLAS/LAPACK and METIS 5 (set METIS_DIR "
        "to its prefix), and they were not all found, so it is not used and the direct MQS "
        "solve falls back to Eigen's much slower SparseLU.")
    return()
endif()

if(NOT STRUMPACK_ROOT)
    if(USE_OPENMP)
        set(_strumpack_openmp ON)
    else()
        set(_strumpack_openmp OFF)
    endif()
    set(_strumpack_prefix "${CMAKE_BINARY_DIR}/tpl/strumpack")
    set(_strumpack_stamp "${_strumpack_prefix}/.built-${ELECTROMAG_STRUMPACK_VERSION}-openmp-${_strumpack_openmp}")
    if(NOT EXISTS "${_strumpack_stamp}")
        message(STATUS "Building STRUMPACK ${ELECTROMAG_STRUMPACK_VERSION} (one-time, at configure)...")
        FetchContent_Declare(
            strumpack
            GIT_REPOSITORY https://github.com/pghysels/STRUMPACK.git
            GIT_TAG        ${ELECTROMAG_STRUMPACK_VERSION}
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
            SOURCE_SUBDIR  do-not-configure-strumpack  # built below, not added
        )
        FetchContent_MakeAvailable(strumpack)

        # Shared memory only (one rank calls it), METIS ordering, and none of
        # the optional compression or distributed libraries.
        set(_strumpack_build "${CMAKE_BINARY_DIR}/tpl/strumpack-build")
        set(_strumpack_config Release)
        execute_process(
            COMMAND ${CMAKE_COMMAND}
                -S "${strumpack_SOURCE_DIR}" -B "${_strumpack_build}"
                -G "${CMAKE_GENERATOR}"
                -DCMAKE_BUILD_TYPE=${_strumpack_config}
                -DCMAKE_INSTALL_PREFIX=${_strumpack_prefix}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                -DBUILD_SHARED_LIBS=OFF
                -DBUILD_TESTING=OFF
                -DSTRUMPACK_USE_MPI=OFF
                -DSTRUMPACK_USE_OPENMP=${_strumpack_openmp}
                -DTPL_ENABLE_SLATE=OFF
                -DTPL_ENABLE_PARMETIS=OFF
                -DTPL_ENABLE_SCOTCH=OFF
                -DTPL_ENABLE_PTSCOTCH=OFF
                -DTPL_ENABLE_BPACK=OFF
                -DTPL_ENABLE_ZFP=OFF
                "-DTPL_BLAS_LIBRARIES=${BLAS_LIBRARIES}"
                "-DTPL_LAPACK_LIBRARIES=${LAPACK_LIBRARIES}"
                "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
            RESULT_VARIABLE _strumpack_result)
        if(NOT _strumpack_result EQUAL 0)
            message(FATAL_ERROR "Configuring STRUMPACK failed (see output above). "
                "Configure with -DUSE_STRUMPACK=OFF to build without it.")
        endif()
        include(ProcessorCount)
        ProcessorCount(_ncpu)
        if(_ncpu EQUAL 0)
            set(_ncpu 1)
        endif()
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build "${_strumpack_build}" --config ${_strumpack_config}
                    --target install --parallel ${_ncpu}
            RESULT_VARIABLE _strumpack_result)
        if(NOT _strumpack_result EQUAL 0)
            message(FATAL_ERROR "Building STRUMPACK failed (see output above). "
                "Configure with -DUSE_STRUMPACK=OFF to build without it.")
        endif()
        file(WRITE "${_strumpack_stamp}" "${ELECTROMAG_STRUMPACK_VERSION}\n")
    endif()
    set(STRUMPACK_ROOT "${_strumpack_prefix}")
endif()

find_package(STRUMPACK CONFIG REQUIRED)
message(STATUS "STRUMPACK: ${STRUMPACK_DIR}")
set(ELECTROMAG_HAVE_STRUMPACK ON)
