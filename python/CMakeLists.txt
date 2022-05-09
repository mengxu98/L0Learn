cmake_minimum_required(VERSION 3.12...3.18)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(l0learn_core VERSION 0.0.2)

# Add Armadillo and pybind11
if($ENV{CIBUILDWHEEL})
    set(ALLOW_OPENBLAS_MACOS ON CACHE BOOL "Allow detection of OpenBLAS on macOS" FORCE)
endif()

ADD_SUBDIRECTORY(external/pybind11)
ADD_SUBDIRECTORY(external/carma)

pybind11_add_module(l0learn_core MODULE
        src/l0learn/pyl0learn.cpp
        src/l0learn/src/CDL012LogisticSwaps.cpp
        src/l0learn/src/CDL012SquaredHingeSwaps.cpp
        src/l0learn/src/CDL012Swaps.cpp
        src/l0learn/src/Grid.cpp
        src/l0learn/src/Grid1D.cpp
        src/l0learn/src/Grid2D.cpp
        src/l0learn/src/Normalize.cpp)

# Add -fPIC for Armadillo (and OpenBLAS if compiled)
# clang on Windows does not support -fPIC
if(NOT MSVC AND NOT WIN32)
    target_compile_options(l0learn_core PRIVATE -fPIC)
endif()

target_include_directories(l0learn_core PRIVATE external/carma/)
target_include_directories(l0learn_core PRIVATE external/pybind11/)
target_include_directories(l0learn_core PRIVATE src/l0learn/)
target_include_directories(l0learn_core PRIVATE src/l0learn/src/include/)


if(NOT $ENV{CIBUILDWHEEL})
    FIND_LIBRARY(LAPACK)
    FIND_LIBRARY(BLAS)
    TARGET_LINK_LIBRARIES(l0learn_core PRIVATE carma::carma BLAS::BLAS LAPACK::LAPACK)
else()
    TARGET_LINK_LIBRARIES(l0learn_core PRIVATE carma::carma)
endif()

target_compile_definitions(l0learn_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

target_compile_options(l0learn_core PRIVATE -std=c++14)
add_link_options(-Wl,--no-undefined)
set_target_properties(l0learn_core PROPERTIES OUTPUT_NAME l0learn/l0learn_core)

if($ENV{CIBUILDWHEEL})
    message(STATUS "Building on a CI server. Precompiled libraries used.")
    # Set a flag stating that a precompiled version of OpenBLAS is used
    target_compile_definitions(l0learn_core PRIVATE -DPYARMA_PRECOMPILED_OPENBLAS)
    if(UNIX AND NOT APPLE)
        # Set a flag stating that a precompiled version of HDF5 is used on Linux
        target_compile_definitions(l0learn_core PRIVATE -DPYARMA_PRECOMPILED_HDF5)
    endif()
    # Force use of LAPACK, as OpenBLAS is always used anyway
    target_compile_definitions(l0learn_core PRIVATE -DARMA_USE_LAPACK)
endif()


if(NOT MSVC)
    # add_definitions("-O2")
    target_compile_options(l0learn_core PRIVATE "-O2")
endif()
