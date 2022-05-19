cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF (NOT DEFINED L0LEARN_VERSION_INFO)
    SET(L0LEARN_VERSION_INFO "0.0.0")
ENDIF ()

INCLUDE(GNUInstallDirs)

PROJECT(
        l0learn
        VERSION ${L0LEARN_VERSION_INFO}
        DESCRIPTION "Python bindings for L0Learn CD and CDPSI optimization."
        LANGUAGES CXX
)

# Force static libraries to be built
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared library" FORCE)

ADD_SUBDIRECTORY(external/pybind11)
ADD_SUBDIRECTORY(external/armadillo-code)
ADD_SUBDIRECTORY(external/carma)

pybind11_add_module(_l0learn MODULE
        src/l0learn/pyl0learn.cpp
        src/l0learn/src/Normalize.cpp)

# Add -fPIC for Armadillo (and OpenBLAS if compiled)
if(NOT MSVC)
    # clang on Windows does not support -fPIC
    if(NOT WIN32)
        target_compile_options(armadillo PRIVATE -fPIC)
    endif()
endif()


if($ENV{CIBUILDWHEEL})
    target_compile_definitions(armadillo PRIVATE ARMA_USE_LAPACK)
    target_compile_definitions(armadillo PRIVATE ARMA_USE_BLAS)
endif()

TARGET_LINK_LIBRARIES(_l0learn PRIVATE armadillo-code)
TARGET_LINK_LIBRARIES(_l0learn PRIVATE carma::carma)

target_include_directories(_l0learn PRIVATE external/pybind11/)
target_include_directories(_l0learn PRIVATE external/armadillo-code/)
target_include_directories(_l0learn PRIVATE external/carma/)
target_include_directories(_l0learn PRIVATE src/l0learn/)
target_include_directories(_l0learn PRIVATE src/l0learn/src/include/)

target_compile_options(_l0learn PRIVATE -std=c++14)
add_link_options(-Wl,--no-undefined)
# set_target_properties(l0learn_core PROPERTIES OUTPUT_NAME l0learn/l0learn_core)

if(SKBUILD)
    INSTALL(TARGETS _l0learn LIBRARY DESTINATION "${PROJECT_NAME}/lib")
ELSE ()
    INSTALL(
            TARGETS _l0learn
            LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/l0learn/lib"
    )
ENDIF ()
