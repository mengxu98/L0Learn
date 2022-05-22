cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF (NOT DEFINED L0LEARN_VERSION_INFO)
    SET(L0LEARN_VERSION_INFO "0.0.0")
ENDIF ()

PROJECT(
        _l0learn
        VERSION ${L0LEARN_VERSION_INFO}
        DESCRIPTION "Python bindings for L0Learn CD and CDPSI optimization."
        LANGUAGES CXX
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared library" FORCE)

FIND_PACKAGE(Python3 COMPONENTS Interpreter Development.Module NumPy)
ADD_SUBDIRECTORY(external/pybind11)
ADD_SUBDIRECTORY(external/armadillo-code)
# FIND_PACKAGE(Armadillo REQUIRED) # Does not build with/without this line
FIND_PACKAGE(carma CONFIG REQUIRED)


pybind11_add_module(_l0learn MODULE
        l0learn/pyl0learn.cpp
        l0learn/src/Normalize.cpp)

TARGET_LINK_LIBRARIES(_l0learn PRIVATE carma::carma)

target_include_directories(_l0learn PRIVATE
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/external/pybind11/"
        "${PROJECT_SOURCE_DIR}/l0learn/"
        "${PROJECT_SOURCE_DIR}/l0learn/src/include/")

INSTALL(TARGETS _l0learn DESTINATION .)
