cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF (NOT DEFINED L0LEARN_VERSION_INFO)
    SET(L0LEARN_VERSION_INFO "0.0.0")
ENDIF ()

PROJECT(
        l0learn
        VERSION ${L0LEARN_VERSION_INFO}
        DESCRIPTION "Python bindings for L0Learn CD and CDPSI optimization."
        LANGUAGES CXX
)

# Now we can find pybind11
FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy)
ADD_SUBDIRECTORY(external/pybind11)
ADD_SUBDIRECTORY(external/carma)

pybind11_add_module(l0learn MODULE
        l0learn/pyl0learn.cpp
        l0learn/src/Normalize.cpp)

TARGET_LINK_LIBRARIES(l0learn PRIVATE carma::carma)

target_include_directories(l0learn PRIVATE
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/external/pybind11/"
        "${PROJECT_SOURCE_DIR}/external/carma/"
        "${PROJECT_SOURCE_DIR}/l0learn/"
        "${PROJECT_SOURCE_DIR}/l0learn/src/include/")

target_compile_options(l0learn PRIVATE -std=c++14)
add_link_options(-Wl,--no-undefined)
# set_target_properties(l0learn_core PROPERTIES OUTPUT_NAME l0learn/l0learn_core)

#INCLUDE(GNUInstallDirs REQUIRED)
#
#if(SKBUILD)
#    INSTALL(TARGETS l0learn LIBRARY DESTINATION "${PROJECT_NAME}/lib")
#ELSE ()
#    INSTALL(
#            TARGETS l0learn
#            LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/l0learn/lib"
#    )
#ENDIF ()

install(TARGETS l0learn DESTINATION .)