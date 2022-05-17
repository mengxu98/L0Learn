cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force static libraries to be built
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared library" FORCE)


IF (NOT DEFINED L0LEARN_CORE_VERSION_INFO)
    SET(L0LEARN_CORE_VERSION_INFO "0.0.0")
ENDIF ()

PROJECT(
        l0learn_core
        VERSION ${L0LEARN_CORE_VERSION_INFO}
        DESCRIPTION "Python bindings for L0Learn CD and CDPSI optimization."
        LANGUAGES CXX
)

ADD_SUBDIRECTORY(external/pybind11)
ADD_SUBDIRECTORY(external/armadillo-code)
ADD_SUBDIRECTORY(external/carma)

if($ENV{CIBUILDWHEEL})
    target_compile_definitions(armadillo PRIVATE ARMA_USE_LAPACK)
    target_compile_definitions(armadillo PRIVATE ARMA_USE_BLAS)
endif()


pybind11_add_module(l0learn_core MODULE
        src/l0learn/pyl0learn.cpp
        src/l0learn/src/Normalize.cpp)

TARGET_LINK_LIBRARIES(l0learn_core PUBLIC armadillo)
TARGET_LINK_LIBRARIES(l0learn_core PUBLIC carma)

target_include_directories(l0learn_core PRIVATE external/pybind11/)
target_include_directories(l0learn_core PRIVATE external/armadillo-code/)
target_include_directories(l0learn_core PRIVATE external/carma/)
target_include_directories(l0learn_core PRIVATE src/l0learn/)
target_include_directories(l0learn_core PRIVATE src/l0learn/src/include/)

target_compile_options(l0learn_core PRIVATE -std=c++14)
add_link_options(-Wl,--no-undefined)
# set_target_properties(l0learn_core PROPERTIES OUTPUT_NAME l0learn/l0learn_core)

if(SKBUILD)
    INSTALL(TARGETS l0learn_core LIBRARY DESTINATION "${PROJECT_NAME}/lib")
ELSE ()
    INSTALL(
            TARGETS l0learn_core
            LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/l0learn_core/lib"
    )
ENDIF ()
