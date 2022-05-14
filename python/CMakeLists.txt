cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(Python3_EXECUTABLE ${PYTHON_EXECUTABLE}) # Required for carma!
SET(Python3_INCLUDE_DIR ${PYTHON_INCLUDE_DIR}) # Required for carma!
SET(Python3_LIBRARY ${PYTHON_LIBRARY}) # Required for carma!

project(l0learn_core VERSION 0.0.2)

ADD_SUBDIRECTORY(external/pybind11)
ADD_SUBDIRECTORY(external/carma)

pybind11_add_module(l0learn_core MODULE
        src/l0learn/pyl0learn.cpp
        src/l0learn/src/Normalize.cpp)

target_include_directories(l0learn_core PRIVATE external/carma/)
target_include_directories(l0learn_core PRIVATE external/pybind11/)
target_include_directories(l0learn_core PRIVATE src/l0learn/)
target_include_directories(l0learn_core PRIVATE src/l0learn/src/include/)

TARGET_LINK_LIBRARIES(l0learn_core PRIVATE carma::carma)

target_compile_definitions(l0learn_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

target_compile_options(l0learn_core PRIVATE -std=c++14)
add_link_options(-Wl,--no-undefined)
set_target_properties(l0learn_core PROPERTIES OUTPUT_NAME l0learn/l0learn_core)
