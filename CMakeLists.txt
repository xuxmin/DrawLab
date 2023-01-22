project(drawlab)

cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake"
  ${CMAKE_MODULE_PATH}
)

set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../" CACHE PATH "Path to OptiX installed location.")

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target which defaults to Debug when you first start it.
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
if (WIN32)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

add_subdirectory(ext)
add_subdirectory(src/core)
add_subdirectory(src/opengl)

set(sources

    src/editor/gui.cpp

    src/opengl/texture.cpp
    src/opengl/display.cpp

    src/tracer/scene.cpp
    src/tracer/mesh/mesh.cpp
    src/tracer/mesh/obj.cpp
    src/tracer/accel/accel.cpp
    src/tracer/accel/octree.cpp
    src/tracer/emitter/area.cpp
    src/tracer/bsdf/diffuse.cpp
    src/tracer/bsdf/dielectric.cpp
    src/tracer/bsdf/microfacet.cpp
    src/tracer/bsdf/mirror.cpp
    src/tracer/integrator/normal.cpp
    src/tracer/integrator/path.cpp
    src/tracer/integrator/whitted.cpp
    src/tracer/sampler/independent.cpp
    src/tracer/camera/perspective.cpp
    src/tracer/rfilter/rfilter.cpp
    src/tracer/main.cpp
)

add_executable(drawlab ${sources})

target_link_libraries(drawlab PUBLIC ${GLFW_LIB_NAME} imgui corelib gllib tiny_obj_loader)

target_include_directories(drawlab
    PUBLIC
    ext
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # glfw
    ${GLFW_INCLUDE_DIRS}
    # pcg32
    ${PCG32_INCLUDE_DIR}
    # tiny obj loader
    ${TINY_OBJ_INCLUDE_DIR}
)

target_compile_features(drawlab PRIVATE cxx_std_17)

# Test
# add_library(drawlab_lib ${sources})
# target_compile_features(drawlab_lib PRIVATE cxx_std_17)
# enable_testing()
# add_subdirectory(tests)