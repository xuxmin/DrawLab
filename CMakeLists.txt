cmake_minimum_required (VERSION 3.2)
project(drawlab)

# place binaries and libraries according to GNU standards
# In windows, place binaries and libraries into specific directories, the zlib1.all
# will be placed with drawlab.exe.
# In Linux, skip this.
if (WIN32) 
    include(GNUInstallDirs)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
endif(WIN32)


add_subdirectory(ext)

set(sources
    src/main.cpp
    src/core/base/common.cpp
    src/core/base/string.cpp
    src/core/math/math.cpp
    src/core/math/matrix.cpp
    src/core/math/wrap.cpp
    src/core/bitmap/bitmap.cpp
    src/core/parser/proplist.cpp
    src/core/parser/parser.cpp
    src/core/parser/object.cpp

    src/editor/gui.cpp
    src/editor/block.cpp

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
)

add_executable(drawlab ${sources})

target_link_libraries(drawlab PUBLIC ${GLFW_LIB_NAME} imgui glad IlmImf stb_image pugixml tiny_obj_loader tbb_static)

target_include_directories(drawlab
    PUBLIC
    ext
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # Portable filesystem API
    ${FILESYSTEM_INCLUDE_DIR}
    # tinyformat string formatting library
    ${TFM_INCLUDE_DIR}
    # OpenEXR high dynamic range bitmap library
    ${OPENEXR_INCLUDE_DIRS}
    # stb image
    ${STB_IMAGE_INCLUDE_DIR}
    # glad
    ${GLAD_INCLUDE_DIRS}
    # glfw
    ${GLFW_INCLUDE_DIRS}
    # pugixml
    ${PUGIXML_INCLUDE_DIR}
    # pcg32
    ${PCG32_INCLUDE_DIR}
    # tiny obj loader
    ${TINY_OBJ_INCLUDE_DIR}
    # tbb
    ${TBB_INCLUDE_DIR}
)

target_compile_features(drawlab PRIVATE cxx_std_17)

# Test
# add_library(drawlab_lib ${sources})
# target_compile_features(drawlab_lib PRIVATE cxx_std_17)
# enable_testing()
# add_subdirectory(tests)