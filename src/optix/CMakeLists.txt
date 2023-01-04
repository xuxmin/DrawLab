# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake"
  ${CMAKE_MODULE_PATH}
)

include(configure_optix)

OPTIX_add_sample_executable(helloOptix
  main.cpp
  optix_renderer.cpp
  sutil.cpp
  cuda/device_programs.cu
)

target_include_directories(helloOptix
  PRIVATE
  # stb image
  ${STB_IMAGE_INCLUDE_DIR}
)

target_link_libraries(helloOptix
  stb_image
)