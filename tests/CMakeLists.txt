include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(
  drawlab_tests
  core/math/test_vector.cpp
  core/math/test_point.cpp
  core/math/test_normal.cpp
  core/math/test_matrix.cpp
)

target_compile_features(drawlab_tests PRIVATE cxx_std_17)


target_link_libraries(drawlab_tests
  PRIVATE gtest_main drawlab_lib
)

# automatic discovery of unit tests
include(GoogleTest)
gtest_discover_tests(drawlab_tests
  PROPERTIES
    LABELS "unit"
  DISCOVERY_TIMEOUT  # how long to wait (in seconds) before crashing
    240
)