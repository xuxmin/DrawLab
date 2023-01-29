find_package( OpenGL REQUIRED )

include(CheckCXXCompilerFlag)

# Build GLFW
if( UNIX )
  # On Unix-like systems, shared libraries can use the soname system.
  set(GLFW_LIB_NAME glfw)
else()
  set(GLFW_LIB_NAME glfw3)
endif()

# Filter out warnings that cause problems with GLFW.
if( WARNINGS_AS_ERRORS AND (USING_GNU_CXX OR USING_CLANG_CXX))
  string( REPLACE "-Wdeclaration-after-statement" "" filtered_c_flags ${CMAKE_C_FLAGS} )
  string( REPLACE "-Wsign-compare" "" filtered_c_flags ${filtered_c_flags} )
  push_variable( CMAKE_C_FLAGS "${filtered_c_flags} -Wno-format-truncation -Wno-deprecated" )
endif()
add_subdirectory( glfw )
set_property( TARGET glfw PROPERTY C_STANDARD 99 )
if( WARNINGS_AS_ERRORS AND (USING_GNU_CXX OR USING_CLANG_CXX))
  pop_variable( CMAKE_C_FLAGS )
endif()

# Build imgui
add_subdirectory( imgui )

# Build glad
add_subdirectory( glad )

# Build OpenER
set(ILMBASE_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
set(OPENEXR_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
set(ILMBASE_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
set(OPENEXR_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
add_subdirectory(openexr)
set_property(TARGET IexMath eLut toFloat b44ExpLogTable dwaLookups IlmThread Half Iex Imath IlmImf PROPERTY FOLDER "dependencies")

# Build stb_image
add_subdirectory(stb_image)

# Build the pugixml parser
add_library(pugixml STATIC pugixml/src/pugixml.cpp)
set_property(TARGET pugixml PROPERTY
   LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pugixml")
set_property(TARGET pugixml PROPERTY FOLDER "dependencies")

# Build  tiny_obj_loader
add_subdirectory(tiny_obj_loader)

# Build Thread Building Blocks (main shared libraries only)
set(TBB_BUILD_SHARED          OFF  CACHE BOOL " " FORCE)
set(TBB_BUILD_STATIC          ON   CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC       OFF  CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF  CACHE BOOL " " FORCE)
set(TBB_BUILD_TESTS           OFF  CACHE BOOL " " FORCE)
add_subdirectory(tbb)
set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")

if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGL_SILENCE_DEPRECATION=1")
endif()


set(FILESYSTEM_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/filesystem)
set(TFM_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/tinyformat)
set(CMDLINE_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/cmdline)
set(SPDLOG_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/spdlog/include)
set(OPENEXR_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Imath
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Iex
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Half
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/OpenEXR/IlmImf
  ${CMAKE_CURRENT_BINARY_DIR}/openexr/OpenEXR/config
  ${CMAKE_CURRENT_BINARY_DIR}/openexr/IlmBase/config)
set(STB_IMAGE_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/stb_image)
set(GLAD_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/glad)
set(GLFW_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include)
set(IMGUI_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
set(PUGIXML_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/pugixml/src)
set(PCG32_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/pcg32)
set(TINY_OBJ_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/tiny_obj_loader)
set(TBB_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/tbb/include)

set(CompilerFlags
      FILESYSTEM_INCLUDE_DIR TFM_INCLUDE_DIR OPENEXR_INCLUDE_DIRS
      STB_IMAGE_INCLUDE_DIR GLAD_INCLUDE_DIRS GLFW_INCLUDE_DIRS
      IMGUI_INCLUDE_DIRS PUGIXML_INCLUDE_DIR PCG32_INCLUDE_DIR
      TINY_OBJ_INCLUDE_DIR TBB_INCLUDE_DIR CMDLINE_INCLUDE_DIR
      SPDLOG_INCLUDE_DIR
)

foreach(CompilerFlag ${CompilerFlags})
  set(${CompilerFlag} "${${CompilerFlag}}" PARENT_SCOPE)
endforeach()