#------------------------------------------ #
# !!! CUDA 11.2 下, GCC 版本不能超过 10 !!!  #
#------------------------------------------ #


set(CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/cmake"
  ${CMAKE_MODULE_PATH}
  )

find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED)


# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)


# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if(NOT PASSED_FIRST_CONFIGURE)
  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch sm_60)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  set(flag "--use_fast_math")
  list(FIND CUDA_NVCC_FLAGS ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS ${flag})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()
  
  set(flag "-lineinfo")
  list(FIND CUDA_NVCC_FLAGS ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS ${flag})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  if (CUDA_VERSION VERSION_LESS "3.0")
    set(flag "--keep")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()

  if( APPLE )
    # Undef'ing __BLOCKS__ for OSX builds.  This is due to a name clash between OSX 10.6
    # C headers and CUDA headers
    set(flag "-U__BLOCKS__")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()

  if(CMAKE_CXX_STANDARD EQUAL 11)
    set(SAMPLES_NVRTC_CXX "-std=c++11")
  else()
    set(SAMPLES_NVRTC_CXX "")
  endif()
  set(CUDA_NVRTC_FLAGS ${SAMPLES_NVRTC_CXX} -arch compute_60 -use_fast_math -lineinfo -default-device -rdc true -D__x86_64 CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
endif(NOT PASSED_FIRST_CONFIGURE)

mark_as_advanced(CUDA_NVRTC_FLAGS)

# Select whether to use NVRTC or NVCC to generate PTX
set(CUDA_NVRTC_ENABLED ON CACHE BOOL "Use NVRTC to compile PTX at run-time instead of NVCC at build-time")

set(PROJECT_PTX_DIR "${CMAKE_BINARY_DIR}/ptx")
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")
set(CUDA_GENERATED_OUTPUT_DIR ${PROJECT_PTX_DIR})

if (WIN32)
  string(REPLACE "/" "\\\\" PROJECT_PTX_DIR  ${PROJECT_PTX_DIR})
else (WIN32)
  if ( USING_GNU_C AND NOT APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DM_PI=3.14159265358979323846" )
  endif()
endif (WIN32)

# 注意 NVRTC 编译时的包含目录需要在 nvrtcCompileProgram() 运行时指定, 
# target_include_directories 里指定并没有用. 因此设置了下面两个变量, 然后在
# sutil.cpp 中的 getPtxFromCuString 方法会进行 JIT 编译

# NVRTC 编译时的包含目录(相对于SOURCE_DIR的路径)
# set(NVRTC_RELATIVE_INCLUDE_DIRS "")

# NVRTC 编译时的包含目录, 绝对路径
set(NVRTC_ABSOLUTE_INCLUDE_DIRS "\\
  \"${CMAKE_SOURCE_DIR}/include\", \\
  \"${OptiX_INCLUDE}\", \\
  \"${CUDA_INCLUDE_DIRS}\", ")

# Build a null-terminated option list for NVRTC
set(CUDA_NVRTC_OPTIONS)
foreach(flag ${CUDA_NVRTC_FLAGS})
  set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS} \\\n  \"${flag}\",")
endforeach()
set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS}")

configure_file(${PROJECT_SOURCE_DIR}/config/projectConfig.h.in ${CMAKE_BINARY_DIR}/projectConfig.h @ONLY)

#########################################################
# OPTIX_add_sample_executable
#
# 定义一个函数来添加子项目. 注意CMake 中函数有自己的定义域, 但是宏使用的是调用者的定义域。
# ARGC 表示函数参数个数, ARGV0, ARGV1... 表示传给函数的参数. ARGN 存储超出预期定义参数个数
# 的参数列表. 例如如果函数定义的时候只有一个参数, 调用的时候有3个参数, 那么多余的两个参数会
# 存储于 ARGN
function(OPTIX_add_sample_executable target_name)

  # 将 PTX 文件和 CUDA 文件进行归类, 归类后可以在 Visual Studio 项目中看到(筛选器)
  if (NOT CUDA_NVRTC_ENABLED)
    source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  endif()
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

  # 将函数参数分离为源文件, CMake 选项和 CUDA 选项. 该宏来自 FindCUDA.cmake, 直接借用过来.
  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  if (CUDA_NVRTC_ENABLED)

    # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
    set(cu_obj_source_files)
    foreach(file ${source_files})
      get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
      if(${_cuda_source_format} MATCHES "OBJ")
        list(APPEND cu_obj_source_files ${file})
      endif()
    endforeach()

    # CUDA_WRAP_SRCS 应该会创建用于编译 CUDA 文件生成 OBJ 或者 PTX 的脚本. 生成的脚本
    # 在 build 的时候调用 nvcc 进行编译.
    # CUDA_ADD_EXECUTABLE 等宏内部就是调用这个命令, 具体看 FindCUDA.cmake 文件
    CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )
  else()

    # generated_files 包含生成的 PTX 与 OBJ 文件目录.
    CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )
  endif()

  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
  add_executable(${target_name}
    ${source_files}
    ${generated_files}
    ${cmake_options}
    )

  # Most of the samples link against the sutil library and the optix library.  Here is the
  # rule that specifies this linkage.
  target_link_libraries( ${target_name}
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    )

  if(CUDA_NVRTC_ENABLED)
    target_link_libraries(${target_name} ${CUDA_nvrtc_LIBRARY})
  endif()
  if(WIN32)
    target_link_libraries(${target_name} winmm.lib)
  endif()

  target_include_directories( ${target_name}
    PRIVATE 
    ${OptiX_INCLUDE}
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
    )
    
  if( UNIX AND NOT APPLE )
    # Force using RPATH instead of RUNPATH on Debian
    target_link_libraries( ${target_name} "-Wl,--disable-new-dtags" )
  endif()

  if(USING_GNU_CXX)
    target_link_libraries( ${target_name} m ) # Explicitly link against math library (C samples don't do that by default)
  endif()
endfunction()