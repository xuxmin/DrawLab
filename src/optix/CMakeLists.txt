set(TARGET_NAME "optixlib")
set(SOURCES
    sutil.cpp
    texture.cpp
)

find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED)

add_library(${TARGET_NAME} ${SOURCES})

target_include_directories(${TARGET_NAME} 
    PUBLIC
    ${OptiX_INCLUDE}
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
)

# Use gcc rather than g++ to link if we are linking statically against libgcc_s and libstdc++
if(USING_GNU_C OR USING_GNU_CXX)
    if(GCC_LIBSTDCPP_HACK)
        set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE "C")
        target_link_libraries(${TARGET_NAME} LINK_PRIVATE ${STATIC_LIBSTDCPP})
    endif()
endif()


if(CUDA_NVRTC_ENABLED)
    target_link_libraries(${TARGET_NAME} LINK_PRIVATE ${CUDA_nvrtc_LIBRARY})
endif()
if(WIN32)
    target_link_libraries(${TARGET_NAME} LINK_PRIVATE winmm.lib)
endif()
