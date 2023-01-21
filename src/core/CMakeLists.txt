set(TARGET_NAME corelib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_FILES
    base/common.cpp
    base/string.cpp
    math/math.cpp
    math/matrix.cpp
    math/wrap.cpp
    bitmap/bitmap.cpp
    parser/proplist.cpp
    parser/parser.cpp
    parser/object.cpp
)

add_library(${TARGET_NAME} ${SOURCE_FILES})

# Link dependencies
target_link_libraries(${TARGET_NAME} stb_image)
target_link_libraries(${TARGET_NAME} IlmImf)
target_link_libraries(${TARGET_NAME} pugixml)


target_include_directories(${TARGET_NAME} 
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OPENEXR_INCLUDE_DIRS}     # OpenEXR high dynamic range bitmap library
    ${TFM_INCLUDE_DIR}          # tinyformat string formatting library
    ${FILESYSTEM_INCLUDE_DIR}   # Portable filesystem 
    ${PUGIXML_INCLUDE_DIR}      # pugixml
)