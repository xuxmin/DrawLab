set(TARGET_NAME gllib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_FILES
    display.cpp
    texture.cpp
)

add_library(${TARGET_NAME} ${SOURCE_FILES})

# Link dependencies
target_link_libraries(${TARGET_NAME} glad)


target_include_directories(${TARGET_NAME} 
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${GLAD_INCLUDE_DIRS}        # glad
    ${TFM_INCLUDE_DIR}          # tinyformat string formatting library
)