# Build glad
add_library( glad
  KHR/khrplatform.h
  glad/glad.c
  glad/glad.h
)
# target_compile_definitions( glad
#   PRIVATE GLAD_GLAPI_EXPORT_BUILD
#   PUBLIC  GLAD_GLAPI_EXPORT )
target_include_directories( glad PUBLIC . )
target_link_libraries( glad PUBLIC ${OPENGL_LIBRARIES} )
