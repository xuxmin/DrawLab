add_library( stb_image 
    stb_image.h
    stb_image.cc
    stb_image_write.h
  )

target_include_directories( stb_image
  INTERFACE .)
