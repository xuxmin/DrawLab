add_library( tiny_obj_loader 
    tiny_obj_loader.h
    tiny_obj_loader.cc
  )

target_include_directories( tiny_obj_loader
  INTERFACE .)
