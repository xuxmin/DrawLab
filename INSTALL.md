## Requirements

- Optix SDK 7.3.0
- CUDA Toolkit 11.1
- Nvidia card with dirver

## Compile

```
$ git clone https://github.com/xuxmin/DrawLab.git --recursive
$ cd Drawlab
$ mkdir build
$ cd build
$ cmake ..
$ make
```

If the configuration fails, you need to modify these options manually in CMakeCache.txt:
- `OptiX_INSTALL_DIR`: the Optix SDK directory 
- `CUDA_TOOLKIT_INCLUDE`: the cuda toolkit include directory, for example: `/usr/local/cuda-11.1/include`
- `CUDA_TOOLKIT_ROOT_DIR`: the cuda toolkit directory, for example: ``/usr/local/cuda-11.1``

configure again:
```
$ cmake ..
$ make
```