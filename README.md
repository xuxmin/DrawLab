

## Install

### Dependencies

- Optix SDK 7.3.0
- CUDA Toolkit 11.1
- Nvidia card with dirver

### Compile

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

### Usage

```
usage: .\drawlab.exe --scene=string [options] ... 
options:
  -s, --scene      Scene xml file (string)
  -t, --thread     Thread num used in cpu backend (int [=4])
  -b, --backend    Backend:[cpu, optix] (string [=cpu])
      --gui        Show GUI
  -?, --help       print this message
```

  

