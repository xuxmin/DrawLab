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
$ cmake .. -DOptiX_INSTALL_DIR=<optix install dir>
$ make
```