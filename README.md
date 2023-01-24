

## 遇到的问题

一开始 configuration 的时候, 报错 CUDA_CUDA_LIBRARY 找不到. 查了一些资料, 加了下面这一条:

```bash
set(CMAKE_LIBRARY_PATH "/usr/local/cuda/lib64/stubs" CACHE PATH "Path to CUDA_CUDA_LIBRARY location.")
```

编译是成功了, 但是运行到 `optixDeviceContextCreate` 的时候就失败了.

最终看到这里 https://forums.developer.nvidia.com/t/segmentation-fault-in-optix7-1-examples/145412/2 一个人说的:

/usr/lib/x86_64-linux-gnu/libcuda.so.1 to /usr/lib/x86_64-linux-gnu/libcuda.so 缺少一个软连接，我看了一下还真的缺了。把这个软链接加上去就成功了。