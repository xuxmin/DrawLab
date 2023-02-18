#pragma once
#include <cuda_runtime.h>

namespace optix {

struct AnisoGGX {
    float3 pd;
    float3 ps;
    float2 axay;

    cudaTextureObject_t pd_tex;
    cudaTextureObject_t ps_tex;
    cudaTextureObject_t axay_tex;
};

}  // namespace optix