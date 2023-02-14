#pragma once
#include <cuda_runtime.h>

namespace optix {

struct Diffuse {
    float4 albedo;

    cudaTextureObject_t normal_tex;
    cudaTextureObject_t albedo_tex;
};

}  // namespace optix