#pragma once

#include <cuda_runtime.h>

namespace optix {

struct MaterialData {
    enum Type { DIFFUSE = 0 };

    struct Diffuse {
        float4 albedo;

        cudaTextureObject_t normal_tex;
        cudaTextureObject_t albedo_tex;
    };

    Type type;

    union {
        Diffuse diffuse;
    };
};

}  // namespace optix