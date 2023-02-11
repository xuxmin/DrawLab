#pragma once

#include <cuda_runtime.h>

namespace optix {

struct MaterialData {
    enum Type { DIFFUSE = 0, MICROFACET, MIRROR, DIELECTRIC};

    struct Diffuse {
        float4 albedo;

        cudaTextureObject_t normal_tex;
        cudaTextureObject_t albedo_tex;
    };

    struct Microfacet {
        float alpha;
        float intIOR;
        float extIOR;
        float ks;
        float3 kd;
    };

    struct Dielectric {
        float intIOR;
        float extIOR;
    };

    Type type;

    union {
        Diffuse diffuse;
        Microfacet microfacet;
        Dielectric dielectric;
    };
};

}  // namespace optix