#pragma once

#include "optix/material/dielectric.h"
#include "optix/material/diffuse.h"
#include "optix/material/mirror.h"
#include "optix/material/microfacet.h"
#include "optix/material/aniso_ggx.h"
#include <cuda_runtime.h>

namespace optix {

enum {
    MATERIAL_CALLABLE_EVAL = 0,
    MATERIAL_CALLABLE_PDF = 1,
    MATERIAL_CALLABLE_SAMPLE = 2,
    MATERIAL_CALLABLE_NUM = 3
};

struct Material {
    enum Type {
        DIFFUSE = 0,
        MICROFACET = 1,
        MIRROR = 2,
        DIELECTRIC = 3,
        ANISOGGX = 4,
        MATERIAL_NUM = 5
    };

    Type type;

    union {
        Diffuse diffuse;
        Dielectric dielectric;
        Mirror mirror;
        Microfacet microfacet;
        AnisoGGX aniso_ggx;
    };

    cudaTextureObject_t normal_tex = 0;     // normal texture
    cudaTextureObject_t tangent_tex = 0;
    bool is_tangent_space = {false};     // the normal/tangent texture is in tangent space?
    bool is_diffuse = {true};
    bool is_hide = {false};
};

struct MaterialBuffer {
    Material* materials;
    int material_num;
};

}  // namespace optix