#pragma once

#include "optix/material/dielectric.h"
#include "optix/material/diffuse.h"
#include "optix/material/mirror.h"
#include <cuda_runtime.h>

namespace optix {

enum {
    MATERIAL_CALLABLE_EVAL = 0,
    MATERIAL_CALLABLE_PDF = 1,
    MATERIAL_CALLABLE_SAMPLE = 2,
    MATERIAL_CALLABLE_NUM = 3
};

struct Material {
    enum Type { DIFFUSE = 0, MICROFACET, MIRROR, DIELECTRIC, MATERIAL_NUM };

    Type type;

    union {
        Diffuse diffuse;
        Dielectric dielectric;
        Mirror mirror;
    };
    bool is_diffuse = {true};
};

struct MaterialBuffer {
    Material* materials;
    int material_num;
};

}  // namespace optix