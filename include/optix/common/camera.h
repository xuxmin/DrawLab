#pragma once
#include <cuda_runtime.h>


namespace optix{

struct Camera {
    enum Type {
        VIRTUAL = 0,
    };

    struct Virtual {
        float3 eye;
        float3 U;
        float3 V;
        float3 W;

        float3 looat;
        float3 up;
        float fov;
    };

    Type type;

    union {
        Virtual virtual_cam;
    };
    
};
    
} // namespace optix
