#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include "optix/common/geometry_data.h"
#include "optix/common/material_data.h"


namespace optix {

// for this simple example, we have a single ray type
enum { RAY_TYPE_RADIANCE = 0, RAY_TYPE_OCCLUSION, RAY_TYPE_COUNT };

struct LaunchParams {
    int frameID{0};

    // frame
    unsigned int* color_buffer;
    int width;
    int height;

    // camera
    float3 eye;
    float3 U;
    float3 V;
    float3 W;

    OptixTraversableHandle handle;
};

struct RayGenData {};

struct MissData {};

struct HitGroupData {
    // float3 color;
    // float3* vertex;
    // float3* normal;
    // float2* texcoord;
    // int3* index;
    // bool hasTexture;
    // cudaTextureObject_t texture;
    GeometryData geometry_data;
    MaterialData material_data;
};

}  // namespace optix