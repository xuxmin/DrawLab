#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include "optix/shape/shape.h"
#include "optix/material/material.h"
#include "optix/camera/camera.h"
#include "optix/light/light.h"


namespace optix {

// for this simple example, we have a single ray type
enum { RAY_TYPE_RADIANCE = 0, RAY_TYPE_OCCLUSION, RAY_TYPE_COUNT };

struct Params {
    // frame
    float3* color_buffer;
    int width;
    int height;
    int subframe_index;

    // sample
    int spp;
    float epsilon = 1e-3f;

    // camera
    Camera camera;

    // lights
    LightBuffer light_buffer;

    // materials
    MaterialBuffer material_buffer;

    OptixTraversableHandle handle;
};

struct EmptyData {};

struct HitGroupData {
    Shape geometry_data;
    // MaterialData material_data;
    int light_idx;      // This is an area light bind to mesh
    int material_idx;
};

template <typename T> struct Record {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<EmptyData> RayGenRecord;
typedef Record<EmptyData> MissRecord;
typedef Record<HitGroupData> HitgroupRecord;
typedef Record<EmptyData> CallablesRecord;

}  // namespace optix