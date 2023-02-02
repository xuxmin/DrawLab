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
    GeometryData geometry_data;
    MaterialData material_data;
};

/**
 * The SBT(shader binding table) connects geometric data to programs
 *
 * header: Opaque to the application, filled in by optixSbtRecordPackHeader.
 *      uased by Optix 7 to identify different behaviour, such as any-hit,
 *      intersection...
 *
 * data: Opaque to NVIDIA OptiX 7. can store program parameter values.
 */
/*! SBT record for a raygen program */
template <typename T> struct Record {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData> RayGenRecord;
typedef Record<MissData> MissRecord;
typedef Record<HitGroupData> HitgroupRecord;

}  // namespace optix