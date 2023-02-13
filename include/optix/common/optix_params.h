#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include "optix/common/geometry_data.h"
#include "optix/common/material_data.h"
#include "optix/common/camera.h"
#include "optix/common/light_data.h"


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

    // camera
    Camera camera;

    // lights
    LightData light_data;

    OptixTraversableHandle handle;
};

struct RayGenData {};

struct MissData {};

struct HitGroupData {
    GeometryData geometry_data;
    MaterialData material_data;
    int light_idx;  // This is an area light bind to mesh
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

// BSDF sample record in the prev path
struct BSDFSampleRecord {
    float3 fr;  // eval() / pdf() * cos(theta)
    float eta;
    float3 p;   // surface point position
    float3 wo;  // sampled direction in world coordinate.
    float pdf;
    bool is_diffuse;

    BSDFSampleRecord()
        : fr(make_float3(1.f)), eta(1.f), pdf(0.f), is_diffuse(false) {}
};

enum EMeasure { EUnknownMeasure = 0, ESolidAngle, EDiscrete };

struct BSDFQueryRecord {
    /// Reference to the underlying surface interaction
    const Intersection& its;
    /// Incident direction (in the local frame)
    float3 wi;
    /// Outgoing direction (in the local frame)
    float3 wo;
    /// Relative refractive index in the sampled direction
    float eta;
    /// Measure associated with the sample
    EMeasure measure;

    BSDFQueryRecord(const Intersection& its, const float3& wi)
        : its(its), wi(wi), eta(1.f), measure(EUnknownMeasure) {}

    BSDFQueryRecord(const Intersection& its, const float3& wi, const float3& wo,
                    EMeasure measure)
        : its(its), wi(wi), wo(wo), eta(1.f), measure(measure) {}
};

/**
 * The payload is associated with each ray, and is passed to all
 * the intersection, any-hit, closest-hit and miss programs that
 * are executed during this invocation of trace.
 */
struct RadiancePRD {
    float3 radiance;
    bool done;
    BSDFSampleRecord sRec;

    /**
     * The initial seed of each path.
     *
     * We assign a initial seed for each path for the convenience of debuging.
     *
     * Notice:
     * 1. rnd(seed) takes the reference of seed as input, each call of rnd(seed)
     * will change the value of the seed.
     * 2. Initialize the seed at raygen programs
     * 3. Don't copy the seed value to a new variable, use REFERENCE instead!!!
     * 4. Each time call rnd(seed), make sure the prd.seed is changed!!!
     */
    unsigned int seed;
};

}  // namespace optix