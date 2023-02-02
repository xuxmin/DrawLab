#include <optix.h>
#include <optix_device.h>

#include "optix/device/random.h"
#include "optix/common/optix_params.h"
#include "optix/common/vec_math.h"


namespace optix {

/**
 * Launch-varying parameters.
 * 
 * This params can be accessible from any module in a pipeline.
 * - declare with extern "C" and __constant__
 * - set in OptixPipelineCompileOptions
 * - filled in by optix upon optixLaunch
*/
extern "C"  __constant__ LaunchParams params;

/**
 * The payload is associated with each ray, and is passed to all 
 * the intersection, any-hit, closest-hit and miss programs that 
 * are executed during this invocation of trace.
*/
struct RadiancePRD {
    float3       radiance;
};

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ RadiancePRD* getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}

static __forceinline__ __device__ void
traceRadiance(OptixTraversableHandle handle, float3 ray_origin,
              float3 ray_direction, float tmin, float tmax, RadiancePRD* prd) {
    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
               0.0f,  // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               RAY_TYPE_RADIANCE,  // SBT offset
               RAY_TYPE_COUNT,     // SBT stride
               RAY_TYPE_RADIANCE,  // missSBTIndex
               u0, u1);
}

static __forceinline__ __device__ bool
traceOcclusion(OptixTraversableHandle handle, float3 ray_origin,
               float3 ray_direction, float tmin, float tmax) {
    unsigned int occluded = 0u;
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
               0.0f,  // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               RAY_TYPE_OCCLUSION,  // SBT offset
               RAY_TYPE_COUNT,      // SBT stride
               RAY_TYPE_OCCLUSION,  // missSBTIndex
               occluded);
    return occluded;
}

//---------------------------------------------------------------------
// These program types are specified by prefixing the program’s name with the following
//  Ray generation          __raygen__ 
//  Intersection            __intersection__ 
//  Any-hit                 __anyhit__ 
//  Closest-hit             __closesthit__ 
//  Miss                    __miss__ 
//  Direct callable         __direct_callable__ 
//  Continuation callable   __continuation_callable__ 
//  Exception               __exception__
//
// Each program may call a specific set of device-side intrinsics that 
// implement the actual ray-tracing-specific features
//---------------------------------------------------------------------

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
    RadiancePRD* prd = getPRD();
    // set to constant white as background color
    prd->radiance = make_float3(1.f, 0.f, 0.f);
}

extern "C" __global__ void __miss__occlusion() {
    // setPayloadOcclusion(true);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
    const int    w   = params.width;
    const int    h   = params.height;
    const float3 eye = params.eye;
    const float3 U   = params.U;
    const float3 V   = params.V;
    const float3 W   = params.W;

    const uint3  idx = optixGetLaunchIndex();

    const float2 d = 2.0f * make_float2(
             static_cast<float>(idx.x) / static_cast<float>(w),
             static_cast<float>(idx.y) / static_cast<float>(h)) - 1.0f;
    float3 ray_direction = normalize(d.x*U + d.y*V + W);
    float3 ray_origin    = eye;

    // printf("%lf %lf %lf\n", ray_direction.x, ray_direction.y, ray_direction.z);

    RadiancePRD prd;
    prd.radiance = make_float3(0.f);

    traceRadiance(params.handle, 
                  ray_origin,
                  ray_direction,
                  0.01f,  // tmin
                  1e20f,  // tmax
                  &prd);

    const int r = int(255.99f*prd.radiance.x);
    const int g = int(255.99f*prd.radiance.y);
    const int b = int(255.99f*prd.radiance.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const unsigned int rgba = 0xff000000 | (r<<0) | (g<<8) | (b<<16);

    // and write to frame buffer ...
    const unsigned int image_index = idx.x + idx.y * params.width;
    params.color_buffer[image_index] = rgba;
}
}  // namespace optix