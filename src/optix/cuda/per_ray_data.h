#pragma once
#include <cuda_runtime.h>
#include "optix/shape/shape.h"

namespace optix {


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

static __forceinline__ __device__ void* unpackPointer(unsigned int i0,
                                                      unsigned int i1) {
    const unsigned long long uptr =
        static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0,
                                                   unsigned int& i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T> static __forceinline__ __device__ T* getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}

static __forceinline__ __device__ bool
traceOcclusion(OptixTraversableHandle handle, float3 ray_origin,
               float3 ray_direction, float tmin, float tmax) {
    unsigned int occluded = 0u;
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
               0.0f,  // rayTime
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               RAY_TYPE_OCCLUSION,  // SBT offset
               RAY_TYPE_COUNT,      // SBT stride
               RAY_TYPE_OCCLUSION,  // missSBTIndex
               occluded);
    return occluded;
}

}  // namespace optix