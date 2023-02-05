#include <optix.h>
#include <optix_device.h>

#include "optix/common/optix_params.h"
#include "optix/common/vec_math.h"
#include "optix/device/random.h"
#include "optix/device/raygen.h"
#include "optix/device/util.h"

namespace optix {

/**
 * Launch-varying parameters.
 *
 * This params can be accessible from any module in a pipeline.
 * - declare with extern "C" and __constant__
 * - set in OptixPipelineCompileOptions
 * - filled in by optix upon optixLaunch
 */
extern "C" __constant__ LaunchParams params;

static __forceinline__ __device__ void
traceRadiance(OptixTraversableHandle handle, float3 ray_origin,
              float3 ray_direction, float tmin, float tmax, RadiancePRD* prd) {
    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
               0.0f,  // rayTime
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               RAY_TYPE_RADIANCE,  // SBT offset
               RAY_TYPE_COUNT,     // SBT stride
               RAY_TYPE_RADIANCE,  // missSBTIndex
               u0, u1);
}

extern "C" __global__ void __raygen__simple() {
    const uint3 idx = optixGetLaunchIndex();
    float3 ray_origin, ray_direction;
    genCameraRay(params, idx, ray_origin, ray_direction);

    RadiancePRD prd;
    prd.radiance = make_float3(0.f);

    traceRadiance(params.handle, ray_origin, ray_direction,
                  0.01f,  // tmin
                  1e20f,  // tmax
                  &prd);


    // and write to frame buffer ...
    const unsigned int image_index = idx.x + idx.y * params.width;
    params.color_buffer[image_index] = prd.radiance;
}
}  // namespace optix