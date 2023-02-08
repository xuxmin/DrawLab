#include <optix.h>
#include <optix_device.h>

#include "optix/common/optix_params.h"
#include "optix/common/vec_math.h"
#include "optix/device/random.h"
#include "optix/device/util.h"

namespace optix {

static __forceinline__ __device__ void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}

extern "C" __global__ void __miss__radiance() {
    RadiancePRD* prd = getPRD<RadiancePRD>();
    // set to constant white as background color
    prd->radiance = make_float3(1.f, 0.f, 0.f);
}

extern "C" __global__ void __miss__occlusion() {
    // setPayloadOcclusion(true);
}

}  // namespace optix