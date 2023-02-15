#include "optix/math/vec_math.h"
#include "shader_common.h"
#include "per_ray_data.h"

namespace optix {

extern "C" __global__ void __miss__radiance() {
    RadiancePRD* prd = getPRD<RadiancePRD>();
    // set to constant white as background color
    prd->radiance = make_float3(0.f, 0.f, 0.f);
    prd->done = true;
}

extern "C" __global__ void __miss__occlusion() {
    // setPayloadOcclusion(true);
}

}  // namespace optix