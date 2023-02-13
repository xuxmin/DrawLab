#include <optix.h>
#include <optix_device.h>

#include "optix/common/optix_params.h"
#include "optix/math/vec_math.h"
#include "optix/math/random.h"
#include "optix/common/bsdf_common.h"

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