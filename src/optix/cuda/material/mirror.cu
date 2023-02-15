#include "optix/math/random.h"
#include "optix/math/vec_math.h"
#include "optix/math/wrap.h"
#include "../shader_common.h"

namespace optix {

extern "C" __constant__ Params params;

extern "C" __device__ float3 __direct_callable__mirror_eval(
    const Material& mat_data, const BSDFQueryRecord& bRec) {
    return make_float3(0.f);
}

extern "C" __device__ float
__direct_callable__mirror_pdf(const Material& mat_data,
                              const BSDFQueryRecord& bRec) {
    return 0.0f;
}

extern "C" __device__ float3 __direct_callable__mirror_sample(
    const Material& mat_data, unsigned int& seed, BSDFQueryRecord& bRec) {
    if (bRec.wi.z <= 0) {
        return make_float3(0.f);
    }
    bRec.wo = make_float3(-bRec.wi.x, -bRec.wi.y, bRec.wi.z);
    bRec.measure = EDiscrete;
    bRec.eta = 1.f;

    return make_float3(1.f);
}

}  // namespace optix