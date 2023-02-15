#include "optix/math/onb.h"
#include "optix/math/random.h"
#include "optix/math/vec_math.h"
#include "../shader_common.h"

namespace optix {

extern "C" __device__ float3 __direct_callable__dielectric_eval(
    const Material& mat_data, const BSDFQueryRecord& bRec) {
    return make_float3(0.f);
}

extern "C" __device__ float
__direct_callable__dielectric_pdf(const Material& mat_data,
                                  const BSDFQueryRecord& bRec) {
    return 0.0f;
}

extern "C" __device__ float3 __direct_callable__dielectric_sample(
    const Material& mat_data, unsigned int& seed, BSDFQueryRecord& bRec) {
    bRec.measure = EDiscrete;
    float cosThetai = bRec.wi.z;
    float extIOR = mat_data.dielectric.extIOR;
    float intIOR = mat_data.dielectric.intIOR;
    float ri = fresnel(cosThetai, extIOR, intIOR);

    if (rnd(seed) < ri) {
        bRec.wo = make_float3(-bRec.wi.x, -bRec.wi.y, bRec.wi.z);
        bRec.eta = 1;
    }
    else {
        float eta = extIOR / intIOR;
        bRec.wo = normalize(refract(bRec.wi, eta));
        bRec.eta = cosThetai > 0.f ? eta : 1 / eta;
    }
    return make_float3(1.f);
}

}  // namespace optix