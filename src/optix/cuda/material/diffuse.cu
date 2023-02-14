#include "optix/common/bsdf_common.h"
#include "optix/math/random.h"
#include "optix/math/vec_math.h"
#include "optix/math/wrap.h"

namespace optix {

extern "C" __device__ float3 __direct_callable__diffuse_eval(
    const Material& mat_data, const BSDFQueryRecord& bRec) {
    float3 bsdf_val;
    if (bRec.wi.z <= 0 || bRec.wo.z <= 0) {
        bsdf_val = make_float3(0.f);
    }

    float4 albedo = mat_data.diffuse.albedo;
    if (mat_data.diffuse.albedo_tex) {
        albedo = tex2D<float4>(mat_data.diffuse.albedo_tex, bRec.its.uv.x,
                               bRec.its.uv.y);
    }
    bsdf_val = make_float3(albedo) * M_1_PIf;
    return bsdf_val;
}

extern "C" __device__ float
__direct_callable__diffuse_pdf(const Material& mat_data,
                               const BSDFQueryRecord& bRec) {
    if (bRec.wi.z <= 0 || bRec.wo.z <= 0) {
        return 0.f;
    }
    // Note that wo is in local coordinates, so cosTheta(wo)
    // actually just wo.z
    return M_1_PIf * bRec.wo.z;
}

extern "C" __device__ float3 __direct_callable__diffuse_sample(
    const Material& mat_data, unsigned int& seed, BSDFQueryRecord& bRec) {
    if (bRec.wi.z <= 0) {
        return make_float3(0.f);
    }

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    bRec.wo = Wrap::squareToCosineHemisphere(make_float2(z1, z2));

    bRec.eta = 1.f;

    // Return eval() / pdf() * cos(theta) = albedo.
    float4 albedo = mat_data.diffuse.albedo;
    if (mat_data.diffuse.albedo_tex) {
        albedo = tex2D<float4>(mat_data.diffuse.albedo_tex, bRec.its.uv.x,
                               bRec.its.uv.y);
    }
    return make_float3(albedo);
}

}  // namespace optix