#include "optix/math/random.h"
#include "optix/math/vec_math.h"
#include "optix/math/wrap.h"
#include "../shader_common.h"

namespace optix {

static __forceinline__ __device__ float tanTheta(const float3& v) {
    float temp = 1 - v.z * v.z;
    if (temp <= 0.0f)
        return 0.0f;
    return sqrtf(temp) / v.z;
}

static __forceinline__ __device__ 
float G1(const float3& wv, const float3& wh, float alpha) {
    float c = dot(wv, wh) / wv.z;
    if (c <= 0.f) {
        return 0.f;
    }
    float b = 1.0 / (alpha * tanTheta(wv));
    return b < 1.6 ?
               (3.535 * b + 2.181 * b * b) / (1 + 2.276 * b + 2.577 * b * b) :
               1;
}

static __forceinline__ __device__ float3 eval(
    const Material& mat_data, const BSDFQueryRecord& bRec) {
    float alpha = mat_data.microfacet.alpha;
    float3 kd = make_float3(mat_data.microfacet.kd);
    float ks = mat_data.microfacet.ks;

    float3 wh = normalize(bRec.wi + bRec.wo);

    float D = Wrap::squareToBeckmannPdf(wh, alpha);

    // Microfacet model, note cos(wi, wh) not cos(wi, N)
    float F = fresnel(dot(bRec.wi, wh), mat_data.microfacet.extIOR,
                          mat_data.microfacet.intIOR);
    float G = G1(bRec.wi, wh, alpha) * G1(bRec.wo, wh, alpha);
    float deno = 1 / (4 * bRec.wi.z * bRec.wo.z);

    return kd * M_INV_PI + ks * D * F * G * deno;
}

static __forceinline__ __device__ 
float pdf(const Material& mat_data, const BSDFQueryRecord& bRec) {
    if (bRec.measure != ESolidAngle || bRec.wi.z <= 0 || bRec.wo.z <= 0) {
        return 0.f;
    }
    float alpha = mat_data.microfacet.alpha;
    float3 kd = make_float3(mat_data.microfacet.kd);
    float ks = mat_data.microfacet.ks;

    float3 wh = normalize(bRec.wi + bRec.wo);

    float diff_pdf = (1 - ks) * Wrap::squareToCosineHemispherePdf(bRec.wo);
    float spec_pdf = ks * Wrap::squareToBeckmannPdf(wh, alpha) / 4.f / dot(wh, bRec.wo);
    return diff_pdf + spec_pdf;
}

extern "C" __device__ float3 __direct_callable__microfacet_eval(
    const Material& mat_data, const BSDFQueryRecord& bRec) {
    return eval(mat_data, bRec);
}

extern "C" __device__ float
__direct_callable__microfacet_pdf(const Material& mat_data,
                               const BSDFQueryRecord& bRec) {
    return pdf(mat_data, bRec);
}

extern "C" __device__ float3 __direct_callable__microfacet_sample(
    const Material& mat_data, unsigned int& seed, BSDFQueryRecord& bRec) {
    if (bRec.wi.z <= 0) {
        return make_float3(0.f);
    }

    float alpha = mat_data.microfacet.alpha;
    float3 kd = make_float3(mat_data.microfacet.kd);
    float ks = mat_data.microfacet.ks;

    bRec.measure = ESolidAngle;

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    const float2 sample = make_float2(z1, z2);
    // diffuse case
    if (rnd(seed) > ks) {
        bRec.wo = Wrap::squareToCosineHemisphere(sample); 
    }
    else {
        float3 wh = Wrap::squareToBeckmann(sample, alpha);
        bRec.wo = reflect(bRec.wi, wh);
    }

    if (bRec.wo.z <= 0) {
        return make_float3(0.f);
    }

    return eval(mat_data, bRec) * bRec.wo.z / pdf(mat_data, bRec);
}

}  // namespace optix