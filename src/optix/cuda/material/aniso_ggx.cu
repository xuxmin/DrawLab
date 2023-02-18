#include "optix/math/random.h"
#include "optix/math/vec_math.h"
#include "optix/math/wrap.h"
#include "optix/math/onb.h"
#include "../shader_common.h"

namespace optix {

static __forceinline__ __device__ 
float G1(const float3& v, const float ax, const float ay) {
	if (v.z <= 0.0)
		return 0.0;
	float3 vv = make_float3(v.x * ax, v.y * ay, v.z);
	return 2.0 * v.z / (v.z + length(vv));
}

static __forceinline__ __device__
void getGGXData(const Material& mat_data, const BSDFQueryRecord& bRec,
                float3& pd, float3& ps, float2& axay) {
    const float2& uv = bRec.its.uv;

    pd = mat_data.aniso_ggx.pd;
    if (mat_data.aniso_ggx.pd_tex) {
        pd = make_float3(tex2D<float4>(mat_data.aniso_ggx.pd_tex, uv.x, uv.y));
    }

    ps = mat_data.aniso_ggx.ps;
    if (mat_data.aniso_ggx.ps_tex) {
        ps = make_float3(tex2D<float4>(mat_data.aniso_ggx.ps_tex, uv.x, uv.y));
    }

    axay = mat_data.aniso_ggx.axay;
    if (mat_data.aniso_ggx.axay_tex) {
        axay = make_float2(tex2D<float4>(mat_data.aniso_ggx.axay_tex, uv.x, uv.y));
    }
}

static __forceinline__ __device__ float3 eval(
    const Material& mat_data, const BSDFQueryRecord& bRec) {

    if (bRec.wi.z <= 0 || bRec.wo.z <= 0) {
        return make_float3(0.f);
    }

    float3 pd, ps;
    float2 axay;
    getGGXData(mat_data, bRec, pd, ps, axay);

    float3 wh = normalize(bRec.wi + bRec.wo);

    float D = Wrap::squareToGGXPdf(wh, axay);
    float F;
    {
		const float F0 = 0.04;// metalness;
		float ldoth = dot(bRec.wo, wh);
		float temp = fminf(fmaxf(0.0, 1.0 - ldoth), 1.0);
		F = F0 + (1.0 - F0) * temp * temp * temp * temp * temp;
    }
    float G = G1(bRec.wi, axay.x, axay.y) * G1(bRec.wo, axay.x, axay.y);
    float deno = 1.f / (4.f * bRec.wi.z * bRec.wo.z);
    return pd * M_INV_PI + ps * D * F * G * deno;
}

static __forceinline__ __device__ 
float pdf(const Material& mat_data, const BSDFQueryRecord& bRec) {
    if (bRec.measure != ESolidAngle || bRec.wi.z <= 0 || bRec.wo.z <= 0) {
        return 0.f;
    }
    float3 pd, ps;
    float2 axay;
    getGGXData(mat_data, bRec, pd, ps, axay);

    float3 wh = normalize(bRec.wi + bRec.wo);

    float diff_pdf = (1 - 0.5f) * Wrap::squareToCosineHemispherePdf(bRec.wo);
    float spec_pdf = 0.5f * Wrap::squareToGGXPdf(wh, axay) * wh.z / 4.f / dot(wh, bRec.wo);
    return diff_pdf + spec_pdf;
}

extern "C" __device__ float3 __direct_callable__anisoggx_eval(
    const Material& mat_data, const BSDFQueryRecord& bRec) {
    return eval(mat_data, bRec);
}

extern "C" __device__ float
__direct_callable__anisoggx_pdf(const Material& mat_data,
                               const BSDFQueryRecord& bRec) {
    return pdf(mat_data, bRec);
}

extern "C" __device__ float3 __direct_callable__anisoggx_sample(
    const Material& mat_data, unsigned int& seed, BSDFQueryRecord& bRec) {
    if (bRec.wi.z <= 0) {
        return make_float3(0.f);
    }

    float3 pd, ps;
    float2 axay;
    getGGXData(mat_data, bRec, pd, ps, axay);

    bRec.measure = ESolidAngle;

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    const float2 sample = make_float2(z1, z2);

    // diffuse case
    if (rnd(seed) > 0.5f) {
        bRec.wo = Wrap::squareToCosineHemisphere(sample); 
    }
    else {
        float3 wh = Wrap::squareToGGX(sample, axay);
        bRec.wo = reflect(bRec.wi, wh);
    }

    if (bRec.wo.z <= 0) {
        return make_float3(0.f);
    }

    return eval(mat_data, bRec) * bRec.wo.z / pdf(mat_data, bRec);
}

}  // namespace optix