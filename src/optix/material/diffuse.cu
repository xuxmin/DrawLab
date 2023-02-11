#include <optix.h>
#include <optix_device.h>

#include "optix/common/optix_params.h"
#include "optix/common/vec_math.h"
#include "optix/device/random.h"
#include "optix/device/util.h"
#include "optix/device/onb.h"
#include "optix/device/wrap.h"


namespace optix {

extern "C" __constant__ Params params;

#define SFD static __forceinline__ __device__

// Evaluate the BRDF model
SFD float3 eval(const MaterialData& mat_data, const BSDFQueryRecord& bRec) {
    float3 bsdf_val;
    if (bRec.wi.z <= 0 || bRec.wo.z <= 0) {
        bsdf_val = make_float3(0.f);
    }

    float4 albedo = mat_data.diffuse.albedo;
    if (mat_data.diffuse.albedo_tex) {
        albedo = tex2D<float4>(mat_data.diffuse.albedo_tex, bRec.its.uv.x, bRec.its.uv.y);
    }
    bsdf_val = make_float3(albedo) * M_1_PIf;
    return bsdf_val;
}

SFD float pdf(const MaterialData& mat_data, const BSDFQueryRecord& bRec) {
    if (bRec.wi.z <= 0 || bRec.wo.z <= 0) {
        return 0.f;
    }
    // Note that wo is in local coordinates, so cosTheta(wo)
    // actually just wo.z
    return M_1_PIf * bRec.wo.z;
}

SFD float3 sample(const MaterialData& mat_data, unsigned int seed, BSDFQueryRecord& bRec) {
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
        albedo = tex2D<float4>(mat_data.diffuse.albedo_tex, bRec.its.uv.x, bRec.its.uv.y);
    }
    return make_float3(albedo.x, albedo.y, albedo.z);
}

extern "C" __global__ void __closesthit__occlusion() {
    setPayloadOcclusion(true);
}

extern "C" __global__ void __closesthit__radiance() {

    const HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    const MaterialData& mat_data =
        reinterpret_cast<const MaterialData&>(rt_data->material_data);
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 ray_ori = optixGetWorldRayOrigin();
    RadiancePRD* prd = getPRD<RadiancePRD>();
    unsigned int seed = prd->seed;
    Intersection its = getHitData();

    float3 radiance = make_float3(0.f);

    if (its.light_idx >= 0) {
        const Light& light = params.light_data.lights[its.light_idx];
        const BSDFSampleRecord& sRec = prd->sRec;

        float3 light_val = light.eval(its, -ray_dir);

        DirectionSampleRecord dRec(ray_ori, its.p, its.sn, its.mesh);
        float light_pdf = params.light_data.pdfLightDirection(its.light_idx, dRec);
        float bsdf_pdf = sRec.pdf;
        float mis = sRec.is_diffuse ? mis_weight(bsdf_pdf, light_pdf) : 1.f;
        
        radiance += mis * light_val;
    }

    // --------------------- Emitter sampling ---------------------
    DirectionSampleRecord dRec;
    float3 light_val = params.light_data.sampleLightDirection(its, seed, dRec);

    // Trace occlusion
    const bool occluded = traceOcclusion(
        params.handle, its.p + 1e-3f * its.gn, dRec.d,
        0.01f,                      // tmin
        dRec.dist - 0.01f   // tmax
    );

    Onb onb(its.sn);
    const float3 wi = onb.transform(-ray_dir);
    const float3 wo = onb.transform(dRec.d);
    BSDFQueryRecord bRec(its, wi, wo, ESolidAngle);
    if (!occluded && dRec.pdf > 0) {

        float3 bsdf_val = eval(mat_data, bRec);

        // Determine density of sampling that same direction using BSDF
        // sampling
        float bsdf_pdf = pdf(mat_data, bRec);
        float light_pdf = dRec.pdf;
        float weight = dRec.delta ? 1 : mis_weight(light_pdf, bsdf_pdf);

        radiance += weight * bsdf_val * light_val;
    }

    // ----------------------- BSDF sampling ----------------------
    BSDFQueryRecord bsdf_bRec(its, wi);
    float3 fr = sample(mat_data, seed, bsdf_bRec);

    // Record throughput, eta
    prd->sRec.fr = fr;
    prd->sRec.eta = bsdf_bRec.eta;

    // BSDF sampled ray direction, also be used as the next path
    // direction
    prd->sRec.p = its.p;
    prd->sRec.wo = onb.inverse_transform(bsdf_bRec.wo);
    prd->sRec.pdf = pdf(mat_data, bsdf_bRec);

    prd->sRec.is_diffuse = true;

    prd->radiance = radiance;

    if (fmaxf(prd->sRec.fr) <= 0.f) {
        prd->done = true;
    }
}

extern "C" __global__ void __anyhit__radiance() {}

extern "C" __global__ void __anyhit__occlusion() {}

}  // namespace optix