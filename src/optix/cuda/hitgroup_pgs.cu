#include <optix.h>
#include <optix_device.h>

#include "optix/optix_params.h"
#include "optix/math/onb.h"
#include "optix/math/vec_math.h"
#include "per_ray_data.h"
#include "shader_common.h"

#define NEXT_EVENT_ESTIMATION

namespace optix {

extern "C" __constant__ Params params;

extern "C" __global__ void __closesthit__occlusion() {
    setPayloadOcclusion(true);
}

extern "C" __global__ void __closesthit__radiance() {
    const HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    const Material& mat_data =
        params.material_buffer.materials[rt_data->material_idx];

    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 ray_ori = optixGetWorldRayOrigin();
    RadiancePRD* prd = getPRD<RadiancePRD>();
    unsigned int& seed = prd->seed;
    Intersection its = getHitData(mat_data);

    float3 radiance = make_float3(0.f);

    if (its.light_idx >= 0) {
        const Light& light = params.light_buffer.lights[its.light_idx];
        const BSDFSampleRecord& sRec = prd->sRec;

        float3 light_val = light.eval(its, -ray_dir);

        float mis = 1.f;

#ifdef NEXT_EVENT_ESTIMATION
        LightSampleRecord dRec(ray_ori, its.p, its.sn, its.mesh);
        float light_pdf =
            params.light_buffer.pdfLightDirection(its.light_idx, dRec);
        float bsdf_pdf = sRec.pdf;
        mis = sRec.is_diffuse ? powerHeuristic(bsdf_pdf, light_pdf) : 1.f;
#endif

        radiance += mis * light_val;
    }


    // -----------------------------------------------------
    // Hide object?
    // -----------------------------------------------------
    if (mat_data.is_hide) {
        prd->sRec.fr = make_float3(1.f);
        prd->sRec.eta = 1.f;
        prd->sRec.p = its.p;
        prd->sRec.wo = ray_dir;
        if (prd->depth != 0) {
            prd->radiance = radiance;
        }
        return;
    }

    Onb onb(its.sn, its.st);
    const float3 wi = normalize(onb.transform(-ray_dir));

#ifdef NEXT_EVENT_ESTIMATION
    // --------------------- Emitter sampling ---------------------
    LightSampleRecord dRec;
    float3 light_val =
        params.light_buffer.sampleLightDirection(its, seed, dRec);

    if (fmaxf(light_val) > 0.f) {
        // Trace occlusion
        const bool occluded =
            traceOcclusion(params.handle, its.p, dRec.d,
                        params.epsilon,             // tmin
                        dRec.dist - params.epsilon  // tmax
            );

        const float3 wo = onb.transform(dRec.d);
        BSDFQueryRecord bRec(its, wi, wo, ESolidAngle);
        if (!occluded && dRec.pdf > 0) {
            float3 bsdf_val =
                optixDirectCall<float3, const Material&, const BSDFQueryRecord&>(
                    3 * rt_data->material_idx + MATERIAL_CALLABLE_EVAL, mat_data,
                    bRec);

            // Determine density of sampling that same direction using BSDF
            // sampling
            float bsdf_pdf =
                optixDirectCall<float, const Material&, const BSDFQueryRecord&>(
                    3 * rt_data->material_idx + MATERIAL_CALLABLE_PDF, mat_data,
                    bRec);
            float light_pdf = dRec.pdf;
            float weight = dRec.delta ? 1 : powerHeuristic(light_pdf, bsdf_pdf);

            // cos_theta is in sampleLightDirection
            radiance += weight * bsdf_val * light_val;
        }
    }
#endif

    // ----------------------- BSDF sampling ----------------------
    BSDFQueryRecord bsdf_bRec(its, wi);
    float3 fr = optixDirectCall<float3, const Material&, unsigned int&,
                                BSDFQueryRecord&>(3 * rt_data->material_idx +
                                                      MATERIAL_CALLABLE_SAMPLE,
                                                  mat_data, seed, bsdf_bRec);

    // Record throughput, eta
    prd->sRec.fr = fr;
    prd->sRec.eta = bsdf_bRec.eta;

    // BSDF sampled ray direction, also be used as the next path
    // direction
    prd->sRec.wo = normalize(onb.inverse_transform(bsdf_bRec.wo));
    prd->sRec.pdf =
        optixDirectCall<float, const Material&, const BSDFQueryRecord&>(
            3 * rt_data->material_idx + MATERIAL_CALLABLE_PDF, mat_data,
            bsdf_bRec);

    prd->sRec.is_diffuse = mat_data.is_diffuse;

    prd->radiance = radiance;
    prd->depth++;
    prd->its = its;

    if (fmaxf(prd->sRec.fr) <= 0.f) {
        prd->done = true;
    }

    // refine hitpoint
    if (dot(prd->sRec.wo, its.gn) >= 0.f) {     // reflection
        prd->sRec.p = its.p + its.gn * params.epsilon;
    }
    else {                                      // refraction
        prd->sRec.p = its.p - its.gn * params.epsilon;
    }
}

extern "C" __global__ void __anyhit__radiance() {}

extern "C" __global__ void __anyhit__occlusion() {}

}  // namespace optix