#include "optix/common/optix_params.h"

namespace optix {

static __forceinline__ __device__ void genCameraRay(const Params& params,
                                                    const uint3 launch_idx,
                                                    float3& ray_origin,
                                                    float3& ray_direction) {
    const int w = params.width;
    const int h = params.height;

    if (params.camera.type == Camera::VIRTUAL) {
        const float3 eye = params.camera.virtual_cam.eye;
        const float3 U = params.camera.virtual_cam.U;
        const float3 V = -params.camera.virtual_cam.V;
        const float3 W = params.camera.virtual_cam.W;

        const float2 d = 2.0f * make_float2(static_cast<float>(launch_idx.x) /
                                                static_cast<float>(w),
                                            static_cast<float>(launch_idx.y) /
                                                static_cast<float>(h)) -
                         1.0f;
        ray_direction = normalize(d.x * U + d.y * V + W);
        ray_origin = eye;
    }
}

}  // namespace optix