#include "optix/host/material.h"
#include "optix/host/texture.h"

namespace optix {

class Mirror : public Material {
public:
    Mirror(std::string mat_id, DeviceContext& device_context)
        : Material(
              mat_id, MaterialData::MIRROR, device_context,
              "optix/material/mirror.cu",
              {{0, "__closesthit__radiance"}, {1, "__closesthit__occlusion"}},
              {{0, "__anyhit__radiance"}, {1, "__anyhit__occlusion"}}) {}

    void packHitgroupRecord(HitgroupRecord& record) const {
    }

    ~Mirror() {}
};

}  // namespace optix