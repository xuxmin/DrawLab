#include "optix/host/material.h"
#include "optix/host/texture.h"

namespace optix {

class Diffuse : public Material {
public:
    Diffuse(std::string mat_id, DeviceContext& device_context, float4 albedo)
        : Material(
              mat_id, MaterialData::DIFFUSE, device_context,
              "optix/cuda/diffuse.cu",
              {{0, "__closesthit__radiance"}, {1, "__closesthit__occlusion"}},
              {{0, "__anyhit__radiance"}, {1, "__anyhit__occlusion"}}),
          m_albedo_val(albedo), m_albedo_tex(nullptr) {}

    Diffuse(std::string mat_id, DeviceContext& device_context,
            const Texture* albedo)
        : Material(
              mat_id, MaterialData::DIFFUSE, device_context,
              "optix/cuda/diffuse.cu",
              {{0, "__closesthit__radiance"}, {1, "__closesthit__occlusion"}},
              {{0, "__anyhit__radiance"}, {1, "__anyhit__occlusion"}}),
          m_albedo_val(make_float4(0.f, 0.f, 0.f, 0.f)), m_albedo_tex(albedo) {}

    void packHitgroupRecord(HitgroupRecord& record) const {
        MaterialData& mat_data = record.data.material_data;

        mat_data.type = m_material_type;
        mat_data.diffuse.albedo = m_albedo_val;
        mat_data.diffuse.albedo_tex =
            m_albedo_tex ? m_albedo_tex->getObject() : 0;
        mat_data.diffuse.normal_tex = 0;
    }

private:
    float4 m_albedo_val;
    const Texture* m_albedo_tex;
};

}  // namespace optix