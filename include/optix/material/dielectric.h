#include "optix/host/material.h"
#include "optix/host/texture.h"

namespace optix {

class Dielectric : public Material {
public:
    Dielectric(std::string mat_id, DeviceContext& device_context, float intIOR,
               float extIOR)
        : Material(
              mat_id, MaterialData::DIELECTRIC, device_context,
              "optix/material/dielectric.cu",
              {{0, "__closesthit__radiance"}, {1, "__closesthit__occlusion"}},
              {{0, "__anyhit__radiance"}, {1, "__anyhit__occlusion"}}),
          m_extIOR(extIOR), m_intIOR(intIOR) {}

    void packHitgroupRecord(HitgroupRecord& record) const {
        record.data.material_data.dielectric.extIOR = m_extIOR;
        record.data.material_data.dielectric.intIOR = m_intIOR;
    }

    ~Dielectric() {}

private:
    float m_extIOR;
    float m_intIOR;
};

}  // namespace optix