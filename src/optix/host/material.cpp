#include "optix/host/material.h"

namespace optix {

Material::Material(std::string mat_id, MaterialData::Type mat_type,
                   DeviceContext& device_context, std::string cu_file,
                   const std::vector<std::pair<int, const char*>> closet_hits,
                   const std::vector<std::pair<int, const char*>> any_hits)
    : m_material_id(mat_id), m_material_type(mat_type) {
    m_module = device_context.createModuleFromCU(cu_file);

    // TODO: assert ray_type_num ?

    for (int i = 0; i < closet_hits.size(); i++) {
        auto& ch = closet_hits[i];
        auto& ah = any_hits[i];
        m_hitgroup_pgs.push_back(device_context.createHitgroupPrograms(
            m_module, ch.second, ah.second));
    }
}

Material::~Material() {
    OPTIX_CHECK(optixModuleDestroy(m_module));
    for (auto pg : m_hitgroup_pgs) {
        OPTIX_CHECK(optixProgramGroupDestroy(pg));
    }
}

}  // namespace optix