#pragma once

#include "optix/common/optix_params.h"
#include "optix/host/device_context.h"
#include "optix/host/sutil.h"
#include <map>
#include <string>
#include <vector>

namespace optix {

class Material {
public:
    /// @brief
    /// @param device_context
    /// @param cu_file the file path relative to SOURCE_DIR
    Material(std::string mat_id, MaterialData::Type mat_type,
             DeviceContext& device_context, std::string cu_file,
             const std::vector<std::pair<int, const char*>> closet_hits,
             const std::vector<std::pair<int, const char*>> any_hits);

    /// @brief Pack the material data of the hitgroup record.
    virtual void packHitgroupRecord(HitgroupRecord& record) const = 0;

    const OptixProgramGroup& getHitgroupPGs(int ray_type) const {
        return m_hitgroup_pgs[ray_type];
    }

protected:
    std::string m_material_id;
    MaterialData::Type m_material_type;

    /**
     * The module that contains out device programs
     * Programs are first compiled into modules
     */
    OptixModule m_module;

    /// @brief hitgroup programs for all ray types
    std::vector<OptixProgramGroup> m_hitgroup_pgs;
};

}  // namespace optix