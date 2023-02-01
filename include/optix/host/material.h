#pragma once

#include <string>
#include <vector>
#include "optix/host/sutil.h"
#include "optix/host/device_context.h"


namespace optix {

class Material {

public:

    /// @brief 
    /// @param context 
    /// @param cu_file the file path relative to SOURCE_DIR
    Material(const DeviceContext& context, std::string cu_file);


private:
    std::string m_material_id;

    /**
     * The module that contains out device programs
     * 
     * Programs are first compiled into modules
     * 
     *   .cu --nvcc--> .ptx ---> module(optixModule)
    */
    OptixModule module;

    std::vector<OptixProgramGroup> m_hitgroup_pgs;
};

}  // namespace optix