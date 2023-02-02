#pragma once

#include "optix/host/sutil.h"
#include "optix/host/texture.h"
#include <cuda_runtime.h>
#include <string>
#include <map>

namespace optix {

class Material;

class DeviceContext {
public:
    DeviceContext(int deviceID);
    ~DeviceContext();

    std::string getDeviceName() const;
    const int getDeviceId() const;

    /// @brief  configures the optixPipeline link options and compile options,
    ///         based on what values (motion blur on/off, multi-level
    ///         instnacing, etc) are set in the context
    void configurePipelineOptions();

    /**
     * @brief Create module from cu file. A module may include
     *        multiple programs of any program type
     *          .cu --nvcc--> .ptx ---> module(optixModule)
     * @param cu_file file path relative to SOURCE_DIR
    */
    OptixModule createModuleFromCU(std::string cu_file) const;

    OptixProgramGroup createHitgroupPrograms(OptixModule ch_module,
                                             OptixModule ah_module,
                                             std::string ch_func,
                                             std::string ah_func);

    OptixProgramGroup createHitgroupPrograms(OptixModule module,
                                             std::string ch_func,
                                             std::string ah_func);

    const OptixPipelineCompileOptions* getPipelineCompileOptions() const {
        return &m_pipeline_compile_options;
    }

    const OptixPipelineLinkOptions* getPipelineLinkOptions() const {
        return &m_pipeline_link_options;
    }

    const OptixModuleCompileOptions* getModuleCompileOptions() const {
        return &m_module_compile_options;
    }

    const OptixDeviceContext& getOptixDeviceContext() const {
        return m_optix_context;
    }

    const CUstream& getStream() const {
        return m_stream;
    }

    /// @brief Return texture from pool
    const Texture* getTexture(std::string tex_id) const;

    /// @brief Add texture to device context
    void addTexture(std::string tex_id, const Texture* texture);

    const Material* getMaterial(std::string mat_id) const;

    void addMaterial(std::string mat_id, const Material* material);

    const std::map<std::string, OptixProgramGroup>& getHitgroupPGs() const; 

private:
    int m_device_id;
    CUstream m_stream = nullptr;
    CUcontext m_cuda_context = nullptr;
    /// the optix context that our pipeline will run in.
    OptixDeviceContext m_optix_context = nullptr;

    // Two option structs control the parameters of the compilation process:
    // - OptixPipelineCompileOptions: Must be identical for all modules
    //      used to create program groups linked in a single pipeline.
    // - OptixModuleCompileOptions: May vary across the modules within
    //      the same pipeline.
    OptixPipelineCompileOptions m_pipeline_compile_options = {};
    OptixPipelineLinkOptions m_pipeline_link_options = {};
    OptixModuleCompileOptions m_module_compile_options = {};

    std::map<std::string, const Texture*> m_textures;
    std::map<std::string, const Material*> m_materials;
    std::map<std::string, OptixProgramGroup> m_hitgroup_pgs;
};

}  // namespace optix