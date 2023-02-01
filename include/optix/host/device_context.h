#pragma once

#include "optix/host/sutil.h"
#include <cuda_runtime.h>
#include <string>

namespace optix {

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
};

}  // namespace optix