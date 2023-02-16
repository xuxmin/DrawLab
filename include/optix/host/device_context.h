#pragma once

#include "optix/optix_params.h"
#include "optix/host/cuda_buffer.h"
#include "optix/host/sutil.h"
#include "optix/host/optix_accel.h"
#include "optix/host/param_buffer.h"
#include <cuda_runtime.h>
#include <string>

namespace optix {


class DeviceContext {
public:
    DeviceContext(int deviceID);

    ~DeviceContext();

    // Configures the optixPipeline link options and compile options,
    void configurePipelineOptions();

    void buildRaygenProgramsAndBindSBT(std::string cu_file, const char* func);

    /**
     * \param func Store the different miss functions for different rays.
     * The sequence should be based on the ray type definition in optix_params.h
     */
    void buildMissProgramsAndBindSBT(std::string cu_file,
                                      std::vector<const char*> func);

    void buildHitProgramsAndBindSBT(
        std::string cu_file,
        const std::vector<std::pair<int, const char*>> closet_hits,
        const std::vector<std::pair<int, const char*>> any_hits,
        const OptixAccel* accel);

    // Each cu_file corresponds to a material, and has three funcs: eval, pdf, sample
    void buildCallableProgramsAndBindSBT(std::vector<std::string> cu_files, std::vector<std::string> func_names);

    void buildPipeline();

    const OptixDeviceContext& getOptixDeviceContext() const { return m_optix_context; }

    void launch(const ParamBuffer* param_buffer);

private:

    // Load *.cu file and create OptixModule object
    OptixModule createModuleFromCU(std::string cu_file) const;

private:
    int m_device_id;

    CUstream                        m_stream        = nullptr;
    CUcontext                       m_cuda_context  = nullptr;
    OptixDeviceContext              m_optix_context = nullptr;

    OptixPipelineCompileOptions     m_pipeline_compile_options = {};
    OptixPipelineLinkOptions        m_pipeline_link_options = {};
    OptixModuleCompileOptions       m_module_compile_options = {};
    OptixPipeline                   m_pipeline;
    OptixShaderBindingTable         m_sbt = {};

    // raygen
    OptixModule                     m_raygen_module;
    OptixProgramGroup               m_raygen_pg;
    CUDABuffer                      m_raygen_record_buffer;

    // hitgroup
    OptixModule                     m_hit_module;
    std::vector<OptixProgramGroup>  m_hitgroup_pgs;
    CUDABuffer                      m_hitgroup_record_buffer;

    // miss
    OptixModule                     m_miss_module;
    std::vector<OptixProgramGroup>  m_miss_pgs;
    CUDABuffer                      m_miss_record_buffer;

    // callable
    std::vector<OptixModule>        m_callable_modules;
    std::vector<OptixProgramGroup>  m_callable_pgs;
    CUDABuffer                      m_callable_record_buffer;
};

}  // namespace optix