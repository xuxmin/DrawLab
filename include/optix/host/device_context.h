#pragma once

#include "optix/common/optix_params.h"
#include "optix/host/cuda_buffer.h"
#include "optix/host/sutil.h"
#include "optix/host/texture.h"
#include "optix/host/accel.h"
#include "optix/host/resource_manager.h"
#include <cuda_runtime.h>
#include <map>
#include <functional>
#include <string>

namespace optix {

class Material;
class LaunchParam;


class DeviceContext {
public:
    DeviceContext(int deviceID);

    void destroy();

    ~DeviceContext();

    /// @brief  configures the optixPipeline link options and compile options,
    void configurePipelineOptions();

    std::string getDeviceName() const;

    const int getDeviceId() const;

    void createRaygenProgramsAndBindSBT(std::string cu_file, const char* func);

    void createMissProgramsAndBindSBT(const char* cu_file, std::vector<const char*> func);

    void createHitProgramsAndBindSBT(
        std::string cu_file,
        const std::vector<std::pair<int, const char*>> closet_hits,
        const std::vector<std::pair<int, const char*>> any_hits);

    // Each cu_file corresponds to a material, and has three funcs: eval, pdf, sample
    void createCallableProgramsAndBindSBT(std::vector<std::string> cu_files, std::vector<std::string> func_names);

    void createPipeline();

    OptixModule createModuleFromCU(std::string cu_file) const;

    OptixProgramGroup createHitgroupPrograms(OptixModule ch_module,
                                             OptixModule ah_module,
                                             std::string ch_func,
                                             std::string ah_func);

    OptixProgramGroup createHitgroupPrograms(OptixModule module,
                                             std::string ch_func,
                                             std::string ah_func);

    void launch(const LaunchParam& launch_params);

    const CUstream& getStream() const { return m_stream; }

    void addTexture(const Texture* texture);

    const OptixPipeline getPipeline() const { return m_pipeline; }

    const OptixAccel* getAccel() {return m_accel; }

    const OptixTraversableHandle& getHandle() const {return m_as_handle; }

    void createAccel(std::function<void(OptixAccel*)> init);

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

    OptixAccel* m_accel;
    OptixTraversableHandle m_as_handle;

    std::vector<const Texture*> m_textures;

    OptixModule m_raygen_module;
    OptixModule m_miss_module;
    OptixModule m_hit_module;
    std::vector<OptixModule> m_callable_modules;
    OptixProgramGroup m_raygen_pg;
    std::vector<OptixProgramGroup> m_miss_pgs;
    std::vector<OptixProgramGroup> m_hitgroup_pgs;
    std::vector<OptixProgramGroup> m_callable_pgs;

    OptixPipeline m_pipeline;

    CUDABuffer m_raygen_record_buffer;
    CUDABuffer m_miss_record_buffer;
    CUDABuffer m_hitgroup_record_buffer;
    CUDABuffer m_callable_record_buffer;
    /**
     * The shader binding table (SBT) is an array that contains information
     * about the location of programs and their parameters
     *
     * More details about sbt:
     * 1. Only one raygenRecord and exceptionRecord
     * 2. Arrays of SBT records for miss programs.
     *      In optixTrace(), use missSBTIndex parameter to select miss programs.
     * 3. Arrays of SBT records for hit groups.
     *      The computation of the index for the hit group (intersection,
     *      any-hit, closest-hit) is done during traversal.
     *
     * The SBT record index sbtIndex is determined by the following index
     * calculation during traversal: sbt-index = sbt-instance-offset
     *              + (sbt-geometry-acceleration-structure-index *
     * sbt-stride-from-trace-call)
     *              + sbt-offset-from-trace-call
     *
     * sbt-instance-offset:                         0 if only one gas
     *
     * sbt-geometry-acceleration-structure-index:   buildInput index if
     * numSBTRecords=1
     *
     * sbt-stride-from-trace-call: The parameter SBTstride, defined as an index
     * offset, is multiplied by optixTrace with the SBT geometry acceleration
     * structure index. It is required to implement different ray types.
     *
     * sbt-offset-from-trace-call: The optixTrace function takes the parameter
     * SBToffset, allowing for an SBT access shift for this specific ray. It is
     * required to implement different ray types.
     *
     */
    OptixShaderBindingTable m_sbt = {};
};

}  // namespace optix