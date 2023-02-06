#include "optix/host/device_context.h"
#include "optix/host/material.h"
#include "optix/host/sutil.h"
#include "optix/host/launch_param.h"
#include <spdlog/spdlog.h>

namespace optix {

static void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void*) {
    int spd_level = 0;
    if (level == 4) {
        spd_level = 2;  // print -> info
    }
    else if (level == 3) {
        spd_level = 3;  // warning -> warn
    }
    else if (level == 2) {
        spd_level = 4;  // err -> err
    }
    else if (level == 1) {
        spd_level = 5;  // fatal -> critical
    }
    else {
        spd_level = 6;  // disable -> off
    }
    spdlog::log((spdlog::level::level_enum)spd_level, "[{:>14}] {}", tag,
                message);
}

DeviceContext::DeviceContext(int deviceID)
    : m_device_id(deviceID), m_accel(nullptr), m_raygen_pg(nullptr) {
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    spdlog::info("[DeviceContext] Create device context on device: {}",
                 getDeviceName());

    CUresult cuRes = cuCtxGetCurrent(&m_cuda_context);
    if (cuRes != CUDA_SUCCESS)
        spdlog::error(
            "[DEVICE CONTEXT] Error querying current context: error code {}",
            (int)cuRes);

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    OPTIX_CHECK(
        optixDeviceContextCreate(m_cuda_context, &options, &m_optix_context));
}

std::string DeviceContext::getDeviceName() const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, getDeviceId());
    return prop.name;
}

const int DeviceContext::getDeviceId() const { return m_device_id; }

void DeviceContext::configurePipelineOptions() {
    m_module_compile_options.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    m_module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    m_pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    m_pipeline_compile_options.usesMotionBlur = false;
    m_pipeline_compile_options.numPayloadValues = 2;

    // The number of 32-bit words that are reserved to store the attributes.
    // This corresponds to the attribute definition in optixReportIntersection
    m_pipeline_compile_options.numAttributeValues = 2;
    m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

    // The params which are shared with all modules
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    m_pipeline_link_options.maxTraceDepth = 2;
}

OptixModule DeviceContext::createModuleFromCU(std::string cu_file) const {
    OptixModule module;

    size_t ptxSize = 0;
    const char* ptxCode = optix::getInputData(cu_file.c_str(), ptxSize);
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        m_optix_context, &m_module_compile_options, &m_pipeline_compile_options,
        ptxCode, ptxSize, log, &sizeof_log, &module));

    return module;
}

OptixProgramGroup DeviceContext::createHitgroupPrograms(OptixModule ch_module,
                                                        OptixModule ah_module,
                                                        std::string ch_func,
                                                        std::string ah_func) {
    std::string key = std::to_string(int(ch_module)) +
                      std::to_string(int(ah_module)) + ch_func + ah_func;
    if (m_hitgroup_pgs.find(key) != m_hitgroup_pgs.end()) {
        return m_hitgroup_pgs.at(key);
    }

    OptixProgramGroup hitgroupPG;

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = ch_module;
    pgDesc.hitgroup.moduleAH = ah_module;
    pgDesc.hitgroup.entryFunctionNameCH = ch_func.c_str();
    pgDesc.hitgroup.entryFunctionNameAH = ah_func.c_str();

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_optix_context, &pgDesc, 1,
                                            &pgOptions, log, &sizeof_log,
                                            &hitgroupPG));

    m_hitgroup_pgs[key] = hitgroupPG;

    return hitgroupPG;
}

OptixProgramGroup DeviceContext::createHitgroupPrograms(OptixModule module,
                                                        std::string ch_func,
                                                        std::string ah_func) {
    return createHitgroupPrograms(module, module, ch_func, ah_func);
}

void DeviceContext::createRaygenProgramsAndBindSBT(std::string cu_file,
                                                   const char* func) {
    if (m_raygen_pg != nullptr) {
        throw Exception(
            "DeviceContext::createRaygenProgramsAndBindSBT() can be called only once!");
    }
    // Create module
    OptixModule module = createModuleFromCU(cu_file);

    // Create raygen pgs
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = func;

    char log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_optix_context, &pgDesc, 1,
                                            &pgOptions, log, &sizeof_log,
                                            &m_raygen_pg));

    std::vector<RayGenRecord> raygenRecords;
    RayGenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_pg, &rec));
    raygenRecords.push_back(rec);
    m_raygen_record_buffer.allocAndUpload(raygenRecords);
    m_sbt.raygenRecord = m_raygen_record_buffer.devicePtr();
}

void DeviceContext::createMissProgramsAndBindSBT(
    const char* cu_file, std::vector<const char*> func) {
    if (m_miss_pgs.size() > 0) {
        throw Exception("DeviceContext::createMissProgramsAndBindSBT() can "
                        "be called only once!");
    }

    int ray_type_count = func.size();
    m_miss_pgs.resize(ray_type_count);

    OptixModule module = createModuleFromCU(cu_file);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;

    for (int ray_id = 0; ray_id < ray_type_count; ray_id++) {
        pgDesc.miss.entryFunctionName = func[ray_id];

        char log[2048];  // For error reporting from OptiX creation functions
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(m_optix_context, &pgDesc,
                                                1,  // num program groups
                                                &pgOptions, log, &sizeof_log,
                                                &m_miss_pgs[ray_id]));
    }

    std::vector<MissRecord> missRecords;
    for (int i = 0; i < m_miss_pgs.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_pgs[i], &rec));
        missRecords.push_back(rec);
    }
    m_miss_record_buffer.allocAndUpload(missRecords);
    m_sbt.missRecordBase = m_miss_record_buffer.devicePtr();
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = (int)missRecords.size();
}

void DeviceContext::createHitProgramsAndBindSBT(
    int shape_num, int ray_type_num,
    std::function<const Material*(int)> getMaterial) {

    std::vector<HitgroupRecord> hitgroupRecords;
    for (int shape_id = 0; shape_id < shape_num; shape_id++) {
        for (int ray_id = 0; ray_id < ray_type_num; ray_id++) {
            HitgroupRecord rec;
            const Material* mat = getMaterial(shape_id);

            OPTIX_CHECK(
                optixSbtRecordPackHeader(mat->getHitgroupPGs(ray_id), &rec));

            m_accel->packHitgroupRecord(rec, shape_id);
            mat->packHitgroupRecord(rec);

            hitgroupRecords.push_back(rec);
        }
    }

    m_hitgroup_record_buffer.allocAndUpload(hitgroupRecords);
    m_sbt.hitgroupRecordBase = m_hitgroup_record_buffer.devicePtr();
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    m_sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void DeviceContext::createPipeline() {
    std::vector<OptixProgramGroup> programGroups;

    // Add raygen programs
    programGroups.push_back(m_raygen_pg);
    // Add miss programs
    for (auto pg : m_miss_pgs) {
        programGroups.push_back(pg);
    }
    // Add hitgroup programs
    for (auto pg : m_hitgroup_pgs) {
        programGroups.push_back(pg.second);
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_optix_context, &m_pipeline_compile_options, &m_pipeline_link_options,
        programGroups.data(), (int)programGroups.size(), log, &sizeof_log,
        &m_pipeline));

    OPTIX_CHECK_LOG(
        optixPipelineSetStackSize(/* [in] The pipeline to configure the stack
                                     size for */
                                  m_pipeline,
                                  /* [in] The direct stack size requirement for
                                     direct callables invoked from IS or AH. */
                                  2 * 1024,
                                  /* [in] The direct stack size requirement for
                                     direct callables invoked from RG, MS, or
                                     CH.  */
                                  2 * 1024,
                                  /* [in] The continuation stack requirement. */
                                  2 * 1024,
                                  /* [in] The maximum depth of a traversable
                                     graph passed to trace. */
                                  1));
}

const Texture* DeviceContext::getTexture(std::string tex_id) const {
    if (m_textures.find(tex_id) == m_textures.end()) {
        return nullptr;
    }
    return m_textures.at(tex_id);
}

void DeviceContext::addTexture(std::string tex_id, const Texture* texture) {
    if (m_textures.find(tex_id) != m_textures.end()) {
        spdlog::warn("[DEVICE CONTEXT] DeviceContext::addTexture: The tex_id "
                     "{} already in "
                     "pool, override it!",
                     tex_id);
    }
    m_textures[tex_id] = texture;
    spdlog::info("[DEVICE CONTEXT] Add texture {} to DeviceContext", tex_id);
}

const Material* DeviceContext::getMaterial(std::string mat_id) const {
    if (m_materials.find(mat_id) == m_materials.end()) {
        return nullptr;
    }
    return m_materials.at(mat_id);
}

void DeviceContext::addMaterial(std::string mat_id, const Material* material) {
    if (m_materials.find(mat_id) != m_materials.end()) {
        spdlog::warn("[DEVICE CONTEXT] DeviceContext::addMaterial: The mat_id "
                     "{} already in "
                     "pool, override it!",
                     mat_id);
    }
    m_materials[mat_id] = material;
    spdlog::info("[DEVICE CONTEXT] Add material {} to DeviceContext", mat_id);
}

void DeviceContext::createAccel(std::function<void(OptixAccel*)> init) {
    if (m_accel) {
        throw Exception(
            "DeviceContext::createAccel() can only be called once!");
    }
    m_accel = new OptixAccel(m_optix_context);

    init(m_accel);

    m_as_handle = m_accel->build();
}

void DeviceContext::launch(const LaunchParam& launch_params) {
    const CUDABuffer& params_buffer = launch_params.getParamsBuffer();
    OPTIX_CHECK(optixLaunch(m_pipeline, m_stream, params_buffer.devicePtr(),
                            params_buffer.m_size_in_bytes, &m_sbt,
                            launch_params.getWidth(), launch_params.getHeight(),
                            1));
}

}  // namespace optix