#include "optix/host/device_context.h"
#include "optix/host/sutil.h"
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

DeviceContext::DeviceContext(int deviceID) : m_device_id(deviceID) {
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

OptixProgramGroup DeviceContext::createHitgroupPrograms(
    OptixModule ch_module, OptixModule ah_module, std::string ch_func,
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

OptixProgramGroup
DeviceContext::createHitgroupPrograms(OptixModule module, std::string ch_func,
                                      std::string ah_func) {
    return createHitgroupPrograms(module, module, ch_func, ah_func);
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

const std::map<std::string, OptixProgramGroup>& DeviceContext::getHitgroupPGs() const {
    return m_hitgroup_pgs;
}


}  // namespace optix