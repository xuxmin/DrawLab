#include "optix/host/device_context.h"
#include "optix/host/sutil.h"
#include <spdlog/spdlog.h>
#include <optix_stack_size.h>

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
    spdlog::log((spdlog::level::level_enum)spd_level, "[{:>14}] {}", tag, message);
}

DeviceContext::DeviceContext(int deviceID)
    : m_device_id(deviceID), m_raygen_pg(nullptr) {
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, m_device_id);

    spdlog::info("[DeviceContext] Create device context on device: {}", prop.name);

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
    OPTIX_CHECK(optixDeviceContextCreate(m_cuda_context, &options, &m_optix_context));
}

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

void DeviceContext::buildRaygenProgramsAndBindSBT(std::string cu_file,
                                                   const char* func) {
    if (m_raygen_pg != nullptr) {
        throw Exception("DeviceContext::createRaygenProgramsAndBindSBT() can "
                        "be called only once!");
    }
    // Create module
    m_raygen_module = createModuleFromCU(cu_file);

    // Create raygen pgs
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = m_raygen_module;
    pgDesc.raygen.entryFunctionName = func;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_optix_context, &pgDesc, 1,
                                            &pgOptions, log, &sizeof_log,
                                            &m_raygen_pg));

    // Bind SBT
    std::vector<RayGenRecord> raygenRecords;
    RayGenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_pg, &rec));
    raygenRecords.push_back(rec);
    m_raygen_record_buffer.allocAndUpload(raygenRecords);
    m_sbt.raygenRecord = m_raygen_record_buffer.devicePtr();
}

void DeviceContext::buildMissProgramsAndBindSBT(
    std::string cu_file, std::vector<const char*> func) {
    if (m_miss_pgs.size() > 0) {
        throw Exception("DeviceContext::createMissProgramsAndBindSBT() can "
                        "be called only once!");
    }

    int ray_type_count = func.size();
    m_miss_pgs.resize(ray_type_count);

    m_miss_module = createModuleFromCU(cu_file);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = m_miss_module;

    for (int ray_id = 0; ray_id < ray_type_count; ray_id++) {
        pgDesc.miss.entryFunctionName = func[ray_id];

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(m_optix_context, &pgDesc, 1,
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

void DeviceContext::buildHitProgramsAndBindSBT(
    std::string cu_file,
    const std::vector<std::pair<int, const char*>> closet_hits,
    const std::vector<std::pair<int, const char*>> any_hits,
    const OptixAccel* accel) {
    int shape_num = accel->getShapeNum();
    int ray_type_count = closet_hits.size();

    m_hitgroup_pgs.resize(ray_type_count);

    m_hit_module = createModuleFromCU(cu_file);

    for (int i = 0; i < closet_hits.size(); i++) {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = m_hit_module;
        pgDesc.hitgroup.moduleAH = m_hit_module;
        pgDesc.hitgroup.entryFunctionNameCH = closet_hits[i].second;
        pgDesc.hitgroup.entryFunctionNameAH = any_hits[i].second;

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(m_optix_context, &pgDesc, 1,
                                                &pgOptions, log, &sizeof_log,
                                                &m_hitgroup_pgs[i]));
    }

    std::vector<HitgroupRecord> hitgroupRecords;
    for (int shape_id = 0; shape_id < shape_num; shape_id++) {
        for (int ray_id = 0; ray_id < ray_type_count; ray_id++) {
            HitgroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroup_pgs[ray_id], &rec));
            accel->packHitgroupRecord(rec, shape_id);
            hitgroupRecords.push_back(rec);
        }
    }

    m_hitgroup_record_buffer.allocAndUpload(hitgroupRecords);
    m_sbt.hitgroupRecordBase = m_hitgroup_record_buffer.devicePtr();
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    m_sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void DeviceContext::buildCallableProgramsAndBindSBT(
    std::vector<std::string> cu_files, std::vector<std::string> func_names) {
    if (m_callable_modules.size() != 0) {
        throw Exception("DeviceContext::createCallableProgramsAndBindSBT() can "
                        "be called only once!");
    }

    int material_num = cu_files.size();
    if (material_num * MATERIAL_CALLABLE_NUM != func_names.size()) {
        throw Exception(
            "The size of func_names doesn't match to the size of cu_files!");
    }

    m_callable_modules.resize(material_num);
    m_callable_pgs.resize(func_names.size());

    for (int mat_id = 0; mat_id < material_num; mat_id++) {
        m_callable_modules[mat_id] = createModuleFromCU(cu_files[mat_id]);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};

        for (int func_id = 0; func_id < MATERIAL_CALLABLE_NUM; func_id++) {
            int idx = mat_id * 3 + func_id;

            pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            pgDesc.callables.moduleDC = m_callable_modules[mat_id];
            pgDesc.callables.entryFunctionNameDC = func_names[idx].c_str();
            char log[2048];
            size_t sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                m_optix_context, &pgDesc,
                1,  // num program groups
                &pgOptions, log, &sizeof_log, &m_callable_pgs[idx]));
        }
    }

    std::vector<CallablesRecord> callableRecords;
    for (int i = 0; i < m_callable_pgs.size(); i++) {
        CallablesRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_callable_pgs[i], &rec));
        callableRecords.push_back(rec);
    }
    m_callable_record_buffer.allocAndUpload(callableRecords);
    m_sbt.callablesRecordBase = m_callable_record_buffer.devicePtr();
    m_sbt.callablesRecordStrideInBytes = sizeof(CallablesRecord);
    m_sbt.callablesRecordCount = (int)callableRecords.size();
}

void DeviceContext::buildPipeline() {
    std::vector<OptixProgramGroup> programGroups;

    // Add raygen programs
    programGroups.push_back(m_raygen_pg);
    // Add miss programs
    for (auto pg : m_miss_pgs) {
        programGroups.push_back(pg);
    }
    // Add hitgroup programs
    for (auto pg : m_hitgroup_pgs) {
        programGroups.push_back(pg);
    }
    // Add callable programs
    for (auto pg : m_callable_pgs) {
        programGroups.push_back(pg);
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_optix_context, &m_pipeline_compile_options, &m_pipeline_link_options,
        programGroups.data(), (int)programGroups.size(), log, &sizeof_log,
        &m_pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : programGroups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }
    const uint32_t max_trace_depth     = 2;
    const uint32_t max_cc_depth        = 1;
    const uint32_t max_dc_depth        = 1;
    const uint32_t max_traversal_depth = 1;

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;

    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, max_trace_depth, max_cc_depth, max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));

    OPTIX_CHECK_LOG(optixPipelineSetStackSize(
        m_pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        max_traversal_depth));
}

void DeviceContext::launch(const ParamBuffer* param_buffer) {
    const CUDABuffer& params_buffer = param_buffer->getParamsBuffer();
    OPTIX_CHECK(optixLaunch(m_pipeline, m_stream, params_buffer.devicePtr(),
                            params_buffer.m_size_in_bytes, &m_sbt,
                            param_buffer->getWidth(), param_buffer->getHeight(),
                            1));
}

DeviceContext::~DeviceContext() {
    // Release pipeline
    OPTIX_CHECK(optixPipelineDestroy(m_pipeline));

    // Release raygen module and program
    OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_pg));
    OPTIX_CHECK(optixModuleDestroy(m_raygen_module));

    // Release miss module and programs
    for (auto miss_pgs : m_miss_pgs) {
        OPTIX_CHECK(optixProgramGroupDestroy(miss_pgs));
    }
    m_miss_pgs.clear();
    OPTIX_CHECK(optixModuleDestroy(m_miss_module));

    // Release callable module and programs
    for (auto call_pgs : m_callable_pgs) {
        OPTIX_CHECK(optixProgramGroupDestroy(call_pgs));
    }
    m_callable_pgs.clear();
    for (auto module : m_callable_modules) {
        OPTIX_CHECK(optixModuleDestroy(module));
    }
    m_callable_modules.clear();

    // Release optix context
    OPTIX_CHECK(optixDeviceContextDestroy(m_optix_context));

    // Release record buffer
    m_raygen_record_buffer.free();
    m_miss_record_buffer.free();
    m_hitgroup_record_buffer.free();
    m_callable_record_buffer.free();
}

}  // namespace optix