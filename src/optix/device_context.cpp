#include "optix/device_context.h"
#include "optix/sutil.h"
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
        spdlog::error("Error querying current context: error code {}",
                      (int)cuRes);

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    OPTIX_CHECK(optixDeviceContextCreate(m_cuda_context, &options, &m_optix_context));
}

std::string DeviceContext::getDeviceName() const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, getDeviceId());
    return prop.name;
}

const int DeviceContext::getDeviceId() const { return m_device_id; }


void DeviceContext::configurePipelineOptions() {
    m_module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    m_module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    
    m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
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

}  // namespace optix