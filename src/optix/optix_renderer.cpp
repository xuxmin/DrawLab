#include "optix/optix_renderer.h"
#include "optix/sutil.h"
#include "optix/vec_math.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

namespace optix {

/**
 * The SBT(shader binding table) connects geometric data to programs
 * 
 * header: Opaque to the application, filled in by optixSbtRecordPackHeader.
 *      uased by Optix 7 to identify different behaviour, such as any-hit,
 *      intersection...
 * 
 * data: Opaque to NVIDIA OptiX 7. can store program parameter values.
*/
/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    int objectID;
};

OptixRenderer::OptixRenderer(drawlab::Scene* scene): m_scene(scene) {
    initOptix();

    std::cout << "#optix: creating optix context ..." << std::endl;
    createContext();
    std::cout << "#optix: setting up module ..." << std::endl;
    createModule();

    std::cout << "#optix: creating raygen programs ..." << std::endl;
    createRaygenPrograms();
    std::cout << "#optix: creating miss programs ..." << std::endl;
    createMissPrograms();
    std::cout << "#optix: creating hitgroup programs ..." << std::endl;
    createHitgroupPrograms();

    launchParams.handle = buildAccel();

    std::cout << "#optix: setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << "#optix: building SBT ..." << std::endl;
    buildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << "#optix: context, module, pipeline, etc, all set up ..."
              << std::endl;

    std::cout << TERMINAL_GREEN;
    std::cout << "#optix: Optix 7 Sample fully set up" << std::endl;
    std::cout << TERMINAL_DEFAULT;
}

void OptixRenderer::initOptix() {
    std::cout << "#optix: initializing optix..." << std::endl;

    // Initialize CUDA for this device on this thread
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("#optix: no CUDA capable devices found!");
    std::cout << "#optix: found " << numDevices << " CUDA devices" << std::endl;

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());
    std::cout << TERMINAL_GREEN
              << "#optix: successfully initialized optix... yay!"
              << TERMINAL_DEFAULT << std::endl;
}

static void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void*) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void OptixRenderer::createContext() {
    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#optix: running on device: " << deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
        fprintf( stderr, "Error querying current context: error code %d\n", cuRes );

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
}

void OptixRenderer::createModule() {

    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;

    // The number of 32-bit words that are reserved to store the attributes.
    // This corresponds to the attribute definition in optixReportIntersection
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

    // The params which are shared with all modules
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    pipelineLinkOptions.maxTraceDepth = 2;

    size_t ptxSize = 0;
    const char* ptxCode = optix::getInputData("helloOptix", "optix/cuda", "device_programs.cu", ptxSize);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode, ptxSize,
        log, &sizeof_log,
        &module));
}

void OptixRenderer::createRaygenPrograms() {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    char log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &raygenPGs[0]));
}

void OptixRenderer::createMissPrograms() {
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    char log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(optixContext, &pgDesc,
                                            1,  // num program groups
                                            &pgOptions, log, &sizeof_log,
                                            &missPGs[0]));
}

void OptixRenderer::createHitgroupPrograms() {
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(optixContext, &pgDesc,
                                            1,  // num program groups
                                            &pgOptions, log, &sizeof_log,
                                            &hitgroupPGs[0]));
}

void OptixRenderer::createPipeline() {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        optixContext, &pipelineCompileOptions, &pipelineLinkOptions,
        programGroups.data(), (int)programGroups.size(), log, &sizeof_log,
        &pipeline));

    OPTIX_CHECK(
        optixPipelineSetStackSize(/* [in] The pipeline to configure the stack
                                     size for */
                                  pipeline,
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

void OptixRenderer::buildSBT() {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.allocAndUpload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordsBuffer.allocAndUpload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    // we don't actually have any objects in this example, but let's
    // create a dummy one so the SBT doesn't have any null pointers
    // (which the sanity checks in compilation would complain about)
    const std::vector<drawlab::Mesh*> meshs = m_scene->getMeshes();
    int numObjects = (int)meshs.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int i = 0; i < numObjects; i++) {
        int objectType = 0;
        HitgroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
        rec.objectID = i;
        hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.allocAndUpload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

OptixTraversableHandle OptixRenderer::buildAccel() {

    // ------------------------------------------------------------
    // Primitive build inputs
    // ------------------------------------------------------------
    /**
     * A triangle build input references an array of triangle vertex
     * buffers in device memory, one buffer per motion key (a single
     * triangle vertex buffer if there is no motion)
     * Optionally, triangles can be indexed using an index buffer in
     * device memory.
     */

    const std::vector<drawlab::Mesh*> meshs = m_scene->getMeshes();
    int mesh_num = meshs.size();
    vertexBuffers.resize(mesh_num);
    indexBuffers.resize(mesh_num);

    std::vector<OptixBuildInput> buildInputs(mesh_num);
    std::vector<CUdeviceptr> d_vertices(mesh_num);
	std::vector<CUdeviceptr> d_indices(mesh_num);

    for (int i = 0; i < mesh_num; i++) {

        // Upload triangle data to device
        vertexBuffers[i].allocAndUpload(meshs[i]->getVertexPosition());
        indexBuffers[i].allocAndUpload(meshs[i]->getVertexIndex());
    
        // Get the pointer to the device
        d_vertices[i] = vertexBuffers[i].d_pointer();
        d_indices[i]  = indexBuffers[i].d_pointer();

        // Triangle inputs
        OptixBuildInputTriangleArray& buildInput = buildInputs[i].triangleArray;
        buildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        buildInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.vertexStrideInBytes = sizeof(float) * 3;
        buildInput.numVertices = meshs[i]->getVertexCount();
        buildInput.vertexBuffers = &d_vertices[i];

        buildInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.vertexStrideInBytes = sizeof(unsigned int) * 3;
        buildInput.numIndexTriplets = meshs[i]->getTriangleCount();
        buildInput.indexBuffer = d_indices[i];

        // Support a 3x4 transform matrix to transfrom the vertices at build time.
        buildInput.preTransform = 0;

        // Each build input maps to one or more consecutive records in the 
        // shader binding table(SBT)
        buildInput.numSbtRecords = 1;
        buildInput.sbtIndexOffsetBuffer = 0;
        buildInput.sbtIndexOffsetSizeInBytes = 0;
        buildInput.sbtIndexOffsetStrideInBytes = 0;

        uint32_t flagsPerSBTRecord[1];
        flagsPerSBTRecord[0] = OPTIX_GEOMETRY_FLAG_NONE;
        buildInput.flags = flagsPerSBTRecord;
    }

    // ------------------------------------------------------------
    // BLAS setup
    // ------------------------------------------------------------

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    // A numKeys value of zero specifies no motion blu
    accelOptions.motionOptions.numKeys = 0;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optixContext, &accelOptions, buildInputs.data(), mesh_num, &blasBufferSizes));

    // ------------------------------------------------------------
    // prepare compaction
    // ------------------------------------------------------------
    
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(unsigned long long));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ------------------------------------------------------------
    // Build (main stage)
    // ------------------------------------------------------------

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OptixTraversableHandle outputHandle = 0;
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                0,
                                &accelOptions,
                                &buildInputs[0],
                                mesh_num,
                                tempBuffer.d_pointer(),
                                tempBuffer.m_size_in_bytes,
                                outputBuffer.d_pointer(),
                                outputBuffer.m_size_in_bytes,
                                &outputHandle,
                                &emitDesc,
                                1));
    CUDA_SYNC_CHECK();

    // ------------------------------------------------------------
    // perform compaction
    // ------------------------------------------------------------
    unsigned long long compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);
    
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/0,
                                  outputHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.m_size_in_bytes,
                                  &outputHandle));
    CUDA_SYNC_CHECK();

    // ------------------------------------------------------------
    // Finally clean up!
    // ------------------------------------------------------------
    outputBuffer.free();
    tempBuffer.free();
    compactedSizeBuffer.free();

    return outputHandle;
}

void OptixRenderer::render() {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.width == 0)
        return;

    launchParamsBuffer.upload(&launchParams, 1);
    launchParams.frameID++;

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipeline, stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.m_size_in_bytes, &sbt,
                            /*! dimensions of the launch: */
                            launchParams.width, launchParams.height,
                            1));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
}

void OptixRenderer::updateCamera() {
    
	const float aspect_ratio = launchParams.width / (float)launchParams.height;
    const float fov = 15;

    // const Camera* camera = this->m_scene->getCamera();
    float3 from = make_float3(0, 6, 27.5);
    float3 at = make_float3(0, -1.5, 2.5);
    float3 up = make_float3(0, 1, 0);

    float ulen, vlen, wlen;
    float3 W = at - from;       // Do not normalize W -- it implies focal length

    wlen = length(W);
    float3 U = normalize(cross(up, W));
    float3 V = normalize(cross(U, W));

    U = -U;

    vlen = wlen * tanf( 0.5f * fov * M_PIf / 180.0f );
    V *= vlen;
    ulen = vlen * aspect_ratio;
    U *= ulen;

    launchParams.eye = from;
    launchParams.U = U;
    launchParams.V = V;
    launchParams.W = W;
}

/*! resize frame buffer to given resolution */
void OptixRenderer::resize(const int height, const int width) {
    // if window minimized
    if (height == 0 | width == 0)
        return;

    // resize our cuda frame buffer
    colorBuffer.resize(height * width * sizeof(unsigned int));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.width = width;
    launchParams.height = height;
    launchParams.color_buffer = (unsigned int*)colorBuffer.m_device_ptr;
}

void OptixRenderer::downloadPixels(unsigned int h_pixels[]) {
    colorBuffer.download(h_pixels,
                         launchParams.height * launchParams.width);
}

}  // namespace optix