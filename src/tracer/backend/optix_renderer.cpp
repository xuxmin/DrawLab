#include "tracer/backend/optix_renderer.h"
#include "core/bitmap/bitmap.h"
#include "optix/common/vec_math.h"
#include "optix/host/sutil.h"
#include "tracer/camera.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include <spdlog/spdlog.h>
#include "stb_image_write.h"

namespace optix {

OptixRenderer::OptixRenderer(drawlab::Scene* scene) : m_scene(scene) {
    initOptix();

    spdlog::info("[OPTIX RENDERER] Creating optix context ...");
    createContext();
    spdlog::info("[OPTIX RENDERER] Setting up module ...");
    module = deviceContext->createModuleFromCU("optix/cuda/device_programs.cu");

    spdlog::info("[OPTIX RENDERER] Creating raygen programs ...");
    createRaygenPrograms();
    spdlog::info("[OPTIX RENDERER] Creating miss programs ...");
    createMissPrograms();

    launchParams.handle = buildAccel();

    spdlog::info("[OPTIX RENDERER] Building SBT ...");
    buildSBT();

    spdlog::info("[OPTIX RENDERER] Setting up optix pipeline ...");
    createPipeline();

    launchParamsBuffer.alloc(sizeof(launchParams));
    spdlog::info(
        "[OPTIX RENDERER] Context, module, pipeline, etc, all set up ...");

    spdlog::info("[OPTIX RENDERER] Optix 7 Sample fully set up");

    updateCamera();
}

void OptixRenderer::initOptix() {
    spdlog::info("[OPTIX RENDERER] Initializing optix...");

    // Initialize CUDA for this device on this thread
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error(
            "[OPTIX RENDERER] No CUDA capable devices found!");
    spdlog::info("[OPTIX RENDERER] Found {} CUDA devices", numDevices);

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());
    spdlog::info("[OPTIX RENDERER] Successfully initialized optix... yay!");
}

void OptixRenderer::createContext() {
    const int deviceID = 0;

    deviceContext = new DeviceContext(deviceID);
    deviceContext->configurePipelineOptions();
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
        deviceContext->getOptixDeviceContext(), &pgDesc, 1, &pgOptions, log,
        &sizeof_log, &raygenPGs[0]));
}

void OptixRenderer::createMissPrograms() {
    missPGs.resize(RAY_TYPE_COUNT);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    char log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        deviceContext->getOptixDeviceContext(), &pgDesc,
        1,  // num program groups
        &pgOptions, log, &sizeof_log, &missPGs[RAY_TYPE_RADIANCE]));

    // NULL miss program for occlusion rays
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__occlusion";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        deviceContext->getOptixDeviceContext(), &pgDesc,
        1,  // num program groups
        &pgOptions, log, &sizeof_log, &missPGs[RAY_TYPE_OCCLUSION]));
}

void OptixRenderer::createPipeline() {

    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);

    // Add hitgroup programs
    for (auto pg : deviceContext->getHitgroupPGs()) {
        programGroups.push_back(pg.second);
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        deviceContext->getOptixDeviceContext(),
        deviceContext->getPipelineCompileOptions(),
        deviceContext->getPipelineLinkOptions(), programGroups.data(),
        (int)programGroups.size(), log, &sizeof_log, &pipeline));

    OPTIX_CHECK_LOG(
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
    std::vector<RayGenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
        RayGenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.allocAndUpload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.devicePtr();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        missRecords.push_back(rec);
    }
    missRecordsBuffer.allocAndUpload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.devicePtr();
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
        for (int ray_id = 0; ray_id < RAY_TYPE_COUNT; ray_id++) {
            HitgroupRecord rec;

            const optix::Material* mat = meshs[i]->getBSDF()->getOptixMaterial(*deviceContext);
            OPTIX_CHECK(optixSbtRecordPackHeader(mat->getHitgroupPGs(ray_id), &rec));

            rec.data.geometry_data.type = GeometryData::TRIANGLE_MESH;
            rec.data.geometry_data.triangle_mesh.positions =
                (float3*)vertexBuffers[i].devicePtr();
            rec.data.geometry_data.triangle_mesh.indices =
                (int3*)indexBuffers[i].devicePtr();
            rec.data.geometry_data.triangle_mesh.normals =
                (float3*)normalBuffer[i].devicePtr();
            rec.data.geometry_data.triangle_mesh.texcoords =
                (float2*)texcoordBuffer[i].devicePtr();

            mat->packHitgroupRecord(rec);

            hitgroupRecords.push_back(rec);
        }
    }
    hitgroupRecordsBuffer.allocAndUpload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.devicePtr();
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
    normalBuffer.resize(mesh_num);
    texcoordBuffer.resize(mesh_num);

    std::vector<OptixBuildInput> buildInputs(mesh_num);
    std::vector<CUdeviceptr> d_vertices(mesh_num);
    std::vector<CUdeviceptr> d_indices(mesh_num);

    for (int i = 0; i < mesh_num; i++) {
        // Upload triangle data to device
        vertexBuffers[i].allocAndUpload(meshs[i]->getVertexPosition());
        indexBuffers[i].allocAndUpload(meshs[i]->getVertexIndex());
        if (meshs[i]->hasVertexNormal())
            normalBuffer[i].allocAndUpload(meshs[i]->getVertexNormal());
        if (meshs[i]->hasTexCoord())
            texcoordBuffer[i].allocAndUpload(meshs[i]->getVertexTexCoord());

        // Get the pointer to the device
        d_vertices[i] = vertexBuffers[i].devicePtr();
        d_indices[i] = indexBuffers[i].devicePtr();

        // Triangle inputs
        OptixBuildInputTriangleArray& buildInput = buildInputs[i].triangleArray;


        /**
         * Different build type
         * 
         * instance acceleration structures:
         *  OPTIX_BUILD_INPUT_TYPE_INSTANCES
         *  OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS
         * 
         * A geometry acceleration structure containing built-in triangles
         *  OPTIX_BUILD_INPUT_TYPE_TRIANGLES
         * 
         * A geometry acceleration structure containing built-in curve primitives
         *  OPTIX_BUILD_INPUT_TYPE_CURVES
         * 
         * A geometry acceleration structure containing custom primitives
         *  OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES
         * 
         * 
         * Instance acceleration structures have a single build input and 
         * specify an array of instances. Each instance includes a ray 
         * transformation and an OptixTraversableHandle that refers to a 
         * geometry-AS, a transform node, or another instance acceleration 
         * structure.
        */
        buildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        buildInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.vertexStrideInBytes = sizeof(float) * 3;
        buildInput.numVertices = meshs[i]->getVertexCount();
        buildInput.vertexBuffers = &d_vertices[i];

        buildInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.vertexStrideInBytes = sizeof(unsigned int) * 3;
        buildInput.numIndexTriplets = meshs[i]->getTriangleCount();
        buildInput.indexBuffer = d_indices[i];

        // Support a 3x4 transform matrix to transfrom the vertices at build
        // time.
        buildInput.preTransform = 0;

        // Each build input maps to one or more consecutive records in the
        // shader binding table(SBT), which controls program dispatch.
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
        deviceContext->getOptixDeviceContext(), &accelOptions,
        buildInputs.data(), mesh_num, &blasBufferSizes));

    // ------------------------------------------------------------
    // prepare compaction
    // ------------------------------------------------------------

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(unsigned long long));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.devicePtr();

    // ------------------------------------------------------------
    // Build (main stage)
    // ------------------------------------------------------------

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OptixTraversableHandle outputHandle = 0;
    OPTIX_CHECK(optixAccelBuild(
        deviceContext->getOptixDeviceContext(), 0, &accelOptions,
        &buildInputs[0], mesh_num, tempBuffer.devicePtr(),
        tempBuffer.m_size_in_bytes, outputBuffer.devicePtr(),
        outputBuffer.m_size_in_bytes, &outputHandle, &emitDesc, 1));
    CUDA_SYNC_CHECK();

    // ------------------------------------------------------------
    // perform compaction
    // ------------------------------------------------------------
    unsigned long long compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(deviceContext->getOptixDeviceContext(),
                                  /*stream:*/ 0, outputHandle,
                                  asBuffer.devicePtr(),
                                  asBuffer.m_size_in_bytes, &outputHandle));
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
                            pipeline, deviceContext->getStream(),
                            /*! parameters and SBT */
                            launchParamsBuffer.devicePtr(),
                            launchParamsBuffer.m_size_in_bytes, &sbt,
                            /*! dimensions of the launch: */
                            launchParams.width, launchParams.height, 1));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();

    // write image
    std::vector<unsigned int> pixels(m_width * m_height);
    downloadPixels(pixels.data());
    const std::string fileName = "osc_example2.png";
    stbi_write_png(fileName.c_str(), m_width, m_height, 4, pixels.data(),
                            m_width * sizeof(unsigned int));
    spdlog::info("Image rendered, and saved to {} ... done.", fileName);
}

void OptixRenderer::updateCamera() {

    const drawlab::Camera* camera = m_scene->getCamera();
    drawlab::Vector2i outputSize = camera->getOutputSize();
    resize(outputSize[1], outputSize[0]);

    camera->packLaunchParameters(launchParams);
}

/*! resize frame buffer to given resolution */
void OptixRenderer::resize(const int height, const int width) {
    // if window minimized
    if (height == 0 | width == 0)
        return;

    m_width = width;
    m_height = height;

    // resize our cuda frame buffer
    colorBuffer.resize(height * width * sizeof(unsigned int));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.width = width;
    launchParams.height = height;
    launchParams.color_buffer = (unsigned int*)colorBuffer.m_device_ptr;
}

void OptixRenderer::downloadPixels(unsigned int h_pixels[]) {
    colorBuffer.download(h_pixels, launchParams.height * launchParams.width);
}

}  // namespace optix