#pragma once
#include "tracer/scene.h"
#include "optix/host/texture.h"
#include "optix/host/device_context.h"
#include "optix/host/cuda_buffer.h"
#include "optix/common/optix_params.h"


namespace optix {

/**
 * A simple Opitx-7 renderer that demonstrates how to set up
 * context, module, programs, pipeline, SBT, etc, and perform
 * a valid launch that renders some pixel.
 *
 */
class OptixRenderer {
public:
    /**
     * Performs all setup, including initializing optix,
     * creates module, pipeline, programs, SBT, etc.
     */
    OptixRenderer(drawlab::Scene* scene);

    /// Render one frame.
    void render();

    /// Resize framebuffer to a given size
    void resize(const int height, const int width);

    /// Download the rendered color buffer
    void downloadPixels(unsigned int h_pixels[]);

    void updateCamera();

protected:
    /// Initialize optix and check for errors.
    void initOptix();

    /// @brief Creates and configures a optix device context
    void createContext();

    /**
     * Programs, a block of executable code on the GPU, called shader in DXR
     * and Vulkan.
     * There are nine types of user-defined ray interactions: Ray generation,
     * Intersection, Any-hit, Closest-hit, Miss, Exception, Direct callable,
     * Continuation callable
    */
    void createRaygenPrograms();
    void createMissPrograms();

    void createPipeline();

    /// @brief Constructs the shader binding table
    void buildSBT();

    // Build an acceleration structure for the scene
    OptixTraversableHandle buildAccel();

protected:

    DeviceContext* deviceContext;

    /* The module that contains out device programs */

    /**
     * Programs are first compiled into modules
     * 
     *   .cu --nvcc--> .ptx ---> module(optixModule)
    */
    OptixModule module;

    /**
     * Modules are combined to create program groups
     * 
     *  modules ---> program groups(OptixProgramGroup)
     * 
     * OptixProgramGroup objects are created from one to three OptixModule
     * objects and are used to fill the header of the SBT records.
    */
    std::vector<OptixProgramGroup> raygenPGs;
    std::vector<OptixProgramGroup> missPGs;

    CUDABuffer raygenRecordsBuffer;
    CUDABuffer missRecordsBuffer;
    CUDABuffer hitgroupRecordsBuffer;

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
     * The SBT record index sbtIndex is determined by the following index calculation during traversal:
     * sbt-index = sbt-instance-offset
     *              + (sbt-geometry-acceleration-structure-index * sbt-stride-from-trace-call)
     *              + sbt-offset-from-trace-call
     * 
     * sbt-instance-offset:                         0 if only one gas
     * 
     * sbt-geometry-acceleration-structure-index:   buildInput index if numSBTRecords=1
     * 
     * sbt-stride-from-trace-call: The parameter SBTstride, defined as an index offset,
     * is multiplied by optixTrace with the SBT geometry acceleration structure index.
     * It is required to implement different ray types.
     * 
     * sbt-offset-from-trace-call: The optixTrace function takes the parameter SBToffset, 
     * allowing for an SBT access shift for this specific ray. It is required to implement
     * different ray types.
     * 
    */
    OptixShaderBindingTable sbt = {};

    /**
     * Linking the programs to a pipeline
     * 
    */
    OptixPipeline pipeline;


    // OptixPipelineCompileOptions pipelineCompileOptions = {};
    // OptixPipelineLinkOptions pipelineLinkOptions = {};

    /* our launch parameters, on the host, and the buffer to store
       them on the device */
    LaunchParams launchParams;
    CUDABuffer launchParamsBuffer;

    CUDABuffer colorBuffer;

    drawlab::Scene* m_scene;
    std::vector<CUDABuffer> vertexBuffers;
    std::vector<CUDABuffer> indexBuffers;
    std::vector<CUDABuffer> normalBuffer;
    std::vector<CUDABuffer> texcoordBuffer;
    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer;

    std::vector<Texture*> textures;
};

};  // namespace optix