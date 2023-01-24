#pragma once
#include "optix/cuda_buffer.h"
#include "optix/params.h"
#include "tracer/scene.h"


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
     * \brief Creates the module from the .ptx file.
     * 
     * A module may include multiple programs of any program type
    */
    void createModule();

    /**
     * Programs, a block of executable code on the GPU, called shader in DXR
     * and Vulkan.
     * There are nine types of user-defined ray interactions: Ray generation,
     * Intersection, Any-hit, Closest-hit, Miss, Exception, Direct callable,
     * Continuation callable
    */
    void createRaygenPrograms();
    void createMissPrograms();
    void createHitgroupPrograms();

    void createPipeline();

    /// @brief Constructs the shader binding table
    void buildSBT();

    // Build an acceleration structure for the scene
    OptixTraversableHandle buildAccel();

protected:
    CUcontext cudaContext;
    CUstream stream;
    cudaDeviceProp deviceProps;

    /// the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /* The module that contains out device programs */

    /**
     * Programs are first compiled into modules
     * 
     *   .cu --nvcc--> .ptx ---> module(optixModule)
    */
    OptixModule module;
    OptixModuleCompileOptions moduleCompileOptions = {};

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
    std::vector<OptixProgramGroup> hitgroupPGs;

    CUDABuffer raygenRecordsBuffer;
    CUDABuffer missRecordsBuffer;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    /**
     * Linking the programs to a pipeline
     * 
    */
    OptixPipeline pipeline;

    // Two option structs control the parameters of the compilation process:
    // - OptixPipelineCompileOptions: Must be identical for all modules 
    //      used to create program groups linked in a single pipeline.
    // - OptixModuleCompileOptions: May vary across the modules within 
    //      the same pipeline.
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};

    /* our launch parameters, on the host, and the buffer to store
       them on the device */
    LaunchParams launchParams;
    CUDABuffer launchParamsBuffer;

    CUDABuffer colorBuffer;

    drawlab::Scene* m_scene;
    std::vector<CUDABuffer> vertexBuffers;
    std::vector<CUDABuffer> indexBuffers;
    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer;
};

};  // namespace optix