#include "tracer/optix_accel.h"

namespace drawlab {

OptixAccel::OptixAccel(const OptixDeviceContext& device_context, const std::vector<Mesh*> mesh_ptrs)
    : m_device_context(device_context), m_meshPtrs(mesh_ptrs) {}

OptixAccel::~OptixAccel() {}

OptixTraversableHandle OptixAccel::build() {
    /**
     * A triangle build input references an array of triangle vertex
     * buffers in device memory, one buffer per motion key (a single
     * triangle vertex buffer if there is no motion)
     * Optionally, triangles can be indexed using an index buffer in
     * device memory.
     */
    int mesh_num = m_meshPtrs.size();

    m_vertex_buffers.resize(mesh_num);
    m_index_buffers.resize(mesh_num);
    m_normal_buffers.resize(mesh_num);
    m_texcoord_buffers.resize(mesh_num);

    std::vector<OptixBuildInput> buildInputs(mesh_num);
    std::vector<CUdeviceptr> d_vertices(mesh_num);
    std::vector<CUdeviceptr> d_indices(mesh_num);

    for (int i = 0; i < mesh_num; i++) {
        // Upload triangle data to device
        m_vertex_buffers[i].allocAndUpload(m_meshPtrs[i]->getVertexPosition());
        m_index_buffers[i].allocAndUpload(m_meshPtrs[i]->getVertexIndex());
        if (m_meshPtrs[i]->hasVertexNormal())
            m_normal_buffers[i].allocAndUpload(
                m_meshPtrs[i]->getVertexNormal());
        if (m_meshPtrs[i]->hasTexCoord())
            m_texcoord_buffers[i].allocAndUpload(
                m_meshPtrs[i]->getVertexTexCoord());

        // Get the pointer to the device
        d_vertices[i] = m_vertex_buffers[i].devicePtr();
        d_indices[i] = m_index_buffers[i].devicePtr();

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
         * A geometry acceleration structure containing built-in curve
         * primitives OPTIX_BUILD_INPUT_TYPE_CURVES
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
        buildInput.numVertices = m_meshPtrs[i]->getVertexCount();
        buildInput.vertexBuffers = &d_vertices[i];

        buildInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.vertexStrideInBytes = sizeof(unsigned int) * 3;
        buildInput.numIndexTriplets = m_meshPtrs[i]->getTriangleCount();
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
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_device_context, &accelOptions,
                                             buildInputs.data(), mesh_num,
                                             &blasBufferSizes));

    // ------------------------------------------------------------
    // prepare compaction
    // ------------------------------------------------------------

    optix::CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(unsigned long long));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.devicePtr();


    // ------------------------------------------------------------
    // Build (main stage)
    // ------------------------------------------------------------
    optix::CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    optix::CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    m_handle = 0;
    OPTIX_CHECK(optixAccelBuild(
        m_device_context, 0, &accelOptions,
        &buildInputs[0], mesh_num, tempBuffer.devicePtr(),
        tempBuffer.m_size_in_bytes, outputBuffer.devicePtr(),
        outputBuffer.m_size_in_bytes, &m_handle, &emitDesc, 1));
    CUDA_SYNC_CHECK();


    // ------------------------------------------------------------
    // perform compaction
    // ------------------------------------------------------------
    unsigned long long compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    m_as_buffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(m_device_context,
                                  /*stream:*/ 0, m_handle,
                                  m_as_buffer.devicePtr(),
                                  m_as_buffer.m_size_in_bytes, &m_handle));
    CUDA_SYNC_CHECK();

    // ------------------------------------------------------------
    // Finally clean up!
    // ------------------------------------------------------------
    outputBuffer.free();
    tempBuffer.free();
    compactedSizeBuffer.free();

    return m_handle;
}


void OptixAccel::packHitgroupRecord(optix::HitgroupRecord& rec, int mesh_idx) const {
    
    optix::GeometryData& geo = rec.data.geometry_data;
    geo.type = optix::GeometryData::TRIANGLE_MESH;
    geo.triangle_mesh.positions = (float3*)m_vertex_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.indices = (int3*)m_index_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.normals = (float3*)m_normal_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.texcoords = (float2*)m_texcoord_buffers[mesh_idx].devicePtr();
}

}  // namespace drawlab