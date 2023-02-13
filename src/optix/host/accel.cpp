#include "optix/host/accel.h"
#include <spdlog/spdlog.h>

namespace optix {

OptixAccel::OptixAccel(const OptixDeviceContext& device_context)
    : m_device_context(device_context) {}

OptixAccel::~OptixAccel() {
    for (auto vb : m_vertex_buffers) {
        vb.free();
    }
    for (auto ib : m_index_buffers) {
        ib.free();
    }
    for (auto nb : m_normal_buffers) {
        nb.free();
    }
    for (auto tb : m_texcoord_buffers) {
        tb.free();
    }

    m_as_buffer.free();
}

void OptixAccel::addTriangleMesh(const std::vector<float>& positions,
                                 const std::vector<unsigned int>& indices,
                                 const std::vector<float>& normals,
                                 const std::vector<float>& texcoords,
                                 int light_idx, float pdf) {
    m_meshs.push_back(
        TriangleMesh(positions, indices, normals, texcoords, light_idx, pdf));
}

OptixTraversableHandle OptixAccel::build() {
    std::vector<OptixBuildInput> build_inputs(m_meshs.size());
    std::vector<uint32_t> triangleInputFlags(m_meshs.size());
    std::vector<CUdeviceptr> d_vertices(m_meshs.size());
    std::vector<CUdeviceptr> d_indices(m_meshs.size());

    m_vertex_buffers.resize(m_meshs.size());
    m_index_buffers.resize(m_meshs.size());
    m_normal_buffers.resize(m_meshs.size());
    m_texcoord_buffers.resize(m_meshs.size());

    for (int i = 0; i < m_meshs.size(); i++) {
        if (m_meshs[i].light_idx >= 0) {
            m_emitted_mesh[m_meshs[i].light_idx] = i;
        }

        m_vertex_buffers[i].allocAndUpload(m_meshs[i].positions);
        m_index_buffers[i].allocAndUpload(m_meshs[i].indices);
        if (m_meshs[i].normals.size() > 0) {
            m_normal_buffers[i].allocAndUpload(m_meshs[i].normals);
        }
        if (m_meshs[i].texcoords.size() > 0) {
            m_texcoord_buffers[i].allocAndUpload(m_meshs[i].texcoords);
        }

        d_vertices[i] = m_vertex_buffers[i].devicePtr();
        d_indices[i] = m_index_buffers[i].devicePtr();

        OptixBuildInput& buildInput = build_inputs[i];
        OptixBuildInputTriangleArray& triInput = buildInput.triangleArray;

        // Triangle inputs
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triInput.vertexStrideInBytes = sizeof(float) * 3;
        triInput.numVertices = m_meshs[i].positions.size() / 3;
        triInput.vertexBuffers = &d_vertices[i];

        triInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triInput.vertexStrideInBytes = sizeof(unsigned int) * 3;
        triInput.numIndexTriplets = m_meshs[i].indices.size() / 3;
        triInput.indexBuffer = d_indices[i];

        triInput.preTransform = 0;

        // Each build input maps to one or more consecutive records in the
        // shader binding table(SBT), which controls program dispatch.
        triInput.numSbtRecords = 1;
        triInput.sbtIndexOffsetBuffer = 0;
        triInput.sbtIndexOffsetSizeInBytes = 0;
        triInput.sbtIndexOffsetStrideInBytes = 0;

        triangleInputFlags[i] = 0;

        triInput.flags = &triangleInputFlags[i];
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
        m_device_context, &accelOptions, build_inputs.data(),
        build_inputs.size(), &blasBufferSizes));

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
    OPTIX_CHECK(
        optixAccelBuild(m_device_context, 0, &accelOptions, build_inputs.data(),
                        build_inputs.size(), tempBuffer.devicePtr(),
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

void OptixAccel::packHitgroupRecord(optix::HitgroupRecord& rec,
                                    int mesh_idx) const {
    optix::Shape& geo = rec.data.geometry_data;
    geo.type = optix::Shape::TRIANGLE_MESH;
    geo.triangle_mesh.positions =
        (float3*)m_vertex_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.indices = (int3*)m_index_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.normals = (float3*)m_normal_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.texcoords =
        (float2*)m_texcoord_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.face_num =
        m_index_buffers[mesh_idx].m_size_in_bytes / sizeof(int3);
    geo.triangle_mesh.pdf = m_meshs[mesh_idx].pdf;
    rec.data.light_idx = m_meshs[mesh_idx].light_idx;
}

void OptixAccel::packEmittedMesh(std::vector<Light>& lights) const {
    for (auto em : m_emitted_mesh) {
        const int light_idx = em.first;
        const int mesh_idx = em.second;
        if (lights[light_idx].type != Light::Type::AREA) {
            throw Exception("Error in packEmittedMesh");
        }
        lights[light_idx].area.triangle_mesh.positions =
            (float3*)m_vertex_buffers[mesh_idx].devicePtr();
        lights[light_idx].area.triangle_mesh.indices =
            (int3*)m_index_buffers[mesh_idx].devicePtr();
        lights[light_idx].area.triangle_mesh.normals =
            (float3*)m_normal_buffers[mesh_idx].devicePtr();
        lights[light_idx].area.triangle_mesh.texcoords =
            (float2*)m_texcoord_buffers[mesh_idx].devicePtr();
        lights[light_idx].area.triangle_mesh.face_num =
            m_index_buffers[mesh_idx].m_size_in_bytes / sizeof(int3);
        lights[light_idx].area.triangle_mesh.pdf = m_meshs[mesh_idx].pdf;
    }
}

}  // namespace optix