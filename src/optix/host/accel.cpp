#include "optix/host/accel.h"
#include <spdlog/spdlog.h>

namespace optix {

OptixAccel::OptixAccel(const OptixDeviceContext& device_context)
    : m_device_context(device_context) {
    m_triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
}

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
    size_t num = m_build_inputs.size();

    m_light_idx.resize(num + 1);
    m_light_idx[num] = light_idx;
    if (light_idx >= 0) {
        m_emitted_mesh[light_idx] = num;
    }

    m_pdf.resize(num + 1);
    m_pdf[num] = pdf;

    m_vertex_buffers.resize(num + 1);
    m_index_buffers.resize(num + 1);
    m_normal_buffers.resize(num + 1);
    m_texcoord_buffers.resize(num + 1);
    d_vertices.resize(num + 1);
    d_indices.resize(num + 1);
    m_build_inputs.resize(num + 1);

    CUDABuffer& vertex_buffer = m_vertex_buffers[num];
    CUDABuffer& index_buffer = m_index_buffers[num];
    CUDABuffer& normal_buffer = m_normal_buffers[num];
    CUDABuffer& texcoord_buffer = m_texcoord_buffers[num];

    // pad positions
    // std::vector<float> pad_positions(positions.size() / 3 * 4);
    // for (int i = 0; i < positions.size() / 3; i++) {
    //     pad_positions[i * 4] = positions[i * 3];
    //     pad_positions[i * 4 + 1] = positions[i * 3 + 1];
    //     pad_positions[i * 4 + 2] = positions[i * 3 + 2];
    //     pad_positions[i * 4 + 3] = 0;
    // }

    vertex_buffer.allocAndUpload(positions);
    index_buffer.allocAndUpload(indices);
    if (normals.size() > 0) {
        normal_buffer.allocAndUpload(normals);
    }
    if (texcoords.size() > 0) {
        texcoord_buffer.allocAndUpload(texcoords);
    }

    // Get the pointer to the device
    CUdeviceptr& d_vertice = d_vertices[num];
    CUdeviceptr& d_indice = d_indices[num];

    d_vertice = vertex_buffer.devicePtr();
    d_indice = index_buffer.devicePtr();

    OptixBuildInput& buildInput = m_build_inputs[num];
    OptixBuildInputTriangleArray& triInput = buildInput.triangleArray;

    // Triangle inputs
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triInput.vertexStrideInBytes = sizeof(float) * 3;
    triInput.numVertices = positions.size() / 3;
    triInput.vertexBuffers = &d_vertice;

    triInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triInput.vertexStrideInBytes = sizeof(unsigned int) * 3;
    triInput.numIndexTriplets = indices.size() / 3;
    triInput.indexBuffer = d_indice;

    triInput.preTransform = 0;

    // Each build input maps to one or more consecutive records in the
    // shader binding table(SBT), which controls program dispatch.
    triInput.numSbtRecords = 1;
    triInput.sbtIndexOffsetBuffer = 0;
    triInput.sbtIndexOffsetSizeInBytes = 0;
    triInput.sbtIndexOffsetStrideInBytes = 0;

    triInput.flags = m_triangle_input_flags;
}

OptixTraversableHandle OptixAccel::build() {
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
        m_device_context, &accelOptions, m_build_inputs.data(),
        m_build_inputs.size(), &blasBufferSizes));

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
        m_device_context, 0, &accelOptions, m_build_inputs.data(),
        m_build_inputs.size(), tempBuffer.devicePtr(),
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
    optix::GeometryData& geo = rec.data.geometry_data;
    geo.type = optix::GeometryData::TRIANGLE_MESH;
    geo.triangle_mesh.positions =
        (float3*)m_vertex_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.indices = (int3*)m_index_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.normals = (float3*)m_normal_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.texcoords =
        (float2*)m_texcoord_buffers[mesh_idx].devicePtr();
    geo.triangle_mesh.face_num = m_index_buffers[mesh_idx].m_size_in_bytes / sizeof(int3);
    geo.triangle_mesh.pdf = m_pdf[mesh_idx];
    rec.data.light_idx = m_light_idx[mesh_idx];
}

void OptixAccel::packEmittedMesh(std::vector<Light>& lights) const {

    for (auto em : m_emitted_mesh) {
        const int light_idx = em.first;
        const int mesh_idx = em.second;
        if (lights[light_idx].type != Light::Type::AREA) {
            throw Exception("Error in packEmittedMesh");
        }
        lights[light_idx].area.triangle_mesh.positions = (float3*)m_vertex_buffers[mesh_idx].devicePtr();
        lights[light_idx].area.triangle_mesh.indices = (int3*)m_index_buffers[mesh_idx].devicePtr();
        lights[light_idx].area.triangle_mesh.normals = (float3*)m_normal_buffers[mesh_idx].devicePtr();
        lights[light_idx].area.triangle_mesh.texcoords = (float2*)m_texcoord_buffers[mesh_idx].devicePtr();
        lights[light_idx].area.triangle_mesh.face_num = m_index_buffers[mesh_idx].m_size_in_bytes / sizeof(int3);
        lights[light_idx].area.triangle_mesh.pdf = m_pdf[mesh_idx];
    }
}

}  // namespace optix