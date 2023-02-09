#pragma once

#include "optix/common/optix_params.h"
#include "optix/host/cuda_buffer.h"
#include <vector>

namespace optix {

class OptixAccel {
private:
    const OptixDeviceContext& m_device_context;

    std::vector<OptixBuildInput> m_build_inputs;
    uint32_t m_triangle_input_flags[1];

    std::vector<int> m_light_idx;
    std::vector<CUDABuffer> m_vertex_buffers;
    std::vector<CUDABuffer> m_index_buffers;
    std::vector<CUDABuffer> m_normal_buffers;
    std::vector<CUDABuffer> m_texcoord_buffers;

    std::vector<CUdeviceptr> d_vertices;
    std::vector<CUdeviceptr> d_indices;

    //! buffer that keeps the (final, compacted) accel structure
    optix::CUDABuffer m_as_buffer;

    OptixTraversableHandle m_handle;

public:
    OptixAccel(const OptixDeviceContext& device_context);

    void addTriangleMesh(const std::vector<float>& positions,
                         const std::vector<unsigned int>& indices,
                         const std::vector<float>& normals,
                         const std::vector<float>& texcoords,
                         int light_idx);

    ~OptixAccel();

    OptixTraversableHandle build();

    void packHitgroupRecord(HitgroupRecord& record, int mesh_idx) const;
};

}  // namespace optix