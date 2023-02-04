#pragma once

#include "optix/host/cuda_buffer.h"
#include "tracer/mesh.h"
#include <vector>

namespace drawlab {

class OptixAccel {

private:
    const std::vector<Mesh*> m_meshPtrs;

    const OptixDeviceContext& m_device_context;

    std::vector<optix::CUDABuffer> m_vertex_buffers;
    std::vector<optix::CUDABuffer> m_index_buffers;
    std::vector<optix::CUDABuffer> m_normal_buffers;
    std::vector<optix::CUDABuffer> m_texcoord_buffers;
    //! buffer that keeps the (final, compacted) accel structure
    optix::CUDABuffer m_as_buffer;

    OptixTraversableHandle m_handle;

public:
    OptixAccel(const OptixDeviceContext& device_context, const std::vector<Mesh*> mesh_ptrs);

    ~OptixAccel();

    OptixTraversableHandle build();

    void packHitgroupRecord(optix::HitgroupRecord& record, int mesh_idx) const;
};

}  // namespace drawlab