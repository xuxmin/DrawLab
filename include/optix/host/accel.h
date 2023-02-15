#pragma once

#include "optix/optix_params.h"
#include "optix/host/cuda_buffer.h"
#include <map>
#include <vector>

namespace optix {

class OptixAccel {
public:
    struct TriangleMesh {
        const std::vector<float>& positions;
        const std::vector<unsigned int>& indices;
        const std::vector<float>& normals;
        const std::vector<float>& texcoords;
        const int light_idx;
        const int material_idx;
        const float pdf;

        TriangleMesh(const std::vector<float>& positions,
                     const std::vector<unsigned int>& indices,
                     const std::vector<float>& normals,
                     const std::vector<float>& texcoords, const int light_idx,
                     const int material_idx, const float pdf)
            : positions(positions), indices(indices), normals(normals),
              texcoords(texcoords), pdf(pdf), light_idx(light_idx),
              material_idx(material_idx) {}
    };

private:
    const OptixDeviceContext& m_device_context;
    std::vector<TriangleMesh> m_meshs;

    std::map<int, int> m_emitted_mesh;  // light_idx -> mesh_idx
    std::vector<CUDABuffer> m_vertex_buffers;
    std::vector<CUDABuffer> m_index_buffers;
    std::vector<CUDABuffer> m_normal_buffers;
    std::vector<CUDABuffer> m_texcoord_buffers;

    //! buffer that keeps the (final, compacted) accel structure
    optix::CUDABuffer m_as_buffer;

    OptixTraversableHandle m_handle;

public:
    OptixAccel(const OptixDeviceContext& device_context);

    void addTriangleMesh(const std::vector<float>& positions,
                         const std::vector<unsigned int>& indices,
                         const std::vector<float>& normals,
                         const std::vector<float>& texcoords,
                         int light_idx, int material_idx,
                         float pdf);

    ~OptixAccel();

    OptixTraversableHandle build();

    void packHitgroupRecord(HitgroupRecord& record, int mesh_idx) const;

    void packEmittedMesh(std::vector<Light>& lights) const;

    int getShapeNum() const { return m_meshs.size(); }
};

}  // namespace optix