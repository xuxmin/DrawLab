#pragma once

#include "optix/optix_params.h"
#include "optix/host/cuda_buffer.h"
#include <map>
#include <vector>

namespace optix {

struct OptixSceneMesh {
    const std::vector<float>& positions;
    const std::vector<unsigned int>& indices;
    const std::vector<float>& normals;
    const std::vector<float>& tangents;
    const std::vector<float>& texcoords;

    const int light_idx;
    const int material_idx;
    const float pdf;

    OptixSceneMesh(const std::vector<float>& positions,
                   const std::vector<unsigned int>& indices,
                   const std::vector<float>& normals,
                   const std::vector<float>& tangents,
                   const std::vector<float>& texcoords, const int light_idx,
                   const int material_idx, const float pdf)
        : positions(positions), indices(indices), normals(normals),
          tangents(tangents), texcoords(texcoords), pdf(pdf),
          light_idx(light_idx), material_idx(material_idx) {}
};

class OptixAccel {
private:
    const OptixDeviceContext& m_device_context;

    std::map<int, int> m_emitted_mesh;  // light_idx -> mesh_idx
    std::vector<CUDABuffer> m_vertex_buffers;
    std::vector<CUDABuffer> m_index_buffers;
    std::vector<CUDABuffer> m_normal_buffers;
    std::vector<CUDABuffer> m_tangent_buffers;
    std::vector<CUDABuffer> m_texcoord_buffers;
    const std::vector<OptixSceneMesh>* m_scene_meshs;

    //! buffer that keeps the (final, compacted) accel structure
    optix::CUDABuffer m_as_buffer;

    OptixTraversableHandle m_handle;

public:
    OptixAccel(const OptixDeviceContext& device_context);

    ~OptixAccel();

    OptixTraversableHandle build(const std::vector<OptixSceneMesh>& meshs);

    void packHitgroupRecord(HitgroupRecord& record, int mesh_idx) const;

    void packEmittedMesh(std::vector<Light>& lights) const;

    int getShapeNum() const { return m_scene_meshs->size(); }

    OptixTraversableHandle getHandle() const { return m_handle; }
};

}  // namespace optix