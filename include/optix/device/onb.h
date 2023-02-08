
#include <cuda_runtime.h>

namespace optix {

struct Onb {

    // Build local coordinate system based on normal
    __forceinline__ __device__ Onb(const float3& normal) {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z)) {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    // Convert vector from local coordinate system to the xx
    __forceinline__ __device__ float3 inverse_transform(const float3& p) const {
        return p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    // Convert vector to local coordinate system 
    __forceinline__ __device__ float3 transform(const float3& p) const {
        return make_float3(dot(p, m_tangent), dot(p, m_binormal), dot(p, m_normal));
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

}  // namespace optix