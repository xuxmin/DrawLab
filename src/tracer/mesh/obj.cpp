#include "tiny_obj_loader.h"
#include "tracer/mesh.h"
#include "core/utils/timer.h"
#include <filesystem/resolver.h>
#include <fstream>
#include <unordered_map>

namespace drawlab {

/**
 * \brief Loader for Wavefront OBJ triangle meshes
 */
class WavefrontOBJ : public Mesh {
public:
    WavefrontOBJ(const PropertyList& propList) {
        filesystem::path filename =
            getFileResolver()->resolve(propList.getString("filename"));

        std::string err;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        bool ok =
            tinyobj::LoadObj(shapes, materials, err, filename.str().c_str());
        if (!ok) {
            throw Exception("Loadobj fail: %s", err);
        }
        Transform trafo = propList.getTransform("toWorld", Transform());

        cout << "Loading \"" << filename << "\" .. ";
        cout.flush();
        Timer timer;

        size_t max_index = 0;
        for (int shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
            std::vector<float>& positions = shapes[shape_idx].mesh.positions;
            std::vector<float>& normals = shapes[shape_idx].mesh.normals;
            std::vector<float>& texcoords = shapes[shape_idx].mesh.texcoords;
            std::vector<unsigned int>& indices = shapes[shape_idx].mesh.indices;

            size_t m_V_len = m_V.size();
            m_V.resize(m_V_len + positions.size());
            for (int i = 0; i < positions.size() / 3; i++) {
                Point3f p = Point3f(positions[3 * i], positions[3 * i + 1],
                                    positions[3 * i + 2]);
                p = trafo * p;
                m_V[m_V_len + 3 * i] = p[0];
                m_V[m_V_len + 3 * i + 1] = p[1];
                m_V[m_V_len + 3 * i + 2] = p[2];
            }

            size_t m_N_len = m_N.size();
            m_N.resize(m_N_len + normals.size());
            for (int i = 0; i < normals.size() / 3; i++) {
                Normal3f n = Normal3f(normals[3 * i], normals[3 * i + 1],
                                      normals[3 * i + 2]);
                n = trafo * n;
                m_N[m_N_len + 3 * i] = n[0];
                m_N[m_N_len + 3 * i + 1] = n[1];
                m_N[m_N_len + 3 * i + 2] = n[2];
            }

            size_t m_UV_len = m_UV.size();
            m_UV.resize(m_UV_len + texcoords.size());
            std::copy(texcoords.begin(), texcoords.end(),
                      m_UV.begin() + m_UV_len);

            size_t m_F_len = m_F.size();
            m_F.resize(m_F_len + indices.size());
            for (int i = 0; i < indices.size(); i++) {
                m_F[m_F_len + i] = indices[i] + max_index;
            }
            max_index = m_V.size() / 3;
        }

        for (int i = 0; i < m_V.size() / 3; i++) {
            m_bbox.expandBy(
                Point3f(m_V[3 * i], m_V[3 * i + 1], m_V[3 * i + 2]));
        }

        m_name = filename.str();
        cout << "done. (V=" << m_V.size() / 3 << ", F=" << m_F.size() / 3
             << ", took " << timer.elapsedString() << " and "
             << memString(m_F.size() * sizeof(uint32_t) +
                          sizeof(float) *
                              (m_V.size() + m_N.size() + m_UV.size()))
             << ")" << endl;
    }
};

REGISTER_CLASS(WavefrontOBJ, "obj");

}  // namespace drawlab