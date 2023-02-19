#include "tracer/mesh.h"
#include <fstream>
#include <unordered_map>
#include <spdlog/spdlog.h>


namespace drawlab {

class Rectangle : public Mesh {
public:
    Rectangle(const PropertyList& propList) {
        Vector3f pos = propList.getVector("pos");
        Vector3f v0 = propList.getVector("v0");
        Vector3f v1 = propList.getVector("v1");
        Vector3f normal = propList.getVector("normal");

        std::vector<Vector3f> p = {pos, pos + v0, pos + v1, pos + v0 + v1};
        m_V.resize(4 * 3);
        m_N.resize(4 * 3);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                m_V[i * 3 + j] = p[i][j];
                m_N[i * 3 + j] = normal[j];
            }
        }
        m_UV.resize(4 * 2);
        m_UV[0] = 1; m_UV[1] = 0;
        m_UV[2] = 1; m_UV[3] = 1;
        m_UV[4] = 0; m_UV[5] = 0;
        m_UV[6] = 0; m_UV[7] = 1;

        m_F.resize(6);
        m_F[0] = 0; m_F[1] = 1; m_F[2] = 2;
        m_F[3] = 3; m_F[4] = 2; m_F[5] = 1;

        for (int i = 0; i < m_V.size() / 3; i++) {
            m_bbox.expandBy(
                Point3f(m_V[3 * i], m_V[3 * i + 1], m_V[3 * i + 2]));
        }
    }
};

REGISTER_CLASS(Rectangle, "rectangle");

}  // namespace drawlab