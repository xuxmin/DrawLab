#include "core/math/wrap.h"
#include "core/parser/object.h"
#include "tracer/camera.h"
#include "tracer/rfilter.h"

namespace drawlab {

/**
 * Virtual Camera System:
 *
 * http://124.223.26.211:8080/images/2021/11/05/ee52ce9c5ace.png
 *
 */
class VirtualCamera : public Camera {
public:
    VirtualCamera(const PropertyList& propList) {
        m_outputSize[0] = propList.getInteger("width", 1280);
        m_outputSize[1] = propList.getInteger("height", 720);
        m_invOutputSize =
            Vector2f(1.f / m_outputSize[0], 1.f / m_outputSize[1]);

        m_fov = propList.getFloat("fov", 30.0f);
        float aspect = m_outputSize.x() / (float)m_outputSize.y();

        m_eye = propList.getVector("eye");
        Vector3f m_lookat = propList.getVector("lookat");
        Vector3f m_up = propList.getVector("up");

        m_W = m_lookat - m_eye;

        float wlen = m_W.length();

        m_U = m_W.cross(m_up).normalized();
        m_V = m_U.cross(m_W).normalized();

        float vlen = wlen * tanf(0.5f * m_fov * M_PI / 180.f);
        m_V *= vlen;
        float ulen = vlen * aspect;
        m_U *= ulen;

        m_rfilter = nullptr;
    }

    void activate() {
        /* If no reconstruction filter was assigned, instantiate a Gaussian
         * filter */
        if (!m_rfilter)
            m_rfilter = static_cast<ReconstructionFilter*>(
                ObjectFactory::createInstance("gaussian", PropertyList()));
    }

    Color3f sampleRay(Ray3f& ray, const Point2f& samplePosition,
                      const Point2f& apertureSample) const {
        // Convert samplePosition coordinate to camera UV
        Vector2f launch_index(samplePosition.x(),
                              m_outputSize[1] - samplePosition.y() - 1);
        Vector2f pixel = launch_index * m_invOutputSize * 2 - 1;
        Vector2f d = pixel;

        ray.o = Point3f(m_eye[0], m_eye[1], m_eye[2]);
        ray.d = (d.x() * m_U + d.y() * m_V + m_W).normalized();
        ray.update();

        return Color3f(1.0f);
    }

    void addChild(Object* obj) {
        switch (obj->getClassType()) {
            case EReconstructionFilter:
                if (m_rfilter)
                    throw Exception("Camera: tried to register multiple "
                                    "reconstruction filters!");
                m_rfilter = static_cast<ReconstructionFilter*>(obj);
                break;

            default:
                throw Exception("Camera::addChild(<%s>) is not supported!",
                                classTypeName(obj->getClassType()));
        }
    }

    /// Return a human-readable summary
    std::string toString() const {
        return tfm::format("VirtualeCamera[\n"
                           "  eye = %s,\n"
                           "  U = %s,\n"
                           "  V = %s,\n"
                           "  W = %s,\n"
                           "  fov = %f,\n"
                           "  output_size = %f,\n"
                           "  rfilter = %s\n"
                           "]",
                           m_eye.toString(), m_U.toString(), m_V.toString(),
                           m_W.toString(), m_fov, m_outputSize.toString(),
                           indent(m_rfilter->toString()));
    }

    void packLaunchParameters(optix::LaunchParams& launch_params) const {
        launch_params.camera_data.type = optix::CameraData::VIRTUAL;
        launch_params.camera_data.virtual_cam.eye = make_float3(m_eye[0], m_eye[1], m_eye[2]);
        launch_params.camera_data.virtual_cam.U = make_float3(m_U[0], m_U[1], m_U[2]);
        launch_params.camera_data.virtual_cam.V = make_float3(m_V[0], m_V[1], m_V[2]);
        launch_params.camera_data.virtual_cam.W = make_float3(m_W[0], m_W[1], m_W[2]);
    }

private:
    Vector3f m_eye;
    Vector3f m_U;
    Vector3f m_V;
    Vector3f m_W;
    float m_fov;

    Vector2f m_invOutputSize;
};

REGISTER_CLASS(VirtualCamera, "virtual");

}  // namespace drawlab