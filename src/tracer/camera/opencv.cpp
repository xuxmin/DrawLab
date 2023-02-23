#include "core/math/wrap.h"
#include "core/parser/object.h"
#include "tracer/camera.h"
#include "tracer/rfilter.h"

namespace drawlab {

class OpencvCamera : public Camera {
public:
    OpencvCamera(const PropertyList& propList) {

        m_cx = propList.getFloat("cx");
        m_cy = propList.getFloat("cy");
        m_fx = propList.getFloat("fx");
        m_fy = propList.getFloat("fy");
        Transform extrinsic = propList.getTransform("extrinsic");

        m_inv_ext = extrinsic.getInverseMatrix();

        m_outputSize[0] = propList.getInteger("width", 1280);
        m_outputSize[1] = propList.getInteger("height", 720);

        m_eye = m_inv_ext * Vector4f(0, 0, 0, 1);

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
        throw Exception("OpencvCamera::sampleRay() does not implement!");
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
        return tfm::format("OpencvCamera[\n"
                           "  eye = %s,\n"
                           "  cxcy = %s %s,\n"
                           "  fxfy = %s %s,\n"
                           "  inv_ext = %s,\n"
                           "  output_size = %f,\n"
                           "  rfilter = %s\n"
                           "]",
                           m_eye.toString(), m_cx, m_cy, m_fx, m_fy, 
                           m_inv_ext.toString(), m_outputSize.toString(),
                           indent(m_rfilter->toString()));
    }

    optix::Camera getOptixCamera() const {
        optix::Camera camera;
        camera.type = optix::Camera::OPENCV;
        camera.opencv_cam.eye = make_float3(m_eye[0], m_eye[1], m_eye[2]);
        camera.opencv_cam.cx = m_cx;
        camera.opencv_cam.cy = m_cy;
        camera.opencv_cam.fx = m_fx;
        camera.opencv_cam.fy = m_fy;
        for (int i = 0; i < 16; i++) {
            camera.opencv_cam.invExt[i] = m_inv_ext.m[i / 4][i % 4];
        }
        return camera;
    }

private:
    float m_cx, m_cy, m_fx, m_fy;
    Vector4f m_eye;
    Matrix4f m_inv_ext;
};

REGISTER_CLASS(OpencvCamera, "opencv");

}  // namespace drawlab