#include "tracer/backend/context.h"

namespace optix {

void Context::initCameraState(const optix::Camera& cam, float aspect) {
    camera.setEye(cam.virtual_cam.eye);
    camera.setLookat(cam.virtual_cam.looat);
    camera.setUp(cam.virtual_cam.up);
    camera.setFovY(cam.virtual_cam.fov);
    camera.setAspectRatio(aspect);
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f),
                                make_float3(0.0f, 0.0f, 1.0f),
                                make_float3(0.0f, 1.0f, 0.0f));
    trackball.setGimbalLock(true);
}

Camera Context::getOptixCamera() const {
    Camera optix_cam;
    optix_cam.type = Camera::VIRTUAL;
    optix_cam.virtual_cam.eye = camera.eye();
    camera.UVWFrame(optix_cam.virtual_cam.U, optix_cam.virtual_cam.V,
                    optix_cam.virtual_cam.W);
    return optix_cam;
}

}  // namespace drawlab