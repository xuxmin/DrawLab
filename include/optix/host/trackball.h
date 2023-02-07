#pragma once

#include "optix/host/camera.h"

namespace optix {

class Trackball {
public:
    void moveForward(float speed);
    void moveBackward(float speed);
    void moveLeft(float speed);
    void moveRight(float speed);
    void moveUp(float speed);
    void moveDown(float speed);
    void rollLeft(float speed);
    void rollRight(float speed);
    bool wheelEvent(int dir);

    void zoom(int direction);

    void startTracking(int x, int y);
    void updateTracking(int x, int y, int canvasWidth, int canvasHeight);

    // Setting the gimbal lock to 'on' will fix the reference frame (i.e., the
    // singularity of the trackball). In most cases this is preferred. For free
    // scene exploration the gimbal lock can be turned off, which causes the
    // trackball's reference frame to be update on every camera update (adopted
    // from the camera).
    bool gimbalLock() const { return m_gimbalLock; }
    void setGimbalLock(bool val) { m_gimbalLock = val; }

    // Set the camera that will be changed according to user input.
    // Warning, this also initializes the reference frame of the trackball from
    // the camera. The reference frame defines the orbit's singularity.
    inline void setCamera(PerspectiveCamera* camera) {
        m_camera = camera;
        reinitOrientationFromCamera();
    }
    inline const PerspectiveCamera* currentCamera() const { return m_camera; }

    // Adopts the reference frame from the camera.
    // Note that the reference frame of the camera usually has a different 'up'
    // than the 'up' of the camera. Though, typically, it is desired that the
    // trackball's reference frame aligns with the actual up of the camera.
    void reinitOrientationFromCamera();

    // Specify the frame of the orbit that the camera is orbiting around.
    // The important bit is the 'up' of that frame as this is defines the
    // singularity. Here, 'up' is the 'w' component. Typically you want the up
    // of the reference frame to align with the up of the camera. However, to be
    // able to really freely move around, you can also constantly update the
    // reference frame of the trackball. This can be done by calling
    // reinitOrientationFromCamera(). In most cases it is not required though
    // (set the frame/up once, leave it as is).
    void setReferenceFrame(const float3& u, const float3& v, const float3& w);

    enum ViewMode { EyeFixed, LookAtFixed };
    ViewMode viewMode() const { return m_viewMode; }
    void setViewMode(ViewMode val) { m_viewMode = val; }

    float moveSpeed() const { return m_moveSpeed; }
    void setMoveSpeed(const float& val) { m_moveSpeed = val; }

private:
    void updateCamera();

private:
    bool m_gimbalLock = false;
    ViewMode m_viewMode = LookAtFixed;
    PerspectiveCamera* m_camera = nullptr;
    float m_cameraEyeLookatDistance = 0.0f;
    float m_zoomMultiplier = 1.1f;
    float m_moveSpeed = 1.0f;
    float m_rollSpeed = 0.5f;

    float m_latitude = 0.0f;   // in radians
    float m_longitude = 0.0f;  // in radians

    // mouse tracking
    int m_prevPosX = 0;
    int m_prevPosY = 0;
    bool m_performTracking = false;

    // trackball computes camera orientation (eye, lookat) using
    // latitude/longitude with respect to this frame local frame for trackball
    float3 m_u = {0.0f, 0.0f, 0.0f};
    float3 m_v = {0.0f, 0.0f, 0.0f};
    float3 m_w = {0.0f, 0.0f, 0.0f};
};

}  // namespace optix
