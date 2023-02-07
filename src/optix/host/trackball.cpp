#include "optix/host/trackball.h"
#include <iostream>

namespace optix {

namespace {
    float radians(float degrees) { return degrees * M_PIf / 180.0f; }
    float degrees(float radians) { return radians * M_1_PIf * 180.0f; }
}  // namespace

bool Trackball::wheelEvent(int dir) {
    zoom(dir);
    return true;
}

void Trackball::zoom(int direction) {
    float zoom = (direction > 0) ? 1 / m_zoomMultiplier : m_zoomMultiplier;
    m_cameraEyeLookatDistance *= zoom;
    const float3& lookat = m_camera->lookat();
    const float3& eye = m_camera->eye();
    m_camera->setEye(lookat + (eye - lookat) * zoom);
}

void Trackball::startTracking(int x, int y) {
    m_prevPosX = x;
    m_prevPosY = y;
    m_performTracking = true;
}

void Trackball::updateTracking(int x, int y, int, int) {
    if (!m_performTracking) {
        startTracking(x, y);
        return;
    }

    int deltaX = x - m_prevPosX;
    int deltaY = y - m_prevPosY;

    m_prevPosX = x;
    m_prevPosY = y;
    m_latitude = radians(
        std::min(89.0f, std::max(-89.0f, degrees(m_latitude) + 0.5f * deltaY)));
    m_longitude = radians(fmod(degrees(m_longitude) - 0.5f * deltaX, 360.0f));

    updateCamera();

    if (!m_gimbalLock) {
        reinitOrientationFromCamera();
        m_camera->setUp(m_w);
    }
}

void Trackball::updateCamera() {
    // use latlon for view definition
    float3 localDir;
    localDir.x = cos(m_latitude) * sin(m_longitude);
    localDir.y = cos(m_latitude) * cos(m_longitude);
    localDir.z = sin(m_latitude);

    float3 dirWS = m_u * localDir.x + m_v * localDir.y + m_w * localDir.z;

    if (m_viewMode == EyeFixed) {
        const float3& eye = m_camera->eye();
        m_camera->setLookat(eye - dirWS * m_cameraEyeLookatDistance);
    }
    else {
        const float3& lookat = m_camera->lookat();
        m_camera->setEye(lookat + dirWS * m_cameraEyeLookatDistance);
    }
}

void Trackball::reinitOrientationFromCamera() {
    m_camera->UVWFrame(m_u, m_v, m_w);
    m_u = normalize(m_u);
    m_v = normalize(m_v);
    m_w = normalize(-m_w);
    std::swap(m_v, m_w);
    m_latitude = 0.0f;
    m_longitude = 0.0f;
    m_cameraEyeLookatDistance = length(m_camera->lookat() - m_camera->eye());
}

void Trackball::setReferenceFrame(const float3& u, const float3& v,
                                  const float3& w) {
    m_u = u;
    m_v = v;
    m_w = w;
    float3 dirWS = -normalize(m_camera->lookat() - m_camera->eye());
    float3 dirLocal;
    dirLocal.x = dot(dirWS, u);
    dirLocal.y = dot(dirWS, v);
    dirLocal.z = dot(dirWS, w);
    m_longitude = atan2(dirLocal.x, dirLocal.y);
    m_latitude = asin(dirLocal.z);
}

void Trackball::moveForward(float speed) {
    float3 dirWS = normalize(m_camera->lookat() - m_camera->eye());
    m_camera->setEye(m_camera->eye() + dirWS * speed);
    m_camera->setLookat(m_camera->lookat() + dirWS * speed);
}

void Trackball::moveBackward(float speed) {
    float3 dirWS = normalize(m_camera->lookat() - m_camera->eye());
    m_camera->setEye(m_camera->eye() - dirWS * speed);
    m_camera->setLookat(m_camera->lookat() - dirWS * speed);
}

void Trackball::moveLeft(float speed) {
    float3 u, v, w;
    m_camera->UVWFrame(u, v, w);
    u = normalize(u);

    m_camera->setEye(m_camera->eye() - u * speed);
    m_camera->setLookat(m_camera->lookat() - u * speed);
}

void Trackball::moveRight(float speed) {
    float3 u, v, w;
    m_camera->UVWFrame(u, v, w);
    u = normalize(u);

    m_camera->setEye(m_camera->eye() + u * speed);
    m_camera->setLookat(m_camera->lookat() + u * speed);
}

void Trackball::moveUp(float speed) {
    float3 u, v, w;
    m_camera->UVWFrame(u, v, w);
    v = normalize(v);

    m_camera->setEye(m_camera->eye() + v * speed);
    m_camera->setLookat(m_camera->lookat() + v * speed);
}

void Trackball::moveDown(float speed) {
    float3 u, v, w;
    m_camera->UVWFrame(u, v, w);
    v = normalize(v);

    m_camera->setEye(m_camera->eye() - v * speed);
    m_camera->setLookat(m_camera->lookat() - v * speed);
}

void Trackball::rollLeft(float speed) {
    float3 u, v, w;
    m_camera->UVWFrame(u, v, w);
    u = normalize(u);
    v = normalize(v);

    m_camera->setUp(u * cos(radians(90.0f + speed)) +
                    v * sin(radians(90.0f + speed)));
}

void Trackball::rollRight(float speed) {
    float3 u, v, w;
    m_camera->UVWFrame(u, v, w);
    u = normalize(u);
    v = normalize(v);

    m_camera->setUp(u * cos(radians(90.0f - speed)) +
                    v * sin(radians(90.0f - speed)));
}

}  // namespace optix