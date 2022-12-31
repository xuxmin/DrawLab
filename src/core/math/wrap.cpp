#include "core/math/wrap.h"
#include "core/math/math.h"
#include "core/math/point.h"
#include "core/math/vector.h"

namespace drawlab {

Point2f Warp::squareToUniformSquare(const Point2f& sample) { return sample; }

float Warp::squareToUniformSquarePdf(const Point2f& sample) {
    return ((sample >= 0).all() && (sample <= 1).all()) ? 1.0f : 0.0f;
}

/// Warp a uniformly distributed square sample to a 2D tent distribution
Point2f Warp::squareToTent(const Point2f& sample) {
    Point2f s = sample - 0.5;
    float x = s.x() < 0 ? sqrt(2 * s.x() + 1) - 1 : 1 - sqrt(1 - 2 * s.x());
    float y = s.y() < 0 ? sqrt(2 * s.y() + 1) - 1 : 1 - sqrt(1 - 2 * s.y());
    return Point2f(x, y);
}

/// Density of \ref squareToTent per unit area.
float Warp::squareToTentPdf(const Point2f& p) {
    auto p_ = p.cwiseAbs();
    return (p_.x() <= 1 && p_.y() <= 1) ? (1 - p_.x()) * (1 - p_.y()) : 0;
}

/**
 * Uniformly sample a vector on a 2D disk.
 * transforms uniformly distributed 2D points on the unit square into uniformly
 * distributed points on a planar disk with radius 1 centered at the origin.
 */
Point2f Warp::squareToUniformDisk(const Point2f& sample) {
    float r = sqrt(sample.x());
    float theta = 2 * M_PI * sample.y();
    return Point2f(r * cos(theta), r * sin(theta));
}

float Warp::squareToUniformDiskPdf(const Point2f& p) {
    auto p_ = p.cwiseAbs2();
    return (p_.x() + p_.y() <= 1) ? M_INV_PI : 0;
}

Vector3f Warp::squareToUniformSphere(const Point2f& sample) {
    float theta = acos(1 - 2 * sample.x());
    float phi = 2 * M_PI * sample.y();
    return Vector3f(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}

float Warp::squareToUniformSpherePdf(const Vector3f& v) {
    auto d = abs(v.cwiseAbs2().sum() - 1);
    return d <= M_EPSILON ? M_INV_FOURPI : 0;
}

Vector3f Warp::squareToUniformHemisphere(const Point2f& sample) {
    float theta = acos(1 - sample.x());
    float phi = 2 * M_PI * sample.y();
    return Vector3f(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}

float Warp::squareToUniformHemispherePdf(const Vector3f& v) {
    auto d = abs(v.cwiseAbs2().sum() - 1);
    return d <= M_EPSILON && v.z() >= 0 ? M_INV_TWOPI : 0;
}

Vector3f Warp::squareToCosineHemisphere(const Point2f& sample) {
    Point2f p = squareToUniformDisk(sample);
    float z = sqrt(1 - p.cwiseAbs2().sum());
    return Vector3f(p.x(), p.y(), z);
}

float Warp::squareToCosineHemispherePdf(const Vector3f& v) {
    return v.z() <= 0 ? 0 : v.z() * M_INV_PI;
}

/// $p(w_h)=cos\theta_hD(w_h)$
Vector3f Warp::squareToBeckmann(const Point2f& sample, float alpha) {
    float theta = atan(sqrt(-alpha * alpha * log(1 - sample.x())));
    float phi = 2 * M_PI * sample.y();
    return Vector3f(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}

float Warp::squareToBeckmannPdf(const Vector3f& m, float alpha) {
    if (m.z() <= 0)
        return 0;
    float cosTheta = m.z();
    float tanTheta_2 = (1 - cosTheta * cosTheta) / (cosTheta * cosTheta);
    float p = M_INV_TWOPI * (2 * exp(-tanTheta_2 / (alpha * alpha))) /
              (alpha * alpha * cosTheta * cosTheta * cosTheta);
    return p;
}

/// Convert an uniformly distributed square sample into barycentric coordinates
Vector3f Warp::squareToUniformTriangle(const Point2f& sample) {
    float t = sqrt(1.f - sample.x());
    return Vector3f(1 - t, t * sample.y(), t - t * sample.y());
}

}  // namespace drawlab
