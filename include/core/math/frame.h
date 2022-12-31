#pragma once

#include "core/math/math.h"
#include "core/math/normal.h"
#include "core/math/vector.h"

namespace drawlab {

struct Frame {
    Vector3f t, b;
    Normal3f n;

    Frame() {}

    Frame(const Normal3f& n, const Vector3f& t, const Vector3f b)
        : n(n), t(t), b(b) {}

    Frame(const Normal3f& n) : n(n) { buildLocalFrame(n, t, b); }

    Frame(const Vector3f& vn) {
        n[0] = vn[0];
        n[1] = vn[1];
        n[2] = vn[2];
        buildLocalFrame(n, t, b);
    }

    static void buildLocalFrame(const Normal3f& n, Vector3f& t, Vector3f& b) {
        if (std::abs(n.z() - 1.0) > M_EPSILON && abs(n.z() + 1.0) > M_EPSILON)
            t = Vector3f(0.0, 0.0, 1.0);
        else
            t = Vector3f(1.0, 0.0, 0.0);

        Vector3f n_ = n.toVector();
        b = n_.cross(t);
        b.normalize();
        t = b.cross(n_);
    }

    /// Convert from world coordinates to local coordinates
    Vector3f toLocal(const Vector3f& v) const {
        return Vector3f(v.dot(t), v.dot(b), n.dot(v));
    }

    /// Convert from local coordinates to world coordinates
    Vector3f toWorld(const Vector3f& v) const {
        return t * v.x() + b * v.y() + n * v.z();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the cosine of the angle between the normal and v */
    static float cosTheta(const Vector3f& v) { return v.z(); }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the sine of the angle between the normal and v */
    static float sinTheta(const Vector3f& v) {
        float temp = sinTheta2(v);
        if (temp <= 0.0f)
            return 0.0f;
        return std::sqrt(temp);
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the tangent of the angle between the normal and v */
    static float tanTheta(const Vector3f& v) {
        float temp = 1 - v.z() * v.z();
        if (temp <= 0.0f)
            return 0.0f;
        return std::sqrt(temp) / v.z();
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the squared sine of the angle between the normal and v */
    static float sinTheta2(const Vector3f& v) { return 1.0f - v.z() * v.z(); }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the sine of the phi parameter in spherical coordinates */
    static float sinPhi(const Vector3f& v) {
        float sinTheta = Frame::sinTheta(v);
        if (sinTheta == 0.0f)
            return 1.0f;
        return clamp(v.y() / sinTheta, -1.0f, 1.0f);
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the cosine of the phi parameter in spherical coordinates
     */
    static float cosPhi(const Vector3f& v) {
        float sinTheta = Frame::sinTheta(v);
        if (sinTheta == 0.0f)
            return 1.0f;
        return clamp(v.x() / sinTheta, -1.0f, 1.0f);
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the squared sine of the phi parameter in  spherical
     * coordinates */
    static float sinPhi2(const Vector3f& v) {
        return clamp(v.y() * v.y() / sinTheta2(v), 0.0f, 1.0f);
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the squared cosine of the phi parameter in  spherical
     * coordinates */
    static float cosPhi2(const Vector3f& v) {
        return clamp(v.x() * v.x() / sinTheta2(v), 0.0f, 1.0f);
    }

    /// Equality test
    bool operator==(const Frame& frame) const {
        return frame.t == t && frame.b == b && frame.n == n;
    }

    /// Inequality test
    bool operator!=(const Frame& frame) const { return !operator==(frame); }

    /// Return a human-readable string summary of this frame
    std::string toString() const {
        return tfm::format("Frame[\n"
                           "  t = %s,\n"
                           "  b = %s,\n"
                           "  n = %s\n"
                           "]",
                           t.toString(), b.toString(), n.toString());
    }
};

}  // namespace drawlab