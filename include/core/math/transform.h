#pragma once

#include "core/base/common.h"
#include "core/math/math.h"
#include "core/math/matrix.h"
#include "core/math/normal.h"
#include "core/math/ray.h"

namespace drawlab {

/**
 * \brief Homogeneous coordinate transformation
 *
 * This class stores a general homogeneous coordinate tranformation, such as
 * rotation, translation, uniform or non-uniform scaling, and perspective
 * transformations.
 */
struct Transform {
public:
    /// Create the identity transform
    Transform() : m_transform(Matrix4f::getIdentityMatrix()) {}

    /// Create a new transform instance for the given matrix
    Transform(const Matrix4f& trafo) : m_transform(trafo) {}

    /// Return the underlying matrix
    const Matrix4f& getMatrix() const { return m_transform; }

    /// Return the inverse of the underlying matrix
    const Matrix4f getInverseMatrix() const { return m_transform.inv(); }

    /// Concatenate with another transform
    Transform operator*(const Transform& t) const {
        return Transform(m_transform * t.m_transform);
    }

    friend Transform operator*(const Matrix4f& mat, const Transform& t) {
        return Transform(mat * t.m_transform);
    }

    /// Apply the homogeneous transformation to a 3D vector
    Vector3f operator*(const Vector3f& v) const {
        Vector4f result = m_transform * Vector4f(v[0], v[1], v[2], 0.0f);
        return Vector3f(result[0], result[1], result[2]);
    }

    /// Apply the homogeneous transformation to a 3D normal
    Normal3f operator*(const Normal3f& n) const {
        Vector4f result =
            m_transform.inv().transpose() * Vector4f(n[0], n[1], n[2], 0.0f);
        return Normal3f(result[0], result[1], result[2]);
    }

    /// Transform a point by an arbitrary matrix in homogeneous coordinates
    Point3f operator*(const Point3f& p) const {
        Vector4f result = m_transform * Vector4f(p[0], p[1], p[2], 1.0f);
        return Point3f(result[0], result[1], result[2]) / result[3];
    }

    /// Apply the homogeneous transformation to a ray
    Ray3f operator*(const Ray3f& r) const {
        return Ray3f(operator*(r.o), operator*(r.d), r.mint, r.maxt);
    }

    /// Return a string representation
    std::string toString() const {
        return "TRANSFORM" + m_transform.toString();
    }

    /// Set a indentity transform
    void setIdentity() { m_transform = Matrix4f::getIdentityMatrix(); }

private:
    Matrix4f m_transform;
};

}  // namespace drawlab
