#pragma once

#include "core/base/common.h"

namespace drawlab {

template <typename T> struct TNormal3 {
public:
    T m[3];

    TNormal3(T u = (T)0) {
        for (size_t i = 0; i < 3; i++)
            m[i] = u;
    }

    TNormal3(const TNormal3<T>& u) {
        for (size_t i = 0; i < 3; i++)
            m[i] = u.m[i];
    }

    TNormal3(const Vector3f& u) {
        for (size_t i = 0; i < 3; i++)
            m[i] = u.m[i];
    }

    TNormal3(T x, T y, T z) {
        m[0] = x;
        m[1] = y;
        m[2] = z;
    }

    const T& operator[](size_t i) const {
        assert(i < 3);
        return m[i];
    }

    T& operator[](size_t i) {
        assert(i < 3);
        return m[i];
    }

    T x() const { return m[0]; }

    T y() const { return m[1]; }

    T z() const { return m[2]; }

    T* ptr() { return m; }
    const T* ptr() const { return m; }

    T coeff(size_t i) const {
        assert(i < 3);
        return m[i];
    }

    /// @brief Returns the square of the length(magnitude) of the vector.
    T squaredLength() const {
        T sum = 0;
        for (size_t i = 0; i < 3; i++)
            sum += m[i] * m[i];
        return sum;
    }

    /// @brief Returns the length of the vector.
    T length() const { return sqrt(squaredLength()); }

    /// @brief Normalize the vector and return a new vector, the
    /// fuction will not change this vector
    TNormal3<T> normalized() const {
        TNormal3<T> c;
        T len = length();
        for (size_t i = 0; i < 3; i++) {
            c[i] = m[i] / len;
        }
        return c;
    }

    /// @brief Normalize the vector and return nothing
    void normalize() {
        T len = length();
        for (size_t i = 0; i < 3; i++) {
            m[i] /= len;
        }
    }

    T dot(const TVector<3, T>& rhs) const {
        T sum = 0;
        for (size_t i = 0; i < 3; i++)
            sum += m[i] * rhs[i];
        return sum;
    }

    friend T dot(const TVector<3, T>& lhs, const TNormal3<T>& rhs) {
        static_assert(N == 3);
        T sum = 0;
        for (size_t i = 0; i < 3; i++)
            sum += lhs[i] * rhs[i];
        return sum;
    }

    TVector<3, T> operator*(const T& scalar) const {
        TVector<3, T> c;
        for (size_t i = 0; i < 3; i++)
            c.m[i] = this->m[i] * scalar;
        return c;
    }

    friend TVector<3, T> operator*(T scalar, const TNormal3<T>& rhs) {
        TVector<3, T> c;
        for (size_t i = 0; i < 3; i++)
            c.m[i] = rhs[i] * scalar;
        return c;
    }

    bool operator==(const TNormal3<T>& rhs) const {
        for (size_t i = 0; i < 3; i++) {
            if (m[i] != rhs[i])
                return false;
        }
        return true;
    }

    TVector<3, T> toVector() const {
        TVector<3, T> res;
        res[0] = m[0];
        res[1] = m[1];
        res[2] = m[2];
        return res;
    }

    std::string toString() const {
        std::string result;
        for (size_t i = 0; i < 3; ++i) {
            result += std::to_string(this->coeff(i));
            if (i + 1 < 3)
                result += ", ";
        }
        return "Normal: [" + result + "]";
    }
};

}  // namespace drawlab
