#pragma once

#include "core/base/common.h"
#include "core/math/normal.h"
#include <cmath>

namespace drawlab {

/**
 * \brief Generic N-dimensional vector data structure
 */
template <size_t N, typename T> struct TVector {
public:
    T m[N];

    /// @brief Create a new vector with constant component vlaues
    TVector(T u = (T)0) {
        for (size_t i = 0; i < N; i++)
            m[i] = u;
    }

    TVector(const TVector<N, T>& u) {
        for (size_t i = 0; i < N; i++)
            m[i] = u.m[i];
    }

    TVector(T x, T y) {
        static_assert(N == 2, "TVector error");
        m[0] = x;
        m[1] = y;
    }

    TVector(T x, T y, T z) {
        static_assert(N == 3, "TVector error");
        m[0] = x;
        m[1] = y;
        m[2] = z;
    }

    TVector(T x, T y, T z, T w) {
        static_assert(N == 4, "TVector error");
        m[0] = x;
        m[1] = y;
        m[2] = z;
        m[3] = w;
    }

    TVector(const std::initializer_list<T>& u) {
        auto it = u.begin();
        for (size_t i = 0; i < N; i++)
            m[i] = *it++;
    }

    const T& operator[](size_t i) const {
        assert(i < N);
        return m[i];
    }

    T& operator[](size_t i) {
        assert(i < N);
        return m[i];
    }

    T* ptr() { return m; }
    const T* ptr() const { return m; }

    T coeff(size_t i) const { return m[i]; }

    T maxCoeff() const {
        T max_val = m[0];
        for (size_t i = 1; i < N; i++) {
            max_val = max(max_val, m[i]);
        }
        return max_val;
    }

    TVector<N, T> cwiseInverse() const {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = 1 / this->m[i];
        return c;
    }

    TVector<N, T> cwiseAbs() const {
        TVector<N, T> ret;
        for (size_t i = 0; i < N; i++) {
            ret[i] = abs(m[i]);
        }
        return ret;
    }

    TVector<N, T> cwiseAbs2() const {
        TVector<N, T> ret;
        for (size_t i = 0; i < N; i++) {
            ret[i] = m[i] * m[i];
        }
        return ret;
    }

    T sum() const {
        T s = 0;
        for (size_t i = 0; i < N; i++) {
            s += m[i];
        }
        return s;
    }

    const T& x() const { return m[0]; }

    T& x() { return m[0]; }

    const T& y() const { return m[1]; }

    T& y() { return m[1]; }

    const T& z() const { return m[2]; }

    T& z() { return m[2]; }

    const T& w() const { return m[3]; }

    T& w() { return m[3]; }

    bool operator==(const TVector<N, T>& rhs) const {
        for (size_t i = 0; i < N; i++) {
            if (m[i] != rhs[i])
                return false;
        }
        return true;
    }

    bool operator!=(const TVector<N, T>& rhs) const {
        for (size_t i = 0; i < N; i++) {
            if (m[i] != rhs[i])
                return true;
        }
        return false;
    }

    // arithmetic operations
    TVector<N, T> operator+(const TVector<N, T>& rhs) const {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = this->m[i] + rhs[i];
        return c;
    }

    TVector<N, T> operator-(const TVector<N, T>& rhs) const {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = this->m[i] - rhs[i];
        return c;
    }

    TVector<N, T> operator*(const TVector<N, T>& rhs) const {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = this->m[i] * rhs[i];
        return c;
    }

    TVector<N, T> operator*(const T& scalar) const {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = this->m[i] * scalar;
        return c;
    }

    TVector<N, T> operator/(const TVector<N, T>& rhs) const {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++) {
            assert(rhs.m[i] != 0.0);
            c.m[i] = this->m[i] / rhs.m[i];
        }
        return c;
    }

    TVector<N, T> operator/(const T& scalar) const {
        assert(scalar != 0.0);
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = this->m[i] / scalar;
        return c;
    }

    friend TVector<N, T> operator*(T scalar, const TVector<N, T>& rhs) {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = rhs[i] * scalar;
        return c;
    }

    friend TVector<N, T> operator/(T scalar, const TVector<N, T>& rhs) {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = scalar / rhs[i];
        return c;
    }

    friend TVector<N, T> operator+(const TVector<N, T>& lhs, T rhs) {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = lhs[i] + rhs;
        return c;
    }

    friend TVector<N, T> operator+(T lhs, const TVector<N, T>& rhs) {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = lhs + rhs[i];
        return c;
    }

    friend TVector<N, T> operator-(const TVector<N, T>& lhs, T rhs) {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = lhs[i] - rhs;
        return c;
    }

    friend TVector<N, T> operator-(T lhs, const TVector<N, T>& rhs) {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = lhs - rhs[i];
        return c;
    }

    TVector<N, T>& operator+=(const TVector<N, T>& rhs) {
        for (size_t i = 0; i < N; i++)
            m[i] += rhs[i];
        return *this;
    }

    TVector<N, T>& operator+=(T scalar) {
        for (size_t i = 0; i < N; i++)
            m[i] += scalar;
        return *this;
    }

    TVector<N, T>& operator-=(const TVector<N, T>& rhs) {
        for (size_t i = 0; i < N; i++)
            m[i] -= rhs[i];
        return *this;
    }

    TVector<N, T>& operator-=(T scalar) {
        for (size_t i = 0; i < N; i++)
            m[i] -= scalar;
        return *this;
    }

    TVector<N, T>& operator*=(const TVector<N, T>& rhs) {
        for (size_t i = 0; i < N; i++)
            m[i] *= rhs[i];
        return *this;
    }

    TVector<N, T>& operator*=(T scalar) {
        for (size_t i = 0; i < N; i++)
            m[i] *= scalar;
        return *this;
    }

    TVector<N, T>& operator/=(const TVector<N, T>& rhs) {
        for (size_t i = 0; i < N; i++) {
            assert(rhs[i] != 0);
            m[i] /= rhs[i];
        }
        return *this;
    }

    TVector<N, T>& operator/=(T scalar) {
        assert(scalar != 0);
        for (size_t i = 0; i < N; i++)
            m[i] /= scalar;
        return *this;
    }

    const TVector<N, T>& operator+() const { return *this; }

    TVector<N, T> operator-() const {
        TVector<N, T> ret;
        for (size_t i = 0; i < N; i++) {
            ret[i] = -m[i];
        }
        return ret;
    }

    /// @brief Returns the square of the length(magnitude) of the vector.
    T squaredLength() const {
        T sum = 0;
        for (size_t i = 0; i < N; i++)
            sum += m[i] * m[i];
        return sum;
    }

    /// @brief Returns the length of the vector.
    T length() const { return sqrt(squaredLength()); }

    /// @brief Normalize the vector and return a new vector, the
    /// fuction will not change this vector
    TVector<N, T> normalized() const {
        TVector<N, T> c;
        T len = length();
        for (size_t i = 0; i < N; i++) {
            c[i] = m[i] / len;
        }
        return c;
    }

    /// @brief Normalize the vector and return nothing
    void normalize() {
        T len = length();
        for (size_t i = 0; i < N; i++) {
            m[i] /= len;
        }
    }

    /// @brief Product of all elements.
    T prod() const {
        T result = 1;
        for (size_t i = 0; i < N; i++) {
            result *= m[i];
        }
        return result;
    }

    T dot(const TVector<N, T>& rhs) const {
        T sum = 0;
        for (size_t i = 0; i < N; i++)
            sum += m[i] * rhs[i];
        return sum;
    }

    T cross(const TVector<2, T>& rhs) const {
        static_assert(N == 2);
        return m[0] * rhs[1] - m[1] * rhs[0];
    }

    TVector<3, T> cross(const TVector<3, T>& rhs) const {
        static_assert(N == 3);
        return TVector<3, T>(m[1] * rhs[2] - m[2] * rhs[1],
                             m[2] * rhs[0] - m[0] * rhs[2],
                             m[0] * rhs[1] - m[1] * rhs[0]);
    }

    std::string toString() const {
        std::string result;
        for (size_t i = 0; i < N; ++i) {
            result += std::to_string(this->coeff(i));
            if (i + 1 < N)
                result += ", ";
        }
        return "Vector: [" + result + "]";
    }
};

}  // namespace drawlab
