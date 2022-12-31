#pragma once

#include "core/base/common.h"
#include "core/math/array.h"
#include <cmath>

namespace drawlab {

/**
 * \brief Generic N-dimensional point data structure
 */
template <size_t N, typename T> struct TPoint {
public:
    T m[N];

    TPoint(T u = (T)0) {
        for (size_t i = 0; i < N; i++)
            m[i] = u;
    }

    TPoint(const TPoint<N, T>& u) {
        for (size_t i = 0; i < N; i++)
            m[i] = u.m[i];
    }

    TPoint(T x, T y) {
        static_assert(N == 2, "TPoint error");
        m[0] = x;
        m[1] = y;
    }

    TPoint(T x, T y, T z) {
        static_assert(N == 3, "TPoint error");
        m[0] = x;
        m[1] = y;
        m[2] = z;
    }

    TPoint(T x, T y, T z, T w) {
        static_assert(N == 4, "TPoint error");
        m[0] = x;
        m[1] = y;
        m[2] = z;
        m[3] = w;
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

    const T& x() const { return m[0]; }

    T& x() { return m[0]; }

    const T& y() const { return m[1]; }

    T& y() { return m[1]; }

    const T& z() const { return m[2]; }

    T& z() { return m[2]; }

    T coeff(size_t i) const { return m[i]; }

    T maxCoeff() const {
        T max_val = m[0];
        for (size_t i = 1; i < N; i++) {
            max_val = max(max_val, m[i]);
        }
        return max_val;
    }

    TPoint<N, T> cwiseMin(const TPoint<N, T>& rhs) const {
        TPoint<N, T> ret;
        for (size_t i = 0; i < N; i++) {
            ret[i] = std::min(m[i], rhs[i]);
        }
        return ret;
    }

    TPoint<N, T> cwiseMax(const TPoint<N, T>& rhs) const {
        TPoint<N, T> ret;
        for (size_t i = 0; i < N; i++) {
            ret[i] = std::max(m[i], rhs[i]);
        }
        return ret;
    }

    TPoint<N, T> cwiseAbs() const {
        TPoint<N, T> ret;
        for (size_t i = 0; i < N; i++) {
            ret[i] = std::abs(m[i]);
        }
        return ret;
    }

    TPoint<N, T> cwiseAbs2() const {
        TPoint<N, T> ret;
        for (size_t i = 0; i < N; i++) {
            ret[i] = m[i] * m[i];
        }
        return ret;
    }

    /// @brief Normalize the vector and return a new vector, the
    /// fuction will not change this vector
    TVector<N, float> normalized() const {
        TVector<N, float> c;
        float sum = 0;
        for (size_t i = 0; i < N; i++)
            sum += m[i] * m[i];

        float len = sqrt(sum);

        for (size_t i = 0; i < N; i++) {
            c[i] = m[i] / len;
        }
        return c;
    }

    T sum() const {
        T s = 0;
        for (size_t i = 0; i < N; i++) {
            s += m[i];
        }
        return s;
    }

    bool operator==(const TPoint<N, T>& rhs) const {
        for (size_t i = 0; i < N; i++) {
            if (m[i] != rhs[i])
                return false;
        }
        return true;
    }

    bool operator!=(const TPoint<N, T>& rhs) const {
        for (size_t i = 0; i < N; i++) {
            if (m[i] != rhs[i])
                return true;
        }
        return false;
    }

    // arithmetic operations
    TPoint<N, T> operator+(const TPoint<N, T>& rhs) const {
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = this->m[i] + rhs[i];
        return c;
    }

    /// @brief Subtract a point from the other to get a vector
    TVector<N, T> operator-(const TPoint<N, T>& rhs) const {
        TVector<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = this->m[i] - rhs[i];
        return c;
    }

    TArray<N, bool> operator>(const TPoint<N, T>& rhs) const {
        TArray<N, bool> c;
        for (size_t i = 0; i < N; i++) {
            c.m[i] = this->m[i] > rhs.m[i];
        }
        return c;
    }

    TArray<N, bool> operator>=(const TPoint<N, T>& rhs) const {
        TArray<N, bool> c;
        for (size_t i = 0; i < N; i++) {
            c.m[i] = this->m[i] >= rhs.m[i];
        }
        return c;
    }

    TArray<N, bool> operator<(const TPoint<N, T>& rhs) const {
        TArray<N, bool> c;
        for (size_t i = 0; i < N; i++) {
            c.m[i] = this->m[i] < rhs.m[i];
        }
        return c;
    }

    TArray<N, bool> operator<=(const TPoint<N, T>& rhs) const {
        TArray<N, bool> c;
        for (size_t i = 0; i < N; i++) {
            c.m[i] = this->m[i] <= rhs.m[i];
        }
        return c;
    }

    void setConstant(T constant) {
        for (size_t i = 0; i < N; i++) {
            m[i] = constant;
        }
    }

    TPoint<N, T> operator*(const TPoint<N, T>& rhs) const {
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = this->m[i] * rhs[i];
        return c;
    }

    TPoint<N, T> operator*(const T& scalar) const {
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = this->m[i] * scalar;
        return c;
    }

    TPoint<N, T> operator/(const TPoint<N, T>& rhs) const {
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++) {
            assert(rhs.m[i] != 0.0);
            c.m[i] = this->m[i] / rhs.m[i];
        }
        return c;
    }

    TPoint<N, T> operator/(const T& scalar) const {
        assert(scalar != 0.0);
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = this->m[i] / scalar;
        return c;
    }

    friend TPoint<N, T> operator*(T scalar, const TPoint<N, T>& rhs) {
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = rhs[i] * scalar;
        return c;
    }

    friend TPoint<N, T> operator/(T scalar, const TPoint<N, T>& rhs) {
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = scalar / rhs[i];
        return c;
    }

    friend TPoint<N, T> operator+(const TPoint<N, T>& lhs, T rhs) {
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = lhs[i] + rhs;
        return c;
    }

    friend TPoint<N, T> operator+(T lhs, const TPoint<N, T>& rhs) {
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = lhs + rhs[i];
        return c;
    }

    friend TPoint<N, T> operator-(const TPoint<N, T>& lhs, T rhs) {
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = lhs[i] - rhs;
        return c;
    }

    friend TPoint<N, T> operator-(T lhs, const TPoint<N, T>& rhs) {
        TPoint<N, T> c;
        for (size_t i = 0; i < N; i++)
            c.m[i] = lhs - rhs[i];
        return c;
    }

    TPoint<N, T>& operator+=(const TPoint<N, T>& rhs) {
        for (size_t i = 0; i < N; i++)
            m[i] += rhs[i];
        return *this;
    }

    TPoint<N, T>& operator+=(T scalar) {
        for (size_t i = 0; i < N; i++)
            m[i] += scalar;
        return *this;
    }

    TPoint<N, T>& operator-=(const TPoint<N, T>& rhs) {
        for (size_t i = 0; i < N; i++)
            m[i] -= rhs[i];
        return *this;
    }

    TPoint<N, T>& operator-=(T scalar) {
        for (size_t i = 0; i < N; i++)
            m[i] -= scalar;
        return *this;
    }

    TPoint<N, T>& operator*=(const TPoint<N, T>& rhs) {
        for (size_t i = 0; i < N; i++)
            m[i] *= rhs[i];
        return *this;
    }

    TPoint<N, T>& operator*=(T scalar) {
        for (size_t i = 0; i < N; i++)
            m[i] *= scalar;
        return *this;
    }

    TPoint<N, T>& operator/=(const TPoint<N, T>& rhs) {
        for (size_t i = 0; i < N; i++) {
            assert(rhs[i] != 0);
            m[i] /= rhs[i];
        }
        return *this;
    }

    TPoint<N, T>& operator/=(T scalar) {
        assert(scalar != 0);
        for (size_t i = 0; i < N; i++)
            m[i] /= scalar;
        return *this;
    }

    std::string toString() const {
        std::string result;
        for (size_t i = 0; i < N; ++i) {
            result += std::to_string(this->coeff(i));
            if (i + 1 < N)
                result += ", ";
        }
        return "Point: [" + result + "]";
    }
};

}  // namespace drawlab
