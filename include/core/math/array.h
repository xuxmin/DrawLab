#pragma once

#include <cmath>
#include <assert.h>
#include <string>

namespace drawlab {

/**
 * \brief Generic N-dimensional array data structure
 */
template <size_t N, typename T> struct TArray {
public:
    T m[N];

    /// @brief Create a new array with constant component vlaues
    TArray(T u = (T)0) {
        for (size_t i = 0; i < N; i++)
            m[i] = u;
    }

    TArray(const TArray<N, T>& u) {
        for (size_t i = 0; i < N; i++)
            m[i] = u.m[i];
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

    bool operator==(const TArray<N, T>& rhs) const {
        for (size_t i = 0; i < N; i++) {
            if (m[i] != rhs[i])
                return false;
        }
        return true;
    }

    bool operator!=(const TArray<N, T>& rhs) const {
        for (size_t i = 0; i < N; i++) {
            if (m[i] != rhs[i])
                return true;
        }
        return false;
    }

    /// @brief Check whether all elements are unequal to zero.
    bool all() const {
        for (size_t i = 0; i < N; i++) {
            if ((float)m[i] == 0) {
                return false;
            }
        }
        return true;
    }

    std::string toString() const {
        std::string result;
        for (size_t i = 0; i < N; ++i) {
            result += std::to_string(this->coeff(i));
            if (i + 1 < N)
                result += ", ";
        }
        return "Array: [" + result + "]";
    }
};

};  // namespace drawlab
