#pragma once

#include "core/math/vector.h"
#include <cmath>
#include <string>

namespace drawlab {

struct Matrix4f {
    float m[4][4];

    Matrix4f() {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m[i][j] = 0;
            }
        }
    }

    Matrix4f(const std::initializer_list<float>& u) {
        auto it = u.begin();
        for (size_t i = 0; i < 4; i++)
            for (size_t j = 0; j < 4; j++)
                m[i][j] = *it++;
    }

    Matrix4f(const Matrix4f& src) {
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                m[i][j] = src.m[i][j];
            }
        }
    }

    const float* operator[](size_t row) const {
        assert(row < 4);
        return m[row];
    }

    float* operator[](size_t row) {
        assert(row < 4);
        return m[row];
    }

    /// @brief Get a row
    TVector<4, float> row(size_t row) const {
        assert(row < 4);
        TVector<4, float> a;
        for (size_t i = 0; i < 4; i++)
            a[i] = m[row][i];
        return a;
    }

    /// @brief Get a col
    TVector<4, float> col(size_t col) const {
        assert(col < 4);
        TVector<4, float> a;
        for (size_t i = 0; i < 4; i++)
            a[i] = m[i][col];
        return a;
    }

    void setRow(size_t row, const TVector<4, float>& a) {
        assert(row < 4);
        for (size_t i = 0; i < 4; i++)
            m[row][i] = a[i];
    }

    void setCol(size_t col, const TVector<4, float>& a) {
        assert(col < 4);
        for (size_t i = 0; i < 4; i++)
            m[i][col] = a[i];
    }

    bool operator==(const Matrix4f& rhs) const {
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                if (m[i][j] != rhs[i][j])
                    return false;
            }
        }
        return true;
    }

    bool operator!=(const Matrix4f& rhs) const {
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                if (m[i][j] != rhs[i][j])
                    return true;
            }
        }
        return false;
    }

    /// @brief Get identity matrix
    static Matrix4f getIdentityMatrix() {
        Matrix4f ret = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        return ret;
    }

    static Matrix4f getTranslationMatrix(const Vector3f& vec) {
        Matrix4f ret = {1, 0, 0, vec[0], 0, 1, 0, vec[1],
                        0, 0, 1, vec[2], 0, 0, 0, 1};
        return ret;
    }

    static Matrix4f getScaleMatrix(const Vector3f& scale) {
        Matrix4f ret = {scale[0], 0, 0,        0, 0, scale[1], 0, 0,
                        0,        0, scale[2], 0, 0, 0,        0, 1};
        return ret;
    }

    static Matrix4f getRotateMatrix(const Vector3f& axis, const float& angle) {
        float x = axis[0], y = axis[1], z = axis[2];
        float cos_t = std::cos(angle);
        float sin_t = std::sin(angle);
        Matrix4f ret = {cos_t + (1 - cos_t) * x * x,
                        (1 - cos_t) * x * y - sin_t * z,
                        (1 - cos_t) * x * z + sin_t * y,
                        0,
                        (1 - cos_t) * y * x + sin_t * z,
                        cos_t + (1 - cos_t) * y * y,
                        (1 - cos_t) * y * z - sin_t * x,
                        0,
                        (1 - cos_t) * z * x - sin_t * y,
                        (1 - cos_t) * z * y + sin_t * x,
                        cos_t + (1 - cos_t) * z * z,
                        0,
                        0,
                        0,
                        0,
                        1};
        return ret;
    }

    static Matrix4f getLookAtMatrix(const Vector3f& eye, const Vector3f& center,
                                    const Vector3f& up) {
        Vector3f z = (eye - center).normalized();
        Vector3f x = up.normalized().cross(z).normalized();
        Vector3f y = z.cross(x).normalized();

        Matrix4f M = Matrix4f::getIdentityMatrix();

        for (int i = 0; i < 3; i++) {
            M[0][i] = x[i];
            M[1][i] = y[i];
            M[2][i] = z[i];
        }

        M[0][3] = -x.dot(eye);
        M[1][3] = -y.dot(eye);
        M[2][3] = -z.dot(eye);

        return M;
    }

    static Matrix4f getPerspectiveMatrix(float fovy, float aspect, float zNear,
                                         float zFar) {
        float z_range = zFar - zNear;
        assert(z_range > 0 && fovy > 0 && aspect > 0 && zNear > 0 && zFar > 0);

        Matrix4f M = Matrix4f::getIdentityMatrix();

        M[1][1] = 1.0f / (float)std::tan(fovy / 2);
        M[0][0] = M[1][1] / aspect;
        M[2][2] = -(zFar + zNear) / z_range;
        M[2][3] = -2.0f * zFar * zNear / z_range;
        M[3][2] = -1;
        M[3][3] = 0;

        return M;
    }

    Matrix4f transpose() const {
        Matrix4f ret;
        for (size_t r = 0; r < 4; r++) {
            for (size_t c = 0; c < 4; c++)
                ret.m[c][r] = m[r][c];
        }
        return ret;
    }

    std::string toString() const {
        std::string result;
        result += "[";
        for (size_t i = 0; i < 4; ++i) {
            result += "[";
            for (size_t j = 0; j < 4; j++) {
                result += std::to_string(m[i][j]);
                if (j < 3)
                    result += ", ";
            }
            result += "]\n";
        }
        result += "]";
        return result;
    }

    //----------------------------------------------------------------------
    // Matrix Operation
    //----------------------------------------------------------------------

    /// Get the inverse matrix
    Matrix4f inv() const;

    Matrix4f operator*(const Matrix4f& b) {
        Matrix4f out;
        for (size_t j = 0; j < 4; j++) {
            for (size_t i = 0; i < 4; i++) {
                out.m[j][i] = row(j).dot(b.col(i));
            }
        }
        return out;
    }

    friend Matrix4f operator*(const Matrix4f& a, const Matrix4f& b) {
        Matrix4f out;
        for (size_t j = 0; j < 4; j++) {
            for (size_t i = 0; i < 4; i++) {
                out.m[j][i] = a.row(j).dot(b.col(i));
            }
        }
        return out;
    }

    Matrix4f operator*(float x) {
        Matrix4f out;
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                out.m[i][j] = m[i][j] * x;
            }
        }
        return out;
    }

    friend Matrix4f operator*(float x, const Matrix4f& a) {
        Matrix4f out;
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                out.m[i][j] = a[i][j] * x;
            }
        }
        return out;
    }

    Matrix4f operator/(float x) {
        Matrix4f out;
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                out.m[i][j] = m[i][j] / x;
            }
        }
        return out;
    }

    friend TVector<4, float> operator*(const TVector<4, float>& a,
                                       const Matrix4f& m) {
        TVector<4, float> b;
        for (size_t i = 0; i < 4; i++) {
            b[i] = a.dot(m.col(i));
        }
        return b;
    }

    friend TVector<4, float> operator*(const Matrix4f& m,
                                       const TVector<4, float>& a) {
        TVector<4, float> b;
        for (size_t i = 0; i < 4; i++)
            b[i] = a.dot(m.row(i));
        return b;
    }
};

}  // namespace drawlab