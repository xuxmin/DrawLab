#pragma once

#include "core/base/common.h"
#include "core/math/math.h"
#include "core/math/vector.h"

namespace drawlab {

/**
 * \brief Represents a linear RGB color value
 */
struct Color3f : public Vector3f {
public:
    typedef Vector3f Base;

    /// Initialize the color vector with a uniform value
    Color3f(float value = 0.f) : Base(value) {}

    /// Initialize the color vector with specific per-channel values
    Color3f(float r, float g, float b) : Base(r, g, b) {}

    /// @brief Construct a color vector from vector
    Color3f(const Vector3f& p) : Base(p) {}
    Color3f(const Color3f& p) : Base(p) {}

    Color3f& operator=(const Vector3f& p) {
        this->Base::operator=(p);
        return *this;
    }

    float sum() const { return m[0] + m[1] + m[2]; }

    /// Return a reference to the red channel
    float& r() { return m[0]; }
    /// Return a reference to the red channel (const version)
    const float& r() const { return m[0]; }
    /// Return a reference to the green channel
    float& g() { return m[1]; }
    /// Return a reference to the green channel (const version)
    const float& g() const { return m[1]; }
    /// Return a reference to the blue channel
    float& b() { return m[2]; }
    /// Return a reference to the blue channel (const version)
    const float& b() const { return m[2]; }

    /// Clamp to the positive range
    Color3f clamp() const {
        return Color3f(max(r(), 0.0f), max(g(), 0.0f), max(b(), 0.0f));
    }

    /// Check if the color vector contains a NaN/Inf/negative value
    bool isValid() const {
        for (int i = 0; i < 3; i++) {
            float value = this->coeff(i);
            if (value < 0 || !std::isfinite(value)) {
                return false;
            }
        }
        return true;
    }

    /// Convert from sRGB to linear RGB
    Color3f toLinearRGB() const {
        Color3f result;
        for (int i = 0; i < 3; ++i) {
            float value = coeff(i);
            if (value <= 0.04045f)
                result[i] = value * (1.0f / 12.92f);
            else
                result[i] = std::pow((value + 0.055f) * (1.0f / 1.055f), 2.4f);
        }
        return result;
    }

    /// Convert from linear RGB to sRGB
    Color3f toSRGB() const {
        Color3f result;
        for (int i = 0; i < 3; ++i) {
            float value = coeff(i);
            if (value <= 0.0031308f)
                result[i] = 12.92f * value;
            else
                result[i] =
                    (1.0f + 0.055f) * std::pow(value, 1.0f / 2.4f) - 0.055f;
        }
        return result;
    }

    /// Return the associated luminance
    float getLuminance() const {
        return coeff(0) * 0.212671f + coeff(1) * 0.715160f +
               coeff(2) * 0.072169f;
    }

    /// Return a human-readable string summary
    std::string toString() const {
        return tfm::format("Color [%f, %f, %f]", coeff(0), coeff(1), coeff(2));
    }
};

/**
 * \brief Represents a linear RGB color and a weight
 *
 * This is used by Nori's image reconstruction filter code
 */
struct Color4f : public Vector4f {
public:
    typedef Vector4f Base;

    /// Create an zero value
    Color4f() : Base(0.0f, 0.0f, 0.0f, 0.0f) {}

    /// Create from a 3-channel color
    Color4f(const Color3f& c) : Base(c.r(), c.g(), c.b(), 1.0f) {}

    /// Initialize the color vector with specific per-channel values
    Color4f(float r, float g, float b, float w) : Base(r, g, b, w) {}

    /// Divide by the filter weight and convert into a \ref Color3f value
    Color3f divideByFilterWeight() const {
        if (w() != 0)
            return Color3f(x(), y(), z()) / w();
        else
            return Color3f(0.0f);
    }

    /// Return a human-readable string summary
    std::string toString() const {
        return tfm::format("[%f, %f, %f, %f]", coeff(0), coeff(1), coeff(2),
                           coeff(3));
    }
};

}  // namespace drawlab