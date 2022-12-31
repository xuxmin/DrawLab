#pragma once

#if defined(_MSC_VER)
#    undef max
#    undef min
#endif

#include <stdlib.h>
#include "core/math/vector.h"
#include "core/math/point.h"
#include "core/math/normal.h"

namespace drawlab {

/* "Ray epsilon": relative error threshold for ray intersection computations */
static const float M_EPSILON = 0.0001f;

#undef M_PI
static const float M_PI = 3.14159265358979323846f;
static const float M_INV_PI = 0.31830988618379067154f;
static const float M_INV_TWOPI = 0.15915494309189533577f;
static const float M_INV_FOURPI = 0.07957747154594766788f;
static const float M_SQRT_TWO = 1.41421356237309504880f;
static const float M_INV_SQRT_TWO = 0.70710678118654752440f;

template <typename T> T max(T a, T b) { return a > b ? a : b; }

inline float clamp(float value, float min, float max) {
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return value;
}

//// Convert radians to degrees
inline float radToDeg(float value) { return value * (180.0f / M_PI); }

/// Convert degrees to radians
inline float degToRad(float value) { return value * (M_PI / 180.0f); }

/**
 * \brief Calculates the unpolarized fresnel reflection coefficient for a
 * dielectric material. Handles incidence from either side (i.e.
 * \code cosThetaI<0 is allowed).
 *
 * \param cosThetaI
 *      Cosine of the angle between the normal and the incident ray
 * \param extIOR
 *      Refractive index of the side that contains the surface normal
 * \param intIOR
 *      Refractive index of the interior
 */
extern float fresnel(float cosThetaI, float extIOR, float intIOR);

/**
 * \brief Refraction in local coordinates
 * \param eta  m_extIOR / m_intIOR
 */
extern Vector3f refract(const Vector3f& wi, float eta);

extern Vector3f reflect(const Vector3f& wi, const Vector3f n);

};  // namespace drawlab