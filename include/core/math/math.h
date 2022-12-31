#pragma once

#if defined(_MSC_VER)
#    undef max
#    undef min
#endif

namespace drawlab {

/* "Ray epsilon": relative error threshold for ray intersection computations */
static const float M_EPSILON = 0.0001f;

static const float M_PI = 3.14159265358979323846f;
static const float M_INV_PI = 0.31830988618379067154f;
static const float M_INV_TWOPI = 0.15915494309189533577f;
static const float M_INV_FOURPI = 0.07957747154594766788f;
static const float M_SQRT_TWO = 1.41421356237309504880f;
static const float M_INV_SQRT_TWO = 0.70710678118654752440f;

/* Forward declarations */
template <size_t N, typename T> struct TVector;
template <size_t N, typename T> struct TPoint;
template <typename T> struct TNormal3;

/* Basic data structures (vectors, points, rays, bounding boxes,
   kd-trees) are oblivious to the underlying data type and dimension.
   The following list of typedefs establishes some convenient aliases
   for specific types. */
typedef TNormal3<float> Normal3f;
typedef TNormal3<double> Normal3d;
typedef TVector<2, float> Vector2f;
typedef TVector<3, float> Vector3f;
typedef TVector<4, float> Vector4f;
typedef TVector<2, double> Vector2d;
typedef TVector<3, double> Vector3d;
typedef TVector<4, double> Vector4d;
typedef TVector<2, int> Vector2i;
typedef TVector<3, int> Vector3i;
typedef TVector<4, int> Vector4i;
typedef TPoint<2, float> Point2f;
typedef TPoint<3, float> Point3f;
typedef TPoint<4, float> Point4f;
typedef TPoint<2, double> Point2d;
typedef TPoint<3, double> Point3d;
typedef TPoint<4, double> Point4d;
typedef TPoint<2, int> Point2i;
typedef TPoint<3, int> Point3i;
typedef TPoint<4, int> Point4i;

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