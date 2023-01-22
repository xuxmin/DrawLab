#pragma once

#include "core/math/math.h"
#include "core/math/point.h"
#include "core/math/vector.h"
#include <tinyformat.h>

namespace drawlab {

/**
 * \brief Simple n-dimensional ray segment data structure
 *
 * Along with the ray origin and direction, this data structure additionally
 * stores a ray segment [mint, maxt] (whose entries may include
 * positive/negative infinity), as well as the componentwise reciprocals of the
 * ray direction. That is just done for convenience, as these values are
 * frequently required.
 *
 * \remark Important: be careful when changing the ray direction. You must
 * call \ref update() to compute the componentwise reciprocals as well.
 */
template <size_t N, typename T> struct TRay {
    typedef TPoint<N, T> PointType;
    typedef TVector<N, T> VectorType;
    typedef T Scalar;

    PointType o;      // Ray origin
    VectorType d;     // Ray direction
    VectorType dRcp;  // Componentwise reciprocals of the ray direction
    Scalar mint;      // Minimum position on the segment
    Scalar maxt;      // Maximum position on the segment

    TRay() : mint(M_EPSILON), maxt(std::numeric_limits<Scalar>::infinity()) {}

    TRay(const PointType& o, const VectorType& d)
        : o(o), d(d), mint(M_EPSILON),
          maxt(std::numeric_limits<Scalar>::infinity()) {
        update();
    }

    TRay(const PointType& o, const VectorType& d, Scalar mint, Scalar maxt)
        : o(o), d(d), mint(mint), maxt(maxt) {
        update();
    }

    /// Copy constructor
    TRay(const TRay& ray)
        : o(ray.o), d(ray.d), dRcp(ray.dRcp), mint(ray.mint), maxt(ray.maxt) {}

    /// Copy a ray, but change the covered segment of the copy
    TRay(const TRay& ray, Scalar mint, Scalar maxt)
        : o(ray.o), d(ray.d), dRcp(ray.dRcp), mint(mint), maxt(maxt) {}

    /// Update the reciprocal ray directions after changing 'd'
    void update() { dRcp = d.cwiseInverse(); }

    /// Return the position of a point along the ray
    PointType operator()(Scalar t) const { return o + t * d; }

    /// Return a ray that points into the opposite direction
    TRay reverse() const {
        TRay result;
        result.o = o;
        result.d = -d;
        result.dRcp = -dRcp;
        result.mint = mint;
        result.maxt = maxt;
        return result;
    }

    /// Return a human-readable string summary of this ray
    std::string toString() const {
        return tfm::format("Ray[\n"
                           "  o = %s,\n"
                           "  d = %s,\n"
                           "  mint = %f,\n"
                           "  maxt = %f\n"
                           "]",
                           o.toString(), d.toString(), mint, maxt);
    }
};

typedef TRay<2, float> Ray2f;
typedef TRay<3, float> Ray3f;

}  // namespace drawlab