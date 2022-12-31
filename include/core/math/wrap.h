#pragma once

#include "core/base/common.h"
#include "core/math/math.h"

namespace drawlab {

/// A collection of useful warping functions for importance sampling
class Warp {
public:
    /// Dummy warping function: takes uniformly distributed points in a square
    /// and just returns them
    static Point2f squareToUniformSquare(const Point2f& sample);

    /// Probability density of \ref squareToUniformSquare()
    static float squareToUniformSquarePdf(const Point2f& p);

    /// Sample a 2D tent distribution
    static Point2f squareToTent(const Point2f& sample);

    /// Probability density of \ref squareToTent()
    static float squareToTentPdf(const Point2f& p);

    /// Uniformly sample a vector on a 2D disk with radius 1, centered around
    /// the origin
    static Point2f squareToUniformDisk(const Point2f& sample);

    /// Probability density of \ref squareToUniformDisk()
    static float squareToUniformDiskPdf(const Point2f& p);

    /// Uniformly sample a vector on the unit sphere with respect to solid
    /// angles
    static Vector3f squareToUniformSphere(const Point2f& sample);

    /// Probability density of \ref squareToUniformSphere()
    static float squareToUniformSpherePdf(const Vector3f& v);

    /// Uniformly sample a vector on the unit hemisphere around the pole (0,0,1)
    /// with respect to solid angles
    static Vector3f squareToUniformHemisphere(const Point2f& sample);

    /// Probability density of \ref squareToUniformHemisphere()
    static float squareToUniformHemispherePdf(const Vector3f& v);

    /// Uniformly sample a vector on the unit hemisphere around the pole (0,0,1)
    /// with respect to projected solid angles
    static Vector3f squareToCosineHemisphere(const Point2f& sample);

    /// Probability density of \ref squareToCosineHemisphere()
    static float squareToCosineHemispherePdf(const Vector3f& v);

    /// Warp a uniformly distributed square sample to a Beckmann distribution *
    /// cosine for the given 'alpha' parameter
    static Vector3f squareToBeckmann(const Point2f& sample, float alpha);

    /// Probability density of \ref squareToBeckmann()
    static float squareToBeckmannPdf(const Vector3f& m, float alpha);

    static Vector3f squareToUniformTriangle(const Point2f& sample);
};

}  // namespace drawlab
