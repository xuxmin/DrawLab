#include "core/math/math.h"
#include "core/math/vector.h"
#include <algorithm>
#include <iostream>

namespace drawlab {

float fresnel(float cosThetaI, float extIOR, float intIOR) {
    float etaI = extIOR, etaT = intIOR;

    if (extIOR == intIOR)
        return 0.0f;

    /* Swap the indices of refraction if the interaction starts
       at the inside of the object */
    if (cosThetaI < 0.0f) {
        std::swap(etaI, etaT);
        cosThetaI = -cosThetaI;
    }

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float eta = etaI / etaT,
          sinThetaTSqr = eta*eta * (1-cosThetaI*cosThetaI);

    if (sinThetaTSqr > 1.0f)
        return 1.0f;  /* Total internal reflection! */

    float cosThetaT = std::sqrt(1.0f - sinThetaTSqr);

    float Rs = (etaI * cosThetaI - etaT * cosThetaT)
             / (etaI * cosThetaI + etaT * cosThetaT);
    float Rp = (etaT * cosThetaI - etaI * cosThetaT)
             / (etaT * cosThetaI + etaI * cosThetaT);

    return (Rs * Rs + Rp * Rp) / 2.0f;
}

Vector3f refract(const Vector3f& wi, float eta) {
    float cosThetaI = wi.z();
    bool outside = cosThetaI > 0.f;
    eta = outside ? eta : 1 / eta;
    float cosThetaTSquare = 1 - eta * eta * (1 - cosThetaI * cosThetaI);
    float cosThetaT = sqrt(cosThetaTSquare);
    cosThetaT = outside ? -cosThetaT : cosThetaT;
    return Vector3f(-eta * wi.x(), -eta * wi.y(), cosThetaT);
}

Vector3f reflect(const Vector3f& wi, const Vector3f n) {
    return 2 * wi.dot(n) * n - wi;
}

};