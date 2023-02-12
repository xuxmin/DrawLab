#include "optix/material/diffuse.h"
#include "core/base/common.h"
#include "core/math/frame.h"
#include "core/math/wrap.h"
#include "optix/host/device_context.h"
#include "tracer/bsdf.h"
#include "tracer/texture.h"

namespace drawlab {

/**
 * \brief Diffuse / Lambertian BRDF model
 */
class Diffuse : public BSDF {
public:
    Diffuse(const PropertyList& propList) {
        Color3f constant_albedo = propList.getColor("albedo", Color3f(0.5f));
        m_albedo = new ConstantTexture(constant_albedo);
    }

    /// Evaluate the BRDF model
    Color3f eval(const BSDFQueryRecord& bRec) const {
        /* This is a smooth BRDF -- return zero if the measure
           is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0)
            return Color3f(0.0f);

        /* The BRDF is simply the albedo / pi */
        return m_albedo->eval(bRec.its) * M_INV_PI;
    }

    /// Compute the density of \ref sample() wrt. solid angles
    float pdf(const BSDFQueryRecord& bRec) const {
        /* This is a smooth BRDF -- return zero if the measure
           is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0)
            return 0.0f;

        /* Importance sampling density wrt. solid angles:
           cos(theta) / pi.

           Note that the directions in 'bRec' are in local coordinates,
           so Frame::cosTheta() actually just returns the 'z' component.
        */
        return M_INV_PI * Frame::cosTheta(bRec.wo);
    }

    /// Draw a a sample from the BRDF model
    Color3f sample(BSDFQueryRecord& bRec, const Point2f& sample) const {
        if (Frame::cosTheta(bRec.wi) <= 0)
            return Color3f(0.0f);

        bRec.measure = ESolidAngle;

        /* Warp a uniformly distributed sample on [0,1]^2
           to a direction on a cosine-weighted hemisphere */
        bRec.wo = Warp::squareToCosineHemisphere(sample);

        /* Relative index of refraction: no change */
        bRec.eta = 1.0f;

        /* eval() / pdf() * cos(theta) = albedo. There
           is no need to call these functions. */
        return m_albedo->eval(bRec.its);
    }

    void addChild(Object* child) {
        if (child->getClassType() == ETexture) {
            m_albedo = static_cast<Texture*>(child);
        }
        else {
            BSDF::addChild(child);
        }
    }

    bool isDiffuse() const { return true; }

    /// Return a human-readable summary
    std::string toString() const {
        return tfm::format("Diffuse[\n"
                           "  albedo = %s\n"
                           "]",
                           m_albedo->toString());
    }

    EClassType getClassType() const { return EBSDF; }

    const optix::Material*
    createOptixMaterial(optix::DeviceContext& context) const {
        optix::Material* optix_mat = nullptr;
        if (m_albedo->isConstant()) {
            Color3f color = m_albedo->eval(Intersection());
            optix_mat = new optix::Diffuse(
                "diffuse", context,
                make_float4(color[0], color[1], color[2], 1.f));
        }
        else {
            optix_mat = new optix::Diffuse("diffuse", context,
                                           m_albedo->getOptixTexture(context));
        }
        return optix_mat;
    }

private:
    // Color3f m_albedo;
    const Texture* m_albedo;
};

REGISTER_CLASS(Diffuse, "diffuse");

}  // namespace drawlab
