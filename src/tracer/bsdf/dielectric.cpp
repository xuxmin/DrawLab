#include "optix/material/dielectric.h"
#include "core/base/common.h"
#include "core/math/frame.h"
#include "tracer/bsdf.h"

namespace drawlab {

/**
 *
This plugin models an interface between two dielectric materials having
mismatched indices of refraction (for instance, water ↔ air). Exterior and
interior IOR values can be specified independently, where "exterior" refers to
the side that contains the surface normal. When no parameters are given, the
plugin activates the defaults, which describe a borosilicate glass (BK7) ↔ air
interface.
 *
*/

/// Ideal dielectric BSDF
class Dielectric : public BSDF {
public:
    Dielectric(const PropertyList& propList) {
        /* Interior IOR (default: BK7 borosilicate optical glass) */
        m_intIOR = propList.getFloat("intIOR", 1.5046f);

        /* Exterior IOR (default: air) */
        m_extIOR = propList.getFloat("extIOR", 1.000277f);
    }

    Color3f eval(const BSDFQueryRecord&) const {
        /* Discrete BRDFs always evaluate to zero in Nori */
        return Color3f(0.0f);
    }

    float pdf(const BSDFQueryRecord&) const {
        /* Discrete BRDFs always evaluate to zero in Nori */
        return 0.0f;
    }

    Color3f sample(BSDFQueryRecord& bRec, const Point2f& sample) const {
        bRec.measure = EDiscrete;
        float cosThetai = Frame::cosTheta(bRec.wi);
        float ri = fresnel(cosThetai, m_extIOR, m_intIOR);

        // Reflection in local coordinates
        if (sample.x() < ri) {
            bRec.wo = Vector3f(-bRec.wi.x(), -bRec.wi.y(), bRec.wi.z());
            bRec.eta = 1;
        }
        // refraction(transmission)
        else {
            float eta = m_extIOR / m_intIOR;
            bRec.wo = refract(bRec.wi, eta).normalized();
            bRec.eta = cosThetai > 0.f ? eta : 1 / eta;
        }
        // Actually is: eval()/pdf(), here ri/ri or (1-ri)/(1-ri)
        return Color3f(1.f);
    }

    std::string toString() const {
        return tfm::format("Dielectric[\n"
                           "  intIOR = %f,\n"
                           "  extIOR = %f\n"
                           "]",
                           m_intIOR, m_extIOR);
    }

    void createOptixBSDF(optix::DeviceContext& context,
                         optix::Material& bsdf) const {
        bsdf.type = optix::Material::DIELECTRIC;
        bsdf.dielectric.extIOR = m_extIOR;
        bsdf.dielectric.intIOR = m_intIOR;
        bsdf.diffuse.normal_tex = 0;
        bsdf.is_diffuse = false;
    }

    optix::Material::Type getOptixBSDFType() const {
        return optix::Material::DIELECTRIC;
    }

private:
    float m_intIOR, m_extIOR;
};

REGISTER_CLASS(Dielectric, "dielectric");
}  // namespace drawlab