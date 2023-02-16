
#include "core/base/common.h"
#include "core/math/frame.h"
#include "core/math/wrap.h"
#include "tracer/bsdf.h"

namespace drawlab {

class Microfacet : public BSDF {
public:
    Microfacet(const PropertyList& propList) {
        /* RMS surface roughness */
        m_alpha = propList.getFloat("alpha", 0.1f);

        /* Interior IOR (default: BK7 borosilicate optical glass) */
        m_intIOR = propList.getFloat("intIOR", 1.5046f);

        /* Exterior IOR (default: air) */
        m_extIOR = propList.getFloat("extIOR", 1.000277f);

        /* Albedo of the diffuse base material (a.k.a "kd") */
        m_kd = propList.getColor("kd", Color3f(0.5f));

        /* To ensure energy conservation, we must scale the
           specular component by 1-kd.

           While that is not a particularly realistic model of what
           happens in reality, this will greatly simplify the
           implementation. Please see the course staff if you're
           interested in implementing a more realistic version
           of this BRDF. */
        m_ks = 1 - m_kd.maxCoeff();
    }

    static float G1(const Vector3f& wv, const Vector3f& wh, float alpha) {
        float c = wv.dot(wh) / Frame::cosTheta(wv);
        if (c <= 0) {
            return float(0);
        }
        float b = 1.0 / (alpha * Frame::tanTheta(wv));
        return b < 1.6 ? (3.535 * b + 2.181 * b * b) /
                             (1 + 2.276 * b + 2.577 * b * b) :
                         1;
    }

    /// Evaluate the BRDF for the given pair of directions
    Color3f eval(const BSDFQueryRecord& bRec) const {
        Vector3f wh = (bRec.wi + bRec.wo).normalized();

        float D = Warp::squareToBeckmannPdf(wh, m_alpha);
        // Microfacet model, note cos(wi, wh) not cos(wi, N)
        float F = fresnel(bRec.wi.dot(wh), m_extIOR, m_intIOR);
        float G = G1(bRec.wi, wh, m_alpha) * G1(bRec.wo, wh, m_alpha);
        float deno =
            1 / (4 * Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo));

        return m_kd * M_INV_PI + m_ks * D * F * G * deno;
    }

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdf(const BSDFQueryRecord& bRec) const {
        if (bRec.measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0)
            return 0.0f;

        Vector3f wh = (bRec.wi + bRec.wo).normalized();

        float diff_pdf =
            (1 - m_ks) * Warp::squareToCosineHemispherePdf(bRec.wo);
        float spec_pdf =
            m_ks * Warp::squareToBeckmannPdf(wh, m_alpha) / 4 / wh.dot(bRec.wo);
        return diff_pdf + spec_pdf;
    }

    /// Sample the BRDF
    Color3f sample(BSDFQueryRecord& bRec, const Point2f& _sample) const {
        if (Frame::cosTheta(bRec.wi) <= 0)
            return Color3f(0.0f);

        bRec.measure = ESolidAngle;

        Point2f sample(_sample);
        float sample_x = sample.x();
        sample.x() = abs(sample_x - m_ks) / (1 - m_ks);

        // diffuse case
        if (sample_x > m_ks) {
            sample.x() = (sample_x - m_ks) / (1 - m_ks);
            bRec.wo = Warp::squareToCosineHemisphere(sample);
        }
        else {
            sample.x() = sample_x / m_ks;
            Vector3f wh = Warp::squareToBeckmann(sample, m_alpha);
            bRec.wo = reflect(bRec.wi, wh);
        }
        if (Frame::cosTheta(bRec.wo) <= 0) {
            return Color3f(0.0f);
        }
        // Note: Once you have implemented the part that computes the scattered
        // direction, the last part of this function should simply return the
        // BRDF value divided by the solid angle density and multiplied by the
        // cosine factor from the reflection equation, i.e.
        return eval(bRec) * Frame::cosTheta(bRec.wo) / pdf(bRec);
    }

    bool isDiffuse() const {
        /* While microfacet BRDFs are not perfectly diffuse, they can be
           handled by sampling techniques for diffuse/non-specular materials,
           hence we return true here */
        return true;
    }

    std::string toString() const {
        return tfm::format("Microfacet[\n"
                           "  alpha = %f,\n"
                           "  intIOR = %f,\n"
                           "  extIOR = %f,\n"
                           "  kd = %s,\n"
                           "  ks = %f\n"
                           "]",
                           m_alpha, m_intIOR, m_extIOR, m_kd.toString(), m_ks);
    }

    void createOptixMaterial(
        optix::Material& mat,
        std::vector<const optix::CUDATexture*>& textures) const {
        mat.is_diffuse = false;
        mat.type = optix::Material::MICROFACET;
    }

    optix::Material::Type getOptixMaterialType() const {
        return optix::Material::MICROFACET;
    }

private:
    float m_alpha;
    float m_intIOR, m_extIOR;
    float m_ks;
    Color3f m_kd;
};

REGISTER_CLASS(Microfacet, "microfacet");

}  // namespace drawlab
