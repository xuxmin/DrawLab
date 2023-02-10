#include "tracer/emitter.h"

namespace drawlab {

class AreaLight final : public Emitter {
public:
    AreaLight(const PropertyList& propList) {
        m_radiance = propList.getColor("radiance");
    }

    void sampleDirection(const Point2f& sample, const Intersection& its,
                         DirectionSample& ds,
                         Color3f& spectrum) const override {
        if (m_mesh == nullptr) {
            throw Exception("AreaLight: can't sample from an area emitter "
                            "without an associated Shape.");
        }
        // Texture is uniform, try to importance sample the shape wrt. solid
        // angle at 'its'
        Point3f position;
        Normal3f normal;
        m_mesh->samplePosition(sample, position, normal);

        ds = DirectionSample(its.p, position, normal);

        // pdf conversation: pA = pw * |cos(-d·ln)| / ||lp - xp||^2
        float pA = m_mesh->pdfPosition();
        float dp = std::max(normal.dot(-ds.d), (float)0);
        float pw = dp != 0 ? pA * ds.dist * ds.dist / dp : 0;
        ds.pdf = pw;

        // spectrum = (Le * |cos(d·xn)|) / pw
        spectrum =
            pw != 0 ?
                m_radiance * std::max(its.shFrame.n.dot(ds.d), (float)0) / pw :
                Color3f(0.f);
    }

    float pdfDirection(const DirectionSample& ds) const override {
        float pA = m_mesh->pdfPosition();
        float dp = std::max(ds.n.dot(-ds.d), (float)0);
        float pw = dp != 0 ? pA * ds.dist * ds.dist / dp : 0;
        return pw;
    }

    Color3f eval(const Intersection& its, const Vector3f wi) const {
        float cosTheta = its.toLocal(wi).z();
        return cosTheta > 0.f ? m_radiance : Color3f(0, 0, 0);
    }

    std::string toString() const {
        return tfm::format("AreaLight[\n"
                           "m_radiance = \"%f,%f,%f\"\n"
                           "]",
                           m_radiance.x(), m_radiance.y(), m_radiance.z());
    }

    void getOptixLight(optix::Light& light) const {
        light.type = optix::Light::Type::AREA;
        light.area.intensity = make_float3(m_radiance[0], m_radiance[1], m_radiance[2]);
    }

private:
    Color3f m_radiance;
};

REGISTER_CLASS(AreaLight, "area");

}  // namespace drawlab