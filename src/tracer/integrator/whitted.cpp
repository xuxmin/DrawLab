#include "tracer/bsdf.h"
#include "tracer/emitter.h"
#include "tracer/integrator.h"
#include "tracer/sampler.h"
#include "tracer/scene.h"

namespace drawlab {

/**
 * Whitted-style ray tracing
 *
 */

class WhittedIntegrator : public Integrator {
public:
    WhittedIntegrator(const PropertyList& props) {}

    Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            return Color3f(0.0f);

        Color3f Le(0.f), Lr(0.f);

        // self-luminous
        if (its.mesh->isEmitter()) {
            Le = its.mesh->getEmitter()->eval(its, -ray.d);
        }

        float RR = 0.95;
        if (sampler->next1D() >= RR) {
            return Color3f(0.f);
        }

        if (its.mesh->getBSDF()->isDiffuse()) {
            // Sample emitter
            DirectionSample ds;
            Color3f spectrum;
            scene->sampleEmitterDirection(its, sampler->next2D(), ds, spectrum);

            // Query the BSDF for that emitter-sampled direction
            const Vector3f wo = its.toLocal(ds.d);
            const Vector3f wi = its.toLocal(-ray.d.normalized());
            BSDFQueryRecord bRec(wi, wo, ESolidAngle);
            Color3f fr = its.mesh->getBSDF()->eval(bRec);

            Lr = spectrum * fr;
        } else {
            // Sample bsdf
            const Vector3f wi = its.toLocal(-ray.d.normalized());
            BSDFQueryRecord bRec(wi);
            Color3f fr = its.mesh->getBSDF()->sample(bRec, sampler->next2D());

            Lr = Li(scene, sampler, Ray3f(its.p, its.toWorld(bRec.wo))) * fr;
        }
        return (Le + Lr) / RR;
    }

    std::string toString() const { return "WhittedIntegrator[]"; }

protected:
    std::string m_myProperty;
};

REGISTER_CLASS(WhittedIntegrator, "whitted");

}  // namespace drawlab