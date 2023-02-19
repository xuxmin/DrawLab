#include "tracer/bsdf.h"
#include "tracer/emitter.h"
#include "tracer/integrator.h"
#include "tracer/sampler.h"
#include "tracer/scene.h"

namespace drawlab {

class PathIntegrator : public Integrator {
public:
    PathIntegrator(const PropertyList& props) {}

    Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray_) const {
        Ray3f ray(ray_);

        // Tracks radiance scaling due to index of refraction changes
        float eta = 1.f;

        Color3f result(0.f), throughput(1.f);

        // ---------------------- First intersection ----------------------
        Intersection its;
        if (!scene->rayIntersect(ray, its)) {
            return result;
        }
        // self-luminous
        if (its.mesh->isEmitter()) {
            result += throughput * its.mesh->getEmitter()->eval(its, -ray.d);
        }

        for (int depth = 0;; ++depth) {
            // --------------------- Emitter sampling ---------------------

            const BSDF* bsdf = its.mesh->getBSDF();

            DirectionSample ds;
            Color3f emitter_val;
            scene->sampleEmitterDirection(its, sampler->next2D(), ds,
                                          emitter_val);

            // Query the BSDF for that emitter-sampled direction
            const Vector3f wi = its.toLocal(-ray.d.normalized());
            const Vector3f wo = its.toLocal(ds.d);
            BSDFQueryRecord bRec(its, wi, wo, ESolidAngle);

            if (ds.pdf > 0) {
                Color3f bsdf_val = bsdf->eval(bRec);

                // Determine density of sampling that same direction using BSDF
                // sampling
                float bsdf_pdf = bsdf->pdf(bRec);
                float emitter_pdf = ds.pdf;
                float mis = mis_weight(emitter_pdf, bsdf_pdf);

                result += mis * throughput * bsdf_val * emitter_val;
            }

            // ----------------------- BSDF sampling ----------------------

            BSDFQueryRecord bsdf_bRec(its, wi);
            Color3f fr = bsdf->sample(bsdf_bRec, sampler->next2D());

            // Update throughput, eta
            throughput *= fr;
            eta *= bsdf_bRec.eta;

            // BSDF sampled ray direction, also be used as the next path
            // direction
            ray = Ray3f(its.p, its.toWorld(bsdf_bRec.wo));

            // Intersection light_its;
            if (scene->rayIntersect(ray, its)) {
                if (its.mesh->isEmitter()) {
                    const Emitter* emitter = its.mesh->getEmitter();
                    Color3f emitter_val = emitter->eval(its, -ray.d);

                    DirectionSample ds(ray.o, its.p, its.shFrame.n, emitter);
                    float emitter_pdf = scene->pdfEmitterDirection(ds);
                    float bsdf_pdf = bsdf->pdf(bsdf_bRec);
                    float mis = bsdf->isDiffuse() ?
                                    mis_weight(bsdf_pdf, emitter_pdf) :
                                    1.f;
                    result += mis * throughput * emitter_val;
                }
            } else {
                break;
            }

            // Russian Roulette
            if (depth > 3) {
                float p =
                    std::min((throughput * eta * eta).maxCoeff(), (float)0.99);
                if (sampler->next1D() > p) {
                    break;
                }
                throughput /= p;
            }
        }
        return result;
    }

    std::string toString() const { return "PathIntegrator[]"; }

    float mis_weight(float pdf_a, float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return pdf_a / (pdf_a + pdf_b);
    }

    optix::Integrator getOptixIntegrator() const {
        optix::Integrator integrator;
        integrator.type = optix::Integrator::PATH;
        return integrator;
    }

protected:
    std::string m_myProperty;
};

REGISTER_CLASS(PathIntegrator, "path");

}  // namespace drawlab