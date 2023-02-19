#include "tracer/scene.h"
#include <spdlog/spdlog.h>

namespace drawlab {

Scene::Scene(const PropertyList&) { m_accel = new OCTree(); }

Scene::~Scene() {
    delete m_accel;
    delete m_integrator;
    delete m_sampler;
    delete m_camera;
}

void Scene::activate() {
    m_accel->build();

    if (!m_integrator)
        throw Exception("No integrator was specified!");
    if (!m_camera)
        throw Exception("No camera was specified!");

    if (!m_sampler) {
        /* Create a default (independent) sampler */
        m_sampler = static_cast<Sampler*>(
            ObjectFactory::createInstance("independent", PropertyList()));
    }

    spdlog::info("Configuration: {}", toString());
}

void Scene::addChild(Object* obj) {
    switch (obj->getClassType()) {
        case EMesh: {
            Mesh* mesh = static_cast<Mesh*>(obj);
            m_accel->addMesh(mesh);
            
            // Add mesh
            m_meshes.push_back(mesh);

            // Add emitter
            if (mesh->isEmitter()) {
                m_mesh_light_idx.push_back(m_emitters.size());
                m_emitters.push_back(mesh->getEmitter());
            }
            else {
                m_mesh_light_idx.push_back(-1);
            }

            // Add BSDF
            int bsdf_idx = 0;
            for (bsdf_idx = 0; bsdf_idx < m_bsdfs.size(); bsdf_idx++) {
                if (mesh->getBSDF() == m_bsdfs[bsdf_idx]) {
                    m_mesh_bsdf_idx.push_back(bsdf_idx);
                }
            }
            if (bsdf_idx == m_bsdfs.size()) {
                m_mesh_bsdf_idx.push_back(bsdf_idx);
                m_bsdfs.push_back(mesh->getBSDF());
            }

            if (mesh->isEmitter()) {
                m_light_bsdf_idx.push_back(bsdf_idx);
            }
            break;
        }
        case EEmitter: {
            Emitter* emitter = static_cast<Emitter*>(obj);
            m_emitters.push_back(emitter);
            break;
        }
        case ESampler: {
            if (m_sampler)
                throw Exception("There can only be one sampler per scene!");
            m_sampler = static_cast<Sampler*>(obj);
            break;
        }
        case ECamera: {
            if (m_camera)
                throw Exception("There can only be one camera per scene!");
            m_camera = static_cast<Camera*>(obj);
            break;
        }
        case EIntegrator:
            if (m_integrator)
                throw Exception("There can only be one integrator per scene!");
            m_integrator = static_cast<Integrator*>(obj);
            break;

        default:
            throw Exception("Scene::addChild(<%s>) is not supported!",
                            classTypeName(obj->getClassType()));
    }
}

std::string Scene::toString() const {
    std::string meshes;
    for (size_t i = 0; i < m_meshes.size(); ++i) {
        meshes += std::string("  ") + indent(m_meshes[i]->toString(), 2);
        if (i + 1 < m_meshes.size())
            meshes += ",";
        meshes += "\n";
    }

    std::string emitters;
    for (size_t i = 0; i < m_emitters.size(); ++i) {
        emitters += std::string("  ") + indent(m_emitters[i]->toString(), 2);
        if (i + 1 < m_emitters.size())
            emitters += ",";
        emitters += "\n";
    }

    return tfm::format("Scene[\n"
                       "  integrator = %s,\n"
                       "  sampler = %s\n"
                       "  camera = %s,\n"
                       "  meshes = {\n"
                       "  %s  }\n"
                       "  emitters = {\n"
                       "  %s  }\n"
                       "]",
                       indent(m_integrator->toString()),
                       indent(m_sampler->toString()),
                       indent(m_camera->toString()), indent(meshes, 2), indent(emitters, 2));
}

void Scene::sampleEmitterDirection(const Intersection& its,
                                   const Point2f& sample_, DirectionSample& ds,
                                   Color3f& spectrum) const {
    Point2f sample(sample_);
    if (m_emitters.size() == 0) {
        spectrum = Color3f(0.f);
    } else if (m_emitters.size() == 1) {
        m_emitters[0]->sampleDirection(sample, its, ds, spectrum);
    } else {
        size_t emitter_num = m_emitters.size();
        float emitter_pdf = 1.f / emitter_num;

        // Randomly pick a emitter
        size_t index =
            std::min((size_t)(sample.x() * emitter_num), emitter_num - 1);
        Emitter* emitter = m_emitters[index];

        // Rescale sample.x() to lie in [0,1) again, it can "reuse" now !!!
        sample.x() = sample.x() * emitter_num - index;

        emitter->sampleDirection(sample, its, ds, spectrum);
        // Account for the discrete probability of sampling this emitter
        spectrum = spectrum * emitter_num;
        ds.pdf *= emitter_pdf;
    }

    // Perform a visibility test if requested
    if (spectrum.sum() > 0) {
        Ray3f ray(its.p, ds.d, M_EPSILON, ds.dist - M_EPSILON);
        spectrum *= 1 - (float)rayIntersect(ray);
    }
}

float Scene::pdfEmitterDirection(const DirectionSample& ds) const {
    if (m_emitters.size() == 1) {
        return m_emitters[0]->pdfDirection(ds);
    } else {
        const Emitter* emitter = reinterpret_cast<const Emitter*>(ds.obj);
        return emitter->pdfDirection(ds) / m_emitters.size();
    }
}

REGISTER_CLASS(Scene, "scene");

}  // namespace drawlab