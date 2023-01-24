#include "tracer/mesh.h"
#include "core/math/wrap.h"
#include "tracer/emitter.h"

namespace drawlab {

Mesh::Mesh() {}

Mesh::~Mesh() {
    delete m_bsdf;
    delete m_emitter;
    delete m_dpdf;
}

void Mesh::activate() {
    if (!m_bsdf) {
        /* If no material was assigned, instantiate a diffuse BRDF */
        m_bsdf = static_cast<BSDF*>(
            ObjectFactory::createInstance("diffuse", PropertyList()));
    }
    if (!m_dpdf) {
        m_dpdf = new DiscretePDF(0);
        uint32_t face_num = getTriangleCount();
        for (uint32_t i = 0; i < face_num; i++) {
            m_dpdf->append(surfaceArea(i));
        }
        m_dpdf->normalize();
    }
}

void Mesh::addChild(Object* obj) {
    switch (obj->getClassType()) {
        case EBSDF:
            if (m_bsdf)
                throw Exception(
                    "Mesh: tried to register multiple BSDF instances!");
            m_bsdf = static_cast<BSDF*>(obj);
            break;

        case EEmitter: {
            Emitter* emitter = static_cast<Emitter*>(obj);
            if (m_emitter)
                throw Exception(
                    "Mesh: tried to register multiple Emitter instances!");
            emitter->setMesh(this);
            m_emitter = emitter;
        } break;

        default:
            throw Exception("Mesh::addChild(<%s>) is not supported!",
                            classTypeName(obj->getClassType()));
    }
}

std::string Mesh::toString() const {
    return tfm::format(
        "Mesh[\n"
        "  name = \"%s\",\n"
        "  vertexCount = %i,\n"
        "  triangleCount = %i,\n"
        "  bsdf = %s,\n"
        "  emitter = %s\n"
        "]",
        m_name, getVertexCount(), getTriangleCount(),
        m_bsdf ? indent(m_bsdf->toString()) : std::string("null"),
        m_emitter ? indent(m_emitter->toString()) : std::string("null"));
}

std::string Intersection::toString() const {
    if (!mesh)
        return "Intersection[invalid]";

    return tfm::format("Intersection[\n"
                       "  p = %s,\n"
                       "  t = %f,\n"
                       "  uv = %s,\n"
                       "  shFrame = %s,\n"
                       "  geoFrame = %s,\n"
                       "  mesh = %s\n"
                       "]",
                       p.toString(), t, uv.toString(),
                       indent(shFrame.toString()), indent(geoFrame.toString()),
                       mesh ? mesh->toString() : std::string("null"));
}

float Mesh::surfaceArea(uint32_t index) const {
    const Point3f p0 = getVertexPosition(index, 0);
    const Point3f p1 = getVertexPosition(index, 1);
    const Point3f p2 = getVertexPosition(index, 2);

    return 0.5f * Vector3f((p1 - p0).cross(p2 - p0)).length();
}

Point3f Mesh::getCentroid(uint32_t index) const {
    return (1.0f / 3.0f) *
           (getVertexPosition(index, 0) + getVertexPosition(index, 1) +
            getVertexPosition(index, 2));
}

BoundingBox3f Mesh::getBoundingBox(uint32_t index) const {
    BoundingBox3f result(getVertexPosition(index, 0));
    result.expandBy(getVertexPosition(index, 1));
    result.expandBy(getVertexPosition(index, 2));
    return result;
}

Point2f Mesh::getVertexTexCoord(size_t iface, size_t index) const {
    if (m_UV.size() <= 0) {
        return Point2f(0, 0);
    }
    unsigned int p = m_F[3 * iface + index];
    return Point2f(m_UV[2 * p], m_UV[2 * p + 1]);
}

Point3f Mesh::getVertexPosition(size_t iface, size_t index) const {
    unsigned int p = m_F[3 * iface + index];
    return Point3f(m_V[3 * p], m_V[3 * p + 1], m_V[3 * p + 2]);
}

Normal3f Mesh::getVertexNormal(size_t iface, size_t index) const {
    if (m_N.size() <= 0) {
        return Normal3f(0, 0, 0);
    }
    unsigned int p = m_F[3 * iface + index];
    return Normal3f(m_N[3 * p], m_N[3 * p + 1], m_N[3 * p + 2]);
}

const std::vector<float>& Mesh::getVertexPosition() const {
    return m_V;
}

const std::vector<unsigned int>& Mesh::getVertexIndex() const {
    return m_F;
}

bool Mesh::rayIntersect(uint32_t index, const Ray3f& ray, float& u, float& v,
                        float& t) const {
    const Point3f p0 = getVertexPosition(index, 0);
    const Point3f p1 = getVertexPosition(index, 1);
    const Point3f p2 = getVertexPosition(index, 2);

    /* Find vectors for two edges sharing v[0] */
    Vector3f edge1 = p1 - p0, edge2 = p2 - p0;

    /* Begin calculating determinant - also used to calculate U parameter */
    Vector3f pvec = ray.d.cross(edge2);

    /* If determinant is near zero, ray lies in plane of triangle */
    float det = edge1.dot(pvec);

    if (det > -1e-8f && det < 1e-8f)
        return false;
    float inv_det = 1.0f / det;

    /* Calculate distance from v[0] to ray origin */
    Vector3f tvec = ray.o - p0;

    /* Calculate U parameter and test bounds */
    u = tvec.dot(pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    /* Prepare to test V parameter */
    Vector3f qvec = tvec.cross(edge1);

    /* Calculate V parameter and test bounds */
    v = ray.d.dot(qvec) * inv_det;

    if (v < 0.0 || u + v > 1.0)
        return false;

    /* Ray intersects triangle -> compute t */
    t = edge2.dot(qvec) * inv_det;

    return t >= ray.mint && t <= ray.maxt;
}

void Mesh::samplePosition(const Point2f& sample_, Point3f& position,
                          Normal3f& normal) const {
    Point2f sample(sample_);
    size_t f = m_dpdf->sampleReuse(sample.x());
    const Point3f p0 = getVertexPosition(f, 0);
    const Point3f p1 = getVertexPosition(f, 1);
    const Point3f p2 = getVertexPosition(f, 2);

    Vector3f bary = Warp::squareToUniformTriangle(sample);
    position = bary.x() * p0 + bary.y() * p1 + bary.z() * p2;

    if (m_N.size() != 0) {
        const Normal3f n0 = getVertexNormal(f, 0);
        const Normal3f n1 = getVertexNormal(f, 1);
        const Normal3f n2 = getVertexNormal(f, 2);
        normal = bary.x() * n0 + bary.y() * n1 + bary.z() * n2;
    } else {
        Vector3f p01 = p1 - p0;
        Vector3f p12 = p2 - p1;
        normal = p01.cross(p12);
    }
    normal.normalize();
}

float Mesh::pdfPosition() const { return m_dpdf->getNormalization(); }

}  // namespace drawlab