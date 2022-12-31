#include "tracer/accel.h"
#include "tracer/mesh.h"

namespace drawlab {

void Accel::addMesh(Mesh* mesh) {
    m_meshPtrs.push_back(mesh);
    m_bbox.expandBy(mesh->getBoundingBox());
}

void Accel::build() {
    m_octree = new OCTree();
    m_octree->build(m_meshPtrs);
}

bool Accel::rayIntersect(const Ray3f& ray_, Intersection& its,
                         bool shadowRay) const {
    Ray3f ray(ray_);  /// Make a copy of the ray (we will need to update its
                      /// '.maxt' value)

    if (shadowRay) {
        return m_octree->rayAnyhit(ray);
    }

    std::pair<int, int> f(-1, -1);
    Point2f uv(0, 0);
    float t = 0;
    bool foundIntersection = m_octree->rayIntersect(ray, uv, t, f);
    if (foundIntersection) {
        ray.maxt = t;
        its.uv = uv;
        its.mesh = m_meshPtrs[f.first];

        /* Find the barycentric coordinates */
        Vector3f bary(1 - uv.x() - uv.y(), uv.x(), uv.y());

        /* References to all relevant mesh buffers */
        const Mesh* mesh = its.mesh;

        Point3f p0 = mesh->getVertexPosition(f.second, 0);
        Point3f p1 = mesh->getVertexPosition(f.second, 1);
        Point3f p2 = mesh->getVertexPosition(f.second, 2);

        /* Compute the intersection positon accurately
           using barycentric coordinates */
        its.p = bary.x() * p0 + bary.y() * p1 + bary.z() * p2;

        /* Compute proper texture coordinates if provided by the mesh */
        Point2f new_UV = bary.x() * mesh->getVertexTexCoord(f.second, 0) +
                         bary.y() * mesh->getVertexTexCoord(f.second, 1) +
                         bary.z() * mesh->getVertexTexCoord(f.second, 2);
        if (mesh->hasTexCoord()) {
            its.uv = new_UV;
        }

        /* Compute the geometry frame */
        its.geoFrame = Frame((p1 - p0).cross(p2 - p0).normalized());

        if (mesh->hasVertexNormal()) {
            /* Compute the shading frame. Note that for simplicity,
               the current implementation doesn't attempt to provide
               tangents that are continuous across the surface. That
               means that this code will need to be modified to be able
               use anisotropic BRDFs, which need tangent continuity */
            its.shFrame = Frame(
                bary.x() * mesh->getVertexNormal(f.second, 0) +
                bary.y() * mesh->getVertexNormal(f.second, 1) +
                bary.z() * mesh->getVertexNormal(f.second, 2).normalized());
        } else {
            its.shFrame = its.geoFrame;
        }
    }

    return foundIntersection;
}

}  // namespace drawlab
