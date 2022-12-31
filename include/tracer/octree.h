#pragma once

#include "core/base/common.h"
#include "core/math/bbox.h"
#include "tracer/mesh.h"
#include <algorithm>
#include <vector>

namespace drawlab {

struct OCNode {
    BoundingBox3f m_bbox;
    OCNode* m_children[8];
    std::vector<std::pair<int, int>> m_triangles;

    OCNode(BoundingBox3f bbox, std::vector<std::pair<int, int>> triangles) {
        m_bbox = bbox;
        m_triangles.assign(triangles.begin(), triangles.end());
    }
    OCNode(BoundingBox3f bbox) { m_bbox = bbox; }
};

class OCTree {
private:
    std::vector<Mesh*> m_meshPtrs;
    OCNode* m_root;
    int m_maxDepth;
    int m_leaf;
    int m_interior;

    OCNode* recursiveBuild(BoundingBox3f bbox,
                           std::vector<std::pair<int, int>>& triangles,
                           int depth);
    bool recursiveIntersect(OCNode* root, Ray3f& ray, Point2f& uv, float& t,
                            std::pair<int, int>& f);
    bool recursiveAnyhit(OCNode* root, const Ray3f& ray);

public:
    OCTree();
    ~OCTree();

    /// @brief Build tree with the provided meshes.
    void build(std::vector<Mesh*> meshPtr);

    /**
     * \brief Do ray intersection
     * \param ray
     * \param uv UV coordinates, if any
     * \param t Unoccluded distance along the ray
     * \param f Triangle index, the first is the mesh index, and
     *          the second is the triangle index of that mesh.
     */
    bool rayIntersect(const Ray3f& ray, Point2f& uv, float& t,
                      std::pair<int, int>& f);

    bool rayAnyhit(const Ray3f& ray);
};

}  // namespace drawlab