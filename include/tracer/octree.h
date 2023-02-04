#pragma once

#include "core/base/common.h"
#include "core/math/bbox.h"
#include "tracer/mesh.h"
#include "tracer/accel.h"
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


class OCTree : public Accel {
private:
    OCNode* m_root;
    int m_maxDepth;
    int m_leaf;
    int m_interior;

    OCNode* recursiveBuild(BoundingBox3f bbox,
                           std::vector<std::pair<int, int>>& triangles,
                           int depth);

    bool recursiveIntersect(OCNode* root, Ray3f& ray, Point2f& uv, float& t,
                            std::pair<int, int>& f) const;

    bool recursiveAnyhit(OCNode* root, const Ray3f& ray) const;

public:
    OCTree() : m_maxDepth(0), m_leaf(0), m_interior(0) {}
    ~OCTree() {}

    void build();

    bool rayIntersect(const Ray3f& ray_, Intersection& its,
                          bool shadowRay) const;
};

}  // namespace drawlab