#include "tracer/octree.h"
#include "core/utils/timer.h"

namespace drawlab {

OCTree::OCTree() {
    m_maxDepth = 0;
    m_leaf = 0;
    m_interior = 0;
}

OCTree::~OCTree() {
    // TODO
}

void OCTree::build(std::vector<Mesh*> meshPtrs) {
    Timer timer;
    cout << "Build OCTree.. ";
    m_meshPtrs.assign(meshPtrs.begin(), meshPtrs.end());

    BoundingBox3f bbox;
    std::vector<std::pair<int, int>> triangles;

    for (int i = 0; i < m_meshPtrs.size(); i++) {
        for (int j = 0; j < m_meshPtrs[i]->getTriangleCount(); j++) {
            triangles.push_back(std::make_pair(i, j));
        }
        bbox.expandBy(m_meshPtrs[i]->getBoundingBox());
    }
    m_root = recursiveBuild(bbox, triangles, 0);
    cout << "done. (Depth=" << m_maxDepth << ", Leaf=" << m_leaf
         << ", Interior=" << m_interior << ", took " << timer.elapsedString()
         << ")" << endl;
}

OCNode* OCTree::recursiveBuild(BoundingBox3f bbox,
                               std::vector<std::pair<int, int>>& triangles,
                               int depth) {
    if (triangles.size() == 0) {
        return nullptr;
    }
    m_maxDepth = std::max(m_maxDepth, depth);
    if (triangles.size() < 20) {
        m_leaf++;
        return new OCNode(bbox, triangles);
    } else {
        m_interior++;
    }

    std::vector<std::pair<int, int>> triangles_list[8];
    for (int i = 0; i < 8; i++) {
        Point3f corner = bbox.getCorner(i);
        Point3f min_p = corner.cwiseMin(bbox.getCenter());
        Point3f max_p = corner.cwiseMax(bbox.getCenter());
        BoundingBox3f sub_bbox = BoundingBox3f(min_p, max_p);
        for (auto [mesh_idx, tri_idx] : triangles) {
            if (m_meshPtrs[mesh_idx]->getBoundingBox(tri_idx).overlaps(
                    sub_bbox)) {
                triangles_list[i].push_back(std::make_pair(mesh_idx, tri_idx));
            }
        }
        if (triangles.size() == triangles_list[i].size()) {
            return new OCNode(bbox, triangles);
        }
    }

    OCNode* node = new OCNode(bbox);
    for (int i = 0; i < 8; i++) {
        BoundingBox3f bbox;
        for (auto [mesh_idx, tri_idx] : triangles_list[i]) {
            bbox.expandBy(m_meshPtrs[mesh_idx]->getBoundingBox(tri_idx));
        }
        // TODO: why?
        bbox.m_min -= Point3f(0.1f, 0.1f, 0.1f);
        node->m_children[i] =
            recursiveBuild(bbox, triangles_list[i], depth + 1);
    }
    return node;
}

bool OCTree::recursiveIntersect(OCNode* root, Ray3f& ray, Point2f& uv, float& t,
                                std::pair<int, int>& f) {
    if (root == nullptr || !root->m_bbox.rayIntersect(ray)) {
        return false;
    }

    // is leaf node
    if (root->m_triangles.size() != 0) {
        bool hit = false;
        float u_, v_, t_;
        for (auto [mesh_idx, tri_idx] : root->m_triangles) {
            if (m_meshPtrs[mesh_idx]->rayIntersect(tri_idx, ray, u_, v_, t_)) {
                ray.maxt = t_;
                uv = Point2f(u_, v_);
                t = t_;
                f = std::make_pair(mesh_idx, tri_idx);
                hit = true;
            }
        }
        return hit;
    }

    bool hit = false;
    float t_;
    Point2f uv_;
    std::pair<int, int> f_;

    float nearTs[8];
    int index[8];

    for (int i = 0; i < 8; i++) {
        if (root->m_children[i] == nullptr)
            nearTs[i] = std::numeric_limits<float>::infinity();
        else {
            float nearT, farT;
            root->m_children[i]->m_bbox.rayIntersect(ray, nearT, farT);
            nearTs[i] = nearT;
        }
        index[i] = i;
    }
    std::sort(index, index + 8, [&](int idx1, int idx2) -> bool {
        return nearTs[idx1] < nearTs[idx2];
    });

    for (auto idx : index) {
        if (ray.maxt < nearTs[idx]) {
            continue;
        }
        if (recursiveIntersect(root->m_children[idx], ray, uv_, t_, f_)) {
            ray.maxt = t_;
            uv = uv_;
            t = t_;
            f = f_;
            hit = true;
        }
    }
    return hit;
}

bool OCTree::rayIntersect(const Ray3f& ray, Point2f& uv, float& t,
                          std::pair<int, int>& f) {
    Ray3f ray_copy(ray);
    bool hit = recursiveIntersect(m_root, ray_copy, uv, t, f);
    return hit;
}

bool OCTree::recursiveAnyhit(OCNode* root, const Ray3f& ray) {
    if (root == nullptr || !root->m_bbox.rayIntersect(ray)) {
        return false;
    }

    if (root->m_triangles.size() != 0) {
        float u_, v_, t_;
        for (auto [mesh_idx, tri_idx] : root->m_triangles) {
            if (m_meshPtrs[mesh_idx]->rayIntersect(tri_idx, ray, u_, v_, t_)) {
                return true;
            }
        }
        return false;
    }

    for (int i = 0; i < 8; i++) {
        if (recursiveAnyhit(root->m_children[i], ray)) {
            return true;
        }
    }
    return false;
}

bool OCTree::rayAnyhit(const Ray3f& ray) {
    return recursiveAnyhit(m_root, ray);
}

}  // namespace drawlab
