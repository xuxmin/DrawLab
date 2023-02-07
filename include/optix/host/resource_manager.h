#pragma once

#include <map>
#include <string>
#include "optix/host/sutil.h"

namespace optix {

template <typename T> class ResourceManager {
public:
    const T get(std::string key) const {
        if (m_resources.find(key) != m_resources.end()) {
            return m_resources.at(key);
        }
        return nullptr;
    }

    bool existed(std::string key) const { return get(key) != nullptr; }

    void add(std::string key, T res) {
        T data = get(key);
        if (data != nullptr) {
            throw Exception("Error in ResourceManager::add(), the key " + key +
                            " is existed!");
        }
        m_resources[key] = res;
    }

    const std::map<std::string, T>& getResources() const { return m_resources; }

    void clear() { m_resources.clear(); }

private:
    std::map<std::string, T> m_resources;
};

}  // namespace optix