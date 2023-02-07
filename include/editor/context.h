#pragma once

#include "core/math/vector.h"
#include <map>

namespace drawlab {

class Context {

private:
    std::map<std::string, bool> bool_vars;
    std::map<std::string, float> float_vars;
    std::map<std::string, Vector2f> vector2f_vars;
    std::map<std::string, Vector3f> vector3f_vars;

public:

    void setBool(std::string var, bool val) {
        bool_vars[var] = val;
    }
    bool& getBool(std::string var) {
        return bool_vars.at(var);
    }

    void setFloat(std::string var, float val) {
        float_vars[var] = val;
    }
    float& getFloat(std::string var) {
        return float_vars.at(var);
    }


    void setVector2f(std::string var, Vector2f val) {
        vector2f_vars[var] = val;
    }
    Vector2f& getVector2f(std::string var) {
        return vector2f_vars.at(var);
    }

    void setVector3f(std::string var, Vector3f val) {
        vector3f_vars[var] = val;
    }
    Vector3f& getVector3f(std::string var) {
        return vector3f_vars.at(var);
    }
};

}  // namespace drawlab