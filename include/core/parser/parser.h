#pragma once

#include "core/parser/object.h"

namespace drawlab {

/**
 * Load a scene from a xml file, and return its root object
 */
extern Object* loadFromXML(const std::string& filename);

}  // namespace drawlab