#include "core/parser/object.h"

namespace drawlab {

void Object::addChild(Object*) {
    throw Exception(
        "NoriObject::addChild() is not implemented for objects of type '%s'!",
        classTypeName(getClassType()));
}

void Object::activate() { /* Do nothing */
}
void Object::setParent(Object*) { /* Do nothing */
}

std::map<std::string, ObjectFactory::Constructor>*
    ObjectFactory::m_constructors = nullptr;

void ObjectFactory::registerClass(const std::string& name,
                                  const Constructor& constr) {
    if (!m_constructors)
        m_constructors =
            new std::map<std::string, ObjectFactory::Constructor>();
    (*m_constructors)[name] = constr;
}

}  // namespace drawlab