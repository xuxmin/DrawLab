#pragma once

#include "core/base/common.h"
#include "core/parser/proplist.h"
#include <functional>

namespace drawlab {

/**
 * \brief Base class of all objects
 *
 * An object represents an instance that is part of
 * a scene description.
 *
 */
class Object {
public:
    enum EClassType {
        EScene = 0,
        EMesh,
        EBSDF,
        EPhaseFunction,
        EEmitter,
        EMedium,
        ECamera,
        EIntegrator,
        ESampler,
        ETest,
        EReconstructionFilter,
        EClassTypeCount
    };

    virtual ~Object() {}

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.)
     * provided by this instance
     * */
    virtual EClassType getClassType() const = 0;

    /**
     * \brief Add a child object to the current instance
     *
     * The default implementation does not support children and
     * simply throws an exception
     */
    virtual void addChild(Object* child);

    /**
     * \brief Set the parent object
     *
     * Subclasses may choose to override this method to be
     * notified when they are added to a parent object. The
     * default implementation does nothing.
     */
    virtual void setParent(Object* parent);

    /**
     * \brief Perform some action associated with the object
     *
     * The default implementation throws an exception. Certain objects
     * may choose to override it, e.g. to implement initialization,
     * testing, or rendering functionality.
     *
     * This function is called by the XML parser once it has
     * constructed an object and added all of its children
     * using \ref addChild().
     */
    virtual void activate();

    /// Return a brief string summary of the instance (for debugging purposes)
    virtual std::string toString() const = 0;

    /// Turn a class type into a human-readable string
    static std::string classTypeName(EClassType type) {
        switch (type) {
            case EScene: return "scene";
            case EMesh: return "mesh";
            case EBSDF: return "bsdf";
            case EEmitter: return "emitter";
            case ECamera: return "camera";
            case EIntegrator: return "integrator";
            case ESampler: return "sampler";
            case ETest: return "test";
            default: return "<unknown>";
        }
    }
};

/**
 * \brief Factory for objects
 *
 * The class is part of a mini-RTTI framework. we except
 * to instantiate arbitrary objects by their name.
 *
 */
class ObjectFactory {
public:
    typedef std::function<Object*(const PropertyList&)> Constructor;

    /**
     * \brief Register an object constructor with the object factory
     *
     * This function is called by the macro \ref NORI_REGISTER_CLASS
     *
     * \param name
     *     An internal name that is associated with this class. This is the
     *     'type' field found in the scene description XML files
     *
     * \param constr
     *     A function pointer to an anonymous function that is
     *     able to call the constructor of the class.
     */
    static void registerClass(const std::string& name,
                              const Constructor& constr);

    /**
     * \brief Construct an instance from the class of the given name
     *
     * \param name
     *     An internal name that is associated with this class. This is the
     *     'type' field found in the scene description XML files
     *
     * \param propList
     *     A list of properties that will be passed to the constructor
     *     of the class.
     */
    static Object* createInstance(const std::string& name,
                                  const PropertyList& propList) {
        if (!m_constructors ||
            m_constructors->find(name) == m_constructors->end())
            throw Exception(
                "A constructor for class \"%s\" could not be found!", name);
        return (*m_constructors)[name](propList);
    }

private:
    static std::map<std::string, Constructor>* m_constructors;
};

/// Macro for registering an object constructor with the \ref ObjectFactory
#define REGISTER_CLASS(cls, name)                                              \
    cls* cls##_create(const PropertyList& list) { return new cls(list); }      \
    static struct cls##_ {                                                     \
        cls##_() { ObjectFactory::registerClass(name, cls##_create); }         \
    } cls##__HAHA__;

// Scene* Scene_create(const PropertyList& list) {
//     return new Scene(list);
// }
// static struct Scene_ {
//     Scene_() {
//         ObjectFactory::registerClass("scene", Scene_create);
//     }
// } Scene__HAHA__;

}  // namespace drawlab