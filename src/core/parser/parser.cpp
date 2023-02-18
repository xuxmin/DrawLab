/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#include "core/parser/parser.h"
#include "core/base/string.h"
#include "core/math/math.h"
#include "core/math/transform.h"
#include "pugixml.hpp"
#include <fstream>
#include <set>

namespace drawlab {

Object* loadFromXML(const std::string& filename) {
    /* Load the XML file using 'pugi' (a tiny self-contained XML parser
     * implemented in C++) */
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filename.c_str());

    /* Helper function: map a position offset in bytes to a more readable
     * line/column value */
    auto offset = [&](ptrdiff_t pos) -> std::string {
        std::fstream is(filename);
        char buffer[1024];
        int line = 0, linestart = 0, offset = 0;
        while (is.good()) {
            is.read(buffer, sizeof(buffer));
            for (int i = 0; i < is.gcount(); ++i) {
                if (buffer[i] == '\n') {
                    if (offset + i >= pos)
                        return tfm::format("line %i, col %i", line + 1,
                                           pos - linestart);
                    ++line;
                    linestart = offset + i;
                }
            }
            offset += (int)is.gcount();
        }
        return "byte offset " + std::to_string(pos);
    };

    if (!result) /* There was a parser / file IO error */
        throw Exception("Error while parsing \"%s\": %s (at %s)", filename,
                        result.description(), offset(result.offset));

    /* Set of supported XML tags */
    enum ETag {
        /* Object classes */
        EScene = Object::EScene,
        EMesh = Object::EMesh,
        EBSDF = Object::EBSDF,
        EPhaseFunction = Object::EPhaseFunction,
        EEmitter = Object::EEmitter,
        EMedium = Object::EMedium,
        ECamera = Object::ECamera,
        EIntegrator = Object::EIntegrator,
        ESampler = Object::ESampler,
        ETest = Object::ETest,
        EReconstructionFilter = Object::EReconstructionFilter,
        ETexture = Object::ETexture,

        /* Properties */
        EBoolean = Object::EClassTypeCount,
        EInteger,
        EFloat,
        EString,
        EPoint,
        EVector,
        EColor,
        ETransform,
        ETranslate,
        EMatrix,
        ERotate,
        EScale,
        ELookAt,

        EInvalid
    };

    /* Create a mapping from tag names to tag IDs */
    std::map<std::string, ETag> tags;
    tags["scene"] = EScene;
    tags["mesh"] = EMesh;
    tags["bsdf"] = EBSDF;
    tags["emitter"] = EEmitter;
    tags["camera"] = ECamera;
    tags["medium"] = EMedium;
    tags["phase"] = EPhaseFunction;
    tags["integrator"] = EIntegrator;
    tags["sampler"] = ESampler;
    tags["rfilter"] = EReconstructionFilter;
    tags["test"] = ETest;
    tags["texture"] = ETexture;
    tags["boolean"] = EBoolean;
    tags["integer"] = EInteger;
    tags["float"] = EFloat;
    tags["string"] = EString;
    tags["point"] = EPoint;
    tags["vector"] = EVector;
    tags["color"] = EColor;
    tags["transform"] = ETransform;
    tags["translate"] = ETranslate;
    tags["matrix"] = EMatrix;
    tags["rotate"] = ERotate;
    tags["scale"] = EScale;
    tags["lookat"] = ELookAt;

    /* Helper function to check if attributes are fully specified */
    auto check_attributes = [&](const pugi::xml_node& node,
                                std::set<std::string> attrs,
                                std::set<std::string> opt_attrs = {}) {
        for (auto attr : node.attributes()) {
            auto it0 = attrs.find(attr.name());
            auto it1 = opt_attrs.find(attr.name());
            if (it0 == attrs.end()) {
                if (it1 == opt_attrs.end()) {
                    throw Exception("Error while parsing \"%s\": unexpected "
                                    "attribute \"%s\" in \"%s\" at %s",
                                    filename, attr.name(), node.name(),
                                    offset(node.offset_debug()));
                }
            }
            else {
                attrs.erase(it0);
            }
        }
        if (!attrs.empty())
            throw Exception("Error while parsing \"%s\": missing attribute "
                            "\"%s\" in \"%s\" at %s",
                            filename, *attrs.begin(), node.name(),
                            offset(node.offset_debug()));
    };

    Transform transform;

    /* Helper function to parse a Nori XML node (recursive) */
    std::function<Object*(pugi::xml_node&, PropertyList&, int)> parseTag =
        [&](pugi::xml_node& node, PropertyList& list,
            int parentTag) -> Object* {
        /* Skip over comments */
        if (node.type() == pugi::node_comment ||
            node.type() == pugi::node_declaration)
            return nullptr;

        if (node.type() != pugi::node_element)
            throw Exception(
                "Error while parsing \"%s\": unexpected content at %s",
                filename, offset(node.offset_debug()));

        /* Look up the name of the current element */
        auto it = tags.find(node.name());
        if (it == tags.end())
            throw Exception(
                "Error while parsing \"%s\": unexpected tag \"%s\" at %s",
                filename, node.name(), offset(node.offset_debug()));
        int tag = it->second;

        /* Perform some safety checks to make sure that the XML tree really
         * makes sense */
        bool hasParent = parentTag != EInvalid;
        bool parentIsObject = hasParent && parentTag < Object::EClassTypeCount;
        bool currentIsObject = tag < Object::EClassTypeCount;
        bool parentIsTransform = parentTag == ETransform;
        bool currentIsTransformOp = tag == ETranslate || tag == ERotate ||
                                    tag == EScale || tag == ELookAt ||
                                    tag == EMatrix;

        if (!hasParent && !currentIsObject)
            throw Exception("Error while parsing \"%s\": root element \"%s\" "
                            "must be a Nori object (at %s)",
                            filename, node.name(), offset(node.offset_debug()));

        if (parentIsTransform != currentIsTransformOp)
            throw Exception("Error while parsing \"%s\": transform nodes "
                            "can only contain transform operations (at %s)",
                            filename, offset(node.offset_debug()));

        if (hasParent && !parentIsObject &&
            !(parentIsTransform && currentIsTransformOp))
            throw Exception("Error while parsing \"%s\": node \"%s\" requires "
                            "a Nori object as parent (at %s)",
                            filename, node.name(), offset(node.offset_debug()));

        if (tag == EScene)
            node.append_attribute("type") = "scene";
        else if (tag == ETransform)
            transform.setIdentity();

        PropertyList propList;
        std::vector<Object*> children;
        for (pugi::xml_node& ch : node.children()) {
            Object* child = parseTag(ch, propList, tag);
            if (child)
                children.push_back(child);
        }

        Object* result = nullptr;
        try {
            if (currentIsObject) {
                check_attributes(node, {"type"}, {"name"});

                /* This is an object, first instantiate it */
                result = ObjectFactory::createInstance(
                    node.attribute("type").value(), propList,
                    node.attribute("name").value());

                if (result->getClassType() != (int)tag) {
                    throw Exception(
                        "Unexpectedly constructed an object "
                        "of type <%s> (expected type <%s>): %s",
                        Object::classTypeName(result->getClassType()),
                        Object::classTypeName((Object::EClassType)tag),
                        result->toString());
                }

                /* Add all children */
                for (auto ch : children) {
                    result->addChild(ch);
                    ch->setParent(result);
                }

                /* Activate / configure the object */
                result->activate();
            } else {
                /* This is a property */
                switch (tag) {
                    case EString: {
                        check_attributes(node, {"name", "value"});
                        list.setString(node.attribute("name").value(),
                                       node.attribute("value").value());
                    } break;
                    case EFloat: {
                        check_attributes(node, {"name", "value"});
                        list.setFloat(node.attribute("name").value(),
                                      toFloat(node.attribute("value").value()));
                    } break;
                    case EInteger: {
                        check_attributes(node, {"name", "value"});
                        list.setInteger(node.attribute("name").value(),
                                        toInt(node.attribute("value").value()));
                    } break;
                    case EBoolean: {
                        check_attributes(node, {"name", "value"});
                        list.setBoolean(
                            node.attribute("name").value(),
                            toBool(node.attribute("value").value()));
                    } break;
                    case EPoint: {
                        check_attributes(node, {"name", "value"});
                        Vector3f vec =
                            toVector3f(node.attribute("value").value());
                        list.setPoint(node.attribute("name").value(),
                                      Point3f(vec[0], vec[1], vec[2]));
                    } break;
                    case EVector: {
                        check_attributes(node, {"name", "value"});
                        list.setVector(
                            node.attribute("name").value(),
                            toVector3f(node.attribute("value").value()));
                    } break;
                    case EColor: {
                        check_attributes(node, {"name", "value"});
                        Vector3f vec =
                            toVector3f(node.attribute("value").value());
                        list.setColor(node.attribute("name").value(),
                                      Color3f(vec[0], vec[1], vec[2]));
                    } break;
                    case ETransform: {
                        check_attributes(node, {"name"});
                        list.setTransform(node.attribute("name").value(),
                                          transform);
                    } break;
                    case ETranslate: {
                        check_attributes(node, {"value"});
                        Vector3f v =
                            toVector3f(node.attribute("value").value());
                        transform =
                            Matrix4f::getTranslationMatrix(v) * transform;
                    } break;
                    case EMatrix: {
                        check_attributes(node, {"value"});
                        std::vector<std::string> tokens =
                            tokenize(node.attribute("value").value());
                        if (tokens.size() != 16)
                            throw Exception("Expected 16 values");
                        Matrix4f matrix;
                        for (int i = 0; i < 4; ++i)
                            for (int j = 0; j < 4; ++j)
                                matrix[i][j] = toFloat(tokens[i * 4 + j]);
                        transform = matrix * transform;
                    } break;
                    case EScale: {
                        check_attributes(node, {"value"});
                        Vector3f v =
                            toVector3f(node.attribute("value").value());
                        transform = Matrix4f::getScaleMatrix(v) * transform;
                    } break;
                    case ERotate: {
                        check_attributes(node, {"angle", "axis"});
                        float angle =
                            degToRad(toFloat(node.attribute("angle").value()));
                        Vector3f axis =
                            toVector3f(node.attribute("axis").value());
                        transform =
                            Matrix4f::getRotateMatrix(axis, angle) * transform;
                    } break;
                    case ELookAt: {
                        check_attributes(node, {"origin", "target", "up"});
                        Vector3f origin = toVector3f(
                            node.attribute("origin").value());  // center
                        Vector3f target = toVector3f(
                            node.attribute("target").value());  // eye
                        Vector3f up = toVector3f(node.attribute("up").value());

                        Vector3f dir = (target - origin).normalized();
                        Vector3f left = up.normalized().cross(dir).normalized();
                        Vector3f newUp = dir.cross(left).normalized();

                        Matrix4f trafo;
                        trafo[0][0] = left[0];
                        trafo[0][1] = newUp[0];
                        trafo[0][2] = dir[0];
                        trafo[0][3] = origin[0];
                        trafo[1][0] = left[1];
                        trafo[1][1] = newUp[1];
                        trafo[1][2] = dir[1];
                        trafo[1][3] = origin[1];
                        trafo[2][0] = left[2];
                        trafo[2][1] = newUp[2];
                        trafo[2][2] = dir[2];
                        trafo[2][3] = origin[2];
                        trafo[3][3] = 1.f;
                        transform = trafo * transform;
                    } break;

                    default:
                        throw Exception("Unhandled element \"%s\"",
                                        node.name());
                };
            }
        } catch (const Exception& e) {
            throw Exception("Error while parsing \"%s\": %s (at %s)", filename,
                            e.what(), offset(node.offset_debug()));
        }

        return result;
    };

    PropertyList list;
    return parseTag(*doc.begin(), list, EInvalid);
}

}  // namespace drawlab