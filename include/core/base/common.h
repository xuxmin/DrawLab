#pragma once

#if defined(_MSC_VER)
/* Disable some warnings on MSVC++ */
// #pragma warning(disable : 4127 4702 4100 4515 4800 4146 4512)
#    define WIN32_LEAN_AND_MEAN /* Don't ever include MFC on Windows */
#    define NOMINMAX            /* Don't override min/max */
#endif

#include "core/base/string.h"
#include "core/base/exception.h"
#include "core/math/math.h"
#include "filesystem/path.h"
#include "filesystem/resolver.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace drawlab {

/// Import cout, cerr, endl for debugging purposes
using std::cerr;
using std::cout;
using std::endl;

/// Some more forward declarations
class BSDF;
class Bitmap;
class Camera;
class Integrator;
class Emitter;
struct EmitterQueryRecord;
class Mesh;
class Object;
class ObjectFactory;
class ReconstructionFilter;
class Sampler;
class Scene;
class AccelTree;
class ImageBlock;
class Intersection;

/**
 * \brief Return the global file resolver instance
 *
 * This class is used to locate resource files (e.g. mesh or
 * texture files) referenced by a scene being loaded
 */
extern filesystem::resolver* getFileResolver();

}  // namespace drawlab
