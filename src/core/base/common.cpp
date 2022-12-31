#include "core/base/common.h"
#include "filesystem/resolver.h"

namespace drawlab {

filesystem::resolver* getFileResolver() {
    static filesystem::resolver* resolver = new filesystem::resolver();
    return resolver;
}

}  // namespace drawlab
