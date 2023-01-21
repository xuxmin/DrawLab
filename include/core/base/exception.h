#include <iostream>
#include <tinyformat.h>


namespace drawlab {

/// Simple exception class, which stores a human-readable error description
class Exception : public std::runtime_error {
public:
    /// Variadic template constructor to support printf-style arguments
    template <typename... Args>
    Exception(const char* fmt, const Args&... args)
        : std::runtime_error(tfm::format(fmt, args...)) {}

    Exception(std::string msg) : std::runtime_error(msg) {}
};

}