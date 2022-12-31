#include "core/base/string.h"
#include "core/math/vector.h"
#include <iomanip>

namespace drawlab {

std::string toLower(const std::string& value) {
    std::string result;
    result.resize(value.size());
    std::transform(value.begin(), value.end(), result.begin(), ::tolower);
    return result;
}

/// Indent a string by the specified number of spaces
std::string indent(const std::string& string, int amount) {
    /* This could probably be done faster (it's not
       really speed-critical though) */
    std::istringstream iss(string);
    std::ostringstream oss;
    std::string spacer(amount, ' ');
    bool firstLine = true;
    for (std::string line; std::getline(iss, line);) {
        if (!firstLine)
            oss << spacer;
        oss << line;
        if (!iss.eof())
            oss << endl;
        firstLine = false;
    }
    return oss.str();
}

/// Convert a string into an boolean value
bool toBool(const std::string& str) {
    std::string value = toLower(str);
    if (value == "false")
        return false;
    else if (value == "true")
        return true;
    else
        throw Exception("Could not parse boolean value \"%s\"", str);
}

/// Convert a string into a signed integer value
int toInt(const std::string& str) {
    char* end_ptr = nullptr;
    int result = (int)strtol(str.c_str(), &end_ptr, 10);
    if (*end_ptr != '\0')
        throw Exception("Could not parse integer value \"%s\"", str);
    return result;
}

/// Convert a string into an unsigned integer value
unsigned int toUInt(const std::string& str) {
    char* end_ptr = nullptr;
    unsigned int result = (int)strtoul(str.c_str(), &end_ptr, 10);
    if (*end_ptr != '\0')
        throw Exception("Could not parse integer value \"%s\"", str);
    return result;
}

/// Convert a string into a floating point value
float toFloat(const std::string& str) {
    char* end_ptr = nullptr;
    float result = (float)strtof(str.c_str(), &end_ptr);
    if (*end_ptr != '\0')
        throw Exception("Could not parse floating point value \"%s\"", str);
    return result;
}

/// Convert a string into a 3D vector
Vector3f toVector3f(const std::string& str) {
    std::vector<std::string> tokens = tokenize(str);
    if (tokens.size() != 3)
        throw Exception("Expected 3 values");
    Vector3f result;
    for (int i = 0; i < 3; ++i)
        result[i] = toFloat(tokens[i]);
    return result;
}

std::vector<std::string> tokenize(const std::string& string,
                                  const std::string& delim, bool includeEmpty) {
    std::string::size_type lastPos = 0,
                           pos = string.find_first_of(delim, lastPos);
    std::vector<std::string> tokens;

    while (lastPos != std::string::npos) {
        if (pos != lastPos || includeEmpty)
            tokens.push_back(string.substr(lastPos, pos - lastPos));
        lastPos = pos;
        if (lastPos != std::string::npos) {
            lastPos += 1;
            pos = string.find_first_of(delim, lastPos);
        }
    }

    return tokens;
}

bool endsWith(const std::string& value, const std::string& ending) {
    if (ending.size() > value.size()) {
        return false;
    }
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

std::string timeString(double time, bool precise) {
    if (std::isnan(time) || std::isinf(time))
        return "inf";

    std::string suffix = "ms";
    if (time > 1000) {
        time /= 1000;
        suffix = "s";
        if (time > 60) {
            time /= 60;
            suffix = "m";
            if (time > 60) {
                time /= 60;
                suffix = "h";
                if (time > 12) {
                    time /= 12;
                    suffix = "d";
                }
            }
        }
    }

    std::ostringstream os;
    os << std::setprecision(precise ? 4 : 1) << std::fixed << time << suffix;

    return os.str();
}

std::string memString(size_t size, bool precise) {
    double value = (double)size;
    const char* suffixes[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    int suffix = 0;
    while (suffix < 5 && value > 1024.0f) {
        value /= 1024.0f;
        ++suffix;
    }

    std::ostringstream os;
    os << std::setprecision(suffix == 0 ? 0 : (precise ? 4 : 1)) << std::fixed
       << value << " " << suffixes[suffix];

    return os.str();
}

}  // namespace drawlab
