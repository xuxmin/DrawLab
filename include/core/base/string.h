#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include "core/math/math.h"
#include "core/base/common.h"

namespace drawlab {

/// Convert a string to lower case
std::string toLower(const std::string& value);

/// Indent a string by the specified number of spaces
std::string indent(const std::string &string, int amount = 2);

/// Convert a string into an boolean value
bool toBool(const std::string &str);

/// Convert a string into a signed integer value
int toInt(const std::string &str);

/// Convert a string into an unsigned integer value
unsigned int toUInt(const std::string &str);

/// Convert a string into a floating point value
float toFloat(const std::string &str);

/// Convert a string into a 3D vector
Vector3f toVector3f(const std::string &str);

/// Tokenize a string into a list by splitting at 'delim'
std::vector<std::string> tokenize(const std::string &s, const std::string &delim = ", ", bool includeEmpty = false);

/// Check if a string ends with another string
bool endsWith(const std::string& value, const std::string &ending);

/// Convert a time value in milliseconds into a human-readable string
extern std::string timeString(double time, bool precise = false);

/// Convert a memory amount in bytes into a human-readable string
extern std::string memString(size_t size, bool precise = false);

}  // namespace drawlab
