//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Various string manipulation utility functions.
//

#pragma once

#include <functional>
#include <string>
#include <string_view>

namespace vpux {
void splitRangeAndApply(std::string_view::const_iterator begin, std::string_view::const_iterator end, char delim,
                        std::function<void(std::string_view)> callback);

// Sanitize a string for use as a filename component.
// Replaces characters outside [a-zA-Z0-9_.-] with '_', truncates to maxLen,
// and returns "model" as a default for empty inputs (without truncation).
std::string sanitizeFilename(std::string_view input, size_t maxLen = 200);
}  // namespace vpux
