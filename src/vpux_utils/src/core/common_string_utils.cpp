//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/core/common_string_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <stdexcept>

namespace vpux {
void splitRangeAndApply(std::string_view::const_iterator begin, std::string_view::const_iterator end, char delim,
                        std::function<void(std::string_view)> callback) {
    auto curBegin = begin;
    auto curEnd = begin;
    while (curEnd != end) {
        while (curEnd != end && *curEnd != delim) {
            ++curEnd;
        }

        callback(std::string_view(&(*curBegin), static_cast<size_t>(curEnd - curBegin)));

        if (curEnd != end) {
            ++curEnd;
            curBegin = curEnd;
        }
    }
}

std::string sanitizeFilename(std::string_view input, size_t maxLen) {
    std::string s(input.substr(0, maxLen));
    for (auto& c : s) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-' && c != '.') {
            c = '_';
        }
    }
    if (s.empty()) {
        s = "model";
    }
    return s;
}
}  // namespace vpux
