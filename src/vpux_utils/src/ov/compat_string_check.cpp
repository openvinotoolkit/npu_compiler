//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/ov/compat_string_check.hpp"
#include "vpux/utils/ov/compat_string_parser.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace {

uint64_t parseInt(const std::string& str) {
    size_t pos = 0;
    uint64_t value = std::stoull(str, &pos);
    if (pos != str.size()) {
        throw std::runtime_error("Invalid integer: " + str);
    }
    return value;
}

}  // namespace

namespace vpux::compat {

BlobRequirements parseCompatibilityString(std::string_view compatibilityString) {
    Parser parser(compatibilityString, std::array{"compiler", "npu", "t", "elf", "mi"});

    BlobRequirements reqs;
    reqs.platformId = parseInt(parser.getAttribute("npu"));
    reqs.numTiles = parseInt(parser.getAttribute("t"));
    return reqs;
}

}  // namespace vpux::compat
