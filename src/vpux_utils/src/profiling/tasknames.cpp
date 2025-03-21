//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// Task name helpers

#include "vpux/utils/profiling/tasknames.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/location.hpp"

#include <sstream>
#include <vector>

namespace {

struct TokenizedTaskName {
    std::string layerName;
    std::vector<std::string> tokens;
};

std::vector<std::string> splitBySeparator(const std::string& s, char separator) {
    std::istringstream iss(s);
    std::string part;
    std::vector<std::string> parts;
    while (std::getline(iss, part, separator)) {
        parts.push_back(part);
    }
    return parts;
}

TokenizedTaskName tokenizeTaskName(const std::string& taskName) {
    auto nameSepPos = taskName.rfind(vpux::LOCATION_ORIGIN_SEPARATOR);
    VPUX_THROW_WHEN(nameSepPos == std::string::npos, "Malformed task name: '{0}'", taskName);
    auto layerName = taskName.substr(0, nameSepPos);
    auto afterNameSep = taskName.substr(nameSepPos + 1);
    std::vector<std::string> parts = splitBySeparator(afterNameSep, vpux::LOCATION_SEPARATOR);
    return {std::move(layerName), std::move(parts)};
}

}  // namespace

namespace vpux::profiling {

const std::string VARIANT_LEVEL_PROFILING_SUFFIX = "variant";

ParsedTaskName deserializeTaskName(const std::string& fullTaskName) {
    const auto LOC_METADATA_SEPARATOR = '_';  // conventional separator used for attaching metadata to MLIR Locations

    auto tokenized = tokenizeTaskName(fullTaskName);
    std::string layerType = "<unknown>";
    std::string& layerName = tokenized.layerName;

    for (const auto& token : tokenized.tokens) {
        VPUX_THROW_WHEN(token.empty(), "Empty task name token");

        auto parts = splitBySeparator(token, LOC_METADATA_SEPARATOR);
        auto partsNum = parts.size();

        if (partsNum == 2 && parts[0] == "t") {
            layerType = parts[1];
        }
    }

    return {std::move(layerName), std::move(layerType)};
}

std::string getLayerName(const std::string& taskName) {
    return taskName.substr(0, taskName.rfind(vpux::LOCATION_ORIGIN_SEPARATOR));
}

}  // namespace vpux::profiling
