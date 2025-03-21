//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// Task name handling utilities

#pragma once

#include <string>

namespace vpux::profiling {

// Suffix used to create variant name from cluster name
extern const std::string VARIANT_LEVEL_PROFILING_SUFFIX;

struct ParsedTaskName {
    std::string layerName;
    std::string layerType;
};

// Parses the full task nameinto ParsedTaskName, extracting task name, layer type and cluster id
ParsedTaskName deserializeTaskName(const std::string& taskName);

std::string getLayerName(const std::string& taskName);

}  // namespace vpux::profiling
