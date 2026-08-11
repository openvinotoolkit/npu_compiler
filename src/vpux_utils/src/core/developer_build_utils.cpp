//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/core/developer_build_utils.hpp"

#include <cstdlib>

using namespace vpux;

void vpux::parseEnv(StringRef envVarName, std::string& var) {
    if (const auto env = std::getenv(envVarName.data())) {
        var = env;
    }
}

void vpux::parseEnv(StringRef envVarName, bool& var) {
    if (const auto env = std::getenv(envVarName.data())) {
        if (env[0] == '\0') {
            // Empty string treated as disabled
            var = false;
            return;
        }
        try {
            var = std::stoi(env) != 0;
        } catch (const std::exception&) {
            // Treat invalid values as disabled
            var = false;
        }
    }
}

bool vpux::isPerfDebugMode() {
    bool result = false;
    parseEnv("IE_NPU_PERF_DEBUG", result);
    return result;
}
