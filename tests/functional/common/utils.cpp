//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <string>

#include "utils.hpp"
#include "vpux_private_metrics.hpp"

std::string getBackendName(const ov::Core& core) {
    return core.get_property("VPUX", VPUX_METRIC_KEY(BACKEND_NAME)).as<std::string>();
}

std::vector<std::string> getAvailableDevices(const ov::Core& core) {
    return core.get_property("VPUX", METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
}
