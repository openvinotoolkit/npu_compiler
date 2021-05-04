//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <cstdint>
#include <map>

#include "vpux/vpux_plugin_config.hpp"

namespace utils {
uint32_t getSliceIdBySwDeviceId(const uint32_t swDevId);
InferenceEngine::VPUXConfigParams::VPUXPlatform getPlatformBySwDeviceId(const uint32_t swDevId);
std::string getPlatformNameByDeviceName(const std::string& deviceName);
std::string getDeviceNameBySwDeviceId(const uint32_t swDevId);
}  // namespace utils
