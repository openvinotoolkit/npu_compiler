//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <iostream>
#include <openvino/runtime/core.hpp>
#include <string>
#include <string_view>
#include "common/npu_test_env_cfg.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace ov::test::utils {

class VpuTestTool {
public:
    const VpuTestEnvConfig& envConfig;
    const std::string DEVICE_NAME;
    vpux::Logger _log;

public:
    explicit VpuTestTool(const VpuTestEnvConfig& envCfg);

    void exportModel(ov::CompiledModel& compiledModel, const std::string& fsName);
    void exportTensor(const ov::Tensor& tensor, const std::string& fsName);
    void importTensor(ov::Tensor& tensor, const std::string& fsName);
    std::string getDeviceMetric(std::string name);
};

std::string filesysName(const testing::TestInfo* testInfo, const std::string& ext, bool limitAbsPathLength);

constexpr std::string_view testKind(std::string_view filePath) {
    if (filePath.find("subgraph") != std::string::npos) {
        return std::string_view("Subgraph");
    } else if (filePath.find("single_layer") != std::string::npos) {
        return std::string_view("SingleLayer");
    } else if (filePath.find("behavior") != std::string::npos) {
        return std::string_view("Behavior");
    } else {
        return std::string_view("Unknown");
    }
}

}  // namespace ov::test::utils

namespace LayerTestsUtils {
using ov::test::utils::filesysName;
using ov::test::utils::VpuTestTool;
}  // namespace LayerTestsUtils
