//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/ws_base.hpp"

#include <cstdio>

namespace ov {
namespace test {

using namespace ov::test::utils;

void WsBaseTest::TearDown() {
    std::remove(_xmlPath.c_str());
    std::remove(_binPath.c_str());

    VpuOv2LayerTest::TearDown();
}

void WsBaseTest::setupSpecialEnvironment(std::string modelName) {
    const auto filePrefix = ov::test::utils::generateTestFilePrefix();
    _xmlPath = filePrefix + modelName + ".xml";
    _binPath = filePrefix + modelName + ".bin";

    // Note: both PLUGIN and DRIVER are OK here, but PLUGIN is generally
    // preferred for weights separation.
    configuration["NPU_COMPILER_TYPE"] = "PLUGIN";
    configuration["ENABLE_WEIGHTLESS"] = "YES";
    configuration["WEIGHTS_PATH"] = _binPath;
}

}  // namespace test
}  // namespace ov
