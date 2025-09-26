// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/memory_LSTMCell.hpp"
#include <common_test_utils/test_constants.hpp>
#include <shared_test_classes/subgraph/memory_LSTMCell.hpp>
#include "common/npu_test_env_cfg.hpp"
#include "vpu_test_tool.hpp"

using namespace ov::test::utils;

namespace ov::test {

static std::string getTestCaseName(testing::TestParamInfo<memoryLSTMCellParams> obj) {
    MemoryTransformation memoryTransform;
    ov::element::Type precision;
    std::string targetDevice;
    size_t inputSize;
    size_t outputSize;
    ov::AnyMap configuration;
    std::tie(memoryTransform, std::ignore, precision, inputSize, outputSize, configuration) = obj.param;
    const std::string sep = "_";
    std::ostringstream result;
    targetDevice = LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
    result << "targetDevice=" << targetDevice << sep;
    result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
    result << "TestIdx=" << obj.index << sep;
    result << "inSize=" << inputSize << sep;
    result << "outSize=" << outputSize << sep;
    return result.str();
}

std::vector<MemoryTransformation> transformation{
        MemoryTransformation::NONE,
};

std::vector<size_t> inputSizes = {80, 32, 64, 100, 25};

std::vector<size_t> hiddenSizes = {
        128, 200, 300, 24, 32,
};

ov::AnyMap additionalConfig = {};

INSTANTIATE_TEST_SUITE_P(
        smoke_MemoryLSTMCellTest, MemoryLSTMCellTest,
        ::testing::Combine(
                ::testing::ValuesIn(transformation),
                ::testing::Values(std::string(test_utils::TARGET_DEVICE) + "." +
                                  removeDeviceNameOnlyID(ov::test::utils::getTestsPlatformFromEnvironmentOr("3720"))),
                ::testing::Values(ov::element::f32), ::testing::ValuesIn(inputSizes), ::testing::ValuesIn(hiddenSizes),
                ::testing::Values(additionalConfig)),
        getTestCaseName);
}  // namespace ov::test
