//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"
#include <vpu_ov2_layer_test.hpp>
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/preprocess_builders.hpp"

#include "openvino/op/relu.hpp"

using namespace ov::test;
using namespace ov::test::utils;
using namespace ov::preprocess;

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PrePostProcessTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PreProcessTestCommon);

class PreProcessTestCommon : virtual public PrePostProcessTest, virtual public VpuOv2LayerTest {
public:
    void SetUp() override {
        PrePostProcessTest::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<preprocessParamsTuple>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << PrePostProcessTest::getTestCaseName(obj) << sep;

        return result.str();
    }

protected:
    std::map<std::string, std::string> config;
};

TEST_P(PreProcessTestCommon, NPU3720_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto test_type = std::get<0>(GetParam());
        if (test_type.m_name == "resize_nearest_nchw" || test_type.m_name == "resize_nearest_nhwc") {
            skip << "[Tracking number: E#74951] - Resize nearest is currently giving an incorrect output";
        }
    });
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
