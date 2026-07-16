//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/functions.h"
#include "common/npu_test_env_cfg.hpp"

#include <common_test_utils/common_utils.hpp>
#include <shared_test_classes/base/ov_behavior_test_utils.hpp>
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/pass/serialize.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>

namespace {

class CompilationWeightlessTests :
        public ov::test::behavior::OVPluginTestBase,
        public testing::WithParamInterface<std::tuple<std::string, ov::AnyMap>> {
public:
    void SetUp() override {
        std::tie(target_device, _configuration) = GetParam();
        OVPluginTestBase::SetUp();

        const auto filePrefix = ov::test::utils::generateTestFilePrefix();
        _xmlPath = filePrefix + "ws_friendly" + ".xml";
        _binPath = filePrefix + "ws_friendly" + ".bin";

        _configuration["NPU_COMPILER_TYPE"] = "PLUGIN";
        _configuration["ENABLE_WEIGHTLESS"] = "YES";

        auto model = buildSingleWsFriendlyNetwork(ov::element::f32, ov::Shape{1, 2, 3});

        // Note: Model serialization is required due to the nature of the Weight Separation (WS) feature, as it relies
        // on the WeightlessCacheAttribute. This attribute is populated during deserialization, which does not normally
        // occur in typical functional tests.
        ov::pass::Serialize(_xmlPath, _binPath).run_on_model(model);
        _deserializedModel = _core->read_model(_xmlPath, _binPath);
    }

    void TearDown() override {
        std::remove(_xmlPath.c_str());
        std::remove(_binPath.c_str());

        OVPluginTestBase::TearDown();
    }

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<std::string, ov::AnyMap>> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

protected:
    ov::AnyMap _configuration;
    std::shared_ptr<ov::Model> _deserializedModel;
    std::shared_ptr<ov::Core> _core = ov::test::utils::PluginCache::get().core();

    std::string _xmlPath{};
    std::string _binPath{};
};

TEST_P(CompilationWeightlessTests, OneShotAndIterativeMustProduceTheSameBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto cfg = _configuration;
    cfg["NPU_SEPARATE_WEIGHTS_VERSION"] = "ONE_SHOT";
    auto compiledModelOneShot = _core->compile_model(_deserializedModel, target_device, cfg);

    cfg["NPU_SEPARATE_WEIGHTS_VERSION"] = "ITERATIVE";
    auto compiledModelIterative = _core->compile_model(_deserializedModel, target_device, cfg);

    std::stringstream blobStreamOneShot;
    compiledModelOneShot.export_model(blobStreamOneShot);
    std::stringstream blobStreamIterative;
    compiledModelIterative.export_model(blobStreamIterative);

    ASSERT_EQ(blobStreamOneShot.str(), blobStreamIterative.str());
}
const std::vector<ov::AnyMap> configs = {
        {{ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, CompilationWeightlessTests,
                         ::testing::Combine(::testing::Values(test_utils::TARGET_DEVICE), ::testing::ValuesIn(configs)),
                         CompilationWeightlessTests::getTestCaseName);
}  // namespace
