//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_behavior_test_utils.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common/functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"

namespace {

using VclExtensionLibPathTestsParam = std::tuple<std::string, ov::AnyMap, ov::AnyMap>;

class VclExtensionLibPathTests :
        public ov::test::behavior::OVPluginTestBase,
        public testing::WithParamInterface<VclExtensionLibPathTestsParam> {
public:
    void SetUp() override {
        // Skip test according to plugin specific disabled_test_patterns() (if any).
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        ov::AnyMap commonConfig;
        ov::AnyMap extensionLibPathConfig;
        std::tie(target_device, commonConfig, extensionLibPathConfig) = GetParam();

        config = commonConfig;
        config.insert(extensionLibPathConfig.begin(), extensionLibPathConfig.end());
        OVPluginTestBase::SetUp();
    }

    static std::string getTestCaseName(testing::TestParamInfo<VclExtensionLibPathTestsParam> obj) {
        std::string targetDevice;
        ov::AnyMap commonConfig;
        ov::AnyMap extensionLibPathConfig;
        std::tie(targetDevice, commonConfig, extensionLibPathConfig) = obj.param;

        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        result << "commonConfig=";
        for (auto& configItem : commonConfig) {
            result << configItem.first << "=";
            configItem.second.print(result);
            result << "_";
        }
        result << "extensionLibPathConfig=";
        for (auto& configItem : extensionLibPathConfig) {
            result << configItem.first << "=";
            configItem.second.print(result);
            result << "_";
        }
        return result.str();
    }

protected:
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    ov::AnyMap config;
};

class VclExtensionLibNoPathTests : public VclExtensionLibPathTests {};
class VclExtensionLibBadPathTests : public VclExtensionLibPathTests {};

TEST_P(VclExtensionLibNoPathTests, EmptyPathsDoNotBreakCompilation) {
    const auto& ov_model = buildSingleLayerSoftMaxNetwork();

    // Expect to see no exception thrown
    OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_model, target_device, config));
}

TEST_P(VclExtensionLibBadPathTests, InvalidPathFailsCompilation) {
    const auto& ov_model = buildSingleLayerSoftMaxNetwork();

    // Expect throw: compilation should fail when trying to load a non-existent extension library.
    ASSERT_THROW(auto compiled_model = core->compile_model(ov_model, target_device, config), ov::Exception);
}

const std::vector<ov::AnyMap> commonConfigs = {
        {{"NPU_COMPILER_TYPE", "PLUGIN"}, {"NPU_MODEL_SERIALIZER_VERSION", "ALL_WEIGHTS_COPY"}},
        {{"NPU_COMPILER_TYPE", "PLUGIN"}, {"NPU_MODEL_SERIALIZER_VERSION", "NO_WEIGHTS_COPY"}}};

const std::vector<ov::AnyMap> validExtensionLibPathConfigs = {{{"OV_EXTENSION_LIB_PATH", ""}},
                                                              {{"OV_EXTENSION_LIB_PATH", ";;;"}}};

const std::vector<ov::AnyMap> invalidExtensionLibPathConfigs = {
        {{"OV_EXTENSION_LIB_PATH", "__fake_path__/lib.dll"}},
        {{"OV_EXTENSION_LIB_PATH", "__fake_path__/lib1.dll;__fake_path__/lib2.dll"}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest_ExtensionLibPath, VclExtensionLibNoPathTests,
                         ::testing::Combine(::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::ValuesIn(commonConfigs),
                                            ::testing::ValuesIn(validExtensionLibPathConfigs)),
                         VclExtensionLibPathTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest_ExtensionLibPath, VclExtensionLibBadPathTests,
                         ::testing::Combine(::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::ValuesIn(commonConfigs),
                                            ::testing::ValuesIn(invalidExtensionLibPathConfigs)),
                         VclExtensionLibPathTests::getTestCaseName);

}  // namespace
