//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_behavior_test_utils.hpp>
#include <string>
#include <vector>
#include "common/functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

namespace {

class CompilationPipelineCfgConsistencyTests :
        public ov::test::behavior::OVPluginTestBase,
        public testing::WithParamInterface<std::tuple<std::string, ov::AnyMap>> {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        OVPluginTestBase::SetUp();

        ov_stub_model = buildSingleLayerSoftMaxNetwork();
        configuration["NPU_DEFER_WEIGHTS_LOAD"] = true;
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
    ov::AnyMap configuration;
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> ov_stub_model;
};

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationWithBatchUnrollingDefaultOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg[ov::intel_npu::batch_compiler_mode_settings.name()] = "batch-compile-method=unroll";
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg));
    }
}

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationWithBatchUnrollingSkipBatchOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg[ov::intel_npu::batch_compiler_mode_settings.name()] = "batch-compile-method=unroll "
                                                                  "batch-unroll-settings={skip-unroll-batch=true}";
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg));
    }
}

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationWithDebatchDefaultOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg[ov::intel_npu::batch_compiler_mode_settings.name()] = "batch-compile-method=debatch";
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg));
    }
}

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationWithDebatchNonDefaultOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg[ov::intel_npu::batch_compiler_mode_settings.name()] =
                "batch-compile-method=debatch "
                "debatcher-settings={debatching-inlining-method=reordering}";
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg));
    }
}

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationMixUnrollWithDebatchNonDefaultOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg[ov::intel_npu::batch_compiler_mode_settings.name()] =
                "batch-compile-method=unroll "
                "debatcher-settings={debatching-inlining-method=reordering}";
        OV_EXPECT_THROW_HAS_SUBSTRING(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg),
                                      std::runtime_error, "is inconsistent");
    }
}

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationMixDebatchWithBatchUnrollingSkipBatchOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg[ov::intel_npu::batch_compiler_mode_settings.name()] = "batch-compile-method=debatch "
                                                                  "batch-unroll-settings={skip-unroll-batch=true}";
        OV_EXPECT_THROW_HAS_SUBSTRING(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg),
                                      std::runtime_error, "is inconsistent");
    }
}

const std::vector<ov::AnyMap> configs = {
        {{ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())},
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, CompilationPipelineCfgConsistencyTests,
                         ::testing::Combine(::testing::Values(test_utils::TARGET_DEVICE), ::testing::ValuesIn(configs)),
                         CompilationPipelineCfgConsistencyTests::getTestCaseName);
}  // namespace
