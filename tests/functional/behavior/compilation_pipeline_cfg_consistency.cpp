//
// Copyright (C) 2023 - 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <base/ov_behavior_test_utils.hpp>
#include <string>
#include <vector>
#include "common/functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"
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
        cfg["NPU_COMPILATION_MODE_PARAMS"] = "batch-compile-method=unroll";
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg));
    }
}

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationWithBatchUnrollingSkipBatchOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg["NPU_COMPILATION_MODE_PARAMS"] =
                "batch-compile-method=unroll batch-unroll-settings={skip-unroll-batch=true}";
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg));
    }
}

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationWithDebatchDefaultOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg["NPU_COMPILATION_MODE_PARAMS"] = "batch-compile-method=debatch";
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg));
    }
}

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationWithDebatchNonDefaultOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg["NPU_COMPILATION_MODE_PARAMS"] =
                "batch-compile-method=debatch debatcher-settings={debatching-inlining-method=reordering}";
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg));
    }
}

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationMixUnrollWithDebatchNonDefaultOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg["NPU_COMPILATION_MODE_PARAMS"] =
                "batch-compile-method=unroll debatcher-settings={debatching-inlining-method=reordering}";
        std::string device_id = cfg["DEVICE_ID"].as<std::string>();
        if (device_id.find("3720") != std::string::npos) {
            OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg));
        } else {
            OV_EXPECT_THROW_HAS_SUBSTRING(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg),
                                          std::runtime_error, "is inconsistent");
        }
    }
}

TEST_P(CompilationPipelineCfgConsistencyTests, CompilationMixDebatchWithBatchUnrollingSkipBatchOptions) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cfg = configuration;
        cfg["NPU_COMPILATION_MODE_PARAMS"] =
                "batch-compile-method=debatch batch-unroll-settings={skip-unroll-batch=true}";
        std::string device_id = cfg["DEVICE_ID"].as<std::string>();
        if (device_id.find("3720") != std::string::npos) {
            OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg));
        } else {
            OV_EXPECT_THROW_HAS_SUBSTRING(auto compiled_model = core->compile_model(ov_stub_model, target_device, cfg),
                                          std::runtime_error, "is inconsistent");
        }
    }
}

const std::vector<ov::AnyMap> configs = {
        {{ov::device::id("3720")}, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR)},
        {{ov::device::id("4000")}, ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, CompilationPipelineCfgConsistencyTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         CompilationPipelineCfgConsistencyTests::getTestCaseName);
}  // namespace
