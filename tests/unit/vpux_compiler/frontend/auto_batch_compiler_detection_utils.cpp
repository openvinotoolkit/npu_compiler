//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/op/relu.hpp>
#include <string>
#include <vector>

#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "vpux/compiler/frontend/ov_batch_detection.hpp"
#include "vpux/compiler/utils/batch.hpp"

namespace {

using NpuCompilerParams = std::string;
using ExplicitCfgBatchMethodOptionRequested = bool;
using AutoBatchCompilerCfgValidationParams = std::tuple<NpuCompilerParams, ExplicitCfgBatchMethodOptionRequested>;
class AutoBatchCompilerCfgValidationTests : public testing::TestWithParam<AutoBatchCompilerCfgValidationParams> {
public:
    void SetUp() override {
        auto [compileParams, isOptionMustBeFound] = GetParam();

        auto optionDesc = std::make_shared<intel_npu::OptionsDesc>();
        optionDesc->add<intel_npu::BATCH_COMPILER_MODE_SETTINGS>();
        configurationPtr.reset(new intel_npu::Config(optionDesc));

        std::map<std::string, std::string> rawConfig{
                {{ov::intel_npu::batch_compiler_mode_settings.name(), compileParams}}};
        configurationPtr->update(rawConfig, intel_npu::OptionMode::Both);
        explicitCfgBatchMethodOptionExpected = isOptionMustBeFound;
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AutoBatchCompilerCfgValidationParams>& obj) {
        std::string compileParams;
        bool explicitCfgBatchMethodOptionExpected;
        std::tie(compileParams, explicitCfgBatchMethodOptionExpected) = obj.param;
        std::ostringstream result;
        result << "params=" << compileParams
               << "_explicit_batch_method_option_expected=" << explicitCfgBatchMethodOptionExpected;
        return result.str();
    }

protected:
    std::unique_ptr<intel_npu::Config> configurationPtr;
    bool explicitCfgBatchMethodOptionExpected;
    vpux::Logger logger = vpux::Logger::global();
};

class AutoBatchCompilerOptionsOverridingTests : public AutoBatchCompilerCfgValidationTests {};
TEST_P(AutoBatchCompilerOptionsOverridingTests, checkUserPassedOptions) {
    EXPECT_EQ(vpux::isExplicitCfgBatchMethodOptionRequested(*configurationPtr, logger),
              explicitCfgBatchMethodOptionExpected);
}

const vpux::DebatchCoefficients defaultOverridingCoefficients =
        vpux::DebatchCoefficients::create("[0-2],[0-2],[0-1]").value();
INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTest, AutoBatchCompilerOptionsOverridingTests,
        testing::Values(
                AutoBatchCompilerCfgValidationParams{"", false},
                AutoBatchCompilerCfgValidationParams{"batch-compile-method=unroll", true},
                AutoBatchCompilerCfgValidationParams{"batch-compile-method=debatch "
                                                     "debatcher-settings={debatcher-input-coefficients-partitions=" +
                                                             defaultOverridingCoefficients.to_string() +
                                                             ", debatching-inlining-method=naive}",
                                                     true},
                AutoBatchCompilerCfgValidationParams{"batch-compile-method=debatch", false},
                AutoBatchCompilerCfgValidationParams{
                        "batch-compile-method=debatch debatcher-settings={debatching-inlining-method=naive}", false},
                AutoBatchCompilerCfgValidationParams{"debatcher-settings={debatching-inlining-method=naive}", false},
                AutoBatchCompilerCfgValidationParams{"debatcher-settings={debatcher-input-coefficients-partitions=" +
                                                             defaultOverridingCoefficients.to_string() +
                                                             ", debatching-inlining-method=naive}",
                                                     false}),
        AutoBatchCompilerOptionsOverridingTests::getTestCaseName);
}  // namespace
