//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/op/relu.hpp>
#include <string>
#include <vector>

#include "common/utils.hpp"
#include "vpux/compiler/frontend/ov_batch_detection.hpp"
#include "vpux/compiler/utils/batch.hpp"
#include "vpux/utils/ov/config.hpp"
#include "vpux/utils/ov/options.hpp"

namespace {

using NpuCompilerParams = std::string;
using ExplicitCfgBatchMethodOptionRequested = bool;
using AutoBatchCompilerCfgValidationParams = std::tuple<NpuCompilerParams, ExplicitCfgBatchMethodOptionRequested>;
class AutoBatchCompilerCfgValidationTests : public testing::TestWithParam<AutoBatchCompilerCfgValidationParams> {
public:
    void SetUp() override {
        auto [compileParams, isOptionMustBeFound] = GetParam();

        auto optionDesc = std::make_shared<vpux::OV::OptionsDesc>();
        optionDesc->add<vpux::OV::BATCH_COMPILER_MODE_SETTINGS>();
        configurationPtr.reset(new vpux::OV::Config(optionDesc));

        std::map<std::string, std::string> rawConfig{
                {{vpux::OV::BATCH_COMPILER_MODE_SETTINGS::key().data(), compileParams}}};
        configurationPtr->update(rawConfig);
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
    std::unique_ptr<vpux::OV::Config> configurationPtr;
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
