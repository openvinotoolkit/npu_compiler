//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/NPU37XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect_pipeline_strategy.hpp"
#include "vpux/utils/ov/config.hpp"
#include "vpux/utils/ov/options.hpp"

#include <gtest/gtest.h>

using namespace vpux;
using namespace OV;

namespace {
template <class OptionsType>
void checkQDQOptimizationAggressiveEnabled(
        const std::tuple<std::unique_ptr<VPU::InitCompilerOptions>, std::unique_ptr<OptionsType>>& container) {
    bool isAggressiveStrippingEnabled = std::get<0>(container)->enableQDQOptimizationAggressive;

    EXPECT_TRUE(isAggressiveStrippingEnabled);
}

template <class OptionsType>
void checkQDQOptimizationAggressiveDisabled(
        const std::tuple<std::unique_ptr<VPU::InitCompilerOptions>, std::unique_ptr<OptionsType>>& container) {
    bool isAggressiveStrippingEnabled = std::get<0>(container)->enableQDQOptimizationAggressive;

    EXPECT_FALSE(isAggressiveStrippingEnabled);
}

}  // namespace

class QDQOptimizationAggressiveTest : public MLIR_UnitBase, public ::testing::WithParamInterface<config::Platform> {
public:
    void SetUp() override {
        _optionDesc = std::make_shared<OptionsDesc>();
        _optionDesc->add<PLATFORM>();
        _optionDesc->add<QDQ_OPTIMIZATION_AGGRESSIVE>();
    }

    Config makeQDQAggressiveConfig(config::Platform platform,
                                   const std::string& enableQDQOptimizationAggressiveString) {
        Config config(_optionDesc);
        Config::ConfigMap keyValues = {{std::string(PLATFORM::key()), config::stringifyPlatform(platform).str()}};
        if (!enableQDQOptimizationAggressiveString.empty()) {
            keyValues.emplace(std::string(QDQ_OPTIMIZATION_AGGRESSIVE::key()), enableQDQOptimizationAggressiveString);
        }

        config.update(keyValues);
        return config;
    }

    void checkQDQOptimizationAggressive(config::Platform platform,
                                        std::optional<std::string> enableQDQOptimizationAggressive) {
        if (!enableQDQOptimizationAggressive.has_value()) {
            // NOTE: If QDQOptimizationAggressive is unset then expect it to be disabled
            auto config = makeQDQAggressiveConfig(platform, enableQDQOptimizationAggressive.value_or(""));
            isQDQOptimizationAggressiveUnsetOrSetNo(platform, config);
            return;
        }

        if (enableQDQOptimizationAggressive.value() != "YES" && enableQDQOptimizationAggressive.value() != "NO") {
            VPUX_THROW("Expected either 'YES' or 'NO', but got '{0}'", enableQDQOptimizationAggressive.value());
        }

        auto config = makeQDQAggressiveConfig(platform, enableQDQOptimizationAggressive.value());
        if (enableQDQOptimizationAggressive.value() == "YES") {
            isQDQOptimizationAggressiveSetYes(platform, config);
        } else {
            isQDQOptimizationAggressiveUnsetOrSetNo(platform, config);
        }
    }

private:
    void isQDQOptimizationAggressiveSetYes(config::Platform platform, const Config& config) {
        switch (platform) {
        case config::Platform::NPU3720:
            checkQDQOptimizationAggressiveEnabled(createOptionsDefaultHW<DefaultHWOptions37XX>(config));
            break;
        case config::Platform::NPU4000:
            checkQDQOptimizationAggressiveEnabled(createOptionsDefaultHW<DefaultHWOptions40XX>(config));
            break;
        case config::Platform::NPU5010:
        case config::Platform::NPU5020:
            checkQDQOptimizationAggressiveEnabled(createOptionsDefaultHW<DefaultHWOptions50XX>(config));
            break;
        default:
            VPUX_THROW("Unsupported platform: '{0}'", platform);
        }
    }

    void isQDQOptimizationAggressiveUnsetOrSetNo(config::Platform platform, const Config& config) {
        switch (platform) {
        case config::Platform::NPU3720:
            checkQDQOptimizationAggressiveDisabled(createOptionsDefaultHW<DefaultHWOptions37XX>(config));
            break;
        case config::Platform::NPU4000:
            checkQDQOptimizationAggressiveDisabled(createOptionsDefaultHW<DefaultHWOptions40XX>(config));
            break;
        case config::Platform::NPU5010:
        case config::Platform::NPU5020:
            checkQDQOptimizationAggressiveDisabled(createOptionsDefaultHW<DefaultHWOptions50XX>(config));
            break;
        default:
            VPUX_THROW("Unsupported platform: '{0}'", platform);
        }
    }

    std::shared_ptr<OptionsDesc> _optionDesc;
};

TEST_P(QDQOptimizationAggressiveTest, QDQOptimizationAggressiveFlagsEnabled) {
    auto platform = GetParam();
    checkQDQOptimizationAggressive(platform, "YES");
}

TEST_P(QDQOptimizationAggressiveTest, QDQOptimizationAggressiveFlagsDisabled) {
    auto platform = GetParam();
    checkQDQOptimizationAggressive(platform, "NO");
}

TEST_P(QDQOptimizationAggressiveTest, QDQOptimizationAggressiveFlagsUnset) {
    auto platform = GetParam();
    checkQDQOptimizationAggressive(platform, std::nullopt);
}

INSTANTIATE_TEST_SUITE_P(NPU3720_HW, QDQOptimizationAggressiveTest, ::testing::Values(config::Platform::NPU3720));
INSTANTIATE_TEST_SUITE_P(NPU4000_HW, QDQOptimizationAggressiveTest, ::testing::Values(config::Platform::NPU4000));
INSTANTIATE_TEST_SUITE_P(NPU5010_HW, QDQOptimizationAggressiveTest, ::testing::Values(config::Platform::NPU5010));
INSTANTIATE_TEST_SUITE_P(NPU5020_HW, QDQOptimizationAggressiveTest, ::testing::Values(config::Platform::NPU5020));
