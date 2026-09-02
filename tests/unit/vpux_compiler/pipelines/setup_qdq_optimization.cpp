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

namespace {
template <class OptionsType>
void checkQDQOptimizationEnabled(
        const std::tuple<std::unique_ptr<VPU::InitCompilerOptions>, std::unique_ptr<OptionsType>>& container) {
    bool isAdaptiveStrippingEnabled = std::get<0>(container)->enableAdaptiveStripping;
    bool isQuantDequantRemovalEnabled = std::get<1>(container)->enableQuantDequantRemoval;
    bool isFuseOutstandingDequantEnabled = std::get<1>(container)->enableFuseOutstandingDequant;
    bool isFuseOutstandingQuantEnabled = std::get<1>(container)->enableFuseOutstandingQuant;

    EXPECT_TRUE(isAdaptiveStrippingEnabled);
    EXPECT_TRUE(isQuantDequantRemovalEnabled);
    EXPECT_TRUE(isFuseOutstandingDequantEnabled);
    EXPECT_TRUE(isFuseOutstandingQuantEnabled);
}

template <class OptionsType>
void checkQDQOptimizationDisabled(
        const std::tuple<std::unique_ptr<VPU::InitCompilerOptions>, std::unique_ptr<OptionsType>>& container) {
    bool isAdaptiveStrippingEnabled = std::get<0>(container)->enableAdaptiveStripping;
    bool isQuantDequantRemovalEnabled = std::get<1>(container)->enableQuantDequantRemoval;
    bool isFuseOutstandingDequantEnabled = std::get<1>(container)->enableFuseOutstandingDequant;
    bool isFuseOutstandingQuantEnabled = std::get<1>(container)->enableFuseOutstandingQuant;

    EXPECT_FALSE(isAdaptiveStrippingEnabled);
    EXPECT_FALSE(isQuantDequantRemovalEnabled);
    EXPECT_FALSE(isFuseOutstandingDequantEnabled);
    EXPECT_FALSE(isFuseOutstandingQuantEnabled);
}

}  // namespace

class QDQOptimizationTest : public MLIR_UnitBase, public ::testing::WithParamInterface<config::Platform> {
public:
    void SetUp() override {
        _optionDesc = std::make_shared<vpux::OV::OptionsDesc>();
        _optionDesc->add<vpux::OV::PLATFORM>();
        _optionDesc->add<vpux::OV::QDQ_OPTIMIZATION>();
    }

    vpux::OV::Config makeQDQConfig(config::Platform platform, const std::string& enableQDQOptimizationString) {
        vpux::OV::Config config(_optionDesc);
        config.update({
                {std::string(vpux::OV::PLATFORM::key()), config::stringifyPlatform(platform).str()},
                {std::string(vpux::OV::QDQ_OPTIMIZATION::key()), enableQDQOptimizationString},
        });
        return config;
    }

    vpux::OV::Config makeQDQConfig(config::Platform platform) {
        vpux::OV::Config config(_optionDesc);
        config.update({{std::string(vpux::OV::PLATFORM::key()), config::stringifyPlatform(platform).str()}});
        return config;
    }

    void checkQDQOptimization(config::Platform platform, std::optional<std::string> enableQDQOptimization) {
        if (!enableQDQOptimization.has_value()) {
            // NOTE: If QDQOptimization is unset then expect it to be enabled
            auto config = makeQDQConfig(platform);
            isQDQOptimizationEnabled(platform, config);
            return;
        }

        if (enableQDQOptimization.value() != "YES" && enableQDQOptimization.value() != "NO") {
            VPUX_THROW("Expected either 'YES' or 'NO', but got '{0}'", enableQDQOptimization.value());
        }

        auto config = makeQDQConfig(platform, enableQDQOptimization.value());
        if (enableQDQOptimization.value() == "YES") {
            isQDQOptimizationEnabled(platform, config);
        } else {
            isQDQOptimizationDisabled(platform, config);
        }
    }

private:
    void isQDQOptimizationEnabled(config::Platform platform, const vpux::OV::Config& config) {
        switch (platform) {
        case config::Platform::NPU3720:
            checkQDQOptimizationEnabled(createOptionsDefaultHW<DefaultHWOptions37XX>(config));
            break;
        case config::Platform::NPU4000:
            checkQDQOptimizationEnabled(createOptionsDefaultHW<DefaultHWOptions40XX>(config));
            break;
        case config::Platform::NPU5010:
        case config::Platform::NPU5020:
            checkQDQOptimizationEnabled(createOptionsDefaultHW<DefaultHWOptions50XX>(config));
            break;
        default:
            VPUX_THROW("Unsupported platform: '{0}'", platform);
        }
    }

    void isQDQOptimizationDisabled(config::Platform platform, const vpux::OV::Config& config) {
        switch (platform) {
        case config::Platform::NPU3720:
            checkQDQOptimizationDisabled(createOptionsDefaultHW<DefaultHWOptions37XX>(config));
            break;
        case config::Platform::NPU4000:
            checkQDQOptimizationDisabled(createOptionsDefaultHW<DefaultHWOptions40XX>(config));
            break;
        case config::Platform::NPU5010:
        case config::Platform::NPU5020:
            checkQDQOptimizationDisabled(createOptionsDefaultHW<DefaultHWOptions50XX>(config));
            break;
        default:
            VPUX_THROW("Unsupported platform: '{0}'", platform);
        }
    }

    std::shared_ptr<vpux::OV::OptionsDesc> _optionDesc;
};

TEST_P(QDQOptimizationTest, QDQOptimizationFlagsDisabled) {
    auto platform = GetParam();
    checkQDQOptimization(platform, "NO");
}

TEST_P(QDQOptimizationTest, QDQOptimizationFlagsEnabled) {
    auto platform = GetParam();
    checkQDQOptimization(platform, "YES");
}

TEST_P(QDQOptimizationTest, QDQOptimizationFlagsUnset) {
    auto platform = GetParam();
    checkQDQOptimization(platform, std::nullopt);
}

INSTANTIATE_TEST_SUITE_P(NPU3720_HW, QDQOptimizationTest, ::testing::Values(config::Platform::NPU3720));
INSTANTIATE_TEST_SUITE_P(NPU4000_HW, QDQOptimizationTest, ::testing::Values(config::Platform::NPU4000));
INSTANTIATE_TEST_SUITE_P(NPU5010_HW, QDQOptimizationTest, ::testing::Values(config::Platform::NPU5010));
INSTANTIATE_TEST_SUITE_P(NPU5020_HW, QDQOptimizationTest, ::testing::Values(config::Platform::NPU5020));
