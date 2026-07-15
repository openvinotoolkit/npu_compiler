//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/pipeline_options.hpp"
#include "vpux/compiler/pipelines/options_setup.hpp"

#include <gtest/gtest.h>

using namespace vpux;

// Local test helper that mimics DefaultHWSetup40XX::setupOptionsImpl logic
class TestDefaultHWSetup40XX : public OptionsSetupBase<TestDefaultHWSetup40XX, DefaultHWOptions40XX> {
public:
    static void setupOptionsImpl(DefaultHWOptions40XX& options, const intel_npu::Config& config) {
        if (config.get<intel_npu::TURBO>()) {
            overwriteIfUnset(options.optimizationLevel, 3);
        }
        setupParamsAccordingToOptimizationLevel(options.optimizationLevel, options);
    }
};

class OptionsSetupTurboTest : public ::testing::Test {
public:
    void SetUp() override {
        _optionDesc = std::make_shared<intel_npu::OptionsDesc>();
        _optionDesc->add<intel_npu::TURBO>();
        _config = intel_npu::Config(_optionDesc);
        _config->update({{std::string(intel_npu::TURBO::key()), "YES"}});
    }

    std::shared_ptr<intel_npu::OptionsDesc> _optionDesc;
    std::optional<intel_npu::Config> _config;
};

TEST_F(OptionsSetupTurboTest, UserSetEnableReduceNumTilesForSmallModelsPassIsNotOverriddenByTurbo) {
    DefaultHWOptions40XX options;
    options.enableReduceNumTilesForSmallModelsPass = false;  // user sets explicitly

    ASSERT_TRUE(_config.has_value());
    TestDefaultHWSetup40XX::setupOptionsImpl(options, _config.value());

    // Should remain as set by user, not overridden by TURBO logic
    EXPECT_FALSE(options.enableReduceNumTilesForSmallModelsPass);
}
